import re
from collections import OrderedDict

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.readout = nn.Linear(input_dim, num_classes)

    def forward(self, latent):
        logits = self.readout(latent)  # (B, L, num_classes)
        return logits


class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, n_hidden_layers=1, output_shape=None):
        super(AttentionDecoder, self).__init__()
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        flat_output_dim = int(torch.tensor(self.output_shape).prod().item()) if self.output_shape else output_dim
        hidden_dim = hidden_dim or input_dim
        layers = []
        for layer_idx in range(n_hidden_layers):
            in_features = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, flat_output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        attention = 1.0 + torch.tanh(self.layers(x))
        if self.output_shape is not None:
            attention = attention.reshape(*attention.shape[:-1], *self.output_shape)
        return attention


class ARPLmodel(nn.Module):
    """
    Attentive Predictive Learning

    Assumes video format of data (B, L, C, H, W) and processes sequentially along the L dimension.
    """
    def __init__(self,
                 encoder,
                 integrator,
                 predictor,
                 head,
                 decoder,
                 num_tasks,
                 preprocess=None,
                 postprocess=None,
                 freeze_repl=True,
                 eval_frozen=True,
                 decoder_input_dim=None,
                 use_task_embedding=True,
                 class_prompt_value=10.0):
        super(ARPLmodel, self).__init__()
        self.preprocess = preprocess
        self.encoder_first, self.encoder_tail, self.encoder_returns_full_features = self._split_encoder(encoder)
        self.postprocess = postprocess
        self.integrator = integrator
        self.predictor = predictor
        self.use_task_embedding = use_task_embedding
        self.class_prompt_value = float(class_prompt_value)

        if freeze_repl:
            for module in (self.encoder_first, self.encoder_tail, self.integrator, self.predictor):
                for param in module.parameters():
                    param.requires_grad_(False)
                if eval_frozen:
                    module.eval()

        self.head = head
        self.decoder = decoder
        self.context_dim = head.input_dim
        self.decoder_input_dim = decoder_input_dim or head.input_dim
        self.task_embedding = nn.Embedding(num_tasks, self.decoder_input_dim)
        self.class_feedback = nn.Linear(head.num_classes, self.context_dim, bias=False)
        if not self.use_task_embedding:
            for param in self.task_embedding.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _layer_index(name: str) -> int | None:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else None

    @classmethod
    def _split_encoder(cls, encoder):
        if isinstance(encoder, (list, tuple, nn.ModuleList)):
            if len(encoder) < 1:
                raise ValueError("encoder stage list cannot be empty")
            stages = nn.ModuleList(encoder)
            first = stages[0]
            tail = nn.Sequential(*stages[1:]) if len(stages) > 1 else nn.Identity()
            return first, tail, True

        backbone = getattr(encoder, "backbone", None)
        if not isinstance(backbone, nn.Sequential):
            raise TypeError("ARPLmodel expects a staged encoder or an encoder with a Sequential backbone.")

        children = list(backbone.named_children())
        if not children:
            raise ValueError("encoder backbone cannot be empty")

        first_idx = cls._layer_index(children[0][0])
        split_at = 1
        if first_idx is not None:
            for split_at, (name, _) in enumerate(children, start=1):
                if cls._layer_index(name) != first_idx:
                    split_at -= 1
                    break
            else:
                split_at = len(children)

        first = nn.Sequential(OrderedDict(children[:split_at]))
        tail_items = children[split_at:]
        tail = nn.Sequential(OrderedDict(tail_items)) if tail_items else nn.Identity()
        returns_full_features = bool(getattr(encoder, "return_full_feature_map", False))
        return first, tail, returns_full_features

    @staticmethod
    def _forward_video_layers(layers, x):
        if isinstance(layers, nn.Identity):
            return x
        if x.ndim == 5:
            batch_size, seq_len = x.shape[:2]
            x = x.reshape(batch_size * seq_len, *x.shape[2:])
            x = layers(x)
            x = x.reshape(batch_size, seq_len, *x.shape[1:])
            return x.contiguous()
        return layers(x)

    def _forward_encoder_first(self, data_t):
        return self._forward_video_layers(self.encoder_first, data_t)

    def _forward_encoder_tail(self, z0_t):
        z_t = self._forward_video_layers(self.encoder_tail, z0_t)
        if z_t.ndim == 5 and not self.encoder_returns_full_features:
            z_t = z_t.mean(-1).mean(-1)
        return z_t

    @staticmethod
    def _format_attention(attention, target):
        if target.ndim == 5:
            if attention.ndim == 2:
                attention = attention[:, None, :, None, None]
            elif attention.ndim == 3:
                attention = attention[:, :, :, None, None]
            elif attention.ndim != 5:
                raise RuntimeError(
                    "attention decoder output for spatial features must be either "
                    f"(batch, channels), (batch, time, channels), or full-rank (batch, time, channels, height, width); "
                    f"got {tuple(attention.shape)}"
                )
        elif target.ndim == 3 and attention.ndim == 2:
            attention = attention[:, None, :]

        if attention.ndim != target.ndim:
            raise RuntimeError(
                "attention decoder output rank must match first encoder features after formatting; "
                f"(batch, time, channels); got {tuple(attention.shape)} for {tuple(target.shape)}"
            )
        try:
            broadcast_shape = torch.broadcast_shapes(attention.shape, target.shape)
        except RuntimeError as exc:
            raise RuntimeError(
                "attention decoder output must be broadcastable to first encoder features; "
                f"got {tuple(attention.shape)} for {tuple(target.shape)}"
            ) from exc
        if tuple(broadcast_shape) != tuple(target.shape):
            raise RuntimeError(
                "attention decoder output must broadcast exactly to first encoder features; "
                f"got {tuple(attention.shape)} for {tuple(target.shape)}"
            )
        return attention

    def _make_context_feedback(self, task_info, logits_t):
        if task_info is None:
            return self.class_feedback(logits_t)
        task_info = task_info.to(device=logits_t.device, dtype=torch.long)
        target_class = task_info[:, 1]
        prompt_mask = task_info[:, 2].to(dtype=torch.bool).view(-1, 1, 1)
        prompt_logits = torch.zeros_like(logits_t)
        prompt_logits.scatter_(
            dim=-1,
            index=target_class.view(-1, 1, 1),
            value=self.class_prompt_value,
        )
        feedback_logits = torch.where(prompt_mask, prompt_logits, logits_t)
        return self.class_feedback(feedback_logits)

    def _prepare_task_info(self, task_info, data):
        if task_info is None:
            return None
        task_info = task_info.to(device=data.device, dtype=torch.long)
        return task_info.unsqueeze(0) if task_info.ndim == 1 else task_info

    def _initial_decoder_state(self, data, use_attention):
        if not use_attention:
            return None
        return torch.zeros(
            data.size(0),
            1,
            self.decoder_input_dim,
            dtype=data.dtype,
            device=data.device,
        )

    def _decoder_input(self, pred, task_info=None):
        if pred is None:
            return None
        pred = pred.unsqueeze(1) if pred.ndim == 2 else pred
        if pred.ndim != 3 or pred.shape[-1] != self.decoder_input_dim:
            raise RuntimeError(
                "predictor output must be the ARPL decoder input and match decoder_input_dim; "
                f"got {tuple(pred.shape)} with decoder_input_dim={self.decoder_input_dim}"
            )
        if self.use_task_embedding and task_info is not None:
            task_emb = self.task_embedding(task_info[:, 0]).unsqueeze(1)
            pred = pred + task_emb
        return pred

    def forward(self, data, return_logits_only=False, use_attention=True):
        task_info = None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data, task_info = data

        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        zs, cs, ps, ks = [], [], [], []  # lists to store outputs at each timestep
        integrator_hidden_t = getattr(self.integrator, "hidden", None)
        task_info = self._prepare_task_info(task_info, data)
        p_t = self._initial_decoder_state(data, use_attention)

        # Sequential processing of visual stream
        for t in range(data.size(1)):
            data_t = data[:, t:t+1, ...]  # (B, 1, C, H, W)
            decoder_input_t = self._decoder_input(p_t, task_info) if use_attention else None
            z_t = self.encoder_step(data_t, decoder_input_t)
            c_t, integrator_hidden_t = self.integrator_step(z_t, integrator_hidden_t)

            # Classifier is intentionally applied before class-prompt feedback.
            k_t = self.head(c_t)  # (B, 1, num_classes)
            p_t = self.predictor_step(c_t, task_info, use_attention, k_t)

            if not return_logits_only:
                zs.append(z_t)
                cs.append(c_t)
                ps.append(p_t)
            ks.append(k_t)

        # Returns: stacked outputs from all timesteps
        logits = torch.cat(ks, dim=1)  # (B, L, num_classes)
        if return_logits_only:
            return logits

        z = torch.cat(zs, dim=1)  # (B, L, C)
        context_tensor = torch.cat(cs, dim=1)  # (B, L, C)
        pred = torch.cat(ps, dim=1)  # (B, L, C*(num_pred_steps=1))
        return z, context_tensor, pred, logits

    def encoder_step(self,
                     data_t,
                     decoder_input_t=None):
        z0_t = self._forward_encoder_first(data_t)
        # compute dynamic attention with decoder
        if decoder_input_t is not None:
            a0_t = self._format_attention(self.decoder(decoder_input_t), z0_t)
            z0_t = z0_t * a0_t
        # apply attention on first encoder layer, and let the rest of the encoder process the attended representation
        z_t = self._forward_encoder_tail(z0_t)
        if self.postprocess is not None:
            z_t = self.postprocess(z_t)
        return z_t

    def integrator_step(self,
                        z_t,
                        integrator_hidden_t=None):
        backbone = getattr(self.integrator, "backbone")
        if isinstance(backbone, nn.RNNBase):
            context_tensor_t, integrator_hidden_t = backbone(z_t, integrator_hidden_t)
        else:
            context_tensor_t = self.integrator(z_t)
        return context_tensor_t, integrator_hidden_t

    # TODO: context FB should only affect decoder input; data should pass through predictor unaltered
    def predictor_step(self, context_t, task_info, use_attention, logits_t=None):
        if not use_attention:
            return self.predictor(context_t)
        context_FB =self._make_context_feedback(task_info, logits_t)
        return self.predictor(context_t *torch.tanh(context_FB))
