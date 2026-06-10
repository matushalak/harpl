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


class ChannelAttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, n_hidden_layers=1):
        super(ChannelAttentionDecoder, self).__init__()
        hidden_dim = hidden_dim or input_dim
        layers = []
        for layer_idx in range(n_hidden_layers):
            in_features = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.layers(x))


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
                 decoder_input_dim=None):
        super(ARPLmodel, self).__init__()
        self.preprocess = preprocess
        self.encoder_first, self.encoder_tail, self.encoder_returns_full_features = self._split_encoder(encoder)
        self.postprocess = postprocess
        self.integrator = integrator
        self.predictor = predictor

        if freeze_repl:
            for module in (self.encoder_first, self.encoder_tail, self.integrator, self.predictor):
                for param in module.parameters():
                    param.requires_grad_(False)
                if eval_frozen:
                    module.eval()

        self.head = head
        self.decoder = decoder
        self.decoder_input_dim = decoder_input_dim or head.input_dim
        self.task_embedding = nn.Embedding(num_tasks, self.decoder_input_dim)
        self.class_embedding = nn.Embedding(head.num_classes, self.decoder_input_dim)

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
        elif target.ndim == 3 and attention.ndim == 2:
            attention = attention[:, None, :]

        if attention.shape[:3] != target.shape[:3]:
            raise RuntimeError(
                "attention decoder output must match the first encoder block over "
                f"(batch, time, channels); got {tuple(attention.shape)} for {tuple(target.shape)}"
            )
        return attention

    def forward(self, data, return_logits_only=False):
        has_task = False
        task_info = None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data, task_info = data
            has_task = True

        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        zs, cs, ps, ks = [], [], [], []  # lists to store outputs at each timestep
        integrator_hidden_t = getattr(self.integrator, "hidden", None)

        # Visual stream accompanied by prompt for attentional task
        if has_task:
            task_info = task_info.to(device=data.device, dtype=torch.long)
            if task_info.ndim == 1:
                task_info = task_info.unsqueeze(0)
            # task info is of shape (B, 3), where
            #   first column is task ID; and
            task_id = task_info[:, 0]  # (B, 1)
            task_emb = self.task_embedding(task_id)  # (B, decoder_input_dim)
            #   second column is target class ID; and
            #   third column indicates whether target class ID is given as prompt of not
            prompt_mask = task_info[:, 2].to(dtype=task_emb.dtype).unsqueeze(-1)
            prompt_emb = self.class_embedding(task_info[:, 1]) * prompt_mask  # (B, decoder_input_dim)
            # Combine task and class embeddings to form the task-dependent decoder input
            decoder_input_const = (task_emb + prompt_emb).unsqueeze(1)  # (B, 1, decoder_input_dim)
            # Initialize p_t for the first timestep
            p_t = torch.zeros_like(decoder_input_const)  # (B, 1, decoder_input_dim)
        else:
            decoder_input_const = None
            p_t = None

        L = data.size(1) # sequence length

        # Sequential processing of visual stream
        for t in range(L):
            # Get the t-th timestep's data
            data_t = data[:, t:t+1, ...]  # (B, 1, C, H, W)

            # decoder input only if task info is available.
            if has_task:
                if p_t.ndim == 2:
                    p_t = p_t.unsqueeze(1)
                if p_t.shape != decoder_input_const.shape:
                    raise RuntimeError(
                        "predictor output must match task embedding shape for ARPL decoder input; "
                        f"got {tuple(p_t.shape)} and {tuple(decoder_input_const.shape)}"
                    )
                decoder_input_t = decoder_input_const + p_t
            else:
                decoder_input_t = None

            # process the t-th timestep
            z_t, c_t, p_t, integrator_hidden_t = self.forward_step(data_t,
                                                                   integrator_hidden_t,
                                                                   decoder_input_t
                                                                   )
            # classifier based on the context representation at the current timestep
            k_t = self.head(c_t)  # (B, 1, num_classes)

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


    def forward_step(self,
                     data_t,
                     integrator_hidden_t=None,
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

        backbone = getattr(self.integrator, "backbone")
        if isinstance(backbone, nn.RNNBase):
            context_tensor_t, integrator_hidden_t = backbone(z_t, integrator_hidden_t)
        else:
            context_tensor_t = self.integrator(z_t)

        pred_t = self.predictor(context_tensor_t)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)
        return z_t, context_tensor_t, pred_t, integrator_hidden_t
