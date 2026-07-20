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


class FeatureSpatialAttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_shape, hidden_dim=None, n_hidden_layers=1):
        super(FeatureSpatialAttentionDecoder, self).__init__()
        self.output_shape = tuple(int(v) for v in output_shape)
        output_dim = 1
        for value in self.output_shape:
            output_dim *= value
        hidden_dim = hidden_dim or input_dim
        layers = []
        for layer_idx in range(n_hidden_layers):
            in_features = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = torch.tanh(self.layers(x))
        return out.view(*x.shape[:-1], *self.output_shape)


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
                 use_prompt_embedding=True):
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
        self.decoder_embedding_dim = decoder_input_dim or head.input_dim
        self.decoder_input_dim = 2 * self.decoder_embedding_dim
        self.use_task_embedding = use_task_embedding
        self.use_prompt_embedding = use_prompt_embedding
        self.task_embedding = nn.Embedding(num_tasks, self.decoder_embedding_dim)
        self.class_embedding = nn.Embedding(head.num_classes, self.decoder_embedding_dim)

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

    def forward(self, data, return_logits_only=False, return_attention=False):
        has_task = False
        task_info = None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data, task_info = data
            has_task = True

        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        zs, cs, ps, ks, attentions = [], [], [], [], []  # lists to store outputs at each timestep
        integrator_hidden_t = getattr(self.integrator, "hidden", None)

        # Visual stream accompanied by prompt for attentional task
        if has_task:
            task_info = task_info.to(device=data.device, dtype=torch.long)
            if task_info.ndim == 1:
                task_info = task_info.unsqueeze(0)
            # task info is of shape (B, 3), where
            #   first column is task ID; and
            task_id = task_info[:, 0]  # (B, 1)
            task_emb = self.task_embedding(task_id) if self.use_task_embedding else torch.zeros(
                task_id.shape[0],
                self.decoder_embedding_dim,
                dtype=data.dtype,
                device=data.device,
            )
            #   second column is target class ID; and
            #   third column indicates whether target class ID is given as prompt or not
            prompt_mask = task_info[:, 2].to(dtype=task_emb.dtype).unsqueeze(-1)
            if self.use_prompt_embedding:
                prompt_emb = self.class_embedding(task_info[:, 1]) * prompt_mask  # (B, decoder_embedding_dim)
            else:
                prompt_emb = torch.zeros_like(task_emb)
            task_prompt_emb = (task_emb + prompt_emb).unsqueeze(1)  # (B, 1, decoder_embedding_dim)
            # Initialize p_t for the first timestep
            p_t = torch.zeros_like(task_prompt_emb)  # (B, 1, decoder_embedding_dim)
        else:
            task_prompt_emb = None
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
                if p_t.shape != task_prompt_emb.shape:
                    raise RuntimeError(
                        "predictor output must match task/prompt embedding shape before ARPL decoder concatenation; "
                        f"got {tuple(p_t.shape)} and {tuple(task_prompt_emb.shape)}"
                    )
                decoder_input_t = torch.cat((task_prompt_emb, p_t), dim=-1)
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
            if return_attention:
                attentions.append(self._last_attention_t)
            ks.append(k_t)

        # Returns: stacked outputs from all timesteps
        logits = torch.cat(ks, dim=1)  # (B, L, num_classes)
        if return_logits_only and not return_attention:
            return logits
        if return_logits_only and return_attention:
            return logits, torch.cat(attentions, dim=1)

        z = torch.cat(zs, dim=1)  # (B, L, C)
        context_tensor = torch.cat(cs, dim=1)  # (B, L, C)
        pred = torch.cat(ps, dim=1)  # (B, L, C*(num_pred_steps=1))
        if return_attention:
            return z, context_tensor, pred, logits, torch.cat(attentions, dim=1)
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
            self._last_attention_t = a0_t
        else:
            self._last_attention_t = torch.ones_like(z0_t)
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


class HierarchicalARPLmodel(nn.Module):
    def __init__(
        self,
        hierarchical_repl,
        head,
        decoder,
        num_tasks,
        freeze_repl=True,
        eval_frozen=True,
        decoder_input_dim=None,
        readout_area=-1,
        use_task_embedding=True,
        use_prompt_embedding=True,
    ):
        super(HierarchicalARPLmodel, self).__init__()
        self.preprocess = hierarchical_repl.preprocess
        self.postprocess = hierarchical_repl.postprocess
        self.areas = hierarchical_repl.areas
        self.n_areas = hierarchical_repl.n_areas
        self.readout_area = readout_area % self.n_areas

        self.area0_first, self.area0_tail, self.area0_returns_full_features = ARPLmodel._split_encoder(self.areas[0]["encoder"])
        if freeze_repl:
            for area in self.areas:
                for module in (area["encoder"], area["integrator"], area["predictor"]):
                    for param in module.parameters():
                        param.requires_grad_(False)
                    if eval_frozen:
                        module.eval()

        self.head = head
        self.decoder = decoder
        self.decoder_embedding_dim = decoder_input_dim or head.input_dim
        self.decoder_input_dim = 2 * self.decoder_embedding_dim
        self.use_task_embedding = use_task_embedding
        self.use_prompt_embedding = use_prompt_embedding
        self.task_embedding = nn.Embedding(num_tasks, self.decoder_embedding_dim)
        self.class_embedding = nn.Embedding(head.num_classes, self.decoder_embedding_dim)

    def _area0_encode(self, data_t, decoder_input_t):
        z0_t = ARPLmodel._forward_video_layers(self.area0_first, data_t)
        if decoder_input_t is not None:
            a0_t = ARPLmodel._format_attention(self.decoder(decoder_input_t), z0_t)
            z0_t = z0_t * a0_t
        else:
            a0_t = torch.ones_like(z0_t)
        z_t = ARPLmodel._forward_video_layers(self.area0_tail, z0_t)
        if z_t.ndim == 5 and not self.area0_returns_full_features:
            z_t = z_t.mean(-1).mean(-1)
        return z_t, a0_t

    @staticmethod
    def _run_integrator(integrator, z_t, hidden_t):
        backbone = getattr(integrator, "backbone", None)
        if isinstance(backbone, nn.RNNBase):
            return backbone(z_t, hidden_t)
        return integrator(z_t), hidden_t

    def forward(self, data, return_logits_only=False, return_attention=False):
        has_task = False
        task_info = None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data, task_info = data
            has_task = True

        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        if has_task:
            task_info = task_info.to(device=data.device, dtype=torch.long)
            if task_info.ndim == 1:
                task_info = task_info.unsqueeze(0)
            task_emb = self.task_embedding(task_info[:, 0]) if self.use_task_embedding else torch.zeros(
                task_info.shape[0],
                self.decoder_embedding_dim,
                dtype=data.dtype,
                device=data.device,
            )
            prompt_mask = task_info[:, 2].to(dtype=task_emb.dtype).unsqueeze(-1)
            if self.use_prompt_embedding:
                prompt_emb = self.class_embedding(task_info[:, 1]) * prompt_mask
            else:
                prompt_emb = torch.zeros_like(task_emb)
            task_prompt_emb = (task_emb + prompt_emb).unsqueeze(1)
            p_t = torch.zeros_like(task_prompt_emb)
        else:
            task_prompt_emb = None
            p_t = None

        hiddens = [getattr(area["integrator"], "hidden", None) for area in self.areas]
        logits, attentions = [], []
        zs_by_area = [[] for _ in range(self.n_areas)]
        cs_by_area = [[] for _ in range(self.n_areas)]
        ps_by_area = [[] for _ in range(self.n_areas)]

        for t in range(data.size(1)):
            data_t = data[:, t:t + 1, ...]
            if has_task:
                if p_t.ndim == 2:
                    p_t = p_t.unsqueeze(1)
                decoder_input_t = torch.cat((task_prompt_emb, p_t), dim=-1)
            else:
                decoder_input_t = None

            inp = data_t
            final_pred = None
            for area_idx, area in enumerate(self.areas):
                if area_idx == 0:
                    z_raw, a0_t = self._area0_encode(inp, decoder_input_t)
                    z_for_next = z_raw
                else:
                    z_raw = area["encoder"](inp)
                    z_for_next = z_raw

                z_t = self.postprocess(z_raw, *preprocess_args) if self.postprocess is not None else z_raw
                c_t, hiddens[area_idx] = self._run_integrator(area["integrator"], z_t, hiddens[area_idx])
                pred_t = area["predictor"](c_t)

                zs_by_area[area_idx].append(z_t)
                cs_by_area[area_idx].append(c_t)
                ps_by_area[area_idx].append(pred_t)
                if area_idx == self.readout_area:
                    logits.append(self.head(c_t))
                if area_idx == self.n_areas - 1:
                    final_pred = pred_t

                inp = z_for_next.detach() if z_for_next.ndim == 5 else z_for_next.detach().transpose(-2, -1)

            p_t = final_pred
            if return_attention:
                attentions.append(a0_t)

        logits_tensor = torch.cat(logits, dim=1)
        if return_logits_only and not return_attention:
            return logits_tensor
        if return_logits_only and return_attention:
            return logits_tensor, torch.cat(attentions, dim=1)

        z_list = [torch.cat(items, dim=1) for items in zs_by_area]
        c_list = [torch.cat(items, dim=1) for items in cs_by_area]
        p_list = [torch.cat(items, dim=1) for items in ps_by_area]
        if return_attention:
            return z_list, c_list, p_list, logits_tensor, torch.cat(attentions, dim=1)
        return z_list, c_list, p_list, logits_tensor
