import torch
import torch.nn as nn


class RePLModel(nn.Module):
    """Temporal joint-embedding model based on CPC (https://arxiv.org/pdf/1807.03748.pdf)

    Args:
        encoder (nn.Module): Encoder network
        integrator (nn.Module): Autoregressive network
        preprocess (nn.Module): Optional preprocessing network. Default: None
        postprocess (nn.Module): Optional postprocessing network. Default: None
        predictor (nn.Module): Optional predictor network. Default: nn.Identity()
        linear_readout (nn.Module): Optional linear readout network. Default: None

    Attributes:
        encoder (nn.Module): Encoder network
        integrator (nn.Module): Autoregressive network
        preprocess (nn.Module): Preprocessing network
        postprocess (nn.Module): Postprocessing network
        predictor (nn.Module): Predictor network
        linear_readout (nn.Module): Linear readout network

    Notes:
        The `forward` method gets a data tensor as input which can be in the form of
        - (B, C, L) for 1D signal like raw audio
        - (B, L, C, H, W) for video.
        The differences arise because convolutional layers expect the channel dimension to be the second dimension.
        Whereas RNNs expect the channel dimension to be the last dimension.

        The `forward` method outputs the following:
        - context_tensor: Encoded representation from the integrator network
        - encoded_data: Encoded representation from the encoder network
        - pred: Predictions from the predictor network

        All outputs are in the form of (B, L, C) or (B, L, C, H, W), where:
        - B: Batch size
        - L: length of the sequence
        - C: Number of channels or features
        - H: Height of the image
        - W: Width of the image
        Also note that (B, L, C, H, W) is only valid for visual data.
    """

    def __init__(
        self,
        encoder,
        integrator,
        preprocess=None,
        postprocess=None,
        predictor=nn.Identity(),
    ):
        super(RePLModel, self).__init__()
        self.preprocess = preprocess
        self.encoder = encoder
        self.integrator = integrator
        self.predictor = predictor
        self.postprocess = postprocess

    def forward(self, data):
        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        z = self.encoder(data)  # (B, L, C) or (B, L, C, H, W)

        if self.postprocess is not None:
            z = self.postprocess(z, *preprocess_args)

        context_tensor = self.integrator(z)  # (B, L, C) or (B, L, C, H, W)

        pred = self.predictor(context_tensor)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)

        return z, context_tensor, pred


class HierarchicalRePLModel(nn.Module):
    """Hierarchical RePL model for multi-area processing.
    Args:
        encoder_list (list): List of encoder networks for each area
        integrator_list (list): List of integrator networks for each area
        predictor_list (list): List of predictor networks for each area
        preprocess (nn.Module): Optional preprocessing network. Default: None
        postprocess (nn.Module): Optional postprocessing network. Default: None
        integrator_FB_path (bool): Flag to use feedback path in integrator. Default: False
    Attributes:
        areas (nn.ModuleList): List of areas, each containing encoder, integrator, and predictor networks
        n_areas (int): Number of areas in the model
    Notes:
        The `forward` method processes the input data through each area sequentially.
        Each area consists of an encoder, an integrator, and a predictor.
        The output includes the encoded representations, context tensors, and predictions from each area.

        The `forward` method gets a data tensor as input which can be in the form of
        - (B, C, L) for 1D signal like raw audio
        - (B, L, C, H, W) for video.
        The differences arise because convolutional layers expect the channel dimension to be the second dimension.
        Whereas RNNs expect the channel dimension to be the last dimension.
        This dimensionality standard is enforced for each area in the model.

        The `forward` method outputs the following:
        - z_list: List of encoded representations from each area
        - ctx_list: List of context tensors from each area
        - pred_list: List of predictions from each area
        All outputs are in the form of (B, L, C) or (B, L, C, H, W), where:
        - B: Batch size
        - L: length of the sequence
        - C: Number of channels or features
        - H: Height of the image
        - W: Width of the image
        Also note that (B, L, C, H, W) is only valid for visual data.

        The feedback path is not used in the current implementation but will be added in the future.
    """
    def __init__(
        self,
        encoder_list,
        integrator_list,
        predictor_list,
        preprocess=None,
        postprocess=None,
        integrator_FB_path=False,
    ):
        super(HierarchicalRePLModel, self).__init__()
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.integrator_FB_path = integrator_FB_path  # Not used in the current implementation

        self.areas = nn.ModuleList()
        for encoder, integrator, predictor in zip(encoder_list, integrator_list, predictor_list):
            self.areas.append(
                nn.ModuleDict(
                    {
                        "encoder": encoder,
                        "integrator": integrator,
                        "predictor": predictor,
                    }
                )
            )
        self.n_areas = len(self.areas)

    def forward(self, data):
        # preprocess data if available
        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        z_list = []  # list of z's from each block for later loss computation
        ctx_list = []  # list of ctx's from each block for later loss computation
        pred_list = []  # list of predictions from each block for later loss computation

        inp = data  # input to the first block

        for i, area in enumerate(self.areas):
            # Block i's encoder gets input from detached block i-1's z
            z = area["encoder"](inp)  # (B, L, C) or (B, L, C, H, W)

            # apply postprocessing if available (only for inter-block path (and hence the block loss) and not hierarchical FF path)
            if self.postprocess is not None:
                z_processed = self.postprocess(z, *preprocess_args)
            else:
                z_processed = z

            # Block i's integrator gets input from block i's encoder
            ctx = area["integrator"](z_processed)  # (B, L, C) or (B, L, C, H, W)

            # Block i's predictor gets input from block i's ctx
            pred = area["predictor"](ctx)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)

            z_list.append(z_processed)
            ctx_list.append(ctx)
            pred_list.append(pred)

            # Gradient doesn't flow through the instantaneous FF path
            if z.ndim == 5:
                inp = z.detach()
            else:
                inp = z.detach().transpose(-2, -1)  # (B, L, C) -> (B, C, L)

        return z_list, ctx_list, pred_list


class RePLSeq(nn.Module):
    """
    Sequential RePL model for processing sequences one timestep at a time.
    """
    def __init__(
        self,
        encoder,
        integrator,
        preprocess=None,
        postprocess=None,
        predictor=nn.Identity(),
        freeze_repl=True,
        eval_frozen=True,
    ):
        super(RePLSeq, self).__init__()
        self.preprocess = preprocess
        self.encoder = encoder
        self.integrator = integrator
        self.predictor = predictor
        self.postprocess = postprocess

        if freeze_repl:
            for module in (self.encoder, self.integrator, self.predictor):
                for param in module.parameters():
                    param.requires_grad_(False)
                if eval_frozen:
                    module.eval()

    def forward(self, data):
        L = data.size(1) # sequence length

        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        zs, cs, ps = [], [], []  # lists to store outputs at each timestep
        hidden = getattr(self.integrator, "hidden", None)

        for t in range(L):
            # Get the t-th timestep's data
            t_data = data[:, t:t+1, ...]  # (B, 1, C) or (B, 1, C, H, W)

            z_t, c_t, p_t, hidden = self.forward_step(t_data, hidden)  # process the t-th timestep
            zs.append(z_t)
            cs.append(c_t)
            ps.append(p_t)

        # stack outputs from all timesteps
        z = torch.cat(zs, dim=1)  # (B, L, C) or (B, L, C, H, W)
        context_tensor = torch.cat(cs, dim=1)  # (B, L, C) or (B, L, C, H, W)
        pred = torch.cat(ps, dim=1)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)
        return z, context_tensor, pred

    def forward_step(self, t_data, hidden=None):
        z_t = self.encoder(t_data)  # (B, 1, C) or (B, 1, C, H, W)
        if self.postprocess is not None:
            z_t = self.postprocess(z_t)

        backbone = getattr(self.integrator, "backbone")
        context_tensor_t, hidden = backbone(z_t, hidden)

        pred_t = self.predictor(context_tensor_t)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)
        return z_t, context_tensor_t, pred_t, hidden


# added for harpl to enable layerwise modulation
class RePLModelExposed(nn.Module):
    """
    Functionally identical implementation to RePLModel, but exposes the activations
        at all layers of encoder, integrator and predictor for layerwise modulation.
    Furthermore, the forward pass procedes sequentially over timesteps instead of processing the whole sequence at once.
        This allows for dynamic & flexible attentional modulation.
    """

    def __init__(
        self,
        encoder,
        integrator,
        preprocess=None,
        postprocess=None,
        predictor=nn.Identity(),
        freeze_repl=True,
        eval_frozen=True,
        compile_model=False,
        compile_mode=None,
        batch_static_layers=False,
    ):
        super(RePLModelExposed, self).__init__()
        self.preprocess = preprocess
        self.encoder = encoder
        self.integrator = integrator
        self.predictor = predictor
        self.postprocess = postprocess
        self.layer_activations = {}
        self.compile_model = compile_model
        self.batch_static_layers = batch_static_layers

        if freeze_repl:
            for module in (self.encoder, self.integrator, self.predictor):
                for param in module.parameters():
                    param.requires_grad_(False)
                if eval_frozen:
                    module.eval()

        self._compiled_forward = None
        if compile_model and hasattr(torch, "compile"):
            self._compiled_forward = torch.compile(self._forward_impl, mode=compile_mode)

    @staticmethod
    def _sequence_dim(x):
        return -1 if x.ndim == 3 else 1

    @staticmethod
    def _stack_steps(xs, dim=1):
        return torch.cat(xs, dim=dim).contiguous()

    def _stack_context_steps(self, xs):
        context = torch.cat(xs, dim=1)
        if isinstance(getattr(self.integrator, "backbone", None), nn.RNNBase):
            context = context.transpose(0, 1).contiguous().transpose(0, 1)
        return context

    @staticmethod
    def _stack_layer_outputs(xs):
        if len(xs) == 1:
            return xs[0].contiguous()
        sample = xs[0]
        for dim, size in enumerate(sample.shape):
            if size == 1:
                return torch.cat(xs, dim=dim).contiguous()
        return torch.stack(xs, dim=1).contiguous()

    @staticmethod
    def _record(activations, name, x):
        activations.setdefault(name, []).append(x)

    def _forward_layers(self, layers, prefix, x, activations):
        for name, layer in layers.named_children():
            x = layer(x)
            self._record(activations, f"{prefix}.{name}", x)
        return x

    def _encoder_step(self, x, activations):
        layers = getattr(self.encoder, "layers", None)
        if layers is not None:
            if x.ndim == 5:
                x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self._forward_layers(layers, "encoder.layers", x, activations)
            return x.transpose(1, 2).contiguous()

        backbone = getattr(self.encoder, "backbone", None)
        if backbone is None:
            z = self.encoder(x)
            self._record(activations, "encoder", z)
            return z

        backbone_layers = getattr(backbone, "layers", None)
        if backbone_layers is not None:
            if x.ndim == 5:
                x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self._forward_layers(backbone_layers, "encoder.backbone.layers", x, activations)
            return x.transpose(1, 2).contiguous()

        if x.ndim == 3:
            z = self._forward_layers(backbone, "encoder.backbone", x.unsqueeze(1), activations).mean(-2)
            return z.transpose(1, 2).contiguous()

        is_video = x.ndim == 5
        if is_video:
            seq_len = x.size(1)
            x = x.reshape(-1, *x.shape[2:])
        z = self._forward_layers(backbone, "encoder.backbone", x, activations)
        if is_video:
            z = z.reshape(-1, seq_len, z.size(1), z.size(2), z.size(3))
        if getattr(self.encoder, "return_full_feature_map", False):
            return z
        return z.mean(-1).mean(-1)

    def _encoder_full(self, x, activations):
        layers = getattr(self.encoder, "layers", None)
        if layers is not None:
            if x.ndim == 5:
                x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self._forward_layers(layers, "encoder.layers", x, activations)
            return x.transpose(1, 2).contiguous()

        backbone = getattr(self.encoder, "backbone", None)
        if backbone is None:
            z = self.encoder(x)
            self._record(activations, "encoder", z)
            return z

        backbone_layers = getattr(backbone, "layers", None)
        if backbone_layers is not None:
            if x.ndim == 5:
                x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self._forward_layers(backbone_layers, "encoder.backbone.layers", x, activations)
            return x.transpose(1, 2).contiguous()

        if x.ndim == 3:
            z = self._forward_layers(backbone, "encoder.backbone", x.unsqueeze(1), activations).mean(-2)
            return z.transpose(1, 2).contiguous()

        is_video = x.ndim == 5
        if is_video:
            seq_len = x.size(1)
            x = x.reshape(-1, *x.shape[2:])
        z = self._forward_layers(backbone, "encoder.backbone", x, activations)
        if is_video:
            z = z.reshape(-1, seq_len, z.size(1), z.size(2), z.size(3))
        if getattr(self.encoder, "return_full_feature_map", False):
            return z
        return z.mean(-1).mean(-1)

    def _predictor_step(self, x, activations):
        layers = getattr(self.predictor, "layers", None)
        if layers is None:
            pred = self.predictor(x)
            self._record(activations, "predictor", pred)
            return pred
        return self._forward_layers(layers, "predictor.layers", x, activations)

    def _predictor_full(self, x, activations):
        return self._predictor_step(x, activations)

    def _integrator_step(self, x, hidden, activations):
        backbone = getattr(self.integrator, "backbone", None)
        if isinstance(backbone, nn.RNNBase):
            if getattr(self.integrator, "reverse", False):
                x = torch.flip(x, dims=[1])
            out, hidden = backbone(x, hidden)
            self._record(activations, "integrator.backbone", out)
            if getattr(self.integrator, "reverse", False):
                out = torch.flip(out, dims=[1])
            return out, hidden
        out = self.integrator(x)
        self._record(activations, "integrator", out)
        return out, hidden

    def forward(self, data, return_activations=False):
        if self._compiled_forward is not None:
            return self._compiled_forward(data, return_activations)
        return self._forward_impl(data, return_activations)

    def _forward_impl(self, data, return_activations=False):
        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        activations = {}
        ctx_steps = []
        hidden = getattr(self.integrator, "hidden", None)
        if self.batch_static_layers:
            z = self._encoder_full(data, activations)
            if self.postprocess is not None:
                z = self.postprocess(z, *preprocess_args)

            for t in range(z.size(1)):
                ctx_t, hidden = self._integrator_step(z.narrow(1, t, 1), hidden, activations)
                ctx_steps.append(ctx_t)

            context_tensor = self._stack_context_steps(ctx_steps)
            pred = self._predictor_full(context_tensor, activations)
        else:
            z_steps = []
            pred_steps = []
            seq_dim = self._sequence_dim(data)

            for t in range(data.size(seq_dim)):
                data_t = data.narrow(seq_dim, t, 1)
                z_t = self._encoder_step(data_t, activations)

                if self.postprocess is not None:
                    z_t = self.postprocess(z_t, *preprocess_args)

                ctx_t, hidden = self._integrator_step(z_t, hidden, activations)
                pred_t = self._predictor_step(ctx_t, activations)

                z_steps.append(z_t)
                ctx_steps.append(ctx_t)
                pred_steps.append(pred_t)

            z = self._stack_steps(z_steps, dim=1)
            context_tensor = self._stack_context_steps(ctx_steps)
            pred = self._stack_steps(pred_steps, dim=1)

        if hasattr(self.integrator, "save_hidden"):
            self.integrator.save_hidden(hidden)

        self.layer_activations = {
            name: self._stack_layer_outputs(outputs)
            for name, outputs in activations.items()
        }

        if return_activations:
            return z, context_tensor, pred, self.layer_activations
        return z, context_tensor, pred
