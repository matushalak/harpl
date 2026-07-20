import torch
import torch.nn as nn

# Wrapper around 'layer-norm' type normalization (layer & rms)
class DenseNorm2d(nn.Module):
    def __init__(self, norm_type: str = "rmsnorm", *args, **kwargs):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(*args, **kwargs)
        elif norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

# GRN
class GlobalResponseNorm(nn.Module):
    """
    Global response normalization (GRN) as used in ConvNeXtV2.
        GRN normalizes each channel by the global L2 norm across spatial dimensions, 
        with learnable scaling and bias.
    
    Assumes (B, C, H, W) input shape.
    """

    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1, 1)
                                #   *1e-3
                                  )
        self.beta = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor | None = None) -> torch.Tensor:
        if keep_mask is None:
            mask = 1.0
        else:
            mask = keep_mask.to(dtype=x.dtype)

        # L2 norm "pooling" across spatial dimensions.
        gx = torch.sqrt((x.pow(2) * mask).sum(dim=[2, 3], keepdim=True)+self.eps)
        # Competition across channels.
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        # Apply scaling and bias
        return self.gamma * (x * nx) + self.beta + x
    