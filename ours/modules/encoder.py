import torch
import torch.nn as nn
import ours.modules.norms as norms

class Patchify(nn.Module):
    def __init__(self, kernelsize_stride: int = 4, d_model: int = 32, in_channels: int = 3):
        super().__init__()
        self.patch_size = kernelsize_stride
        self.n_channels = d_model
        self.stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=kernelsize_stride,
            stride=kernelsize_stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        spatial_kernel_size: int = 7,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        
        assert spatial_kernel_size % 2 == 1
        padding = spatial_kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=spatial_kernel_size,
                stride=1,
                padding=padding,
                groups=n_channels,
            ),
            norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim)),
            nn.Conv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            norms.GlobalResponseNorm(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
class TransitionConv(Patchify):
    def __init__(self, in_channels: int, down_ratio:int = 2):
        super().__init__(kernelsize_stride=down_ratio, d_model=in_channels*down_ratio, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)
    

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads

    # Can control where q, k, v come from (e.g. ff vs fb)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # input dimensions (B, C, H, W)
        # where we have H*W = N, the number of patches
        # C = d_model
        b, c, h, w = q.size()
        n = h * w

        # flatten spatial dimensions
        q = q.flatten(2).transpose(1, 2)  # (B, N, C)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        # split into heads
        q = q.reshape(q.size(0), q.size(1), self.n_heads, -1).transpose(1, 2)  # (B, n_heads, N, C/H)
        k = k.reshape(k.size(0), k.size(1), self.n_heads, -1).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.n_heads, -1).transpose(1, 2)

        # Perform scaled dot-product attention
        qkt = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_weights = torch.softmax(qkt, dim=-1)
        
        # Concatenate heads and return in (B, N, C) shape
        attn_output = torch.matmul(attn_weights, v) # (B, n_heads, N, C/H)   
        return attn_output.transpose(1, 2).reshape(b, n, -1)  # (B, N, C)
    

class MHA_MLP_block (nn.Module):
    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], n_heads: int = 4, norm_type: str = "rmsnorm"):
        super().__init__()
        self.mha = MultiheadAttention(n_heads=n_heads)
        self.norm1 = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.mlp = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 4, kernel_size=1),
            nn.GELU(),
            norms.GlobalResponseNorm(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=1),
        )
        self.norm2 = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # To decide what is what exactly
        x = q + self.mha(self.norm1(q), self.norm1(k), self.norm1(v))
        x = x + self.mlp(self.norm2(x))
        return x