import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, target_size: tuple[int, int], n_channels: int):
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size = self.target_size, mode="bilinear", align_corners=True)