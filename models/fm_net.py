import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        if t.dim() == 1:
            t = t[:, None]
        return self.mlp(t)


class FMNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.conv_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.time_to_scale = nn.Linear(base_channels, base_channels)
        self.time_to_shift = nn.Linear(base_channels, base_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        t_embed = self.time_embed(t)
        scale = self.time_to_scale(t_embed)[:, :, None, None]
        shift = self.time_to_shift(t_embed)[:, :, None, None]
        h = h * (1 + scale) + shift
        h = self.block1(h)
        h = self.block2(h)
        return self.conv_out(h)
