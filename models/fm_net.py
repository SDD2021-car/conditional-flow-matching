import torch
from torch import nn
from torch.nn import functional as F


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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.time_proj(t_embed).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act2(h))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax((q @ k) / (c**0.5), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, c, h, w)
        out = self.proj(out)
        return h_in + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class FMNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (32, 16),
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, time_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        channels = base_channels
        skip_channels: list[int] = [base_channels]
        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(channels, out_channels, time_dim))
                channels = out_channels
                level_blocks.append(SelfAttention2d(channels))
                skip_channels.append(channels)
            self.down_blocks.append(level_blocks)
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(channels))
                skip_channels.append(channels)
        self.mid_block1 = ResBlock(channels, channels, time_dim)
        self.mid_attn = SelfAttention2d(channels)
        self.mid_block2 = ResBlock(channels, channels, time_dim)
        skip_channels_iter = iter(reversed(skip_channels))
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = next(skip_channels_iter)
                level_blocks.append(ResBlock(channels + skip_ch, out_channels, time_dim))
                channels = out_channels
                level_blocks.append(SelfAttention2d(channels))
            self.up_blocks.append(level_blocks)
            if level != 0:
                self.upsamples.append(Upsample(channels))
        self.norm_out = nn.GroupNorm(8, channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)
        self.attn_resolutions = set(attn_resolutions)

    def _maybe_attend(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(module, SelfAttention2d):
            if x.shape[-1] in self.attn_resolutions:
                return module(x)
            return x
        return module(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        h = self.conv_in(x)
        skips = [h]
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_embed)
                    skips.append(h)
                else:
                    h = self._maybe_attend(block, h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
                skips.append(h)
        # print("num skips saved:", len(skips))

        h = self.mid_block1(h, t_embed)
        h = self._maybe_attend(self.mid_attn, h)
        h = self.mid_block2(h, t_embed)
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, t_embed)
                else:
                    h = self._maybe_attend(block, h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)
        h = self.conv_out(self.act_out(self.norm_out(h)))
        return h
