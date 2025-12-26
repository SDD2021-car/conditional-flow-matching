from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class StructLossConfig:
    eps: float = 0.01
    mask_type: Literal["topk", "soft"] = "topk"
    topk: int = 8
    tau: float = 0.4
    beta: float = 0.07


def rms_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, v.dim()))
    rms = torch.sqrt(torch.mean(v**2, dim=dims, keepdim=True) + eps)
    return v / rms


def _self_similarity(tokens: torch.Tensor) -> torch.Tensor:
    tokens_hat = tokens / (torch.norm(tokens, dim=-1, keepdim=True) + 1e-6)
    return tokens_hat @ tokens_hat.transpose(-1, -2)


def _topk_mask(sim: torch.Tensor, topk: int) -> torch.Tensor:
    batch, n, _ = sim.shape
    sim_no_diag = sim.clone()
    diag = torch.eye(n, device=sim.device, dtype=sim.dtype).bool()
    sim_no_diag[:, diag] = -torch.inf
    _, indices = torch.topk(sim_no_diag, k=min(topk, n - 1), dim=-1)
    mask = torch.zeros_like(sim)
    mask.scatter_(-1, indices, 1.0)
    return mask


def _soft_mask(sim: torch.Tensor, tau: float, beta: float) -> torch.Tensor:
    return torch.sigmoid((sim - tau) / beta)


class StructLoss(nn.Module):
    def __init__(self, phi: nn.Module, config: StructLossConfig):
        super().__init__()
        self.phi = phi
        self.config = config

    def forward(self, x_t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        v_norm = rms_normalize(v_pred)
        x_probe = x_t + self.config.eps * v_norm

        tokens_t = self.phi(x_t)
        tokens_probe = self.phi(x_probe)
        sim_t = _self_similarity(tokens_t)
        sim_probe = _self_similarity(tokens_probe)

        if self.config.mask_type == "soft":
            mask = _soft_mask(sim_t, self.config.tau, self.config.beta)
        else:
            mask = _topk_mask(sim_t, self.config.topk)

        diff = mask * (sim_probe - sim_t)
        loss = torch.mean(torch.sum(diff**2, dim=(1, 2)))
        return loss
