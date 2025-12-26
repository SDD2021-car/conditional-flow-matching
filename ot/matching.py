from dataclasses import dataclass
from typing import Literal

import torch

from ot.sinkhorn import sinkhorn


@dataclass
class OTConfig:
    eps_ot: float = 0.05
    iters_ot: int = 50
    match_type: Literal["argmax", "sample"] = "argmax"


def match_batches(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    config: OTConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = feats_a.shape[0]
    cost = torch.cdist(feats_a, feats_b, p=2.0) ** 2
    cost = cost.unsqueeze(0)
    plan = sinkhorn(cost, eps=config.eps_ot, iters=config.iters_ot)[0]

    if config.match_type == "sample":
        idx_b = torch.multinomial(plan, num_samples=1).squeeze(-1)
    else:
        idx_b = torch.argmax(plan, dim=-1)

    idx_a = torch.arange(batch, device=feats_a.device)
    return idx_a, idx_b
