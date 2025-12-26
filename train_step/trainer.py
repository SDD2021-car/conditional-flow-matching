from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from losses.struct_loss import StructLoss, StructLossConfig
from ot.matching import OTConfig, match_batches


@dataclass
class LambdaSchedule:
    lambda0: float = 1.0
    p: float = 2.0
    lambda_min: float = 0.1

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.lambda0 * (1 - t) ** self.p + self.lambda_min


@dataclass
class TrainerConfig:
    device: str = "cuda"
    amp: bool = True
    grad_clip: float = 1.0
    log_every: int = 50
    sample_every: int = 500
    output_dir: str = "outputs"
    mode: str = "paired"
    checkpoint_every: int = 10000

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        phi: nn.Module,
        struct_config: StructLossConfig,
        lambda_schedule: LambdaSchedule,
        ot_config: Optional[OTConfig] = None,
        psi: Optional[nn.Module] = None,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.phi = phi
        self.struct_loss = StructLoss(phi, struct_config)
        self.lambda_schedule = lambda_schedule
        self.ot_config = ot_config
        self.psi = psi or phi
        self.config = config or TrainerConfig()
        self.scaler = GradScaler(enabled=self.config.amp)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_samples(
        self,
        step: int,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> None:
        grid = torch.cat([x_a, x_pred, x_b], dim=0)
        save_image(grid, self.output_dir / f"samples_{step:06d}.png", nrow=x_a.size(0))

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint_dir / f"ckpt_{step:06d}.pt",
        )

    def _paired_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return batch

    def _unpaired_step(
        self,
        batch_a: torch.Tensor,
        batch_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feats_a = self.psi.pooled(batch_a)
            feats_b = self.psi.pooled(batch_b)
            idx_a, idx_b = match_batches(feats_a, feats_b, self.ot_config)
        return batch_a[idx_a], batch_b[idx_b]

    def fit(
        self,
        dataloader_a: DataLoader,
        dataloader_b: Optional[DataLoader] = None,
        num_steps: int = 10000,
    ) -> None:
        device = torch.device(self.config.device)
        self.model.to(device)
        self.phi.to(device)
        self.psi.to(device)
        self.model.train()
        self.phi.eval()
        self.psi.eval()

        if self.config.mode == "unpaired":
            if dataloader_b is None:
                raise ValueError("Unpaired mode requires dataloader_b")
            iter_a = itertools.cycle(dataloader_a)
            iter_b = itertools.cycle(dataloader_b)
        else:
            iter_a = iter(itertools.cycle(dataloader_a))
            iter_b = None

        for step in range(1, num_steps + 1):
            if self.config.mode == "unpaired":
                batch_a = next(iter_a)
                batch_b = next(iter_b)
                x_a, x_b = self._unpaired_step(batch_a.to(device), batch_b.to(device))
            else:
                batch = next(iter_a)
                x_a, x_b = self._paired_step((batch[0].to(device), batch[1].to(device)))

            t = torch.rand(x_a.size(0), device=device)
            t_view = t.view(-1, 1, 1, 1)
            x_t = (1 - t_view) * x_a + t_view * x_b
            v_tgt = x_b - x_a

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.config.amp):
                v_pred = self.model(x_t, t)
                loss_fm = torch.mean((v_pred - v_tgt) ** 2)

                # Insert L_struct here
                loss_struct = self.struct_loss(x_t, v_pred)
                lambda_t = self.lambda_schedule(t).mean()
                loss = loss_fm + lambda_t * loss_struct

            self.scaler.scale(loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if step % self.config.log_every == 0:
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"fm={loss_fm.item():.4f} struct={loss_struct.item():.4f}"
                )

            if step % self.config.sample_every == 0:
                with torch.no_grad():
                    x_pred = x_t + (1 - t_view) * v_pred
                self._save_samples(step, x_a, x_b, x_pred)

            if self.config.checkpoint_every > 0 and step % self.config.checkpoint_every == 0:
                self._save_checkpoint(step)