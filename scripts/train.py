import argparse
import json
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from data.datasets import PairedImageDataset, UnpairedImageDataset
from losses.struct_loss import StructLossConfig
from models.fm_net import FMNet
from models.phi_dino import build_phi
from ot.matching import OTConfig
from train_step.trainer import LambdaSchedule, Trainer, TrainerConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def build_dataloader(cfg: dict):
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg["resolution"], cfg["resolution"])),
            transforms.ToTensor(),
        ]
    )
    if cfg["mode"] == "paired":
        dataset = PairedImageDataset(cfg["data_root"], transform=tfm)
        return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    dataset_a = UnpairedImageDataset(cfg["data_root"], "A", transform=tfm)
    dataset_b = UnpairedImageDataset(cfg["data_root"], "B", transform=tfm)
    loader_a = DataLoader(dataset_a, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    loader_b = DataLoader(dataset_b, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    return loader_a, loader_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = FMNet(in_channels=cfg["model"]["in_channels"], base_channels=cfg["model"]["base_channels"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"])

    phi = build_phi(
        backbone=cfg["phi"]["backbone"],
        model_name=cfg["phi"]["model_name"],
        layer=cfg["phi"].get("layer", -1),
        use_cls_token=cfg["phi"].get("use_cls_token", False),
    )

    struct_config = StructLossConfig(
        eps=cfg["struct"]["eps"],
        mask_type=cfg["struct"]["mask_type"],
        topk=cfg["struct"].get("topk", 8),
        tau=cfg["struct"].get("tau", 0.4),
        beta=cfg["struct"].get("beta", 0.07),
    )
    lambda_schedule = LambdaSchedule(
        lambda0=cfg["struct"]["lambda0"],
        p=cfg["struct"]["p"],
        lambda_min=cfg["struct"]["lambda_min"],
    )

    trainer_cfg = TrainerConfig(
        device=cfg["trainer"]["device"],
        amp=cfg["trainer"]["amp"],
        grad_clip=cfg["trainer"]["grad_clip"],
        log_every=cfg["trainer"]["log_every"],
        sample_every=cfg["trainer"]["sample_every"],
        output_dir=cfg["trainer"]["output_dir"],
        mode=cfg["mode"],
    )

    ot_config = None
    if cfg["mode"] == "unpaired":
        ot_config = OTConfig(
            eps_ot=cfg["ot"]["eps_ot"],
            iters_ot=cfg["ot"]["iters_ot"],
            match_type=cfg["ot"]["match_type"],
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        phi=phi,
        struct_config=struct_config,
        lambda_schedule=lambda_schedule,
        ot_config=ot_config,
        config=trainer_cfg,
    )

    if cfg["mode"] == "paired":
        loader = build_dataloader(cfg)
        trainer.fit(loader, num_steps=cfg["trainer"]["num_steps"])
    else:
        loader_a, loader_b = build_dataloader(cfg)
        trainer.fit(loader_a, dataloader_b=loader_b, num_steps=cfg["trainer"]["num_steps"])


if __name__ == "__main__":
    main()
