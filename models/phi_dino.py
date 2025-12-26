from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn


@dataclass
class DinoConfig:
    backbone: Literal["dinov2", "dinov3"] = "dinov2"
    model_name: str = "dinov2_vits14"
    layer: int = -1
    use_cls_token: bool = False


class DinoFeatureExtractor(nn.Module):
    def __init__(self, config: DinoConfig):
        super().__init__()
        self.config = config
        if config.backbone == "dinov3":
            repo = "facebookresearch/dinov3"
        else:
            repo = "facebookresearch/dinov2"
        self.model = torch.hub.load(repo, config.model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_intermediate_layers"):
            n_layers = abs(self.config.layer) if self.config.layer < 0 else 1
            layers = self.model.get_intermediate_layers(
                x, n=n_layers, return_class_token=self.config.use_cls_token
            )
            if self.config.layer < 0:
                tokens = layers[self.config.layer]
            else:
                tokens = layers[-1]
            if self.config.use_cls_token:
                if tokens.dim() == 2:
                    tokens = tokens[:, None, :]
            return tokens
        output = self.model(x)
        if output.dim() == 2:
            output = output[:, None, :]
        return output

    @torch.no_grad()
    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward(x)
        if tokens.dim() == 2:
            return tokens
        return tokens.mean(dim=1)


def build_phi(
    backbone: Literal["dinov2", "dinov3"] = "dinov2",
    model_name: str = "dinov2_vits14",
    layer: int = -1,
    use_cls_token: bool = False,
) -> DinoFeatureExtractor:
    config = DinoConfig(
        backbone=backbone,
        model_name=model_name,
        layer=layer,
        use_cls_token=use_cls_token,
    )
    return DinoFeatureExtractor(config)
