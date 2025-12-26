from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DinoConfig:
    backbone: Literal["dinov2", "dinov3"] = "dinov2"
    model_name: str = "dinov2_vits14"
    layer: int = -1
    use_cls_token: bool = False
    pad_to_patch: bool = True
    pad_mode: str = "constant"


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

    def _get_patch_size(self) -> Optional[tuple[int, int]]:
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is None:
            return None
        patch_size = getattr(patch_embed, "patch_size", None)
        if patch_size is None:
            return None
        if isinstance(patch_size, tuple):
            return patch_size
        return (int(patch_size), int(patch_size))

    def _pad_to_patch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.pad_to_patch:
            return x
        patch_size = self._get_patch_size()
        if patch_size is None:
            return x
        _, _, height, width = x.shape
        pad_h = (patch_size[0] - height % patch_size[0]) % patch_size[0]
        pad_w = (patch_size[1] - width % patch_size[1]) % patch_size[1]
        if pad_h == 0 and pad_w == 0:
            return x
        padding = (0, pad_w, 0, pad_h)
        return F.pad(x, padding, mode=self.config.pad_mode, value=0.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_patch(x)
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
    pad_to_patch: bool = True,
    pad_mode: str = "constant",
) -> DinoFeatureExtractor:
    config = DinoConfig(
        backbone=backbone,
        model_name=model_name,
        layer=layer,
        use_cls_token=use_cls_token,
        pad_to_patch=pad_to_patch,
        pad_mode=pad_mode,
    )
    return DinoFeatureExtractor(config)
