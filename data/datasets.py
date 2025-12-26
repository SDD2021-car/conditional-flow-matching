from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


def _default_loader(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


class PairedImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable = _default_loader,
    ) -> None:
        root_path = Path(root)
        self.a_paths = _list_images(root_path / "A")
        self.b_paths = _list_images(root_path / "B")
        if len(self.a_paths) != len(self.b_paths):
            raise ValueError("Paired dataset expects equal number of A and B images")
        self.transform = transform
        self.loader = loader

    def __len__(self) -> int:
        return len(self.a_paths)

    def __getitem__(self, index: int):
        img_a = self.loader(self.a_paths[index])
        img_b = self.loader(self.b_paths[index])
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        domain: str,
        transform: Optional[Callable] = None,
        loader: Callable = _default_loader,
    ) -> None:
        root_path = Path(root)
        self.paths = _list_images(root_path / domain)
        self.transform = transform
        self.loader = loader

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self.loader(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img
