import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models.fm_net import FMNet


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/NAS_data/yjy/SEN1-2-BIT_matched/Test/A", required=False)
    parser.add_argument("--output_dir", type=str, default="/data/yjy_data/conditional-flow-matching/outputs_change_struct_loss/result_10000", required=False)
    parser.add_argument("--checkpoint", type=str, default="/data/yjy_data/conditional-flow-matching/outputs_change_struct_loss/checkpoints/ckpt_010000.pt", required=False)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--base_channels", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = FMNet(in_channels=args.in_channels, base_channels=args.base_channels)
    load_checkpoint(args.checkpoint, model, device)
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
        ]
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            x_t = x
            for i in range(args.steps):
                t = torch.full((x.size(0),), float(i) / args.steps, device=device)
                v = model(x_t, t)
                x_t = x_t + (1.0 / args.steps) * v
            x_out = torch.clamp(x_t, 0.0, 1.0)
            save_image(x_out, output_dir / path.name)


if __name__ == "__main__":
    main()
