import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .config import SIZE


def load_inputs(paths) -> torch.Tensor:
    if len(paths) != 4:
        raise ValueError("Exactly four input PNGs required")
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (SIZE, SIZE):
                raise ValueError(f"{path}: expected grayscale 8-bit PNG {SIZE}x{SIZE}, got {image.format} {image.mode} {image.size}")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"{path}: animated PNG is unsupported")
            array = np.asarray(image, dtype=np.float32).copy() / 255.
        tensors.append(torch.from_numpy(array))
    result = torch.stack(tensors)[None]
    return result / result.amax((-2, -1), keepdim=True).clamp_min(1e-8)


def phase_to_png(phase: torch.Tensor, path) -> None:
    if phase.shape != (SIZE, SIZE):
        raise ValueError(f"Phase must have shape ({SIZE}, {SIZE})")
    if not torch.isfinite(phase).all():
        raise ValueError("Phase contains NaN or Inf")
    phase = torch.atan2(phase.detach().float().sin(), phase.detach().float().cos())
    array = torch.round((phase + math.pi) * 255 / (2 * math.pi)).clamp(0, 255).byte().cpu().numpy()
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Output path must end with .png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def phase_from_png(path, device="cpu") -> torch.Tensor:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (SIZE, SIZE):
            raise ValueError(f"{path}: expected grayscale 8-bit phase PNG {SIZE}x{SIZE}")
        array = np.asarray(image, dtype=np.float32).copy()
    return (torch.from_numpy(array).to(device) * (2 * math.pi / 255) - math.pi)[None, None]
