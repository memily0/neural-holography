from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .model import phase_from_ready_unet
from .physics import norm_amp, reconstruct_four
from .preprocessing import load_inputs, phase_from_png, phase_to_png


def select_device(name="auto") -> torch.device:
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    return torch.device(name)


@torch.inference_mode()
def predict(paths, output, model, device="cpu", reconstruction_dir=None) -> None:
    output = Path(output)
    input_paths = {Path(path).resolve() for path in paths}
    if output.resolve() in input_paths:
        raise ValueError("Output must differ from input paths")
    targets = load_inputs(paths).to(device)
    phase = phase_from_ready_unet(model, targets)
    phase_to_png(phase[0, 0], output)
    if reconstruction_dir is not None:
        directory = Path(reconstruction_dir)
        reconstruction_paths = [directory / f"reconstructed_{i + 1}.png" for i in range(4)]
        if any(path.resolve() in input_paths or path.resolve() == output.resolve() for path in reconstruction_paths):
            raise ValueError("Reconstruction paths must differ from input and output paths")
        restored = norm_amp(reconstruct_four(phase_from_png(output, device)))
        if not torch.isfinite(restored).all():
            raise FloatingPointError("Fresnel reconstruction contains NaN or Inf")
        directory.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(reconstruction_paths):
            pixels = (restored[0, i].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(path, format="PNG")
