from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from .config import ARCHITECTURE, DEFAULT_CHECKPOINT, SIZE
from .physics import physics_seed


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Checkpoint architecture must be {ARCHITECTURE}")
    state = saved.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a model state_dict")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint contains invalid tensors")
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=8, classes=2, activation=None
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {key: value for key, value in saved.items() if key != "model"}


def phase_from_ready_unet(model, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 4 or targets.shape[0] == 0 or targets.shape[1:] != (4, SIZE, SIZE):
        raise ValueError(f"Targets must have shape (B, 4, {SIZE}, {SIZE})")
    if not targets.is_floating_point() or not torch.isfinite(targets).all():
        raise ValueError("Targets must contain finite floating-point amplitudes")
    seed = physics_seed(targets)
    coord = torch.linspace(-1, 1, SIZE, device=targets.device, dtype=torch.float32)
    ygrid, xgrid = torch.meshgrid(coord, coord, indexing="ij")
    xy = torch.stack((xgrid, ygrid), dim=0)[None]
    img_features = (targets.float() - 0.45) / 0.225
    inputs = torch.cat((img_features, seed, xy.expand(len(targets), -1, -1, -1)), dim=1)
    with torch.autocast(device_type=targets.device.type, enabled=targets.device.type == "cuda"):
        delta = model(inputs)
    field = 0.7 * seed + delta.float()
    phase = torch.atan2(field[:, 1:2], field[:, 0:1])
    if not torch.isfinite(phase).all():
        raise FloatingPointError("Model produced NaN or Inf")
    return phase
