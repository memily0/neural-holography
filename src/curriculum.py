import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def sample_forward_targets(batch_size, propagator, scales, generator):
    """Generate amplitudes by forward physics; sampled phases are never labels."""
    device = scales.device
    fields = []
    for size in (8, 32, 128):
        field = torch.randn(batch_size, 1, size, size, generator=generator, device=device)
        fields.append(F.interpolate(field, size=(128, 128), mode="bilinear", align_corners=False))
    weights = torch.rand(batch_size, 3, 1, 1, generator=generator, device=device)
    weights = weights / weights.sum(1, keepdim=True)
    phase = math.pi * torch.tanh(2 * (torch.cat(fields, dim=1) * weights).sum(1, keepdim=True))
    raw = propagator.amplitude(phase) * scales
    target = (raw.clamp(0, 1) * 255).round() / 255
    stats = {
        "synthetic_clipped_fraction": float((raw > 1).float().mean()),
        "synthetic_png_mse": float((raw - target).square().mean()),
        "synthetic_target_rms": float(target.square().mean().sqrt()),
    }
    return target, stats
