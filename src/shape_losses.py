"""Scale-invariant raw-amplitude shape regularizers, independent of synthesis."""

import math

import torch


def log_lp_rms(amplitude, p=8.0, eps=1e-12):
    """Return [B,C] log(Lp/RMS) using spatially averaged nonnegative moments.

    Lp/RMS is one for a spatially constant nonzero field and grows as energy
    concentrates. Rescaling by a detached maximum is only a numerical device:
    it cancels algebraically between both moments and avoids overflow in a^p.
    All-zero fields return zero. Half-precision moment arithmetic uses float32.
    """
    if amplitude.ndim != 4 or min(amplitude.shape) < 1:
        raise ValueError("Expected nonempty [batch, planes, height, width] amplitudes")
    if not amplitude.is_floating_point():
        raise TypeError("Amplitudes must be floating point")
    if not math.isfinite(p) or p <= 2:
        raise ValueError("p must be finite and greater than 2")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    values = amplitude.float() if amplitude.dtype in (torch.float16, torch.bfloat16) else amplitude
    values = values.clamp_min(0)
    maximum = values.amax((-2, -1), keepdim=True)
    scaled = values / maximum.detach().clamp_min(torch.finfo(values.dtype).tiny)
    second = scaled.square().mean((-2, -1))
    higher = scaled.pow(p).mean((-2, -1))
    concentration = higher.clamp_min(eps).log() / p - second.clamp_min(eps).log() / 2
    return torch.where(maximum.squeeze(-1).squeeze(-1) > 0, concentration, torch.zeros_like(concentration))


def excess_concentration_loss(raw_amplitude, target, p=8.0, margin=0.0, eps=1e-12):
    """Return [B,C] squared excess log(Lp/RMS) relative to detached targets.

    A nonnegative log-space margin permits a concentration ratio exp(margin)
    above the target. Only excessive concentration is penalized; this is an
    auxiliary regularizer, not a substitute for reconstruction/shape metrics.
    Undefined zero-target profiles contribute zero. No phase labels are used.
    """
    if raw_amplitude.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    target = target.detach()
    prediction_shape = log_lp_rms(raw_amplitude, p, eps)
    target_shape = log_lp_rms(target, p, eps)
    excess = (prediction_shape - target_shape - margin).clamp_min(0).square()
    valid_target = target.clamp_min(0).amax((-2, -1)) > 0
    return torch.where(valid_target, excess, torch.zeros_like(excess))
