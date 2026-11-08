"""Auxiliary image losses; no change to optical synthesis or evaluation."""

import math

import torch


def _check_amplitudes(reconstruction, target):
    if reconstruction.ndim != 4 or reconstruction.shape != target.shape:
        raise ValueError("Expected matching [batch, planes, height, width] images")
    if min(reconstruction.shape) < 1:
        raise ValueError("Image dimensions must be nonempty")
    if not reconstruction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Amplitude losses require floating-point images")


def amplitude_cosine_loss(raw_amplitude, target, eps=1e-8):
    """Return [B,C] uncentered cosine losses for nonnegative amplitudes.

    Positive scaling of either nonzero image does not change the loss above
    the numerical norm floor. Zero targets have undefined direction and make
    no contribution. A zero prediction with a nonzero target has loss one
    and a finite useful gradient. Targets are never optimized by this loss.
    """
    _check_amplitudes(raw_amplitude, target)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    prediction = raw_amplitude.flatten(2)
    target = target.detach().flatten(2)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    prediction_unit = prediction / prediction_norm.clamp_min(eps)
    target_unit = target / target_norm.clamp_min(eps)
    cosine = (prediction_unit * target_unit).sum(-1).clamp(-1, 1)
    return torch.where(target_norm.squeeze(-1) > 0, 1 - cosine, torch.zeros_like(cosine))


def balanced_pixel_mse(reconstruction, target, threshold=0.1):
    """Return [B,C] MSE with equal weights for present foreground/background.

    Foreground is defined by detached target > threshold. If one region is
    absent, the remaining region receives all weight, avoiding an artificial
    factor of one half for completely black or completely foreground images.
    """
    _check_amplitudes(reconstruction, target)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    target = target.detach()
    squared = (reconstruction - target).square()
    foreground = target > threshold
    background = ~foreground
    foreground_count = foreground.sum((-2, -1))
    background_count = background.sum((-2, -1))
    foreground_mse = (squared * foreground).sum((-2, -1)) / foreground_count.clamp_min(1)
    background_mse = (squared * background).sum((-2, -1)) / background_count.clamp_min(1)
    present = (foreground_count > 0).to(squared.dtype) + (background_count > 0).to(squared.dtype)
    return (foreground_mse + background_mse) / present.clamp_min(1)
