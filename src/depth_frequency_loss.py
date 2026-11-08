"""Training-only high-pass differences between RMS-normalized depth planes."""

import math

import torch
from torch.nn import functional as F


def depth_frequency_loss(raw_amplitude, target, window_size=5, eps=1e-6):
    """Compare fine amplitude differences across four planes on valid windows.

    Normalize each image by its own RMS, subtract the mean across planes,
    and apply H(x)=center_crop(x)-valid_avgpool(x). Return mean squared error
    over batch, planes and the common valid spatial region. Targets are
    detached. Scale invariance holds above the numerical RMS floor.
    """
    if raw_amplitude.ndim != 4 or raw_amplitude.shape != target.shape or raw_amplitude.shape[1] != 4:
        raise ValueError("Expected matching [batch, 4, height, width] amplitudes")
    if min(raw_amplitude.shape) < 1 or not raw_amplitude.is_floating_point() or not target.is_floating_point():
        raise ValueError("Amplitudes must be nonempty floating tensors")
    if type(window_size) is not int or window_size < 3 or window_size % 2 == 0 or min(raw_amplitude.shape[-2:]) < window_size:
        raise ValueError("window_size must be odd, at least three, and fit inside the image")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    prediction = raw_amplitude.float() if raw_amplitude.dtype in (torch.float16, torch.bfloat16) else raw_amplitude
    reference = target.detach().to(prediction.dtype)
    prediction_rms = prediction.square().mean((-2, -1), keepdim=True).clamp_min(eps**2).sqrt()
    target_rms = reference.square().mean((-2, -1), keepdim=True).clamp_min(eps**2).sqrt()
    difference = prediction / prediction_rms - reference / target_rms
    difference = difference - difference.mean(1, keepdim=True)
    radius = window_size // 2
    highpass_difference = difference[..., radius:-radius, radius:-radius] - F.avg_pool2d(difference, window_size, stride=1)
    return highpass_difference.square().mean()
