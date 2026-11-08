"""Optional biased peak-normalization gradient for shared-network training."""

import torch


def peak_normalize_surrogate(raw, detach_peak=False):
    """Keep the exact peak-normalized forward, optionally stop peak gradients.

    The detached branch is a training surrogate, not the derivative of the
    reported normalized objective. Its clamp uses an explicit identity
    backward, including at the normalized maximum, so the intended derivative
    is 1 / detached_peak for nonzero nonnegative amplitude images. Evaluation
    should keep the usual protocol. Entirely zero images have zero gradient.
    """
    if not raw.is_floating_point():
        raise TypeError("Normalization requires floating-point amplitudes")
    maximum = raw.amax(dim=(-2, -1), keepdim=True)
    denominator = maximum.detach() if detach_peak else maximum
    normalized = raw / denominator.clamp_min(torch.finfo(raw.dtype).tiny)
    normalized = torch.where(maximum > 0, normalized, torch.zeros_like(normalized))
    clipped = normalized.clamp(0, 1)
    # The clamp is redundant for valid amplitudes, but its endpoint derivative
    # differs across torch runtimes. Specify the surrogate's boundary choice.
    return normalized + (clipped - normalized).detach() if detach_peak else clipped
