"""Versioned image normalization; optical propagation is unchanged."""

import torch


LEGACY_PROTOCOL = "fresnel_png_v1"
PEAK_PROTOCOL = "fresnel_png_v2_peak"
PROTOCOLS = (LEGACY_PROTOCOL, PEAK_PROTOCOL)


def normalize_peak(image):
    """Normalize each image independently; an all-zero image remains zero."""
    if not image.is_floating_point():
        raise TypeError("Normalization requires floating-point amplitudes")
    maximum = image.amax(dim=(-2, -1), keepdim=True)
    return image / maximum.clamp_min(torch.finfo(image.dtype).tiny)


def prepare_target(image, protocol):
    if protocol == LEGACY_PROTOCOL:
        return image
    if protocol == PEAK_PROTOCOL:
        return normalize_peak(image)
    raise ValueError(f"Unknown protocol: {protocol}")


def prepare_reconstruction(amplitude, protocol):
    if protocol == LEGACY_PROTOCOL:
        return amplitude.clamp(0, 1)
    if protocol == PEAK_PROTOCOL:
        return normalize_peak(amplitude).clamp(0, 1)
    raise ValueError(f"Unknown protocol: {protocol}")
