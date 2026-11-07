import math
import cmath
import torch

DISTANCES = (0.0241, 0.0253, 0.0265, 0.0277)


def fresnel_field(phase: torch.Tensor, distance: float,
                  wavelength: float = 532e-9,
                  pixel_pitch: float = 10e-6) -> torch.Tensor:
    """Single-FFT Fresnel propagation. Input: phase (B, 1, N, N), rad; output: complex field."""
    if phase.ndim != 4 or phase.shape[1] != 1 or phase.shape[-2] != phase.shape[-1]:
        raise ValueError("phase must have shape (B, 1, N, N)")
    if not phase.is_floating_point():
        raise TypeError("phase must be a floating-point tensor")
    if distance <= 0:
        raise ValueError("distance must be positive")

    n = phase.shape[-1]
    indices = torch.arange(n, device=phase.device, dtype=phase.dtype) - n // 2
    u, v = torch.meshgrid(indices * pixel_pitch, indices * pixel_pitch, indexing="ij")
    chirp_in = torch.exp(1j * math.pi / (wavelength * distance) * (u**2 + v**2))
    field = torch.exp(1j * phase) * chirp_in

    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )

    output_pitch = wavelength * distance / (n * pixel_pitch)
    x, y = torch.meshgrid(indices * output_pitch, indices * output_pitch, indexing="ij")
    chirp_out = torch.exp(1j * math.pi / (wavelength * distance) * (x**2 + y**2))
    prefactor = (pixel_pitch**2 / (1j * wavelength * distance)) * cmath.exp(
        1j * 2 * math.pi * distance / wavelength
    )
    return prefactor * chirp_out * spectrum


def fresnel_propagate(phase: torch.Tensor, distance: float) -> torch.Tensor:
    return fresnel_field(phase, distance).abs()


def reconstruct_four(phase: torch.Tensor) -> torch.Tensor:
    return torch.cat([fresnel_propagate(phase, z) for z in DISTANCES], dim=1)

