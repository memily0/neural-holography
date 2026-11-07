import cmath
import math

import torch

from .config import DISTANCES, PIXEL_PITCH, SIZE, WAVELENGTH


def fresnel_field(phase: torch.Tensor, distance: float,
                  wavelength: float = WAVELENGTH,
                  pixel_pitch: float = PIXEL_PITCH) -> torch.Tensor:
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


def norm_amp(amplitude: torch.Tensor) -> torch.Tensor:
    return amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def physics_seed(targets: torch.Tensor) -> torch.Tensor:
    ind = (torch.arange(SIZE, device=targets.device, dtype=torch.float32) - SIZE // 2) * PIXEL_PITCH
    uy, ux = torch.meshgrid(ind, ind, indexing="ij")
    chirps = torch.stack([
        torch.exp(1j * math.pi / (WAVELENGTH * z) * (ux.square() + uy.square()))
        for z in DISTANCES
    ], dim=0).to(torch.complex64)
    t = targets.float().to(torch.complex64)
    back_planes = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(t, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1)
    ) * chirps[None].conj()
    back = back_planes.mean(dim=1)
    seed = torch.stack((back.real, back.imag), dim=1).float()
    maxabs = seed.square().sum(dim=1, keepdim=True).sqrt().amax(dim=(-2, -1), keepdim=True)
    return seed / maxabs.clamp_min(1e-6)
