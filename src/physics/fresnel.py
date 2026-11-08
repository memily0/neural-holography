import math

import torch
from torch import nn


DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)


class FresnelPropagator(nn.Module):
    """Single-FFT Fresnel integral with uniform unit-amplitude illumination.

    Input and output coordinates are centered at index ``size // 2``.
    The output pitch is wavelength * distance / (size * source_pitch).
    Real buffers retain their initial float64 precision for reference checks;
    forward uses float32/complex64 or float64/complex128 to match the phase.
    """

    def __init__(self, size=128, wavelength=532e-9, pitch=10e-6, distances=DISTANCES):
        super().__init__()
        if not isinstance(size, int) or size < 2:
            raise ValueError("size must be an integer >= 2")
        if not math.isfinite(wavelength) or not math.isfinite(pitch) or min(wavelength, pitch) <= 0:
            raise ValueError("wavelength and pitch must be finite and positive")
        distances = torch.as_tensor(distances, dtype=torch.float64)
        if distances.ndim != 1 or not distances.numel() or not torch.isfinite(distances).all() or (distances <= 0).any():
            raise ValueError("distances must be a nonempty sequence of positive finite values")

        self.size = size
        self.wavelength = wavelength
        self.pitch = pitch
        indices = torch.arange(size, dtype=torch.float64) - size // 2
        yy, xx = torch.meshgrid(indices, indices, indexing="ij")
        radius_squared = xx.square() + yy.square()
        output_pitch = wavelength * distances / (size * pitch)
        input_phase = math.pi * pitch**2 * radius_squared / (wavelength * distances[:, None, None])
        output_phase = math.pi * output_pitch[:, None, None].square() * radius_squared / (wavelength * distances[:, None, None])
        global_phase = torch.remainder(2 * math.pi * distances / wavelength, 2 * math.pi) - math.pi / 2
        output_phase = output_phase + global_phase[:, None, None]

        self.register_buffer("distances", distances)
        self.register_buffer("output_pitch", output_pitch)
        self.register_buffer("field_rms", size * pitch**2 / (wavelength * distances))
        self.register_buffer("input_phase", input_phase)
        self.register_buffer("output_phase", output_phase)

    def forward(self, phi):
        if phi.ndim == 4 and phi.shape[1] == 1:
            phi = phi[:, 0]
        if phi.ndim != 3 or phi.shape[-2:] != (self.size, self.size):
            raise ValueError("phase must have shape [B,H,W] or [B,1,H,W] matching size")
        if not phi.is_floating_point():
            raise TypeError("phase must be floating point, in radians")
        dtype = torch.float64 if phi.dtype == torch.float64 else torch.float32
        phi = phi.to(dtype)
        input_phase = self.input_phase.to(device=phi.device, dtype=dtype)
        output_phase = self.output_phase.to(device=phi.device, dtype=dtype)
        scale = self.field_rms.to(device=phi.device, dtype=dtype)
        field = torch.exp(1j * (phi[:, None] + input_phase))
        field = torch.fft.ifftshift(field, dim=(-2, -1))
        field = torch.fft.fft2(field, norm="ortho")
        field = torch.fft.fftshift(field, dim=(-2, -1))
        return field * torch.exp(1j * output_phase) * scale[None, :, None, None]

    def amplitude(self, phi):
        return self(phi).abs()


def phase_to_uint8(phi):
    """Encode [-pi, pi] as PNG codes 0..255, including both endpoints."""
    return torch.round((phi.clamp(-math.pi, math.pi) + math.pi) * (255 / (2 * math.pi))).clamp(0, 255).to(torch.uint8)


def uint8_to_phase(values, dtype=torch.float32):
    if values.dtype != torch.uint8:
        raise TypeError("phase image must be a uint8 tensor")
    return values.to(dtype) * (2 * math.pi / 255) - math.pi


def quantize_phase(phi, straight_through=False):
    quantized = uint8_to_phase(phase_to_uint8(phi), dtype=phi.dtype)
    if straight_through:
        return phi + (quantized - phi).detach()
    return quantized
