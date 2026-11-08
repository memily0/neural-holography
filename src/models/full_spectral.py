"""Learned full-band spectral features, with no optical inverse in the generator.

Conceptual reference: Dong et al., Optics Letters 48, 759 (2023),
https://doi.org/10.1364/OL.477630. This is a fresh implementation, using the
project's convolution blocks, normalization, phase heads and four input planes.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.networks import ConvBlock, phase_from_logits


class FullSpectralBlock(nn.Module):
    """Combine local convolutions with nonlinear mixing of every rFFT bin."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local = ConvBlock(in_channels, out_channels)
        self.frequency = nn.Sequential(
            nn.Conv2d(2 * in_channels, 2 * out_channels, 1, bias=False),
            nn.GroupNorm(math.gcd(8, 2 * out_channels), 2 * out_channels),
            nn.SiLU(),
            nn.Conv2d(2 * out_channels, 2 * out_channels, 1, bias=False),
        )
        self.spectral_norm = nn.GroupNorm(math.gcd(8, out_channels), out_channels)
        self.spectral_gain = nn.Parameter(torch.tensor(0.1))

    def spectral_features(self, x):
        shape = x.shape[-2:]
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        encoded = self.frequency(torch.cat((spectrum.real, spectrum.imag), dim=1))
        real, imag = encoded.float().chunk(2, dim=1)
        decoded = torch.fft.irfft2(torch.complex(real, imag), s=shape, norm="ortho")
        return self.spectral_norm(decoded.to(x.dtype))

    def forward(self, x):
        return self.local(x) + self.spectral_gain * self.spectral_features(x)


class FullSpectralGenerator(nn.Module):
    """Multiscale spatial/spectral encoder-decoder to one phase-only mask."""

    def __init__(self, width=16, phase_head="circular", position="xy"):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if phase_head not in ("tanh", "circular", "wrapped"):
            raise ValueError("phase_head must be tanh, circular, or wrapped")
        if position not in ("none", "xy"):
            raise ValueError("position must be none or xy")
        self.phase_head = phase_head
        self.position = position
        self.radius_eps = 1e-4
        coordinates = torch.arange(128, dtype=torch.float32) / 64 - 1
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("full_coordinates", torch.stack((xx, yy))[None])
        self.full_encoder1 = FullSpectralBlock(6 if position == "xy" else 4, width)
        self.full_encoder2 = FullSpectralBlock(width, 2 * width)
        self.full_encoder3 = FullSpectralBlock(2 * width, 4 * width)
        self.full_bottleneck = FullSpectralBlock(4 * width, 8 * width)
        self.full_up3 = nn.ConvTranspose2d(8 * width, 4 * width, 2, stride=2)
        self.full_decoder3 = FullSpectralBlock(8 * width, 4 * width)
        self.full_up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.full_decoder2 = FullSpectralBlock(4 * width, 2 * width)
        self.full_up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.full_decoder1 = FullSpectralBlock(2 * width, width)
        self.full_output = nn.Conv2d(width, 2 if phase_head == "circular" else 1, 1)

    def forward(self, images, return_diagnostics=False):
        if images.ndim != 4 or images.shape[1:] != (4, 128, 128):
            raise ValueError("images must have shape (batch, 4, 128, 128)")
        if self.position == "xy":
            coordinates = self.full_coordinates.to(images.dtype).expand(images.shape[0], -1, -1, -1)
            images = torch.cat((images, coordinates), dim=1)
        first = self.full_encoder1(images)
        second = self.full_encoder2(F.avg_pool2d(first, 2))
        third = self.full_encoder3(F.avg_pool2d(second, 2))
        latent = self.full_bottleneck(F.avg_pool2d(third, 2))
        decoded = self.full_decoder3(torch.cat((self.full_up3(latent), third), dim=1))
        decoded = self.full_decoder2(torch.cat((self.full_up2(decoded), second), dim=1))
        decoded = self.full_decoder1(torch.cat((self.full_up1(decoded), first), dim=1))
        return phase_from_logits(self.full_output(decoded), self.phase_head, self.radius_eps, return_diagnostics)
