"""Learned spectral context inside the target-phase encoder, not mask synthesis.

The nonlinear full-band mixer follows this project's FullSpectralBlock idea
(conceptual reference: Dong et al., https://doi.org/10.1364/OL.477630).
The hypothesis is spatially structured global context, not that the original
GroupNorm encoder has a strictly finite receptive field.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.transport import TransportEncoder


class TargetSpectralMixer(nn.Module):
    """Nonlinear real/imaginary channel mixing over all bottleneck rFFT bins."""

    def __init__(self, channels):
        super().__init__()
        self.frequency = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 1, bias=False),
            nn.GroupNorm(math.gcd(8, 2 * channels), 2 * channels),
            nn.SiLU(),
            nn.Conv2d(2 * channels, 2 * channels, 1, bias=False),
        )
        self.output_norm = nn.GroupNorm(math.gcd(8, channels), channels)

    def forward(self, features):
        spectrum = torch.fft.rfft2(features.float(), norm="ortho")
        mixed = self.frequency(torch.cat((spectrum.real, spectrum.imag), dim=1))
        real, imaginary = mixed.float().chunk(2, dim=1)
        decoded = torch.fft.irfft2(torch.complex(real, imaginary), s=features.shape[-2:], norm="ortho")
        return self.output_norm(decoded.to(features.dtype))


class GlobalTargetEncoder(TransportEncoder):
    """Preserve base keys and add h + g*S(h) at the 32x32 target bottleneck."""

    def __init__(self, in_channels, out_channels, width):
        super().__init__(in_channels, out_channels, width)
        # CPU-only fork restores RNG before the source decoder is constructed.
        # This creates learned random weights without changing any base weight.
        with torch.random.fork_rng(devices=[]):
            self.global_mixer = TargetSpectralMixer(4 * width)
        self.global_gain = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        latent = self.bottleneck(F.avg_pool2d(second, 2))
        latent = latent + self.global_gain * self.global_mixer(latent)
        decoded = self.decoder2(torch.cat((self.up2(latent), second), dim=1))
        decoded = self.decoder1(torch.cat((self.up1(decoded), first), dim=1))
        return self.output(decoded)
