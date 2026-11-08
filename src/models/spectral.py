import math

import torch
from torch import nn

from src.models.networks import ConvBlock, phase_from_logits


def spectral_features(images):
    """Input features only: preserve both components of the centered target FFT."""
    centered = torch.fft.ifftshift(images.float(), dim=(-2, -1))
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered, norm="ortho"), dim=(-2, -1))
    # Invertible compression keeps a bright DC coefficient from dominating.
    return torch.asinh(torch.cat((spectrum.real, spectrum.imag), dim=1))


def position_features(size=128):
    axis = (torch.arange(size, dtype=torch.float32) - size / 2) / (size / 2)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = [xx, yy]
    for frequency in (1, 2, 4, 8, 16, 32):
        for value in (xx, yy):
            angle = math.pi * frequency * value
            coordinates.extend((angle.sin(), angle.cos()))
    return torch.stack(coordinates)[None]


class SpectralResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = ConvBlock(channels, channels)

    def forward(self, features):
        return (features + self.block(features)) / math.sqrt(2)


class SpectralPhaseGenerator(nn.Module):
    """Learn a phase from global spectral features, without analytic synthesis.

    ``spectral`` removes the direct target-pixel/phase-pixel shortcut. ``hybrid``
    also exposes spatial target channels as an explicit ablation. The network
    outputs all phase pixels through learned subpixel upsampling; no FFT is
    inverted to a complex optical field or converted directly into a mask.
    """

    def __init__(self, width=32, phase_head="wrapped", position="xy", feature_mode="spectral"):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if phase_head not in ("tanh", "circular", "wrapped"):
            raise ValueError("phase_head must be tanh, circular, or wrapped")
        if position not in ("none", "xy"):
            raise ValueError("spectral model supports position='none' or 'xy'")
        if feature_mode not in ("spectral", "hybrid"):
            raise ValueError("feature_mode must be spectral or hybrid")
        self.phase_head = phase_head
        self.position = position
        self.feature_mode = feature_mode
        self.radius_eps = 1e-4
        coordinates = position_features() if position == "xy" else None
        self.register_buffer("spectral_coordinates", coordinates)
        channels = 8 + (4 if feature_mode == "hybrid" else 0)
        channels += 0 if coordinates is None else coordinates.shape[1]
        self.spectral_stem = ConvBlock(channels, width)
        self.spectral_down1 = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            ConvBlock(2 * width, 2 * width),
        )
        self.spectral_down2 = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1),
            SpectralResidualBlock(4 * width),
            SpectralResidualBlock(4 * width),
        )
        self.spectral_up2 = nn.Sequential(nn.Conv2d(4 * width, 8 * width, 1), nn.PixelShuffle(2))
        self.spectral_decoder2 = ConvBlock(4 * width, 2 * width)
        self.spectral_up1 = nn.Sequential(nn.Conv2d(2 * width, 4 * width, 1), nn.PixelShuffle(2))
        self.spectral_decoder1 = ConvBlock(2 * width, width)
        self.spectral_output = nn.Conv2d(width, 2 if phase_head == "circular" else 1, 3, padding=1)

    def forward(self, images, return_diagnostics=False):
        if images.ndim != 4 or images.shape[1:] != (4, 128, 128):
            raise ValueError("images must have shape (batch, 4, 128, 128)")
        if not images.is_floating_point():
            raise TypeError("images must be floating point amplitudes")
        features = spectral_features(images)
        if self.feature_mode == "hybrid":
            features = torch.cat((features, images.float()), dim=1)
        if self.spectral_coordinates is not None:
            coordinates = self.spectral_coordinates.expand(images.shape[0], -1, -1, -1)
            features = torch.cat((features, coordinates), dim=1)
        first = self.spectral_stem(features)
        second = self.spectral_down1(first)
        latent = self.spectral_down2(second)
        decoded = self.spectral_decoder2(torch.cat((self.spectral_up2(latent), second), dim=1))
        decoded = self.spectral_decoder1(torch.cat((self.spectral_up1(decoded), first), dim=1))
        logits = self.spectral_output(decoded)
        return phase_from_logits(logits, self.phase_head, self.radius_eps, return_diagnostics)
