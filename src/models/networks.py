import math

import torch
from torch import nn
from torch.nn import functional as F


def phase_from_logits(logits, phase_head, radius_eps=1e-4, return_diagnostics=False):
    diagnostics = {}
    if phase_head == "tanh":
        phase = math.pi * torch.tanh(logits.float())
    elif phase_head == "wrapped":
        angle = math.pi * logits.float()
        phase = torch.atan2(angle.sin(), angle.cos())
    else:
        real, imag = logits.float().chunk(2, dim=1)
        radius_sq = real.square() + imag.square()
        near_origin = radius_sq < radius_eps ** 2
        # Guard the arguments before atan2 to avoid NaN gradients at (0, 0).
        safe_real = torch.where(near_origin, torch.ones_like(real), real)
        safe_imag = torch.where(near_origin, torch.zeros_like(imag), imag)
        phase = torch.atan2(safe_imag, safe_real)
        if return_diagnostics:
            diagnostics = {
                "radius_min": radius_sq.detach().min().sqrt(),
                "near_origin_fraction": near_origin.detach().float().mean(),
            }
    if return_diagnostics:
        return phase, diagnostics
    return phase


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = math.gcd(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)


class FourierMixing(nn.Module):
    """Learned feature mixing, independent of the optical propagation model."""

    def __init__(self, channels, modes=12):
        super().__init__()
        self.modes = modes
        self.weight_top = nn.Parameter(0.02 * torch.randn(channels, modes, modes, 2))
        self.weight_bottom = nn.Parameter(0.02 * torch.randn(channels, modes, modes, 2))
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        # Float32 FFT also works when the surrounding convolutions use autocast.
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        filtered = torch.zeros_like(spectrum)
        rows = min(self.modes, height // 2)
        cols = min(self.modes, spectrum.shape[-1])
        top = torch.view_as_complex(self.weight_top.float().contiguous())
        bottom = torch.view_as_complex(self.weight_bottom.float().contiguous())
        filtered[:, :, :rows, :cols] = spectrum[:, :, :rows, :cols] * top[:, :rows, :cols]
        filtered[:, :, -rows:, :cols] = spectrum[:, :, -rows:, :cols] * bottom[:, :rows, :cols]
        global_features = torch.fft.irfft2(filtered, s=(height, width), norm="ortho")
        return x + self.mix(F.silu(global_features.to(x.dtype)))


class LocalMixing(nn.Module):
    """Residual local control with almost the same parameter count as FourierMixing."""

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.normal_(self.local.weight, std=0.02)
        nn.init.zeros_(self.local.bias)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.mix(F.silu(self.local(x)))


class AttentionMixing(nn.Module):
    """Content-dependent global mixing of 8 x 8 pooled spatial tokens."""

    def __init__(self, channels, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.position = nn.Linear(2, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, math.gcd(4, channels), batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 3 * channels),
            nn.SiLU(),
            nn.Linear(3 * channels, channels),
        )

    def forward(self, x):
        height, width = min(self.grid_size, x.shape[-2]), min(self.grid_size, x.shape[-1])
        pooled = F.adaptive_avg_pool2d(x, (height, width))
        tokens = pooled.flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        normalized = self.norm1(tokens)
        query_key = normalized + self.position(coordinates)
        global_tokens, _ = self.attention(query_key, query_key, normalized, need_weights=False)
        mixed = tokens + global_tokens
        mixed = mixed + self.feed_forward(self.norm2(mixed))
        delta = (mixed - tokens).transpose(1, 2).reshape_as(pooled)
        return x + F.interpolate(delta, size=x.shape[-2:], mode="bilinear", align_corners=False)


class PhaseGenerator(nn.Module):
    """One feed-forward pass maps four target amplitudes to one phase mask."""

    def __init__(self, width=16, use_skips=True, use_fourier=False, phase_head="tanh", position="none", use_local=False, use_attention=False):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if phase_head not in ("tanh", "circular", "wrapped"):
            raise ValueError("phase_head must be 'tanh', 'circular', or 'wrapped'")
        if position not in ("none", "xy", "chirp"):
            raise ValueError("position must be 'none', 'xy', or 'chirp'")
        if sum((use_fourier, use_local, use_attention)) > 1:
            raise ValueError("choose only one of Fourier, local, or attention mixing")
        self.use_skips = use_skips
        self.phase_head = phase_head
        self.position = position
        self.radius_eps = 1e-4
        self.encoder1 = ConvBlock(4, width)
        self.encoder2 = ConvBlock(width, 2 * width)
        self.bottleneck = ConvBlock(2 * width, 4 * width)
        self.global_mixing = nn.Identity()
        if use_fourier:
            self.global_mixing = FourierMixing(4 * width)
        elif use_local:
            self.global_mixing = LocalMixing(4 * width)
        elif use_attention:
            self.global_mixing = AttentionMixing(4 * width)
        self.up2 = nn.Conv2d(4 * width, 2 * width, 1)
        self.decoder2 = ConvBlock(2 * width, 2 * width)
        self.up1 = nn.Conv2d(2 * width, width, 1)
        self.decoder1 = ConvBlock(width, width)
        self.output = nn.Conv2d(width, 2 if phase_head == "circular" else 1, 1)
        # Construct the extra branch last, preserving all baseline weight shapes
        # and their seeded initialization. These are features, not a phase mask.
        features = None
        self.position_projection = None
        if position != "none":
            indices = torch.arange(128, dtype=torch.float64) - 64
            yy, xx = torch.meshgrid(indices, indices, indexing="ij")
            features = torch.stack((xx / 64, yy / 64))
            if position == "chirp":
                distances = torch.tensor((0.0241, 0.0253, 0.0265, 0.0277), dtype=torch.float64)
                phase = math.pi * (10e-6) ** 2 * (xx.square() + yy.square())
                phase = phase[None] / (532e-9 * distances[:, None, None])
                features = torch.cat((features, phase.sin(), phase.cos()))
            features = features.float().unsqueeze(0)
            self.position_projection = nn.Conv2d(features.shape[1], width, 1, bias=False)
            nn.init.zeros_(self.position_projection.weight)
        self.register_buffer("position_features", features)

    def to_phase(self, logits, return_diagnostics=False):
        return phase_from_logits(logits, self.phase_head, self.radius_eps, return_diagnostics)

    def forward(self, images, return_diagnostics=False):
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError("images must have shape (batch, 4, height, width)")
        if min(images.shape[-2:]) < 8:
            raise ValueError("image dimensions must be at least 8")
        if self.position != "none" and images.shape[-2:] != (128, 128):
            raise ValueError("fixed position features require 128 x 128 images")
        first = self.encoder1(images)
        if self.position_projection is not None:
            first = first + self.position_projection(self.position_features.to(first.dtype))
        second = self.encoder2(F.avg_pool2d(first, 2))
        latent = self.bottleneck(F.avg_pool2d(second, 2))
        latent = self.global_mixing(latent)
        decoded = self.up2(F.interpolate(latent, size=second.shape[-2:], mode="bilinear", align_corners=False))
        if self.use_skips:
            decoded = decoded + second
        decoded = self.decoder2(decoded)
        decoded = self.up1(F.interpolate(decoded, size=first.shape[-2:], mode="bilinear", align_corners=False))
        if self.use_skips:
            decoded = decoded + first
        decoded = self.decoder1(decoded)
        return self.to_phase(self.output(decoded), return_diagnostics)


class InterleavedGenerator(nn.Module):
    """Learn all sixteen subpixel phases without interpolating the output mask."""

    def __init__(self, width=16, phase_head="tanh", position="none"):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if phase_head not in ("tanh", "circular", "wrapped"):
            raise ValueError("phase_head must be 'tanh', 'circular', or 'wrapped'")
        if position != "none":
            raise ValueError("interleave currently supports only position='none'")
        self.phase_head = phase_head
        self.position = position
        self.radius_eps = 1e-4
        self.subpixel_features = nn.Sequential(
            nn.PixelUnshuffle(4),
            ConvBlock(64, width),
            ConvBlock(width, 2 * width),
            ConvBlock(2 * width, 4 * width),
            ConvBlock(4 * width, 2 * width),
            ConvBlock(2 * width, width),
        )
        self.subpixel_output = nn.Sequential(
            nn.Conv2d(width, 32 if phase_head == "circular" else 16, 1),
            nn.PixelShuffle(4),
        )

    def to_phase(self, logits, return_diagnostics=False):
        return phase_from_logits(logits, self.phase_head, self.radius_eps, return_diagnostics)

    def forward(self, images, return_diagnostics=False):
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError("images must have shape (batch, 4, height, width)")
        if min(images.shape[-2:]) < 8 or any(size % 4 for size in images.shape[-2:]):
            raise ValueError("interleave image dimensions must be at least 8 and divisible by 4")
        features = self.subpixel_features(images)
        logits = self.subpixel_output(features)
        return self.to_phase(logits, return_diagnostics)


def coordinate_features(size):
    axis = (torch.arange(size, dtype=torch.float32) + 0.5) * 2 / size - 1
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=-1).reshape(1, size * size, 2)
    frequencies = torch.tensor((1, 2, 4, 8), dtype=torch.float32) * math.pi
    angles = coordinates.unsqueeze(-1) * frequencies
    return torch.cat((coordinates, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)


class CrossAttentionGenerator(nn.Module):
    """Coordinate-only phase queries read all target tokens in one forward pass."""

    def __init__(self, width=16, phase_head="tanh", position="xy"):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if phase_head not in ("tanh", "circular", "wrapped"):
            raise ValueError("phase_head must be 'tanh', 'circular', or 'wrapped'")
        if position not in ("none", "xy"):
            raise ValueError("cross_attention uses its own XY features; choose position='none' or 'xy'")
        self.phase_head = phase_head
        self.position = position
        self.radius_eps = 1e-4
        channels = 4 * width
        # Distinct names prevent accidental loading of aligned CNN decoder weights.
        self.cross_encoder = nn.Sequential(
            ConvBlock(4, width),
            nn.AvgPool2d(2),
            ConvBlock(width, 2 * width),
            nn.AvgPool2d(2),
            ConvBlock(2 * width, channels),
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.register_buffer("cross_target_coordinates", coordinate_features(16))
        self.register_buffer("cross_phase_coordinates", coordinate_features(32))
        self.cross_target_position = nn.Linear(18, channels)
        self.cross_query = nn.Sequential(nn.Linear(18, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.cross_query_norm = nn.LayerNorm(channels)
        self.cross_target_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(channels, math.gcd(4, channels), batch_first=True)
        self.cross_output_norm = nn.LayerNorm(channels)
        self.cross_feed_forward = nn.Sequential(
            nn.Linear(channels, 3 * channels), nn.SiLU(), nn.Linear(3 * channels, channels),
        )
        self.cross_phase_output = nn.Linear(channels, 32 if phase_head == "circular" else 16)

    def to_phase(self, logits, return_diagnostics=False):
        return phase_from_logits(logits, self.phase_head, self.radius_eps, return_diagnostics)

    def forward(self, images, return_diagnostics=False):
        if images.ndim != 4 or images.shape[1:] != (4, 128, 128):
            raise ValueError("cross_attention images must have shape (batch, 4, 128, 128)")
        tokens = self.cross_encoder(images).flatten(2).transpose(1, 2)
        values = self.cross_target_norm(tokens)
        keys = values + self.cross_target_position(self.cross_target_coordinates.to(tokens.dtype))
        query = self.cross_query(self.cross_phase_coordinates.to(tokens.dtype)).expand(images.shape[0], -1, -1)
        attended, _ = self.cross_attention(self.cross_query_norm(query), keys, values, need_weights=False)
        features = query + attended
        features = features + self.cross_feed_forward(self.cross_output_norm(features))
        logits = self.cross_phase_output(features).transpose(1, 2).reshape(images.shape[0], -1, 32, 32)
        logits = F.pixel_shuffle(logits, 4)
        return self.to_phase(logits, return_diagnostics)


def build_model(name, width=16, phase_head="tanh", position="none"):
    transport_names = (
        "transport", "transport_gauge", "transport_random", "transport_gauge_random",
        "transport_linear", "transport_random_linear", "transport_gauge_random_linear",
        "transport_subpixel", "transport_random_subpixel", "transport_gauge_random_subpixel",
        "transport_complex", "transport_random_complex", "transport_gauge_random_complex",
        "transport_pointwise", "transport_random_pointwise", "transport_gauge_random_pointwise",
        "transport_subtarget_complex", "transport_residualtarget_complex",
        "transport_spatial_complex", "transport_energy_complex", "transport_envelope_complex",
        "transport_consensus_complex", "transport_globaltarget_complex",
    )
    if name in transport_names:
        from src.models.transport import TransportPhaseGenerator
        decoder = next((kind for kind in ("subpixel", "complex", "pointwise") if name.endswith("_" + kind)), "unet")
        target_encoder = "subpixel" if "subtarget" in name else "residual" if "residualtarget" in name else "envelope" if "envelope" in name else "consensus" if "consensus" in name else "global" if "globaltarget" in name else "unet"
        return TransportPhaseGenerator(
            width=width, phase_head=phase_head, position=position,
            include_output_chirp="gauge" not in name,
            phase_bias_init="random" if "random" in name else "zero",
            linear_skip=name.endswith("_linear"), phase_decoder=decoder,
            target_encoder=target_encoder, complex_mix_kernel=3 if "spatial" in name else 1,
            energy_balanced="energy" in name,
        )
    if name.endswith("_bias"):
        from src.models.phase_bias import SharedPhaseBias
        return SharedPhaseBias(build_model(name[:-5], width, phase_head, position))
    if name == "full_spectral":
        from src.models.full_spectral import FullSpectralGenerator
        return FullSpectralGenerator(width=width, phase_head=phase_head, position=position)
    if name in ("spectral", "spectral_hybrid"):
        from src.models.spectral import SpectralPhaseGenerator
        mode = "spectral" if name == "spectral" else "hybrid"
        return SpectralPhaseGenerator(width=width, phase_head=phase_head, position=position, feature_mode=mode)
    if name == "interleave":
        return InterleavedGenerator(width=width, phase_head=phase_head, position=position)
    if name == "cross_attention":
        return CrossAttentionGenerator(width=width, phase_head=phase_head, position=position)
    if name not in ("cnn", "fourier", "ae", "local", "attention"):
        raise ValueError(f"Unknown model: {name}. Choose cnn, fourier, ae, local, attention, cross_attention, or interleave.")
    return PhaseGenerator(
        width=width,
        use_skips=name != "ae",
        use_fourier=name == "fourier",
        phase_head=phase_head,
        position=position,
        use_local=name == "local",
        use_attention=name == "attention",
    )
