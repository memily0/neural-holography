"""Learned complex features transported between target and DOE coordinates.

The fixed adjoint is a feature layer, not a phase-only synthesis algorithm.
Both target phases and the final four-plane phase-only fusion are learned.
There is no analytic angle extraction, classical encoding, iterative mask
update or phase-label teacher. Conceptual reference: HoloNet, Peng et al.,
ACM TOG 39(6), 2020, https://doi.org/10.1145/3414685.3417802.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.models.networks import ConvBlock, phase_from_logits
from src.physics.fresnel import FresnelPropagator
from src.latent_losses import balanced_source_features


class UnitaryFresnelTransport(nn.Module):
    """Plane-wise unitary transport on the task's exact discrete grids.

    Each of four feature channels is a separate complex field on its plane.
    Positive physical prefactors are omitted to obtain unitary feature maps.
    Omitting the output chirp chooses an amplitude-equivalent target-phase
    gauge; it does not change the physical propagation used for evaluation.
    """

    def __init__(self, size=128, include_output_chirp=True):
        super().__init__()
        physics = FresnelPropagator(size=size)
        self.size = size
        self.plane_count = len(physics.distances)
        self.include_output_chirp = include_output_chirp
        self.register_buffer("input_phase", physics.input_phase.clone())
        self.register_buffer("output_phase", physics.output_phase.clone())

    def rotations(self, field):
        if field.ndim != 4 or field.shape[1:] != (self.plane_count, self.size, self.size):
            raise ValueError("complex fields must have shape [batch, planes, size, size]")
        if not field.is_complex():
            raise TypeError("transport requires complex-valued fields")
        input_phase = self.input_phase.to(device=field.device, dtype=field.real.dtype)
        output_phase = self.output_phase.to(device=field.device, dtype=field.real.dtype)
        return torch.exp(1j * input_phase), torch.exp(1j * output_phase)

    def forward(self, field):
        input_rotation, output_rotation = self.rotations(field)
        field = torch.fft.ifftshift(field * input_rotation, dim=(-2, -1))
        field = torch.fft.fftshift(torch.fft.fft2(field, norm="ortho"), dim=(-2, -1))
        return field * output_rotation if self.include_output_chirp else field

    def adjoint(self, field):
        input_rotation, output_rotation = self.rotations(field)
        if self.include_output_chirp:
            field = field * output_rotation.conj()
        field = torch.fft.ifftshift(field, dim=(-2, -1))
        field = torch.fft.fftshift(torch.fft.ifft2(field, norm="ortho"), dim=(-2, -1))
        return field * input_rotation.conj()


class TransportEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, width):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, width)
        self.encoder2 = ConvBlock(width, 2 * width)
        self.bottleneck = ConvBlock(2 * width, 4 * width)
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.decoder2 = ConvBlock(4 * width, 2 * width)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.decoder1 = ConvBlock(2 * width, width)
        self.output = nn.Conv2d(width, out_channels, 1)

    def forward(self, images):
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        latent = self.bottleneck(F.avg_pool2d(second, 2))
        decoded = self.decoder2(torch.cat((self.up2(latent), second), dim=1))
        decoded = self.decoder1(torch.cat((self.up1(decoded), first), dim=1))
        return self.output(decoded)


class TransportSubpixelEncoder(nn.Module):
    """Preserve input pixels as channels and learn every output subpixel.

    Pixel rearrangements are exact permutations. The output channels are free
    learned logits, not double-phase encoding or another analytic phase code.
    """

    def __init__(self, in_channels, out_channels, width):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(4)
        self.features = nn.Sequential(
            ConvBlock(16 * in_channels, 2 * width),
            ConvBlock(2 * width, 4 * width),
            ConvBlock(4 * width, 8 * width),
            ConvBlock(8 * width, 4 * width),
            ConvBlock(4 * width, 2 * width),
        )
        self.output = nn.Conv2d(2 * width, 16 * out_channels, 1)
        self.shuffle = nn.PixelShuffle(4)

    def forward(self, features):
        encoded = self.features(self.unshuffle(features))
        return self.shuffle(self.output(encoded))


class TransportResidualTargetEncoder(nn.Module):
    """Add input-dependent subpixel detail to a learned target-phase U-Net."""

    def __init__(self, in_channels, out_channels, width):
        super().__init__()
        self.base = TransportEncoder(in_channels, out_channels, width)
        self.detail = TransportSubpixelEncoder(in_channels, out_channels, width)
        self.reset_detail_output()

    def reset_detail_output(self):
        nn.init.zeros_(self.detail.output.weight)
        nn.init.zeros_(self.detail.output.bias)

    def forward(self, images):
        return self.base(images) + self.detail(images)


class TransportPhaseGenerator(nn.Module):
    """Learn target phases, transport features, then learn one phase-only mask."""

    def __init__(self, width=16, phase_head="circular", position="xy", include_output_chirp=True, phase_bias_init="zero", linear_skip=False, phase_decoder="unet", target_encoder="unet", complex_mix_kernel=1, energy_balanced=False):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if phase_head not in ("tanh", "circular", "wrapped"):
            raise ValueError("phase_head must be tanh, circular, or wrapped")
        if position not in ("none", "xy"):
            raise ValueError("position must be none or xy")
        if phase_bias_init not in ("zero", "random"):
            raise ValueError("phase_bias_init must be zero or random")
        if phase_decoder not in ("unet", "subpixel", "complex", "pointwise"):
            raise ValueError("phase_decoder must be unet, subpixel, complex or pointwise")
        if phase_decoder in ("complex", "pointwise") and phase_head != "circular":
            raise ValueError("Complex phase decoders require the circular head")
        if complex_mix_kernel not in (1, 3) or (complex_mix_kernel != 1 and phase_decoder not in ("complex", "pointwise")):
            raise ValueError("A spatial complex mix requires a complex phase decoder")
        if target_encoder not in ("unet", "subpixel", "residual", "envelope", "consensus", "global"):
            raise ValueError("target_encoder must be unet, subpixel, residual, envelope, consensus or global")
        self.phase_head = phase_head
        self.position = position
        self.phase_bias_init = phase_bias_init
        self.phase_decoder = phase_decoder
        self.target_encoder = target_encoder
        self.complex_mix_kernel = complex_mix_kernel
        self.energy_balanced = energy_balanced
        self.radius_eps = 1e-4
        coordinates = torch.arange(128, dtype=torch.float32) / 64 - 1
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("transport_coordinates", torch.stack((xx, yy))[None])
        extra = 2 if position == "xy" else 0
        if target_encoder in ("envelope", "consensus"):
            from src.models.target_envelope import TargetEnvelopeEncoder
            target_class = TargetEnvelopeEncoder
        elif target_encoder == "global":
            from src.models.target_global import GlobalTargetEncoder
            target_class = GlobalTargetEncoder
        else:
            target_class = {"unet": TransportEncoder, "subpixel": TransportSubpixelEncoder, "residual": TransportResidualTargetEncoder}[target_encoder]
        self.transport_target_encoder = target_class(4 + extra, 4, width)
        # A shared network parameter, never initialized as a lens or fitted per input.
        self.transport_target_bias = nn.Parameter(torch.zeros(1, 4, 128, 128))
        self.transport_operator = UnitaryFresnelTransport(include_output_chirp=include_output_chirp)
        if phase_decoder in ("complex", "pointwise"):
            from src.models.complex_decoder import ComplexPhaseEncoder
            self.transport_phase_encoder = ComplexPhaseEncoder(8 + extra, 2, width, spatial_context=phase_decoder == "complex", mix_kernel=complex_mix_kernel)
        else:
            decoder_class = TransportEncoder if phase_decoder == "unet" else TransportSubpixelEncoder
            self.transport_phase_encoder = decoder_class(8 + extra, 2 if phase_head == "circular" else 1, width)
        # Sample only after both CNNs, keeping their seeded initial weights equal
        # in zero/random ablations. This remains one dataset-wide learned bias.
        if phase_bias_init == "random":
            nn.init.uniform_(self.transport_target_bias, -math.pi, math.pi)
        # Construct last so the other seeded weights, including random bias,
        # remain identical. This learned path preserves fine complex features;
        # it contains no fixed summation coefficients or analytic mask encoding.
        self.transport_linear_skip = None
        if linear_skip:
            self.transport_linear_skip = nn.Conv2d(8, 2 if phase_head == "circular" else 1, 1, bias=False)
            nn.init.zeros_(self.transport_linear_skip.weight)
        if target_encoder == "consensus":
            # One shared learned scalar, never fitted or updated for an input.
            # Zeros consume no RNG; every existing model keeps its old state keys.
            self.transport_consensus_gain = nn.Parameter(torch.zeros(()))

    def load_pointwise_state_dict(self, state):
        """Expand a matching 1x1 decoder checkpoint; caller validates config flags."""
        from src.models.complex_decoder import expand_complex_mixing_state
        if self.complex_mix_kernel != 3:
            raise ValueError("Pointwise transfer requires a 3x3 complex decoder")
        prefix = "transport_phase_encoder."
        decoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        expanded = expand_complex_mixing_state(decoder_state, self.transport_phase_encoder.state_dict())
        mapped = dict(state)
        mapped.update({prefix + key: value for key, value in expanded.items()})
        expected = self.state_dict()
        if set(mapped) != set(expected):
            raise ValueError("Pointwise transfer architecture mismatch")
        if any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape for key, value in mapped.items()):
            raise ValueError("Pointwise transfer tensor shape mismatch")
        return self.load_state_dict(mapped, strict=True)

    def load_envelope_state_dict(self, state):
        """Copy an ordinary transport_complex state and zero the new envelope.

        Validate every key, shape and dtype before changing any parameter.
        The only missing keys must be the two new envelope tensors. Config
        flags absent from state_dict (chirp gauge, initialization labels) must
        additionally be checked by the caller against the source config.
        No optimizer state is transferred and no inference input is fitted.
        """
        if self.target_encoder != "envelope" or self.phase_decoder != "complex" or self.complex_mix_kernel != 1 or self.energy_balanced or self.transport_linear_skip is not None:
            raise ValueError("Envelope transfer requires an ordinary complex decoder without other architecture changes")
        if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
            raise ValueError("Expected a nonempty model state dictionary with string keys")
        expected = self.state_dict()
        envelope_keys = {"transport_target_encoder.envelope.weight", "transport_target_encoder.envelope.bias"}
        if set(expected) - set(state) != envelope_keys or set(state) - set(expected):
            raise ValueError("Envelope transfer mismatch: only the new envelope head may be missing")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError(f"Envelope transfer tensor shape/dtype mismatch: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite envelope source tensor: {key}")
        mapped = dict(state)
        mapped.update({key: torch.zeros_like(expected[key]) for key in envelope_keys})
        return self.load_state_dict(mapped, strict=True)

    def load_consensus_state_dict(self, state):
        """Copy ordinary transport_complex weights and zero envelope plus gain.

        Only the three new tensors may be absent. Validate all source keys,
        shapes, dtypes and finite values before any mutation. The caller must
        also verify the source model/config, including flags absent from its
        state dictionary. No optimizer or per-input phase state is transferred.
        """
        if self.target_encoder != "consensus" or self.phase_decoder != "complex" or self.complex_mix_kernel != 1 or self.energy_balanced or self.transport_linear_skip is not None or not self.transport_operator.include_output_chirp or self.phase_bias_init != "zero":
            raise ValueError("Consensus transfer requires the ordinary full-chirp complex architecture")
        if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
            raise ValueError("Expected a nonempty model state dictionary with string keys")
        expected = self.state_dict()
        new_keys = {"transport_target_encoder.envelope.weight", "transport_target_encoder.envelope.bias", "transport_consensus_gain"}
        if set(expected) - set(state) != new_keys or set(state) - set(expected):
            raise ValueError("Consensus transfer mismatch: only envelope and consensus gain may be missing")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError(f"Consensus transfer tensor shape/dtype mismatch: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite consensus source tensor: {key}")
        mapped = dict(state)
        mapped.update({key: torch.zeros_like(expected[key]) for key in new_keys})
        return self.load_state_dict(mapped, strict=True)

    def load_globaltarget_state_dict(self, state):
        """Copy ordinary transport_complex weights, retaining a closed mixer.

        Validate the complete mapping before mutation. Only new target mixer
        tensors and its scalar gate may be missing. The randomly initialized
        mixer is retained, and the gate is reset to zero without consuming RNG.
        The caller must also validate the source architecture/config flags.
        """
        if self.target_encoder != "global" or self.phase_decoder != "complex" or self.complex_mix_kernel != 1 or self.energy_balanced or self.transport_linear_skip is not None or not self.transport_operator.include_output_chirp or self.phase_bias_init != "zero":
            raise ValueError("Global-target transfer requires the ordinary full-chirp complex architecture")
        if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
            raise ValueError("Expected a nonempty model state dictionary with string keys")
        expected = self.state_dict()
        gain_key = "transport_target_encoder.global_gain"
        new_keys = {key for key in expected if key.startswith("transport_target_encoder.global_mixer.")} | {gain_key}
        if set(expected) - set(state) != new_keys or set(state) - set(expected):
            raise ValueError("Global-target transfer mismatch: only the new mixer and gain may be missing")
        mapped = dict(state)
        mapped.update({key: expected[key] for key in new_keys})
        mapped[gain_key] = torch.zeros_like(expected[gain_key])
        for key, value in mapped.items():
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError(f"Global-target transfer tensor shape/dtype mismatch: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite global-target tensor: {key}")
        return self.load_state_dict(mapped, strict=True)

    def load_base_state_dict(self, state):
        """Initialize a residual target model from an otherwise identical U-Net.

        Every existing weight/buffer must match. Only the new detail branch
        may be missing, and its final layer is reset to zero. Validate all keys
        and tensor shapes before changing state. Configuration flags absent
        from state_dict (e.g. the output-chirp gauge) require caller validation.
        """
        if self.target_encoder != "residual":
            raise ValueError("Base-state loading requires target_encoder='residual'")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("Expected a nonempty model state dictionary")
        prefix = "transport_target_encoder."
        expected_target = {prefix + key for key in self.transport_target_encoder.base.state_dict()}
        actual_target = {key for key in state if isinstance(key, str) and key.startswith(prefix)}
        if actual_target != expected_target:
            raise ValueError("Base checkpoint must contain the matching U-Net target encoder")
        mapped = {}
        for key, value in state.items():
            if not isinstance(key, str):
                raise ValueError("Model state keys must be strings")
            mapped[prefix + "base." + key[len(prefix):] if key.startswith(prefix) else key] = value
        expected = self.state_dict()
        detail_keys = {key for key in expected if key.startswith(prefix + "detail.")}
        missing = set(expected) - set(mapped)
        unexpected = set(mapped) - set(expected)
        if missing != detail_keys or unexpected:
            raise ValueError("Base checkpoint architecture mismatch: only detail-branch keys may be missing")
        for key, value in mapped.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
                raise ValueError(f"Base checkpoint tensor shape mismatch: {key}")
        loaded = self.load_state_dict(mapped, strict=False)
        if set(loaded.missing_keys) != detail_keys or loaded.unexpected_keys:
            raise RuntimeError("Unexpected residual initialization state")
        self.transport_target_encoder.reset_detail_output()
        return loaded

    def add_coordinates(self, features):
        if self.position == "none":
            return features
        coordinates = self.transport_coordinates.to(features.dtype).expand(features.shape[0], -1, -1, -1)
        return torch.cat((features, coordinates), dim=1)

    def forward(self, images, return_diagnostics=False, return_latent=False):
        """Return phase, or (phase, complex source features) for training losses.

        Latent fields are raw, before optional feature-energy balancing, and
        retain their autograd graph. Diagnostics and latent
        returns are separate contracts and cannot be requested together.
        """
        if return_diagnostics and return_latent:
            raise ValueError("Request either return_diagnostics or return_latent, not both")
        if images.ndim != 4 or images.shape[1:] != (4, 128, 128):
            raise ValueError("images must have shape (batch, 4, 128, 128)")
        if self.target_encoder in ("envelope", "consensus"):
            phase_logits, envelope_delta = self.transport_target_encoder(self.add_coordinates(images))
            target_phase = math.pi * phase_logits.float()
            field_envelope = images.float() + envelope_delta.float()
            if self.target_encoder == "consensus":
                # Internal feature preconditioning only: the original targets
                # and physical evaluation stay unchanged. The free gain can
                # attenuate or amplify depth differences; no clamp or teacher.
                difference = images.float().mean(dim=1, keepdim=True) - images.float()
                field_envelope = field_envelope + self.transport_consensus_gain * difference
        else:
            target_phase = math.pi * self.transport_target_encoder(self.add_coordinates(images)).float()
            field_envelope = images.float()
        target_phase = target_phase + self.transport_target_bias
        target_field = field_envelope * torch.exp(1j * target_phase)
        source_features = self.transport_operator.adjoint(target_field)
        decoder_fields = balanced_source_features(source_features, images) if self.energy_balanced else source_features
        features = torch.cat((decoder_fields.real, decoder_fields.imag), dim=1)
        logits = self.transport_phase_encoder(self.add_coordinates(features))
        if self.transport_linear_skip is not None:
            logits = logits + self.transport_linear_skip(features)
        phase = phase_from_logits(logits, self.phase_head, self.radius_eps, return_diagnostics)
        return (phase, source_features) if return_latent else phase
