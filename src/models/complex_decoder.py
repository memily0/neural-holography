"""A phase-equivariant learned decoder for transported complex features.

Complex mixing preserves global phase covariance. Optional 3x3 residual
mixes also learn from neighboring phases; input/output mixes remain 1x1.
Real magnitude gates provide learned nonlinear context without mixing axes.
All mixing coefficients are randomly initialized, shared learned weights.
This module returns two learned logits; it does not extract a mask from an
input field. Conceptual references (fresh implementation, no copied code):
https://arxiv.org/abs/1705.09792 and https://arxiv.org/abs/2112.01525.
"""

from collections.abc import Mapping
import re

import torch
from torch import nn
from torch.nn import functional as F


class ComplexPointwise(nn.Module):
    """Complex-linear mixing, pointwise by default, in paired real tensors."""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        if type(kernel_size) is not int or kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        self.kernel_size = kernel_size
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, features):
        real = torch.cat((self.real.weight, -self.imag.weight), dim=1)
        imag = torch.cat((self.imag.weight, self.real.weight), dim=1)
        weights = torch.cat((real, imag), dim=0)
        if self.kernel_size == 1:
            return F.conv2d(features, weights)
        # Keep the old 1x1 arithmetic at initialization while allowing nonzero
        # gradients to every initially zero neighboring coefficient.
        center = weights[:, :, 1:2, 1:2].contiguous()
        neighbors = weights - F.pad(center, (1, 1, 1, 1))
        return F.conv2d(features, center) + F.conv2d(features, neighbors, padding=1)


class ComplexMagnitudeGate(nn.Module):
    """A real gate depending only on magnitudes and optional coordinates."""

    def __init__(self, channels, coordinates, spatial_context):
        super().__init__()
        kernel = 3 if spatial_context else 1
        self.context = nn.Conv2d(channels + coordinates, channels, kernel, padding=kernel // 2)

    def forward(self, features, coordinates):
        real, imag = features.chunk(2, dim=1)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        context = torch.cat((torch.log1p(magnitude), coordinates), dim=1)
        gate = 2 * torch.sigmoid(self.context(context))
        return torch.cat((real * gate, imag * gate), dim=1)


class ComplexPhaseEncoder(nn.Module):
    """Map four complex fields (+ optional XY) to two learned phase logits.

    Input order is real[4], imag[4], optional XY[2]. Coordinates only condition
    magnitude gates. For any common scalar phase theta the output obeys
    f(exp(i*theta) * fields, XY) = exp(i*theta) * f(fields, XY).
    The circular output head therefore changes only the physically irrelevant
    global mask phase. There is no spatial pooling, interpolation, real/imag
    GroupNorm, fixed field summation, or analytic phase-only encoding.
    """

    def __init__(self, in_channels=10, out_channels=2, width=16, spatial_context=True, depth=3, mix_kernel=1):
        super().__init__()
        if in_channels not in (8, 10):
            raise ValueError("in_channels must be 8 complex components or 8 plus XY")
        if out_channels != 2:
            raise ValueError("complex decoder requires the two-channel circular phase head")
        if not isinstance(width, int) or width < 1 or not isinstance(depth, int) or depth < 1:
            raise ValueError("width and depth must be positive integers")
        if type(mix_kernel) is not int or mix_kernel not in (1, 3):
            raise ValueError("mix_kernel must be 1 or 3")
        self.in_channels = in_channels
        self.mix_kernel = mix_kernel
        self.input = ComplexPointwise(4, width)
        self.mixes = nn.ModuleList([ComplexPointwise(width, width, mix_kernel) for _ in range(depth)])
        self.gates = nn.ModuleList([
            ComplexMagnitudeGate(width, in_channels - 8, spatial_context) for _ in range(depth)
        ])
        self.output = ComplexPointwise(width, 1)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError("features must have shape [batch, in_channels, height, width]")
        fields, coordinates = features[:, :8], features[:, 8:]
        # One real scalar per example preserves phase covariance and does not
        # collect running statistics or change any parameter during inference.
        rms = fields.square().mean((1, 2, 3), keepdim=True).add(1e-8).sqrt()
        encoded = self.input(fields / rms)
        for mix, gate in zip(self.mixes, self.gates):
            encoded = encoded + gate(mix(encoded), coordinates)
        return self.output(encoded)


def expand_complex_mixing_state(state, target_state):
    """Clone a decoder state, expanding only residual mixes from 1x1 to 3x3.

    Pass unprefixed decoder state dictionaries, with target_state taken from
    the desired mix_kernel=3 decoder. Keys, widths, depth, coordinates, gate
    kernels and dtypes must match. Old coefficients become the center; all
    eight neighbors start at zero. Other tensors are cloned unchanged, on
    their original devices. Neither dictionary, any module nor RNG is changed.

    No optimizer state is migrated. A caller loading a larger model must also
    validate its remaining architecture/configuration before replacing this
    decoder's state. There are no analytic phase or carrier coefficients.
    """
    for mapping in (state, target_state):
        if not isinstance(mapping, Mapping) or not mapping or any(not isinstance(key, str) for key in mapping):
            raise ValueError("Expected nonempty decoder state dictionaries with string keys")
        if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.layout != torch.strided for value in mapping.values()):
            raise ValueError("Decoder state values must be real floating-point dense tensors")
    if state.keys() != target_state.keys():
        raise ValueError("Source and target decoder state keys must match exactly")
    pattern = re.compile(r"mixes\.(0|[1-9][0-9]*)\.(real|imag)\.weight")
    matches = {key: pattern.fullmatch(key) for key in state}
    mix_keys = {key for key, match in matches.items() if match}
    indices = {int(matches[key].group(1)) for key in mix_keys}
    if not indices or indices != set(range(len(indices))):
        raise ValueError("Residual mix indices must form a nonempty consecutive sequence")
    expected_keys = {f"{layer}.{part}.weight" for layer in ("input", "output") for part in ("real", "imag")}
    expected_keys |= {f"mixes.{index}.{part}.weight" for index in indices for part in ("real", "imag")}
    expected_keys |= {f"gates.{index}.context.{part}" for index in indices for part in ("weight", "bias")}
    if state.keys() != expected_keys:
        raise ValueError("Expected the complete ComplexPhaseEncoder decoder state without prefixes or extra keys")
    first = state["input.real.weight"]
    if first.ndim != 4 or first.shape[1:] != (4, 1, 1) or first.shape[0] < 1:
        raise ValueError("Invalid input complex-mixing shape")
    width = first.shape[0]
    gate_shape = state["gates.0.context.weight"].shape
    if len(gate_shape) != 4 or gate_shape[0] != width or gate_shape[1] not in (width, width + 2) or gate_shape[2:] not in ((1, 1), (3, 3)):
        raise ValueError("Invalid magnitude-gate shape")
    for key, value in state.items():
        expected = target_state[key]
        if value.dtype != first.dtype or expected.dtype != value.dtype:
            raise ValueError(f"Decoder dtype mismatch: {key}")
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise ValueError(f"Nonfinite decoder tensor: {key}")
        if key in mix_keys:
            source_shape, target_shape = (width, width, 1, 1), (width, width, 3, 3)
        elif key.startswith("input."):
            source_shape = target_shape = (width, 4, 1, 1)
        elif key.startswith("output."):
            source_shape = target_shape = (1, width, 1, 1)
        else:
            source_shape = target_shape = gate_shape if key.endswith("weight") else (width,)
        if value.shape != source_shape or expected.shape != target_shape:
            raise ValueError(f"Decoder architecture/tensor shape mismatch: {key}")
    expanded = {}
    for key, value in state.items():
        if key in mix_keys:
            expanded[key] = value.detach().new_zeros(target_state[key].shape)
            expanded[key][:, :, 1:2, 1:2] = value.detach()
        else:
            expanded[key] = value.detach().clone()
    return expanded
