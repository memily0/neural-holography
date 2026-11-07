import math
import torch

from src.physics.fresnel import DISTANCES, fresnel_field, reconstruct_four


def test_shapes_values_and_gradients():
    phase = (torch.rand(2, 1, 128, 128) * 2 * math.pi - math.pi).requires_grad_()
    reconstructed = reconstruct_four(phase)
    assert reconstructed.shape == (2, 4, 128, 128)
    assert torch.isfinite(reconstructed).all()
    loss = (reconstructed - torch.rand_like(reconstructed)).square().mean()
    loss.backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()
    assert phase.grad.abs().sum() > 0


def test_energy_conservation_single_plane():
    phase = torch.randn(1, 1, 128, 128)
    wavelength, pitch, distance = 532e-9, 10e-6, DISTANCES[0]
    field = fresnel_field(phase, distance, wavelength, pitch)
    output_pitch = wavelength * distance / (128 * pitch)
    input_energy = 128 * 128 * pitch**2
    output_energy = field.abs().square().sum().item() * output_pitch**2
    assert math.isclose(input_energy, output_energy, rel_tol=1e-4)
