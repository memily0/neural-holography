from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_holography.config import DISTANCES, PIXEL_PITCH, WAVELENGTH
from neural_holography.physics import fresnel_field, norm_amp, physics_seed, reconstruct_four


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.reference = np.load(ROOT / "tests" / "fixtures" / "notebook_cpu.npz")

    def test_constants(self):
        self.assertEqual(DISTANCES, (0.0241, 0.0253, 0.0265, 0.0277))
        self.assertEqual(WAVELENGTH, 532e-9)
        self.assertEqual(PIXEL_PITCH, 10e-6)

    def test_field_matches_notebook_at_every_plane(self):
        phase = torch.from_numpy(self.reference["phase"])
        for i, z in enumerate(DISTANCES):
            with self.subTest(distance=z):
                np.testing.assert_allclose(fresnel_field(phase, z).numpy(), self.reference[f"field_{i}"], rtol=1e-6, atol=1e-6)

    def test_seed_matches_notebook(self):
        targets = torch.from_numpy(self.reference["targets"])
        np.testing.assert_allclose(physics_seed(targets).numpy(), self.reference["seed"], rtol=1e-6, atol=1e-6)

    def test_reconstruction_is_normalized_amplitude(self):
        phase = torch.from_numpy(self.reference["phase"])
        expected = np.concatenate([np.abs(self.reference[f"field_{i}"]) for i in range(4)], axis=1)
        actual = reconstruct_four(phase)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
        normalized = norm_amp(actual)
        self.assertTrue(torch.isfinite(normalized).all().item())
        torch.testing.assert_close(normalized.amax((-2, -1)), torch.ones(1, 4))

    def test_zero_seed_and_normalization_remain_finite(self):
        targets = torch.zeros(1, 4, 128, 128)
        self.assertEqual(physics_seed(targets).abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(norm_amp(targets)).all().item())

    def test_fresnel_retains_autograd(self):
        phase = torch.from_numpy(self.reference["phase"]).requires_grad_()
        reconstruct_four(phase).mean().backward()
        self.assertTrue(torch.isfinite(phase.grad).all().item())
        self.assertGreater(phase.grad.abs().sum().item(), 0)

    def test_invalid_phase_is_rejected(self):
        for phase in (torch.zeros(128, 128), torch.zeros(1, 2, 128, 128), torch.zeros(1, 1, 128, 64)):
            with self.assertRaises(ValueError):
                fresnel_field(phase, DISTANCES[0])
        with self.assertRaises(TypeError):
            fresnel_field(torch.zeros(1, 1, 128, 128, dtype=torch.uint8), DISTANCES[0])
        with self.assertRaises(ValueError):
            fresnel_field(torch.zeros(1, 1, 128, 128), 0)
