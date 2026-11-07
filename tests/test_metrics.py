from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_holography.metrics import calc_metrics, global_ssim_per_plane, local_ssim_per_plane, pearson_per_plane


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.reference = np.load(ROOT / "tests" / "fixtures" / "notebook_cpu.npz")

    def test_matches_notebook(self):
        p = torch.from_numpy(self.reference["metric_prediction"])
        t = torch.from_numpy(self.reference["targets"])
        for name, function in (("cc", pearson_per_plane), ("local", local_ssim_per_plane), ("global", global_ssim_per_plane)):
            with self.subTest(metric=name):
                np.testing.assert_allclose(function(p, t).numpy(), self.reference[name], rtol=1e-6, atol=1e-6)

    def test_identity_and_constant_convention(self):
        t = torch.from_numpy(self.reference["targets"])
        metrics = calc_metrics(t, t)
        for key in ("cc_mean", "local_mean", "global_mean"):
            self.assertAlmostEqual(metrics[key], 1, places=5)
        zeros = torch.zeros_like(t)
        self.assertEqual(pearson_per_plane(zeros, zeros).abs().sum().item(), 0)
        torch.testing.assert_close(local_ssim_per_plane(zeros, zeros), torch.ones(1, 4))
