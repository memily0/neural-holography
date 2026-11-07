from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_holography.config import ARCHITECTURE, DEFAULT_CHECKPOINT
from neural_holography.model import load_model, phase_from_ready_unet


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        with patch.object(socket.socket, "connect", side_effect=RuntimeError("Network disabled")), patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Download disabled")):
            cls.model, cls.metadata = load_model(DEFAULT_CHECKPOINT, "cpu")
        cls.reference = np.load(ROOT / "tests" / "fixtures" / "notebook_cpu.npz")

    def test_actual_checkpoint_architecture(self):
        self.assertEqual(self.metadata["architecture"], ARCHITECTURE)
        self.assertEqual(tuple(self.model.encoder.conv1.weight.shape), (64, 8, 7, 7))
        self.assertEqual(self.model.segmentation_head[0].out_channels, 2)
        self.assertFalse(self.model.training)
        self.assertEqual(len(self.model.state_dict()), 350)

    def test_neural_phase_matches_notebook(self):
        targets = torch.from_numpy(self.reference["targets"])
        with torch.inference_mode():
            actual = phase_from_ready_unet(self.model, targets)
        expected = torch.from_numpy(self.reference["neural_phase"])
        torch.testing.assert_close(torch.exp(1j * actual), torch.exp(1j * expected), rtol=1e-5, atol=2e-5)

    def test_missing_and_incompatible_checkpoint_fail(self):
        with self.assertRaises(FileNotFoundError):
            load_model(ROOT / "checkpoints" / "absent.pt", "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.pt"
            torch.save({"architecture": "other", "model": {}}, path)
            with self.assertRaises(ValueError):
                load_model(path, "cpu")
            torch.save({"architecture": ARCHITECTURE, "model": {}}, path)
            with self.assertRaises(RuntimeError):
                load_model(path, "cpu")

    def test_invalid_targets_fail(self):
        for targets in (torch.zeros(1, 3, 128, 128), torch.full((1, 4, 128, 128), float("nan"))):
            with self.assertRaises(ValueError):
                phase_from_ready_unet(self.model, targets)
