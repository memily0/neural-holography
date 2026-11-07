import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_holography.config import DEFAULT_CHECKPOINT
from neural_holography.inference import predict, select_device
from neural_holography.model import load_model
from neural_holography.physics import norm_amp, reconstruct_four
from neural_holography.preprocessing import load_inputs, phase_from_png, phase_to_png


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model, _ = load_model(DEFAULT_CHECKPOINT, "cpu")
        cls.inputs = [ROOT / "examples" / f"{i}.png" for i in range(1, 5)]

    def test_cpu_repeatability_and_saved_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.png", Path(tmp) / "b.png"
            recon_dir = Path(tmp) / "recon"
            predict(self.inputs, first, self.model, "cpu", recon_dir)
            predict(self.inputs, second, self.model, "cpu")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with Image.open(first) as image:
                pixels = np.asarray(image)
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (128, 128))
                self.assertEqual(pixels.dtype, np.uint8)
                self.assertGreaterEqual(int(pixels.min()), 0)
                self.assertLessEqual(int(pixels.max()), 255)
            phase = phase_from_png(first)
            recon = norm_amp(reconstruct_four(phase))
            self.assertTrue(torch.isfinite(recon).all().item())
            for i in range(4):
                with Image.open(recon_dir / f"reconstructed_{i + 1}.png") as image:
                    expected = (recon[0, i].numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    np.testing.assert_array_equal(np.asarray(image), expected)

    def test_input_normalization(self):
        target = load_inputs(self.inputs)
        self.assertEqual(tuple(target.shape), (1, 4, 128, 128))
        self.assertEqual(target.dtype, torch.float32)
        torch.testing.assert_close(target.amax((-2, -1)), torch.ones(1, 4))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            load_inputs(self.inputs[:3])
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "invalid.png"
            for mode, size, fmt in (("RGB", (128, 128), "PNG"), ("L", (64, 128), "PNG"), ("L", (128, 128), "JPEG"), ("I;16", (128, 128), "PNG")):
                Image.new(mode, size).save(invalid, format=fmt)
                with self.assertRaises(ValueError):
                    load_inputs([invalid] + self.inputs[1:])
            invalid.write_bytes(b"not an image")
            with self.assertRaises((OSError, ValueError)):
                load_inputs([invalid] + self.inputs[1:])
            invalid.unlink()
            with self.assertRaises(FileNotFoundError):
                load_inputs([invalid] + self.inputs[1:])

    def test_black_targets_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "black.png"
            Image.new("L", (128, 128)).save(path)
            targets = load_inputs([path] * 4)
            self.assertEqual(targets.abs().sum().item(), 0)
            predict([path] * 4, Path(tmp) / "doe.png", self.model, "cpu")

    def test_phase_byte_decoding_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase.png"
            pixels = np.tile(np.arange(256, dtype=np.uint8), 64).reshape(128, 128)
            Image.fromarray(pixels).save(path)
            decoded = phase_from_png(path)
            self.assertAlmostEqual(decoded.min().item(), -np.pi, places=6)
            self.assertAlmostEqual(decoded.max().item(), np.pi, places=6)
            phase = torch.linspace(-3.14, 3.14, 128 * 128).reshape(128, 128)
            phase_to_png(phase, path)
            restored = phase_from_png(path)[0, 0]
            self.assertLessEqual((restored - phase).abs().max().item(), np.pi / 255 + 1e-6)
            with self.assertRaises(ValueError):
                phase_to_png(torch.full((128, 128), float("inf")), path)

    def test_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "doe.png"
            command = [sys.executable, str(ROOT / "scripts" / "predict.py"), "--input", *map(str, self.inputs), "--output", str(output), "--device", "cpu"]
            result = subprocess.run(command, cwd=tmp, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertTrue(output.is_file())

    def test_cpu_selection(self):
        self.assertEqual(select_device("cpu").type, "cpu")
        with self.assertRaises(ValueError):
            select_device("invalid")

    def test_cli_works_with_network_connections_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "offline.png"
            script = "import socket,sys,runpy; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            command = [sys.executable, "-c", script, str(ROOT / "scripts" / "predict.py"), "--input", *map(str, self.inputs), "--output", str(output), "--device", "cpu"]
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertTrue(output.is_file())

    def test_input_cannot_be_overwritten(self):
        before = self.inputs[0].read_bytes()
        with self.assertRaises(ValueError):
            predict(self.inputs, self.inputs[0], self.model, "cpu")
        self.assertEqual(before, self.inputs[0].read_bytes())

    def test_benchmark_cli_evaluates_saved_png(self):
        import json

        from neural_holography.metrics import calc_metrics

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.json"
            command = [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--input", *map(str, self.inputs), "--output", str(output), "--device", "cpu", "--repeat", "1"]
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["quartets"], 1)
            self.assertFalse(report["official_score"])
            self.assertEqual(len(report["metrics"]["cc_planes"]), 4)
            self.assertGreater(report["timing"]["neural_seconds_mean"], 0)
            restored = norm_amp(reconstruct_four(phase_from_png(Path(tmp) / "benchmark_doe" / "001.png")))
            self.assertEqual(report["metrics"], calc_metrics(restored, load_inputs(self.inputs)))
