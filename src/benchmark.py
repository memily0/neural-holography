"""Benchmark a portable checkpoint on four existing PNGs without training."""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import PIL
import torch

from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.predict import file_hash, read_input
from src.protocol import LEGACY_PROTOCOL, PROTOCOLS, prepare_reconstruction, prepare_target


REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def processor_name():
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(operation, device, repeats, warmups, batch_size):
    for _ in range(warmups):
        operation()
    synchronize(device)
    elapsed = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter_ns()
        operation()
        synchronize(device)
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    median = float(np.median(elapsed))
    return {
        "unit": "milliseconds_per_batch", "median": median,
        "p95": float(np.quantile(elapsed, 0.95, method="linear")),
        "mean": float(np.mean(elapsed)), "minimum": float(min(elapsed)),
        "maximum": float(max(elapsed)), "samples": elapsed,
        "quartets_per_second_at_median": batch_size * 1000 / median,
    }


def cold_cli(checkpoint, inputs, device, protocol, expected_codes):
    """A fresh process, not an OS page-cache flush or cold hardware measurement."""
    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="holography_benchmark_", dir=temporary_root) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != temporary_root:
            raise RuntimeError("Unexpected temporary output location")
        output = temporary / "prediction"
        command = [sys.executable, "-m", "src.predict", "--checkpoint", str(checkpoint.resolve()),
                   "--inputs", *[str(path.resolve()) for path in inputs], "--output", str(output),
                   "--device", str(device), "--protocol", protocol]
        started = time.perf_counter_ns()
        process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        seconds = (time.perf_counter_ns() - started) / 1e9
        if process.returncode:
            raise RuntimeError(f"Cold prediction CLI failed with exit code {process.returncode}")
        result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        phase = read_input(output / "phase.png")
        reconstruction = np.stack([read_input(output / f"reconstruction_{plane}.png") for plane in range(1, 5)])
        phase_difference = int(np.abs(phase.astype(np.int16) - expected_codes[0].astype(np.int16)).max())
        reconstruction_difference = int(np.abs(reconstruction.astype(np.int16) - expected_codes[1].astype(np.int16)).max())
    return {
        "wall_seconds": seconds, "runs": 1, "batch_size": 1, "threads": 4,
        "scope": "Fresh Python subprocess: interpreter/imports, checkpoint loading, four PNG reads, inference, saved-and-reread phase/reconstruction PNGs, metric calculation and output writes",
        "cache_state": "New process only; OS file cache and hardware caches are not flushed",
        "temporary_outputs_removed": True, "checkpoint_sha256": result["checkpoint_sha256"],
        "max_phase_code_difference_from_warm_compute": phase_difference,
        "max_reconstruction_code_difference_from_warm_compute": reconstruction_difference,
        "metrics": result["mean"],
    }


@torch.inference_mode()
def benchmark(checkpoint, inputs, device="cpu", repeats=50, warmups=10, batch_size=1, threads=4, protocol=None, include_cold_cli=False):
    if repeats < 1 or warmups < 0 or batch_size < 1 or threads < 1:
        raise ValueError("repeats, batch-size and threads must be positive; warmups nonnegative")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = Path(checkpoint)
    inputs = [Path(path) for path in inputs]
    if len(inputs) != 4:
        raise ValueError("Exactly four PNGs are required, in depth order")
    torch.set_num_threads(threads)
    torch.manual_seed(20261108)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    input_protocol = config.get("protocol", LEGACY_PROTOCOL)
    protocol = protocol or input_protocol
    if input_protocol not in PROTOCOLS or protocol not in PROTOCOLS:
        raise ValueError("Unsupported checkpoint input or reconstruction protocol")
    scales = torch.as_tensor(config["amplitude_scales"], dtype=torch.float32, device=device)
    if scales.shape != (4,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Expected four finite positive amplitude scales")
    scales = scales.reshape(1, 4, 1, 1)
    model = build_model(config["model"], config["width"], config["phase_head"], position=config.get("position", "none"))
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device).eval()
    original = torch.from_numpy(np.stack([read_input(path) for path in inputs])).to(device=device, dtype=torch.float32)[None] / 255
    original = original.repeat(batch_size, 1, 1, 1)
    model_input = prepare_target(original, input_protocol)
    propagator = FresnelPropagator().to(device)

    def model_only():
        return model(model_input)

    def compute_pipeline():
        phase = model(prepare_target(original, input_protocol))
        phase_codes = phase_to_uint8(phase)
        raw = propagator.amplitude(uint8_to_phase(phase_codes)) * scales
        reconstructed_codes = (prepare_reconstruction(raw, protocol) * 255).round().to(torch.uint8)
        return phase_codes, reconstructed_codes, reconstructed_codes.float() / 255

    phase = model_only()
    if phase.shape != (batch_size, 1, 128, 128) or not torch.isfinite(phase).all():
        raise ValueError("Invalid phase output from checkpoint")
    phase_codes, reconstruction_codes, reconstruction = compute_pipeline()
    if not torch.isfinite(reconstruction).all():
        raise ValueError("Nonfinite reconstruction")
    expected_codes = (phase_codes[0, 0].cpu().numpy(), reconstruction_codes[0].cpu().numpy())
    # Both measurements use resident tensors; no disk or host/device copies occur inside them.
    model_timing = measure(model_only, device, repeats, warmups, batch_size)
    pipeline_timing = measure(compute_pipeline, device, repeats, warmups, batch_size)
    hardware = {
        "processor": processor_name(), "logical_cpu_count": os.cpu_count(),
        "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                           "compute_capability": [properties.major, properties.minor], "multiprocessor_count": properties.multi_processor_count}
    config_keys = ("model", "width", "phase_head", "position", "protocol", "amplitude_scales")
    sources = {Path(__file__).resolve(), REPO_ROOT / "src/predict.py", REPO_ROOT / "src/protocol.py",
               REPO_ROOT / "src/metrics.py", REPO_ROOT / "src/latent_losses.py", REPO_ROOT / "src/physics/fresnel.py", *list((REPO_ROOT / "src/models").glob("*.py"))}
    report = {
        "checkpoint": {"path": portable_path(checkpoint), "sha256": file_hash(checkpoint), "step": saved.get("step"), "weights_only_load": True},
        "inputs": [{"path": portable_path(path), "sha256": file_hash(path)} for path in inputs],
        "config": {key: config[key] for key in config_keys if key in config},
        "model_input_protocol": input_protocol, "reconstruction_protocol": protocol,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "dtype": "float32", "batch_size": batch_size, "repeats": repeats,
        "warmups_per_measurement": warmups, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "seed": 20261108,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "pillow": PIL.__version__, "torch_cuda_build": torch.version.cuda, "hardware": hardware},
        "backend": {"cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "timing": {"clock": "time.perf_counter_ns", "cuda_synchronization": "before and after every measured operation when CUDA is selected",
                   "execution": "eval mode, torch.inference_mode, no autocast, no compilation",
                   "order": "model-only then compute pipeline; each has its own warmups",
                   "input_reuse": "One fixed quartet; batches larger than one contain repeated copies",
                   "model_only_scope": "One model forward on resident, preprocessed input tensors",
                   "compute_pipeline_scope": "Input normalization, one model forward, phase uint8 encode/decode, four-plane Fresnel amplitude, amplitude normalization, reconstruction uint8 encode/decode",
                   "compute_excludes": "Disk IO, PNG serialization, input host/device transfer, model loading, metric calculations",
                   "limitations": "Observed warm latency on this host; other processes, power and OS scheduling can affect timings. p95 describes these repeated measurements, not a confidence interval."},
        "model_only": model_timing, "compute_pipeline": pipeline_timing,
        "source_sha256": {portable_path(path): file_hash(path) for path in sorted(sources)},
    }
    if include_cold_cli:
        synchronize(device)
        report["cold_cli"] = cold_cli(checkpoint, inputs, device, protocol, expected_codes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--cold-cli", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.checkpoint, args.inputs, args.device, args.repeats, args.warmups, args.batch_size, args.threads, args.protocol, args.cold_cli)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": portable_path(args.output), "model_median_ms": result["model_only"]["median"],
                      "pipeline_median_ms": result["compute_pipeline"]["median"], "cold_cli_seconds": result.get("cold_cli", {}).get("wall_seconds")}))


if __name__ == "__main__":
    main()
