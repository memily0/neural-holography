import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from neural_holography.config import DEFAULT_CHECKPOINT, DISTANCES
from neural_holography.inference import select_device
from neural_holography.metrics import calc_metrics
from neural_holography.model import load_model, phase_from_ready_unet
from neural_holography.physics import norm_amp, reconstruct_four
from neural_holography.preprocessing import load_inputs, phase_from_png, phase_to_png


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved 8-bit DOE reconstructions")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs=4, type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.repeat < 1 or args.threads < 1:
        parser.error("--repeat and --threads must be positive")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    load_start = time.perf_counter()
    model, metadata = load_model(args.checkpoint, device)
    load_seconds = time.perf_counter() - load_start
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        groups = [[args.manifest.parent / name for name in group] for group in manifest["quartets"]]
        source_name = manifest.get("source", str(args.manifest))
    else:
        groups = [args.input]
        source_name = "user supplied images"
    if not groups:
        raise ValueError("Benchmark requires at least one quartet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output.parent / (args.output.stem + "_doe")
    records = []
    network_times, pipeline_times = [], []
    with torch.inference_mode():
        warmup = load_inputs(groups[0]).to(device)
        phase_from_ready_unet(model, warmup)
        synchronize(device)
        for index, paths in enumerate(groups):
            target = load_inputs(paths).to(device)
            path = artifact_dir / f"{index + 1:03d}.png"
            timings = []
            for _ in range(args.repeat):
                synchronize(device)
                pipeline_start = time.perf_counter()
                inputs = load_inputs(paths).to(device)
                synchronize(device)
                start = time.perf_counter()
                phase = phase_from_ready_unet(model, inputs)
                synchronize(device)
                elapsed = time.perf_counter() - start
                network_times.append(elapsed)
                timings.append(elapsed)
                phase_to_png(phase[0, 0], path)
                saved_phase = phase_from_png(path, device)
                synchronize(device)
                pipeline_times.append(time.perf_counter() - pipeline_start)
            restored = norm_amp(reconstruct_four(saved_phase))
            if not torch.isfinite(restored).all():
                raise FloatingPointError("Nonfinite reconstruction")
            records.append({
                "quartet": index + 1,
                "inputs": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                "metrics": calc_metrics(restored, target),
                "neural_seconds": timings,
            })
    keys = records[0]["metrics"]
    means = {key: np.mean([record["metrics"][key] for record in records], axis=0).tolist() for key in keys}
    report = {
        "source": source_name,
        "official_score": False,
        "quartets": len(groups),
        "repeats_per_quartet": args.repeat,
        "warmup_quartets": 1,
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": metadata.get("epoch"),
        "distances_mm": [z * 1000 for z in DISTANCES],
        "quantization": "metrics from reloaded uint8 PNG, no phase optimization",
        "metrics": means,
        "timing": {
            "checkpoint_load_seconds": load_seconds,
            "neural_seconds_mean": float(np.mean(network_times)),
            "neural_seconds_median": float(np.median(network_times)),
            "pipeline_seconds_mean": float(np.mean(pipeline_times)),
            "pipeline_includes": "input IO, preprocessing, neural synthesis, PNG write and read; excludes Fresnel and metrics",
        },
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
