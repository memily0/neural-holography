"""Export a trusted research checkpoint as a portable inference artifact."""

import argparse
import hashlib
import json
import math
import numbers
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator
from src.protocol import LEGACY_PROTOCOL, PROTOCOLS


TRAINING_FIELDS = (
    "seed", "data_seed", "steps", "batch_size", "lr", "ssim_weight",
    "pearson_weight", "rank_weight", "rank_margin", "synthetic_steps",
    "lr_schedule", "warmup_steps", "lr_min_ratio", "cosine_weight",
    "balanced_weight", "concentration_weight", "latent_weight", "depth_frequency_weight", "worst_weight",
    "quantize_train", "amp", "fixed_train_quartets", "detach_peak_gradient",
    "resume_mode", "resume_checkpoint_sha256",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_json(value):
    """Strip scalar subclasses and reject objects requiring custom unpicklers."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Metadata contains a nonfinite number")
        return result
    if isinstance(value, (tuple, list)):
        return [plain_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Metadata keys must be strings")
        return {str(key): plain_json(item) for key, item in value.items()}
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def inference_config(config):
    result = plain_json({
        "model": config["model"],
        "width": config["width"],
        "phase_head": config["phase_head"],
        "position": config.get("position", "none"),
        "protocol": config.get("protocol", LEGACY_PROTOCOL),
        "amplitude_scales": config["amplitude_scales"],
    })
    if result["protocol"] not in PROTOCOLS:
        raise ValueError("Checkpoint uses an unsupported evaluation protocol")
    if type(result["width"]) is not int or result["width"] <= 0:
        raise ValueError("Model width must be a positive integer")
    scales = result["amplitude_scales"]
    if not isinstance(scales, list) or len(scales) != 4:
        raise ValueError("Expected four positive amplitude scales")
    if any(type(item) not in (float, int) or item <= 0 for item in scales):
        raise ValueError("Expected four positive amplitude scales")
    return result


def portable_state_dict(state):
    result = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Model state must map names to tensors")
        dtype = torch.complex64 if value.is_complex() else torch.float32 if value.is_floating_point() else value.dtype
        tensor = value.detach().to(device="cpu", dtype=dtype).contiguous().clone()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite model tensor: {name}")
        result[str(name)] = tensor
    return result


@torch.inference_mode()
def validate_model(config, state):
    model = build_model(config["model"], config["width"], config["phase_head"], position=config["position"]).float().eval()
    model.load_state_dict(state, strict=True)
    generator = torch.Generator(device="cpu").manual_seed(0)
    phase = model(torch.rand(1, 4, 128, 128, generator=generator))
    if phase.shape != (1, 1, 128, 128) or not torch.isfinite(phase).all():
        raise ValueError("Exported model did not produce a finite 128x128 phase")
    if phase.abs().max().item() > math.pi + 1e-6:
        raise ValueError("Exported phase is outside [-pi, pi]")
    return {
        "class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_tensors": len(state),
        "state_dtypes": sorted({str(value.dtype) for value in state.values()}),
        "input_shape": [1, 4, 128, 128],
        "output_shape": list(phase.shape),
        "phase_range_radians": [-math.pi, math.pi],
    }


def source_hashes():
    paths = set((PROJECT_ROOT / "src" / "models").glob("*.py"))
    paths.update((PROJECT_ROOT / "src" / "physics").glob("*.py"))
    paths.update(PROJECT_ROOT / "src" / name for name in ("predict.py", "protocol.py", "metrics.py", "latent_losses.py"))
    paths.add(Path(__file__).resolve())
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_hash(path) for path in sorted(paths)}


def export_solution(checkpoint, output_dir=Path("models"), overwrite=False):
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    weights_path, metadata_path = output_dir / "solution.pt", output_dir / "metadata.json"
    if checkpoint.resolve() in (weights_path.resolve(), metadata_path.resolve()):
        raise ValueError("Export must not overwrite its source checkpoint")
    if not overwrite and (weights_path.exists() or metadata_path.exists()):
        raise FileExistsError("Export already exists; choose another directory or pass --overwrite")
    source_sha = file_hash(checkpoint)
    # Research checkpoints may contain TorchVersion/config objects: only use trusted local files.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["config"].get("diagnostic_only", False) or saved["config"].get("fixed_train_quartets", 0) > 0:
        raise ValueError("Diagnostic or fixed-quartet memorization checkpoints cannot be exported as a solution")
    config = inference_config(saved["config"])
    step = plain_json(saved["step"])
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    state = portable_state_dict(saved["model"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    try:
        model_summary = validate_model(config, state)
    finally:
        torch.set_num_threads(previous_threads)
    provenance = {
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_name": checkpoint.name,
        "source_run": checkpoint.parent.name,
        "selected_step": step,
    }
    payload = {"format_version": 1, "model": state, "config": config, "step": step, "provenance": provenance}
    propagator = FresnelPropagator()
    metadata = {
        "format_version": 1,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": weights_path.name,
        "protocol": config["protocol"],
        "selected_step": step,
        "source_checkpoint_sha256": source_sha,
        "provenance": provenance,
        "config": config,
        "model_summary": model_summary,
        "physics": {
            "size": propagator.size,
            "wavelength_m": propagator.wavelength,
            "pixel_pitch_m": propagator.pitch,
            "distances_m": propagator.distances.tolist(),
            "phase_png": "uint8 round((phase + pi) * 255 / (2 * pi)); 0=-pi, 255=pi",
            "reconstruction": "Amplitude abs(E); evaluated after saved phase and reconstruction PNG quantization",
        },
        "training_hyperparameters": plain_json({name: saved["config"][name] for name in TRAINING_FIELDS if name in saved["config"]}),
        "export_environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "source_sha256": source_hashes(),
        "validation": {"strict_model_state_load": True, "finite_forward": True, "weights_only_load": True},
        "optimizer_included": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_temp = output_dir / "solution.pt.partial"
    metadata_temp = output_dir / "metadata.json.partial"
    if weights_temp.exists() or metadata_temp.exists():
        raise FileExistsError("A partial export already exists; inspect it before retrying")
    try:
        torch.save(payload, weights_temp)
        restored = torch.load(weights_temp, map_location="cpu", weights_only=True)
        if restored["config"] != config or restored["step"] != step:
            raise RuntimeError("Portable checkpoint round-trip changed metadata")
        for name, tensor in state.items():
            if not torch.equal(restored["model"][name], tensor):
                raise RuntimeError(f"Portable checkpoint round-trip changed tensor: {name}")
        metadata["checkpoint_sha256"] = file_hash(weights_temp)
        metadata["checkpoint_bytes"] = weights_temp.stat().st_size
        metadata_temp.write_text(json.dumps(plain_json(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if file_hash(checkpoint) != source_sha:
            raise RuntimeError("Source checkpoint changed while it was being exported")
        weights_temp.replace(weights_path)
        metadata_temp.replace(metadata_path)
    finally:
        weights_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local research checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = export_solution(args.checkpoint, args.output_dir, args.overwrite)
    print(json.dumps({"output": str(args.output_dir), "step": metadata["selected_step"], "sha256": metadata["checkpoint_sha256"], "protocol": metadata["protocol"]}))


if __name__ == "__main__":
    main()
