import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.metrics import global_ssim, pearson, ssim
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import LEGACY_PROTOCOL, PROTOCOLS, prepare_target, prepare_reconstruction


def read_input(path):
    path = Path(path)
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (128, 128):
            raise ValueError(f"Expected 128x128 grayscale 8-bit PNG: {path}")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Animated PNG is not an input image: {path}")
        values = np.array(image)
    if values.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels: {path}")
    return values


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def predict(checkpoint, inputs, output, device="cpu", protocol=None):
    torch.set_num_threads(4)
    checkpoint, output = Path(checkpoint), Path(output)
    inputs = [Path(path) for path in inputs]
    if len(inputs) != 4:
        raise ValueError("Exactly four input PNGs are required, in depth order")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    targets = np.stack([read_input(path) for path in inputs])
    # Checkpoints are local training artifacts produced by src.train.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    input_protocol = cfg.get("protocol", LEGACY_PROTOCOL)
    protocol = protocol or input_protocol
    if input_protocol not in PROTOCOLS or protocol not in PROTOCOLS:
        raise ValueError("Checkpoint uses an unsupported evaluation protocol")
    scales = torch.as_tensor(cfg["amplitude_scales"], dtype=torch.float32, device=device)
    if scales.shape != (4,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Checkpoint must contain four finite positive amplitude_scales")
    model = build_model(cfg["model"], cfg["width"], cfg["phase_head"], position=cfg.get("position", "none"))
    model.load_state_dict(saved["model"])
    model = model.to(device).eval()
    target = torch.from_numpy(targets).to(device=device, dtype=torch.float32)[None] / 255
    phase = model(prepare_target(target, input_protocol))
    target = prepare_target(target, protocol)
    if not torch.isfinite(phase).all():
        raise ValueError("Model produced a nonfinite phase")

    output.mkdir(parents=True, exist_ok=False)
    phase_path = output / "phase.png"
    Image.fromarray(phase_to_uint8(phase)[0, 0].cpu().numpy()).save(phase_path)
    phase_codes = torch.from_numpy(read_input(phase_path)).to(device)[None, None]
    decoded_phase = uint8_to_phase(phase_codes)
    propagator = FresnelPropagator().to(device)
    raw = propagator.amplitude(decoded_phase) * scales[None, :, None, None]
    if not torch.isfinite(raw).all():
        raise ValueError("Reconstruction contains nonfinite amplitudes")
    displayed = prepare_reconstruction(raw, protocol)
    codes = (displayed * 255).round().to(torch.uint8)[0].cpu().numpy()
    reconstructed = []
    input_metadata = []
    for number, path in enumerate(inputs, start=1):
        copied_path = output / f"target_{number}.png"
        shutil.copy2(path, copied_path)
        digest = file_hash(path)
        if file_hash(copied_path) != digest:
            raise RuntimeError(f"Input copy verification failed: {path}")
        input_metadata.append({"path": path.name, "copy": copied_path.name, "sha256": digest})
        reconstruction_path = output / f"reconstruction_{number}.png"
        Image.fromarray(codes[number - 1]).save(reconstruction_path)
        reconstructed.append(read_input(reconstruction_path))
    reconstruction = torch.from_numpy(np.stack(reconstructed)).to(device=device, dtype=torch.float32)[None] / 255

    correlations, valid = pearson(reconstruction, target, return_valid=True)
    local_scores = ssim(reconstruction, target)
    global_scores = global_ssim(reconstruction, target)
    squared_errors = (reconstruction - target).square().mean((-2, -1))
    per_plane = []
    for plane in range(4):
        is_valid = bool(valid[0, plane])
        per_plane.append({
            "plane": plane + 1,
            "distance_m": float(propagator.distances[plane]),
            "pearson": float(correlations[0, plane]) if is_valid else None,
            "pearson_valid": is_valid,
            "ssim": float(local_scores[0, plane]),
            "global_ssim": float(global_scores[0, plane]),
            "mse": float(squared_errors[0, plane]),
        })
    result = {
        "protocol": protocol,
        "model_input_protocol": input_protocol,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_hash(checkpoint),
        "checkpoint_step": saved.get("step"),
        "device": device,
        "inputs": input_metadata,
        "phase_png": phase_path.name,
        "phase_sha256": file_hash(phase_path),
        "amplitude_scales": scales.cpu().tolist(),
        "per_plane": per_plane,
        "mean": {
            "pearson": float(correlations.mean()) if bool(valid.all()) else None,
            "ssim": float(local_scores.mean()),
            "global_ssim": float(global_scores.mean()),
            "mse": float(squared_errors.mean()),
        },
        "invalid_pearson_count": int((~valid).sum()),
        "clipped_fraction": float((raw > 1).float().mean()) if protocol == LEGACY_PROTOCOL else 0.0,
        "excess_energy": float((raw.square() - 1).clamp_min(0).mean()) if protocol == LEGACY_PROTOCOL else 0.0,
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="One-pass phase synthesis and PNG-based Fresnel evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protocol", choices=PROTOCOLS)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.inputs, args.output, args.device, args.protocol)
    print(json.dumps({"output": str(args.output.resolve()), "mean": result["mean"], "invalid_pearson_count": result["invalid_pearson_count"]}, allow_nan=False))


if __name__ == "__main__":
    main()
