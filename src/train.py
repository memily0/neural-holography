import argparse
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.data import ImageBank
from src.curriculum import sample_forward_targets
from src.losses import depth_ranking_loss
from src.amplitude_losses import amplitude_cosine_loss, balanced_pixel_mse
from src.shape_losses import excess_concentration_loss
from src.training_schedule import cosine_lr_multiplier
from src.training_normalization import peak_normalize_surrogate
from src.training_probe import fixed_training_manifest
from src.latent_losses import latent_source_consistency
from src.depth_frequency_loss import depth_frequency_loss
from src.metrics import global_ssim, pearson, ssim
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, quantize_phase, uint8_to_phase
from src.protocol import LEGACY_PROTOCOL, PEAK_PROTOCOL, PROTOCOLS, prepare_target, prepare_reconstruction


@dataclass
class Config:
    run: str = "pilot_cnn"
    model: str = "cnn"
    width: int = 16
    phase_head: str = "tanh"
    position: str = "none"
    steps: int = 1000
    batch_size: int = 16
    lr: float = 2e-4
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    lr_min_ratio: float = 0.1
    seed: int = 42
    data_seed: int = 42
    domain: str = "mixed"
    val_per_group: int = 64
    eval_every: int = 250
    ssim_weight: float = 0.1
    pearson_weight: float = 0.1
    worst_weight: float = 0.0
    contrast_weight: float = 0.0
    rank_weight: float = 0.0
    rank_margin: float = 0.2
    cosine_weight: float = 0.0
    balanced_weight: float = 0.0
    concentration_weight: float = 0.0
    latent_weight: float = 0.0
    depth_frequency_weight: float = 0.0
    detach_peak_gradient: bool = False
    save_every_eval: bool = False
    fixed_train_quartets: int = 0
    quantize_train: bool = False
    amp: bool = False
    blur_start: float = 0.0
    synthetic_steps: int = 0
    resume: str = ""
    device: str = "cuda"
    protocol: str = PEAK_PROTOCOL


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def png_roundtrip(values):
    shape = values.shape
    arrays = values.detach().cpu().numpy().reshape(-1, *shape[-2:])
    decoded = []
    for array in arrays:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        buffer.seek(0)
        with Image.open(buffer) as image:
            decoded.append(np.array(image, dtype=np.uint8))
    return torch.from_numpy(np.stack(decoded).reshape(shape)).to(values.device)


def image_metrics(reconstruction, target):
    correlation, valid = pearson(reconstruction, target, return_valid=True)
    return {
        "pearson": correlation,
        "pearson_valid": valid,
        "ssim": ssim(reconstruction, target),
        "global_ssim": global_ssim(reconstruction, target),
        "mse": (reconstruction - target).square().mean((-2, -1)),
    }


def summarize(values):
    result = {}
    for key, array in values.items():
        if key == "pearson_valid":
            result["invalid_pearson_count"] = int((~array).sum())
            continue
        if key == "pearson" and not values["pearson_valid"].all():
            # Undefined values are not silently dropped to improve the score.
            result[key] = None
            result[key + "_per_plane"] = [None if not values["pearson_valid"][:, j].all() else float(array[:, j].mean()) for j in range(4)]
            continue
        result[key] = float(array.mean())
        result[key + "_per_plane"] = array.mean(0).tolist()
        result[key + "_p10"] = float(np.quantile(array.mean(1), 0.1))
    return result


def save_examples(directory, target, phase_codes, reconstructed_codes, display_target=None):
    directory.mkdir(parents=True, exist_ok=True)
    target_codes = (target[0].cpu().numpy() * 255).round().astype(np.uint8)
    phase = phase_codes[0, 0].cpu().numpy()
    reconstruction = reconstructed_codes[0].cpu().numpy()
    Image.fromarray(phase).save(directory / "phase.png")
    for j in range(4):
        Image.fromarray(target_codes[j]).save(directory / f"target_{j + 1}.png")
        Image.fromarray(reconstruction[j]).save(directory / f"reconstruction_{j + 1}.png")
    grid = Image.new("L", (5 * 128, 2 * 128), 255)
    display_codes = target_codes if display_target is None else (display_target[0].cpu().numpy() * 255).round().astype(np.uint8)
    for j in range(4):
        grid.paste(Image.fromarray(display_codes[j]), (j * 128, 0))
        grid.paste(Image.fromarray(reconstruction[j]), (j * 128, 128))
    grid.paste(Image.fromarray(phase), (4 * 128, 128))
    grid.save(directory / "comparison.png")


@torch.no_grad()
def evaluate(model, bank, propagator, scales, manifest, cfg, output=None, diagnostics=False, evaluation_protocol=None):
    model.eval()
    protocol = evaluation_protocol or cfg.protocol
    arrays = {}
    groups = {}
    for group, records in manifest.items():
        batches = {}
        for start in range(0, len(records), cfg.batch_size):
            original_target = bank.fetch(records[start:start + cfg.batch_size], cfg.device)
            target = prepare_target(original_target, protocol)
            phase = model(prepare_target(original_target, cfg.protocol))
            codes = png_roundtrip(phase_to_uint8(phase))
            decoded_phase = uint8_to_phase(codes)
            raw = propagator.amplitude(decoded_phase) * scales
            displayed = prepare_reconstruction(raw, protocol)
            reconstructed_codes = png_roundtrip((displayed * 255).round().to(torch.uint8))
            reconstruction = reconstructed_codes.float() / 255
            scores = image_metrics(reconstruction, target)
            clipping_input = raw if protocol == LEGACY_PROTOCOL else displayed
            scores["clipped_fraction"] = (clipping_input > 1).float().mean((-2, -1))
            scores["excess_energy"] = (clipping_input.square() - 1).clamp_min(0).mean((-2, -1))
            scores["raw_peak"] = raw.amax((-2, -1))
            scores["foreground_mse"] = ((reconstruction - target).square() * (target > 0.1)).sum((-2, -1)) / (target > 0.1).sum((-2, -1)).clamp_min(1)
            if diagnostics:
                float_amp = prepare_reconstruction(propagator.amplitude(phase) * scales, protocol)
                scores["float_ssim"] = ssim(float_amp, target)
                scores["float_pearson"] = pearson(float_amp, target)
                scores["phase_quantized_ssim"] = ssim(displayed, target)
            for key, value in scores.items():
                batches.setdefault(key, []).append(value.cpu().numpy())
            if output is not None and start == 0:
                save_examples(output / "examples" / group, original_target, codes, reconstructed_codes, display_target=target)
        arrays[group] = {key: np.concatenate(value) for key, value in batches.items()}
        groups[group] = summarize(arrays[group])
    pooled = {key: np.concatenate([arrays[group][key] for group in arrays]) for key in next(iter(arrays.values()))}
    result = {"aggregate": summarize(pooled), "groups": groups, "tuples": len(pooled["ssim"]), "protocol": protocol, "model_input_protocol": cfg.protocol}
    if output is not None:
        np.savez_compressed(output / "per_example_metrics.npz", **{f"{group}__{key}": value for group, scores in arrays.items() for key, value in scores.items()})
        write_json(output / "metrics.json", result)
    return result


def snapshot_source(output):
    hashes = {}
    for path in sorted(Path("src").rglob("*.py")):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = output / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(output / "source_hashes.json", hashes)
    return hashes


def train(cfg):
    if cfg.lr_schedule not in ("constant", "cosine"):
        raise ValueError("lr_schedule must be constant or cosine")
    if cfg.lr_schedule == "constant" and cfg.warmup_steps:
        raise ValueError("warmup_steps requires cosine schedule")
    if cfg.lr_schedule == "cosine":
        cosine_lr_multiplier(1, cfg.steps, cfg.warmup_steps, cfg.lr_min_ratio)
    if min(cfg.cosine_weight, cfg.balanced_weight, cfg.concentration_weight, cfg.latent_weight, cfg.depth_frequency_weight) < 0:
        raise ValueError("Auxiliary loss weights must be nonnegative")
    if cfg.fixed_train_quartets < 0 or cfg.fixed_train_quartets % 4:
        raise ValueError("fixed_train_quartets must be nonnegative and divisible by four")
    if cfg.amp and cfg.device != "cuda":
        raise ValueError("amp training is supported on CUDA only")
    if cfg.protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {cfg.protocol}")
    if cfg.detach_peak_gradient and cfg.protocol != PEAK_PROTOCOL:
        raise ValueError("detach_peak_gradient requires peak-normalized v2")
    if not 0 <= cfg.synthetic_steps <= cfg.steps:
        raise ValueError("synthetic_steps must be between zero and steps")
    if cfg.rank_weight < 0 or not 0 <= cfg.rank_margin <= 1:
        raise ValueError("rank_weight must be nonnegative and rank_margin in [0, 1]")
    output = Path("outputs") / cfg.run
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(cfg.model, cfg.width, cfg.phase_head, position=cfg.position)
    if cfg.latent_weight and not hasattr(model, "transport_operator"):
        raise ValueError("latent_weight requires an unwrapped transport model")
    # Match shared initial weights across architectural comparisons.
    base_model = getattr(model, "base_generator", model)
    base_name = cfg.model.removesuffix("_bias")
    if base_name in ("cnn", "fourier", "ae", "local", "attention"):
        torch.manual_seed(cfg.seed)
        reference = build_model("cnn", cfg.width, cfg.phase_head)
        base_model.load_state_dict(reference.state_dict(), strict=False)
        del reference
    elif base_name.startswith("transport"):
        torch.manual_seed(cfg.seed)
        reference_name = "transport_random" if "random" in base_name else "transport"
        changed_target = "subtarget" in base_name or "residualtarget" in base_name
        if changed_target:
            reference_name = base_name.replace("_subtarget", "").replace("_residualtarget", "")
        if "spatial" in base_name:
            reference_name = base_name.replace("_spatial", "")
        if "envelope" in base_name:
            reference_name = base_name.replace("_envelope", "")
        if "consensus" in base_name:
            reference_name = base_name.replace("_consensus", "")
        if "globaltarget" in base_name:
            reference_name = base_name.replace("_globaltarget", "")
        reference = build_model(reference_name, cfg.width, cfg.phase_head, cfg.position)
        if "globaltarget" in base_name:
            base_model.load_globaltarget_state_dict(reference.state_dict())
        elif "consensus" in base_name:
            base_model.load_consensus_state_dict(reference.state_dict())
        elif "envelope" in base_name:
            base_model.load_envelope_state_dict(reference.state_dict())
        elif "spatial" in base_name:
            base_model.load_pointwise_state_dict(reference.state_dict())
        elif "residualtarget" in base_name:
            base_model.load_base_state_dict(reference.state_dict())
        else:
            if not changed_target:
                base_model.transport_target_encoder.load_state_dict(reference.transport_target_encoder.state_dict())
            else:
                base_model.transport_phase_encoder.load_state_dict(reference.transport_phase_encoder.state_dict())
            with torch.no_grad():
                base_model.transport_target_bias.copy_(reference.transport_target_bias)
        del reference
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    resume_mode = None
    resume_sha = None
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        source_config = checkpoint["config"]
        for key in ("width", "phase_head", "position", "protocol", "data_seed"):
            if source_config.get(key, getattr(cfg, key)) != getattr(cfg, key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        if source_config["model"] == cfg.model:
            model.load_state_dict(checkpoint["model"])
            resume_mode = "weights_only_new_optimizer"
        elif "residualtarget" in cfg.model and source_config["model"] == cfg.model.replace("_residualtarget", ""):
            model.load_base_state_dict(checkpoint["model"])
            resume_mode = "weights_only_residual_target_transfer_new_optimizer"
        elif "spatial" in cfg.model and source_config["model"] == cfg.model.replace("_spatial", ""):
            model.load_pointwise_state_dict(checkpoint["model"])
            resume_mode = "weights_only_spatial_complex_transfer_new_optimizer"
        elif "envelope" in cfg.model and source_config["model"] == cfg.model.replace("_envelope", ""):
            model.load_envelope_state_dict(checkpoint["model"])
            resume_mode = "weights_only_target_envelope_transfer_new_optimizer"
        elif cfg.model == "transport_consensus_complex" and source_config["model"] == "transport_complex":
            model.load_consensus_state_dict(checkpoint["model"])
            resume_mode = "weights_only_target_consensus_transfer_new_optimizer"
        elif cfg.model == "transport_globaltarget_complex" and source_config["model"] == "transport_complex":
            model.load_globaltarget_state_dict(checkpoint["model"])
            resume_mode = "weights_only_global_target_transfer_new_optimizer"
        else:
            raise ValueError("Resume model must match or be the exact residual-target base")
        resume_sha = hashlib.sha256(Path(cfg.resume).read_bytes()).hexdigest()
    bank = ImageBank(seed=cfg.data_seed)
    manifest = bank.validation_manifest(cfg.val_per_group)
    bank.save_manifest(output / "data_manifest.json", per_group=cfg.val_per_group)
    write_json(output / "validation_tuples.json", {key: np.asarray(value).tolist() for key, value in manifest.items()})
    propagator = FresnelPropagator().to(cfg.device)
    target_rms = bank.fixed_train_rms(cfg.device)
    target_rms = float(target_rms)
    scales = (target_rms / propagator.field_rms.float()).reshape(1, 4, 1, 1)
    config = asdict(cfg)
    config.update({"target_rms": target_rms, "amplitude_scales": scales.flatten().tolist(), "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__, "numpy": np.__version__, "python": sys.version, "gpu": torch.cuda.get_device_name() if cfg.device == "cuda" else "cpu", "cuda": torch.version.cuda, "protocol": cfg.protocol, "resume_mode": "weights_only_new_optimizer" if cfg.resume else None})
    config.update({"resume_mode": resume_mode, "resume_checkpoint_sha256": resume_sha})
    write_json(output / "config.json", config)
    snapshot_source(output)
    initial = evaluate(model, bank, propagator, scales, manifest, cfg)
    write_json(output / "initial_metrics.json", initial)
    print(json.dumps({"run": cfg.run, "step": 0, **initial["aggregate"]}), flush=True)
    rng = np.random.default_rng(cfg.seed + 1000)
    fixed_records = None
    if cfg.fixed_train_quartets:
        fixed_groups = fixed_training_manifest(bank, per_group=cfg.fixed_train_quartets // 4, seed=50042)
        fixed_records = np.concatenate(list(fixed_groups.values()))
        write_json(output / "fixed_training_tuples.json", {name: records.tolist() for name, records in fixed_groups.items()})
    synthetic_rng = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 2000)
    history = []
    started = time.perf_counter()
    training_seconds = 0.0
    best_ssim = -float("inf")
    best_pearson = -float("inf")
    for step in range(1, cfg.steps + 1):
        model.train()
        lr_multiplier = cosine_lr_multiplier(step, cfg.steps, cfg.warmup_steps, cfg.lr_min_ratio) if cfg.lr_schedule == "cosine" else 1.0
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_multiplier
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        records = bank.sample_train(cfg.batch_size, rng, cfg.domain) if fixed_records is None else fixed_records[rng.integers(0, len(fixed_records), size=cfg.batch_size)]
        target = bank.fetch(records, cfg.device)
        synthetic_stats = {}
        if step <= cfg.synthetic_steps:
            target, synthetic_stats = sample_forward_targets(cfg.batch_size, propagator, scales, synthetic_rng)
        target = prepare_target(target, cfg.protocol)
        if cfg.blur_start > 0 and step < cfg.steps // 2:
            weight = cfg.blur_start * (1 - step / max(1, cfg.steps // 2))
            target = (1 - weight) * target + weight * F.avg_pool2d(target, 9, stride=1, padding=4)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16, enabled=cfg.amp):
            if cfg.latent_weight:
                phase, latent_fields = model(target, return_latent=True)
            else:
                phase = model(target)
        if cfg.quantize_train:
            phase = quantize_phase(phase, straight_through=True)
        raw = propagator.amplitude(phase) * scales
        reconstructed = prepare_reconstruction(raw, cfg.protocol)
        if cfg.detach_peak_gradient:
            reconstructed = peak_normalize_surrogate(raw, detach_peak=True)
        loss_amplitude = raw if cfg.protocol == LEGACY_PROTOCOL else reconstructed
        mse = (loss_amplitude - target).square().mean((-2, -1))
        correlation = pearson(reconstructed, target)
        structural = ssim(reconstructed, target)
        losses = mse + cfg.pearson_weight * (1 - correlation) + cfg.ssim_weight * (1 - structural)
        loss = losses.mean() + cfg.worst_weight * losses.max(1).values.mean()
        plane_contrast = ((loss_amplitude - loss_amplitude.mean(1, keepdim=True)) - (target - target.mean(1, keepdim=True))).square().mean()
        loss = loss + cfg.contrast_weight * plane_contrast
        depth_rank = depth_ranking_loss(reconstructed, target, cfg.rank_margin) if cfg.rank_weight else raw.new_zeros(())
        loss = loss + cfg.rank_weight * depth_rank
        cosine_loss = amplitude_cosine_loss(raw, target).mean() if cfg.cosine_weight else raw.new_zeros(())
        balanced_loss = balanced_pixel_mse(reconstructed, target).mean() if cfg.balanced_weight else raw.new_zeros(())
        loss = loss + cfg.cosine_weight * cosine_loss + cfg.balanced_weight * balanced_loss
        concentration_loss = excess_concentration_loss(raw, target).mean() if cfg.concentration_weight else raw.new_zeros(())
        loss = loss + cfg.concentration_weight * concentration_loss
        latent_loss = latent_source_consistency(latent_fields, phase, target) if cfg.latent_weight else raw.new_zeros(())
        loss = loss + cfg.latent_weight * latent_loss
        frequency_loss = depth_frequency_loss(raw, target) if cfg.depth_frequency_weight else raw.new_zeros(())
        loss = loss + cfg.depth_frequency_weight * frequency_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - tick
        if step % 50 == 0 or step == 1:
            entry = {"step": step, "loss": loss.item(), "mse": mse.mean().item(), "pearson": correlation.mean().item(), "ssim": structural.mean().item(), "plane_contrast": plane_contrast.item(), "grad_norm": float(grad_norm), "training_seconds": training_seconds, "target_source": "synthetic_forward" if step <= cfg.synthetic_steps else "real", **synthetic_stats}
            history.append(entry)
            entry["depth_rank"] = float(depth_rank)
            entry["amplitude_cosine_loss"] = float(cosine_loss)
            entry["balanced_pixel_mse"] = float(balanced_loss)
            entry["excess_concentration_loss"] = float(concentration_loss)
            entry["latent_consistency_loss"] = float(latent_loss)
            entry["depth_frequency_loss"] = float(frequency_loss)
            entry["learning_rate"] = cfg.lr * lr_multiplier
            if hasattr(model, "transport_consensus_gain"):
                entry["consensus_gain"] = float(model.transport_consensus_gain.detach())
            if getattr(model, "target_encoder", None) == "global":
                entry["global_target_gain"] = float(model.transport_target_encoder.global_gain.detach())
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps({"run": cfg.run, **entry}), flush=True)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            validation = evaluate(model, bank, propagator, scales, manifest, cfg)
            score = validation["aggregate"]["ssim"]
            entry = {"step": step, "validation": validation, "training_seconds": training_seconds}
            if hasattr(model, "transport_consensus_gain"):
                entry["consensus_gain"] = float(model.transport_consensus_gain.detach())
            if getattr(model, "target_encoder", None) == "global":
                entry["global_target_gain"] = float(model.transport_target_encoder.global_gain.detach())
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps({"run": cfg.run, "step": step, "val_ssim": score, "val_pearson": validation["aggregate"]["pearson"]}), flush=True)
            if fixed_records is not None:
                probe = evaluate(model, bank, propagator, scales, fixed_groups, cfg)
                probe_entry = {"step": step, "training_probe": probe}
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(probe_entry) + "\n")
                print(json.dumps({"run": cfg.run, "step": step, "training_probe_ssim": probe["aggregate"]["ssim"], "training_probe_pearson": probe["aggregate"]["pearson"]}), flush=True)
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "step": step, "training_seconds": training_seconds}
            torch.save(checkpoint, output / "last.pt")
            if cfg.save_every_eval:
                torch.save(checkpoint, output / f"step_{step:06d}.pt")
            if validation["aggregate"]["pearson"] > best_pearson:
                best_pearson = validation["aggregate"]["pearson"]
                torch.save(checkpoint, output / "best_pearson.pt")
            if score > best_ssim:
                best_ssim = score
                torch.save(checkpoint, output / "best.pt")
    if fixed_records is not None:
        probe_output = output / "training_probe_last"
        probe_output.mkdir()
        probe = evaluate(model, bank, propagator, scales, fixed_groups, cfg, output=probe_output, diagnostics=True)
        probe.update({"training_only_diagnostic": True, "test_used": False, "checkpoint": "last.pt", "step": cfg.steps})
        write_json(probe_output / "metrics.json", probe)
    selected = torch.load(output / "best.pt", map_location=cfg.device, weights_only=False)
    model.load_state_dict(selected["model"])
    final = evaluate(model, bank, propagator, scales, manifest, cfg, output=output, diagnostics=True)
    inputs = bank.fetch(next(iter(manifest.values()))[:1], cfg.device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        for _ in range(50):
            model(inputs)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
    final.update({"training_seconds": training_seconds, "total_seconds": time.perf_counter() - started, "inference_ms_batch1": (time.perf_counter() - tick) * 1000 / 50, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if cfg.device == "cuda" else 0, "selected_checkpoint": "best.pt", "selected_step": selected["step"]})
    write_json(output / "metrics.json", final)
    print(json.dumps({"run": cfg.run, "finished": True, **final["aggregate"], "training_seconds": training_seconds}), flush=True)


def parse_config():
    parser = argparse.ArgumentParser()
    for key, field in Config.__dataclass_fields__.items():
        if field.type is bool:
            parser.add_argument("--" + key.replace("_", "-"), action="store_true")
        else:
            parser.add_argument("--" + key.replace("_", "-"), type=field.type, default=field.default)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_config())
