import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import PIL

from src.data import ImageBank
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator
from src.train import Config, evaluate, snapshot_source, write_json
from src.protocol import PROTOCOLS


def manifest_records(manifest, bank):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Expected a nonempty dictionary of tuple groups")
    groups = []
    for name, values in manifest.items():
        records = np.asarray(values)
        if records.ndim != 3 or records.shape[1:] != (4, 2) or not np.issubdtype(records.dtype, np.integer):
            raise ValueError(f"Invalid tuple group: {name}")
        groups.append(records.reshape(-1, 2))
    records = np.concatenate(groups)
    if np.any((records[:, 0] < 0) | (records[:, 0] >= len(bank.names))):
        raise ValueError("Invalid dataset index in exclusion manifest")
    for dataset in range(len(bank.names)):
        indices = records[records[:, 0] == dataset, 1]
        if np.any((indices < 0) | (indices >= len(bank.images[dataset]))):
            raise ValueError("Invalid image index in exclusion manifest")
    return records


def source_counts(records, bank):
    return {name: int(len(np.unique(records[records[:, 0] == dataset, 1]))) for dataset, name in enumerate(bank.names)}


def reserve_manifest(bank, per_group=448, selection_per_group=64, exclude_manifests=(), manifest_seed=None, return_provenance=False):
    if per_group < 1 or selection_per_group < 0:
        raise ValueError("per_group must be positive and selection_per_group nonnegative")
    records = np.empty((0, 2), dtype=np.int64)
    if selection_per_group:
        records = manifest_records(bank.validation_manifest(selection_per_group), bank)
    selection_counts = source_counts(records, bank)
    exclusions = []
    for path in exclude_manifests:
        path = Path(path)
        payload = path.read_bytes()
        excluded = manifest_records(json.loads(payload), bank)
        records = np.concatenate((records, excluded))
        exclusions.append({"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "unique_sources": source_counts(excluded, bank)})
    pools = [np.setdiff1d(indices, records[records[:, 0] == dataset, 1]) for dataset, indices in enumerate(bank.val)]
    for name, pool in zip(bank.names, pools):
        if len(pool) < 4:
            raise ValueError(f"At least four unused validation images required for {name}; remaining: {len(pool)}")
    seed = 30_000 + bank.seed if manifest_seed is None else manifest_seed
    groups = {}
    for number, (name, domain) in enumerate(zip(bank.names + ["mixed"], ["cifar", "mnist", "fashion", "independent"])):
        rng = np.random.default_rng(seed + number)
        groups[name] = np.concatenate([bank._sample(1, rng, pools, domain) for _ in range(per_group)])
    if return_provenance:
        provenance = {
            "split_seed": bank.seed,
            "manifest_seed": seed,
            "per_group": per_group,
            "selection_per_group": selection_per_group,
            "selection_unique_sources": selection_counts,
            "exclude_manifests": exclusions,
            "excluded_unique_sources": source_counts(records, bank),
            "available_validation_sources": {name: len(pool) for name, pool in zip(bank.names, pools)},
            "sampled_unique_sources": source_counts(manifest_records(groups, bank), bank),
            "source_reuse_between_tuples": True,
            "test_used": False,
        }
        return groups, provenance
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--per-group", type=int, default=512)
    parser.add_argument("--skip", type=int, default=64)
    parser.add_argument("--tag", default="validation_holdout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reserve-images", action="store_true")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--manifest-seed", type=int)
    parser.add_argument("--protocol", choices=PROTOCOLS)
    args = parser.parse_args()
    if not 0 <= args.skip < args.per_group:
        raise ValueError("skip must be in [0, per_group)")
    if (args.exclude_manifest or args.manifest_seed is not None) and not args.reserve_images:
        parser.error("--exclude-manifest and --manifest-seed require --reserve-images")
    output = args.run / args.tag
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    saved = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    cfg = Config(**{key: value for key, value in saved.items() if key in Config.__dataclass_fields__})
    cfg.device = args.device
    model = build_model(cfg.model, cfg.width, cfg.phase_head, position=cfg.position).to(cfg.device)
    checkpoint_path = args.checkpoint or args.run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    embedded = checkpoint.get("config", saved)
    for key in ("model", "width", "phase_head", "position", "protocol", "amplitude_scales", "data_seed"):
        if key in embedded and key in saved and embedded[key] != saved[key]:
            raise ValueError(f"Checkpoint/run config mismatch for {key}")
    model.load_state_dict(checkpoint["model"])
    bank = ImageBank(seed=cfg.data_seed)
    provenance = None
    if args.reserve_images:
        manifest, provenance = reserve_manifest(bank, per_group=args.per_group - args.skip, selection_per_group=args.skip, exclude_manifests=args.exclude_manifest, manifest_seed=args.manifest_seed, return_provenance=True)
        write_json(output / "manifest_provenance.json", provenance)
    else:
        manifest = {key: value[args.skip:] for key, value in bank.validation_manifest(args.per_group).items()}
    write_json(output / "validation_tuples.json", {key: np.asarray(value).tolist() for key, value in manifest.items()})
    propagator = FresnelPropagator().to(cfg.device)
    scales = torch.tensor(saved["amplitude_scales"], device=cfg.device).reshape(1, 4, 1, 1)
    result = evaluate(model, bank, propagator, scales, manifest, cfg, output=output, diagnostics=True, evaluation_protocol=args.protocol)
    result["selected_step"] = checkpoint["step"]
    result["checkpoint_file"] = checkpoint_path.name
    result["validation_prefix_skipped"] = args.skip
    result["test_used"] = False
    result["source_disjoint_from_selection"] = args.reserve_images
    result["source_disjoint_from_excluded_manifests"] = bool(args.exclude_manifest)
    write_json(output / "metrics.json", result)
    source_hashes = snapshot_source(output)
    write_json(output / "evaluation_config.json", {"training_config": saved, "evaluation_protocol": args.protocol or cfg.protocol, "model_input_protocol": cfg.protocol, "per_group": args.per_group, "skip": args.skip, "source_disjoint_from_selection": args.reserve_images, "manifest_provenance": provenance, "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__, "cudnn_deterministic": True, "source_hashes": source_hashes, "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()})
    print(json.dumps({"run": str(args.run), "evaluation": args.tag, "tuples": result["tuples"], "ssim": result["aggregate"]["ssim"], "pearson": result["aggregate"]["pearson"]}), flush=True)


if __name__ == "__main__":
    main()
