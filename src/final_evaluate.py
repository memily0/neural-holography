"""Evaluate a frozen checkpoint on the public datasets' untouched test files.

Importing this module and generating manifests do not read any test images.
The test loader is called only by the explicit final-evaluation entry point.
These public test splits are not the organizers' undisclosed challenge inputs.
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch

from src.data import ImageBank, NAMES, PREPROCESS_VERSION, read_idx
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator
from src.protocol import LEGACY_PROTOCOL, PEAK_PROTOCOL, PROTOCOLS
from src.train import Config, evaluate, snapshot_source, write_json


TEST_NAMESPACE = "public_dataset_official_test"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cifar_test(path):
    with Path(path).open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    data = batch[b"data"]
    if data.shape != (10_000, 3072) or data.dtype != np.uint8:
        raise ValueError("Expected the original CIFAR10 test_batch with 10000 uint8 images")
    rgb = data.reshape(-1, 3, 32, 32)
    gray = np.empty((len(rgb), 32, 32), dtype=np.uint8)
    for start in range(0, len(rgb), 512):
        pixels = rgb[start:start + 512].astype(np.float64)
        gray[start:start + 512] = np.rint(
            0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        ).clip(0, 255).astype(np.uint8)
    return gray


class FinalTestBank:
    """Test-only bank; never calls ImageBank's training-data constructor."""

    names = NAMES
    fetch = ImageBank.fetch

    def __init__(self, root=Path("data")):
        self.root = Path(root)
        paths = [
            self.root / "cifar-10-batches-py" / "test_batch",
            self.root / "MNIST" / "raw" / "t10k-images-idx3-ubyte",
            self.root / "FashionMNIST" / "raw" / "t10k-images-idx3-ubyte",
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.images = [load_cifar_test(paths[0]), read_idx(paths[1]), read_idx(paths[2])]
        self.sources = []
        for dataset, (name, path, images) in enumerate(zip(NAMES, paths, self.images)):
            shape = (10_000, 32, 32) if dataset == 0 else (10_000, 28, 28)
            if images.shape != shape or images.dtype != np.uint8:
                raise ValueError(f"Unexpected original test image array for {name}: {images.shape}")
            self.sources.append({
                "dataset_index": dataset, "dataset": name, "namespace": TEST_NAMESPACE,
                "file": path.relative_to(self.root).as_posix(), "sha256": file_sha256(path),
                "source_rows": len(images), "array_shape": list(images.shape),
                "source_index": "zero-based row in the named official test file, not a training-file row",
            })


def make_final_manifest(bank, per_group=256, seed=20261108):
    if not isinstance(per_group, int) or per_group < 1:
        raise ValueError("per_group must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(bank.images) != len(NAMES) or any(len(images) < 4 for images in bank.images):
        raise ValueError("Each of the three test pools must contain at least four sources")
    pools = [np.arange(len(images), dtype=np.int64) for images in bank.images]
    groups = {}
    for number, (name, domain) in enumerate(zip(NAMES + ["mixed"], ("cifar", "mnist", "fashion", "independent"))):
        rng = np.random.default_rng(seed + number)
        # Independent group RNGs and per-row draws preserve exact short prefixes.
        groups[name] = np.concatenate([ImageBank._sample(bank, 1, rng, pools, domain) for _ in range(per_group)])
    return groups


def used_source_counts(manifest):
    records = np.concatenate(list(manifest.values())).reshape(-1, 2)
    return {name: int(np.unique(records[records[:, 0] == dataset, 1]).size) for dataset, name in enumerate(NAMES)}


def require_final_candidate(config, source):
    """Reject diagnostic weights before constructing a model or opening test data."""
    if config.get("diagnostic_only", False) or config.get("fixed_train_quartets", 0) > 0:
        raise ValueError(f"{source} contains diagnostic-only weights; final test evaluation is forbidden")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/final_test"))
    parser.add_argument("--per-group", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.per_group < 1 or args.seed < 0 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("per-group/batch-size must be positive and seed nonnegative")
    output = args.output_root / args.run.name
    if output.exists():
        raise FileExistsError(f"Final evaluation is never overwritten: {output}")
    saved = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    require_final_candidate(saved, "Run config")
    values = {key: value for key, value in saved.items() if key in Config.__dataclass_fields__}
    values["protocol"] = saved.get("protocol", LEGACY_PROTOCOL)
    cfg = Config(**values)
    cfg.device = args.device
    if cfg.protocol not in PROTOCOLS:
        raise ValueError(f"Unknown checkpoint input protocol: {cfg.protocol}")
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_path = args.checkpoint or args.run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    embedded = checkpoint.get("config", saved)
    require_final_candidate(embedded, "Checkpoint config")
    for key in ("model", "width", "phase_head", "position", "amplitude_scales"):
        if key in embedded and key in saved and embedded[key] != saved[key]:
            raise ValueError(f"Checkpoint/run config mismatch for {key}")
    if embedded.get("protocol", LEGACY_PROTOCOL) != cfg.protocol:
        raise ValueError("Checkpoint/run input protocol mismatch")
    model = build_model(cfg.model, cfg.width, cfg.phase_head, position=cfg.position).to(cfg.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scales = torch.tensor(saved["amplitude_scales"], device=cfg.device).reshape(1, 4, 1, 1)
    propagator = FresnelPropagator().to(cfg.device)
    output.mkdir(parents=True)
    frozen = {
        "run": args.run.name, "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"), "training_config": saved,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_input_protocol": cfg.protocol, "evaluation_protocol": PEAK_PROTOCOL,
        "rule": "One checkpoint fixed before loading any official test image; test scores cannot select weights",
    }
    write_json(output / "frozen_model.json", frozen)
    source_hashes = snapshot_source(output)

    # This is the only point in this command that opens official test images.
    bank = FinalTestBank(args.data_root)
    manifest = make_final_manifest(bank, args.per_group, args.seed)
    write_json(output / "test_tuples.json", {name: records.tolist() for name, records in manifest.items()})
    provenance = {
        "namespace": TEST_NAMESPACE, "test_used": True, "organizer_hidden_test": False,
        "source_identity": "(source file SHA256, zero-based image row); numerical indices can overlap train indices in different files",
        "data_root": str(args.data_root), "sources": bank.sources, "preprocessing": PREPROCESS_VERSION,
        "seed": args.seed, "quadruples_per_group": args.per_group, "quadruples": 4 * args.per_group,
        "groups": NAMES + ["mixed"], "unique_sources_used": used_source_counts(manifest),
        "mixed_sampler": "independent uniformly selected dataset per plane; distinct source rows within each quadruple",
        "source_reuse_between_quadruples": True, "classes_used_for_sampling": False,
        "test_tuples_sha256": file_sha256(output / "test_tuples.json"),
    }
    write_json(output / "test_provenance.json", provenance)
    started = time.perf_counter()
    result = evaluate(model, bank, propagator, scales, manifest, cfg, output=output, diagnostics=True, evaluation_protocol=PEAK_PROTOCOL)
    result.update({"test_used": True, "organizer_hidden_test": False, "source_namespace": TEST_NAMESPACE, "selected_step": checkpoint.get("step"), "checkpoint_sha256": frozen["checkpoint_sha256"], "evaluation_seconds": time.perf_counter() - started, "checkpoint_selection_used_test": False})
    write_json(output / "metrics.json", result)
    write_json(output / "evaluation_config.json", {
        **frozen, "test_provenance": provenance, "source_hashes": source_hashes,
        "batch_size": cfg.batch_size, "device": cfg.device, "torch": torch.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__, "python": sys.version,
        "cudnn_deterministic": True,
    })
    print(json.dumps({"run": args.run.name, "output": str(output), "test_used": True, "quadruples": result["tuples"], "ssim": result["aggregate"]["ssim"], "pearson": result["aggregate"]["pearson"]}), flush=True)


if __name__ == "__main__":
    main()
