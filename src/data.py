import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


NAMES = ["CIFAR10", "MNIST", "FashionMNIST"]
SIZE = 128
PREPROCESS_VERSION = "gray-f64-rint_bilinear-align-false_rint-u8_v1"


def read_idx(path):
    with Path(path).open("rb") as stream:
        magic, kind, dimensions = struct.unpack(">HBB", stream.read(4))
        if magic != 0 or kind != 8:
            raise ValueError(f"Expected unsigned-byte IDX: {path}")
        shape = struct.unpack(">" + "I" * dimensions, stream.read(4 * dimensions))
        values = np.frombuffer(stream.read(), dtype=np.uint8).copy()
    if values.size != np.prod(shape):
        raise ValueError(f"Invalid IDX size: {path}")
    return values.reshape(shape)


def load_cifar(root):
    batches = []
    labels = []
    for number in range(1, 6):
        with (root / "cifar-10-batches-py" / f"data_batch_{number}").open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        rgb = batch[b"data"].reshape(-1, 3, 32, 32)
        gray = np.empty((len(rgb), 32, 32), dtype=np.uint8)
        for start in range(0, len(rgb), 512):
            pixels = rgb[start:start + 512].astype(np.float64)
            gray[start:start + 512] = np.rint(
                0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
            ).clip(0, 255).astype(np.uint8)
        batches.append(gray)
        labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
    return np.concatenate(batches), np.concatenate(labels)


def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts * 0.1
    val_counts = np.floor(exact).astype(int)
    remainder = round(len(labels) * 0.1) - val_counts.sum()
    order = np.argsort(-(exact - val_counts), kind="stable")
    val_counts[order[:remainder]] += 1
    for label, count in zip(classes, val_counts):
        indices = rng.permutation(np.flatnonzero(labels == label))
        val.append(indices[:count])
        train.append(indices[count:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


class ImageBank:
    """Local training images and fixed image-level splits; test is never loaded."""

    names = NAMES

    def __init__(self, root=Path("data"), seed=42):
        self.root = Path(root)
        self.seed = seed
        cifar_images, cifar_labels = load_cifar(self.root)
        self.images = [cifar_images]
        self.labels = [cifar_labels]
        for name in NAMES[1:]:
            raw = self.root / name / "raw"
            self.images.append(read_idx(raw / "train-images-idx3-ubyte"))
            self.labels.append(read_idx(raw / "train-labels-idx1-ubyte").astype(np.int64))
        self.train, self.val = [], []
        for images, labels in zip(self.images, self.labels):
            if len(images) != len(labels):
                raise ValueError("Images and labels have different lengths")
            train, val = stratified_split(labels, seed)
            self.train.append(train)
            self.val.append(val)
        self._rms = None

    def _sample(self, batch_size, rng, pools, domain):
        aliases = {"cifar": 0, "cifar10": 0, "mnist": 1, "fashion": 2, "fashionmnist": 2}
        domain = domain.lower()
        if domain == "mixed":
            domains = rng.integers(0, 3, size=(batch_size, 4))
            homogeneous = rng.random(batch_size) < 0.5
            domains[homogeneous] = rng.integers(0, 3, size=(homogeneous.sum(), 1))
        elif domain == "independent":
            domains = rng.integers(0, 3, size=(batch_size, 4))
        elif domain in aliases:
            domains = np.full((batch_size, 4), aliases[domain], dtype=np.int64)
        else:
            raise ValueError(f"Unknown domain: {domain}")
        records = np.empty((batch_size, 4, 2), dtype=np.int64)
        records[:, :, 0] = domains
        for row in range(batch_size):
            for dataset in np.unique(domains[row]):
                positions = np.flatnonzero(domains[row] == dataset)
                records[row, positions, 1] = rng.choice(pools[dataset], len(positions), replace=False)
        return records

    def sample_train(self, batch_size, rng, domain="mixed"):
        return self._sample(batch_size, rng, self.train, domain)

    def validation_manifest(self, per_group=512):
        # Each group has its own RNG: group size does not change other groups.
        groups = {}
        for number, (name, domain) in enumerate(zip(NAMES + ["mixed"], ["cifar", "mnist", "fashion", "independent"])):
            rng = np.random.default_rng(10_000 + self.seed + number)
            # Draw per row so shorter manifests are exact prefixes of larger ones.
            groups[name] = np.concatenate([
                self._sample(1, rng, self.val, domain) for _ in range(per_group)
            ])
        return groups

    def fetch(self, records, device="cpu"):
        records = np.asarray(records, dtype=np.int64)
        if records.ndim != 3 or records.shape[1:] != (4, 2):
            raise ValueError("Expected records with shape [batch, 4, 2]")
        flat = records.reshape(-1, 2)
        if np.any((flat[:, 0] < 0) | (flat[:, 0] >= len(NAMES))):
            raise ValueError("Invalid dataset index")
        output = torch.empty((len(flat), SIZE, SIZE), dtype=torch.float32, device=device)
        for dataset in range(len(NAMES)):
            positions = np.flatnonzero(flat[:, 0] == dataset)
            if not len(positions):
                continue
            indices = flat[positions, 1]
            if np.any((indices < 0) | (indices >= len(self.images[dataset]))):
                raise ValueError("Invalid source image index")
            images = torch.from_numpy(self.images[dataset][indices]).to(device=device, dtype=torch.float32)
            resized = F.interpolate(images[:, None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)
            output[torch.as_tensor(positions, device=device)] = resized[:, 0].round().clamp(0, 255) / 255
        return output.reshape(len(records), 4, SIZE, SIZE)

    def fixed_train_rms(self, device="cpu"):
        """Balanced RMS over 1024 fixed training images per dataset, no test fit."""
        if self._rms is None:
            sums = []
            for dataset in range(len(NAMES)):
                rng = np.random.default_rng(20_000 + self.seed + dataset)
                indices = rng.choice(self.train[dataset], 1024, replace=False)
                total = 0.0
                for start in range(0, len(indices), 64):
                    subset = indices[start:start + 64]
                    records = np.stack([np.full(len(subset), dataset), subset], axis=1).reshape(-1, 4, 2)
                    images = self.fetch(records, device=device)
                    total += images.double().square().sum().item()
                sums.append(total / (len(indices) * SIZE * SIZE))
            self._rms = float(np.sqrt(np.mean(sums)))
        return self._rms

    def manifest(self, per_group=512):
        return {
            "version": PREPROCESS_VERSION,
            "split_seed": self.seed,
            "names": NAMES,
            "size": SIZE,
            "test_used": False,
            "train": {name: indices.tolist() for name, indices in zip(NAMES, self.train)},
            "val": {name: indices.tolist() for name, indices in zip(NAMES, self.val)},
            "validation": {name: records.tolist() for name, records in self.validation_manifest(per_group).items()},
            "validation_mixed_sampler": "independent equally likely dataset per plane",
            "train_mixed_sampler": "half homogeneous, half independent; uniform domains",
        }

    def save_manifest(self, path, per_group=512):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.manifest(per_group), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path.write_text(payload, encoding="utf-8")
        return signature
