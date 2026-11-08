"""Fixed TRAIN-only inputs for shared-network memorization diagnostics."""

import numpy as np


def fixed_training_manifest(bank, per_group=8, seed=50042):
    """Return balanced groups with no repeated source identity anywhere.

    The returned indices select input images, never per-example parameters or
    inputs to the neural network. Original ImageBank pools remain unchanged.
    Mixed-domain positions are independently sampled with equal probability;
    the mixed group need not contain an exactly balanced number of images.
    """
    if not isinstance(per_group, int) or isinstance(per_group, bool) or per_group < 1:
        raise ValueError("per_group must be a positive integer")
    names = ("CIFAR10", "MNIST", "FashionMNIST")
    if tuple(bank.names) != names or len(bank.train) != 3:
        raise ValueError("Expected CIFAR10, MNIST and FashionMNIST TRAIN pools")
    pools = [np.asarray(pool).copy() for pool in bank.train]
    for pool in pools:
        if pool.ndim != 1 or not np.issubdtype(pool.dtype, np.integer):
            raise ValueError("TRAIN pools must be one-dimensional integer arrays")
        if np.unique(pool).size != pool.size or (pool < 0).any():
            raise ValueError("TRAIN pools must contain unique nonnegative source indices")
        if len(pool) < 4 * per_group:
            raise ValueError("Not enough distinct TRAIN sources for homogeneous groups")
    rng = np.random.default_rng(seed)
    groups = {}
    for name, domain in zip(names + ("mixed",), ("cifar", "mnist", "fashion", "independent")):
        rows = []
        for _ in range(per_group):
            try:
                row = np.asarray(bank._sample(1, rng, pools, domain), dtype=np.int64)
            except ValueError as error:
                raise ValueError("Not enough distinct TRAIN sources for the requested probe") from error
            if row.shape != (1, 4, 2) or len(set(map(tuple, row[0]))) != 4:
                raise ValueError("Sampler must return four distinct source identities")
            for dataset, source in row[0]:
                if not 0 <= dataset < 3 or source not in pools[dataset]:
                    raise ValueError("Sampler returned a source outside the unused TRAIN pools")
            for dataset in range(3):
                used = row[0, :, 1][row[0, :, 0] == dataset]
                pools[dataset] = pools[dataset][~np.isin(pools[dataset], used)]
            rows.append(row)
        groups[name] = np.concatenate(rows, axis=0)
    return groups
