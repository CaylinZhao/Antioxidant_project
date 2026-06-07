"""
Reproducible dataset splitting utilities.
Deterministic train/valid/test partitioning based on unique SMILES strings,
ensuring no data leakage across splits.
"""

import json
from pathlib import Path

import numpy as np


def make_smiles_split(smiles_list, seed=0, train_ratio=0.8, val_ratio=0.1):
    """Deterministic split based on unique SMILES.

    Procedure:
    1. Deduplicate and sort all SMILES → guarantees identical sets across runs.
    2. Shuffle indices with a fixed seed, then partition into train/valid/test.
    3. Boundary protection: ensures each partition has at least 1 sample.

    Returns: {"train": [...], "valid": [...], "test": [...]}
    """
    smiles = sorted({str(s) for s in smiles_list if s is not None and str(s)})
    n = len(smiles)
    if n == 0:
        return {"train": [], "valid": [], "test": []}

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * float(train_ratio)))
    n_valid = int(round(n * float(val_ratio)))
    # Clamp to ensure at least one sample per split and no out-of-bounds
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 1)
    n_valid = max(1, min(n_valid, n - n_train - 1)) if (n - n_train) >= 2 else max(0, n - n_train - 1)

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]

    # Prevent empty test/valid partitions
    if len(test_idx) == 0 and len(valid_idx) > 0:
        test_idx = valid_idx[-1:]
        valid_idx = valid_idx[:-1]
    if len(valid_idx) == 0 and len(test_idx) > 1:
        valid_idx = test_idx[:1]
        test_idx = test_idx[1:]

    return {
        "train": [smiles[i] for i in train_idx.tolist()],
        "valid": [smiles[i] for i in valid_idx.tolist()],
        "test": [smiles[i] for i in test_idx.tolist()],
    }


def save_split(split_obj, split_file):
    """Save a split dictionary as JSON for reproducibility."""
    fp = Path(split_file)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split(split_file):
    """Load a previously saved split from a JSON file."""
    fp = Path(split_file)
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))
