from __future__ import annotations

import json

import numpy as np


def make_event_split(n_events: int, *, frac_train: float = 0.8, seed: int = 0):
    """Returns (train_idx, test_idx): sorted int64 arrays, disjoint, union == range(n_events)."""
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_events)
    n_train = int(round(frac_train * n_events))
    train_idx = np.sort(permutation[:n_train])
    test_idx = np.sort(permutation[n_train:])
    return train_idx, test_idx


def write_split(path: str, n_events: int, *, frac_train: float = 0.8, seed: int = 0):
    """Persist a seeded event split to path; returns (train_idx, test_idx)."""
    train_idx, test_idx = make_event_split(n_events, frac_train=frac_train, seed=seed)
    with open(path, "w") as handle:
        json.dump({"n_events": n_events, "seed": seed, "frac_train": frac_train,
                   "train": train_idx.tolist(), "test": test_idx.tolist()}, handle)
    return train_idx, test_idx


def load_split(path: str, side: str) -> np.ndarray:
    """side: "train" or "test". Returns the event-index array for that side."""
    with open(path) as handle:
        obj = json.load(handle)
    return np.asarray(obj[side], dtype=np.int64)
