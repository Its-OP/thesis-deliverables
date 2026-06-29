from __future__ import annotations

import numpy as np

from scripts.python.train_triplet_filter import _curve, _factor_at_floor


def test_curve_recall_and_compression_monotone():
    rng = np.random.default_rng(0)
    gt_scores = rng.uniform(0, 1, 200)
    sub_scores = rng.uniform(0, 1, 5000)
    sub_weight = np.full(5000, 3.0)
    points = _curve(gt_scores, sub_scores, sub_weight, recon=250, n_full=100000)
    points = sorted(points, key=lambda p: p[0])  # by recall
    recalls = [p[0] for p in points]
    # As the threshold rises both recall and compression-ratio fall (non-increasing).
    taus_order = _curve(gt_scores, sub_scores, sub_weight, recon=250, n_full=100000)
    rec = [p[0] for p in taus_order]
    comp = [p[1] for p in taus_order]
    # _curve iterates thresholds ascending -> recall and compression non-increasing.
    assert all(rec[i] >= rec[i + 1] - 1e-9 for i in range(len(rec) - 1))
    assert all(comp[i] >= comp[i + 1] - 1e-9 for i in range(len(comp) - 1))


def test_factor_at_floor_picks_max_compression_above_floor():
    # points: (recall, compression_ratio)
    points = [(1.0, 0.5), (0.98, 0.2), (0.96, 0.05), (0.90, 0.01)]
    # at floor 0.97 only first two qualify -> min ratio 0.2 -> factor 5
    assert _factor_at_floor(points, 0.97) == 5.0
    # at floor 0.95 the 0.96/0.05 point qualifies -> factor 20
    assert _factor_at_floor(points, 0.95) == 20.0
    # impossible floor -> None
    assert _factor_at_floor(points, 1.01) is None
