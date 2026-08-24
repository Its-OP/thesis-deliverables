from __future__ import annotations

import numpy as np

from scripts.python.t2_survivor_stats import serving_stats


def test_serving_stats_counts_gated_rows_and_finds_gt_within_cap():
    scores = np.array([0.9, 0.8, 0.7, 0.2])
    kinds = np.zeros(4, dtype=np.int64)
    is_gt = np.array([False, False, True, False])
    n_survivors, gt_survives, gt_within_cap = serving_stats(
        scores, kinds, is_gt, tau=0.5, cap=2)
    assert n_survivors == 3
    assert gt_survives is True
    # GT sits at survivor position 2 (0-indexed), beyond cap 2.
    assert gt_within_cap is False


def test_serving_stats_gate_kills_gt():
    scores = np.array([0.9, 0.1])
    kinds = np.zeros(2, dtype=np.int64)
    is_gt = np.array([False, True])
    n_survivors, gt_survives, gt_within_cap = serving_stats(
        scores, kinds, is_gt, tau=0.5, cap=10)
    assert n_survivors == 1
    assert gt_survives is False and gt_within_cap is False


def test_serving_stats_ignores_non_serving_rows():
    scores = np.array([0.9, 0.9, 0.9])
    kinds = np.array([0, 1, 0])
    is_gt = np.array([False, True, True])
    n_survivors, gt_survives, gt_within_cap = serving_stats(
        scores, kinds, is_gt, tau=0.5, cap=10)
    assert n_survivors == 2
    assert gt_survives is True and gt_within_cap is True
