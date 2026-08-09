from __future__ import annotations

import numpy as np
import pytest

from scripts.python.analyze_filter_floors import (
    bottom_fraction_indices,
    compression_grid,
    jaccard,
    max_abs_spearman,
    percentile_table,
)


def test_compression_grid_reports_one_entry_per_floor():
    gt = np.linspace(0.0, 1.0, 100)
    sub = np.linspace(0.0, 1.0, 1000)
    weights = np.full(1000, 10.0)
    grid = compression_grid(gt, sub, weights, recon=100, n_full=100000,
                            floors=(0.99, 0.95, 0.90))
    assert sorted(grid) == [0.90, 0.95, 0.99]
    assert all(entry['compression'] > 0 for entry in grid.values())


def test_compression_rises_as_the_recall_floor_relaxes():
    generator = np.random.default_rng(0)
    gt = generator.normal(1.0, 0.5, 500)
    sub = generator.normal(0.0, 0.5, 5000)
    weights = np.full(5000, 20.0)
    grid = compression_grid(gt, sub, weights, recon=500, n_full=200000,
                            floors=(0.99, 0.95, 0.90))
    assert grid[0.90]['compression'] >= grid[0.95]['compression'] >= \
        grid[0.99]['compression']


def test_compression_grid_reports_the_ground_truth_actually_lost():
    gt = np.linspace(0.0, 1.0, 200)
    sub = np.linspace(0.0, 1.0, 200)
    weights = np.ones(200)
    grid = compression_grid(gt, sub, weights, recon=200, n_full=1000,
                            floors=(0.95,))
    assert grid[0.95]['gt_lost'] == pytest.approx(200 * 0.05, abs=2)


def test_percentile_table_is_monotonic():
    scores = np.random.default_rng(0).random(10000)
    table = percentile_table(scores, (1, 5, 25, 50))
    values = [table[p] for p in (1, 5, 25, 50)]
    assert values == sorted(values)


def test_percentile_table_matches_numpy():
    scores = np.random.default_rng(1).normal(size=5000)
    table = percentile_table(scores, (1, 10))
    assert table[1] == pytest.approx(float(np.percentile(scores, 1)))
    assert table[10] == pytest.approx(float(np.percentile(scores, 10)))


def test_bottom_fraction_picks_the_lowest_scores():
    scores = np.array([0.5, 0.1, 0.9, 0.2, 0.7])
    assert sorted(bottom_fraction_indices(scores, 0.4).tolist()) == [1, 3]


def test_bottom_fraction_keeps_at_least_one_row():
    assert len(bottom_fraction_indices(np.array([0.3, 0.4]), 0.001)) == 1


def test_jaccard_of_identical_sets_is_one():
    assert jaccard(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(1.0)


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard(np.array([1, 2]), np.array([3, 4])) == pytest.approx(0.0)


def test_jaccard_counts_the_shared_fraction():
    assert jaccard(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])) == \
        pytest.approx(2 / 6)


def test_jaccard_of_two_empty_sets_is_nan():
    assert np.isnan(jaccard(np.array([], dtype=int), np.array([], dtype=int)))


def test_max_abs_spearman_finds_a_perfect_monotone_partner():
    generator = np.random.default_rng(0)
    legacy = generator.random((200, 3))
    # A monotone transform of legacy column 1 must score 1.0.
    new = np.exp(legacy[:, [1]] * 2.0)
    best, partner = max_abs_spearman(new, legacy)
    assert best[0] == pytest.approx(1.0, abs=1e-9)
    assert partner[0] == 1


def test_max_abs_spearman_detects_inversion():
    generator = np.random.default_rng(0)
    legacy = generator.random((200, 2))
    new = -legacy[:, [0]]
    best, _ = max_abs_spearman(new, legacy)
    assert best[0] == pytest.approx(1.0, abs=1e-9)


def test_max_abs_spearman_is_low_for_independent_columns():
    generator = np.random.default_rng(3)
    legacy = generator.random((4000, 2))
    new = generator.random((4000, 1))
    best, _ = max_abs_spearman(new, legacy)
    assert best[0] < 0.1
