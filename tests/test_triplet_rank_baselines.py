from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from eval_triplet_rank_baselines import (
    OPERATING_POINTS,
    ORDERINGS,
    deduped_gt_rank,
    load_operating_points,
)


def test_deduped_gt_rank_manual():
    # Ordered candidate keys (already in ranking order); duplicates collapse to the
    # first occurrence. Unique sequence: A, B, C, D -> GT=C sits at rank 3.
    keys = np.array([
        [0, 1, 2],   # A
        [0, 1, 3],   # B
        [0, 1, 2],   # A duplicate
        [0, 2, 3],   # C  <- GT
        [0, 1, 3],   # B duplicate
        [1, 2, 3],   # D
    ])
    is_gt = np.array([False, False, False, True, False, False])
    assert deduped_gt_rank(keys, is_gt) == 3

    # GT as a duplicate: both decompositions map to the same 3-set; the first
    # occurrence (position 1 -> unique rank 2) counts.
    keys2 = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 3]])
    is_gt2 = np.array([False, True, True])
    assert deduped_gt_rank(keys2, is_gt2) == 2

    # GT absent -> None.
    assert deduped_gt_rank(keys, np.zeros(len(keys), dtype=bool)) is None


def test_dedup_encoding_survives_track_indices_beyond_2048():
    # The eval shards pad to 2,100 tracks; the old base-2048 encoding collided
    # there. Two DISTINCT 3-sets that collide under radix 2048:
    # (0, 0, 2099) -> 0*2048^2 + 0*2048 + 2099 = 2099
    # (0, 1, 51)   -> 0*2048^2 + 1*2048 + 51   = 2099
    keys = np.array([
        [0, 1, 51],
        [0, 0, 2099],
    ])
    is_gt = np.array([False, True])
    assert deduped_gt_rank(keys, is_gt) == 2


def test_tierh_applies_no_learned_gate():
    assert OPERATING_POINTS['tierH'] == ('filter_score', -np.inf)


def test_stale_gbdt_taus_are_gone():
    assert 'd6@0.99' not in OPERATING_POINTS
    assert 'd8@0.95' not in OPERATING_POINTS


def test_operating_points_load_from_json(tmp_path):
    path = tmp_path / 'operating_points.json'
    path.write_text(json.dumps({
        'score_column': 'filter_score',
        'taus': {'p99': 0.026845, 'p95': 0.264811},
    }))
    points = load_operating_points(str(path))
    assert points['tierH'] == ('filter_score', -np.inf)
    assert points['p99'] == ('filter_score', pytest.approx(0.026845))
    assert points['p95'] == ('filter_score', pytest.approx(0.264811))


def test_orderings_rank_by_the_single_filter_score():
    assert 'filter' in ORDERINGS
    assert 'gbdt' not in ORDERINGS
