from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from eval_triplet_rank_baselines import (
    OPERATING_POINTS,
    deduped_gt_rank,
    evaluate_baselines,
)

_DELIVERABLES = os.path.join(os.path.dirname(__file__), '..')
_DUMP = os.path.join(_DELIVERABLES, 'data', 'low-pt', 'eval', 'perstage_couples_val.parquet')
_SRC_GLOB = '/Users/oleh/Projects/masters/part/data/low-pt/val/val_*.parquet'
_GBDT6 = os.path.join(_DELIVERABLES, 'models', 'third_pion_filter_gbdt_full_P2.joblib')
_HAVE_REAL_VAL = os.path.exists(_DUMP) and glob.glob(_SRC_GLOB) and os.path.exists(_GBDT6)


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


def test_operating_points_frozen():
    assert OPERATING_POINTS['d6@0.99'] == ('gbdt6_score', pytest.approx(0.003824))
    assert OPERATING_POINTS['d8@0.95'] == ('gbdt8_score', pytest.approx(0.090546))
    assert OPERATING_POINTS['tierH'][1] == 0.0


@pytest.fixture(scope='module')
def real_artifact(tmp_path_factory):
    if not _HAVE_REAL_VAL:
        pytest.skip('real VAL dump/src/models not present')
    from build_triplet_rank_candidates import main as build_main
    out_dir = str(tmp_path_factory.mktemp('triplet_rank_baselines'))
    build_main(['--out-dir', out_dir, '--tag', 'val', '--max-events', '60'])
    return out_dir


def test_evaluate_baselines_consistency(real_artifact):
    result = evaluate_baselines(
        os.path.join(real_artifact, 'candidates_val.parquet'),
        operating_point='d6@0.99', seed=0,
    )
    assert result['n_events'] == 60
    ceiling = result['ceiling']
    assert 0.0 < ceiling <= 1.0
    for ordering in ['gbdt', 'couple_rank_lex', 'random']:
        curve = result['t_at_k'][ordering]
        values = [curve[str(k)] for k in result['k_values']]
        # Monotone in K, bounded by the ceiling, and the loosest K reaches it only
        # if every surviving GT ranks inside max(K).
        assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
        assert values[-1] <= ceiling + 1e-9
    # GBDT ordering at K>=1 can't beat the ceiling and must find at least one event.
    assert result['t_at_k']['gbdt']['100'] > 0.0
