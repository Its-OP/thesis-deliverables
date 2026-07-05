from __future__ import annotations

import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from autopsy_triplet_rank import autopsy_event, main as autopsy_main

from test_triplet_rank_data import _write_synthetic_artifacts


def _row(**overrides):
    # Base event: 4 candidates over 6 tracks, GT 3-set {0,1,2} enumerated once at
    # position 0; impostor sharing the couple (0,1) at position 1; disjoint fakes
    # at positions 2-3. All survive tau=0.1 except position 3.
    row = {
        'recon': True,
        'gt_i': 0, 'gt_j': 1, 'gt_k': 2,
        'cand_i': np.array([0, 0, 3, 4]),
        'cand_j': np.array([1, 1, 4, 5]),
        'cand_k': np.array([2, 3, 5, 3]),
        'couple_rank': np.array([0, 0, 1, 2]),
        'gbdt6_score': np.array([0.8, 0.7, 0.6, 0.05]),
        'is_gt': np.array([True, False, False, False]),
    }
    row.update(overrides)
    return row


def test_autopsy_not_reconstructable():
    result = autopsy_event(_row(recon=False, is_gt=np.array([False] * 4)),
                           window_positions=np.array([0, 1, 2]), tau=0.1, ks=(10,))
    assert result['bucket'] == 'not_reconstructable'


def test_autopsy_gt_not_enumerated():
    result = autopsy_event(_row(is_gt=np.array([False] * 4)),
                           window_positions=np.array([0, 1, 2]), tau=0.1, ks=(10,))
    assert result['bucket'] == 'gt_not_enumerated'


def test_autopsy_gt_killed_by_tau():
    row = _row(gbdt6_score=np.array([0.05, 0.7, 0.6, 0.05]))
    result = autopsy_event(row, window_positions=np.array([1, 2]), tau=0.1, ks=(10,))
    assert result['bucket'] == 'gt_killed_by_tau'


def test_autopsy_gt_outside_window():
    result = autopsy_event(_row(), window_positions=np.array([1, 2]), tau=0.1, ks=(10,))
    assert result['bucket'] == 'gt_outside_window'


def test_autopsy_hit_and_ranked_stats():
    # Window order = E1 score desc: impostor first, GT second.
    result = autopsy_event(_row(), window_positions=np.array([1, 0, 2]), tau=0.1,
                           ks=(1, 10))
    assert result['bucket'] == 'gt_in_window'
    assert result['gt_window_rank'] == 2
    assert result['hits'] == {1: False, 10: True}
    # Top impostor = window position 1 = candidate (0,1,3): shares couple with GT.
    assert result['impostor_shared_tracks'] == 2
    assert result['impostor_gbdt6'] == 0.7
    assert result['gt_gbdt6'] == 0.8
    assert result['n_surviving'] == 3


def test_autopsy_gt_rank_one_has_no_impostor_above():
    result = autopsy_event(_row(), window_positions=np.array([0, 1, 2]), tau=0.1,
                           ks=(1,))
    assert result['gt_window_rank'] == 1
    assert result['hits'] == {1: True}
    # Impostor = best-scored non-GT anywhere in window (rank 2 here).
    assert result['impostor_shared_tracks'] == 2


def test_autopsy_main_on_synthetic_artifacts(tmp_path):
    cand, tracks, _ = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    window_path = os.path.join(str(tmp_path), 'window.parquet')
    pq.write_table(pa.table({
        'window_positions': [[0, 1, 2], [0, 1]],
        'window_scores': [[0.9, 0.5, 0.2], [0.7, 0.6]],
    }), window_path)
    out_path = os.path.join(str(tmp_path), 'autopsy.json')
    autopsy_main(['--candidates', cand, '--window', window_path,
                  '--tau', '0.0', '--out-json', out_path])
    with open(out_path) as fh:
        report = json.load(fh)
    assert report['n_events'] == 2
    assert report['funnel']['gt_in_window'] == 2
    # Both synthetic events have GT at window rank 1.
    assert report['hits_by_k']['10'] == 2
    assert 'gt_window_rank_histogram' in report
    assert 'impostor_shared_tracks_histogram' in report
