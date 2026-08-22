from __future__ import annotations

import numpy as np

from scripts.python.gate_recall_curve import (recall_at_top_n,
                                              threshold_curve)


def _toy_events():
    # Event 0: GT at score 0.9 (rank 1 of 3). Event 1: GT at 0.2 (rank 3 of 3).
    # Event 2: no GT row.
    scores = [np.array([0.9, 0.5, 0.1]), np.array([0.8, 0.4, 0.2]),
              np.array([0.7, 0.3])]
    is_gt = [np.array([True, False, False]), np.array([False, False, True]),
             np.array([False, False])]
    return scores, is_gt


def test_recall_at_top_n_counts_events_with_surviving_gt():
    scores, is_gt = _toy_events()
    assert recall_at_top_n(scores, is_gt, n=1) == (1, 3)
    assert recall_at_top_n(scores, is_gt, n=3) == (2, 3)


def test_load_event_arrays_prefers_external_scores(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.python.gate_recall_curve import load_event_arrays

    candidates = pa.table({
        'filter_score': pa.array([[0.9, 0.1], [0.2, 0.8]],
                                 type=pa.list_(pa.float32())),
        'is_gt': pa.array([[True, False], [True, False]],
                          type=pa.list_(pa.bool_())),
        'row_kind': pa.array([[0, 0], [0, 0]], type=pa.list_(pa.uint8())),
    })
    external = pa.table({
        'scores': pa.array([[0.1, 0.9], [0.9, 0.1]],
                           type=pa.list_(pa.float32())),
    })
    pq.write_table(candidates, tmp_path / 'candidates.parquet')
    pq.write_table(external, tmp_path / 'scores.parquet')

    scores, is_gt = load_event_arrays(str(tmp_path / 'candidates.parquet'),
                                      'filter_score')
    assert scores[0][0] == np.float32(0.9)

    scores, is_gt = load_event_arrays(
        str(tmp_path / 'candidates.parquet'), 'filter_score',
        scores_parquet=str(tmp_path / 'scores.parquet'))
    # External ordering flips the GT ranks: event 0 GT falls to rank 2,
    # event 1 GT rises to rank 1.
    assert recall_at_top_n(scores, is_gt, n=1) == (1, 2)


def test_threshold_curve_returns_budget_recall_pairs():
    scores, is_gt = _toy_events()
    curve = threshold_curve(scores, is_gt, thresholds=[0.15, 0.6])
    # tau=0.15: survivors 2+3+2=7 -> mean 7/3; both GT rows survive -> 2/3.
    # tau=0.6: survivors 1+1+1=3 -> mean 1.0; only event-0 GT survives -> 1/3.
    assert np.isclose(curve[0]['mean_survivors'], 7 / 3)
    assert np.isclose(curve[0]['recall'], 2 / 3)
    assert np.isclose(curve[1]['mean_survivors'], 1.0)
    assert np.isclose(curve[1]['recall'], 1 / 3)
