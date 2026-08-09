from __future__ import annotations

import numpy as np
import pytest

from scripts.python.analyze_triplet_probe import (
    cross_fold_scores,
    event_level_flag,
    per_event_hit,
    restricted_metrics,
)


def _probe_table(n_events=12, rows_per_event=5, seed=0, signal=1.0):
    generator = np.random.default_rng(seed)
    event_index = np.repeat(np.arange(n_events), rows_per_event)
    labels = np.zeros(len(event_index))
    labels[::rows_per_event] = 1.0
    feature = generator.normal(size=len(event_index)) + signal * labels
    return dict(
        labels=labels,
        event_index=event_index,
        columns={'a': feature.astype(np.float32),
                 'b': generator.normal(size=len(event_index)).astype(np.float32),
                 'has_sv': np.repeat(
                     (np.arange(n_events) % 2).astype(np.float32),
                     rows_per_event)},
    )


def test_cross_fold_scores_returns_one_score_per_evaluation_row():
    train = _probe_table(seed=0)
    evaluate = _probe_table(seed=1)
    scores = cross_fold_scores(train, evaluate, ['a', 'b'], n_folds=3)
    assert scores.shape == evaluate['labels'].shape
    assert np.isfinite(scores).all()


def test_cross_fold_scores_never_trains_on_the_evaluated_event():
    # A model that memorizes its training rows would score the held-out event
    # perfectly; folds must place every event's rows on one side only.
    train = _probe_table(seed=0)
    evaluate = _probe_table(seed=0)
    all_events = np.union1d(train['event_index'], evaluate['event_index'])
    fold_of_event = {event: index % 3 for index, event in enumerate(all_events)}
    train_folds = np.array([fold_of_event[e] for e in train['event_index']])
    evaluate_folds = np.array([fold_of_event[e] for e in evaluate['event_index']])
    for fold in range(3):
        assert not set(train['event_index'][train_folds != fold]) & \
            set(evaluate['event_index'][evaluate_folds == fold])


def test_cross_fold_scores_honours_dropped_columns():
    train = _probe_table(seed=0)
    evaluate = _probe_table(seed=1)
    with_all = cross_fold_scores(train, evaluate, ['a', 'b'], n_folds=3)
    without_signal = cross_fold_scores(
        train, evaluate, ['a', 'b'], n_folds=3, drop=('a',))
    assert not np.allclose(with_all, without_signal)


def test_cross_fold_scores_rejects_an_unknown_column():
    train = _probe_table(seed=0)
    with pytest.raises(KeyError):
        cross_fold_scores(train, train, ['a', 'missing'], n_folds=3)


def test_per_event_hit_flags_events_by_the_rank_of_their_truth():
    labels = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    event_index = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.1, 0.2, 0.1, 0.8, 0.7])
    hits = per_event_hit(scores, labels, event_index, k=1)
    assert hits == {0: True, 1: False}


def test_per_event_hit_skips_events_without_truth():
    labels = np.array([1.0, 0.0, 0.0, 0.0])
    event_index = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.1, 0.5, 0.4])
    assert set(per_event_hit(scores, labels, event_index, k=1)) == {0}


def test_event_level_flag_reads_one_value_per_event():
    table = _probe_table(n_events=4, rows_per_event=3)
    flags = event_level_flag(table['columns'], table['event_index'], 'has_sv')
    assert flags == {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0}


def test_restricted_metrics_only_scores_the_named_events():
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    event_index = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.1, 0.1, 0.9])
    assert restricted_metrics(scores, labels, event_index, {0}, k=1)[
        'hit_at_k'] == pytest.approx(1.0)
    assert restricted_metrics(scores, labels, event_index, {1}, k=1)[
        'hit_at_k'] == pytest.approx(0.0)


def test_restricted_metrics_reports_an_empty_selection():
    labels = np.array([1.0, 0.0])
    event_index = np.array([0, 0])
    scores = np.array([0.9, 0.1])
    result = restricted_metrics(scores, labels, event_index, set(), k=1)
    assert result['n_events'] == 0
    assert np.isnan(result['hit_at_k'])
