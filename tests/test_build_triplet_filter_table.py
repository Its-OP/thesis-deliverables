from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.python.build_triplet_filter_table import (
    SRC_COLS,
    _h6_inputs_for_event,
    _sample_negative_rows,
    _select_hard_negative_rows,
)
from utils.triplet_join import H6_INPUT_KEYS


def _source_columns():
    return {
        "track_vertex_x": [[0.1, 0.2, 0.3]],
        "track_vertex_y": [[0.4, 0.5, 0.6]],
        "track_vertex_z": [[0.7, 0.8, 0.9]],
        "track_dz": [[1.0, 1.1, 1.2]],
        "event_primary_vertex_x": [0.05],
        "event_primary_vertex_y": [-0.05],
        "sv_x": [[1.0, 2.0]],
        "sv_y": [[1.5, 2.5]],
        "sv_z": [[1.7, 2.7]],
        "sv_dlen_sig": [[3.0, 4.0]],
        "sv_mass": [[0.5, 0.6]],
        "other_track_pt": [[0.4]],
        "other_track_eta": [[0.1]],
        "other_track_phi": [[0.2]],
        "other_track_dz": [[0.3]],
    }


def test_source_columns_cover_every_h6_input():
    for name in ("track_vertex_x", "track_vertex_y", "track_vertex_z",
                 "track_dz", "event_primary_vertex_x",
                 "event_primary_vertex_y", "sv_x", "sv_y", "sv_z",
                 "sv_dlen_sig", "sv_mass", "other_track_pt",
                 "other_track_eta", "other_track_phi", "other_track_dz"):
        assert name in SRC_COLS


def test_h6_inputs_for_event_builds_every_key_as_a_tensor():
    inputs = _h6_inputs_for_event(_source_columns(), 0)
    assert set(inputs) == set(H6_INPUT_KEYS)
    assert all(torch.is_tensor(value) for value in inputs.values())
    assert inputs["vertex_x"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert inputs["dz_raw"].tolist() == pytest.approx([1.0, 1.1, 1.2])
    assert float(inputs["primary_vertex_x"]) == pytest.approx(0.05)
    assert inputs["sv_dlen_sig"].tolist() == pytest.approx([3.0, 4.0])
    assert inputs["other_pt"].tolist() == pytest.approx([0.4])


def test_h6_inputs_tolerate_empty_secondary_vertex_and_companion_lists():
    columns = _source_columns()
    for name in ("sv_x", "sv_y", "sv_z", "sv_dlen_sig", "sv_mass",
                 "other_track_pt", "other_track_eta", "other_track_phi",
                 "other_track_dz"):
        columns[name] = [[]]
    inputs = _h6_inputs_for_event(columns, 0)
    assert inputs["sv_x"].numel() == 0
    assert inputs["other_pt"].numel() == 0


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def _candidate_pool():
    # 12 negatives spread unevenly over 3 couples, plus one positive.
    couple_row = np.array([0] * 7 + [1] * 4 + [2] * 1 + [0], dtype=np.int64)
    is_gt = np.zeros(len(couple_row), dtype=bool)
    is_gt[-1] = True
    return couple_row, is_gt


def test_uniform_mode_draws_the_requested_count_without_replacement():
    couple_row, is_gt = _candidate_pool()
    rows = _sample_negative_rows(
        is_gt, couple_row, 5, "uniform", np.random.default_rng(0))
    assert len(rows) == 5
    assert len(set(rows.tolist())) == 5
    assert not is_gt[rows].any()


def test_uniform_mode_is_capped_by_the_available_negatives():
    couple_row, is_gt = _candidate_pool()
    rows = _sample_negative_rows(
        is_gt, couple_row, 50, "uniform", np.random.default_rng(0))
    assert len(rows) == int((~is_gt).sum())


def test_per_couple_mode_visits_every_couple_before_repeating():
    couple_row, is_gt = _candidate_pool()
    rows = _sample_negative_rows(
        is_gt, couple_row, 3, "per_couple", np.random.default_rng(0))
    assert sorted(couple_row[rows].tolist()) == [0, 1, 2]


def test_per_couple_mode_spreads_the_budget_round_robin():
    couple_row, is_gt = _candidate_pool()
    rows = _sample_negative_rows(
        is_gt, couple_row, 6, "per_couple", np.random.default_rng(0))
    counts = np.bincount(couple_row[rows], minlength=3)
    # Couple 2 has a single candidate, so the remainder spreads over 0 and 1.
    assert counts.tolist() == [3, 2, 1]
    assert len(set(rows.tolist())) == 6


def test_per_couple_mode_exhausts_without_duplicating():
    couple_row, is_gt = _candidate_pool()
    rows = _sample_negative_rows(
        is_gt, couple_row, 100, "per_couple", np.random.default_rng(0))
    assert sorted(rows.tolist()) == sorted(np.flatnonzero(~is_gt).tolist())


def test_hard_mode_takes_the_top_scoring_negatives_first():
    couple_row, is_gt = _candidate_pool()
    scores = np.linspace(0.0, 1.0, len(couple_row))
    rows = _select_hard_negative_rows(
        is_gt, scores, hard_top=3, random_count=0,
        gen=np.random.default_rng(0))
    # Row 12 is the positive, so the highest-scoring negatives are 11, 10, 9.
    assert sorted(rows.tolist()) == [9, 10, 11]


def test_hard_mode_adds_random_draws_from_outside_the_top():
    couple_row, is_gt = _candidate_pool()
    scores = np.linspace(0.0, 1.0, len(couple_row))
    rows = _select_hard_negative_rows(
        is_gt, scores, hard_top=3, random_count=4,
        gen=np.random.default_rng(0))
    assert len(rows) == 7
    assert len(set(rows.tolist())) == 7
    assert {9, 10, 11}.issubset(set(rows.tolist()))
    assert not is_gt[rows].any()


def test_hard_mode_never_returns_the_positive_rows():
    couple_row, is_gt = _candidate_pool()
    scores = np.zeros(len(couple_row))
    scores[np.flatnonzero(is_gt)] = 10.0
    rows = _select_hard_negative_rows(
        is_gt, scores, hard_top=5, random_count=5,
        gen=np.random.default_rng(0))
    assert not is_gt[rows].any()


def test_unknown_negative_mode_is_rejected():
    couple_row, is_gt = _candidate_pool()
    with pytest.raises(ValueError, match="neg_mode"):
        _sample_negative_rows(
            is_gt, couple_row, 3, "hardest", np.random.default_rng(0))
