from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.python.build_triplet_filter_table import (
    IDENTITY_COLS,
    SRC_COLS,
    assert_dump_aligned,
    dump_row_order,
    identity_keys,
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


def test_parallel_build_matches_the_serial_build():
    """Workers must change only the wall clock: per-event seeding makes the
    sampling independent of how events are distributed over processes."""
    import glob
    import os

    from scripts.python.build_triplet_filter_table import _shard_blocks, build_train

    root = os.path.join(os.path.dirname(__file__), '..')
    dump = os.path.join(root, 'data', 'dumps', 'couples_k125_dump.parquet')
    shards = sorted(glob.glob(os.path.join(root, 'data', 'low-pt', 'eval', '*.parquet')))
    if not (os.path.exists(dump) and shards):
        pytest.skip('eval dump or shards not present')

    outputs = {}
    for workers in (0, 3):
        blocks = _shard_blocks(dump, os.path.join(root, 'data', 'low-pt', 'eval', '*.parquet'), 24)
        path = os.path.join(
            os.path.dirname(__file__), f'_parallel_probe_{workers}.parquet')
        build_train(blocks, 100, 20, 'uniform', np.random.default_rng(0), path,
                    True, workers=workers, seed=0)
        outputs[workers] = pq.read_table(path)
        os.remove(path)

    assert outputs[0].num_rows == outputs[3].num_rows
    for name in outputs[0].schema.names:
        serial = np.asarray(outputs[0].column(name))
        parallel = np.asarray(outputs[3].column(name))
        if serial.dtype.kind == 'f':
            assert np.allclose(serial, parallel, equal_nan=True), name
        else:
            assert (serial == parallel).all(), name


def _identity_table(keys):
    columns = {name: [key[index] for key in keys]
               for index, name in enumerate(IDENTITY_COLS)}
    return pa.table(columns)


def test_identity_keys_reads_all_five_columns():
    keys = [(1, 10, 100, 9, 1), (1, 10, 100, 9, 2)]
    assert identity_keys(_identity_table(keys)) == keys


def test_dump_row_order_inverts_a_permuted_dump():
    source = [(1, 10, 100, 9, index) for index in range(5)]
    permuted = [source[index] for index in (3, 0, 4, 1, 2)]
    order = dump_row_order(permuted, source)
    assert [permuted[index] for index in order] == source


def test_dump_row_order_is_identity_for_an_aligned_dump():
    source = [(1, 10, 100, 9, index) for index in range(4)]
    assert dump_row_order(source, source).tolist() == [0, 1, 2, 3]


def test_dump_row_order_rejects_a_duplicated_key():
    duplicated = [(1, 10, 100, 9, 1), (1, 10, 100, 9, 1)]
    with pytest.raises(ValueError, match='duplicate identity key'):
        dump_row_order(duplicated, duplicated)


def test_dump_row_order_rejects_a_dump_missing_source_events():
    source = [(1, 10, 100, 9, index) for index in range(3)]
    with pytest.raises(ValueError, match='absent from the dump'):
        dump_row_order(source[:2], source)


def test_identity_columns_are_the_documented_five():
    assert IDENTITY_COLS == ['event_run', 'event_id', 'event_luminosity_block',
                             'source_batch_id', 'source_microbatch_id']


def test_alignment_guard_accepts_an_aligned_dump():
    couples = [[[0, 1], [2, 3]], [[0, 4]]]
    assert assert_dump_aligned(couples, [5, 5]) == 2


def test_alignment_guard_rejects_an_out_of_range_track_index():
    couples = [[[0, 1]], [[3, 630]]]
    with pytest.raises(ValueError, match="not aligned"):
        assert_dump_aligned(couples, [5, 474])


def test_alignment_guard_skips_events_without_couples():
    assert assert_dump_aligned([[], [[0, 1]]], [5, 5]) == 1


def test_alignment_guard_stops_after_the_sample():
    couples = [[[0, 1]]] * 10 + [[[999, 1000]]]
    assert assert_dump_aligned(couples, [5] * 10 + [5], sample=10) == 10


def test_parallel_eval_build_matches_the_serial_build():
    import glob
    import json
    import os

    from scripts.python.build_triplet_filter_table import _shard_blocks, build_eval

    root = os.path.join(os.path.dirname(__file__), '..')
    dump = os.path.join(root, 'data', 'dumps', 'couples_k125_dump.parquet')
    pattern = os.path.join(root, 'data', 'low-pt', 'eval', '*.parquet')
    if not (os.path.exists(dump) and glob.glob(pattern)):
        pytest.skip('eval dump or shards not present')

    outputs = {}
    for workers in (0, 3):
        blocks = _shard_blocks(dump, pattern, 24)
        path = os.path.join(
            os.path.dirname(__file__), f'_parallel_eval_{workers}.parquet')
        build_eval(blocks, 100, 20, np.random.default_rng(0), path, True,
                   workers=workers, seed=0)
        outputs[workers] = {
            part: pq.read_table(path.replace('.parquet', f'_{part}.parquet'))
            for part in ('gt', 'sub')}
        with open(path.replace('.parquet', '_meta.json')) as handle:
            outputs[workers]['meta'] = json.load(handle)
        for suffix in ('_gt.parquet', '_sub.parquet', '_meta.json'):
            os.remove(path.replace('.parquet', suffix))

    assert outputs[0]['meta'] == outputs[3]['meta']
    for part in ('gt', 'sub'):
        serial, parallel = outputs[0][part], outputs[3][part]
        assert serial.num_rows == parallel.num_rows
        for name in serial.schema.names:
            left = np.asarray(serial.column(name))
            right = np.asarray(parallel.column(name))
            if left.dtype.kind == 'f':
                assert np.allclose(left, right, equal_nan=True), f'{part}.{name}'
            else:
                assert (left == right).all(), f'{part}.{name}'


def test_unknown_negative_mode_is_rejected():
    couple_row, is_gt = _candidate_pool()
    with pytest.raises(ValueError, match="neg_mode"):
        _sample_negative_rows(
            is_gt, couple_row, 3, "hardest", np.random.default_rng(0))
