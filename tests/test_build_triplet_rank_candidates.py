from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from build_triplet_rank_candidates import (
    CANDIDATE_SCHEMA,
    FILTER_MODEL_GLOB,
    _dump_blocks,
    _window_selection,
    load_filter_model,
    main,
)
from build_triplet_filter_table import IDENTITY_COLS

_HAVE_FILTER = bool(glob.glob(FILTER_MODEL_GLOB))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_v2_has_the_single_filter_score_and_row_kind():
    names = CANDIDATE_SCHEMA.names
    assert 'filter_score' in names
    assert 'row_kind' in names
    assert 'n_tierh' in names
    assert 'gbdt6_score' not in names
    assert 'gbdt8_score' not in names
    assert CANDIDATE_SCHEMA.field('filter_score').type == pa.list_(pa.float32())
    assert CANDIDATE_SCHEMA.field('row_kind').type == pa.list_(pa.uint8())
    assert CANDIDATE_SCHEMA.field('n_tail_total').type == pa.int32()
    for name in IDENTITY_COLS:
        assert name in names


# ---------------------------------------------------------------------------
# Window selection (pure)
# ---------------------------------------------------------------------------

def test_window_head_is_descending_by_score():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3], dtype=np.float32)
    is_gt = np.zeros(5, dtype=bool)
    rows, kinds, n_tail = _window_selection(
        scores, is_gt, window=3, tail_sample=0,
        generator=np.random.default_rng(0))
    assert rows.tolist() == [1, 3, 2]
    assert kinds.tolist() == [0, 0, 0]
    assert n_tail == 2


def test_gt_rows_beyond_the_window_are_force_included():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.05], dtype=np.float32)
    is_gt = np.array([False, False, False, False, True])
    rows, kinds, n_tail = _window_selection(
        scores, is_gt, window=2, tail_sample=0,
        generator=np.random.default_rng(0))
    assert rows.tolist()[:2] == [0, 1]
    assert kinds.tolist()[:2] == [0, 0]
    assert 4 in rows.tolist()
    assert kinds[rows.tolist().index(4)] == 1
    # Forced GT never counts toward the reweighting tail.
    assert n_tail == 2


def test_gt_inside_the_window_stays_a_window_row():
    scores = np.array([0.9, 0.8, 0.1], dtype=np.float32)
    is_gt = np.array([True, False, False])
    rows, kinds, _ = _window_selection(
        scores, is_gt, window=2, tail_sample=0,
        generator=np.random.default_rng(0))
    assert rows.tolist() == [0, 1]
    assert kinds.tolist() == [0, 0]


def test_tail_sample_draws_beyond_the_window_without_replacement():
    scores = np.linspace(1.0, 0.0, 40, dtype=np.float32)
    is_gt = np.zeros(40, dtype=bool)
    rows, kinds, n_tail = _window_selection(
        scores, is_gt, window=10, tail_sample=5,
        generator=np.random.default_rng(0))
    head, tail = rows[:10], rows[10:]
    assert kinds[:10].tolist() == [0] * 10
    assert kinds[10:].tolist() == [2] * 5
    assert len(set(tail.tolist())) == 5
    assert all(index >= 10 for index in np.argsort(-scores)[tail])
    assert n_tail == 30


def test_no_window_returns_the_full_list_sorted_descending():
    scores = np.array([0.2, 0.9, 0.4], dtype=np.float32)
    is_gt = np.array([False, True, False])
    rows, kinds, n_tail = _window_selection(
        scores, is_gt, window=None, tail_sample=0,
        generator=np.random.default_rng(0))
    assert rows.tolist() == [1, 2, 0]
    assert kinds.tolist() == [0, 0, 0]
    assert n_tail == 0


def test_window_selection_is_deterministic_under_a_seed():
    scores = np.random.default_rng(3).random(100).astype(np.float32)
    is_gt = np.zeros(100, dtype=bool)
    first = _window_selection(scores, is_gt, window=20, tail_sample=10,
                              generator=np.random.default_rng(7))
    second = _window_selection(scores, is_gt, window=20, tail_sample=10,
                               generator=np.random.default_rng(7))
    assert first[0].tolist() == second[0].tolist()


# ---------------------------------------------------------------------------
# Filter model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_FILTER, reason='sweep filter joblib not present')
def test_load_filter_model_resolves_the_100_feature_champion():
    model = load_filter_model(FILTER_MODEL_GLOB)
    assert int(model.n_features_in_) == 100
    # CPU inference must work in a forked worker: predict on a zero row.
    probabilities = model.predict_proba(np.zeros((1, 100), dtype=np.float32))
    assert probabilities.shape == (1, 2)


def test_load_filter_model_rejects_an_ambiguous_glob(tmp_path):
    for name in ('a.joblib', 'b.joblib'):
        (tmp_path / name).write_bytes(b'x')
    with pytest.raises(AssertionError, match='exactly one'):
        load_filter_model(str(tmp_path / '*.joblib'))


# ---------------------------------------------------------------------------
# Synthetic end-to-end fixture
# ---------------------------------------------------------------------------

def _synthetic_events(n_events, seed=0):
    """5-track events; GT = tracks 0,1,2; couples cover the GT couple."""
    generator = np.random.default_rng(seed)
    events = []
    for index in range(n_events):
        jitter = 0.02 * generator.standard_normal(5)
        events.append(dict(
            src={
                'event_n_tracks': 5,
                'track_pt': (np.array([1.0, 1.2, 0.9, 1.1, 0.8]) + jitter).tolist(),
                'track_eta': [0.10, 0.15, 0.12, 0.18, 0.30],
                'track_phi': [0.05, 0.10, 0.08, 0.12, 0.50],
                'track_charge': [1.0, 1.0, -1.0, -1.0, 1.0],
                'track_dz_significance': [0.20, 0.25, 0.22, 0.28, 9.00],
                'track_dxy_significance': [0.5, 0.6, 0.7, 0.8, 0.9],
                'track_dca_significance': [1.0, 1.1, 1.2, 1.3, 1.4],
                'track_n_valid_pixel_hits': [4.0, 4.0, 3.0, 5.0, 2.0],
                'track_norm_chi2': [1.0, 1.2, 0.9, 1.1, 2.0],
                'track_pt_error': [0.01, 0.02, 0.03, 0.04, 0.05],
                'track_covariance_phi_phi': [0.001, 0.002, 0.003, 0.004, 0.005],
                'track_covariance_lambda_lambda': [0.0011, 0.0021, 0.0031,
                                                   0.0041, 0.0051],
                'track_label_from_tau': [1.0, 1.0, 1.0, 0.0, 0.0],
                'track_vertex_x': [0.10, 0.11, 0.09, 0.02, -0.50],
                'track_vertex_y': [0.05, 0.06, 0.04, 0.01, 0.60],
                'track_vertex_z': [1.00, 1.02, 0.98, 0.20, -4.00],
                'track_dz': [0.30, 0.34, 0.28, 0.10, 7.00],
                'event_primary_vertex_x': 0.0,
                'event_primary_vertex_y': 0.0,
                'sv_x': [0.10], 'sv_y': [0.05], 'sv_z': [1.00],
                'sv_dlen_sig': [4.5], 'sv_mass': [0.62],
                'other_track_pt': [0.40], 'other_track_eta': [0.13],
                'other_track_phi': [0.09], 'other_track_dz': [0.31],
                'event_run': 1, 'event_id': 1000 + index,
                'event_luminosity_block': 7,
                'source_batch_id': index // 2, 'source_microbatch_id': index % 2,
            },
            dump={
                'stage1_sorted_indices': [2, 0, 3, 1, 4],
                'stage1_scores': [0.9, 0.8, 0.7, 0.6, 0.5],
                'stage2_sorted_indices': [2, 0, 3, 1],
                'stage2_scores': [0.5, 0.4, 0.35, 0.3],
                'stage3_sorted_couples': [[0, 1], [0, 2], [2, 3]],
                'stage3_couple_scores': [0.95, 0.85, 0.75],
                'event_run': 1, 'event_id': 1000 + index,
                'event_luminosity_block': 7,
                'source_batch_id': index // 2, 'source_microbatch_id': index % 2,
            },
        ))
    return events


def _write_fixture(tmp_path, events, dump_order):
    src_dir = tmp_path / 'shards'
    src_dir.mkdir()
    half = len(events) // 2
    for shard, chunk in enumerate((events[:half], events[half:])):
        columns = {key: [event['src'][key] for event in chunk]
                   for key in chunk[0]['src']}
        pq.write_table(pa.table(columns), src_dir / f'src_{shard:03d}.parquet')
    dump_columns = {key: [events[r]['dump'][key] for r in dump_order]
                    for key in events[0]['dump']}
    dump_path = tmp_path / 'dump.parquet'
    pq.write_table(pa.table(dump_columns), dump_path)
    return str(dump_path), str(src_dir / '*.parquet')


@pytest.mark.skipif(not _HAVE_FILTER, reason='sweep filter joblib not present')
def test_builder_end_to_end_on_a_permuted_dump(tmp_path):
    events = _synthetic_events(4)
    dump_path, src_glob = _write_fixture(tmp_path, events, dump_order=[2, 0, 3, 1])
    out_dir = tmp_path / 'out'
    main(['--role', 'train', '--dump', dump_path, '--src-glob', src_glob,
          '--out-dir', str(out_dir), '--window', '4', '--tail-sample', '2',
          '--top-c', '125'])

    table = pq.read_table(out_dir / 'candidates_train.parquet')
    assert table.num_rows == 4
    assert table.schema.equals(CANDIDATE_SCHEMA)
    rows = table.to_pylist()
    # Output rows follow SOURCE-SHARD order despite the permuted dump.
    assert [row['event_id'] for row in rows] == [1000, 1001, 1002, 1003]
    for row in rows:
        n = len(row['cand_i'])
        assert n == len(row['filter_score']) == len(row['row_kind'])
        assert row['n_tierh'] >= n - sum(kind != 0 for kind in row['row_kind'])
        window_scores = [score for score, kind
                         in zip(row['filter_score'], row['row_kind']) if kind == 0]
        assert window_scores == sorted(window_scores, reverse=True)
        assert len(window_scores) <= 4
        assert row['recon'] is True
        assert any(row['is_gt'])
        # The stage-1 scatter survived the identity join: couple members carry
        # finite stage-2 scores.
        track_s2 = row['track_s2']
        for i, kind in zip(row['cand_i'], row['row_kind']):
            assert np.isfinite(track_s2[i])
    manifest = json.loads((out_dir / 'build_manifest_train.json').read_text())
    assert manifest['window'] == 4
    assert manifest['top_c'] == 125
    ops = json.loads((out_dir / 'operating_points.json').read_text())
    assert ops['score_column'] == 'filter_score'
    assert set(ops['taus']) == {'p99', 'p95'}


@pytest.mark.skipif(not _HAVE_FILTER, reason='sweep filter joblib not present')
def test_eval_role_stores_the_full_sorted_list(tmp_path):
    events = _synthetic_events(4)
    dump_path, src_glob = _write_fixture(tmp_path, events, dump_order=[1, 3, 0, 2])
    out_dir = tmp_path / 'out'
    main(['--role', 'eval', '--dump', dump_path, '--src-glob', src_glob,
          '--out-dir', str(out_dir), '--top-c', '125'])
    rows = pq.read_table(out_dir / 'candidates_eval.parquet').to_pylist()
    for row in rows:
        assert all(kind == 0 for kind in row['row_kind'])
        assert len(row['cand_i']) == row['n_tierh']
        assert row['n_tail_total'] == 0
        scores = row['filter_score']
        assert scores == sorted(scores, reverse=True)


def test_misaligned_dump_is_rejected(tmp_path):
    events = _synthetic_events(4)
    dump_path, src_glob = _write_fixture(tmp_path, events, dump_order=[2, 0, 3, 1])
    # Corrupt one identity so a source event is absent from the dump.
    table = pq.read_table(dump_path)
    ids = table['event_id'].to_pylist()
    ids[0] = 999999
    table = table.set_column(table.schema.get_field_index('event_id'),
                             'event_id', pa.array(ids))
    pq.write_table(table, dump_path)
    with pytest.raises(ValueError, match='absent from the dump'):
        list(_dump_blocks(dump_path, src_glob, None))
