from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from build_triplet_rank_candidates import (
    CANDIDATE_SCHEMA,
    event_candidates,
    load_gbdt_models,
    main,
)

_DELIVERABLES = os.path.join(os.path.dirname(__file__), '..')
_GBDT6 = os.path.join(_DELIVERABLES, 'models', 'third_pion_filter_gbdt_full_P2.joblib')
_GBDT8 = os.path.join(_DELIVERABLES, 'models', 'third_pion_filter_gbdt8_full_P2.joblib')
_DUMP = os.path.join(_DELIVERABLES, 'data', 'low-pt', 'eval', 'perstage_couples_val.parquet')
_SRC_GLOB = '/Users/oleh/Projects/masters/part/data/low-pt/val/val_*.parquet'

_LIST_FIELDS = ['cand_i', 'cand_j', 'cand_k', 'couple_rank',
                'gbdt6_score', 'gbdt8_score', 'is_gt']

_HAVE_MODELS = os.path.exists(_GBDT6) and os.path.exists(_GBDT8)


def _synthetic_event(couples):
    # 5 tracks, charges (+,+,-,-,+); GT = tracks 0,1,2. Mirrors tests/test_triplet_join.py.
    dump_cols = ([list(range(5))], [couples])
    src_cols = {
        'event_n_tracks': [5],
        'track_pt': [[1.0, 1.2, 0.9, 1.1, 0.8]],
        'track_eta': [[0.10, 0.15, 0.12, 0.18, 3.00]],
        'track_phi': [[0.05, 0.10, 0.08, 0.12, 2.50]],
        'track_charge': [[1.0, 1.0, -1.0, -1.0, 1.0]],
        'track_dz_significance': [[0.20, 0.25, 0.22, 0.28, 9.00]],
        'track_dxy_significance': [[0.5, 0.6, 0.7, 0.8, 0.9]],
        'track_dca_significance': [[1.0, 1.1, 1.2, 1.3, 1.4]],
        'track_n_valid_pixel_hits': [[4.0, 4.0, 3.0, 5.0, 2.0]],
        'track_norm_chi2': [[1.0, 1.2, 0.9, 1.1, 2.0]],
        'track_pt_error': [[0.01, 0.02, 0.03, 0.04, 0.05]],
        'track_covariance_phi_phi': [[0.001, 0.002, 0.003, 0.004, 0.005]],
        'track_covariance_lambda_lambda': [[0.0011, 0.0021, 0.0031, 0.0041, 0.0051]],
        'track_label_from_tau': [[1.0, 1.0, 1.0, 0.0, 0.0]],
    }
    return dump_cols, src_cols


def test_candidate_schema_fields():
    names = CANDIDATE_SCHEMA.names
    for field_name in ['n_tracks', 'n_candidates', 'gt_i', 'gt_j', 'gt_k', 'recon'] + _LIST_FIELDS:
        assert field_name in names
    assert CANDIDATE_SCHEMA.field('cand_i').type == pa.list_(pa.int16())
    assert CANDIDATE_SCHEMA.field('gbdt6_score').type == pa.list_(pa.float32())
    assert CANDIDATE_SCHEMA.field('is_gt').type == pa.list_(pa.bool_())
    assert CANDIDATE_SCHEMA.field('gt_i').type == pa.int16()
    assert CANDIDATE_SCHEMA.field('recon').type == pa.bool_()


@pytest.mark.skipif(not _HAVE_MODELS, reason='GBDT joblibs not present')
def test_event_candidates_synthetic():
    from utils.triplet_join import build_track_lorentz, candidates_for_tier

    dump_cols, src_cols = _synthetic_event([[0, 1], [0, 2]])
    models = load_gbdt_models()
    row = event_candidates(0, dump_cols, src_cols, top_c=100, gbdt_models=models)

    assert row['n_tracks'] == 5
    assert row['recon'] is True
    assert (row['gt_i'], row['gt_j'], row['gt_k']) == (0, 1, 2)
    n = row['n_candidates']
    for field_name in _LIST_FIELDS:
        assert len(row[field_name]) == n

    # Candidate indices reproduce the Tier-H enumeration exactly.
    lorentz = build_track_lorentz(torch.tensor(src_cols['track_pt'][0]),
                                  torch.tensor(src_cols['track_eta'][0]),
                                  torch.tensor(src_cols['track_phi'][0]))
    triplets, couple_row = candidates_for_tier(
        'H', torch.tensor([[0, 1], [0, 2]]), torch.arange(5),
        lorentz=lorentz, charge=torch.tensor(src_cols['track_charge'][0]),
    )
    assert row['cand_i'] == triplets[:, 0].tolist()
    assert row['cand_j'] == triplets[:, 1].tolist()
    assert row['cand_k'] == triplets[:, 2].tolist()
    assert row['couple_rank'] == couple_row.tolist()

    # is_gt marks decompositions of the GT 3-set {0,1,2} only.
    for flag, i, j, k in zip(row['is_gt'], row['cand_i'], row['cand_j'], row['cand_k']):
        assert flag == (sorted((i, j, k)) == [0, 1, 2])
    assert sum(row['is_gt']) >= 1
    assert all(0.0 <= s <= 1.0 for s in row['gbdt6_score'])
    assert all(0.0 <= s <= 1.0 for s in row['gbdt8_score'])


@pytest.mark.skipif(not _HAVE_MODELS, reason='GBDT joblibs not present')
def test_event_candidates_non_reconstructable():
    # GT couple (0,1) absent from the couple list -> not reconstructable.
    dump_cols, src_cols = _synthetic_event([[0, 3]])
    models = load_gbdt_models()
    row = event_candidates(0, dump_cols, src_cols, top_c=100, gbdt_models=models)
    assert row['recon'] is False
    assert (row['gt_i'], row['gt_j'], row['gt_k']) == (-1, -1, -1)
    assert sum(row['is_gt']) == 0


@pytest.mark.skipif(
    not (_HAVE_MODELS and os.path.exists(_DUMP) and glob.glob(_SRC_GLOB)),
    reason='VAL dump, source parquet, or GBDT joblibs not present',
)
def test_builder_integration_real_val(tmp_path):
    main([
        '--out-dir', str(tmp_path),
        '--tag', 'val',
        '--max-events', '40',
    ])
    table = pq.read_table(str(tmp_path / 'candidates_val.parquet'))
    assert table.num_rows == 40
    assert table.schema.equals(CANDIDATE_SCHEMA)
    n_recon = 0
    for row in table.to_pylist():
        n = row['n_candidates']
        assert n == len(row['cand_i']) == len(row['gbdt6_score']) == len(row['is_gt'])
        assert 0 < n <= 32768
        assert max(row['cand_k']) < row['n_tracks']
        gt_flags = sum(row['is_gt'])
        if row['recon']:
            n_recon += 1
            assert 1 <= gt_flags <= 3
            gt_set = sorted((row['gt_i'], row['gt_j'], row['gt_k']))
            for flag, i, j, k in zip(row['is_gt'], row['cand_i'], row['cand_j'], row['cand_k']):
                if flag:
                    assert sorted((i, j, k)) == gt_set
        else:
            assert gt_flags == 0
            assert row['gt_i'] == -1
    assert n_recon > 0


def _replicate(dump_cols, src_cols, n):
    dump = ([dump_cols[0][0]] * n, [dump_cols[1][0]] * n)
    src = {key: value * n for key, value in src_cols.items()}
    return dump, src


@pytest.mark.skipif(not _HAVE_MODELS, reason='GBDT joblibs not present')
def test_event_failure_writes_empty_row(tmp_path, monkeypatch):
    import build_triplet_rank_candidates as builder

    dump_cols, src_cols = _replicate(*_synthetic_event([[0, 1], [0, 2]]), 3)
    models = load_gbdt_models()
    real_event_features = builder.event_features

    def flaky(r, *args, **kwargs):
        if r == 1:
            raise ValueError('boom')
        return real_event_features(r, *args, **kwargs)

    monkeypatch.setattr(builder, 'event_features', flaky)
    out_path = str(tmp_path / 'cand.parquet')
    builder.write_candidates(dump_cols, src_cols, range(3), 100, models, out_path, chunk_size=2)

    table = pq.read_table(out_path)
    assert table.num_rows == 3
    rows = table.to_pylist()
    assert rows[1]['n_candidates'] == 0
    assert rows[1]['recon'] is False
    assert rows[1]['gt_i'] == -1
    assert rows[1]['cand_i'] == []
    assert rows[0]['n_candidates'] > 0
    assert rows[2]['n_candidates'] > 0
    assert not glob.glob(out_path + '.chunk*')


@pytest.mark.skipif(not _HAVE_MODELS, reason='GBDT joblibs not present')
def test_chunk_resume_skips_complete_chunks(tmp_path):
    import build_triplet_rank_candidates as builder

    dump_cols, src_cols = _replicate(*_synthetic_event([[0, 1], [0, 2]]), 3)
    models = load_gbdt_models()
    out_path = str(tmp_path / 'cand.parquet')

    # Pre-existing complete chunk 0 with sentinel content simulates a resumed run;
    # the builder must skip it rather than recompute.
    sentinel = builder._empty_row(src_cols, 0)
    sentinel['n_tracks'] = 999
    columns = {field.name: [sentinel[field.name]] for field in CANDIDATE_SCHEMA}
    pq.write_table(pa.table(columns, schema=CANDIDATE_SCHEMA), out_path + '.chunk000000')

    builder.write_candidates(dump_cols, src_cols, range(3), 100, models, out_path, chunk_size=1)

    rows = pq.read_table(out_path).to_pylist()
    assert len(rows) == 3
    assert rows[0]['n_tracks'] == 999
    assert rows[1]['n_candidates'] > 0
    assert rows[2]['n_candidates'] > 0
    assert not glob.glob(out_path + '.chunk*')


@pytest.mark.skipif(not _HAVE_MODELS, reason='GBDT joblibs not present')
def test_batched_chunk_scores_match_per_event():
    import build_triplet_rank_candidates as builder

    dump_cols, src_cols = _replicate(*_synthetic_event([[0, 1], [0, 2]]), 3)
    models = load_gbdt_models()
    rows, n_failed = builder._process_chunk(list(range(3)), 0, dump_cols, src_cols, 100, models)
    assert n_failed == 0
    for r, row in enumerate(rows):
        reference = event_candidates(r, dump_cols, src_cols, top_c=100, gbdt_models=models)
        assert row['gbdt6_score'] == reference['gbdt6_score']
        assert row['gbdt8_score'] == reference['gbdt8_score']
        assert row['cand_i'] == reference['cand_i']
        assert row['is_gt'] == reference['is_gt']


@pytest.mark.skipif(
    not (_HAVE_MODELS and os.path.exists(_DUMP) and glob.glob(_SRC_GLOB)),
    reason='VAL dump, source parquet, or GBDT joblibs not present',
)
def test_parallel_workers_match_single_process(tmp_path):
    single_dir, parallel_dir = tmp_path / 'single', tmp_path / 'parallel'
    common = ['--tag', 'val', '--max-events', '40', '--chunk-size', '10', '--skip-tracks']
    main(['--out-dir', str(single_dir), '--workers', '1'] + common)
    main(['--out-dir', str(parallel_dir), '--workers', '2'] + common)
    single = pq.read_table(str(single_dir / 'candidates_val.parquet'))
    parallel = pq.read_table(str(parallel_dir / 'candidates_val.parquet'))
    assert single.num_rows == parallel.num_rows == 40
    assert single.equals(parallel)
    assert not glob.glob(str(parallel_dir / 'candidates_val.parquet.chunk*'))


@pytest.mark.skipif(
    not glob.glob(_SRC_GLOB),
    reason='source parquet not present',
)
def test_tracks_consolidation(tmp_path):
    main([
        '--out-dir', str(tmp_path),
        '--tag', 'val',
        '--max-events', '10',
        '--tracks-only',
    ])
    tracks = pq.read_table(str(tmp_path / 'tracks_val.parquet'))
    assert tracks.num_rows == 10
    assert 'track_pt' in tracks.schema.names
    assert 'track_label_from_tau' in tracks.schema.names
    first = tracks.to_pylist()[0]
    assert len(first['track_pt']) == first['event_n_tracks']
