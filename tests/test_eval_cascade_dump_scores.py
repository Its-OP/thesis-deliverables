from __future__ import annotations

import glob
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from eval_cascade_pipeline import OUTPUT_SCHEMA, _write_parquet, main

_DELIVERABLES = os.path.join(os.path.dirname(__file__), '..')
_STAGE1 = os.path.join(_DELIVERABLES, 'models', 'prefilter_best.pt')
_STAGE2 = os.path.join(_DELIVERABLES, 'models', 'stage2_best.pt')
_STAGE3 = os.path.join(_DELIVERABLES, 'models', 'couple_reranker_best.pt')
# The frozen legacy sidecar config — the production checkpoints under test
# were trained on its 16-channel feature set; the live config is now 32.
_DATA_CONFIG = os.path.join(
    _DELIVERABLES, 'data', 'low-pt',
    'lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml')
_SUBSET_DIR = '/Users/oleh/Projects/masters/part/data/low-pt/subset/val'

_SCORE_FIELDS = ['stage1_scores', 'stage2_scores', 'stage3_couple_scores']


def test_output_schema_has_score_fields():
    names = OUTPUT_SCHEMA.names
    for field_name in _SCORE_FIELDS:
        assert field_name in names
        assert OUTPUT_SCHEMA.field(field_name).type == pa.list_(pa.float32())


def test_write_parquet_round_trip_with_scores(tmp_path):
    row = {
        'event_run': 1, 'event_id': 2, 'event_luminosity_block': 3,
        'source_batch_id': 0, 'source_microbatch_id': 0, 'stage': 'couples',
        'stage1_sorted_indices': [4, 0, 2],
        'stage2_sorted_indices': [0, 4],
        'stage3_sorted_couples': [[0, 4]],
        'stage1_scores': [2.5, 1.0, -0.5],
        'stage2_scores': [1.5, 0.25],
        'stage3_couple_scores': [0.75],
    }
    out = str(tmp_path / 'dump.parquet')
    _write_parquet([row], out)
    table = pq.read_table(out)
    assert table.schema.equals(OUTPUT_SCHEMA)
    read = table.to_pylist()[0]
    for field_name in _SCORE_FIELDS:
        assert read[field_name] == pytest.approx(row[field_name])
    assert read['stage1_sorted_indices'] == row['stage1_sorted_indices']
    assert read['stage3_sorted_couples'] == row['stage3_sorted_couples']


@pytest.mark.skipif(
    not (os.path.exists(_STAGE1) and os.path.exists(_STAGE2) and os.path.exists(_STAGE3)
         and os.path.exists(_DATA_CONFIG) and glob.glob(f'{_SUBSET_DIR}/*.parquet')),
    reason='cascade checkpoints, data config, or subset data not present',
)
def test_couples_dump_scores_align_with_indices(tmp_path):
    out = str(tmp_path / 'scores_dump.parquet')
    main([
        '--stage', 'couples',
        '--stage1-weights', _STAGE1,
        '--stage2-weights', _STAGE2,
        '--stage3-weights', _STAGE3,
        '--val-data-dir', _SUBSET_DIR,
        '--data-config', _DATA_CONFIG,
        '--output', out,
        '--device', 'cpu',
        '--batch-size', '4',
        '--max-events', '8',
        '--num-couples', '50',
    ])
    table = pq.read_table(out)
    assert table.num_rows == 8
    for row in table.to_pylist():
        # Scores are parallel to their index columns and sorted descending.
        assert len(row['stage1_scores']) == len(row['stage1_sorted_indices'])
        assert len(row['stage2_scores']) == len(row['stage2_sorted_indices'])
        assert len(row['stage3_couple_scores']) == len(row['stage3_sorted_couples'])
        for scores in (row['stage1_scores'], row['stage2_scores'], row['stage3_couple_scores']):
            assert all(a >= b for a, b in zip(scores, scores[1:]))
            assert all(s == s and abs(s) != float('inf') for s in scores)
        assert len(row['stage3_sorted_couples']) > 0
