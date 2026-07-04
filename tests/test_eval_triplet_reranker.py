from __future__ import annotations

import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

_DELIVERABLES = os.path.join(os.path.dirname(__file__), '..')
_DUMP = os.path.join(_DELIVERABLES, 'data', 'low-pt', 'eval', 'perstage_couples_val.parquet')
_SRC_GLOB = '/Users/oleh/Projects/masters/part/data/low-pt/val/val_*.parquet'
_GBDT6 = os.path.join(_DELIVERABLES, 'models', 'third_pion_filter_gbdt_full_P2.joblib')
_HAVE_REAL_VAL = os.path.exists(_DUMP) and glob.glob(_SRC_GLOB) and os.path.exists(_GBDT6)


@pytest.fixture(scope='module')
def artifact_and_checkpoint(tmp_path_factory):
    if not _HAVE_REAL_VAL:
        pytest.skip('real VAL dump/src/models not present')
    from build_triplet_rank_candidates import main as build_main
    from train_triplet_reranker import main as train_main

    out_dir = str(tmp_path_factory.mktemp('triplet_eval'))
    build_main(['--out-dir', out_dir, '--tag', 'val', '--max-events', '40'])
    run_dir = os.path.join(out_dir, 'run')
    train_main([
        '--candidates', os.path.join(out_dir, 'candidates_val.parquet'),
        '--tracks', os.path.join(out_dir, 'tracks_val.parquet'),
        '--norm-stats', os.path.join(out_dir, 'norm_stats.json'),
        '--experiment-dir', run_dir,
        '--tau', '0.0',
        '--epochs', '1',
        '--batch-size', '4',
        '--num-negatives', '10',
        '--device', 'cpu',
        '--norm-stats-events', '20',
    ])
    return out_dir, os.path.join(run_dir, 'checkpoints', 'best_model.pt')


def test_eval_script_reports_model_vs_baselines(artifact_and_checkpoint, tmp_path, capsys):
    from eval_triplet_reranker import main as eval_main

    out_dir, checkpoint = artifact_and_checkpoint
    out_json = str(tmp_path / 'triplet_reranker_eval.json')
    eval_main([
        '--checkpoint', checkpoint,
        '--candidates', os.path.join(out_dir, 'candidates_val.parquet'),
        '--tracks', os.path.join(out_dir, 'tracks_val.parquet'),
        '--out-json', out_json,
        '--device', 'cpu',
    ])
    printed = capsys.readouterr().out
    for label in ('model', 'gbdt', 'couple_rank_lex', 'random'):
        assert label in printed

    with open(out_json) as fh:
        result = json.load(fh)
    assert result['n_events'] == 40
    model_curve = result['model']['t_at_k']
    for k in result['k_values']:
        assert 0.0 <= model_curve[str(k)] <= 1.0
    assert result['baselines']['t_at_k']['gbdt']['100'] > 0.0
    # The model ranks over the same survivor lists as the baselines.
    assert result['model']['ceiling'] == pytest.approx(result['baselines']['ceiling'])
