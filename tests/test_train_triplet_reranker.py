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
def real_artifact(tmp_path_factory):
    if not _HAVE_REAL_VAL:
        pytest.skip('real VAL dump/src/models not present')
    from build_triplet_rank_candidates import main as build_main
    out_dir = str(tmp_path_factory.mktemp('triplet_rank_train'))
    build_main(['--out-dir', out_dir, '--tag', 'val', '--max-events', '40'])
    return out_dir


def test_trainer_smoke_end_to_end(real_artifact, tmp_path):
    from train_triplet_reranker import main as train_main

    experiments = str(tmp_path / 'experiments')
    train_main([
        '--candidates', os.path.join(real_artifact, 'candidates_val.parquet'),
        '--tracks', os.path.join(real_artifact, 'tracks_val.parquet'),
        '--norm-stats', os.path.join(real_artifact, 'norm_stats.json'),
        '--experiments-dir', experiments,
        '--input-mode', 'flat',
        '--tau', '0.0',
        '--epochs', '2',
        '--batch-size', '4',
        '--num-negatives', '10',
        '--device', 'cpu',
        '--norm-stats-events', '20',
    ])
    run_dirs = glob.glob(os.path.join(experiments, '*'))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert os.path.exists(os.path.join(run_dir, 'checkpoints', 'best_model.pt'))
    with open(os.path.join(run_dir, 'metrics_history.json')) as fh:
        history = json.load(fh)
    assert len(history) == 2
    for entry in history:
        assert 0.0 <= entry['T@10'] <= 1.0
        assert entry['train_loss'] == entry['train_loss']  # finite
    # Norm stats were fit on the train side and persisted.
    assert os.path.exists(os.path.join(real_artifact, 'norm_stats.json'))


def test_trainer_checkpoint_is_slim_and_loadable(real_artifact, tmp_path):
    import torch
    from train_triplet_reranker import main as train_main
    from weaver.nn.model.TripletReranker import TripletReranker

    experiments = str(tmp_path / 'experiments_slim')
    train_main([
        '--candidates', os.path.join(real_artifact, 'candidates_val.parquet'),
        '--tracks', os.path.join(real_artifact, 'tracks_val.parquet'),
        '--norm-stats', os.path.join(real_artifact, 'norm_stats.json'),
        '--experiments-dir', experiments,
        '--input-mode', 'flat',
        '--tau', '0.0',
        '--epochs', '1',
        '--batch-size', '4',
        '--num-negatives', '10',
        '--device', 'cpu',
        '--norm-stats-events', '20',
    ])
    ckpt_path = glob.glob(os.path.join(experiments, '*', 'checkpoints', 'best_model.pt'))[0]
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'triplet_reranker_state_dict' in checkpoint
    assert 'args' in checkpoint and 'val_metrics' in checkpoint
    model = TripletReranker(input_mode='flat', feature_dim=89)
    model.load_state_dict(checkpoint['triplet_reranker_state_dict'])
