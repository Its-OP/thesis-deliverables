from __future__ import annotations

import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from triplet_rank_fixture import synthetic_events, write_fixture


def _have_filter():
    from build_triplet_rank_candidates import FILTER_MODEL_GLOB
    return bool(glob.glob(FILTER_MODEL_GLOB))


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    if not _have_filter():
        pytest.skip('sweep filter joblib not present')
    from build_triplet_rank_candidates import main as build_main
    root = tmp_path_factory.mktemp('trainer_v2')
    events = synthetic_events(6)
    dump_path, src_glob = write_fixture(root, events, dump_order=[4, 2, 0, 5, 3, 1])
    out_dir = root / 'artifact'
    build_main(['--role', 'train', '--dump', dump_path, '--src-glob', src_glob,
                '--out-dir', str(out_dir), '--window', '6', '--tail-sample', '2',
                '--top-c', '125'])
    build_main(['--role', 'eval', '--dump', dump_path, '--src-glob', src_glob,
                '--out-dir', str(out_dir), '--top-c', '125'])
    return dict(out_dir=str(out_dir), src_glob=src_glob)


def _base_args(artifacts, run_dir, extra):
    out_dir = artifacts['out_dir']
    return [
        '--candidates', os.path.join(out_dir, 'candidates_train.parquet'),
        '--src-glob', artifacts['src_glob'],
        '--eval-candidates', os.path.join(out_dir, 'candidates_eval.parquet'),
        '--eval-src-glob', artifacts['src_glob'],
        '--operating-points', os.path.join(out_dir, 'operating_points.json'),
        '--norm-stats', os.path.join(run_dir, 'norm_stats.json'),
        '--experiment-dir', run_dir,
        '--gate', 'p95', '--eval-gates', 'p95,p99',
        '--device', 'cpu', '--epochs', '2', '--batch-size', '2',
        '--num-negatives', '4', '--eval-events', '6', '--eval-batch-size', '2',
        '--hidden-dim', '32', '--num-residual-blocks', '1',
        '--norm-stats-events', '6', '--log-every', '0',
    ] + extra


def test_fusion_trainer_end_to_end(artifacts, tmp_path):
    from train_triplet_reranker import main as train_main
    run_dir = str(tmp_path / 'run')
    train_main(_base_args(artifacts, run_dir, ['--fusion']))

    history = json.load(open(os.path.join(run_dir, 'metrics_history.json')))
    assert len(history) == 2
    assert 'T@10' in history[-1]
    assert 'p99/T@10' in history[-1]

    final = json.load(open(os.path.join(run_dir, 'final_eval.json')))
    assert 0.0 <= final['T@10'] <= 1.0
    assert final['n_eval_events'] == 6

    import torch
    best = torch.load(os.path.join(run_dir, 'checkpoints', 'best_model.pt'),
                      map_location='cpu', weights_only=False)
    assert best['operating_point']['gate'] == 'p95'
    assert best['operating_point']['score_column'] == 'filter_score'
    from weaver.nn.model.TripletReranker import TripletReranker
    model = TripletReranker(
        input_mode='flat', feature_names=best['feature_names'],
        hidden_dim=32, num_residual_blocks=1, loss_mode='full',
        trunk_norm='layer', fusion=True)
    model.load_state_dict(best['triplet_reranker_state_dict'])


def test_static_fit_and_aux_and_tail_variants_train(artifacts, tmp_path):
    from train_triplet_reranker import main as train_main
    run_dir = str(tmp_path / 'run_fit')
    train_main(_base_args(artifacts, run_dir, [
        '--fusion', '--fit-mode', 'static', '--aux-fromb-weight', '0.2',
        '--tail-weighting']))
    history = json.load(open(os.path.join(run_dir, 'metrics_history.json')))
    assert len(history) == 2


def test_layer_fit_mode_trains(artifacts, tmp_path):
    from train_triplet_reranker import main as train_main
    run_dir = str(tmp_path / 'run_layer')
    train_main(_base_args(artifacts, run_dir, ['--fusion', '--fit-mode', 'layer']))
    final = json.load(open(os.path.join(run_dir, 'final_eval.json')))
    assert 0.0 <= final['T@10'] <= 1.0
