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
    # Single-file deploy: the checkpoint carries everything needed to rebuild.
    assert checkpoint['feature_names'][:89] and 'norm_stats' in checkpoint
    assert checkpoint['operating_point'] == {'score_column': 'gbdt6_score', 'tau': 0.0}
    model = TripletReranker(input_mode='flat', feature_names=checkpoint['feature_names'])
    model.load_state_dict(checkpoint['triplet_reranker_state_dict'])


def test_trainer_two_artifact_full_loss_and_final_eval(real_artifact, tmp_path):
    from train_triplet_reranker import main as train_main

    run_dir = str(tmp_path / 'two_artifact_run')
    train_main([
        '--candidates', os.path.join(real_artifact, 'candidates_val.parquet'),
        '--tracks', os.path.join(real_artifact, 'tracks_val.parquet'),
        '--eval-candidates', os.path.join(real_artifact, 'candidates_val.parquet'),
        '--eval-tracks', os.path.join(real_artifact, 'tracks_val.parquet'),
        '--norm-stats', os.path.join(real_artifact, 'norm_stats.json'),
        '--experiment-dir', run_dir,
        '--input-mode', 'flat',
        '--loss-mode', 'full',
        '--tau', '0.0',
        '--eval-events', '10',
        '--epochs', '2',
        '--batch-size', '4',
        '--num-negatives', '10',
        '--device', 'cpu',
        '--norm-stats-events', '20',
    ])
    # Explicit --experiment-dir is honored verbatim (the .sh owns run-dir naming).
    with open(os.path.join(run_dir, 'metrics_history.json')) as fh:
        history = json.load(fh)
    assert len(history) == 2
    for entry in history:
        assert entry['n_eval_events'] == 10
        assert 0.0 <= entry['T@10'] <= 1.0
    with open(os.path.join(run_dir, 'final_eval.json')) as fh:
        final = json.load(fh)
    assert final['n_eval_events'] == 40
    assert 0.0 <= final['T@10'] <= 1.0


def test_eval_every_skips_intermediate_epochs(real_artifact, tmp_path):
    from train_triplet_reranker import main as train_main

    run_dir = str(tmp_path / 'eval_every_run')
    train_main([
        '--candidates', os.path.join(real_artifact, 'candidates_val.parquet'),
        '--tracks', os.path.join(real_artifact, 'tracks_val.parquet'),
        '--norm-stats', os.path.join(real_artifact, 'norm_stats.json'),
        '--experiment-dir', run_dir,
        '--tau', '0.0',
        '--epochs', '4',
        '--eval-every', '2',
        '--batch-size', '4',
        '--num-negatives', '10',
        '--device', 'cpu',
        '--norm-stats-events', '20',
    ])
    with open(os.path.join(run_dir, 'metrics_history.json')) as fh:
        history = json.load(fh)
    assert [entry['epoch'] for entry in history] == [0, 1, 2, 3]
    for entry in history:
        assert 'train_loss' in entry and 'lr' in entry
        if entry['epoch'] in (1, 3):
            assert 'T@10' in entry
        else:
            assert 'T@10' not in entry
    assert os.path.exists(os.path.join(run_dir, 'checkpoints', 'best_model.pt'))
    assert os.path.exists(os.path.join(run_dir, 'final_eval.json'))


def test_evaluate_parallel_loader_matches_reference(real_artifact):
    import numpy as np
    import torch
    from eval_triplet_rank_baselines import K_VALUES, deduped_gt_rank
    from train_triplet_reranker import evaluate
    from utils.triplet_rank_data import TripletRankDataset
    from weaver.nn.model.TripletReranker import TripletReranker

    dataset = TripletRankDataset(
        os.path.join(real_artifact, 'candidates_val.parquet'),
        os.path.join(real_artifact, 'tracks_val.parquet'),
        tau=0.0, mode='eval', extra_features='auto')
    torch.manual_seed(0)
    model = TripletReranker(input_mode='flat', feature_names=dataset.feature_names)
    model.eval()
    device = torch.device('cpu')
    event_indices = np.arange(40)

    # Reference: the original unbatched per-event loop, inlined verbatim.
    hits = {k: 0 for k in K_VALUES}
    gt_ranks = []
    with torch.no_grad():
        for r in event_indices:
            item = dataset[int(r)]
            if item['features'].shape[0] == 0:
                continue
            scores = model(item['features'].T.unsqueeze(0).to(device)).squeeze(0).cpu()
            order = torch.argsort(scores, descending=True).numpy()
            rank = deduped_gt_rank(item['keys'].numpy()[order],
                                   item['pos_mask'].numpy()[order])
            if rank is None:
                continue
            gt_ranks.append(rank)
            for k in K_VALUES:
                if rank <= k:
                    hits[k] += 1
    reference = {f'T@{k}': hits[k] / len(event_indices) for k in K_VALUES}
    reference['gt_rank_median'] = float(np.median(gt_ranks))
    reference['n_gt_surviving'] = len(gt_ranks)

    result = evaluate(model, dataset, event_indices, device)
    for key, value in reference.items():
        assert result[key] == value, f'{key}: {result[key]} != {value}'
    assert result['n_eval_events'] == 40

    # Worker processes must not change the metrics either.
    workers = evaluate(model, dataset, event_indices, device, num_workers=2)
    for key, value in reference.items():
        assert workers[key] == value, f'workers {key}: {workers[key]} != {value}'


def test_trainer_resume_continues_epochs(real_artifact, tmp_path):
    from train_triplet_reranker import main as train_main

    run_dir = str(tmp_path / 'resume_run')
    common = [
        '--candidates', os.path.join(real_artifact, 'candidates_val.parquet'),
        '--tracks', os.path.join(real_artifact, 'tracks_val.parquet'),
        '--norm-stats', os.path.join(real_artifact, 'norm_stats.json'),
        '--experiment-dir', run_dir,
        '--input-mode', 'flat',
        '--tau', '0.0',
        '--batch-size', '4',
        '--num-negatives', '10',
        '--device', 'cpu',
        '--norm-stats-events', '20',
    ]
    train_main(common + ['--epochs', '1'])
    ckpt = os.path.join(run_dir, 'checkpoints', 'best_model.pt')
    assert os.path.exists(ckpt)
    train_main(common + ['--epochs', '2', '--resume', ckpt])
    with open(os.path.join(run_dir, 'metrics_history.json')) as fh:
        history = json.load(fh)
    # The resumed run continues at epoch 1 and keeps the prior history entry.
    assert [entry['epoch'] for entry in history] == [0, 1]
