from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.triplet_rank_data import TripletRankDataset
from weaver.nn.model.TripletReranker import TripletReranker
try:
    from scripts.python.eval_triplet_rank_baselines import (
        K_VALUES,
        OPERATING_POINTS,
        deduped_gt_rank,
        evaluate_baselines,
    )
except ImportError:  # direct-file invocation: scripts/python is sys.path[0]
    from eval_triplet_rank_baselines import (
        K_VALUES,
        OPERATING_POINTS,
        deduped_gt_rank,
        evaluate_baselines,
    )

TRIPLET_RANK_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                                'data', 'low-pt', 'eval', 'triplet_rank')


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    trainer_args = checkpoint['args']
    model = TripletReranker(
        input_mode=trainer_args['input_mode'],
        hidden_dim=trainer_args['hidden_dim'],
        num_residual_blocks=trainer_args['num_residual_blocks'],
        dropout=trainer_args['dropout'],
        ranking_num_samples=trainer_args['num_negatives'],
        ranking_temperature=trainer_args.get('temperature', 1.0),
        label_smoothing=trainer_args['label_smoothing'],
        projector_dim=trainer_args['projector_dim'],
        feature_names=checkpoint['feature_names'],
        loss_mode=trainer_args.get('loss_mode', 'sampled'),
        num_attention_layers=trainer_args.get('attention_layers', 0),
        attention_heads=trainer_args.get('attention_heads', 8),
    )
    model.load_state_dict(checkpoint['triplet_reranker_state_dict'])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def model_t_at_k(model, dataset, event_indices, device) -> dict:
    hits = {k: 0 for k in K_VALUES}
    gt_ranks = []
    for r in tqdm(event_indices, desc='model', mininterval=10):
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
    n_events = len(event_indices)
    ranks = np.asarray(gt_ranks) if gt_ranks else np.asarray([0])
    return {
        't_at_k': {str(k): hits[k] / n_events for k in K_VALUES},
        'ceiling': len(gt_ranks) / n_events,
        'gt_rank_median': float(np.median(ranks)),
        'gt_rank_p90': float(np.percentile(ranks, 90)),
    }


def _print_table(model_result: dict, baselines: dict) -> None:
    header = '| ranking | ceiling | ' + ' | '.join(f'T@{k}' for k in K_VALUES) + ' |'
    print(header)
    print('|---' * (len(K_VALUES) + 2) + '|')
    rows = [('model', model_result['ceiling'], model_result['t_at_k'])]
    for ordering, curve in baselines['t_at_k'].items():
        rows.append((ordering, baselines['ceiling'], curve))
    for label, ceiling, curve in rows:
        cells = ' | '.join(f"{curve[str(k)]:.4f}" for k in K_VALUES)
        print(f'| {label} | {ceiling:.4f} | {cells} |')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Full-VAL T@K of a trained triplet '
                                                 'reranker vs the zero-train baselines.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates', default=os.path.join(TRIPLET_RANK_DIR, 'candidates_val.parquet'))
    parser.add_argument('--tracks', default=os.path.join(TRIPLET_RANK_DIR, 'tracks_val.parquet'))
    parser.add_argument('--out-json', default=os.path.join(os.path.dirname(__file__), '..', '..',
                                                           'reports', 'triplet_reranker_eval.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--window-artifact', default=None,
                        help='stage-A window dump matching --candidates; required '
                             'when the checkpoint was trained window-mode')
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    operating_point = checkpoint['operating_point']
    score_column, tau = operating_point['score_column'], operating_point['tau']
    print(f"checkpoint epoch {checkpoint['epoch']} | {score_column} >= {tau}")

    if checkpoint['args'].get('window_artifact') and not args.window_artifact:
        raise SystemExit('checkpoint was trained window-mode; pass --window-artifact')
    dataset = TripletRankDataset(
        args.candidates, args.tracks, tau=tau, score_column=score_column,
        mode='eval', norm_stats=checkpoint['norm_stats'], seed=0,
        extra_features=checkpoint['args'].get('extra_features', 'none'),
        weaver_track_blocks=checkpoint['args'].get('weaver_track_blocks', False),
        context_features=checkpoint['args'].get('context_features', False),
        window_artifact=args.window_artifact,
        attention_window=checkpoint['args'].get('attention_window', 512))
    if dataset.feature_names != checkpoint['feature_names']:
        raise SystemExit('candidates artifact resolves different feature names than '
                         'the checkpoint was trained with')
    n_events = dataset.table.num_rows
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    event_indices = np.arange(n_events)

    model_result = model_t_at_k(model, dataset, event_indices, device)
    # The baselines rank the same survivor lists: evaluate at the checkpoint's exact
    # (score_column, tau), which may differ from the named canon if overridden.
    OPERATING_POINTS['__checkpoint__'] = (score_column, tau)
    try:
        baselines = evaluate_baselines(args.candidates, operating_point='__checkpoint__',
                                       event_indices=event_indices, seed=0)
    finally:
        del OPERATING_POINTS['__checkpoint__']

    _print_table(model_result, baselines)
    result = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint['epoch'],
        'operating_point': operating_point,
        'n_events': int(n_events),
        'k_values': K_VALUES,
        'model': model_result,
        'baselines': baselines,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f'wrote {args.out_json}')


if __name__ == '__main__':
    main()
