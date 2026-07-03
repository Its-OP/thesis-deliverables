from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'python'))

from eval_triplet_rank_baselines import K_VALUES, deduped_gt_rank
from utils.checkpointing import CheckpointManager
from utils.experiment import build_experiment_directory
from utils.training import build_warmup_scheduler
from utils.triplet_rank_data import (
    TripletRankDataset,
    collate_triplet_rank,
    fit_norm_stats,
    load_norm_stats,
    save_norm_stats,
)
from utils.triplet_split import load_split
from weaver.nn.model.TripletReranker import TripletReranker

logger = logging.getLogger('train_triplet_reranker')

TRIPLET_RANK_DIR = os.path.join(os.path.dirname(__file__), 'data', 'low-pt', 'eval', 'triplet_rank')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Stage-4 triplet reranker trainer '
                                                 '(offline candidate artifacts, InfoNCE top-1).')
    parser.add_argument('--candidates', default=os.path.join(TRIPLET_RANK_DIR, 'candidates_val.parquet'))
    parser.add_argument('--tracks', default=os.path.join(TRIPLET_RANK_DIR, 'tracks_val.parquet'))
    parser.add_argument('--norm-stats', default=os.path.join(TRIPLET_RANK_DIR, 'norm_stats.json'))
    parser.add_argument('--norm-stats-events', type=int, default=2000)
    parser.add_argument('--split-json', default=None,
                        help='train on the train side, evaluate on the test side; '
                             'omit to train on all trainable events and evaluate on all')
    parser.add_argument('--score-column', default='gbdt6_score')
    parser.add_argument('--tau', type=float, default=0.003824,
                        help='operating-point threshold (default: d6@0.99)')
    parser.add_argument('--input-mode', choices=('flat', 'hierarchical'), default='flat')
    parser.add_argument('--num-negatives', type=int, default=50)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-residual-blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.10)
    parser.add_argument('--projector-dim', type=int, default=32)
    parser.add_argument('--warm-start-projector', default=None,
                        help='couple reranker checkpoint to initialize track_projector from')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--cosine-power', type=float, default=2.0)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--experiments-dir', default=os.path.join(os.path.dirname(__file__), 'experiments'))
    parser.add_argument('--run-name', default='triplet_reranker')
    return parser


def _norm_stats(args, train_events) -> dict:
    if os.path.exists(args.norm_stats):
        logger.info(f'loading norm stats from {args.norm_stats}')
        return load_norm_stats(args.norm_stats)
    logger.info('fitting norm stats on the train side')
    stats = fit_norm_stats(args.candidates, args.tracks, n_events=args.norm_stats_events,
                           seed=args.seed, events=train_events)
    save_norm_stats(stats, args.norm_stats)
    logger.info(f'wrote {args.norm_stats}')
    return stats


def _warm_start_projector(model: TripletReranker, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['couple_reranker_state_dict']
    projector_state = {key[len('couple_projector.'):]: value
                       for key, value in state.items() if key.startswith('couple_projector.')}
    model.track_projector.load_state_dict(projector_state)
    logger.info(f'warm-started track_projector from {checkpoint_path}')


@torch.no_grad()
def evaluate(model, dataset, event_indices, device) -> dict:
    model.eval()
    hits = {k: 0 for k in K_VALUES}
    gt_ranks = []
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
    n_events = len(event_indices)
    ranks = np.asarray(gt_ranks) if gt_ranks else np.asarray([0])
    metrics = {f'T@{k}': hits[k] / n_events for k in K_VALUES}
    metrics['gt_rank_median'] = float(np.median(ranks))
    metrics['gt_rank_p90'] = float(np.percentile(ranks, 90))
    metrics['n_gt_surviving'] = len(gt_ranks)
    return metrics


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    train_events = None
    if args.split_json:
        train_events = load_split(args.split_json, 'train')
    norm_stats = _norm_stats(args, train_events)

    train_dataset = TripletRankDataset(
        args.candidates, args.tracks, tau=args.tau, score_column=args.score_column,
        num_negatives=args.num_negatives, mode='train', norm_stats=norm_stats, seed=args.seed)
    eval_dataset = TripletRankDataset(
        args.candidates, args.tracks, tau=args.tau, score_column=args.score_column,
        mode='eval', norm_stats=norm_stats, seed=args.seed)

    n_rows = train_dataset.table.num_rows
    if args.split_json:
        train_side = np.asarray(load_split(args.split_json, 'train'))
        eval_side = np.asarray(load_split(args.split_json, 'test'))
        train_side, eval_side = train_side[train_side < n_rows], eval_side[eval_side < n_rows]
        trainable = np.intersect1d(train_dataset.trainable_indices, train_side)
    else:
        trainable = train_dataset.trainable_indices
        eval_side = np.arange(n_rows)
    logger.info(f'{len(trainable)} trainable events, {len(eval_side)} eval events')

    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        sampler=SubsetRandomSampler([int(x) for x in trainable]),
                        collate_fn=collate_triplet_rank, num_workers=args.num_workers,
                        drop_last=True)
    steps_per_epoch = max(1, len(trainable) // args.batch_size)

    model = TripletReranker(
        input_mode=args.input_mode, hidden_dim=args.hidden_dim,
        num_residual_blocks=args.num_residual_blocks, dropout=args.dropout,
        ranking_num_samples=args.num_negatives, label_smoothing=args.label_smoothing,
        projector_dim=args.projector_dim,
    ).to(device)
    if args.warm_start_projector:
        _warm_start_projector(model, args.warm_start_projector)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_scheduler(optimizer, args, steps_per_epoch, logger)

    experiment_dir, checkpoints_dir, _ = build_experiment_directory(
        args.experiments_dir, args.run_name, None)
    manager = CheckpointManager(checkpoints_dir, keep_best_k=3,
                                criterion_mode='max', criterion_name='T@10')
    with open(os.path.join(experiment_dir, 'args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    history = []
    best_criterion = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            features = batch['features'].to(device)
            pos_mask = batch['pos_mask'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            out = model.compute_loss(features, pos_mask, valid_mask)
            optimizer.zero_grad(set_to_none=True)
            out['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step_batch()
            losses.append(float(out['total_loss'].detach()))

        metrics = evaluate(model, eval_dataset, eval_side, device)
        metrics['train_loss'] = float(np.mean(losses)) if losses else float('nan')
        metrics['epoch'] = epoch
        metrics['lr'] = scheduler.get_last_lr()[0]
        history.append(metrics)
        logger.info(f"epoch {epoch}: loss {metrics['train_loss']:.4f} "
                    f"T@10 {metrics['T@10']:.4f} median rank {metrics['gt_rank_median']}")
        scheduler.step_epoch(metrics['train_loss'])

        is_best = metrics['T@10'] > best_criterion
        best_criterion = max(best_criterion, metrics['T@10'])
        manager.save_checkpoint(
            {'triplet_reranker_state_dict': model.state_dict(),
             'args': vars(args), 'val_metrics': metrics, 'epoch': epoch},
            epoch, metrics['T@10'], is_best)
        with open(os.path.join(experiment_dir, 'metrics_history.json'), 'w') as fh:
            json.dump(history, fh, indent=2)

    best = max(history, key=lambda m: m['T@10'])
    logger.info(f"best epoch {best['epoch']}: " +
                ' '.join(f"T@{k} {best[f'T@{k}']:.4f}" for k in K_VALUES))


if __name__ == '__main__':
    main()
