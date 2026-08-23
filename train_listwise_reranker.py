from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'python'))

from eval_triplet_rank_baselines import (
    K_VALUES,
    OPERATING_POINTS,
    deduped_gt_rank,
    load_operating_points,
)
from utils.checkpointing import CheckpointManager
from utils.experiment import build_experiment_directory
from utils.training import build_warmup_scheduler
from utils.triplet_rank_data import (
    TripletRankDataset,
    collate_listwise_rank,
    fit_norm_stats,
    load_norm_stats,
    save_norm_stats,
)
from weaver.nn.model.ListwiseTripletReranker import (
    ListwiseTripletReranker,
    within_couple_contrast,
)

logger = logging.getLogger('train_listwise_reranker')

TRIPLET_RANK_DIR = os.path.join(os.path.dirname(__file__), 'data',
                                'triplet_rank_v3')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'low-pt')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates',
                        default=os.path.join(TRIPLET_RANK_DIR,
                                             'candidates_train.parquet'))
    parser.add_argument('--src-glob',
                        default=os.path.join(DATA_DIR, 'train', '*.parquet'))
    parser.add_argument('--eval-candidates',
                        default=os.path.join(TRIPLET_RANK_DIR,
                                             'candidates_eval.parquet'))
    parser.add_argument('--eval-src-glob',
                        default=os.path.join(DATA_DIR, 'eval', '*.parquet'))
    parser.add_argument('--operating-points',
                        default=os.path.join(TRIPLET_RANK_DIR,
                                             'operating_points.json'))
    parser.add_argument('--gate', default='p99')
    parser.add_argument('--eval-gates', default='p99,p95')
    parser.add_argument('--norm-stats',
                        default=os.path.join(TRIPLET_RANK_DIR,
                                             'norm_stats_listwise.json'))
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.10)
    parser.add_argument('--contrast-weight', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--cosine-power', type=float, default=2.0)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--eval-events', type=int, default=20000)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--max-list', type=int, default=3072)
    parser.add_argument('--max-train-events', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--experiments-dir',
                        default=os.path.join(os.path.dirname(__file__),
                                             'experiments'))
    parser.add_argument('--experiment-dir', default=None)
    parser.add_argument('--run-name', default='listwise_reranker')
    return parser


def listwise_loss(scores: torch.Tensor, pos_mask: torch.Tensor,
                  valid_mask: torch.Tensor, couple_ids: torch.Tensor,
                  label_smoothing: float,
                  contrast_weight: float) -> torch.Tensor:
    """scores, pos_mask, valid_mask, couple_ids: (B, N). Scalar loss:
    smoothed listwise CE over valid slots + within-couple contrast."""
    masked = scores.masked_fill(~valid_mask, float('-inf'))
    log_probabilities = torch.log_softmax(masked, dim=1)
    events = []
    for b in range(scores.shape[0]):
        positives = pos_mask[b] & valid_mask[b]
        if not positives.any():
            continue
        n_valid = int(valid_mask[b].sum())
        target = torch.zeros_like(scores[b])
        target[positives] = (1.0 - label_smoothing) / int(positives.sum())
        target[valid_mask[b]] += label_smoothing / n_valid
        events.append(-(target[valid_mask[b]]
                        * log_probabilities[b][valid_mask[b]]).sum())
    if not events:
        return scores.new_zeros(())
    cross_entropy = torch.stack(events).mean()
    if contrast_weight <= 0.0:
        return cross_entropy
    contrast = within_couple_contrast(scores, couple_ids, pos_mask, valid_mask)
    return cross_entropy + contrast_weight * contrast


def _forward(model, batch, device) -> torch.Tensor:
    return model(batch['features'].to(device),
                 keys=batch['keys'].to(device),
                 valid_mask=batch['valid_mask'].to(device),
                 filter_logit=batch['filter_logit'].to(device))


@torch.no_grad()
def evaluate(model, dataset, event_indices, device, *, gates: dict[str, float],
             batch_size: int, num_workers: int = 0,
             score_override=None) -> dict:
    model.eval()
    loader = DataLoader(Subset(dataset, [int(r) for r in event_indices]),
                        batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate_listwise_rank)
    logit_taus = {gate: (math.log(tau / (1.0 - tau))
                         if 0.0 < tau < 1.0 else -float('inf'))
                  for gate, tau in gates.items()}
    hits = {gate: {k: 0 for k in K_VALUES} for gate in gates}
    gt_ranks = {gate: [] for gate in gates}
    for batch in loader:
        if score_override is not None:
            scores = score_override(batch)
        else:
            scores = _forward(model, batch, device).cpu()
        scores = scores.masked_fill(~batch['valid_mask'], float('-inf'))
        counts = batch['counts']
        for b in range(scores.shape[0]):
            n = int(counts[b])
            if n == 0:
                continue
            event_scores = scores[b, :n]
            logits = batch['filter_logit'][b, :n]
            keys = batch['keys'][b, :n].numpy()
            pos = batch['pos_mask'][b, :n].numpy()
            for gate, logit_tau in logit_taus.items():
                surviving = (logits >= logit_tau).numpy()
                if not pos[surviving].any():
                    continue
                order = torch.argsort(event_scores[surviving],
                                      descending=True).numpy()
                rank = deduped_gt_rank(keys[surviving][order],
                                       pos[surviving][order])
                if rank is None:
                    continue
                gt_ranks[gate].append(rank)
                for k in K_VALUES:
                    if rank <= k:
                        hits[gate][k] += 1
    n_events = len(event_indices)
    metrics = {}
    primary = next(iter(gates))
    for gate in gates:
        ranks = np.asarray(gt_ranks[gate]) if gt_ranks[gate] else np.asarray([0])
        prefix = '' if gate == primary else f'{gate}/'
        for k in K_VALUES:
            metrics[f'{prefix}T@{k}'] = hits[gate][k] / n_events
        metrics[f'{prefix}gt_rank_median'] = float(np.median(ranks))
    metrics['n_eval_events'] = n_events
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    points = dict(OPERATING_POINTS)
    if os.path.exists(args.operating_points):
        points = load_operating_points(args.operating_points)
    train_tau = points[args.gate][1]
    eval_gates = {name.strip(): points[name.strip()][1]
                  for name in args.eval_gates.split(',')
                  if name.strip() in points}
    eval_tau = min(eval_gates.values())
    logger.info(f'gate {args.gate}: tau {train_tau}; eval gates {eval_gates}')

    dataset_kwargs = dict(extra_features='auto', context_features=True,
                          vertex_fit='static', max_serving_rows=args.max_list)
    train_dataset = TripletRankDataset(
        args.candidates, args.src_glob, tau=train_tau, mode='eval',
        seed=args.seed, **dataset_kwargs)
    feature_names = train_dataset.feature_names
    if os.path.exists(args.norm_stats):
        norm_stats = load_norm_stats(args.norm_stats)
    else:
        logger.info('fitting norm stats on the train side')
        norm_stats = fit_norm_stats(args.candidates, args.src_glob,
                                    feature_names=feature_names,
                                    seed=args.seed, tau=train_tau,
                                    context_features=True,
                                    vertex_fit='static')
        os.makedirs(os.path.dirname(os.path.abspath(args.norm_stats)),
                    exist_ok=True)
        save_norm_stats(norm_stats, args.norm_stats)
        logger.info(f'wrote {args.norm_stats}')
    train_dataset.norm_stats = norm_stats

    eval_dataset = TripletRankDataset(
        args.eval_candidates, args.eval_src_glob, tau=eval_tau, mode='eval',
        norm_stats=norm_stats, seed=args.seed, **dataset_kwargs)
    if eval_dataset.feature_names != feature_names:
        raise SystemExit('eval artifact resolves different feature names')

    trainable = train_dataset.trainable_indices
    if args.max_train_events and args.max_train_events < len(trainable):
        trainable = trainable[:args.max_train_events]
    full_eval = np.arange(eval_dataset.table.num_rows)
    if args.eval_events and args.eval_events < len(full_eval):
        eval_side = np.random.default_rng(args.seed).choice(
            full_eval, args.eval_events, replace=False)
    else:
        eval_side = full_eval
    logger.info(f'{len(trainable)} trainable events, {len(eval_side)} eval '
                f'events per epoch, {len(full_eval)} in the final eval')

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        sampler=SubsetRandomSampler([int(x) for x in trainable]),
                        collate_fn=collate_listwise_rank,
                        num_workers=args.num_workers, drop_last=True,
                        **loader_kwargs)
    steps_per_epoch = max(1, len(trainable) // args.batch_size)

    model = ListwiseTripletReranker(
        feature_dim=len(feature_names), hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout).to(device)
    logger.info(f'{sum(p.numel() for p in model.parameters())} parameters, '
                f'{len(feature_names)} features')

    sample = eval_side[:min(len(eval_side), 512)]
    model_metrics = evaluate(model, eval_dataset, sample, device,
                             gates=eval_gates, batch_size=args.eval_batch_size)
    filter_metrics = evaluate(model, eval_dataset, sample, device,
                              gates=eval_gates, batch_size=args.eval_batch_size,
                              score_override=lambda b: b['filter_logit'])
    for key in model_metrics:
        if key.split('/')[-1].startswith('T@') \
                and abs(model_metrics[key] - filter_metrics[key]) > 1e-9:
            raise SystemExit(f'epoch-0 fusion mismatch on {key}: '
                             f'{model_metrics[key]:.6f} vs '
                             f'{filter_metrics[key]:.6f}')
    logger.info(f'epoch-0 fusion assert passed on {len(sample)} events '
                f"(T@10 {model_metrics['T@10']:.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_warmup_scheduler(optimizer, args, steps_per_epoch, logger)
    experiment_dir, checkpoints_dir, _ = build_experiment_directory(
        args.experiments_dir, args.run_name, args.experiment_dir)
    manager = CheckpointManager(checkpoints_dir, keep_best_k=3,
                                criterion_mode='max', criterion_name='T@10')
    best = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch_index, batch in enumerate(loader):
            scores = _forward(model, batch, device)
            loss = listwise_loss(scores, batch['pos_mask'].to(device),
                                 batch['valid_mask'].to(device),
                                 batch['couple_ids'].to(device),
                                 args.label_smoothing, args.contrast_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step_batch()
            step += 1
            running += float(loss)
            if (batch_index + 1) % args.log_every == 0:
                logger.info(f'epoch {epoch} | batch {batch_index + 1}/'
                            f'{steps_per_epoch} | loss '
                            f'{running / (batch_index + 1):.4f} | lr '
                            f'{scheduler.get_last_lr()[0]:.2e}')
        metrics = evaluate(model, eval_dataset, eval_side, device,
                           gates=eval_gates, batch_size=args.eval_batch_size,
                           num_workers=args.num_workers)
        logger.info(f"epoch {epoch}: loss {running / steps_per_epoch:.4f} "
                    f"T@10 {metrics['T@10']:.4f} median rank "
                    f"{metrics['gt_rank_median']}")
        is_best = metrics['T@10'] > best
        best = max(best, metrics['T@10'])
        payload = {
            'listwise_reranker_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args), 'val_metrics': metrics, 'epoch': epoch,
            'feature_names': feature_names, 'norm_stats': norm_stats,
        }
        manager.save_checkpoint(payload, epoch, metrics['T@10'], is_best)
        if is_best:
            logger.info(f'New best model (T@10={best:.5f})')

    checkpoint = torch.load(os.path.join(checkpoints_dir, 'best_model.pt'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['listwise_reranker_state_dict'])
    final = evaluate(model, eval_dataset, full_eval, device, gates=eval_gates,
                     batch_size=args.eval_batch_size,
                     num_workers=args.num_workers)
    with open(os.path.join(experiment_dir, 'final_eval.json'), 'w') as handle:
        json.dump(final, handle, indent=2)
    logger.info(f"final full eval: T@10 {final['T@10']:.4f} over "
                f"{len(full_eval)} events")


if __name__ == '__main__':
    main()
