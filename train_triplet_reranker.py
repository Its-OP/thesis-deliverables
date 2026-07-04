from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'python'))

from eval_triplet_rank_baselines import K_VALUES, OPERATING_POINTS, deduped_gt_rank
from utils.checkpointing import CheckpointManager
from utils.experiment import build_experiment_directory
from utils.training import build_warmup_scheduler
from utils.triplet_rank_data import (
    TripletRankDataset,
    collate_triplet_rank,
    collate_triplet_rank_eval,
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
    parser.add_argument('--eval-candidates', default=None,
                        help='separate eval-side candidates (train-on-train / eval-on-val); '
                             'mutually exclusive with --split-json')
    parser.add_argument('--eval-tracks', default=None)
    parser.add_argument('--eval-events', type=int, default=20000,
                        help='fixed-seed eval subsample per epoch; the final eval always '
                             'runs on the full eval side')
    parser.add_argument('--eval-batch-size', type=int, default=1,
                        help='events per eval forward; >1 changes BatchNorm batch '
                             'statistics and lets padding leak into them — keep 1 '
                             'for exact per-event scoring')
    parser.add_argument('--eval-every', type=int, default=1,
                        help='run the per-epoch eval every Nth epoch (the last epoch '
                             'and the final full eval always run)')
    parser.add_argument('--norm-stats', default=os.path.join(TRIPLET_RANK_DIR, 'norm_stats_train.json'))
    parser.add_argument('--norm-stats-events', type=int, default=2000)
    parser.add_argument('--split-json', default=None,
                        help='train on the train side, evaluate on the test side; '
                             'omit to train on all trainable events and evaluate on all')
    parser.add_argument('--operating-point', default='d6@0.99', choices=sorted(OPERATING_POINTS),
                        help='survivor mask tau/score column (both sides)')
    parser.add_argument('--score-column', default=None,
                        help='override the operating-point score column')
    parser.add_argument('--tau', type=float, default=None,
                        help='override the operating-point threshold')
    parser.add_argument('--input-mode', choices=('flat', 'hierarchical'), default='flat')
    parser.add_argument('--extra-features', choices=('none', 'gbdt', 'all', 'auto'), default='auto',
                        help='inputs beyond the 89 geometry features (gbdt scores, cascade scores)')
    parser.add_argument('--loss-mode', choices=('sampled', 'full'), default='sampled')
    parser.add_argument('--temperature', type=float, default=1.0)
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
    parser.add_argument('--log-every', type=int, default=50,
                        help='in-epoch progress log cadence in batches (0 disables)')
    parser.add_argument('--resume', default=None,
                        help='checkpoint to continue from (model + optimizer + epoch)')
    parser.add_argument('--experiments-dir', default=os.path.join(os.path.dirname(__file__), 'experiments'))
    parser.add_argument('--experiment-dir', default=None,
                        help='exact run directory (overrides the timestamped default)')
    parser.add_argument('--run-name', default='triplet_reranker')
    return parser


def _norm_stats(args, feature_names, train_events) -> dict:
    if os.path.exists(args.norm_stats):
        logger.info(f'loading norm stats from {args.norm_stats}')
        stats = load_norm_stats(args.norm_stats)
        missing = [name for name in feature_names if name not in stats]
        if missing:
            raise SystemExit(f'{args.norm_stats} lacks {len(missing)} feature keys '
                             f'(e.g. {missing[:3]}); delete it to refit')
        return stats
    logger.info('fitting norm stats on the train side')
    stats = fit_norm_stats(args.candidates, args.tracks, feature_names=feature_names,
                           n_events=args.norm_stats_events, seed=args.seed,
                           events=train_events)
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
def evaluate(model, dataset, event_indices, device, *, batch_size: int = 1,
             num_workers: int = 0) -> dict:
    """Feature building parallelizes across `num_workers`; scoring stays exact at
    batch_size=1 (NanSafeBatchNorm1d uses batch statistics even in eval mode, so
    batching events together or padding would change per-candidate scores)."""
    model.eval()
    subset = Subset(dataset, [int(r) for r in event_indices])
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate_triplet_rank_eval)
    hits = {k: 0 for k in K_VALUES}
    gt_ranks = []
    for batch in loader:
        counts = batch['counts']
        if int(counts.max()) == 0:
            continue
        scores = model(batch['features'].to(device))
        scores = scores.masked_fill(~batch['valid_mask'].to(device), float('-inf')).cpu()
        for b in range(scores.shape[0]):
            n = int(counts[b])
            if n == 0:
                continue
            order = torch.argsort(scores[b, :n], descending=True).numpy()
            rank = deduped_gt_rank(batch['keys'][b].numpy()[order],
                                   batch['pos_mask'][b, :n].numpy()[order])
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
    metrics['n_eval_events'] = n_events
    return metrics


def _checkpoint_payload(model, optimizer, args, metrics, epoch, best_criterion,
                        feature_names, norm_stats, score_column, tau) -> dict:
    return {
        'triplet_reranker_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'val_metrics': metrics,
        'epoch': epoch,
        'best_criterion': best_criterion,
        'feature_names': feature_names,
        'norm_stats': norm_stats,
        'operating_point': {'score_column': score_column, 'tau': tau},
    }


def _flush_history(experiment_dir: str, history: list) -> None:
    with open(os.path.join(experiment_dir, 'metrics_history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    if args.eval_candidates and args.split_json:
        raise SystemExit('--eval-candidates and --split-json are mutually exclusive')
    if bool(args.eval_candidates) != bool(args.eval_tracks):
        raise SystemExit('--eval-candidates and --eval-tracks must be given together')
    if args.eval_every < 1:
        raise SystemExit('--eval-every must be >= 1')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    score_column, tau = OPERATING_POINTS[args.operating_point]
    if args.score_column is not None:
        score_column = args.score_column
    if args.tau is not None:
        tau = args.tau
    logger.info(f'operating point {args.operating_point}: {score_column} >= {tau}')

    train_events = None
    if args.split_json:
        train_events = load_split(args.split_json, 'train')

    train_dataset = TripletRankDataset(
        args.candidates, args.tracks, tau=tau, score_column=score_column,
        num_negatives=args.num_negatives, mode='train', seed=args.seed,
        extra_features=args.extra_features)
    feature_names = train_dataset.feature_names
    norm_stats = _norm_stats(args, feature_names, train_events)
    train_dataset.norm_stats = norm_stats

    if args.eval_candidates:
        eval_dataset = TripletRankDataset(
            args.eval_candidates, args.eval_tracks, tau=tau, score_column=score_column,
            mode='eval', norm_stats=norm_stats, seed=args.seed,
            extra_features=args.extra_features)
        if eval_dataset.feature_names != feature_names:
            raise SystemExit('eval artifact resolves different feature names than the '
                             'train artifact (extra columns mismatch)')
    else:
        eval_dataset = TripletRankDataset(
            args.candidates, args.tracks, tau=tau, score_column=score_column,
            mode='eval', norm_stats=norm_stats, seed=args.seed,
            extra_features=args.extra_features)

    n_rows = train_dataset.table.num_rows
    if args.split_json:
        train_side = np.asarray(load_split(args.split_json, 'train'))
        full_eval = np.asarray(load_split(args.split_json, 'test'))
        train_side, full_eval = train_side[train_side < n_rows], full_eval[full_eval < n_rows]
        trainable = np.intersect1d(train_dataset.trainable_indices, train_side)
    else:
        trainable = train_dataset.trainable_indices
        full_eval = np.arange(eval_dataset.table.num_rows)
    if args.eval_events and args.eval_events < len(full_eval):
        # Fixed-seed subsample: the SAME events every epoch, so across-epoch T@K
        # comparisons (checkpoint selection) are paired.
        eval_side = np.random.default_rng(args.seed).choice(
            full_eval, args.eval_events, replace=False)
    else:
        eval_side = full_eval
    logger.info(f'{len(trainable)} trainable events, {len(eval_side)} eval events '
                f'per epoch, {len(full_eval)} in the final eval')

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        sampler=SubsetRandomSampler([int(x) for x in trainable]),
                        collate_fn=collate_triplet_rank, num_workers=args.num_workers,
                        drop_last=True, **loader_kwargs)
    steps_per_epoch = max(1, len(trainable) // args.batch_size)

    model = TripletReranker(
        input_mode=args.input_mode, hidden_dim=args.hidden_dim,
        num_residual_blocks=args.num_residual_blocks, dropout=args.dropout,
        ranking_num_samples=args.num_negatives, ranking_temperature=args.temperature,
        label_smoothing=args.label_smoothing, projector_dim=args.projector_dim,
        feature_names=feature_names, loss_mode=args.loss_mode,
    ).to(device)
    if args.warm_start_projector:
        _warm_start_projector(model, args.warm_start_projector)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_scheduler(optimizer, args, steps_per_epoch, logger)

    experiment_dir, checkpoints_dir, _ = build_experiment_directory(
        args.experiments_dir, args.run_name, args.experiment_dir)
    manager = CheckpointManager(checkpoints_dir, keep_best_k=3,
                                criterion_mode='max', criterion_name='T@10')
    with open(os.path.join(experiment_dir, 'args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    history = []
    start_epoch = 0
    best_criterion = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['triplet_reranker_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_criterion = checkpoint.get('best_criterion',
                                        checkpoint['val_metrics'].get('T@10', -1.0))
        # The cosine scheduler carries no state dict; replay its steps.
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step_batch()
        for _ in range(start_epoch):
            scheduler.step_epoch(float('nan'))
        history_path = os.path.join(experiment_dir, 'metrics_history.json')
        if os.path.exists(history_path):
            with open(history_path) as fh:
                history = [entry for entry in json.load(fh) if entry['epoch'] < start_epoch]
        logger.info(f'resumed from {args.resume}: continuing at epoch {start_epoch}, '
                    f'best T@10 {best_criterion:.4f}')

    _flush_history(experiment_dir, history)
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            losses = []
            epoch_start = time.time()
            for step, batch in enumerate(loader):
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
                if args.log_every and step % args.log_every == 0:
                    elapsed = max(time.time() - epoch_start, 1e-9)
                    rate = (step + 1) / elapsed
                    eta_min = (steps_per_epoch - step - 1) / rate / 60
                    logger.info(f'epoch {epoch} | batch {step}/{steps_per_epoch} '
                                f'| loss {np.mean(losses):.4f} '
                                f'| lr {scheduler.get_last_lr()[0]:.2e} '
                                f'| {rate:.2f} batch/s | ETA {eta_min:.1f} min')

            train_loss = float(np.mean(losses)) if losses else float('nan')
            current_lr = scheduler.get_last_lr()[0]
            run_eval = (epoch % args.eval_every == args.eval_every - 1
                        or epoch == args.epochs - 1)
            if run_eval:
                metrics = evaluate(model, eval_dataset, eval_side, device,
                                   batch_size=args.eval_batch_size,
                                   num_workers=args.num_workers)
                metrics['train_loss'] = train_loss
                metrics['epoch'] = epoch
                metrics['lr'] = current_lr
                history.append(metrics)
                logger.info(f"epoch {epoch}: loss {train_loss:.4f} "
                            f"T@10 {metrics['T@10']:.4f} median rank {metrics['gt_rank_median']}")
                is_best = metrics['T@10'] > best_criterion
                best_criterion = max(best_criterion, metrics['T@10'])
                manager.save_checkpoint(
                    _checkpoint_payload(model, optimizer, args, metrics, epoch, best_criterion,
                                        feature_names, norm_stats, score_column, tau),
                    epoch, metrics['T@10'], is_best)
            else:
                history.append({'train_loss': train_loss, 'epoch': epoch, 'lr': current_lr})
                logger.info(f'epoch {epoch}: loss {train_loss:.4f} (eval skipped)')
            scheduler.step_epoch(train_loss)
            _flush_history(experiment_dir, history)
    except BaseException:
        logger.error(f'training crashed at epoch {epoch}:\n{traceback.format_exc()}')
        crash_path = os.path.join(checkpoints_dir, f'crash_epoch{epoch}.pt')
        torch.save(_checkpoint_payload(model, optimizer, args, {}, epoch, best_criterion,
                                       feature_names, norm_stats, score_column, tau),
                   crash_path)
        _flush_history(experiment_dir, history)
        logger.error(f'saved emergency checkpoint {crash_path}')
        raise

    evaluated = [entry for entry in history if 'T@10' in entry]
    if evaluated:
        best = max(evaluated, key=lambda m: m['T@10'])
        logger.info(f"best epoch {best['epoch']}: " +
                    ' '.join(f"T@{k} {best[f'T@{k}']:.4f}" for k in K_VALUES))

    best_path = os.path.join(checkpoints_dir, 'best_model.pt')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['triplet_reranker_state_dict'])
        model.to(device)
        logger.info(f'final full eval on {len(full_eval)} events '
                    f'(best epoch {checkpoint["epoch"]})')
        final = evaluate(model, eval_dataset, full_eval, device,
                         batch_size=args.eval_batch_size,
                         num_workers=args.num_workers)
        final['best_epoch'] = checkpoint['epoch']
        with open(os.path.join(experiment_dir, 'final_eval.json'), 'w') as fh:
            json.dump(final, fh, indent=2)
        logger.info('final eval: ' + ' '.join(f"T@{k} {final[f'T@{k}']:.4f}" for k in K_VALUES))


if __name__ == '__main__':
    main()
