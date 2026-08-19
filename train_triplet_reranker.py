from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
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
    collate_triplet_rank,
    collate_triplet_rank_eval,
    fit_norm_stats,
    is_track32_name,
    load_norm_stats,
    save_norm_stats,
)
from utils.triplet_split import load_split
from weaver.nn.model.TripletReranker import TripletReranker
from weaver.nn.model.VertexFit import FIT_NAMES

logger = logging.getLogger('train_triplet_reranker')

TRIPLET_RANK_DIR = os.path.join(os.path.dirname(__file__), 'data', 'triplet_rank_v2')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'low-pt')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Stage-4 triplet reranker trainer '
                                                 '(schema-v2 window artifacts).')
    parser.add_argument('--candidates', default=os.path.join(TRIPLET_RANK_DIR, 'candidates_train.parquet'))
    parser.add_argument('--src-glob', default=os.path.join(DATA_DIR, 'train', '*.parquet'))
    parser.add_argument('--eval-candidates', default=os.path.join(TRIPLET_RANK_DIR, 'candidates_eval.parquet'))
    parser.add_argument('--eval-src-glob', default=os.path.join(DATA_DIR, 'eval', '*.parquet'))
    parser.add_argument('--eval-events', type=int, default=20000,
                        help='fixed-seed eval subsample per epoch; the final eval always '
                             'runs on the full eval side')
    parser.add_argument('--eval-batch-size', type=int, default=1,
                        help='events per eval forward; >1 requires --trunk-norm layer '
                             '(batch statistics leak across events otherwise)')
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--norm-stats', default=os.path.join(TRIPLET_RANK_DIR, 'norm_stats_train.json'))
    parser.add_argument('--norm-stats-events', type=int, default=2000)
    parser.add_argument('--split-json', default=None,
                        help='train on the train side, evaluate on the test side of '
                             'the SAME artifact; omit for train-on-train / eval-on-eval')
    parser.add_argument('--operating-points', default=os.path.join(
        TRIPLET_RANK_DIR, 'operating_points.json'))
    parser.add_argument('--gate', default='p95',
                        help='training gate name from operating_points.json '
                             '(p95/p99/tierH)')
    parser.add_argument('--eval-gates', default='p95,p99',
                        help='gates reported every eval, all from one forward pass')
    parser.add_argument('--tau', type=float, default=None,
                        help='override the training-gate threshold')
    parser.add_argument('--input-mode', choices=('flat', 'hierarchical'), default='flat')
    parser.add_argument('--track-embed-dim', type=int, default=32)
    parser.add_argument('--context-features', action='store_true')
    parser.add_argument('--fit-mode', choices=('off', 'static', 'layer'), default='off',
                        help='static: 21 WLS fit columns as inputs; layer: the '
                             'differentiable fit computes them in-model')
    parser.add_argument('--fusion', action='store_true',
                        help='score = alpha * filter_logit + f(x); epoch 0 IS the '
                             'filter ordering (asserted)')
    parser.add_argument('--aux-fromb-weight', type=float, default=0.0)
    parser.add_argument('--trunk-norm', choices=('batch', 'layer'), default='layer')
    parser.add_argument('--tail-weighting', action='store_true',
                        help='append the stored tail sample with log reweighting '
                             '(unbiased full-list loss beyond the window)')
    parser.add_argument('--attention-layers', type=int, default=0)
    parser.add_argument('--attention-heads', type=int, default=8)
    parser.add_argument('--warm-start-checkpoint', default=None)
    parser.add_argument('--extra-features', choices=('none', 'filter', 'all', 'auto'),
                        default='auto')
    parser.add_argument('--loss-mode', choices=('sampled', 'full'), default='full')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num-negatives', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-residual-blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.10)
    parser.add_argument('--projector-dim', type=int, default=32)
    parser.add_argument('--warm-start-projector', default=None)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--cosine-power', type=float, default=2.0)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--experiments-dir', default=os.path.join(os.path.dirname(__file__), 'experiments'))
    parser.add_argument('--experiment-dir', default=None)
    parser.add_argument('--run-name', default='triplet_reranker')
    return parser


def _resolve_gates(args) -> tuple[dict, float]:
    points = dict(OPERATING_POINTS)
    if os.path.exists(args.operating_points):
        points = load_operating_points(args.operating_points)
    elif args.tau is None and args.gate != 'tierH':
        raise SystemExit(f'{args.operating_points} not found and no --tau override')
    if args.gate not in points and args.tau is None:
        raise SystemExit(f'gate {args.gate!r} not in {sorted(points)}')
    train_tau = args.tau if args.tau is not None else points[args.gate][1]
    eval_gates = {}
    for name in args.eval_gates.split(','):
        name = name.strip()
        if name == args.gate and args.tau is not None:
            eval_gates[name] = train_tau
        elif name in points:
            eval_gates[name] = points[name][1]
    if not eval_gates:
        eval_gates = {args.gate: train_tau}
    return eval_gates, float(train_tau)


def _norm_stats(args, feature_names, train_events, tau) -> dict:
    if os.path.exists(args.norm_stats):
        logger.info(f'loading norm stats from {args.norm_stats}')
        stats = load_norm_stats(args.norm_stats)
        missing = [name for name in feature_names
                   if name not in stats and not is_track32_name(name)]
        if missing:
            raise SystemExit(f'{args.norm_stats} lacks {len(missing)} feature keys '
                             f'(e.g. {missing[:3]}); delete it to refit')
        return stats
    logger.info('fitting norm stats on the train side')
    stats = fit_norm_stats(args.candidates, args.src_glob,
                           feature_names=feature_names,
                           n_events=args.norm_stats_events, seed=args.seed,
                           events=train_events, tau=tau,
                           context_features=args.context_features,
                           vertex_fit='static' if args.fit_mode != 'off' else 'off')
    os.makedirs(os.path.dirname(os.path.abspath(args.norm_stats)), exist_ok=True)
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


def _warm_start_checkpoint(model: TripletReranker, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['triplet_reranker_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f'warm-start checkpoint has unexpected keys: {unexpected[:5]}')
    fresh = [key for key in missing if not key.startswith(('attention_blocks.',
                                                           'attention_gates.'))]
    if fresh:
        raise SystemExit(f'warm-start checkpoint lacks trunk keys: {fresh[:5]}')
    logger.info(f'warm-started {len(state)} tensors from {checkpoint_path} '
                f'({len(missing)} attention tensors stay fresh)')


def _batch_kwargs(batch, device, *, for_loss: bool) -> dict:
    kwargs = {}
    loss_only = ('log_weights', 'from_b') if for_loss else ()
    for key in ('filter_logit',) + loss_only:
        if key in batch:
            kwargs[key] = batch[key].to(device)
    fit_keys = [key for key in batch if key.startswith('fit_')] \
        + (['primary_vertex'] if 'primary_vertex' in batch else [])
    if any(key.startswith('fit_') for key in fit_keys):
        kwargs['fit_inputs'] = {key: batch[key].to(device) for key in fit_keys}
    return kwargs


@torch.no_grad()
def evaluate(model, dataset, event_indices, device, *, gates: dict[str, float],
             batch_size: int = 1, num_workers: int = 0,
             score_override=None) -> dict:
    """One forward pass per event; every gate in `gates` (name -> tau) is a
    mask applied afterwards, so all reported operating points are paired.
    score_override(batch) replaces the model scores (e.g. the raw filter
    ordering for the epoch-0 fusion assert)."""
    model.eval()
    subset = Subset(dataset, [int(r) for r in event_indices])
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate_triplet_rank_eval)
    hits = {gate: {k: 0 for k in K_VALUES} for gate in gates}
    gt_ranks = {gate: [] for gate in gates}
    # The dataset serves the loosest gate; tighter gates mask by filter logit.
    logit_taus = {gate: (math.log(tau / (1.0 - tau))
                         if 0.0 < tau < 1.0 else -float('inf'))
                  for gate, tau in gates.items()}
    for batch in loader:
        counts = batch['counts']
        if int(counts.max()) == 0:
            continue
        valid_mask = batch['valid_mask'].to(device)
        if score_override is not None:
            scores = score_override(batch).to(device)
        else:
            scores = model(batch['features'].to(device), valid_mask=valid_mask,
                           **_batch_kwargs(batch, device, for_loss=False))
        scores = scores.masked_fill(~valid_mask, float('-inf')).cpu()
        filter_logit = batch['filter_logit']
        for b in range(scores.shape[0]):
            n = int(counts[b])
            if n == 0:
                continue
            event_scores = scores[b, :n]
            event_logits = filter_logit[b, :n]
            keys = batch['keys'][b].numpy()
            pos = batch['pos_mask'][b, :n].numpy()
            for gate, logit_tau in logit_taus.items():
                surviving = (event_logits >= logit_tau).numpy()
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
        metrics[f'{prefix}gt_rank_p90'] = float(np.percentile(ranks, 90))
        metrics[f'{prefix}n_gt_surviving'] = len(gt_ranks[gate])
    metrics['n_eval_events'] = n_events
    return metrics


def _assert_epoch0_fusion(model, dataset, eval_side, device, gates, args) -> None:
    """With the zero-initialized fusion head the model ordering must equal the
    filter ordering exactly — the wiring gate for every later delta."""
    sample = eval_side[:min(len(eval_side), 512)]
    model_metrics = evaluate(model, dataset, sample, device, gates=gates,
                             batch_size=args.eval_batch_size)
    filter_metrics = evaluate(model, dataset, sample, device, gates=gates,
                              batch_size=args.eval_batch_size,
                              score_override=lambda batch: batch['filter_logit'])
    for key in model_metrics:
        if key.split('/')[-1].startswith('T@'):
            if abs(model_metrics[key] - filter_metrics[key]) > 1e-9:
                raise SystemExit(
                    f'epoch-0 fusion mismatch on {key}: model '
                    f'{model_metrics[key]:.6f} vs filter {filter_metrics[key]:.6f} '
                    f'— the fusion wiring is broken')
    logger.info(f'epoch-0 fusion assert passed on {len(sample)} events '
                f"(T@10 {model_metrics['T@10']:.4f})")


def _checkpoint_payload(model, optimizer, args, metrics, epoch, best_criterion,
                        feature_names, norm_stats, gates, train_tau) -> dict:
    return {
        'triplet_reranker_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'val_metrics': metrics,
        'epoch': epoch,
        'best_criterion': best_criterion,
        'feature_names': feature_names,
        'norm_stats': norm_stats,
        'operating_point': {'score_column': 'filter_score', 'gate': args.gate,
                            'tau': train_tau, 'eval_gates': gates},
    }


def _flush_history(experiment_dir: str, history: list) -> None:
    with open(os.path.join(experiment_dir, 'metrics_history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Two full worker pools (train + eval) share tensors over file descriptors
    # by default and exhaust the fd limit mid-run; the file_system strategy
    # shares over named files instead.
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = build_parser().parse_args(argv)
    if args.eval_every < 1:
        raise SystemExit('--eval-every must be >= 1')
    if args.eval_batch_size > 1 and args.trunk_norm != 'layer':
        raise SystemExit('--eval-batch-size > 1 requires --trunk-norm layer')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    eval_gates, train_tau = _resolve_gates(args)
    # The eval dataset serves the LOOSEST gate; tighter gates mask at eval time.
    eval_tau = min(eval_gates.values())
    logger.info(f'training gate {args.gate}: filter_score >= {train_tau}; '
                f'eval gates {eval_gates}')

    train_events = None
    if args.split_json:
        train_events = load_split(args.split_json, 'train')

    dataset_kwargs = dict(
        extra_features=args.extra_features,
        context_features=args.context_features,
        vertex_fit=args.fit_mode,
        from_b_targets=args.aux_fromb_weight > 0.0,
        track32=args.input_mode == 'hierarchical' and args.track_embed_dim == 32)
    train_dataset = TripletRankDataset(
        args.candidates, args.src_glob, tau=train_tau,
        num_negatives=args.num_negatives, mode='train', seed=args.seed,
        tail_weighting=args.tail_weighting, **dataset_kwargs)
    feature_names = train_dataset.feature_names
    norm_stats = _norm_stats(args, feature_names, train_events, train_tau)
    train_dataset.norm_stats = norm_stats

    if args.split_json:
        eval_dataset = TripletRankDataset(
            args.candidates, args.src_glob, tau=eval_tau, mode='eval',
            norm_stats=norm_stats, seed=args.seed, **dataset_kwargs)
    else:
        eval_dataset = TripletRankDataset(
            args.eval_candidates, args.eval_src_glob, tau=eval_tau, mode='eval',
            norm_stats=norm_stats, seed=args.seed, **dataset_kwargs)
        if eval_dataset.feature_names != feature_names:
            raise SystemExit('eval artifact resolves different feature names than '
                             'the train artifact')

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

    fit_stats = None
    if args.fit_mode == 'layer':
        fit_stats = {name: norm_stats[name] for name in FIT_NAMES}
    model = TripletReranker(
        input_mode=args.input_mode, hidden_dim=args.hidden_dim,
        num_residual_blocks=args.num_residual_blocks, dropout=args.dropout,
        ranking_num_samples=args.num_negatives, ranking_temperature=args.temperature,
        label_smoothing=args.label_smoothing, projector_dim=args.projector_dim,
        feature_names=feature_names, loss_mode=args.loss_mode,
        num_attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        track_embed_dim=args.track_embed_dim, trunk_norm=args.trunk_norm,
        fusion=args.fusion, aux_from_b_weight=args.aux_fromb_weight,
        vertex_fit_layer=args.fit_mode == 'layer', fit_norm_stats=fit_stats,
    ).to(device)
    if args.warm_start_checkpoint:
        _warm_start_checkpoint(model, args.warm_start_checkpoint)
    elif args.warm_start_projector:
        _warm_start_projector(model, args.warm_start_projector)
    if args.fusion:
        _assert_epoch0_fusion(model, eval_dataset, eval_side, device,
                              eval_gates, args)

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
                out = model.compute_loss(features, pos_mask, valid_mask,
                                         **_batch_kwargs(batch, device,
                                                         for_loss=True))
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
                                   gates=eval_gates,
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
                                        feature_names, norm_stats, eval_gates, train_tau),
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
                                       feature_names, norm_stats, eval_gates, train_tau),
                   crash_path)
        _flush_history(experiment_dir, history)
        logger.error(f'saved emergency checkpoint {crash_path}')
        raise

    best_path = os.path.join(checkpoints_dir, 'best_model.pt')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['triplet_reranker_state_dict'])
        logger.info(f"final eval with the best checkpoint "
                    f"(epoch {checkpoint['epoch']}, "
                    f"T@10 {checkpoint['val_metrics'].get('T@10', float('nan')):.4f})")
    final = evaluate(model, eval_dataset, full_eval, device, gates=eval_gates,
                     batch_size=args.eval_batch_size, num_workers=args.num_workers)
    with open(os.path.join(experiment_dir, 'final_eval.json'), 'w') as fh:
        json.dump(final, fh, indent=2)
    logger.info(f"final full eval: T@10 {final['T@10']:.4f} over "
                f"{final['n_eval_events']} events")


if __name__ == '__main__':
    main()
