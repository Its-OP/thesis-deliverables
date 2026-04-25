from __future__ import annotations

import argparse
import logging
import os

import torch

from utils.dataset_helpers import (
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)
from utils.metrics import MetricsAccumulator
from utils.optimizers import OPTIMIZER_NAMES, build_optimizer as build_optimizer_factory
from utils.training import (
    add_common_training_args,
    build_data_loaders,
    copy_auto_yaml,
    run_training,
    setup_experiment,
)

logger = logging.getLogger('train_prefilter')

torch.set_float32_matmul_precision('high')


VAL_METRICS_K_VALUES: tuple[int, ...] = (
    10, 20, 30, 50, 100, 200, 256, 300, 400, 500, 600, 800,
)
CHECKPOINT_CRITERIA: tuple[str, ...] = (
    'recall_at_200', 'recall_at_256', 'perfect_at_200', 'perfect_at_256',
)

METRIC_LABELS: dict[str, str] = {
    'train': 'Train loss (per-track ranking, mean per epoch)',
    'val': 'Validation loss (per-track ranking)',
    'lr': 'Learning rate',
    'd_prime': "Cohen's d' between GT and background score distributions (val)",
    'median_gt_rank': 'Median rank of GT pions in the per-event score order (val)',
}
for _k in VAL_METRICS_K_VALUES:
    METRIC_LABELS[f'recall_at_{_k}'] = (
        f'R@{_k}: per-event recall at top-{_k} tracks '
        f'(fraction of GT pions in the model top-{_k}, val-averaged)'
    )
    METRIC_LABELS[f'perfect_at_{_k}'] = (
        f'P@{_k}: per-event perfect recall at top-{_k} tracks '
        f'(fraction of events with all 3 GT pions in top-{_k}, val-averaged)'
    )
del _k


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train TrackPreFilter (Stage 1)')
    add_common_training_args(
        parser, default_lr=1e-3, default_batch_size=96, default_epochs=50,
    )
    parser.add_argument('--model-name', type=str, default='PreFilter')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-neighbors', type=int, default=16)
    parser.add_argument('--num-message-rounds', type=int, default=3)
    parser.add_argument('--aggregation-mode', type=str, default='max',
                        choices=['max'])
    parser.add_argument('--use-edge-features',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--loss-type', type=str, default='pairwise',
                        choices=['pairwise', 'listwise_ce', 'infonce', 'logit_adjust'])
    parser.add_argument('--logit-adjust-tau', type=float, default=1.0)
    parser.add_argument('--listwise-temperature', type=float, default=1.0)
    parser.add_argument('--use-augmentation', action='store_true')
    parser.add_argument('--clustering-dim', type=int, default=8)
    parser.add_argument('--feature-embed-mode', type=str, default='per_feature',
                        choices=('none', 'per_feature'))
    parser.add_argument('--feature-embed-dim', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=OPTIMIZER_NAMES)
    parser.add_argument('--checkpoint-criterion', type=str,
                        default='recall_at_200', choices=CHECKPOINT_CRITERIA)

    parser.add_argument('--profile-steps', type=int, default=0)
    parser.add_argument('--profile-output', type=str, default=None)
    parser.add_argument('--profile-record-shapes', action='store_true')
    parser.add_argument('--profile-memory', action='store_true')
    parser.add_argument('--profile-chrome-trace', action='store_true')
    return parser


def _model_kwargs_from_args(args) -> dict:
    return {
        'dropout': args.dropout,
        'num_neighbors': args.num_neighbors,
        'num_message_rounds': args.num_message_rounds,
        'aggregation_mode': args.aggregation_mode,
        'use_edge_features': args.use_edge_features,
        'loss_type': args.loss_type,
        'logit_adjust_tau': args.logit_adjust_tau,
        'listwise_temperature': args.listwise_temperature,
        'clustering_dim': args.clustering_dim,
        'feature_embed_mode': args.feature_embed_mode,
        'feature_embed_dim': args.feature_embed_dim,
    }


def _format_metrics(metrics: dict[str, float]) -> str:
    perfect_200 = metrics.get('perfect_at_200', 0.0)
    perfect_256 = metrics.get('perfect_at_256', 0.0)
    recall_256 = metrics.get('recall_at_256', 0.0)
    recall_500 = metrics.get('recall_at_500', 0.0)
    recall_600 = metrics.get('recall_at_600', 0.0)
    rank_p90 = metrics.get('gt_rank_p90', 0.0)
    return (
        f'R@30: {metrics["recall_at_30"]:.4f} | '
        f'R@100: {metrics["recall_at_100"]:.4f} | '
        f'R@200: {metrics["recall_at_200"]:.4f} | '
        f'R@256: {recall_256:.4f} | '
        f'R@500: {recall_500:.4f} | '
        f'R@600: {recall_600:.4f} | '
        f'P@200: {perfect_200:.4f} | '
        f'P@256: {perfect_256:.4f} | '
        f'd\': {metrics["d_prime"]:.3f} | '
        f'rank: {metrics["median_gt_rank"]:.0f} '
        f'(p90={rank_p90:.0f})'
    )


def _run_profile_mode(args) -> None:
    from torch.profiler import ProfilerActivity, profile, schedule

    device = torch.device(args.device)
    experiment_dir, _, _ = setup_experiment(args)
    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    (
        train_loader, _, data_config,
        _steps_per_epoch, mask_input_index, label_input_index,
    ) = build_data_loaders(args, logger)
    copy_auto_yaml(args, experiment_dir, logger)

    network_module = load_network_module(args.network)
    model, _ = network_module.get_model(data_config, **_model_kwargs_from_args(args))
    model = model.to(device)

    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    output_dir = args.profile_output or os.path.join(experiment_dir, 'profile')
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'profile_summary.txt')
    trace_path = os.path.join(output_dir, 'profile_trace.json')

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    wait_steps, warmup_steps = 2, 3
    profiler_schedule = schedule(
        wait=wait_steps, warmup=warmup_steps,
        active=args.profile_steps, repeat=1,
    )
    total_batches = wait_steps + warmup_steps + args.profile_steps

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-8,
    )
    model.train()
    logger.info(
        f'Profiling: wait={wait_steps} warmup={warmup_steps} '
        f'active={args.profile_steps} → total {total_batches} batches | '
        f'shapes={args.profile_record_shapes} memory={args.profile_memory} '
        f'trace={args.profile_chrome_trace}',
    )

    with profile(
        activities=activities, schedule=profiler_schedule,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_memory, with_stack=False,
    ) as profiler:
        batch_count = 0
        for batch in train_loader:
            if batch_count >= total_batches:
                break
            X = batch[0]
            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=grad_scaler is not None):
                loss_dict = model.compute_loss(
                    points, features, lorentz_vectors, mask, track_labels,
                )
                loss_dict.pop('_scores', None)
                loss = loss_dict['total_loss']
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            profiler.step()
            batch_count += 1

    if args.profile_chrome_trace:
        profiler.export_chrome_trace(trace_path)
        logger.info(f'Chrome trace written: {trace_path}')
    summary = profiler.key_averages().table(
        sort_by=('cuda_time_total' if device.type == 'cuda' else 'cpu_time_total'),
        row_limit=40,
    )
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f'Summary written: {summary_path}')
    print(summary)


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()
    if args.amp and args.optimizer in ('soap', 'muon'):
        logger.warning(
            f'--amp is incompatible with --optimizer {args.optimizer}; '
            'force-disabling AMP for this run.',
        )
        args.amp = False

    if args.profile_steps > 0:
        logger.info(
            f'=== Profiling mode: {args.profile_steps} active steps ===',
        )
        _run_profile_mode(args)
        return

    state: dict = {'augmentation': None}

    def optimizer_factory(model, args):
        return build_optimizer_factory(
            name=args.optimizer, model=model, lr=args.lr,
            weight_decay=args.weight_decay, amp_enabled=args.amp,
        )

    def on_epoch_start(original_model, model, epoch, args):
        if args.use_augmentation and state['augmentation'] is None:
            from utils.set_augmentation import SetAugmentation
            device = next(original_model.parameters()).device
            state['augmentation'] = SetAugmentation().to(device)

        progress = (epoch - 1) / max(1, args.epochs - 1)
        original_model.set_temperature_progress(progress)
        drw_warmup_epochs = int(
            args.epochs * original_model.drw_warmup_fraction,
        )
        drw_active = epoch > drw_warmup_epochs
        original_model.set_drw_active(drw_active)
        logger.info(
            f'Schedule | T={original_model.current_ranking_temperature:.3f}'
            f' | σ={original_model.current_denoising_sigma:.3f}'
            f' | DRW={"ON" if drw_active else "off"}'
            f' (warmup={drw_warmup_epochs})',
        )

    def preprocess_train_inputs(points, features, lorentz, mask, labels):
        if state['augmentation'] is None:
            return points, features, lorentz, mask
        return state['augmentation'](points, features, lorentz, mask, labels)

    def compute_loss_val(model, points, features, lorentz, mask, labels):
        return model.compute_loss(
            points, features, lorentz, mask, labels,
            use_contrastive_denoising=False,
        )

    def metrics_factory():
        return MetricsAccumulator(k_values=VAL_METRICS_K_VALUES)

    def make_checkpoint_dict(*, epoch, original_model, optimizer,
                              best_selection_value, best_val_loss,
                              best_val_epoch, global_batch_count,
                              val_losses, val_metrics, args):
        return {
            'epoch': epoch,
            'model_state_dict': original_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_score': best_selection_value,
            'checkpoint_criterion': args.checkpoint_criterion,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
        }

    def on_resume(checkpoint, original_model, optimizer, args, device, resume_state):
        if 'best_val_score' in checkpoint:
            resume_state['best_selection_value'] = checkpoint['best_val_score']
        elif args.checkpoint_criterion == 'recall_at_200':
            resume_state['best_selection_value'] = checkpoint.get(
                'best_val_recall_at_200', 0.0,
            )
        else:
            resume_state['best_selection_value'] = 0.0

    def log_summary(*, epoch, val_metrics, train_eval_metrics, val_losses,
                     train_losses, is_best, best_value, best_epoch,
                     selection_value):
        criterion = args.checkpoint_criterion
        val_loss = val_losses['total_loss']
        if train_eval_metrics:
            logger.info(
                f'Epoch {epoch} train_eval | '
                f'total: {train_losses["total_loss"]:.5f} | '
                f'{_format_metrics(train_eval_metrics)}',
            )
        val_summary = _format_metrics(val_metrics)
        if is_best:
            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f} '
                f'{criterion}: {selection_value:.4f} ★ new best | '
                f'{val_summary}',
            )
        else:
            epochs_since_best = epoch - best_epoch
            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f} '
                f'(best {criterion}: {best_value:.4f}, '
                f'{epochs_since_best} epochs ago) | '
                f'{val_summary}',
            )

    run_training(
        args=args,
        logger=logger,
        network_module_path=args.network,
        get_model_kwargs=_model_kwargs_from_args(args),
        selection_metric=args.checkpoint_criterion,
        criterion_name_short='R@200',
        optimizer_factory=optimizer_factory,
        metric_labels=METRIC_LABELS,
        compute_loss_val_fn=compute_loss_val,
        metrics_accumulator_factory=metrics_factory,
        preprocess_train_inputs=preprocess_train_inputs,
        on_epoch_start=on_epoch_start,
        bn_train_mode_for_val=True,
        log_metrics_summary=log_summary,
        make_checkpoint_dict_fn=make_checkpoint_dict,
        on_resume=on_resume,
        use_torch_compile=not args.no_compile,
    )


if __name__ == '__main__':
    main()
