from __future__ import annotations

import argparse
import logging
import math
import os

import torch
from torch.utils.data import DataLoader

from utils.dataset_helpers import (
    extract_label_from_inputs,
    trim_to_max_valid_tracks,
)
from utils.metrics import (
    CoupleMetricsAccumulator,
    format_couple_metrics_table,
)
from utils.training import (
    add_common_training_args,
    run_training,
)

logger = logging.getLogger('train_couple_reranker')

torch.set_float32_matmul_precision('high')


def _build_metric_labels(
    k_values_tracks: list[int],
    k_values_couples: list[int],
) -> dict[str, str]:
    labels: dict[str, str] = {
        'train': 'Train loss (couple ranking, mean per epoch)',
        'val': 'Validation loss (couple ranking)',
        'lr': 'Learning rate',
    }
    for k in k_values_tracks:
        labels[f'val_d_at_{k}_tracks'] = (
            f'D@{k}_tracks: events with ≥2 GT pions in ParT top-{k} tracks'
        )
    for k in k_values_couples:
        labels[f'val_c_at_{k}_couples'] = (
            f'C@{k}_couples: events with ≥1 GT couple in top-{k} of reranker output'
        )
        labels[f'val_rc_at_{k}_couples'] = (
            f'RC@{k}_couples: C@{k}_couples AND full triplet in cascade Stage 1 top-K1'
        )
    return labels


def _compute_batch_mean_first_gt_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    batch_size = scores.shape[0]
    ranks: list[float] = []
    for batch_index in range(batch_size):
        valid_mask = mask[batch_index] > 0.5
        if not valid_mask.any():
            continue
        gt_mask = (labels[batch_index] > 0.5) & valid_mask
        if not gt_mask.any():
            continue
        event_scores = scores[batch_index].clone()
        event_scores = event_scores.masked_fill(~valid_mask, float('-inf'))
        sorted_indices = torch.argsort(event_scores, descending=True)
        sorted_gt = gt_mask[sorted_indices]
        first_gt_position = int(sorted_gt.float().argmax().item())
        ranks.append(float(first_gt_position + 1))
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return 0.0
    stat = spearmanr(xs, ys).statistic
    if stat is None:
        return 0.0
    return 0.0 if math.isnan(stat) else float(stat)


def calibrate_reranker_batchnorm(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    calibration_steps: int = 200,
) -> None:
    """Reset and recalibrate CoupleReranker BN running stats post-training."""
    for module in model.couple_reranker.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()
    model.couple_reranker.train()
    model.cascade.train()
    with torch.no_grad():
        for batch_index, (X, _, _) in enumerate(train_loader):
            if batch_index >= calibration_steps:
                break
            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, _ = extract_label_from_inputs(inputs, label_input_index)
            points, features, lorentz_vectors, mask = model_inputs
            dummy_labels = torch.zeros_like(mask)
            couple_inputs = model._build_couple_inputs(
                points, features, lorentz_vectors, mask, dummy_labels,
            )
            model.couple_reranker(couple_inputs['couple_features'])
            if (batch_index + 1) % 50 == 0:
                logger.info(
                    f'BN calibration: {batch_index + 1}/{calibration_steps}',
                )
    model.eval()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train CoupleReranker (Stage 3) on top of a frozen cascade',
    )
    add_common_training_args(
        parser, default_lr=5e-4, default_batch_size=16, default_epochs=50,
    )
    parser.add_argument('--cascade-checkpoint', type=str, required=True)
    parser.add_argument('--top-k2', type=int, default=50)
    parser.add_argument('--model-name', type=str, default='CoupleReranker')
    parser.add_argument('--cosine-power', type=float, default=2.0)

    parser.add_argument('--couple-hidden-dim', type=int, default=256)
    parser.add_argument('--couple-num-residual-blocks', type=int, default=4)
    parser.add_argument('--couple-dropout', type=float, default=0.1)
    parser.add_argument('--couple-ranking-num-samples', type=int, default=50)
    parser.add_argument('--couple-ranking-temperature', type=float, default=1.0)
    parser.add_argument('--couple-label-smoothing', type=float, default=0.10)
    parser.add_argument('--couple-projector-dim', type=int, default=32)

    parser.add_argument('--bn-calibration-steps', type=int, default=200)

    parser.add_argument(
        '--k-values-tracks', type=int, nargs='+',
        default=[30, 50, 75, 100, 200],
    )
    parser.add_argument(
        '--k-values-couples', type=int, nargs='+',
        default=[50, 75, 100, 200],
        help='Must include 100 (selection criterion is C@100_couples).',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if 100 not in args.k_values_couples:
        parser.error(
            '--k-values-couples must include 100 (the selection criterion '
            f'is C@100_couples). Got: {args.k_values_couples}',
        )

    state = {'batch_loss_samples': [], 'batch_mean_gt_rank_samples': []}

    def compute_loss_train(model, points, features, lorentz, mask, labels):
        out = model.compute_loss(points, features, lorentz, mask, labels)
        scores_t = out.get('_scores')
        labels_t = out.get('_couple_labels')
        mask_t = out.get('_couple_mask')
        if scores_t is not None and labels_t is not None and mask_t is not None:
            with torch.no_grad():
                rank = _compute_batch_mean_first_gt_rank(scores_t, labels_t, mask_t)
                if rank is not None:
                    state['batch_mean_gt_rank_samples'].append(rank)
                    state['batch_loss_samples'].append(
                        float(out['total_loss'].item()),
                    )
        return out

    def update_val_metrics(accumulator, popped, model_inputs, labels, model):
        accumulator.update(
            popped['_scores'].detach(),
            popped['_couple_labels'].detach(),
            popped['_couple_mask'].detach(),
            n_gt_in_top_k1=popped['_n_gt_in_top_k1'].detach(),
            n_gt_in_top_k_tracks=popped['_n_gt_in_top_k_tracks'].detach(),
        )

    def make_checkpoint_dict(*, epoch, original_model, optimizer,
                              best_selection_value, best_val_loss,
                              best_val_epoch, global_batch_count,
                              val_losses, val_metrics, args):
        return {
            'epoch': epoch,
            'couple_reranker_state_dict':
                original_model.couple_reranker.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_c_at_100': best_selection_value,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
        }

    def log_summary(*, epoch, val_metrics, train_eval_metrics, val_losses,
                     train_losses, is_best, best_value, best_epoch,
                     selection_value):
        loss_samples = state['batch_loss_samples']
        rank_samples = state['batch_mean_gt_rank_samples']
        if rank_samples:
            mean_rank = sum(rank_samples) / len(rank_samples)
            tail_start = len(loss_samples) // 2
            rho = _spearman_rho(loss_samples[tail_start:], rank_samples[tail_start:])
            train_losses['mean_first_gt_rank_train'] = mean_rank
            train_losses['loss_rank_spearman_rho'] = rho
            logger.info(
                f'Epoch {epoch} train | mean_first_gt_rank_train: '
                f'{mean_rank:.3f} | ρ(loss, rank): {rho:+.3f} '
                f'(last {len(loss_samples) - tail_start} batches)',
            )
        state['batch_loss_samples'] = []
        state['batch_mean_gt_rank_samples'] = []

        val_table = format_couple_metrics_table(
            val_metrics,
            train_loss=train_losses['total_loss'],
            val_loss=val_losses['total_loss'],
            epoch=epoch,
            is_best=is_best,
            best_val_criterion=best_value,
            best_val_epoch=best_epoch,
            criterion_name='C@100c',
            k_values_tracks=tuple(args.k_values_tracks),
            k_values_couples=tuple(args.k_values_couples),
        )
        logger.info('\n' + val_table)

    def final_cleanup(*, args, model, original_model, optimizer, train_loader,
                       val_loader, data_config, mask_input_index,
                       label_input_index, device, checkpoints_dir,
                       best_selection_value, best_val_loss, best_val_epoch,
                       global_batch_count, val_losses, val_metrics, logger):
        if args.bn_calibration_steps <= 0:
            return
        logger.info(
            f'Calibrating CoupleReranker BN running stats '
            f'({args.bn_calibration_steps} steps)...',
        )
        calibrate_reranker_batchnorm(
            model, train_loader, device, data_config,
            mask_input_index, label_input_index,
            calibration_steps=args.bn_calibration_steps,
        )
        calibrated_path = os.path.join(checkpoints_dir, 'best_model_calibrated.pt')
        torch.save({
            'epoch': args.epochs,
            'couple_reranker_state_dict':
                original_model.couple_reranker.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_c_at_100': best_selection_value,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
        }, calibrated_path)
        logger.info(f'Saved calibrated checkpoint: {calibrated_path}')

    def epoch_metrics_extras(args, epoch):
        return {'top_k2': args.top_k2}

    def metrics_factory():
        return CoupleMetricsAccumulator(
            k_values_couples=tuple(args.k_values_couples),
            k_values_tracks=tuple(args.k_values_tracks),
        )

    def optimizer_factory(model, args):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    run_training(
        args=args,
        logger=logger,
        network_module_path=args.network,
        get_model_kwargs={
            'cascade_checkpoint': args.cascade_checkpoint,
            'top_k2': args.top_k2,
            'k_values_tracks': tuple(args.k_values_tracks),
            'couple_hidden_dim': args.couple_hidden_dim,
            'couple_num_residual_blocks': args.couple_num_residual_blocks,
            'couple_dropout': args.couple_dropout,
            'couple_ranking_num_samples': args.couple_ranking_num_samples,
            'couple_ranking_temperature': args.couple_ranking_temperature,
            'couple_label_smoothing': args.couple_label_smoothing,
            'couple_projector_dim': args.couple_projector_dim,
        },
        selection_metric='c_at_100_couples',
        criterion_name_short='C@100',
        optimizer_factory=optimizer_factory,
        metric_labels=_build_metric_labels(
            args.k_values_tracks, args.k_values_couples,
        ),
        compute_loss_train_fn=compute_loss_train,
        metrics_accumulator_factory=metrics_factory,
        pop_keys_train=('_couple_labels', '_couple_mask',
                        '_n_gt_in_top_k1', '_n_gt_in_top_k_tracks'),
        pop_keys_val=('_couple_labels', '_couple_mask',
                      '_n_gt_in_top_k1', '_n_gt_in_top_k_tracks'),
        update_val_metrics_fn=update_val_metrics,
        bn_train_mode_for_val=True,
        also_validate_train=False,
        train_eval_steps_divisor=2,
        log_metrics_summary=log_summary,
        make_checkpoint_dict_fn=make_checkpoint_dict,
        final_cleanup_fn=final_cleanup,
        epoch_metrics_extras_fn=epoch_metrics_extras,
        use_torch_compile=False,
    )


if __name__ == '__main__':
    main()
