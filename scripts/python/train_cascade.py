from __future__ import annotations

import argparse
import logging
import os

import torch

from utils.checkpointing import CheckpointManager
from utils.ema import (
    build_ema_stage2,
    resume_ema_state,
    use_ema_stage2_for_validation,
)
from utils.metrics import MetricsAccumulator
from utils.optimizers import OPTIMIZER_NAMES, build_optimizer as build_optimizer_factory
from utils.training import (
    add_common_training_args,
    run_training,
)

logger = logging.getLogger('train_cascade')

torch.set_float32_matmul_precision('high')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train CascadeModel (Stage 1 → Stage 2)',
    )
    add_common_training_args(
        parser, default_lr=1e-3, default_batch_size=96, default_epochs=50,
    )
    parser.add_argument('--stage1-checkpoint', type=str, required=True)
    parser.add_argument('--stage1-num-neighbors', type=int, default=16)
    parser.add_argument('--top-k1', type=int, default=256)
    parser.add_argument('--model-name', type=str, default='Cascade')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=OPTIMIZER_NAMES)

    parser.add_argument('--stage2-embed-dim', type=int, default=512)
    parser.add_argument('--stage2-num-heads', type=int, default=8)
    parser.add_argument('--stage2-num-layers', type=int, default=2)
    parser.add_argument('--stage2-pair-embed-dims', type=str, default='64,64,64')
    parser.add_argument('--stage2-ffn-ratio', type=int, default=4)
    parser.add_argument('--stage2-dropout', type=float, default=0.1)
    parser.add_argument('--stage2-pair-extra-dim', type=int, default=6)
    parser.add_argument('--stage2-pair-embed-mode', type=str, default='concat',
                        choices=['concat', 'sum'])
    parser.add_argument('--stage2-loss-mode', type=str, default='pairwise',
                        choices=['pairwise', 'lambda_rank', 'rs_at_k', 'hybrid_lambda'])
    parser.add_argument('--stage2-rs-at-k-target', type=int, default=200)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.amp and args.optimizer in ('soap', 'muon'):
        logger.warning(
            f'--amp is incompatible with --optimizer {args.optimizer}; '
            'force-disabling AMP for this run.',
        )
        args.amp = False

    pair_embed_dims = [int(x) for x in args.stage2_pair_embed_dims.split(',')]
    state = {'ema_stage2': None, 'original_model': None}

    def optimizer_factory(model, args):
        # Cascade uses utils.optimizers.build_optimizer (SOAP/Muon support).
        return build_optimizer_factory(
            name=args.optimizer, model=model, lr=args.lr,
            weight_decay=args.weight_decay, amp_enabled=args.amp,
        )

    def on_epoch_start(original_model, model, epoch, args):
        # hybrid_lambda annealing.
        training_progress = (epoch - 1) / max(1, args.epochs - 1)
        stage2 = getattr(model, 'stage2', None)
        if stage2 is not None and hasattr(stage2, 'set_training_progress'):
            stage2.set_training_progress(training_progress)
        # Lazy EMA build on first epoch (after model is built + on device).
        if state['original_model'] is None:
            state['original_model'] = original_model
            state['ema_stage2'] = build_ema_stage2(
                original_model, decay=args.ema_decay,
                device=next(original_model.parameters()).device,
            )
            if state['ema_stage2'] is not None:
                logger.info(
                    f'EMA enabled on Stage 2: decay={args.ema_decay}',
                )

    def on_step_end(model, loss_dict, popped, batch_index):
        if state['ema_stage2'] is not None:
            state['ema_stage2'].update_parameters(model.stage2)

    def update_val_metrics(accumulator, popped, model_inputs, labels, model):
        # Reuse _run_stage1 to scatter Stage 2 scores back to full-event positions.
        per_track_scores = popped['_scores'].detach()
        points, features, lorentz_vectors, mask = model_inputs
        with torch.no_grad():
            filtered = model._run_stage1(
                points, features, lorentz_vectors, mask, labels,
            )
        selected_indices = filtered['selected_indices']
        full_scores = torch.full_like(mask.squeeze(1), float('-inf'))
        full_scores.scatter_(1, selected_indices, per_track_scores)
        accumulator.update(full_scores, labels, mask)

    def make_checkpoint_dict(*, epoch, original_model, optimizer,
                              best_selection_value, best_val_loss,
                              best_val_epoch, global_batch_count,
                              val_losses, val_metrics, args):
        ema_stage2 = state['ema_stage2']
        return {
            'epoch': epoch,
            'model_state_dict': original_model.stage2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_state_dict': (
                ema_stage2.state_dict() if ema_stage2 is not None else None
            ),
            'best_val_loss': best_val_loss,
            'best_val_recall_at_50': best_selection_value,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
        }

    def load_model_fn(model, checkpoint):
        # Per-stage save: 'model_state_dict' is bare Stage 2 weights.
        model.stage2.load_state_dict(checkpoint['model_state_dict'])

    def on_resume(checkpoint, original_model, optimizer, args, device, resume_state):
        state['ema_stage2'] = resume_ema_state(
            ema_stage2=state['ema_stage2'],
            checkpoint=checkpoint,
            cascade_model=original_model,
            decay=args.ema_decay,
            device=device,
        )
        resume_state['best_selection_value'] = checkpoint.get(
            'best_val_recall_at_50', 0.0,
        )

    def metrics_factory():
        return MetricsAccumulator(
            k_values=tuple(k for k in (10, 20, 30, 50, 100, 200) if k < args.top_k1),
        )

    run_training(
        args=args,
        logger=logger,
        network_module_path=args.network,
        get_model_kwargs={
            'stage1_checkpoint': args.stage1_checkpoint,
            'stage1_num_neighbors': args.stage1_num_neighbors,
            'top_k1': args.top_k1,
            'stage2_embed_dim': args.stage2_embed_dim,
            'stage2_num_heads': args.stage2_num_heads,
            'stage2_num_layers': args.stage2_num_layers,
            'stage2_pair_embed_dims': pair_embed_dims,
            'stage2_pair_extra_dim': args.stage2_pair_extra_dim,
            'stage2_pair_embed_mode': args.stage2_pair_embed_mode,
            'stage2_ffn_ratio': args.stage2_ffn_ratio,
            'stage2_dropout': args.stage2_dropout,
            'stage2_loss_mode': args.stage2_loss_mode,
            'stage2_rs_at_k_target': args.stage2_rs_at_k_target,
        },
        selection_metric='recall_at_50',
        criterion_name_short='R@50',
        optimizer_factory=optimizer_factory,
        metrics_accumulator_factory=metrics_factory,
        on_epoch_start=on_epoch_start,
        on_step_end=on_step_end,
        update_val_metrics_fn=update_val_metrics,
        make_checkpoint_dict_fn=make_checkpoint_dict,
        on_resume=on_resume,
        load_model_fn=load_model_fn,
        epoch_metrics_extras_fn=lambda args, epoch: {'top_k1': args.top_k1},
        use_torch_compile=not args.no_compile,
        train_eval_steps_divisor=4,
    )


if __name__ == '__main__':
    main()
