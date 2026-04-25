from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def build_ema_stage2(
    cascade_model: torch.nn.Module,
    decay: float,
    device: torch.device,
):
    """Returns AveragedModel wrapping cascade_model.stage2, or None when decay<=0.
    use_buffers=False so BN running stats (already an EMA via BN momentum) are
    not double-averaged; eval_swap copies live BN buffers in directly."""
    if decay <= 0.0:
        return None
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    return AveragedModel(
        cascade_model.stage2,
        device=device,
        multi_avg_fn=get_ema_multi_avg_fn(decay=decay),
        use_buffers=False,
    )


@contextmanager
def use_ema_stage2_for_validation(
    cascade_model: torch.nn.Module,
    ema_stage2,
):
    """Swap cascade_model.stage2 → EMA module for validation, restore on exit.
    Pass the PRE-COMPILE cascade_model — torch.compile must never see the swap.
    No-op when ema_stage2 is None."""
    if ema_stage2 is None:
        yield
        return

    live_stage2 = cascade_model.stage2

    # Sync live BN running stats → EMA buffers (named_buffers only, not
    # state_dict — must not overwrite EMA-averaged parameters).
    with torch.no_grad():
        live_buffers = dict(live_stage2.named_buffers())
        for buffer_name, ema_buffer in ema_stage2.module.named_buffers():
            if buffer_name in live_buffers:
                ema_buffer.copy_(live_buffers[buffer_name])

    cascade_model.stage2 = ema_stage2.module
    try:
        yield
    finally:
        cascade_model.stage2 = live_stage2


def resume_ema_state(
    ema_stage2,
    checkpoint: dict,
    cascade_model: torch.nn.Module,
    decay: float,
    device: torch.device,
):
    """Restore EMA from checkpoint, or rebuild from post-resume live weights
    if the checkpoint lacks ema_state_dict. Returns None when ema_stage2 is None."""
    if ema_stage2 is None:
        return None

    saved_ema_state = checkpoint.get('ema_state_dict')
    if saved_ema_state is not None:
        ema_stage2.load_state_dict(saved_ema_state)
        logger.info(
            f'Resumed EMA state from checkpoint '
            f'(n_averaged={int(ema_stage2.n_averaged.item())})',
        )
        return ema_stage2

    logger.warning(
        f'Checkpoint has no ema_state_dict but this run has '
        f'--ema-decay={decay}. Rebuilding EMA from the post-resume '
        f'live weights — first ~{int(1.0 / max(1e-6, 1.0 - decay))} '
        f'steps will be a warm-up.',
    )
    return build_ema_stage2(cascade_model, decay=decay, device=device)
