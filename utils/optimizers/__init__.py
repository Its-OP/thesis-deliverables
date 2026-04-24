"""Optimizer factory for training scripts.

Provides a small `build_optimizer(name, model, lr, weight_decay, amp_enabled, **kwargs)`
that returns a PyTorch-compatible optimizer for one of:

- ``'adamw'`` — default, ``torch.optim.AdamW`` (preserves the existing setup).
- ``'soap'``  — second-order-ish preconditioned optimizer from Vyas et al. 2024
  (https://arxiv.org/abs/2409.11321), vendored from
  https://github.com/nikhilvyas/SOAP.
- ``'muon'``  — Newton-Schulz orthogonalization of 2D momentum from
  Keller Jordan 2024 (https://kellerjordan.github.io/posts/muon/), vendored
  from https://github.com/KellerJordan/Muon.

Both SOAP and Muon are **not compatible with torch.amp mixed precision**:
SOAP has no lower-precision support yet, and Muon is not compatible with
``torch.cuda.amp.GradScaler``. If ``amp_enabled=True`` is passed with either
of these names, the factory logs a WARNING but still builds the optimizer —
callers are expected to honor the warning and force-disable AMP in their
training loops (training scripts do this automatically). The
``amp_enabled`` parameter is therefore defensive only; the factory itself
does not configure AMP, just flags the mismatch.

Example::

    from utils.optimizers import build_optimizer
    optimizer = build_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp_enabled=args.amp,
    )
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .muon import SingleDeviceMuonWithAuxAdam
from .soap import SOAP

logger = logging.getLogger(__name__)


# Known optimizer names exposed via the factory. Training scripts reuse this
# tuple for argparse ``choices=`` to keep the CLI in sync.
OPTIMIZER_NAMES: tuple[str, ...] = ('adamw', 'soap', 'muon')


def build_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    amp_enabled: bool,
    **extra_kwargs,
) -> torch.optim.Optimizer:
    """Build and return an optimizer by name.

    Trainable parameters (``p.requires_grad is True``) are collected from
    ``model.parameters()``. Frozen parameters are excluded so cascades with a
    frozen Stage 1 pre-filter don't accidentally feed dead parameters into
    the optimizer.

    Args:
        name: One of :data:`OPTIMIZER_NAMES` (``'adamw'``, ``'soap'``, ``'muon'``).
        model: The model whose trainable parameters will be optimized.
        lr: Peak learning rate (before any warmup/decay schedule).
        weight_decay: Weight decay coefficient. Applied to all trainable
            parameters; the factory does NOT currently exclude biases or
            normalization parameters (matches existing AdamW behavior).
        amp_enabled: Whether mixed-precision (``torch.amp.GradScaler``) will be
            used in the training loop. For ``'soap'`` or ``'muon'`` this must
            be ``False`` — if ``True``, the factory logs a WARNING (but still
            builds the optimizer), and the caller is expected to ignore
            ``--amp`` in its training loop. The training scripts force-disable
            AMP automatically when either of these optimizers is selected.
        **extra_kwargs: Optimizer-specific extras. ``'soap'`` understands
            ``precondition_frequency`` (default 10).

    Returns:
        A ``torch.optim.Optimizer`` subclass instance.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(
            'No trainable parameters to optimize — check that the model '
            'has any parameter with requires_grad=True.'
        )

    if name == 'adamw':
        return torch.optim.AdamW(
            trainable_parameters, lr=lr, weight_decay=weight_decay,
        )

    if name == 'soap':
        if amp_enabled:
            logger.warning(
                'SOAP does not yet support torch.amp mixed precision '
                '(the upstream repo promises lower-precision support in a '
                'future release). The caller is expected to ignore '
                '--amp in its training loop when using SOAP.'
            )
        precondition_frequency = int(
            extra_kwargs.pop('precondition_frequency', 10),
        )
        return SOAP(
            trainable_parameters,
            lr=lr,
            betas=(0.95, 0.95),
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
        )

    if name == 'muon':
        if amp_enabled:
            logger.warning(
                'Muon is not compatible with torch.cuda.amp.GradScaler. '
                'The caller is expected to ignore --amp in its training '
                'loop when using Muon.'
            )
        muon_param_groups = _split_muon_parameters(
            trainable_parameters, lr=lr, weight_decay=weight_decay,
        )
        return SingleDeviceMuonWithAuxAdam(muon_param_groups)

    raise ValueError(
        f'Unknown optimizer {name!r}. Known: {OPTIMIZER_NAMES}.'
    )


def _split_muon_parameters(
    trainable_parameters: list[nn.Parameter],
    lr: float,
    weight_decay: float,
) -> list[dict]:
    """Split trainable parameters into Muon and auxiliary AdamW groups.

    Muon applies only to **true 2D weight matrices** (i.e. ``p.ndim == 2`` —
    the weight of an ``nn.Linear``). Everything else (biases, LayerNorm /
    BatchNorm gains, 1D parameters, Conv1d / Conv2d kernels with ``ndim >= 3``,
    embedding tables which are 2D but semantically row-vector collections)
    goes into the auxiliary AdamW group.

    Conv1d with ``kernel_size=1`` is mathematically a Linear layer but has
    ``ndim == 3`` (``(out, in, 1)``), so it ends up in the AdamW group — safe
    and explicit. Users with such models will see Muon apply to few or no
    parameters; a warning is logged in that case.

    The returned list has exactly the key set required by
    :class:`SingleDeviceMuonWithAuxAdam` (which asserts the exact key names).

    Args:
        trainable_parameters: Parameters to partition.
        lr: Peak learning rate, applied to both groups.
        weight_decay: Weight decay coefficient, applied to both groups.

    Returns:
        List of 1 or 2 parameter-group dicts.
    """
    muon_parameters: list[nn.Parameter] = []
    adam_parameters: list[nn.Parameter] = []
    for parameter in trainable_parameters:
        if parameter.ndim == 2:
            muon_parameters.append(parameter)
        else:
            adam_parameters.append(parameter)

    muon_numel = sum(parameter.numel() for parameter in muon_parameters)
    adam_numel = sum(parameter.numel() for parameter in adam_parameters)
    total_numel = muon_numel + adam_numel
    if total_numel == 0:
        muon_pct = adam_pct = 0.0
    else:
        muon_pct = 100.0 * muon_numel / total_numel
        adam_pct = 100.0 * adam_numel / total_numel

    logger.info(
        'Muon param split | Muon(2D): %d params, %.0f numel (%.1f%%) | '
        'AuxAdam(non-2D): %d params, %.0f numel (%.1f%%)',
        len(muon_parameters), muon_numel, muon_pct,
        len(adam_parameters), adam_numel, adam_pct,
    )
    if not muon_parameters:
        logger.warning(
            'Muon received zero 2D parameters — the model has no nn.Linear '
            'weights, so Muon degenerates to AdamW on the auxiliary group. '
            'Consider using --optimizer adamw or --optimizer soap instead.'
        )

    groups: list[dict] = []
    if muon_parameters:
        groups.append(
            dict(
                params=muon_parameters,
                lr=lr,
                momentum=0.95,
                weight_decay=weight_decay,
                use_muon=True,
            ),
        )
    if adam_parameters:
        groups.append(
            dict(
                params=adam_parameters,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=weight_decay,
                use_muon=False,
            ),
        )
    return groups
