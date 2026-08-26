from __future__ import annotations

import inspect

import torch

from utils.ema import build_ema_stage2, use_ema_stage2_for_validation
from utils.training import run_training


class _TinyCascade(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stage2 = torch.nn.Linear(4, 2)


def test_run_training_accepts_validation_hooks():
    signature = inspect.signature(run_training)
    assert 'validation_context' in signature.parameters
    assert 'validation_model' in signature.parameters
    assert signature.parameters['validation_context'].default is None
    assert signature.parameters['validation_model'].default is None


def test_validation_context_swaps_stage2_to_ema_and_restores():
    cascade = _TinyCascade()
    ema = build_ema_stage2(cascade, decay=0.5, device=torch.device('cpu'))
    with torch.no_grad():
        for parameter in cascade.stage2.parameters():
            parameter.add_(1.0)
    ema.update_parameters(cascade.stage2)

    live_stage2 = cascade.stage2
    with use_ema_stage2_for_validation(cascade, ema):
        assert cascade.stage2 is ema.module
    assert cascade.stage2 is live_stage2


def test_validation_context_noop_without_ema():
    cascade = _TinyCascade()
    live_stage2 = cascade.stage2
    with use_ema_stage2_for_validation(cascade, None):
        assert cascade.stage2 is live_stage2
    assert cascade.stage2 is live_stage2
