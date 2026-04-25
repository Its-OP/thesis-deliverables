"""TDD tests for the EMA path in train_cascade.py.

Covers the new ``--ema-decay`` flag, the ``build_ema_stage2()`` helper,
the ``use_ema_stage2_for_validation()`` context manager, the EMA-aware
checkpoint save format, and the four-case ``resume_ema_state()`` branch
logic. Tests use a tiny synthetic CascadeModel built from a small
``TrackPreFilter`` plus a ``TinyStage2`` containing one ``BatchNorm1d``
and one ``Conv1d`` so the BatchNorm-buffer-copy path has something to
exercise. This avoids invoking the production 6M-param ``CascadeReranker``
in unit tests and keeps the file fast on any device.
"""
from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from train_cascade import (
    _build_parser,
    build_ema_stage2,
    resume_ema_state,
    use_ema_stage2_for_validation,
)
from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---------------------------------------------------------------------------
# Synthetic cascade fixture
# ---------------------------------------------------------------------------

INPUT_DIM = 8
HIDDEN_DIM = 16
TOP_K1 = 10


class TinyStage2(nn.Module):
    """Minimal Stage 2 model for the EMA tests.

    Has one ``BatchNorm1d`` (so the buffer-copy path is exercised) and one
    ``Conv1d`` scoring head whose ``.weight`` is the easy thing to perturb
    in the drift / swap tests below. Implements the Stage 2 interface
    (``forward`` + ``compute_loss`` with ``use_contrastive_denoising`` kwarg)
    so it can drop into ``CascadeModel`` without modification.
    """

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(INPUT_DIM)
        self.linear = nn.Conv1d(INPUT_DIM, 1, kernel_size=1)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.bn(features)
        scores = self.linear(normalized).squeeze(1)
        valid_mask = mask.squeeze(1).bool()
        return scores.masked_fill(~valid_mask, float('-inf'))

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
        use_contrastive_denoising: bool = True,
    ) -> dict[str, torch.Tensor]:
        del use_contrastive_denoising  # unused; matches CascadeReranker API
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        valid_mask = mask.squeeze(1).bool()
        loss = scores[valid_mask].mean()
        return {'total_loss': loss, '_scores': scores}


def _make_tiny_cascade() -> CascadeModel:
    """Build a small CascadeModel suitable for unit testing the EMA helpers."""
    stage1 = TrackPreFilter(
        mode='mlp',
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_message_rounds=1,
    )
    return CascadeModel(stage1=stage1, stage2=TinyStage2(), top_k1=TOP_K1)


# ---------------------------------------------------------------------------
# TestEmaArgparse
# ---------------------------------------------------------------------------

class TestEmaArgparse:
    """The new ``--ema-decay`` CLI flag, default and override."""

    REQUIRED_ARGS = [
        '--data-config', 'dummy.yaml',
        '--data-dir', 'dummy/',
        '--network', 'dummy.py',
        '--stage1-checkpoint', 'dummy.pt',
    ]

    def test_ema_decay_default_is_zero(self):
        """Default must be 0.0 — EMA must be opt-in, not opt-out."""
        parser = _build_parser()
        args = parser.parse_args(self.REQUIRED_ARGS)
        assert args.ema_decay == 0.0

    def test_ema_decay_cli_override(self):
        parser = _build_parser()
        args = parser.parse_args(
            self.REQUIRED_ARGS + ['--ema-decay', '0.999'],
        )
        assert args.ema_decay == 0.999

    def test_ema_decay_float_type(self):
        parser = _build_parser()
        args = parser.parse_args(
            self.REQUIRED_ARGS + ['--ema-decay', '0.9999'],
        )
        assert isinstance(args.ema_decay, float)
        assert args.ema_decay == 0.9999


# ---------------------------------------------------------------------------
# TestBuildEmaStage2
# ---------------------------------------------------------------------------

class TestBuildEmaStage2:
    """``build_ema_stage2()`` helper construction semantics."""

    def test_returns_none_when_decay_zero(self):
        """Critical zero-impact guard: no AveragedModel when disabled."""
        cascade = _make_tiny_cascade()
        result = build_ema_stage2(
            cascade, decay=0.0, device=torch.device('cpu'),
        )
        assert result is None

    def test_returns_none_when_decay_negative(self):
        cascade = _make_tiny_cascade()
        result = build_ema_stage2(
            cascade, decay=-0.5, device=torch.device('cpu'),
        )
        assert result is None

    def test_returns_averaged_model_when_decay_positive(self):
        from torch.optim.swa_utils import AveragedModel
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        assert isinstance(ema, AveragedModel)

    def test_deepcopies_stage2(self):
        """The wrapped module is a separate object with byte-equal init."""
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        assert ema.module is not cascade.stage2
        for ema_param, live_param in zip(
            ema.module.parameters(), cascade.stage2.parameters(),
        ):
            assert torch.equal(ema_param, live_param)

    def test_n_averaged_starts_at_zero(self):
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        assert ema.n_averaged.item() == 0

    def test_respects_device_argument(self):
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        for parameter in ema.module.parameters():
            assert parameter.device.type == 'cpu'


# ---------------------------------------------------------------------------
# TestEmaUpdateDrift
# ---------------------------------------------------------------------------

class TestEmaUpdateDrift:
    """Numerical behavior of ``AveragedModel.update_parameters``.

    Tests the standard EMA update rule:
        θ_ema ← decay · θ_ema + (1 − decay) · θ_live
    """

    def test_parameters_drift_from_live_after_updates(self):
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.5, device=torch.device('cpu'),
        )
        initial_ema_weight = ema.module.linear.weight.clone()

        # Push live weights forward, update EMA — repeat 5 times.
        for _ in range(5):
            with torch.no_grad():
                cascade.stage2.linear.weight.add_(0.1)
            ema.update_parameters(cascade.stage2)

        # The EMA must have moved from its initial state.
        assert not torch.equal(
            ema.module.linear.weight, initial_ema_weight,
        )
        # AND must lag the live weights (decay=0.5 → tracks but lags).
        assert not torch.equal(
            ema.module.linear.weight, cascade.stage2.linear.weight,
        )

    def test_lag_smaller_with_smaller_decay(self):
        """Bigger decay (slower averaging) → bigger lag from live."""

        def measure_lag(decay: float) -> float:
            cascade = _make_tiny_cascade()
            ema = build_ema_stage2(
                cascade, decay=decay, device=torch.device('cpu'),
            )
            for _ in range(10):
                with torch.no_grad():
                    cascade.stage2.linear.weight.add_(0.1)
                ema.update_parameters(cascade.stage2)
            return (
                ema.module.linear.weight - cascade.stage2.linear.weight
            ).abs().mean().item()

        lag_fast = measure_lag(0.1)
        lag_slow = measure_lag(0.9)
        assert lag_slow > lag_fast, (
            f'Higher decay should produce more lag: '
            f'lag(decay=0.9)={lag_slow:.4f} vs lag(decay=0.1)={lag_fast:.4f}'
        )

    def test_n_averaged_increments_per_call(self):
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        for expected in range(1, 6):
            ema.update_parameters(cascade.stage2)
            assert ema.n_averaged.item() == expected


# ---------------------------------------------------------------------------
# TestSwapContextManager
# ---------------------------------------------------------------------------

class TestSwapContextManager:
    """``use_ema_stage2_for_validation`` context manager semantics."""

    def test_swap_installs_ema_module_inside_context(self):
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        with use_ema_stage2_for_validation(cascade, ema):
            assert cascade.stage2 is ema.module

    def test_swap_restores_live_module_on_normal_exit(self):
        cascade = _make_tiny_cascade()
        live_stage2 = cascade.stage2
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        with use_ema_stage2_for_validation(cascade, ema):
            pass
        assert cascade.stage2 is live_stage2

    def test_swap_restores_on_exception(self):
        cascade = _make_tiny_cascade()
        live_stage2 = cascade.stage2
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        with pytest.raises(RuntimeError, match='intentional'):
            with use_ema_stage2_for_validation(cascade, ema):
                raise RuntimeError('intentional')
        assert cascade.stage2 is live_stage2

    def test_swap_copies_bn_running_mean_into_ema(self):
        """The EMA's BN running stats must equal the live ones inside the
        context — verifies the copy_buffers step of the swap.
        """
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        sentinel = torch.arange(INPUT_DIM, dtype=torch.float32)
        with torch.no_grad():
            cascade.stage2.bn.running_mean.copy_(sentinel)

        # Before entering, the EMA's BN running_mean is still at init (zeros).
        assert not torch.equal(ema.module.bn.running_mean, sentinel)

        with use_ema_stage2_for_validation(cascade, ema):
            assert torch.equal(ema.module.bn.running_mean, sentinel)

    def test_swap_does_not_clobber_ema_parameters(self):
        """The buffer-copy step must NOT overwrite EMA-averaged parameters.

        Iterating ``named_buffers()`` (not ``state_dict()``) is what makes
        this safe; this test enforces that contract.
        """
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.5, device=torch.device('cpu'),
        )
        # Drift the live weights so EMA lags by a known amount.
        with torch.no_grad():
            cascade.stage2.linear.weight.add_(1.0)
        ema.update_parameters(cascade.stage2)
        ema_weight_before_swap = ema.module.linear.weight.clone()

        with use_ema_stage2_for_validation(cascade, ema):
            assert torch.equal(
                ema.module.linear.weight, ema_weight_before_swap,
            )

    def test_swap_is_noop_when_ema_is_none(self):
        """The disabled-EMA path must touch nothing."""
        cascade = _make_tiny_cascade()
        live_stage2 = cascade.stage2
        sentinel = torch.arange(INPUT_DIM, dtype=torch.float32)
        with torch.no_grad():
            cascade.stage2.bn.running_mean.copy_(sentinel)

        with use_ema_stage2_for_validation(cascade, None):
            assert cascade.stage2 is live_stage2
            assert torch.equal(cascade.stage2.bn.running_mean, sentinel)

        assert cascade.stage2 is live_stage2
        assert torch.equal(cascade.stage2.bn.running_mean, sentinel)


# ---------------------------------------------------------------------------
# TestCheckpointLayout
# ---------------------------------------------------------------------------

class TestCheckpointLayout:
    """``ema_state_dict`` round-trip through ``torch.save`` / ``torch.load``."""

    def test_checkpoint_contains_ema_state_dict_when_enabled(self, tmp_path):
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.999, device=torch.device('cpu'),
        )
        ema.update_parameters(cascade.stage2)  # n_averaged = 1

        checkpoint = {
            'epoch': 1,
            'model_state_dict': cascade.state_dict(),
            'ema_state_dict': ema.state_dict(),
        }
        path = tmp_path / 'ck.pt'
        torch.save(checkpoint, path)
        loaded = torch.load(path, weights_only=False)

        assert 'ema_state_dict' in loaded
        assert loaded['ema_state_dict'] is not None
        assert 'n_averaged' in loaded['ema_state_dict']
        assert loaded['ema_state_dict']['n_averaged'].item() == 1
        module_keys = [
            k for k in loaded['ema_state_dict'] if k.startswith('module.')
        ]
        assert len(module_keys) > 0

    def test_checkpoint_ema_state_dict_is_none_when_disabled(self, tmp_path):
        """Disabled-EMA checkpoints carry an explicit None marker so
        downstream loaders can distinguish 'EMA off' from 'pre-EMA file'.
        """
        cascade = _make_tiny_cascade()
        checkpoint = {
            'epoch': 1,
            'model_state_dict': cascade.state_dict(),
            'ema_state_dict': None,
        }
        path = tmp_path / 'ck.pt'
        torch.save(checkpoint, path)
        loaded = torch.load(path, weights_only=False)

        assert 'ema_state_dict' in loaded
        assert loaded['ema_state_dict'] is None

    def test_checkpoint_ema_round_trip(self, tmp_path):
        cascade = _make_tiny_cascade()
        original_ema = build_ema_stage2(
            cascade, decay=0.5, device=torch.device('cpu'),
        )
        for _ in range(3):
            with torch.no_grad():
                cascade.stage2.linear.weight.add_(0.1)
            original_ema.update_parameters(cascade.stage2)

        checkpoint = {'ema_state_dict': original_ema.state_dict()}
        path = tmp_path / 'ck.pt'
        torch.save(checkpoint, path)
        loaded = torch.load(path, weights_only=False)

        # Build a fresh wrapper from a fresh cascade and load the saved state.
        fresh_cascade = _make_tiny_cascade()
        loaded_ema = build_ema_stage2(
            fresh_cascade, decay=0.5, device=torch.device('cpu'),
        )
        loaded_ema.load_state_dict(loaded['ema_state_dict'])

        for original_param, loaded_param in zip(
            original_ema.module.parameters(),
            loaded_ema.module.parameters(),
        ):
            assert torch.equal(original_param, loaded_param)
        assert loaded_ema.n_averaged.item() == original_ema.n_averaged.item()


# ---------------------------------------------------------------------------
# TestResumeBranching
# ---------------------------------------------------------------------------

class TestResumeBranching:
    """``resume_ema_state()`` four-case branching from the plan."""

    def test_resume_loads_ema_state_when_present(self):
        """Case (a): EMA on, checkpoint has ema_state_dict → load it."""
        cascade = _make_tiny_cascade()
        saved_ema = build_ema_stage2(
            cascade, decay=0.5, device=torch.device('cpu'),
        )
        with torch.no_grad():
            cascade.stage2.linear.weight.add_(0.5)
        saved_ema.update_parameters(cascade.stage2)
        saved_ema.update_parameters(cascade.stage2)
        saved_state = saved_ema.state_dict()

        resumed_cascade = _make_tiny_cascade()
        resumed_ema = build_ema_stage2(
            resumed_cascade, decay=0.5, device=torch.device('cpu'),
        )
        result = resume_ema_state(
            ema_stage2=resumed_ema,
            checkpoint={'ema_state_dict': saved_state},
            cascade_model=resumed_cascade,
            decay=0.5,
            device=torch.device('cpu'),
        )

        assert result is resumed_ema  # in-place load, same object
        for original_param, loaded_param in zip(
            saved_ema.module.parameters(),
            result.module.parameters(),
        ):
            assert torch.equal(original_param, loaded_param)
        assert result.n_averaged.item() == saved_ema.n_averaged.item()

    def test_resume_missing_ema_warns_and_rebuilds(self, caplog):
        """Case (b): EMA on, checkpoint has no ema_state_dict → rebuild + warn."""
        cascade = _make_tiny_cascade()
        ema = build_ema_stage2(
            cascade, decay=0.5, device=torch.device('cpu'),
        )
        # Simulate that the live model has been re-loaded post-resume to a
        # different parameter state than the freshly-built EMA.
        with torch.no_grad():
            cascade.stage2.linear.weight.add_(2.0)

        with caplog.at_level(logging.WARNING):
            result = resume_ema_state(
                ema_stage2=ema,
                checkpoint={'model_state_dict': cascade.state_dict()},
                cascade_model=cascade,
                decay=0.5,
                device=torch.device('cpu'),
            )

        assert result is not None
        # The rebuilt EMA must reflect the post-load live weights, NOT the
        # pre-load init that the original ema_stage2 was deepcopied from.
        for live_param, rebuilt_param in zip(
            cascade.stage2.parameters(),
            result.module.parameters(),
        ):
            assert torch.equal(live_param, rebuilt_param)
        assert result.n_averaged.item() == 0
        assert any(
            'no ema_state_dict' in record.message.lower()
            for record in caplog.records
        )

    def test_resume_ema_disabled_returns_none(self):
        """Case (c) and (d): EMA off → always returns None, ignores any
        ema_state_dict in the checkpoint.
        """
        cascade = _make_tiny_cascade()
        result_with_ck_ema = resume_ema_state(
            ema_stage2=None,
            checkpoint={'ema_state_dict': {'n_averaged': torch.tensor(5)}},
            cascade_model=cascade,
            decay=0.0,
            device=torch.device('cpu'),
        )
        assert result_with_ck_ema is None

        result_without_ck_ema = resume_ema_state(
            ema_stage2=None,
            checkpoint={'model_state_dict': cascade.state_dict()},
            cascade_model=cascade,
            decay=0.0,
            device=torch.device('cpu'),
        )
        assert result_without_ck_ema is None
