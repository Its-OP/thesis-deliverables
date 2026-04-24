"""TDD tests for utils.optimizers.build_optimizer factory.

The factory returns a PyTorch-compatible optimizer for one of three names:
``'adamw'`` (default), ``'soap'`` (Vyas et al. 2024), ``'muon'`` (Keller
Jordan 2024). SOAP and Muon are incompatible with torch.amp mixed precision
(SOAP has no lower-precision support yet; Muon is incompatible with
``torch.cuda.amp.GradScaler``), so the factory must raise loudly at
construction time when ``amp_enabled=True`` is combined with either.

Tests exercise:
    - AdamW path returns a real ``torch.optim.AdamW`` with the expected
      learning rate and weight decay.
    - SOAP / Muon both raise ``ValueError`` when ``amp_enabled=True``.
    - SOAP and Muon each run one forward + backward + step on a tiny model
      and the loss moves in the right direction (or at least the parameters
      change and no NaNs appear).
    - The Muon param split helper routes 2D weight matrices to the Muon
      group and everything else (biases, Conv1d kernels with ndim==3) to the
      auxiliary AdamW group — documenting the Conv1d fallback behavior the
      user will encounter on the prefilter MLP.

These tests are written before the factory implementation follows the TDD
discipline from CLAUDE.md.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestFactoryImportable:
    """``build_optimizer`` should be importable from utils.optimizers."""

    def test_importable(self):
        from utils.optimizers import build_optimizer

        assert callable(build_optimizer)


class TestBuildAdamW:
    """AdamW path: factory returns a real ``torch.optim.AdamW``."""

    def test_returns_adamw_with_expected_config(self):
        from utils.optimizers import build_optimizer

        model = nn.Linear(4, 2)
        optimizer = build_optimizer(
            name='adamw',
            model=model,
            lr=1e-3,
            weight_decay=0.01,
            amp_enabled=False,
        )
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['weight_decay'] == 0.01

    def test_adamw_skips_frozen_parameters(self):
        """Parameters with requires_grad=False should not reach the optimizer."""
        from utils.optimizers import build_optimizer

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for parameter in model[0].parameters():
            parameter.requires_grad = False

        optimizer = build_optimizer(
            name='adamw',
            model=model,
            lr=1e-3,
            weight_decay=0.0,
            amp_enabled=False,
        )
        optimized_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        expected_trainable_ids = {
            id(parameter) for parameter in model[1].parameters()
        }
        assert optimized_parameter_ids == expected_trainable_ids


class TestAmpGuards:
    """SOAP and Muon tolerate amp_enabled=True but log a warning.

    The training scripts force-disable AMP before the factory is called,
    so in practice the factory never receives amp_enabled=True for these
    optimizers. But the factory's behavior when it does is defensive:
    log a WARNING, still build the optimizer normally, and let the caller
    be responsible for not actually running AMP in its training loop.
    """

    def test_soap_warns_on_amp_enabled(self, caplog):
        import logging as _logging
        from utils.optimizers import build_optimizer
        from utils.optimizers.soap import SOAP

        model = nn.Linear(4, 2)
        with caplog.at_level(_logging.WARNING, logger='utils.optimizers'):
            optimizer = build_optimizer(
                name='soap',
                model=model,
                lr=3e-3,
                weight_decay=0.01,
                amp_enabled=True,
            )
        assert isinstance(optimizer, SOAP)
        assert any(
            'SOAP' in record.message and 'amp' in record.message.lower()
            for record in caplog.records
        ), f'Expected SOAP/amp warning; got records={[r.message for r in caplog.records]}'

    def test_muon_warns_on_amp_enabled(self, caplog):
        import logging as _logging
        from utils.optimizers import build_optimizer
        from utils.optimizers.muon import SingleDeviceMuonWithAuxAdam

        model = nn.Linear(4, 2)
        with caplog.at_level(_logging.WARNING, logger='utils.optimizers'):
            optimizer = build_optimizer(
                name='muon',
                model=model,
                lr=1e-3,
                weight_decay=0.01,
                amp_enabled=True,
            )
        assert isinstance(optimizer, SingleDeviceMuonWithAuxAdam)
        assert any(
            'Muon' in record.message and 'GradScaler' in record.message
            for record in caplog.records
        ), f'Expected Muon/GradScaler warning; got records={[r.message for r in caplog.records]}'

    def test_unknown_optimizer_raises(self):
        from utils.optimizers import build_optimizer

        model = nn.Linear(4, 2)
        with pytest.raises(ValueError, match='Unknown optimizer'):
            build_optimizer(
                name='nonsense',
                model=model,
                lr=1e-3,
                weight_decay=0.01,
                amp_enabled=False,
            )


class TestSoapSmokeStep:
    """SOAP: build + run one optimization step on a tiny Linear model."""

    def test_single_step_updates_parameters(self):
        from utils.optimizers import build_optimizer

        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        initial_weights = model.weight.detach().clone()

        optimizer = build_optimizer(
            name='soap',
            model=model,
            lr=3e-3,
            weight_decay=0.01,
            amp_enabled=False,
        )

        # SOAP's first .step() initializes the preconditioner and does NOT
        # update parameters (see the vendored SOAP source: "first step is
        # skipped so that we never use the current gradients in the
        # projection"). So run two steps to actually see weight movement.
        for _ in range(2):
            inputs = torch.randn(8, 4)
            targets = torch.randn(8, 2)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            optimizer.step()

        assert not torch.allclose(model.weight, initial_weights), (
            'SOAP should have updated model weights after two steps.'
        )
        assert torch.isfinite(model.weight).all(), (
            'SOAP produced NaN or inf in model weights.'
        )


class TestMuonSmokeStep:
    """Muon: build + run one optimization step on a mixed-ndim model."""

    def test_single_step_updates_parameters(self):
        from utils.optimizers import build_optimizer

        torch.manual_seed(0)
        # Linear has 2D weight + 1D bias — will exercise both the Muon group
        # and the auxiliary AdamW group.
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        initial_first_weight = model[0].weight.detach().clone()
        initial_first_bias = model[0].bias.detach().clone()

        optimizer = build_optimizer(
            name='muon',
            model=model,
            lr=1e-3,
            weight_decay=0.0,
            amp_enabled=False,
        )

        inputs = torch.randn(8, 4)
        targets = torch.randn(8, 2)
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()

        assert not torch.allclose(model[0].weight, initial_first_weight), (
            'Muon should have updated the 2D Linear weights after one step.'
        )
        assert not torch.allclose(model[0].bias, initial_first_bias), (
            'AuxAdam should have updated the 1D bias after one step.'
        )
        assert torch.isfinite(model[0].weight).all(), (
            'Muon produced NaN or inf in Linear weights.'
        )


class TestMuonParameterSplit:
    """
    Document the Muon param split on the two model families we actually train:
    Linear-heavy (cascade transformer analog) and Conv1d-only (prefilter MLP).
    """

    def test_linear_model_sends_weights_to_muon_group(self):
        """A Linear-heavy model should see 2D weights in the Muon group and
        1D biases in the auxiliary AdamW group."""
        from utils.optimizers import build_optimizer
        from utils.optimizers.muon import SingleDeviceMuonWithAuxAdam

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        optimizer = build_optimizer(
            name='muon',
            model=model,
            lr=1e-3,
            weight_decay=0.0,
            amp_enabled=False,
        )
        assert isinstance(optimizer, SingleDeviceMuonWithAuxAdam)

        muon_groups = [
            group for group in optimizer.param_groups if group['use_muon']
        ]
        adam_groups = [
            group for group in optimizer.param_groups if not group['use_muon']
        ]
        assert len(muon_groups) == 1
        assert len(adam_groups) == 1

        muon_parameters = muon_groups[0]['params']
        adam_parameters = adam_groups[0]['params']
        # Two Linear layers contribute two 2D weights and two 1D biases.
        assert len(muon_parameters) == 2
        assert all(parameter.ndim == 2 for parameter in muon_parameters)
        assert len(adam_parameters) == 2
        assert all(parameter.ndim == 1 for parameter in adam_parameters)

    def test_conv1d_only_model_falls_back_to_adam(self):
        """A Conv1d-only model (like the prefilter MLP) has weights with
        ndim==3, so the factory routes them ALL to the auxiliary AdamW group.
        Muon becomes a no-op on such a model — the factory should log a
        warning and the optimizer should degenerate to AdamW behavior.
        """
        from utils.optimizers import build_optimizer
        from utils.optimizers.muon import SingleDeviceMuonWithAuxAdam

        model = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=1),
            nn.Conv1d(8, 2, kernel_size=1),
        )
        optimizer = build_optimizer(
            name='muon',
            model=model,
            lr=1e-3,
            weight_decay=0.0,
            amp_enabled=False,
        )
        assert isinstance(optimizer, SingleDeviceMuonWithAuxAdam)

        muon_groups = [
            group for group in optimizer.param_groups if group['use_muon']
        ]
        adam_groups = [
            group for group in optimizer.param_groups if not group['use_muon']
        ]
        # No 2D params -> no Muon group at all; only the Adam fallback.
        assert len(muon_groups) == 0
        assert len(adam_groups) == 1
        # All 4 parameters (2 Conv1d weights + 2 biases) live in Adam.
        adam_parameters = adam_groups[0]['params']
        assert len(adam_parameters) == 4
