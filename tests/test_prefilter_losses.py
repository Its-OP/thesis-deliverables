"""Unit tests for weaver.nn.model.prefilter_losses.

Covers: listwise_ce_loss, infonce_in_event, logit_adjust_offset,
object_condensation_loss.
"""
from __future__ import annotations

import math

import pytest
import torch

from weaver.nn.model.prefilter_losses import (
    infonce_in_event,
    listwise_ce_loss,
    logit_adjust_offset,
    object_condensation_loss,
)


BATCH_SIZE = 3
NUM_TRACKS = 40


def _make_scores_and_labels(seed: int = 0) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Make synthetic scores / labels / valid_mask.

    Event layout: 3 GT pions at indices {5, 10, 15}, rest noise. Last 5
    tracks are padded.
    """
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(BATCH_SIZE, NUM_TRACKS, generator=generator)
    labels = torch.zeros(BATCH_SIZE, NUM_TRACKS)
    labels[:, [5, 10, 15]] = 1.0
    valid_mask = torch.ones(BATCH_SIZE, NUM_TRACKS, dtype=torch.bool)
    valid_mask[:, -5:] = False  # padded tail
    return scores, labels, valid_mask


class TestListwiseCELoss:
    def test_finite_on_random_scores(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        loss = listwise_ce_loss(scores, labels, valid_mask)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_loss_is_smaller_when_positives_score_higher(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        # Scenario A: random scores
        loss_random = listwise_ce_loss(scores.clone(), labels, valid_mask)
        # Scenario B: push positives up by a big margin
        boosted = scores.clone()
        boosted[labels == 1.0] += 10.0
        loss_boosted = listwise_ce_loss(boosted, labels, valid_mask)
        assert loss_boosted < loss_random

    def test_gradient_flow(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        scores = scores.clone().requires_grad_(True)
        loss = listwise_ce_loss(scores, labels, valid_mask)
        loss.backward()
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()
        # Gradient must be nonzero on at least some tracks
        assert (scores.grad.abs() > 0).any()

    def test_zero_loss_when_no_positives(self):
        scores = torch.randn(1, 5)
        labels = torch.zeros(1, 5)
        valid_mask = torch.ones(1, 5, dtype=torch.bool)
        loss = listwise_ce_loss(scores, labels, valid_mask)
        assert loss.item() == 0.0


class TestInfoNCEInEvent:
    def test_finite(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        loss = infonce_in_event(scores, labels, valid_mask)
        assert torch.isfinite(loss)

    def test_decreases_on_better_ranking(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        loss_random = infonce_in_event(scores.clone(), labels, valid_mask)
        boosted = scores.clone()
        boosted[labels == 1.0] += 10.0
        loss_boosted = infonce_in_event(boosted, labels, valid_mask)
        assert loss_boosted < loss_random

    def test_gradient_flow(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        scores = scores.clone().requires_grad_(True)
        loss = infonce_in_event(scores, labels, valid_mask)
        loss.backward()
        assert torch.isfinite(scores.grad).all()


class TestLogitAdjustOffset:
    def test_formula(self):
        # π_neg = 1127, π_pos = 3 → log(1127/3) ≈ 5.929
        offset = logit_adjust_offset(num_positives=3, num_negatives=1127, tau=1.0)
        assert offset == pytest.approx(math.log(1127 / 3), rel=1e-6)

    def test_tau_scales_offset(self):
        base = logit_adjust_offset(3, 1127, tau=1.0)
        scaled = logit_adjust_offset(3, 1127, tau=2.0)
        assert scaled == pytest.approx(2.0 * base)

    def test_degenerate_cases_return_zero(self):
        assert logit_adjust_offset(0, 100, tau=1.0) == 0.0
        assert logit_adjust_offset(100, 0, tau=1.0) == 0.0


class TestObjectCondensationLoss:
    def test_finite(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        embeddings = torch.randn(BATCH_SIZE, 8, NUM_TRACKS)
        beta = torch.sigmoid(scores)
        loss = object_condensation_loss(
            embeddings, beta, labels, valid_mask,
        )
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        scores, labels, valid_mask = _make_scores_and_labels()
        embeddings = torch.randn(BATCH_SIZE, 8, NUM_TRACKS, requires_grad=True)
        beta = torch.sigmoid(scores).clone().requires_grad_(True)
        loss = object_condensation_loss(
            embeddings, beta, labels, valid_mask,
        )
        loss.backward()
        assert torch.isfinite(embeddings.grad).all()
        assert torch.isfinite(beta.grad).all()


class TestTrackPreFilterLossDispatch:
    """End-to-end check that TrackPreFilter._ranking_loss honors loss_type."""

    @staticmethod
    def _make_model_and_inputs(loss_type: str, **extra):
        from weaver.nn.model.TrackPreFilter import TrackPreFilter

        model = TrackPreFilter(
            mode='mlp', input_dim=7, num_message_rounds=1,
            loss_type=loss_type, **extra,
        )
        points = torch.randn(2, 2, 40)
        features = torch.randn(2, 7, 40)
        lorentz_vectors = torch.randn(2, 4, 40).abs()
        mask = torch.ones(2, 1, 40)
        mask[:, :, -5:] = 0.0
        labels = torch.zeros(2, 1, 40)
        labels[:, 0, [5, 10, 15]] = 1.0
        return model, points, features, lorentz_vectors, mask, labels

    @pytest.mark.parametrize(
        'loss_type',
        ['pairwise', 'listwise_ce', 'infonce', 'logit_adjust'],
    )
    def test_finite_loss_for_each_loss_type(self, loss_type):
        model, points, features, lorentz_vectors, mask, labels = (
            self._make_model_and_inputs(loss_type)
        )
        model.train()
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, labels,
            use_contrastive_denoising=False,
        )
        assert torch.isfinite(loss_dict['ranking_loss'])
        assert torch.isfinite(loss_dict['total_loss'])

    def test_logit_adjust_changes_loss(self):
        """Logit adjustment should shift the loss vs. plain pairwise."""
        model_plain, points, features, lv, mask, labels = (
            self._make_model_and_inputs('pairwise')
        )
        torch.manual_seed(0)
        with torch.no_grad():
            loss_plain = model_plain.compute_loss(
                points, features, lv, mask, labels,
                use_contrastive_denoising=False,
            )['ranking_loss'].item()

        model_adjust, _, _, _, _, _ = self._make_model_and_inputs(
            'logit_adjust', logit_adjust_tau=2.0,
        )
        # Copy weights so the two models are otherwise identical
        model_adjust.load_state_dict(model_plain.state_dict())
        torch.manual_seed(0)
        with torch.no_grad():
            loss_adjust = model_adjust.compute_loss(
                points, features, lv, mask, labels,
                use_contrastive_denoising=False,
            )['ranking_loss'].item()

        assert loss_adjust != pytest.approx(loss_plain, rel=1e-3)
