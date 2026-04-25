"""Unit tests for CascadeModel (Stage 1 → top-K → Stage 2 pipeline).

TDD: Written before implementation. Tests cover:
    - Construction with frozen Stage 1 + pluggable Stage 2
    - Forward pass chains both stages correctly
    - Stage 1 remains frozen (no gradients flow back)
    - Filtered tensors have correct K1 dimension
    - Stage 1 scores are passed to Stage 2
    - compute_loss() returns Stage 2 loss dict with cascade metadata
    - Dummy Stage 2 (identity scorer) works end-to-end
"""
import pytest
import torch
import torch.nn as nn

from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 4
NUM_TRACKS = 200
INPUT_DIM = 16
TOP_K1 = 60  # Smaller than production 600 for test speed


def _make_inputs(batch_size=BATCH_SIZE, num_tracks=NUM_TRACKS, seed=42):
    """Create physically sensible inputs matching the data pipeline."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        batch_size, INPUT_DIM, num_tracks, generator=generator,
    )

    transverse_momentum = (
        torch.rand(batch_size, 1, num_tracks, generator=generator) * 5 + 0.5
    )
    px = transverse_momentum * torch.cos(phi)
    py = transverse_momentum * torch.sin(phi)
    pz = transverse_momentum * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px**2 + py**2 + pz**2 + pion_mass**2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, -30:] = 0.0

    track_labels = torch.zeros(batch_size, 1, num_tracks)
    for batch_index in range(batch_size):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


class DummyStage2(nn.Module):
    """Minimal Stage 2 model for testing the cascade skeleton.

    Takes filtered inputs and returns per-track scores via a single linear
    layer. Implements the Stage 2 interface: forward() and compute_loss().
    """

    def __init__(self, input_dim):
        super().__init__()
        self.scorer = nn.Conv1d(input_dim, 1, kernel_size=1)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Score each track in the filtered set.

        Returns:
            scores: (B, K1) per-track scores.
        """
        valid_mask = mask.squeeze(1).bool()
        scores = self.scorer(features).squeeze(1)
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        return scores

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        valid_mask = mask.squeeze(1).bool()
        labels = track_labels.squeeze(1)[:, :scores.shape[1]]

        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]

        loss = nn.functional.binary_cross_entropy_with_logits(
            valid_scores, valid_labels,
        )
        return {
            'total_loss': loss,
            'stage2_loss': loss,
            '_scores': scores,
        }


def _make_cascade(top_k1=TOP_K1):
    """Create a CascadeModel with a frozen Stage 1 and dummy Stage 2."""
    stage1 = TrackPreFilter(
        mode='mlp', input_dim=INPUT_DIM,
        hidden_dim=64, num_message_rounds=1,
    )
    stage2 = DummyStage2(input_dim=INPUT_DIM)
    model = CascadeModel(
        stage1=stage1,
        stage2=stage2,
        top_k1=top_k1,
    )
    return model


# ---- Construction ----

class TestCascadeConstruction:
    """Test CascadeModel initialization."""

    def test_stage1_is_frozen(self):
        """Stage 1 parameters should have requires_grad=False."""
        model = _make_cascade()
        for name, parameter in model.stage1.named_parameters():
            assert not parameter.requires_grad, (
                f'Stage 1 param {name} should be frozen'
            )

    def test_stage2_is_trainable(self):
        """Stage 2 parameters should have requires_grad=True."""
        model = _make_cascade()
        for name, parameter in model.stage2.named_parameters():
            assert parameter.requires_grad, (
                f'Stage 2 param {name} should be trainable'
            )

    def test_top_k1_stored(self):
        """top_k1 should be accessible on the model."""
        model = _make_cascade(top_k1=600)
        assert model.top_k1 == 600

    def test_stage1_bn_stays_on_batch_stats_in_eval(self):
        """Model-level eval() must leave stage1 BN layers using batch
        statistics (not the stale running stats that would drop R@600
        from 0.90 to 0.70). The cascade achieves this by disabling each
        BN's running-stat tracking, so eval forward falls through to
        batch-stat computation.
        """
        model = _make_cascade()
        model.eval()
        stage1_bn_layers = [
            m for m in model.stage1.modules()
            if isinstance(m, nn.BatchNorm1d)
        ]
        assert stage1_bn_layers, 'stage1 has no BatchNorm layers to check'
        for batch_norm in stage1_bn_layers:
            assert batch_norm.track_running_stats is False
            assert batch_norm.running_mean is None
            assert batch_norm.running_var is None

    def test_stage1_bn_stays_on_batch_stats_in_train(self):
        model = _make_cascade()
        model.train()
        for m in model.stage1.modules():
            if isinstance(m, nn.BatchNorm1d):
                assert m.track_running_stats is False
                assert m.running_mean is None
                assert m.running_var is None

    def test_stage2_follows_outer_mode(self):
        """Stage 2 has no BN, so its submodules should track the cascade
        mode normally — eval() puts them into eval, train() into train.
        """
        model = _make_cascade()
        model.eval()
        for submodule in model.stage2.modules():
            assert submodule.training is False
        model.train()
        for submodule in model.stage2.modules():
            assert submodule.training is True


# ---- Forward pass ----

class TestCascadeForward:
    """Test the cascade forward pass."""

    def test_forward_returns_scores(self):
        """forward() should return (B, K1) scores from Stage 2."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, _ = _make_inputs()

        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, TOP_K1)

    def test_forward_scores_finite_for_valid(self):
        """Valid track scores should be finite."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, _ = _make_inputs()

        scores = model(points, features, lorentz_vectors, mask)
        # All K1 tracks should be valid (selected from valid tracks)
        assert torch.isfinite(scores).all()

    def test_no_gradients_to_stage1(self):
        """Backward through the cascade should not produce Stage 1 gradients."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        for name, parameter in model.stage1.named_parameters():
            assert parameter.grad is None or parameter.grad.abs().sum() == 0, (
                f'Stage 1 param {name} received gradients but should be frozen'
            )

    def test_gradients_flow_to_stage2(self):
        """Backward should produce gradients in Stage 2 parameters."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.stage2.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        assert params_with_grad > 0, 'No gradients reached Stage 2'


# ---- Compute loss ----

class TestCascadeLoss:
    """Test the cascade compute_loss() method."""

    def test_loss_dict_has_total_loss(self):
        """compute_loss() should return a dict with 'total_loss'."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_loss_dict_has_stage2_components(self):
        """Loss dict should contain Stage 2 loss components."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'stage2_loss' in loss_dict

    def test_loss_dict_has_stage1_recall(self):
        """Loss dict should include Stage 1 R@K1 for monitoring."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'stage1_recall_at_k1' in loss_dict


# ---- Filtering ----

class TestCascadeFiltering:
    """Test that Stage 1 filtering produces correct dimensions."""

    def test_filtered_track_count(self):
        """Filtered tensors should have exactly K1 tracks."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        filtered = model.stage1.filter_tracks(
            points, features, lorentz_vectors, mask, track_labels,
            top_k=TOP_K1,
        )
        assert filtered['features'].shape[2] == TOP_K1
        assert filtered['points'].shape[2] == TOP_K1
        assert filtered['lorentz_vectors'].shape[2] == TOP_K1
        assert filtered['mask'].shape[2] == TOP_K1
        assert filtered['track_labels'].shape[2] == TOP_K1

    def test_selected_indices_returned(self):
        """_run_stage1 should return selected_indices for metric scatter."""
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        filtered = model._run_stage1(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'selected_indices' in filtered
        assert filtered['selected_indices'].shape == (BATCH_SIZE, TOP_K1)
        # Indices should be in valid range [0, NUM_TRACKS)
        assert (filtered['selected_indices'] >= 0).all()
        assert (filtered['selected_indices'] < NUM_TRACKS).all()

    def test_end_to_end_recall_uses_full_event_gt(self):
        """End-to-end recall must count GT tracks from the FULL event, not
        just those that survived Stage 1 filtering.

        If 3 GT pions exist in the full event but Stage 1 only selects 2,
        the recall denominator should still be 3.
        """
        model = _make_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()

        # Get cascade output
        with torch.no_grad():
            filtered = model._run_stage1(
                points, features, lorentz_vectors, mask, track_labels,
            )
            stage2_scores = model.stage2(
                filtered['points'], filtered['features'],
                filtered['lorentz_vectors'], filtered['mask'],
                filtered['stage1_scores'],
            )

        # Scatter Stage 2 scores back to full positions
        full_scores = torch.full(
            (BATCH_SIZE, NUM_TRACKS), float('-inf'),
        )
        full_scores.scatter_(
            1, filtered['selected_indices'], stage2_scores,
        )

        # Count GT in full event
        full_gt_count = (
            (track_labels.squeeze(1) == 1.0)
            & mask.squeeze(1).bool()
        ).sum(dim=1)  # Should be 3 per event

        # Count GT in Stage 2's top-K (e.g. top-30 within K1=60)
        from utils.metrics import MetricsAccumulator
        accumulator = MetricsAccumulator(k_values=(30,))
        accumulator.update(full_scores, track_labels, mask)
        metrics = accumulator.compute()

        # Denominator should be 3 (full event GT count), not fewer
        assert metrics['total_gt_tracks'] == int(full_gt_count.sum().item())
        # Recall should be in [0, 1]
        assert 0.0 <= metrics['recall_at_30'] <= 1.0

    def test_stage1_scores_passed_to_stage2(self):
        """Stage 2 should receive stage1_scores as input."""
        # Use a Stage 2 that records its inputs
        class RecordingStage2(DummyStage2):
            def __init__(self, input_dim):
                super().__init__(input_dim)
                self.received_stage1_scores = None

            def forward(self, points, features, lorentz_vectors, mask,
                        stage1_scores):
                self.received_stage1_scores = stage1_scores
                return super().forward(
                    points, features, lorentz_vectors, mask, stage1_scores,
                )

        stage1 = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
        )
        stage2 = RecordingStage2(input_dim=INPUT_DIM)
        model = CascadeModel(stage1=stage1, stage2=stage2, top_k1=TOP_K1)

        points, features, lorentz_vectors, mask, _ = _make_inputs()
        model(points, features, lorentz_vectors, mask)

        assert stage2.received_stage1_scores is not None
        assert stage2.received_stage1_scores.shape == (BATCH_SIZE, TOP_K1)
        assert torch.isfinite(stage2.received_stage1_scores).all()
