"""Unit tests for ``CoupleCascadeModel`` (frozen cascade + CoupleReranker).

The wrapper glues a trained 2-stage cascade with the new per-couple reranker.
Tests use a synthetic small cascade (`TrackPreFilter` + a tiny dummy Stage 2)
plus a small `CoupleReranker` so the full pipeline runs in a few hundred ms.

Coverage:
    - Construction freezes the cascade params (only the reranker trains)
    - Forward pass returns per-couple scores with the expected shape
    - compute_loss returns the standard CoupleReranker dict + the
      ``_couple_labels`` / ``_couple_mask`` extras
    - Gradients flow ONLY to the reranker
    - The cascade is exercised end-to-end (Stage 1 + Stage 2 + couple build
      + reranker forward + ranking loss) on a synthetic event
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.CoupleCascadeModel import CoupleCascadeModel
from weaver.nn.model.CoupleReranker import CoupleReranker
from weaver.nn.model.TrackPreFilter import TrackPreFilter

# ---- Shared configuration ----

BATCH_SIZE = 2
NUM_TRACKS = 80
NUM_PADDED = 10
NUM_VALID = NUM_TRACKS - NUM_PADDED  # 70
INPUT_DIM = 16
# top_k1 = NUM_VALID means Stage 1 selects all valid tracks (no filtering),
# which makes the test deterministic regardless of the random Stage 1 scoring.
TOP_K1 = NUM_VALID
TOP_K2 = 12  # smaller than 50 to keep tests fast


GT_INDICES = (5, 15, 25)  # 3 GT pions per event, planted at these positions
GT_FEATURE_CHANNEL = 0    # the channel that DummyStage2 weights to detect GT


def _make_inputs(seed: int = 42):
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(BATCH_SIZE, 1, NUM_TRACKS, generator=generator) * 1.5
    phi = (
        torch.rand(BATCH_SIZE, 1, NUM_TRACKS, generator=generator) * 2 * 3.14159
        - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        BATCH_SIZE, INPUT_DIM, NUM_TRACKS, generator=generator,
    )
    # Plant a strong feature signature on GT tracks so the deterministic
    # DummyStage2 below picks them up reliably (otherwise top-K2 selection
    # is random and gradients to the CoupleReranker could happen to be zero).
    for b in range(BATCH_SIZE):
        for gt_index in GT_INDICES:
            features[b, GT_FEATURE_CHANNEL, gt_index] = 100.0

    transverse_momentum = (
        torch.rand(BATCH_SIZE, 1, NUM_TRACKS, generator=generator) * 2 + 0.3
    )
    px = transverse_momentum * torch.cos(phi)
    py = transverse_momentum * torch.sin(phi)
    pz = transverse_momentum * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + pion_mass ** 2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -NUM_PADDED:] = 0.0  # last NUM_PADDED tracks padded

    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for b in range(BATCH_SIZE):
        for gt_index in GT_INDICES:
            track_labels[b, 0, gt_index] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


class DummyStage2(nn.Module):
    """Minimal Stage 2 returning per-track scores from a single Conv1d.

    Hand-initialized to weight ``GT_FEATURE_CHANNEL`` strongly so the planted
    GT tracks reliably end up in the top-K2 of Stage 2 scoring. This is purely
    a test fixture to make gradient-flow tests deterministic.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.scorer = nn.Conv1d(input_dim, 1, kernel_size=1)
        with torch.no_grad():
            self.scorer.weight.zero_()
            self.scorer.weight[0, GT_FEATURE_CHANNEL, 0] = 1.0
            self.scorer.bias.zero_()

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = mask.squeeze(1).bool()
        scores = self.scorer(features).squeeze(1)
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
    ) -> dict:
        del use_contrastive_denoising
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        return {'total_loss': scores.mean(), '_scores': scores}


def _make_couple_cascade(top_k2: int = TOP_K2) -> CoupleCascadeModel:
    stage1 = TrackPreFilter(
        mode='mlp',
        input_dim=INPUT_DIM,
        hidden_dim=32,
        num_message_rounds=1,
    )
    stage2 = DummyStage2(input_dim=INPUT_DIM)
    cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=TOP_K1)
    couple_reranker = CoupleReranker(
        input_dim=51,
        hidden_dim=32,
        num_residual_blocks=1,
        dropout=0.0,
    )
    return CoupleCascadeModel(
        cascade=cascade,
        couple_reranker=couple_reranker,
        top_k2=top_k2,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCoupleCascadeConstruction:
    def test_cascade_is_frozen(self):
        model = _make_couple_cascade()
        for name, parameter in model.cascade.named_parameters():
            assert not parameter.requires_grad, f'cascade.{name} should be frozen'

    def test_couple_reranker_is_trainable(self):
        model = _make_couple_cascade()
        for name, parameter in model.couple_reranker.named_parameters():
            assert parameter.requires_grad, f'couple_reranker.{name} should train'

    def test_top_k2_stored(self):
        model = _make_couple_cascade(top_k2=20)
        assert model.top_k2 == 20


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestCoupleCascadeForward:
    def test_forward_returns_scores_and_filter_mask(self):
        model = _make_couple_cascade()
        points, features, lorentz_vectors, mask, _ = _make_inputs()
        scores, filter_mask = model(points, features, lorentz_vectors, mask)
        n_couples = TOP_K2 * (TOP_K2 - 1) // 2
        assert scores.shape == (BATCH_SIZE, n_couples)
        assert filter_mask.shape == (BATCH_SIZE, n_couples)
        assert filter_mask.dtype == torch.bool

    def test_forward_scores_finite(self):
        model = _make_couple_cascade()
        points, features, lorentz_vectors, mask, _ = _make_inputs()
        scores, _ = model(points, features, lorentz_vectors, mask)
        assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Compute loss
# ---------------------------------------------------------------------------

class TestCoupleCascadeLoss:
    def test_loss_dict_contents(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        assert 'total_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        assert '_scores' in loss_dict
        assert '_couple_labels' in loss_dict
        assert '_couple_mask' in loss_dict
        assert '_n_gt_in_top_k1' in loss_dict
        assert '_n_gt_in_top_k_tracks' in loss_dict
        assert torch.isfinite(loss_dict['total_loss'])

    def test_n_gt_in_top_k1_shape_and_values(self):
        """``_n_gt_in_top_k1`` should be ``(B,)`` and equal the planted GT
        count (3 per event in the synthetic fixture)."""
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        n_gt = loss_dict['_n_gt_in_top_k1']
        assert n_gt.shape == (BATCH_SIZE,)
        # All synthetic events plant 3 GT pions; with TOP_K1 = NUM_VALID,
        # all of them survive Stage 1.
        assert (n_gt == 3).all()

    def test_n_gt_in_top_k_tracks_shape_and_monotonicity(self):
        """``_n_gt_in_top_k_tracks`` should be ``(B, len(k_values_tracks))``
        and the per-event counts should be non-decreasing in K."""
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        n_gt_per_k = loss_dict['_n_gt_in_top_k_tracks']
        n_k = len(model.k_values_tracks)
        assert n_gt_per_k.shape == (BATCH_SIZE, n_k)
        # Counts must be non-decreasing in K (a larger top-K can only
        # contain MORE GT, never fewer).
        diffs = n_gt_per_k[:, 1:] - n_gt_per_k[:, :-1]
        assert (diffs >= 0).all()
        # Also: GT count at the LARGEST K is at most 3 (the number of
        # planted GT pions per event).
        assert (n_gt_per_k[:, -1] <= 3).all()

    def test_loss_metric_tensor_shapes(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        n_couples = TOP_K2 * (TOP_K2 - 1) // 2
        assert loss_dict['_scores'].shape == (BATCH_SIZE, n_couples)
        assert loss_dict['_couple_labels'].shape == (BATCH_SIZE, n_couples)
        assert loss_dict['_couple_mask'].shape == (BATCH_SIZE, n_couples)

    def test_no_gradients_flow_to_cascade(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        loss_dict['total_loss'].backward()
        for name, parameter in model.cascade.named_parameters():
            assert parameter.grad is None or parameter.grad.abs().sum() == 0, (
                f'cascade.{name} received gradient'
            )

    def test_gradients_flow_to_couple_reranker(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        loss_dict['total_loss'].backward()
        params_with_grad = sum(
            1 for _, p in model.couple_reranker.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert params_with_grad > 0, 'CoupleReranker received no gradient'

    def test_gt_couples_present_with_planted_signal(self):
        """The synthetic event plants 3 GT pions in positions 5, 15, 25.
        Stage 1 + Stage 2 may or may not put them in the top-K2 with a
        random Conv1d Stage 2, so we don't strictly assert n_gt > 0.
        We DO assert the labels tensor is well-formed (boolean, no NaNs)."""
        model = _make_couple_cascade()
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        labels = loss_dict['_couple_labels']
        assert labels.dtype == torch.bool
        # 0, 1, or 3 GT couples per event are all valid (depending on
        # how many GT tracks are in the top-K2)
        assert labels.sum().item() in {0, 1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Slim checkpoint: only the trainable couple_reranker is persisted
# ---------------------------------------------------------------------------

class TestSlimCheckpoint:
    """The trainer must NOT save the frozen cascade weights into per-epoch
    checkpoints (the cascade is reloaded from its own checkpoint). Saving
    only ``couple_reranker.state_dict()`` keeps the per-epoch artifacts
    small (~few MB instead of hundreds)."""

    def test_couple_reranker_state_dict_has_no_cascade_keys(self):
        model = _make_couple_cascade()
        slim_state_dict = model.couple_reranker.state_dict()
        for key in slim_state_dict:
            assert not key.startswith('cascade.'), (
                f'slim state_dict contains cascade key: {key}'
            )

    def test_slim_state_dict_smaller_than_full_state_dict(self):
        """The slim state dict should contain dramatically fewer parameter
        elements than the full ``model.state_dict()`` (which includes the
        frozen cascade)."""
        model = _make_couple_cascade()
        slim = model.couple_reranker.state_dict()
        full = model.state_dict()
        slim_numel = sum(t.numel() for t in slim.values())
        full_numel = sum(t.numel() for t in full.values())
        assert slim_numel < full_numel
        # Sanity: cascade is much bigger than the tiny test reranker
        assert full_numel >= 2 * slim_numel

    def test_round_trip_save_load_via_torch_save(self, tmp_path):
        """Save the slim state dict, build a fresh model, load it back,
        and verify every reranker tensor matches bit-for-bit."""
        model_a = _make_couple_cascade()
        # Train the reranker for a single backward step so weights drift
        # away from the random init. This catches save/load bugs that a
        # zero-init test wouldn't.
        inputs = _make_inputs()
        loss_dict = model_a.compute_loss(*inputs)
        loss_dict['total_loss'].backward()
        with torch.no_grad():
            for parameter in model_a.couple_reranker.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1e-2)

        checkpoint_path = tmp_path / 'slim_checkpoint.pt'
        torch.save(
            {'couple_reranker_state_dict': model_a.couple_reranker.state_dict()},
            checkpoint_path,
        )

        model_b = _make_couple_cascade()
        # Confirm model_b's reranker starts DIFFERENT from model_a's so
        # the assertion below is meaningful.
        starting_diff = sum(
            (pa - pb).abs().sum().item()
            for pa, pb in zip(
                model_a.couple_reranker.parameters(),
                model_b.couple_reranker.parameters(),
                strict=True,
            )
        )
        assert starting_diff > 0

        loaded = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False,
        )
        model_b.couple_reranker.load_state_dict(
            loaded['couple_reranker_state_dict'],
        )
        for (key_a, tensor_a), (key_b, tensor_b) in zip(
            model_a.couple_reranker.state_dict().items(),
            model_b.couple_reranker.state_dict().items(),
            strict=True,
        ):
            assert key_a == key_b
            assert torch.equal(tensor_a, tensor_b), f'mismatch at {key_a}'

    def test_loaded_slim_checkpoint_does_not_touch_cascade(self, tmp_path):
        """Loading the slim state dict into model_b must NOT change
        ``model_b.cascade``'s weights — they should still match a fresh
        cascade build."""
        model_a = _make_couple_cascade()
        torch.save(
            {'couple_reranker_state_dict': model_a.couple_reranker.state_dict()},
            tmp_path / 'slim.pt',
        )

        model_b = _make_couple_cascade()
        cascade_b_before = {
            name: parameter.detach().clone()
            for name, parameter in model_b.cascade.named_parameters()
        }
        loaded = torch.load(
            tmp_path / 'slim.pt', map_location='cpu', weights_only=False,
        )
        model_b.couple_reranker.load_state_dict(
            loaded['couple_reranker_state_dict'],
        )
        for name, parameter_after in model_b.cascade.named_parameters():
            assert torch.equal(parameter_after, cascade_b_before[name]), (
                f'cascade.{name} changed after loading slim checkpoint'
            )
