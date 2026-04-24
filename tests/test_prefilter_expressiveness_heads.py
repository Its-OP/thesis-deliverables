"""Tests for prefilter expressiveness plug-in heads (P1, P2, P3).

The prefilter-expressiveness sweep (target P@256 > 0.90) inserts three
optional modules into the TrackPreFilter forward path without changing
the E2a-default behaviour when they are disabled:

* ``PerFeatureEmbedding`` — per-feature 1×1 grouped-Conv projection
  (P1): each of the 16 raw features goes through its own learnable
  embedding before any cross-feature mixing.
* ``FeatureGate`` — SE-style squeeze-excite gate on track_mlp output
  (P2): learns per-dim importance per track.
* ``FiLMHead`` — event-level context → (γ, β) modulation (P3):
  conditions per-track features on event-wide statistics.

Tests pin forward shape, gradient flow, correctness invariants, and
param-count sanity. The wiring into ``TrackPreFilter`` is tested
separately (`test_track_prefilter_expressiveness_wiring.py`).
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# P1 — PerFeatureEmbedding
# ---------------------------------------------------------------------------


class TestPerFeatureEmbedding:
    def test_output_shape(self):
        from weaver.nn.model.prefilter_expressiveness import PerFeatureEmbedding

        module = PerFeatureEmbedding(num_features=16, embed_dim=32)
        x = torch.randn(2, 16, 50)
        y = module(x)
        assert y.shape == (2, 16 * 32, 50)

    def test_rejects_wrong_channel_count(self):
        import pytest

        from weaver.nn.model.prefilter_expressiveness import PerFeatureEmbedding

        module = PerFeatureEmbedding(num_features=16, embed_dim=32)
        x = torch.randn(2, 8, 50)
        with pytest.raises(ValueError):
            module(x)

    def test_gradient_flows_to_input(self):
        from weaver.nn.model.prefilter_expressiveness import PerFeatureEmbedding

        module = PerFeatureEmbedding(num_features=16, embed_dim=32)
        x = torch.randn(2, 16, 50, requires_grad=True)
        y = module(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_grouped_independence_feature_channel_isolation(self):
        """Changing only feature ``f``'s values must leave the output
        slice for every other feature unchanged (grouped conv has no
        cross-feature mixing at this stage)."""
        from weaver.nn.model.prefilter_expressiveness import PerFeatureEmbedding

        torch.manual_seed(0)
        module = PerFeatureEmbedding(num_features=4, embed_dim=8)
        module.eval()
        x_a = torch.randn(1, 4, 10)
        x_b = x_a.clone()
        # Perturb feature 1 only.
        x_b[:, 1, :] += 5.0

        with torch.no_grad():
            y_a = module(x_a).view(1, 4, 8, 10)
            y_b = module(x_b).view(1, 4, 8, 10)

        # Feature 0, 2, 3 slices are unchanged.
        for feature_index in (0, 2, 3):
            torch.testing.assert_close(
                y_a[:, feature_index], y_b[:, feature_index],
                atol=1e-6, rtol=1e-6,
            )
        # Feature 1 slice did change (sanity).
        assert not torch.allclose(y_a[:, 1], y_b[:, 1])

    def test_param_count_matches_analytic(self):
        """Grouped Conv1d: weight (F*E, 1, 1) + bias (F*E,) + LayerNorm
        (γ, β) of shape (E,)."""
        from weaver.nn.model.prefilter_expressiveness import PerFeatureEmbedding

        num_features, embed_dim = 16, 32
        module = PerFeatureEmbedding(num_features, embed_dim)
        total = sum(p.numel() for p in module.parameters())
        expected = num_features * embed_dim * 2 + embed_dim * 2
        assert total == expected


# ---------------------------------------------------------------------------
# P2 — FeatureGate (SE-style squeeze-excite)
# ---------------------------------------------------------------------------


class TestFeatureGate:
    def test_output_shape_matches_input(self):
        from weaver.nn.model.prefilter_expressiveness import FeatureGate

        module = FeatureGate(hidden_dim=256, bottleneck=16)
        x = torch.randn(3, 256, 40)
        mask = torch.ones(3, 1, 40)
        y = module(x, mask)
        assert y.shape == x.shape

    def test_masked_positions_are_preserved_if_input_masked(self):
        """Gate shouldn't leak through a masked position when the
        caller respects ``mask``. FeatureGate itself multiplies by the
        sigmoid gate; the forward does NOT zero-out masked positions
        automatically — the caller handles that."""
        from weaver.nn.model.prefilter_expressiveness import FeatureGate

        module = FeatureGate(hidden_dim=256, bottleneck=16)
        module.eval()
        x = torch.randn(2, 256, 30)
        mask = torch.ones(2, 1, 30)
        mask[:, :, 15:] = 0.0
        # Zero the masked part first (caller contract).
        x_masked = x * mask
        y = module(x_masked, mask)
        # Masked region stays zero (gate * 0 = 0 everywhere).
        assert torch.allclose(
            y[:, :, 15:], torch.zeros_like(y[:, :, 15:]),
            atol=1e-6,
        )

    def test_gradient_flows_to_input_and_mask_ignored(self):
        from weaver.nn.model.prefilter_expressiveness import FeatureGate

        module = FeatureGate(hidden_dim=64, bottleneck=8)
        x = torch.randn(2, 64, 20, requires_grad=True)
        mask = torch.ones(2, 1, 20)
        y = module(x, mask)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_param_count_is_reasonable(self):
        """SE-style with two Conv1ds: hidden→bottleneck and
        bottleneck→hidden."""
        from weaver.nn.model.prefilter_expressiveness import FeatureGate

        module = FeatureGate(hidden_dim=256, bottleneck=16)
        total = sum(p.numel() for p in module.parameters())
        # Conv1d(256→16): 256*16 + 16. Conv1d(16→256): 16*256 + 256.
        expected = (256 * 16 + 16) + (16 * 256 + 256)
        assert total == expected


# ---------------------------------------------------------------------------
# P3 — FiLMHead (event-context → per-track (γ, β))
# ---------------------------------------------------------------------------


class TestFiLMHead:
    def test_output_shape(self):
        from weaver.nn.model.prefilter_expressiveness import FiLMHead

        module = FiLMHead(
            num_features=16, hidden_dim=256, context_dim=32,
        )
        features = torch.randn(3, 16, 40)
        mask = torch.ones(3, 1, 40)
        track_embedding = torch.randn(3, 256, 40)
        y = module(track_embedding, features, mask)
        assert y.shape == (3, 256, 40)

    def test_context_invariant_to_masked_tokens(self):
        """The event-level (mean, std) context should ignore masked
        tokens. Adding a spike at a masked position must not change
        the output."""
        from weaver.nn.model.prefilter_expressiveness import FiLMHead

        torch.manual_seed(0)
        module = FiLMHead(
            num_features=16, hidden_dim=64, context_dim=16,
        )
        module.eval()

        features_a = torch.randn(1, 16, 30)
        features_b = features_a.clone()
        mask = torch.ones(1, 1, 30)
        mask[:, :, 20:] = 0.0
        # Put a spike in the masked region of features_b.
        features_b[:, :, 25] = 100.0

        track_embedding = torch.randn(1, 64, 30)
        with torch.no_grad():
            y_a = module(track_embedding, features_a, mask)
            y_b = module(track_embedding, features_b, mask)
        torch.testing.assert_close(y_a, y_b, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        from weaver.nn.model.prefilter_expressiveness import FiLMHead

        module = FiLMHead(
            num_features=16, hidden_dim=64, context_dim=16,
        )
        features = torch.randn(2, 16, 20, requires_grad=True)
        mask = torch.ones(2, 1, 20)
        track_embedding = torch.randn(2, 64, 20, requires_grad=True)
        y = module(track_embedding, features, mask)
        y.sum().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        assert track_embedding.grad is not None
        assert torch.isfinite(track_embedding.grad).all()


# ---------------------------------------------------------------------------
# P4 — SoftAttentionAggregator (replaces max-pool in message passing)
# ---------------------------------------------------------------------------


class TestSoftAttentionAggregator:
    def test_output_shape_matches_max_pool_output(self):
        """Max-pool over K neighbours produces ``(B, H, P)``. The
        soft-attention path must match this shape — it replaces max,
        not extends it."""
        from weaver.nn.model.prefilter_expressiveness import (
            SoftAttentionAggregator,
        )

        module = SoftAttentionAggregator(
            hidden_dim=64, edge_dim=4, bottleneck=16,
        )
        current = torch.randn(2, 64, 10)                 # (B, H, P)
        neighbor_feats = torch.randn(2, 64, 10, 8)       # (B, H, P, K)
        neighbor_valid = torch.ones(2, 1, 10, 8)         # (B, 1, P, K)
        edge_feats = torch.randn(2, 4, 10, 8)            # (B, E, P, K)
        pooled = module(current, neighbor_feats, neighbor_valid, edge_feats)
        assert pooled.shape == (2, 64, 10)

    def test_attention_weights_sum_to_one_over_valid_neighbors(self):
        """Soft-attention's signature property: softmax over valid K."""
        from weaver.nn.model.prefilter_expressiveness import (
            SoftAttentionAggregator,
        )

        module = SoftAttentionAggregator(
            hidden_dim=8, edge_dim=4, bottleneck=4,
        )
        current = torch.randn(1, 8, 3)
        neighbor_feats = torch.randn(1, 8, 3, 5)
        # Mark the last 2 neighbours of track 0 as invalid.
        neighbor_valid = torch.ones(1, 1, 3, 5)
        neighbor_valid[0, 0, 0, 3:] = 0.0
        edge_feats = torch.randn(1, 4, 3, 5)
        weights = module.compute_weights(
            current, neighbor_feats, neighbor_valid, edge_feats,
        )
        assert weights.shape == (1, 3, 5)
        # Masked-out neighbours carry zero weight (softmax with -inf).
        assert weights[0, 0, 3:].abs().max().item() < 1e-6
        # Sum over K equals 1 for every track that has ≥1 valid neighbour.
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(1, 3),
            atol=1e-5, rtol=1e-5,
        )

    def test_gradient_flows(self):
        from weaver.nn.model.prefilter_expressiveness import (
            SoftAttentionAggregator,
        )

        module = SoftAttentionAggregator(
            hidden_dim=8, edge_dim=4, bottleneck=4,
        )
        current = torch.randn(1, 8, 3, requires_grad=True)
        neighbor_feats = torch.randn(1, 8, 3, 5, requires_grad=True)
        neighbor_valid = torch.ones(1, 1, 3, 5)
        edge_feats = torch.randn(1, 4, 3, 5, requires_grad=True)
        pooled = module(current, neighbor_feats, neighbor_valid, edge_feats)
        pooled.sum().backward()
        assert current.grad is not None and torch.isfinite(current.grad).all()
        assert (
            neighbor_feats.grad is not None
            and torch.isfinite(neighbor_feats.grad).all()
        )

    def test_all_invalid_produces_safe_output(self):
        """If every neighbour is masked, we fall back to the center's
        own features — returning zeros or the current feature vector,
        never NaN."""
        from weaver.nn.model.prefilter_expressiveness import (
            SoftAttentionAggregator,
        )

        module = SoftAttentionAggregator(
            hidden_dim=8, edge_dim=4, bottleneck=4,
        )
        module.eval()
        current = torch.randn(1, 8, 3)
        neighbor_feats = torch.randn(1, 8, 3, 5)
        neighbor_valid = torch.zeros(1, 1, 3, 5)
        edge_feats = torch.randn(1, 4, 3, 5)
        with torch.no_grad():
            pooled = module(current, neighbor_feats, neighbor_valid, edge_feats)
        assert torch.isfinite(pooled).all()
