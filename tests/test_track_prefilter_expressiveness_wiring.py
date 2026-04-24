"""Wiring tests: TrackPreFilter accepts the expressiveness CLI flags
and routes inputs through the optional heads (P1, P2, P3).

Full forward-pass checks with small synthetic tensors. Confirms:
* Module attribute presence flipped correctly per flag.
* Disabled flags produce a state-dict compatible with the E2a baseline.
* Enabled flags add parameters (param-count delta).
* Forward works end-to-end and the scorer still returns ``(B, P)``.
"""
from __future__ import annotations

import torch

from weaver.nn.model.TrackPreFilter import TrackPreFilter


def _make_batch(batch_size: int = 2, num_tracks: int = 30, input_dim: int = 16):
    """Synthetic batch with physically-valid Lorentz 4-vectors.

    Lorentz layout is (px, py, pz, E); edge features (``ln kT``, ``ln z``,
    ``ln ΔR``, ``ln m²``) need ``E² − |p|² > 0`` and ``ΔR > 0`` to stay
    finite under the ``log`` path in ``pairwise_lv_fts``. We sample
    (px, py, pz) as scaled normals, add a pion-like mass ``m²``, and
    set ``E = √(|p|² + m²)``.
    """
    points = torch.randn(batch_size, 2, num_tracks) * 2.0  # η, φ spread
    features = torch.randn(batch_size, input_dim, num_tracks)
    momentum = torch.randn(batch_size, 3, num_tracks) * 2.0
    mass_sq = torch.full((batch_size, 1, num_tracks), 0.0196)  # (0.14 GeV)²
    energy = torch.sqrt(
        (momentum ** 2).sum(dim=1, keepdim=True) + mass_sq,
    )
    lorentz = torch.cat([momentum, energy], dim=1)
    mask = torch.ones(batch_size, 1, num_tracks)
    # Mask the last 5 tracks as padding to exercise the masked path.
    mask[:, :, -5:] = 0.0
    return points, features, lorentz, mask


def _make_model(**overrides) -> TrackPreFilter:
    base_config = dict(
        mode='mlp',
        input_dim=16,
        hidden_dim=64,
        num_neighbors=8,
        num_message_rounds=2,
        use_edge_features=True,
        dropout=0.0,
    )
    base_config.update(overrides)
    return TrackPreFilter(**base_config)


class TestBaselineIsUnchangedByDefault:
    def test_no_expressiveness_heads_by_default(self):
        model = _make_model()
        assert getattr(model, 'feature_embedder', None) is None
        assert getattr(model, 'feature_gate', None) is None
        assert getattr(model, 'film_head', None) is None
        assert model.feature_embed_mode == 'none'
        assert model.use_feature_gate is False
        assert model.use_film_head is False


class TestP1PerFeatureEmbedding:
    def test_module_is_installed(self):
        model = _make_model(
            feature_embed_mode='per_feature', feature_embed_dim=16,
        )
        assert model.feature_embedder is not None
        assert model.feature_embedder.num_features == 16
        assert model.feature_embedder.embed_dim == 16

    def test_forward_produces_expected_shape(self):
        model = _make_model(
            feature_embed_mode='per_feature', feature_embed_dim=16,
        )
        model.eval()
        points, features, lorentz, mask = _make_batch()
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        assert scores.shape == (points.shape[0], points.shape[2])
        # Scorer emits -inf at masked positions so they never rank; only
        # valid positions must be finite.
        valid = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid]).all()

    def test_param_count_delta_positive(self):
        baseline = sum(p.numel() for p in _make_model().parameters())
        with_p1 = sum(
            p.numel() for p in _make_model(
                feature_embed_mode='per_feature', feature_embed_dim=16,
            ).parameters()
        )
        assert with_p1 > baseline

    def test_invalid_mode_raises(self):
        import pytest

        with pytest.raises(ValueError):
            _make_model(feature_embed_mode='nonsense')


class TestP2FeatureGate:
    def test_module_is_installed(self):
        model = _make_model(use_feature_gate=True, feature_gate_bottleneck=8)
        assert model.feature_gate is not None
        assert model.feature_gate.bottleneck == 8

    def test_forward_produces_expected_shape(self):
        model = _make_model(use_feature_gate=True)
        model.eval()
        points, features, lorentz, mask = _make_batch()
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        assert scores.shape == (points.shape[0], points.shape[2])
        # Scorer emits -inf at masked positions so they never rank; only
        # valid positions must be finite.
        valid = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid]).all()


class TestP3FiLMHead:
    def test_module_is_installed(self):
        model = _make_model(use_film_head=True, film_context_dim=16)
        assert model.film_head is not None
        assert model.film_head.context_dim == 16

    def test_init_outputs_zero_gamma_beta(self):
        """γ, β heads are zero-initialised so FiLM starts as identity
        modulation: ``(1 + 0) · h + 0 = h``. Verify by calling the
        head directly on an arbitrary (track_embedding, features,
        mask) triple and checking the output matches the input."""
        from weaver.nn.model.prefilter_expressiveness import FiLMHead

        head = FiLMHead(
            num_features=16, hidden_dim=64, context_dim=16,
        )
        head.eval()
        track_embedding = torch.randn(2, 64, 20)
        features = torch.randn(2, 16, 20)
        mask = torch.ones(2, 1, 20)
        with torch.no_grad():
            y = head(track_embedding, features, mask)
        torch.testing.assert_close(
            y, track_embedding, atol=1e-6, rtol=1e-6,
        )


class TestP4SoftAttentionAggregation:
    def test_module_list_is_installed(self):
        model = _make_model(
            use_soft_attention_aggregation=True,
            soft_attention_bottleneck=8,
            num_message_rounds=3,
        )
        assert model.soft_attention_aggregators is not None
        assert len(model.soft_attention_aggregators) == 3

    def test_forward_produces_expected_shape(self):
        model = _make_model(
            use_soft_attention_aggregation=True,
            soft_attention_bottleneck=8,
        )
        model.eval()
        points, features, lorentz, mask = _make_batch()
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        assert scores.shape == (points.shape[0], points.shape[2])
        valid = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid]).all()

    def test_gradient_flows_through_aggregators(self):
        model = _make_model(
            use_soft_attention_aggregation=True,
            soft_attention_bottleneck=8,
        )
        model.train()
        points, features, lorentz, mask = _make_batch()
        scores = model(points, features, lorentz, mask)
        scores.sum().backward()
        for aggregator in model.soft_attention_aggregators:
            for param in aggregator.parameters():
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()


class TestAllThreeCombined:
    def test_forward_works_with_all_three_heads_enabled(self):
        model = _make_model(
            feature_embed_mode='per_feature', feature_embed_dim=8,
            use_feature_gate=True, feature_gate_bottleneck=8,
            use_film_head=True, film_context_dim=16,
        )
        model.eval()
        points, features, lorentz, mask = _make_batch()
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        assert scores.shape == (points.shape[0], points.shape[2])
        # Scorer emits -inf at masked positions so they never rank; only
        # valid positions must be finite.
        valid = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid]).all()

    def test_gradient_flows_through_all_heads(self):
        model = _make_model(
            feature_embed_mode='per_feature', feature_embed_dim=8,
            use_feature_gate=True, feature_gate_bottleneck=8,
            use_film_head=True, film_context_dim=16,
        )
        model.train()
        points, features, lorentz, mask = _make_batch()
        scores = model(points, features, lorentz, mask)
        scores.sum().backward()
        grad_ok = {
            'embedder': all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.feature_embedder.parameters()
            ),
            'gate': all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.feature_gate.parameters()
            ),
            'film': all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.film_head.parameters()
            ),
        }
        assert all(grad_ok.values()), grad_ok
