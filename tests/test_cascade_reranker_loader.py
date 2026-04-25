"""Tests for the Stage-1 auto-detection helper used by the cascade
reranker wrapper. Verifies state-dict → kwargs round-trip across every
checkpoint shape the lineage has produced: pre-P1 anchor, P1 default,
edges on/off, dropout on/off, varying hidden_dim + rounds.
"""
import pytest
import torch

from weaver.nn.model.TrackPreFilter import TrackPreFilter

from networks.lowpt_tau_CascadeReranker import infer_stage1_kwargs


def _build_and_extract(**overrides):
    """Build a TrackPreFilter with ``overrides`` then return its state dict."""
    base = dict(
        mode='mlp',
        input_dim=16,
        hidden_dim=64,
        num_message_rounds=2,
        num_neighbors=16,
        use_edge_features=True,
        dropout=0.0,
        feature_embed_mode='none',
    )
    base.update(overrides)
    model = TrackPreFilter(**base)
    return model.state_dict(), base


class TestInferStage1Kwargs:
    def test_pre_p1_anchor_roundtrips(self):
        state, original = _build_and_extract(feature_embed_mode='none')
        inferred = infer_stage1_kwargs(state, stage1_num_neighbors=16)
        assert inferred['feature_embed_mode'] == 'none'
        assert inferred['input_dim'] == 16
        assert inferred['hidden_dim'] == original['hidden_dim']
        assert inferred['num_message_rounds'] == original['num_message_rounds']
        assert inferred['use_edge_features'] is True
        # Load back into a fresh instance — shape mismatch would raise.
        TrackPreFilter(**inferred).load_state_dict(state)

    def test_p1_default_roundtrips(self):
        state, _ = _build_and_extract(
            feature_embed_mode='per_feature', feature_embed_dim=32,
        )
        inferred = infer_stage1_kwargs(state)
        assert inferred['feature_embed_mode'] == 'per_feature'
        assert inferred['feature_embed_dim'] == 32
        assert inferred['input_dim'] == 16
        TrackPreFilter(**inferred).load_state_dict(state)

    def test_p1_nonstandard_embed_dim(self):
        state, _ = _build_and_extract(
            feature_embed_mode='per_feature', feature_embed_dim=8,
        )
        inferred = infer_stage1_kwargs(state)
        assert inferred['feature_embed_dim'] == 8
        assert inferred['input_dim'] == 16
        TrackPreFilter(**inferred).load_state_dict(state)

    def test_edges_off_detected(self):
        state, _ = _build_and_extract(use_edge_features=False)
        inferred = infer_stage1_kwargs(state)
        assert inferred['use_edge_features'] is False

    def test_dropout_on_detected(self):
        state, _ = _build_and_extract(dropout=0.1)
        inferred = infer_stage1_kwargs(state)
        assert inferred['dropout'] == 0.1
        # Shape match is the real gate — dropout>0 shifts Sequential indices.
        TrackPreFilter(**inferred).load_state_dict(state)

    def test_hidden_dim_variants(self):
        for hidden in (64, 192, 256):
            state, _ = _build_and_extract(hidden_dim=hidden)
            inferred = infer_stage1_kwargs(state)
            assert inferred['hidden_dim'] == hidden

    def test_rounds_variants(self):
        for rounds in (1, 2, 3, 4):
            state, _ = _build_and_extract(num_message_rounds=rounds)
            inferred = infer_stage1_kwargs(state)
            assert inferred['num_message_rounds'] == rounds

    def test_num_neighbors_is_pass_through(self):
        state, _ = _build_and_extract()
        for k in (8, 16, 32):
            inferred = infer_stage1_kwargs(state, stage1_num_neighbors=k)
            assert inferred['num_neighbors'] == k

    def test_missing_track_mlp_raises(self):
        with pytest.raises(ValueError, match='track_mlp.0.weight'):
            infer_stage1_kwargs({}, stage1_num_neighbors=16)

    def test_full_combo_p1_edges_dropout(self):
        state, _ = _build_and_extract(
            hidden_dim=256,
            num_message_rounds=3,
            use_edge_features=True,
            dropout=0.1,
            feature_embed_mode='per_feature',
            feature_embed_dim=32,
        )
        inferred = infer_stage1_kwargs(state)
        assert inferred['hidden_dim'] == 256
        assert inferred['num_message_rounds'] == 3
        assert inferred['use_edge_features'] is True
        assert inferred['dropout'] == 0.1
        assert inferred['feature_embed_mode'] == 'per_feature'
        assert inferred['feature_embed_dim'] == 32
        TrackPreFilter(**inferred).load_state_dict(state)
