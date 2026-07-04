from __future__ import annotations

import pytest
import torch

from utils.triplet_join import FEATURE_NAMES
from weaver.nn.model.CoupleReranker import CoupleReranker, NanSafeBatchNorm1d
from weaver.nn.model.TripletReranker import TripletReranker


def _batch(feature_dim, batch_size=3, num_candidates=12, num_positives=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch_size, feature_dim, num_candidates, generator=generator)
    pos_mask = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    pos_mask[:, :num_positives] = True
    valid_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)
    valid_mask[:, -2:] = False
    return features, pos_mask, valid_mask


def test_flat_forward_shape_and_grad():
    model = TripletReranker(input_mode='flat', feature_dim=89)
    features, pos_mask, valid_mask = _batch(89)
    scores = model(features)
    assert scores.shape == (3, 12)
    out = model.compute_loss(features, pos_mask, valid_mask)
    assert torch.isfinite(out['total_loss'])
    out['total_loss'].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_hierarchical_forward_shape():
    model = TripletReranker(input_mode='hierarchical', projector_dim=32,
                            feature_names=FEATURE_NAMES)
    features, pos_mask, valid_mask = _batch(len(FEATURE_NAMES))
    scores = model(features)
    assert scores.shape == (3, 12)
    out = model.compute_loss(features, pos_mask, valid_mask)
    assert torch.isfinite(out['total_loss'])


def test_hierarchical_requires_feature_names():
    with pytest.raises(ValueError):
        TripletReranker(input_mode='hierarchical', projector_dim=32)


def test_hierarchical_input_dim_check():
    model = TripletReranker(input_mode='hierarchical', projector_dim=32,
                            feature_names=FEATURE_NAMES)
    bad_features = torch.randn(2, len(FEATURE_NAMES) - 1, 5)
    with pytest.raises(AssertionError):
        model(bad_features)


def test_hierarchical_channel_order():
    # φ_k must be the projector applied to the tk_-prefixed channels, wherever they
    # sit in FEATURE_NAMES ([RICH, ti, tj, tk, couple-unit] — NOT track-blocks-first).
    model = TripletReranker(input_mode='hierarchical', projector_dim=32,
                            feature_names=FEATURE_NAMES)
    model.eval()
    features, _, _ = _batch(len(FEATURE_NAMES))
    tk_idx = torch.tensor([index for index, name in enumerate(FEATURE_NAMES)
                           if name.startswith('tk_')])
    assert len(tk_idx) == 16
    with torch.no_grad():
        assembled = model._assemble_hierarchical(features)
        tk_block = features.index_select(1, tk_idx)
        expected_phi_k = model.track_projector(
            tk_block.transpose(1, 2)).transpose(1, 2)
    p = model.projector_dim
    torch.testing.assert_close(assembled[:, p:2 * p, :], expected_phi_k)
    # rest block = every non-track channel, in order, after the 4p-dim head.
    rest_idx = torch.tensor([index for index, name in enumerate(FEATURE_NAMES)
                             if not name.startswith(('ti_', 'tj_', 'tk_'))])
    torch.testing.assert_close(assembled[:, 4 * p:, :],
                               features.index_select(1, rest_idx))


def test_flat_feature_dim_from_names():
    names = FEATURE_NAMES + ['gbdt6_score', 'gbdt8_score']
    model = TripletReranker(input_mode='flat', feature_names=names)
    features, pos_mask, valid_mask = _batch(len(names))
    assert model(features).shape == (3, 12)


def test_loss_is_couple_loss():
    # Imported, not copied: the InfoNCE top-1 reference implementation is the couple
    # stage's; the vectorized losses are pinned to it by test_triplet_loss_vectorized.
    assert TripletReranker._softmax_ce_loss is CoupleReranker._softmax_ce_loss


def test_loss_no_positives_returns_zero():
    model = TripletReranker(input_mode='flat', feature_dim=10)
    features = torch.randn(2, 10, 6)
    pos_mask = torch.zeros(2, 6, dtype=torch.bool)
    valid_mask = torch.ones(2, 6, dtype=torch.bool)
    out = model.compute_loss(features, pos_mask, valid_mask)
    assert out['total_loss'].item() == 0.0


def test_track_projector_warm_start_from_couple_checkpoint():
    couple = CoupleReranker(couple_projector_dim=32)
    triplet = TripletReranker(input_mode='hierarchical', projector_dim=32,
                              feature_names=FEATURE_NAMES)
    triplet.track_projector.load_state_dict(couple.couple_projector.state_dict())
    for ours, theirs in zip(triplet.track_projector.parameters(),
                            couple.couple_projector.parameters()):
        assert torch.equal(ours, theirs)


def test_uses_nan_safe_batchnorm():
    model = TripletReranker(input_mode='flat', feature_dim=89)
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert bn_layers
    assert all(isinstance(m, NanSafeBatchNorm1d) for m in bn_layers)
    assert all(not m.track_running_stats for m in bn_layers)
