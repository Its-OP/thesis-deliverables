from __future__ import annotations

import pytest
import torch

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
    model = TripletReranker(input_mode='hierarchical', projector_dim=32, rest_dim=41)
    features, pos_mask, valid_mask = _batch(16 * 3 + 41)
    scores = model(features)
    assert scores.shape == (3, 12)
    out = model.compute_loss(features, pos_mask, valid_mask)
    assert torch.isfinite(out['total_loss'])


def test_hierarchical_input_dim_check():
    model = TripletReranker(input_mode='hierarchical', projector_dim=32, rest_dim=40)
    bad_features = torch.randn(2, 89, 5)  # expects 3*16 + 40 = 88 channels
    with pytest.raises(AssertionError):
        model(bad_features)


def test_loss_is_couple_loss():
    # Imported, not copied: the InfoNCE top-1 implementation is the couple stage's.
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
    triplet = TripletReranker(input_mode='hierarchical', projector_dim=32, rest_dim=41)
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
