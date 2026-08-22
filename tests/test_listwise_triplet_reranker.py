from __future__ import annotations

import torch

from weaver.nn.model.ListwiseTripletReranker import (ListwiseTripletReranker,
                                                     shared_track_counts)


def test_shared_track_counts_pairwise_overlap():
    # Candidates: (1,2,3), (1,2,4) share 2; (1,2,3), (5,6,7) share 0;
    # (1,2,4), (4,6,7) share 1. Padded slot (-1s) shares 0 with everyone.
    keys = torch.tensor([[[1, 2, 3], [1, 2, 4], [5, 6, 7], [-1, -1, -1]]])
    counts = shared_track_counts(keys)
    assert counts.shape == (1, 4, 4)
    assert counts[0, 0, 1] == 2 and counts[0, 1, 0] == 2
    assert counts[0, 0, 2] == 0
    assert counts[0, 1, 2] == 0
    assert counts[0, 0, 3] == 0 and counts[0, 3, 3] == 0
    assert counts[0, 0, 0] == 3  # self-overlap: full triple


def test_forward_shapes_and_padding_isolation():
    torch.manual_seed(0)
    model = ListwiseTripletReranker(feature_dim=16, hidden_dim=32,
                                    num_layers=2, num_heads=4)
    features = torch.randn(2, 16, 5)
    keys = torch.randint(0, 50, (2, 5, 3))
    valid_mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    filter_logit = torch.randn(2, 5)
    scores = model(features, keys=keys, valid_mask=valid_mask,
                   filter_logit=filter_logit)
    assert scores.shape == (2, 5)
    assert torch.isfinite(scores[valid_mask]).all()


def test_zero_init_fusion_reproduces_filter_ordering():
    # With the zero-initialized head the model score must be a monotone
    # transform of filter_logit: identical candidate ordering at epoch 0.
    torch.manual_seed(1)
    model = ListwiseTripletReranker(feature_dim=8, hidden_dim=16,
                                    num_layers=1, num_heads=2)
    features = torch.randn(1, 8, 6)
    keys = torch.randint(0, 30, (1, 6, 3))
    valid_mask = torch.ones(1, 6, dtype=torch.bool)
    filter_logit = torch.randn(1, 6)
    scores = model(features, keys=keys, valid_mask=valid_mask,
                   filter_logit=filter_logit)
    assert torch.equal(torch.argsort(scores[0], descending=True),
                       torch.argsort(filter_logit[0], descending=True))


def test_gradients_flow_to_attention_and_bias():
    torch.manual_seed(2)
    model = ListwiseTripletReranker(feature_dim=8, hidden_dim=16,
                                    num_layers=1, num_heads=2)
    features = torch.randn(1, 8, 4)
    keys = torch.tensor([[[1, 2, 3], [1, 2, 4], [1, 5, 6], [7, 8, 9]]])
    valid_mask = torch.ones(1, 4, dtype=torch.bool)
    filter_logit = torch.randn(1, 4)
    scores = model(features, keys=keys, valid_mask=valid_mask,
                   filter_logit=filter_logit)
    scores.sum().backward()
    assert torch.isfinite(model.overlap_bias.grad).all()
    head_weight = model.scorer_head[-1].weight
    assert head_weight.grad is not None


def test_within_couple_contrast_targets_same_couple_siblings():
    from weaver.nn.model.ListwiseTripletReranker import within_couple_contrast
    # Candidates 0,1,2 share couple 7; candidate 1 is GT. Contrast loss must
    # push GT above its couple siblings only — candidate 3 (couple 9) excluded.
    scores = torch.tensor([[2.0, 1.0, 0.5, 5.0]])
    couple_ids = torch.tensor([[7, 7, 7, 9]])
    pos_mask = torch.tensor([[False, True, False, False]])
    valid_mask = torch.ones(1, 4, dtype=torch.bool)
    loss = within_couple_contrast(scores, couple_ids, pos_mask, valid_mask)
    assert loss.item() > 0.0
    # GT far above its siblings -> loss near zero; unrelated candidate 3
    # stays high and must not contribute.
    scores_good = torch.tensor([[2.0, 10.0, 0.5, 50.0]])
    loss_good = within_couple_contrast(scores_good, couple_ids, pos_mask,
                                       valid_mask)
    assert loss_good.item() < loss.item()
    assert loss_good.item() < 1e-3
