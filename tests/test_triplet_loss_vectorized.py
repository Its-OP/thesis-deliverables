from __future__ import annotations

import math

import pytest
import torch

from test_couple_loss_vectorized import (
    _inject_negatives,
    reference_softmax_ce_loss,
)
from weaver.nn.model.TripletReranker import (
    TripletReranker,
    full_list_softmax_ce_loss,
)


def _random_case(seed, batch_size, num_candidates, max_positives, drop_negatives=(),
                 pad_tail=0):
    # Returns scores (leaf, requires_grad), bool pos/valid masks. Events listed in
    # drop_negatives get zero valid negatives (all non-positive slots invalidated).
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(batch_size, num_candidates, generator=generator)
    pos_mask = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    valid_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)
    n_pos = torch.randint(0, max_positives + 1, (batch_size,), generator=generator)
    for event, count in enumerate(n_pos.tolist()):
        pos_mask[event, :count] = True
    if pad_tail:
        valid_mask[:, -pad_tail:] = False
        pos_mask[:, -pad_tail:] = False
    for event in drop_negatives:
        valid_mask[event] = pos_mask[event]
    return scores.requires_grad_(True), pos_mask, valid_mask


def _reference_model(num_samples, temperature, label_smoothing):
    return TripletReranker(input_mode='flat', feature_dim=4,
                           ranking_num_samples=num_samples,
                           ranking_temperature=temperature,
                           label_smoothing=label_smoothing)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
@pytest.mark.parametrize('temperature', [1.0, 2.0])
def test_sampled_matches_reference(seed, label_smoothing, temperature):
    num_samples = 7
    model = _reference_model(num_samples, temperature, label_smoothing)
    scores_ref, pos_mask, valid_mask = _random_case(
        seed, batch_size=4, num_candidates=20, max_positives=3,
        drop_negatives=(1,), pad_tail=3)
    scores_vec = scores_ref.detach().clone().requires_grad_(True)
    neg_idx, neg_valid = _inject_negatives(
        pos_mask, valid_mask, num_samples, seed + 500)

    reference = reference_softmax_ce_loss(
        scores_ref, pos_mask.float(), valid_mask.float(), num_samples,
        temperature, label_smoothing, neg_idx=neg_idx, neg_valid=neg_valid)
    vectorized = model._softmax_ce_loss(
        scores_vec, pos_mask, valid_mask, neg_idx=neg_idx, neg_valid=neg_valid)

    torch.testing.assert_close(vectorized, reference, rtol=1e-6, atol=1e-7)
    reference.backward()
    vectorized.backward()
    torch.testing.assert_close(scores_vec.grad, scores_ref.grad, rtol=1e-6, atol=1e-7)


def test_sampled_matches_reference_fewer_negatives_than_samples():
    # n_neg < num_samples exercises the min(S, n_neg) ragged pool width.
    num_samples = 50
    model = _reference_model(num_samples=num_samples, temperature=1.0,
                             label_smoothing=0.1)
    scores_ref, pos_mask, valid_mask = _random_case(
        seed=11, batch_size=3, num_candidates=6, max_positives=2)
    scores_vec = scores_ref.detach().clone().requires_grad_(True)
    neg_idx, neg_valid = _inject_negatives(pos_mask, valid_mask, num_samples, 21)
    assert bool((neg_valid.sum(dim=1) < num_samples).all())

    reference = reference_softmax_ce_loss(
        scores_ref, pos_mask.float(), valid_mask.float(), num_samples,
        temperature=1.0, label_smoothing=0.1,
        neg_idx=neg_idx, neg_valid=neg_valid)
    vectorized = model._softmax_ce_loss(
        scores_vec, pos_mask, valid_mask, neg_idx=neg_idx, neg_valid=neg_valid)

    torch.testing.assert_close(vectorized, reference, rtol=1e-6, atol=1e-7)
    reference.backward()
    vectorized.backward()
    torch.testing.assert_close(scores_vec.grad, scores_ref.grad, rtol=1e-6, atol=1e-7)


def test_sampled_injected_negatives_manual():
    # One event, one positive (slot 0), negatives injected as slots [2, 3]:
    # pool = [s0, s2, s3], nll = -s0 + lse(pool); eps=0 keeps it bare.
    model = _reference_model(num_samples=2, temperature=1.0, label_smoothing=0.0)
    scores = torch.tensor([[1.0, 9.0, -1.0, 0.5]], requires_grad=True)
    pos_mask = torch.tensor([[True, False, False, False]])
    valid_mask = torch.tensor([[True, False, True, True]])
    neg_idx = torch.tensor([[2, 3]])
    loss = model._softmax_ce_loss(scores, pos_mask, valid_mask, neg_idx=neg_idx)
    pool = [1.0, -1.0, 0.5]
    expected = -1.0 + math.log(sum(math.exp(value) for value in pool))
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert scores.grad[0, 1].item() == 0.0  # invalid slot never contributes


def test_full_matches_double_loop():
    scores, pos_mask, valid_mask = _random_case(
        seed=3, batch_size=4, num_candidates=15, max_positives=3,
        drop_negatives=(2,), pad_tail=2)
    temperature, eps = 2.0, 0.1
    loss = full_list_softmax_ce_loss(
        scores, pos_mask, valid_mask, temperature=temperature, label_smoothing=eps)

    event_losses = []
    for event in range(scores.shape[0]):
        positives = (pos_mask[event] & valid_mask[event]).nonzero(as_tuple=True)[0]
        negatives = (~pos_mask[event] & valid_mask[event]).nonzero(as_tuple=True)[0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        per_positive = []
        for position in positives:
            pool = torch.cat([scores[event, position].unsqueeze(0),
                              scores[event, negatives]]).detach()
            scaled = pool / temperature
            lse = torch.logsumexp(scaled, dim=0)
            nll = -scores[event, position].detach() / temperature + lse
            per_positive.append((1 - eps) * nll + eps * (-scaled.mean() + lse))
        event_losses.append(torch.stack(per_positive).mean())
    expected = torch.stack(event_losses).mean()
    torch.testing.assert_close(loss.detach(), expected, rtol=1e-6, atol=1e-7)


def test_full_duplicate_positives_do_not_suppress_each_other():
    # Two positives; each denominator holds only its own score + the negatives,
    # so both positives get an identical loss when their scores are equal.
    scores = torch.tensor([[2.0, 2.0, 0.0, -1.0]])
    pos_mask = torch.tensor([[True, True, False, False]])
    valid_mask = torch.ones(1, 4, dtype=torch.bool)
    loss = full_list_softmax_ce_loss(
        scores, pos_mask, valid_mask, temperature=1.0, label_smoothing=0.0)
    negatives = torch.tensor([0.0, -1.0])
    pool = torch.cat([torch.tensor([2.0]), negatives])
    expected = (-2.0 + torch.logsumexp(pool, dim=0)).item()
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_padding_invariance():
    scores, pos_mask, valid_mask = _random_case(
        seed=7, batch_size=2, num_candidates=10, max_positives=2)
    padded_scores = torch.cat(
        [scores.detach(), torch.full((2, 4), 123.0)], dim=1).requires_grad_(True)
    padded_pos = torch.cat([pos_mask, torch.zeros(2, 4, dtype=torch.bool)], dim=1)
    padded_valid = torch.cat([valid_mask, torch.zeros(2, 4, dtype=torch.bool)], dim=1)

    full_a = full_list_softmax_ce_loss(
        scores, pos_mask, valid_mask, temperature=1.0, label_smoothing=0.1)
    full_b = full_list_softmax_ce_loss(
        padded_scores, padded_pos, padded_valid, temperature=1.0, label_smoothing=0.1)
    torch.testing.assert_close(full_a.detach(), full_b.detach(), rtol=1e-6, atol=1e-7)

    # The sampler draws rand(B, S) and ranks over each event's negatives, so
    # padded-only columns change neither the draw stream nor the mapping.
    model = _reference_model(num_samples=5, temperature=1.0, label_smoothing=0.1)
    torch.manual_seed(0)
    sampled_a = model._softmax_ce_loss(scores, pos_mask, valid_mask)
    torch.manual_seed(0)
    sampled_b = model._softmax_ce_loss(padded_scores, padded_pos, padded_valid)
    torch.testing.assert_close(sampled_a.detach(), sampled_b.detach(),
                               rtol=1e-6, atol=1e-7)


def test_no_contributing_events_returns_graph_zero():
    model = _reference_model(num_samples=5, temperature=1.0, label_smoothing=0.1)
    scores = torch.randn(2, 6, requires_grad=True)
    pos_mask = torch.zeros(2, 6, dtype=torch.bool)
    valid_mask = torch.ones(2, 6, dtype=torch.bool)
    for loss in (
        model._softmax_ce_loss(scores, pos_mask, valid_mask),
        full_list_softmax_ce_loss(scores, pos_mask, valid_mask,
                                  temperature=1.0, label_smoothing=0.1),
    ):
        assert loss.item() == 0.0
        assert loss.requires_grad


def test_compute_loss_dispatch():
    features = torch.randn(2, 8, 10)
    pos_mask = torch.zeros(2, 10, dtype=torch.bool)
    pos_mask[:, 0] = True
    valid_mask = torch.ones(2, 10, dtype=torch.bool)

    sampled_model = TripletReranker(input_mode='flat', feature_dim=8,
                                    loss_mode='sampled')
    full_model = TripletReranker(input_mode='flat', feature_dim=8, loss_mode='full')
    full_model.load_state_dict(sampled_model.state_dict())

    sampled_out = sampled_model.compute_loss(features, pos_mask, valid_mask)
    full_out = full_model.compute_loss(features, pos_mask, valid_mask)
    for out in (sampled_out, full_out):
        assert set(out) == {'total_loss', 'ranking_loss', '_scores'}
        assert torch.isfinite(out['total_loss'])
    out = full_out['total_loss']
    out.backward()

    with pytest.raises(ValueError):
        TripletReranker(input_mode='flat', feature_dim=8, loss_mode='bogus')
