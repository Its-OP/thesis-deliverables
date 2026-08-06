"""Pins the vectorized ``CoupleReranker._softmax_ce_loss`` to the per-event
reference loop it replaced. ``reference_softmax_ce_loss`` below is that loop,
copied verbatim and extended to accept injected negative draws so both
implementations can be fed identical negatives."""
from __future__ import annotations

import pytest
import torch

from weaver.nn.model.CoupleReranker import (
    CoupleReranker,
    _sample_negative_indices,
)


def reference_softmax_ce_loss(
    scores: torch.Tensor,
    couple_labels: torch.Tensor,
    couple_mask: torch.Tensor,
    num_samples: int,
    temperature: float,
    label_smoothing: float,
    neg_idx: torch.Tensor | None = None,
    neg_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """scores, couple_labels, couple_mask: (B, C). neg_idx, neg_valid:
    optional (B, S) injected draws — when given, each event uses exactly its
    valid injected negatives instead of sampling."""
    batch_size = scores.shape[0]
    eps = label_smoothing
    event_losses: list[torch.Tensor] = []

    for event_index in range(batch_size):
        event_scores = scores[event_index]
        event_labels = couple_labels[event_index]
        event_valid = couple_mask[event_index] > 0.5

        positive_indices = (
            (event_labels > 0.5) & event_valid
        ).nonzero(as_tuple=True)[0]
        negative_indices = (
            (event_labels < 0.5) & event_valid
        ).nonzero(as_tuple=True)[0]

        if neg_idx is None:
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            num_drawn = min(num_samples, len(negative_indices))
            sample_positions = torch.randint(
                0, len(negative_indices), (num_drawn,),
                device=event_scores.device,
            )
            sampled_negatives = negative_indices[sample_positions]
        else:
            event_neg_valid = (
                neg_valid[event_index] if neg_valid is not None
                else torch.ones_like(neg_idx[event_index], dtype=torch.bool)
            )
            sampled_negatives = neg_idx[event_index][event_neg_valid]
            if len(positive_indices) == 0 or len(sampled_negatives) == 0:
                continue
        negative_scores = event_scores[sampled_negatives]

        positive_losses: list[torch.Tensor] = []
        for positive_index in positive_indices:
            positive_score = event_scores[positive_index]
            pool_scores = torch.cat(
                [positive_score.unsqueeze(0), negative_scores],
            )
            scaled_pool = pool_scores / temperature
            log_normalizer = torch.logsumexp(scaled_pool, dim=0)
            nll = -positive_score / temperature + log_normalizer
            if eps > 0.0:
                uniform_nll = -scaled_pool.mean() + log_normalizer
                positive_losses.append((1.0 - eps) * nll + eps * uniform_nll)
            else:
                positive_losses.append(nll)
        event_losses.append(torch.stack(positive_losses).mean())

    if not event_losses:
        return scores.sum() * 0.0
    return torch.stack(event_losses).mean()


def _loss_model(num_samples, temperature, label_smoothing):
    return CoupleReranker(
        hidden_dim=8, num_residual_blocks=1, couple_projector_dim=4,
        ranking_num_samples=num_samples,
        ranking_temperature=temperature,
        label_smoothing=label_smoothing,
    )


def _random_case(seed, batch_size=6, num_candidates=24, pad_tail=3):
    # Deterministic mix of event shapes: random 0-3 positives per event, one
    # narrow event (n_neg < num_samples), one zero-negative event, one
    # zero-positive event, plus an all-padded tail on every event.
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(batch_size, num_candidates, generator=generator)
    labels = torch.zeros(batch_size, num_candidates)
    mask = torch.ones(batch_size, num_candidates)
    if pad_tail:
        mask[:, -pad_tail:] = 0.0
    valid_width = num_candidates - pad_tail
    positives_per_event = torch.randint(
        0, 4, (batch_size,), generator=generator,
    )
    for event, count in enumerate(positives_per_event.tolist()):
        columns = torch.randperm(valid_width, generator=generator)[:count]
        labels[event, columns] = 1.0
    # Event 1: one positive among 5 valid slots -> 4 negatives (< num_samples).
    labels[1] = 0.0
    labels[1, 0] = 1.0
    mask[1] = 0.0
    mask[1, :5] = 1.0
    # Event 2: positives only (zero valid negatives).
    labels[2] = 0.0
    labels[2, [0, 3]] = 1.0
    mask[2] = labels[2].clone()
    # Last event: zero positives.
    labels[batch_size - 1] = 0.0
    return scores.requires_grad_(True), labels, mask


def _inject_negatives(labels, mask, num_samples, seed):
    # Mimics the sampler contract: per event, min(num_samples, n_neg) draws
    # with replacement from that event's valid negatives; the rest invalid.
    generator = torch.Generator().manual_seed(seed)
    batch_size, _ = labels.shape
    neg_idx = torch.zeros(batch_size, num_samples, dtype=torch.long)
    neg_valid = torch.zeros(batch_size, num_samples, dtype=torch.bool)
    for event in range(batch_size):
        negative_columns = (
            (labels[event] < 0.5) & (mask[event] > 0.5)
        ).nonzero(as_tuple=True)[0]
        if len(negative_columns) == 0:
            continue
        count = min(num_samples, len(negative_columns))
        draws = torch.randint(
            0, len(negative_columns), (count,), generator=generator,
        )
        neg_idx[event, :count] = negative_columns[draws]
        neg_valid[event, :count] = True
    return neg_idx, neg_valid


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
@pytest.mark.parametrize('temperature', [1.0, 2.0])
def test_injected_forward_and_grad_match_oracle(seed, label_smoothing,
                                                temperature):
    num_samples = 10
    scores_ref, labels, mask = _random_case(seed)
    scores_vec = scores_ref.detach().clone().requires_grad_(True)
    neg_idx, neg_valid = _inject_negatives(
        labels, mask, num_samples, seed + 1000,
    )
    model = _loss_model(num_samples, temperature, label_smoothing)

    reference = reference_softmax_ce_loss(
        scores_ref, labels, mask, num_samples, temperature, label_smoothing,
        neg_idx=neg_idx, neg_valid=neg_valid,
    )
    vectorized = model._softmax_ce_loss(
        scores_vec, labels, mask, neg_idx=neg_idx, neg_valid=neg_valid,
    )

    torch.testing.assert_close(vectorized, reference, rtol=1e-6, atol=1e-7)
    reference.backward()
    vectorized.backward()
    torch.testing.assert_close(
        scores_vec.grad, scores_ref.grad, rtol=1e-6, atol=1e-7,
    )


def test_degenerate_batch_returns_graph_zero():
    model = _loss_model(5, 1.0, 0.1)
    # No positives anywhere.
    scores = torch.randn(3, 8, requires_grad=True)
    loss = model._softmax_ce_loss(
        scores, torch.zeros(3, 8), torch.ones(3, 8),
    )
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert scores.grad is not None
    # Positives everywhere -> no negatives -> equally degenerate.
    scores_all_positive = torch.randn(2, 6, requires_grad=True)
    loss_all_positive = model._softmax_ce_loss(
        scores_all_positive, torch.ones(2, 6), torch.ones(2, 6),
    )
    assert loss_all_positive.item() == 0.0
    assert loss_all_positive.requires_grad


def test_sampler_uniform_over_negatives():
    torch.manual_seed(0)
    num_candidates = 40
    negative_columns = [3, 7, 11, 19, 22, 28, 31, 37]
    num_negatives = len(negative_columns)
    num_rows = 2500  # 2500 rows x 8 draws = 20k samples
    negative = torch.zeros(num_rows, num_candidates, dtype=torch.bool)
    negative[:, negative_columns] = True

    neg_idx, neg_valid = _sample_negative_indices(negative, num_negatives)
    assert neg_valid.all()
    flat = neg_idx.flatten()
    assert set(flat.tolist()) <= set(negative_columns)

    total_draws = num_rows * num_negatives
    probability = 1.0 / num_negatives
    expected = total_draws * probability
    sigma = (total_draws * probability * (1.0 - probability)) ** 0.5
    for column in negative_columns:
        count = int((flat == column).sum())
        assert abs(count - expected) <= 5.0 * sigma, (
            f'column {column}: count {count} outside 5 sigma of {expected}'
        )
    # With replacement: some row must repeat a column.
    has_duplicate_row = any(
        len(set(row.tolist())) < num_negatives for row in neg_idx
    )
    assert has_duplicate_row


def test_sampler_ragged_validity():
    torch.manual_seed(1)
    negative = torch.zeros(2, 12, dtype=torch.bool)
    negative[0, [2, 5, 9]] = True  # n_neg = 3 < num_samples
    # Event 1: zero negatives.
    neg_idx, neg_valid = _sample_negative_indices(negative, 10)
    assert neg_idx.shape == (2, 10)
    assert neg_valid[0, :3].all()
    assert not neg_valid[0, 3:].any()
    assert not neg_valid[1].any()
    assert set(neg_idx[0, :3].tolist()) <= {2, 5, 9}


def test_float_and_bool_masks_equal():
    num_samples = 6
    scores_float, labels, mask = _random_case(0)
    scores_bool = scores_float.detach().clone().requires_grad_(True)
    neg_idx, neg_valid = _inject_negatives(labels, mask, num_samples, 7)
    model = _loss_model(num_samples, 1.0, 0.1)

    loss_float = model._softmax_ce_loss(
        scores_float, labels, mask, neg_idx=neg_idx, neg_valid=neg_valid,
    )
    loss_bool = model._softmax_ce_loss(
        scores_bool, labels.bool(), mask.bool(),
        neg_idx=neg_idx, neg_valid=neg_valid,
    )
    torch.testing.assert_close(loss_bool, loss_float, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_autocast_fp16_smoke():
    model = _loss_model(20, 1.0, 0.1)
    scores = torch.randn(4, 50, device='cuda', requires_grad=True)
    labels = torch.zeros(4, 50, device='cuda')
    labels[:, [0, 1]] = 1.0
    mask = torch.ones(4, 50, device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model._softmax_ce_loss(scores.half(), labels, mask)
    assert torch.isfinite(loss)
