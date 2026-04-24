from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def listwise_ce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """scores: (B, P). labels: (B, P) 0/1. valid_mask: (B, P) bool."""
    scores_scaled = scores / temperature
    scores_scaled = scores_scaled.masked_fill(~valid_mask, float('-inf'))

    batch_size = scores_scaled.shape[0]
    event_losses: list[torch.Tensor] = []
    for event_index in range(batch_size):
        event_valid = valid_mask[event_index]
        event_labels = labels[event_index]
        positives = (event_labels == 1.0) & event_valid
        has_negative = ((event_labels == 0.0) & event_valid).any()
        if not positives.any() or not has_negative:
            continue

        log_probabilities = functional.log_softmax(
            scores_scaled[event_index], dim=0,
        )
        event_losses.append(-log_probabilities[positives].mean())

    if not event_losses:
        return torch.zeros(
            (), device=scores.device, dtype=scores.dtype, requires_grad=True,
        )
    return torch.stack(event_losses).mean()


def infonce_in_event(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """scores: (B, P). labels: (B, P) 0/1. valid_mask: (B, P) bool."""
    batch_size = scores.shape[0]
    event_losses: list[torch.Tensor] = []
    for event_index in range(batch_size):
        event_valid = valid_mask[event_index]
        event_labels = labels[event_index]
        event_scores = scores[event_index] / temperature

        positives = ((event_labels == 1.0) & event_valid).nonzero(as_tuple=True)[0]
        negatives = ((event_labels == 0.0) & event_valid).nonzero(as_tuple=True)[0]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue

        log_negatives_sum = torch.logsumexp(event_scores[negatives], dim=0)
        per_positive_losses: list[torch.Tensor] = []
        for positive_index in positives:
            anchor_score = event_scores[positive_index]
            log_denominator = torch.logaddexp(anchor_score, log_negatives_sum)
            per_positive_losses.append(log_denominator - anchor_score)
        event_losses.append(torch.stack(per_positive_losses).mean())

    if not event_losses:
        return torch.zeros(
            (), device=scores.device, dtype=scores.dtype, requires_grad=True,
        )
    return torch.stack(event_losses).mean()


def logit_adjust_offset(
    num_positives: int | float,
    num_negatives: int | float,
    tau: float = 1.0,
) -> float:
    if num_positives <= 0 or num_negatives <= 0:
        return 0.0
    return tau * math.log(num_negatives / num_positives)
