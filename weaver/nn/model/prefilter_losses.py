"""Supervision losses for TrackPreFilter.

All losses operate on ``(B, P)`` score tensors with paired ``(B, P)``
labels (1 = GT pion, 0 = noise) and ``(B, P)`` valid masks. Padded
tracks have ``valid_mask == False`` and contribute no gradient.

Functions
---------
- ``listwise_ce_loss`` — event-wise softmax cross-entropy (Sec. 3 / 6
  of the prefilter-updates plan). All in-event negatives act as
  contrastive partners for each positive.
- ``infonce_in_event`` — InfoNCE with each positive as anchor and the
  rest of the event as negatives. Closely related to listwise CE but
  normalizes per-positive.
- ``logit_adjust_offset`` — Menon 2007.07314's class-prior offset, added
  to negative logits during training only. Orthogonal to the pairwise
  ranking loss and to listwise CE.
- ``object_condensation_loss`` — Kieseler 2002.03605 attractive /
  repulsive potential with a β-weighted confidence head. Requires the
  model to expose a per-track embedding and β tensor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def listwise_ce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Event-wise softmax cross-entropy.

    Math (per event with at least one positive):
        log_softmax_i = log( exp(s_i / T) / Σ_j∈valid exp(s_j / T) )
        L_event = − (1 / |positives|) · Σ_{i∈positives} log_softmax_i

    The full loss is the mean over events that have at least one positive
    and one negative.

    Args:
        scores: ``(B, P)`` per-track scores.
        labels: ``(B, P)`` 0/1 labels. 1 = GT pion.
        valid_mask: ``(B, P)`` boolean validity (True = real track).
        temperature: softmax temperature. Default 1.0.

    Returns:
        scalar loss.
    """
    scores_scaled = scores / temperature
    # Mask padding so it never contributes to the softmax denominator.
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

        # log_softmax over the valid tracks, -inf for padding already set
        log_probabilities = functional.log_softmax(
            scores_scaled[event_index], dim=0,
        )  # (P,)
        event_loss = -log_probabilities[positives].mean()
        event_losses.append(event_loss)

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
    """InfoNCE with each positive as an anchor, rest of event as negatives.

    Math (per event, per positive track i):
        L_i = − log( exp(s_i / T) / (exp(s_i / T) + Σ_{j∈negatives} exp(s_j / T)) )

    Averages per-positive losses within an event, then across events.
    Unlike listwise CE, the denominator excludes other positives (only
    the anchor + negatives contribute), which sharpens the per-positive
    ranking signal.

    Args:
        scores: ``(B, P)`` per-track scores.
        labels: ``(B, P)`` labels.
        valid_mask: ``(B, P)`` boolean.
        temperature: softmax temperature.

    Returns:
        scalar loss.
    """
    batch_size = scores.shape[0]
    event_losses: list[torch.Tensor] = []
    for event_index in range(batch_size):
        event_valid = valid_mask[event_index]
        event_labels = labels[event_index]
        event_scores = scores[event_index] / temperature

        positives = ((event_labels == 1.0) & event_valid).nonzero(
            as_tuple=True,
        )[0]
        negatives = ((event_labels == 0.0) & event_valid).nonzero(
            as_tuple=True,
        )[0]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue

        negative_scores = event_scores[negatives]  # (N_neg,)
        log_negatives_sum = torch.logsumexp(negative_scores, dim=0)

        per_positive_losses: list[torch.Tensor] = []
        for positive_index in positives:
            anchor_score = event_scores[positive_index]
            # log denominator = logsumexp(anchor_score, negatives)
            # = log(exp(anchor) + Σ exp(negatives))
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
    """Menon 2007.07314 class-prior offset.

    Math:
        offset = τ · log(π_neg / π_pos)
               = τ · log(n_neg / n_pos)     (for equal sample weights)

    Added to *negative* logits (or subtracted from *positive* logits)
    during training, never at inference. Provably optimizes balanced
    error even when the class distribution is extreme.

    Args:
        num_positives: positive count (per event or aggregate).
        num_negatives: negative count.
        tau: scaling hyperparameter. Default 1.0.

    Returns:
        offset (float). Add to negative logits at training time.
    """
    if num_positives <= 0 or num_negatives <= 0:
        return 0.0
    import math
    return tau * math.log(num_negatives / num_positives)


def object_condensation_loss(
    embeddings: torch.Tensor,
    beta: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    q_min: float = 0.1,
    potential_weight: float = 1.0,
    beta_weight: float = 1.0,
) -> torch.Tensor:
    """Kieseler 2002.03605 condensation loss (single-object-per-event flavour).

    Math (per event):
        q_i       = arctanh(β_i)² + q_min                     (charge)
        α         = argmax_{i∈GT} q_i                         (condensation point)
        V_attr_i  = q_i · q_α · ‖emb_i − emb_α‖²  for i∈GT
        V_rep_i   = q_i · q_α · max(0, 1 − ‖emb_i − emb_α‖)²  for i∈noise
        L_pot     = Σ (V_attr + V_rep) / |valid|
        L_beta    = − log(max_{i∈GT} β_i) + ⟨β_i⟩_{i∈noise}
        L         = potential_weight · L_pot + beta_weight · L_beta

    Averaged across events with at least one GT pion and one noise track.

    Args:
        embeddings: ``(B, D, P)`` per-track embeddings.
        beta: ``(B, P)`` confidence in (0, 1).
        labels: ``(B, P)`` GT labels.
        valid_mask: ``(B, P)`` boolean.
        q_min: minimum charge. Default 0.1.
        potential_weight: scalar weight on the potential term.
        beta_weight: scalar weight on the β-regulariser term.

    Returns:
        scalar loss.
    """
    batch_size = embeddings.shape[0]
    event_losses: list[torch.Tensor] = []
    beta_clamped = beta.clamp(1e-6, 1.0 - 1e-6)
    q_values = torch.atanh(beta_clamped).pow(2) + q_min  # (B, P)

    for event_index in range(batch_size):
        event_valid = valid_mask[event_index]
        event_labels = labels[event_index]
        event_embeddings = embeddings[event_index]  # (D, P)
        event_beta = beta_clamped[event_index]  # (P,)
        event_q = q_values[event_index]

        positive_indices = ((event_labels == 1.0) & event_valid).nonzero(
            as_tuple=True,
        )[0]
        negative_indices = ((event_labels == 0.0) & event_valid).nonzero(
            as_tuple=True,
        )[0]
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        positive_q = event_q[positive_indices]
        alpha_index = positive_indices[positive_q.argmax()]
        alpha_embedding = event_embeddings[:, alpha_index]  # (D,)
        alpha_q = event_q[alpha_index]

        # Distances to the condensation point
        differences = event_embeddings - alpha_embedding.unsqueeze(-1)  # (D, P)
        distances_squared = differences.pow(2).sum(dim=0)  # (P,)
        distances = distances_squared.clamp_min(1e-12).sqrt()

        # Attractive for other positives (including α → 0 contribution)
        attractive = (
            event_q[positive_indices]
            * alpha_q
            * distances_squared[positive_indices]
        )

        # Repulsive for negatives
        repulsive_margin = (1.0 - distances[negative_indices]).clamp_min(0.0)
        repulsive = (
            event_q[negative_indices] * alpha_q * repulsive_margin.pow(2)
        )

        num_valid = event_valid.sum().clamp_min(1)
        potential_loss = (attractive.sum() + repulsive.sum()) / num_valid

        # Confidence β-regulariser
        beta_loss = (
            -torch.log(event_beta[positive_indices].max())
            + event_beta[negative_indices].mean()
        )

        event_losses.append(
            potential_weight * potential_loss + beta_weight * beta_loss,
        )

    if not event_losses:
        return torch.zeros(
            (), device=embeddings.device, dtype=embeddings.dtype,
            requires_grad=True,
        )
    return torch.stack(event_losses).mean()
