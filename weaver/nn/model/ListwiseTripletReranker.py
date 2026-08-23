from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MAX_SHARED = 3


def shared_track_counts(keys: torch.Tensor,
                        block: int = 1024) -> torch.Tensor:
    """keys: (B, N, 3) track-index triples, -1 on padded slots.
    Returns (B, N, N) pairwise shared-track counts in [0, 3]. Computed in
    row blocks: the (B, N, N, 3, 3) comparison tensor never materializes."""
    batch, length, _ = keys.shape
    counts = keys.new_zeros(batch, length, length, dtype=torch.uint8)
    right = keys.unsqueeze(1).unsqueeze(-2)
    for start in range(0, length, block):
        chunk = keys[:, start:start + block]
        matches = (chunk.unsqueeze(2).unsqueeze(-1) == right).any(dim=-1)
        matches = matches & (chunk.unsqueeze(2) >= 0)
        counts[:, start:start + block] = matches.sum(dim=-1).to(torch.uint8)
    return counts


def within_couple_contrast(scores: torch.Tensor, couple_ids: torch.Tensor,
                           pos_mask: torch.Tensor,
                           valid_mask: torch.Tensor) -> torch.Tensor:
    """scores, couple_ids, pos_mask, valid_mask: (B, N). Returns a scalar:
    mean softplus margin of same-couple siblings over the GT score."""
    losses = []
    for b in range(scores.shape[0]):
        gt_slots = pos_mask[b] & valid_mask[b]
        if not gt_slots.any():
            continue
        gt_index = int(gt_slots.nonzero()[0])
        siblings = (valid_mask[b] & ~pos_mask[b]
                    & (couple_ids[b] == couple_ids[b, gt_index]))
        if not siblings.any():
            continue
        losses.append(F.softplus(scores[b][siblings]
                                 - scores[b, gt_index]).mean())
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


class _BiasedAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm_feedforward = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N, H); bias: (B, heads, N, N); padding_mask: (B, N)
        True on valid slots. Returns (B, N, H)."""
        batch, length, _ = hidden.shape
        normed = self.norm_attention(hidden)
        qkv = self.qkv(normed).reshape(batch, length, 3, self.num_heads,
                                       self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        # SDPA with the overlap bias folded into an additive float mask keeps
        # peak memory linear in N (no materialized N x N softmax input).
        mask = bias.masked_fill(~padding_mask.unsqueeze(1).unsqueeze(2),
                                float('-inf'))
        merged = F.scaled_dot_product_attention(query, key, value,
                                                attn_mask=mask)
        # Fully-padded query rows attend over -inf only; zero them instead of
        # letting NaN propagate into the residual stream.
        merged = torch.nan_to_num(merged, nan=0.0)
        merged = merged.transpose(1, 2).reshape(batch, length, -1)
        hidden = hidden + self.dropout(self.projection(merged))
        return hidden + self.dropout(self.feedforward(
            self.norm_feedforward(hidden)))


class ListwiseTripletReranker(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU())
        self.overlap_bias = nn.Parameter(
            torch.zeros(num_heads, _MAX_SHARED + 1))
        self.blocks = nn.ModuleList([
            _BiasedAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)])
        self.fusion_alpha = nn.Parameter(torch.ones(()))
        self.scorer_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        # Zero-initialized residual: epoch 0 scores are exactly
        # fusion_alpha * filter_logit — the gate ordering, asserted at launch.
        nn.init.zeros_(self.scorer_head[-1].weight)
        nn.init.zeros_(self.scorer_head[-1].bias)

    def forward(self, features: torch.Tensor, *, keys: torch.Tensor,
                valid_mask: torch.Tensor,
                filter_logit: torch.Tensor) -> torch.Tensor:
        """features: (B, F, N); keys: (B, N, 3); valid_mask, filter_logit:
        (B, N). Returns (B, N) scores."""
        hidden = self.input_projection(features.transpose(1, 2))
        counts = shared_track_counts(keys).long().clamp(0, _MAX_SHARED)
        bias = self.overlap_bias[:, counts].permute(1, 0, 2, 3)
        for block in self.blocks:
            hidden = block(hidden, bias, valid_mask)
        residual = self.scorer_head(hidden).squeeze(-1)
        return self.fusion_alpha * filter_logit + residual
