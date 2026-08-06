from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


_TRACK_EMBED_DIM = 32
# Block 2 (10 pairwise physics) + Block 3 (5 derived geom) + Block 4 (4
# cascade scores) + pair_physics_v3 (5 extra) + h6 couple block (11) = 35.
# Must equal utils/couple_features.COUPLE_REST_DIM (pinned by a test — the
# weaver package cannot import utils directly).
_REST_DIM = 35


def _positive_slots(pos_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pos_mask: (B, N) bool. Returns slots (B, P) long, slot_valid (B, P) bool."""
    batch_size, num_candidates = pos_mask.shape
    slot_range = torch.arange(num_candidates, device=pos_mask.device)
    key = torch.where(pos_mask, slot_range.expand(batch_size, -1),
                      torch.full((batch_size, num_candidates), num_candidates,
                                 device=pos_mask.device, dtype=torch.long))
    sorted_key, _ = key.sort(dim=1)
    max_positives = int(pos_mask.sum(dim=1).max()) if pos_mask.any() else 0
    slots = sorted_key[:, :max_positives]
    slot_valid = slots < num_candidates
    return slots.clamp_max(max(num_candidates - 1, 0)), slot_valid


def _sample_negative_indices(
    negative: torch.Tensor,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """negative: (B, C) bool. Returns neg_idx (B, S) long, neg_valid (B, S)
    bool with S = num_samples; per event the first min(S, n_neg) positions
    hold uniform-with-replacement draws from that event's negatives. Sync-free:
    ranks come from one rand(B, S) and map to columns via searchsorted over
    the negatives' cumulative count."""
    batch_size, num_candidates = negative.shape
    negative_count = negative.sum(dim=1)
    safe_count = negative_count.clamp(min=1)
    uniform = torch.rand(batch_size, num_samples, device=negative.device)
    draw_rank = (uniform * safe_count.unsqueeze(1)).long()
    draw_rank = torch.minimum(draw_rank, (safe_count - 1).unsqueeze(1))
    cumulative_count = negative.cumsum(dim=1)
    neg_idx = torch.searchsorted(
        cumulative_count.contiguous(), (draw_rank + 1).contiguous(),
    ).clamp(max=num_candidates - 1)
    sample_position = torch.arange(num_samples, device=negative.device)
    neg_valid = (
        sample_position.unsqueeze(0)
        < negative_count.clamp(max=num_samples).unsqueeze(1)
    )
    return neg_idx, neg_valid


class NanSafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that skips running-stat updates on non-finite inputs.
    Stat update with NaN corrupts running_mean permanently; on a non-finite
    batch we replace NaN/Inf with 0 and run an eval-mode forward instead."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(x)
        if torch.isfinite(x).all():
            return super().forward(x)
        clean_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.training = False
        output = super().forward(clean_x)
        self.training = True
        return output


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.batchnorm_1 = NanSafeBatchNorm1d(hidden_dim, track_running_stats=False)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.batchnorm_2 = NanSafeBatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv_1(x)
        out = self.batchnorm_1(out)
        out = functional.relu(out)
        out = self.dropout(out)
        out = self.conv_2(out)
        out = self.batchnorm_2(out)
        out = out + identity
        return functional.relu(out)


class CoupleReranker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
        label_smoothing: float = 0.0,
        couple_projector_dim: int = 32,
        rest_dim: int = _REST_DIM,
        track_embed_dim: int = _TRACK_EMBED_DIM,
    ):
        super().__init__()
        if couple_projector_dim <= 0:
            raise ValueError(
                f'couple_projector_dim must be > 0, got {couple_projector_dim}',
            )
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        self.dropout_rate = dropout
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.label_smoothing = label_smoothing
        self.couple_projector_dim = couple_projector_dim
        self.rest_dim = rest_dim
        self.track_embed_dim = track_embed_dim

        # projected-InferSent block-1 rebuild: φ(t) = LayerNorm(ReLU(Linear(d→p)(t)))
        # applied to each track, then assembled as [φ_i, φ_j, |φ_i−φ_j|, φ_i⊙φ_j].
        self.couple_projector = nn.Sequential(
            nn.Linear(track_embed_dim, couple_projector_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(couple_projector_dim),
        )
        self.input_dim = 4 * couple_projector_dim + rest_dim

        self.input_projection = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

        intermediate_dim = hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Conv1d(hidden_dim, intermediate_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(intermediate_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(intermediate_dim, 1, kernel_size=1),
        )

    def _rebuild_block1(self, couple_features: torch.Tensor) -> torch.Tensor:
        """couple_features: (B, F, C). Returns (B, 4*p + rest_dim, C)."""
        d = self.track_embed_dim
        expected = 2 * d + self.rest_dim
        if couple_features.shape[1] != expected:
            raise ValueError(
                f'CoupleReranker expected {expected} couple-feature channels '
                f'(2 x {d} track + {self.rest_dim} rest), got '
                f'{couple_features.shape[1]}.'
            )
        track_i = couple_features[:, :d, :]
        track_j = couple_features[:, d:2 * d, :]
        rest = couple_features[:, 2 * d:, :]

        # nn.Linear expects (..., d) — transpose to (B, C, d) and back.
        phi_i = self.couple_projector(track_i.transpose(1, 2)).transpose(1, 2)
        phi_j = self.couple_projector(track_j.transpose(1, 2)).transpose(1, 2)
        diff = (phi_i - phi_j).abs()
        prod = phi_i * phi_j
        block1 = torch.cat([phi_i, phi_j, diff, prod], dim=1)
        return torch.cat([block1, rest], dim=1)

    def _encode(self, couple_features: torch.Tensor) -> torch.Tensor:
        """couple_features: (B, F, C). Returns (B, hidden_dim, C)."""
        x = self._rebuild_block1(couple_features)
        x = self.input_projection(x)
        for block in self.residual_blocks:
            x = block(x)
        return x

    def forward(self, couple_features: torch.Tensor) -> torch.Tensor:
        """couple_features: (B, F, C). Returns (B, C) per-couple scores."""
        h = self._encode(couple_features)
        return self.scorer(h).squeeze(1)

    def compute_loss(
        self,
        couple_features: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """couple_features: (B, F, C). couple_labels: (B, C). couple_mask: (B, C)."""
        h = self._encode(couple_features)
        scores = self.scorer(h).squeeze(1)
        ranking_loss = self._softmax_ce_loss(scores, couple_labels, couple_mask)
        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }

    def _softmax_ce_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
        *,
        neg_idx: torch.Tensor | None = None,
        neg_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """scores: (B, C). couple_labels, couple_mask: (B, C) float or bool.
        neg_idx, neg_valid: optional (B, S) injected negative draws (tests) —
        when given, sampling is skipped and validity trims each pool."""
        temperature = self.ranking_temperature
        label_smoothing = self.label_smoothing
        valid = couple_mask > 0.5
        positive = (couple_labels > 0.5) & valid
        negative = (couple_labels < 0.5) & valid

        if neg_idx is None:
            neg_idx, neg_valid = _sample_negative_indices(
                negative, self.ranking_num_samples,
            )
        elif neg_valid is None:
            neg_valid = torch.ones_like(neg_idx, dtype=torch.bool)
        contributing = positive.any(dim=1) & neg_valid.any(dim=1)
        if not contributing.any():
            return scores.sum() * 0.0

        slots, slot_valid = _positive_slots(positive)
        max_positives = slots.shape[1]
        positive_scores = scores.gather(1, slots)
        negative_scores = scores.gather(1, neg_idx)

        # Per-positive pool [s_pos] ++ sampled negatives, width 1 + S; the
        # positive's own entry is always valid, so every row's logsumexp is
        # finite even for skipped events.
        pool = torch.cat([
            positive_scores.unsqueeze(2),
            negative_scores.unsqueeze(1).expand(-1, max_positives, -1),
        ], dim=2)
        entry_valid = torch.cat([
            torch.ones_like(slot_valid).unsqueeze(2),
            neg_valid.unsqueeze(1).expand(-1, max_positives, -1),
        ], dim=2)

        scaled = pool / temperature
        log_normalizer = scaled.masked_fill(
            ~entry_valid, float('-inf'),
        ).logsumexp(dim=2)
        nll = -positive_scores / temperature + log_normalizer
        if label_smoothing > 0.0:
            mean_scaled = (
                (scaled * entry_valid).sum(dim=2) / entry_valid.sum(dim=2)
            )
            per_positive = (
                (1.0 - label_smoothing) * nll
                + label_smoothing * (-mean_scaled + log_normalizer)
            )
        else:
            per_positive = nll

        row_valid = slot_valid & contributing.unsqueeze(1)
        per_event = (
            (per_positive * row_valid).sum(dim=1)
            / row_valid.sum(dim=1).clamp_min(1)
        )
        return per_event[contributing].mean()
