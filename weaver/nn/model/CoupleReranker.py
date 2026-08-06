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
    ) -> torch.Tensor:
        """ListMLE top-1 with optional label smoothing.
        For each positive p_i: L_i = (1−ε)·(−s_{p_i}/T + logsumexp(s_pool/T))
                                      + ε·(−mean(s_pool)/T + logsumexp(s_pool/T))."""
        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        eps = self.label_smoothing
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

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_positions = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=event_scores.device,
            )
            sampled_negatives = negative_indices[sample_positions]
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
