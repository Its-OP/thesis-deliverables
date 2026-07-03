from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.CoupleReranker import (
    CoupleReranker,
    NanSafeBatchNorm1d,
    ResidualBlock,
    _TRACK_EMBED_DIM,
)

# ti/tj/tk standardized 16-blocks followed by the rest block (triplet geometry +
# couple-unit physics + cascade-context ranks). See utils/triplet_rank_data.py.
_HIERARCHICAL_TRACK_BLOCKS = 3


class TripletReranker(nn.Module):
    def __init__(
        self,
        *,
        input_mode: str = 'flat',
        feature_dim: int = 89,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
        label_smoothing: float = 0.10,
        projector_dim: int = 32,
        rest_dim: int = 41,
    ):
        super().__init__()
        if input_mode not in ('flat', 'hierarchical'):
            raise ValueError(f'unknown input_mode {input_mode!r}')
        self.input_mode = input_mode
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.label_smoothing = label_smoothing
        self.projector_dim = projector_dim
        self.rest_dim = rest_dim

        if input_mode == 'hierarchical':
            # φ(t) = LayerNorm(ReLU(Linear(16→p)(t))) shared over i, j, k — same block as
            # CoupleReranker.couple_projector, so it can warm-start from its state dict.
            self.track_projector = nn.Sequential(
                nn.Linear(_TRACK_EMBED_DIM, projector_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(projector_dim),
            )
            # g compresses the couple assembly [φ_i, φ_j, |φ_i−φ_j|, φ_i⊙φ_j] to one
            # couple-unit embedding; the candidate head re-runs the InferSent assembly
            # on (g_c, φ_k) so the third track stays distinguished.
            self.couple_encoder = nn.Sequential(
                nn.Linear(4 * projector_dim, projector_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(projector_dim),
            )
            self.input_dim = 4 * projector_dim + rest_dim
        else:
            self.input_dim = feature_dim

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

    def _assemble_hierarchical(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, 48 + rest_dim, N). Returns (B, 4*p + rest_dim, N)."""
        d = _TRACK_EMBED_DIM
        expected = _HIERARCHICAL_TRACK_BLOCKS * d + self.rest_dim
        assert features.shape[1] == expected, \
            f'hierarchical input expects {expected} channels, got {features.shape[1]}'
        project = lambda block: self.track_projector(block.transpose(1, 2)).transpose(1, 2)
        phi_i = project(features[:, :d, :])
        phi_j = project(features[:, d:2 * d, :])
        phi_k = project(features[:, 2 * d:3 * d, :])
        rest = features[:, 3 * d:, :]

        couple_assembly = torch.cat(
            [phi_i, phi_j, (phi_i - phi_j).abs(), phi_i * phi_j], dim=1)
        g_c = self.couple_encoder(couple_assembly.transpose(1, 2)).transpose(1, 2)
        head = torch.cat([g_c, phi_k, (g_c - phi_k).abs(), g_c * phi_k], dim=1)
        return torch.cat([head, rest], dim=1)

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        if self.input_mode == 'hierarchical':
            features = self._assemble_hierarchical(features)
        x = self.input_projection(features)
        for block in self.residual_blocks:
            x = block(x)
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, F, N). Returns (B, N) per-candidate scores."""
        return self.scorer(self._encode(features)).squeeze(1)

    def compute_loss(
        self,
        features: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """features: (B, F, N). pos_mask, valid_mask: (B, N) bool."""
        scores = self.forward(features)
        ranking_loss = self._softmax_ce_loss(scores, pos_mask.float(), valid_mask.float())
        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }

    # InfoNCE top-1 with label smoothing — the couple stage's implementation, reused.
    _softmax_ce_loss = CoupleReranker._softmax_ce_loss
