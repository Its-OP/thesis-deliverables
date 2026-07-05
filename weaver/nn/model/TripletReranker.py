from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.CoupleReranker import (
    CoupleReranker,
    NanSafeBatchNorm1d,
    ResidualBlock,
    _TRACK_EMBED_DIM,
)

_TRACK_PREFIXES = ('ti_', 'tj_', 'tk_')


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


def _sample_negatives(
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pos_mask, neg_mask: (B, N) bool. Returns neg_idx (B, S) long, neg_valid
    (B, S) bool, contrib (B,) bool. Consumes the global RNG exactly like the
    reference loop: one randint(0, n_neg, (min(S, n_neg),)) per contributing event,
    in event order, skipping 0-pos/0-neg events before the draw."""
    batch_size = pos_mask.shape[0]
    neg_idx = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
    neg_valid = torch.zeros(batch_size, num_samples, dtype=torch.bool, device=device)
    contrib = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for event in range(batch_size):
        negative_indices = neg_mask[event].nonzero(as_tuple=True)[0]
        if not pos_mask[event].any() or len(negative_indices) == 0:
            continue
        contrib[event] = True
        count = min(num_samples, len(negative_indices))
        draw = torch.randint(0, len(negative_indices), (count,), device=device)
        neg_idx[event, :count] = negative_indices[draw]
        neg_valid[event, :count] = True
    return neg_idx, neg_valid, contrib


def sampled_softmax_ce_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_samples: int,
    temperature: float,
    label_smoothing: float,
    neg_idx: torch.Tensor | None = None,
    neg_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """scores: (B, N). pos_mask, valid_mask: (B, N) bool. neg_idx, neg_valid:
    optional (B, S) injected negatives (tests). Returns scalar loss."""
    positive = pos_mask & valid_mask
    negative = ~pos_mask & valid_mask
    if neg_idx is None:
        neg_idx, neg_valid, contrib = _sample_negatives(
            positive, negative, num_samples, scores.device)
    else:
        if neg_valid is None:
            neg_valid = torch.ones_like(neg_idx, dtype=torch.bool)
        contrib = positive.any(dim=1) & neg_valid.any(dim=1)
    if not contrib.any():
        return scores.sum() * 0.0

    slots, slot_valid = _positive_slots(positive)
    max_positives = slots.shape[1]
    positive_scores = scores.gather(1, slots)
    negative_scores = scores.gather(1, neg_idx)

    pool = torch.cat([
        positive_scores.unsqueeze(2),
        negative_scores.unsqueeze(1).expand(-1, max_positives, -1),
    ], dim=2)
    entry_valid = torch.cat([
        torch.ones_like(slot_valid).unsqueeze(2),
        neg_valid.unsqueeze(1).expand(-1, max_positives, -1),
    ], dim=2)

    scaled = pool / temperature
    log_normalizer = scaled.masked_fill(~entry_valid, float('-inf')).logsumexp(dim=2)
    nll = -positive_scores / temperature + log_normalizer
    if label_smoothing > 0.0:
        mean_scaled = (scaled * entry_valid).sum(dim=2) / entry_valid.sum(dim=2)
        per_positive = ((1.0 - label_smoothing) * nll
                        + label_smoothing * (-mean_scaled + log_normalizer))
    else:
        per_positive = nll

    row_valid = slot_valid & contrib.unsqueeze(1)
    per_event = (per_positive * row_valid).sum(dim=1) / row_valid.sum(dim=1).clamp_min(1)
    return per_event[contrib].mean()


def full_list_softmax_ce_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
    label_smoothing: float,
) -> torch.Tensor:
    """scores: (B, N). pos_mask, valid_mask: (B, N) bool. Returns scalar loss.
    Each positive's pool is itself plus ALL valid negatives — other positives are
    excluded from every denominator, so duplicate GT 3-sets never suppress each
    other."""
    positive = pos_mask & valid_mask
    negative = ~pos_mask & valid_mask
    contrib = positive.any(dim=1) & negative.any(dim=1)
    if not contrib.any():
        return scores.sum() * 0.0

    scaled = scores / temperature
    negative_lse = scaled.masked_fill(~negative, float('-inf')).logsumexp(dim=1)
    negative_sum = (scaled * negative).sum(dim=1)
    negative_count = negative.sum(dim=1)

    slots, slot_valid = _positive_slots(positive)
    positive_scaled = scores.gather(1, slots) / temperature
    log_normalizer = torch.logaddexp(positive_scaled, negative_lse.unsqueeze(1))
    nll = -positive_scaled + log_normalizer
    if label_smoothing > 0.0:
        mean_scaled = ((positive_scaled + negative_sum.unsqueeze(1))
                       / (1 + negative_count).unsqueeze(1))
        per_positive = ((1.0 - label_smoothing) * nll
                        + label_smoothing * (-mean_scaled + log_normalizer))
    else:
        per_positive = nll

    row_valid = slot_valid & contrib.unsqueeze(1)
    per_event = (per_positive * row_valid).sum(dim=1) / row_valid.sum(dim=1).clamp_min(1)
    return per_event[contrib].mean()


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
        feature_names: list[str] | None = None,
        loss_mode: str = 'sampled',
        num_attention_layers: int = 0,
        attention_heads: int = 8,
    ):
        super().__init__()
        if input_mode not in ('flat', 'hierarchical'):
            raise ValueError(f'unknown input_mode {input_mode!r}')
        if loss_mode not in ('sampled', 'full'):
            raise ValueError(f'unknown loss_mode {loss_mode!r}')
        self.input_mode = input_mode
        self.hidden_dim = hidden_dim
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.label_smoothing = label_smoothing
        self.projector_dim = projector_dim
        self.loss_mode = loss_mode
        self.num_attention_layers = num_attention_layers
        self.feature_names = list(feature_names) if feature_names is not None else None

        if input_mode == 'hierarchical':
            if self.feature_names is None:
                raise ValueError('hierarchical input_mode requires feature_names '
                                 'to locate the ti_/tj_/tk_ channel blocks')
            track_indices = {prefix: [index for index, name in enumerate(self.feature_names)
                                      if name.startswith(prefix)]
                             for prefix in _TRACK_PREFIXES}
            for prefix, indices in track_indices.items():
                if len(indices) != _TRACK_EMBED_DIM:
                    raise ValueError(f'expected {_TRACK_EMBED_DIM} {prefix}* features, '
                                     f'got {len(indices)}')
            rest_indices = [index for index, name in enumerate(self.feature_names)
                            if not name.startswith(_TRACK_PREFIXES)]
            self.register_buffer('ti_idx', torch.tensor(track_indices['ti_']),
                                 persistent=False)
            self.register_buffer('tj_idx', torch.tensor(track_indices['tj_']),
                                 persistent=False)
            self.register_buffer('tk_idx', torch.tensor(track_indices['tk_']),
                                 persistent=False)
            self.register_buffer('rest_idx', torch.tensor(rest_indices),
                                 persistent=False)
            self.rest_dim = len(rest_indices)
            self.feature_dim = len(self.feature_names)
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
            self.input_dim = 4 * projector_dim + self.rest_dim
        else:
            self.feature_dim = (len(self.feature_names)
                                if self.feature_names is not None else feature_dim)
            self.input_dim = self.feature_dim

        self.input_projection = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])
        if num_attention_layers > 0:
            self.attention_blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=attention_heads,
                    dim_feedforward=2 * hidden_dim, dropout=dropout,
                    batch_first=True, norm_first=True)
                for _ in range(num_attention_layers)
            ])
            # ReZero gates: zero-init makes the attention stack an exact identity, so
            # a trunk warm-started from a per-candidate checkpoint scores identically
            # at epoch 0 and any later gain is attributable to cross-candidate mixing.
            self.attention_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(num_attention_layers)
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
        """features: (B, len(feature_names), N). Returns (B, 4*p + rest_dim, N)."""
        assert features.shape[1] == self.feature_dim, \
            f'hierarchical input expects {self.feature_dim} channels, got {features.shape[1]}'
        project = lambda block: self.track_projector(block.transpose(1, 2)).transpose(1, 2)
        phi_i = project(features.index_select(1, self.ti_idx))
        phi_j = project(features.index_select(1, self.tj_idx))
        phi_k = project(features.index_select(1, self.tk_idx))
        rest = features.index_select(1, self.rest_idx)

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

    def _attend(self, tokens: torch.Tensor,
                key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        """tokens: (B, N, C). key_padding_mask: (B, N) bool, True = padded.
        Returns (B, N, C)."""
        for block, gate in zip(self.attention_blocks, self.attention_gates):
            tokens = tokens + gate * (
                block(tokens, src_key_padding_mask=key_padding_mask) - tokens)
        return tokens

    def forward(self, features: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """features: (B, F, N). valid_mask: optional (B, N) bool, True = real
        candidate. Returns (B, N) per-candidate scores."""
        x = self._encode(features)
        if self.num_attention_layers > 0:
            key_padding_mask = ~valid_mask.bool() if valid_mask is not None else None
            x = self._attend(x.transpose(1, 2), key_padding_mask).transpose(1, 2)
        return self.scorer(x).squeeze(1)

    def compute_loss(
        self,
        features: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """features: (B, F, N). pos_mask, valid_mask: (B, N) bool."""
        scores = self.forward(features, valid_mask=valid_mask)
        if self.loss_mode == 'full':
            ranking_loss = full_list_softmax_ce_loss(
                scores, pos_mask.bool(), valid_mask.bool(),
                temperature=self.ranking_temperature,
                label_smoothing=self.label_smoothing)
        else:
            ranking_loss = sampled_softmax_ce_loss(
                scores, pos_mask.bool(), valid_mask.bool(),
                num_samples=self.ranking_num_samples,
                temperature=self.ranking_temperature,
                label_smoothing=self.label_smoothing)
        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }

    # InfoNCE top-1 with label smoothing — the couple stage's loop implementation,
    # kept as the numerical reference the vectorized losses are pinned to.
    _softmax_ce_loss = CoupleReranker._softmax_ce_loss
