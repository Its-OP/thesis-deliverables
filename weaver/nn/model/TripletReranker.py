from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.CoupleReranker import (
    CoupleReranker,
    NanSafeBatchNorm1d,
    ResidualBlock,
    _positive_slots,
)

# Stage-4 candidate tables carry the frozen legacy 16-wide ti_/tj_/tk_
# per-track blocks; the couple stage has since widened to 32, so this
# constant is local and intentionally NOT shared with CoupleReranker.
_TRACK_EMBED_DIM = 16

_TRACK_PREFIXES = ('ti_', 'tj_', 'tk_')


def full_list_softmax_ce_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
    label_smoothing: float,
    log_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """scores: (B, N). pos_mask, valid_mask: (B, N) bool. Returns scalar loss.
    Each positive's pool is itself plus ALL valid negatives — other positives are
    excluded from every denominator, so duplicate GT 3-sets never suppress each
    other. log_weights (B, N), zero by default, multiply each NEGATIVE's
    contribution by exp(log_weight): a stored tail sample with
    log(n_tail_total / n_tail_stored) becomes an unbiased stand-in for the
    full serving list."""
    positive = pos_mask & valid_mask
    negative = ~pos_mask & valid_mask
    contrib = positive.any(dim=1) & negative.any(dim=1)
    if not contrib.any():
        return scores.sum() * 0.0

    scaled = scores / temperature
    if log_weights is None:
        log_weights = torch.zeros_like(scaled)
    weights = log_weights.exp()
    negative_lse = (scaled + log_weights).masked_fill(
        ~negative, float('-inf')).logsumexp(dim=1)
    negative_sum = (scaled * weights * negative).sum(dim=1)
    negative_count = (weights * negative).sum(dim=1)

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


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of (B, C, N) tensors: every candidate's
    score becomes a pure function of its own channels — invariant to batch
    composition, padding and the gate, unlike batch statistics."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


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
        track_embed_dim: int = _TRACK_EMBED_DIM,
        trunk_norm: str = 'batch',
        fusion: bool = False,
        aux_from_b_weight: float = 0.0,
        vertex_fit_layer: bool = False,
        fit_norm_stats: dict | None = None,
    ):
        super().__init__()
        if input_mode not in ('flat', 'hierarchical'):
            raise ValueError(f'unknown input_mode {input_mode!r}')
        if loss_mode not in ('sampled', 'full'):
            raise ValueError(f'unknown loss_mode {loss_mode!r}')
        if trunk_norm not in ('batch', 'layer'):
            raise ValueError(f'unknown trunk_norm {trunk_norm!r}')
        self.input_mode = input_mode
        self.hidden_dim = hidden_dim
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.label_smoothing = label_smoothing
        self.projector_dim = projector_dim
        self.loss_mode = loss_mode
        self.num_attention_layers = num_attention_layers
        self.track_embed_dim = track_embed_dim
        self.trunk_norm = trunk_norm
        self.fusion = fusion
        self.aux_from_b_weight = aux_from_b_weight
        self.feature_names = list(feature_names) if feature_names is not None else None

        if input_mode == 'hierarchical':
            if self.feature_names is None:
                raise ValueError('hierarchical input_mode requires feature_names '
                                 'to locate the ti_/tj_/tk_ channel blocks')
            track_indices = {prefix: [index for index, name in enumerate(self.feature_names)
                                      if name.startswith(prefix)]
                             for prefix in _TRACK_PREFIXES}
            for prefix, indices in track_indices.items():
                if len(indices) != track_embed_dim:
                    raise ValueError(f'expected {track_embed_dim} {prefix}* features, '
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
            # φ(t) = LayerNorm(ReLU(Linear(d→p)(t))) shared over i, j, k — same block as
            # CoupleReranker.couple_projector, so it can warm-start from its state dict.
            self.track_projector = nn.Sequential(
                nn.Linear(track_embed_dim, projector_dim),
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

        # The differentiable fit layer contributes its channels after the
        # hierarchical assembly (they are candidate-level "rest" channels).
        self.vertex_fit_layer = None
        if vertex_fit_layer:
            from weaver.nn.model.VertexFit import FIT_NAMES, VertexFitLayer
            if fit_norm_stats is None:
                raise ValueError('vertex_fit_layer requires fit_norm_stats '
                                 'for the FIT_NAMES channels')
            self.vertex_fit_layer = VertexFitLayer()
            self.register_buffer('fit_log1p', torch.tensor(
                [bool(fit_norm_stats[name]['log1p']) for name in FIT_NAMES]))
            self.register_buffer('fit_center', torch.tensor(
                [float(fit_norm_stats[name]['center']) for name in FIT_NAMES]))
            self.register_buffer('fit_scale', torch.tensor(
                [float(fit_norm_stats[name]['scale']) for name in FIT_NAMES]))
            self.input_dim += len(FIT_NAMES)

        def _norm(dim):
            return (ChannelLayerNorm(dim) if trunk_norm == 'layer'
                    else NanSafeBatchNorm1d(dim, track_running_stats=False))

        self.input_projection = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1, bias=False),
            _norm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])
        if trunk_norm == 'layer':
            # ResidualBlock hardcodes batch statistics; swap them so the whole
            # trunk is batch-invariant, not just the projection.
            for block in self.residual_blocks:
                block.batchnorm_1 = ChannelLayerNorm(hidden_dim)
                block.batchnorm_2 = ChannelLayerNorm(hidden_dim)
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
            _norm(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(intermediate_dim, 1, kernel_size=1),
        )
        if fusion:
            # Residual-on-filter: with the last scorer layer zero-initialized,
            # epoch 0 reproduces the filter ordering exactly and the trunk
            # learns a correction rather than the whole problem.
            self.fusion_alpha = nn.Parameter(torch.ones(1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
        if aux_from_b_weight > 0.0:
            self.aux_from_b_head = nn.Sequential(
                nn.Conv1d(hidden_dim, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 4, kernel_size=1),
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

    def _fit_channels(self, fit_inputs: dict) -> torch.Tensor:
        channels = self.vertex_fit_layer(
            reference=fit_inputs['fit_reference'], eta=fit_inputs['fit_eta'],
            phi=fit_inputs['fit_phi'], var_dxy=fit_inputs['fit_var_dxy'],
            var_dsz=fit_inputs['fit_var_dsz'],
            primary_vertex=fit_inputs['primary_vertex'],
            momentum=fit_inputs['fit_momentum'], mass=fit_inputs['fit_mass'],
            quality=fit_inputs['fit_quality'])
        log1p = self.fit_log1p.view(1, -1, 1)
        center = self.fit_center.view(1, -1, 1)
        scale = self.fit_scale.view(1, -1, 1)
        transformed = torch.where(
            log1p, torch.sign(channels) * torch.log1p(channels.abs()), channels)
        standardized = torch.nan_to_num(
            torch.clamp((transformed - center) / scale, -10.0, 10.0), nan=0.0)
        # Padded and degenerate candidates sit on non-differentiable corners of
        # the fit (zero norms, parallel lines); their upstream gradient is zero
        # but 0 x inf local derivatives poison shared weight grads as NaN.
        # Those gradients are exactly zero in the limit — sanitize them.
        if standardized.requires_grad:
            standardized.register_hook(
                lambda gradient: torch.nan_to_num(gradient, nan=0.0,
                                                  posinf=0.0, neginf=0.0))
        return standardized

    def _encode(self, features: torch.Tensor,
                fit_inputs: dict | None = None) -> torch.Tensor:
        if self.input_mode == 'hierarchical':
            features = self._assemble_hierarchical(features)
        if self.vertex_fit_layer is not None:
            if fit_inputs is None:
                raise ValueError('vertex_fit_layer requires fit_inputs')
            features = torch.cat([features, self._fit_channels(fit_inputs)],
                                 dim=1)
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
                valid_mask: torch.Tensor | None = None,
                filter_logit: torch.Tensor | None = None,
                fit_inputs: dict | None = None) -> torch.Tensor:
        """features: (B, F, N). valid_mask: optional (B, N) bool, True = real
        candidate. filter_logit: (B, N) raw filter logits, required under
        fusion. Returns (B, N) per-candidate scores."""
        x = self._encode(features, fit_inputs=fit_inputs)
        if self.num_attention_layers > 0:
            key_padding_mask = ~valid_mask.bool() if valid_mask is not None else None
            x = self._attend(x.transpose(1, 2), key_padding_mask).transpose(1, 2)
        self._trunk_activation = x
        scores = self.scorer(x).squeeze(1)
        if self.fusion:
            if filter_logit is None:
                raise ValueError('fusion requires filter_logit')
            scores = self.fusion_alpha * filter_logit + scores
        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        filter_logit: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
        from_b: torch.Tensor | None = None,
        fit_inputs: dict | None = None,
        **_unused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """features: (B, F, N). pos_mask, valid_mask: (B, N) bool.
        filter_logit, log_weights: (B, N). from_b: (B, N) long counts 0-3."""
        scores = self.forward(features, valid_mask=valid_mask,
                              filter_logit=filter_logit, fit_inputs=fit_inputs)
        if self.loss_mode == 'full':
            ranking_loss = full_list_softmax_ce_loss(
                scores, pos_mask.bool(), valid_mask.bool(),
                temperature=self.ranking_temperature,
                label_smoothing=self.label_smoothing,
                log_weights=log_weights)
        else:
            ranking_loss = self._softmax_ce_loss(
                scores, pos_mask.bool(), valid_mask.bool())
        out = {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }
        if self.aux_from_b_weight > 0.0 and from_b is not None:
            logits = self.aux_from_b_head(self._trunk_activation)
            per_candidate = torch.nn.functional.cross_entropy(
                logits, from_b.clamp(0, 3), reduction='none')
            mask = valid_mask.bool().float()
            aux_loss = (per_candidate * mask).sum() / mask.sum().clamp_min(1.0)
            out['aux_from_b_loss'] = aux_loss
            out['total_loss'] = ranking_loss + self.aux_from_b_weight * aux_loss
        return out

    # Sampled InfoNCE top-1 with label smoothing — the couple stage's vectorized
    # implementation, shared so both stages keep a single loss source.
    _softmax_ce_loss = CoupleReranker._softmax_ce_loss
