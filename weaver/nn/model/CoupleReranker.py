"""Stage 3 reranker: per-couple scoring head for the post-ParT cascade.

Designed by `reports/triplet_reranking/triplet_research_plan_20260408.md`
direction A. Each couple drawn from the ParT top-50 (after the loose
``m(ij) <= m_tau`` Filter A) is encoded as a flat 51-dim feature vector by
``part/utils/couple_features.py``, and this module scores each couple
independently — no cross-couple interactions.

Architecture (per-couple, no graph between couples):

    input projection : Conv1d(51 → 256, kernel_size=1) + BN + ReLU
    encoder          : 4 × ResidualBlock(256, dropout=0.1)
    scoring head     : Conv1d(256 → 128) + BN + ReLU + Dropout
                       + Conv1d(128 → 1)

The Conv1d-with-kernel-size-1 idiom means each couple is processed
independently along the ``C_max`` axis — exactly the per-element scoring
shape used by the prefilter (``weaver/weaver/nn/model/TrackPreFilter.py``)
for tracks. Variable couple counts per event are handled by padding to
``C_max`` and masking padded positions in the loss.

Loss: pairwise ranking loss in the same form used by the prefilter and the
CascadeReranker (``L = T · softplus((s_neg − s_pos) / T)``), with N=50
random negatives sampled per positive in each event. Events with no GT
couples are skipped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class NanSafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that skips running stat updates when inputs contain NaN/Inf.

    Standard BatchNorm1d uses an EMA to update running_mean and running_var:
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean

    If even one element is NaN, batch_mean becomes NaN, and the running
    stats are permanently corrupted. This wrapper detects non-finite values
    and skips the stat update for that batch, while still producing finite
    output by replacing NaN/Inf with 0 before the BN forward pass.

    State dict is identical to nn.BatchNorm1d — fully compatible for
    loading/saving checkpoints.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(x)

        has_non_finite = not torch.isfinite(x).all()
        if not has_non_finite:
            # Fast path: clean batch, normal BN forward
            return super().forward(x)

        # Slow path: replace non-finite values with 0, run BN without
        # updating running stats (eval-mode forward), then restore training.
        clean_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.training = False
        output = super().forward(clean_x)
        self.training = True
        return output


class ResidualBlock(nn.Module):
    """Post-activation residual block: 2× Conv1d(1×1) + BN + skip + ReLU.

    Math:
        y = ReLU( BN_2( conv_2( ReLU( BN_1( conv_1(x) ) ) ) ) + x )

    Same shape in / out. The skip is an identity (no projection) because
    the channel dimension is preserved at every block. Dropout sits after
    the inner ReLU.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=1, bias=False,
        )
        self.batchnorm_1 = NanSafeBatchNorm1d(hidden_dim, track_running_stats=False)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=1, bias=False,
        )
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


_SUPPORTED_EMBED_MODES = (
    'concat',
    'infersent',
    'symmetric',
    'bilinear_lrb',
    'projected_infersent',
    'ft_transformer',
    'per_track_tokens',
)

_TOKEN_MODES = ('ft_transformer', 'per_track_tokens')

# Layout constants that mirror part/utils/couple_features.py:
# Block 1 = [t_i (16) ‖ t_j (16)]    indices 0:32
# Block 2/3/4 (physics + geom + cascade scores) indices 32: — default 19
# dims, 23 with pair_kinematics_v2 (4 extra features).
_TRACK_EMBED_DIM = 16
_BLOCK_REST_DIM = 19
_BILINEAR_RANK = 8


def _compute_input_dim(
    couple_embed_mode: str,
    couple_projector_dim: int,
    rest_dim: int = _BLOCK_REST_DIM,
    hidden_dim: int = 256,
) -> int:
    """Size of the per-couple feature tensor handed to the input projection
    after the model-side Block-1 rebuild. See TestCoupleEmbedMode docstring
    for the formula table. ``rest_dim`` is the number of non-Block-1 feature
    dims (physics + geom + cascade scores); 19 by default, 23 with
    pair_kinematics_v2. ``hidden_dim`` is only used for the token modes,
    where the TokenEncoder's CLS readout is already ``hidden_dim``-wide.
    """
    d = _TRACK_EMBED_DIM
    if couple_embed_mode == 'concat':
        return 2 * d + rest_dim
    if couple_embed_mode == 'infersent':
        return 4 * d + rest_dim
    if couple_embed_mode == 'symmetric':
        return 3 * d + rest_dim
    if couple_embed_mode == 'bilinear_lrb':
        return 2 * d + _BILINEAR_RANK + rest_dim
    if couple_embed_mode == 'projected_infersent':
        return 4 * couple_projector_dim + rest_dim
    if couple_embed_mode in _TOKEN_MODES:
        # Token modes bypass the raw Block-1 rebuild. The TokenEncoder
        # returns a (B, hidden_dim, C) tensor directly, so the main
        # input_projection operates on hidden_dim channels.
        return hidden_dim
    raise ValueError(
        f"couple_embed_mode must be one of {_SUPPORTED_EMBED_MODES}, "
        f"got '{couple_embed_mode}'",
    )


class TokenEncoder(nn.Module):
    """Per-couple feature-as-token encoder.

    Input: raw couple feature tensor ``(B, F, C)`` where F = 32 (two 16-d
    track concat halves) + rest_dim (physics + geometry + cascade scores).
    Each of the F scalars is tokenised into a ``d_token``-dimensional
    token via a per-feature learnable weight. A ``[CLS]`` token is
    prepended; the resulting sequence of length F+1 is run through a
    stack of Transformer encoder blocks; the [CLS] output is linearly
    projected to ``hidden_dim``.

    Math per couple:
        z_f  = x_f · w_f       (per-feature embed, w_f ∈ ℝ^{d_token})
        z_f  += track_embed[role(f)]   (only in per_track_tokens mode)
        z    = [cls; z_0; ...; z_{F-1}]  ∈ ℝ^{(F+1) × d_token}
        h    = Transformer(z)            ∈ ℝ^{(F+1) × d_token}
        out  = Linear(h[0])              ∈ ℝ^{hidden_dim}

    Two modes:
        ft_transformer   — flat feature tokens, no role/track identity.
        per_track_tokens — role IDs 0 (track i raw features), 1 (track j
                           raw features), 2 (physics + geometry +
                           cascade scores), added as a learnable per-role
                           embedding to every token.
    """

    def __init__(
        self,
        raw_input_dim: int,
        rest_dim: int,
        d_token: int = 16,
        num_blocks: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        mode: str = 'ft_transformer',
    ):
        super().__init__()
        if mode not in _TOKEN_MODES:
            raise ValueError(
                f'mode must be one of {_TOKEN_MODES}, got {mode!r}',
            )
        if d_token % num_heads != 0:
            raise ValueError(
                f'd_token ({d_token}) must be divisible by num_heads '
                f'({num_heads})',
            )
        self.raw_input_dim = raw_input_dim
        self.rest_dim = rest_dim
        self.d_token = d_token
        self.hidden_dim = hidden_dim
        self.mode = mode

        # Per-feature linear projection from scalar → d_token. Weights are
        # unique per feature — there is no weight sharing across features,
        # which mirrors the FT-Transformer "features-as-tokens" recipe
        # (Gorishniy 2021, arXiv:2106.11959). Initialised with a small
        # std so the initial token pre-attention norm is O(1).
        self.feature_embed = nn.Parameter(
            torch.randn(raw_input_dim, d_token) * 0.02,
        )
        self.feature_bias = nn.Parameter(torch.zeros(raw_input_dim, d_token))

        # [CLS] token embedding; learnable, shared across couples.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        if mode == 'per_track_tokens':
            # role IDs: 0 = track-i raw, 1 = track-j raw, 2 = physics/
            # geometry/cascade. Added to every non-CLS token.
            self.track_embed = nn.Parameter(
                torch.randn(3, d_token) * 0.02,
            )
            role_ids = torch.cat([
                torch.full((_TRACK_EMBED_DIM,), 0, dtype=torch.long),
                torch.full((_TRACK_EMBED_DIM,), 1, dtype=torch.long),
                torch.full((rest_dim,), 2, dtype=torch.long),
            ])
            assert role_ids.shape[0] == raw_input_dim, (
                f'role_ids {role_ids.shape[0]} != raw_input_dim '
                f'{raw_input_dim}'
            )
            self.register_buffer('role_ids', role_ids)

        # Pre-norm Transformer blocks; pre-norm is more stable than post
        # for small/medium models (Xiong 2020, arXiv:2002.04745).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=num_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks,
        )

        # CLS → hidden_dim projection, consumed downstream by the
        # existing residual stack.
        self.cls_to_hidden = nn.Linear(d_token, hidden_dim)

    def forward(self, couple_features: torch.Tensor) -> torch.Tensor:
        """Consume ``(B, F, C)`` and emit ``(B, hidden_dim, C)``.

        Runs F+1 token self-attention per couple, so the effective batch
        is B*C sequences. Memory scales as O(B*C * (F+1)² * d_token).
        """
        B, F, C = couple_features.shape
        if F != self.raw_input_dim:
            raise RuntimeError(
                f'TokenEncoder expected F={self.raw_input_dim}, got F={F}',
            )

        # (B, F, C) → (B, C, F) → (B*C, F, 1)
        scalars = couple_features.permute(0, 2, 1).reshape(B * C, F, 1)
        tokens = scalars * self.feature_embed.unsqueeze(0) \
            + self.feature_bias.unsqueeze(0)

        if self.mode == 'per_track_tokens':
            tokens = tokens + self.track_embed[self.role_ids]

        cls = self.cls_token.expand(B * C, 1, self.d_token)
        tokens = torch.cat([cls, tokens], dim=1)

        # PyTorch's flash-/efficient-SDP kernels cap the leading batch at
        # 65 535. With training batch=96 and C=1770 couples/event,
        # B·C = 169 920 — well above the cap. Chunk along dim 0 to stay
        # under it while keeping the fast attention kernels.
        _SDPA_BATCH_CAP = 65_000
        if tokens.shape[0] > _SDPA_BATCH_CAP:
            outputs = [
                self.transformer(tokens[start:start + _SDPA_BATCH_CAP])
                for start in range(0, tokens.shape[0], _SDPA_BATCH_CAP)
            ]
            out = torch.cat(outputs, dim=0)
        else:
            out = self.transformer(tokens)
        cls_out = out[:, 0, :]

        cls_hidden = self.cls_to_hidden(cls_out)
        cls_hidden = cls_hidden.reshape(B, C, self.hidden_dim)
        return cls_hidden.permute(0, 2, 1).contiguous()


class CoupleReranker(nn.Module):
    """Per-couple Stage 3 reranker.

    Args:
        hidden_dim: width of the encoder and the first scoring layer
            (default 256).
        num_residual_blocks: number of ResidualBlock(hidden_dim) layers
            in the encoder (default 4 → 8 hidden Conv1d layers).
        dropout: dropout rate inside residual blocks and the scoring head
            (default 0.1).
        ranking_num_samples: negatives sampled per positive in the
            pairwise ranking loss (default 50, matches the prefilter and
            CascadeReranker).
        ranking_temperature: softplus temperature in the ranking loss
            (default 1.0).
        couple_embed_mode: how Block 1 of the couple feature tensor is
            rebuilt before the input projection. Default ``'concat'``
            matches the legacy behavior (no rebuild, the feature builder's
            32-dim concat is consumed as-is). Other modes rebuild Block 1
            inside ``forward`` so any learnable projector/bilinear weights
            receive gradients. See ``_compute_input_dim`` for shape.
        couple_projector_dim: per-track projector width ``p`` used only
            when ``couple_embed_mode='projected_infersent'``. The projector
            is ``Linear(16 → p) → ReLU → LayerNorm(p)``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
        couple_loss: str = 'pairwise',
        label_smoothing: float = 0.0,
        hardneg_fraction: float = 0.0,
        hardneg_margin: float = 0.1,
        ndcg_K: int = 100,
        lambda_sigma: float = 1.0,
        ndcg_alpha: float = 5.0,
        multi_positive: str = 'none',
        use_full_negative_list: bool = False,
        aux_vertex_weight: float = 0.0,
        event_context: str = 'none',
        context_dim: int = 32,
        couple_embed_mode: str = 'concat',
        couple_projector_dim: int = 0,
        rest_dim: int = _BLOCK_REST_DIM,
        tokenize_d: int = 16,
        tokenize_blocks: int = 2,
        tokenize_heads: int = 4,
        input_dim: int | None = None,
    ):
        super().__init__()
        if couple_loss not in (
            'pairwise', 'softmax-ce', 'lambda_ndcg2pp', 'approx_ndcg',
        ):
            raise ValueError(
                f"couple_loss must be 'pairwise', 'softmax-ce', "
                f"'lambda_ndcg2pp', or 'approx_ndcg', got '{couple_loss}'",
            )
        if ndcg_alpha <= 0.0:
            raise ValueError(
                f'ndcg_alpha must be > 0, got {ndcg_alpha}',
            )
        if multi_positive not in ('none', 'uniform', 'soft_or'):
            raise ValueError(
                f"multi_positive must be 'none', 'uniform', or 'soft_or', "
                f"got '{multi_positive}'",
            )
        if not 0.0 <= hardneg_fraction <= 1.0:
            raise ValueError(
                f'hardneg_fraction must be in [0, 1], got {hardneg_fraction}',
            )
        if couple_embed_mode not in _SUPPORTED_EMBED_MODES:
            raise ValueError(
                f"couple_embed_mode must be one of {_SUPPORTED_EMBED_MODES}, "
                f"got '{couple_embed_mode}'",
            )
        if couple_embed_mode == 'projected_infersent' and couple_projector_dim <= 0:
            raise ValueError(
                "couple_projector_dim must be > 0 when "
                "couple_embed_mode='projected_infersent' "
                f"(got {couple_projector_dim})",
            )
        self.couple_embed_mode = couple_embed_mode
        self.couple_projector_dim = couple_projector_dim
        self.rest_dim = rest_dim
        self.tokenize_d = tokenize_d
        self.tokenize_blocks = tokenize_blocks
        self.tokenize_heads = tokenize_heads
        # input_dim derived from mode + projector dim + rest_dim + hidden
        # dim. In token modes, input_dim == hidden_dim because the
        # TokenEncoder's CLS projection produces a (B, hidden_dim, C)
        # tensor which feeds directly into the downstream residual stack.
        if input_dim is None:
            self.input_dim = _compute_input_dim(
                couple_embed_mode,
                couple_projector_dim,
                rest_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        self.dropout_rate = dropout
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.couple_loss = couple_loss
        self.label_smoothing = label_smoothing
        self.hardneg_fraction = hardneg_fraction
        self.hardneg_margin = hardneg_margin
        self.ndcg_K = ndcg_K
        self.lambda_sigma = lambda_sigma
        self.ndcg_alpha = ndcg_alpha
        self.multi_positive = multi_positive
        self.use_full_negative_list = use_full_negative_list
        self.aux_vertex_weight = aux_vertex_weight
        if event_context not in ('none', 'deepset_film'):
            raise ValueError(
                f"event_context must be 'none' or 'deepset_film', "
                f"got '{event_context}'",
            )
        self.event_context = event_context
        self.context_dim = context_dim

        # Mode-specific Block 1 parameters (only allocated when needed).
        if couple_embed_mode == 'bilinear_lrb':
            # Low-rank bilinear: (U t_i) ⊙ (V t_j), both 16→r
            self.bilinear_u = nn.Linear(
                _TRACK_EMBED_DIM, _BILINEAR_RANK, bias=False,
            )
            self.bilinear_v = nn.Linear(
                _TRACK_EMBED_DIM, _BILINEAR_RANK, bias=False,
            )
        elif couple_embed_mode == 'projected_infersent':
            # φ(t) = LayerNorm(ReLU(Linear(16 → p)(t)))
            # Weight-shared across i and j (applied to each track).
            self.couple_projector = nn.Sequential(
                nn.Linear(_TRACK_EMBED_DIM, couple_projector_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(couple_projector_dim),
            )
        elif couple_embed_mode in _TOKEN_MODES:
            # Per-couple Transformer tokeniser. Replaces the block-1
            # rebuild entirely; emits a (B, hidden_dim, C) tensor that
            # feeds directly into the downstream residual stack.
            self.token_encoder = TokenEncoder(
                raw_input_dim=2 * _TRACK_EMBED_DIM + rest_dim,
                rest_dim=rest_dim,
                d_token=tokenize_d,
                num_blocks=tokenize_blocks,
                num_heads=tokenize_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                mode=couple_embed_mode,
            )

        # Input projection: input_dim → hidden_dim. For token modes this
        # is an identity-style projection since input_dim == hidden_dim,
        # but we keep the same Conv1d+BN+ReLU structure for the downstream
        # residual blocks and BN-calibration pipeline to stay unchanged.
        self.input_projection = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        # Encoder: stack of residual blocks at width hidden_dim
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

        # Scoring head: hidden_dim → hidden_dim/2 → 1
        intermediate_dim = hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Conv1d(hidden_dim, intermediate_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(intermediate_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(intermediate_dim, 1, kernel_size=1),
        )
        # B3 H6: event-context DeepSet encoder + FiLM conditioner.
        # Pools the K2=60 track feature tensor into a per-event vector
        # via mean/max/std along the track axis, projects it to
        # context_dim, and FiLM-conditions the per-couple hidden
        # representation just before scoring: h' = (1+γ)⊙h + β.
        if event_context == 'deepset_film':
            self.event_context_encoder = nn.Sequential(
                nn.Linear(3 * _TRACK_EMBED_DIM, context_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(context_dim),
            )
            self.film_gamma = nn.Linear(context_dim, hidden_dim)
            self.film_beta = nn.Linear(context_dim, hidden_dim)
            # Zero-init the FiLM projections so early training matches
            # the baseline without context (γ=0, β=0 → h' = h).
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        # B3 H7: vertex-compatibility auxiliary head — predicts the same
        # couple_label (1 if both tracks share the tau decay vertex, 0
        # otherwise) from the encoder's hidden representation. Acts as a
        # secondary supervision signal that regularises the backbone
        # without sharing weights with the main ranking scorer.
        if aux_vertex_weight > 0.0:
            self.aux_vertex_head = nn.Sequential(
                nn.Conv1d(
                    hidden_dim, intermediate_dim,
                    kernel_size=1, bias=False,
                ),
                NanSafeBatchNorm1d(intermediate_dim, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(intermediate_dim, 1, kernel_size=1),
            )

    def _encode(
        self,
        couple_features: torch.Tensor,
        k2_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the couple encoder (block-1 rebuild → residual stack)
        plus optional event-context FiLM conditioning.

        ``k2_features`` is the (B, 16, K2) per-track tensor. When
        ``event_context == 'deepset_film'`` it is required; otherwise
        ignored.
        """
        if self.couple_embed_mode in _TOKEN_MODES:
            x = self.token_encoder(couple_features)
        else:
            couple_features = self._rebuild_block1(couple_features)
            x = couple_features
        x = self.input_projection(x)
        for block in self.residual_blocks:
            x = block(x)
        if self.event_context == 'deepset_film':
            if k2_features is None:
                raise RuntimeError(
                    "event_context='deepset_film' requires k2_features",
                )
            # (B, 16, K2) → pool along K2 axis.
            pooled_mean = k2_features.mean(dim=2)
            pooled_max = k2_features.amax(dim=2)
            pooled_std = k2_features.std(dim=2, unbiased=False)
            pooled = torch.cat(
                [pooled_mean, pooled_max, pooled_std], dim=1,
            )  # (B, 48)
            c_ev = self.event_context_encoder(pooled)  # (B, context_dim)
            gamma = self.film_gamma(c_ev).unsqueeze(-1)  # (B, hid, 1)
            beta = self.film_beta(c_ev).unsqueeze(-1)
            x = (1.0 + gamma) * x + beta
        return x  # (B, hidden_dim, C)

    def forward(
        self,
        couple_features: torch.Tensor,
        k2_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score each couple independently.

        The feature builder always emits a couple tensor with the raw
        `[t_i (16) ‖ t_j (16)]` concat as Block 1 (indices 0:32). When
        ``couple_embed_mode`` is anything other than ``'concat'``, Block 1
        is rebuilt here so any learnable projection/bilinear/tokenizer
        weights sit in the gradient path. In token modes the entire
        Block-1 rebuild is replaced by a per-couple self-attention
        tokenizer that returns a ``(B, hidden_dim, C)`` tensor directly.

        Args:
            couple_features: ``(B, F, C)`` per-couple feature tensor where
                F = 32 + rest_dim (typically 51 or 55).

        Returns:
            ``(B, C)`` per-couple scores.
        """
        h = self._encode(couple_features, k2_features=k2_features)
        scores = self.scorer(h)
        return scores.squeeze(1)

    def _rebuild_block1(self, couple_features: torch.Tensor) -> torch.Tensor:
        """Rebuild Block 1 according to ``couple_embed_mode``.

        The input carries the raw 32-dim concat ``[t_i, t_j]`` in its first
        two track slots. Blocks 2-4 (pairwise physics + derived geometry +
        cascade scores) are preserved as-is.
        """
        if self.couple_embed_mode == 'concat':
            return couple_features

        d = _TRACK_EMBED_DIM
        track_i = couple_features[:, :d, :]
        track_j = couple_features[:, d:2 * d, :]
        rest = couple_features[:, 2 * d:, :]

        if self.couple_embed_mode == 'infersent':
            # InferSent-style pair fusion: [u, v, |u − v|, u ⊙ v]
            diff = (track_i - track_j).abs()
            prod = track_i * track_j
            block1 = torch.cat([track_i, track_j, diff, prod], dim=1)
        elif self.couple_embed_mode == 'symmetric':
            # Permutation-invariant aggregation: [max, mean, |diff|]
            track_max = torch.maximum(track_i, track_j)
            track_mean = 0.5 * (track_i + track_j)
            diff = (track_i - track_j).abs()
            block1 = torch.cat([track_max, track_mean, diff], dim=1)
        elif self.couple_embed_mode == 'bilinear_lrb':
            # Low-rank bilinear interaction term appended to the concat.
            # u_i, v_j : (B, r, C) where r = _BILINEAR_RANK
            u_weight = self.bilinear_u.weight  # (r, d)
            v_weight = self.bilinear_v.weight
            u_i = torch.einsum('rd,bdc->brc', u_weight, track_i)
            v_j = torch.einsum('rd,bdc->brc', v_weight, track_j)
            interaction = u_i * v_j
            block1 = torch.cat([track_i, track_j, interaction], dim=1)
        elif self.couple_embed_mode == 'projected_infersent':
            # φ(t) applied per-track (weight-shared). Input shape (B, d, C),
            # nn.Linear expects (..., d) so we transpose (d ↔ C).
            phi_i = self.couple_projector(
                track_i.transpose(1, 2),
            ).transpose(1, 2)
            phi_j = self.couple_projector(
                track_j.transpose(1, 2),
            ).transpose(1, 2)
            diff = (phi_i - phi_j).abs()
            prod = phi_i * phi_j
            block1 = torch.cat([phi_i, phi_j, diff, prod], dim=1)
        else:
            raise RuntimeError(
                f'unreachable: mode {self.couple_embed_mode}',
            )

        return torch.cat([block1, rest], dim=1)

    def compute_loss(
        self,
        couple_features: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
        k2_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Couple ranking loss — dispatches to the configured branch.

        Two branches are supported (selected via ``self.couple_loss``):

        * ``'pairwise'`` (default): softplus pairwise ranking loss
          ``L = T · softplus((s_neg − s_pos) / T)`` averaged over all
          (positive, negative) pairs per event, then over events.
          Mirrors ``CascadeReranker._pairwise_ranking_loss``.

        * ``'softmax-ce'``: listwise softmax cross-entropy (ListMLE
          top-1). For each positive ``p_i`` in an event, the loss is
          ``L_i = −s_{p_i}/T + logsumexp([s_{p_i}, s_{negs}]/T)``,
          optionally smoothed by ``self.label_smoothing``.

        Args:
            couple_features: ``(B, input_dim, C)`` per-couple features.
            couple_labels: ``(B, C)`` binary labels (1 = GT couple).
            couple_mask: ``(B, C)`` validity mask (1 = real, 0 = padded).

        Returns:
            dict with ``'total_loss'``, ``'ranking_loss'``, ``'_scores'``.
        """
        h = self._encode(
            couple_features, k2_features=k2_features,
        )  # (B, hidden_dim, C)
        scores = self.scorer(h).squeeze(1)  # (B, C)
        if self.couple_loss == 'softmax-ce':
            ranking_loss = self._softmax_ce_loss(
                scores, couple_labels, couple_mask,
            )
        elif self.couple_loss == 'lambda_ndcg2pp':
            ranking_loss = self._lambda_ndcg_loss(
                scores, couple_labels, couple_mask,
            )
        elif self.couple_loss == 'approx_ndcg':
            ranking_loss = self._approx_ndcg_loss(
                scores, couple_labels, couple_mask,
            )
        else:
            ranking_loss = self._pairwise_loss(
                scores, couple_labels, couple_mask,
            )
        total_loss = ranking_loss
        aux_loss = None
        if self.aux_vertex_weight > 0.0:
            # H7: binary cross-entropy on the couple-label prediction
            # from the aux head, masked by couple validity.
            aux_logits = self.aux_vertex_head(h).squeeze(1)  # (B, C)
            valid = couple_mask > 0.5
            if valid.any():
                aux_loss = functional.binary_cross_entropy_with_logits(
                    aux_logits[valid],
                    couple_labels[valid].float(),
                )
                total_loss = total_loss + self.aux_vertex_weight * aux_loss
            else:
                aux_loss = scores.sum() * 0.0
        out = {
            'total_loss': total_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }
        if aux_loss is not None:
            out['aux_vertex_loss'] = aux_loss
        return out

    def _pairwise_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
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

            sampled_negatives = self._sample_negative_indices(
                event_scores, positive_indices, negative_indices,
            )

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)
            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            return scores.sum() * 0.0
        return torch.stack(event_losses).mean()

    def _sample_negative_indices(
        self,
        event_scores: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``ranking_num_samples`` negative couple indices.

        When ``hardneg_fraction == 0`` the sample is fully random (legacy
        behavior). When > 0, the first ``⌈N · frac⌉`` slots are filled
        from the top-scoring negatives that pass a positive-aware margin
        filter (``score < max(pos_scores) − margin``); this is the ANCE /
        NV-Retriever recipe and prevents false negatives from being
        selected as hard negatives. The remaining slots are filled with
        random negatives to preserve gradient diversity.

        Math:
            N     = min(ranking_num_samples, |negs|)
            M     = min(⌈N · hardneg_fraction⌉, |negs|)
            pool  = top-M negatives by score, filtered by
                    score < max(s_pos) − hardneg_margin
            fill  = N − |pool| random negatives
            out   = concat(pool, fill)
        """
        if self.use_full_negative_list:
            # H4: bypass sampling and use every valid negative in the
            # event. Moves gradient from a 50-sample estimate to the
            # full 1 770-couple denominator; larger per-step compute but
            # eliminates sampling noise at the rank-100 boundary.
            return negative_indices
        num_samples = min(self.ranking_num_samples, len(negative_indices))
        if self.hardneg_fraction <= 0.0:
            sample_indices = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=event_scores.device,
            )
            return negative_indices[sample_indices]

        hardneg_count = min(
            int(round(num_samples * self.hardneg_fraction)),
            len(negative_indices),
        )
        if hardneg_count > 0:
            with torch.no_grad():
                negative_scores_detached = event_scores[
                    negative_indices
                ].detach()
                top_values, top_positions = torch.topk(
                    negative_scores_detached,
                    k=min(hardneg_count, len(negative_indices)),
                )
                threshold = (
                    event_scores[positive_indices].detach().max()
                    - self.hardneg_margin
                )
                keep_mask = top_values < threshold
                hard_positions = top_positions[keep_mask]
            hard_negatives = negative_indices[hard_positions]
        else:
            hard_negatives = negative_indices.new_empty((0,))

        remaining = num_samples - len(hard_negatives)
        if remaining > 0:
            random_positions = torch.randint(
                0, len(negative_indices), (remaining,),
                device=event_scores.device,
            )
            random_negatives = negative_indices[random_positions]
            sampled = torch.cat([hard_negatives, random_negatives], dim=0)
        else:
            sampled = hard_negatives[:num_samples]
        return sampled

    def _softmax_ce_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Listwise softmax cross-entropy loss.

        For each positive ``p_i`` in an event, build the candidate pool
        ``{p_i} ∪ sample(negs, N)`` and compute

            L_i = −log p_i + ε · (−1/M · Σ_j log p_j − log p_i)
                = (1 − ε) · (−s_{p_i}/T + logsumexp(s_pool/T))
                  + ε · (−mean(s_pool)/T + logsumexp(s_pool/T))

        where ``M = len(pool)`` and ``ε = self.label_smoothing``. With
        ``ε = 0`` this reduces to the plain listwise softmax-CE (a.k.a.
        ListMLE top-1), which directly optimizes P(positive ranked first).
        """
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

            sampled_negatives = self._sample_negative_indices(
                event_scores, positive_indices, negative_indices,
            )
            negative_scores = event_scores[sampled_negatives]

            if self.multi_positive in ('uniform', 'soft_or'):
                # Single pool spanning ALL positives and sampled negatives.
                # 'uniform': target distributes 1/k mass across GT couples.
                #   Loss: L = logsumexp(pool/T) − mean(s_pos)/T.
                # 'soft_or' (H3): MIL / "at least one GT survives" aligned
                #   numerator — log-sum-exp over positives instead of mean:
                #   Loss: L = logsumexp(pool/T) − logsumexp(s_pos/T).
                positive_scores = event_scores[positive_indices]
                pool_scores = torch.cat(
                    [positive_scores, negative_scores], dim=0,
                )
                scaled_pool = pool_scores / temperature
                log_normalizer = torch.logsumexp(scaled_pool, dim=0)
                if self.multi_positive == 'soft_or':
                    # Numerator in the soft-OR formulation.
                    pos_log_sum = torch.logsumexp(
                        positive_scores / temperature, dim=0,
                    )
                    multi_nll = log_normalizer - pos_log_sum
                else:
                    multi_nll = (
                        log_normalizer
                        - positive_scores.mean() / temperature
                    )
                if eps > 0.0:
                    uniform_nll = log_normalizer - scaled_pool.mean()
                    event_loss = (1.0 - eps) * multi_nll + eps * uniform_nll
                else:
                    event_loss = multi_nll
                event_losses.append(event_loss)
            else:
                # Legacy single-positive branch: one pool per positive,
                # mean across positives. With k=1 this is the standard
                # ListMLE top-1; with k>1 it treats each positive
                # independently against the same negative pool.
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
                        uniform_nll = (
                            -scaled_pool.mean() + log_normalizer
                        )
                        smoothed = (
                            (1.0 - eps) * nll + eps * uniform_nll
                        )
                        positive_losses.append(smoothed)
                    else:
                        positive_losses.append(nll)

                event_losses.append(torch.stack(positive_losses).mean())

        if not event_losses:
            return scores.sum() * 0.0
        return torch.stack(event_losses).mean()

    def _lambda_ndcg_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> torch.Tensor:
        """LambdaLoss with NDCG-Loss2++ weighting and top-K truncation.

        Per Wang et al. (CIKM 2018, arXiv:1811.04768) and the top-K
        lambdarank derivation (Jagerman 2022, arXiv:2211.04372). For each
        event and each pair of valid couples (i, j) with different
        relevance (one positive, one negative), contribute

            δ_ij = |G_i − G_j| · |1/D_i − 1/D_j|
                 = |1/log₂(rank_i+1) − 1/log₂(rank_j+1)|
            L_ij = δ_ij · log(1 + exp(−σ · (s_i − s_j)))

        where the positive's score is ``s_i`` and the negative's is
        ``s_j``. Pairs whose BOTH ranks are strictly greater than
        ``self.ndcg_K`` are dropped from the sum — only swaps that could
        move probability mass across the top-K frontier are retained.
        Ranks are derived from the current detached scores (positive
        infinity for masked positions, so they sort to the tail and are
        guaranteed rank > K).

        The loss is averaged over events that contain at least one
        (positive, negative) pair; events with no positives or no
        negatives contribute zero.
        """
        batch_size = scores.shape[0]
        sigma = self.lambda_sigma
        ndcg_K = self.ndcg_K
        event_losses: list[torch.Tensor] = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = couple_labels[event_index]
            event_valid = couple_mask[event_index] > 0.5

            positive_mask = (event_labels > 0.5) & event_valid
            negative_mask = (event_labels < 0.5) & event_valid

            if not positive_mask.any() or not negative_mask.any():
                continue

            # Compute ranks among valid positions. Masked positions get
            # −inf so they sort to the tail; their ranks end up > C_valid
            # and past any reasonable K — they are therefore excluded by
            # the K-truncation.
            ranks_of_valid = self._compute_ranks(event_scores, event_valid)

            # Gain and discount (1-indexed rank).
            gain = event_labels.float()
            discount = 1.0 / torch.log2(ranks_of_valid.float() + 1.0)

            pos_indices = positive_mask.nonzero(as_tuple=True)[0]
            neg_indices = negative_mask.nonzero(as_tuple=True)[0]

            pos_scores = event_scores[pos_indices]
            neg_scores = event_scores[neg_indices]
            pos_ranks = ranks_of_valid[pos_indices]
            neg_ranks = ranks_of_valid[neg_indices]
            pos_gain = gain[pos_indices]
            neg_gain = gain[neg_indices]
            pos_disc = discount[pos_indices]
            neg_disc = discount[neg_indices]

            # Pairwise matrices: (|pos|, |neg|).
            gain_diff = (pos_gain.unsqueeze(1) - neg_gain.unsqueeze(0)).abs()
            disc_diff = (pos_disc.unsqueeze(1) - neg_disc.unsqueeze(0)).abs()
            delta = gain_diff * disc_diff

            # Top-K truncation: drop pair if BOTH ranks exceed K. When
            # ndcg_K <= 0 the truncation is disabled and every (pos, neg)
            # pair contributes — the NDCG log-discount already decays
            # with rank so far-tail pairs have low weight organically.
            if ndcg_K > 0:
                pos_in_K = pos_ranks.unsqueeze(1) <= ndcg_K
                neg_in_K = neg_ranks.unsqueeze(0) <= ndcg_K
                keep = pos_in_K | neg_in_K
                delta = delta * keep.float()

            # Pairwise ranknet-style term. `softplus(-σ·(s_pos-s_neg))`
            # is `log(1 + exp(−σ·(s_pos-s_neg)))`; numerically stable via
            # torch.nn.functional.softplus.
            score_margin = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            loss_mat = delta * functional.softplus(-sigma * score_margin)

            # Ideal-DCG normalisation per Wang et al. 2018: divide by the
            # DCG of the optimal ordering (all positives at the top).
            # For binary gains with n_pos positives, ideal ordering is
            # ranks 1..n_pos, so ideal_DCG = Σ_{r=1}^{n_pos} 1/log2(r+1).
            n_pos = int(pos_indices.numel())
            ideal_ranks = torch.arange(
                1, n_pos + 1, device=event_scores.device,
                dtype=event_scores.dtype,
            )
            ideal_dcg = (1.0 / torch.log2(ideal_ranks + 1.0)).sum()
            # Normalise by ideal_DCG (NDCG-style) and by pair count so
            # loss magnitude stays comparable across events with
            # different |pos|·|neg|.
            num_pairs = max(1, delta.shape[0] * delta.shape[1])
            event_losses.append(loss_mat.sum() / (ideal_dcg * num_pairs))

        if not event_losses:
            return scores.sum() * 0.0
        return torch.stack(event_losses).mean()

    def _approx_ndcg_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ApproxNDCG loss (Qin, Liu, Li 2010 — "A General Approximation
        Framework for Direct Optimization of Information Retrieval
        Measures").

        Per event with binary relevance ``gain ∈ {0, 1}``:

            approx_rank_i = 1 + Σ_{j ≠ i} sigmoid(α · (s_j − s_i))
            DCG          = Σ_i gain_i / log₂(approx_rank_i + 1)
            ideal_DCG    = Σ_{r=1}^{n_pos} 1 / log₂(r + 1)
            NDCG         = DCG / ideal_DCG     ∈ [0, 1]
            loss         = −NDCG               ∈ [−1, 0]

        The sigmoid smooths the rank step function; as ``α → ∞`` the
        approximation becomes exact (non-differentiable). ``α = 5`` is a
        reasonable default for this problem scale.

        Masked positions contribute no gain and are excluded from the
        approximate rank sum (their score is pushed to −∞ before the
        pairwise sigmoid so they always rank last).

        Events with no positives or no valid couples contribute zero to
        the final reduction.
        """
        batch_size = scores.shape[0]
        alpha = self.ndcg_alpha
        event_losses: list[torch.Tensor] = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = couple_labels[event_index]
            event_valid = couple_mask[event_index] > 0.5

            positive_mask = (event_labels > 0.5) & event_valid
            if not positive_mask.any() or not event_valid.any():
                continue

            # Mask out invalid positions by pushing their scores very low
            # so they never rank above valid ones in the sigmoid sum.
            # Use the dtype's min instead of the literal -1e6, which
            # overflows fp16 under AMP autocast.
            neg_fill = torch.finfo(event_scores.dtype).min
            scores_masked = torch.where(
                event_valid,
                event_scores,
                torch.full_like(event_scores, neg_fill),
            )

            # Pairwise sigmoid differences. rank_i = 1 + Σ_{j ≠ i}
            # sigmoid(α·(s_j − s_i)) — "count of positions j beating i".
            # With broadcasting `diff[i, j] = s[j] - s[i]`, the sum for
            # position i runs over columns (dim=1). Invalid `i`
            # positions end up with huge rank and zero gain, so they
            # don't contribute to DCG.
            diff = (
                scores_masked.unsqueeze(0)
                - scores_masked.unsqueeze(1)
            )  # (C, C): [i, j] = s_j − s_i
            sig = torch.sigmoid(alpha * diff)
            # Zero self-pairs to exclude i==j from the sum.
            diag_mask = 1.0 - torch.eye(
                sig.shape[0], device=sig.device, dtype=sig.dtype,
            )
            approx_ranks = 1.0 + (sig * diag_mask).sum(dim=1)

            gain = event_labels.float() * event_valid.float()
            # DCG contributions: only GT positions matter because gain=0
            # elsewhere. log2(rank + 1) is safe: approx_ranks >= 1.
            dcg = (gain / torch.log2(approx_ranks + 1.0)).sum()

            n_pos = int(positive_mask.sum().item())
            ideal_ranks = torch.arange(
                1, n_pos + 1, device=event_scores.device,
                dtype=event_scores.dtype,
            )
            ideal_dcg = (1.0 / torch.log2(ideal_ranks + 1.0)).sum()

            ndcg = dcg / ideal_dcg
            event_losses.append(-ndcg)

        if not event_losses:
            return scores.sum() * 0.0
        return torch.stack(event_losses).mean()

    @staticmethod
    def _compute_ranks(
        event_scores: torch.Tensor,
        event_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return 1-indexed rank of every position among the valid ones.

        Invalid (masked) positions are pushed to the tail by replacing
        their scores with −inf before the sort; their resulting rank is
        larger than any K of interest, which is what the caller relies
        on to exclude them from the LambdaLoss sum.
        """
        with torch.no_grad():
            scores_for_rank = event_scores.detach().clone()
            scores_for_rank[~event_valid] = float('-inf')
            # argsort descending gives index-order; inverse permutation
            # yields rank (0 = highest score).
            order = scores_for_rank.argsort(descending=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(
                order.shape[0], device=order.device,
            )
            # 1-index.
            return ranks + 1
