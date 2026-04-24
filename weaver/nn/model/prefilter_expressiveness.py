"""Expressiveness plug-in heads for the Stage-1 TrackPreFilter.

All three modules are inserted behind explicit CLI flags and default to
disabled. When disabled, the TrackPreFilter reverts to the E2a baseline
(edge features on, k=16, r=3, raw 16→256 input Conv1d) with no
parameter-count or state-dict change.

Mathematical contracts:

``PerFeatureEmbedding`` (P1) — per-feature learnable embedding via
grouped 1×1 convolution. For input ``x ∈ ℝ^{B×F×P}`` with ``F`` raw
features, each channel ``f`` is projected independently to ``E``
dimensions:

    y_f(p) = ReLU( LayerNorm( W_f · x_f(p) + b_f ) ),  f ∈ {0, …, F−1}

The ``F·E`` output channels are concatenated along the channel axis:
``y ∈ ℝ^{B×FE×P}``. No cross-feature mixing at this stage.

``FeatureGate`` (P2) — Squeeze-Excite gate on the per-track embedding.
For ``h ∈ ℝ^{B×H×P}`` with mask ``m ∈ {0,1}^{B×1×P}``:

    z  = MaskedMean_p( h, m )                        ∈ ℝ^{B×H}
    g  = σ( W_exp · ReLU( W_squeeze · z ) )         ∈ ℝ^{B×H}
    y  = h · g[:, :, None]

Per-track gating via a per-event gate vector; cheap regularisation.

``FiLMHead`` (P3) — Feature-wise Linear Modulation from an event-level
context. For raw features ``x ∈ ℝ^{B×F×P}``, mask ``m``, and per-track
embedding ``h ∈ ℝ^{B×H×P}``:

    μ_event = MaskedMean_p( x, m )                   ∈ ℝ^{B×F}
    σ_event = MaskedStd_p( x, m )                    ∈ ℝ^{B×F}
    c       = ReLU( W_ctx · [μ_event, σ_event] )    ∈ ℝ^{B×C}
    γ, β    = W_γ · c, W_β · c                       ∈ ℝ^{B×H}
    y       = (1 + γ[:, :, None]) · h + β[:, :, None]

γ defaults to 0 at init so the head starts as an identity modulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# P1 — PerFeatureEmbedding
# ---------------------------------------------------------------------------


class PerFeatureEmbedding(nn.Module):
    """Per-feature learnable embedding with grouped 1×1 conv + LN + ReLU.

    Each of ``num_features`` input channels is projected independently
    to ``embed_dim`` output channels — no cross-feature mixing happens
    here; that is the downstream ``track_mlp``'s job. LayerNorm is
    applied per feature, so the normalisation statistics are shared
    across tracks within the same feature-embedding group but not
    across features.

    Args:
        num_features: number of raw input features (``F``); must equal
            the ``input_dim`` expected by the downstream ``track_mlp``.
        embed_dim: per-feature embedding width ``E``. Output channel
            count is ``F × E``.
    """

    def __init__(self, num_features: int = 16, embed_dim: int = 32):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.per_feature_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * embed_dim,
            kernel_size=1,
            groups=num_features,
            bias=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, num_features, P)`` standardised raw features.

        Returns:
            ``(B, num_features * embed_dim, P)`` — concatenated
            per-feature embeddings ordered by feature index.
        """
        if x.shape[1] != self.num_features:
            raise ValueError(
                f'PerFeatureEmbedding expects {self.num_features} input '
                f'channels, got {x.shape[1]}',
            )
        batch_size, num_features, num_positions = x.shape
        embed_dim = self.embed_dim

        # Grouped 1×1 conv: per-feature weights, no cross-feature mixing.
        embedded_flat = self.per_feature_conv(x)  # (B, F*E, P)

        # Per-feature LayerNorm: reshape so the embed dim is last.
        embedded = (
            embedded_flat
            .view(batch_size, num_features, embed_dim, num_positions)
            .permute(0, 1, 3, 2)
            .contiguous()
        )  # (B, F, P, E)
        embedded = self.layer_norm(embedded)
        embedded = self.activation(embedded)

        # Restore channel-first layout and flatten (F, E) back to F*E.
        embedded = (
            embedded
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(batch_size, num_features * embed_dim, num_positions)
        )
        return embedded


# ---------------------------------------------------------------------------
# P2 — FeatureGate (SE-style squeeze-excite)
# ---------------------------------------------------------------------------


class FeatureGate(nn.Module):
    """Squeeze-Excite gate on the per-track embedding.

    Aggregates a per-event descriptor via masked mean over the track
    dim, passes it through a ``hidden_dim → bottleneck → hidden_dim``
    bottleneck, and applies the sigmoid gate per channel. Broadcasts
    the per-event gate across tracks — i.e., every track in an event
    shares the same channel-wise modulation.

    Args:
        hidden_dim: channel count of the track embedding ``h``.
        bottleneck: width of the squeeze step (``hidden_dim //
            reduction`` in the SE paper; exposed directly here).
    """

    def __init__(self, hidden_dim: int = 256, bottleneck: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.squeeze = nn.Conv1d(hidden_dim, bottleneck, kernel_size=1)
        self.excite = nn.Conv1d(bottleneck, hidden_dim, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        track_embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            track_embedding: ``(B, H, P)`` per-track features.
            mask: ``(B, 1, P)`` ``{0, 1}`` — valid-track indicator.

        Returns:
            ``(B, H, P)`` gated per-track features.
        """
        mask_float = mask.float()
        valid_count = mask_float.sum(dim=2, keepdim=True).clamp(min=1.0)
        # Masked mean → (B, H, 1).
        event_mean = (track_embedding * mask_float).sum(
            dim=2, keepdim=True,
        ) / valid_count
        squeezed = self.activation(self.squeeze(event_mean))  # (B, B', 1)
        gate = torch.sigmoid(self.excite(squeezed))  # (B, H, 1)
        return track_embedding * gate


# ---------------------------------------------------------------------------
# P3 — FiLMHead (event-context → per-track (γ, β))
# ---------------------------------------------------------------------------


class FiLMHead(nn.Module):
    """Event-conditioned per-track (γ, β) modulation.

    Event-level context built from the **masked** mean and std of the
    raw feature tensor — masked tokens never contribute. Context goes
    through one hidden layer, then two output heads produce (γ, β)
    vectors of width ``hidden_dim``. γ initialised at 0 so at the
    start of training the modulation is an identity pass-through.

    Args:
        num_features: raw feature count ``F`` (used to size the
            context input, which is ``[μ, σ] ∈ ℝ^{2F}``).
        hidden_dim: channel count of the per-track embedding to
            modulate.
        context_dim: width of the event-context hidden layer.
    """

    def __init__(
        self,
        num_features: int = 16,
        hidden_dim: int = 256,
        context_dim: int = 32,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.context_net = nn.Sequential(
            nn.Linear(2 * num_features, context_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma_head = nn.Linear(context_dim, hidden_dim)
        self.beta_head = nn.Linear(context_dim, hidden_dim)
        # γ starts at 0 → (1 + γ) = 1 at init, so the head is an
        # identity modulation at the start of training.
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(
        self,
        track_embedding: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            track_embedding: ``(B, hidden_dim, P)`` per-track features
                to be modulated.
            features: ``(B, num_features, P)`` raw standardised input
                features (used for the event-context statistics).
            mask: ``(B, 1, P)`` ``{0, 1}``.

        Returns:
            ``(B, hidden_dim, P)`` — modulated embedding.
        """
        mask_float = mask.float()
        valid_count = mask_float.sum(dim=2, keepdim=True).clamp(min=1.0)

        # Masked mean over valid tracks: (B, F, 1).
        event_mean = (features * mask_float).sum(
            dim=2, keepdim=True,
        ) / valid_count
        # Masked std = sqrt( masked_mean(x²) − masked_mean(x)² ).
        event_mean_sq = (features ** 2 * mask_float).sum(
            dim=2, keepdim=True,
        ) / valid_count
        event_var = (event_mean_sq - event_mean ** 2).clamp(min=0.0)
        event_std = torch.sqrt(event_var + 1e-6)

        context_input = torch.cat(
            [event_mean.squeeze(-1), event_std.squeeze(-1)], dim=1,
        )  # (B, 2F)
        context_hidden = self.context_net(context_input)  # (B, C)
        gamma = self.gamma_head(context_hidden).unsqueeze(-1)  # (B, H, 1)
        beta = self.beta_head(context_hidden).unsqueeze(-1)  # (B, H, 1)
        return (1.0 + gamma) * track_embedding + beta


# ---------------------------------------------------------------------------
# P4 — SoftAttentionAggregator (replaces max-pool in message passing)
# ---------------------------------------------------------------------------


class SoftAttentionAggregator(nn.Module):
    """Learned soft-attention pooling over the K kNN neighbours.

    Replaces the hard max-pool aggregator in the prefilter's message-
    passing rounds. For each center track ``i`` and its K neighbours
    ``{j}``, a small MLP scores the (center, neighbour, edge) triple:

        logits_ij = MLP( [h_center_i, h_neighbor_j, edge_feat_ij] )
        α_ij     = softmax_j( logits_ij over valid neighbours )
        pooled_i = Σ_j α_ij · h_neighbor_j

    Masked-out neighbours get ``-inf`` logits so softmax assigns them
    zero weight. Centers with **no** valid neighbours at all (every
    row masked) fall back to a zero aggregation — no NaN, no gradient
    surprises — with the upstream message-passing concat handling the
    hidden-state fallback. One ``SoftAttentionAggregator`` is created
    per message-passing round in ``TrackPreFilter``.

    Args:
        hidden_dim: channel count of both center and neighbour
            features (they share the embedding dim).
        edge_dim: channel count of per-edge features (``4`` for the
            E2a Lorentz-scalar set, ``0`` if edges are disabled).
        bottleneck: width of the score MLP's hidden layer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 4,
        bottleneck: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        input_dim = 2 * hidden_dim + edge_dim
        self.score_mlp = nn.Sequential(
            nn.Conv2d(input_dim, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, 1, kernel_size=1),
        )

    def compute_weights(
        self,
        current: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_validity: torch.Tensor,
        edge_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the softmax attention weights of shape ``(B, P, K)``.

        Masked-out neighbours carry weight 0. Rows whose every
        neighbour is masked also carry weight 0 (fallback path).
        """
        batch_size, hidden_dim, num_positions, num_neighbors = (
            neighbor_features.shape
        )
        center_expanded = current.unsqueeze(-1).expand_as(neighbor_features)
        cat_list = [center_expanded, neighbor_features]
        if edge_features is not None:
            cat_list.append(edge_features)
        score_input = torch.cat(cat_list, dim=1)  # (B, 2H+E, P, K)
        logits = self.score_mlp(score_input).squeeze(1)  # (B, P, K)

        valid_mask = (neighbor_validity > 0.5).squeeze(1)  # (B, P, K)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        has_any_valid = valid_mask.any(dim=-1, keepdim=True)  # (B, P, 1)

        # Softmax is undefined on all-(-inf) rows. Substitute a zero-
        # logit vector for those rows so softmax returns uniform weights,
        # then zero those rows out after the fact so the downstream sum
        # effectively skips them.
        logits_safe = torch.where(
            has_any_valid, logits, torch.zeros_like(logits),
        )
        weights = torch.softmax(logits_safe, dim=-1)
        weights = weights * has_any_valid.float() * valid_mask.float()
        return weights

    def forward(
        self,
        current: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_validity: torch.Tensor,
        edge_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            current: ``(B, hidden_dim, P)`` — centre tracks' current
                embedding (the message-passing state).
            neighbor_features: ``(B, hidden_dim, P, K)`` — gathered
                neighbour embeddings.
            neighbor_validity: ``(B, 1, P, K)`` with ``{0, 1}`` entries;
                1 where the neighbour is a real (non-padded) track.
            edge_features: ``(B, edge_dim, P, K)`` pairwise edge
                descriptors, or ``None`` when edges are disabled.

        Returns:
            ``(B, hidden_dim, P)`` — attention-pooled neighbour
            summary, intended as a drop-in replacement for the old
            max-pooled tensor.
        """
        weights = self.compute_weights(
            current, neighbor_features, neighbor_validity, edge_features,
        )
        pooled = (neighbor_features * weights.unsqueeze(1)).sum(dim=-1)
        return pooled
