from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.graph_ops import pairwise_lv_fts

SUPPORTED_PAIR_EXTRA_DIMS = (0, 6, 10)


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        modules = []
        for dim in dims:
            modules.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*modules)

    def forward(self, x):
        """x: (B, C, P). Returns (P, B, embed_dim)."""
        if self.input_bn is not None:
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        mode='sum',
        normalize_input=True,
        activation='gelu',
        eps=1e-8,
    ):
        super().__init__()
        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.pairwise_lv_fts = partial(
            pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps,
        )
        self.out_dim = dims[-1]

        def make_block(in_dim):
            modules = [nn.BatchNorm1d(in_dim)] if normalize_input else []
            for dim in dims:
                modules.extend([
                    nn.Conv1d(in_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                in_dim = dim
            if use_pre_activation_pair:
                modules = modules[:-1]
            return nn.Sequential(*modules)

        if mode == 'concat':
            self.embed = make_block(pairwise_lv_dim + pairwise_input_dim)
        elif mode == 'sum':
            if pairwise_lv_dim > 0:
                self.embed = make_block(pairwise_lv_dim)
            if pairwise_input_dim > 0:
                self.fts_embed = make_block(pairwise_input_dim)
        else:
            raise RuntimeError("`mode` must be 'sum' or 'concat'")

    def forward(self, x, uu=None):
        """x: (B, 4, P) Lorentz vectors. uu: (B, C, P, P) extra pair features."""
        assert x is not None or uu is not None
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric:
                i, j = torch.tril_indices(
                    seq_len, seq_len,
                    offset=-1 if self.remove_self_pair else 0,
                    device=(x if x is not None else uu).device,
                )
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        idx = torch.arange(0, seq_len, device=x.device)
                        x[:, :, idx, idx] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)
        else:
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric:
            y = torch.zeros(
                batch_size, self.out_dim, seq_len, seq_len,
                dtype=elements.dtype, device=elements.device,
            )
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation='gelu',
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids else None
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """x: (P, B, embed_dim). padding_mask: (B, P). attn_mask: (B*H, P, P)."""
        if x_cls is not None:
            with torch.no_grad():
                padding_mask = torch.cat(
                    (torch.zeros_like(padding_mask[:, :1]), padding_mask),
                    dim=1,
                )
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(
                x, x, x,
                key_padding_mask=padding_mask, attn_mask=attn_mask,
            )[0]

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class CascadeReranker(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        pair_input_dim: int = 4,
        pair_extra_dim: int = 0,
        pair_embed_dims: list[int] | None = None,
        pair_embed_mode: str = 'concat',
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
        loss_mode: str = 'pairwise',
        rs_at_k_target: int = 200,
        rs_at_k_tau1: float = 1.0,
        rs_at_k_tau2: float = 1.0,
    ):
        super().__init__()
        if pair_extra_dim not in SUPPORTED_PAIR_EXTRA_DIMS:
            raise ValueError(
                f'pair_extra_dim={pair_extra_dim} is not supported; '
                f'choose one of {SUPPORTED_PAIR_EXTRA_DIMS}.'
            )
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.pair_extra_dim = pair_extra_dim
        self.loss_mode = loss_mode
        self.rs_at_k_target = rs_at_k_target
        self.rs_at_k_tau1 = rs_at_k_tau1
        self.rs_at_k_tau2 = rs_at_k_tau2
        # hybrid_lambda: pure pairwise until progress=warmup_start, ramp to
        # full LambdaRank by warmup_end (epoch fractions of total training).
        self.lambda_rank_warmup_start = 0.4
        self.lambda_rank_warmup_end = 0.7
        self._training_progress: float = 0.0

        if pair_embed_dims is None:
            pair_embed_dims = [64, 64]

        # +2 input channels: stage1_score, energy-sharing fraction z_pt.
        self.embed = Embed(
            input_dim + 2,
            dims=[embed_dim],
            normalize_input=True,
        )

        self.pair_embed = PairEmbed(
            pairwise_lv_dim=pair_input_dim,
            pairwise_input_dim=pair_extra_dim,
            dims=pair_embed_dims + [num_heads],
            remove_self_pair=False,
            use_pre_activation_pair=True,
            mode=pair_embed_mode,
        )

        block_config = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            attn_dropout=dropout,
            activation_dropout=dropout,
            activation='gelu',
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        self.blocks = nn.ModuleList([
            Block(**block_config) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)
        self.scoring_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        """points: (B,>=2,K1). features: (B,F,K1). lorentz_vectors: (B,4,K1).
        mask: (B,1,K1). stage1_scores: (B,K1). Returns (B,K1) with -inf at padded."""
        valid_mask = mask.squeeze(1).bool()
        padding_mask = ~valid_mask
        mask_float = mask.float()

        # select_top_k pads with -inf; (-inf * 0.0) is NaN in float arithmetic.
        safe_stage1_scores = stage1_scores.masked_fill(~valid_mask, 0.0)
        stage1_channel = safe_stage1_scores.unsqueeze(1)

        px_track = lorentz_vectors[:, 0:1, :]
        py_track = lorentz_vectors[:, 1:2, :]
        pt_track = torch.sqrt(px_track ** 2 + py_track ** 2)
        sum_pt = (pt_track * mask_float).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_pt = (pt_track / sum_pt) * mask_float

        combined_features = torch.cat(
            [features, stage1_channel, z_pt], dim=1,
        ) * mask_float

        track_embeddings = self.embed(combined_features)
        track_embeddings = track_embeddings.masked_fill(
            ~mask.bool().permute(2, 0, 1), 0,
        )

        # .detach(): pairwise_lv_fts has 1/sqrt(0)=inf gradient at ΔR=0
        # (self-pairs). Pair features are physics constants for attention bias —
        # no gradient w.r.t. 4-vectors needed. .float() keeps ln/sqrt in fp32
        # under AMP.
        lorentz_for_pairs = (lorentz_vectors * mask_float).detach().float()

        extra_pairwise = self._compute_extra_pairwise_features(
            points, features, lorentz_for_pairs, mask_float,
        ) if self.pair_extra_dim > 0 else None
        if extra_pairwise is not None and extra_pairwise.shape[1] != self.pair_extra_dim:
            raise ValueError(
                f'pair_extra_dim={self.pair_extra_dim} but the builder '
                f'produced {extra_pairwise.shape[1]} pairwise channels — '
                'builder/config mismatch.'
            )

        attention_bias = self.pair_embed(lorentz_for_pairs, uu=extra_pairwise)
        num_heads = attention_bias.shape[1]
        attention_bias = attention_bias.view(
            -1, num_heads, attention_bias.shape[2], attention_bias.shape[3],
        ).reshape(-1, attention_bias.shape[2], attention_bias.shape[3])

        encoded = track_embeddings
        for block in self.blocks:
            encoded = block(
                encoded,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=attention_bias,
            )

        encoded = self.output_norm(encoded)
        encoded = encoded.permute(1, 0, 2)
        scores = self.scoring_head(encoded).squeeze(-1)
        scores = scores.masked_fill(padding_mask, float('-inf'))
        return scores

    def _compute_extra_pairwise_features(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_for_pairs: torch.Tensor,
        mask_float: torch.Tensor,
    ) -> torch.Tensor:
        """points: (B, >=2, K1) — >=9 when pair_extra_dim=10. features:
        (B, F, K1). lorentz_for_pairs: (B, 4, K1). mask_float: (B, 1, K1).
        Returns (B, pair_extra_dim, K1, K1). Channels 0-5: q_i*q_j, |Δdz_sig|,
        ρ(770) Gaussian, OS×ρ, φ-corrected |Δdxy|, Lorentz dot; dim 10
        appends raw |Δdz|, ln 3D POCA, same-other-PV,
        both-lifetime-positive."""
        pair_mask = mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)

        # charge feature is standardized (center=1.0, scale=0.5); recover raw.
        charge_raw = (features[:, 5:6, :] / 0.5 + 1.0) * mask_float
        charge_product = charge_raw.unsqueeze(-1) * charge_raw.unsqueeze(-2)

        # feature index 7 = track_log_dz_significance.
        dz_sig = features[:, 7:8, :] * mask_float
        dz_diff = (dz_sig.unsqueeze(-1) - dz_sig.unsqueeze(-2)).abs()

        lv = lorentz_for_pairs
        px, py, pz, energy = lv[:, 0:1], lv[:, 1:2], lv[:, 2:3], lv[:, 3:4]
        sum_energy = energy.unsqueeze(-1) + energy.unsqueeze(-2)
        sum_px = px.unsqueeze(-1) + px.unsqueeze(-2)
        sum_py = py.unsqueeze(-1) + py.unsqueeze(-2)
        sum_pz = pz.unsqueeze(-1) + pz.unsqueeze(-2)
        m_squared = (
            sum_energy ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
        ).clamp(min=1e-10)
        m_ij = m_squared.sqrt()
        rho_indicator = torch.exp(-0.5 * ((m_ij - 0.770) / 0.075) ** 2)

        is_opposite_sign = (charge_product < 0).float()
        rho_os_indicator = is_opposite_sign * rho_indicator

        # |Δdxy| / |2 sin(Δφ/2)| removes the φ-dependence for tracks sharing a
        # vertex. Clamp denominator to avoid blow-up for nearly parallel pairs.
        dxy_sig = features[:, 6:7, :] * mask_float
        dxy_diff = (dxy_sig.unsqueeze(-1) - dxy_sig.unsqueeze(-2)).abs()
        phi_raw = points[:, 1:2, :]
        delta_phi = phi_raw.unsqueeze(-1) - phi_raw.unsqueeze(-2)
        delta_phi = (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi
        sin_half_dphi = torch.abs(torch.sin(delta_phi / 2.0))
        dxy_phi_corrected = dxy_diff / (2.0 * sin_half_dphi).clamp(min=0.05)

        lorentz_dot = (
            energy.unsqueeze(-1) * energy.unsqueeze(-2)
            - px.unsqueeze(-1) * px.unsqueeze(-2)
            - py.unsqueeze(-1) * py.unsqueeze(-2)
            - pz.unsqueeze(-1) * pz.unsqueeze(-2)
        )

        channels = [
            charge_product, dz_diff, rho_indicator,
            rho_os_indicator, dxy_phi_corrected, lorentz_dot,
        ]
        if self.pair_extra_dim >= 10:
            if points.shape[1] < 9:
                raise ValueError(
                    f'pair_extra_dim={self.pair_extra_dim} needs the '
                    '9-channel pf_points transport block, got '
                    f'{points.shape[1]} point channels.'
                )
            # point index 2 = signed track_dz (cm): the raw gap beats the
            # significance gap head-to-head (H6 S2.2) and is kept alongside it.
            dz_raw = points[:, 2:3, :].float() * mask_float
            raw_dz_gap = (dz_raw.unsqueeze(-1) - dz_raw.unsqueeze(-2)).abs()
            channels.append(raw_dz_gap)
            channels.extend(self._compute_vertex_pair_channels(
                points, features, px, py,
            ))

        return torch.cat(channels, dim=1) * pair_mask

    def _compute_vertex_pair_channels(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
    ) -> list[torch.Tensor]:
        """points: (B, >=9, K1). features: (B, F, K1). px, py: (B, 1, K1).
        Returns three (B, 1, K1, K1) tensors: ln 3D POCA distance,
        same-other-PV flag, both-lifetime-positive flag."""
        eta = points[:, 0:1, :].float()
        phi = points[:, 1:2, :].float()
        cosh_eta = torch.cosh(eta)
        direction = torch.cat([
            torch.cos(phi) / cosh_eta,
            torch.sin(phi) / cosh_eta,
            torch.tanh(eta),
        ], dim=1)
        reference = points[:, 3:6, :].float()

        # Skew-line closest approach between the linearized tracks; parallel
        # pairs (denominator ~ 0, incl. every self-pair on the diagonal) fall
        # back to the point-to-line distance. torch.where evaluates both
        # branches, so denominators are made safe before dividing.
        direction_i = direction.unsqueeze(-1)
        direction_j = direction.unsqueeze(-2)
        separation = reference.unsqueeze(-1) - reference.unsqueeze(-2)
        direction_i_squared = (
            direction_i * direction_i).sum(dim=1, keepdim=True)
        direction_j_squared = (
            direction_j * direction_j).sum(dim=1, keepdim=True)
        direction_dot = (direction_i * direction_j).sum(dim=1, keepdim=True)
        separation_dot_i = (direction_i * separation).sum(dim=1, keepdim=True)
        separation_dot_j = (direction_j * separation).sum(dim=1, keepdim=True)

        denominator = (
            direction_i_squared * direction_j_squared - direction_dot ** 2
        )
        parallel = denominator < 1e-12
        safe_denominator = torch.where(
            parallel, torch.ones_like(denominator), denominator)
        safe_direction_j_squared = torch.where(
            direction_j_squared > 0,
            direction_j_squared, torch.ones_like(direction_j_squared))
        parameter_i = torch.where(
            parallel, torch.zeros_like(denominator),
            (direction_dot * separation_dot_j
             - direction_j_squared * separation_dot_i) / safe_denominator)
        parameter_j = torch.where(
            parallel, separation_dot_j / safe_direction_j_squared,
            (direction_i_squared * separation_dot_j
             - direction_dot * separation_dot_i) / safe_denominator)
        closest_gap = (
            separation + parameter_i * direction_i - parameter_j * direction_j
        )
        poca_distance = closest_gap.square().sum(dim=1, keepdim=True).sqrt()
        # ln: POCA spans µm (GT-GT) to meters and PairEmbed opens with a
        # plain BatchNorm1d; the log keeps the discriminative decades apart.
        ln_poca = torch.log(poca_distance + 1e-6)

        # point index 6 = nearest-other-PV index; feature index 24 =
        # track_closer_to_other_pv (null-standardized 0/1). Together they
        # reproduce the H6 nearest-vertex test over [PV] + OtherPV.
        nearest_index = points[:, 6:7, :]
        closer_flag = features[:, 24:25, :]
        same_index = (
            nearest_index.unsqueeze(-1) == nearest_index.unsqueeze(-2))
        both_closer = (
            (closer_flag.unsqueeze(-1) > 0.5)
            & (closer_flag.unsqueeze(-2) > 0.5))
        same_other_pv = (same_index & both_closer).float()

        # point indices 7-8 = transverse PCA displacement w.r.t. the stored
        # PV; sign each track's impact parameter by its projection on the
        # pair momentum axis (b-tagging convention).
        displacement_x = points[:, 7:8, :].float()
        displacement_y = points[:, 8:9, :].float()
        magnitude = torch.hypot(displacement_x, displacement_y)
        axis_px = px.unsqueeze(-1) + px.unsqueeze(-2)
        axis_py = py.unsqueeze(-1) + py.unsqueeze(-2)
        projection_i = (
            displacement_x.unsqueeze(-1) * axis_px
            + displacement_y.unsqueeze(-1) * axis_py)
        projection_j = (
            displacement_x.unsqueeze(-2) * axis_px
            + displacement_y.unsqueeze(-2) * axis_py)
        positive_i = (projection_i >= 0) & (magnitude.unsqueeze(-1) > 0)
        positive_j = (projection_j >= 0) & (magnitude.unsqueeze(-2) > 0)
        both_lifetime_positive = (positive_i & positive_j).float()

        return [ln_poca, same_other_pv, both_lifetime_positive]

    def set_training_progress(self, progress: float) -> None:
        self._training_progress = max(0.0, min(1.0, progress))

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        valid_mask = mask.squeeze(1).bool()
        labels = (
            track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()
        )

        if self.loss_mode == 'rs_at_k':
            loss_dict = self._rs_at_k_loss(scores, labels, valid_mask)
        elif self.loss_mode == 'hybrid_lambda':
            loss_dict = self._hybrid_lambda_loss(scores, labels, valid_mask)
        else:
            loss_dict = self._pairwise_ranking_loss(scores, labels, valid_mask)

        loss_dict['_scores'] = scores
        return loss_dict

    def _pairwise_ranking_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pairwise softplus, optionally weighted by |ΔR@K| if loss_mode=lambda_rank."""
        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        k_boundary = self.rs_at_k_target
        event_losses = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = labels[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels == 0.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=scores.device,
            )
            sampled_negatives = negative_indices[sample_idx]

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)

            if self.loss_mode == 'lambda_rank':
                with torch.no_grad():
                    masked_scores = event_scores.clone()
                    masked_scores[~event_valid] = float('-inf')
                    ranks = torch.argsort(
                        torch.argsort(masked_scores, descending=True),
                    )
                    pos_ranks = ranks[positive_indices]
                    neg_ranks = ranks[sampled_negatives]
                    pos_in_topk = (pos_ranks < k_boundary).float().unsqueeze(1)
                    neg_in_topk = (neg_ranks < k_boundary).float().unsqueeze(0)
                    n_gt = max(1, len(positive_indices))
                    lambda_weights = (
                        (pos_in_topk * (1.0 - neg_in_topk))
                        + ((1.0 - pos_in_topk) * neg_in_topk)
                    ) / n_gt

                pairwise_loss = pairwise_loss * lambda_weights

            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            ranking_loss = torch.tensor(
                0.0, device=scores.device, dtype=scores.dtype,
            )
        else:
            ranking_loss = torch.stack(event_losses).mean()

        return {'total_loss': ranking_loss, 'ranking_loss': ranking_loss}

    def _rs_at_k_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """RS@K = (1/|P|) Σ_p σ_τ1(K − Σ_n σ_τ2(s_n − s_p)). Loss = 1 − RS@K."""
        batch_size = scores.shape[0]
        k_target = self.rs_at_k_target
        tau1 = self.rs_at_k_tau1
        tau2 = self.rs_at_k_tau2
        event_losses = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = labels[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels == 0.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            pos_scores = event_scores[positive_indices]
            neg_scores = event_scores[negative_indices]

            score_diffs = neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
            soft_rank_contributions = torch.sigmoid(score_diffs / tau2)
            soft_ranks = soft_rank_contributions.sum(dim=1)
            in_top_k = torch.sigmoid((k_target - soft_ranks) / tau1)
            event_losses.append(1.0 - in_top_k.mean())

        if not event_losses:
            rs_loss = torch.tensor(
                0.0, device=scores.device, dtype=scores.dtype,
            )
        else:
            rs_loss = torch.stack(event_losses).mean()

        return {
            'total_loss': rs_loss,
            'ranking_loss': rs_loss,
            'rs_at_k_loss': rs_loss,
        }

    def _hybrid_lambda_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pairwise + LambdaRank with α ramping 0→3 between warmup_start and warmup_end.
        Pairwise weight decays 1.0→0.5; LambdaRank weight grows 0→3 (lambda dominates late)."""
        start = self.lambda_rank_warmup_start
        end = self.lambda_rank_warmup_end
        progress = self._training_progress

        if progress <= start:
            alpha = 0.0
        elif progress >= end:
            alpha = 3.0
        else:
            alpha = 3.0 * (progress - start) / (end - start)

        original_mode = self.loss_mode

        self.loss_mode = 'pairwise'
        pairwise_dict = self._pairwise_ranking_loss(scores, labels, valid_mask)
        self.loss_mode = 'lambda_rank'
        lambda_dict = self._pairwise_ranking_loss(scores, labels, valid_mask)
        self.loss_mode = original_mode

        pairwise_loss = pairwise_dict['ranking_loss']
        lambda_loss = lambda_dict['ranking_loss']
        pairwise_weight = 1.0 - alpha / 6.0
        combined = pairwise_weight * pairwise_loss + alpha * lambda_loss

        return {
            'total_loss': combined,
            'ranking_loss': combined,
            'pairwise_loss': pairwise_loss,
            'lambda_rank_loss': lambda_loss,
            'lambda_alpha': torch.tensor(alpha, device=scores.device),
        }
