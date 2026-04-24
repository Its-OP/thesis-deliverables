from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.graph_ops import (
    cross_set_gather,
    cross_set_knn,
    pairwise_lv_fts,
)
from weaver.nn.model.prefilter_expressiveness import PerFeatureEmbedding
from weaver.nn.model.prefilter_losses import (
    infonce_in_event,
    listwise_ce_loss,
    logit_adjust_offset,
)


class TrackPreFilter(nn.Module):
    def __init__(
        self,
        mode: str = 'mlp',
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_neighbors: int = 16,
        ranking_num_samples: int = 20,
        num_message_rounds: int = 1,
        denoising_sigma_start: float = 0.3,
        denoising_sigma_end: float = 0.3,
        ranking_temperature_start: float = 1.0,
        ranking_temperature_end: float = 1.0,
        drw_warmup_fraction: float = 1.0,
        drw_positive_weight: float = 1.0,
        aggregation_mode: str = 'max',
        dropout: float = 0.0,
        use_edge_features: bool = False,
        loss_type: str = 'pairwise',
        logit_adjust_tau: float = 1.0,
        listwise_temperature: float = 1.0,
        clustering_dim: int = 8,
        feature_embed_mode: str = 'none',
        feature_embed_dim: int = 32,
    ):
        super().__init__()
        if mode != 'mlp':
            raise ValueError(f"Only mode='mlp' is supported, got {mode!r}.")
        if aggregation_mode != 'max':
            raise ValueError(
                f"Only aggregation_mode='max' is supported, got {aggregation_mode!r}.",
            )
        if loss_type not in ('pairwise', 'listwise_ce', 'infonce', 'logit_adjust'):
            raise ValueError(f"Unknown loss_type {loss_type!r}.")
        if feature_embed_mode not in ('none', 'per_feature'):
            raise ValueError(
                f"feature_embed_mode must be 'none' or 'per_feature', "
                f"got {feature_embed_mode!r}",
            )

        self.mode = mode
        self.input_dim = input_dim
        self.num_neighbors = num_neighbors
        self.ranking_num_samples = ranking_num_samples
        self.num_message_rounds = num_message_rounds
        self.dropout = dropout

        self.denoising_sigma_start = denoising_sigma_start
        self.denoising_sigma_end = denoising_sigma_end
        self.ranking_temperature_start = ranking_temperature_start
        self.ranking_temperature_end = ranking_temperature_end
        self._temperature_progress: float = 0.0

        self.drw_warmup_fraction = drw_warmup_fraction
        self.drw_positive_weight = drw_positive_weight
        self._drw_active: bool = False

        self.aggregation_mode = aggregation_mode
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = 4 if use_edge_features else 0

        self.loss_type = loss_type
        self.logit_adjust_tau = logit_adjust_tau
        self.listwise_temperature = listwise_temperature
        self.clustering_dim = clustering_dim

        self.feature_embed_mode = feature_embed_mode
        if feature_embed_mode == 'per_feature':
            self.feature_embedder = PerFeatureEmbedding(
                num_features=input_dim,
                embed_dim=feature_embed_dim,
            )
            track_mlp_input_dim = input_dim * feature_embed_dim
        else:
            self.feature_embedder = None
            track_mlp_input_dim = input_dim

        # Dropout inserted only when > 0 so state_dict keys for
        # `track_mlp.*`, `neighbor_mlps.*.*`, `scorer.*` stay stable across
        # the zero-dropout baseline (no Sequential index shift).
        use_dropout = dropout > 0

        track_mlp_layers: list[nn.Module] = [
            nn.Conv1d(track_mlp_input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        if use_dropout:
            track_mlp_layers.append(nn.Dropout(p=dropout))
        track_mlp_layers += [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        if use_dropout:
            track_mlp_layers.append(nn.Dropout(p=dropout))
        self.track_mlp = nn.Sequential(*track_mlp_layers)

        # Standard max-pool aggregation: cat([current, max_pooled]) = 2H.
        neighbor_input_dim = 2 * hidden_dim + self.edge_feature_dim

        def _build_neighbor_mlp() -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv1d(neighbor_input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
            if use_dropout:
                layers.append(nn.Dropout(p=dropout))
            return nn.Sequential(*layers)

        self.neighbor_mlps = nn.ModuleList(
            _build_neighbor_mlp() for _ in range(num_message_rounds)
        )

        scorer_layers: list[nn.Module] = [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        if use_dropout:
            scorer_layers.append(nn.Dropout(p=dropout))
        scorer_layers.append(nn.Conv1d(hidden_dim, 1, kernel_size=1))
        self.scorer = nn.Sequential(*scorer_layers)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """points: (B, 2, P) (eta, phi). features: (B, F, P). lorentz_vectors: (B, 4, P). mask: (B, 1, P). Returns (B, P); padded tracks -inf."""
        valid_mask = mask.squeeze(1).bool()
        scores = self._forward_mlp(points, features, lorentz_vectors, mask)
        return scores.masked_fill(~valid_mask, float('-inf'))

    def _forward_mlp(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.float()

        features_for_mlp = (
            self.feature_embedder(features)
            if self.feature_embedder is not None
            else features
        )
        track_embedding = self.track_mlp(features_for_mlp) * mask_float

        with torch.no_grad():
            neighbor_indices = cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=self.num_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )

        edge_max_pooled = (
            self._compute_edge_max_pooled(lorentz_vectors, mask_float, neighbor_indices)
            if self.use_edge_features
            else None
        )

        # Gather neighbour validity once — invariant across rounds.
        neighbor_validity = cross_set_gather(mask_float, neighbor_indices)
        neighbor_invalid = neighbor_validity == 0

        current = track_embedding
        for round_index in range(self.num_message_rounds):
            neighbor_features = cross_set_gather(current, neighbor_indices)
            masked = neighbor_features.masked_fill(neighbor_invalid, float('-inf'))
            pooled = masked.max(dim=-1)[0]
            # Guard all-invalid rows: keep zeros instead of leaking -inf into the MLP.
            pooled = torch.where(
                torch.isfinite(pooled), pooled, pooled.new_zeros(()),
            )

            if edge_max_pooled is not None:
                aggregated = torch.cat([current, pooled, edge_max_pooled], dim=1)
            else:
                aggregated = torch.cat([current, pooled], dim=1)

            current = self.neighbor_mlps[round_index](aggregated) * mask_float

        return self.scorer(current).squeeze(1)

    def _compute_edge_max_pooled(
        self,
        lorentz_vectors: torch.Tensor,
        mask_float: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, 4, P) pairwise_lv_fts (ln kT, ln z, ln ΔR, ln m²) max-pooled over kNN."""
        neighbor_lorentz = cross_set_gather(lorentz_vectors, neighbor_indices)
        neighbor_validity = cross_set_gather(mask_float, neighbor_indices)
        center_lorentz = lorentz_vectors.unsqueeze(-1).expand_as(neighbor_lorentz)

        # amp off + detach + fp32: avoids sqrt(ΔR²) backward NaN on near-collinear edges.
        with torch.amp.autocast('cuda', enabled=False):
            lv_features = pairwise_lv_fts(
                center_lorentz.detach().float(),
                neighbor_lorentz.detach().float(),
                num_outputs=4,
            )
        lv_features = lv_features.to(lorentz_vectors.dtype)
        lv_for_max = lv_features.masked_fill(neighbor_validity == 0, float('-inf'))
        lv_max_pooled = lv_for_max.max(dim=-1)[0]
        return lv_max_pooled.masked_fill(lv_max_pooled == float('-inf'), 0.0)

    def set_temperature_progress(self, progress: float) -> None:
        """progress in [0, 1]; 0 = start, 1 = end."""
        self._temperature_progress = max(0.0, min(1.0, progress))

    @property
    def current_denoising_sigma(self) -> float:
        return (
            self.denoising_sigma_start
            + self._temperature_progress
            * (self.denoising_sigma_end - self.denoising_sigma_start)
        )

    @property
    def current_ranking_temperature(self) -> float:
        return (
            self.ranking_temperature_start
            + self._temperature_progress
            * (self.ranking_temperature_end - self.ranking_temperature_start)
        )

    def set_drw_active(self, active: bool) -> None:
        self._drw_active = active

    def _ranking_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == 'listwise_ce':
            return listwise_ce_loss(
                scores, labels, valid_mask,
                temperature=self.listwise_temperature,
            )
        if self.loss_type == 'infonce':
            return infonce_in_event(
                scores, labels, valid_mask,
                temperature=self.listwise_temperature,
            )

        batch_size = scores.shape[0]
        temperature = self.current_ranking_temperature
        event_losses = []

        for event_index in range(batch_size):
            event_labels = labels[event_index]
            event_scores = scores[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = ((event_labels == 1.0) & event_valid).nonzero(as_tuple=True)[0]
            negative_indices = ((event_labels == 0.0) & event_valid).nonzero(as_tuple=True)[0]
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,), device=scores.device,
            )
            sampled_negatives = negative_indices[sample_idx]

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            if self.loss_type == 'logit_adjust':
                offset = logit_adjust_offset(
                    num_positives=len(positive_indices),
                    num_negatives=len(negative_indices),
                    tau=self.logit_adjust_tau,
                )
                if offset != 0.0:
                    negative_scores = negative_scores + offset

            # L = T * softplus((s_neg - s_pos) / T)
            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)

            if self._drw_active:
                pairwise_loss = pairwise_loss * self.drw_positive_weight

            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        return torch.stack(event_losses).mean()

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        use_contrastive_denoising: bool = True,
    ) -> dict[str, torch.Tensor]:
        scores = self.forward(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        labels_flat = (
            track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()
        )

        ranking_loss = self._ranking_loss(scores, labels_flat, valid_mask)
        total_loss = ranking_loss
        loss_dict: dict[str, torch.Tensor] = {'ranking_loss': ranking_loss}

        if use_contrastive_denoising and self.training:
            denoising_loss = self._contrastive_denoising_loss(
                points, features, lorentz_vectors, mask, track_labels, scores,
            )
            total_loss = total_loss + 0.5 * denoising_loss
            loss_dict['denoising_loss'] = denoising_loss

        loss_dict['total_loss'] = total_loss
        loss_dict['_scores'] = scores
        return loss_dict

    def _contrastive_denoising_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = mask.squeeze(1).bool()
        labels_flat = (
            track_labels.squeeze(1)[:, :valid_mask.shape[1]] * valid_mask.float()
        )

        gt_mask = (labels_flat == 1.0) & valid_mask
        if not gt_mask.any():
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        positive_noise = torch.randn_like(features) * self.current_denoising_sigma
        gt_mask_expanded = gt_mask.unsqueeze(1)
        positive_noised_features = torch.where(
            gt_mask_expanded, features + positive_noise, features,
        )
        positive_noised_scores = self.forward(
            points, positive_noised_features, lorentz_vectors, mask,
        )

        batch_size = features.shape[0]
        temperature = self.current_ranking_temperature
        event_losses = []

        for event_index in range(batch_size):
            gt_positions = gt_mask[event_index].nonzero(as_tuple=True)[0]
            if len(gt_positions) == 0:
                continue

            pos_scores = positive_noised_scores[event_index, gt_positions]

            negative_indices = (
                (labels_flat[event_index] == 0.0) & valid_mask[event_index]
            ).nonzero(as_tuple=True)[0]
            if len(negative_indices) == 0:
                continue

            num_samples = min(20, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,), device=features.device,
            )
            bg_scores = original_scores[event_index, negative_indices[sample_idx]]

            positive_pairwise = temperature * functional.softplus(
                (bg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)) / temperature,
            )
            event_losses.append(positive_pairwise.mean())

        if not event_losses:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        return torch.stack(event_losses).mean()

    def select_top_k(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        top_k: int = 200,
    ) -> torch.Tensor:
        """scores: (B, P). mask: (B, 1, P). Returns (B, min(top_k, P)) long."""
        valid_mask = mask.squeeze(1).bool()
        masked_scores = scores.clone()
        masked_scores[~valid_mask] = float('-inf')
        actual_k = min(top_k, scores.shape[1])
        _, top_indices = masked_scores.topk(actual_k, dim=1)
        return top_indices

    def filter_tracks(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        top_k: int = 200,
    ) -> dict[str, torch.Tensor]:
        scores = self.forward(points, features, lorentz_vectors, mask)
        selected_indices = self.select_top_k(scores, mask, top_k)

        def gather_tracks(tensor, indices):
            num_channels = tensor.shape[1]
            expanded = indices.unsqueeze(1).expand(-1, num_channels, -1)
            return tensor.gather(2, expanded)

        return {
            'points': gather_tracks(points, selected_indices),
            'features': gather_tracks(features, selected_indices),
            'lorentz_vectors': gather_tracks(lorentz_vectors, selected_indices),
            'mask': gather_tracks(mask, selected_indices),
            'track_labels': gather_tracks(track_labels, selected_indices),
        }
