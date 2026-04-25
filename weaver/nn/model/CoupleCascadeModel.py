import torch
import torch.nn as nn


class CoupleCascadeModel(nn.Module):
    def __init__(
        self,
        cascade: nn.Module,
        couple_reranker: nn.Module,
        top_k2: int = 50,
        k_values_tracks: tuple[int, ...] = (30, 50, 75, 100, 200),
    ):
        super().__init__()
        self.cascade = cascade
        self.couple_reranker = couple_reranker
        self.top_k2 = top_k2
        self.k_values_tracks = tuple(k_values_tracks)

        # Cascade is frozen; trainer optimizer must filter requires_grad.
        for parameter in self.cascade.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _run_cascade_to_top_k2(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """points: (B,2,P). features: (B,F,P). lorentz_vectors: (B,4,P). mask: (B,1,P)."""
        filtered = self.cascade._run_stage1(
            points, features, lorentz_vectors, mask, track_labels,
        )

        stage2_scores = self.cascade.stage2(
            filtered['points'],
            filtered['features'],
            filtered['lorentz_vectors'],
            filtered['mask'],
            filtered['stage1_scores'],
        )

        filtered_labels_flat = filtered['track_labels'].squeeze(1) > 0.5
        filtered_mask_flat = filtered['mask'].squeeze(1) > 0.5
        gt_in_k1_mask = filtered_labels_flat & filtered_mask_flat
        n_gt_in_top_k1 = gt_in_k1_mask.sum(dim=1)

        sorted_stage2_indices = torch.argsort(
            stage2_scores, dim=1, descending=True,
        )
        sorted_gt_in_k1 = gt_in_k1_mask.gather(1, sorted_stage2_indices)
        max_k = sorted_gt_in_k1.shape[1]
        n_gt_in_top_k_tracks_columns = []
        for k_tracks in self.k_values_tracks:
            effective_k = min(k_tracks, max_k)
            n_gt_in_top_k_tracks_columns.append(
                sorted_gt_in_k1[:, :effective_k].sum(dim=1),
            )
        n_gt_in_top_k_tracks = torch.stack(n_gt_in_top_k_tracks_columns, dim=1)

        top_k2_in_k1 = stage2_scores.topk(self.top_k2, dim=1).indices

        def gather_along_track_dim(tensor: torch.Tensor) -> torch.Tensor:
            num_channels = tensor.shape[1]
            expanded_indices = top_k2_in_k1.unsqueeze(1).expand(-1, num_channels, -1)
            return tensor.gather(2, expanded_indices)

        # Padding tracks get -inf Stage 2 scores, so they sort to tail; mask
        # them so the couple feature builder excludes those couples.
        top_k2_stage2_scores = stage2_scores.gather(1, top_k2_in_k1)
        track_valid_mask = torch.isfinite(top_k2_stage2_scores)

        return {
            'features': gather_along_track_dim(filtered['features']),
            'points': gather_along_track_dim(filtered['points']),
            'lorentz_vectors': gather_along_track_dim(filtered['lorentz_vectors']),
            'stage1_scores': filtered['stage1_scores'].gather(1, top_k2_in_k1),
            'stage2_scores': top_k2_stage2_scores,
            'track_labels': filtered['track_labels'].squeeze(1).gather(1, top_k2_in_k1),
            'track_valid_mask': track_valid_mask,
            'n_gt_in_top_k1': n_gt_in_top_k1,
            'n_gt_in_top_k_tracks': n_gt_in_top_k_tracks,
        }

    def _build_couple_inputs(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Lazy import keeps the weaver-side module free of the part-side dep.
        from utils.couple_features import build_couple_features_batched

        top_k2_data = self._run_cascade_to_top_k2(
            points, features, lorentz_vectors, mask, track_labels,
        )
        couple_inputs = build_couple_features_batched(
            top_k2_features=top_k2_data['features'],
            top_k2_points=top_k2_data['points'],
            top_k2_lorentz=top_k2_data['lorentz_vectors'],
            top_k2_stage1_scores=top_k2_data['stage1_scores'],
            top_k2_stage2_scores=top_k2_data['stage2_scores'],
            top_k2_track_labels=top_k2_data['track_labels'],
            track_valid_mask=top_k2_data['track_valid_mask'],
        )
        couple_inputs['n_gt_in_top_k1'] = top_k2_data['n_gt_in_top_k1']
        couple_inputs['n_gt_in_top_k_tracks'] = top_k2_data['n_gt_in_top_k_tracks']
        return couple_inputs

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores, filter_a_mask). scores: (B, n_couples). filter_a_mask: (B, n_couples) bool."""
        dummy_track_labels = torch.zeros_like(mask)
        couple_inputs = self._build_couple_inputs(
            points, features, lorentz_vectors, mask, dummy_track_labels,
        )
        scores = self.couple_reranker(couple_inputs['couple_features'])
        return scores, couple_inputs['filter_a_mask']

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        couple_inputs = self._build_couple_inputs(
            points, features, lorentz_vectors, mask, track_labels,
        )
        couple_features = couple_inputs['couple_features']
        couple_labels = couple_inputs['couple_labels']
        filter_a_mask = couple_inputs['filter_a_mask']

        loss_dict = self.couple_reranker.compute_loss(
            couple_features=couple_features,
            couple_labels=couple_labels.to(couple_features.dtype),
            couple_mask=filter_a_mask.to(couple_features.dtype),
        )
        loss_dict['_couple_labels'] = couple_labels
        loss_dict['_couple_mask'] = filter_a_mask
        loss_dict['_n_gt_in_top_k1'] = couple_inputs['n_gt_in_top_k1']
        loss_dict['_n_gt_in_top_k_tracks'] = couple_inputs['n_gt_in_top_k_tracks']
        return loss_dict
