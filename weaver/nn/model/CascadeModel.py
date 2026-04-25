import torch
import torch.nn as nn


class CascadeModel(nn.Module):
    def __init__(
        self,
        stage1: nn.Module,
        stage2: nn.Module,
        top_k1: int = 600,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.top_k1 = top_k1

        # Stage 1 BNs were built with track_running_stats=False, so eval mode
        # still uses batch stats. Freezing params here is enough.
        for parameter in self.stage1.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _run_stage1(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """points: (B,2,P). features: (B,F,P). lorentz_vectors: (B,4,P). mask: (B,1,P)."""
        scores = self.stage1(points, features, lorentz_vectors, mask)
        selected_indices = self.stage1.select_top_k(scores, mask, self.top_k1)

        def gather_tracks(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            num_channels = tensor.shape[1]
            expanded_indices = indices.unsqueeze(1).expand(-1, num_channels, -1)
            return tensor.gather(2, expanded_indices)

        stage1_scores = scores.gather(1, selected_indices)

        result = {
            'points': gather_tracks(points, selected_indices),
            'features': gather_tracks(features, selected_indices),
            'lorentz_vectors': gather_tracks(lorentz_vectors, selected_indices),
            'mask': gather_tracks(mask, selected_indices),
            'stage1_scores': stage1_scores,
            'selected_indices': selected_indices,
        }
        if track_labels is not None:
            result['track_labels'] = gather_tracks(track_labels, selected_indices)

        return result

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """points: (B,2,P). features: (B,F,P). lorentz_vectors: (B,4,P). mask: (B,1,P)."""
        filtered = self._run_stage1(points, features, lorentz_vectors, mask)
        return self.stage2(
            filtered['points'],
            filtered['features'],
            filtered['lorentz_vectors'],
            filtered['mask'],
            filtered['stage1_scores'],
        )

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        filtered = self._run_stage1(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict = self.stage2.compute_loss(
            filtered['points'],
            filtered['features'],
            filtered['lorentz_vectors'],
            filtered['mask'],
            filtered['track_labels'],
            filtered['stage1_scores'],
        )

        filtered_labels = filtered['track_labels'].squeeze(1)
        filtered_mask = filtered['mask'].squeeze(1).bool()
        original_labels = track_labels.squeeze(1)[:, :mask.shape[2]]
        original_mask = mask.squeeze(1).bool()

        gt_in_filtered = (
            (filtered_labels == 1.0) & filtered_mask
        ).sum(dim=1).float()
        gt_in_original = (
            (original_labels == 1.0) & original_mask
        ).sum(dim=1).float()

        has_gt = gt_in_original > 0
        if has_gt.any():
            recall_at_k1 = (
                gt_in_filtered[has_gt] / gt_in_original[has_gt]
            ).mean()
        else:
            recall_at_k1 = torch.tensor(0.0, device=points.device)

        loss_dict['stage1_recall_at_k1'] = recall_at_k1
        return loss_dict
