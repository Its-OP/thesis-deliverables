import torch
import torch.nn as nn


class CoupleDumpModel(nn.Module):
    """Dump-fed twin of ``CoupleCascadeModel``: same couple-input construction
    and ``compute_loss`` contract, but consumes precomputed cascade dumps
    instead of running the frozen Stage 1 + Stage 2 in every batch."""

    def __init__(
        self,
        couple_reranker: nn.Module,
        top_k2: int = 50,
        k_values_tracks: tuple[int, ...] = (30, 50, 75, 100, 200),
    ):
        super().__init__()
        self.couple_reranker = couple_reranker
        self.top_k2 = top_k2
        self.k_values_tracks = tuple(k_values_tracks)

    def _build_couple_inputs(
        self, batch: dict[str, torch.Tensor], with_metrics: bool = True,
    ) -> dict[str, torch.Tensor]:
        """batch: features (B,32,K1), points (B,26,K1), lorentz (B,4,K1),
        stage1_scores/stage2_scores/labels (B,K1), original_indices (B,K1)
        long, cone_points (B,3,P), cone_lorentz (B,4,P),
        cone_valid_mask (B,P) bool."""
        # Lazy import keeps the weaver-side module free of the part-side dep.
        from utils.couple_features import build_couple_features_batched

        with torch.no_grad():
            stage2_scores = batch['stage2_scores']
            # Padded K1 slots carry -inf Stage 2 scores in the dump; the
            # finite-score mask doubles as the validity mask.
            valid_in_k1 = torch.isfinite(stage2_scores)
            safe_stage2_scores = stage2_scores.masked_fill(~valid_in_k1, -1e9)
            top_k2_in_k1 = safe_stage2_scores.topk(self.top_k2, dim=1).indices

            def gather_along_track_dim(tensor: torch.Tensor) -> torch.Tensor:
                num_channels = tensor.shape[1]
                expanded_indices = top_k2_in_k1.unsqueeze(1).expand(
                    -1, num_channels, -1,
                )
                return tensor.gather(2, expanded_indices)

            track_valid_mask = valid_in_k1.gather(1, top_k2_in_k1)
            zero = torch.zeros(
                1, device=stage2_scores.device, dtype=stage2_scores.dtype,
            )
            # torch.where, NOT multiplication: -inf * 0 = NaN.
            top_k2_stage1_scores = torch.where(
                track_valid_mask,
                batch['stage1_scores'].gather(1, top_k2_in_k1),
                zero,
            )
            top_k2_stage2_scores = torch.where(
                track_valid_mask,
                stage2_scores.gather(1, top_k2_in_k1),
                zero,
            )
            top_k2_track_labels = batch['labels'].gather(1, top_k2_in_k1)
            member_full_indices = batch['original_indices'].gather(
                1, top_k2_in_k1,
            )

            couple_inputs = build_couple_features_batched(
                top_k2_features=gather_along_track_dim(batch['features']),
                top_k2_points=gather_along_track_dim(batch['points']),
                top_k2_lorentz=gather_along_track_dim(batch['lorentz']),
                top_k2_stage1_scores=top_k2_stage1_scores,
                top_k2_stage2_scores=top_k2_stage2_scores,
                full_points=batch['cone_points'],
                full_lorentz=batch['cone_lorentz'],
                full_valid_mask=batch['cone_valid_mask'],
                member_full_indices=member_full_indices,
                top_k2_track_labels=top_k2_track_labels,
                track_valid_mask=track_valid_mask,
                precomputed_cone=batch.get('precomputed_cone'),
            )
            couple_inputs['member_full_indices'] = member_full_indices

            if with_metrics:
                couple_inputs['n_gt_in_top_k1'] = (
                    batch['labels'] * valid_in_k1.float()
                ).sum(dim=1)

                sorted_stage2_indices = torch.argsort(
                    safe_stage2_scores, dim=1, descending=True,
                )
                gt_in_k1_mask = (batch['labels'] > 0.5) & valid_in_k1
                sorted_gt_in_k1 = gt_in_k1_mask.gather(
                    1, sorted_stage2_indices,
                )
                max_k = sorted_gt_in_k1.shape[1]
                n_gt_in_top_k_tracks_columns = []
                for k_tracks in self.k_values_tracks:
                    effective_k = min(k_tracks, max_k)
                    n_gt_in_top_k_tracks_columns.append(
                        sorted_gt_in_k1[:, :effective_k].sum(dim=1),
                    )
                couple_inputs['n_gt_in_top_k_tracks'] = torch.stack(
                    n_gt_in_top_k_tracks_columns, dim=1,
                )
        return couple_inputs

    @torch.no_grad()
    def build_cone_cache(
        self,
        dataset,
        batch_size: int,
        device: torch.device,
        num_workers: int = 4,
    ) -> torch.Tensor:
        """dataset: CoupleDumpDataset (without a cache attached). Returns
        (num_events, 4, n_couples) float16 cpu — the companion-cone block
        [count, sum_pt, min_dr, has_companion], deterministic per
        (event, K2), computed once so training batches skip it."""
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, collate_fn=dataset.collate,
        )
        cache: torch.Tensor | None = None
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            couple_inputs = self._build_couple_inputs(
                batch, with_metrics=False,
            )
            # h6 block tail layout: [.., cone(4), sv(3), has_sv] — the cone
            # block is channels -8:-4 of the couple vector.
            cone_block = couple_inputs['couple_features'][:, -8:-4, :]
            if cache is None:
                cache = torch.empty(
                    len(dataset), 4, cone_block.shape[2],
                    dtype=torch.float16,
                )
            cache[batch['event_index'].cpu()] = (
                cone_block.half().cpu())
        return cache

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores, filter_a_mask). scores: (B, n_couples). filter_a_mask: (B, n_couples) bool."""
        couple_inputs = self._build_couple_inputs(batch)
        scores = self.couple_reranker(couple_inputs['couple_features'])
        return scores, couple_inputs['filter_a_mask']

    def compute_loss(
        self, batch: dict[str, torch.Tensor], with_metrics: bool = True,
    ) -> dict[str, torch.Tensor]:
        couple_inputs = self._build_couple_inputs(
            batch, with_metrics=with_metrics,
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
        if with_metrics:
            loss_dict['_n_gt_in_top_k1'] = couple_inputs['n_gt_in_top_k1']
            loss_dict['_n_gt_in_top_k_tracks'] = (
                couple_inputs['n_gt_in_top_k_tracks']
            )
        return loss_dict
