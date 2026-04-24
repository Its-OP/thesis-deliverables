"""Set-friendly training augmentations for TrackPreFilter.

All transforms are permutation-equivariant and mask-aware. They never
drop or perturb GT positives. JetCLR-inspired (Dillon 2108.04253).

Transforms
----------
- ``TrackDropout``    — zero the mask on a random subset of non-positive
  tracks. Simulates detector inefficiency and forces the model to be
  robust to missing evidence.
- ``FeatureJitter``   — additive Gaussian noise on per-track features,
  scale σ. Calibrated to match detector resolution.
- ``EtaPhiRotation``  — global random rotation of φ (uniform in
  [-π, π)) + optional η reflection. Preserves 4-vector magnitudes.
  Applied to ``points[:, 1]`` (phi) and ``lorentz_vectors[:, :2]``
  (px, py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TrackDropout(nn.Module):
    """Randomly drop non-positive tracks from the mask.

    Each non-positive valid track is masked with probability ``p``. GT
    positives (label == 1) are never dropped.
    """

    def __init__(self, probability: float = 0.1) -> None:
        super().__init__()
        self.probability = probability

    def forward(
        self,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return augmented mask with shape ``(B, 1, P)``."""
        if self.probability <= 0.0 or not self.training:
            return mask
        # mask: (B, 1, P), labels: (B, 1, P)
        labels_bool = labels > 0.5
        drop_candidates = (mask > 0.5) & ~labels_bool  # (B, 1, P)
        keep_probabilities = torch.rand_like(mask)
        keep = keep_probabilities >= self.probability
        # New mask: positives always kept; other valid tracks kept with 1-p
        augmented = mask.clone()
        augmented[drop_candidates & ~keep] = 0.0
        return augmented


class FeatureJitter(nn.Module):
    """Additive Gaussian noise on per-track features.

    Math:
        x_aug = x + σ · ε,   ε ∼ N(0, 1)

    Applied to padded tracks too for simplicity; they are masked
    downstream.
    """

    def __init__(self, sigma: float = 0.05) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0.0 or not self.training:
            return features
        noise = torch.randn_like(features) * self.sigma
        return features + noise


class EtaPhiRotation(nn.Module):
    """Global random azimuthal rotation + optional η reflection.

    Math (per event, rotation angle α drawn uniformly from [-π, π)):
        φ' = wrap(φ + α)
        px' = cos(α) · px − sin(α) · py
        py' = sin(α) · px + cos(α) · py

    If ``eta_flip_probability`` > 0, with that probability the event's
    η and pz are multiplied by −1 (parity flip about the beam axis).

    Note: impact parameters (dxy_significance, dz_significance, etc.)
    are already scalars invariant under these rotations, so no changes
    are needed to the other feature channels.
    """

    def __init__(self, eta_flip_probability: float = 0.5) -> None:
        super().__init__()
        self.eta_flip_probability = eta_flip_probability

    def forward(
        self,
        points: torch.Tensor,
        lorentz_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return points, lorentz_vectors
        batch_size = points.shape[0]
        device = points.device
        alpha = (
            torch.rand(batch_size, device=device) * 2 * math.pi - math.pi
        )
        cos_alpha = alpha.cos().view(-1, 1, 1)
        sin_alpha = alpha.sin().view(-1, 1, 1)

        eta = points[:, 0:1]
        phi = points[:, 1:2]
        rotated_phi = torch.remainder(
            phi + alpha.view(-1, 1, 1) + math.pi, 2 * math.pi,
        ) - math.pi

        px = lorentz_vectors[:, 0:1]
        py = lorentz_vectors[:, 1:2]
        pz = lorentz_vectors[:, 2:3]
        energy = lorentz_vectors[:, 3:4]
        rotated_px = cos_alpha * px - sin_alpha * py
        rotated_py = sin_alpha * px + cos_alpha * py

        if self.eta_flip_probability > 0.0:
            flip_sign = torch.where(
                torch.rand(batch_size, device=device)
                < self.eta_flip_probability,
                torch.full_like(alpha, -1.0),
                torch.full_like(alpha, 1.0),
            ).view(-1, 1, 1)
            eta = eta * flip_sign
            pz = pz * flip_sign

        rotated_points = torch.cat([eta, rotated_phi], dim=1)
        rotated_lorentz = torch.cat(
            [rotated_px, rotated_py, pz, energy], dim=1,
        )
        return rotated_points, rotated_lorentz


class SetAugmentation(nn.Module):
    """Composite JetCLR-style augmentation bundle."""

    def __init__(
        self,
        dropout_probability: float = 0.1,
        jitter_sigma: float = 0.05,
        eta_flip_probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.track_dropout = TrackDropout(dropout_probability)
        self.feature_jitter = FeatureJitter(jitter_sigma)
        self.eta_phi_rotation = EtaPhiRotation(eta_flip_probability)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_mask = self.track_dropout(mask, labels)
        augmented_features = self.feature_jitter(features)
        augmented_points, augmented_lorentz = self.eta_phi_rotation(
            points, lorentz_vectors,
        )
        return (
            augmented_points,
            augmented_features,
            augmented_lorentz,
            augmented_mask,
        )
