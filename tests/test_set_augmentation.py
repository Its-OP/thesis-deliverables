"""Tests for part.utils.set_augmentation."""
from __future__ import annotations

import math

import torch

from utils.set_augmentation import (
    EtaPhiRotation,
    FeatureJitter,
    SetAugmentation,
    TrackDropout,
)


BATCH_SIZE = 4
NUM_TRACKS = 50
INPUT_DIM = 16


def _make_inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    eta = torch.randn(BATCH_SIZE, 1, NUM_TRACKS, generator=generator) * 1.5
    phi = (
        torch.rand(BATCH_SIZE, 1, NUM_TRACKS, generator=generator)
        * 2 * math.pi - math.pi
    )
    points = torch.cat([eta, phi], dim=1)
    features = torch.randn(
        BATCH_SIZE, INPUT_DIM, NUM_TRACKS, generator=generator,
    )
    pt = (
        torch.rand(BATCH_SIZE, 1, NUM_TRACKS, generator=generator) * 5 + 0.5
    )
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px * px + py * py + pz * pz + 0.13957 ** 2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -10:] = 0.0
    labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    labels[:, 0, [5, 10, 15]] = 1.0
    return points, features, lorentz_vectors, mask, labels


class TestTrackDropout:
    def test_positives_preserved(self):
        augmentation = TrackDropout(probability=1.0).train()
        _, _, _, mask, labels = _make_inputs()
        augmented = augmentation(mask, labels)
        # All non-positive valid tracks zeroed; positives kept
        positive_indices = (labels > 0.5)
        assert (augmented[positive_indices] == 1.0).all()
        # Some non-positives should be dropped
        assert augmented.sum() < mask.sum()

    def test_eval_mode_noop(self):
        augmentation = TrackDropout(probability=0.5).eval()
        _, _, _, mask, labels = _make_inputs()
        assert torch.equal(augmentation(mask, labels), mask)


class TestFeatureJitter:
    def test_adds_noise_in_train(self):
        jitter = FeatureJitter(sigma=0.1).train()
        _, features, _, _, _ = _make_inputs()
        augmented = jitter(features)
        assert not torch.equal(augmented, features)
        assert augmented.shape == features.shape

    def test_eval_mode_noop(self):
        jitter = FeatureJitter(sigma=0.1).eval()
        _, features, _, _, _ = _make_inputs()
        assert torch.equal(jitter(features), features)


class TestEtaPhiRotation:
    def test_phi_stays_in_range(self):
        rotation = EtaPhiRotation(eta_flip_probability=0.0).train()
        points, _, lv, _, _ = _make_inputs()
        rotated_points, _ = rotation(points, lv)
        rotated_phi = rotated_points[:, 1]
        assert (rotated_phi >= -math.pi).all()
        assert (rotated_phi <= math.pi).all()

    def test_pt_magnitude_preserved(self):
        """Rotation preserves transverse momentum magnitude."""
        rotation = EtaPhiRotation(eta_flip_probability=0.0).train()
        _, _, lv, _, _ = _make_inputs()
        _, rotated_lv = rotation(torch.zeros(BATCH_SIZE, 2, NUM_TRACKS), lv)
        pt_before = torch.sqrt(lv[:, 0] ** 2 + lv[:, 1] ** 2)
        pt_after = torch.sqrt(
            rotated_lv[:, 0] ** 2 + rotated_lv[:, 1] ** 2,
        )
        torch.testing.assert_close(pt_before, pt_after, atol=1e-5, rtol=1e-5)

    def test_eval_mode_noop(self):
        rotation = EtaPhiRotation().eval()
        points, _, lv, _, _ = _make_inputs()
        rotated_points, rotated_lv = rotation(points, lv)
        assert torch.equal(rotated_points, points)
        assert torch.equal(rotated_lv, lv)


class TestSetAugmentation:
    def test_shapes_preserved(self):
        augmentation = SetAugmentation().train()
        points, features, lv, mask, labels = _make_inputs()
        result = augmentation(points, features, lv, mask, labels)
        (
            augmented_points,
            augmented_features,
            augmented_lv,
            augmented_mask,
        ) = result
        assert augmented_points.shape == points.shape
        assert augmented_features.shape == features.shape
        assert augmented_lv.shape == lv.shape
        assert augmented_mask.shape == mask.shape

    def test_positives_still_valid(self):
        augmentation = SetAugmentation(
            dropout_probability=1.0,
        ).train()
        points, features, lv, mask, labels = _make_inputs()
        _, _, _, augmented_mask = augmentation(
            points, features, lv, mask, labels,
        )
        positive_mask = labels > 0.5
        assert (augmented_mask[positive_mask] == 1.0).all()
