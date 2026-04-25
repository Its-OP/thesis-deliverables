"""Unit tests for `utils/couple_features.py`.

Covers the canonical-ordering enumerator, the Filter A mass cut, the 51-dim
per-couple feature builder, the GT-couple label computation, and the all-in-one
``enumerate_and_featurize_filter_a`` entry point used by both the trainer and
the diagnostic. The tests use small synthetic events with hand-computed
expected values so any drift in the feature ordering or formulas is caught.
"""
from __future__ import annotations

import math

import pytest
import torch

from utils.couple_features import (
    CASCADE_SCORE_DIM,
    COUPLE_FEATURE_DIM,
    DERIVED_GEOM_DIM,
    M_TAU_GEV,
    PAIRWISE_PHYSICS_DIM,
    RHO_MASS_GEV,
    build_couple_feature_vector,
    build_couple_features_batched,
    compute_couple_labels,
    compute_invariant_mass,
    enumerate_and_featurize_filter_a,
    enumerate_couples_canonical,
    filter_a_mask,
    recover_raw_charges,
)

# ---------------------------------------------------------------------------
# Synthetic single-event fixture
# ---------------------------------------------------------------------------

NUM_TRACKS = 5  # small enough to enumerate by hand


def _make_synthetic_event(seed: int = 42) -> dict[str, torch.Tensor]:
    """Build a (deterministic) tiny single-event tensor batch for testing.

    Tracks 0, 2 are GT pions; tracks 1, 3, 4 are background. Their 4-vectors
    are constructed so the invariant masses are well within the τ mass.
    """
    generator = torch.Generator().manual_seed(seed)

    # 16 standardized features per track. Channel 5 = standardized charge.
    # Standardization: raw {-1, +1} → standardized {-1, 0}.
    # We seed it manually to keep charge ordering predictable.
    features = torch.randn(16, NUM_TRACKS, generator=generator) * 0.5
    # Tracks 0 and 2 are positively charged (raw +1, standardized 0)
    # Tracks 1, 3, 4 are negatively charged (raw -1, standardized -1)
    features[5, [0, 2]] = 0.0
    features[5, [1, 3, 4]] = -1.0

    # Raw points (eta, phi) — small spread, easy to verify Δη / Δφ
    points = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50],   # eta
            [0.05, 0.10, 0.15, 0.20, 0.25],   # phi
        ],
        dtype=torch.float32,
    )

    # Raw 4-vectors. Pion mass = 0.13957 GeV.
    # Construct so all 2-track masses are well below m_τ.
    lorentz = torch.tensor(
        [
            [0.5, 0.6, 0.4, 0.7, 0.8],          # px
            [0.0, 0.1, 0.2, 0.0, 0.1],          # py
            [0.3, 0.4, 0.5, 0.6, 0.7],          # pz
            [0.7, 0.8, 0.7, 0.95, 1.1],         # E (designed to make m_pi-ish)
        ],
        dtype=torch.float32,
    )

    # Stage 1 / Stage 2 scores (raw, ranked descending so track 0 is best)
    stage1_scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    stage2_scores = torch.tensor([6.0, 4.5, 3.5, 2.5, 1.5], dtype=torch.float32)

    # Track labels: 1 = GT pion, 0 = background. Tracks 0 and 2 are GT.
    track_labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    return {
        'features': features,
        'points': points,
        'lorentz': lorentz,
        'stage1_scores': stage1_scores,
        'stage2_scores': stage2_scores,
        'track_labels': track_labels,
    }


# ---------------------------------------------------------------------------
# Constants and dimension assertions
# ---------------------------------------------------------------------------

class TestConstants:
    def test_feature_dim_is_51(self):
        assert COUPLE_FEATURE_DIM == 51
        assert PAIRWISE_PHYSICS_DIM == 10
        assert DERIVED_GEOM_DIM == 5
        assert CASCADE_SCORE_DIM == 4
        # 16 + 16 + 10 + 5 + 4
        assert 16 * 2 + PAIRWISE_PHYSICS_DIM + DERIVED_GEOM_DIM + CASCADE_SCORE_DIM == 51

    def test_m_tau_pdg_value(self):
        # PDG 2024 τ mass
        assert abs(M_TAU_GEV - 1.77693) < 1e-6


# ---------------------------------------------------------------------------
# Charge recovery
# ---------------------------------------------------------------------------

class TestRecoverRawCharges:
    def test_standardized_zero_recovers_to_plus_one(self):
        # raw +1 standardizes to (1 - 1) * 0.5 = 0.0; reverse should give +1
        x = torch.tensor([0.0])
        assert torch.equal(recover_raw_charges(x), torch.tensor([1.0]))

    def test_standardized_minus_one_recovers_to_minus_one(self):
        # raw -1 standardizes to (-1 - 1) * 0.5 = -1.0; reverse should give -1
        x = torch.tensor([-1.0])
        assert torch.equal(recover_raw_charges(x), torch.tensor([-1.0]))

    def test_batch_recovery(self):
        x = torch.tensor([0.0, -1.0, 0.0, -1.0])
        expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.equal(recover_raw_charges(x), expected)


# ---------------------------------------------------------------------------
# Couple enumeration
# ---------------------------------------------------------------------------

class TestEnumerateCouplesCanonical:
    def test_returns_upper_triangular_indices(self):
        upper_i, upper_j = enumerate_couples_canonical(5, torch.device('cpu'))
        # 5 choose 2 = 10 pairs
        assert upper_i.shape == (10,)
        assert upper_j.shape == (10,)

    def test_canonical_ordering_i_lt_j(self):
        upper_i, upper_j = enumerate_couples_canonical(5, torch.device('cpu'))
        assert (upper_i < upper_j).all()

    def test_pairs_are_unique(self):
        upper_i, upper_j = enumerate_couples_canonical(5, torch.device('cpu'))
        pairs = set()
        for i, j in zip(upper_i.tolist(), upper_j.tolist()):
            assert (i, j) not in pairs
            pairs.add((i, j))
        assert len(pairs) == 10

    def test_n_choose_2_count(self):
        for n in [3, 5, 10, 50]:
            upper_i, _ = enumerate_couples_canonical(n, torch.device('cpu'))
            assert upper_i.shape[0] == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# Invariant mass + Filter A
# ---------------------------------------------------------------------------

class TestComputeInvariantMass:
    def test_two_back_to_back_photons(self):
        # Two massless particles back-to-back with E = 1 each gives m = 2
        lorentz = torch.tensor(
            [
                [1.0, -1.0],   # px
                [0.0, 0.0],    # py
                [0.0, 0.0],    # pz
                [1.0, 1.0],    # E
            ],
            dtype=torch.float32,
        )
        upper_i = torch.tensor([0])
        upper_j = torch.tensor([1])
        m = compute_invariant_mass(lorentz, upper_i, upper_j)
        assert torch.allclose(m, torch.tensor([2.0]), atol=1e-6)

    def test_collinear_photons_have_zero_mass(self):
        # Two parallel photons → m² = 0
        lorentz = torch.tensor(
            [
                [1.0, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        upper_i = torch.tensor([0])
        upper_j = torch.tensor([1])
        m = compute_invariant_mass(lorentz, upper_i, upper_j)
        assert torch.allclose(m, torch.tensor([0.0]), atol=1e-6)


class TestFilterAMask:
    def test_high_mass_pair_excluded(self):
        # 2 GeV + 2 GeV opposite, 4 GeV total mass — above m_tau
        lorentz = torch.tensor(
            [
                [2.0, -2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 2.0],
            ],
            dtype=torch.float32,
        )
        upper_i = torch.tensor([0])
        upper_j = torch.tensor([1])
        mask = filter_a_mask(lorentz, upper_i, upper_j)
        assert not mask.item()

    def test_low_mass_pair_kept(self):
        # 0.5 GeV + 0.5 GeV opposite, 1 GeV total mass — below m_tau
        lorentz = torch.tensor(
            [
                [0.5, -0.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        upper_i = torch.tensor([0])
        upper_j = torch.tensor([1])
        mask = filter_a_mask(lorentz, upper_i, upper_j)
        assert mask.item()

    def test_no_charge_or_mass_window_in_filter(self):
        """Filter A is purely kinematic — same-sign and far-from-rho pairs
        must still pass as long as m(ij) <= m_tau."""
        # Two same-sign tracks with low mass should pass
        lorentz = torch.tensor(
            [
                [0.3, 0.3],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.4, 0.4],
            ],
            dtype=torch.float32,
        )
        upper_i = torch.tensor([0])
        upper_j = torch.tensor([1])
        # Same-sign pair (charge cut would kill it) — should still pass Filter A
        assert filter_a_mask(lorentz, upper_i, upper_j).item()


# ---------------------------------------------------------------------------
# Couple labels
# ---------------------------------------------------------------------------

class TestComputeCoupleLabels:
    def test_both_gt_is_gt_couple(self):
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        upper_i, upper_j = enumerate_couples_canonical(4, torch.device('cpu'))
        couple_labels = compute_couple_labels(labels, upper_i, upper_j)
        # The only GT couple is (0, 2)
        n_gt_couples = couple_labels.sum().item()
        assert n_gt_couples == 1

    def test_three_gt_pions_make_three_gt_couples(self):
        labels = torch.tensor([1.0, 1.0, 1.0, 0.0])
        upper_i, upper_j = enumerate_couples_canonical(4, torch.device('cpu'))
        couple_labels = compute_couple_labels(labels, upper_i, upper_j)
        # GT couples: (0,1), (0,2), (1,2) → C(3,2) = 3
        assert couple_labels.sum().item() == 3

    def test_no_gt_pions_means_no_gt_couples(self):
        labels = torch.zeros(5)
        upper_i, upper_j = enumerate_couples_canonical(5, torch.device('cpu'))
        couple_labels = compute_couple_labels(labels, upper_i, upper_j)
        assert couple_labels.sum().item() == 0


# ---------------------------------------------------------------------------
# Build couple feature vector
# ---------------------------------------------------------------------------

class TestBuildCoupleFeatureVector:
    def test_output_shape_is_51_x_n_pairs(self):
        event = _make_synthetic_event()
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        couple_features = build_couple_feature_vector(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            upper_i=upper_i,
            upper_j=upper_j,
        )
        n_pairs = NUM_TRACKS * (NUM_TRACKS - 1) // 2
        assert couple_features.shape == (51, n_pairs)

    def test_block_1_is_per_track_concat(self):
        """First 32 dims should be ``[features_i (16) ‖ features_j (16)]``."""
        event = _make_synthetic_event()
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        couple_features = build_couple_feature_vector(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            upper_i=upper_i,
            upper_j=upper_j,
        )
        # Check first couple — should be (i=0, j=1)
        first_pair_features = couple_features[:32, 0]
        expected = torch.cat([event['features'][:, 0], event['features'][:, 1]])
        assert torch.allclose(first_pair_features, expected, atol=1e-6)

    def test_block_3_invariant_mass_matches_compute_function(self):
        """Block 3 dim 0 = m(ij), should match compute_invariant_mass."""
        event = _make_synthetic_event()
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        couple_features = build_couple_feature_vector(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            upper_i=upper_i,
            upper_j=upper_j,
        )
        # m(ij) is at block 1 (32) + block 2 (10) + position 0 of block 3 = index 42
        m_from_features = couple_features[42, :]
        m_from_helper = compute_invariant_mass(event['lorentz'], upper_i, upper_j)
        assert torch.allclose(m_from_features, m_from_helper, atol=1e-6)

    def test_block_4_cascade_scores_have_correct_order(self):
        """Block 4 = [s1(i), s2(i), s1(j), s2(j)] in that order."""
        event = _make_synthetic_event()
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        couple_features = build_couple_feature_vector(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            upper_i=upper_i,
            upper_j=upper_j,
        )
        # Block 4 starts at 32 + 10 + 5 = 47
        # First couple: i=0, j=1
        # Expected: s1(0)=5, s2(0)=6, s1(1)=4, s2(1)=4.5
        expected = torch.tensor([5.0, 6.0, 4.0, 4.5])
        assert torch.allclose(couple_features[47:51, 0], expected, atol=1e-6)

    def test_pairwise_charge_product_at_block_2_position_4(self):
        """The 5th channel of block 2 (overall index 32 + 4 = 36) is q_i × q_j."""
        event = _make_synthetic_event()
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        couple_features = build_couple_feature_vector(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            upper_i=upper_i,
            upper_j=upper_j,
        )
        # Track 0 is +1 (standardized 0.0), track 1 is -1 (standardized -1.0)
        # First couple (0, 1): q_0 × q_1 = +1 × -1 = -1
        assert torch.allclose(
            couple_features[36, 0], torch.tensor(-1.0), atol=1e-6,
        )
        # Couple (0, 2): both +1 → product = +1
        # Index of (0, 2) in upper-tri enumeration: depends on order
        # For n=5, upper order is (0,1)(0,2)(0,3)(0,4)(1,2)(1,3)(1,4)(2,3)(2,4)(3,4)
        idx_0_2 = 1
        assert (upper_i[idx_0_2].item(), upper_j[idx_0_2].item()) == (0, 2)
        assert torch.allclose(
            couple_features[36, idx_0_2], torch.tensor(1.0), atol=1e-6,
        )


# ---------------------------------------------------------------------------
# enumerate_and_featurize_filter_a (the all-in-one entry point)
# ---------------------------------------------------------------------------

class TestEnumerateAndFeaturizeFilterA:
    def test_returns_expected_keys(self):
        event = _make_synthetic_event()
        result = enumerate_and_featurize_filter_a(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            top50_track_labels=event['track_labels'],
        )
        assert 'upper_i' in result
        assert 'upper_j' in result
        assert 'couple_features' in result
        assert 'couple_labels' in result

    def test_couple_features_dimension_matches_n_kept(self):
        event = _make_synthetic_event()
        result = enumerate_and_featurize_filter_a(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            top50_track_labels=event['track_labels'],
        )
        n_kept = result['upper_i'].shape[0]
        assert result['couple_features'].shape == (51, n_kept)
        assert result['couple_labels'].shape == (n_kept,)

    def test_no_track_labels_omits_couple_labels_key(self):
        event = _make_synthetic_event()
        result = enumerate_and_featurize_filter_a(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            top50_track_labels=None,
        )
        assert 'couple_labels' not in result

    def test_filter_a_keeps_only_low_mass_pairs(self):
        """All synthetic-event pairs are constructed with low mass — Filter A
        should keep all C(5, 2) = 10 of them."""
        event = _make_synthetic_event()
        result = enumerate_and_featurize_filter_a(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            top50_track_labels=event['track_labels'],
        )
        n_pairs = NUM_TRACKS * (NUM_TRACKS - 1) // 2
        assert result['upper_i'].shape[0] == n_pairs

    def test_gt_couple_label_correct_for_known_event(self):
        """In the synthetic event tracks 0 and 2 are GT — exactly one GT couple
        should exist (the (0, 2) pair)."""
        event = _make_synthetic_event()
        result = enumerate_and_featurize_filter_a(
            top50_features=event['features'],
            top50_points=event['points'],
            top50_lorentz=event['lorentz'],
            top50_stage1_scores=event['stage1_scores'],
            top50_stage2_scores=event['stage2_scores'],
            top50_track_labels=event['track_labels'],
        )
        n_gt_couples = result['couple_labels'].sum().item()
        assert n_gt_couples == 1


# ---------------------------------------------------------------------------
# build_couple_features_batched (the trainer-side vectorized helper)
# ---------------------------------------------------------------------------

class TestBuildCoupleFeaturesBatched:
    def _make_batch(self, batch_size=2):
        events = [_make_synthetic_event(seed=i) for i in range(batch_size)]
        return {
            'features': torch.stack([e['features'] for e in events]),
            'points': torch.stack([e['points'] for e in events]),
            'lorentz': torch.stack([e['lorentz'] for e in events]),
            'stage1_scores': torch.stack([e['stage1_scores'] for e in events]),
            'stage2_scores': torch.stack([e['stage2_scores'] for e in events]),
            'track_labels': torch.stack([e['track_labels'] for e in events]),
        }

    def test_output_shapes(self):
        batch = self._make_batch(batch_size=3)
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
        )
        n_couples = NUM_TRACKS * (NUM_TRACKS - 1) // 2
        assert result['couple_features'].shape == (3, 51, n_couples)
        assert result['filter_a_mask'].shape == (3, n_couples)
        assert result['couple_labels'].shape == (3, n_couples)
        assert result['filter_a_mask'].dtype == torch.bool
        assert result['couple_labels'].dtype == torch.bool

    def test_no_track_labels_omits_couple_labels_key(self):
        batch = self._make_batch()
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=None,
        )
        assert 'couple_labels' not in result

    def test_batched_matches_per_event_helper(self):
        """The batched version must produce identical features to the
        per-event helper applied to each event individually."""
        batch = self._make_batch(batch_size=2)
        batched = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
        )
        # Per-event reference for event 0
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        per_event_features_0 = build_couple_feature_vector(
            top50_features=batch['features'][0],
            top50_points=batch['points'][0],
            top50_lorentz=batch['lorentz'][0],
            top50_stage1_scores=batch['stage1_scores'][0],
            top50_stage2_scores=batch['stage2_scores'][0],
            upper_i=upper_i,
            upper_j=upper_j,
        )
        assert torch.allclose(
            batched['couple_features'][0], per_event_features_0, atol=1e-6,
        )

    def test_batched_filter_a_matches_per_event(self):
        batch = self._make_batch(batch_size=2)
        batched = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
        )
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        per_event_mask_0 = filter_a_mask(batch['lorentz'][0], upper_i, upper_j)
        assert torch.equal(batched['filter_a_mask'][0], per_event_mask_0)

    def test_batched_labels_match_per_event(self):
        batch = self._make_batch(batch_size=2)
        batched = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
        )
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        per_event_labels_0 = compute_couple_labels(
            batch['track_labels'][0], upper_i, upper_j,
        )
        assert torch.equal(batched['couple_labels'][0], per_event_labels_0)

    def test_n_couples_is_k2_choose_2(self):
        """n_couples should always equal C(K2, 2), regardless of batch size."""
        batch = self._make_batch(batch_size=4)
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=None,
        )
        expected_n = NUM_TRACKS * (NUM_TRACKS - 1) // 2
        assert result['couple_features'].shape[2] == expected_n
        assert result['filter_a_mask'].shape[1] == expected_n

    # --- track_valid_mask tests (padding exclusion) ---

    def test_track_valid_mask_excludes_padding_couples_from_filter_a(self):
        """Couples where either track is invalid must be False in filter_a_mask."""
        batch = self._make_batch(batch_size=2)
        # Mark track 3 and 4 as padding (invalid) in event 0
        track_valid_mask = torch.ones(2, NUM_TRACKS, dtype=torch.bool)
        track_valid_mask[0, 3] = False
        track_valid_mask[0, 4] = False

        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
            track_valid_mask=track_valid_mask,
        )
        upper_i, upper_j = enumerate_couples_canonical(NUM_TRACKS, torch.device('cpu'))
        # Every couple involving track 3 or 4 must be masked out
        involves_invalid = (upper_i == 3) | (upper_i == 4) | (upper_j == 3) | (upper_j == 4)
        assert not result['filter_a_mask'][0, involves_invalid].any(), (
            'Couples involving invalid tracks must be excluded from filter_a_mask'
        )
        # Event 1 (all valid) should be unaffected
        result_no_mask = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
        )
        assert torch.equal(result['filter_a_mask'][1], result_no_mask['filter_a_mask'][1])

    def test_track_valid_mask_zeroes_padding_features(self):
        """Features/scores for invalid tracks must be zeroed before couple
        feature computation to prevent Inf from entering the pipeline."""
        batch = self._make_batch(batch_size=2)
        # Inject -inf into scores of invalid tracks (mimics cascade padding)
        batch['stage1_scores'][0, 3] = float('-inf')
        batch['stage1_scores'][0, 4] = float('-inf')
        batch['stage2_scores'][0, 3] = float('-inf')
        batch['stage2_scores'][0, 4] = float('-inf')

        track_valid_mask = torch.ones(2, NUM_TRACKS, dtype=torch.bool)
        track_valid_mask[0, 3] = False
        track_valid_mask[0, 4] = False

        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
            track_valid_mask=track_valid_mask,
        )
        # ALL couple features must be finite (no Inf from -inf scores)
        assert torch.isfinite(result['couple_features']).all(), (
            'Couple features must be finite when track_valid_mask zeroes out padding'
        )

    def test_no_track_valid_mask_preserves_original_behavior(self):
        """Omitting track_valid_mask should behave identically to before."""
        batch = self._make_batch(batch_size=2)
        result_with = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
            track_valid_mask=None,
        )
        result_without = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
        )
        assert torch.equal(result_with['couple_features'], result_without['couple_features'])
        assert torch.equal(result_with['filter_a_mask'], result_without['filter_a_mask'])

    def test_all_tracks_valid_mask_matches_no_mask(self):
        """An all-True mask should produce identical results to no mask."""
        batch = self._make_batch(batch_size=2)
        all_valid = torch.ones(2, NUM_TRACKS, dtype=torch.bool)
        result_mask = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            track_valid_mask=all_valid,
        )
        result_no_mask = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
        )
        assert torch.allclose(
            result_mask['couple_features'], result_no_mask['couple_features'], atol=1e-6,
        )
        assert torch.equal(result_mask['filter_a_mask'], result_no_mask['filter_a_mask'])


class TestPairKinematicsV2:
    """Extra four pair-kinematic features (T2.2)."""

    def _make_batch(self, batch_size=2):
        events = [_make_synthetic_event(seed=i) for i in range(batch_size)]
        return {
            'features': torch.stack([e['features'] for e in events]),
            'points': torch.stack([e['points'] for e in events]),
            'lorentz': torch.stack([e['lorentz'] for e in events]),
            'stage1_scores': torch.stack([e['stage1_scores'] for e in events]),
            'stage2_scores': torch.stack([e['stage2_scores'] for e in events]),
            'track_labels': torch.stack([e['track_labels'] for e in events]),
        }

    def test_v2_output_has_55_dims(self):
        from utils.couple_features import COUPLE_FEATURE_DIM_V2
        assert COUPLE_FEATURE_DIM_V2 == 55
        batch = self._make_batch(batch_size=2)
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            top_k2_track_labels=batch['track_labels'],
            pair_kinematics_v2=True,
        )
        n_couples = NUM_TRACKS * (NUM_TRACKS - 1) // 2
        assert result['couple_features'].shape == (2, 55, n_couples)

    def test_v2_first_51_channels_identical_to_v1(self):
        """With `pair_kinematics_v2=True` the first 51 channels of the
        output must bytewise match the v1 output (extras go at the end)."""
        batch = self._make_batch(batch_size=2)
        kwargs = {
            'top_k2_features': batch['features'],
            'top_k2_points': batch['points'],
            'top_k2_lorentz': batch['lorentz'],
            'top_k2_stage1_scores': batch['stage1_scores'],
            'top_k2_stage2_scores': batch['stage2_scores'],
            'top_k2_track_labels': batch['track_labels'],
        }
        v1 = build_couple_features_batched(**kwargs)
        v2 = build_couple_features_batched(pair_kinematics_v2=True, **kwargs)
        assert torch.allclose(
            v1['couple_features'],
            v2['couple_features'][:, :51, :],
            atol=1e-6,
        )

    def test_v2_cos_opening_angle_in_expected_range(self):
        """Channel 51 = cos(opening angle) ∈ [-1, 1]."""
        batch = self._make_batch(batch_size=2)
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            pair_kinematics_v2=True,
        )
        cos_angle = result['couple_features'][:, 51, :]
        assert (cos_angle >= -1.0001).all()
        assert (cos_angle <= 1.0001).all()

    def test_v2_mass_residual_sign_makes_sense(self):
        """Channel 52 = (m − m_τ) / σ_m: negative below τ mass, positive
        above. Synthetic pions are low-pT so most couples are below τ."""
        batch = self._make_batch(batch_size=4)
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
            pair_kinematics_v2=True,
        )
        mass_residual = result['couple_features'][:, 52, :]
        # Synthetic low-pT batch → below-tau couples dominate
        assert (mass_residual < 0).sum() > (mass_residual > 0).sum()

    def test_v2_is_opt_in_default_stays_51(self):
        """Default `pair_kinematics_v2=False` must still emit 51 dims
        (backwards compat for pretrained reranker checkpoints)."""
        batch = self._make_batch(batch_size=2)
        result = build_couple_features_batched(
            top_k2_features=batch['features'],
            top_k2_points=batch['points'],
            top_k2_lorentz=batch['lorentz'],
            top_k2_stage1_scores=batch['stage1_scores'],
            top_k2_stage2_scores=batch['stage2_scores'],
        )
        assert result['couple_features'].shape[1] == 51
