from __future__ import annotations

import math

import pytest
import torch

from utils.triplet_join import (
    CONE_DELTA_R_MAX,
    CONE_DZ_MAX,
    COMPANION_MIN_DR_SENTINEL,
    FEATURE_NAMES,
    FEATURE_NAMES_EXTENDED,
    GATE4_NAMES,
    H6_NAMES,
    LOW_IP_SIGNIFICANCE_MAX,
    SV_MATCH_RADIUS_CM,
    build_track_lorentz,
    triplet_feature_columns,
)

# 5 tracks; tracks 0-2 form the tau-like triplet, 3 is a nearby companion,
# 4 sits far away in (eta, phi, dz). Vertex reference points put 0-2 on a
# common displaced vertex and 3-4 elsewhere.


def _event():
    pt = torch.tensor([1.0, 1.2, 0.9, 1.1, 0.8])
    eta = torch.tensor([0.10, 0.15, 0.12, 0.18, 3.00])
    phi = torch.tensor([0.05, 0.10, 0.08, 0.12, 2.50])
    charge = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0])
    dz_sig = torch.tensor([0.20, 0.25, 0.22, 0.28, 9.00])
    return dict(
        lorentz=build_track_lorentz(pt, eta, phi),
        charge=charge, eta=eta, phi=phi, dz=dz_sig,
        dxy_sig=torch.tensor([0.5, 2.6, 0.7, 0.8, 0.9]),
        dca_sig=torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
        n_pixel=torch.tensor([4.0, 4.0, 3.0, 5.0, 2.0]),
        norm_chi2=torch.tensor([1.0, 1.2, 0.9, 1.1, 2.0]),
        pt_error=torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05]),
        cov_phi_phi=torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005]),
        cov_lambda_lambda=torch.tensor([0.0011, 0.0021, 0.0031, 0.0041, 0.0051]),
    )


def _h6_inputs(**overrides):
    inputs = dict(
        vertex_x=torch.tensor([0.10, 0.11, 0.09, 0.02, -0.50]),
        vertex_y=torch.tensor([0.05, 0.06, 0.04, 0.01, 0.60]),
        vertex_z=torch.tensor([1.00, 1.02, 0.98, 0.20, -4.00]),
        dz_raw=torch.tensor([0.30, 0.34, 0.28, 0.10, 7.00]),
        primary_vertex_x=torch.tensor(0.0),
        primary_vertex_y=torch.tensor(0.0),
        sv_x=torch.tensor([0.10, 3.00]),
        sv_y=torch.tensor([0.05, -2.00]),
        sv_z=torch.tensor([1.00, 5.00]),
        sv_dlen_sig=torch.tensor([4.5, 1.5]),
        sv_mass=torch.tensor([0.62, 2.20]),
        other_pt=torch.tensor([0.40, 0.35]),
        other_eta=torch.tensor([0.13, 2.90]),
        other_phi=torch.tensor([0.09, 2.40]),
        other_dz=torch.tensor([0.31, 6.50]),
    )
    inputs.update(overrides)
    return inputs


def _columns(h6=None, triplet=(0, 1, 2), event=None):
    event = event or _event()
    i, j, k = (torch.tensor([value]) for value in triplet)
    return triplet_feature_columns(
        i, j, k, torch.tensor([0]), h6_inputs=h6, **event)


def _feature(name, h6=None, triplet=(0, 1, 2), event=None):
    columns = _columns(h6=h6, triplet=triplet, event=event)
    return float(columns[0, FEATURE_NAMES_EXTENDED.index(name)])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_legacy_layout_is_untouched():
    assert len(FEATURE_NAMES) == 89
    assert GATE4_NAMES == FEATURE_NAMES[:4]
    assert FEATURE_NAMES_EXTENDED[:89] == FEATURE_NAMES
    assert FEATURE_NAMES_EXTENDED[89:] == H6_NAMES
    assert len(H6_NAMES) == 22
    assert len(set(FEATURE_NAMES_EXTENDED)) == 111


def test_builder_returns_legacy_width_without_h6_inputs():
    assert _columns().shape == (1, 89)


def test_builder_returns_extended_width_with_h6_inputs():
    assert _columns(h6=_h6_inputs()).shape == (1, 111)


def test_h6_inputs_do_not_disturb_the_legacy_columns():
    legacy = _columns()
    extended = _columns(h6=_h6_inputs())
    assert torch.allclose(extended[:, :89], legacy, equal_nan=True)


# ---------------------------------------------------------------------------
# Vertex block
# ---------------------------------------------------------------------------

def _brute_force_closest_approach(h6, a, b, event):
    """Minimum distance between two linearized tracks, by grid search."""
    def line(index):
        eta, phi = float(event['eta'][index]), float(event['phi'][index])
        direction = torch.tensor([
            math.cos(phi) / math.cosh(eta),
            math.sin(phi) / math.cosh(eta),
            math.tanh(eta)])
        reference = torch.tensor([float(h6['vertex_x'][index]),
                                  float(h6['vertex_y'][index]),
                                  float(h6['vertex_z'][index])])
        return reference, direction

    reference_a, direction_a = line(a)
    reference_b, direction_b = line(b)
    center_a = center_b = 0.0
    half_width = 40.0
    # Adaptive refinement: a single coarse grid only bounds the minimum from
    # above by its own step size, which is far looser than the analytic solution.
    for _ in range(8):
        grid_a = torch.linspace(center_a - half_width,
                                center_a + half_width, 401)
        grid_b = torch.linspace(center_b - half_width,
                                center_b + half_width, 401)
        points_a = reference_a + grid_a.unsqueeze(1) * direction_a
        points_b = reference_b + grid_b.unsqueeze(1) * direction_b
        distances = torch.cdist(points_a, points_b)
        best = int(distances.argmin())
        center_a = float(grid_a[best // distances.shape[1]])
        center_b = float(grid_b[best % distances.shape[1]])
        half_width *= 4.0 / 401
    return float(distances.min())


def test_poca_matches_a_brute_force_grid_search():
    event, h6 = _event(), _h6_inputs()
    distances = [_brute_force_closest_approach(h6, a, b, event)
                 for a, b in ((0, 1), (0, 2), (1, 2))]
    assert _feature('poca_max', h6) == pytest.approx(max(distances), abs=2e-3)
    assert _feature('poca_mean', h6) == pytest.approx(
        sum(distances) / 3.0, abs=2e-3)


def test_raw_dz_gap_uses_raw_dz_not_significance():
    h6 = _h6_inputs()
    expected = max(abs(0.30 - 0.34), abs(0.30 - 0.28), abs(0.34 - 0.28))
    assert _feature('raw_dz_gap_max', h6) == pytest.approx(expected, abs=1e-6)


def test_crossing_z_gap_is_positive_and_finite_for_crossing_tracks():
    assert _feature('crossing_z_gap_max', _h6_inputs()) >= 0.0


def test_lifetime_positive_count_counts_members_displaced_along_the_axis():
    forward = _h6_inputs()
    assert _feature('lifetime_positive_count', forward) == pytest.approx(3.0)
    # Mirroring every reference point through the primary vertex flips all
    # three projections onto the triplet axis.
    backward = _h6_inputs(
        vertex_x=-forward['vertex_x'], vertex_y=-forward['vertex_y'])
    assert _feature('lifetime_positive_count', backward) == pytest.approx(0.0)


def test_dxy_significance_spread_is_the_population_std_over_members():
    values = torch.tensor([0.5, 2.6, 0.7])
    expected = float(values.std(unbiased=False))
    assert _feature('dxy_sig_spread', _h6_inputs()) == pytest.approx(
        expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Physics block
# ---------------------------------------------------------------------------

def test_corrected_mass_equals_the_invariant_mass_when_flight_is_collinear():
    event = _event()
    momentum_x = float(event['lorentz'][0, :3].sum())
    momentum_y = float(event['lorentz'][1, :3].sum())
    scale = 0.5 / math.hypot(momentum_x, momentum_y)
    # Put every member reference point on the momentum direction so the
    # reconstructed flight direction is exactly collinear with the triplet pT.
    h6 = _h6_inputs(
        vertex_x=torch.full((5,), momentum_x * scale),
        vertex_y=torch.full((5,), momentum_y * scale),
        vertex_z=torch.zeros(5))
    mass = _feature('m_ijk', h6)
    assert _feature('corrected_mass', h6) == pytest.approx(mass, abs=1e-4)


def test_corrected_mass_exceeds_the_invariant_mass_when_flight_is_transverse():
    h6 = _h6_inputs()
    assert _feature('corrected_mass', h6) > _feature('m_ijk', h6)


def test_signed_ip_flips_with_the_displacement_direction():
    forward = _h6_inputs()
    backward = _h6_inputs(
        vertex_x=-forward['vertex_x'], vertex_y=-forward['vertex_y'])
    assert _feature('signed_ip_k', forward) == pytest.approx(
        -_feature('signed_ip_k', backward), abs=1e-6)


def test_min_and_scalar_sum_pt_use_the_three_members():
    h6 = _h6_inputs()
    assert _feature('min_pt_ijk', h6) == pytest.approx(0.9, abs=1e-5)
    assert _feature('scalar_sum_pt_ijk', h6) == pytest.approx(3.1, abs=1e-5)


def test_low_ip_cone_count_excludes_members_and_high_ip_tracks():
    h6 = _h6_inputs()
    # Track 3 sits inside the cone with |dxy_sig| = 0.8 <= the cut; track 4 is
    # outside the cone entirely; members are always excluded.
    assert _feature('n_low_ip_in_cone', h6) == pytest.approx(1.0)
    raised = _event()
    raised['dxy_sig'] = raised['dxy_sig'].clone()
    raised['dxy_sig'][3] = LOW_IP_SIGNIFICANCE_MAX + 1.0
    assert _feature('n_low_ip_in_cone', h6, event=raised) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Isolation block
# ---------------------------------------------------------------------------

def test_kept_cone_counts_companions_inside_the_window_only():
    h6 = _h6_inputs()
    assert _feature('kept_cone_count', h6) == pytest.approx(1.0)
    assert _feature('kept_cone_sum_pt', h6) == pytest.approx(1.1, abs=1e-5)


def test_kept_cone_drops_companions_outside_the_dz_window():
    h6 = _h6_inputs(dz_raw=torch.tensor([0.30, 0.34, 0.28, 5.00, 7.00]))
    assert _feature('kept_cone_count', h6) == pytest.approx(0.0)
    assert _feature('cone_min_dr', h6) == pytest.approx(
        COMPANION_MIN_DR_SENTINEL)


def test_cone_min_dr_reports_the_nearest_companion():
    event, h6 = _event(), _h6_inputs()
    delta_eta = float(event['eta'][3]) - float(
        torch.asinh(event['lorentz'][2, :3].sum()
                    / torch.hypot(event['lorentz'][0, :3].sum(),
                                  event['lorentz'][1, :3].sum())))
    assert _feature('cone_min_dr', h6) < CONE_DELTA_R_MAX
    assert _feature('cone_min_dr', h6) >= abs(delta_eta) - 1e-6


def test_other_track_cone_uses_the_sub_cutoff_collection():
    h6 = _h6_inputs()
    assert _feature('other_cone_count', h6) == pytest.approx(1.0)
    assert _feature('other_cone_sum_pt', h6) == pytest.approx(0.40, abs=1e-5)


def test_other_track_cone_is_empty_when_the_collection_is_empty():
    empty = torch.zeros(0)
    h6 = _h6_inputs(other_pt=empty, other_eta=empty, other_phi=empty,
                    other_dz=empty)
    assert _feature('other_cone_count', h6) == pytest.approx(0.0)
    assert _feature('other_cone_sum_pt', h6) == pytest.approx(0.0)


def test_cone_windows_are_the_documented_constants():
    assert CONE_DELTA_R_MAX == 0.4
    assert CONE_DZ_MAX == 0.5


# ---------------------------------------------------------------------------
# Secondary-vertex block
# ---------------------------------------------------------------------------

def test_sv_block_attaches_the_nearest_vertex():
    h6 = _h6_inputs()
    assert _feature('sv_nearest_dlen_sig', h6) == pytest.approx(4.5, abs=1e-5)
    assert _feature('sv_nearest_mass', h6) == pytest.approx(0.62, abs=1e-5)
    assert _feature('has_sv', h6) == pytest.approx(1.0)
    # The three members are nearly collinear, so the pairwise closest-approach
    # midpoints sit well along the axis and the reconstructed vertex is only
    # loosely localized — the nearest SV still lands inside the match radius.
    assert _feature('sv_min_distance', h6) < SV_MATCH_RADIUS_CM


def test_n_sv_matched_counts_vertices_within_the_match_radius():
    h6 = _h6_inputs()
    assert _feature('n_sv_matched', h6) == pytest.approx(1.0)
    far = _h6_inputs(
        sv_x=torch.tensor([50.0]), sv_y=torch.tensor([50.0]),
        sv_z=torch.tensor([50.0]), sv_dlen_sig=torch.tensor([4.5]),
        sv_mass=torch.tensor([0.62]))
    assert _feature('n_sv_matched', far) == pytest.approx(0.0)
    assert _feature('sv_min_distance', far) > SV_MATCH_RADIUS_CM


def test_sv_block_is_nan_without_secondary_vertices():
    empty = torch.zeros(0)
    h6 = _h6_inputs(sv_x=empty, sv_y=empty, sv_z=empty,
                    sv_dlen_sig=empty, sv_mass=empty)
    for name in ('sv_min_distance', 'sv_nearest_dlen_sig',
                 'sv_nearest_mass', 'sv_pointing_cos'):
        assert math.isnan(_feature(name, h6))
    assert _feature('has_sv', h6) == pytest.approx(0.0)
    assert _feature('n_sv_matched', h6) == pytest.approx(0.0)


def test_sv_pointing_cos_is_bounded():
    value = _feature('sv_pointing_cos', _h6_inputs())
    assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_candidate_features_keep_the_legacy_output_without_h6_inputs():
    from utils.triplet_join import triplet_candidate_features
    event = _event()
    couples = torch.tensor([[0, 1], [1, 2]])
    pool = torch.arange(5)
    features, names, _, _ = triplet_candidate_features(
        couples, pool, **event)
    assert features.shape[1] == 89
    assert names == FEATURE_NAMES


def test_candidate_features_widen_and_agree_with_the_legacy_columns():
    from utils.triplet_join import triplet_candidate_features
    event, h6 = _event(), _h6_inputs()
    couples = torch.tensor([[0, 1], [1, 2]])
    pool = torch.arange(5)
    legacy, _, legacy_is_gt, legacy_rows = triplet_candidate_features(
        couples, pool, **event)
    extended, names, is_gt, rows = triplet_candidate_features(
        couples, pool, h6_inputs=h6, **event)
    assert extended.shape[1] == 111
    assert names == FEATURE_NAMES_EXTENDED
    assert torch.equal(is_gt, legacy_is_gt)
    assert torch.equal(rows, legacy_rows)
    assert torch.allclose(extended[:, :89], legacy, equal_nan=True)


def test_batched_candidates_match_per_candidate_evaluation():
    event, h6 = _event(), _h6_inputs()
    triplets = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 4)]
    batched = triplet_feature_columns(
        torch.tensor([t[0] for t in triplets]),
        torch.tensor([t[1] for t in triplets]),
        torch.tensor([t[2] for t in triplets]),
        torch.arange(len(triplets)), h6_inputs=h6, **event)
    for row, triplet in enumerate(triplets):
        single = triplet_feature_columns(
            torch.tensor([triplet[0]]), torch.tensor([triplet[1]]),
            torch.tensor([triplet[2]]), torch.tensor([row]),
            h6_inputs=h6, **event)
        assert torch.allclose(batched[row], single[0], atol=1e-5,
                              equal_nan=True)
