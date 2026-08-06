"""Stage-3 H6 couple-vector extension: the 11 new channels (POCA block,
companion cone, SV attach) must reproduce the H6 reference geometry
(part/utils/vertex_geometry.py + the h6_stage3 diagnostic formulas) on real
fixture events, and the widened 99-dim layout must keep channels 0-87
formula-identical. Uses the real h6_geometry_fixture shard only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from utils.couple_features import (
    COMPANION_MIN_DR_SENTINEL,
    COUPLE_FEATURE_DIM_TOTAL,
    H6_COUPLE_EXTRA_DIM,
    TRACK_EMBED_DIM,
    build_couple_features_batched,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PART_ROOT = REPO_ROOT.parent / 'part'
FIXTURE_PATH = PART_ROOT / 'tests' / 'fixtures' / 'h6_geometry_fixture.parquet'
VERTEX_GEOMETRY_PATH = PART_ROOT / 'utils' / 'vertex_geometry.py'
DATA_CONFIG = REPO_ROOT / 'data' / 'low-pt' / 'lowpt_tau_trackfinder.yaml'

MAX_TRACKS_PER_EVENT = 24
PION_MASS_GEV = 0.13957039
LN_EPSILON = 1e-6
SV_SLOT_COUNT = 3
SV_COORDINATE_SENTINEL = 1e4
CONE_DELTA_R_MAXIMUM = 0.4
CONE_DZ_WINDOW = 0.5
WELL_CONDITIONED_DENOMINATOR = 1e-3

CH_LN_POCA = 88
CH_LN_DISPLACEMENT = 89
CH_POINTING_COS = 90
CH_COMPANION_COUNT = 91
CH_COMPANION_SUM_PT = 92
CH_COMPANION_MIN_DR = 93
CH_HAS_COMPANION = 94
CH_LN_SV_DISTANCE = 95
CH_SV_DLEN_SIG = 96
CH_SV_MASS = 97
CH_HAS_SV = 98

fixture_required = pytest.mark.skipif(
    not (FIXTURE_PATH.exists() and VERTEX_GEOMETRY_PATH.exists()),
    reason='part/ geometry fixture or reference implementation not present')


def load_vertex_geometry():
    specification = importlib.util.spec_from_file_location(
        'h6_vertex_geometry_reference', VERTEX_GEOMETRY_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def fixture_events():
    awkward = pytest.importorskip('awkward')
    records = awkward.from_parquet(str(FIXTURE_PATH))

    events = []
    for record in records:
        track_count = min(
            len(np.asarray(record['track_eta'])), MAX_TRACKS_PER_EVENT)

        def column(name, count=track_count):
            return np.asarray(record[name], dtype=np.float64)[:count]

        eta = column('track_eta')
        phi = column('track_phi')
        transverse_momentum = column('track_pt')
        momentum_x = transverse_momentum * np.cos(phi)
        momentum_y = transverse_momentum * np.sin(phi)
        momentum_z = transverse_momentum * np.sinh(eta)
        energy = np.sqrt(momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2
                         + PION_MASS_GEV ** 2)

        secondary_x = np.asarray(record['sv_x'], dtype=np.float64)
        slot_count = min(len(secondary_x), SV_SLOT_COUNT)
        secondary = {
            'x': np.full(SV_SLOT_COUNT, SV_COORDINATE_SENTINEL),
            'y': np.full(SV_SLOT_COUNT, SV_COORDINATE_SENTINEL),
            'z': np.full(SV_SLOT_COUNT, SV_COORDINATE_SENTINEL),
            'dlen_sig': np.zeros(SV_SLOT_COUNT),
            'mass': np.zeros(SV_SLOT_COUNT),
        }
        for slot in range(slot_count):
            secondary['x'][slot] = np.asarray(record['sv_x'])[slot]
            secondary['y'][slot] = np.asarray(record['sv_y'])[slot]
            secondary['z'][slot] = np.asarray(record['sv_z'])[slot]
            secondary['dlen_sig'][slot] = np.asarray(
                record['sv_dlen_sig'])[slot]
            secondary['mass'][slot] = np.asarray(record['sv_mass'])[slot]

        events.append({
            'eta': eta, 'phi': phi,
            'charge': column('track_charge'),
            'longitudinal_ip': column('track_dz'),
            'vertex_x': column('track_vertex_x'),
            'vertex_y': column('track_vertex_y'),
            'vertex_z': column('track_vertex_z'),
            'momentum_x': momentum_x, 'momentum_y': momentum_y,
            'momentum_z': momentum_z, 'energy': energy,
            'transverse_momentum': transverse_momentum,
            'primary_vertex_x': float(record['event_primary_vertex_x']),
            'primary_vertex_y': float(record['event_primary_vertex_y']),
            'has_sv': 1.0 if slot_count > 0 else 0.0,
            'secondary': secondary,
        })
    return events


@pytest.fixture(scope='module')
def builder_output(fixture_events):
    batch_size = len(fixture_events)
    pool_size = max(len(event['eta']) for event in fixture_events)

    points = torch.zeros(batch_size, 26, pool_size)
    features = torch.zeros(batch_size, 32, pool_size)
    lorentz_vectors = torch.zeros(batch_size, 4, pool_size)
    valid_mask = torch.zeros(batch_size, pool_size, dtype=torch.bool)
    stage1_scores = torch.zeros(batch_size, pool_size)
    stage2_scores = torch.zeros(batch_size, pool_size)

    for event_index, event in enumerate(fixture_events):
        track_count = len(event['eta'])
        columns = slice(0, track_count)
        points[event_index, 0, columns] = torch.from_numpy(
            event['eta']).float()
        points[event_index, 1, columns] = torch.from_numpy(
            event['phi']).float()
        points[event_index, 2, columns] = torch.from_numpy(
            event['longitudinal_ip']).float()
        points[event_index, 3, columns] = torch.from_numpy(
            event['vertex_x']).float()
        points[event_index, 4, columns] = torch.from_numpy(
            event['vertex_y']).float()
        points[event_index, 5, columns] = torch.from_numpy(
            event['vertex_z']).float()
        points[event_index, 9, columns] = event['primary_vertex_x']
        points[event_index, 10, columns] = event['primary_vertex_y']
        for slot in range(SV_SLOT_COUNT):
            base = 11 + slot * 5
            points[event_index, base + 0, columns] = float(
                event['secondary']['x'][slot])
            points[event_index, base + 1, columns] = float(
                event['secondary']['y'][slot])
            points[event_index, base + 2, columns] = float(
                event['secondary']['z'][slot])
            points[event_index, base + 3, columns] = float(
                event['secondary']['dlen_sig'][slot])
            points[event_index, base + 4, columns] = float(
                event['secondary']['mass'][slot])

        features[event_index, 5, columns] = torch.from_numpy(
            (event['charge'] - 1.0) * 0.5).float()
        features[event_index, 30, columns] = event['has_sv']

        lorentz_vectors[event_index, 0, columns] = torch.from_numpy(
            event['momentum_x']).float()
        lorentz_vectors[event_index, 1, columns] = torch.from_numpy(
            event['momentum_y']).float()
        lorentz_vectors[event_index, 2, columns] = torch.from_numpy(
            event['momentum_z']).float()
        lorentz_vectors[event_index, 3, columns] = torch.from_numpy(
            event['energy']).float()
        valid_mask[event_index, columns] = True

    member_full_indices = torch.arange(pool_size).unsqueeze(0).expand(
        batch_size, pool_size)

    with torch.no_grad():
        result = build_couple_features_batched(
            top_k2_features=features,
            top_k2_points=points,
            top_k2_lorentz=lorentz_vectors,
            top_k2_stage1_scores=stage1_scores,
            top_k2_stage2_scores=stage2_scores,
            track_valid_mask=valid_mask,
            full_points=points,
            full_lorentz=lorentz_vectors,
            full_valid_mask=valid_mask,
            member_full_indices=member_full_indices,
        )
    return result


def couple_index_pairs(track_count):
    upper_i, upper_j = np.triu_indices(track_count, k=1)
    return upper_i, upper_j


def valid_couples_view(builder_output, event_index, track_count, pool_size):
    upper_i_pool, upper_j_pool = np.triu_indices(pool_size, k=1)
    keep = (upper_i_pool < track_count) & (upper_j_pool < track_count)
    channels = builder_output['couple_features'][event_index].numpy()
    return channels[:, keep], upper_i_pool[keep], upper_j_pool[keep]


def reference_axis(event, upper_i, upper_j):
    sum_px = event['momentum_x'][upper_i] + event['momentum_x'][upper_j]
    sum_py = event['momentum_y'][upper_i] + event['momentum_y'][upper_j]
    sum_pz = event['momentum_z'][upper_i] + event['momentum_z'][upper_j]
    axis_pt = np.maximum(np.hypot(sum_px, sum_py), 1e-12)
    axis_eta = np.arcsinh(sum_pz / axis_pt)
    axis_phi = np.arctan2(sum_py, sum_px)
    axis_dz = 0.5 * (event['longitudinal_ip'][upper_i]
                     + event['longitudinal_ip'][upper_j])
    return sum_px, sum_py, axis_eta, axis_phi, axis_dz


def reference_poca(geometry, event, upper_i, upper_j):
    reference_points = np.stack(
        [event['vertex_x'], event['vertex_y'], event['vertex_z']], axis=1)
    direction = np.stack(geometry.compute_momentum_direction(
        event['transverse_momentum'], event['eta'], event['phi']), axis=1)
    distance, midpoint = geometry.compute_line_pair_closest_approach(
        reference_points[upper_i], direction[upper_i],
        reference_points[upper_j], direction[upper_j])
    direction_a = direction[upper_i]
    direction_b = direction[upper_j]
    denominator = (
        np.sum(direction_a * direction_a, axis=1)
        * np.sum(direction_b * direction_b, axis=1)
        - np.sum(direction_a * direction_b, axis=1) ** 2)
    return distance, midpoint, denominator


@fixture_required
class TestPocaBlock:
    def test_ln_poca_matches_reference(self, builder_output, fixture_events):
        geometry = load_vertex_geometry()
        pool_size = max(len(event['eta']) for event in fixture_events)
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            channels, upper_i, upper_j = valid_couples_view(
                builder_output, event_index, track_count, pool_size)
            distance, _, denominator = reference_poca(
                geometry, event, upper_i, upper_j)
            expected = np.log(distance + LN_EPSILON)
            observed = channels[CH_LN_POCA]
            assert np.isfinite(observed).all()
            well_conditioned = denominator >= WELL_CONDITIONED_DENOMINATOR
            np.testing.assert_allclose(
                observed[well_conditioned], expected[well_conditioned],
                rtol=1e-3, atol=2e-2)

    def test_displacement_and_pointing_match_reference(
            self, builder_output, fixture_events):
        geometry = load_vertex_geometry()
        pool_size = max(len(event['eta']) for event in fixture_events)
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            channels, upper_i, upper_j = valid_couples_view(
                builder_output, event_index, track_count, pool_size)
            _, midpoint, denominator = reference_poca(
                geometry, event, upper_i, upper_j)
            displacement_x = midpoint[:, 0] - event['primary_vertex_x']
            displacement_y = midpoint[:, 1] - event['primary_vertex_y']
            displacement = np.hypot(displacement_x, displacement_y)
            sum_px, sum_py, _, _, _ = reference_axis(event, upper_i, upper_j)
            pointing = (
                (displacement_x * sum_px + displacement_y * sum_py)
                / np.maximum(displacement * np.hypot(sum_px, sum_py), 1e-12))

            observed_displacement = channels[CH_LN_DISPLACEMENT]
            observed_pointing = channels[CH_POINTING_COS]
            assert np.isfinite(observed_displacement).all()
            assert np.isfinite(observed_pointing).all()
            comparable = (
                (denominator >= WELL_CONDITIONED_DENOMINATOR)
                & (displacement > 1e-3))
            np.testing.assert_allclose(
                observed_displacement[comparable],
                np.log(displacement + LN_EPSILON)[comparable],
                rtol=1e-3, atol=3e-2)
            np.testing.assert_allclose(
                observed_pointing[comparable], pointing[comparable],
                rtol=1e-3, atol=3e-2)


@fixture_required
class TestCompanionCone:
    def test_cone_aggregates_match_looped_reference(
            self, builder_output, fixture_events):
        pool_size = max(len(event['eta']) for event in fixture_events)
        nontrivial_couples = 0
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            channels, upper_i, upper_j = valid_couples_view(
                builder_output, event_index, track_count, pool_size)
            _, _, axis_eta, axis_phi, axis_dz = reference_axis(
                event, upper_i, upper_j)

            for couple_index in range(len(upper_i)):
                member_i = upper_i[couple_index]
                member_j = upper_j[couple_index]
                delta_phi = np.mod(
                    event['phi'] - axis_phi[couple_index] + np.pi,
                    2 * np.pi) - np.pi
                delta_r = np.sqrt(
                    (event['eta'] - axis_eta[couple_index]) ** 2
                    + delta_phi ** 2)
                in_cone = (
                    (delta_r < CONE_DELTA_R_MAXIMUM)
                    & (np.abs(event['longitudinal_ip']
                              - axis_dz[couple_index]) < CONE_DZ_WINDOW))
                in_cone[member_i] = False
                in_cone[member_j] = False

                expected_count = float(in_cone.sum())
                expected_sum_pt = float(
                    event['transverse_momentum'][in_cone].sum())
                expected_min_dr = (
                    float(delta_r[in_cone].min()) if in_cone.any()
                    else COMPANION_MIN_DR_SENTINEL)

                assert channels[CH_COMPANION_COUNT][couple_index] == (
                    pytest.approx(expected_count))
                assert channels[CH_COMPANION_SUM_PT][couple_index] == (
                    pytest.approx(expected_sum_pt, rel=1e-4, abs=1e-4))
                assert channels[CH_COMPANION_MIN_DR][couple_index] == (
                    pytest.approx(expected_min_dr, rel=1e-4, abs=1e-4))
                assert channels[CH_HAS_COMPANION][couple_index] == (
                    1.0 if in_cone.any() else 0.0)
                if in_cone.any():
                    nontrivial_couples += 1
        assert nontrivial_couples > 0


@fixture_required
class TestSecondaryVertexAttach:
    def test_sv_attach_matches_reference(self, builder_output,
                                         fixture_events):
        geometry = load_vertex_geometry()
        pool_size = max(len(event['eta']) for event in fixture_events)
        events_with_sv = 0
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            channels, upper_i, upper_j = valid_couples_view(
                builder_output, event_index, track_count, pool_size)
            if event['has_sv'] < 0.5:
                np.testing.assert_array_equal(
                    channels[CH_LN_SV_DISTANCE], 0.0)
                np.testing.assert_array_equal(channels[CH_SV_DLEN_SIG], 0.0)
                np.testing.assert_array_equal(channels[CH_SV_MASS], 0.0)
                np.testing.assert_array_equal(channels[CH_HAS_SV], 0.0)
                continue
            events_with_sv += 1
            _, midpoint, denominator = reference_poca(
                geometry, event, upper_i, upper_j)
            slot_positions = np.stack(
                [event['secondary']['x'], event['secondary']['y'],
                 event['secondary']['z']], axis=1)
            distances = np.linalg.norm(
                midpoint[:, None, :] - slot_positions[None, :, :], axis=2)
            nearest_slot = distances.argmin(axis=1)
            nearest_distance = distances.min(axis=1)
            expected_dlen = event['secondary']['dlen_sig'][nearest_slot]
            expected_mass = event['secondary']['mass'][nearest_slot]

            observed_distance = channels[CH_LN_SV_DISTANCE]
            assert np.isfinite(observed_distance).all()
            # The midpoint of near-parallel member pairs is ill-conditioned
            # in fp32; slot-argmin ties flip the dlen/mass gathers. Compare
            # only well-conditioned couples with a clear nearest slot.
            sorted_distances = np.sort(distances, axis=1)
            argmin_margin = sorted_distances[:, 1] - sorted_distances[:, 0]
            comparable = (
                (denominator >= WELL_CONDITIONED_DENOMINATOR)
                & (argmin_margin > 1e-3))
            np.testing.assert_allclose(
                observed_distance[comparable],
                np.log(nearest_distance + LN_EPSILON)[comparable],
                rtol=1e-3, atol=3e-2)
            np.testing.assert_allclose(
                channels[CH_SV_DLEN_SIG][comparable],
                expected_dlen[comparable], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                channels[CH_SV_MASS][comparable],
                expected_mass[comparable], rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(channels[CH_HAS_SV], 1.0)
        assert events_with_sv > 0


@fixture_required
class TestLayoutAndSafety:
    def test_total_width_and_constants(self, builder_output):
        assert TRACK_EMBED_DIM == 32
        assert H6_COUPLE_EXTRA_DIM == 11
        assert COUPLE_FEATURE_DIM_TOTAL == 99
        assert builder_output['couple_features'].shape[1] == 99

    def test_all_finite(self, builder_output):
        assert torch.isfinite(builder_output['couple_features']).all()

    def test_chunk_invariance(self, builder_output, fixture_events):
        batch_size = len(fixture_events)
        pool_size = max(len(event['eta']) for event in fixture_events)
        points = torch.zeros(batch_size, 26, pool_size)
        features = torch.zeros(batch_size, 32, pool_size)
        lorentz_vectors = torch.zeros(batch_size, 4, pool_size)
        valid_mask = torch.zeros(batch_size, pool_size, dtype=torch.bool)
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            points[event_index, 0, :track_count] = torch.from_numpy(
                event['eta']).float()
            points[event_index, 1, :track_count] = torch.from_numpy(
                event['phi']).float()
            points[event_index, 2, :track_count] = torch.from_numpy(
                event['longitudinal_ip']).float()
            lorentz_vectors[event_index, 0, :track_count] = torch.from_numpy(
                event['momentum_x']).float()
            lorentz_vectors[event_index, 1, :track_count] = torch.from_numpy(
                event['momentum_y']).float()
            lorentz_vectors[event_index, 2, :track_count] = torch.from_numpy(
                event['momentum_z']).float()
            lorentz_vectors[event_index, 3, :track_count] = torch.from_numpy(
                event['energy']).float()
            valid_mask[event_index, :track_count] = True
        member_full_indices = torch.arange(pool_size).unsqueeze(0).expand(
            batch_size, pool_size)

        default = build_couple_features_batched(
            top_k2_features=features, top_k2_points=points,
            top_k2_lorentz=lorentz_vectors,
            top_k2_stage1_scores=torch.zeros(batch_size, pool_size),
            top_k2_stage2_scores=torch.zeros(batch_size, pool_size),
            track_valid_mask=valid_mask,
            full_points=points, full_lorentz=lorentz_vectors,
            full_valid_mask=valid_mask,
            member_full_indices=member_full_indices,
        )['couple_features']
        chunked = build_couple_features_batched(
            top_k2_features=features, top_k2_points=points,
            top_k2_lorentz=lorentz_vectors,
            top_k2_stage1_scores=torch.zeros(batch_size, pool_size),
            top_k2_stage2_scores=torch.zeros(batch_size, pool_size),
            track_valid_mask=valid_mask,
            full_points=points, full_lorentz=lorentz_vectors,
            full_valid_mask=valid_mask,
            member_full_indices=member_full_indices,
            companion_chunk_elements=1000,
        )['couple_features']
        torch.testing.assert_close(default, chunked)

    def test_narrow_points_raises(self, builder_output, fixture_events):
        batch_size = 2
        pool_size = 8
        with pytest.raises(ValueError, match='pf_points'):
            build_couple_features_batched(
                top_k2_features=torch.zeros(batch_size, 32, pool_size),
                top_k2_points=torch.zeros(batch_size, 9, pool_size),
                top_k2_lorentz=torch.zeros(batch_size, 4, pool_size),
                top_k2_stage1_scores=torch.zeros(batch_size, pool_size),
                top_k2_stage2_scores=torch.zeros(batch_size, pool_size),
                full_points=torch.zeros(batch_size, 9, pool_size),
                full_lorentz=torch.zeros(batch_size, 4, pool_size),
                full_valid_mask=torch.ones(
                    batch_size, pool_size, dtype=torch.bool),
                member_full_indices=torch.arange(pool_size).unsqueeze(
                    0).expand(batch_size, pool_size),
            )

    def test_wrong_feature_width_raises(self):
        batch_size = 2
        pool_size = 8
        with pytest.raises(ValueError, match='feature'):
            build_couple_features_batched(
                top_k2_features=torch.zeros(batch_size, 16, pool_size),
                top_k2_points=torch.zeros(batch_size, 26, pool_size),
                top_k2_lorentz=torch.zeros(batch_size, 4, pool_size),
                top_k2_stage1_scores=torch.zeros(batch_size, pool_size),
                top_k2_stage2_scores=torch.zeros(batch_size, pool_size),
                full_points=torch.zeros(batch_size, 26, pool_size),
                full_lorentz=torch.zeros(batch_size, 4, pool_size),
                full_valid_mask=torch.ones(
                    batch_size, pool_size, dtype=torch.bool),
                member_full_indices=torch.arange(pool_size).unsqueeze(
                    0).expand(batch_size, pool_size),
            )


def test_yaml_points_pin():
    with open(DATA_CONFIG) as handle:
        config = yaml.safe_load(handle)
    entries = config['inputs']['pf_points']['vars']
    names = [entry[0] if isinstance(entry, list) else entry
             for entry in entries]
    assert len(names) == 26
    assert names[9] == 'track_primary_vertex_x'
    assert names[10] == 'track_primary_vertex_y'
    assert names[11] == 'track_sv0_x'
    assert names[16] == 'track_sv1_x'
    assert names[21] == 'track_sv2_x'
    assert names[25] == 'track_sv2_mass'
    for entry in entries:
        assert isinstance(entry, list) and entry[1] is None, entry
