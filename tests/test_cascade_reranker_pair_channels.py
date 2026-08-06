"""Stage-2 U-matrix extension: the 10-channel extra-pairwise builder must
reproduce the H6 reference geometry (part/utils/vertex_geometry.py) on real
fixture events, keep the legacy 6-channel path intact, and stay finite under
padding. Uses the real h6_geometry_fixture shard only."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from weaver.nn.model.CascadeReranker import (
    SUPPORTED_PAIR_EXTRA_DIMS,
    CascadeReranker,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PART_ROOT = REPO_ROOT.parent / 'part'
FIXTURE_PATH = PART_ROOT / 'tests' / 'fixtures' / 'h6_geometry_fixture.parquet'
VERTEX_GEOMETRY_PATH = PART_ROOT / 'utils' / 'vertex_geometry.py'
DATA_CONFIG = REPO_ROOT / 'data' / 'low-pt' / 'lowpt_tau_trackfinder.yaml'

MAX_TRACKS_PER_EVENT = 64
PION_MASS_GEV = 0.13957039
LN_POCA_EPSILON = 1e-6
CLOSER_TO_OTHER_PV_FEATURE_INDEX = 24
WELL_CONDITIONED_DENOMINATOR = 1e-3

fixture_required = pytest.mark.skipif(
    not (FIXTURE_PATH.exists() and VERTEX_GEOMETRY_PATH.exists()),
    reason='part/ geometry fixture or reference implementation not present')


def load_vertex_geometry():
    specification = importlib.util.spec_from_file_location(
        'h6_vertex_geometry_reference', VERTEX_GEOMETRY_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def make_reranker(pair_extra_dim):
    torch.manual_seed(0)
    return CascadeReranker(
        input_dim=32,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        pair_input_dim=4,
        pair_extra_dim=pair_extra_dim,
        pair_embed_dims=[16, 16],
    )


@pytest.fixture(scope='module')
def fixture_events():
    awkward = pytest.importorskip('awkward')
    geometry = load_vertex_geometry()
    records = awkward.from_parquet(str(FIXTURE_PATH))

    events = []
    for record in records:
        track_count = min(
            len(np.asarray(record['track_eta'])), MAX_TRACKS_PER_EVENT)
        eta = np.asarray(record['track_eta'], dtype=np.float64)[:track_count]
        phi = np.asarray(record['track_phi'], dtype=np.float64)[:track_count]
        transverse_momentum = np.asarray(
            record['track_pt'], dtype=np.float64)[:track_count]
        charge = np.asarray(
            record['track_charge'], dtype=np.float64)[:track_count]
        longitudinal_ip = np.asarray(
            record['track_dz'], dtype=np.float64)[:track_count]
        vertex_x = np.asarray(
            record['track_vertex_x'], dtype=np.float64)[:track_count]
        vertex_y = np.asarray(
            record['track_vertex_y'], dtype=np.float64)[:track_count]
        vertex_z = np.asarray(
            record['track_vertex_z'], dtype=np.float64)[:track_count]
        dxy_significance = np.asarray(
            record['track_dxy_significance'], dtype=np.float64)[:track_count]
        dz_significance = np.asarray(
            record['track_dz_significance'], dtype=np.float64)[:track_count]

        primary_vertex_x = float(record['event_primary_vertex_x'])
        primary_vertex_y = float(record['event_primary_vertex_y'])
        primary_vertex_z = float(record['event_primary_vertex_z'])
        other_pv_z = np.asarray(record['event_other_pv_z'], dtype=np.float64)

        global_z = primary_vertex_z + longitudinal_ip
        if other_pv_z.size:
            gaps = np.abs(global_z[:, None] - other_pv_z[None, :])
            nearest_other_pv_index = gaps.argmin(axis=1).astype(np.float64)
            pileup_min_gap = gaps.min(axis=1)
        else:
            nearest_other_pv_index = np.zeros(track_count)
            pileup_min_gap = np.full(track_count, np.inf)
        closer_to_other_pv = (
            pileup_min_gap < np.abs(longitudinal_ip)).astype(np.float64)

        all_vertex_z = np.concatenate([[primary_vertex_z], other_pv_z])
        nearest_any_vertex_index = np.abs(
            global_z[:, None] - all_vertex_z[None, :]).argmin(axis=1)

        pca_x, pca_y, _ = geometry.compute_point_of_closest_approach_to_point(
            vertex_x, vertex_y, vertex_z, eta, phi,
            primary_vertex_x, primary_vertex_y)
        lifetime_disp_x = pca_x - primary_vertex_x
        lifetime_disp_y = pca_y - primary_vertex_y

        momentum_x = transverse_momentum * np.cos(phi)
        momentum_y = transverse_momentum * np.sin(phi)
        momentum_z = transverse_momentum * np.sinh(eta)
        energy = np.sqrt(
            momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2
            + PION_MASS_GEV ** 2)

        log_dz_significance = (
            np.sign(dz_significance) * np.log1p(np.abs(dz_significance)))

        events.append({
            'eta': eta, 'phi': phi, 'charge': charge,
            'longitudinal_ip': longitudinal_ip,
            'vertex_x': vertex_x, 'vertex_y': vertex_y, 'vertex_z': vertex_z,
            'dxy_significance': dxy_significance,
            'log_dz_significance': log_dz_significance,
            'nearest_other_pv_index': nearest_other_pv_index,
            'closer_to_other_pv': closer_to_other_pv,
            'nearest_any_vertex_index': nearest_any_vertex_index,
            'lifetime_disp_x': lifetime_disp_x,
            'lifetime_disp_y': lifetime_disp_y,
            'momentum_x': momentum_x, 'momentum_y': momentum_y,
            'momentum_z': momentum_z, 'energy': energy,
            'transverse_momentum': transverse_momentum,
        })
    return events


@pytest.fixture(scope='module')
def builder_inputs(fixture_events):
    batch_size = len(fixture_events)
    pool_size = max(len(event['eta']) for event in fixture_events)

    points = torch.zeros(batch_size, 9, pool_size)
    features = torch.zeros(batch_size, 32, pool_size)
    lorentz_vectors = torch.zeros(batch_size, 4, pool_size)
    mask = torch.zeros(batch_size, 1, pool_size)

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
        points[event_index, 6, columns] = torch.from_numpy(
            event['nearest_other_pv_index']).float()
        points[event_index, 7, columns] = torch.from_numpy(
            event['lifetime_disp_x']).float()
        points[event_index, 8, columns] = torch.from_numpy(
            event['lifetime_disp_y']).float()

        features[event_index, 5, columns] = torch.from_numpy(
            (event['charge'] - 1.0) * 0.5).float()
        features[event_index, 6, columns] = torch.from_numpy(
            event['dxy_significance']).float()
        features[event_index, 7, columns] = torch.from_numpy(
            event['log_dz_significance']).float()
        features[event_index, CLOSER_TO_OTHER_PV_FEATURE_INDEX, columns] = (
            torch.from_numpy(event['closer_to_other_pv']).float())

        lorentz_vectors[event_index, 0, columns] = torch.from_numpy(
            event['momentum_x']).float()
        lorentz_vectors[event_index, 1, columns] = torch.from_numpy(
            event['momentum_y']).float()
        lorentz_vectors[event_index, 2, columns] = torch.from_numpy(
            event['momentum_z']).float()
        lorentz_vectors[event_index, 3, columns] = torch.from_numpy(
            event['energy']).float()
        mask[event_index, 0, columns] = 1.0

    return {
        'points': points,
        'features': features,
        'lorentz_vectors': lorentz_vectors,
        'mask': mask,
    }


def compute_extra_pairwise(model, inputs):
    mask_float = inputs['mask'].float()
    lorentz_for_pairs = (
        inputs['lorentz_vectors'] * mask_float).detach().float()
    return model._compute_extra_pairwise_features(
        inputs['points'], inputs['features'], lorentz_for_pairs, mask_float)


@fixture_required
class TestNewChannelsMatchReference:
    @pytest.fixture(scope='class')
    def extra_pairwise(self, builder_inputs):
        model = make_reranker(pair_extra_dim=10)
        with torch.no_grad():
            return compute_extra_pairwise(model, builder_inputs)

    def test_ln_poca_matches_reference(self, extra_pairwise, fixture_events):
        geometry = load_vertex_geometry()
        well_conditioned_pairs = 0
        total_pairs = 0
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            row_index, column_index = np.meshgrid(
                np.arange(track_count), np.arange(track_count), indexing='ij')
            reference_points = np.stack(
                [event['vertex_x'], event['vertex_y'], event['vertex_z']],
                axis=1)
            direction = np.stack(geometry.compute_momentum_direction(
                event['transverse_momentum'], event['eta'], event['phi']),
                axis=1)
            distance, _ = geometry.compute_line_pair_closest_approach(
                reference_points[row_index.ravel()],
                direction[row_index.ravel()],
                reference_points[column_index.ravel()],
                direction[column_index.ravel()])
            expected = np.log(distance + LN_POCA_EPSILON)

            direction_a = direction[row_index.ravel()]
            direction_b = direction[column_index.ravel()]
            denominator = (
                np.sum(direction_a * direction_a, axis=1)
                * np.sum(direction_b * direction_b, axis=1)
                - np.sum(direction_a * direction_b, axis=1) ** 2)
            well_conditioned = denominator >= WELL_CONDITIONED_DENOMINATOR

            observed = extra_pairwise[
                event_index, 7, :track_count, :track_count].numpy().ravel()
            assert np.isfinite(observed).all()
            # atol on the ln scale: the builder runs fp32 against a float64
            # reference; 2e-2 = 2% distance agreement.
            np.testing.assert_allclose(
                observed[well_conditioned], expected[well_conditioned],
                rtol=1e-3, atol=2e-2)
            well_conditioned_pairs += int(well_conditioned.sum())
            total_pairs += well_conditioned.size
        assert well_conditioned_pairs > 0.9 * total_pairs

    def test_raw_dz_gap_appended_alongside_dz_sig(
            self, extra_pairwise, fixture_events):
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            longitudinal_ip = event['longitudinal_ip']
            expected = np.abs(
                longitudinal_ip[:, None] - longitudinal_ip[None, :])
            observed = extra_pairwise[
                event_index, 6, :track_count, :track_count].numpy()
            np.testing.assert_allclose(observed, expected,
                                       rtol=1e-5, atol=1e-6)
            dz_significance_gap = np.abs(
                event['log_dz_significance'][:, None]
                - event['log_dz_significance'][None, :])
            observed_significance = extra_pairwise[
                event_index, 1, :track_count, :track_count].numpy()
            np.testing.assert_allclose(observed_significance,
                                       dz_significance_gap,
                                       rtol=1e-4, atol=1e-5)
            assert not np.allclose(observed, observed_significance)

    def test_same_other_pv_matches_h6_reference(
            self, extra_pairwise, fixture_events):
        positive_pairs = 0
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            nearest = event['nearest_any_vertex_index']
            expected = (
                (nearest[:, None] == nearest[None, :])
                & (nearest[:, None] > 0)
                & (nearest[None, :] > 0)).astype(np.float32)
            observed = extra_pairwise[
                event_index, 8, :track_count, :track_count].numpy()
            np.testing.assert_array_equal(observed, expected)
            positive_pairs += int(expected.sum())
        assert positive_pairs > 0

    def test_both_lifetime_positive_matches_reference(
            self, extra_pairwise, fixture_events):
        geometry = load_vertex_geometry()
        positive_pairs = 0
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            axis_px = (event['momentum_x'][:, None]
                       + event['momentum_x'][None, :])
            axis_py = (event['momentum_y'][:, None]
                       + event['momentum_y'][None, :])
            magnitude = np.hypot(
                event['lifetime_disp_x'], event['lifetime_disp_y'])
            projection_rows = (
                event['lifetime_disp_x'][:, None] * axis_px
                + event['lifetime_disp_y'][:, None] * axis_py)
            projection_columns = (
                event['lifetime_disp_x'][None, :] * axis_px
                + event['lifetime_disp_y'][None, :] * axis_py)
            signed_rows = np.where(
                projection_rows >= 0, magnitude[:, None], -magnitude[:, None])
            signed_columns = np.where(
                projection_columns >= 0,
                magnitude[None, :], -magnitude[None, :])
            expected = (
                (signed_rows > 0) & (signed_columns > 0)).astype(np.float32)
            observed = extra_pairwise[
                event_index, 9, :track_count, :track_count].numpy()
            # The PCA displacement is exactly perpendicular to the track's own
            # transverse momentum, so self-pairs (and duplicated tracks) sit
            # on the sign boundary — projection == 0 in exact arithmetic and
            # fp32/fp64 round its sign arbitrarily. Compare only pairs a safe
            # normalized distance away from the boundary.
            axis_norm = np.hypot(axis_px, axis_py)
            boundary_rows = np.abs(projection_rows) / np.maximum(
                magnitude[:, None] * axis_norm, 1e-300)
            boundary_columns = np.abs(projection_columns) / np.maximum(
                magnitude[None, :] * axis_norm, 1e-300)
            decided = (boundary_rows > 1e-9) & (boundary_columns > 1e-9)
            np.testing.assert_array_equal(
                observed[decided], expected[decided])
            assert decided.mean() > 0.9
            positive_pairs += int(expected[decided].sum())
        assert positive_pairs > 0
        assert callable(geometry.compute_lifetime_signed_impact_parameter)

    def test_legacy_channels_keep_their_positions(
            self, extra_pairwise, builder_inputs, fixture_events):
        legacy_model = make_reranker(pair_extra_dim=6)
        legacy_inputs = dict(builder_inputs)
        legacy_inputs['points'] = builder_inputs['points'][:, 0:2, :]
        with torch.no_grad():
            legacy = compute_extra_pairwise(legacy_model, legacy_inputs)
        for legacy_channel in (0, 1, 2, 3, 4, 5):
            torch.testing.assert_close(
                extra_pairwise[:, legacy_channel],
                legacy[:, legacy_channel])


@fixture_required
class TestLegacyPathAndSafety:
    def test_legacy_dim6_builder_unchanged(self, builder_inputs,
                                           fixture_events):
        model = make_reranker(pair_extra_dim=6)
        legacy_inputs = dict(builder_inputs)
        legacy_inputs['points'] = builder_inputs['points'][:, 0:2, :]
        with torch.no_grad():
            extra_pairwise = compute_extra_pairwise(model, legacy_inputs)
        assert extra_pairwise.shape[1] == 6
        for event_index, event in enumerate(fixture_events):
            track_count = len(event['eta'])
            expected = np.abs(
                event['log_dz_significance'][:, None]
                - event['log_dz_significance'][None, :])
            observed = extra_pairwise[
                event_index, 1, :track_count, :track_count].numpy()
            np.testing.assert_allclose(observed, expected,
                                       rtol=1e-4, atol=1e-5)

    def test_padded_pairs_exactly_zero_and_no_nan(self, builder_inputs):
        model = make_reranker(pair_extra_dim=10)
        with torch.no_grad():
            extra_pairwise = compute_extra_pairwise(model, builder_inputs)
        assert torch.isfinite(extra_pairwise).all()
        valid = builder_inputs['mask'].float()
        pair_mask = valid.unsqueeze(-1) * valid.unsqueeze(-2)
        padded_values = extra_pairwise * (1.0 - pair_mask)
        assert padded_values.abs().max().item() == 0.0

    def test_dim10_forward_backward_finite(self, builder_inputs):
        model = make_reranker(pair_extra_dim=10)
        stage1_scores = torch.randn(
            builder_inputs['mask'].shape[0], builder_inputs['mask'].shape[2])
        scores = model(
            builder_inputs['points'],
            builder_inputs['features'],
            builder_inputs['lorentz_vectors'],
            builder_inputs['mask'],
            stage1_scores,
        )
        valid = builder_inputs['mask'].squeeze(1).bool()
        assert torch.isfinite(scores[valid]).all()
        scores[valid].sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()

    def test_unsupported_dim_raises(self):
        with pytest.raises(ValueError):
            make_reranker(pair_extra_dim=5)

    def test_supported_dims_exported(self):
        assert set(SUPPORTED_PAIR_EXTRA_DIMS) == {0, 6, 10}

    def test_dim10_requires_nine_point_channels(self, builder_inputs):
        model = make_reranker(pair_extra_dim=10)
        narrow_inputs = dict(builder_inputs)
        narrow_inputs['points'] = builder_inputs['points'][:, 0:2, :]
        with pytest.raises(ValueError, match='pf_points'):
            with torch.no_grad():
                compute_extra_pairwise(model, narrow_inputs)


@fixture_required
def test_checkpoint_roundtrip_dim10(tmp_path, builder_inputs):
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 'python'))
    from eval_cascade_pipeline import _load_stage2

    model = make_reranker(pair_extra_dim=10)
    checkpoint_path = tmp_path / 'stage2_dim10.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': {
            'stage2_embed_dim': 32,
            'stage2_num_heads': 2,
            'stage2_num_layers': 1,
            'stage2_pair_embed_dims': '16,16',
            'stage2_pair_extra_dim': 10,
            'stage2_pair_embed_mode': 'concat',
            'stage2_ffn_ratio': 4,
            'stage2_dropout': 0.1,
            'stage2_loss_mode': 'pairwise',
            'stage2_rs_at_k_target': 200,
            'top_k1': 256,
        },
    }, checkpoint_path)
    reloaded, top_k1 = _load_stage2(str(checkpoint_path), input_dim=32,
                                    device='cpu')
    assert reloaded.pair_extra_dim == 10
    assert top_k1 == 256
    with torch.no_grad():
        extra_pairwise = compute_extra_pairwise(reloaded, builder_inputs)
    assert extra_pairwise.shape[1] == 10


def test_closer_flag_feature_index_pinned():
    with open(DATA_CONFIG) as handle:
        config = yaml.safe_load(handle)
    entries = config['inputs']['pf_features']['vars']
    names = [entry[0] if isinstance(entry, list) else entry
             for entry in entries]
    assert names[CLOSER_TO_OTHER_PV_FEATURE_INDEX] == 'track_closer_to_other_pv'
