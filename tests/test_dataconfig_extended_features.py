"""Stage-1 H6 feature extension: the data config must expose the 16 new
per-track channels (32 total) while preserving the legacy 16-channel prefix,
and the loader must yield finite, non-degenerate batches for all of them.
Uses real shards only."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from weaver.utils.data.config import _md5
from weaver.utils.dataset import SimpleIterDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / 'data' / 'low-pt' / 'lowpt_tau_trackfinder.yaml'
EVAL_SHARD = REPO_ROOT / 'data' / 'low-pt' / 'eval' / 'eval_000.parquet'

LEGACY_FEATURE_NAMES = [
    'track_px', 'track_py', 'track_pz', 'track_eta', 'track_phi',
    'track_charge', 'track_dxy_significance', 'track_log_dz_significance',
    'track_log_norm_chi2', 'track_log_pt_error',
    'track_n_valid_pixel_hits', 'track_dca_significance',
    'track_log_covariance_phi_phi', 'track_log_covariance_lambda_lambda',
    'track_log_pt', 'track_log_relative_pt_error',
]
NEW_FEATURE_NAMES = [
    'track_absolute_dxy', 'track_absolute_dz',
    'track_log_covariance_dxy_dxy', 'track_log_covariance_dsz_dsz',
    'track_log_absolute_covariance_dxy_dsz',
    'track_log_absolute_covariance_phi_dxy',
    'track_n_valid_hits', 'track_pileup_min_gap',
    'track_closer_to_other_pv', 'track_n_pvs',
    'track_muon_min_delta_r', 'track_muon_min_dz_gap',
    'track_has_soft_muon', 'track_sv_min_line_distance', 'track_has_sv',
    'track_pv_line_distance_3d',
]
UNSTANDARDIZED_FLAGS = ['track_closer_to_other_pv', 'track_has_soft_muon',
                        'track_has_sv']
POINTS_TRANSPORT_NAMES = [
    'track_eta_raw', 'track_phi_raw', 'track_dz',
    'track_vertex_x', 'track_vertex_y', 'track_vertex_z',
    'track_nearest_other_pv_index',
    'track_lifetime_disp_x', 'track_lifetime_disp_y',
    'track_primary_vertex_x', 'track_primary_vertex_y',
    'track_sv0_x', 'track_sv0_y', 'track_sv0_z',
    'track_sv0_dlen_sig', 'track_sv0_mass',
    'track_sv1_x', 'track_sv1_y', 'track_sv1_z',
    'track_sv1_dlen_sig', 'track_sv1_mass',
    'track_sv2_x', 'track_sv2_y', 'track_sv2_z',
    'track_sv2_dlen_sig', 'track_sv2_mass',
]
TRANSPORT_NEW_VARIABLES = [
    'track_nearest_other_pv_index', 'track_pca_parameter',
    'track_lifetime_disp_x', 'track_lifetime_disp_y',
    'track_primary_vertex_x', 'track_primary_vertex_y',
    'track_sv0_x', 'track_sv1_x', 'track_sv2_x',
    'track_sv0_mass', 'track_sv1_mass', 'track_sv2_mass',
]

eval_shard_required = pytest.mark.skipif(
    not EVAL_SHARD.exists(), reason='eval_000.parquet not present')


def feature_entries():
    with open(DATA_CONFIG) as handle:
        config = yaml.safe_load(handle)
    return config['inputs']['pf_features']['vars']


def entry_name(entry):
    return entry[0] if isinstance(entry, list) else entry


class TestConfigStructure:
    def test_thirty_two_features_with_legacy_prefix(self):
        names = [entry_name(entry) for entry in feature_entries()]
        assert len(names) == 32
        assert names[:16] == LEGACY_FEATURE_NAMES
        assert names[16:] == NEW_FEATURE_NAMES

    def test_flags_bypass_standardization(self):
        entries_by_name = {entry_name(entry): entry
                           for entry in feature_entries()}
        for flag_name in UNSTANDARDIZED_FLAGS:
            entry = entries_by_name[flag_name]
            assert isinstance(entry, list) and entry[1] is None, flag_name

    def test_new_variables_define_all_new_features(self):
        with open(DATA_CONFIG) as handle:
            config = yaml.safe_load(handle)
        defined = set(config['new_variables'])
        for name in NEW_FEATURE_NAMES:
            if name == 'track_n_valid_hits':
                continue
            assert name in defined, name


class TestPointsStructure:
    def points_entries(self):
        with open(DATA_CONFIG) as handle:
            config = yaml.safe_load(handle)
        return config['inputs']['pf_points']['vars']

    def test_transport_channels_names_and_null_standardization(self):
        entries = self.points_entries()
        assert [entry_name(entry) for entry in entries] == (
            POINTS_TRANSPORT_NAMES)
        for entry in entries:
            assert isinstance(entry, list) and entry[1] is None, entry

    def test_transport_new_variables_defined(self):
        with open(DATA_CONFIG) as handle:
            config = yaml.safe_load(handle)
        defined = set(config['new_variables'])
        for name in TRANSPORT_NEW_VARIABLES:
            assert name in defined, name


@eval_shard_required
class TestBatchContents:
    @pytest.fixture(scope='class')
    def batch_samples(self):
        sidecar_path = str(DATA_CONFIG).replace(
            '.yaml', f'.{_md5(str(DATA_CONFIG))}.auto.yaml')
        if not os.path.exists(sidecar_path):
            pytest.skip('standardization sidecar for the current config md5 '
                        'not generated yet (produced on the training server '
                        'at first launch)')
        dataset = SimpleIterDataset(
            file_dict={'all': [str(EVAL_SHARD)]},
            data_config_file=str(DATA_CONFIG),
            for_training=False,
            fetch_by_files=True,
            fetch_step=1,
            async_load=False,
        )
        iterator = iter(dataset)
        features, points, masks = [], [], []
        for _ in range(30):
            model_inputs, _, _ = next(iterator)
            features.append(np.asarray(model_inputs['pf_features']))
            points.append(np.asarray(model_inputs['pf_points']))
            masks.append(np.asarray(model_inputs['pf_mask']))
        return {
            'features': np.stack(features),
            'points': np.stack(points),
            'masks': np.stack(masks),
        }

    @pytest.fixture(scope='class')
    def feature_samples(self, batch_samples):
        return batch_samples['features']

    def test_shape_and_finiteness(self, feature_samples):
        assert feature_samples.shape[1:] == (32, 2100)
        assert np.isfinite(feature_samples).all()

    def test_new_channels_are_not_degenerate(self, feature_samples):
        for channel_offset, name in enumerate(NEW_FEATURE_NAMES):
            channel = feature_samples[:, 16 + channel_offset, :]
            assert float(channel.std()) > 0, name

    def test_points_shape_and_finiteness(self, batch_samples):
        assert batch_samples['points'].shape[1:] == (9, 2100)
        assert np.isfinite(batch_samples['points']).all()

    def test_signed_dz_keeps_both_signs(self, batch_samples):
        valid = batch_samples['masks'][:, 0, :] > 0
        signed_dz = batch_samples['points'][:, 2, :][valid]
        assert (signed_dz < 0).any() and (signed_dz > 0).any()

    def test_nearest_index_is_integer_valued(self, batch_samples):
        valid = batch_samples['masks'][:, 0, :] > 0
        nearest_index = batch_samples['points'][:, 6, :][valid]
        assert np.allclose(nearest_index, np.round(nearest_index))
        assert (nearest_index >= 0).all()

    def test_nearest_index_matches_numpy_recompute(self, batch_samples):
        awkward = pytest.importorskip('awkward')
        records = awkward.from_parquet(
            str(EVAL_SHARD),
            columns=['track_dz', 'event_primary_vertex_z',
                     'event_other_pv_z'])
        for event_index in range(batch_samples['points'].shape[0]):
            record = records[event_index]
            longitudinal_ip = np.asarray(record['track_dz'])[:2100]
            track_count = len(longitudinal_ip)
            other_pv_z = np.asarray(record['event_other_pv_z'])
            global_z = float(record['event_primary_vertex_z']) + (
                longitudinal_ip)
            if other_pv_z.size:
                expected = np.abs(
                    global_z[:, None] - other_pv_z[None, :]).argmin(axis=1)
            else:
                expected = np.zeros(track_count)
            observed = batch_samples['points'][event_index, 6, :track_count]
            np.testing.assert_array_equal(observed, expected)
