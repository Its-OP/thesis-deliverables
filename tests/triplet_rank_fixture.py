from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Shared synthetic fixture for the schema-v2 triplet-rank stack: 5-track
# events (GT = tracks 0,1,2 on a common displaced vertex), a permutable
# per-stage dump, and two source shards carrying every column the builder
# and the dataset read.


def synthetic_events(n_events, seed=0):
    generator = np.random.default_rng(seed)
    events = []
    for index in range(n_events):
        jitter = 0.02 * generator.standard_normal(5)
        events.append(dict(
            src={
                'event_n_tracks': 5,
                'track_pt': (np.array([1.0, 1.2, 0.9, 1.1, 0.8]) + jitter).tolist(),
                'track_eta': [0.10, 0.15, 0.12, 0.18, 0.30],
                'track_phi': [0.05, 0.10, 0.08, 0.12, 0.50],
                'track_charge': [1.0, 1.0, -1.0, -1.0, 1.0],
                'track_dz_significance': [0.20, 0.25, 0.22, 0.28, 9.00],
                'track_dxy_significance': [0.5, 0.6, 0.7, 0.8, 0.9],
                'track_dca_significance': [1.0, 1.1, 1.2, 1.3, 1.4],
                'track_n_valid_pixel_hits': [4.0, 4.0, 3.0, 5.0, 2.0],
                'track_norm_chi2': [1.0, 1.2, 0.9, 1.1, 2.0],
                'track_pt_error': [0.01, 0.02, 0.03, 0.04, 0.05],
                'track_covariance_phi_phi': [0.001, 0.002, 0.003, 0.004, 0.005],
                'track_covariance_lambda_lambda': [0.0011, 0.0021, 0.0031,
                                                   0.0041, 0.0051],
                'track_label_from_tau': [1.0, 1.0, 1.0, 0.0, 0.0],
                'track_label_from_b': [0.0, 0.0, 0.0, 1.0, 0.0],
                'track_vertex_x': [0.10, 0.11, 0.09, 0.02, -0.50],
                'track_vertex_y': [0.05, 0.06, 0.04, 0.01, 0.60],
                'track_vertex_z': [1.00, 1.02, 0.98, 0.20, -4.00],
                'track_dz': [0.30, 0.34, 0.28, 0.10, 7.00],
                'track_covariance_dxy_dxy': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
                'track_covariance_dsz_dsz': [2e-4, 1e-4, 2e-4, 3e-4, 4e-4],
                'track_covariance_dxy_dsz': [1e-5, -1e-5, 2e-5, -2e-5, 3e-5],
                'track_covariance_phi_dxy': [1e-6, 2e-6, -1e-6, 3e-6, -2e-6],
                'track_n_valid_hits': [12.0, 14.0, 11.0, 15.0, 8.0],
                'event_primary_vertex_x': 0.0,
                'event_primary_vertex_y': 0.0,
                'event_primary_vertex_z': 0.9,
                'sv_x': [0.10], 'sv_y': [0.05], 'sv_z': [1.00],
                'sv_dlen_sig': [4.5], 'sv_mass': [0.62],
                'other_track_pt': [0.40], 'other_track_eta': [0.13],
                'other_track_phi': [0.09], 'other_track_dz': [0.31],
                'event_run': 1, 'event_id': 1000 + index,
                'event_luminosity_block': 7,
                'source_batch_id': index // 2, 'source_microbatch_id': index % 2,
            },
            dump={
                'stage1_sorted_indices': [2, 0, 3, 1, 4],
                'stage1_scores': [0.9, 0.8, 0.7, 0.6, 0.5],
                'stage2_sorted_indices': [2, 0, 3, 1],
                'stage2_scores': [0.5, 0.4, 0.35, 0.3],
                'stage3_sorted_couples': [[0, 1], [0, 2], [2, 3]],
                'stage3_couple_scores': [0.95, 0.85, 0.75],
                'event_run': 1, 'event_id': 1000 + index,
                'event_luminosity_block': 7,
                'source_batch_id': index // 2, 'source_microbatch_id': index % 2,
            },
        ))
    return events


def write_fixture(tmp_path, events, dump_order):
    src_dir = tmp_path / 'shards'
    src_dir.mkdir()
    half = len(events) // 2
    for shard, chunk in enumerate((events[:half], events[half:])):
        columns = {key: [event['src'][key] for event in chunk]
                   for key in chunk[0]['src']}
        pq.write_table(pa.table(columns), src_dir / f'src_{shard:03d}.parquet')
    dump_columns = {key: [events[r]['dump'][key] for r in dump_order]
                    for key in events[0]['dump']}
    dump_path = tmp_path / 'dump.parquet'
    pq.write_table(pa.table(dump_columns), dump_path)
    return str(dump_path), str(src_dir / '*.parquet')
