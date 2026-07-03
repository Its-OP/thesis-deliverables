from __future__ import annotations

import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import Dataset

from utils.triplet_join import FEATURE_NAMES, build_track_lorentz, triplet_feature_columns

DEFAULT_AUTO_YAML = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'low-pt',
    'lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml',
)

# pf_features channel order from the data-config yaml; formulas from its new_variables.
TRACK16_VAR_NAMES = [
    'track_px', 'track_py', 'track_pz', 'track_eta', 'track_phi', 'track_charge',
    'track_dxy_significance', 'track_log_dz_significance', 'track_log_norm_chi2',
    'track_log_pt_error', 'track_n_valid_pixel_hits', 'track_dca_significance',
    'track_log_covariance_phi_phi', 'track_log_covariance_lambda_lambda',
    'track_log_pt', 'track_log_relative_pt_error',
]

# FEATURE_NAMES substrings flagged for sign(x)*log1p(|x|) before the affine transform:
# raw significances, covariances, pt errors, chi2, and the Minkowski dot span decades.
LOG1P_MARKERS = ('dz', 'dxy', 'dca', 'chi2', 'pt_error', 'rel_pt_err', 'cov_', 'lorentz_dot')


def load_track16_params(auto_yaml_path: str = DEFAULT_AUTO_YAML):
    with open(auto_yaml_path) as fh:
        config = yaml.safe_load(fh)
    all_params = config['preprocess']['params']
    params = []
    for name in TRACK16_VAR_NAMES:
        entry = all_params[name]
        params.append((name, {'center': float(entry['center']), 'scale': float(entry['scale']),
                              'min': float(entry['min']), 'max': float(entry['max'])}))
    return params


def track16_std(*, pt, eta, phi, charge, dxy_sig, dz_sig, norm_chi2, pt_error,
                n_pixel, dca_sig, cov_phi_phi, cov_lambda_lambda, params):
    """All track inputs: (N,). Returns (N, 16) weaver-standardized pf_features."""
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    raw_channels = [
        px, py, pz, eta, phi, charge,
        dxy_sig,
        torch.sign(dz_sig) * torch.log1p(dz_sig.abs()),
        torch.log1p(norm_chi2),
        torch.log(torch.clamp_min(pt_error, 1e-12)),
        n_pixel,
        dca_sig,
        torch.log(torch.clamp_min(cov_phi_phi, 1e-12)),
        torch.log(torch.clamp_min(cov_lambda_lambda, 1e-12)),
        torch.log(pt + 1e-6),
        torch.log(torch.clamp_min(pt_error / (pt + 1e-6), 1e-12)),
    ]
    standardized = [
        torch.clamp((channel - p['center']) * p['scale'], p['min'], p['max'])
        for channel, (_, p) in zip(raw_channels, params)
    ]
    return torch.stack(standardized, dim=1)


def standardize_features(X: torch.Tensor, names: list[str], stats: dict) -> torch.Tensor:
    """X: (M, F) in `names` order. Returns (M, F) sign-log1p + affine, clipped to +-10."""
    columns = []
    for column, name in zip(X.unbind(dim=1), names):
        s = stats[name]
        if s['log1p']:
            column = torch.sign(column) * torch.log1p(column.abs())
        columns.append(torch.clamp((column - s['center']) / s['scale'], -10.0, 10.0))
    return torch.stack(columns, dim=1)


def save_norm_stats(stats: dict, path: str) -> None:
    with open(path, 'w') as fh:
        json.dump(stats, fh, indent=2)


def load_norm_stats(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _plain_array(chunked):
    # Single plain (Large)ListArray for random access. Source parquets written by awkward
    # carry ARROW:extension metadata: once awkward is imported anywhere in the process,
    # pyarrow resolves them to ExtensionArray, whose scalars have no .values — unwrap.
    import pyarrow as pa
    array = chunked.combine_chunks()
    if isinstance(array, pa.ChunkedArray):
        array = array.chunk(0)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    return array


class _EventTable:
    """Row-position-aligned access to the candidates and tracks parquet artifacts."""

    _CAND_COLS = ['n_tracks', 'n_candidates', 'cand_i', 'cand_j', 'cand_k',
                  'couple_rank', 'gbdt6_score', 'gbdt8_score', 'is_gt', 'recon']
    _TRACK_COLS = ['track_pt', 'track_eta', 'track_phi', 'track_charge',
                   'track_dz_significance', 'track_dxy_significance',
                   'track_dca_significance', 'track_n_valid_pixel_hits',
                   'track_norm_chi2', 'track_pt_error',
                   'track_covariance_phi_phi', 'track_covariance_lambda_lambda']

    def __init__(self, candidates_path: str, tracks_path: str):
        candidates = pq.read_table(candidates_path, columns=self._CAND_COLS)
        tracks = pq.read_table(tracks_path, columns=self._TRACK_COLS)
        assert candidates.num_rows == tracks.num_rows, 'candidates/tracks row mismatch'
        self.num_rows = candidates.num_rows
        self.candidates = {name: _plain_array(candidates[name]) for name in self._CAND_COLS}
        self.tracks = {name: _plain_array(tracks[name]) for name in self._TRACK_COLS}

    def candidate_arrays(self, r: int) -> dict[str, np.ndarray]:
        out = {}
        for name in ['cand_i', 'cand_j', 'cand_k', 'couple_rank', 'gbdt6_score',
                     'gbdt8_score', 'is_gt']:
            out[name] = np.asarray(self.candidates[name][r].values)
        return out

    def track_kw(self, r: int) -> dict[str, torch.Tensor]:
        column = lambda name: torch.tensor(np.asarray(self.tracks[name][r].values),
                                           dtype=torch.float32)
        pt, eta, phi = column('track_pt'), column('track_eta'), column('track_phi')
        return dict(
            lorentz=build_track_lorentz(pt, eta, phi),
            charge=column('track_charge'), eta=eta, phi=phi,
            dz=column('track_dz_significance'), dxy_sig=column('track_dxy_significance'),
            dca_sig=column('track_dca_significance'),
            n_pixel=column('track_n_valid_pixel_hits'),
            norm_chi2=column('track_norm_chi2'), pt_error=column('track_pt_error'),
            cov_phi_phi=column('track_covariance_phi_phi'),
            cov_lambda_lambda=column('track_covariance_lambda_lambda'),
        )


def _segment_any(flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate([[0], np.cumsum(flat.astype(np.int64))])
    return (cumulative[offsets[1:]] - cumulative[offsets[:-1]]) > 0


def fit_norm_stats(candidates_path: str, tracks_path: str, *, n_events: int = 2000,
                   per_event: int = 50, seed: int = 0, events=None) -> dict:
    """Median/IQR stats over sampled surviving candidates; keys = FEATURE_NAMES.

    events: optional event-row pool to sample from (e.g. the train split side);
    defaults to all rows.
    """
    table = _EventTable(candidates_path, tracks_path)
    generator = np.random.default_rng(seed)
    pool = np.arange(table.num_rows) if events is None else np.asarray(events)
    events = generator.choice(pool, min(n_events, len(pool)), replace=False)
    samples = []
    for r in events:
        arrays = table.candidate_arrays(int(r))
        n = len(arrays['cand_i'])
        if n == 0:
            continue
        take = generator.choice(n, min(per_event, n), replace=False)
        kw = table.track_kw(int(r))
        X = triplet_feature_columns(
            torch.tensor(arrays['cand_i'][take], dtype=torch.long),
            torch.tensor(arrays['cand_j'][take], dtype=torch.long),
            torch.tensor(arrays['cand_k'][take], dtype=torch.long),
            torch.tensor(arrays['couple_rank'][take], dtype=torch.long),
            **kw,
        )
        samples.append(X.numpy())
    sample = np.concatenate(samples)
    stats = {}
    for column_index, name in enumerate(FEATURE_NAMES):
        values = sample[:, column_index].astype(np.float64)
        log1p = any(marker in name for marker in LOG1P_MARKERS)
        if log1p:
            values = np.sign(values) * np.log1p(np.abs(values))
        low, median, high = np.percentile(values, [25, 50, 75])
        stats[name] = {'log1p': log1p, 'center': float(median),
                       'scale': float(max(high - low, 1e-6))}
    return stats


class TripletRankDataset(Dataset):
    """Event-major candidate lists for listwise (InfoNCE) training and full-list eval.

    Train items: 89-feature rows for all surviving positives + `num_negatives` random
    negatives (with replacement, mirroring the couple loss sampler). Eval items: the
    full surviving list + sorted 3-set keys for dedup.
    """

    def __init__(self, candidates_path: str, tracks_path: str, *, tau: float,
                 score_column: str = 'gbdt6_score', num_negatives: int = 50,
                 mode: str = 'train', norm_stats: dict | None = None, seed: int = 0):
        assert mode in ('train', 'eval')
        self.table = _EventTable(candidates_path, tracks_path)
        self.tau = tau
        self.score_column = score_column
        self.num_negatives = num_negatives
        self.mode = mode
        self.norm_stats = norm_stats
        self.generator = np.random.default_rng(seed)

        scores = self.table.candidates[score_column]
        flat_survive = np.asarray(scores.values) >= tau
        offsets = np.asarray(scores.offsets)
        flat_gt = np.asarray(self.table.candidates['is_gt'].values)
        self.trainable_indices = np.where(_segment_any(flat_survive & flat_gt, offsets))[0]

    def __len__(self) -> int:
        return self.table.num_rows

    def _select(self, r: int):
        arrays = self.table.candidate_arrays(r)
        survive = arrays[self.score_column] >= self.tau
        return arrays, np.where(survive)[0]

    def __getitem__(self, r: int) -> dict[str, torch.Tensor]:
        arrays, surviving = self._select(int(r))
        is_gt = arrays['is_gt'][surviving]
        if self.mode == 'train':
            positive_positions = surviving[is_gt]
            negative_positions = surviving[~is_gt]
            sampled = self.generator.integers(0, len(negative_positions), self.num_negatives)
            selected = np.concatenate([positive_positions, negative_positions[sampled]])
            pos_mask = np.zeros(len(selected), dtype=bool)
            pos_mask[:len(positive_positions)] = True
        else:
            selected = surviving
            pos_mask = is_gt.astype(bool)

        kw = self.table.track_kw(int(r))
        i = torch.tensor(arrays['cand_i'][selected], dtype=torch.long)
        j = torch.tensor(arrays['cand_j'][selected], dtype=torch.long)
        k = torch.tensor(arrays['cand_k'][selected], dtype=torch.long)
        rank = torch.tensor(arrays['couple_rank'][selected], dtype=torch.long)
        features = triplet_feature_columns(i, j, k, rank, **kw)
        if self.norm_stats is not None:
            features = standardize_features(features, FEATURE_NAMES, self.norm_stats)
        item = {'features': features, 'pos_mask': torch.from_numpy(pos_mask)}
        if self.mode == 'eval':
            item['keys'] = torch.stack([i, j, k], dim=1).sort(dim=1).values
        return item


def collate_triplet_rank(items: list[dict]) -> dict[str, torch.Tensor]:
    """Pads to the batch-max candidate count. Returns features (B, F, N) for Conv1d."""
    batch_size = len(items)
    max_candidates = max(item['features'].shape[0] for item in items)
    num_features = items[0]['features'].shape[1]
    features = torch.zeros(batch_size, num_features, max_candidates)
    pos_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    valid_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    for b, item in enumerate(items):
        n = item['features'].shape[0]
        features[b, :, :n] = item['features'].T
        pos_mask[b, :n] = item['pos_mask']
        valid_mask[b, :n] = True
    return {'features': features, 'pos_mask': pos_mask, 'valid_mask': valid_mask}
