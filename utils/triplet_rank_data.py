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

# Optional per-candidate inputs beyond the 89 geometry features: the GBDT soft-filter
# scores (always present in the candidates artifact) and the frozen-cascade scores
# gathered from the per-event track_s1/track_s2/couple_scores columns (present only in
# artifacts built with the cascade-column schema).
GBDT_EXTRA_NAMES = ['gbdt6_score', 'gbdt8_score']
CASCADE_EXTRA_NAMES = ['s1_i', 's1_j', 's1_k', 's2_i', 's2_j', 's2_k',
                       's2_k_isvalid', 's3_couple']


def resolve_feature_names(table: '_EventTable', extra_features: str) -> list[str]:
    if extra_features not in ('none', 'gbdt', 'all', 'auto'):
        raise ValueError(f'unknown extra_features {extra_features!r}')
    if extra_features == 'none':
        return list(FEATURE_NAMES)
    if extra_features == 'all' and not table.has_cascade_columns:
        raise ValueError('extra_features=all requires track_s1/track_s2/couple_scores '
                         'columns in the candidates artifact')
    names = list(FEATURE_NAMES) + GBDT_EXTRA_NAMES
    if extra_features == 'all' or (extra_features == 'auto' and table.has_cascade_columns):
        names += CASCADE_EXTRA_NAMES
    return names


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
    """X: (M, F) in `names` order. Returns (M, F) sign-log1p + affine, clipped to +-10.
    NaN inputs (e.g. Stage-2 scores outside the top-K1) standardize to 0; their
    companion *_isvalid flag channel carries the missingness."""
    log1p_mask = torch.tensor([stats[name]['log1p'] for name in names])
    center = torch.tensor([stats[name]['center'] for name in names], dtype=X.dtype)
    scale = torch.tensor([stats[name]['scale'] for name in names], dtype=X.dtype)
    transformed = torch.where(log1p_mask, torch.sign(X) * torch.log1p(X.abs()), X)
    return torch.nan_to_num(torch.clamp((transformed - center) / scale, -10.0, 10.0),
                            nan=0.0)


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
    # list<> uses int32 offsets; the 300k-event TRAIN candidates exceed 2^31 total
    # elements, so cast to large_list (int64 offsets) before combining to avoid overflow.
    if isinstance(chunked, pa.ChunkedArray) and pa.types.is_list(chunked.type):
        chunked = chunked.cast(pa.large_list(chunked.type.value_type))
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
    _CASCADE_COLS = ['track_s1', 'track_s2', 'couple_scores']
    _TRACK_COLS = ['track_pt', 'track_eta', 'track_phi', 'track_charge',
                   'track_dz_significance', 'track_dxy_significance',
                   'track_dca_significance', 'track_n_valid_pixel_hits',
                   'track_norm_chi2', 'track_pt_error',
                   'track_covariance_phi_phi', 'track_covariance_lambda_lambda']

    def __init__(self, candidates_path: str, tracks_path: str):
        schema_names = set(pq.read_schema(candidates_path).names)
        self.has_cascade_columns = all(name in schema_names for name in self._CASCADE_COLS)
        cand_cols = self._CAND_COLS + (self._CASCADE_COLS if self.has_cascade_columns else [])
        candidates = pq.read_table(candidates_path, columns=cand_cols)
        tracks = pq.read_table(tracks_path, columns=self._TRACK_COLS)
        assert candidates.num_rows == tracks.num_rows, 'candidates/tracks row mismatch'
        self.num_rows = candidates.num_rows
        self.candidates = {name: _plain_array(candidates[name]) for name in cand_cols}
        self.tracks = {name: _plain_array(tracks[name]) for name in self._TRACK_COLS}

    def candidate_arrays(self, r: int) -> dict[str, np.ndarray]:
        out = {}
        for name in ['cand_i', 'cand_j', 'cand_k', 'couple_rank', 'gbdt6_score',
                     'gbdt8_score', 'is_gt']:
            out[name] = np.asarray(self.candidates[name][r].values)
        return out

    def cascade_arrays(self, r: int) -> dict[str, np.ndarray]:
        return {name: np.asarray(self.candidates[name][r].values)
                for name in self._CASCADE_COLS}

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


def build_candidate_features(table: _EventTable, r: int, arrays: dict, selected,
                             feature_names: list[str]):
    """arrays: candidate_arrays(r). selected: (M,) candidate positions.
    Returns features (M, len(feature_names)) plus the i/j/k index tensors."""
    kw = table.track_kw(r)
    i = torch.tensor(arrays['cand_i'][selected], dtype=torch.long)
    j = torch.tensor(arrays['cand_j'][selected], dtype=torch.long)
    k = torch.tensor(arrays['cand_k'][selected], dtype=torch.long)
    rank = torch.tensor(arrays['couple_rank'][selected], dtype=torch.long)
    features = triplet_feature_columns(i, j, k, rank, **kw)
    extra_names = feature_names[len(FEATURE_NAMES):]
    if extra_names:
        values = {name: torch.tensor(arrays[name][selected], dtype=torch.float32)
                  for name in extra_names if name in GBDT_EXTRA_NAMES}
        if any(name in CASCADE_EXTRA_NAMES for name in extra_names):
            cascade = table.cascade_arrays(r)
            track_s1 = torch.tensor(cascade['track_s1'], dtype=torch.float32)
            track_s2 = torch.tensor(cascade['track_s2'], dtype=torch.float32)
            couple_scores = torch.tensor(cascade['couple_scores'], dtype=torch.float32)
            s2_k = track_s2[k]
            values.update({
                's1_i': track_s1[i], 's1_j': track_s1[j], 's1_k': track_s1[k],
                's2_i': track_s2[i], 's2_j': track_s2[j], 's2_k': s2_k,
                's2_k_isvalid': torch.isfinite(s2_k).float(),
                's3_couple': couple_scores[rank],
            })
        extras = torch.stack([values[name] for name in extra_names], dim=1)
        features = torch.cat([features, extras], dim=1)
    return features, i, j, k


def fit_norm_stats(candidates_path: str, tracks_path: str, *,
                   feature_names: list[str] | None = None, n_events: int = 2000,
                   per_event: int = 50, seed: int = 0, events=None) -> dict:
    """Median/IQR stats over sampled surviving candidates; keys = feature_names
    (default FEATURE_NAMES). NaN entries (missing Stage-2 scores) are ignored by
    the percentiles; binary *_isvalid flags get passthrough stats.

    events: optional event-row pool to sample from (e.g. the train split side);
    defaults to all rows.
    """
    table = _EventTable(candidates_path, tracks_path)
    if feature_names is None:
        feature_names = list(FEATURE_NAMES)
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
        X, _, _, _ = build_candidate_features(table, int(r), arrays, take, feature_names)
        samples.append(X.numpy())
    sample = np.concatenate(samples)
    stats = {}
    for column_index, name in enumerate(feature_names):
        if name.endswith('_isvalid'):
            stats[name] = {'log1p': False, 'center': 0.0, 'scale': 1.0}
            continue
        values = sample[:, column_index].astype(np.float64)
        log1p = any(marker in name for marker in LOG1P_MARKERS)
        if log1p:
            values = np.sign(values) * np.log1p(np.abs(values))
        low, median, high = np.nanpercentile(values, [25, 50, 75])
        if not np.isfinite(median):
            median, low, high = 0.0, 0.0, 1.0
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
                 mode: str = 'train', norm_stats: dict | None = None, seed: int = 0,
                 extra_features: str = 'none'):
        assert mode in ('train', 'eval')
        self.table = _EventTable(candidates_path, tracks_path)
        self.tau = tau
        self.score_column = score_column
        self.num_negatives = num_negatives
        self.mode = mode
        self.norm_stats = norm_stats
        self.generator = np.random.default_rng(seed)
        self.feature_names = resolve_feature_names(self.table, extra_features)

        scores = self.table.candidates[score_column]
        flat_survive = np.asarray(scores.values) >= tau
        offsets = np.asarray(scores.offsets)
        flat_gt = np.asarray(self.table.candidates['is_gt'].values)
        self.trainable_indices = np.where(_segment_any(flat_survive & flat_gt, offsets))[0]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

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
            if len(negative_positions) == 0:
                selected = positive_positions
            else:
                sampled = self.generator.integers(0, len(negative_positions),
                                                  self.num_negatives)
                selected = np.concatenate([positive_positions, negative_positions[sampled]])
            pos_mask = np.zeros(len(selected), dtype=bool)
            pos_mask[:len(positive_positions)] = True
        else:
            selected = surviving
            pos_mask = is_gt.astype(bool)

        features, i, j, k = build_candidate_features(
            self.table, int(r), arrays, selected, self.feature_names)
        if self.norm_stats is not None:
            features = standardize_features(features, self.feature_names, self.norm_stats)
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


def collate_triplet_rank_eval(items: list[dict]) -> dict:
    """collate_triplet_rank plus each item's dedup keys (variable-length) and its
    true candidate count."""
    batch = collate_triplet_rank(items)
    batch['keys'] = [item['keys'] for item in items]
    batch['counts'] = torch.tensor([item['features'].shape[0] for item in items])
    return batch
