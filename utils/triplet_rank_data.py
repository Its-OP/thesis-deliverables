from __future__ import annotations

import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import Dataset

from utils.triplet_join import (
    FEATURE_NAMES,
    FEATURE_NAMES_EXTENDED,
    build_track_lorentz,
    triplet_feature_columns,
)
from utils.vertex_fit_features import FIT_NAMES, static_fit_columns

DEFAULT_AUTO_YAML = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'low-pt',
    'lowpt_tau_trackfinder.4bb52a63a2023c146396395a612cbe3f.auto.yaml',
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
# raw significances, covariances, pt errors, chi2, fit geometry and the Minkowski
# dot span decades.
LOG1P_MARKERS = ('dz', 'dxy', 'dca', 'chi2', 'pt_error', 'rel_pt_err', 'cov_',
                 'lorentz_dot', 'fit_res', 'fit_lxy', 'fit_sigma', 'fit_arc',
                 'fit_dlen', 'fitpv_lxy')

# The reranker's per-candidate base is the h6s4 champion's 100-feature layout:
# the legacy 89 plus the vertex (6) and physics (5) blocks. Isolation and
# secondary-vertex blocks measured inert and are never computed.
H6_FEATURE_WIDTH = 11
BASE_FEATURE_NAMES = list(FEATURE_NAMES_EXTENDED[:len(FEATURE_NAMES) + H6_FEATURE_WIDTH])

# The single soft-filter score column (the gbdt6/gbdt8 pair died with the old
# cascade) and the frozen-cascade extras gathered per candidate.
FILTER_EXTRA_NAMES = ['filter_score']
CASCADE_EXTRA_NAMES = ['s1_i', 's1_j', 's1_k', 's2_i', 's2_j', 's2_k',
                       's2_k_isvalid', 's3_couple']
# Per-candidate standings within the event's serving list, computed over the
# full surviving list on both train and eval sides.
CONTEXT_FEATURE_NAMES = ['ctx_filter_rank_frac', 'ctx_filter_top_gap',
                         'ctx_filter_z', 'ctx_log_n_surviving',
                         'ctx_couple_rank_frac']

IDENTITY_COLS = ['event_run', 'event_id', 'event_luminosity_block',
                 'source_batch_id', 'source_microbatch_id']

_LOGIT_EPS = 1e-7


def resolve_feature_names(table: '_EventTable', extra_features: str) -> list[str]:
    if extra_features not in ('none', 'filter', 'all', 'auto'):
        raise ValueError(f'unknown extra_features {extra_features!r}')
    if extra_features == 'none':
        return list(BASE_FEATURE_NAMES)
    if extra_features == 'all' and not table.has_cascade_columns:
        raise ValueError('extra_features=all requires track_s1/track_s2/'
                         'couple_scores columns in the candidates artifact')
    names = list(BASE_FEATURE_NAMES) + FILTER_EXTRA_NAMES
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
    """Schema-v2 candidates plus per-track columns read straight from the
    source shards — the candidates artifact is written in source-shard order
    and echoes the 5-column identity key, which is asserted here at load."""

    _CAND_COLS = ['n_tracks', 'n_tierh', 'n_tail_total', 'cand_i', 'cand_j',
                  'cand_k', 'couple_rank', 'filter_score', 'is_gt', 'row_kind',
                  'recon']
    _CASCADE_COLS = ['track_s1', 'track_s2', 'couple_scores']
    _TRACK_COLS = ['track_pt', 'track_eta', 'track_phi', 'track_charge',
                   'track_dz_significance', 'track_dxy_significance',
                   'track_dca_significance', 'track_n_valid_pixel_hits',
                   'track_norm_chi2', 'track_pt_error',
                   'track_covariance_phi_phi', 'track_covariance_lambda_lambda',
                   'track_vertex_x', 'track_vertex_y', 'track_vertex_z',
                   'track_dz', 'track_label_from_b',
                   'track_covariance_dxy_dxy', 'track_covariance_dsz_dsz',
                   'track_covariance_dxy_dsz', 'track_covariance_phi_dxy',
                   'track_n_valid_hits',
                   'other_track_pt', 'other_track_eta', 'other_track_phi',
                   'other_track_dz']
    _EVENT_COLS = ['event_primary_vertex_x', 'event_primary_vertex_y',
                   'event_primary_vertex_z']

    def __init__(self, candidates_path: str, src_glob: str):
        schema_names = set(pq.read_schema(candidates_path).names)
        self.has_cascade_columns = all(name in schema_names for name in self._CASCADE_COLS)
        cand_cols = self._CAND_COLS + IDENTITY_COLS \
            + (self._CASCADE_COLS if self.has_cascade_columns else [])
        candidates = pq.read_table(candidates_path, columns=cand_cols)

        shards = sorted(glob.glob(src_glob))
        assert shards, f'no source shards matched {src_glob}'
        import pyarrow as pa
        src = pa.concat_tables([
            pq.read_table(shard, columns=self._TRACK_COLS + self._EVENT_COLS
                          + IDENTITY_COLS)
            for shard in shards])
        # A candidates artifact may cover a prefix of the shards (smoke builds).
        assert candidates.num_rows <= src.num_rows, \
            f'candidates rows {candidates.num_rows} exceed src rows {src.num_rows}'
        src = src.slice(0, candidates.num_rows)
        for name in IDENTITY_COLS:
            expected = np.asarray(candidates[name])
            actual = np.asarray(src[name])
            assert (expected == actual).all(), (
                f'identity echo mismatch on {name}: the candidates artifact was '
                f'built against different shards or a different shard order')
        self.num_rows = candidates.num_rows
        self.candidates = {name: _plain_array(candidates[name])
                           for name in cand_cols}
        self.tracks = {name: _plain_array(src[name]) for name in self._TRACK_COLS}
        self.events = {name: src[name].to_numpy(zero_copy_only=False)
                       for name in self._EVENT_COLS}

    def candidate_arrays(self, r: int) -> dict[str, np.ndarray]:
        out = {}
        for name in ['cand_i', 'cand_j', 'cand_k', 'couple_rank',
                     'filter_score', 'is_gt', 'row_kind']:
            out[name] = np.asarray(self.candidates[name][r].values)
        out['n_tail_total'] = int(self.candidates['n_tail_total'][r].as_py())
        return out

    def cascade_arrays(self, r: int) -> dict[str, np.ndarray]:
        return {name: np.asarray(self.candidates[name][r].values)
                for name in self._CASCADE_COLS}

    def _track_tensor(self, name: str, r: int) -> torch.Tensor:
        return torch.tensor(np.asarray(self.tracks[name][r].values),
                            dtype=torch.float32)

    def track_kw(self, r: int) -> dict[str, torch.Tensor]:
        column = lambda name: self._track_tensor(name, r)
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

    def h6_inputs(self, r: int) -> dict[str, torch.Tensor]:
        """Inputs for the leading H6_FEATURE_WIDTH feature columns. The
        secondary-vertex keys are supplied empty: the SV block sits beyond
        width 11 and is never computed."""
        empty = torch.zeros(0, dtype=torch.float32)
        return dict(
            vertex_x=self._track_tensor('track_vertex_x', r),
            vertex_y=self._track_tensor('track_vertex_y', r),
            vertex_z=self._track_tensor('track_vertex_z', r),
            dz_raw=self._track_tensor('track_dz', r),
            primary_vertex_x=torch.tensor(self.events['event_primary_vertex_x'][r],
                                          dtype=torch.float32),
            primary_vertex_y=torch.tensor(self.events['event_primary_vertex_y'][r],
                                          dtype=torch.float32),
            sv_x=empty, sv_y=empty, sv_z=empty, sv_dlen_sig=empty, sv_mass=empty,
            other_pt=self._track_tensor('other_track_pt', r),
            other_eta=self._track_tensor('other_track_eta', r),
            other_phi=self._track_tensor('other_track_phi', r),
            other_dz=self._track_tensor('other_track_dz', r),
        )

    def primary_vertex(self, r: int) -> torch.Tensor:
        return torch.tensor([self.events[name][r] for name in self._EVENT_COLS],
                            dtype=torch.float32)

    def vertex_fit_kw(self, r: int) -> dict[str, torch.Tensor]:
        return dict(
            vertex_x=self._track_tensor('track_vertex_x', r),
            vertex_y=self._track_tensor('track_vertex_y', r),
            vertex_z=self._track_tensor('track_vertex_z', r),
            var_dxy=self._track_tensor('track_covariance_dxy_dxy', r),
            var_dsz=self._track_tensor('track_covariance_dsz_dsz', r),
        )

    def quality_channels(self, r: int) -> torch.Tensor:
        """Returns (T, 12) log-compressed per-track quality channels feeding
        the learned fit-weight head."""
        log = lambda name: torch.log(
            torch.clamp_min(self._track_tensor(name, r), 1e-12))
        log_abs = lambda name: torch.log1p(self._track_tensor(name, r).abs())
        return torch.stack([
            log('track_covariance_dxy_dxy'),
            log('track_covariance_dsz_dsz'),
            log_abs('track_covariance_dxy_dsz'),
            log_abs('track_covariance_phi_dxy'),
            log('track_covariance_phi_phi'),
            log('track_covariance_lambda_lambda'),
            log('track_pt_error'),
            torch.log1p(self._track_tensor('track_norm_chi2', r)),
            self._track_tensor('track_n_valid_pixel_hits', r),
            self._track_tensor('track_n_valid_hits', r),
            log_abs('track_dxy_significance'),
            log_abs('track_dz_significance'),
        ], dim=1)

    def from_b_counts(self, r: int, i, j, k) -> torch.Tensor:
        labels = self._track_tensor('track_label_from_b', r) > 0.5
        return (labels[i].long() + labels[j].long() + labels[k].long())


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
    features = triplet_feature_columns(
        i, j, k, rank, h6_inputs=table.h6_inputs(r),
        h6_width=H6_FEATURE_WIDTH, **kw)
    extra_names = feature_names[len(BASE_FEATURE_NAMES):]
    if extra_names:
        values = {}
        if 'filter_score' in extra_names:
            values['filter_score'] = torch.tensor(
                arrays['filter_score'][selected], dtype=torch.float32)
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


def _rank_fractions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks / len(scores)


def event_context_features(arrays: dict, cascade: dict, surviving: np.ndarray,
                           selected) -> torch.Tensor:
    """arrays/cascade: candidate_arrays(r)/cascade_arrays(r). surviving: (S,) and
    selected: (M,) candidate positions, selected a subset of surviving.
    Returns (M, len(CONTEXT_FEATURE_NAMES)) float32."""
    n_surviving = len(surviving)
    if n_surviving == 0:
        return torch.zeros((0, len(CONTEXT_FEATURE_NAMES)), dtype=torch.float32)
    scores = arrays['filter_score'][surviving].astype(np.float64)
    n_couples = len(cascade['couple_scores'])
    couple_rank = arrays['couple_rank'][surviving].astype(np.float64)
    context = np.stack([
        _rank_fractions(scores),
        scores.max() - scores,
        (scores - scores.mean()) / (scores.std() + 1e-6),
        np.full(n_surviving, np.log1p(n_surviving)),
        couple_rank / n_couples if n_couples else np.zeros(n_surviving),
    ], axis=1)
    position_in_surviving = np.empty(int(surviving.max()) + 1, dtype=np.int64)
    position_in_surviving[surviving] = np.arange(n_surviving)
    lookup = position_in_surviving[np.asarray(selected)]
    return torch.tensor(context[lookup], dtype=torch.float32)


def _static_fit_block(table: _EventTable, r: int, i, j, k) -> torch.Tensor:
    kw = table.track_kw(r)
    return static_fit_columns(
        i, j, k, lorentz=kw['lorentz'], eta=kw['eta'], phi=kw['phi'],
        primary_vertex=table.primary_vertex(r), **table.vertex_fit_kw(r))


def fit_norm_stats(candidates_path: str, src_glob: str, *,
                   feature_names: list[str] | None = None, n_events: int = 2000,
                   per_event: int = 50, seed: int = 0, events=None,
                   tau: float | None = None,
                   context_features: bool = False,
                   vertex_fit: str = 'off') -> dict:
    """Median/IQR stats over sampled surviving window candidates; keys =
    feature_names (default BASE_FEATURE_NAMES) plus the static fit block when
    vertex_fit != 'off' and the context block when context_features."""
    if context_features and tau is None:
        raise ValueError('context_features=True requires tau (context is defined '
                         'over the tau-surviving list)')
    table = _EventTable(candidates_path, src_glob)
    if feature_names is None:
        feature_names = list(BASE_FEATURE_NAMES)
    base_names = [name for name in feature_names
                  if name not in CONTEXT_FEATURE_NAMES and name not in FIT_NAMES]
    with_fit = vertex_fit != 'off' or any(name in FIT_NAMES for name in feature_names)
    generator = np.random.default_rng(seed)
    pool = np.arange(table.num_rows) if events is None else np.asarray(events)
    events = generator.choice(pool, min(n_events, len(pool)), replace=False)
    samples = []
    for r in events:
        arrays = table.candidate_arrays(int(r))
        window = arrays['row_kind'] == 0
        if tau is not None:
            window &= arrays['filter_score'] >= tau
        surviving = np.where(window)[0]
        if len(surviving) == 0:
            continue
        take = surviving[generator.choice(len(surviving),
                                          min(per_event, len(surviving)),
                                          replace=False)]
        X, i, j, k = build_candidate_features(table, int(r), arrays, take,
                                              base_names)
        if with_fit:
            X = torch.cat([X, _static_fit_block(table, int(r), i, j, k)], dim=1)
        if context_features:
            context = event_context_features(arrays, table.cascade_arrays(int(r)),
                                             surviving, take)
            X = torch.cat([X, context], dim=1)
        samples.append(X.numpy())
    sample = np.concatenate(samples)
    names = list(base_names) + (list(FIT_NAMES) if with_fit else []) \
        + (list(CONTEXT_FEATURE_NAMES) if context_features else [])
    stats = {}
    for column_index, name in enumerate(names):
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
    """Event-major candidate lists for listwise training and full-list eval
    over the schema-v2 window artifact.

    Serving list = window rows (row_kind 0) at or above the gate tau. Train
    items: all serving positives + `num_negatives` negatives sampled with
    replacement; optionally the tail-sample rows (row_kind 2) with log
    reweighting for an unbiased full-list loss. Eval items: the full serving
    list + sorted 3-set keys for dedup."""

    def __init__(self, candidates_path: str, src_glob: str, *, tau: float,
                 num_negatives: int = 512, mode: str = 'train',
                 norm_stats: dict | None = None, seed: int = 0,
                 extra_features: str = 'auto', context_features: bool = False,
                 vertex_fit: str = 'off', tail_weighting: bool = False,
                 from_b_targets: bool = False):
        assert mode in ('train', 'eval')
        assert vertex_fit in ('off', 'static', 'layer')
        self.table = _EventTable(candidates_path, src_glob)
        self.tau = tau
        self.num_negatives = num_negatives
        self.mode = mode
        self.norm_stats = norm_stats
        self.generator = np.random.default_rng(seed)
        self.vertex_fit = vertex_fit
        self.tail_weighting = tail_weighting and mode == 'train'
        self.from_b_targets = from_b_targets
        self._base_feature_names = resolve_feature_names(self.table, extra_features)
        self.feature_names = list(self._base_feature_names)
        if vertex_fit == 'static':
            self.feature_names += list(FIT_NAMES)
        self.context_features = context_features
        if context_features:
            if not self.table.has_cascade_columns:
                raise ValueError('context_features requires track_s1/track_s2/'
                                 'couple_scores columns in the candidates artifact')
            self.feature_names = self.feature_names + CONTEXT_FEATURE_NAMES

        flat_scores = np.asarray(self.table.candidates['filter_score'].values)
        flat_kind = np.asarray(self.table.candidates['row_kind'].values)
        offsets = np.asarray(self.table.candidates['filter_score'].offsets)
        flat_gt = np.asarray(self.table.candidates['is_gt'].values)
        serving = (flat_scores >= tau) & (flat_kind == 0)
        self.trainable_indices = np.where(
            _segment_any(serving & flat_gt, offsets))[0]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def __len__(self) -> int:
        return self.table.num_rows

    def _select(self, r: int):
        arrays = self.table.candidate_arrays(r)
        serving = (arrays['filter_score'] >= self.tau) & (arrays['row_kind'] == 0)
        return arrays, np.where(serving)[0]

    def __getitem__(self, r: int) -> dict[str, torch.Tensor]:
        arrays, surviving = self._select(int(r))
        is_gt = arrays['is_gt'][surviving]
        log_weights = None
        if self.mode == 'train':
            positive_positions = surviving[is_gt.astype(bool)]
            negative_positions = surviving[~is_gt.astype(bool)]
            if len(negative_positions) == 0:
                selected = positive_positions
            else:
                sampled = self.generator.integers(0, len(negative_positions),
                                                  self.num_negatives)
                selected = np.concatenate([positive_positions,
                                           negative_positions[sampled]])
            pos_mask = np.zeros(len(selected), dtype=bool)
            pos_mask[:len(positive_positions)] = True
            if self.tail_weighting:
                tail = np.where(arrays['row_kind'] == 2)[0]
                if len(tail):
                    log_weight = float(np.log(arrays['n_tail_total'] / len(tail)))
                    log_weights = np.concatenate([
                        np.zeros(len(selected)), np.full(len(tail), log_weight)])
                    selected = np.concatenate([selected, tail])
                    pos_mask = np.concatenate(
                        [pos_mask, np.zeros(len(tail), dtype=bool)])
                else:
                    log_weights = np.zeros(len(selected))
        else:
            selected = surviving
            pos_mask = is_gt.astype(bool)

        features, i, j, k = build_candidate_features(
            self.table, int(r), arrays, selected, self._base_feature_names)
        if self.vertex_fit == 'static':
            features = torch.cat(
                [features, _static_fit_block(self.table, int(r), i, j, k)], dim=1)
        if self.context_features:
            context = event_context_features(
                arrays, self.table.cascade_arrays(int(r)), surviving, selected)
            features = torch.cat([features, context], dim=1)
        if self.norm_stats is not None:
            features = standardize_features(features, self.feature_names,
                                            self.norm_stats)

        scores = np.clip(arrays['filter_score'][selected].astype(np.float64),
                         _LOGIT_EPS, 1.0 - _LOGIT_EPS)
        item = {
            'features': features,
            'pos_mask': torch.from_numpy(pos_mask),
            'filter_logit': torch.tensor(np.log(scores / (1.0 - scores)),
                                         dtype=torch.float32),
        }
        if log_weights is not None:
            item['log_weights'] = torch.tensor(log_weights, dtype=torch.float32)
        if self.from_b_targets:
            item['from_b'] = self.table.from_b_counts(int(r), i, j, k)
        if self.vertex_fit == 'layer':
            item.update(self._fit_layer_inputs(int(r), i, j, k))
        if self.mode == 'eval':
            item['keys'] = torch.stack([i, j, k], dim=1).sort(dim=1).values
        return item

    def _fit_layer_inputs(self, r: int, i, j, k) -> dict[str, torch.Tensor]:
        kw = self.table.track_kw(r)
        fit = self.table.vertex_fit_kw(r)
        members = torch.stack([i, j, k], dim=0)
        reference = torch.stack([
            torch.stack([fit['vertex_x'][members[m]], fit['vertex_y'][members[m]],
                         fit['vertex_z'][members[m]]], dim=0)
            for m in range(3)], dim=0)
        momentum = sum(kw['lorentz'][:3, members[m]] for m in range(3))
        energy = sum(kw['lorentz'][3, members[m]] for m in range(3))
        mass = (energy.square() - momentum.square().sum(dim=0)) \
            .clamp_min(0.0).sqrt()
        quality = self.table.quality_channels(r)
        return {
            'fit_reference': reference,
            'fit_eta': torch.stack([kw['eta'][members[m]] for m in range(3)]),
            'fit_phi': torch.stack([kw['phi'][members[m]] for m in range(3)]),
            'fit_var_dxy': torch.stack(
                [fit['var_dxy'][members[m]] for m in range(3)]),
            'fit_var_dsz': torch.stack(
                [fit['var_dsz'][members[m]] for m in range(3)]),
            'fit_momentum': momentum,
            'fit_mass': mass,
            'fit_quality': torch.stack(
                [quality[members[m]] for m in range(3)], dim=1).permute(2, 1, 0),
            'primary_vertex': self.table.primary_vertex(r),
        }


_FIT_PAD_KEYS = ['fit_reference', 'fit_eta', 'fit_phi', 'fit_var_dxy',
                 'fit_var_dsz', 'fit_momentum', 'fit_quality']


def collate_triplet_rank(items: list[dict]) -> dict[str, torch.Tensor]:
    """Pads to the batch-max candidate count. Returns features (B, F, N) for Conv1d."""
    batch_size = len(items)
    max_candidates = max(item['features'].shape[0] for item in items)
    num_features = items[0]['features'].shape[1]
    features = torch.zeros(batch_size, num_features, max_candidates)
    pos_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    valid_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    filter_logit = torch.zeros(batch_size, max_candidates)
    for b, item in enumerate(items):
        n = item['features'].shape[0]
        features[b, :, :n] = item['features'].T
        pos_mask[b, :n] = item['pos_mask']
        valid_mask[b, :n] = True
        filter_logit[b, :n] = item['filter_logit']
    batch = {'features': features, 'pos_mask': pos_mask,
             'valid_mask': valid_mask, 'filter_logit': filter_logit}
    if 'log_weights' in items[0]:
        log_weights = torch.zeros(batch_size, max_candidates)
        for b, item in enumerate(items):
            log_weights[b, :item['log_weights'].shape[0]] = item['log_weights']
        batch['log_weights'] = log_weights
    if 'from_b' in items[0]:
        from_b = torch.zeros(batch_size, max_candidates, dtype=torch.long)
        for b, item in enumerate(items):
            from_b[b, :item['from_b'].shape[0]] = item['from_b']
        batch['from_b'] = from_b
    if 'fit_reference' in items[0]:
        for key in _FIT_PAD_KEYS:
            shape = items[0][key].shape[:-1]
            padded = torch.zeros(batch_size, *shape, max_candidates)
            for b, item in enumerate(items):
                padded[b, ..., :item[key].shape[-1]] = item[key]
            batch[key] = padded
        mass = torch.zeros(batch_size, max_candidates)
        for b, item in enumerate(items):
            mass[b, :item['fit_mass'].shape[0]] = item['fit_mass']
        batch['fit_mass'] = mass
        batch['primary_vertex'] = torch.stack(
            [item['primary_vertex'] for item in items])
    return batch


def collate_triplet_rank_eval(items: list[dict]) -> dict:
    """collate_triplet_rank plus each item's dedup keys (variable-length) and its
    true candidate count."""
    batch = collate_triplet_rank(items)
    batch['keys'] = [item['keys'] for item in items]
    batch['counts'] = torch.tensor([item['features'].shape[0] for item in items])
    return batch
