"""Extract the collaboration's baseline blocks of the abcbatch production into
one parquet file per ROOT job file, one row per ROOT entry (no event
selection), keyed by the same composite event key as the model-input
conversion (event_identity):

* identity + selection: the 5-key, event_source_job / entry, the converter's
  clean mask, our funnel bucket (ntuple_truth.select_event, cut first then
  count) and the GT raw track indices;
* gen truth: tau / top-most B indices, true PV, decay vertex, decay lengths,
  gen pion indices and their matched raw track indices;
* KstarCutflow_* counters;
* K* candidates (KstarToPiKAll, shared map with the converter), which of them
  are true, and the selected K* (KstarToPiK[0]);
* ABCNet per-(K*, track) rows: raw track index, K* row, score, dz, dxy;
* the TauTo3Pi builder triplets with the GT-hit flag and per-triplet
  aggregates of the B candidates (count, best svprob and its K* row,
  fit mass of that row) plus event-level B counts.

Index spaces: *_root_index columns address the raw Track table (the model
parquet's track_root_index); *_row columns address this row's own lists
(kstar_* = KstarToPiKAll order, tau_* = TauTo3Pi order).

Usage:
    python3 extract_baseline_blocks.py \\
        --input-dir /eos/.../abcbatch0_2024 \\
        --pattern 'abcbatch*_job*_ditaus_mc.root' \\
        --output-dir /eos/.../abcbatch_2024/baseline/batch0 \\
        [--stripe 0 --num-stripes 1] [--entry-stop N] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import time

import awkward as ak
import numpy as np

import convert_root_to_parquet as converter
import event_identity
import ntuple_truth
from ntuple_truth import FUNNEL_BUCKETS

DEFAULT_PATTERN = 'abcbatch*_job*_ditaus_mc.root'
CUTFLOW_PREFIX = 'KstarCutflow_'

TRACK_BRANCHES = ['nTrack', 'Track_pt', 'Track_pdgId', 'Track_trackFromTau',
                  'Track_trackFromKstar']
PV_BRANCHES = ['PV_x', 'PV_y', 'PV_z']
GEN_BRANCHES = ['GenPart_pdgId', 'GenPart_status', 'GenPart_genPartIdxMother',
                'GenPart_pt', 'GenPart_vx', 'GenPart_vy', 'GenPart_vz',
                'GenPart_gentrackIdx']
KSTAR_SELECTED_BRANCHES = list(converter.KSTAR_SELECTED_BRANCH_MAP)
KSTAR_ALL_BRANCH_MAP = dict(converter.KSTAR_ALL_BRANCH_MAP)
KSTAR_ALL_INTEGER_COLUMNS = set(converter.KSTAR_ALL_INTEGER_COLUMNS)
ABCNET_BRANCH_MAP = {
    'ABCNetTau_trk_idx': 'abcnet_track_root_index',
    'ABCNetTau_kstar_idx': 'abcnet_kstar_row',
    'ABCNetTau_score': 'abcnet_score',
    'ABCNetTau_dz': 'abcnet_dz',
    'ABCNetTau_dxy': 'abcnet_dxy',
}
ABCNET_INTEGER_COLUMNS = {'abcnet_track_root_index', 'abcnet_kstar_row'}
# The builder's 11 branches of the converter's --builder-blocks (same column
# names) plus the four mass branches new in the abcbatch production.
TAU_BRANCH_MAP = {
    **converter.TAU_BUILDER_BRANCH_MAP,
    'TauTo3Pi_rho_mass_best': 'tau_rho_mass_best',
    'TauTo3Pi_rho_mass_high': 'tau_rho_mass_high',
    'TauTo3Pi_rho_mass_low': 'tau_rho_mass_low',
    'TauTo3Pi_ss_pair_mass': 'tau_same_sign_pair_mass',
}
TAU_INTEGER_COLUMNS = set(converter.TAU_BUILDER_INTEGER_COLUMNS)
TAU_B_AGGREGATE_COLUMNS = ['tau_n_b_rows', 'tau_b_best_kstar_row',
                           'tau_b_best_vertex_probability', 'tau_b_best_fit_mass']
# B candidates: only the four branches the aggregation needs (of 56).
B_BRANCHES = ['BToKstarTauTau_tau_idx', 'BToKstarTauTau_kstar_idx',
              'BToKstarTauTau_svprob', 'BToKstarTauTau_fit_mass']

GEN_SCALAR_FLOAT_COLUMNS = [
    'gen_tau_pt', 'gen_tau_vertex_x', 'gen_tau_vertex_y', 'gen_tau_vertex_z',
    'gen_tau_decay_vertex_x', 'gen_tau_decay_vertex_y', 'gen_tau_decay_vertex_z',
    'gen_true_pv_x', 'gen_true_pv_y', 'gen_true_pv_z',
    'gen_decay_length_3d', 'gen_decay_length_xy',
]


# ---------------------------------------------------------------------------
# Per-event helpers (numpy; raw Track / GenPart / KstarToPiKAll index spaces)
# ---------------------------------------------------------------------------

def converter_clean_mask_chunk(tree, entry_start, entry_stop):
    """tree: uproot Events tree; [entry_start, entry_stop) window. Returns
    (N,) bool: the converter's own event-quality mask, computed by the same
    function and branch map the model conversion uses."""
    _, clean = converter.read_and_filter_pion_tracks(
        tree, converter.build_track_branch_map(converter.DEFAULT_COLUMNS),
        entry_start, entry_stop)
    return np.asarray(clean, dtype=bool)


def selection_block(track_pt, track_pdg, track_from_tau, clean, pt_cutoff):
    """track_pt, track_pdg, track_from_tau: (T,) raw Track arrays; clean:
    bool. Returns the funnel bucket ('quality_dropped' when not clean), the
    sorted GT raw indices (3,) or an empty array, and the track / GT counts."""
    is_pion = np.abs(track_pdg) == ntuple_truth.PION_PDG
    is_gt = is_pion & (track_from_tau == 1)
    above = track_pt >= pt_cutoff
    if clean:
        bucket, gt_indices = ntuple_truth.select_event(track_pt, track_pdg, track_from_tau,
                                                       pt_cutoff=pt_cutoff)
    else:
        bucket, gt_indices = 'quality_dropped', None
    gt_root_index = (np.asarray(gt_indices, dtype=np.int32) if gt_indices is not None
                     else np.empty(0, dtype=np.int32))
    return {
        'bucket': bucket,
        'gt_root_index': gt_root_index,
        'n_tracks_raw': int(track_pt.size),
        'n_tracks': int(np.sum(is_pion & above)),
        'n_gt_flagged': int(np.sum(is_gt)),
        'n_gt_above_cutoff': int(np.sum(is_gt & above)),
        'is_selected': int(bucket == 'selected'),
    }


def gen_truth_block(gen_pdg, gen_mother, gen_status, gen_pt, gen_vertex, gen_track_index):
    """gen_pdg, gen_mother, gen_status, gen_pt, gen_track_index: (G,);
    gen_vertex: (G, 3). Returns the gen scalars (NaN / -1 when the tau chain
    is absent) and the gen pion lists (GenPart rows, matched raw track
    indices, pT)."""
    truth = ntuple_truth.gen_truth(gen_pdg, gen_mother, gen_status, gen_pt, gen_vertex)
    pions = truth['pion_indices']
    block = {
        'gen_tau_index': int(truth['tau_index']),
        'gen_b_index': int(truth['b_index']),
        'gen_b_pdg_id': int(gen_pdg[truth['b_index']]) if truth['b_index'] >= 0 else 0,
        'gen_pion_index': pions.astype(np.int32),
        'gen_pion_track_root_index': (gen_track_index[pions].astype(np.int32) if pions.size
                                      else np.empty(0, dtype=np.int32)),
        'gen_pion_pt': (gen_pt[pions].astype(np.float32) if pions.size
                        else np.empty(0, dtype=np.float32)),
    }
    for name in GEN_SCALAR_FLOAT_COLUMNS:
        block[name] = np.nan
    if truth['tau_index'] >= 0:
        block['gen_tau_pt'] = truth['tau_pt']
        for axis, suffix in enumerate('xyz'):
            block[f'gen_tau_vertex_{suffix}'] = truth['tau_vertex'][axis]
            block[f'gen_tau_decay_vertex_{suffix}'] = truth['decay_vertex'][axis]
            if truth['true_pv'] is not None:
                block[f'gen_true_pv_{suffix}'] = truth['true_pv'][axis]
        block['gen_decay_length_3d'] = truth['decay_length_3d']
        block['gen_decay_length_xy'] = truth['decay_length_xy']
    return block


def kstar_block(all_trk1, all_trk2, from_kstar, selected_trk1, selected_trk2):
    """all_trk1, all_trk2: (K,) KstarToPiKAll daughters; from_kstar: (T,);
    selected_trk1/2: the KstarToPiK[0] pair or None. Returns kstar_is_true
    (K,) int32, the selected row (-1) and the selected pair (-1, -1)."""
    rows = ntuple_truth.kstar_rows(all_trk1, all_trk2, from_kstar, selected_trk1, selected_trk2)
    is_true = np.zeros(all_trk1.size, dtype=np.int32)
    is_true[rows['true_rows']] = 1
    selected_row = -1 if rows['selected_row'] is None else int(rows['selected_row'])
    pair = (-1 if selected_trk1 is None else int(selected_trk1),
            -1 if selected_trk2 is None else int(selected_trk2))
    return {'kstar_is_true': is_true, 'selected_row': selected_row, 'selected_pair': pair}


def builder_block(idx1, idx2, idx3, gt_root_index):
    """idx1, idx2, idx3: (C,) TauTo3Pi raw track indices; gt_root_index:
    (3,) or empty. Returns is_gt_triplet (C,) int32 and the first hit row
    (-1 when none)."""
    if idx1.size == 0 or gt_root_index.size != 3:
        hits = np.zeros(idx1.size, dtype=np.int32)
    else:
        hits = ntuple_truth.sorted_triplet_hit(idx1, idx2, idx3, gt_root_index).astype(np.int32)
    hit_rows = np.flatnonzero(hits)
    return {'is_gt_triplet': hits, 'gt_row': int(hit_rows[0]) if hit_rows.size else -1}


def aggregate_b_rows(tau_idx, kstar_idx, svprob, fit_mass, n_tau_rows):
    """tau_idx, kstar_idx: (R,) B-candidate rows -> TauTo3Pi row /
    KstarToPiKAll row; svprob, fit_mass: (R,); n_tau_rows: int. Returns
    (per builder row, per event): the B-row count per builder row, the row
    with the highest finite svprob among them (first on ties: its svprob,
    K* row and fit mass; -1 / -1.0 without such a row), and the event-level
    row count, distinct-triplet count and overall best row. Rows with a
    non-finite svprob (NaN fits) count in n_rows but never rank."""
    n_tau_rows = int(n_tau_rows)
    per_row = {
        'n_rows': np.zeros(n_tau_rows, dtype=np.int32),
        'best_vertex_probability': np.full(n_tau_rows, -1.0, dtype=np.float32),
        'best_kstar_row': np.full(n_tau_rows, -1, dtype=np.int32),
        'best_fit_mass': np.full(n_tau_rows, -1.0, dtype=np.float32),
    }
    event = {'n_rows': int(tau_idx.size), 'n_unique_triplets': 0, 'best_tau_row': -1,
             'best_kstar_row': -1, 'best_vertex_probability': -1.0}
    valid = (tau_idx >= 0) & (tau_idx < n_tau_rows)
    if not np.any(valid):
        return per_row, event
    tau_idx, kstar_idx, svprob, fit_mass = (np.asarray(array)[valid]
                                            for array in (tau_idx, kstar_idx, svprob, fit_mass))
    per_row['n_rows'] = np.bincount(tau_idx, minlength=n_tau_rows).astype(np.int32)
    event['n_unique_triplets'] = int(np.unique(tau_idx).size)
    finite = np.isfinite(svprob)
    if not np.any(finite):
        return per_row, event
    tau_idx, kstar_idx, svprob, fit_mass = (array[finite]
                                            for array in (tau_idx, kstar_idx, svprob, fit_mass))
    # primary key tau row, then svprob descending, then the original row
    # order (lexsort sorts by the last key first)
    order = np.lexsort((np.arange(tau_idx.size), -svprob.astype(np.float64), tau_idx))
    sorted_tau = tau_idx[order]
    first_of_group = np.ones(order.size, dtype=bool)
    first_of_group[1:] = sorted_tau[1:] != sorted_tau[:-1]
    best_rows = order[first_of_group]
    builder_rows = tau_idx[best_rows]
    per_row['best_vertex_probability'][builder_rows] = svprob[best_rows]
    per_row['best_kstar_row'][builder_rows] = kstar_idx[best_rows]
    per_row['best_fit_mass'][builder_rows] = fit_mass[best_rows]
    best = int(np.argmax(svprob))
    event.update(best_tau_row=int(tau_idx[best]), best_kstar_row=int(kstar_idx[best]),
                 best_vertex_probability=float(svprob[best]))
    return per_row, event


def rename_and_cast(data, branch_map, integer_columns, event_mask=None):
    """data: awkward record; branch_map: ROOT branch -> column. Returns the
    renamed jagged columns cast to int32 / float32."""
    block = {column: (data[branch] if event_mask is None else data[branch][event_mask])
             for branch, column in branch_map.items()}
    return converter.cast_to_float32(block, integer_columns=integer_columns)


def jagged(arrays, dtype):
    """arrays: list of 1-D numpy arrays (one per event). Returns a typed
    jagged awkward array."""
    counts = np.asarray([array.size for array in arrays], dtype=np.int64)
    flat = (np.concatenate(arrays).astype(dtype) if counts.sum()
            else np.empty(0, dtype=dtype))
    return ak.unflatten(flat, counts)


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------

def cutflow_branches_of(tree):
    return sorted(name for name in tree.keys() if name.startswith(CUTFLOW_PREFIX))


def build_chunk_table(tree, path, entry_start, entry_stop, cutflow_branches, pt_cutoff):
    """tree: uproot Events tree; path: its file; [entry_start, entry_stop)
    window. Returns the awkward record of the baseline columns for the
    window (one row per entry)."""
    identity = event_identity.identity_columns(tree, path, entry_start, entry_stop)
    clean = converter_clean_mask_chunk(tree, entry_start, entry_stop)
    branches = (TRACK_BRANCHES + PV_BRANCHES + GEN_BRANCHES + KSTAR_SELECTED_BRANCHES
                + list(KSTAR_ALL_BRANCH_MAP) + list(ABCNET_BRANCH_MAP) + list(TAU_BRANCH_MAP)
                + cutflow_branches)
    data = tree.arrays(expressions=branches, entry_start=entry_start, entry_stop=entry_stop)
    b_data = tree.arrays(expressions=B_BRANCHES, entry_start=entry_start, entry_stop=entry_stop)
    n_events = entry_stop - entry_start

    columns = dict(identity)
    columns['event_n_tracks_raw'] = np.asarray(data['nTrack']).astype(np.int32)
    columns['event_converter_clean'] = clean.astype(np.int32)
    for branch, column in zip(PV_BRANCHES, ('event_primary_vertex_x', 'event_primary_vertex_y',
                                            'event_primary_vertex_z')):
        columns[column] = np.asarray(data[branch]).astype(np.float32)
    for branch in cutflow_branches:
        columns['cutflow_' + branch[len(CUTFLOW_PREFIX):]] = \
            np.asarray(data[branch]).astype(np.int32)

    per_event = {name: [] for name in (
        'bucket', 'gt_root_index', 'n_tracks', 'n_gt_flagged', 'n_gt_above_cutoff', 'is_selected',
        'gen_tau_index', 'gen_b_index', 'gen_b_pdg_id', 'gen_pion_index',
        'gen_pion_track_root_index', 'gen_pion_pt', *GEN_SCALAR_FLOAT_COLUMNS,
        'kstar_is_true', 'kstar_selected_row', 'kstar_selected_track1', 'kstar_selected_track2',
        'n_kstar_selected', 'n_abcnet_lists',
        'tau_is_gt_triplet', 'tau_gt_row', *TAU_B_AGGREGATE_COLUMNS,
        'b_n_rows', 'b_n_unique_triplets', 'b_n_gt_rows', 'b_best_tau_row', 'b_best_kstar_row',
        'b_best_vertex_probability')}
    for index in range(n_events):
        track_pt = ak.to_numpy(data['Track_pt'][index])
        track_pdg = ak.to_numpy(data['Track_pdgId'][index])
        track_from_tau = ak.to_numpy(data['Track_trackFromTau'][index])
        track_from_kstar = ak.to_numpy(data['Track_trackFromKstar'][index])

        selection = selection_block(track_pt, track_pdg, track_from_tau, bool(clean[index]), pt_cutoff)
        for name in ('bucket', 'gt_root_index', 'n_tracks', 'n_gt_flagged',
                     'n_gt_above_cutoff', 'is_selected'):
            per_event[name].append(selection[name])

        gen_vertex = np.stack([ak.to_numpy(data[branch][index])
                               for branch in ('GenPart_vx', 'GenPart_vy', 'GenPart_vz')], axis=1)
        gen = gen_truth_block(
            ak.to_numpy(data['GenPart_pdgId'][index]),
            ak.to_numpy(data['GenPart_genPartIdxMother'][index]).astype(np.int64),
            ak.to_numpy(data['GenPart_status'][index]),
            ak.to_numpy(data['GenPart_pt'][index]), gen_vertex,
            ak.to_numpy(data['GenPart_gentrackIdx'][index]).astype(np.int64))
        for name, value in gen.items():
            per_event[name].append(value)

        selected_trk1 = ak.to_numpy(data['KstarToPiK_trk1_idx'][index])
        selected_trk2 = ak.to_numpy(data['KstarToPiK_trk2_idx'][index])
        kstar = kstar_block(
            ak.to_numpy(data['KstarToPiKAll_trk1_idx'][index]).astype(np.int64),
            ak.to_numpy(data['KstarToPiKAll_trk2_idx'][index]).astype(np.int64),
            track_from_kstar,
            int(selected_trk1[0]) if selected_trk1.size else None,
            int(selected_trk2[0]) if selected_trk2.size else None)
        per_event['kstar_is_true'].append(kstar['kstar_is_true'])
        per_event['kstar_selected_row'].append(kstar['selected_row'])
        per_event['kstar_selected_track1'].append(kstar['selected_pair'][0])
        per_event['kstar_selected_track2'].append(kstar['selected_pair'][1])
        per_event['n_kstar_selected'].append(int(selected_trk1.size))

        abcnet_trk = ak.to_numpy(data['ABCNetTau_trk_idx'][index])
        abcnet_kstar = ak.to_numpy(data['ABCNetTau_kstar_idx'][index])
        per_event['n_abcnet_lists'].append(ntuple_truth.abcnet_lists(abcnet_trk, abcnet_kstar)['n_lists'])

        tau_idx1 = ak.to_numpy(data['TauTo3Pi_idx1'][index]).astype(np.int64)
        builder = builder_block(tau_idx1, ak.to_numpy(data['TauTo3Pi_idx2'][index]).astype(np.int64),
                                ak.to_numpy(data['TauTo3Pi_idx3'][index]).astype(np.int64),
                                selection['gt_root_index'])
        per_event['tau_is_gt_triplet'].append(builder['is_gt_triplet'])
        per_event['tau_gt_row'].append(builder['gt_row'])
        b_per_row, b_event = aggregate_b_rows(
            ak.to_numpy(b_data['BToKstarTauTau_tau_idx'][index]).astype(np.int64),
            ak.to_numpy(b_data['BToKstarTauTau_kstar_idx'][index]).astype(np.int64),
            ak.to_numpy(b_data['BToKstarTauTau_svprob'][index]),
            ak.to_numpy(b_data['BToKstarTauTau_fit_mass'][index]), tau_idx1.size)
        per_event['tau_n_b_rows'].append(b_per_row['n_rows'])
        per_event['tau_b_best_kstar_row'].append(b_per_row['best_kstar_row'])
        per_event['tau_b_best_vertex_probability'].append(b_per_row['best_vertex_probability'])
        per_event['tau_b_best_fit_mass'].append(b_per_row['best_fit_mass'])
        per_event['b_n_rows'].append(b_event['n_rows'])
        per_event['b_n_unique_triplets'].append(b_event['n_unique_triplets'])
        per_event['b_n_gt_rows'].append(int(b_per_row['n_rows'][builder['is_gt_triplet'] == 1].sum()))
        per_event['b_best_tau_row'].append(b_event['best_tau_row'])
        per_event['b_best_kstar_row'].append(b_event['best_kstar_row'])
        per_event['b_best_vertex_probability'].append(b_event['best_vertex_probability'])

    columns['event_n_tracks'] = np.asarray(per_event['n_tracks'], dtype=np.int32)
    columns['event_selection_bucket'] = ak.Array(per_event['bucket'])
    columns['event_is_selected'] = np.asarray(per_event['is_selected'], dtype=np.int32)
    columns['event_n_gt_flagged'] = np.asarray(per_event['n_gt_flagged'], dtype=np.int32)
    columns['event_n_gt_above_cutoff'] = np.asarray(per_event['n_gt_above_cutoff'], dtype=np.int32)
    columns['gt_root_index'] = jagged(per_event['gt_root_index'], np.int32)

    for name in ('gen_tau_index', 'gen_b_index', 'gen_b_pdg_id'):
        columns[name] = np.asarray(per_event[name], dtype=np.int32)
    for name in GEN_SCALAR_FLOAT_COLUMNS:
        columns[name] = np.asarray(per_event[name], dtype=np.float32)
    columns['gen_pion_index'] = jagged(per_event['gen_pion_index'], np.int32)
    columns['gen_pion_track_root_index'] = jagged(per_event['gen_pion_track_root_index'], np.int32)
    columns['gen_pion_pt'] = jagged(per_event['gen_pion_pt'], np.float32)

    columns.update(rename_and_cast(data, KSTAR_ALL_BRANCH_MAP, KSTAR_ALL_INTEGER_COLUMNS))
    columns['kstar_is_true'] = jagged(per_event['kstar_is_true'], np.int32)
    columns['event_n_kstar_all'] = np.asarray(ak.num(data['KstarToPiKAll_trk1_idx'], axis=1)).astype(np.int32)
    columns['event_n_kstar_selected'] = np.asarray(per_event['n_kstar_selected'], dtype=np.int32)
    columns['event_kstar_selected_row'] = np.asarray(per_event['kstar_selected_row'], dtype=np.int32)
    columns['event_kstar_selected_track1_root_index'] = \
        np.asarray(per_event['kstar_selected_track1'], dtype=np.int32)
    columns['event_kstar_selected_track2_root_index'] = \
        np.asarray(per_event['kstar_selected_track2'], dtype=np.int32)

    columns.update(rename_and_cast(data, ABCNET_BRANCH_MAP, ABCNET_INTEGER_COLUMNS))
    columns['event_n_abcnet_rows'] = np.asarray(ak.num(data['ABCNetTau_trk_idx'], axis=1)).astype(np.int32)
    columns['event_n_abcnet_lists'] = np.asarray(per_event['n_abcnet_lists'], dtype=np.int32)

    columns.update(rename_and_cast(data, TAU_BRANCH_MAP, TAU_INTEGER_COLUMNS))
    columns['tau_is_gt_triplet'] = jagged(per_event['tau_is_gt_triplet'], np.int32)
    columns['tau_n_b_rows'] = jagged(per_event['tau_n_b_rows'], np.int32)
    columns['tau_b_best_kstar_row'] = jagged(per_event['tau_b_best_kstar_row'], np.int32)
    columns['tau_b_best_vertex_probability'] = jagged(per_event['tau_b_best_vertex_probability'], np.float32)
    columns['tau_b_best_fit_mass'] = jagged(per_event['tau_b_best_fit_mass'], np.float32)
    columns['event_n_tau_candidates'] = np.asarray(ak.num(data['TauTo3Pi_idx1'], axis=1)).astype(np.int32)
    columns['event_tau_gt_row'] = np.asarray(per_event['tau_gt_row'], dtype=np.int32)
    for name, column in (('b_n_rows', 'event_n_b_rows'),
                         ('b_n_unique_triplets', 'event_n_b_unique_triplets'),
                         ('b_n_gt_rows', 'event_n_b_gt_rows'),
                         ('b_best_tau_row', 'event_b_best_tau_row'),
                         ('b_best_kstar_row', 'event_b_best_kstar_row')):
        columns[column] = np.asarray(per_event[name], dtype=np.int32)
    columns['event_b_best_vertex_probability'] = \
        np.asarray(per_event['b_best_vertex_probability'], dtype=np.float32)
    del b_data

    return ak.Array(columns)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def output_name_for(path):
    """path: ROOT file. Returns baseline_batch{B}_job{J}.parquet, or
    baseline_{stem}.parquet when the name carries no job token."""
    job = event_identity.job_number_from_filename(path)
    if job is None:
        return f'baseline_{os.path.splitext(os.path.basename(path))[0]}.parquet'
    batch = event_identity.batch_number_from_filename(path) or 0
    return f'baseline_batch{batch}_job{job}.parquet'


def extract_file(path, output_path, entry_stop=None, chunk_size=100, pt_cutoff=0.5):
    """path: ROOT file; output_path: parquet to write. Processes the first
    entry_stop entries (all when None) in windows of chunk_size. Returns the
    number of rows written (one per entry)."""
    tree = converter.load_root_tree(path)
    n_entries = tree.num_entries if entry_stop is None else min(int(entry_stop), tree.num_entries)
    cutflow_branches = cutflow_branches_of(tree)
    chunks = [build_chunk_table(tree, path, start, min(start + chunk_size, n_entries),
                                cutflow_branches, pt_cutoff)
              for start in range(0, n_entries, chunk_size)]
    if not chunks:
        raise ValueError(f'{path} has no entries to extract')
    table = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    # Written and validated under a temporary name, then renamed: a job
    # killed mid-write never leaves a partial file that a resubmission
    # would skip as "exists".
    temporary_path = output_path + '.tmp'
    converter.write_parquet_shard(table, temporary_path)
    validate_baseline_file(temporary_path, n_entries)
    os.replace(temporary_path, output_path)
    return len(table)


JAGGED_BLOCKS = {
    'kstar': ['kstar_track1_root_index', 'kstar_track2_root_index', 'kstar_vertex_x',
              'kstar_vertex_probability', 'kstar_charge', 'kstar_is_true'],
    'abcnet': ['abcnet_track_root_index', 'abcnet_kstar_row', 'abcnet_score', 'abcnet_dz', 'abcnet_dxy'],
    'tau': ['tau_track1_index', 'tau_track2_index', 'tau_track3_index', 'tau_vertex_probability',
            'tau_rho_mass_best', 'tau_same_sign_pair_mass', 'tau_is_gt_triplet', *TAU_B_AGGREGATE_COLUMNS],
    'gen_pion': ['gen_pion_index', 'gen_pion_track_root_index', 'gen_pion_pt'],
}


def validate_baseline_file(path, expected_rows):
    """path: baseline parquet; expected_rows: ROOT entries processed. Checks
    the row count, the uniqueness of the 5-key, equal list lengths inside
    every jagged block, gt_root_index of length 0 or 3, the bucket /
    is_selected agreement and at most one GT triplet row per event.

    Raises:
        ValueError: If any constraint is violated.
    """
    data = ak.from_parquet(path)
    basename = os.path.basename(path)
    if len(data) != expected_rows:
        raise ValueError(f'{basename}: {len(data)} rows, expected {expected_rows} rows')
    event_identity.assert_unique_identity(
        {name: ak.to_numpy(data[name]) for name in
         event_identity.IDENTITY_COLUMNS + event_identity.SOURCE_DETAIL_COLUMNS},
        context=basename)
    for block, columns in JAGGED_BLOCKS.items():
        missing = [column for column in columns if column not in data.fields]
        if missing:
            raise ValueError(f'{basename}: block {block} lacks {missing}')
        reference = ak.num(data[columns[0]], axis=1)
        for column in columns[1:]:
            if not ak.all(ak.num(data[column], axis=1) == reference):
                raise ValueError(f'{basename}: {column} length differs inside block {block}')
    gt_lengths = ak.to_numpy(ak.num(data['gt_root_index'], axis=1))
    if not np.all(np.isin(gt_lengths, (0, 3))):
        raise ValueError(f'{basename}: gt_root_index must hold 0 or 3 indices')
    buckets = np.asarray(ak.to_list(data['event_selection_bucket']))
    selected = ak.to_numpy(data['event_is_selected']) == 1
    if not np.array_equal(buckets == 'selected', selected):
        raise ValueError(f'{basename}: event_is_selected disagrees with the bucket')
    if not ak.all(ak.sum(data['tau_is_gt_triplet'], axis=1) <= 1):
        raise ValueError(f'{basename}: more than one GT triplet row in an event')


def run(input_dir, pattern, output_dir, entry_stop=None, chunk_size=100, pt_cutoff=0.5,
        stripe=0, num_stripes=1, overwrite=False):
    """Processes the files of input_dir matching pattern whose index in the
    shared sorted order satisfies index % num_stripes == stripe. Returns
    [(output path, rows written or None when skipped)]."""
    files = converter.discover_root_files(input_dir, pattern=pattern)
    event_identity.assert_unique_file_keys(files)
    files = [path for index, path in enumerate(files) if index % num_stripes == stripe]
    print(f'{len(files)} file(s) in stripe {stripe}/{num_stripes}')
    start_time = time.time()
    written = []
    for position, path in enumerate(files, 1):
        output_path = os.path.join(output_dir, output_name_for(path))
        if os.path.exists(output_path) and not overwrite:
            print(f'[{position}/{len(files)}] {os.path.basename(output_path)} exists, skipping')
            written.append((output_path, None))
            continue
        rows = extract_file(path, output_path, entry_stop=entry_stop,
                            chunk_size=chunk_size, pt_cutoff=pt_cutoff)
        elapsed = time.time() - start_time
        print(f'[{position}/{len(files)}] {os.path.basename(path)} -> '
              f'{os.path.basename(output_path)} ({rows} rows, {elapsed:.0f} s elapsed)')
        written.append((output_path, rows))
    return written


def main():
    parser = argparse.ArgumentParser(
        description='Extract the ABCNet / K* / builder / B-candidate baseline blocks '
                    'and truth of abcbatch job files into per-file parquets.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--pattern', default=DEFAULT_PATTERN)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--entry-stop', type=int, default=None,
                        help='Process only the first N entries of every file')
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--pt-cutoff', type=float, default=0.5,
                        help='Must equal the model conversion cutoff (selection bucket)')
    parser.add_argument('--stripe', type=int, default=0,
                        help='Process files whose index in the sorted list satisfies '
                             'index %% num_stripes == stripe')
    parser.add_argument('--num-stripes', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true',
                        help='Rewrite existing outputs (default: skip them)')
    arguments = parser.parse_args()
    run(arguments.input_dir, arguments.pattern, arguments.output_dir,
        entry_stop=arguments.entry_stop, chunk_size=arguments.chunk_size,
        pt_cutoff=arguments.pt_cutoff, stripe=arguments.stripe,
        num_stripes=arguments.num_stripes, overwrite=arguments.overwrite)


if __name__ == '__main__':
    main()
