"""Survey of the abcbatch0_2024 ditau ntuples (ABCNet output, analysis chain,
gen truth) against the schema and selection our cascade is trained on.

Read-only. Reads only the branches it needs (the combinatorial
BToKstarTauTau / KstarToPiKAll kinematics are never loaded), computes one
flat record per event, and summarises per file and pooled into JSON plus
markdown tables for reports/abcbatch_ntuple_survey_<date>.md.

Usage:
    python -m scripts.python.survey_abcbatch_ntuples \\
        --output-json reports/abcbatch_ntuple_survey_20261104.json \\
        --output-markdown reports/abcbatch_ntuple_survey_tables_20261104.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import awkward as ak
import numpy as np
import uproot

from scripts.python.probe_triplet_features import roc_auc
from scripts.python.pv_reassociation_check import distribution_summary
from utils import tau_muon_proxy as proxy

ABCNET_WORKING_POINT = 0.1443
RECALL_KS = (3, 5, 10, 20, 50)
PV_FAR_CM = (0.1, 0.2, 0.5)
IMPOSTOR_WINDOWS = {'wide': (0.5, 0.2), 'tight': (0.2, 0.05)}
GEN_PT_BINS = (0.0, 0.3, 0.5, 1.0, 2.0, np.inf)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONVERTER_DIR = os.path.join(REPO_ROOT, 'scripts', 'condor', 'convert_to_parquet')
MERGE_DIR = os.path.join(REPO_ROOT, 'scripts', 'condor', 'merge_root_batches')
if CONVERTER_DIR not in sys.path:
    sys.path.insert(0, CONVERTER_DIR)
from ntuple_truth import (  # noqa: E402  (shared with the condor scripts; re-exported)
    B_PDGS, FUNNEL_BUCKETS, PION_PDG, PT_CUTOFF, TAU_PDG, abcnet_lists, gen_tau_pions,
    gen_truth, kstar_rows, select_event, sorted_triplet_hit, top_b_ancestor, walk_to_ancestor,
)
DEFAULT_INPUT_GLOB = os.path.join(
    REPO_ROOT, 'data', 'low-pt', 'abcbatch', 'abcbatch0_2024_job*_ditaus_mc.root')
DEFAULT_REFERENCE_ROOT = os.path.join(
    REPO_ROOT, 'data', 'thesis_files', 'step_MINI_10_nano_ditaus_mc.root')

SURVEY_BRANCHES = [
    'run', 'event', 'luminosityBlock',
    'nTrack', 'Track_pt', 'Track_eta', 'Track_phi', 'Track_charge',
    'Track_pdgId', 'Track_dz', 'Track_trackFromTau', 'Track_trackFromB',
    'Track_trackFromKstar', 'Track_trackgenIdx', 'Track_trackgenFlav',
    'Track_isMatchedToMuon',
    'nKstarToPiK', 'KstarToPiK_trk1_idx', 'KstarToPiK_trk2_idx',
    'nKstarToPiKAll', 'KstarToPiKAll_trk1_idx', 'KstarToPiKAll_trk2_idx',
    'nGenPart', 'GenPart_pdgId', 'GenPart_status', 'GenPart_genPartIdxMother',
    'GenPart_pt', 'GenPart_vx', 'GenPart_vy', 'GenPart_vz',
    'GenPart_gentrackIdx', 'GenPart_gentrackMatchDr', 'GenPart_gentrackMatchDPt',
    'nGenFinalState', 'GenFinalState_pdgId', 'GenFinalState_mother_pdgId',
    'GenFinalState_mother_genPartIdx', 'GenFinalState_B_genPartIdx',
    'GenFinalState_Bmother_pdgId',
    'nABCNetTau', 'ABCNetTau_trk_idx', 'ABCNetTau_pf_idx', 'ABCNetTau_kstar_idx',
    'ABCNetTau_score', 'ABCNetTau_pt', 'ABCNetTau_eta', 'ABCNetTau_phi',
    'nTauTo3Pi', 'TauTo3Pi_idx1', 'TauTo3Pi_idx2', 'TauTo3Pi_idx3', 'TauTo3Pi_vprob',
    'nTauTo3PiWithMatch', 'TauTo3PiWithMatch_idx1', 'TauTo3PiWithMatch_idx2',
    'TauTo3PiWithMatch_idx3', 'TauTo3PiWithMatch_mcMatch',
    'nBToKstarTauTau', 'BToKstarTauTau_pi1_idx', 'BToKstarTauTau_pi2_idx',
    'BToKstarTauTau_pi3_idx', 'BToKstarTauTau_tau_idx', 'BToKstarTauTau_kstar_idx',
    'BToKstarTauTau_svprob', 'BToKstarTauTau_fit_mass',
    'PV_x', 'PV_y', 'PV_z', 'nPVtx', 'PVtx_vx', 'PVtx_vy', 'PVtx_vz', 'PVtx_score',
    'nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_charge', 'Muon_dz',
    'Muon_softId', 'Muon_muonFromTau', 'Muon_muonGenIdx',
]
KSTAR_CUTFLOW_PREFIX = 'KstarCutflow_'


# ---------------------------------------------------------------------------
# Loading and inventory
# ---------------------------------------------------------------------------

def load_events(path, entry_stop=None):
    """path: ROOT file with an Events tree. Returns an awkward record array
    of SURVEY_BRANCHES plus every KstarCutflow_* scalar, first entry_stop
    entries (all when None)."""
    tree = uproot.open(path)['Events']
    cutflow = sorted(name for name in tree.keys()
                     if name.startswith(KSTAR_CUTFLOW_PREFIX))
    return tree.arrays(SURVEY_BRANCHES + cutflow, entry_stop=entry_stop)


def _converter_module():
    if CONVERTER_DIR not in sys.path:
        sys.path.insert(0, CONVERTER_DIR)
    import convert_root_to_parquet
    return convert_root_to_parquet


def converter_clean_mask(path, entry_stop=None):
    """path: ROOT file with an Events tree. Returns (N,) bool: the
    converter's event-quality mask (corrupt eta/phi/pT/charge, NaN/Inf in
    the legacy branches) computed by the converter's own function."""
    converter = _converter_module()
    tree = uproot.open(path)['Events']
    _, clean = converter.read_and_filter_pion_tracks(
        tree, converter.build_track_branch_map(converter.DEFAULT_COLUMNS),
        entry_stop=entry_stop)
    return np.asarray(clean, dtype=bool)


def converter_branches():
    """Returns the set of ROOT branches the extended + builder-blocks
    conversion reads (scripts/condor/convert_to_parquet)."""
    converter = _converter_module()
    names = set(converter.build_track_branch_map(converter.EXTENDED_TRACK_COLUMNS))
    names.add('Track_pdgId')
    names.update(converter.MUON_BRANCH_MAP)
    names.update(converter.SV_BRANCH_MAP)
    names.update(converter.EXTENDED_EVENT_BRANCHES)
    names.update(converter.EVENT_BRANCHES)
    names.update(converter.KSTAR_BRANCHES)
    names.update(converter.TAU_BUILDER_BRANCH_MAP)
    names.update(converter.BEAM_SPOT_BRANCH_MAP)
    return names


def merge_kept_branches():
    """Returns the set of ROOT branches the merge step whitelists
    (scripts/condor/merge_root_batches)."""
    if MERGE_DIR not in sys.path:
        sys.path.insert(0, MERGE_DIR)
    import merge_batches
    return set(merge_batches.KEEP_BRANCHES)


def classify_branches(branch_names, reference_names, kept_names, converter_names):
    """All arguments: sets of branch names. Returns {branch: class} with
    classes 'converter' (read by the conversion), 'merge_kept' (whitelisted
    at merge only), 'reference_not_kept' (in the reference raw nano, dropped
    at merge) and 'new' (absent from the reference raw nano)."""
    classes = {}
    for name in branch_names:
        if name in converter_names:
            classes[name] = 'converter'
        elif name in kept_names:
            classes[name] = 'merge_kept'
        elif name in reference_names:
            classes[name] = 'reference_not_kept'
        else:
            classes[name] = 'new'
    return classes


def branch_family(name):
    """Returns the collection prefix of a branch; counters nX join X."""
    if '_' not in name and len(name) > 1 and name[0] == 'n' and name[1].isupper():
        return name[1:]
    return name.split('_', 1)[0]


def inventory(tree, reference_path=None):
    """tree: uproot Events tree. Returns {family: {'n', 'classes': {class:
    [branch]}, 'new_dtypes': {branch: typename}}}; no data is read."""
    branch_names = set(tree.keys())
    reference_names = set()
    if reference_path is not None:
        reference_names = set(uproot.open(reference_path)['Events'].keys())
    classes = classify_branches(branch_names, reference_names,
                                merge_kept_branches(), converter_branches())
    families = {}
    for name in sorted(branch_names):
        entry = families.setdefault(branch_family(name),
                                    {'n': 0, 'classes': {}, 'new_dtypes': {}})
        entry['n'] += 1
        entry['classes'].setdefault(classes[name], []).append(name)
        if classes[name] == 'new':
            entry['new_dtypes'][name] = tree[name].typename
    return families


# ---------------------------------------------------------------------------
# Gen-level walks
# ---------------------------------------------------------------------------

# walk_to_ancestor, top_b_ancestor and gen_tau_pions live in ntuple_truth.


# ---------------------------------------------------------------------------
# Selection and labels
# ---------------------------------------------------------------------------

# FUNNEL_BUCKETS and select_event live in ntuple_truth.


def track_walks_to_tau(track_gen_idx, gen_pdg, gen_mother):
    """track_gen_idx: (T,); gen_pdg, gen_mother: (N,). Returns (T,) bool:
    gen-matched to a charged pion with a tau ancestor."""
    walks = np.zeros(track_gen_idx.size, dtype=bool)
    for track, gen_index in enumerate(track_gen_idx):
        gen_index = int(gen_index)
        if gen_index < 0 or gen_index >= gen_pdg.size \
                or abs(int(gen_pdg[gen_index])) != PION_PDG:
            continue
        walks[track] = walk_to_ancestor(
            gen_pdg, gen_mother, int(gen_mother[gen_index]), {TAU_PDG}) >= 0
    return walks


def label_consistency(track_gen_idx, track_from_tau, gen_pdg, gen_mother):
    """track_gen_idx, track_from_tau: (T,); gen_pdg, gen_mother: (N,).
    Returns counts of GT tracks whose gen match walks to a tau, GT tracks
    without such a walk, and non-GT tracks with one."""
    walks = track_walks_to_tau(track_gen_idx, gen_pdg, gen_mother)
    gt = track_from_tau == 1
    return {'gt_walks': int(np.sum(gt & walks)),
            'gt_no_walk': int(np.sum(gt & ~walks)),
            'walk_not_gt': int(np.sum(~gt & walks))}


# ---------------------------------------------------------------------------
# K* candidates and the ABCNet block
# ---------------------------------------------------------------------------

# kstar_rows and abcnet_lists live in ntuple_truth.


def abcnet_index_check(events):
    """events: awkward record array of SURVEY_BRANCHES. Returns the largest
    |ABCNetTau_pt/eta/phi - Track_*[trk_idx]| over rows with a track index,
    whether every kstar_idx is below nKstarToPiKAll, the row count and the
    rows without a track index."""
    trk_idx = events['ABCNetTau_trk_idx']
    valid = trk_idx >= 0
    result = {}
    for name in ('pt', 'eta', 'phi'):
        table = events[f'Track_{name}'][trk_idx[valid]]
        rows = events[f'ABCNetTau_{name}'][valid]
        difference = ak.flatten(abs(table - rows))
        result[f'max_abs_delta_{name}'] = \
            float(ak.max(difference)) if len(difference) else 0.0
    kstar_idx = events['ABCNetTau_kstar_idx']
    in_range = (kstar_idx >= 0) & (kstar_idx < events['nKstarToPiKAll'])
    result['all_kstar_idx_in_range'] = bool(ak.all(ak.all(in_range, axis=1)))
    result['n_rows'] = int(ak.sum(ak.num(trk_idx, axis=1)))
    result['n_missing_track'] = int(ak.sum(~valid))
    return result


def list_ranks(trk_idx, score, kstar_idx, list_row, gt_triplet):
    """trk_idx, score, kstar_idx: (A,) ABCNetTau rows; list_row: the
    kstar_idx of one list; gt_triplet: (3,). Returns the 0-based
    score-descending rank of each GT track inside that list (-1 when
    absent), the list size, the rows above the working point, the GT scores
    (NaN when absent) and whether the stored order is score-descending."""
    rows = np.flatnonzero(kstar_idx == list_row)
    order = rows[np.argsort(-score[rows], kind='stable')]
    ranks = np.full(3, -1, dtype=np.int64)
    gt_scores = np.full(3, np.nan)
    for position, row in enumerate(order):
        for member in range(3):
            if trk_idx[row] == gt_triplet[member]:
                ranks[member] = position
                gt_scores[member] = score[row]
    return {'ranks': ranks,
            'list_size': int(rows.size),
            'n_above_wp': int(np.sum(score[rows] > ABCNET_WORKING_POINT)),
            'gt_scores': gt_scores,
            'score_sorted': bool(np.all(np.diff(score[rows]) <= 0))}


def hit_at_k(ranks, top_k):
    """ranks: (3,). Returns True when every rank lies in [0, top_k)."""
    return bool(np.all((ranks >= 0) & (ranks < top_k)))


# ---------------------------------------------------------------------------
# Builder and analysis-level candidates
# ---------------------------------------------------------------------------

# sorted_triplet_hit lives in ntuple_truth.


def builder_candidates(idx1, idx2, idx3, gt_triplet):
    """idx1, idx2, idx3: (R,) TauTo3Pi rows; gt_triplet: (3,). Returns the
    row count, the exact-hit rows, the number of distinct tracks the rows
    draw from (the builder's pool) and whether rows are stored sorted."""
    hits = sorted_triplet_hit(idx1, idx2, idx3, gt_triplet)
    pool = np.unique(np.concatenate([idx1, idx2, idx3]))
    return {'n_rows': int(idx1.size),
            'hit_rows': [int(row) for row in np.flatnonzero(hits)],
            'pool_size': int(pool.size),
            'stored_sorted': bool(np.all((idx1 < idx2) & (idx2 < idx3)))}


def withmatch_check(events):
    """events: awkward record array. Returns whether TauTo3PiWithMatch has
    the same row counts and track indices as TauTo3Pi in every event, the
    distinct mcMatch values and the WithMatch row count."""
    counts_equal = bool(ak.all(events['nTauTo3Pi'] == events['nTauTo3PiWithMatch']))
    indices_equal = counts_equal and all(
        bool(ak.all(ak.flatten(events[f'TauTo3Pi_idx{member}'])
                    == ak.flatten(events[f'TauTo3PiWithMatch_idx{member}'])))
        for member in (1, 2, 3))
    mcmatch = ak.to_numpy(ak.flatten(events['TauTo3PiWithMatch_mcMatch']))
    return {'all_counts_equal': counts_equal,
            'all_indices_equal': indices_equal,
            'mcmatch_values': [int(value) for value in np.unique(mcmatch)],
            'n_rows': int(mcmatch.size)}


def b_candidates(pi1, pi2, pi3, tau_idx, kstar_idx, builder_idx1, builder_idx2,
                 builder_idx3, n_kstar_all, gt_triplet):
    """pi1, pi2, pi3, tau_idx, kstar_idx: (R,) BToKstarTauTau rows;
    builder_idx1..3: (C,) TauTo3Pi rows; n_kstar_all: int; gt_triplet: (3,).
    Returns the row count, the distinct sorted triplets, the exact-hit rows,
    whether every row's pions equal the TauTo3Pi row it points to, and
    whether every kstar_idx addresses KstarToPiKAll."""
    hits = sorted_triplet_hit(pi1, pi2, pi3, gt_triplet)
    triplets = np.sort(np.stack([pi1, pi2, pi3], axis=1).astype(np.int64), axis=1)
    unique_triplets = np.unique(triplets, axis=0).shape[0] if triplets.size else 0
    tau_valid = (tau_idx >= 0) & (tau_idx < builder_idx1.size)
    consistent = bool(np.all(tau_valid)) and bool(np.all(
        (pi1 == builder_idx1[tau_idx]) & (pi2 == builder_idx2[tau_idx])
        & (pi3 == builder_idx3[tau_idx])))
    return {'n_rows': int(pi1.size),
            'n_unique_triplets': int(unique_triplets),
            'n_hit_rows': int(np.sum(hits)),
            'index_consistent': consistent,
            'kstar_in_range': bool(np.all((kstar_idx >= 0) & (kstar_idx < n_kstar_all)))}


# ---------------------------------------------------------------------------
# Truth: vertices, impostor origins, tau-muon
# ---------------------------------------------------------------------------

# gen_truth lives in ntuple_truth.


def pv_truth(pv_xyz, pvtx_xyz, pvtx_score, true_xyz):
    """pv_xyz: (3,) stored PV; pvtx_xyz: (V, 3); pvtx_score: (V,);
    true_xyz: (3,). Returns the stored PV's 3-D distance and z gap to the
    true PV, the nearest PVtx index (-1 when empty), its distance and its
    rank by descending score (0 = highest)."""
    pv_xyz = np.asarray(pv_xyz, dtype=np.float64)
    true_xyz = np.asarray(true_xyz, dtype=np.float64)
    pvtx_xyz = np.asarray(pvtx_xyz, dtype=np.float64).reshape(-1, 3)
    result = {'stored_distance_3d': float(np.linalg.norm(pv_xyz - true_xyz)),
              'stored_dz': float(pv_xyz[2] - true_xyz[2]),
              'nearest_index': -1, 'nearest_distance': None,
              'nearest_score_rank': None}
    if pvtx_xyz.shape[0] == 0:
        return result
    distances = np.linalg.norm(pvtx_xyz - true_xyz, axis=1)
    nearest = int(np.argmin(distances))
    score_order = np.argsort(-np.asarray(pvtx_score, dtype=np.float64), kind='stable')
    result.update(nearest_index=nearest,
                  nearest_distance=float(distances[nearest]),
                  nearest_score_rank=int(np.flatnonzero(score_order == nearest)[0]))
    return result


TRACK_FIELDS = {
    'track_pt': 'Track_pt', 'track_eta': 'Track_eta', 'track_phi': 'Track_phi',
    'track_charge': 'Track_charge', 'track_dz': 'Track_dz', 'track_pdg': 'Track_pdgId',
    'track_from_tau': 'Track_trackFromTau', 'track_from_b': 'Track_trackFromB',
    'track_from_kstar': 'Track_trackFromKstar', 'track_gen_idx': 'Track_trackgenIdx',
    'track_matched_to_muon': 'Track_isMatchedToMuon',
    'muon_pt': 'Muon_pt', 'muon_eta': 'Muon_eta', 'muon_phi': 'Muon_phi',
    'muon_charge': 'Muon_charge', 'muon_dz': 'Muon_dz', 'muon_soft': 'Muon_softId',
}


def event_tracks(events, entry, gt_indices=None):
    """events: awkward record array; entry: int; gt_indices: (3,) selected
    GT tracks or None. Returns a dict of numpy arrays (float64 kinematics,
    int64 flags) in the layout of utils.tau_muon_proxy; 'track_label' marks
    gt_indices when given, else every flagged GT pion."""
    event = {}
    for key, branch in TRACK_FIELDS.items():
        values = ak.to_numpy(events[branch][entry])
        event[key] = values.astype(np.float64) if values.dtype.kind == 'f' \
            else values.astype(np.int64)
    if gt_indices is None:
        event['track_label'] = ((event['track_from_tau'] == 1)
                                & (np.abs(event['track_pdg']) == PION_PDG)).astype(np.int64)
    else:
        event['track_label'] = np.zeros(event['track_pt'].size, dtype=np.int64)
        event['track_label'][np.asarray(gt_indices, dtype=np.int64)] = 1
    return event


IMPOSTOR_CATEGORIES = ('kstar_daughter', 'matched_to_muon', 'b_chain',
                       'other_gen', 'unmatched')


def impostor_taxonomy(event, gt_triplet, muon_gen_idx, window):
    """event: dict from event_tracks; gt_triplet: (3,); muon_gen_idx: (M,)
    GenPart indices of the reco muons; window: (max delta R to the 3-pion
    system, max |dz - median GT dz|). Returns the non-GT charged pions above
    PT_CUTOFF inside the window, split by origin with priority
    kstar_daughter > matched_to_muon > b_chain > other_gen > unmatched."""
    delta_r_max, dz_max = window
    gt_triplet = np.asarray(gt_triplet, dtype=np.int64)
    system = proxy.system_from_members(event, gt_triplet[None, :])
    delta_r = proxy.delta_r(event['track_eta'], event['track_phi'],
                            system['eta'][0], system['phi'][0])
    in_window = ((np.abs(event['track_pdg']) == PION_PDG)
                 & (event['track_pt'] >= PT_CUTOFF)
                 & (delta_r < delta_r_max)
                 & (np.abs(event['track_dz'] - system['dz'][0]) < dz_max))
    in_window[gt_triplet] = False
    gen_idx = event['track_gen_idx']
    muon_gen_idx = np.asarray(muon_gen_idx, dtype=np.int64)
    muon_matched = ((event['track_matched_to_muon'] == 1)
                    | np.isin(gen_idx, muon_gen_idx[muon_gen_idx >= 0]))
    categories = {name: 0 for name in IMPOSTOR_CATEGORIES}
    for track in np.flatnonzero(in_window):
        if event['track_from_kstar'][track] == 1:
            name = 'kstar_daughter'
        elif muon_matched[track]:
            name = 'matched_to_muon'
        elif event['track_from_b'][track] == 1:
            name = 'b_chain'
        elif gen_idx[track] >= 0:
            name = 'other_gen'
        else:
            name = 'unmatched'
        categories[name] += 1
    return {'n_window': int(in_window.sum()), 'categories': categories}


def proxy_muon_truth(event, muon_from_tau):
    """event: dict from event_tracks; muon_from_tau: (M,). Returns the proxy
    filter's pick (-1 when none), whether it is a truth tau-muon, the truth
    and other muon counts, and for the first truth muon its per-cut pass
    flags and relations to the GT system (None without one)."""
    muon_from_tau = np.asarray(muon_from_tau)
    truth = np.flatnonzero(muon_from_tau == 1)
    result = {'picked': -1, 'picked_is_tau': False,
              'n_tau_muons': int(truth.size),
              'n_other_muons': int(muon_from_tau.size - truth.size),
              'tau_cut_flags': None, 'tau_relations': None}
    if int(event['track_label'].sum()) != 3 or muon_from_tau.size == 0:
        return result
    relations = proxy.gt_relations(event)
    picked = int(proxy.select_proxy_muon(relations, event['muon_soft'], 'full'))
    result['picked'] = picked
    result['picked_is_tau'] = bool(picked >= 0 and muon_from_tau[picked] == 1)
    if truth.size:
        muon = int(truth[0])
        values = {name: float(relations[name][0, muon])
                  for name in ('delta_r', 'dz_gap', 'mass', 'charge_product')}
        result['tau_relations'] = values
        result['tau_cut_flags'] = {
            'soft': bool(event['muon_soft'][muon] == 1),
            'opposite_charge': values['charge_product'] < 0,
            'dz': values['dz_gap'] < proxy.DZ_MAX,
            'delta_r': values['delta_r'] < proxy.DR_MAX,
            'mass': values['mass'] < proxy.B_MASS,
        }
    return result


def kstar_cutflow(events, entry):
    """events: awkward record array; entry: int. Returns {counter: value}
    for the KstarCutflow_* scalars (prefix stripped)."""
    return {name[len(KSTAR_CUTFLOW_PREFIX):]: int(events[name][entry])
            for name in events.fields if name.startswith(KSTAR_CUTFLOW_PREFIX)}


# ---------------------------------------------------------------------------
# Per-event record
# ---------------------------------------------------------------------------

LIST_CAP = 50
BUILDER_CANDIDATE_THRESHOLD = 50
PV_PROXY_CM = 0.2


def _column(events, branch, entry):
    return ak.to_numpy(events[branch][entry])


def _first_or_none(values):
    return int(values[0]) if values.size else None


def event_record(events, entry, converter_clean=True):
    """events: awkward record array; entry: int; converter_clean: the
    converter's quality mask for this entry. Returns one flat dict with
    every per-event measurement; blocks that need tracks (labels, kstar) or
    a selected event (abcnet, builder, b, impostors, muon) are None
    otherwise; an event the converter would drop keeps only its truth and
    PV blocks under the bucket 'quality_dropped'. 'abcnet_pools' holds numpy
    (score, is_gt) pairs for the pooled AUCs."""
    track_pt = _column(events, 'Track_pt', entry)
    track_pdg = _column(events, 'Track_pdgId', entry)
    track_from_tau = _column(events, 'Track_trackFromTau', entry)
    bucket, gt_indices = select_event(track_pt, track_pdg, track_from_tau)
    selected = bucket == 'selected'

    gen_pdg = _column(events, 'GenPart_pdgId', entry)
    gen_mother = _column(events, 'GenPart_genPartIdxMother', entry)
    gen_status = _column(events, 'GenPart_status', entry)
    gen_pt = _column(events, 'GenPart_pt', entry)
    gen_vertex = np.stack([_column(events, f'GenPart_v{axis}', entry)
                           for axis in 'xyz'], axis=1)
    truth = gen_truth(gen_pdg, gen_mother, gen_status, gen_pt, gen_vertex)
    gen_track_idx = _column(events, 'GenPart_gentrackIdx', entry)
    pion_reco = gen_track_idx[truth['pion_indices']] >= 0
    b_pointers = np.unique(_column(events, 'GenFinalState_B_genPartIdx', entry).astype(np.int64))
    b_pointer_mismatch = bool(truth['b_index'] >= 0 and b_pointers.size
                              and truth['b_index'] not in b_pointers)

    record = {
        'entry': int(entry),
        'run': int(events['run'][entry]),
        'event': int(events['event'][entry]),
        'bucket': bucket,
        'selected': selected,
        'gt_indices': None if gt_indices is None else [int(index) for index in gt_indices],
        'n_tracks': int(track_pt.size),
        'n_tracks_above_cutoff': int(np.sum((np.abs(track_pdg) == PION_PDG)
                                            & (track_pt >= PT_CUTOFF))),
        'n_kstar_selected': int(events['nKstarToPiK'][entry]),
        'n_kstar_all': int(events['nKstarToPiKAll'][entry]),
        'n_abcnet': int(events['nABCNetTau'][entry]),
        'n_tau3pi': int(events['nTauTo3Pi'][entry]),
        'n_b': int(events['nBToKstarTauTau'][entry]),
        'n_muons': int(events['nMuon'][entry]),
        'n_pvtx': int(events['nPVtx'][entry]),
        'cutflow': kstar_cutflow(events, entry),
        'truth': {
            'tau_index': truth['tau_index'],
            'b_index': truth['b_index'],
            'n_gen_pions': int(truth['pion_indices'].size),
            'n_gen_pions_reco': int(pion_reco.sum()),
            'gen_pion_pt': [float(value) for value in gen_pt[truth['pion_indices']]],
            'gen_pion_reco': [bool(flag) for flag in pion_reco],
            'tau_pt': truth['tau_pt'],
            'decay_length_3d': truth['decay_length_3d'],
            'decay_length_xy': truth['decay_length_xy'],
            'b_pointer_mismatch': b_pointer_mismatch,
        },
        'labels': None, 'kstar': None, 'abcnet': None, 'builder': None,
        'b': None, 'pv': None, 'impostors': None, 'muon': None,
    }

    if truth['true_pv'] is not None:
        pv_xyz = np.array([float(events[f'PV_{axis}'][entry]) for axis in 'xyz'])
        pvtx_xyz = np.stack([_column(events, f'PVtx_v{axis}', entry)
                             for axis in 'xyz'], axis=1)
        record['pv'] = pv_truth(pv_xyz, pvtx_xyz, _column(events, 'PVtx_score', entry),
                                truth['true_pv'])
        record['pv']['gt_median_dz'] = None
        if gt_indices is not None and converter_clean:
            record['pv']['gt_median_dz'] = float(
                np.median(_column(events, 'Track_dz', entry)[gt_indices]))

    if not converter_clean:
        record.update(bucket='quality_dropped', selected=False, gt_indices=None)
        return record
    if track_pt.size == 0:
        return record

    track_gen_idx = _column(events, 'Track_trackgenIdx', entry)
    record['labels'] = label_consistency(track_gen_idx, track_from_tau, gen_pdg, gen_mother)
    if gt_indices is not None:
        gt_gen = track_gen_idx[gt_indices]
        valid = gt_gen >= 0
        track_dz = _column(events, 'Track_dz', entry)
        record['labels'].update(
            gt_gen_flav=[int(flavour) for flavour in
                         _column(events, 'Track_trackgenFlav', entry)[gt_indices]],
            gt_match_dr=[float(value) for value in
                         _column(events, 'GenPart_gentrackMatchDr', entry)[gt_gen[valid]]],
            gt_match_dpt=[float(abs(value)) for value in
                          _column(events, 'GenPart_gentrackMatchDPt', entry)[gt_gen[valid]]],
            gt_dz_spread=float(np.max(np.abs(track_dz[gt_indices]
                                             - np.median(track_dz[gt_indices])))),
            bijection_failures=int(np.sum(gen_track_idx[gt_gen[valid]] != gt_indices[valid])))

    kstar = kstar_rows(
        _column(events, 'KstarToPiKAll_trk1_idx', entry),
        _column(events, 'KstarToPiKAll_trk2_idx', entry),
        _column(events, 'Track_trackFromKstar', entry),
        _first_or_none(_column(events, 'KstarToPiK_trk1_idx', entry)),
        _first_or_none(_column(events, 'KstarToPiK_trk2_idx', entry)))
    record['kstar'] = {'n_true_rows': len(kstar['true_rows']),
                       'has_selected_row': kstar['selected_row'] is not None,
                       'selected_is_true': kstar['selected_is_true']}
    if not selected:
        return record

    trk_idx = _column(events, 'ABCNetTau_trk_idx', entry)
    score = _column(events, 'ABCNetTau_score', entry).astype(np.float64)
    kstar_idx = _column(events, 'ABCNetTau_kstar_idx', entry)
    lists = abcnet_lists(trk_idx, kstar_idx)
    is_gt_row = np.isin(trk_idx, gt_indices)
    valid_rows = trk_idx >= 0
    unique_tracks, inverse = np.unique(trk_idx[valid_rows], return_inverse=True)
    track_max = np.full(unique_tracks.size, -np.inf)
    np.maximum.at(track_max, inverse, score[valid_rows])
    pools = {'all_rows': (score[valid_rows], is_gt_row[valid_rows]),
             'track_max': (track_max, np.isin(unique_tracks, gt_indices))}
    list_blocks = {}
    list_rows = {'true': kstar['true_rows'][0] if kstar['true_rows'] else None,
                 'selected': kstar['selected_row']}
    for list_type, row in list_rows.items():
        block = {'exists': row is not None}
        if row is not None:
            ranks = list_ranks(trk_idx, score, kstar_idx, row, gt_indices)
            block.update(
                ranks=[int(rank) for rank in ranks['ranks']],
                list_size=ranks['list_size'],
                n_above_wp=ranks['n_above_wp'],
                score_sorted=ranks['score_sorted'],
                all3_present=bool(np.all(ranks['ranks'] >= 0)),
                gt_scores=[None if np.isnan(value) else float(value)
                           for value in ranks['gt_scores']],
                hits={str(top_k): hit_at_k(ranks['ranks'], top_k) for top_k in RECALL_KS})
            rows = kstar_idx == row
            pools[f'{list_type}_list'] = (score[rows], is_gt_row[rows])
        list_blocks[list_type] = block
    gt_in_any = [bool(np.any(trk_idx == index)) for index in gt_indices]
    record['abcnet'] = {
        'lists': list_blocks,
        'n_rows': lists['n_rows'],
        'n_lists': lists['n_lists'],
        'rows_per_list_max': int(lists['rows_per_list'].max()) if lists['n_lists'] else 0,
        'n_lists_at_cap': int(np.sum(lists['rows_per_list'] == LIST_CAP)),
        'n_missing_track': lists['n_missing_track'],
        'n_unique_tracks': int(unique_tracks.size),
        'gt_in_any_list': gt_in_any,
        'all3_any_list': all(gt_in_any),
    }
    record['abcnet_pools'] = pools

    builder_idx = tuple(_column(events, f'TauTo3Pi_idx{member}', entry)
                        for member in (1, 2, 3))
    builder = builder_candidates(*builder_idx, gt_indices)
    record['builder'] = {'n_rows': builder['n_rows'],
                         'hit': bool(builder['hit_rows']),
                         'pool_size': builder['pool_size'],
                         'stored_sorted': builder['stored_sorted']}

    b_columns = tuple(_column(events, name, entry) for name in (
        'BToKstarTauTau_pi1_idx', 'BToKstarTauTau_pi2_idx', 'BToKstarTauTau_pi3_idx',
        'BToKstarTauTau_tau_idx', 'BToKstarTauTau_kstar_idx'))
    record['b'] = b_candidates(*b_columns, *builder_idx, record['n_kstar_all'], gt_indices)
    record['b']['hit'] = record['b']['n_hit_rows'] > 0

    event = event_tracks(events, entry, gt_indices)
    muon_gen_idx = _column(events, 'Muon_muonGenIdx', entry)
    record['impostors'] = {name: impostor_taxonomy(event, gt_indices, muon_gen_idx, window)
                           for name, window in IMPOSTOR_WINDOWS.items()}
    record['muon'] = proxy_muon_truth(event, _column(events, 'Muon_muonFromTau', entry))
    return record


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def wilson_interval(successes, total, z_score=1.96):
    """successes, total: ints. Returns the (low, high) Wilson score interval
    of the rate; (0.0, 0.0) when total is 0."""
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z_score * z_score / total
    centre = (proportion + z_score * z_score / (2.0 * total)) / denominator
    half_width = z_score * np.sqrt(proportion * (1.0 - proportion) / total
                                   + z_score * z_score / (4.0 * total * total)) / denominator
    return (max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width)))


def rate_entry(successes, total):
    """Returns {'successes', 'total', 'rate' (None when total is 0), 'ci'}."""
    successes, total = int(successes), int(total)
    return {'successes': successes, 'total': total,
            'rate': successes / total if total else None,
            'ci': list(wilson_interval(successes, total))}


def _rate_of(items, predicate):
    return rate_entry(sum(bool(predicate(item)) for item in items), len(items))


def _tercile_rates(values, flags):
    """values, flags: (n,). Returns {tercile: rate_entry} by ascending value."""
    values = np.asarray(values, dtype=np.float64)
    flags = np.asarray(flags, dtype=bool)
    if values.size < 3:
        return {}
    edges = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    bins = np.digitize(values, edges)
    return {f'tercile_{tercile}_max_{edges[tercile] if tercile < 2 else values.max():.3g}':
            rate_entry(flags[bins == tercile].sum(), int(np.sum(bins == tercile)))
            for tercile in range(3)}


def _pool_auc(records, pool_name):
    scores, labels = [], []
    for record in records:
        pools = record.get('abcnet_pools') or {}
        if pool_name in pools:
            scores.append(pools[pool_name][0])
            labels.append(pools[pool_name][1])
    if not scores:
        return None, np.array([]), np.array([], dtype=bool)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    auc = float(roc_auc(scores, labels)) if labels.any() and not labels.all() else None
    return auc, scores, labels


def _summarize_abcnet(selected):
    with_block = [record for record in selected if record['abcnet']]
    lists = {list_type: [record['abcnet']['lists'][list_type] for record in with_block]
             for list_type in ('true', 'selected')}
    total_lists = sum(record['abcnet']['n_lists'] for record in with_block)
    total_rows = sum(record['abcnet']['n_rows'] for record in with_block)
    summary = {
        'n_events': len(with_block),
        'rows_per_event': distribution_summary([record['abcnet']['n_rows'] for record in with_block]),
        'lists_per_event': distribution_summary([record['abcnet']['n_lists'] for record in with_block]),
        'rows_per_list_max': distribution_summary(
            [record['abcnet']['rows_per_list_max'] for record in with_block]),
        'lists_at_cap': rate_entry(sum(record['abcnet']['n_lists_at_cap'] for record in with_block),
                                   total_lists),
        'rows_without_track': rate_entry(
            sum(record['abcnet']['n_missing_track'] for record in with_block), total_rows),
        'unique_tracks_per_event': distribution_summary(
            [record['abcnet']['n_unique_tracks'] for record in with_block]),
        'coverage': {
            'per_track_any_list': rate_entry(
                sum(sum(record['abcnet']['gt_in_any_list']) for record in with_block),
                3 * len(with_block)),
            'all3_any_list': _rate_of(with_block, lambda record: record['abcnet']['all3_any_list']),
        },
        'recall': {}, 'pool_at_wp': {}, 'score_sorted': {}, 'auc': {},
    }
    for list_type, blocks in lists.items():
        existing = [block for block in blocks if block['exists']]
        summary['coverage'][f'{list_type}_list_exists'] = rate_entry(len(existing), len(blocks))
        summary['coverage'][f'all3_in_{list_type}_list_conditional'] = _rate_of(
            existing, lambda block: block['all3_present'])
        summary['coverage'][f'all3_in_{list_type}_list_unconditional'] = rate_entry(
            sum(block['all3_present'] for block in existing), len(blocks))
        summary['recall'][list_type] = {
            'conditional': {str(top_k): _rate_of(
                existing, lambda block, top_k=top_k: block['hits'][str(top_k)])
                for top_k in RECALL_KS},
            'unconditional': {str(top_k): rate_entry(
                sum(block['hits'][str(top_k)] for block in existing), len(blocks))
                for top_k in RECALL_KS},
        }
        summary['pool_at_wp'][list_type] = distribution_summary(
            [block['n_above_wp'] for block in existing])
        summary['score_sorted'][list_type] = _rate_of(existing, lambda block: block['score_sorted'])
    for pool_name in ('all_rows', 'track_max', 'true_list', 'selected_list'):
        auc, scores, labels = _pool_auc(with_block, pool_name)
        summary['auc'][pool_name] = auc
        if pool_name == 'all_rows' and scores.size:
            summary['gt_rows_above_wp'] = rate_entry(
                np.sum(scores[labels] > ABCNET_WORKING_POINT), int(labels.sum()))
            summary['other_rows_above_wp'] = rate_entry(
                np.sum(scores[~labels] > ABCNET_WORKING_POINT), int((~labels).sum()))
            summary['gt_row_scores'] = distribution_summary(scores[labels])
            summary['other_row_scores'] = distribution_summary(scores[~labels])
    return summary


def _summarize_pv(population):
    far = {str(threshold): _rate_of(
        population,
        lambda record, threshold=threshold: record['pv']['stored_distance_3d'] > threshold)
        for threshold in PV_FAR_CM}
    far_z = {str(threshold): _rate_of(
        population,
        lambda record, threshold=threshold: abs(record['pv']['stored_dz']) > threshold)
        for threshold in PV_FAR_CM}
    with_pvtx = [record for record in population if record['pv']['nearest_index'] >= 0]
    stored_near = [record['pv']['stored_distance_3d'] <= PV_PROXY_CM for record in with_pvtx]
    any_near = [record['pv']['nearest_distance'] <= PV_PROXY_CM for record in with_pvtx]
    near_pairs = list(zip(stored_near, any_near))
    rank_counts = Counter(min(record['pv']['nearest_score_rank'], 3) for record in with_pvtx)
    return {
        'n': len(population),
        'misassigned_3d': far,
        'misassigned_z': far_z,
        'stored_distance_3d': distribution_summary(
            [record['pv']['stored_distance_3d'] for record in population]),
        'stored_dz_abs': distribution_summary(
            [abs(record['pv']['stored_dz']) for record in population]),
        'nearest_distance': distribution_summary(
            [record['pv']['nearest_distance'] for record in with_pvtx]),
        'nearest_within_0.2': _rate_of(with_pvtx,
                                       lambda record: record['pv']['nearest_distance'] <= PV_PROXY_CM),
        'nearest_is_index_0': _rate_of(with_pvtx, lambda record: record['pv']['nearest_index'] == 0),
        'nearest_score_rank_counts': {str(rank) if rank < 3 else '3+': int(count)
                                      for rank, count in sorted(rank_counts.items())},
        'stored_near_x_any_near': {
            'stored_near_any_near': int(sum(
                stored_flag and any_flag for stored_flag, any_flag in near_pairs)),
            'stored_far_any_near': int(sum(
                (not stored_flag) and any_flag for stored_flag, any_flag in near_pairs)),
            'stored_far_none_near': int(sum(
                (not stored_flag) and (not any_flag) for stored_flag, any_flag in near_pairs)),
        },
    }


def summarize(records):
    """records: list of event_record dicts. Returns the nested summary
    (funnel, multiplicities, cutflow, gen, labels, kstar, abcnet, builder,
    b, pv, impostors, muon) with counts, rates with Wilson intervals and
    distribution quantiles; JSON-serialisable."""
    selected = [record for record in records if record['selected']]
    track_events_raw = [record for record in records if record['n_tracks'] > 0]
    track_events = [record for record in track_events_raw
                    if record['bucket'] != 'quality_dropped']
    with_truth = [record for record in records if record['pv'] is not None]
    summary = {'n_events': len(records)}

    buckets = {name: sum(record['bucket'] == name for record in records)
               for name in FUNNEL_BUCKETS}
    summary['funnel'] = {
        'buckets': buckets,
        'n_track_events_raw': len(track_events_raw),
        'n_track_events': len(track_events),
        'n_tracks_no_kstar': sum(record['n_kstar_selected'] == 0 for record in track_events),
        'n_selected': len(selected),
        'n_selected_with_kstar': sum(record['n_kstar_selected'] > 0 for record in selected),
        'selected_of_all': rate_entry(len(selected), len(records)),
        'selected_of_track_events': rate_entry(len(selected), len(track_events)),
    }
    summary['multiplicities'] = {
        name: distribution_summary([record[name] for record in selected])
        for name in ('n_tracks', 'n_tracks_above_cutoff', 'n_kstar_all', 'n_abcnet',
                     'n_tau3pi', 'n_b', 'n_muons', 'n_pvtx')}
    counters = list(records[0]['cutflow']) if records else []
    summary['cutflow'] = {
        name: {'positive': _rate_of(
            records, lambda record, counter=name: record['cutflow'][counter] > 0),
            'mean': float(np.mean([record['cutflow'][name] for record in records]))}
        for name in counters}

    pion_pt = np.array([value for record in records for value in record['truth']['gen_pion_pt']])
    pion_reco = np.array([flag for record in records for flag in record['truth']['gen_pion_reco']],
                         dtype=bool)
    pion_reco_track = np.array([flag for record in track_events
                                for flag in record['truth']['gen_pion_reco']], dtype=bool)
    reco_by_pt = {}
    for low, high in zip(GEN_PT_BINS[:-1], GEN_PT_BINS[1:]):
        mask = (pion_pt >= low) & (pion_pt < high)
        reco_by_pt[f'{low:g}-{high:g}'] = rate_entry(pion_reco[mask].sum(), int(mask.sum()))
    summary['gen'] = {
        'n_gen_pions_counts': {str(count): int(number) for count, number in sorted(Counter(
            record['truth']['n_gen_pions'] for record in records).items())},
        'pion_reco_all_events': rate_entry(pion_reco.sum(), pion_reco.size),
        'pion_reco_track_events': rate_entry(pion_reco_track.sum(), pion_reco_track.size),
        'pion_reco_by_pt': reco_by_pt,
        'all3_reco_all_events': _rate_of(
            records, lambda record: record['truth']['n_gen_pions'] == 3
            and record['truth']['n_gen_pions_reco'] == 3),
        'all3_reco_track_events': _rate_of(
            track_events, lambda record: record['truth']['n_gen_pions'] == 3
            and record['truth']['n_gen_pions_reco'] == 3),
        'tau_pt_all': distribution_summary(
            [record['truth']['tau_pt'] for record in records if record['truth']['tau_pt'] is not None]),
        'tau_pt_selected': distribution_summary(
            [record['truth']['tau_pt'] for record in selected if record['truth']['tau_pt'] is not None]),
        'decay_length_3d_selected': distribution_summary(
            [record['truth']['decay_length_3d'] for record in selected
             if record['truth']['decay_length_3d'] is not None]),
        'decay_length_xy_selected': distribution_summary(
            [record['truth']['decay_length_xy'] for record in selected
             if record['truth']['decay_length_xy'] is not None]),
        'b_pointer_mismatch': _rate_of(with_truth, lambda record: record['truth']['b_pointer_mismatch']),
    }

    labelled = [record['labels'] for record in track_events if record['labels']]
    gt_labelled = [labels for labels in labelled if 'gt_dz_spread' in labels]
    match_dr = [value for labels in gt_labelled for value in labels['gt_match_dr']]
    match_dpt = [value for labels in gt_labelled for value in labels['gt_match_dpt']]
    summary['labels'] = {
        'gt_walks': sum(labels['gt_walks'] for labels in labelled),
        'gt_no_walk': sum(labels['gt_no_walk'] for labels in labelled),
        'walk_not_gt': sum(labels['walk_not_gt'] for labels in labelled),
        'gt_gen_flav_counts': {str(flavour): int(count) for flavour, count in sorted(Counter(
            flavour for labels in gt_labelled for flavour in labels['gt_gen_flav']).items())},
        'gt_match_dr': distribution_summary(match_dr),
        'gt_match_dr_above_0.05': rate_entry(sum(value > 0.05 for value in match_dr), len(match_dr)),
        'gt_match_dpt': distribution_summary(match_dpt),
        'gt_match_dpt_above_0.2': rate_entry(sum(value > 0.2 for value in match_dpt), len(match_dpt)),
        'gt_dz_spread': distribution_summary([labels['gt_dz_spread'] for labels in gt_labelled]),
        'gt_dz_spread_above_0.2': _rate_of(gt_labelled, lambda labels: labels['gt_dz_spread'] > 0.2),
        'gt_dz_spread_above_1.0': _rate_of(gt_labelled, lambda labels: labels['gt_dz_spread'] > 1.0),
        'bijection_failures': sum(labels['bijection_failures'] for labels in gt_labelled),
    }

    with_kstar = [record['kstar'] for record in track_events if record['kstar']]
    with_selected_row = [kstar for kstar in with_kstar if kstar['has_selected_row']]
    summary['kstar'] = {
        'selected_is_true': _rate_of(with_selected_row, lambda kstar: kstar['selected_is_true']),
        'selected_row_found': rate_entry(len(with_selected_row), len(with_kstar)),
        'true_row_exists': _rate_of(with_kstar, lambda kstar: kstar['n_true_rows'] > 0),
        'true_row_multiplicity': {str(count) if count < 2 else '2+': int(number)
                                  for count, number in sorted(Counter(
                                      min(kstar['n_true_rows'], 2) for kstar in with_kstar).items())},
    }

    summary['abcnet'] = _summarize_abcnet(selected)

    builder_hits = [record['builder']['hit'] for record in selected]
    summary['builder'] = {
        'hit': rate_entry(sum(builder_hits), len(selected)),
        'candidates': distribution_summary([record['builder']['n_rows'] for record in selected]),
        'zero_candidates': _rate_of(selected, lambda record: record['builder']['n_rows'] == 0),
        'over_50_candidates': _rate_of(
            selected, lambda record: record['builder']['n_rows'] > BUILDER_CANDIDATE_THRESHOLD),
        'pool_size': distribution_summary([record['builder']['pool_size'] for record in selected]),
        'stored_sorted': _rate_of(selected, lambda record: record['builder']['stored_sorted']),
        'hit_by_tau_pt_tercile': _tercile_rates(
            [record['truth']['tau_pt'] or 0.0 for record in selected], builder_hits),
        'hit_by_multiplicity_tercile': _tercile_rates(
            [record['n_tracks_above_cutoff'] for record in selected], builder_hits),
    }

    builder_hit_events = [record for record in selected if record['builder']['hit']]
    summary['b'] = {
        'hit': _rate_of(selected, lambda record: record['b']['hit']),
        'candidates': distribution_summary([record['n_b'] for record in selected]),
        'zero_candidates': _rate_of(selected, lambda record: record['n_b'] == 0),
        'hit_given_builder_hit': _rate_of(builder_hit_events, lambda record: record['b']['hit']),
        'zero_given_builder_hit': _rate_of(builder_hit_events, lambda record: record['n_b'] == 0),
        'index_consistent': _rate_of(selected, lambda record: record['b']['index_consistent']),
        'kstar_in_range': _rate_of(selected, lambda record: record['b']['kstar_in_range']),
        'unique_triplets': distribution_summary(
            [record['b']['n_unique_triplets'] for record in selected]),
    }

    pv_track = [record for record in track_events if record['pv'] is not None]
    pv_gt = [record for record in pv_track if record['pv']['gt_median_dz'] is not None]
    truth_far = [record['pv']['stored_distance_3d'] > PV_PROXY_CM for record in pv_gt]
    proxy_far = [abs(record['pv']['gt_median_dz']) > PV_PROXY_CM for record in pv_gt]
    far_pairs = list(zip(truth_far, proxy_far))
    summary['pv'] = {
        'all_events': _summarize_pv(with_truth),
        'track_events': _summarize_pv(pv_track),
        'selected_events': _summarize_pv([record for record in selected if record['pv'] is not None]),
        'dz_proxy_far': rate_entry(sum(proxy_far), len(pv_gt)),
        'dz_proxy_x_truth': {
            'truth_far_proxy_far': int(sum(
                truth_flag and proxy_flag for truth_flag, proxy_flag in far_pairs)),
            'truth_far_proxy_near': int(sum(
                truth_flag and not proxy_flag for truth_flag, proxy_flag in far_pairs)),
            'truth_near_proxy_far': int(sum(
                (not truth_flag) and proxy_flag for truth_flag, proxy_flag in far_pairs)),
            'truth_near_proxy_near': int(sum(
                (not truth_flag) and (not proxy_flag) for truth_flag, proxy_flag in far_pairs)),
        },
    }

    summary['impostors'] = {}
    for window in IMPOSTOR_WINDOWS:
        blocks = [record['impostors'][window] for record in selected]
        counts = Counter()
        for block in blocks:
            counts.update(block['categories'])
        total = sum(counts.values())
        summary['impostors'][window] = {
            'counts': {name: int(counts[name]) for name in IMPOSTOR_CATEGORIES},
            'fractions': {name: (counts[name] / total if total else None)
                          for name in IMPOSTOR_CATEGORIES},
            'n_window_per_event': distribution_summary([block['n_window'] for block in blocks]),
            'events_with_kstar_daughter': _rate_of(
                blocks, lambda block: block['categories']['kstar_daughter'] > 0),
            'events_with_empty_window': _rate_of(blocks, lambda block: block['n_window'] == 0),
        }

    muons = [record['muon'] for record in selected]
    with_truth_muon = [muon for muon in muons if muon['n_tau_muons'] > 0]
    without_truth_muon = [muon for muon in muons if muon['n_tau_muons'] == 0]
    summary['muon'] = {
        'events_with_truth_muon': rate_entry(len(with_truth_muon), len(muons)),
        'recall': _rate_of(with_truth_muon, lambda muon: muon['picked_is_tau']),
        'picked_any': _rate_of(muons, lambda muon: muon['picked'] >= 0),
        'false_pass': _rate_of(without_truth_muon, lambda muon: muon['picked'] >= 0),
        'n_other_muons': sum(muon['n_other_muons'] for muon in muons),
        'cut_pass_rates': {
            cut: _rate_of(with_truth_muon,
                          lambda muon, cut_name=cut: muon['tau_cut_flags'][cut_name])
            for cut in ('soft', 'opposite_charge', 'dz', 'delta_r', 'mass')},
    }
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_rate(entry):
    """entry: rate_entry dict. Returns 'successes/total = rate [low, high]'."""
    if entry['total'] == 0:
        return '0/0'
    return (f"{entry['successes']}/{entry['total']} = {entry['rate']:.3f} "
            f"[{entry['ci'][0]:.3f}, {entry['ci'][1]:.3f}]")


def format_quantiles(entry):
    """entry: distribution_summary dict. Returns 'p10 / p50 / p90 (max)'."""
    if not entry['count']:
        return 'n=0'
    return f"{entry['p10']:.3g} / {entry['p50']:.3g} / {entry['p90']:.3g} (max {entry['max']:.3g})"


def _table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(lines)


def render_markdown(per_file, pooled, inventory_table, withmatch, index_checks):
    """per_file: {name: summary}; pooled: summary; inventory_table: output
    of inventory(); withmatch / index_checks: {name: check}. Returns the
    markdown tables of the survey."""
    columns = list(per_file) + ['pooled']
    summaries = {**per_file, 'pooled': pooled}
    sections = ['# abcbatch ntuple survey — generated tables']

    sections.append('## Inventory\n' + _table(
        ['family', 'branches', 'converter', 'merge_kept', 'reference_not_kept', 'new'],
        [[family, entry['n']] + [len(entry['classes'].get(name, []))
                                 for name in ('converter', 'merge_kept', 'reference_not_kept', 'new')]
         for family, entry in sorted(inventory_table.items())]))

    funnel_rows = [[name] + [summaries[column]['funnel']['buckets'][name] for column in columns]
                   for name in FUNNEL_BUCKETS]
    funnel_rows += [['track events (raw)'] + [summaries[column]['funnel']['n_track_events_raw'] for column in columns],
                    ['track events (converter-clean)'] + [summaries[column]['funnel']['n_track_events'] for column in columns],
                    ['tracks, no selected K*'] + [summaries[column]['funnel']['n_tracks_no_kstar'] for column in columns],
                    ['selected of clean track events'] + [format_rate(summaries[column]['funnel']['selected_of_track_events']) for column in columns]]
    sections.append('## Funnel\n' + _table(['bucket'] + columns, funnel_rows))

    sections.append('## Cutflow (fraction of events with counter > 0, pooled)\n' + _table(
        ['counter', 'positive', 'mean'],
        [[name, format_rate(entry['positive']), f"{entry['mean']:.2f}"]
         for name, entry in pooled['cutflow'].items()]))

    gen = pooled['gen']
    sections.append('## Gen truth and labels (pooled)\n' + _table(
        ['quantity', 'value'],
        [['gen tau-pion count per event', gen['n_gen_pions_counts']],
         ['gen pion reconstructed, all events', format_rate(gen['pion_reco_all_events'])],
         ['gen pion reconstructed, track events', format_rate(gen['pion_reco_track_events'])],
         ['all 3 reconstructed, track events', format_rate(gen['all3_reco_track_events'])]]
        + [[f'reco efficiency, gen pT {name}', format_rate(entry)] for name, entry in gen['pion_reco_by_pt'].items()]
        + [['gen tau pT, selected (p10/p50/p90)', format_quantiles(gen['tau_pt_selected'])],
           ['decay length 3D, selected [cm]', format_quantiles(gen['decay_length_3d_selected'])],
           ['decay length xy, selected [cm]', format_quantiles(gen['decay_length_xy_selected'])],
           ['GenFinalState_B pointer != top B', format_rate(gen['b_pointer_mismatch'])],
           ['GT walks to tau / GT no walk / walk not GT',
            f"{pooled['labels']['gt_walks']} / {pooled['labels']['gt_no_walk']} / {pooled['labels']['walk_not_gt']}"],
           ['GT trackgenFlav counts', pooled['labels']['gt_gen_flav_counts']],
           ['GT gentrackMatchDr > 0.05', format_rate(pooled['labels']['gt_match_dr_above_0.05'])],
           ['GT |gentrackMatchDPt| > 0.2', format_rate(pooled['labels']['gt_match_dpt_above_0.2'])],
           ['GT dz spread > 0.2 cm', format_rate(pooled['labels']['gt_dz_spread_above_0.2'])],
           ['GT dz spread > 1 cm', format_rate(pooled['labels']['gt_dz_spread_above_1.0'])],
           ['gen<->track bijection failures', pooled['labels']['bijection_failures']]]))

    sections.append('## K* selection\n' + _table(
        ['quantity'] + columns,
        [['selected K* is the true K*'] + [format_rate(summaries[column]['kstar']['selected_is_true']) for column in columns],
         ['true K* row exists (track events)'] + [format_rate(summaries[column]['kstar']['true_row_exists']) for column in columns],
         ['true-row multiplicity'] + [summaries[column]['kstar']['true_row_multiplicity'] for column in columns]]))

    abcnet_rows = [
        ['rows per event (p10/p50/p90)'] + [format_quantiles(summaries[column]['abcnet']['rows_per_event']) for column in columns],
        ['lists at the 50-row cap'] + [format_rate(summaries[column]['abcnet']['lists_at_cap']) for column in columns],
        ['rows without a track index'] + [format_rate(summaries[column]['abcnet']['rows_without_track']) for column in columns],
        ['GT track in any list'] + [format_rate(summaries[column]['abcnet']['coverage']['per_track_any_list']) for column in columns],
        ['all 3 GT in any list'] + [format_rate(summaries[column]['abcnet']['coverage']['all3_any_list']) for column in columns],
    ]
    for list_type in ('true', 'selected'):
        coverage = {column: summaries[column]['abcnet']['coverage'] for column in columns}
        abcnet_rows += [
            [f'{list_type}-K* list exists'] + [format_rate(coverage[column][f'{list_type}_list_exists']) for column in columns],
            [f'all 3 GT in {list_type}-K* list (conditional)'] + [format_rate(coverage[column][f'all3_in_{list_type}_list_conditional']) for column in columns],
            [f'all 3 GT in {list_type}-K* list (unconditional)'] + [format_rate(coverage[column][f'all3_in_{list_type}_list_unconditional']) for column in columns],
        ]
        for top_k in RECALL_KS:
            abcnet_rows.append([f'{list_type}-K* list R@{top_k} (unconditional)']
                               + [format_rate(summaries[column]['abcnet']['recall'][list_type]['unconditional'][str(top_k)]) for column in columns])
        abcnet_rows.append([f'{list_type}-K* rows above WP (p10/p50/p90)']
                           + [format_quantiles(summaries[column]['abcnet']['pool_at_wp'][list_type]) for column in columns])
    for pool_name in ('all_rows', 'track_max', 'true_list', 'selected_list'):
        abcnet_rows.append([f'AUC {pool_name}'] + [
            'n/a' if summaries[column]['abcnet']['auc'].get(pool_name) is None
            else f"{summaries[column]['abcnet']['auc'][pool_name]:.3f}" for column in columns])
    abcnet_rows += [
        ['GT rows above WP'] + [format_rate(summaries[column]['abcnet'].get('gt_rows_above_wp', rate_entry(0, 0))) for column in columns],
        ['non-GT rows above WP'] + [format_rate(summaries[column]['abcnet'].get('other_rows_above_wp', rate_entry(0, 0))) for column in columns],
        ['GT row scores (p10/p50/p90)'] + [format_quantiles(summaries[column]['abcnet'].get('gt_row_scores', distribution_summary([]))) for column in columns],
    ]
    sections.append('## ABCNet block (selected events)\n' + _table(['quantity'] + columns, abcnet_rows))

    sections.append('## Builder (TauTo3Pi, selected events)\n' + _table(
        ['quantity'] + columns,
        [['exact GT triplet hit'] + [format_rate(summaries[column]['builder']['hit']) for column in columns],
         ['candidates per event (p10/p50/p90)'] + [format_quantiles(summaries[column]['builder']['candidates']) for column in columns],
         ['events with > 50 candidates'] + [format_rate(summaries[column]['builder']['over_50_candidates']) for column in columns],
         ['track pool size (p10/p50/p90)'] + [format_quantiles(summaries[column]['builder']['pool_size']) for column in columns],
         ['WithMatch identical to TauTo3Pi'] + [
             str(withmatch[column]['all_indices_equal']) if column in withmatch else '—'
             for column in columns],
         ['mcMatch values'] + [str(withmatch[column]['mcmatch_values']) if column in withmatch else '—'
                               for column in columns]]))
    sections.append('### Builder hit by tercile (pooled)\n' + _table(
        ['stratum', 'exact GT triplet hit'],
        [[f'gen tau pT {name}', format_rate(entry)]
         for name, entry in pooled['builder']['hit_by_tau_pt_tercile'].items()]
        + [[f'tracks above cutoff {name}', format_rate(entry)]
           for name, entry in pooled['builder']['hit_by_multiplicity_tercile'].items()]))

    sections.append('## B candidates (selected events)\n' + _table(
        ['quantity'] + columns,
        [['exact GT triplet in a B candidate'] + [format_rate(summaries[column]['b']['hit']) for column in columns],
         ['B hit given builder hit'] + [format_rate(summaries[column]['b']['hit_given_builder_hit']) for column in columns],
         ['zero B candidates given builder hit'] + [format_rate(summaries[column]['b']['zero_given_builder_hit']) for column in columns],
         ['zero B candidates'] + [format_rate(summaries[column]['b']['zero_candidates']) for column in columns],
         ['pi indices consistent with tau_idx'] + [format_rate(summaries[column]['b']['index_consistent']) for column in columns],
         ['B candidates per event (p10/p50/p90)'] + [format_quantiles(summaries[column]['b']['candidates']) for column in columns]]))

    pv_rows = []
    for population in ('track_events', 'selected_events'):
        block = {column: summaries[column]['pv'][population] for column in columns}
        pv_rows += [[f'{population}: n'] + [block[column]['n'] for column in columns]]
        for threshold in PV_FAR_CM:
            pv_rows.append([f'{population}: stored PV > {threshold} cm from true PV (3D)']
                           + [format_rate(block[column]['misassigned_3d'][str(threshold)]) for column in columns])
        pv_rows += [
            [f'{population}: |dz| > 0.2 cm'] + [format_rate(block[column]['misassigned_z']['0.2']) for column in columns],
            [f'{population}: nearest PVtx within 0.2 cm'] + [format_rate(block[column]['nearest_within_0.2']) for column in columns],
            [f'{population}: nearest PVtx is index 0'] + [format_rate(block[column]['nearest_is_index_0']) for column in columns],
            [f'{population}: nearest PVtx score rank'] + [block[column]['nearest_score_rank_counts'] for column in columns],
            [f'{population}: stored-near x any-near'] + [block[column]['stored_near_x_any_near'] for column in columns],
        ]
    pv_rows += [['dz proxy |median GT dz| > 0.2 cm'] + [format_rate(summaries[column]['pv']['dz_proxy_far']) for column in columns],
                ['dz proxy x truth (far/near)'] + [summaries[column]['pv']['dz_proxy_x_truth'] for column in columns]]
    sections.append('## PV truth\n' + _table(['quantity'] + columns, pv_rows))

    impostor_rows = []
    for window in IMPOSTOR_WINDOWS:
        for name in IMPOSTOR_CATEGORIES:
            impostor_rows.append([f'{window}: {name}'] + [
                f"{summaries[column]['impostors'][window]['counts'][name]} "
                f"({summaries[column]['impostors'][window]['fractions'][name] or 0:.3f})" for column in columns])
        impostor_rows += [
            [f'{window}: tracks in window per event (p10/p50/p90)'] + [format_quantiles(summaries[column]['impostors'][window]['n_window_per_event']) for column in columns],
            [f'{window}: events with a K* daughter in window'] + [format_rate(summaries[column]['impostors'][window]['events_with_kstar_daughter']) for column in columns],
        ]
    sections.append('## Impostor origins (selected events)\n' + _table(['quantity'] + columns, impostor_rows))

    sections.append('## Tau-muon proxy vs truth (selected events)\n' + _table(
        ['quantity'] + columns,
        [['events with a truth tau-muon'] + [format_rate(summaries[column]['muon']['events_with_truth_muon']) for column in columns],
         ['proxy recall'] + [format_rate(summaries[column]['muon']['recall']) for column in columns],
         ['proxy false pass (events without truth muon)'] + [format_rate(summaries[column]['muon']['false_pass']) for column in columns],
         ['non-tau muons in table'] + [summaries[column]['muon']['n_other_muons'] for column in columns]]
        + [[f'truth muon passes {cut}'] + [format_rate(summaries[column]['muon']['cut_pass_rates'][cut]) for column in columns]
           for cut in ('soft', 'opposite_charge', 'dz', 'delta_r', 'mass')]))

    sections.append('## Index checks\n' + _table(
        ['file', 'max |dpt|', 'max |deta|', 'max |dphi|', 'kstar_idx in range', 'rows', 'rows without track'],
        [[name, check['max_abs_delta_pt'], check['max_abs_delta_eta'], check['max_abs_delta_phi'],
          check['all_kstar_idx_in_range'], check['n_rows'], check['n_missing_track']]
         for name, check in index_checks.items()]))
    return '\n\n'.join(sections) + '\n'


def run(input_glob, output_json, output_markdown, reference_root=None, entry_stop=None):
    """input_glob: ROOT file glob; output_json / output_markdown: paths;
    reference_root: raw nano for the inventory classification (None skips
    it). Returns the summary dict after writing both files."""
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f'no ROOT files match {input_glob}')
    per_file, withmatch, index_checks, all_records = {}, {}, {}, []
    for path in paths:
        events = load_events(path, entry_stop)
        clean = converter_clean_mask(path, entry_stop)
        records = [event_record(events, entry, converter_clean=bool(clean[entry]))
                   for entry in range(len(events))]
        name = os.path.basename(path)
        per_file[name] = summarize(records)
        withmatch[name] = withmatch_check(events)
        index_checks[name] = abcnet_index_check(events)
        all_records.extend(records)
    inventory_table = inventory(uproot.open(paths[0])['Events'], reference_root)
    summary = {'inputs': paths, 'per_file': per_file, 'pooled': summarize(all_records),
               'withmatch': withmatch, 'index_checks': index_checks,
               'inventory': inventory_table}
    for directory in (os.path.dirname(output_json), os.path.dirname(output_markdown)):
        if directory:
            os.makedirs(directory, exist_ok=True)
    with open(output_json, 'w') as handle:
        json.dump(summary, handle, indent=2)
    with open(output_markdown, 'w') as handle:
        handle.write(render_markdown(per_file, summary['pooled'], inventory_table,
                                     withmatch, index_checks))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--input-glob', default=DEFAULT_INPUT_GLOB)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-markdown', required=True)
    parser.add_argument('--reference-root', default=DEFAULT_REFERENCE_ROOT,
                        help='raw nano whose branch set defines "reference_not_kept"')
    parser.add_argument('--entry-stop', type=int, default=None,
                        help='read only the first N entries of every file')
    arguments = parser.parse_args()
    reference = arguments.reference_root if os.path.exists(arguments.reference_root) else None
    summary = run(arguments.input_glob, arguments.output_json, arguments.output_markdown,
                  reference, arguments.entry_stop)
    pooled = summary['pooled']
    print(f"events {pooled['n_events']} | track events {pooled['funnel']['n_track_events']} "
          f"| selected {pooled['funnel']['n_selected']} "
          f"| builder hit {format_rate(pooled['builder']['hit'])} "
          f"| AUC all rows {pooled['abcnet']['auc']['all_rows']}")


if __name__ == '__main__':
    main()
