"""Selection, gen-truth, K* and builder helpers shared by the baseline
extractor and the survey (scripts/python/survey_abcbatch_ntuples.py imports
and re-exports them). Dependency-free (numpy only) so the condor scripts can
import it on lxplus by bare module name.

Index spaces: track indices are raw Track-table positions; gen indices are
GenPart rows; K* rows are KstarToPiKAll rows.
"""
from __future__ import annotations

import numpy as np

PT_CUTOFF = 0.5
PION_PDG = 211
TAU_PDG = 15
B_PDGS = {511, 521, 531}
FUNNEL_BUCKETS = ('quality_dropped', 'no_tracks', 'n_gt_0', 'n_gt_1', 'n_gt_2',
                  'n_gt_4plus', '3_gt_below_cutoff', 'selected')


# ---------------------------------------------------------------------------
# Gen truth walks
# ---------------------------------------------------------------------------

def walk_to_ancestor(gen_pdg, gen_mother, start, target_abs_pdgs, max_depth=25):
    """gen_pdg, gen_mother: (N,). Returns the first particle on the mother
    chain from start (start included) whose |pdgId| is in target_abs_pdgs,
    or -1."""
    current = int(start)
    for _ in range(max_depth):
        if current < 0 or current >= gen_pdg.size:
            return -1
        if abs(int(gen_pdg[current])) in target_abs_pdgs:
            return current
        current = int(gen_mother[current])
    return -1


def top_b_ancestor(gen_pdg, gen_mother, tau_index):
    """gen_pdg, gen_mother: (N,). Returns the index of the top-most B meson
    on the mother chain of tau_index (an oscillated copy's mother is the
    original), or -1."""
    best = -1
    current = walk_to_ancestor(gen_pdg, gen_mother, tau_index, B_PDGS)
    while current >= 0:
        best = current
        current = walk_to_ancestor(gen_pdg, gen_mother, int(gen_mother[current]), B_PDGS)
    return best


def gen_tau_pions(gen_pdg, gen_mother, gen_status):
    """gen_pdg, gen_mother, gen_status: (N,). Returns (M,) int64 indices of
    final-state charged pions with a tau ancestor."""
    candidates = np.flatnonzero((np.abs(gen_pdg) == PION_PDG) & (gen_status == 1))
    pions = [int(index) for index in candidates
             if walk_to_ancestor(gen_pdg, gen_mother, int(gen_mother[index]),
                                 {TAU_PDG}) >= 0]
    return np.asarray(pions, dtype=np.int64)


def gen_truth(gen_pdg, gen_mother, gen_status, gen_pt, gen_vertex):
    """gen_pdg, gen_mother, gen_status, gen_pt: (N,); gen_vertex: (N, 3)
    production vertices. Returns the hadronic tau index, its top-most B
    ancestor, its final-state charged pions, the tau production vertex (3,),
    the tau decay vertex (3,) (= production vertex of its first pion), the
    true primary vertex (3,) (= production vertex of the top-most B), the
    decay lengths and the tau pT; indices are -1 and the rest None when
    the chain is absent."""
    gen_vertex = np.asarray(gen_vertex, dtype=np.float64)
    pion_indices = gen_tau_pions(gen_pdg, gen_mother, gen_status)
    result = {'tau_index': -1, 'b_index': -1, 'pion_indices': pion_indices,
              'tau_vertex': None, 'decay_vertex': None, 'true_pv': None,
              'decay_length_3d': None, 'decay_length_xy': None, 'tau_pt': None}
    if pion_indices.size == 0:
        return result
    tau_index = walk_to_ancestor(gen_pdg, gen_mother,
                                 int(gen_mother[pion_indices[0]]), {TAU_PDG})
    b_index = top_b_ancestor(gen_pdg, gen_mother, tau_index)
    flight = gen_vertex[pion_indices[0]] - gen_vertex[tau_index]
    result.update(tau_index=tau_index, b_index=b_index,
                  tau_vertex=gen_vertex[tau_index],
                  decay_vertex=gen_vertex[pion_indices[0]],
                  decay_length_3d=float(np.linalg.norm(flight)),
                  decay_length_xy=float(np.linalg.norm(flight[:2])),
                  tau_pt=float(gen_pt[tau_index]))
    if b_index >= 0:
        result['true_pv'] = gen_vertex[b_index]
    return result


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_event(track_pt, track_pdg, track_from_tau, pt_cutoff=PT_CUTOFF):
    """track_pt, track_pdg, track_from_tau: (T,). Returns (funnel bucket,
    sorted GT track indices (3,)): for 'selected' the three GT pions above
    the cutoff, for '3_gt_below_cutoff' the three GT pions, else None.
    GT = trackFromTau == 1 and |pdgId| == 211; 'selected' = exactly three
    GT remain after dropping tracks below pt_cutoff (the converter's order:
    cut first, then count)."""
    if track_pt.size == 0:
        return 'no_tracks', None
    gt_indices = np.flatnonzero((track_from_tau == 1) & (np.abs(track_pdg) == PION_PDG))
    above_cutoff = gt_indices[track_pt[gt_indices] >= pt_cutoff]
    if above_cutoff.size == 3:
        return 'selected', above_cutoff
    if gt_indices.size == 3:
        return '3_gt_below_cutoff', gt_indices
    bucket = 'n_gt_4plus' if gt_indices.size > 3 else f'n_gt_{gt_indices.size}'
    return bucket, None


# ---------------------------------------------------------------------------
# K*, ABCNet and builder blocks
# ---------------------------------------------------------------------------

def kstar_rows(all_trk1, all_trk2, from_kstar, selected_trk1, selected_trk2):
    """all_trk1, all_trk2: (K,) daughter track indices of the KstarToPiKAll
    rows; from_kstar: (T,); selected_trk1/2: the KstarToPiK[0] pair or None.
    Returns the rows whose daughters are both flagged trackFromKstar, the row
    equal to the selected pair (either order; None when absent) and whether
    that row is a true one (None without a selected row)."""
    valid = ((all_trk1 >= 0) & (all_trk2 >= 0)
             & (all_trk1 < from_kstar.size) & (all_trk2 < from_kstar.size))
    true_mask = np.zeros(all_trk1.size, dtype=bool)
    true_mask[valid] = ((from_kstar[all_trk1[valid]] == 1)
                        & (from_kstar[all_trk2[valid]] == 1))
    true_rows = [int(row) for row in np.flatnonzero(true_mask)]
    selected_row = None
    if selected_trk1 is not None and selected_trk2 is not None:
        matches = np.flatnonzero(
            ((all_trk1 == selected_trk1) & (all_trk2 == selected_trk2))
            | ((all_trk1 == selected_trk2) & (all_trk2 == selected_trk1)))
        if matches.size:
            selected_row = int(matches[0])
    selected_is_true = None if selected_row is None else selected_row in true_rows
    return {'true_rows': true_rows, 'selected_row': selected_row,
            'selected_is_true': selected_is_true}


def abcnet_lists(trk_idx, kstar_idx):
    """trk_idx, kstar_idx: (A,) ABCNetTau rows. Returns the row count, the
    number of distinct lists (kstar_idx values), rows per list (L,) and the
    rows without a Track index."""
    lists, rows_per_list = np.unique(kstar_idx, return_counts=True)
    return {'n_rows': int(trk_idx.size),
            'n_lists': int(lists.size),
            'rows_per_list': rows_per_list.astype(np.int64),
            'n_missing_track': int(np.sum(trk_idx < 0))}


def sorted_triplet_hit(idx1, idx2, idx3, gt_triplet):
    """idx1, idx2, idx3: (R,) track indices of candidate triplets;
    gt_triplet: (3,). Returns (R,) bool: the candidate's sorted triple
    equals the sorted GT triple (the thesis definition of an exact hit)."""
    candidates = np.sort(np.stack([idx1, idx2, idx3], axis=1).astype(np.int64), axis=1)
    return np.all(candidates == np.sort(np.asarray(gt_triplet, dtype=np.int64)), axis=1)
