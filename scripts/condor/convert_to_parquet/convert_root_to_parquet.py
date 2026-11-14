"""
Convert low-pT tau dataset from CMS NanoAOD ROOT format to sharded Parquet.

Reads multiple merged_noBKstar_batch*.root files sequentially, applies:
  1. Event quality cleaning (corrupt tracks, NaN/Inf)
  2. Pion track selection (|pdgId| == 211)
  3. pT cutoff: drops all pion tracks with pT below threshold
  4. GT pion filter: keeps only events with exactly N tau-origin pions
     surviving the pT cutoff (default: 3)

Output is split into train/ and val/ subdirectories, each containing
sharded Parquet files with jagged awkward arrays consumable by weaver.
With --extended --builder-blocks the shards also carry the inputs of the
CERN TauTo3PiBuilder emulation: per-event K* veto scalars, the builder's
own tau output block and the beam-spot reference point.

After writing, all output files are validated against the pT cutoff
and GT pion count constraints.

Usage:
    python convert_root_to_parquet.py \\
        --input-dir ./root \\
        --output-dir ./parquet_low_pt_cutoff \\
        --train-events 300000 --val-events 75000 \\
        --pt-cutoff 0.5 --gt-pions 3

abcbatch evaluation conversion (every qualifying event, no quota):
    python3 convert_root_to_parquet.py \\
        --input-dir /eos/.../abcbatch0_2024 \\
        --pattern 'abcbatch*_job*_ditaus_mc.root' \\
        --extended --eval-only --shard-events 5000 \\
        --output-dir /eos/.../abcbatch_2024/model/batch0 --val-subdir model

Event identity (event_identity.py): the 5-key (event_run, event_id,
event_luminosity_block, source_batch_id, source_microbatch_id) is unique by
construction. Merged production: the source ids are copied from the
branches. abcbatch job files (no such branches): source_batch_id = batch
number from the file name, source_microbatch_id = job * 100000 + entry
index; event_source_job / event_source_entry carry the decoded values.
Every shard also carries track_root_index (the kept track's position in
the raw Track table, the index space of the baseline blocks) and, in
extended mode, the optional truth / vertex / K* blocks (OPTIONAL_BLOCKS)
present in the first input file.

References:
  - Dataset description: part/data/low-pt/description.tex
  - YAML config: part/data/low-pt/lowpt_tau_trackfinder.yaml

Versions:
  - uproot >= 5.1
  - awkward >= 2.8
  - pyarrow >= 22.0
"""

import os
import sys
import glob
import time
import argparse
from collections import Counter

import numpy as np
import awkward as ak
import uproot

import event_identity  # sibling module: composite event key (batch / job / entry)
import miniaod_reader  # sibling module: raw MiniAOD reader (imports ROOT only when a MiniAOD file is read)
import miniaod_source  # sibling module: MiniAOD file as an ntuple-like Events tree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PION_PDG_ID = 211

ROOT_TREE_NAME = 'Events'

# Synthetic key under which read_and_filter_pion_tracks carries each kept
# track's position in the raw Track table (never requested from uproot);
# written as the track_root_index column.
TRACK_ROOT_INDEX_KEY = '__track_root_index__'
TRACK_ROOT_INDEX_COLUMN = 'track_root_index'

# Complete mapping from ROOT branch names to output column names for track features.
# Only raw features are stored — derived features (log pT, 4-vectors, tanh(dxy),
# vertex displacements, etc.) are computed by weaver at runtime via YAML new_variables.
TRACK_BRANCH_MAP_ALL = {
    'Track_pt': 'track_pt',
    'Track_eta': 'track_eta',
    'Track_phi': 'track_phi',
    'Track_mass': 'track_mass',
    'Track_charge': 'track_charge',
    'Track_dxy': 'track_dxy',
    'Track_dxyS': 'track_dxy_significance',
    'Track_dz': 'track_dz',
    'Track_dzS': 'track_dz_significance',
    'Track_dzTrg': 'track_dz_trigger',
    'Track_normChi2': 'track_norm_chi2',
    'Track_nValidHits': 'track_n_valid_hits',
    'Track_nValidPixelHits': 'track_n_valid_pixel_hits',
    'Track_ptErr': 'track_pt_error',
    'Track_DCASig': 'track_dca_significance',
    'Track_covQopQop': 'track_covariance_qop_qop',
    'Track_covQopLam': 'track_covariance_qop_lambda',
    'Track_covQopPhi': 'track_covariance_qop_phi',
    'Track_covLamLam': 'track_covariance_lambda_lambda',
    'Track_covLamPhi': 'track_covariance_lambda_phi',
    'Track_covPhiPhi': 'track_covariance_phi_phi',
    'Track_vx': 'track_vertex_x',
    'Track_vy': 'track_vertex_y',
    'Track_vz': 'track_vertex_z',
    'Track_trackFromTau': 'track_label_from_tau',
    'Track_covDxyDxy': 'track_covariance_dxy_dxy',
    'Track_covDszDsz': 'track_covariance_dsz_dsz',
    'Track_covDxyDsz': 'track_covariance_dxy_dsz',
    'Track_covLamDxy': 'track_covariance_lambda_dxy',
    'Track_covPhiDxy': 'track_covariance_phi_dxy',
    'Track_covQopDxy': 'track_covariance_qop_dxy',
    'Track_isLostTrk': 'track_is_lost',
    'Track_isMatchedToMuon': 'track_is_matched_to_muon',
    'Track_isMatchedToEle': 'track_is_matched_to_ele',
    'Track_trackFromB': 'track_label_from_b',
}

# Default output columns matching the YAML data config
# (part/data/low-pt/lowpt_tau_trackfinder.yaml).
DEFAULT_COLUMNS = [
    # Kinematic features
    'track_pt',
    'track_eta',
    'track_phi',
    'track_charge',
    # Displacement features
    'track_dxy_significance',
    'track_dz_significance',
    # Track quality / measurement precision
    'track_pt_error',
    'track_n_valid_pixel_hits',
    'track_dca_significance',
    # Helix fit precision (covariance matrix elements)
    'track_covariance_phi_phi',
    'track_covariance_lambda_lambda',
    'track_norm_chi2',
    # Per-track binary label: 1 = tau-origin pion, 0 = background
    'track_label_from_tau',
]

# Extended (H6) track columns: DEFAULT_COLUMNS + raw helix geometry, the
# dxy/dsz covariance rows, quality/lepton-overlap flags, and the from-B
# auxiliary label.
EXTENDED_TRACK_COLUMNS = DEFAULT_COLUMNS + [
    'track_dxy',
    'track_dz',
    'track_vertex_x',
    'track_vertex_y',
    'track_vertex_z',
    'track_covariance_dxy_dxy',
    'track_covariance_dsz_dsz',
    'track_covariance_dxy_dsz',
    'track_covariance_lambda_dxy',
    'track_covariance_phi_dxy',
    'track_covariance_qop_dxy',
    # q/p variance: the one diagonal of the TauTo3PiBuilder's 8-element
    # packed covariance that the legacy columns do not carry.
    'track_covariance_qop_qop',
    'track_n_valid_hits',
    'track_is_lost',
    'track_is_matched_to_muon',
    'track_is_matched_to_ele',
    'track_label_from_b',
]

# Event-cleaning is FROZEN to the ROOT branches behind DEFAULT_COLUMNS.
# The NaN/Inf/extreme artifact scan drops whole events, so letting it run
# over newly requested branches would change which events survive and
# shift the event order — invalidating every positionally-aligned
# downstream artifact (cascade dumps, window dumps). New branches are
# sanitized in place instead of dropping events.
LEGACY_CLEAN_BRANCHES = frozenset([
    'Track_pt', 'Track_eta', 'Track_phi', 'Track_charge',
    'Track_dxyS', 'Track_dzS', 'Track_ptErr', 'Track_nValidPixelHits',
    'Track_DCASig', 'Track_covPhiPhi', 'Track_covLamLam', 'Track_normChi2',
    'Track_trackFromTau',
])

# Muon collection (H6.M): real PF muons — the only true non-hadron reco
# objects in the dataset. Never filtered, never event-dropping.
MUON_BRANCH_MAP = {
    'Muon_pt': 'muon_pt',
    'Muon_eta': 'muon_eta',
    'Muon_phi': 'muon_phi',
    'Muon_charge': 'muon_charge',
    'Muon_dz': 'muon_dz',
    'Muon_dzErr': 'muon_dz_error',
    'Muon_dxy': 'muon_dxy',
    'Muon_dxyErr': 'muon_dxy_error',
    'Muon_ip3d': 'muon_ip3d',
    'Muon_sip3d': 'muon_sip3d',
    'Muon_softId': 'muon_soft_id',
    'Muon_mediumId': 'muon_medium_id',
    'Muon_isTriggering': 'muon_is_triggering',
    'Muon_pfRelIso03_all': 'muon_pf_rel_iso03',
    'Muon_vx': 'muon_vertex_x',
    'Muon_vy': 'muon_vertex_y',
    'Muon_vz': 'muon_vertex_z',
}

MUON_INTEGER_COLUMNS = {
    'muon_charge', 'muon_soft_id', 'muon_medium_id', 'muon_is_triggering',
}

# Inclusive secondary-vertex collection (H6.S).
SV_BRANCH_MAP = {
    'SV_x': 'sv_x',
    'SV_y': 'sv_y',
    'SV_z': 'sv_z',
    'SV_dlen': 'sv_dlen',
    'SV_dlenSig': 'sv_dlen_sig',
    'SV_pAngle': 'sv_pangle',
    'SV_mass': 'sv_mass',
    'SV_ntracks': 'sv_n_tracks',
    'SV_chi2': 'sv_chi2',
    'SV_dxySig': 'sv_dxy_sig',
}

SV_INTEGER_COLUMNS = {'sv_n_tracks'}

# Sub-cutoff track complement (H6.C companions): tracks with
# OTHER_TRACK_PT_FLOOR <= pT < pt_cutoff. The pdgId filter is vacuous
# (every charged hadron carries pdgId ±211), so no pdg column is stored;
# label_from_tau is provably always 0 for these (the exactly-3-GT filter
# forbids sub-cutoff GT pions) and is omitted too.
OTHER_TRACK_BRANCH_MAP = {
    'Track_pt': 'other_track_pt',
    'Track_eta': 'other_track_eta',
    'Track_phi': 'other_track_phi',
    'Track_dz': 'other_track_dz',
    'Track_charge': 'other_track_charge',
    'Track_isMatchedToMuon': 'other_track_is_matched_to_muon',
    'Track_isMatchedToEle': 'other_track_is_matched_to_ele',
    'Track_trackFromB': 'other_track_label_from_b',
}

OTHER_TRACK_INTEGER_COLUMNS = {
    'other_track_charge', 'other_track_is_matched_to_muon',
    'other_track_is_matched_to_ele', 'other_track_label_from_b',
}

OTHER_TRACK_PT_FLOOR = 0.3

# Pileup context (H6.P).
EXTENDED_EVENT_BRANCHES = ['OtherPV_z', 'PV_npvs', 'PV_npvsGood']

# CERN TauTo3PiBuilder emulation inputs (--builder-blocks). Never filtered,
# never event-dropping. The K* pair table (all pairs in the sample files,
# thousands of rows per event) is reduced to per-event scalars: the first
# row is kstarcandidates->front() of an all-pairs collection, the
# highest-pT row stands in for the best-by-fitted-pT collection the
# current DiTrackBuilder emits. Indices address the unfiltered NanoAOD
# Track table (= the builder's SelectedTracks); -1 when the event has no
# K* pair.
KSTAR_BRANCHES = ['KstarToPiK_trk1_idx', 'KstarToPiK_trk2_idx', 'KstarToPiK_pt']

KSTAR_SCALAR_COLUMNS = [
    'event_n_kstar',
    'event_kstar_front_track1_index', 'event_kstar_front_track2_index',
    'event_kstar_leading_track1_index', 'event_kstar_leading_track2_index',
]

# The builder's own output for the same event: validation target of the
# emulation (indices again address the unfiltered Track table).
TAU_BUILDER_BRANCH_MAP = {
    'TauTo3Pi_idx1': 'tau_track1_index',
    'TauTo3Pi_idx2': 'tau_track2_index',
    'TauTo3Pi_idx3': 'tau_track3_index',
    'TauTo3Pi_vprob': 'tau_vertex_probability',
    'TauTo3Pi_v4trkprob': 'tau_four_track_vertex_probability',
    'TauTo3Pi_fit_pt': 'tau_fit_pt',
    'TauTo3Pi_mass': 'tau_mass',
    'TauTo3Pi_vx': 'tau_vertex_x',
    'TauTo3Pi_vy': 'tau_vertex_y',
    'TauTo3Pi_vz': 'tau_vertex_z',
    'TauTo3Pi_declen_sig': 'tau_decay_length_significance',
}

TAU_BUILDER_INTEGER_COLUMNS = {
    'tau_track1_index', 'tau_track2_index', 'tau_track3_index',
}

# Beam spot: the builder's reference point for ip / declen / lxyip.
BEAM_SPOT_BRANCH_MAP = {
    'BeamSpot_x': 'event_beam_spot_x',
    'BeamSpot_y': 'event_beam_spot_y',
    'BeamSpot_z': 'event_beam_spot_z',
}

# Event-level branches (scalars per event).
# PV_x/y/z: primary vertex coordinates (float).
# run, event, luminosityBlock: CMS event identifiers for tracing back to ROOT source.
# source_batch_id, source_microbatch_id: added during merge_batches to uniquely
#   identify the tau candidate together with (run, event, luminosityBlock).
# source_batch_id, source_microbatch_id: added during merge_batches to
#   disambiguate tau candidates — (run, event, luminosityBlock) is NOT unique
#   because the same CMS event appears in multiple microbatch files.
EVENT_BRANCHES = [
    'PV_x', 'PV_y', 'PV_z',
    'run', 'event', 'luminosityBlock',
    'source_batch_id', 'source_microbatch_id',
]

# Integer columns for type casting (int32 instead of float32).
# Optional blocks (extended mode only): truth, vertex and K*-candidate
# branches of the raw / abcbatch productions that the merged production
# lacks. A block is written when ALL its branches exist in the first input
# file of a run and is then required in every later file (fixed schema per
# run). Index columns point into the raw Track table (= track_root_index).
OPTIONAL_TRACK_BRANCH_MAP = {
    'Track_trackFromKstar': 'track_label_from_kstar',
    'Track_trackgenIdx': 'track_gen_index',
    'Track_trackgenFlav': 'track_gen_flavour',
    'Track_isPacked': 'track_is_packed',
}
OPTIONAL_TRACK_INTEGER_COLUMNS = set(OPTIONAL_TRACK_BRANCH_MAP.values())
OPTIONAL_EVENT_SCALAR_MAP = {
    'PV_score': 'event_primary_vertex_score',
    'PV_sumpt2': 'event_primary_vertex_sum_pt2',
}
OPTIONAL_EVENT_JAGGED_MAP = {'OtherPV_score': 'event_other_pv_score'}
PVTX_BRANCH_MAP = {
    'PVtx_vx': 'pvtx_x', 'PVtx_vy': 'pvtx_y', 'PVtx_vz': 'pvtx_z',
    'PVtx_covXX': 'pvtx_cov_xx', 'PVtx_covXY': 'pvtx_cov_xy', 'PVtx_covXZ': 'pvtx_cov_xz',
    'PVtx_covYY': 'pvtx_cov_yy', 'PVtx_covYZ': 'pvtx_cov_yz', 'PVtx_covZZ': 'pvtx_cov_zz',
    'PVtx_ndof': 'pvtx_ndof', 'PVtx_chi2': 'pvtx_chi2', 'PVtx_score': 'pvtx_score',
    'PVtx_ntracks': 'pvtx_n_tracks',
}
PVTX_INTEGER_COLUMNS = {'pvtx_n_tracks'}
GENPART_BRANCH_MAP = {
    'GenPart_pdgId': 'gen_pdg_id', 'GenPart_status': 'gen_status',
    'GenPart_genPartIdxMother': 'gen_mother_index',
    'GenPart_pt': 'gen_pt', 'GenPart_eta': 'gen_eta', 'GenPart_phi': 'gen_phi',
    'GenPart_vx': 'gen_vertex_x', 'GenPart_vy': 'gen_vertex_y', 'GenPart_vz': 'gen_vertex_z',
    'GenPart_gentrackIdx': 'gen_track_index',
    'GenPart_gentrackMatchDr': 'gen_track_match_delta_r',
    'GenPart_gentrackMatchDPt': 'gen_track_match_delta_pt',
}
GENPART_INTEGER_COLUMNS = {'gen_pdg_id', 'gen_status', 'gen_mother_index', 'gen_track_index'}
OPTIONAL_MUON_BRANCH_MAP = {
    'Muon_muonFromTau': 'muon_label_from_tau',
    'Muon_muonFromB': 'muon_label_from_b',
    'Muon_muonFromKstar': 'muon_label_from_kstar',
    'Muon_muonGenFlav': 'muon_gen_flavour',
    'Muon_muonGenIdx': 'muon_gen_index',
}
OPTIONAL_MUON_INTEGER_COLUMNS = set(OPTIONAL_MUON_BRANCH_MAP.values())
KSTAR_ALL_BRANCH_MAP = {
    'KstarToPiKAll_trk1_idx': 'kstar_track1_root_index',
    'KstarToPiKAll_trk2_idx': 'kstar_track2_root_index',
    'KstarToPiKAll_vtx_x': 'kstar_vertex_x', 'KstarToPiKAll_vtx_y': 'kstar_vertex_y',
    'KstarToPiKAll_vtx_z': 'kstar_vertex_z',
    'KstarToPiKAll_svprob': 'kstar_vertex_probability',
    'KstarToPiKAll_fitted_mass_Kpi': 'kstar_fitted_mass_kpi',
    'KstarToPiKAll_fitted_mass_piK': 'kstar_fitted_mass_pik',
    'KstarToPiKAll_fitted_pt': 'kstar_fitted_pt',
    'KstarToPiKAll_lxy': 'kstar_lxy', 'KstarToPiKAll_lxyerr': 'kstar_lxy_error',
    'KstarToPiKAll_pt': 'kstar_pt', 'KstarToPiKAll_eta': 'kstar_eta',
    'KstarToPiKAll_phi': 'kstar_phi', 'KstarToPiKAll_charge': 'kstar_charge',
}
KSTAR_ALL_INTEGER_COLUMNS = {'kstar_track1_root_index', 'kstar_track2_root_index', 'kstar_charge'}
KSTAR_SELECTED_BRANCH_MAP = {
    'KstarToPiK_trk1_idx': 'event_kstar_selected_track1_root_index',
    'KstarToPiK_trk2_idx': 'event_kstar_selected_track2_root_index',
}
KSTAR_SELECTED_INTEGER_COLUMNS = set(KSTAR_SELECTED_BRANCH_MAP.values())
# Blocks only the MiniAOD source provides (miniaod_source.py): vertex
# association, anchor-muon and generator-match columns per track, trigger
# and anchor flags per muon, trigger / pileup / beam-spot scalars per event,
# and the K* candidate distances. Trigger bit i = path i of
# miniaod_source.MUON_TRIGGER_PATHS.
MINIAOD_TRACK_BRANCH_MAP = {
    'Track_fromPV': 'track_from_pv',
    'Track_pvAssocQuality': 'track_pv_association_quality',
    'Track_pvIdx': 'track_pv_index',
    'Track_lostInnerHits': 'track_lost_inner_hits',
    'Track_puppiWeight': 'track_puppi_weight',
    'Track_covLamDsz': 'track_covariance_lambda_dsz',
    'Track_dzTrg': 'track_dz_to_anchor_muon',
    'Track_genMatchDr': 'track_gen_match_delta_r',
    'Track_genAssocIdx': 'track_gen_association_index',
}
MINIAOD_TRACK_INTEGER_COLUMNS = {
    'track_from_pv', 'track_pv_association_quality', 'track_pv_index', 'track_lost_inner_hits',
    'track_gen_association_index',
}
MINIAOD_MUON_BRANCH_MAP = {
    'Muon_isPFcand': 'muon_is_pf',
    'Muon_triggerBits': 'muon_trigger_bits',
    'Muon_isAnchor': 'muon_is_anchor',
}
MINIAOD_EVENT_BRANCH_MAP = {
    'Trigger_bits': 'event_trigger_bits',
    'Trigger_firedAny': 'event_fired_any_muon_path',
    'Trigger_firedIP6': 'event_fired_ip6',
    'Pileup_nTrueInt': 'event_n_true_interactions',
    'BeamSpot_x': 'event_beam_spot_x',
    'BeamSpot_y': 'event_beam_spot_y',
    'BeamSpot_z': 'event_beam_spot_z',
}
MINIAOD_EVENT_INTEGER_COLUMNS = {'event_trigger_bits', 'event_fired_any_muon_path', 'event_fired_ip6'}
MINIAOD_KSTAR_BRANCH_MAP = {
    'KstarToPiKAll_muon_dca': 'kstar_muon_dca',
    'KstarToPiKAll_pair_dca': 'kstar_pair_dca',
}
# block name -> (ROOT branch -> column map, integer column names)
OPTIONAL_BLOCKS = {
    'track': (OPTIONAL_TRACK_BRANCH_MAP, OPTIONAL_TRACK_INTEGER_COLUMNS),
    'event_scalar': (OPTIONAL_EVENT_SCALAR_MAP, set()),
    'event_jagged': (OPTIONAL_EVENT_JAGGED_MAP, set()),
    'pvtx': (PVTX_BRANCH_MAP, PVTX_INTEGER_COLUMNS),
    'genpart': (GENPART_BRANCH_MAP, GENPART_INTEGER_COLUMNS),
    'muon_truth': (OPTIONAL_MUON_BRANCH_MAP, OPTIONAL_MUON_INTEGER_COLUMNS),
    'kstar_all': (KSTAR_ALL_BRANCH_MAP, KSTAR_ALL_INTEGER_COLUMNS),
    'kstar_selected': (KSTAR_SELECTED_BRANCH_MAP, KSTAR_SELECTED_INTEGER_COLUMNS),
    'track_miniaod': (MINIAOD_TRACK_BRANCH_MAP, MINIAOD_TRACK_INTEGER_COLUMNS),
    'muon_miniaod': (MINIAOD_MUON_BRANCH_MAP, set(MINIAOD_MUON_BRANCH_MAP.values())),
    'event_miniaod': (MINIAOD_EVENT_BRANCH_MAP, MINIAOD_EVENT_INTEGER_COLUMNS),
    'kstar_miniaod': (MINIAOD_KSTAR_BRANCH_MAP, set()),
}
# Blocks that ride the track branch map (filtered with the tracks) and
# blocks of per-event scalars.
PER_TRACK_OPTIONAL_BLOCKS = ('track', 'track_miniaod')
EVENT_SCALAR_OPTIONAL_BLOCKS = ('event_scalar', 'event_miniaod')
# List whose length each jagged optional block must share (default: the
# block's own first column).
OPTIONAL_BLOCK_LENGTH_REFERENCE = {
    'track': 'track_pt', 'track_miniaod': 'track_pt', 'muon_truth': 'muon_pt', 'muon_miniaod': 'muon_pt',
    'event_jagged': 'event_other_pv_z', 'kstar_miniaod': 'kstar_track1_root_index',
}

# Track selections (--track-selection): every pion track, or only those not
# used in the fit of another primary vertex (Track_fromPV > 0, the pool of
# the collaborator's ntuple; MiniAOD input only).
TRACK_SELECTIONS = ('all', 'from_pv')
FROM_PV_BRANCH = 'Track_fromPV'

# A MiniAOD run skips unreadable input files (CRAB leftovers, storage
# errors) and fails only above this fraction of the files.
MAXIMUM_UNREADABLE_FRACTION = 0.02

TRACK_INTEGER_COLUMNS = {
    'track_charge', 'track_n_valid_hits',
    'track_n_valid_pixel_hits', 'track_label_from_tau',
    'track_is_lost', 'track_is_matched_to_muon',
    'track_is_matched_to_ele', 'track_label_from_b',
    TRACK_ROOT_INDEX_COLUMN,
} | OPTIONAL_TRACK_INTEGER_COLUMNS | MINIAOD_TRACK_INTEGER_COLUMNS

# The H6-extension float columns of the kept-track block. These are NOT
# covered by the legacy clean-event mask (frozen to LEGACY_CLEAN_BRANCHES
# for positional determinism), so source-borne NaN/Inf must be sanitized
# instead of event-dropped.
NEW_TRACK_FLOAT_COLUMNS = [
    column for column in EXTENDED_TRACK_COLUMNS
    if column not in DEFAULT_COLUMNS and column not in TRACK_INTEGER_COLUMNS
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_root_files(input_dir, pattern='merged_noBKstar_batch*.root'):
    """Find and sort ROOT files numerically by (batch number, job number).

    Args:
        input_dir: Directory containing ROOT files, or a list of such
            directories (e.g. the batch directories of one production).
        pattern: Glob pattern for matching files; '**/' prefixes descend
            into subdirectories (CRAB-style nested productions).

    Returns:
        List of file paths sorted by event_identity.file_sort_key (merged
        production: batch number; abcbatch production: batch, then job).

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    input_dirs = [input_dir] if isinstance(input_dir, str) else list(input_dir)
    matches = [path for directory in input_dirs
               for path in glob.glob(os.path.join(directory, pattern), recursive=True)]
    if not matches:
        raise FileNotFoundError(
            f'No ROOT files matching "{pattern}" found in {input_dirs}'
        )

    matches.sort(key=event_identity.file_sort_key)
    return matches


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def load_root_tree(filepath):
    """Open a ROOT file and return the Events TTree.

    Args:
        filepath: Path to the ROOT file.

    Returns:
        An uproot TTree object for the Events tree of an ntuple; for a
        MiniAOD file a miniaod_source.MiniAODTree, which exposes the same
        keys() / num_entries / arrays() with the ntuple's branch names.

    Raises:
        FileNotFoundError: If the ROOT file does not exist.
        KeyError: If the Events tree is not found in the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'ROOT file not found: {filepath}')

    root_file = uproot.open(filepath)

    if ROOT_TREE_NAME not in root_file:
        available_keys = list(root_file.keys())
        raise KeyError(
            f'Tree "{ROOT_TREE_NAME}" not found in {filepath}. '
            f'Available keys: {available_keys}'
        )

    tree = root_file[ROOT_TREE_NAME]
    if miniaod_reader.is_miniaod_tree(tree):
        return miniaod_source.MiniAODTree(filepath, num_entries=tree.num_entries)
    return tree


def read_and_filter_pion_tracks(tree, track_branch_map, entry_start=None, entry_stop=None,
                                track_selection='all'):
    """Read Track branches and filter to keep only charged pions (|pdgId| == 211).

    With track_selection='from_pv' only the pions with Track_fromPV > 0
    are kept (a tree without that branch raises ValueError).

    The filtering applies a boolean mask per event, so the output jagged arrays
    have variable length (only pion entries survive).

    Also drops events where any track has corrupt eta (|eta| > 5 or NaN).
    These are simulation artefacts (e.g. event 11960 in merged_noBKstar.root
    has 24 tracks with eta up to ~7e31 and 2 with NaN).

    Args:
        tree: An uproot TTree object.
        track_branch_map: Dict mapping ROOT branch names to output column names.
        entry_start: Optional first entry index to read.
        entry_stop: Optional last entry index (exclusive) to read.

    Returns:
        Tuple of (filtered_data, clean_event_mask) where:
          - filtered_data: dict mapping ROOT branch names to filtered awkward arrays.
          - clean_event_mask: boolean numpy array (True = event is clean).
    """
    if track_selection not in TRACK_SELECTIONS:
        raise ValueError(f'track_selection must be one of {TRACK_SELECTIONS}, got {track_selection!r}')
    if track_selection == 'from_pv' and FROM_PV_BRANCH not in tree.keys():
        raise ValueError(f"track_selection='from_pv' needs the {FROM_PV_BRANCH} branch "
                         f'(MiniAOD input); this tree does not have it')

    # Read all track branches needed, plus pdgId and eta for filtering
    branches_to_read = list(track_branch_map.keys()) + ['Track_pdgId']
    if 'Track_eta' not in branches_to_read:
        branches_to_read.append('Track_eta')
    if track_selection == 'from_pv' and FROM_PV_BRANCH not in branches_to_read:
        branches_to_read.append(FROM_PV_BRANCH)
    track_data = tree.arrays(
        expressions=branches_to_read,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    # ---- Event cleaning ----
    # Drop events with data quality issues that would corrupt training.
    clean_event_mask = np.ones(len(track_data), dtype=bool)
    drop_reasons = {}

    # 1. Corrupt eta: |eta| > 5 or NaN (CMS tracking covers |eta| < 2.5)
    eta = track_data['Track_eta']
    corrupt_eta = ak.to_numpy(ak.any((np.abs(eta) > 5.0) | np.isnan(eta), axis=1))
    if corrupt_eta.any():
        drop_reasons['corrupt_eta'] = int(corrupt_eta.sum())
        clean_event_mask &= ~corrupt_eta

    # 2. Unphysical phi: |phi| > pi
    if 'Track_phi' in track_data.fields:
        phi = track_data['Track_phi']
        corrupt_phi = ak.to_numpy(ak.any(np.abs(phi) > (np.pi + 0.01), axis=1))
        if corrupt_phi.any():
            drop_reasons['corrupt_phi'] = int(corrupt_phi.sum())
            clean_event_mask &= ~corrupt_phi

    # 3. Unphysical pT: <= 0, > 500 GeV, or float16 max (65504)
    pt = track_data['Track_pt']
    bad_pt = ak.to_numpy(ak.any((pt <= 0) | (pt > 500) | (pt >= 65504), axis=1))
    if bad_pt.any():
        drop_reasons['corrupt_pt'] = int(bad_pt.sum())
        clean_event_mask &= ~bad_pt

    # 4. Invalid charge: not +/-1
    charge = track_data['Track_charge']
    bad_charge = ak.to_numpy(ak.any((charge != 1) & (charge != -1), axis=1))
    if bad_charge.any():
        drop_reasons['invalid_charge'] = int(bad_charge.sum())
        clean_event_mask &= ~bad_charge

    # 5. NaN, Inf, or extreme values (|val| > 1e10) — scanned ONLY over the
    # frozen legacy branch set so requesting extra columns can never change
    # which events survive (see LEGACY_CLEAN_BRANCHES).
    for root_branch_name in [b for b in track_branch_map
                             if b in LEGACY_CLEAN_BRANCHES]:
        branch_data = track_data[root_branch_name]
        has_nan = ak.any(np.isnan(branch_data), axis=1)
        has_inf = ak.any(np.isinf(branch_data), axis=1)
        has_extreme = ak.any(np.abs(branch_data) > 1e10, axis=1)
        artifact_mask = ak.to_numpy(has_nan | has_inf | has_extreme)
        new_drops = int(np.sum(artifact_mask & clean_event_mask))
        if new_drops > 0:
            drop_reasons[f'artifacts_{root_branch_name}'] = new_drops
            clean_event_mask &= ~artifact_mask

    num_dropped = int(np.sum(~clean_event_mask))
    if num_dropped > 0 and os.environ.get('CONVERT_VERBOSE'):
        print(f'    Dropping {num_dropped} event(s):')
        for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f'      {reason}: {count}')

    track_data = track_data[clean_event_mask]

    # Filter: keep only charged pions where |pdgId| == 211
    pion_mask = abs(track_data['Track_pdgId']) == PION_PDG_ID
    if track_selection == 'from_pv':
        pion_mask = pion_mask & (track_data[FROM_PV_BRANCH] > 0)

    filtered_data = {}
    for root_branch_name in track_branch_map:
        filtered_data[root_branch_name] = track_data[root_branch_name][pion_mask]
    # Raw Track-table position of every kept track (the index space of the
    # builder / ABCNet / K* / GenPart track references), carried through the
    # same masks as the features.
    filtered_data[TRACK_ROOT_INDEX_KEY] = ak.local_index(
        track_data['Track_pdgId'], axis=1)[pion_mask]

    return filtered_data, clean_event_mask


def read_event_info(tree, clean_event_mask, entry_start=None, entry_stop=None,
                    filepath=None):
    """Read event-level scalar branches (primary vertex, CMS ids) and the
    source ids, filtered to clean events.

    Source ids (event_identity.source_ids): copied from the
    source_batch_id / source_microbatch_id branches when present; for
    abcbatch job files without them, derived from the file name (batch,
    job) and the ROOT entry index; -1 otherwise (also when filepath is
    None). The decoded job number and entry index are returned as
    event_source_job / event_source_entry.

    Args:
        tree: An uproot TTree object.
        clean_event_mask: Boolean numpy array from read_and_filter_pion_tracks
            (True = event is clean, False = event dropped).
        entry_start: Optional first entry index to read.
        entry_stop: Optional last entry index (exclusive) to read.
        filepath: Path of the ROOT file (needed for file-name derived ids).

    Returns:
        A dict mapping branch names (plus the two event_source_* columns)
        to numpy arrays (one value per clean event).
    """
    available = set(tree.keys())
    scalar_branches = [b for b in EVENT_BRANCHES
                       if b not in event_identity.SOURCE_ID_BRANCHES]
    branches_to_read = [b for b in scalar_branches if b in available]
    event_data = tree.arrays(
        expressions=branches_to_read,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    result = {}
    n_clean = int(clean_event_mask.sum())
    for branch in scalar_branches:
        if branch in available:
            result[branch] = ak.to_numpy(event_data[branch])[clean_event_mask]
        else:
            result[branch] = np.full(n_clean, -1, dtype=np.int32)
    ids = event_identity.source_ids(tree, filepath or '', entry_start, entry_stop)
    result['source_batch_id'] = ids.batch[clean_event_mask]
    result['source_microbatch_id'] = ids.microbatch[clean_event_mask]
    result['event_source_job'] = ids.job[clean_event_mask]
    result['event_source_entry'] = ids.entry[clean_event_mask]
    return result


# ---------------------------------------------------------------------------
# Extended (H6) blocks: sanitization + collection readers
# ---------------------------------------------------------------------------

def sanitize_jagged(array):
    """array: jagged awkward float array. Returns a float32 copy with
    NaN/Inf and |value| > 1e10 replaced by 0.0."""
    cleaned = ak.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = ak.where(abs(cleaned) > 1e10, 0.0, cleaned)
    return ak.values_astype(cleaned, np.float32)


def sanitize_float_block(block, integer_columns):
    """block: dict of jagged awkward arrays. Sanitizes every non-integer
    column; integer columns pass through unchanged."""
    return {
        name: array if name in integer_columns else sanitize_jagged(array)
        for name, array in block.items()
    }


def sanitize_new_track_columns(track_features):
    """track_features: renamed+cast kept-track block. Sanitizes the
    NEW_TRACK_FLOAT_COLUMNS; every other column (the 13 legacy ones and
    the integer flags) passes through untouched, preserving bit-identity
    with the legacy conversion."""
    return {
        name: sanitize_jagged(array) if name in NEW_TRACK_FLOAT_COLUMNS
        else array
        for name, array in track_features.items()
    }


def extract_other_tracks(raw_track_data, pt_cutoff):
    """raw_track_data: clean-masked jagged dict keyed by ROOT branch names
    (must contain every OTHER_TRACK_BRANCH_MAP branch). Returns the
    renamed, cast, sanitized other_track_* block for tracks with
    OTHER_TRACK_PT_FLOOR <= pT < pt_cutoff."""
    pt = raw_track_data['Track_pt']
    window_mask = (pt >= OTHER_TRACK_PT_FLOOR) & (pt < pt_cutoff)
    selected = {
        branch: raw_track_data[branch][window_mask]
        for branch in OTHER_TRACK_BRANCH_MAP
    }
    renamed = rename_branches(selected, OTHER_TRACK_BRANCH_MAP)
    casted = cast_to_float32(renamed, integer_columns=OTHER_TRACK_INTEGER_COLUMNS)
    return sanitize_float_block(casted, OTHER_TRACK_INTEGER_COLUMNS)


def read_extended_collections(tree, clean_event_mask, entry_start=None,
                              entry_stop=None):
    """Reads the Muon/SV collections and pileup-context event branches,
    masked to clean events. Missing branches raise (the extended pipeline
    requires a re-merged source with the full branch set). Returns a dict
    of output-named, cast, sanitized columns."""
    branches = (list(MUON_BRANCH_MAP) + list(SV_BRANCH_MAP)
                + EXTENDED_EVENT_BRANCHES)
    data = tree.arrays(
        expressions=branches,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    result = {}
    for branch_map, integer_columns in (
        (MUON_BRANCH_MAP, MUON_INTEGER_COLUMNS),
        (SV_BRANCH_MAP, SV_INTEGER_COLUMNS),
    ):
        block = {
            output_name: data[branch][clean_event_mask]
            for branch, output_name in branch_map.items()
        }
        block = cast_to_float32(block, integer_columns=integer_columns)
        result.update(sanitize_float_block(block, integer_columns))

    result['event_other_pv_z'] = sanitize_jagged(
        ak.values_astype(data['OtherPV_z'][clean_event_mask], np.float32)
    )
    result['event_n_pvs'] = \
        ak.to_numpy(data['PV_npvs'])[clean_event_mask].astype(np.int32)
    result['event_n_pvs_good'] = \
        ak.to_numpy(data['PV_npvsGood'])[clean_event_mask].astype(np.int32)
    return result


def resolve_optional_branches(tree):
    """tree: uproot Events tree (the first input file of a run). Returns
    {block name: branch map} for every OPTIONAL_BLOCKS block whose branches
    are all present; partially present blocks are skipped with a note."""
    available = set(tree.keys())
    resolved = {}
    for name, (branch_map, _) in OPTIONAL_BLOCKS.items():
        present = [branch for branch in branch_map if branch in available]
        if len(present) == len(branch_map):
            resolved[name] = dict(branch_map)
        elif present:
            print(f'  optional block {name} skipped: missing '
                  f'{sorted(set(branch_map) - set(present))}')
    return resolved


def check_optional_branches(tree, optional_blocks, basename):
    """tree: uproot Events tree; optional_blocks: resolve_optional_branches
    output. Raises ValueError when a resolved branch is absent (the schema
    is fixed per run from the first file)."""
    available = set(tree.keys())
    missing = [branch for branch_map in optional_blocks.values()
               for branch in branch_map if branch not in available]
    if missing:
        raise ValueError(f'{basename} lacks optional branches resolved from the first '
                         f'file: {missing}')


def read_optional_collections(tree, clean_event_mask, entry_start, entry_stop,
                              optional_blocks):
    """Reads the resolved event-level optional blocks (everything but the
    per-track blocks, which ride the track branch map), masked to clean
    events. Returns a dict of output-named, cast, sanitized columns; the
    selected-K* pair becomes two int32 scalars (-1 without a pair)."""
    event_blocks = [name for name in optional_blocks if name not in PER_TRACK_OPTIONAL_BLOCKS]
    if not event_blocks:
        return {}
    branches = [branch for name in event_blocks for branch in optional_blocks[name]]
    data = tree.arrays(
        expressions=branches,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    result = {}
    for name in event_blocks:
        branch_map = optional_blocks[name]
        integer_columns = OPTIONAL_BLOCKS[name][1]
        if name in EVENT_SCALAR_OPTIONAL_BLOCKS:
            for branch, column in branch_map.items():
                result[column] = ak.to_numpy(data[branch])[clean_event_mask].astype(
                    np.int32 if column in integer_columns else np.float32)
        elif name == 'kstar_selected':
            for branch, column in branch_map.items():
                firsts = ak.fill_none(ak.firsts(data[branch][clean_event_mask]), -1)
                result[column] = ak.to_numpy(firsts).astype(np.int32)
        else:
            block = {column: data[branch][clean_event_mask]
                     for branch, column in branch_map.items()}
            block = cast_to_float32(block, integer_columns=integer_columns)
            result.update(sanitize_float_block(block, integer_columns))
    return result


def reduce_kstar_pairs(track1_index, track2_index, pair_pt):
    """track1_index, track2_index, pair_pt: jagged awkward arrays of the
    per-event K* pair table (any row count). Returns the
    KSTAR_SCALAR_COLUMNS dict of int32 numpy arrays: row count, the first
    row's track indices and the highest-pT row's track indices (-1 for
    events without a pair)."""
    leading_row = ak.argmax(pair_pt, axis=1, keepdims=True)

    def first_or_minus_one(array):
        return ak.to_numpy(ak.fill_none(ak.firsts(array), -1)).astype(np.int32)

    return {
        'event_n_kstar': ak.to_numpy(ak.num(track1_index, axis=1)).astype(np.int32),
        'event_kstar_front_track1_index': first_or_minus_one(track1_index),
        'event_kstar_front_track2_index': first_or_minus_one(track2_index),
        'event_kstar_leading_track1_index': first_or_minus_one(track1_index[leading_row]),
        'event_kstar_leading_track2_index': first_or_minus_one(track2_index[leading_row]),
    }


def read_builder_blocks(tree, clean_event_mask, entry_start=None,
                        entry_stop=None):
    """Reads the TauTo3PiBuilder emulation inputs (K* pair table reduced
    to per-event scalars, the builder's tau output block, beam spot),
    masked to clean events. Missing branches raise. Returns a dict of
    output-named, cast, sanitized columns."""
    branches = (KSTAR_BRANCHES + list(TAU_BUILDER_BRANCH_MAP)
                + list(BEAM_SPOT_BRANCH_MAP))
    data = tree.arrays(
        expressions=branches,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    result = reduce_kstar_pairs(
        *(data[branch][clean_event_mask] for branch in KSTAR_BRANCHES)
    )
    tau_block = {
        output_name: data[branch][clean_event_mask]
        for branch, output_name in TAU_BUILDER_BRANCH_MAP.items()
    }
    tau_block = cast_to_float32(tau_block, integer_columns=TAU_BUILDER_INTEGER_COLUMNS)
    result.update(sanitize_float_block(tau_block, TAU_BUILDER_INTEGER_COLUMNS))
    for branch, output_name in BEAM_SPOT_BRANCH_MAP.items():
        result[output_name] = \
            ak.to_numpy(data[branch])[clean_event_mask].astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Track filtering
# ---------------------------------------------------------------------------

def filter_tracks_by_pt(track_data, pt_key, pt_cutoff):
    """Remove pion tracks with transverse momentum below cutoff.

    Applies a per-track boolean mask: tracks with pT < pt_cutoff are dropped.
    All branches in track_data are filtered consistently using the same mask.

    Args:
        track_data: Dict mapping ROOT branch names to jagged awkward arrays.
        pt_key: Key for the pT branch (e.g. 'Track_pt').
        pt_cutoff: Minimum pT threshold in GeV (inclusive: pT >= cutoff kept).

    Returns:
        Dict with same keys, low-pT tracks removed from each event.
    """
    # Boolean mask: True = track survives, False = track dropped
    pt_mask = track_data[pt_key] >= pt_cutoff
    return {key: array[pt_mask] for key, array in track_data.items()}


def filter_events_by_gt_pion_count(track_data, label_key, required_gt_pions):
    """Keep only events with exactly the required number of tau-origin (GT) pions.

    Counts tracks with label == 1 per event. Events where this count does not
    equal required_gt_pions are dropped.

    Args:
        track_data: Dict mapping branch names to jagged awkward arrays.
        label_key: Key for the track label branch (1 = GT pion, 0 = background).
        required_gt_pions: Exact number of GT pions required per event.

    Returns:
        Tuple of (filtered_track_data, event_mask) where:
          - filtered_track_data: dict with only qualifying events.
          - event_mask: boolean numpy array (True = event kept).
    """
    labels = track_data[label_key]
    # Sum of (label == 1) per event gives GT pion count
    gt_count_per_event = ak.sum(labels == 1, axis=1)
    event_mask = ak.to_numpy(gt_count_per_event == required_gt_pions)
    filtered = {key: array[event_mask] for key, array in track_data.items()}
    return filtered, event_mask


# ---------------------------------------------------------------------------
# Branch renaming and type casting
# ---------------------------------------------------------------------------

def rename_branches(data_dict, branch_map):
    """Rename dictionary keys from ROOT branch names to output column names.

    Args:
        data_dict: Dict mapping ROOT branch names to arrays.
        branch_map: Dict mapping ROOT names to desired output names.

    Returns:
        A new dict with renamed keys.
    """
    return {branch_map[root_name]: array for root_name, array in data_dict.items()}


def cast_to_float32(data_dict, integer_columns=None):
    """Cast all arrays to float32, except specified integer columns which use int32.

    Args:
        data_dict: Dict mapping column names to awkward or numpy arrays.
        integer_columns: Set of column names that should remain int32.

    Returns:
        A new dict with consistently typed arrays.
    """
    if integer_columns is None:
        integer_columns = set()

    casted = {}
    for column_name, array in data_dict.items():
        if column_name in integer_columns:
            if isinstance(array, ak.Array):
                casted[column_name] = ak.values_astype(array, np.int32)
            else:
                casted[column_name] = array.astype(np.int32)
        else:
            if isinstance(array, ak.Array):
                casted[column_name] = ak.values_astype(array, np.float32)
            else:
                casted[column_name] = array.astype(np.float32)

    return casted


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output_table(track_features, event_info, pion_counts,
                       extra_columns=None):
    """Combine track features and event-level info into output dict.

    Args:
        track_features: Dict of renamed, typed track columns (jagged arrays).
        event_info: Dict of event-level scalars (numpy arrays).
        pion_counts: Numpy array with number of pion tracks per event.
        extra_columns: Optional dict of already-named extended columns
            (muon_*/sv_*/other_track_*/event_*) appended verbatim.

    Returns:
        A dict ready to be wrapped in ak.Array and written to Parquet.
    """
    output = {}

    # Event-level scalars
    output['event_primary_vertex_x'] = event_info['PV_x'].astype(np.float32)
    output['event_primary_vertex_y'] = event_info['PV_y'].astype(np.float32)
    output['event_primary_vertex_z'] = event_info['PV_z'].astype(np.float32)
    output['event_n_tracks'] = pion_counts.astype(np.int32)

    # CMS event identifiers — for tracing events back to ROOT source.
    # (run, luminosityBlock, event) is NOT unique per row because the
    # upstream CMSSW analysis writes one entry per tau candidate.
    # source_batch_id + source_microbatch_id disambiguate: in the merged
    # production they are the merge batch / microbatch file (added by
    # merge_batches); in the abcbatch production they are the batch number
    # and job * ENTRIES_PER_JOB_LIMIT + entry index (event_identity).
    # event_source_job / event_source_entry carry the decoded values.
    output['event_run'] = event_info['run'].astype(np.int32)
    output['event_id'] = event_info['event'].astype(np.int64)
    output['event_luminosity_block'] = event_info['luminosityBlock'].astype(np.int32)
    output['source_batch_id'] = event_info['source_batch_id'].astype(np.int32)
    output['source_microbatch_id'] = event_info['source_microbatch_id'].astype(np.int32)
    output['event_source_job'] = event_info['event_source_job'].astype(np.int32)
    output['event_source_entry'] = event_info['event_source_entry'].astype(np.int32)

    # Track (pion) features and per-track label — jagged arrays
    output.update(track_features)

    if extra_columns:
        output.update(extra_columns)

    return output


def build_track_branch_map(output_columns=None):
    """Build a ROOT-to-output branch map for the requested output columns.

    Args:
        output_columns: List of output column names to include, or None
            for DEFAULT_COLUMNS.

    Returns:
        Dict mapping ROOT branch names to output column names.

    Raises:
        ValueError: If a requested column is not in TRACK_BRANCH_MAP_ALL.
    """
    if output_columns is None:
        output_columns = DEFAULT_COLUMNS

    reverse_map = {v: k for k, v in TRACK_BRANCH_MAP_ALL.items()}

    track_branch_map = {}
    for output_name in output_columns:
        if output_name not in reverse_map:
            available = sorted(TRACK_BRANCH_MAP_ALL.values())
            raise ValueError(
                f'Unknown output column: "{output_name}". '
                f'Available columns: {available}'
            )
        track_branch_map[reverse_map[output_name]] = output_name

    return track_branch_map


# ---------------------------------------------------------------------------
# Sharded Parquet output
# ---------------------------------------------------------------------------

def write_parquet_shard(output_array, filepath):
    """Write an awkward array to a single Parquet file with LZ4 compression.

    Creates parent directories if they don't exist.

    Args:
        output_array: Awkward array to write.
        filepath: Output file path.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ak.to_parquet(output_array, filepath, compression='LZ4', compression_level=4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_builder_blocks(data, basename):
    """data: one shard's awkward array. Checks the --builder-blocks
    columns: presence, tau-block length consistency, integer dtypes,
    K* scalars (-1 indices exactly for events without a pair) and
    finiteness of the float columns.

    Raises:
        ValueError: If any builder-block constraint is violated.
    """
    expected_columns = (
        set(KSTAR_SCALAR_COLUMNS)
        | set(TAU_BUILDER_BRANCH_MAP.values())
        | set(BEAM_SPOT_BRANCH_MAP.values())
    )
    missing = expected_columns - set(data.fields)
    if missing:
        raise ValueError(f'Missing builder-block columns in {basename}: '
                         f'{sorted(missing)}')

    tau_counts = ak.num(data['tau_track1_index'], axis=1)
    for column in TAU_BUILDER_BRANCH_MAP.values():
        if ak.any(ak.num(data[column], axis=1) != tau_counts):
            raise ValueError(f'Tau block length mismatch in {basename}: '
                             f'{column}')

    for column in TAU_BUILDER_INTEGER_COLUMNS:
        if ak.to_numpy(ak.flatten(data[column])).dtype.kind != 'i':
            raise ValueError(f'Wrong dtype for {column} in {basename}: '
                             f'expected integer')

    n_kstar = ak.to_numpy(data['event_n_kstar'])
    for column in KSTAR_SCALAR_COLUMNS:
        values = ak.to_numpy(data[column])
        if values.dtype.kind != 'i':
            raise ValueError(f'Wrong dtype for {column} in {basename}: '
                             f'expected integer')
        if column != 'event_n_kstar' and ((values < 0) != (n_kstar == 0)).any():
            raise ValueError(f'K* index/count mismatch in {basename}: '
                             f'{column}')

    float_columns = (
        sorted(set(TAU_BUILDER_BRANCH_MAP.values()) - TAU_BUILDER_INTEGER_COLUMNS)
        + list(BEAM_SPOT_BRANCH_MAP.values())
    )
    for column in float_columns:
        flat = ak.to_numpy(ak.ravel(data[column]))
        if not np.isfinite(flat).all():
            raise ValueError(f'Non-finite values in {column} of {basename} '
                             f'(sanitization failed)')


def validate_extended_blocks(data, filepath, pt_cutoff, builder_blocks=False):
    """data: one shard's awkward array. Checks the extended-mode blocks:
    column presence, other_track pT window, muon length consistency,
    integer flag dtypes, and finiteness of sanitized float columns; with
    builder_blocks=True also the --builder-blocks columns.

    Raises:
        ValueError: If any extended-block constraint is violated.
    """
    basename = os.path.basename(filepath)
    expected_columns = (
        set(EXTENDED_TRACK_COLUMNS)
        | set(MUON_BRANCH_MAP.values())
        | set(SV_BRANCH_MAP.values())
        | set(OTHER_TRACK_BRANCH_MAP.values())
        | {'event_other_pv_z', 'event_n_pvs', 'event_n_pvs_good'}
    )
    missing = expected_columns - set(data.fields)
    if missing:
        raise ValueError(f'Missing extended columns in {basename}: '
                         f'{sorted(missing)}')

    other_pt = ak.to_numpy(ak.flatten(data['other_track_pt']))
    if other_pt.size and (
        float(other_pt.min()) < OTHER_TRACK_PT_FLOOR
        or float(other_pt.max()) >= pt_cutoff
    ):
        raise ValueError(
            f'other_track pT window violation in {basename}: '
            f'range [{other_pt.min():.4f}, {other_pt.max():.4f}] outside '
            f'[{OTHER_TRACK_PT_FLOOR}, {pt_cutoff})'
        )

    muon_counts = ak.num(data['muon_pt'], axis=1)
    for column in MUON_BRANCH_MAP.values():
        if ak.any(ak.num(data[column], axis=1) != muon_counts):
            raise ValueError(f'Muon block length mismatch in {basename}: '
                             f'{column}')

    integer_flag_columns = (
        {'track_is_lost', 'track_is_matched_to_muon',
         'track_is_matched_to_ele', 'track_label_from_b'}
        | MUON_INTEGER_COLUMNS | SV_INTEGER_COLUMNS
        | OTHER_TRACK_INTEGER_COLUMNS
    )
    for column in integer_flag_columns:
        if ak.to_numpy(ak.flatten(data[column])).dtype.kind != 'i':
            raise ValueError(f'Wrong dtype for {column} in {basename}: '
                             f'expected integer')

    for column in ['track_dxy', 'track_vertex_x', 'track_covariance_dxy_dxy',
                   'muon_pt', 'muon_vertex_x', 'sv_x', 'other_track_pt',
                   'event_other_pv_z']:
        flat = ak.to_numpy(ak.flatten(data[column]))
        if not np.isfinite(flat).all():
            raise ValueError(f'Non-finite values in {column} of {basename} '
                             f'(sanitization failed)')

    if builder_blocks:
        validate_builder_blocks(data, basename)


def _dtype_kind(array):
    values = ak.flatten(array) if array.ndim > 1 else array
    return ak.to_numpy(values).dtype.kind


def validate_optional_blocks(data, basename):
    """data: one shard's awkward array. For every optional block present in
    the shard: all its columns present, integer dtypes, and equal list
    lengths (per-track columns vs track_pt, muon truth vs muon_pt,
    OtherPV score vs event_other_pv_z, PVtx / GenPart / K* blocks within
    themselves). Blocks absent from the shard are skipped.

    Raises:
        ValueError: If any constraint is violated.
    """
    fields = set(data.fields)
    length_reference = OPTIONAL_BLOCK_LENGTH_REFERENCE
    # the beam-spot columns are also written by --builder-blocks: alone they
    # do not make an optional block present
    shared_columns = set(BEAM_SPOT_BRANCH_MAP.values())
    for name, (branch_map, integer_columns) in OPTIONAL_BLOCKS.items():
        columns = list(branch_map.values())
        present = [column for column in columns if column in fields]
        if not set(present) - shared_columns:
            continue
        if len(present) != len(columns):
            raise ValueError(f'optional block {name} is partial in {basename}: missing '
                             f'{sorted(set(columns) - set(present))}')
        for column in integer_columns:
            if _dtype_kind(data[column]) != 'i':
                raise ValueError(f'{column} in {basename} is not integer')
        if name in EVENT_SCALAR_OPTIONAL_BLOCKS or name == 'kstar_selected':
            continue
        reference = data[length_reference.get(name, columns[0])]
        for column in columns:
            if not ak.all(ak.num(data[column], axis=1) == ak.num(reference, axis=1)):
                raise ValueError(f'optional block {name}: {column} length differs from '
                                 f'its reference in {basename}')


def validate_track_root_index(data, basename):
    """data: one shard's awkward array. Checks the track_root_index column:
    present, integer, one entry per kept track, non-negative and strictly
    increasing within each event (kept tracks keep the raw table order).

    Raises:
        ValueError: If any constraint is violated.
    """
    if TRACK_ROOT_INDEX_COLUMN not in data.fields:
        raise ValueError(f'Missing column {TRACK_ROOT_INDEX_COLUMN} in {basename}')
    root_index = data[TRACK_ROOT_INDEX_COLUMN]
    if ak.to_numpy(ak.flatten(root_index)).dtype.kind != 'i':
        raise ValueError(f'{TRACK_ROOT_INDEX_COLUMN} in {basename} is not integer')
    if not ak.all(ak.num(root_index, axis=1) == ak.num(data['track_pt'], axis=1)):
        raise ValueError(f'{TRACK_ROOT_INDEX_COLUMN} length differs from track_pt in {basename}')
    flat = ak.to_numpy(ak.flatten(root_index))
    if flat.size and flat.min() < 0:
        raise ValueError(f'{TRACK_ROOT_INDEX_COLUMN} has negative entries in {basename}')
    steps = root_index[:, 1:] - root_index[:, :-1]
    if not ak.all(ak.flatten(steps) > 0):
        raise ValueError(f'{TRACK_ROOT_INDEX_COLUMN} is not strictly increasing in {basename}')


def validate_identity_unique(parquet_files):
    """parquet_files: list of shard paths. Checks that the 5-key
    (event_run, event_id, event_luminosity_block, source_batch_id,
    source_microbatch_id) is unique across all of them: a duplicate among
    rows with file-derived or copied source ids raises, duplicates among
    legacy -1 rows only warn (event_identity.assert_unique_identity).

    Raises:
        ValueError: On a duplicate key among rows with real source ids.
    """
    names = list(event_identity.IDENTITY_COLUMNS + event_identity.SOURCE_DETAIL_COLUMNS)
    collected = {name: [] for name in names}
    for path in parquet_files:
        data = ak.from_parquet(path)
        for name in names:
            if name in data.fields:
                collected[name].append(ak.to_numpy(data[name]))
    columns = {name: np.concatenate(arrays) for name, arrays in collected.items() if arrays}
    context = f'{len(parquet_files)} shard(s) under {os.path.dirname(parquet_files[0])}'
    event_identity.assert_unique_identity(columns, context=context)


def validate_parquet_output(output_dir, pt_cutoff, required_gt_pions,
                            extended=False, builder_blocks=False,
                            shard_glob='*.parquet'):
    """Validate the Parquet files of a directory (those matching shard_glob)
    against quality constraints.

    Reads each file and checks:
      1. Every pion track has pT >= pt_cutoff.
      2. Every event has exactly required_gt_pions tracks with label == 1.
      3. With extended=True: the extended blocks (see
         validate_extended_blocks); with builder_blocks=True also the
         --builder-blocks columns.

    Args:
        output_dir: Directory containing .parquet files.
        pt_cutoff: Minimum pT in GeV.
        required_gt_pions: Required GT pion count per event.
        extended: Whether to also validate the extended-mode blocks.
        builder_blocks: Whether to also validate the builder blocks.
        shard_glob: File-name pattern of the shards to validate (a run
            that shares its split directory with other jobs validates
            only its own prefix).

    Raises:
        FileNotFoundError: If no parquet files are found.
        ValueError: If any constraint is violated.
    """
    parquet_files = sorted(glob.glob(os.path.join(output_dir, shard_glob)))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {output_dir}')

    total_events = 0
    total_tracks = 0

    for filepath in parquet_files:
        data = ak.from_parquet(filepath)
        n_events = len(data)
        total_events += n_events

        # Validate pT >= cutoff for ALL pion tracks
        pt = data['track_pt']
        flat_pt = ak.to_numpy(ak.flatten(pt))
        n_tracks = len(flat_pt)
        total_tracks += n_tracks
        min_pt = float(flat_pt.min())

        if min_pt < pt_cutoff:
            raise ValueError(
                f'pT violation in {os.path.basename(filepath)}: '
                f'found track with pT={min_pt:.6f} < cutoff {pt_cutoff}'
            )

        # Validate exactly required_gt_pions GT pions per event
        labels = data['track_label_from_tau']
        gt_counts = ak.to_numpy(ak.sum(labels == 1, axis=1))
        bad_mask = gt_counts != required_gt_pions
        n_bad = int(bad_mask.sum())

        if n_bad > 0:
            bad_counts = sorted(set(gt_counts[bad_mask].tolist()))
            raise ValueError(
                f'GT pion count violation in {os.path.basename(filepath)}: '
                f'{n_bad} events have wrong count. '
                f'Found: {bad_counts}, required: {required_gt_pions}'
            )

        # Validate CMS event identifiers and source IDs exist with correct types
        for id_col, expected_kind in [
            ('event_run', 'i'), ('event_id', 'i'), ('event_luminosity_block', 'i'),
            ('source_batch_id', 'i'), ('source_microbatch_id', 'i'),
        ]:
            if id_col not in data.fields:
                raise ValueError(
                    f'Missing column {id_col} in {os.path.basename(filepath)}'
                )
            id_values = ak.to_numpy(data[id_col])
            if id_values.dtype.kind != expected_kind:
                raise ValueError(
                    f'Wrong dtype for {id_col} in {os.path.basename(filepath)}: '
                    f'{id_values.dtype} (expected integer)'
                )

        validate_track_root_index(data, os.path.basename(filepath))

        if extended:
            validate_extended_blocks(data, filepath, pt_cutoff,
                                     builder_blocks=builder_blocks)
            validate_optional_blocks(data, os.path.basename(filepath))

        print(f'    OK {os.path.basename(filepath)}: '
              f'{n_events} events, {n_tracks} tracks, min pT={min_pt:.4f}')

    print(f'    Total: {total_events} events, {total_tracks} tracks — all valid')


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def print_progress(total_scanned, total_qualifying, total_needed,
                   current_shard_index, total_shards, start_time):
    """Print a single-line progress indicator that updates in place.

    Shows: events scanned, qualifying collected / needed, acceptance rate,
    shards written, and elapsed time.
    """
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if total_scanned > 0:
        acceptance_percent = 100.0 * total_qualifying / total_scanned
    else:
        acceptance_percent = 0.0

    qualifying = (f'Qualifying {total_qualifying:,}' if total_needed is None
                  else f'Qualifying {total_qualifying:,}/{total_needed:,}')
    shards = (f'Shards {current_shard_index}' if total_shards is None
              else f'Shards {current_shard_index}/{total_shards}')
    line = (
        f'  [{minutes:02d}:{seconds:02d}] '
        f'Scanned {total_scanned:,} | '
        f'{qualifying} '
        f'({acceptance_percent:.1f}% accept) | '
        f'{shards}'
    )

    # Overwrite previous line; pad with spaces to clear leftover characters
    sys.stdout.write(f'\r{line:<100}')
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(input_dir, output_dir, train_events, val_events,
            train_files, val_files, pt_cutoff, required_gt_pions,
            chunk_size, output_columns, extended=False,
            pattern='merged_noBKstar_batch*.root',
            train_subdir='train', val_subdir='val', builder_blocks=False,
            eval_only=False, shard_events=5000, track_selection='all',
            shard_prefix=None, max_unreadable_fraction=MAXIMUM_UNREADABLE_FRACTION):
    """Convert ROOT files to sharded Parquet with pT cutoff and GT pion filter.

    Processes ROOT files sequentially in chunks, applying quality cleaning,
    pion selection, pT cutoff, and GT pion count filtering. Output is split
    into train/val sharded Parquet files. All output is validated after writing.

    Memory is bounded: events are accumulated in a buffer and flushed to disk
    as soon as a shard's worth of events is collected.

    Args:
        input_dir: Directory containing the ROOT files, or a list of
            directories (e.g. several batch directories of one production).
        output_dir: Base output directory (creates train/ and val/ subdirs).
        train_events: Total number of training events to collect.
        val_events: Total number of validation events to collect.
        train_files: Number of train parquet shards.
        val_files: Number of val parquet shards.
        pt_cutoff: Minimum pT in GeV for all pion tracks.
        required_gt_pions: Exact number of GT pions required per event.
        chunk_size: Events per ROOT reading chunk.
        output_columns: List of output column names, None for defaults, 'all' for all.
        extended: Extract the H6 extended schema (EXTENDED_TRACK_COLUMNS +
            muon/SV/other_track/pileup blocks). Mutually exclusive with
            output_columns.
        pattern: Glob pattern for input ROOT files.
        train_subdir: Subdirectory (and shard-name prefix) for the train split.
        val_subdir: Subdirectory (and shard-name prefix) for the val split.
        builder_blocks: With extended=True, also extract the TauTo3PiBuilder
            emulation inputs (K* veto scalars, builder tau output, beam spot).
        eval_only: Write EVERY qualifying event into the val split as
            shards of shard_events events (the last one partial), with no
            event quota; train_events/val_events/train_files/val_files are
            ignored. Used for the abcbatch evaluation conversion.
        shard_events: Events per shard in eval_only mode.
        track_selection: 'all' (every pion track) or 'from_pv' (only
            tracks with Track_fromPV > 0; MiniAOD input).
        shard_prefix: eval_only shard-name prefix (default: val_subdir).
            When given, only shards with this prefix are validated, so
            several jobs can fill one split directory.
        max_unreadable_fraction: A MiniAOD run skips and reports input
            files that cannot be read (never the first one) and raises
            RuntimeError after writing when more than this fraction of
            the files was skipped. Ntuple runs fail on the first
            unreadable file.
    """
    # --- Validate configuration ---
    if builder_blocks and not extended:
        raise ValueError('builder_blocks=True requires extended=True')
    if eval_only:
        if shard_events <= 0:
            raise ValueError('shard_events must be positive in eval_only mode')
        print('eval-only mode: every qualifying event is written to '
              f'{val_subdir}/ in shards of {shard_events:,}; the train/val '
              'quota arguments are ignored')
    else:
        if train_events % train_files != 0:
            raise ValueError(
                f'train_events ({train_events}) must be evenly divisible by '
                f'train_files ({train_files})'
            )
        if val_events % val_files != 0:
            raise ValueError(
                f'val_events ({val_events}) must be evenly divisible by '
                f'val_files ({val_files})'
            )

    train_dir = os.path.join(output_dir, train_subdir)
    val_dir = os.path.join(output_dir, val_subdir)

    # Shard plan: ordered list of (directory, filename, events_per_shard);
    # train shards are filled first, then val shards. In eval_only mode the
    # plan is open-ended: shard i of the val split holds shard_events events.
    if eval_only:
        shard_plan = None
        total_needed = None
    else:
        train_events_per_file = train_events // train_files
        val_events_per_file = val_events // val_files
        total_needed = train_events + val_events
        shard_plan = []
        for i in range(train_files):
            shard_plan.append(
                (train_dir, f'{train_subdir}_{i:03d}.parquet', train_events_per_file))
        for i in range(val_files):
            shard_plan.append(
                (val_dir, f'{val_subdir}_{i:03d}.parquet', val_events_per_file))
    total_shards = None if shard_plan is None else len(shard_plan)

    def plan_complete():
        return shard_plan is not None and current_shard_index >= len(shard_plan)

    def shard_spec(index):
        if shard_plan is None:
            return (val_dir, f'{shard_prefix or val_subdir}_{index:03d}.parquet', shard_events)
        return shard_plan[index]

    # Build branch map
    if extended:
        if output_columns is not None:
            raise ValueError('extended=True fixes the column set; '
                             'do not pass output_columns')
        track_branch_map = build_track_branch_map(EXTENDED_TRACK_COLUMNS)
    elif output_columns == 'all':
        track_branch_map = dict(TRACK_BRANCH_MAP_ALL)
    else:
        track_branch_map = build_track_branch_map(output_columns)

    # Discover input files
    root_files = discover_root_files(input_dir, pattern=pattern)
    event_identity.assert_unique_file_keys(root_files)

    # Optional truth / vertex / K* blocks: resolved once from the first file
    # (extended mode only), then required in every file of the run.
    first_tree = load_root_tree(root_files[0])
    miniaod_input = isinstance(first_tree, miniaod_source.MiniAODTree)
    unreadable_files = []
    optional_blocks = {}
    if extended:
        optional_blocks = resolve_optional_branches(first_tree)
        for name in PER_TRACK_OPTIONAL_BLOCKS:
            track_branch_map = {**track_branch_map, **optional_blocks.get(name, {})}
        print(f'Optional blocks: {sorted(optional_blocks) or "none"}')
    print(f'Input format: {"MiniAOD" if miniaod_input else "ntuple"} | track selection: {track_selection}')

    print(f'Found {len(root_files)} ROOT files in {input_dir}')
    if not isinstance(input_dir, str):
        print(f'  from {len(input_dir)} input directories')
    print(f'Extracting {len(track_branch_map)} track columns')
    print(f'pT cutoff: >= {pt_cutoff} GeV | Required GT pions: {required_gt_pions}')
    if eval_only:
        print(f'Target: every qualifying event, {shard_events:,} per shard')
    else:
        print(f'Target: {train_events} train + {val_events} val = {total_needed} events')
    if not eval_only:
        print(f'Output: {train_files} x {train_events_per_file} train + '
              f'{val_files} x {val_events_per_file} val shards')
    print()

    # --- Process ROOT files ---
    current_shard_index = 0
    buffer = []
    buffer_count = 0
    total_scanned = 0
    total_qualifying = 0
    total_dropped_quality = 0
    total_dropped_filter = 0
    start_time = time.time()

    for file_index, root_filepath in enumerate(root_files):
        if plan_complete():
            break

        basename = os.path.basename(root_filepath)
        # Clear progress line before printing file header
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f'[{file_index + 1}/{len(root_files)}] {basename}')
        if file_index == 0:
            tree = first_tree
        else:
            try:
                tree = load_root_tree(root_filepath)
            except Exception as error:
                if not miniaod_input:
                    raise
                unreadable_files.append(root_filepath)
                print(f'  UNREADABLE, skipped: {root_filepath} ({type(error).__name__}: {error})')
                continue
        number_of_entries = tree.num_entries
        file_qualifying = 0
        identity_origin = event_identity.source_ids(
            tree, root_filepath, 0, min(1, number_of_entries)).origin
        print(f'  identity: source ids from {identity_origin}')
        if eval_only and identity_origin == 'missing':
            raise ValueError(
                f'{basename}: eval-only conversion needs source ids (the '
                f'source_batch_id/source_microbatch_id branches, or a job token '
                f'in the file name); without them the event keys would collide')
        check_optional_branches(tree, optional_blocks, basename)

        for chunk_start in range(0, number_of_entries, chunk_size):
            if plan_complete():
                break

            chunk_end = min(chunk_start + chunk_size, number_of_entries)
            chunk_events = chunk_end - chunk_start
            total_scanned += chunk_events

            # Step 1: Read pion tracks + quality cleaning
            raw_track_data, clean_event_mask = read_and_filter_pion_tracks(
                tree, track_branch_map, chunk_start, chunk_end,
                track_selection=track_selection,
            )
            event_info = read_event_info(
                tree, clean_event_mask, chunk_start, chunk_end,
                filepath=root_filepath,
            )
            if extended:
                extended_collections = read_extended_collections(
                    tree, clean_event_mask, chunk_start, chunk_end,
                )
                extended_collections.update(read_optional_collections(
                    tree, clean_event_mask, chunk_start, chunk_end, optional_blocks,
                ))
                other_track_data = extract_other_tracks(
                    raw_track_data, pt_cutoff,
                )
                if builder_blocks:
                    extended_collections.update(read_builder_blocks(
                        tree, clean_event_mask, chunk_start, chunk_end,
                    ))

            number_of_clean = int(clean_event_mask.sum())
            total_dropped_quality += (chunk_events - number_of_clean)

            if number_of_clean == 0:
                print_progress(total_scanned, total_qualifying, total_needed,
                               current_shard_index, total_shards, start_time)
                continue

            # Step 2: Apply pT cutoff — drop all pion tracks with pT < threshold
            pt_filtered_data = filter_tracks_by_pt(
                raw_track_data, 'Track_pt', pt_cutoff,
            )

            # Step 3: Keep only events with exactly required_gt_pions GT pions
            # (counted after pT cutoff, so events where a GT pion was below
            # the cutoff are automatically excluded)
            gt_filtered_data, gt_event_mask = filter_events_by_gt_pion_count(
                pt_filtered_data, 'Track_trackFromTau', required_gt_pions,
            )

            number_after_filter = int(gt_event_mask.sum())
            total_dropped_filter += (number_of_clean - number_after_filter)

            if number_after_filter == 0:
                print_progress(total_scanned, total_qualifying, total_needed,
                               current_shard_index, total_shards, start_time)
                continue

            total_qualifying += number_after_filter
            file_qualifying += number_after_filter

            # Filter event info to matching events
            event_info_filtered = {
                key: value[gt_event_mask] for key, value in event_info.items()
            }
            if extended:
                extra_columns = {
                    key: value[gt_event_mask]
                    for key, value in {**extended_collections,
                                       **other_track_data}.items()
                }
            else:
                extra_columns = None

            # Compute pion counts per event (after pT cutoff + GT filter)
            first_key = next(iter(gt_filtered_data))
            pion_counts_per_event = ak.to_numpy(
                ak.num(gt_filtered_data[first_key], axis=1)
            )

            # Rename ROOT branch names -> output column names, cast types
            track_features = rename_branches(
                gt_filtered_data,
                {**track_branch_map, TRACK_ROOT_INDEX_KEY: TRACK_ROOT_INDEX_COLUMN},
            )
            track_features = cast_to_float32(
                track_features, integer_columns=TRACK_INTEGER_COLUMNS,
            )
            track_features = sanitize_new_track_columns(track_features)

            chunk_output = build_output_table(
                track_features, event_info_filtered, pion_counts_per_event,
                extra_columns=extra_columns,
            )
            # to_packed severs all view chains: filtered/cast awkward arrays
            # are VIEWS holding the full raw chunk read (~80 chunks/shard
            # pinned = tens of GB); packing copies only the kept events.
            buffer.append(ak.to_packed(ak.Array(chunk_output)))
            buffer_count += number_after_filter

            # Flush full shards from buffer to disk
            while (not plan_complete()
                   and buffer_count >= shard_spec(current_shard_index)[2]):
                shard_dir, shard_filename, shard_n_events = shard_spec(current_shard_index)

                all_buffered = ak.concatenate(buffer)
                shard_data = all_buffered[:shard_n_events]
                remainder = all_buffered[shard_n_events:]

                shard_path = os.path.join(shard_dir, shard_filename)
                write_parquet_shard(shard_data, shard_path)
                shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
                # Clear progress line, print shard info, then resume progress
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                print(f'  -> {shard_filename} '
                      f'({shard_n_events:,} events, {shard_size_mb:.1f} MB)')

                if len(remainder) > 0:
                    # to_packed materializes the slice — without it the
                    # remainder is an offset VIEW that keeps the whole
                    # concatenated parent array (a full shard) alive
                    # until the next flush.
                    buffer = [ak.to_packed(remainder)]
                    buffer_count = len(remainder)
                else:
                    buffer = []
                    buffer_count = 0

                current_shard_index += 1

            print_progress(total_scanned, total_qualifying, total_needed,
                           current_shard_index, total_shards, start_time)

            # Free chunk memory
            del raw_track_data, pt_filtered_data, gt_filtered_data
            del event_info, event_info_filtered, track_features, chunk_output

        # Clear progress line for per-file summary
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f'  {basename}: {file_qualifying:,} qualifying / {number_of_entries:,} total '
              f'(cumulative: {total_qualifying:,})')

    # eval-only: the remaining buffered events form the last, partial shard
    if eval_only and buffer_count > 0:
        shard_dir, shard_filename, _ = shard_spec(current_shard_index)
        shard_path = os.path.join(shard_dir, shard_filename)
        write_parquet_shard(ak.concatenate(buffer), shard_path)
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f'  -> {shard_filename} ({buffer_count:,} events, final partial shard)')
        buffer = []
        buffer_count = 0
        current_shard_index += 1

    # --- Summary ---
    elapsed = time.time() - start_time
    elapsed_minutes = int(elapsed // 60)
    elapsed_seconds = int(elapsed % 60)

    print(f'\n{"=" * 50}')
    print(f'Conversion Summary')
    print(f'{"=" * 50}')
    print(f'  Elapsed time:            {elapsed_minutes:02d}:{elapsed_seconds:02d}')
    print(f'  ROOT files processed:    {len(root_files)}')
    print(f'  Events scanned:          {total_scanned:,}')
    print(f'  Dropped (quality):       {total_dropped_quality:,}')
    print(f'  Dropped (pT cutoff/GT):  {total_dropped_filter:,}')
    print(f'  Qualifying events:       {total_qualifying:,}')
    shards_written = (f'{current_shard_index}' if total_shards is None
                      else f'{current_shard_index}/{total_shards}')
    print(f'  Shards written:          {shards_written}')
    if miniaod_input:
        print(f'  Unreadable files skipped: {len(unreadable_files)}')
        for path in unreadable_files:
            print(f'    {path}')
    too_many_unreadable = len(unreadable_files) > max_unreadable_fraction * len(root_files)

    if eval_only and current_shard_index == 0:
        print(f'\n  WARNING: No qualifying events; nothing written.')
        if too_many_unreadable:
            raise RuntimeError(f'{len(unreadable_files)} of {len(root_files)} input files were unreadable')
        return
    if not eval_only and current_shard_index < len(shard_plan):
        print(f'\n  WARNING: Not enough qualifying events!')
        print(f'  Collected {total_qualifying}, needed {total_needed}.')
        print(f'  Only {current_shard_index}/{len(shard_plan)} shards written.')
        return

    # --- Validate output ---
    print(f'\n{"=" * 50}')
    print(f'Validating Output')
    print(f'{"=" * 50}')

    split_dirs = [val_dir] if eval_only else [train_dir, val_dir]
    shard_glob = f'{shard_prefix}_*.parquet' if eval_only and shard_prefix else '*.parquet'
    for split_dir in split_dirs:
        print(f'  {os.path.basename(split_dir)} ({split_dir}):')
        validate_parquet_output(split_dir, pt_cutoff, required_gt_pions,
                                extended=extended, builder_blocks=builder_blocks,
                                shard_glob=shard_glob)
    written_files = [path for split_dir in split_dirs
                     for path in sorted(glob.glob(os.path.join(split_dir, shard_glob)))]
    validate_identity_unique(written_files)
    print(f'    event keys unique across {len(written_files)} shard(s)')

    if too_many_unreadable:
        raise RuntimeError(
            f'{len(unreadable_files)} of {len(root_files)} input files were unreadable (more than '
            f'{max_unreadable_fraction:.0%}); the shards of the readable files are written and valid')
    print(f'\nDone. Output directory: {output_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    available_columns = sorted(TRACK_BRANCH_MAP_ALL.values())

    parser = argparse.ArgumentParser(
        description='Convert low-pT tau ROOT dataset to sharded Parquet with pT cutoff.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'Available track columns:\n  ' + '\n  '.join(available_columns),
    )

    # Input / output paths
    parser.add_argument(
        '--input-dir', nargs='+',
        default=['/eos/user/o/oprostak/tau_data/root1'],
        help='Directory (or several directories, e.g. the batch directories of '
             'one production) holding the ROOT files (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        default='/eos/user/o/oprostak/tau_data/parquet_low_pt_cutoff',
        help='Base output directory (creates train/ and val/ subdirs) (default: %(default)s)',
    )

    # Event counts and sharding
    parser.add_argument(
        '--train-events', type=int, default=300000,
        help='Total number of training events (default: %(default)s)',
    )
    parser.add_argument(
        '--val-events', type=int, default=75000,
        help='Total number of validation events (default: %(default)s)',
    )
    parser.add_argument(
        '--train-files', type=int, default=10,
        help='Number of train parquet shards (default: %(default)s)',
    )
    parser.add_argument(
        '--val-files', type=int, default=10,
        help='Number of val parquet shards (default: %(default)s)',
    )

    # Filtering
    parser.add_argument(
        '--pt-cutoff', type=float, default=0.5,
        help='Minimum pT in GeV for all pion tracks (default: %(default)s)',
    )
    parser.add_argument(
        '--gt-pions', type=int, default=3,
        help='Required number of GT (tau-origin) pions per event (default: %(default)s)',
    )

    # Processing
    parser.add_argument(
        '--chunk-size', type=int, default=2500,
        help='Events per ROOT reading chunk (default: %(default)s)',
    )

    # Column selection
    column_group = parser.add_mutually_exclusive_group()
    column_group.add_argument(
        '--columns', type=str, default=None,
        help=f'Comma-separated output columns (default: {",".join(DEFAULT_COLUMNS)})',
    )
    column_group.add_argument(
        '--all-columns', action='store_true',
        help='Extract all available track columns',
    )
    column_group.add_argument(
        '--extended', action='store_true',
        help='Extract the H6 extended schema: EXTENDED_TRACK_COLUMNS plus '
             'the muon/SV/other_track/pileup blocks',
    )
    parser.add_argument(
        '--builder-blocks', action='store_true',
        help='With --extended: also extract the CERN TauTo3PiBuilder '
             'emulation inputs (K* veto scalars, builder tau output, beam '
             'spot); needs a source merged with those branches',
    )

    parser.add_argument(
        '--pattern', type=str, default='merged_noBKstar_batch*.root',
        help='Glob pattern for input ROOT files (default: %(default)s)',
    )
    parser.add_argument(
        '--train-subdir', type=str, default='train',
        help='Subdirectory and shard prefix for the train split '
             '(default: %(default)s)',
    )
    parser.add_argument(
        '--val-subdir', type=str, default='val',
        help='Subdirectory and shard prefix for the val split '
             '(default: %(default)s)',
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Write every qualifying event into the val split as shards of '
             '--shard-events events, no quota (train/val counts ignored); '
             'the abcbatch evaluation conversion',
    )
    parser.add_argument(
        '--shard-events', type=int, default=5000,
        help='Events per shard with --eval-only (default: %(default)s)',
    )
    parser.add_argument(
        '--shard-prefix', type=str, default=None,
        help='Shard-name prefix with --eval-only (default: the --val-subdir '
             'name); lets several jobs fill one split directory, each '
             'validating only its own shards',
    )
    parser.add_argument(
        '--track-selection', choices=TRACK_SELECTIONS, default='all',
        help='all: every pion track; from_pv: only tracks not used in the '
             'fit of another primary vertex (Track_fromPV > 0, MiniAOD '
             'input) (default: %(default)s)',
    )

    arguments = parser.parse_args()

    # Resolve column selection
    if arguments.all_columns:
        output_columns = 'all'
    elif arguments.columns is not None:
        output_columns = [c.strip() for c in arguments.columns.split(',')]
    else:
        output_columns = None  # uses DEFAULT_COLUMNS (or the extended set)

    convert(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        train_events=arguments.train_events,
        val_events=arguments.val_events,
        train_files=arguments.train_files,
        val_files=arguments.val_files,
        pt_cutoff=arguments.pt_cutoff,
        required_gt_pions=arguments.gt_pions,
        chunk_size=arguments.chunk_size,
        output_columns=output_columns,
        extended=arguments.extended,
        pattern=arguments.pattern,
        train_subdir=arguments.train_subdir,
        val_subdir=arguments.val_subdir,
        builder_blocks=arguments.builder_blocks,
        eval_only=arguments.eval_only,
        shard_events=arguments.shard_events,
        track_selection=arguments.track_selection,
        shard_prefix=arguments.shard_prefix,
    )


if __name__ == '__main__':
    main()
