"""A MiniAOD file presented as an Events tree with the branch names of the
collaborator's ntuple, so that convert_root_to_parquet.py selects, cleans,
shards and validates it with the code it already runs on ntuples.

Conventions taken from the ntuple (measured on abcbatch0):
  Track    packed PF candidates and lost tracks with track details,
           |pdgId| = 211, high purity for packed ones, sorted by pT.
           pt / eta / phi = candidate momentum, vx / vy / vz = reference
           point, dxy / dz with respect to PV[0], dxyS = dxy / dxyError,
           dzS = dz(PV[0] z) / dzError, DCASig = |transverse distance to the
           beam spot| / dxyError.
  PVtx     PV[0] and every other good vertex (ndof > 4, |z| <= 24, rho <= 2);
           chi2 is not normalised; ntracks is 0 (MiniAOD keeps no track list).
  OtherPV  vertices 1..3.
  SV       slimmedSecondaryVertices with flight significance > 3 (NanoAOD).

Differences from the ntuple (nothing else is available without CMSSW):
  Muon     every PF muon candidate with pT >= 3, |eta| <= 2.4 instead of the
           trigger muon only; softId = 1, mediumId = -1, pfRelIso03_all = -1.
  Track    isMatchedToMuon / isMatchedToEle = 0.
  truth    labels by a tight match (miniaod_truth.match_tracks_to_generator)
           of the signal chain only: tau pions, K* daughters, tau muon;
           GenPart_genPartIdxMother is filled for that chain only.
  K*       closest-approach vertex instead of a kinematic fit (miniaod_kstar).
"""
from __future__ import annotations

import re

import awkward as ak
import numpy as np

import miniaod_geometry as geometry
import miniaod_kstar as kstar
import miniaod_reader as reader
import miniaod_truth as truth
import packed_candidate as packed

TRACK_MINIMUM_PT = 0.3
TRACK_MAXIMUM_ABS_ETA = 3.0
MUON_MINIMUM_PT = 3.0
MUON_MAXIMUM_ABS_ETA = 2.4
TRIGGER_MATCH_DELTA_R = 0.1
SV_MINIMUM_SIGNIFICANCE = 3.0
GOOD_VERTEX_MINIMUM_NDOF = 4.0
GOOD_VERTEX_MAXIMUM_ABS_Z = 24.0
GOOD_VERTEX_MAXIMUM_RHO = 2.0
OTHER_VERTEX_COUNT = 3
SUM_PT2_MAXIMUM_DZ = 0.2
PION_PDG, MUON_PDG = 211, 13
TAU_FLAVOUR, KSTAR_FLAVOUR = 15, 313
TINY = 1e-12
VERSION_SUFFIX = re.compile(r'_v\d+$')

# Single-muon parking paths; bit i of Trigger_bits / Muon_triggerBits = path i.
MUON_TRIGGER_PATHS = (
    'HLT_Mu0_Barrel', 'HLT_Mu0_Barrel_L1HP6', 'HLT_Mu0_Barrel_L1HP7', 'HLT_Mu0_Barrel_L1HP8',
    'HLT_Mu0_Barrel_L1HP9', 'HLT_Mu0_Barrel_L1HP10', 'HLT_Mu0_Barrel_L1HP11',
    'HLT_Mu0_Barrel_L1HP6_IP6', 'HLT_Mu6_Barrel_L1HP7_IP6', 'HLT_Mu7_Barrel_L1HP8_IP6',
    'HLT_Mu8_Barrel_L1HP9_IP6', 'HLT_Mu9_Barrel_L1HP10_IP6', 'HLT_Mu10_Barrel_L1HP11_IP6')
IP6_PATH_MASK = sum(1 << bit for bit, name in enumerate(MUON_TRIGGER_PATHS) if name.endswith('_IP6'))

# position of a symmetric 3x3 term in the stored lower-triangular 4x4 vertex covariance
VERTEX_COVARIANCE_POSITION = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (0, 2): 3, (1, 2): 4, (2, 2): 5}
VERTEX_COVARIANCE_BRANCHES = {'XX': (0, 0), 'XY': (0, 1), 'XZ': (0, 2), 'YY': (1, 1), 'YZ': (1, 2), 'ZZ': (2, 2)}

SCALAR_BRANCHES = (
    'run', 'luminosityBlock', 'event', 'PV_x', 'PV_y', 'PV_z', 'PV_npvs', 'PV_npvsGood', 'PV_score', 'PV_sumpt2',
    'BeamSpot_x', 'BeamSpot_y', 'BeamSpot_z', 'Pileup_nTrueInt', 'Trigger_bits', 'Trigger_firedAny',
    'Trigger_firedIP6')
VERTEX_BRANCHES = ('OtherPV_z', 'OtherPV_score', 'PVtx_vx', 'PVtx_vy', 'PVtx_vz', 'PVtx_ndof', 'PVtx_chi2',
                   'PVtx_score', 'PVtx_ntracks') + tuple(f'PVtx_cov{name}' for name in VERTEX_COVARIANCE_BRANCHES)
TRACK_BRANCHES = (
    'Track_pt', 'Track_eta', 'Track_phi', 'Track_charge', 'Track_pdgId', 'Track_vx', 'Track_vy', 'Track_vz',
    'Track_dxy', 'Track_dxyS', 'Track_dz', 'Track_dzS', 'Track_DCASig', 'Track_ptErr', 'Track_normChi2',
    'Track_nValidHits', 'Track_nValidPixelHits', 'Track_covQopQop', 'Track_covLamLam', 'Track_covPhiPhi',
    'Track_covDxyDxy', 'Track_covDszDsz', 'Track_covDxyDsz', 'Track_covPhiDxy', 'Track_covLamDsz',
    'Track_covLamDxy', 'Track_covQopDxy', 'Track_isPacked', 'Track_isLostTrk', 'Track_isMatchedToMuon',
    'Track_isMatchedToEle', 'Track_fromPV', 'Track_pvAssocQuality', 'Track_pvIdx', 'Track_lostInnerHits',
    'Track_puppiWeight', 'Track_genAssocIdx', 'Track_dzTrg', 'Track_trackFromTau', 'Track_trackFromB',
    'Track_trackFromKstar', 'Track_trackgenIdx', 'Track_trackgenFlav', 'Track_genMatchDr')
MUON_BRANCHES = (
    'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_charge', 'Muon_vx', 'Muon_vy', 'Muon_vz', 'Muon_dxy', 'Muon_dxyErr',
    'Muon_dz', 'Muon_dzErr', 'Muon_ip3d', 'Muon_sip3d', 'Muon_softId', 'Muon_mediumId', 'Muon_pfRelIso03_all',
    'Muon_isTriggering', 'Muon_isPFcand', 'Muon_triggerBits', 'Muon_isAnchor', 'Muon_muonFromTau', 'Muon_muonFromB',
    'Muon_muonFromKstar', 'Muon_muonGenFlav', 'Muon_muonGenIdx')
SV_BRANCHES = ('SV_x', 'SV_y', 'SV_z', 'SV_dlen', 'SV_dlenSig', 'SV_dxySig', 'SV_pAngle', 'SV_mass', 'SV_ntracks',
               'SV_chi2')
GENERATOR_BRANCHES = (
    'GenPart_pdgId', 'GenPart_status', 'GenPart_genPartIdxMother', 'GenPart_pt', 'GenPart_eta', 'GenPart_phi',
    'GenPart_vx', 'GenPart_vy', 'GenPart_vz', 'GenPart_gentrackIdx', 'GenPart_gentrackMatchDr',
    'GenPart_gentrackMatchDPt')
KSTAR_BRANCHES = tuple(f'KstarToPiKAll_{name}' for name in kstar.CANDIDATE_FIELDS) + (
    'KstarToPiK_trk1_idx', 'KstarToPiK_trk2_idx')
BRANCH_NAMES = (SCALAR_BRANCHES + VERTEX_BRANCHES + TRACK_BRANCHES + MUON_BRANCHES + SV_BRANCHES
                + GENERATOR_BRANCHES + KSTAR_BRANCHES)
INTEGER_BRANCHES = frozenset((
    'run', 'luminosityBlock', 'event', 'PV_npvs', 'PV_npvsGood', 'Trigger_bits', 'Trigger_firedAny',
    'Trigger_firedIP6', 'PVtx_ntracks', 'Track_charge', 'Track_pdgId', 'Track_nValidHits', 'Track_nValidPixelHits',
    'Track_isPacked', 'Track_isLostTrk', 'Track_isMatchedToMuon', 'Track_isMatchedToEle', 'Track_fromPV',
    'Track_pvAssocQuality', 'Track_pvIdx', 'Track_lostInnerHits', 'Track_genAssocIdx', 'Track_trackFromTau',
    'Track_trackFromB', 'Track_trackFromKstar', 'Track_trackgenIdx', 'Track_trackgenFlav', 'Muon_charge',
    'Muon_softId', 'Muon_mediumId', 'Muon_isTriggering', 'Muon_isPFcand', 'Muon_triggerBits', 'Muon_isAnchor',
    'Muon_muonFromTau', 'Muon_muonFromB', 'Muon_muonFromKstar', 'Muon_muonGenFlav', 'Muon_muonGenIdx',
    'SV_ntracks', 'GenPart_pdgId', 'GenPart_status', 'GenPart_genPartIdxMother', 'GenPart_gentrackIdx',
    'KstarToPiKAll_trk1_idx', 'KstarToPiKAll_trk2_idx', 'KstarToPiKAll_charge', 'KstarToPiK_trk1_idx',
    'KstarToPiK_trk2_idx'))

_covariance_tables = None


def covariance_tables():
    """Returns the packing tables, loaded once per process."""
    global _covariance_tables
    if _covariance_tables is None:
        _covariance_tables = packed.CovarianceTables.load()
    return _covariance_tables


# ---------------------------------------------------------------------------
# Vertices
# ---------------------------------------------------------------------------

def covariance_matrices(stored):
    """stored: (N, 10) lower-triangular 4x4 vertex covariance (xx, xy, yy,
    xz, yz, zz, then the time terms). Returns (N, 3, 3)."""
    values = np.asarray(stored, dtype=np.float64).reshape(-1, reader.VERTEX_COVARIANCE_SIZE)
    matrices = np.zeros((values.shape[0], 3, 3))
    for (row, column), position in VERTEX_COVARIANCE_POSITION.items():
        matrices[:, row, column] = matrices[:, column, row] = values[:, position]
    return matrices


def primary_vertices(block, score):
    """block: one event's {PV member: (V,)}; score: (V,). Returns a dict:
    position (V, 3), chi2, ndof, score (V,), covariance (V, 3, 3), good
    (V,) bool (NanoAOD's good-vertex cut), listed (V,) bool (PV[0] and the
    other good vertices) and list_position (V,) (row in that list or -1)."""
    position = np.stack([np.asarray(block[member], dtype=np.float64) for member in reader.PV_MEMBERS[:3]], axis=1)
    chi2 = np.asarray(block['chi2_'], dtype=np.float64)
    ndof = np.asarray(block['ndof_'], dtype=np.float64)
    good = (~((chi2 == 0) & (ndof == 0)) & (ndof > GOOD_VERTEX_MINIMUM_NDOF)
            & (np.abs(position[:, 2]) <= GOOD_VERTEX_MAXIMUM_ABS_Z)
            & (np.hypot(position[:, 0], position[:, 1]) <= GOOD_VERTEX_MAXIMUM_RHO))
    listed = good.copy()
    listed[:1] = True
    return {'position': position, 'chi2': chi2, 'ndof': ndof, 'score': np.asarray(score, dtype=np.float64),
            'covariance': covariance_matrices(block['covariance_[10]']), 'good': good, 'listed': listed,
            'list_position': np.where(listed, np.cumsum(listed) - 1, -1)}


def vertex_branches(vertices):
    """vertices: primary_vertices output. Returns the PV / OtherPV / PVtx
    branches of one event."""
    position, score, listed = vertices['position'], vertices['score'], vertices['listed']
    first = position[0] if position.shape[0] else np.zeros(3)
    others = slice(1, 1 + OTHER_VERTEX_COUNT)
    branches = {
        'PV_x': first[0], 'PV_y': first[1], 'PV_z': first[2], 'PV_npvs': position.shape[0],
        'PV_npvsGood': int(vertices['good'].sum()), 'PV_score': score[0] if score.size else 0.0,
        'OtherPV_z': position[others, 2], 'OtherPV_score': score[others],
        'PVtx_vx': position[listed, 0], 'PVtx_vy': position[listed, 1], 'PVtx_vz': position[listed, 2],
        'PVtx_ndof': vertices['ndof'][listed], 'PVtx_chi2': vertices['chi2'][listed], 'PVtx_score': score[listed],
        'PVtx_ntracks': np.zeros(int(listed.sum())),
    }
    for name, (row, column) in VERTEX_COVARIANCE_BRANCHES.items():
        branches[f'PVtx_cov{name}'] = vertices['covariance'][listed, row, column]
    return branches


def sum_pt2(candidates, vertex_position):
    """candidates: unpacked PF candidates of one event; vertex_position:
    (V, 3). Returns NanoAOD's PV_sumpt2: the summed pT^2 of the charged
    candidates within 0.2 cm in z of PV[0] and closer to it than to any
    other vertex."""
    charged = candidates['charge'] != 0
    if not charged.any() or not vertex_position.shape[0]:
        return 0.0
    phi, eta = candidates['track_phi'][charged], candidates['track_eta'][charged]
    offset = candidates['point'][charged][:, None, :] - vertex_position[None, :, :]
    along = offset[..., 0] * np.cos(phi)[:, None] + offset[..., 1] * np.sin(phi)[:, None]
    distance = np.abs(offset[..., 2] - along * np.sinh(eta)[:, None])
    included = (distance[:, 0] < SUM_PT2_MAXIMUM_DZ) & np.all(distance[:, 1:] >= distance[:, :1], axis=1)
    return float(np.sum(candidates['pt'][charged][included] ** 2))


def secondary_vertex_branches(block, vertices):
    """block: one event's {SV member: (S,)} plus covariance (S, 10);
    vertices: primary_vertices output. Returns the SV branches of the
    vertices with a 3D flight significance above 3 with respect to PV[0]."""
    position = np.stack([np.asarray(block[f'm_state.vertex_.fCoordinates.f{axis}'], dtype=np.float64)
                         for axis in 'XYZ'], axis=1)
    if not position.shape[0] or not vertices['position'].shape[0]:
        return {name: np.zeros(0) for name in SV_BRANCHES}
    primary = vertices['position'][0]
    distance = geometry.vertex_distance(primary, vertices['covariance'][0], position,
                                        covariance_matrices(block['covariance']))
    pt, eta, phi = (np.asarray(block[f'm_state.p4Polar_.fCoordinates.f{name}'], dtype=np.float64)
                    for name in ('Pt', 'Eta', 'Phi'))
    momentum = pt[:, None] * np.stack([np.cos(phi), np.sin(phi), np.sinh(eta)], axis=1)
    # NanoAOD convention: angle between the momentum and the line from the vertex back to PV[0]
    back = primary[None, :] - position
    cosine = np.einsum('ij,ij->i', back, momentum) / np.maximum(
        np.linalg.norm(back, axis=1) * np.linalg.norm(momentum, axis=1), TINY)
    ndof = np.asarray(block['ndof_'], dtype=np.float64)
    keep = distance['significance_3d'] > SV_MINIMUM_SIGNIFICANCE
    return {
        'SV_x': position[keep, 0], 'SV_y': position[keep, 1], 'SV_z': position[keep, 2],
        'SV_dlen': distance['distance_3d'][keep], 'SV_dlenSig': distance['significance_3d'][keep],
        'SV_dxySig': distance['significance_xy'][keep], 'SV_pAngle': np.arccos(np.clip(cosine, -1.0, 1.0))[keep],
        'SV_mass': np.asarray(block['m_state.p4Polar_.fCoordinates.fM'], dtype=np.float64)[keep],
        'SV_ntracks': np.asarray(block['dau@.size()'])[keep],
        'SV_chi2': (np.asarray(block['chi2_'], dtype=np.float64) / np.where(ndof != 0, ndof, 1.0))[keep],
    }


# ---------------------------------------------------------------------------
# Candidates: tracks and muons
# ---------------------------------------------------------------------------

def merge_candidates(pf, lost, pf_association, lost_association):
    """pf, lost: packed_candidate.unpack_candidates outputs of one event;
    *_association: generator row of each candidate (-1 none). Returns one
    dict over the PF candidates followed by the lost tracks, with is_lost
    and gen_association added."""
    merged = {name: np.concatenate([pf[name], lost[name]]) for name in pf if name != 'covariance'}
    merged['covariance'] = {element: np.concatenate([pf['covariance'][element], lost['covariance'][element]])
                            for element in pf['covariance']}
    merged['is_lost'] = np.concatenate([np.zeros(pf['pdg'].size, dtype=bool), np.ones(lost['pdg'].size, dtype=bool)])
    merged['gen_association'] = np.concatenate([pf_association, lost_association]).astype(np.int64)
    return merged


def take(candidates, rows):
    """candidates: merge_candidates output; rows: (R,) indices. Returns the
    same dict restricted to rows."""
    taken = {name: values[rows] for name, values in candidates.items() if name != 'covariance'}
    taken['covariance'] = {element: values[rows] for element, values in candidates['covariance'].items()}
    return taken


def track_branches(tracks, vertices, beam_spot):
    """tracks: candidate fields of the track-table rows; vertices:
    primary_vertices output; beam_spot: (3,). Returns the reconstructed
    Track branches of one event (truth and anchor-muon columns excluded)."""
    covariance = tracks['covariance']
    vertex_z = vertices['position'][:, 2]
    primary = vertices['position'][0] if vertex_z.size else np.zeros(3)
    key = tracks['pv_key']
    safe_key = np.clip(key, 0, max(vertex_z.size - 1, 0))
    own_z = np.where(key >= 0, vertex_z[safe_key], 0.0) if vertex_z.size else np.zeros(key.size)
    list_position = np.where(key >= 0, vertices['list_position'][safe_key], -1) if vertex_z.size else key
    dxy = packed.dxy_to_point(tracks['point'], tracks['track_phi'], primary)
    beam_distance = geometry.beam_spot_dca(tracks['point'][:, 0], tracks['point'][:, 1], tracks['track_phi'],
                                           beam_spot[0], beam_spot[1])
    dxy_error, dz_error = tracks['dxy_error'], tracks['dz_error']
    zeros = np.zeros(key.size)
    return {
        'Track_pt': tracks['pt'], 'Track_eta': tracks['eta'], 'Track_phi': tracks['phi'],
        'Track_charge': tracks['charge'], 'Track_pdgId': tracks['pdg'],
        'Track_vx': tracks['point'][:, 0], 'Track_vy': tracks['point'][:, 1], 'Track_vz': tracks['point'][:, 2],
        'Track_dxy': dxy, 'Track_dxyS': dxy / dxy_error,
        'Track_dz': packed.dz_to_point(tracks['point'], tracks['track_phi'], tracks['track_eta'], primary),
        'Track_dzS': (tracks['dz_associated'] + own_z - primary[2]) / dz_error,
        'Track_DCASig': np.where(dxy_error > 0, np.abs(beam_distance) / np.where(dxy_error > 0, dxy_error, 1.0), -1.0),
        'Track_ptErr': tracks['pt_error'], 'Track_normChi2': tracks['norm_chi2'],
        'Track_nValidHits': tracks['n_hits'], 'Track_nValidPixelHits': tracks['n_pixel_hits'],
        'Track_covQopQop': covariance[(0, 0)], 'Track_covLamLam': covariance[(1, 1)],
        'Track_covPhiPhi': covariance[(2, 2)], 'Track_covDxyDxy': covariance[(3, 3)],
        'Track_covDszDsz': covariance[(4, 4)], 'Track_covDxyDsz': covariance[(3, 4)],
        'Track_covPhiDxy': covariance[(2, 3)], 'Track_covLamDsz': covariance[(1, 4)],
        'Track_covLamDxy': zeros, 'Track_covQopDxy': zeros,
        'Track_isPacked': ~tracks['is_lost'], 'Track_isLostTrk': tracks['is_lost'],
        'Track_isMatchedToMuon': zeros, 'Track_isMatchedToEle': zeros,
        'Track_fromPV': tracks['from_pv'], 'Track_pvAssocQuality': tracks['pv_association_quality'],
        'Track_pvIdx': list_position, 'Track_lostInnerHits': tracks['lost_inner_hits'],
        'Track_puppiWeight': tracks['puppi_weight'], 'Track_genAssocIdx': tracks['gen_association'],
    }


def muon_branches(muons, vertices, trigger_bits, anchor, tau_muon_position, tau_muon_generator_row):
    """muons: candidate fields of the muon-table rows; vertices:
    primary_vertices output; trigger_bits: (M,) path bits of the matched
    trigger objects; anchor, tau_muon_position: muon-table rows or -1;
    tau_muon_generator_row: GenPart row of the generated tau muon. Returns
    the Muon branches of one event."""
    covariance = muons['covariance']
    has_vertex = vertices['position'].shape[0] > 0
    primary = vertices['position'][0] if has_vertex else np.zeros(3)
    primary_covariance = vertices['covariance'][0] if has_vertex else np.zeros((3, 3))
    offset = geometry.point_line_offset(muons['point'], geometry.unit_direction(muons['track_eta'], muons['track_phi']),
                                        primary)
    impact = np.linalg.norm(offset, axis=1)
    normal = offset / np.maximum(impact, TINY)[:, None]
    variance = (geometry.perpendicular_variance(normal, muons['track_phi'], muons['track_eta'], covariance[(3, 3)],
                                                covariance[(4, 4)], covariance[(3, 4)])
                + np.einsum('ni,ij,nj->n', normal, primary_covariance, normal))
    count = muons['pt'].size
    from_tau = np.arange(count) == tau_muon_position
    ones = np.ones(count)
    return {
        'Muon_pt': muons['pt'], 'Muon_eta': muons['eta'], 'Muon_phi': muons['phi'], 'Muon_charge': muons['charge'],
        'Muon_vx': muons['point'][:, 0], 'Muon_vy': muons['point'][:, 1], 'Muon_vz': muons['point'][:, 2],
        'Muon_dxy': packed.dxy_to_point(muons['point'], muons['track_phi'], primary),
        'Muon_dxyErr': muons['dxy_error'],
        'Muon_dz': packed.dz_to_point(muons['point'], muons['track_phi'], muons['track_eta'], primary),
        'Muon_dzErr': muons['dz_error'] * np.cosh(muons['track_eta']),
        'Muon_ip3d': impact, 'Muon_sip3d': impact / np.sqrt(np.maximum(variance, TINY)),
        'Muon_softId': ones, 'Muon_mediumId': -ones, 'Muon_pfRelIso03_all': -ones,
        'Muon_isTriggering': (trigger_bits & IP6_PATH_MASK) != 0, 'Muon_isPFcand': ones,
        'Muon_triggerBits': trigger_bits, 'Muon_isAnchor': np.arange(count) == anchor,
        'Muon_muonFromTau': from_tau, 'Muon_muonFromB': from_tau, 'Muon_muonFromKstar': np.zeros(count),
        'Muon_muonGenFlav': np.where(from_tau, TAU_FLAVOUR, 0),
        'Muon_muonGenIdx': np.where(from_tau, tau_muon_generator_row, -1),
    }


def kstar_branches(tracks, anchor_point, anchor_direction, beam_spot):
    """tracks: candidate fields of the track-table rows; anchor_point,
    anchor_direction: (3,) anchor-muon line or None; beam_spot: (3,).
    Returns the KstarToPiKAll branches and the highest-pT pair as
    KstarToPiK (no anchor muon: no candidates)."""
    if anchor_point is None:
        candidates = kstar.empty_candidates()
    else:
        covariance = tracks['covariance']
        candidates = kstar.kstar_candidates(
            tracks['point'], tracks['pt'], tracks['eta'], tracks['phi'], tracks['charge'], covariance[(3, 3)],
            covariance[(4, 4)], covariance[(3, 4)], anchor_point, anchor_direction, beam_spot)
    branches = {f'KstarToPiKAll_{name}': candidates[name] for name in kstar.CANDIDATE_FIELDS}
    leading = [int(np.argmax(candidates['pt']))] if candidates['pt'].size else []
    branches['KstarToPiK_trk1_idx'] = candidates['trk1_idx'][leading]
    branches['KstarToPiK_trk2_idx'] = candidates['trk2_idx'][leading]
    return branches


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def trigger_path_lookup(hlt_paths):
    """hlt_paths: ordered path names of the file's HLT menu. Returns the
    (13,) menu index of every MUON_TRIGGER_PATHS name (versions ignored)."""
    stripped = [VERSION_SUFFIX.sub('', name) for name in hlt_paths]
    missing = [name for name in MUON_TRIGGER_PATHS if name not in stripped]
    if missing:
        raise ValueError(f'trigger paths absent from the HLT menu: {missing}')
    return np.array([stripped.index(name) for name in MUON_TRIGGER_PATHS], dtype=np.int64)


def event_trigger_bits(status, lookup):
    """status: (P,) raw HLT path status words of one event; lookup:
    trigger_path_lookup output. Returns the bit mask of the accepted
    MUON_TRIGGER_PATHS."""
    accepted = (np.asarray(status).astype(np.int64)[lookup] & reader.PATH_STATE_MASK) == reader.PATH_ACCEPTED_STATE
    return int(np.sum(accepted.astype(np.int64) << np.arange(lookup.size)))


def trigger_object_bits(path_indices, path_accepted, object_of_entry, n_objects, lookup):
    """path_indices, path_accepted, object_of_entry: (E,) menu path, last
    filter flag and trigger object of every (object, path) entry of one
    event; lookup: trigger_path_lookup output. Returns (n_objects,) bit
    masks of the MUON_TRIGGER_PATHS each object fired."""
    bits = np.zeros(n_objects, dtype=np.int64)
    for bit, menu_index in enumerate(lookup):
        bits[object_of_entry[(path_indices == menu_index) & (path_accepted != 0)]] |= 1 << bit
    return bits


def muon_trigger_bits(muon_eta, muon_phi, object_eta, object_phi, object_bits):
    """muon_eta, muon_phi: (M,); object_eta, object_phi, object_bits: (O,).
    Returns (M,) bit masks: the paths of the trigger objects within
    TRIGGER_MATCH_DELTA_R of each muon."""
    if not muon_eta.size or not object_eta.size:
        return np.zeros(muon_eta.size, dtype=np.int64)
    close = geometry.delta_r(muon_eta[:, None], muon_phi[:, None], object_eta[None, :],
                             object_phi[None, :]) < TRIGGER_MATCH_DELTA_R
    return np.bitwise_or.reduce(np.where(close, object_bits[None, :], 0), axis=1)


# ---------------------------------------------------------------------------
# Generator truth
# ---------------------------------------------------------------------------

def generator_table(pruned, packed_generator):
    """pruned, packed_generator: one event's generator members as read from
    MiniAOD. Returns (table, signal): table = dict of (G,) arrays (pdg,
    status, pt, eta, phi, mother) and vertex (G, 3) over the pruned rows
    followed by the two K* daughters recovered from the packed collection;
    signal = (rows, charges, kinds) of the particles to match to tracks,
    kind = TAU_FLAVOUR (tau pion), KSTAR_FLAVOUR (K* daughter) or MUON_PDG
    (tau muon)."""
    pdg = np.asarray(pruned['pdgId_']).astype(np.int64)
    status = np.asarray(pruned['status_']).astype(np.int64)
    pt, eta, phi = (np.asarray(pruned[f'p4Polar_.fCoordinates.f{name}'], dtype=np.float64)
                    for name in ('Pt', 'Eta', 'Phi'))
    vertex = np.stack([np.asarray(pruned[f'vertex_.fCoordinates.f{axis}'], dtype=np.float64) for axis in 'XYZ'],
                      axis=1)
    chain = truth.signal_chain(pdg, status, vertex, eta, phi)
    mother = np.full(pdg.size, -1, dtype=np.int64)
    mother[chain['tau_pions']] = chain['hadronic_tau']
    if chain['tau_muon'] >= 0:
        mother[chain['tau_muon']] = chain['muonic_tau']
    for row in (chain['hadronic_tau'], chain['muonic_tau'], chain['kstar']):
        if row >= 0:
            mother[row] = chain['b_meson']
    if chain['b_meson'] != chain['b_meson_origin']:
        mother[chain['b_meson']] = chain['b_meson_origin']
    rows = [int(row) for row in chain['tau_pions']]
    charges = [int(np.sign(pdg[row])) for row in rows]
    kinds = [TAU_FLAVOUR] * len(rows)
    if chain['tau_muon'] >= 0:
        rows.append(chain['tau_muon'])
        charges.append(-int(np.sign(pdg[chain['tau_muon']])))
        kinds.append(MUON_PDG)
    table = {'pdg': pdg, 'status': status, 'pt': pt, 'eta': eta, 'phi': phi, 'vertex': vertex, 'mother': mother}
    if chain['kstar'] >= 0:
        unpacked = truth.unpack_packed_generator(*(packed_generator[member] for member in reader.PACKED_GEN_MEMBERS))
        daughters = truth.kstar_daughters(pt[chain['kstar']], eta[chain['kstar']], phi[chain['kstar']], unpacked)
        if daughters is not None:
            added = np.array(daughters)
            rows += [pdg.size, pdg.size + 1]
            charges += [int(np.sign(unpacked['pdg'][row])) for row in added]
            kinds += [KSTAR_FLAVOUR, KSTAR_FLAVOUR]
            table = {
                'pdg': np.concatenate([pdg, unpacked['pdg'][added]]), 'status': np.concatenate([status, [1, 1]]),
                'pt': np.concatenate([pt, unpacked['pt'][added]]), 'eta': np.concatenate([eta, unpacked['eta'][added]]),
                'phi': np.concatenate([phi, unpacked['phi'][added]]),
                'vertex': np.concatenate([vertex, np.repeat(vertex[[chain['kstar']]], 2, axis=0)]),
                'mother': np.concatenate([mother, [chain['kstar'], chain['kstar']]]),
            }
    return table, (np.array(rows, dtype=np.int64), np.array(charges, dtype=np.int64), kinds)


def truth_branches(table, signal, candidates, track_rows, muon_rows):
    """table, signal: generator_table output; candidates: merge_candidates
    output; track_rows, muon_rows: candidate rows of the track and muon
    tables. Every signal particle is matched one-to-one to the charged
    candidates with track details (not only the table rows, so that a
    particle reconstructed outside the tables cannot take a wrong table
    track); labels are written for the matches inside the tables. Returns
    (Track truth and GenPart branches, muon-table row of the tau muon or -1,
    its GenPart row or -1)."""
    rows, charges, kinds = signal
    pool = np.flatnonzero(candidates['has_track_details'] & (candidates['charge'] != 0))
    matched, distance = truth.match_tracks_to_generator(
        candidates['pt'][pool], candidates['eta'][pool], candidates['phi'][pool], candidates['charge'][pool],
        table['pt'][rows], table['eta'][rows], table['phi'][rows], charges)
    track_position = np.full(candidates['pt'].size, -1, dtype=np.int64)
    track_position[track_rows] = np.arange(track_rows.size)
    muon_position = np.full(candidates['pt'].size, -1, dtype=np.int64)
    muon_position[muon_rows] = np.arange(muon_rows.size)
    track_pt = candidates['pt'][track_rows]
    track_generator = np.full(track_rows.size, -1, dtype=np.int64)
    track_flavour = np.zeros(track_rows.size, dtype=np.int64)
    track_distance = np.full(track_rows.size, -1.0)
    generator_track = np.full(table['pdg'].size, -1, dtype=np.int64)
    generator_distance = np.full(table['pdg'].size, -1.0)
    generator_pt_difference = np.full(table['pdg'].size, -1.0)
    tau_muon_position = tau_muon_generator_row = -1
    for generator_row, pool_row, pair_distance, kind in zip(rows.tolist(), matched.tolist(), distance.tolist(), kinds):
        if pool_row < 0:
            continue
        candidate = pool[pool_row]
        if kind == MUON_PDG:
            tau_muon_generator_row = generator_row
            tau_muon_position = int(muon_position[candidate])
            continue
        position = int(track_position[candidate])
        if position < 0:
            continue
        track_generator[position], track_flavour[position], track_distance[position] = generator_row, kind, pair_distance
        generator_track[generator_row], generator_distance[generator_row] = position, pair_distance
        generator_pt_difference[generator_row] = (abs(track_pt[position] - table['pt'][generator_row])
                                                  / table['pt'][generator_row])
    from_tau, from_kstar = track_flavour == TAU_FLAVOUR, track_flavour == KSTAR_FLAVOUR
    branches = {
        'Track_trackFromTau': from_tau, 'Track_trackFromKstar': from_kstar, 'Track_trackFromB': from_tau | from_kstar,
        'Track_trackgenIdx': track_generator, 'Track_trackgenFlav': track_flavour, 'Track_genMatchDr': track_distance,
        'GenPart_pdgId': table['pdg'], 'GenPart_status': table['status'], 'GenPart_genPartIdxMother': table['mother'],
        'GenPart_pt': table['pt'], 'GenPart_eta': table['eta'], 'GenPart_phi': table['phi'],
        'GenPart_vx': table['vertex'][:, 0], 'GenPart_vy': table['vertex'][:, 1], 'GenPart_vz': table['vertex'][:, 2],
        'GenPart_gentrackIdx': generator_track, 'GenPart_gentrackMatchDr': generator_distance,
        'GenPart_gentrackMatchDPt': generator_pt_difference,
    }
    return branches, tau_muon_position, tau_muon_generator_row


# ---------------------------------------------------------------------------
# Event and file assembly
# ---------------------------------------------------------------------------

def _sorted_rows(candidates, mask):
    rows = np.flatnonzero(mask)
    return rows[np.argsort(-candidates['pt'][rows], kind='stable')]


def build_event(event, tables, lookup):
    """event: one event's raw products (pf, lost, pv, gen, packed_gen, sv:
    {member: array}; pf_match, lost_match, pv_score, trigger_status,
    beam_spot, object_*: arrays; run, luminosityBlock, event, pileup:
    scalars); tables: CovarianceTables; lookup: trigger_path_lookup output.
    Returns {branch: scalar or 1-D array} for every BRANCH_NAMES entry."""
    vertices = primary_vertices(event['pv'], event['pv_score'])
    pf = packed.unpack_candidates(event['pf'], vertices['position'], vertices['ndof'], tables)
    lost = packed.unpack_candidates(event['lost'], vertices['position'], vertices['ndof'], tables)
    candidates = merge_candidates(pf, lost, event['pf_match'], event['lost_match'])
    beam_spot = np.asarray(event['beam_spot'], dtype=np.float64)

    detailed = candidates['has_track_details']
    track_rows = _sorted_rows(candidates, (
        (np.abs(candidates['pdg']) == PION_PDG) & detailed & (candidates['high_purity'] | candidates['is_lost'])
        & (candidates['pt'] >= TRACK_MINIMUM_PT) & (np.abs(candidates['eta']) <= TRACK_MAXIMUM_ABS_ETA)))
    muon_rows = _sorted_rows(candidates, (
        (np.abs(candidates['pdg']) == MUON_PDG) & detailed & ~candidates['is_lost']
        & (candidates['pt'] >= MUON_MINIMUM_PT) & (np.abs(candidates['eta']) <= MUON_MAXIMUM_ABS_ETA)))
    tracks, muons = take(candidates, track_rows), take(candidates, muon_rows)

    bits = event_trigger_bits(event['trigger_status'], lookup)
    object_bits = trigger_object_bits(event['object_path_indices'], event['object_path_accepted'],
                                      event['object_of_entry'], event['object_eta'].size, lookup)
    muon_bits = muon_trigger_bits(muons['eta'], muons['phi'], event['object_eta'], event['object_phi'], object_bits)
    triggering = np.flatnonzero((muon_bits & IP6_PATH_MASK) != 0)
    anchor = int(triggering[0]) if triggering.size else (0 if muon_rows.size else -1)

    branches = {
        'run': event['run'], 'luminosityBlock': event['luminosityBlock'], 'event': event['event'],
        'PV_sumpt2': sum_pt2(pf, vertices['position']),
        'BeamSpot_x': beam_spot[0], 'BeamSpot_y': beam_spot[1], 'BeamSpot_z': beam_spot[2],
        'Pileup_nTrueInt': event['pileup'], 'Trigger_bits': bits, 'Trigger_firedAny': int(bits != 0),
        'Trigger_firedIP6': int((bits & IP6_PATH_MASK) != 0),
    }
    branches.update(vertex_branches(vertices))
    branches.update(track_branches(tracks, vertices, beam_spot))
    table, signal = generator_table(event['gen'], event['packed_gen'])
    labels, tau_muon_position, tau_muon_generator_row = truth_branches(table, signal, candidates, track_rows,
                                                                       muon_rows)
    branches.update(labels)
    branches.update(muon_branches(muons, vertices, muon_bits, anchor, tau_muon_position, tau_muon_generator_row))
    if anchor >= 0:
        anchor_point = muons['point'][anchor]
        anchor_direction = geometry.unit_direction(muons['track_eta'][[anchor]], muons['track_phi'][[anchor]])[0]
        branches['Track_dzTrg'] = tracks['point'][:, 2] - anchor_point[2]
    else:
        anchor_point = anchor_direction = None
        branches['Track_dzTrg'] = np.zeros(track_rows.size)
    branches.update(kstar_branches(tracks, anchor_point, anchor_direction, beam_spot))
    branches.update(secondary_vertex_branches(event['sv'], vertices))
    return branches


def _per_event(block):
    """block: {member: awkward (events, var[, k])}. Returns a list over
    events of {member: numpy array}."""
    members = list(block)
    counts = ak.to_numpy(ak.num(block[members[0]], axis=1))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    flat = {member: ak.to_numpy(ak.flatten(block[member], axis=1)) for member in members}
    return [{member: values[offsets[row]:offsets[row + 1]] for member, values in flat.items()}
            for row in range(counts.size)]


def build_branches(products):
    """products: miniaod_reader.read_miniaod_file output (at least one
    event). Returns {branch: (events,) numpy array for SCALAR_BRANCHES,
    awkward (events, var) array otherwise} for every BRANCH_NAMES entry."""
    n_events = products['n_events']
    if len(products['event']['run']) != n_events or len(products['beam_spot']) != n_events:
        raise ValueError('event-level products do not hold one row per event')
    lookup = trigger_path_lookup(products['hlt_paths'])
    tables = covariance_tables()
    blocks = {name: _per_event(products[name]) for name in ('pf', 'lost', 'pv', 'gen', 'packed_gen', 'sv')}
    singles = {name: _per_event({'values': products[name]})
               for name in ('pf_match', 'lost_match', 'pv_score', 'trigger_status')}
    objects = products['trigger_objects']
    rows = []
    for row in range(n_events):
        event = {name: values[row] for name, values in blocks.items()}
        event.update({name: values[row]['values'] for name, values in singles.items()})
        path_counts = ak.to_numpy(ak.num(objects['path_indices'][row], axis=1))
        event.update(
            run=products['event']['run'][row], luminosityBlock=products['event']['luminosityBlock'][row],
            event=products['event']['event'][row], pileup=products['pileup'][row],
            beam_spot=products['beam_spot'][row],
            object_eta=ak.to_numpy(objects['eta'][row]), object_phi=ak.to_numpy(objects['phi'][row]),
            object_path_indices=ak.to_numpy(ak.flatten(objects['path_indices'][row], axis=1)),
            object_path_accepted=ak.to_numpy(ak.flatten(objects['path_accepted'][row], axis=1)),
            object_of_entry=np.repeat(np.arange(path_counts.size), path_counts))
        rows.append(build_event(event, tables, lookup))
    if set(rows[0]) != set(BRANCH_NAMES):
        raise RuntimeError(f'built branches differ from BRANCH_NAMES: {sorted(set(rows[0]) ^ set(BRANCH_NAMES))}')
    branches = {}
    for name in BRANCH_NAMES:
        dtype = np.int64 if name in INTEGER_BRANCHES else np.float64
        if name in SCALAR_BRANCHES:
            branches[name] = np.array([row[name] for row in rows], dtype=dtype)
        else:
            values = [np.asarray(row[name]) for row in rows]
            branches[name] = ak.unflatten(np.concatenate(values).astype(dtype), [value.size for value in values])
    return branches


class MiniAODTree:
    """Events tree of a MiniAOD file with the interface the converter uses
    on uproot trees: keys(), num_entries and arrays(). The file is read and
    every branch built when the object is created."""

    def __init__(self, path, num_entries=None):
        """path: MiniAOD file; num_entries: its entry count when already
        known."""
        import uproot
        self.path = path
        if num_entries is None:
            with uproot.open(path) as root_file:
                num_entries = root_file['Events'].num_entries
        self.num_entries = int(num_entries)
        self._branches = build_branches(reader.read_miniaod_file(path)) if self.num_entries else {}

    def keys(self):
        return list(BRANCH_NAMES)

    def arrays(self, expressions, entry_start=None, entry_stop=None):
        """expressions: branch names; [entry_start, entry_stop): entry
        window. Returns an awkward record array with one field per name."""
        start = 0 if entry_start is None else int(entry_start)
        stop = self.num_entries if entry_stop is None else int(entry_stop)
        return ak.Array({name: self._branches[name][start:stop] for name in expressions})
