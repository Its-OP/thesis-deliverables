"""Numpy port of the unpacking of pat::PackedCandidate (CMSSW
DataFormats/PatCandidates: PackedCandidate.h/.cc,
CovarianceParameterization.cc, DataFormats/Math liblogintpack.h).

MiniAOD stores every PF candidate and lost track in a compressed integer
form. The functions below reproduce what the candidate's C++ methods
return: momentum, the track reference point (the point of closest approach
to the associated primary vertex), impact parameters, hit counts, quality
flags, fromPV() and the 5x5 track covariance in the (q/p, lambda, phi, dxy,
dsz) basis, of which MiniAOD keeps eight terms.
"""
from __future__ import annotations

import os

import numpy as np

INT16_MAX = 32767.0
ANGLE_SCALE = 3.2
ETA_SCALE = 6.0
IMPACT_PARAMETER_SCALE = 100.0
UNASSOCIATED_DZ_SCALE = 40.0
PIXEL_HITS_MASK = 7
STRIP_HITS_SHIFT = 3
ASSIGNMENT_QUALITY_MASK = 0x7
HIGH_PURITY_MASK = 0x8
LOST_INNER_HITS_MASK = 0x30
LOST_INNER_HITS_SHIFT = 4
PUPPI_WEIGHT_SCALE = 255.0

# pat::PackedCandidate::PVAssociationQuality and PVAssoc
QUALITY_COMPATIBILITY_BTAG = 4
QUALITY_USED_IN_FIT_LOOSE = 6
QUALITY_USED_IN_FIT_TIGHT = 7
NO_PV, PV_LOOSE, PV_TIGHT, PV_USED_IN_FIT = 0, 1, 2, 3
MINIMUM_VERTEX_NDOF = 4.0

# packed member -> (i, j) of the track covariance, PackedCandidate::unpackCovariance
PACKED_COVARIANCE_ELEMENTS = {
    'packedCovariance_.dptdpt': (0, 0), 'packedCovariance_.detadeta': (1, 1),
    'packedCovariance_.dphidphi': (2, 2), 'packedCovariance_.dxydxy': (3, 3),
    'packedCovariance_.dzdz': (4, 4), 'packedCovariance_.dxydz': (3, 4),
    'packedCovariance_.dlambdadz': (1, 4), 'packedCovariance_.dphidxy': (2, 3),
}
# CompressionElement::Method and ::Target
METHOD_FLOAT16, METHOD_REDUCE_MANTISSA, METHOD_LOG_PACK, METHOD_TAN_LOG_PACK, METHOD_ZERO, METHOD_ONE = range(6)
TARGET_REAL_VALUE, TARGET_RATIO_TO_REFERENCE, TARGET_DIFFERENCE_TO_REFERENCE = range(3)
MINIMUM_HITS_FOR_REFERENCE, MAXIMUM_HITS_FOR_REFERENCE = 2, 32
MAXIMUM_LOG_BASE = 32768
EMPTY_DIAGONAL = 1e-9
DEFAULT_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                              'covariance_parameterization_version1.npz')


def half_to_float(values):
    """values: (N,) IEEE half-precision bit patterns (any numeric dtype).
    Returns (N,) float64 (MiniFloatConverter::float16to32)."""
    return np.asarray(values).astype(np.uint16).view(np.float16).astype(np.float64)


def to_signed16(values):
    """values: (N,) uint16 patterns (any numeric dtype). Returns (N,)
    float64 of the int16 reinterpretation."""
    return np.asarray(values).astype(np.uint16).view(np.int16).astype(np.float64)


def charge(pdg):
    """pdg: (N,) particle codes. Returns (N,) int charges as
    PackedCandidate::charge(): defined for pions, electrons, muons, taus
    and W, zero otherwise."""
    pdg = np.asarray(pdg, dtype=np.int64)
    sign = np.sign(pdg)
    absolute = np.abs(pdg)
    return np.where(np.isin(absolute, (211, 24)), sign, np.where(np.isin(absolute, (11, 13, 15)), -sign, 0))


def unpack_momentum(packed_pt, packed_eta, packed_phi, packed_mass):
    """packed_*: (N,) uint16. Returns a dict of (N,) arrays pt, eta, phi,
    mass (PackedCandidate::unpack, including its small phi dither)."""
    pt = half_to_float(packed_pt)
    shift = np.where(pt < 1.0, 0.1 * pt, 0.1 / np.where(pt > 0, pt, 1.0))
    sign = np.where((pt * 10.0).astype(np.int64) % 2 == 0, 1.0, -1.0)
    phi = to_signed16(packed_phi) * ANGLE_SCALE / INT16_MAX + sign * shift * ANGLE_SCALE / INT16_MAX
    return {'pt': pt, 'eta': to_signed16(packed_eta) * ETA_SCALE / INT16_MAX, 'phi': phi,
            'mass': half_to_float(packed_mass)}


def unpack_track_momentum(pt, eta, phi, packed_dphi, packed_deta, packed_dtrkpt):
    """pt, eta, phi: (N,) candidate momentum; packed_*: (N,) uint16.
    Returns a dict of (N,) arrays: track_pt (ptTrk), track_eta (etaAtVtx),
    track_phi (phiAtVtx, wrapped into [-pi, pi])."""
    track_phi = phi + to_signed16(packed_dphi) * ANGLE_SCALE / INT16_MAX
    track_phi = np.where(track_phi > np.pi, track_phi - 2.0 * np.pi, track_phi)
    track_phi = np.where(track_phi < -np.pi, track_phi + 2.0 * np.pi, track_phi)
    return {'track_pt': pt + half_to_float(packed_dtrkpt), 'track_eta': eta + half_to_float(packed_deta),
            'track_phi': track_phi}


def hit_counts(packed_hits, packed_layers):
    """packed_hits, packed_layers: (N,) uint8. Returns a dict of (N,) int
    arrays n_hits, n_pixel_hits, n_pixel_layers, n_strip_layers and the bool
    has_track_details (packedHits_ holds the hits beyond one per layer)."""
    hits = np.asarray(packed_hits).astype(np.int64)
    layers = np.asarray(packed_layers).astype(np.int64)
    n_pixel_layers = layers & PIXEL_HITS_MASK
    n_strip_layers = layers >> STRIP_HITS_SHIFT
    n_pixel_hits = (hits & PIXEL_HITS_MASK) + n_pixel_layers
    return {'n_hits': (hits >> STRIP_HITS_SHIFT) + n_strip_layers + n_pixel_hits, 'n_pixel_hits': n_pixel_hits,
            'n_pixel_layers': n_pixel_layers, 'n_strip_layers': n_strip_layers,
            'has_track_details': (hits != 0) | (layers != 0)}


def quality(quality_flags):
    """quality_flags: (N,) uint16. Returns a dict of (N,) arrays:
    high_purity (bool), pv_association_quality (0..7), lost_inner_hits
    (-1 valid hit in the first pixel barrel layer, 0 none lost, 1, 2)."""
    flags = np.asarray(quality_flags).astype(np.int64)
    return {'high_purity': (flags & HIGH_PURITY_MASK) != 0,
            'pv_association_quality': flags & ASSIGNMENT_QUALITY_MASK,
            'lost_inner_hits': ((flags & LOST_INNER_HITS_MASK) >> LOST_INNER_HITS_SHIFT) - 1}


def unpack_vertex(packed_dxy, packed_dz, pv_key, pv_position, track_phi):
    """packed_dxy, packed_dz: (N,) uint16; pv_key: (N,) index of the
    associated primary vertex or -1; pv_position: (V, 3); track_phi: (N,).
    Returns a dict: dxy_associated, dz_associated ((N,), w.r.t. the
    associated vertex) and point ((N, 3) reference point = the track's
    closest approach to that vertex in the transverse plane)."""
    pv_key = np.asarray(pv_key, dtype=np.int64)
    associated = pv_key >= 0
    pv_position = np.asarray(pv_position, dtype=np.float64).reshape(-1, 3)
    reference = np.zeros((pv_key.size, 3))
    if pv_position.shape[0]:
        reference = np.where(associated[:, None], pv_position[np.clip(pv_key, 0, pv_position.shape[0] - 1)], 0.0)
    dxy = half_to_float(packed_dxy) / IMPACT_PARAMETER_SCALE
    dz = np.where(associated, half_to_float(packed_dz) / IMPACT_PARAMETER_SCALE,
                  to_signed16(packed_dz) * UNASSOCIATED_DZ_SCALE / INT16_MAX)
    point = np.stack([reference[:, 0] - dxy * np.sin(track_phi), reference[:, 1] + dxy * np.cos(track_phi),
                      reference[:, 2] + dz], axis=1)
    return {'dxy_associated': dxy, 'dz_associated': dz, 'point': point}


def dxy_to_point(point, track_phi, target):
    """point: (N, 3) track reference points; track_phi: (N,); target: (3,).
    Returns (N,) transverse impact parameters w.r.t. target
    (PackedCandidate::dxy(const Point&))."""
    target = np.asarray(target, dtype=np.float64)
    return -(point[:, 0] - target[0]) * np.sin(track_phi) + (point[:, 1] - target[1]) * np.cos(track_phi)


def dz_to_point(point, track_phi, track_eta, target):
    """point: (N, 3); track_phi, track_eta: (N,); target: (3,). Returns (N,)
    longitudinal impact parameters w.r.t. target
    (PackedCandidate::dz(const Point&))."""
    target = np.asarray(target, dtype=np.float64)
    along = (point[:, 0] - target[0]) * np.cos(track_phi) + (point[:, 1] - target[1]) * np.sin(track_phi)
    return (point[:, 2] - target[2]) - along * np.sinh(track_eta)


def from_pv(pdg, pv_key, pv_association_quality, dz_associated, pv_z, pv_ndof):
    """pdg, pv_key, pv_association_quality, dz_associated: (N,); pv_z,
    pv_ndof: (V,). Returns (N,) int PackedCandidate::fromPV(0): 3 used in
    the fit of PV[0], 2 tight, 1 loose, 0 used in the fit of another
    vertex. Candidates without a vertex reference get 0."""
    pdg = np.asarray(pdg, dtype=np.int64)
    pv_key = np.asarray(pv_key, dtype=np.int64)
    pv_association_quality = np.asarray(pv_association_quality, dtype=np.int64)
    pv_z = np.asarray(pv_z, dtype=np.float64)
    pv_ndof = np.asarray(pv_ndof, dtype=np.float64)
    associated = (pv_key >= 0) & (pv_key < pv_z.size)
    safe_key = np.clip(pv_key, 0, max(pv_z.size - 1, 0))
    own_z = pv_z[safe_key] if pv_z.size else np.zeros(pv_key.size)
    own_ndof = pv_ndof[safe_key] if pv_ndof.size else np.zeros(pv_key.size)
    first_z = pv_z[0] if pv_z.size else 0.0
    dz_first = dz_associated + own_z - first_z
    result = np.full(pv_key.size, NO_PV, dtype=np.int64)
    loose = (pv_association_quality < QUALITY_USED_IN_FIT_LOOSE) | (own_ndof < MINIMUM_VERTEX_NDOF)
    result[loose] = PV_LOOSE
    btag = (pv_association_quality == QUALITY_COMPATIBILITY_BTAG) & (np.abs(dz_associated) > np.abs(dz_first))
    result[btag] = PV_TIGHT
    result[(pv_key == 0) | np.isin(np.abs(pdg), (13, 11))] = PV_TIGHT
    result[(pv_association_quality == QUALITY_USED_IN_FIT_TIGHT) & (pv_key == 0)] = PV_USED_IN_FIT
    result[~associated] = NO_PV
    return result


def element_index(i, j):
    """i <= j in 0..4. Returns the row of the element in the tables."""
    return ((9 - i) * i) // 2 + j


class CovarianceTables:
    """Mean covariance values and packing schemas of
    CovarianceParameterization (see build_covariance_tables.py)."""

    def __init__(self, pt_edges, eta_edges, hits_edges, reference, schema_ids, schema_method,
                 schema_target, schema_bits, schema_params):
        self.pt_edges, self.eta_edges, self.hits_edges = pt_edges, eta_edges, hits_edges
        self.reference = reference
        self.schema_row = {int(schema_id): row for row, schema_id in enumerate(schema_ids)}
        self.schema_method, self.schema_target = schema_method, schema_target
        self.schema_bits, self.schema_params = schema_bits, schema_params

    @classmethod
    def load(cls, path=DEFAULT_TABLES):
        """path: npz written by build_covariance_tables.py."""
        with np.load(path) as data:
            return cls(*(data[name] for name in (
                'pt_edges', 'eta_edges', 'hits_edges', 'reference', 'schema_ids', 'schema_method',
                'schema_target', 'schema_bits', 'schema_params')))

    def mean_value(self, i, j, pt, eta, n_hits, n_pixel_hits):
        """i <= j; pt, eta, n_hits, n_pixel_hits: (N,). Returns (N,) mean
        values (CovarianceParameterization::meanValue; TAxis::FindBin
        semantics, under- and overflow bins hold 0)."""
        hits = np.clip(n_hits, MINIMUM_HITS_FOR_REFERENCE, MAXIMUM_HITS_FOR_REFERENCE)
        pt_bin = np.searchsorted(self.pt_edges, pt, side='right')
        eta_bin = np.searchsorted(self.eta_edges, np.abs(eta), side='right')
        hits_bin = np.searchsorted(self.hits_edges, hits, side='right')
        has_pixel = (np.asarray(n_pixel_hits) > 0).astype(np.int64)
        return self.reference[element_index(i, j), has_pixel, pt_bin, eta_bin, hits_bin].astype(np.float64)

    def unpack_element(self, packed, schema, i, j, pt, eta, n_hits, n_pixel_hits):
        """packed: (N,) uint16; schema: (N,) packing schema of each
        candidate; i <= j; pt, eta: (N,) candidate momentum; n_hits,
        n_pixel_hits: (N,). Returns (N,) covariance element
        (CovarianceParameterization::unpack)."""
        packed = np.asarray(packed)
        pt = np.asarray(pt, dtype=np.float64)
        reference = self.mean_value(i, j, pt, eta, n_hits, n_pixel_hits)
        schema_used = np.where(reference == 0, 0, np.asarray(schema).astype(np.int64))
        if i == j and i in (0, 2):
            reference = np.where(schema_used == 0, 1.0 / np.where(pt > 0, pt * pt, 1.0), reference)
        column = element_index(i, j)
        result = np.zeros(packed.size)
        for schema_id in np.unique(schema_used):
            if int(schema_id) not in self.schema_row:
                raise ValueError(f'covariance packing schema {int(schema_id)} is not in the tables')
            row = self.schema_row[int(schema_id)]
            rows = schema_used == schema_id
            method, target = int(self.schema_method[row, column]), int(self.schema_target[row, column])
            first, second = self.schema_params[row, column]
            if method == METHOD_FLOAT16:
                unpacked = half_to_float(packed[rows]) / first
            elif method == METHOD_REDUCE_MANTISSA:
                unpacked = packed[rows].astype(np.float64)
            elif method == METHOD_LOG_PACK:
                signed = to_signed16(packed[rows])
                base = float(min(int(self.schema_bits[row, column]), MAXIMUM_LOG_BASE))
                magnitude = np.exp(first + np.abs(signed) / base * (second - first))
                unpacked = np.where(signed < 0, -magnitude, magnitude)
            elif method == METHOD_ZERO:
                unpacked = np.zeros(int(rows.sum()))
            else:
                unpacked = np.ones(int(rows.sum()))
            if target == TARGET_RATIO_TO_REFERENCE:
                unpacked = unpacked * reference[rows]
            elif target == TARGET_DIFFERENCE_TO_REFERENCE:
                unpacked = unpacked + reference[rows]
            result[rows] = unpacked
        if i == j:
            result = np.where(result == 0, EMPTY_DIAGONAL, result)
        return result


def unpack_covariance(tables, block, schema, pt, eta, n_hits, n_pixel_hits):
    """tables: CovarianceTables; block: {packed member name: (N,) uint16}
    holding the eight PACKED_COVARIANCE_ELEMENTS; schema, pt, eta, n_hits,
    n_pixel_hits: (N,). Returns {(i, j): (N,)} for the eight stored terms."""
    return {element: tables.unpack_element(block[member], schema, element[0], element[1], pt, eta, n_hits, n_pixel_hits)
            for member, element in PACKED_COVARIANCE_ELEMENTS.items()}


def pt_error(track_pt, track_eta, cov_qoverp_qoverp, cov_lambda_lambda):
    """track_pt, track_eta: (N,) pseudo-track momentum; cov_*: (N,).
    Returns (N,) pT errors (reco::TrackBase::ptError with the q/p-lambda
    covariance MiniAOD does not keep set to zero)."""
    momentum = track_pt * np.cosh(track_eta)
    longitudinal = track_pt * np.sinh(track_eta)
    return np.sqrt(np.maximum(track_pt ** 2 * momentum ** 2 * cov_qoverp_qoverp
                              + longitudinal ** 2 * cov_lambda_lambda, 0.0))


def unpack_candidates(block, pv_position, pv_ndof, tables):
    """block: {member: (N,)} one event's packed candidates as read from
    MiniAOD (miniaod_reader.CANDIDATE_MEMBERS); pv_position: (V, 3);
    pv_ndof: (V,); tables: CovarianceTables. Returns a dict of (N,) arrays
    (point: (N, 3); covariance: {(i, j): (N,)}, zero where the candidate
    has no track details)."""
    pdg = np.asarray(block['pdgId_']).astype(np.int64)
    momentum = unpack_momentum(block['packedPt_'], block['packedEta_'], block['packedPhi_'], block['packedM_'])
    track = unpack_track_momentum(momentum['pt'], momentum['eta'], momentum['phi'], block['packedDPhi_'],
                                  block['packedDEta_'], block['packedDTrkPt_'])
    hits = hit_counts(block['packedHits_'], block['packedLayers_'])
    flags = quality(block['qualityFlags_'])
    pv_position = np.asarray(pv_position, dtype=np.float64).reshape(-1, 3)
    key = np.asarray(block['pvRefKey_']).astype(np.int64)
    pv_key = np.where((key >= 0) & (key < pv_position.shape[0]), key, -1)
    vertex = unpack_vertex(block['packedDxy_'], block['packedDz_'], pv_key, pv_position, track['track_phi'])
    detailed = hits['has_track_details']
    covariance = {element: np.zeros(pdg.size) for element in PACKED_COVARIANCE_ELEMENTS.values()}
    if detailed.any():
        unpacked = unpack_covariance(
            tables, {member: np.asarray(block[member])[detailed] for member in PACKED_COVARIANCE_ELEMENTS},
            np.asarray(block['covarianceSchema_'])[detailed], momentum['pt'][detailed], momentum['eta'][detailed],
            hits['n_hits'][detailed], hits['n_pixel_hits'][detailed])
        for element, values in unpacked.items():
            covariance[element][detailed] = values
    degrees_of_freedom = hits['n_hits'] + hits['n_pixel_hits'] - 5
    result = {'pdg': pdg, 'charge': charge(pdg), 'pv_key': pv_key, 'covariance': covariance,
              'from_pv': from_pv(pdg, pv_key, flags['pv_association_quality'], vertex['dz_associated'],
                                 pv_position[:, 2], pv_ndof),
              'norm_chi2': np.where(degrees_of_freedom != 0, np.asarray(block['normalizedChi2_']).astype(np.float64), 0.0),
              'puppi_weight': np.asarray(block['packedPuppiweight_']).astype(np.float64) / PUPPI_WEIGHT_SCALE,
              'pt_error': pt_error(track['track_pt'], track['track_eta'], covariance[(0, 0)], covariance[(1, 1)]),
              'dxy_error': np.sqrt(covariance[(3, 3)]), 'dz_error': np.sqrt(covariance[(4, 4)])}
    for part in (momentum, track, hits, flags, vertex):
        result.update(part)
    return result
