"""Generator truth for the MiniAOD source (numpy only): the signal chain
B0 -> K*0 tau tau with tau -> 3 pi (pi0) nu and tau -> mu nu nu, and the
matching of generator particles to reconstructed tracks.

Plain readers cannot follow MiniAOD's mother links, so the chain is found
from shared production vertices of the pruned generator collection: the
three pions and a tau neutrino produced at one point are the hadronic tau's
daughters, a muon with its two neutrinos the other tau's, and both taus are
produced at the B decay vertex, where the K*0 is produced too. The K*
daughters exist only in the packed collection (no vertex stored there) and
are recovered as the opposite-sign K pi pair whose summed momentum equals the
K* momentum.
"""
from __future__ import annotations

import collections

import numpy as np

import miniaod_geometry as geometry

TAU_PDG, MUON_PDG, PION_PDG, KAON_PDG, KSTAR_PDG, B_ZERO_PDG = 15, 13, 211, 321, 313, 511
TAU_NEUTRINO_PDG, MUON_NEUTRINO_PDG, ELECTRON_PDG = 16, 14, 11
INT16_MAX = 32767.0
VERTEX_TOLERANCE = 1e-5
DIRECTION_TOLERANCE = 1e-4
KSTAR_MOMENTUM_TOLERANCE = 0.02
MATCH_DELTA_R = 0.05
MATCH_PT_RATIO = 0.5


def unpack_packed_generator(pdg, packed_pt, packed_rapidity, packed_phi, packed_mass):
    """pdg: (P,) int; packed_pt, packed_rapidity, packed_phi, packed_mass:
    (P,) uint16 as stored in pat::PackedGenParticle. Returns a dict of (P,)
    arrays: pdg, pt, eta, phi, mass."""
    def half(values):
        return np.ascontiguousarray(values, dtype=np.uint16).view(np.float16).astype(np.float64)

    def signed(values):
        return np.ascontiguousarray(values, dtype=np.uint16).view(np.int16).astype(np.float64)

    pt, mass = half(packed_pt), half(packed_mass)
    rapidity = signed(packed_rapidity) * 6.0 / INT16_MAX
    longitudinal = np.sqrt(pt ** 2 + mass ** 2) * np.sinh(rapidity)
    eta = np.arcsinh(longitudinal / np.where(pt > 0, pt, 1.0))
    return {'pdg': np.asarray(pdg, dtype=np.int64), 'pt': pt, 'eta': np.where(pt > 0, eta, 0.0),
            'phi': signed(packed_phi) * 3.2 / INT16_MAX, 'mass': mass}


def signal_chain(pdg, status, vertex, eta, phi):
    """pdg, status, eta, phi: (G,) pruned generator particles; vertex:
    (G, 3) their production vertices. Returns a dict: tau_pions ((3,)
    indices, empty when absent), tau_muon, hadronic_tau, muonic_tau, kstar,
    b_meson (the decaying signal B0 row) and b_meson_origin (its first row:
    differs for an oscillated B0), all -1 when absent; tau_decay_vertex,
    b_decay_vertex, true_primary_vertex ((3,) arrays or None)."""
    pdg = np.asarray(pdg, dtype=np.int64)
    status = np.asarray(status, dtype=np.int64)
    vertex = np.asarray(vertex, dtype=np.float64).reshape(-1, 3)
    chain = {'tau_pions': np.zeros(0, dtype=np.int64), 'tau_muon': -1, 'hadronic_tau': -1, 'muonic_tau': -1,
             'kstar': -1, 'b_meson': -1, 'b_meson_origin': -1, 'tau_decay_vertex': None, 'b_decay_vertex': None,
             'true_primary_vertex': None}
    groups = collections.defaultdict(list)
    for index in np.flatnonzero(status == 1):
        groups[tuple(vertex[index].tolist())].append(int(index))
    for position, members in groups.items():
        kinds = collections.Counter(np.abs(pdg[members]).tolist())
        if (kinds[PION_PDG] == 3 and kinds[TAU_NEUTRINO_PDG] >= 1
                and kinds[MUON_PDG] == 0 and kinds[ELECTRON_PDG] == 0):
            chain['tau_pions'] = np.array([index for index in members if abs(pdg[index]) == PION_PDG], dtype=np.int64)
            chain['tau_decay_vertex'] = np.array(position)
        if kinds[MUON_PDG] == 1 and kinds[MUON_NEUTRINO_PDG] >= 1 and kinds[TAU_NEUTRINO_PDG] >= 1:
            chain['tau_muon'] = [index for index in members if abs(pdg[index]) == MUON_PDG][0]
    taus = np.flatnonzero(np.abs(pdg) == TAU_PDG)
    if chain['tau_pions'].size == 3 and taus.size:
        # tau- carries pdgId +15, so its pdgId sign is opposite to its charge
        pion_charge = int(np.sum(np.sign(pdg[chain['tau_pions']])))
        hadronic = taus[np.sign(pdg[taus]) == -pion_charge]
        muonic = taus[np.sign(pdg[taus]) == pion_charge]
        if hadronic.size:
            chain['hadronic_tau'] = int(hadronic[-1])
            chain['b_decay_vertex'] = vertex[hadronic[-1]].copy()
        if muonic.size:
            chain['muonic_tau'] = int(muonic[-1])
    if chain['b_decay_vertex'] is not None:
        kstars = [int(index) for index in np.flatnonzero(np.abs(pdg) == KSTAR_PDG)
                  if np.allclose(vertex[index], chain['b_decay_vertex'], atol=VERTEX_TOLERANCE)]
        if kstars:
            chain['kstar'] = kstars[0]
    b_mesons = np.flatnonzero(np.abs(pdg) == B_ZERO_PDG)
    if b_mesons.size:
        chain['true_primary_vertex'] = vertex[b_mesons[0]].copy()
        signal = b_mesons
        if chain['b_decay_vertex'] is not None:
            # the event's other b hadron may be a B0 too: the signal B0 rows
            # (an oscillated one appears twice) fly towards the B decay vertex
            flight = chain['b_decay_vertex'] - chain['true_primary_vertex']
            length = np.linalg.norm(flight)
            if length > 0:
                cosine = geometry.unit_direction(np.asarray(eta)[b_mesons], np.asarray(phi)[b_mesons]) @ (flight / length)
                signal = b_mesons[cosine > cosine.max() - DIRECTION_TOLERANCE]
        chain['b_meson'] = int(signal[-1])
        chain['b_meson_origin'] = int(signal[0])
    return chain


def kstar_daughters(kstar_pt, kstar_eta, kstar_phi, packed):
    """kstar_pt, kstar_eta, kstar_phi: the pruned K* momentum; packed:
    unpack_packed_generator output. Returns (kaon index, pion index) into
    the packed collection for the opposite-sign pair whose summed momentum
    matches the K*, or None."""
    def momentum(pt, eta, phi):
        return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], axis=-1)

    kaons = np.flatnonzero(np.abs(packed['pdg']) == KAON_PDG)
    pions = np.flatnonzero(np.abs(packed['pdg']) == PION_PDG)
    if not kaons.size or not pions.size:
        return None
    target = momentum(np.float64(kstar_pt), np.float64(kstar_eta), np.float64(kstar_phi))
    all_momenta = momentum(packed['pt'], packed['eta'], packed['phi'])
    total = all_momenta[kaons][:, None, :] + all_momenta[pions][None, :, :]
    mismatch = np.linalg.norm(total - target[None, None, :], axis=2) / np.linalg.norm(target)
    opposite = np.sign(packed['pdg'][kaons])[:, None] != np.sign(packed['pdg'][pions])[None, :]
    mismatch = np.where(opposite, mismatch, np.inf)
    kaon, pion = np.unravel_index(int(np.argmin(mismatch)), mismatch.shape)
    if mismatch[kaon, pion] >= KSTAR_MOMENTUM_TOLERANCE:
        return None
    return int(kaons[kaon]), int(pions[pion])


def match_tracks_to_generator(track_pt, track_eta, track_phi, track_charge,
                              gen_pt, gen_eta, gen_phi, gen_charge,
                              max_delta_r=MATCH_DELTA_R, max_pt_ratio=MATCH_PT_RATIO):
    """track_*: (T,) tracks; gen_*: (G,) generator particles. A pair is
    allowed when the charges agree, deltaR < max_delta_r and
    |track pT / generator pT - 1| < max_pt_ratio; pairs are assigned
    one-to-one in order of increasing deltaR. Returns ((G,) track index or
    -1, (G,) deltaR of the assigned pair or -1)."""
    track_pt, track_eta, track_phi = (np.asarray(value, dtype=np.float64) for value in (track_pt, track_eta, track_phi))
    gen_pt, gen_eta, gen_phi = (np.asarray(value, dtype=np.float64) for value in (gen_pt, gen_eta, gen_phi))
    matched = np.full(gen_pt.size, -1, dtype=np.int64)
    matched_delta_r = np.full(gen_pt.size, -1.0)
    if not track_pt.size or not gen_pt.size:
        return matched, matched_delta_r
    distance = geometry.delta_r(gen_eta[:, None], gen_phi[:, None], track_eta[None, :], track_phi[None, :])
    allowed = ((distance < max_delta_r)
               & (np.asarray(gen_charge)[:, None] == np.asarray(track_charge)[None, :])
               & (np.abs(track_pt[None, :] / gen_pt[:, None] - 1.0) < max_pt_ratio))
    gen_rows, track_rows = np.nonzero(allowed)
    order = np.argsort(distance[gen_rows, track_rows], kind='stable')
    used_tracks = set()
    for position in order:
        gen_row, track_row = int(gen_rows[position]), int(track_rows[position])
        if matched[gen_row] >= 0 or track_row in used_tracks:
            continue
        matched[gen_row] = track_row
        matched_delta_r[gen_row] = distance[gen_row, track_row]
        used_tracks.add(track_row)
    return matched, matched_delta_r
