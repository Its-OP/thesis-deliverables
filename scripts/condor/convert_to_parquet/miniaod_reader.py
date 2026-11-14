"""Raw reader of CMS MiniAOD files without CMSSW.

MiniAOD stores a few large collections split into one branch per data member
(packed PF candidates, primary vertices, generator particles, trigger
objects); uproot reads those and the beam spot. Everything else is stored
object-wise and is read member by member through plain ROOT's TTree::Draw
on emulated classes (lost tracks, secondary vertices, event ids, trigger
results, pileup summary, vertex scores, the candidate-to-generator
associations).

The trigger objects must NOT be read through ROOT: the class holds a
std::string, ROOT relocates emulated collection elements byte-wise, and a
libstdc++ string keeps a pointer into itself, so the element destructor
frees an invalid pointer (segmentation fault on Linux; libc++ on macOS is
immune). The classes still read through ROOT hold no string inside a
collection element.

Not readable either way: slimmedMuons and the generator mother links.

MINIAOD_READER_TRACE=1 prints each product before it is read (stderr), to
locate a crash inside ROOT.

All per-object quantities are returned as awkward arrays jagged per event,
still in MiniAOD's packed integer form; packed_candidate.py unpacks them.
"""
from __future__ import annotations

import os
import re
import sys

import awkward as ak
import numpy as np
import uproot

PF_PREFIX = 'patPackedCandidates_packedPFCandidates__PAT.obj.'
LOST_PREFIX = 'patPackedCandidates_lostTracks__PAT.obj.'
PV_PREFIX = 'recoVertexs_offlineSlimmedPrimaryVertices__PAT.obj.'
GEN_PREFIX = 'recoGenParticles_prunedGenParticles__PAT.obj.m_state.'
PACKED_GEN_PREFIX = 'patPackedGenParticles_packedGenParticles__PAT.obj.'
SV_PREFIX = 'recoVertexCompositePtrCandidates_slimmedSecondaryVertices__PAT.obj.'
TRIGGER_OBJECT_PREFIX = 'patTriggerObjectStandAlones_slimmedPatTrigger__PAT.obj.'
PF_MATCH = 'recoGenParticlesedmAssociation_packedPFCandidateToGenAssociation__PAT.obj.values_'
LOST_MATCH = 'recoGenParticlesedmAssociation_lostTracksToGenAssociation__PAT.obj.values_'
TRIGGER_STATUS = 'edmTriggerResults_TriggerResults__HLT.obj.paths_.status_'
PILEUP_TRUE = 'PileupSummaryInfos_slimmedAddPileupInfo__PAT.obj.TrueNumInteractions_'
PV_SCORE = 'floatedmValueMap_offlineSlimmedPrimaryVertices__PAT.obj.values_'
BEAM_SPOT = 'recoBeamSpot_offlineBeamSpot__RECO.obj'

COVARIANCE_MEMBERS = tuple('packedCovariance_.' + name for name in (
    'dptdpt', 'detadeta', 'dphidphi', 'dxydxy', 'dzdz', 'dxydz', 'dlambdadz', 'dphidxy'))
CANDIDATE_MEMBERS = ('packedPt_', 'packedEta_', 'packedPhi_', 'packedM_', 'packedDxy_', 'packedDz_',
                     'packedDPhi_', 'packedDEta_', 'packedDTrkPt_', 'pdgId_', 'qualityFlags_', 'pvRefKey_',
                     'packedHits_', 'packedLayers_', 'normalizedChi2_', 'covarianceSchema_',
                     'packedPuppiweight_') + COVARIANCE_MEMBERS
PV_MEMBERS = ('position_.fCoordinates.fX', 'position_.fCoordinates.fY', 'position_.fCoordinates.fZ',
              'chi2_', 'ndof_', 'covariance_[10]')
GEN_MEMBERS = ('pdgId_', 'status_', 'p4Polar_.fCoordinates.fPt', 'p4Polar_.fCoordinates.fEta',
               'p4Polar_.fCoordinates.fPhi', 'p4Polar_.fCoordinates.fM', 'vertex_.fCoordinates.fX',
               'vertex_.fCoordinates.fY', 'vertex_.fCoordinates.fZ')
PACKED_GEN_MEMBERS = ('pdgId_', 'packedPt_', 'packedY_', 'packedPhi_', 'packedM_')
SV_MEMBERS = ('m_state.vertex_.fCoordinates.fX', 'm_state.vertex_.fCoordinates.fY',
              'm_state.vertex_.fCoordinates.fZ', 'chi2_', 'ndof_', 'm_state.p4Polar_.fCoordinates.fPt',
              'm_state.p4Polar_.fCoordinates.fEta', 'm_state.p4Polar_.fCoordinates.fPhi',
              'm_state.p4Polar_.fCoordinates.fM', 'm_state.qx3_', 'dau@.size()')
TRIGGER_OBJECT_MEMBERS = ('m_state.p4Polar_.fCoordinates.fPt', 'm_state.p4Polar_.fCoordinates.fEta',
                          'm_state.p4Polar_.fCoordinates.fPhi', 'pathIndices_', 'pathLastFilterAccepted_')
SPLIT_BLOCKS = {'pf': (PF_PREFIX, CANDIDATE_MEMBERS), 'pv': (PV_PREFIX, PV_MEMBERS),
                'gen': (GEN_PREFIX, GEN_MEMBERS), 'packed_gen': (PACKED_GEN_PREFIX, PACKED_GEN_MEMBERS),
                'trigger_objects': (TRIGGER_OBJECT_PREFIX, TRIGGER_OBJECT_MEMBERS)}
VERTEX_COVARIANCE_SIZE = 10
DRAW_ESTIMATE = 20_000_000
EXPRESSIONS_PER_DRAW = 3
PATH_ACCEPTED_STATE = 1
PATH_STATE_MASK = 3


def is_miniaod_tree(tree):
    """tree: uproot Events tree. True when it holds the packed PF candidate
    product of MiniAOD."""
    return PF_PREFIX.rstrip('.').rsplit('.', 1)[0] + '.' in tree.keys(recursive=False)


def is_miniaod(path):
    """path: ROOT file. True when its Events tree holds the packed PF
    candidate product of MiniAOD."""
    with uproot.open(path) as root_file:
        return 'Events' in root_file and is_miniaod_tree(root_file['Events'])


def _trace(label):
    if os.environ.get('MINIAOD_READER_TRACE'):
        print(f'miniaod_reader: reading {label}', file=sys.stderr, flush=True)


def read_split_products(path):
    """path: MiniAOD file. Returns {'n_events': int, block: {member: awkward
    array jagged per event}} for the blocks pf, pv, gen, packed_gen, plus
    trigger_objects (pt, eta, phi per object; path_indices and
    path_accepted per object and path) and beam_spot ((n_events, 3))."""
    _trace('split products (uproot)')
    events = uproot.open(path)['Events']
    names = [prefix + member for prefix, members in SPLIT_BLOCKS.values() for member in members]
    data = events.arrays(names)
    products = {'n_events': int(events.num_entries)}
    for block, (prefix, members) in SPLIT_BLOCKS.items():
        products[block] = {member: data[prefix + member] for member in members}
    objects = products['trigger_objects']
    products['trigger_objects'] = {
        'pt': objects['m_state.p4Polar_.fCoordinates.fPt'], 'eta': objects['m_state.p4Polar_.fCoordinates.fEta'],
        'phi': objects['m_state.p4Polar_.fCoordinates.fPhi'], 'path_indices': objects['pathIndices_'],
        'path_accepted': objects['pathLastFilterAccepted_']}
    position = events[BEAM_SPOT].array()['position_']['fCoordinates']
    products['beam_spot'] = np.stack([ak.to_numpy(position[axis]) for axis in ('fX', 'fY', 'fZ')], axis=1)
    return products


def hlt_path_names(path, expected_length=None):
    """path: EDM file. Returns the ordered HLT path names stored in the
    file's configuration blobs: the `@trigger_paths` list of the requested
    length, else the longest one."""
    tree = uproot.open(path)['ParameterSets']
    text = b''.join(bytes(branch.basket(index).data) for branch in tree.branches
                    for index in range(branch.num_baskets)).decode('latin-1')
    candidates = []
    for match in re.finditer(r'@trigger_paths=\+z\(\{([^}]*)\}\)', text):
        cleaned = re.sub(r'[^\x21-\x7e]+', ' ', match.group(1)).replace(',', ' ')
        candidates.append([name for name in cleaned.split() if name])
    if not candidates:
        raise ValueError(f'{path}: no trigger path list in the stored configuration')
    if expected_length is not None:
        exact = [names for names in candidates if len(names) == expected_length]
        if not exact:
            raise ValueError(f'{path}: no trigger path list of length {expected_length}')
        return exact[0]
    return max(candidates, key=len)


def _draw(tree, expressions):
    """tree: ROOT TTree; expressions: per-instance expressions over one
    collection. Returns ((N,) int64 entry of each instance, {expression:
    (N,) float64})."""
    columns = {}
    entry = None
    for start in range(0, len(expressions), EXPRESSIONS_PER_DRAW):
        chunk = expressions[start:start + EXPRESSIONS_PER_DRAW]
        count = tree.Draw('Entry$:' + ':'.join(chunk), '', 'goff')
        if count < 0:
            raise RuntimeError(f'TTree::Draw failed for {chunk}')
        buffers = (tree.GetV1(), tree.GetV2(), tree.GetV3(), tree.GetV4())
        arrays = [np.frombuffer(buffers[slot], dtype=np.float64, count=count).copy() if count else np.zeros(0)
                  for slot in range(len(chunk) + 1)]
        if entry is None:
            entry = arrays[0].astype(np.int64)
        elif arrays[0].size != entry.size:
            raise RuntimeError(f'TTree::Draw returned misaligned columns for {chunk}')
        for expression, values in zip(chunk, arrays[1:]):
            columns[expression] = values
    return entry, columns


def _jagged(entry, values, n_events):
    """entry: (N,) event of each value; values: (N,). Returns an awkward
    array of n_events lists."""
    return ak.unflatten(values, np.bincount(entry, minlength=n_events))


def _draw_block(tree, prefix, members, n_events):
    entry, columns = _draw(tree, [prefix + member for member in members])
    return {member: _jagged(entry, columns[prefix + member], n_events) for member in members}


def _draw_single(tree, expression, n_events):
    entry, columns = _draw(tree, [expression])
    return _jagged(entry, columns[expression], n_events)


def read_unsplit_products(path, n_events):
    """path: MiniAOD file; n_events: its entry count. Returns a dict:
    lost (same members as the pf block), pf_match / lost_match (generator
    index per candidate, -1 unmatched), trigger_status (per event, one value
    per HLT path), event (run, luminosityBlock, event, bunchCrossing as
    (n_events,) int64), pileup ((n_events,) true interactions), sv (members
    + covariance (n_sv, 10)), pv_score."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise OSError(f'{path}: ROOT cannot open the file')
    tree = root_file.Get('Events')
    tree.SetEstimate(DRAW_ESTIMATE)
    products = {}
    for name, read in (
            ('lost', lambda: _draw_block(tree, LOST_PREFIX, CANDIDATE_MEMBERS, n_events)),
            ('pf_match', lambda: _draw_single(tree, PF_MATCH, n_events)),
            ('lost_match', lambda: _draw_single(tree, LOST_MATCH, n_events)),
            ('trigger_status', lambda: _draw_single(tree, TRIGGER_STATUS, n_events)),
            ('pv_score', lambda: _draw_single(tree, PV_SCORE, n_events)),
            ('sv', lambda: _draw_block(tree, SV_PREFIX, SV_MEMBERS, n_events))):
        _trace(f'{name} (ROOT)')
        products[name] = read()
    _trace('sv covariance, event ids, pileup (ROOT)')
    sv_counts = ak.num(products['sv']['chi2_'])
    covariance = _draw_single(tree, SV_PREFIX + 'covariance_', n_events)
    products['sv']['covariance'] = ak.unflatten(
        ak.flatten(covariance), VERTEX_COVARIANCE_SIZE) if ak.sum(sv_counts) else ak.Array(np.zeros((0, VERTEX_COVARIANCE_SIZE)))
    products['sv']['covariance'] = ak.unflatten(products['sv']['covariance'], sv_counts)

    entry, columns = _draw(tree, ['EventAuxiliary.id_.run_', 'EventAuxiliary.id_.luminosityBlock_',
                                  'EventAuxiliary.id_.event_'])
    bunch_entry, bunch_columns = _draw(tree, ['EventAuxiliary.bunchCrossing_'])
    products['event'] = {
        'run': columns['EventAuxiliary.id_.run_'].astype(np.int64),
        'luminosityBlock': columns['EventAuxiliary.id_.luminosityBlock_'].astype(np.int64),
        'event': columns['EventAuxiliary.id_.event_'].astype(np.int64),
        'bunchCrossing': bunch_columns['EventAuxiliary.bunchCrossing_'].astype(np.int64),
    }
    pileup = _draw_single(tree, PILEUP_TRUE, n_events)
    products['pileup'] = ak.to_numpy(ak.fill_none(ak.firsts(pileup), np.nan)).astype(np.float64)
    _trace('closing the file (ROOT)')
    root_file.Close()
    return products


def read_miniaod_file(path):
    """path: MiniAOD file. Returns the split and unsplit products merged in
    one dict (see read_split_products and read_unsplit_products), plus
    hlt_paths: the ordered HLT path names."""
    products = read_split_products(path)
    products.update(read_unsplit_products(path, products['n_events']))
    n_paths = int(ak.num(products['trigger_status'])[0]) if products['n_events'] else None
    products['hlt_paths'] = hlt_path_names(path, n_paths)
    return products


def accepted_paths(trigger_status):
    """trigger_status: awkward (events, paths) raw HLTPathStatus words.
    Returns a boolean awkward array: path accepted."""
    return (ak.values_astype(trigger_status, np.int64) & PATH_STATE_MASK) == PATH_ACCEPTED_STATE
