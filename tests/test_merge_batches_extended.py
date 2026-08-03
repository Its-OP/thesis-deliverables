"""Merge smoke on real data: treat the two sample ROOT files as the
microbatches of one batch and run merge_batches.merge_batch end-to-end.
Catches uproot-writing issues (jagged Muon/SV blocks, bool branches)
before anything reaches lxplus."""
from __future__ import annotations

import os
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

MERGE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'condor', 'merge_root_batches',
))
if MERGE_DIR not in sys.path:
    sys.path.insert(0, MERGE_DIR)

from merge_batches import KEEP_BRANCHES, merge_batch, parse_microbatch_id

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'part', 'data', 'low-pt',
))
EXAMPLE = os.path.join(DATA_DIR, 'example_root.root')
DITAUS = os.path.join(DATA_DIR, 'step_MINI_10_nano_ditaus_mc.root')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(EXAMPLE) and os.path.exists(DITAUS)),
    reason='sample ROOT files not present',
)


def test_parse_microbatch_id():
    assert parse_microbatch_id(DITAUS) == 10
    assert parse_microbatch_id(EXAMPLE) == -1


@pytest.fixture(scope='module')
def merged_path(tmp_path_factory):
    output = str(tmp_path_factory.mktemp('merged') / 'merged_ext_batch1.root')
    merge_batch(1, [EXAMPLE, DITAUS], output)
    return output


def test_merged_entry_count_and_order(merged_path):
    tree = uproot.open(merged_path)['Events']
    assert tree.num_entries == 193
    # Order preserved: first file's events first.
    merged_event = tree['event'].array(entry_stop=100)
    source_event = uproot.open(EXAMPLE)['Events']['event'].array()
    assert ak.to_list(merged_event) == ak.to_list(source_event)


def test_merged_keeps_extended_branches(merged_path):
    tree = uproot.open(merged_path)['Events']
    available = set(tree.keys())
    source_available = set(uproot.open(EXAMPLE)['Events'].keys())
    expected = {b for b in KEEP_BRANCHES if b in source_available}
    missing = expected - available
    assert not missing, f'branches lost at merge: {sorted(missing)}'
    for branch in ('Muon_pt', 'Muon_softId', 'SV_dlenSig', 'OtherPV_z',
                   'Track_covDxyDxy', 'Track_isMatchedToMuon',
                   'Track_trackFromB', 'PV_npvs'):
        assert branch in available, branch


def test_merged_source_ids(merged_path):
    tree = uproot.open(merged_path)['Events']
    batch_ids = ak.to_numpy(tree['source_batch_id'].array())
    micro_ids = ak.to_numpy(tree['source_microbatch_id'].array())
    assert set(batch_ids.tolist()) == {1}
    assert micro_ids[:100].tolist() == [-1] * 100
    assert micro_ids[100:].tolist() == [10] * 93


def test_merged_jagged_content_roundtrip(merged_path):
    tree = uproot.open(merged_path)['Events']
    merged_muon_pt = tree['Muon_pt'].array(entry_stop=100)
    source_muon_pt = uproot.open(EXAMPLE)['Events']['Muon_pt'].array()
    assert ak.to_list(merged_muon_pt) == ak.to_list(source_muon_pt)
    merged_soft_id = tree['Muon_softId'].array(entry_stop=100)
    source_soft_id = uproot.open(EXAMPLE)['Events']['Muon_softId'].array()
    assert ak.to_list(ak.values_astype(merged_soft_id, np.int32)) == \
        ak.to_list(ak.values_astype(source_soft_id, np.int32))
