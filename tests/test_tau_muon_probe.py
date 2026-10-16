import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from utils import tau_muon_proxy as proxy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARD = os.path.join(ROOT, 'data', 'low-pt', 'eval', 'eval_000.parquet')
COUPLE_DUMP = os.path.join(ROOT, 'data', 'dumps', 'couples_k125_dump.parquet')
CANDIDATES = os.path.join(ROOT, 'data', 'thesis_files', 'triplet_rank', 'candidates_val.parquet')
N_EVENTS = 300

data_required = pytest.mark.skipif(
    not (os.path.exists(SHARD) and os.path.exists(COUPLE_DUMP)),
    reason='real eval shard / couple dump not present')
candidates_required = pytest.mark.skipif(
    not os.path.exists(CANDIDATES), reason='thesis-era candidates_val.parquet not present')


@pytest.fixture(scope='module')
def events():
    table = pq.ParquetFile(SHARD).read_row_group(0, columns=proxy.SHARD_COLUMNS).slice(0, N_EVENTS)
    return proxy.events_from_table(table)


def _manual_relations(event, muon_index):
    gt = proxy.gt_members(event['track_label'])
    pt, eta, phi = (event['track_pt'][gt], event['track_eta'][gt], event['track_phi'][gt])
    px = (pt * np.cos(phi)).sum()
    py = (pt * np.sin(phi)).sum()
    pz = (pt * np.sinh(eta)).sum()
    energy = np.sqrt(pt ** 2 * np.cosh(eta) ** 2 + proxy.PION_MASS ** 2).sum()
    mu_pt, mu_eta, mu_phi = (event['muon_pt'][muon_index], event['muon_eta'][muon_index],
                             event['muon_phi'][muon_index])
    mu_px, mu_py, mu_pz = mu_pt * np.cos(mu_phi), mu_pt * np.sin(mu_phi), mu_pt * np.sinh(mu_eta)
    mu_energy = np.sqrt(mu_pt ** 2 * np.cosh(mu_eta) ** 2 + proxy.MUON_MASS ** 2)
    total_energy = energy + mu_energy
    mass = np.sqrt(max(total_energy ** 2 - (px + mu_px) ** 2 - (py + mu_py) ** 2 - (pz + mu_pz) ** 2, 0.0))
    system_p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    system_eta = np.arctanh(pz / system_p)
    system_phi = np.arctan2(py, px)
    delta_phi = (mu_phi - system_phi + np.pi) % (2 * np.pi) - np.pi
    delta_r = np.sqrt((mu_eta - system_eta) ** 2 + delta_phi ** 2)
    dz_gap = abs(event['muon_dz'][muon_index] - np.median(event['track_dz'][gt]))
    charge_product = event['muon_charge'][muon_index] * event['track_charge'][gt].sum()
    return dict(delta_r=delta_r, mass=mass, dz_gap=dz_gap, charge_product=charge_product)


@data_required
def test_events_from_table_shapes(events):
    assert len(events) == N_EVENTS
    for event in events:
        assert len(proxy.gt_members(event['track_label'])) == 3
        assert event['muon_pt'].shape == event['muon_soft'].shape == event['muon_dz'].shape
        assert len(event['identity']) == 5


@data_required
def test_gt_relations_match_manual_computation(events):
    checked = 0
    for event in events:
        if len(event['muon_pt']) == 0:
            continue
        relations = proxy.gt_relations(event)
        for muon_index in range(len(event['muon_pt'])):
            manual = _manual_relations(event, muon_index)
            for name, value in manual.items():
                assert relations[name][0, muon_index] == pytest.approx(value, rel=1e-4, abs=1e-6), name
        checked += 1
        if checked >= 20:
            break
    assert checked == 20


@data_required
def test_proxy_selection_obeys_cuts_and_variant_nesting(events):
    passing = {name: 0 for name in proxy.PROXY_VARIANTS}
    for event in events:
        relations = proxy.gt_relations(event)
        for name, variant in proxy.PROXY_VARIANTS.items():
            index = proxy.select_proxy_muon(relations, event['muon_soft'], name)
            if index < 0:
                continue
            passing[name] += 1
            assert event['muon_soft'][index] == 1
            assert relations['charge_product'][0, index] < 0
            if variant['use_dz']:
                assert relations['dz_gap'][0, index] < proxy.DZ_MAX
            if variant['use_dr']:
                assert relations['delta_r'][0, index] < proxy.DR_MAX
            if variant['use_mass']:
                assert relations['mass'][0, index] < proxy.B_MASS
        full = proxy.select_proxy_muon(relations, event['muon_soft'], 'full')
        if full >= 0:
            for name in ('no_dz', 'no_dr', 'no_mass'):
                assert proxy.select_proxy_muon(relations, event['muon_soft'], name) >= 0
    # Roughly one selected event in six on the selected population (calibrated on the 85k batch).
    assert 0.05 < passing['full'] / N_EVENTS < 0.35
    assert passing['no_dr'] >= passing['full']


@data_required
def test_policy_indices_and_gather(events):
    event = next(event for event in events if len(event['muon_pt']) >= 2)
    gt = proxy.gt_members(event['track_label'])
    members = np.array([[gt[0], gt[1]], [gt[0], gt[2]], [gt[1], gt[2]], [0, 1]])
    system = proxy.system_from_members(event, members)
    relations = proxy.muon_relations(event, system)
    assert relations['delta_r'].shape == (4, len(event['muon_pt']))
    for policy in ('min_dr', 'min_dz'):
        indices = proxy.policy_indices(relations, event['muon_soft'], policy)
        assert indices.shape == (4,)
        features = proxy.gather(relations, indices)
        for row, index in enumerate(indices):
            if index < 0:
                assert np.isnan(features['delta_r'][row])
            else:
                assert features['delta_r'][row] == relations['delta_r'][row, index]
                assert event['muon_soft'][index] == 1
    oracle = proxy.policy_indices(relations, event['muon_soft'], 'oracle', oracle_index=-1)
    assert (oracle == -1).all()
    assert np.isnan(proxy.gather(relations, oracle)['mass']).all()


@data_required
def test_couple_rows_from_dump(events):
    from scripts.python import tau_muon_probe as probe
    dump_rows = probe.load_couple_dump(COUPLE_DUMP, {event['identity'] for event in events})
    matched = [event for event in events if event['identity'] in dump_rows]
    assert len(matched) >= 0.9 * N_EVENTS
    seen_gt_present = 0
    for event in matched[:100]:
        rows = probe.build_couple_rows(event, dump_rows[event['identity']], top_k2=125, oracle_index=-1)
        n_rows = len(rows['label'])
        assert n_rows == len(dump_rows[event['identity']]['couples'])
        gt = set(proxy.gt_members(event['track_label']).tolist())
        for row, (first, second) in enumerate(dump_rows[event['identity']]['couples']):
            assert bool(rows['label'][row]) == (first in gt and second in gt)
        assert np.all(np.diff(rows['baseline_score']) <= 1e-6)
        if rows['label'].any():
            seen_gt_present += 1
            assert rows['gt_rank'] == int(np.flatnonzero(rows['label'])[0])
        else:
            assert rows['gt_rank'] < 0
        for policy in probe.POLICIES:
            assert rows[f'{policy}__delta_r'].shape == (n_rows,)
    assert seen_gt_present > 50


@data_required
@candidates_required
def test_triplet_rows_from_candidates(events):
    from scripts.python import tau_muon_probe as probe
    candidate_rows = probe.load_candidate_rows(CANDIDATES, start=0, count=len(events))
    assert len(candidate_rows) == len(events)
    events_with_gt_rows = 0
    for event, candidate in zip(events[:50], candidate_rows[:50]):
        assert candidate['n_tracks'] == len(event['track_pt'])
        rows = probe.build_triplet_rows(event, candidate, impostor_cap=50, oracle_index=-1)
        assert len(rows['label']) <= 50 + int(rows['label'].sum())
        gt = set(proxy.gt_members(event['track_label']).tolist())
        if candidate['recon']:
            assert set(candidate['gt']) == gt
        if candidate['is_gt'].any():
            events_with_gt_rows += 1
            # Every GT row is kept; the GT triplet is reachable from each of its couples.
            assert rows['label'].sum() == candidate['is_gt'].sum() <= 3
            assert rows['gt_rank'] >= 0
            for members in candidate['cand'][candidate['is_gt']]:
                assert set(members.tolist()) == gt
        else:
            assert rows['gt_rank'] < 0
        assert np.all(np.diff(rows['baseline_score'][~rows['label'].astype(bool)]) <= 1e-6)
    assert events_with_gt_rows >= 10


@data_required
def test_run_probe_small_couples():
    from scripts.python import tau_muon_probe as probe
    result = probe.run(level='couples', shard_glob=SHARD, source=COUPLE_DUMP, variant='full',
                       max_events=N_EVENTS, top_k2=125, impostor_cap=50, n_folds=2, k_values=(10, 50, 100))
    assert result['n_events'] == N_EVENTS
    assert result['n_subset'] + result['n_complement'] == result['n_events_with_gt_in_list']
    for group in ('subset', 'complement'):
        block = result[group]
        assert 0.0 <= block['baseline']['c_at_k']['100'] <= 1.0
        for policy in probe.POLICIES:
            assert 0.0 <= block['feature_auc'][policy]['delta_r'] <= 1.0 or np.isnan(block['feature_auc'][policy]['delta_r'])
        for policy in probe.PROBE_POLICIES:
            assert 'c_at_k' in block['probe'][policy]
        assert 'none' not in block['feature_auc']


@data_required
def test_gt_rank_is_tie_aware():
    from scripts.python import tau_muon_probe as probe
    scores = np.array([0.9, 0.9, 0.9, 0.5, 0.1])
    labels = np.array([0, 1, 0, 0, 0])
    assert probe.gt_rank(scores, labels) == 1.0
    assert probe.gt_rank(np.array([0.3, 0.9]), np.array([1, 0])) == 1.0
    assert probe.gt_rank(np.array([0.3, 0.9]), np.array([0, 0])) == -1.0
