"""Truth-based primary-vertex check on a raw nano file that carries the
GenPart and PVtx blocks but not the abcbatch analysis blocks (e.g. the old
production's step_MINI_10): stored PV vs the true PV, nearest PVtx, the dz
proxy, and the selection buckets. Companion of survey_abcbatch_ntuples for
the cross-production comparison; applies no converter quality cleaning.

Usage:
    python -m scripts.python.pv_truth_check \\
        --input data/thesis_files/step_MINI_10_nano_ditaus_mc.root
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import awkward as ak
import numpy as np
import uproot

from scripts.python import survey_abcbatch_ntuples as survey

BRANCHES = [
    'nTrack', 'Track_pt', 'Track_pdgId', 'Track_dz', 'Track_trackFromTau',
    'GenPart_pdgId', 'GenPart_status', 'GenPart_genPartIdxMother', 'GenPart_pt',
    'GenPart_vx', 'GenPart_vy', 'GenPart_vz',
    'PV_x', 'PV_y', 'PV_z', 'nPVtx', 'PVtx_vx', 'PVtx_vy', 'PVtx_vz', 'PVtx_score',
]
POPULATIONS = {
    'all': lambda row: True,
    'track_events': lambda row: row['n_tracks'] > 0,
    'selected': lambda row: row['bucket'] == 'selected',
}


def event_row(events, entry):
    """events: awkward record array of BRANCHES; entry: int. Returns the
    bucket, track count, pv_truth dict (None without a true PV) and the
    median GT dz (None without three GT pions)."""
    track_pt = ak.to_numpy(events['Track_pt'][entry])
    bucket, gt_indices = survey.select_event(
        track_pt, ak.to_numpy(events['Track_pdgId'][entry]),
        ak.to_numpy(events['Track_trackFromTau'][entry]))
    gen_vertex = np.stack([ak.to_numpy(events[f'GenPart_v{axis}'][entry])
                           for axis in 'xyz'], axis=1)
    truth = survey.gen_truth(
        ak.to_numpy(events['GenPart_pdgId'][entry]),
        ak.to_numpy(events['GenPart_genPartIdxMother'][entry]),
        ak.to_numpy(events['GenPart_status'][entry]),
        ak.to_numpy(events['GenPart_pt'][entry]), gen_vertex)
    row = {'bucket': bucket, 'n_tracks': int(track_pt.size), 'pv': None, 'gt_median_dz': None}
    if truth['true_pv'] is None:
        return row
    pv_xyz = np.array([float(events[f'PV_{axis}'][entry]) for axis in 'xyz'])
    pvtx_xyz = np.stack([ak.to_numpy(events[f'PVtx_v{axis}'][entry]) for axis in 'xyz'], axis=1)
    row['pv'] = survey.pv_truth(pv_xyz, pvtx_xyz, ak.to_numpy(events['PVtx_score'][entry]),
                                truth['true_pv'])
    if gt_indices is not None:
        row['gt_median_dz'] = float(np.median(ak.to_numpy(events['Track_dz'][entry])[gt_indices]))
    return row


def survey_file(path):
    """path: ROOT file with an Events tree. Returns the selection bucket
    counts and, per population (all / track_events / selected, events with
    a true PV only), the stored-PV-far, nearest-PVtx and dz-proxy rates."""
    events = uproot.open(path)['Events'].arrays(BRANCHES)
    rows = [event_row(events, entry) for entry in range(len(events))]
    buckets = Counter(row['bucket'] for row in rows)
    populations = {}
    for name, keep in POPULATIONS.items():
        subset = [row for row in rows if keep(row) and row['pv'] is not None]
        with_proxy = [row for row in subset if row['gt_median_dz'] is not None]
        populations[name] = {
            'n': len(subset),
            'stored_far': survey.rate_entry(
                sum(row['pv']['stored_distance_3d'] > survey.PV_PROXY_CM for row in subset),
                len(subset)),
            'nearest_within': survey.rate_entry(
                sum(row['pv']['nearest_index'] >= 0
                    and row['pv']['nearest_distance'] <= survey.PV_PROXY_CM for row in subset),
                len(subset)),
            'nearest_is_index_0': survey.rate_entry(
                sum(row['pv']['nearest_index'] == 0 for row in subset), len(subset)),
            'nearest_score_rank_0': survey.rate_entry(
                sum(row['pv']['nearest_score_rank'] == 0 for row in subset), len(subset)),
            'proxy_far': survey.rate_entry(
                sum(abs(row['gt_median_dz']) > survey.PV_PROXY_CM for row in with_proxy),
                len(with_proxy)),
        }
    return {'n_events': len(rows),
            'buckets': {name: int(buckets[name]) for name in survey.FUNNEL_BUCKETS},
            'populations': populations}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-json', default=None)
    arguments = parser.parse_args()
    summary = survey_file(arguments.input)
    if arguments.output_json:
        with open(arguments.output_json, 'w') as handle:
            json.dump(summary, handle, indent=2)
    for name, population in summary['populations'].items():
        print(f"{name}: n={population['n']} stored PV far {survey.format_rate(population['stored_far'])} "
              f"| nearest PVtx within 0.2 cm {survey.format_rate(population['nearest_within'])} "
              f"| dz proxy far {survey.format_rate(population['proxy_far'])}")
    print('buckets', summary['buckets'])


if __name__ == '__main__':
    main()
