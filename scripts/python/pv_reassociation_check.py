from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    'event_primary_vertex_z',
    'event_other_pv_z',
    'event_n_pvs',
    'track_label_from_tau',
    'track_dz',
    'track_pt',
]

V1_PROXY_DEFINITION = (
    'PV nearest to the z proxy computed from the 2 highest-pt GT tracks. '
    'Stage-3 couple predictions are not available in the raw eval shards, so '
    'the 2 highest-pt GT tracks stand in for the top couple; this is an '
    'UPPER BOUND on the V1 re-association rule accuracy.'
)


def true_vertex_z_proxy(stored_pv_z, track_dz_values):
    """stored_pv_z: scalar; track_dz_values: (n_tracks,). Returns scalar z."""
    values = np.asarray(track_dz_values, dtype=np.float64)
    return float(np.median(values + float(stored_pv_z)))


def nearest_pv_index(proxy_z, pv_z_candidates):
    """proxy_z: scalar; pv_z_candidates: (n_candidates,). Returns int index."""
    distances = np.abs(np.asarray(pv_z_candidates, dtype=np.float64) - float(proxy_z))
    return int(np.argmin(distances))


def analyze_event(stored_pv_z, other_pv_z, gt_track_dz, gt_track_pt):
    """stored_pv_z: scalar; other_pv_z: (n_other,); gt_track_dz, gt_track_pt:
    (n_gt,) with n_gt >= 3. Returns a per-event record dict."""
    other_pv_z = np.asarray(other_pv_z, dtype=np.float64)
    gt_track_dz = np.asarray(gt_track_dz, dtype=np.float64)
    gt_track_pt = np.asarray(gt_track_pt, dtype=np.float64)
    proxy_z = true_vertex_z_proxy(stored_pv_z, gt_track_dz)
    candidates = np.concatenate(([float(stored_pv_z)], other_pv_z))
    correct_index = nearest_pv_index(proxy_z, candidates)
    top_two_by_pt = np.argsort(-gt_track_pt, kind='stable')[:2]
    couple_proxy_z = true_vertex_z_proxy(stored_pv_z, gt_track_dz[top_two_by_pt])
    v1_index = nearest_pv_index(couple_proxy_z, candidates)
    best_other_distance = (
        float(np.min(np.abs(other_pv_z - proxy_z))) if other_pv_z.size else None)
    return {
        'stored_pv_correct': correct_index == 0,
        'v1_pick_correct': v1_index == correct_index,
        'abs_stored_pv_minus_proxy': abs(float(stored_pv_z) - proxy_z),
        'abs_best_other_pv_minus_proxy': best_other_distance,
    }


def process_events(event_pv_z, event_other_pv_z, event_n_pvs,
                   track_labels, track_dz, track_pt):
    """All args: (n_events,) sequences; track_* entries are per-event arrays
    over tracks. Returns (records, analyzed_n_pvs, analyzed_n_other_pvs) for
    events with >= 3 GT tracks."""
    records = []
    analyzed_n_pvs = []
    analyzed_n_other_pvs = []
    for index in range(len(event_pv_z)):
        gt_mask = np.asarray(track_labels[index]) == 1
        if int(gt_mask.sum()) < 3:
            continue
        gt_dz = np.asarray(track_dz[index], dtype=np.float64)[gt_mask]
        gt_pt = np.asarray(track_pt[index], dtype=np.float64)[gt_mask]
        other_pv_z = np.asarray(event_other_pv_z[index], dtype=np.float64)
        records.append(analyze_event(event_pv_z[index], other_pv_z, gt_dz, gt_pt))
        analyzed_n_pvs.append(int(event_n_pvs[index]))
        analyzed_n_other_pvs.append(int(other_pv_z.size))
    return records, analyzed_n_pvs, analyzed_n_other_pvs


def distribution_summary(values):
    """values: (n,) sequence. Returns dict of count/mean/std/min/quantiles/max."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'p10': None,
                'p25': None, 'p50': None, 'p75': None, 'p90': None, 'max': None}
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'p10': float(quantiles[0]),
        'p25': float(quantiles[1]),
        'p50': float(quantiles[2]),
        'p75': float(quantiles[3]),
        'p90': float(quantiles[4]),
        'max': float(values.max()),
    }


def summarize(records, analyzed_n_pvs, analyzed_n_other_pvs, n_events_total):
    """records: list of analyze_event dicts; analyzed_n_pvs,
    analyzed_n_other_pvs: (n_analyzed,); n_events_total: scalar. Returns the
    aggregate summary dict."""
    n_analyzed = len(records)
    n_stored_correct = sum(1 for record in records if record['stored_pv_correct'])
    n_v1_correct = sum(1 for record in records if record['v1_pick_correct'])
    stored_distances = [record['abs_stored_pv_minus_proxy'] for record in records]
    best_other_distances = [
        record['abs_best_other_pv_minus_proxy'] for record in records
        if record['abs_best_other_pv_minus_proxy'] is not None]
    fraction_stored_correct = n_stored_correct / n_analyzed
    return {
        'n_events_total': int(n_events_total),
        'n_events_with_ge3_gt_tracks': n_analyzed,
        'fraction_events_analyzed': n_analyzed / int(n_events_total),
        'fraction_stored_pv_correct': fraction_stored_correct,
        'fraction_stored_pv_wrong': 1.0 - fraction_stored_correct,
        'v1_reassociation_accuracy': n_v1_correct / n_analyzed,
        'v1_proxy_definition': V1_PROXY_DEFINITION,
        'abs_stored_pv_minus_proxy': distribution_summary(stored_distances),
        'abs_best_other_pv_minus_proxy': distribution_summary(best_other_distances),
        'n_pvs': distribution_summary(analyzed_n_pvs),
        'n_other_pv_candidates': distribution_summary(analyzed_n_other_pvs),
    }


def run(src_glob, output_path):
    """src_glob: shard glob string; output_path: JSON destination. Returns the
    summary dict."""
    paths = sorted(glob.glob(src_glob))
    assert paths, f'no shards match {src_glob}'
    records = []
    analyzed_n_pvs = []
    analyzed_n_other_pvs = []
    n_events_total = 0
    for path in paths:
        table = pq.read_table(path, columns=REQUIRED_COLUMNS)
        n_events_total += table.num_rows
        shard_records, shard_n_pvs, shard_n_other_pvs = process_events(
            table['event_primary_vertex_z'].to_numpy(zero_copy_only=False),
            table['event_other_pv_z'].to_numpy(zero_copy_only=False),
            table['event_n_pvs'].to_numpy(zero_copy_only=False),
            table['track_label_from_tau'].to_numpy(zero_copy_only=False),
            table['track_dz'].to_numpy(zero_copy_only=False),
            table['track_pt'].to_numpy(zero_copy_only=False),
        )
        records.extend(shard_records)
        analyzed_n_pvs.extend(shard_n_pvs)
        analyzed_n_other_pvs.extend(shard_n_other_pvs)
    summary = summarize(records, analyzed_n_pvs, analyzed_n_other_pvs, n_events_total)
    summary['src_glob'] = src_glob
    summary['n_shards'] = len(paths)
    with open(output_path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(
        f'PV re-association: stored-PV correct {summary["fraction_stored_pv_correct"]:.4f} '
        f'of {summary["n_events_with_ge3_gt_tracks"]} events (>=3 GT) | '
        f'V1 top2-pt-GT-proxy accuracy {summary["v1_reassociation_accuracy"]:.4f} (upper bound) | '
        f'median |stored_pv-proxy| {summary["abs_stored_pv_minus_proxy"]["p50"]:.4f} vs '
        f'|best_other_pv-proxy| {summary["abs_best_other_pv_minus_proxy"]["p50"]:.4f} | '
        f'mean n_pvs {summary["n_pvs"]["mean"]:.2f}'
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description='C-PV V1 measurement: quantify primary-vertex '
                    're-association on the eval shards.')
    parser.add_argument('--src-glob', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.src_glob, args.output)


if __name__ == '__main__':
    main()
