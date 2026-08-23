from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TOP_KS = [10, 50, 100, 500]
LEADING = (1, 5, 12)


def third_slot_stats(thirds: np.ndarray, k: int,
                     leading=LEADING) -> dict[str, float]:
    """thirds: (n,) third-track indices of the top-k rows (gate order)."""
    head = thirds[:k]
    counts = Counter(head.tolist())
    ranked = [count for _, count in counts.most_common()]
    stats = {'n_distinct': len(counts)}
    for m in leading:
        stats[f'coverage_top{m}'] = sum(ranked[:m]) / len(head)
    return stats


def attachment_counts(thirds: np.ndarray) -> Counter:
    """thirds: (n,). How many candidates each third track appears in."""
    return Counter(thirds.tolist())


def percentile_of(value: float, population: np.ndarray) -> float:
    """Fraction of the population <= value."""
    return float((population <= value).sum() / len(population))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--src-glob', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-events', type=int, default=0)
    args = parser.parse_args()

    candidates = pq.read_table(
        args.candidates,
        columns=['cand_k', 'is_gt', 'row_kind', 'gt_k', 'track_s1'])
    shards = sorted(glob.glob(args.src_glob))
    source = pa.concat_tables([
        pq.read_table(shard, columns=['track_pt', 'track_label_from_b'])
        for shard in shards]).slice(0, candidates.num_rows)

    n_rows = candidates.num_rows if not args.max_events \
        else min(args.max_events, candidates.num_rows)
    per_event: list[dict] = []
    for r in range(n_rows):
        kinds = np.asarray(candidates['row_kind'][r].values)
        thirds = np.asarray(candidates['cand_k'][r].values)[kinds == 0]
        if thirds.size == 0:
            continue
        track_s1 = np.asarray(candidates['track_s1'][r].values)
        track_pt = np.asarray(source['track_pt'][r].values)
        from_b = np.asarray(source['track_label_from_b'][r].values)
        gt_k = int(candidates['gt_k'][r].as_py())

        record: dict = {'event': r}
        for k in TOP_KS:
            stats = third_slot_stats(thirds, k)
            for name, value in stats.items():
                record[f'{name}_top{k}'] = value
        head = thirds[:100]
        distinct = np.unique(head)
        record['s1_pctile_median_top100'] = float(np.median(
            [percentile_of(track_s1[t], track_s1) for t in distinct]))
        record['pt_pctile_median_top100'] = float(np.median(
            [percentile_of(track_pt[t], track_pt) for t in distinct]))
        record['from_b_share_top100'] = float(np.mean(
            [from_b[t] > 0.5 for t in distinct]))
        counts = attachment_counts(head)
        record['max_attachment_top100'] = int(max(counts.values()))
        if gt_k >= 0:
            record['gt_attachment_top100'] = int(counts.get(gt_k, 0))
            record['gt_s1_pctile'] = percentile_of(track_s1[gt_k], track_s1)
        per_event.append(record)

    frame_keys = sorted({key for record in per_event for key in record})
    summary = {}
    for key in frame_keys:
        values = np.asarray([record[key] for record in per_event
                             if key in record], dtype=np.float64)
        summary[key] = {
            'mean': float(values.mean()),
            'p10': float(np.percentile(values, 10)),
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
        }
    with open(args.output, 'w') as handle:
        json.dump({'n_events': len(per_event), 'summary': summary}, handle,
                  indent=2)
    for key in ('n_distinct_top10', 'n_distinct_top100', 'n_distinct_top500',
                'coverage_top5_top100', 's1_pctile_median_top100',
                'pt_pctile_median_top100', 'from_b_share_top100',
                'max_attachment_top100', 'gt_attachment_top100'):
        if key in summary:
            entry = summary[key]
            print(f"{key}: mean {entry['mean']:.3f} "
                  f"p10/p50/p90 {entry['p10']:.2f}/{entry['p50']:.2f}/"
                  f"{entry['p90']:.2f}")


if __name__ == '__main__':
    main()
