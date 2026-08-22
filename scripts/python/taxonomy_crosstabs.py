from __future__ import annotations

import argparse

import pandas as pd


def class_share_table(records: pd.DataFrame, column: str,
                      n_bins: int = 4) -> pd.DataFrame:
    """records: per-event taxonomy rows with `class` and `column`. Returns one
    row per quantile bin of `column`: per-class shares plus event count `n`."""
    rows = records.dropna(subset=[column]).copy()
    rows[column] = pd.qcut(rows[column], n_bins, duplicates='drop')
    counts = rows.groupby(column, observed=True)['class'] \
        .value_counts().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares['n'] = counts.sum(axis=1)
    return shares


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--gate', default='tierH')
    parser.add_argument('--bins', type=int, default=4)
    args = parser.parse_args()

    records = pd.read_parquet(args.records)
    records = records[records['gate'] == args.gate]
    print(f'gate {args.gate}: {len(records)} events')
    for column in ('gt_pt_visible', 'n_tierh', 'n_couples_top10'):
        table = class_share_table(records, column, n_bins=args.bins)
        median_rank = records.dropna(subset=[column]).copy()
        median_rank[column] = pd.qcut(median_rank[column], args.bins,
                                      duplicates='drop')
        table['gt_rank_median'] = median_rank.groupby(column, observed=True)[
            'gt_rank'].median()
        print(f'\n== class shares by {column} quantile ==')
        print(table.round(3).to_string())


if __name__ == '__main__':
    main()
