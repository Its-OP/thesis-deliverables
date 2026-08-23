from __future__ import annotations

import argparse

import pandas as pd

DEFAULT_BUDGETS = [5, 12, 25, 50, 75, 100, 125]


def couple_budget_curve(gt_couple_rank: pd.Series,
                        budgets: list[int]) -> dict[int, float]:
    """gt_couple_rank: per-event stage-3 rank of the GT couple (NaN when the
    GT is absent). Fractions denominate over ALL events."""
    total = len(gt_couple_rank)
    return {budget: float((gt_couple_rank < budget).sum() / total)
            for budget in budgets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--gate', default='tierH')
    args = parser.parse_args()

    records = pd.read_parquet(args.records)
    records = records[records['gate'] == args.gate]
    print(f'gate {args.gate}: {len(records)} events')

    print('\n== couple-budget ceiling: P(GT couple within stage-3 top-C) ==')
    curve = couple_budget_curve(records['gt_couple_rank'], DEFAULT_BUDGETS)
    print(' '.join(f'C={budget}:{fraction:.4f}'
                   for budget, fraction in curve.items()))

    for column in [name for name in records.columns
                   if name.startswith('n_couples_top')]:
        values = records[column].dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
        print(f'\n== {column} (n={len(values)}) ==')
        print(f'mean {values.mean():.2f} p10/p25/p50/p75/p90 '
              + '/'.join(f'{q:.0f}' for q in quantiles))
        hit = records[records['class'] == 'hit'][column].dropna()
        miss = records[~records['class'].isin(['hit', 'absent'])][column].dropna()
        if not hit.empty and not miss.empty:
            print(f'median hit {hit.median():.0f} vs in-list miss '
                  f'{miss.median():.0f}')

    coverage_columns = [name for name in records.columns
                        if name.startswith('cov_top')]
    if coverage_columns:
        print('\n== coverage: share of top-K candidates from the leading m '
              'couples ==')
        for column in sorted(coverage_columns):
            values = records[column].dropna()
            print(f'{column}: mean {values.mean():.3f} median '
                  f'{values.median():.3f} p90 {values.quantile(0.9):.3f}')


if __name__ == '__main__':
    main()
