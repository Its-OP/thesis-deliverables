from __future__ import annotations

import pandas as pd

from scripts.python.taxonomy_crosstabs import class_share_table


def test_class_share_table_bins_by_quantile_and_normalizes_rows():
    records = pd.DataFrame({
        'gt_pt_visible': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'class': ['hit', 'hit', 'hit', 'absent',
                  'third_pion_confusion', 'third_pion_confusion', 'hit', 'hit'],
    })
    table = class_share_table(records, 'gt_pt_visible', n_bins=2)
    assert list(table.index.names) == ['gt_pt_visible']
    assert (table['n'] == [4, 4]).all()
    shares = table.drop(columns='n')
    assert (abs(shares.sum(axis=1) - 1.0) < 1e-9).all()
    low, high = shares.iloc[0], shares.iloc[1]
    assert low['hit'] == 0.75 and low['absent'] == 0.25
    assert high['third_pion_confusion'] == 0.5 and high['hit'] == 0.5


def test_class_share_table_drops_rows_without_the_binning_value():
    records = pd.DataFrame({
        'gt_pt_visible': [1.0, None, 2.0, None],
        'class': ['hit', 'absent', 'hit', 'absent'],
    })
    table = class_share_table(records, 'gt_pt_visible', n_bins=1)
    assert int(table['n'].sum()) == 2
