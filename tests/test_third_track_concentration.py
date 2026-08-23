from __future__ import annotations

import numpy as np

from scripts.python.third_track_concentration import (
    attachment_counts, percentile_of, third_slot_stats)


def test_third_slot_stats_counts_distinct_and_coverage():
    thirds = np.array([4, 4, 7, 4, 9, 7, 4, 11])
    stats = third_slot_stats(thirds, k=8, leading=(1, 3))
    assert stats['n_distinct'] == 4
    assert stats['coverage_top1'] == 4 / 8   # track 4 appears 4 times
    assert stats['coverage_top3'] == 7 / 8   # tracks 4, 7, 9 (or 4,7,11)


def test_attachment_counts_max_and_gt():
    thirds = np.array([4, 4, 7, 4, 9])
    assert attachment_counts(thirds)[4] == 3
    assert attachment_counts(thirds).get(9) == 1


def test_percentile_of_value_within_population():
    population = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert percentile_of(0.5, population) == 1.0
    assert percentile_of(0.1, population) == 0.2
    assert percentile_of(0.3, population) == 0.6
