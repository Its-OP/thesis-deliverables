from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.python.couple_concentration_report import couple_budget_curve
from scripts.python.triplet_failure_taxonomy import top_couple_coverage


def test_couple_budget_curve_fractions_over_all_events():
    gt_couple_rank = pd.Series([0, 3, 11, 40, None, 120])
    curve = couple_budget_curve(gt_couple_rank, budgets=[5, 25, 125])
    # Denominator is ALL events (None = GT absent counts as unreachable).
    assert curve[5] == 2 / 6
    assert curve[25] == 3 / 6
    assert curve[125] == 5 / 6


def test_top_couple_coverage_counts_first_m_couples_share():
    # Deduped ranking order of couple ids; first distinct couples: 7, 3, 9.
    couple_ids = np.array([7, 7, 3, 7, 9, 3, 5, 7])
    assert top_couple_coverage(couple_ids, k=8, m=1) == 4 / 8
    assert top_couple_coverage(couple_ids, k=8, m=3) == 7 / 8
    assert top_couple_coverage(couple_ids, k=4, m=1) == 3 / 4
    # m larger than distinct couples present: everything covered.
    assert top_couple_coverage(couple_ids, k=8, m=10) == 1.0
