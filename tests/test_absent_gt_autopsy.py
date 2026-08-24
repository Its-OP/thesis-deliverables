from __future__ import annotations

import numpy as np

from scripts.python.absent_gt_autopsy import classify_event


def test_classify_event_present_when_any_gt_row_survives_tau():
    assert classify_event(np.array([0.2, 0.9]), tau=0.8) == 'present'


def test_classify_event_gate_killed_when_all_gt_rows_below_tau():
    assert classify_event(np.array([0.2, 0.5]), tau=0.8) == 'gate_killed'


def test_classify_event_not_stored_without_gt_rows():
    assert classify_event(np.array([]), tau=0.8) == 'not_stored'
