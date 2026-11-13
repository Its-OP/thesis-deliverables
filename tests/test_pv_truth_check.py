"""Tests for scripts/python/pv_truth_check.py on the old production's raw
nano (data/thesis_files/step_MINI_10_nano_ditaus_mc.root, 93 events)."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, 'data', 'thesis_files',
                         'step_MINI_10_nano_ditaus_mc.root')

pytestmark = pytest.mark.skipif(not os.path.exists(REFERENCE),
                                reason='reference raw nano not present')

from scripts.python import pv_truth_check  # noqa: E402


def test_old_production_pv_truth_numbers():
    summary = pv_truth_check.survey_file(REFERENCE)
    assert summary['buckets']['selected'] == 13
    assert summary['buckets']['3_gt_below_cutoff'] == 57
    assert summary['buckets']['no_tracks'] == 11
    population = summary['populations']
    assert population['all']['n'] == 93
    assert population['all']['stored_far']['successes'] == 68
    assert population['all']['nearest_within']['successes'] == 93
    assert population['all']['nearest_is_index_0']['successes'] == 21
    assert population['all']['nearest_score_rank_0']['successes'] == 21
    assert population['all']['proxy_far']['successes'] == 49
    assert population['all']['proxy_far']['total'] == 70
    assert population['track_events']['n'] == 82
    assert population['track_events']['stored_far']['successes'] == 59
    assert population['selected']['n'] == 13
    assert population['selected']['stored_far']['successes'] == 6
