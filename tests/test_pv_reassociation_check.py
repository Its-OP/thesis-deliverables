from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.python.pv_reassociation_check import (
    analyze_event,
    distribution_summary,
    nearest_pv_index,
    process_events,
    run,
    summarize,
    true_vertex_z_proxy,
)


def test_true_vertex_z_proxy_is_median_of_stored_plus_dz():
    assert true_vertex_z_proxy(1.0, [0.1, 0.2, 0.4]) == pytest.approx(1.2)


def test_true_vertex_z_proxy_even_count_averages_middle_pair():
    assert true_vertex_z_proxy(-2.0, [0.0, 1.0]) == pytest.approx(-1.5)


def test_nearest_pv_index_picks_argmin_abs_distance():
    assert nearest_pv_index(2.05, [0.0, 2.0, 5.0]) == 1
    assert nearest_pv_index(-4.9, [0.0, -5.0]) == 1


def test_nearest_pv_index_breaks_ties_toward_first_candidate():
    assert nearest_pv_index(1.0, [0.5, 1.5]) == 0


def test_analyze_event_stored_pv_correct_when_nearest_to_proxy():
    record = analyze_event(
        stored_pv_z=0.0,
        other_pv_z=[5.0],
        gt_track_dz=[0.01, -0.01, 0.02],
        gt_track_pt=[1.0, 2.0, 3.0],
    )
    assert record['stored_pv_correct'] is True
    assert record['v1_pick_correct'] is True
    assert record['abs_stored_pv_minus_proxy'] == pytest.approx(0.01)
    assert record['abs_best_other_pv_minus_proxy'] == pytest.approx(4.99)


def test_analyze_event_other_pv_correct_when_gt_tracks_point_elsewhere():
    record = analyze_event(
        stored_pv_z=0.0,
        other_pv_z=[3.0, 7.0],
        gt_track_dz=[3.1, 2.9, 3.0],
        gt_track_pt=[1.0, 1.0, 1.0],
    )
    assert record['stored_pv_correct'] is False
    assert record['v1_pick_correct'] is True
    assert record['abs_stored_pv_minus_proxy'] == pytest.approx(3.0)
    assert record['abs_best_other_pv_minus_proxy'] == pytest.approx(0.0)


def test_analyze_event_v1_uses_only_two_highest_pt_gt_tracks():
    record = analyze_event(
        stored_pv_z=0.0,
        other_pv_z=[4.0],
        gt_track_dz=[0.0, 0.2, 4.1, 3.9, 4.0],
        gt_track_pt=[10.0, 9.0, 1.0, 1.0, 1.0],
    )
    assert record['stored_pv_correct'] is False
    assert record['v1_pick_correct'] is False


def test_analyze_event_without_other_pvs_keeps_stored_pv():
    record = analyze_event(
        stored_pv_z=1.0,
        other_pv_z=[],
        gt_track_dz=[0.1, 0.1, 0.1],
        gt_track_pt=[1.0, 2.0, 3.0],
    )
    assert record['stored_pv_correct'] is True
    assert record['v1_pick_correct'] is True
    assert record['abs_best_other_pv_minus_proxy'] is None


def test_process_events_masks_to_gt_tracks_and_skips_below_three():
    records, analyzed_n_pvs, analyzed_n_other_pvs = process_events(
        event_pv_z=[0.0, 0.0, 0.0],
        event_other_pv_z=[[6.0], [1.0], [5.0]],
        event_n_pvs=[2, 3, 4],
        track_labels=[
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 1]),
        ],
        track_dz=[
            np.array([0.0, 0.02, 0.01, 9.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([5.0, 5.1, 4.9]),
        ],
        track_pt=[
            np.array([5.0, 4.0, 3.0, 99.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
        ],
    )
    assert len(records) == 2
    assert analyzed_n_pvs == [2, 4]
    assert analyzed_n_other_pvs == [1, 1]
    assert records[0]['stored_pv_correct'] is True
    assert records[0]['v1_pick_correct'] is True
    assert records[1]['stored_pv_correct'] is False


def test_distribution_summary_reports_quantiles():
    summary = distribution_summary([1.0, 2.0, 3.0, 4.0])
    assert summary['count'] == 4
    assert summary['mean'] == pytest.approx(2.5)
    assert summary['min'] == pytest.approx(1.0)
    assert summary['p50'] == pytest.approx(2.5)
    assert summary['max'] == pytest.approx(4.0)


def test_distribution_summary_empty_input():
    summary = distribution_summary([])
    assert summary['count'] == 0
    assert summary['mean'] is None
    assert summary['p50'] is None


def test_summarize_fractions_and_json_serializability():
    records = [
        {'stored_pv_correct': True, 'v1_pick_correct': True,
         'abs_stored_pv_minus_proxy': 0.0, 'abs_best_other_pv_minus_proxy': 0.5},
        {'stored_pv_correct': False, 'v1_pick_correct': True,
         'abs_stored_pv_minus_proxy': 1.0, 'abs_best_other_pv_minus_proxy': 0.0},
        {'stored_pv_correct': False, 'v1_pick_correct': False,
         'abs_stored_pv_minus_proxy': 2.0, 'abs_best_other_pv_minus_proxy': None},
        {'stored_pv_correct': False, 'v1_pick_correct': True,
         'abs_stored_pv_minus_proxy': 3.0, 'abs_best_other_pv_minus_proxy': 0.1},
    ]
    summary = summarize(records, analyzed_n_pvs=[2, 3, 4, 5],
                        analyzed_n_other_pvs=[1, 1, 2, 3], n_events_total=6)
    assert summary['n_events_total'] == 6
    assert summary['n_events_with_ge3_gt_tracks'] == 4
    assert summary['fraction_events_analyzed'] == pytest.approx(4 / 6)
    assert summary['fraction_stored_pv_correct'] == pytest.approx(0.25)
    assert summary['fraction_stored_pv_wrong'] == pytest.approx(0.75)
    assert summary['v1_reassociation_accuracy'] == pytest.approx(0.75)
    assert 'upper' in summary['v1_proxy_definition'].lower()
    assert summary['abs_stored_pv_minus_proxy']['p50'] == pytest.approx(1.5)
    assert summary['abs_best_other_pv_minus_proxy']['count'] == 3
    assert summary['n_pvs']['mean'] == pytest.approx(3.5)
    assert summary['n_other_pv_candidates']['max'] == pytest.approx(3.0)
    json.dumps(summary)


def test_run_end_to_end_on_synthetic_parquet(tmp_path):
    table = pa.table({
        'event_primary_vertex_z': pa.array([0.0, 0.0, 0.0], type=pa.float32()),
        'event_other_pv_z': pa.array([[5.0], [3.0], []],
                                     type=pa.large_list(pa.float32())),
        'event_n_pvs': pa.array([2, 2, 1], type=pa.int32()),
        'track_label_from_tau': pa.array([[1, 1, 1, 0], [1, 1, 1], [1, 1]],
                                         type=pa.large_list(pa.int32())),
        'track_dz': pa.array([[0.0, 0.01, -0.01, 9.0], [3.0, 3.1, 2.9], [0.0, 0.0]],
                             type=pa.large_list(pa.float32())),
        'track_pt': pa.array([[3.0, 2.0, 1.0, 99.0], [1.0, 2.0, 3.0], [1.0, 1.0]],
                             type=pa.large_list(pa.float32())),
    })
    pq.write_table(table, tmp_path / 'shard_000.parquet')
    output_path = tmp_path / 'summary.json'

    summary = run(str(tmp_path / '*.parquet'), str(output_path))

    assert summary['n_shards'] == 1
    assert summary['n_events_total'] == 3
    assert summary['n_events_with_ge3_gt_tracks'] == 2
    assert summary['fraction_stored_pv_correct'] == pytest.approx(0.5)
    assert summary['v1_reassociation_accuracy'] == pytest.approx(1.0)
    assert summary['n_pvs']['mean'] == pytest.approx(2.0)
    persisted = json.loads(output_path.read_text())
    assert persisted['fraction_stored_pv_correct'] == pytest.approx(0.5)
    assert persisted['src_glob'] == str(tmp_path / '*.parquet')
