"""Tests for the numpy geometry used by the MiniAOD source: closest approach
of two track lines, vertex flight distances, pair masses and the beam-spot
impact parameter.

Real data only: every expectation is a quantity the batch-0 ntuple stores
(computed there by CMSSW), recomputed from the ntuple's own track and vertex
columns."""
from __future__ import annotations

import os
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'condor', 'convert_to_parquet',
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import miniaod_geometry as geometry  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JOB0 = os.path.join(ROOT, 'data', 'low-pt', 'abcbatch', 'abcbatch0_2024_job0_ditaus_mc.root')
job0_required = pytest.mark.skipif(not os.path.exists(JOB0), reason='abcbatch job0 file not present')


@pytest.fixture(scope='module')
def events():
    branches = ['PV_x', 'PV_y', 'PV_z', 'SV_x', 'SV_y', 'SV_z', 'SV_dlen', 'SV_dxy',
                'Track_pt', 'Track_eta', 'Track_phi', 'Track_vx', 'Track_vy', 'Track_vz',
                'Track_covDxyDxy', 'Track_DCASig', 'BeamSpot_x', 'BeamSpot_y',
                'KstarToPiKAll_trk1_idx', 'KstarToPiKAll_trk2_idx', 'KstarToPiKAll_trk1_mass',
                'KstarToPiKAll_trk2_mass', 'KstarToPiKAll_mass', 'KstarToPiKAll_vtx_x',
                'KstarToPiKAll_vtx_y', 'KstarToPiKAll_vtx_z']
    return uproot.open(JOB0)['Events'].arrays(branches)


@job0_required
def test_vertex_distance_reproduces_sv_flight_lengths(events):
    checked = 0
    for row in range(len(events)):
        positions = np.stack([ak.to_numpy(events[f'SV_{axis}'][row]) for axis in 'xyz'], axis=1)
        if not positions.shape[0]:
            continue
        reference = np.array([events['PV_x'][row], events['PV_y'][row], events['PV_z'][row]], dtype=np.float64)
        result = geometry.vertex_distance(reference, np.zeros((3, 3)), positions,
                                          np.tile(np.eye(3) * 1e-6, (positions.shape[0], 1, 1)))
        # the ntuple stores both lengths with a 10-bit mantissa
        np.testing.assert_allclose(result['distance_3d'], ak.to_numpy(events['SV_dlen'][row]), rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(result['distance_xy'], ak.to_numpy(events['SV_dxy'][row]), rtol=2e-3, atol=2e-4)
        checked += positions.shape[0]
    assert checked > 100


@job0_required
def test_vertex_distance_significance_is_distance_over_projected_error():
    reference = np.zeros(3)
    positions = np.array([[0.3, 0.0, 0.4]])
    covariance = np.diag([0.01, 0.04, 0.09])[None, :, :]
    result = geometry.vertex_distance(reference, np.diag([0.01, 0.0, 0.0]), positions, covariance)
    # error^2 = d^T (C_ref + C_sv) d / |d|^2 with d = (0.3, 0, 0.4)
    expected_error = np.sqrt((0.3 ** 2 * 0.02 + 0.4 ** 2 * 0.09) / 0.25)
    assert result['distance_3d'][0] == pytest.approx(0.5)
    assert result['significance_3d'][0] == pytest.approx(0.5 / expected_error)
    assert result['significance_xy'][0] == pytest.approx(0.3 / np.sqrt(0.02))


@job0_required
def test_pair_mass_reproduces_kstar_unfitted_mass(events):
    checked = 0
    for row in range(len(events)):
        first = ak.to_numpy(events['KstarToPiKAll_trk1_idx'][row])
        if not first.size:
            continue
        second = ak.to_numpy(events['KstarToPiKAll_trk2_idx'][row])
        pt, eta, phi = (ak.to_numpy(events[name][row]) for name in ('Track_pt', 'Track_eta', 'Track_phi'))
        mass = geometry.pair_mass(pt[first], eta[first], phi[first], ak.to_numpy(events['KstarToPiKAll_trk1_mass'][row]),
                                  pt[second], eta[second], phi[second], ak.to_numpy(events['KstarToPiKAll_trk2_mass'][row]))
        np.testing.assert_allclose(mass, ak.to_numpy(events['KstarToPiKAll_mass'][row]), atol=2e-3)
        checked += first.size
    assert checked > 1000


@job0_required
def test_closest_approach_midpoint_lies_near_the_fitted_kstar_vertex(events):
    distances, pulls = [], []
    for row in range(len(events)):
        first = ak.to_numpy(events['KstarToPiKAll_trk1_idx'][row])
        if not first.size:
            continue
        second = ak.to_numpy(events['KstarToPiKAll_trk2_idx'][row])
        points = np.stack([ak.to_numpy(events[f'Track_v{axis}'][row]) for axis in 'xyz'], axis=1).astype(np.float64)
        directions = geometry.unit_direction(ak.to_numpy(events['Track_eta'][row]), ak.to_numpy(events['Track_phi'][row]))
        separation, midpoint = geometry.line_distance(points[first], directions[first], points[second], directions[second])
        fitted = np.stack([ak.to_numpy(events[f'KstarToPiKAll_vtx_{axis}'][row]) for axis in 'xyz'], axis=1)
        error = np.sqrt(ak.to_numpy(events['Track_covDxyDxy'][row]))
        distances.append(np.linalg.norm(midpoint - fitted, axis=1))
        pulls.append(separation / np.hypot(error[first], error[second]))
    distances, pulls = np.concatenate(distances), np.concatenate(pulls)
    assert distances.size > 5000
    # the stored candidates passed a fit probability >= 0.1, i.e. their tracks
    # cross within about 1.6 combined position errors
    assert np.percentile(pulls, 90) < 1.7
    # the midpoint of closest approach stands in for the fitted vertex
    assert np.median(distances) < 0.01
    assert np.percentile(distances, 90) < 0.1


@job0_required
def test_line_distance_of_crossing_and_parallel_lines():
    separation, midpoint = geometry.line_distance(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.5, 2.0, 0.3], [0.0, 3.0, 4.0]]), np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert separation[0] == pytest.approx(0.3)
    np.testing.assert_allclose(midpoint[0], [0.5, 0.0, 0.15], atol=1e-9)
    assert separation[1] == pytest.approx(5.0)


@job0_required
def test_beam_spot_dca_reproduces_track_dca_significance(events):
    ratios = []
    for row in range(len(events)):
        significance = ak.to_numpy(events['Track_DCASig'][row])
        if not significance.size:
            continue
        impact = geometry.beam_spot_dca(ak.to_numpy(events['Track_vx'][row]), ak.to_numpy(events['Track_vy'][row]),
                                        ak.to_numpy(events['Track_phi'][row]),
                                        float(events['BeamSpot_x'][row]), float(events['BeamSpot_y'][row]))
        ours = np.abs(impact) / np.sqrt(ak.to_numpy(events['Track_covDxyDxy'][row]))
        valid = (significance > 0.5) & np.isfinite(ours)
        ratios.append(ours[valid] / significance[valid])
    ratios = np.concatenate(ratios)
    assert ratios.size > 5000
    assert np.median(ratios) == pytest.approx(1.0, abs=0.01)
    assert np.percentile(np.abs(ratios - 1.0), 90) < 0.05


@job0_required
def test_perpendicular_variance_along_the_dxy_and_dsz_axes():
    phi, eta = np.array([0.7]), np.array([0.4])
    dxy_axis = np.array([[-np.sin(0.7), np.cos(0.7), 0.0]])
    sin_lambda, cos_lambda = np.tanh(0.4), 1.0 / np.cosh(0.4)
    dsz_axis = np.array([[-np.cos(0.7) * sin_lambda, -np.sin(0.7) * sin_lambda, cos_lambda]])
    assert geometry.perpendicular_variance(dxy_axis, phi, eta, np.array([4.0]), np.array([9.0]), np.array([1.0]))[0] == pytest.approx(4.0)
    assert geometry.perpendicular_variance(dsz_axis, phi, eta, np.array([4.0]), np.array([9.0]), np.array([1.0]))[0] == pytest.approx(9.0)
    mixed = (dxy_axis + dsz_axis) / np.sqrt(2.0)
    assert geometry.perpendicular_variance(mixed, phi, eta, np.array([4.0]), np.array([9.0]), np.array([1.0]))[0] == pytest.approx(7.5)
