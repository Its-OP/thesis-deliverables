from __future__ import annotations

import pytest
import torch

from utils.triplet_join import build_track_lorentz
from utils.vertex_fit_features import FIT_NAMES, static_fit_columns
from weaver.nn.model.VertexFit import VertexFitLayer, physics_log_weights


def _tracks():
    return dict(
        pt=torch.tensor([1.0, 1.2, 0.9, 1.1, 0.8]),
        eta=torch.tensor([0.10, 0.15, 0.12, 0.18, 3.00]),
        phi=torch.tensor([0.05, 0.10, 0.08, 0.12, 2.50]),
        vertex_x=torch.tensor([0.10, 0.11, 0.09, 0.02, -0.50]),
        vertex_y=torch.tensor([0.05, 0.06, 0.04, 0.01, 0.60]),
        vertex_z=torch.tensor([1.00, 1.02, 0.98, 0.20, -4.00]),
        var_dxy=torch.tensor([1e-4, 2e-4, 3e-4, 4e-4, 5e-4]),
        var_dsz=torch.tensor([2e-4, 1e-4, 2e-4, 3e-4, 4e-4]),
    )


def _columns(triplets=((0, 1, 2), (0, 1, 3))):
    tracks = _tracks()
    i, j, k = (torch.tensor([t[m] for t in triplets]) for m in range(3))
    return static_fit_columns(
        i, j, k,
        lorentz=build_track_lorentz(tracks['pt'], tracks['eta'], tracks['phi']),
        primary_vertex=torch.tensor([0.01, -0.02, 0.90]),
        **tracks)


def test_static_columns_have_one_column_per_fit_name():
    columns = _columns()
    assert columns.shape == (2, len(FIT_NAMES))
    assert torch.isfinite(columns).all()


def test_static_log_weight_channels_are_the_physics_weights():
    tracks = _tracks()
    columns = _columns(triplets=((0, 1, 2),))
    expected = physics_log_weights(
        tracks['var_dxy'][[0, 1, 2]].unsqueeze(0),
        tracks['var_dsz'][[0, 1, 2]].unsqueeze(0))
    start = FIT_NAMES.index('fit_logw_i')
    assert torch.allclose(columns[0, start:start + 3], expected[0], atol=1e-6)


def test_static_twin_equals_the_zero_init_layer():
    tracks = _tracks()
    triplets = ((0, 1, 2), (0, 1, 3), (2, 3, 4))
    columns = _columns(triplets=triplets)

    lorentz = build_track_lorentz(tracks['pt'], tracks['eta'], tracks['phi'])
    members = torch.tensor(triplets)
    reference = torch.stack([
        torch.stack([tracks['vertex_x'][members[:, m]],
                     tracks['vertex_y'][members[:, m]],
                     tracks['vertex_z'][members[:, m]]], dim=0)
        for m in range(3)], dim=0).unsqueeze(0)
    momentum = sum(lorentz[:3, members[:, m]] for m in range(3)).unsqueeze(0)
    energy = sum(lorentz[3, members[:, m]] for m in range(3))
    mass = (energy.square()
            - momentum[0].square().sum(dim=0)).clamp_min(0.0).sqrt().unsqueeze(0)

    layer = VertexFitLayer()
    outputs = layer(
        reference=reference,
        eta=torch.stack([tracks['eta'][members[:, m]] for m in range(3)]).unsqueeze(0),
        phi=torch.stack([tracks['phi'][members[:, m]] for m in range(3)]).unsqueeze(0),
        var_dxy=torch.stack([tracks['var_dxy'][members[:, m]] for m in range(3)]).unsqueeze(0),
        var_dsz=torch.stack([tracks['var_dsz'][members[:, m]] for m in range(3)]).unsqueeze(0),
        primary_vertex=torch.tensor([[0.01, -0.02, 0.90]]),
        momentum=momentum,
        mass=mass,
        quality=torch.randn(1, 12, 3, len(triplets)))
    assert torch.allclose(outputs[0].T, columns, atol=1e-5)


def test_pv_free_channels_ignore_the_primary_vertex():
    tracks = _tracks()
    i, j, k = (torch.tensor([v]) for v in (0, 1, 2))
    lorentz = build_track_lorentz(tracks['pt'], tracks['eta'], tracks['phi'])
    base = static_fit_columns(
        i, j, k, lorentz=lorentz,
        primary_vertex=torch.tensor([0.0, 0.0, 0.0]), **tracks)
    moved = static_fit_columns(
        i, j, k, lorentz=lorentz,
        primary_vertex=torch.tensor([0.5, -0.5, 2.0]), **tracks)
    for index, name in enumerate(FIT_NAMES):
        if name.startswith('fitpv_'):
            continue
        assert torch.allclose(base[0, index], moved[0, index], atol=1e-6), name


def test_a_common_vertex_triplet_scores_near_zero_chi2():
    columns = _columns(triplets=((0, 1, 2),))
    chi2 = float(columns[0, FIT_NAMES.index('fit_chi2')])
    spread = _columns(triplets=((2, 3, 4),))
    spread_chi2 = float(spread[0, FIT_NAMES.index('fit_chi2')])
    assert chi2 < spread_chi2
    assert float(columns[0, FIT_NAMES.index('fit_res_max')]) < \
        float(spread[0, FIT_NAMES.index('fit_res_max')])
