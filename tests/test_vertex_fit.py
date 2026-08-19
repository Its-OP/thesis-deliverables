from __future__ import annotations

import math

import pytest
import torch

from utils.triplet_join import _closest_approach
from weaver.nn.model.VertexFit import (
    FIT_NAMES,
    FIT_PV_PREFIX,
    VertexFitLayer,
    physics_log_weights,
    wls_vertex_fit,
)


def _lines_through(vertex, directions, offsets):
    """Reference points offset along each line from a common vertex."""
    directions = torch.nn.functional.normalize(directions, dim=-1)
    points = vertex.unsqueeze(0) + offsets.unsqueeze(1) * directions
    return points, directions


def _spread_lines():
    """Three lines that do NOT meet: perturbed off a common vertex."""
    vertex = torch.tensor([0.4, -0.2, 1.1], dtype=torch.float64)
    directions = torch.tensor([
        [0.9, 0.1, 0.4],
        [-0.2, 0.8, 0.5],
        [0.3, -0.6, 0.7],
    ], dtype=torch.float64)
    points, directions = _lines_through(
        vertex, directions, torch.tensor([-3.0, 2.0, 4.0], dtype=torch.float64))
    points = points + torch.tensor([
        [0.02, -0.01, 0.03],
        [-0.03, 0.02, 0.01],
        [0.01, 0.03, -0.02],
    ], dtype=torch.float64)
    return points, directions


def _brute_force_vertex(points, directions, weights):
    """Adaptive 3-D grid search minimizing the weighted point-to-line chi2."""
    center = points.mean(dim=0)
    half_width = 20.0
    for _ in range(10):
        axes = [torch.linspace(float(center[a]) - half_width,
                               float(center[a]) + half_width, 41,
                               dtype=torch.float64)
                for a in range(3)]
        grid = torch.cartesian_prod(*axes)
        separation = grid.unsqueeze(1) - points.unsqueeze(0)
        along = (separation * directions).sum(dim=2)
        perpendicular = separation - along.unsqueeze(2) * directions
        chi2 = (weights * perpendicular.square().sum(dim=2)).sum(dim=1)
        center = grid[int(chi2.argmin())]
        half_width *= 2.5 / 41
    return center, float(chi2.min())


def test_fit_recovers_a_known_common_vertex():
    vertex = torch.tensor([0.7, -0.3, 2.4], dtype=torch.float64)
    directions = torch.tensor([
        [1.0, 0.2, 0.1],
        [-0.3, 1.0, 0.4],
        [0.2, -0.5, 1.0],
    ], dtype=torch.float64)
    points, directions = _lines_through(
        vertex, directions, torch.tensor([-2.0, 3.0, 1.5], dtype=torch.float64))
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         torch.zeros(1, 3, dtype=torch.float64))
    # The relative Tikhonov ridge (1e-4, sized for BACKWARD stability on
    # near-parallel production triplets) biases an exact intersection at the
    # ~1e-4 level — far below any physical scale in the problem.
    assert torch.allclose(fit.vertex[0], vertex, atol=2e-3)
    assert float(fit.chi2[0]) == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(fit.residuals[0], torch.zeros(3, dtype=torch.float64),
                          atol=2e-3)


def test_fit_matches_a_brute_force_grid_minimizer():
    points, directions = _spread_lines()
    log_weights = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         log_weights.unsqueeze(0))
    expected_vertex, expected_chi2 = _brute_force_vertex(
        points, directions, log_weights.exp())
    assert torch.allclose(fit.vertex[0], expected_vertex, atol=2e-3)
    assert float(fit.chi2[0]) == pytest.approx(expected_chi2, rel=1e-3)


def test_chi2_equals_the_weighted_sum_of_squared_residuals():
    points, directions = _spread_lines()
    log_weights = torch.tensor([0.1, 0.4, -0.3], dtype=torch.float64)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         log_weights.unsqueeze(0))
    expected = float((log_weights.exp() * fit.residuals[0].square()).sum())
    assert float(fit.chi2[0]) == pytest.approx(expected, rel=1e-9)


def test_two_equal_weight_tracks_meet_at_the_closest_approach_midpoint():
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.5],
    ], dtype=torch.float64)
    directions = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.3, 0.2],
        [-0.4, 1.0, 0.6],
    ], dtype=torch.float64), dim=-1)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         torch.zeros(1, 2, dtype=torch.float64))
    _, midpoint = _closest_approach(
        points[0:1].float(), directions[0:1].float(),
        points[1:2].float(), directions[1:2].float())
    assert torch.allclose(fit.vertex[0].float(), midpoint[0], atol=2e-3)


def test_a_dominant_weight_pins_the_vertex_onto_that_line():
    points, directions = _spread_lines()
    log_weights = torch.tensor([30.0, 0.0, 0.0], dtype=torch.float64)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         log_weights.unsqueeze(0))
    assert float(fit.residuals[0, 0]) == pytest.approx(0.0, abs=2e-3)
    assert float(fit.residuals[0, 1]) > 1e-3


def test_fit_is_invariant_under_track_permutation():
    points, directions = _spread_lines()
    log_weights = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         log_weights.unsqueeze(0))
    order = torch.tensor([2, 0, 1])
    permuted = wls_vertex_fit(points[order].unsqueeze(0),
                              directions[order].unsqueeze(0),
                              log_weights[order].unsqueeze(0))
    assert torch.allclose(fit.vertex, permuted.vertex, atol=1e-9)
    assert torch.allclose(fit.residuals[0][order], permuted.residuals[0],
                          atol=1e-9)


def test_uniform_weight_rescaling_leaves_the_vertex_unchanged():
    points, directions = _spread_lines()
    log_weights = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    base = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                          log_weights.unsqueeze(0))
    scaled = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                            (log_weights + 3.0).unsqueeze(0))
    assert torch.allclose(base.vertex, scaled.vertex, atol=1e-9)


def test_parallel_tracks_stay_finite_with_finite_gradients():
    direction = torch.nn.functional.normalize(
        torch.tensor([0.2, 0.3, 1.0], dtype=torch.float64), dim=-1)
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float64).requires_grad_(True)
    directions = direction.repeat(3, 1)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         torch.zeros(1, 3, dtype=torch.float64))
    assert torch.isfinite(fit.vertex).all()
    assert torch.isfinite(fit.chi2).all()
    fit.chi2.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_non_unit_directions_are_normalized_internally():
    points, directions = _spread_lines()
    log_weights = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    unit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                          log_weights.unsqueeze(0))
    stretched = wls_vertex_fit(points.unsqueeze(0),
                               (directions * 7.0).unsqueeze(0),
                               log_weights.unsqueeze(0))
    assert torch.allclose(unit.vertex, stretched.vertex, atol=1e-9)
    assert torch.allclose(unit.chi2, stretched.chi2, atol=1e-9)


def test_gradcheck_through_the_regularized_solve():
    points, directions = _spread_lines()
    log_weights = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    points = points.clone().requires_grad_(True)
    directions = directions.clone().requires_grad_(True)
    log_weights = log_weights.clone().requires_grad_(True)

    def objective(p, d, w):
        fit = wls_vertex_fit(p.unsqueeze(0), d.unsqueeze(0), w.unsqueeze(0))
        return fit.vertex.sum() + fit.chi2.sum()

    assert torch.autograd.gradcheck(
        objective, (points, directions, log_weights), atol=1e-6)


def test_signed_arcs_are_negative_for_backward_crossings():
    # A vertex BEHIND the reference point along the direction of motion gives
    # a negative arc: place the reference beyond the common vertex.
    vertex = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    directions = torch.tensor([
        [1.0, 0.0, 0.1],
        [0.0, 1.0, 0.1],
        [1.0, 1.0, 0.3],
    ], dtype=torch.float64)
    points, directions = _lines_through(
        vertex, directions, torch.tensor([2.0, 2.0, -2.0], dtype=torch.float64))
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         torch.zeros(1, 3, dtype=torch.float64))
    arcs = fit.arcs[0]
    assert float(arcs[0]) == pytest.approx(-2.0, abs=2e-3)
    assert float(arcs[1]) == pytest.approx(-2.0, abs=2e-3)
    assert float(arcs[2]) == pytest.approx(2.0, abs=2e-3)


# ---------------------------------------------------------------------------
# physics_log_weights
# ---------------------------------------------------------------------------

def test_physics_log_weights_are_the_negative_log_position_variance():
    var_dxy = torch.tensor([[1e-4, 4e-4, 9e-4]])
    var_dsz = torch.tensor([[1e-4, 1e-4, 1e-4]])
    logw = physics_log_weights(var_dxy, var_dsz)
    expected = -torch.log(var_dxy + var_dsz + 1e-12)
    assert torch.allclose(logw, expected)


def test_physics_log_weights_survive_zero_variances():
    logw = physics_log_weights(torch.zeros(1, 3), torch.zeros(1, 3))
    assert torch.isfinite(logw).all()


# ---------------------------------------------------------------------------
# VertexFitLayer
# ---------------------------------------------------------------------------

def _layer_batch(batch=2, candidates=5, generator=None):
    generator = generator or torch.Generator().manual_seed(7)
    ref = torch.randn(batch, 3, 3, candidates, generator=generator) * 0.1
    eta = torch.randn(batch, 3, candidates, generator=generator) * 0.8
    phi = torch.randn(batch, 3, candidates, generator=generator) * math.pi / 2
    var_dxy = torch.rand(batch, 3, candidates, generator=generator) * 1e-3
    var_dsz = torch.rand(batch, 3, candidates, generator=generator) * 1e-3
    pv = torch.randn(batch, 3, generator=generator) * 0.01
    momentum = torch.randn(batch, 3, candidates, generator=generator) * 2.0
    mass = torch.rand(batch, candidates, generator=generator) + 0.5
    quality = torch.randn(batch, 12, 3, candidates, generator=generator)
    return dict(reference=ref, eta=eta, phi=phi, var_dxy=var_dxy,
                var_dsz=var_dsz, primary_vertex=pv, momentum=momentum,
                mass=mass, quality=quality)


def test_fit_names_layout():
    assert len(FIT_NAMES) == 21
    assert len(set(FIT_NAMES)) == 21
    pv_channels = [name for name in FIT_NAMES if name.startswith(FIT_PV_PREFIX)]
    assert pv_channels == ['fitpv_cos', 'fitpv_mcorr', 'fitpv_lxy']
    assert FIT_NAMES.index('fit_logw_i') > FIT_NAMES.index('fitpv_lxy')


def test_layer_emits_one_channel_per_fit_name():
    layer = VertexFitLayer()
    outputs = layer(**_layer_batch())
    assert outputs.shape == (2, len(FIT_NAMES), 5)
    assert torch.isfinite(outputs).all()


def test_zero_init_layer_reproduces_the_physics_weight_fit():
    layer = VertexFitLayer()
    batch = _layer_batch()
    outputs = layer(**batch)
    logw_start = FIT_NAMES.index('fit_logw_i')
    expected = physics_log_weights(
        batch['var_dxy'].permute(0, 2, 1).reshape(-1, 3),
        batch['var_dsz'].permute(0, 2, 1).reshape(-1, 3))
    produced = outputs[:, logw_start:logw_start + 3, :].permute(0, 2, 1)
    assert torch.allclose(produced.reshape(-1, 3), expected, atol=1e-5)


def test_only_the_pv_block_depends_on_the_primary_vertex():
    layer = VertexFitLayer()
    batch = _layer_batch()
    base = layer(**batch)
    moved = dict(batch)
    moved['primary_vertex'] = batch['primary_vertex'] + torch.tensor([0.3, -0.2, 0.5])
    shifted = layer(**moved)
    pv_indices = [index for index, name in enumerate(FIT_NAMES)
                  if name.startswith(FIT_PV_PREFIX)]
    other = [index for index in range(len(FIT_NAMES)) if index not in pv_indices]
    assert torch.allclose(base[:, other, :], shifted[:, other, :], atol=1e-6)
    assert not torch.allclose(base[:, pv_indices, :], shifted[:, pv_indices, :],
                              atol=1e-4)


def test_gradients_reach_the_weight_head_once_it_is_nonzero():
    layer = VertexFitLayer()
    with torch.no_grad():
        layer.weight_head[-1].weight.normal_(0.0, 0.1)
        layer.weight_head[-1].bias.fill_(0.05)
    outputs = layer(**_layer_batch())
    outputs.square().mean().backward()
    grads = [parameter.grad for parameter in layer.weight_head.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(float(grad.abs().sum()) > 0 for grad in grads)


def test_the_fit_block_is_permutation_consistent_across_members():
    # Swapping members i and j swaps their residual/arc/logw channels and
    # leaves the vertex-level channels unchanged.
    layer = VertexFitLayer()
    batch = _layer_batch()
    swapped = {key: value.clone() for key, value in batch.items()}
    for key in ('reference', 'eta', 'phi', 'var_dxy', 'var_dsz'):
        swapped[key][:, [0, 1]] = swapped[key][:, [1, 0]]
    swapped['quality'][:, :, [0, 1]] = swapped['quality'][:, :, [1, 0]]
    base = layer(**batch)
    permuted = layer(**swapped)
    for vertex_level in ('fit_chi2', 'fit_res_max', 'fit_sigma_xy', 'fit_sigma_z',
                         'fit_logdet_a', 'fit_lxy_beam'):
        index = FIT_NAMES.index(vertex_level)
        assert torch.allclose(base[:, index], permuted[:, index], atol=1e-5)
    res_i, res_j = FIT_NAMES.index('fit_res_i'), FIT_NAMES.index('fit_res_j')
    assert torch.allclose(base[:, res_i], permuted[:, res_j], atol=1e-5)
    assert torch.allclose(base[:, res_j], permuted[:, res_i], atol=1e-5)
