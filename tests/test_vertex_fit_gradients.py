from __future__ import annotations

import pytest
import torch

from weaver.nn.model.VertexFit import (VertexFitLayer, physics_log_weights,
                                       wls_vertex_fit)


def test_backward_is_finite_at_an_exact_intersection():
    # Zero residuals sit exactly at sqrt(0), whose unfloored gradient is
    # infinite — the failure that NaN'd training on GT-like triplets.
    vertex = torch.tensor([0.1, -0.05, 0.9], dtype=torch.float64)
    directions = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.1, 0.2],
        [-0.2, 1.0, 0.3],
        [0.3, -0.4, 1.0],
    ], dtype=torch.float64), dim=-1)
    points = (vertex.unsqueeze(0)
              + torch.tensor([[-2.0], [1.5], [3.0]], dtype=torch.float64)
              * directions).requires_grad_(True)
    log_weights = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    fit = wls_vertex_fit(points.unsqueeze(0), directions.unsqueeze(0),
                         log_weights)
    (fit.residuals.sum() + fit.chi2.sum() + fit.sigma_xy.sum()).backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(log_weights.grad).all()


def test_physics_log_weights_is_finite_on_garbage_variance():
    variance_dxy = torch.tensor([[-1.0, 0.0, float('inf'), float('nan')]])
    variance_dsz = torch.zeros(1, 4)
    log_weights = physics_log_weights(variance_dxy, variance_dsz)
    assert torch.isfinite(log_weights).all()


def _clean_layer_inputs(candidates: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    return dict(
        reference=rand(1, 3, 3, candidates) * 0.1,
        eta=rand(1, 3, candidates) * 0.8,
        phi=rand(1, 3, candidates),
        var_dxy=rand(1, 3, candidates).square() * 1e-3 + 1e-5,
        var_dsz=rand(1, 3, candidates).square() * 1e-3 + 1e-5,
        primary_vertex=torch.zeros(1, 3),
        momentum=rand(1, 3, candidates),
        mass=torch.full((1, candidates), 0.7),
        quality=rand(1, 12, 3, candidates),
    )


@pytest.mark.parametrize('poison', ['negative_variance', 'inf_variance',
                                    'inf_quality'])
def test_poisoned_candidate_does_not_corrupt_clean_gradients(poison):
    # A single candidate with garbage inputs must neither produce non-finite
    # channels nor leak NaN into the shared weight-head gradients of the
    # clean candidates — the failure that killed S3 training.
    torch.manual_seed(3)
    layer = VertexFitLayer()
    with torch.no_grad():
        for parameter in layer.weight_head.parameters():
            parameter.normal_(0.0, 0.1)

    inputs = _clean_layer_inputs(2)
    if poison == 'negative_variance':
        inputs['var_dxy'][0, 0, 1] = -1.0
    elif poison == 'inf_variance':
        inputs['var_dxy'][0, :, 1] = float('inf')
    else:
        inputs['quality'][0, 3, :, 1] = float('inf')

    clean_only = {key: value if key == 'primary_vertex' else value[..., :1]
                  for key, value in inputs.items()}
    layer(**clean_only).sum().backward()
    reference_gradients = [parameter.grad.clone()
                           for parameter in layer.weight_head.parameters()]

    layer.zero_grad()
    outputs = layer(**inputs)
    assert torch.isfinite(outputs).all()
    outputs[..., 0].sum().backward()
    for parameter, reference in zip(layer.weight_head.parameters(),
                                    reference_gradients):
        assert torch.isfinite(parameter.grad).all()
        assert torch.allclose(parameter.grad, reference, atol=1e-6)


def test_layer_backward_is_finite_on_degenerate_geometry():
    layer = VertexFitLayer()
    with torch.no_grad():
        layer.weight_head[-1].weight.normal_(0.0, 0.1)
    # Members share one reference point AND the primary vertex sits on it:
    # every norm in the beam/PV blocks passes through zero.
    reference = torch.zeros(1, 3, 3, 2)
    eta = torch.zeros(1, 3, 2)
    phi = torch.tensor([[[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]])
    quality = torch.randn(1, 12, 3, 2, requires_grad=True)
    outputs = layer(reference=reference, eta=eta, phi=phi,
                    var_dxy=torch.full((1, 3, 2), 1e-4),
                    var_dsz=torch.full((1, 3, 2), 1e-4),
                    primary_vertex=torch.zeros(1, 3),
                    momentum=torch.ones(1, 3, 2),
                    mass=torch.full((1, 2), 0.7),
                    quality=quality)
    outputs.sum().backward()
    assert torch.isfinite(outputs).all()
    assert torch.isfinite(quality.grad).all()
