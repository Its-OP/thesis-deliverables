from __future__ import annotations

import torch

from weaver.nn.model.VertexFit import VertexFitLayer, wls_vertex_fit


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
