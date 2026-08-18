from __future__ import annotations

import torch

from weaver.nn.model.VertexFit import (
    FIT_NAMES,
    FIT_PV_PREFIX,
    VertexFitLayer,
)

__all__ = ['FIT_NAMES', 'FIT_PV_PREFIX', 'static_fit_columns']

_LAYER = VertexFitLayer()


def static_fit_columns(
    i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, *,
    lorentz: torch.Tensor,
    eta: torch.Tensor, phi: torch.Tensor,
    vertex_x: torch.Tensor, vertex_y: torch.Tensor, vertex_z: torch.Tensor,
    var_dxy: torch.Tensor, var_dsz: torch.Tensor,
    primary_vertex: torch.Tensor,
    **_ignored: torch.Tensor,
) -> torch.Tensor:
    """i, j, k: (M,) member indices; lorentz: (4, T); eta, phi, vertex_*,
    var_*: (T,); primary_vertex: (3,). Returns (M, len(FIT_NAMES)) — the
    zero-init VertexFitLayer's channels with fixed physics weights."""
    members = torch.stack([i, j, k], dim=0)
    reference = torch.stack([
        torch.stack([vertex_x[members[m]], vertex_y[members[m]],
                     vertex_z[members[m]]], dim=0)
        for m in range(3)], dim=0).unsqueeze(0)
    momentum = sum(lorentz[:3, members[m]] for m in range(3)).unsqueeze(0)
    energy = sum(lorentz[3, members[m]] for m in range(3))
    mass = (energy.square() - momentum[0].square().sum(dim=0)) \
        .clamp_min(0.0).sqrt().unsqueeze(0)
    with torch.no_grad():
        outputs = _LAYER(
            reference=reference,
            eta=torch.stack([eta[members[m]] for m in range(3)]).unsqueeze(0),
            phi=torch.stack([phi[members[m]] for m in range(3)]).unsqueeze(0),
            var_dxy=torch.stack(
                [var_dxy[members[m]] for m in range(3)]).unsqueeze(0),
            var_dsz=torch.stack(
                [var_dsz[members[m]] for m in range(3)]).unsqueeze(0),
            primary_vertex=primary_vertex.unsqueeze(0),
            momentum=momentum,
            mass=mass,
            quality=torch.zeros(1, 12, 3, int(i.shape[0])),
        )
    return outputs[0].T.contiguous()
