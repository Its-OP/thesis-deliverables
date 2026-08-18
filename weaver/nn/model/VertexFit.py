from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

# Channel layout of the fit block. The fitpv_* prefix isolates every channel
# that depends on the stored primary vertex, which is mis-assigned in a large
# fraction of events — one flag can drop the whole block.
FIT_PV_PREFIX = 'fitpv_'
FIT_NAMES = [
    'fit_chi2',
    'fit_res_i', 'fit_res_j', 'fit_res_k', 'fit_res_max',
    'fit_arc_i', 'fit_arc_j', 'fit_arc_k',
    'fit_sigma_xy', 'fit_sigma_z', 'fit_logdet_a',
    'fit_lxy_beam', 'fit_cos_xy_beam', 'fit_mcorr_beam', 'fit_dlen_sig_beam',
    'fitpv_cos', 'fitpv_mcorr', 'fitpv_lxy',
    'fit_logw_i', 'fit_logw_j', 'fit_logw_k',
]

_TIKHONOV_RELATIVE = 1e-6
_TIKHONOV_FLOOR = 1e-9
_VARIANCE_FLOOR = 1e-12
_EPSILON = 1e-12


class WLSFit(NamedTuple):
    vertex: torch.Tensor
    residuals: torch.Tensor
    arcs: torch.Tensor
    chi2: torch.Tensor
    sigma_xy: torch.Tensor
    sigma_z: torch.Tensor
    log_det_a: torch.Tensor


def physics_log_weights(variance_dxy: torch.Tensor,
                        variance_dsz: torch.Tensor) -> torch.Tensor:
    """variance_dxy, variance_dsz: (..., T). Returns (..., T)."""
    return -torch.log(variance_dxy + variance_dsz + _VARIANCE_FLOOR)


def wls_vertex_fit(points: torch.Tensor, directions: torch.Tensor,
                   log_weights: torch.Tensor) -> WLSFit:
    """points, directions: (..., T, 3); log_weights: (..., T).
    Returns per-batch vertex (..., 3), residuals/arcs (..., T), and scalars
    chi2 / sigma_xy / sigma_z / log_det_a (...,)."""
    dtype = points.dtype
    points64 = points.double()
    directions64 = torch.nn.functional.normalize(directions.double(), dim=-1)
    log_weights64 = log_weights.double()

    # Mean-shifted weights make the solve exactly invariant under a uniform
    # rescaling of the weights; reported quantities are corrected back below.
    mean_log_weight = log_weights64.mean(dim=-1, keepdim=True)
    shifted_weights = (log_weights64 - mean_log_weight).exp()

    identity = torch.eye(3, dtype=torch.float64, device=points.device)
    identity = identity.expand(*points64.shape[:-1], 3, 3)
    projectors = identity - directions64.unsqueeze(-1) * directions64.unsqueeze(-2)

    weighted = shifted_weights.unsqueeze(-1).unsqueeze(-1) * projectors
    matrix = weighted.sum(dim=-3)
    rhs = (weighted @ points64.unsqueeze(-1)).sum(dim=-3)
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    ridge = (_TIKHONOV_RELATIVE * trace / 3.0 + _TIKHONOV_FLOOR)
    regularized = matrix + ridge.unsqueeze(-1).unsqueeze(-1) \
        * torch.eye(3, dtype=torch.float64, device=points.device)
    vertex = torch.linalg.solve(regularized, rhs).squeeze(-1)

    separation = vertex.unsqueeze(-2) - points64
    arcs = (separation * directions64).sum(dim=-1)
    perpendicular = separation - arcs.unsqueeze(-1) * directions64
    residuals = perpendicular.square().sum(dim=-1).clamp_min(0.0).sqrt()
    chi2 = (log_weights64.exp() * residuals.square()).sum(dim=-1)

    covariance = torch.linalg.inv(regularized) \
        * (-mean_log_weight.squeeze(-1)).exp().unsqueeze(-1).unsqueeze(-1)
    diagonal = covariance.diagonal(dim1=-2, dim2=-1)
    sigma_xy = (diagonal[..., 0] + diagonal[..., 1]).clamp_min(0.0).sqrt()
    sigma_z = diagonal[..., 2].clamp_min(0.0).sqrt()
    log_det_a = torch.logdet(regularized) \
        + 3.0 * mean_log_weight.squeeze(-1)

    return WLSFit(vertex.to(dtype), residuals.to(dtype), arcs.to(dtype),
                  chi2.to(dtype), sigma_xy.to(dtype), sigma_z.to(dtype),
                  log_det_a.to(dtype))


def _corrected_mass(mass: torch.Tensor,
                    transverse_momentum: torch.Tensor) -> torch.Tensor:
    """mass, transverse_momentum: (M,). Returns (M,)."""
    return (mass.square() + transverse_momentum.square()).sqrt() \
        + transverse_momentum


class VertexFitLayer(nn.Module):
    def __init__(self, quality_channels: int = 12, hidden_dim: int = 16):
        super().__init__()
        self.weight_head = nn.Sequential(
            nn.Linear(quality_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-initialized correction: at epoch 0 the layer IS the
        # covariance-weighted physics fit.
        nn.init.zeros_(self.weight_head[-1].weight)
        nn.init.zeros_(self.weight_head[-1].bias)

    def forward(self, *, reference: torch.Tensor, eta: torch.Tensor,
                phi: torch.Tensor, var_dxy: torch.Tensor,
                var_dsz: torch.Tensor, primary_vertex: torch.Tensor,
                momentum: torch.Tensor, mass: torch.Tensor,
                quality: torch.Tensor) -> torch.Tensor:
        """reference: (B, 3, 3, N) member-major; eta, phi, var_dxy, var_dsz:
        (B, 3, N); primary_vertex: (B, 3); momentum: (B, 3, N) candidate sum;
        mass: (B, N); quality: (B, Q, 3, N). Returns (B, len(FIT_NAMES), N)."""
        batch, _, _, candidates = reference.shape
        flat = batch * candidates

        points = reference.permute(0, 3, 1, 2).reshape(flat, 3, 3)
        eta_flat = eta.permute(0, 2, 1).reshape(flat, 3)
        phi_flat = phi.permute(0, 2, 1).reshape(flat, 3)
        cosh_eta = torch.cosh(eta_flat)
        directions = torch.stack([
            torch.cos(phi_flat) / cosh_eta,
            torch.sin(phi_flat) / cosh_eta,
            torch.tanh(eta_flat),
        ], dim=-1)

        base_log_weights = physics_log_weights(
            var_dxy.permute(0, 2, 1).reshape(flat, 3),
            var_dsz.permute(0, 2, 1).reshape(flat, 3))
        correction = self.weight_head(
            quality.permute(0, 3, 2, 1).reshape(flat, 3, -1)).squeeze(-1)
        log_weights = base_log_weights + correction

        fit = wls_vertex_fit(points, directions, log_weights)

        momentum_flat = momentum.permute(0, 2, 1).reshape(flat, 3)
        mass_flat = mass.reshape(flat)
        vertex = fit.vertex

        # Beam-frame block: flight direction taken from the beamline origin in
        # the transverse plane — independent of the stored primary vertex.
        lxy_beam = vertex[:, :2].square().sum(dim=-1).clamp_min(0.0).sqrt()
        pt_total = momentum_flat[:, :2].square().sum(dim=-1) \
            .clamp_min(0.0).sqrt()
        cos_xy_beam = (vertex[:, 0] * momentum_flat[:, 0]
                       + vertex[:, 1] * momentum_flat[:, 1]) \
            / (lxy_beam * pt_total + _EPSILON)
        transverse_xy = (momentum_flat[:, 0] * vertex[:, 1]
                         - momentum_flat[:, 1] * vertex[:, 0]).abs() \
            / (lxy_beam + _EPSILON)
        mcorr_beam = _corrected_mass(mass_flat, transverse_xy)
        dlen_sig_beam = lxy_beam / (fit.sigma_xy + _EPSILON)

        # PV block: everything downstream of the stored primary vertex.
        flight = vertex - primary_vertex.unsqueeze(1) \
            .expand(batch, candidates, 3).reshape(flat, 3)
        flight_norm = flight.square().sum(dim=-1).clamp_min(0.0).sqrt()
        momentum_norm = momentum_flat.square().sum(dim=-1).clamp_min(0.0).sqrt()
        pv_cos = (flight * momentum_flat).sum(dim=-1) \
            / (flight_norm * momentum_norm + _EPSILON)
        flight_unit = flight / (flight_norm + _EPSILON).unsqueeze(-1)
        along = (momentum_flat * flight_unit).sum(dim=-1, keepdim=True)
        perpendicular = momentum_flat - along * flight_unit
        transverse_3d = perpendicular.square().sum(dim=-1).clamp_min(0.0).sqrt()
        pv_mcorr = _corrected_mass(mass_flat, transverse_3d)
        pv_lxy = flight[:, :2].square().sum(dim=-1).clamp_min(0.0).sqrt()

        channels = torch.stack([
            fit.chi2,
            fit.residuals[:, 0], fit.residuals[:, 1], fit.residuals[:, 2],
            fit.residuals.max(dim=-1).values,
            fit.arcs[:, 0], fit.arcs[:, 1], fit.arcs[:, 2],
            fit.sigma_xy, fit.sigma_z, fit.log_det_a,
            lxy_beam, cos_xy_beam, mcorr_beam, dlen_sig_beam,
            pv_cos, pv_mcorr, pv_lxy,
            log_weights[:, 0], log_weights[:, 1], log_weights[:, 2],
        ], dim=-1)
        return channels.reshape(batch, candidates, len(FIT_NAMES)) \
            .permute(0, 2, 1)
