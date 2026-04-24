from __future__ import annotations

import math

import torch


@torch.jit.script
def delta_phi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1: torch.Tensor, phi1: torch.Tensor, eta2: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    m = torch.sqrt(to_m2(x, eps=eps))
    return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    return x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def _delta_r_squared(
    eta_a: torch.Tensor,
    phi_a: torch.Tensor,
    eta_b: torch.Tensor,
    phi_b: torch.Tensor,
) -> torch.Tensor:
    return (eta_a - eta_b).square() + delta_phi(phi_a, phi_b).square()


def cross_set_knn(
    query_coordinates: torch.Tensor,
    reference_coordinates: torch.Tensor,
    num_neighbors: int,
    reference_mask: torch.Tensor | None = None,
    query_reference_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """query_coordinates: (B, 2, M). reference_coordinates: (B, 2, P). Returns (B, M, K) long."""
    query_eta = query_coordinates[:, 0:1, :]
    query_phi = query_coordinates[:, 1:2, :]
    reference_eta = reference_coordinates[:, 0:1, :]
    reference_phi = reference_coordinates[:, 1:2, :]

    delta_eta = query_eta.unsqueeze(-1) - reference_eta.unsqueeze(-2)
    delta_phi_val = delta_phi(
        query_phi.unsqueeze(-1), reference_phi.unsqueeze(-2),
    )
    distances = (delta_eta.square() + delta_phi_val.square()).squeeze(1)

    if reference_mask is not None:
        distances = distances.masked_fill(~reference_mask.bool(), float('inf'))

    # Exclude self-matches so downstream log/sqrt on ΔR=0 pairs can't NaN.
    if query_reference_indices is not None:
        num_reference = distances.shape[-1]
        self_indices = query_reference_indices.unsqueeze(-1)
        reference_range = torch.arange(
            num_reference, device=distances.device,
        ).view(1, 1, -1)
        distances = distances.masked_fill(reference_range == self_indices, float('inf'))

    return distances.topk(k=num_neighbors, dim=-1, largest=False, sorted=False)[1]


def cross_set_gather(
    reference_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """reference_features: (B, C, P). neighbor_indices: (B, M, K). Returns (B, C, M, K)."""
    batch_size, num_channels, _ = reference_features.shape
    _, num_queries, num_neighbors = neighbor_indices.shape

    flat_indices = neighbor_indices.reshape(
        batch_size, 1, num_queries * num_neighbors,
    ).expand(batch_size, num_channels, num_queries * num_neighbors)
    gathered = reference_features.gather(dim=2, index=flat_indices)
    return gathered.view(batch_size, num_channels, num_queries, num_neighbors)
