"""Track and vertex geometry for the MiniAOD source (numpy only).

Tracks are straight lines through their reference point (the point of closest
approach to the associated primary vertex) along the momentum direction at
that point; within the few millimetres around the beam line that matter here
the neglected curvature is below the track resolution.

Position-error basis of a track with azimuth phi and pseudorapidity eta
(sin(lambda) = tanh(eta), cos(lambda) = 1 / cosh(eta)):
    dxy axis: (-sin(phi), cos(phi), 0)
    dsz axis: (-cos(phi) sin(lambda), -sin(phi) sin(lambda), cos(lambda))
with variances cov(dxy, dxy), cov(dsz, dsz) and covariance cov(dxy, dsz) of
the track covariance matrix.
"""
from __future__ import annotations

import numpy as np

PARALLEL_TOLERANCE = 1e-12


def delta_r(eta_a, phi_a, eta_b, phi_b):
    """eta_a, phi_a, eta_b, phi_b: broadcastable arrays. Returns deltaR with
    the azimuth difference wrapped into [-pi, pi)."""
    delta_phi = np.mod(np.asarray(phi_a) - np.asarray(phi_b) + np.pi, 2.0 * np.pi) - np.pi
    return np.hypot(np.asarray(eta_a) - np.asarray(eta_b), delta_phi)


def unit_direction(eta, phi):
    """eta, phi: (N,). Returns (N, 3) unit momentum directions."""
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    return np.stack([np.cos(phi), np.sin(phi), np.sinh(eta)], axis=-1) / np.cosh(eta)[..., None]


def line_distance(points_a, directions_a, points_b, directions_b):
    """points_a, directions_a: (N, 3) lines; points_b, directions_b: (N, 3) or
    (3,) lines (unit directions). Returns the (N,) distances of closest
    approach and the (N, 3) midpoints of the closest-approach segments; for
    parallel lines the midpoint is taken at points_a."""
    points_a = np.asarray(points_a, dtype=np.float64)
    directions_a = np.asarray(directions_a, dtype=np.float64)
    points_b = np.broadcast_to(np.asarray(points_b, dtype=np.float64), points_a.shape)
    directions_b = np.broadcast_to(np.asarray(directions_b, dtype=np.float64), directions_a.shape)
    offset = points_b - points_a
    cosine = np.einsum('ij,ij->i', directions_a, directions_b)
    along_a = np.einsum('ij,ij->i', offset, directions_a)
    along_b = np.einsum('ij,ij->i', offset, directions_b)
    denominator = 1.0 - cosine ** 2
    parallel = denominator < PARALLEL_TOLERANCE
    safe = np.where(parallel, 1.0, denominator)
    step_a = np.where(parallel, 0.0, (along_a - cosine * along_b) / safe)
    step_b = np.where(parallel, -along_b, (cosine * along_a - along_b) / safe)
    closest_a = points_a + step_a[:, None] * directions_a
    closest_b = points_b + step_b[:, None] * directions_b
    return np.linalg.norm(closest_b - closest_a, axis=1), 0.5 * (closest_a + closest_b)


def point_line_offset(points, directions, target):
    """points, directions: (N, 3) lines (unit directions); target: (3,) or
    (N, 3). Returns the (N, 3) vectors from each line to the target,
    perpendicular to the line."""
    points = np.asarray(points, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    offset = np.broadcast_to(np.asarray(target, dtype=np.float64), points.shape) - points
    along = np.einsum('ij,ij->i', offset, directions)
    return offset - along[:, None] * directions


def perpendicular_variance(normal, phi, eta, cov_dxy_dxy, cov_dsz_dsz, cov_dxy_dsz):
    """normal: (N, 3) unit vectors; phi, eta: (N,) track angles; cov_*: (N,)
    track covariance terms. Returns the (N,) variance of the track position
    along normal."""
    phi = np.asarray(phi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    sin_lambda, cos_lambda = np.tanh(eta), 1.0 / np.cosh(eta)
    dxy_axis = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    dsz_axis = np.stack([-np.cos(phi) * sin_lambda, -np.sin(phi) * sin_lambda, cos_lambda], axis=-1)
    along_dxy = np.einsum('ij,ij->i', normal, dxy_axis)
    along_dsz = np.einsum('ij,ij->i', normal, dsz_axis)
    return (along_dxy ** 2 * cov_dxy_dxy + along_dsz ** 2 * cov_dsz_dsz
            + 2.0 * along_dxy * along_dsz * cov_dxy_dsz)


def vertex_distance(reference, reference_covariance, positions, covariances):
    """reference: (3,) vertex; reference_covariance: (3, 3); positions:
    (N, 3) vertices; covariances: (N, 3, 3). Returns a dict of (N,) arrays:
    distance_3d, significance_3d, distance_xy, significance_xy, each
    significance being distance / sqrt(d^T (C_ref + C) d) * |d|."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 3, 3)
    total = covariances + np.asarray(reference_covariance, dtype=np.float64)[None, :, :]
    result = {}
    for label, mask in (('3d', np.ones(3)), ('xy', np.array([1.0, 1.0, 0.0]))):
        difference = (positions - np.asarray(reference, dtype=np.float64)[None, :]) * mask[None, :]
        distance = np.linalg.norm(difference, axis=1)
        projected = np.einsum('ni,nij,nj->n', difference, total, difference)
        error = np.sqrt(np.maximum(projected, 0.0)) / np.where(distance > 0, distance, 1.0)
        result[f'distance_{label}'] = distance
        result[f'significance_{label}'] = np.where(error > 0, distance / np.where(error > 0, error, 1.0), 0.0)
    return result


def pair_mass(pt_a, eta_a, phi_a, mass_a, pt_b, eta_b, phi_b, mass_b):
    """pt, eta, phi, mass of two particles: broadcastable arrays. Returns
    the invariant mass of each pair."""
    def four_vector(pt, eta, phi, mass):
        pt, eta, phi = (np.asarray(value, dtype=np.float64) for value in (pt, eta, phi))
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        return px, py, pz, np.sqrt(px ** 2 + py ** 2 + pz ** 2 + np.asarray(mass, dtype=np.float64) ** 2)
    px_a, py_a, pz_a, energy_a = four_vector(pt_a, eta_a, phi_a, mass_a)
    px_b, py_b, pz_b, energy_b = four_vector(pt_b, eta_b, phi_b, mass_b)
    mass_squared = ((energy_a + energy_b) ** 2 - (px_a + px_b) ** 2 - (py_a + py_b) ** 2 - (pz_a + pz_b) ** 2)
    return np.sqrt(np.maximum(mass_squared, 0.0))


def beam_spot_dca(vertex_x, vertex_y, track_phi, beam_x, beam_y):
    """vertex_x, vertex_y: (N,) track reference point; track_phi: (N,);
    beam_x, beam_y: beam-spot position. Returns the (N,) signed transverse
    impact parameter with respect to the beam-spot point."""
    track_phi = np.asarray(track_phi, dtype=np.float64)
    return (-(np.asarray(vertex_x, dtype=np.float64) - beam_x) * np.sin(track_phi)
            + (np.asarray(vertex_y, dtype=np.float64) - beam_y) * np.cos(track_phi))
