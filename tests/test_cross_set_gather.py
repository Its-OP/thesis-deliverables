"""Equivalence + correctness tests for ``cross_set_gather``.

Introduced as part of the HierarchicalGraphBackbone.cross_set_gather
rewrite (transpose + reshape + advanced-index + permute + contiguous →
single ``torch.gather`` call). These tests lock in the semantics against
a reference implementation preserved here so that any subsequent
regression is caught regardless of which impl ships in the source tree.

Shape contract:
    ``cross_set_gather((B, C, P), (B, M, K)) → (B, C, M, K)``
    where ``neighbor_indices[b, m, k] ∈ [0, P)``.
"""
from __future__ import annotations

import pytest
import torch

from weaver.nn.model.HierarchicalGraphBackbone import cross_set_gather


def _cross_set_gather_reference(
    reference_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """Original transpose/reshape/index/permute/contiguous impl.

    Kept as the ground-truth reference so tests pin the new impl to the
    exact numerical behavior that shipped through epochs 1-40 of the
    compile-on sweep.
    """
    batch_size, num_channels, num_reference_points = reference_features.shape
    _, num_queries, num_neighbors = neighbor_indices.shape
    batch_offset = (
        torch.arange(batch_size, device=reference_features.device)
        .view(-1, 1, 1) * num_reference_points
    )
    flat_indices = (neighbor_indices + batch_offset).reshape(-1)
    flat_features = reference_features.transpose(1, 2).reshape(-1, num_channels)
    gathered = flat_features[flat_indices]
    gathered = gathered.view(
        batch_size, num_queries, num_neighbors, num_channels,
    )
    gathered = gathered.permute(0, 3, 1, 2).contiguous()
    return gathered


def _random_inputs(
    batch_size: int,
    num_channels: int,
    num_reference_points: int,
    num_queries: int,
    num_neighbors: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random `(features, indices)` matching the contract."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    features = torch.randn(
        batch_size, num_channels, num_reference_points,
        dtype=dtype, device=device, generator=generator,
    )
    indices = torch.randint(
        0, num_reference_points,
        (batch_size, num_queries, num_neighbors),
        device=device, generator=generator,
    )
    return features, indices


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'batch_size, num_channels, num_reference_points, num_queries, num_neighbors',
    [
        (2, 8, 64, 64, 6),           # baseline: M == P
        (1, 16, 128, 32, 4),         # M < P (hierarchical centroids)
        (4, 1, 32, 32, 8),           # single channel
        (3, 4, 16, 16, 1),           # K=1
        (1, 2, 5, 5, 5),             # tiny, M == K
    ],
)
def test_shape(
    batch_size, num_channels, num_reference_points,
    num_queries, num_neighbors,
):
    """Output shape matches ``(B, C, M, K)`` for a battery of sizes."""
    features, indices = _random_inputs(
        batch_size, num_channels, num_reference_points,
        num_queries, num_neighbors,
    )
    gathered = cross_set_gather(features, indices)
    assert gathered.shape == (
        batch_size, num_channels, num_queries, num_neighbors,
    )


# ---------------------------------------------------------------------------
# Value parity with reference impl
# ---------------------------------------------------------------------------
def test_values_match_reference_fp32():
    """fp32 values bit-match the original transpose/reshape impl."""
    features, indices = _random_inputs(2, 8, 64, 16, 6, seed=1)
    new_output = cross_set_gather(features, indices)
    reference_output = _cross_set_gather_reference(features, indices)
    assert torch.equal(new_output, reference_output)


def test_values_match_reference_fp16():
    """fp16 path (AMP forward): values match reference within fp16 tol."""
    features, indices = _random_inputs(
        2, 8, 64, 16, 6, dtype=torch.float16, seed=2,
    )
    new_output = cross_set_gather(features, indices)
    reference_output = _cross_set_gather_reference(features, indices)
    assert torch.allclose(new_output, reference_output, atol=1e-3, rtol=1e-3)


def test_values_match_reference_uneven_shapes():
    """Non-square hierarchical setup: M != P and C != 1."""
    features, indices = _random_inputs(3, 7, 50, 20, 4, seed=3)
    new_output = cross_set_gather(features, indices)
    reference_output = _cross_set_gather_reference(features, indices)
    assert torch.equal(new_output, reference_output)


# ---------------------------------------------------------------------------
# Gradient parity
# ---------------------------------------------------------------------------
def test_gradients_match_reference():
    """input.grad matches the reference after `.sum().backward()`."""
    features_new, indices = _random_inputs(2, 4, 32, 16, 5, seed=4)
    features_new = features_new.clone().requires_grad_(True)
    features_ref = features_new.detach().clone().requires_grad_(True)

    cross_set_gather(features_new, indices).sum().backward()
    _cross_set_gather_reference(features_ref, indices).sum().backward()

    assert torch.equal(features_new.grad, features_ref.grad)


def test_gradients_match_reference_weighted():
    """Weighted backward (not uniform sum) still matches."""
    features_new, indices = _random_inputs(2, 4, 32, 16, 5, seed=5)
    features_new = features_new.clone().requires_grad_(True)
    features_ref = features_new.detach().clone().requires_grad_(True)

    new_output = cross_set_gather(features_new, indices)
    reference_output = _cross_set_gather_reference(features_ref, indices)
    weights = torch.randn_like(new_output)
    (new_output * weights).sum().backward()
    (reference_output * weights.clone()).sum().backward()

    assert torch.allclose(
        features_new.grad, features_ref.grad, atol=1e-6, rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Semantic correctness
# ---------------------------------------------------------------------------
def test_single_channel_identity_selection():
    """When features are 1:P along the P axis, output equals the index itself."""
    num_reference_points = 10
    features = (
        torch.arange(num_reference_points, dtype=torch.float32)
        .view(1, 1, num_reference_points)
    )
    indices = torch.tensor([[[3, 1, 9, 0]]])  # (1, 1, 4)
    gathered = cross_set_gather(features, indices)
    expected = torch.tensor([[[[3.0, 1.0, 9.0, 0.0]]]])
    assert torch.equal(gathered, expected)


def test_all_channels_share_neighbor():
    """Every channel reads the same index for a given (m, k)."""
    features, indices = _random_inputs(1, 5, 8, 4, 3, seed=6)
    gathered = cross_set_gather(features, indices)
    # Compare channel 0 and channel 3 at a specific (m, k).
    for query_index in range(4):
        for neighbor_index in range(3):
            reference_index = int(indices[0, query_index, neighbor_index])
            for channel_index in range(5):
                assert gathered[0, channel_index, query_index, neighbor_index] == \
                    features[0, channel_index, reference_index]


# ---------------------------------------------------------------------------
# CUDA parity (skipped when no GPU)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA not available',
)
def test_cuda_parity_values_and_grads():
    """Same values and grads on CUDA as on CPU."""
    features, indices = _random_inputs(2, 4, 32, 16, 5, seed=7)
    cpu_output = cross_set_gather(features, indices)

    features_cuda = features.to('cuda').requires_grad_(True)
    indices_cuda = indices.to('cuda')
    cuda_output = cross_set_gather(features_cuda, indices_cuda)

    assert torch.allclose(cuda_output.cpu(), cpu_output, atol=1e-6)
    cuda_output.sum().backward()
    cpu_grad = torch.ones_like(cpu_output)  # Uniform weight equivalent.
    # Build cpu grad via autograd for an exact comparison.
    features_cpu_grad = features.clone().requires_grad_(True)
    cross_set_gather(features_cpu_grad, indices).sum().backward()
    assert torch.allclose(
        features_cuda.grad.cpu(), features_cpu_grad.grad, atol=1e-6,
    )
