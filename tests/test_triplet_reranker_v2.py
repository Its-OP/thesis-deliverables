from __future__ import annotations

import math

import pytest
import torch

from weaver.nn.model.TripletReranker import (
    TripletReranker,
    full_list_softmax_ce_loss,
)
from weaver.nn.model.VertexFit import FIT_NAMES


def _flat_model(**kwargs):
    defaults = dict(input_mode='flat', feature_dim=12, hidden_dim=32,
                    num_residual_blocks=1, loss_mode='full')
    defaults.update(kwargs)
    return TripletReranker(**defaults)


# ---------------------------------------------------------------------------
# Tail-weighted loss
# ---------------------------------------------------------------------------

def test_zero_log_weights_reduce_to_the_unweighted_loss():
    generator = torch.Generator().manual_seed(0)
    scores = torch.randn(3, 8, generator=generator)
    pos = torch.zeros(3, 8, dtype=torch.bool)
    pos[:, 0] = True
    valid = torch.ones(3, 8, dtype=torch.bool)
    base = full_list_softmax_ce_loss(scores, pos, valid, temperature=1.0,
                                     label_smoothing=0.1)
    weighted = full_list_softmax_ce_loss(scores, pos, valid, temperature=1.0,
                                         label_smoothing=0.1,
                                         log_weights=torch.zeros(3, 8))
    assert torch.allclose(base, weighted, atol=1e-7)


def test_a_log2_weight_equals_duplicating_the_row():
    scores = torch.tensor([[2.0, 1.0, 0.5]])
    pos = torch.tensor([[True, False, False]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    log_weights = torch.tensor([[0.0, 0.0, math.log(2.0)]])
    weighted = full_list_softmax_ce_loss(scores, pos, valid, temperature=1.0,
                                         label_smoothing=0.1,
                                         log_weights=log_weights)
    duplicated_scores = torch.tensor([[2.0, 1.0, 0.5, 0.5]])
    duplicated_pos = torch.tensor([[True, False, False, False]])
    duplicated = full_list_softmax_ce_loss(
        duplicated_scores, duplicated_pos, torch.ones(1, 4, dtype=torch.bool),
        temperature=1.0, label_smoothing=0.1)
    assert torch.allclose(weighted, duplicated, atol=1e-6)


def test_weights_apply_only_to_negatives():
    scores = torch.tensor([[2.0, 1.0]])
    pos = torch.tensor([[True, False]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    # A weight on the positive row must not change anything.
    weighted = full_list_softmax_ce_loss(
        scores, pos, valid, temperature=1.0, label_smoothing=0.0,
        log_weights=torch.tensor([[5.0, 0.0]]))
    base = full_list_softmax_ce_loss(scores, pos, valid, temperature=1.0,
                                     label_smoothing=0.0)
    assert torch.allclose(weighted, base, atol=1e-6)


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

def test_fusion_model_reproduces_the_filter_ordering_at_init():
    model = _flat_model(fusion=True)
    model.eval()
    features = torch.randn(2, 12, 7)
    filter_logit = torch.randn(2, 7)
    with torch.no_grad():
        scores = model(features, valid_mask=torch.ones(2, 7, dtype=torch.bool),
                       filter_logit=filter_logit)
    assert torch.allclose(scores, filter_logit, atol=1e-6)


def test_fusion_alpha_scales_the_anchor():
    model = _flat_model(fusion=True)
    with torch.no_grad():
        model.fusion_alpha.fill_(2.0)
    model.eval()
    filter_logit = torch.randn(1, 5)
    with torch.no_grad():
        scores = model(torch.randn(1, 12, 5),
                       valid_mask=torch.ones(1, 5, dtype=torch.bool),
                       filter_logit=filter_logit)
    assert torch.allclose(scores, 2.0 * filter_logit, atol=1e-6)


def test_fusion_requires_the_filter_logit():
    model = _flat_model(fusion=True)
    with pytest.raises(ValueError, match='filter_logit'):
        model(torch.randn(1, 12, 4),
              valid_mask=torch.ones(1, 4, dtype=torch.bool))


def test_gradient_reaches_the_residual_trunk_through_fusion():
    model = _flat_model(fusion=True)
    batch = dict(features=torch.randn(2, 12, 6),
                 pos_mask=torch.zeros(2, 6, dtype=torch.bool),
                 valid_mask=torch.ones(2, 6, dtype=torch.bool),
                 filter_logit=torch.randn(2, 6))
    batch['pos_mask'][:, 0] = True
    out = model.compute_loss(**batch)
    out['total_loss'].backward()
    first_conv = model.input_projection[0].weight
    assert first_conv.grad is not None
    assert torch.isfinite(first_conv.grad).all()


# ---------------------------------------------------------------------------
# LayerNorm trunk
# ---------------------------------------------------------------------------

def test_layer_norm_scores_are_batch_invariant():
    model = _flat_model(trunk_norm='layer')
    model.eval()
    features = torch.randn(1, 12, 5)
    with torch.no_grad():
        alone = model(features, valid_mask=torch.ones(1, 5, dtype=torch.bool))
        padded_features = torch.zeros(3, 12, 9)
        padded_features[0, :, :5] = features[0]
        padded_features[1] = torch.randn(12, 9)
        padded_features[2] = torch.randn(12, 9)
        valid = torch.zeros(3, 9, dtype=torch.bool)
        valid[0, :5] = True
        valid[1:] = True
        batched = model(padded_features, valid_mask=valid)
    # Different batch shapes take different kernel paths; equality holds to
    # float rounding, which is the invariance that matters.
    assert torch.allclose(alone[0], batched[0, :5], atol=1e-6)


def test_batch_norm_scores_are_not_batch_invariant():
    # The documented pathology the LN option removes.
    model = _flat_model(trunk_norm='batch')
    model.eval()
    features = torch.randn(1, 12, 5)
    with torch.no_grad():
        alone = model(features, valid_mask=torch.ones(1, 5, dtype=torch.bool))
        stacked = torch.cat([features, torch.randn(1, 12, 5)], dim=0)
        batched = model(stacked, valid_mask=torch.ones(2, 5, dtype=torch.bool))
    assert not torch.allclose(alone[0], batched[0], atol=1e-5)


# ---------------------------------------------------------------------------
# From-B auxiliary head
# ---------------------------------------------------------------------------

def test_aux_head_adds_a_masked_loss_term():
    model = _flat_model(aux_from_b_weight=0.5)
    batch = dict(features=torch.randn(2, 12, 6),
                 pos_mask=torch.zeros(2, 6, dtype=torch.bool),
                 valid_mask=torch.ones(2, 6, dtype=torch.bool),
                 from_b=torch.randint(0, 4, (2, 6)))
    batch['pos_mask'][:, 0] = True
    out = model.compute_loss(**batch)
    assert 'aux_from_b_loss' in out
    expected = out['ranking_loss'] + 0.5 * out['aux_from_b_loss']
    assert torch.allclose(out['total_loss'], expected, atol=1e-6)


def test_aux_head_ignores_padded_candidates():
    model = _flat_model(aux_from_b_weight=1.0)
    model.eval()  # dropout off so the two passes see identical activations
    features = torch.randn(1, 12, 4)
    pos = torch.tensor([[True, False, False, False]])
    valid = torch.tensor([[True, True, False, False]])
    targets_a = torch.tensor([[1, 2, 0, 0]])
    targets_b = torch.tensor([[1, 2, 3, 3]])  # differs only on padding
    loss_a = model.compute_loss(features=features, pos_mask=pos,
                                valid_mask=valid, from_b=targets_a)
    loss_b = model.compute_loss(features=features, pos_mask=pos,
                                valid_mask=valid, from_b=targets_b)
    assert torch.allclose(loss_a['aux_from_b_loss'], loss_b['aux_from_b_loss'],
                          atol=1e-7)


# ---------------------------------------------------------------------------
# Track embed dim + warm start shapes
# ---------------------------------------------------------------------------

def test_hierarchical_track_embed_dim_is_configurable():
    names = [f'{prefix}{c}' for prefix in ('ti_', 'tj_', 'tk_')
             for c in range(32)] + ['rest_a', 'rest_b']
    model = TripletReranker(input_mode='hierarchical', feature_names=names,
                            track_embed_dim=32, projector_dim=32,
                            hidden_dim=32, num_residual_blocks=1)
    assert model.track_projector[0].weight.shape == (32, 32)
    scores = model(torch.randn(2, len(names), 5),
                   valid_mask=torch.ones(2, 5, dtype=torch.bool))
    assert scores.shape == (2, 5)


def test_the_couple_checkpoint_projector_loads_into_32_wide_track_blocks():
    import os
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models',
                                   'couple_reranker_best.pt')
    if not os.path.exists(checkpoint_path):
        pytest.skip('promoted couple checkpoint not present')
    state = torch.load(checkpoint_path, map_location='cpu',
                       weights_only=False)
    weights = state['couple_reranker_state_dict']
    projector = {key.split('couple_projector.', 1)[1]: value
                 for key, value in weights.items()
                 if key.startswith('couple_projector.')}
    names = [f'{prefix}{c}' for prefix in ('ti_', 'tj_', 'tk_')
             for c in range(32)] + ['rest']
    model = TripletReranker(input_mode='hierarchical', feature_names=names,
                            track_embed_dim=32, projector_dim=32,
                            hidden_dim=32, num_residual_blocks=1)
    model.track_projector.load_state_dict(projector)


# ---------------------------------------------------------------------------
# In-model differentiable fit
# ---------------------------------------------------------------------------

def _fit_batch(batch=2, candidates=4):
    generator = torch.Generator().manual_seed(11)
    return dict(
        fit_reference=torch.randn(batch, 3, 3, candidates, generator=generator) * 0.1,
        fit_eta=torch.randn(batch, 3, candidates, generator=generator) * 0.5,
        fit_phi=torch.randn(batch, 3, candidates, generator=generator),
        fit_var_dxy=torch.rand(batch, 3, candidates, generator=generator) * 1e-3,
        fit_var_dsz=torch.rand(batch, 3, candidates, generator=generator) * 1e-3,
        fit_momentum=torch.randn(batch, 3, candidates, generator=generator) * 2,
        fit_mass=torch.rand(batch, candidates, generator=generator) + 0.5,
        fit_quality=torch.randn(batch, 12, 3, candidates, generator=generator),
        primary_vertex=torch.randn(batch, 3, generator=generator) * 0.01,
    )


def _fit_stats():
    return {name: {'log1p': False, 'center': 0.0, 'scale': 1.0}
            for name in FIT_NAMES}


def test_layer_mode_matches_static_columns_through_the_same_trunk():
    # Same trunk weights, same standardization: a model consuming the static
    # fit block as input columns and a model computing the fit in-layer must
    # produce identical scores at zero-init.
    from weaver.nn.model.VertexFit import VertexFitLayer
    torch.manual_seed(4)
    static_model = _flat_model(feature_dim=8 + len(FIT_NAMES))
    layer_model = _flat_model(feature_dim=8, vertex_fit_layer=True,
                              fit_norm_stats=_fit_stats())
    layer_model.load_state_dict(static_model.state_dict(), strict=False)

    fit_batch = _fit_batch(batch=2, candidates=4)
    reference_layer = VertexFitLayer()
    with torch.no_grad():
        fit_channels = reference_layer(
            reference=fit_batch['fit_reference'], eta=fit_batch['fit_eta'],
            phi=fit_batch['fit_phi'], var_dxy=fit_batch['fit_var_dxy'],
            var_dsz=fit_batch['fit_var_dsz'],
            primary_vertex=fit_batch['primary_vertex'],
            momentum=fit_batch['fit_momentum'], mass=fit_batch['fit_mass'],
            quality=fit_batch['fit_quality'])
    base = torch.randn(2, 8, 4)
    static_model.eval()
    layer_model.eval()
    # The layer path standardizes its channels (identity affine here, but the
    # +-10 clamp and NaN policy still apply); the static path must see the
    # same transform for the equivalence to be exact.
    standardized = torch.nan_to_num(fit_channels.clamp(-10.0, 10.0), nan=0.0)
    with torch.no_grad():
        static_scores = static_model(
            torch.cat([base, standardized], dim=1),
            valid_mask=torch.ones(2, 4, dtype=torch.bool))
        layer_scores = layer_model(
            base, valid_mask=torch.ones(2, 4, dtype=torch.bool),
            fit_inputs=fit_batch)
    assert torch.allclose(static_scores, layer_scores, atol=1e-5)


def test_layer_mode_standardizes_fit_channels_with_the_stored_stats():
    stats = _fit_stats()
    stats['fit_chi2'] = {'log1p': True, 'center': 1.0, 'scale': 2.0}
    model = _flat_model(feature_dim=8, vertex_fit_layer=True,
                        fit_norm_stats=stats, fusion=True)
    model.eval()
    fit_batch = _fit_batch(batch=1, candidates=3)
    # With fusion zero-init the trunk contributes nothing; scores equal the
    # anchor regardless of fit channels — this only asserts the path runs and
    # stays finite under a log1p transform.
    with torch.no_grad():
        scores = model(torch.randn(1, 8, 3),
                       valid_mask=torch.ones(1, 3, dtype=torch.bool),
                       filter_logit=torch.zeros(1, 3), fit_inputs=fit_batch)
    assert torch.isfinite(scores).all()
