"""Unit tests for `weaver/weaver/nn/model/CoupleReranker.py`.

The CoupleReranker is the per-couple Stage 3 head designed in
`reports/triplet_reranking/triplet_research_plan_20260408.md`. It consumes a
flat 51-dim feature vector per couple (built by ``utils.couple_features``) and
produces a single scalar score per couple. Architecture: input projection
(Conv1d 51→256), 4 ResidualBlock(256), scoring head (Conv1d 256→128→1).

Tests cover the model contract: shape, padding mask, residual connectivity,
the pairwise ranking loss interface, gradient flow, and a synthetic
loss-decreases-during-overfitting check.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from weaver.nn.model.CoupleReranker import CoupleReranker, ResidualBlock


# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------

class TestCoupleRerankerConstruction:
    def test_default_input_dim_is_51(self):
        model = CoupleReranker()
        assert model.input_dim == 51

    def test_default_hidden_dim_is_256(self):
        model = CoupleReranker()
        assert model.hidden_dim == 256

    def test_default_num_residual_blocks_is_4(self):
        model = CoupleReranker()
        assert len(model.residual_blocks) == 4

    def test_param_count_in_expected_range(self):
        """At default settings, the model should have ~580K params."""
        model = CoupleReranker()
        n_params = sum(p.numel() for p in model.parameters())
        # Allow ±20% slack for BN counting and minor variations
        assert 450_000 < n_params < 800_000, f'Got {n_params} params'

    def test_custom_input_dim_accepted(self):
        model = CoupleReranker(input_dim=64)
        assert model.input_dim == 64

    def test_custom_hidden_dim_accepted(self):
        model = CoupleReranker(hidden_dim=128)
        assert model.hidden_dim == 128

    def test_custom_num_blocks_accepted(self):
        model = CoupleReranker(num_residual_blocks=2)
        assert len(model.residual_blocks) == 2


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class TestResidualBlock:
    def test_output_shape_matches_input(self):
        block = ResidualBlock(hidden_dim=256, dropout=0.1)
        x = torch.randn(2, 256, 10)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_at_zero_weights_is_identity(self):
        """If we zero the conv weights inside the block, the output should
        be ReLU(x) (the residual passes through, the conv branch is zero,
        and the trailing ReLU is applied)."""
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
            # BN running stats: mean 0, var 1, weight 0, bias 0 → BN(x) = 0
        block.eval()
        x = torch.randn(2, 256, 10)
        y = block(x)
        assert torch.allclose(y, torch.relu(x), atol=1e-6)

    def test_gradient_flows_through_skip(self):
        """Even with all conv weights zero, gradient should still propagate
        through the identity skip path."""
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        x = torch.randn(2, 256, 10, requires_grad=True)
        y = block(x).sum()
        y.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape_is_batch_x_n_couples(self):
        model = CoupleReranker()
        # 2 events, 51 features per couple, 100 couples per event
        x = torch.randn(2, 51, 100)
        scores = model(x)
        assert scores.shape == (2, 100)

    def test_handles_single_event_batch(self):
        model = CoupleReranker()
        x = torch.randn(1, 51, 50)
        scores = model(x)
        assert scores.shape == (1, 50)

    def test_handles_variable_n_couples(self):
        """The Conv1d-1 architecture must work for any number of couples."""
        model = CoupleReranker()
        for n_couples in [1, 50, 200, 860, 1225]:
            x = torch.randn(2, 51, n_couples)
            scores = model(x)
            assert scores.shape == (2, n_couples)

    def test_scores_are_finite(self):
        model = CoupleReranker()
        model.eval()  # disable dropout for determinism
        x = torch.randn(4, 51, 100)
        scores = model(x)
        assert torch.isfinite(scores).all()

    def test_gradient_flows_to_all_params(self):
        model = CoupleReranker()
        x = torch.randn(2, 51, 50)
        loss = model(x).sum()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f'{name} has no grad'
            # At least the input projection and the scoring head should
            # always receive non-zero gradient.
            if 'input_projection' in name or 'scorer' in name:
                assert parameter.grad.abs().sum() > 0, f'{name} grad is zero'


# ---------------------------------------------------------------------------
# Compute loss
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_loss_dict_shape(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        # 5 GT couples per event, marked at fixed positions
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, [0, 1, 2, 3, 4]] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert 'total_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        assert loss_dict['total_loss'].dim() == 0  # scalar
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_handles_padded_couples(self):
        """Padded positions (mask = 0) must be ignored in the loss."""
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, 0] = 1.0  # 1 GT couple per event
        # Mask: only first 50 positions are valid; rest are padded
        couple_mask = torch.zeros(2, 100)
        couple_mask[:, :50] = 1.0
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_skips_events_with_no_gt(self):
        """Events with 0 GT couples should not contribute to the loss."""
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        # First event has 1 GT couple; second event has 0
        couple_labels = torch.zeros(2, 100)
        couple_labels[0, 0] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        # Loss should be finite and based only on event 0
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_zero_when_no_events_have_gt(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)  # no GT anywhere
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        # No positives → no loss to compute → returns zero
        assert loss_dict['total_loss'].item() == 0.0


# ---------------------------------------------------------------------------
# Softmax-CE loss branch (T1.1 of the couple-reranker improvement sweep)
# ---------------------------------------------------------------------------

class TestSoftmaxCELoss:
    def test_invalid_loss_name_rejected(self):
        with pytest.raises(ValueError):
            CoupleReranker(couple_loss='bce')

    def test_softmax_ce_loss_dict_shape(self):
        model = CoupleReranker(couple_loss='softmax-ce')
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, [0, 1, 2]] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(
            couple_features, couple_labels, couple_mask,
        )
        assert 'total_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        assert '_scores' in loss_dict
        assert loss_dict['total_loss'].dim() == 0
        assert torch.isfinite(loss_dict['total_loss'])

    def test_softmax_ce_handles_padding(self):
        model = CoupleReranker(couple_loss='softmax-ce')
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, 0] = 1.0
        couple_mask = torch.zeros(2, 100)
        couple_mask[:, :50] = 1.0
        loss_dict = model.compute_loss(
            couple_features, couple_labels, couple_mask,
        )
        assert torch.isfinite(loss_dict['total_loss'])

    def test_softmax_ce_zero_when_no_gt_anywhere(self):
        model = CoupleReranker(couple_loss='softmax-ce')
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(
            couple_features, couple_labels, couple_mask,
        )
        assert loss_dict['total_loss'].item() == 0.0

    def test_softmax_ce_loss_is_negative_log_prob_of_positive(self):
        """When scores are fixed and exactly 1 positive + N negatives, the
        loss must equal -log(exp(s_pos) / sum_j exp(s_j))."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='softmax-ce')
        # Override the model's forward to return a deterministic score
        # tensor so we can compute the analytic answer.
        s_pos = 3.0
        s_neg = 1.0
        n_neg = 10
        scores = torch.tensor([[s_pos] + [s_neg] * n_neg])  # (1, 1+N)
        couple_labels = torch.tensor([[1.0] + [0.0] * n_neg])
        couple_mask = torch.ones_like(couple_labels)
        model.ranking_num_samples = n_neg
        loss = model._softmax_ce_loss(scores, couple_labels, couple_mask)
        import math as _math
        expected = -s_pos + _math.log(_math.exp(s_pos) + n_neg * _math.exp(s_neg))
        assert abs(loss.item() - expected) < 1e-5

    def test_softmax_ce_overfits_tiny_task(self):
        """softmax-CE must also be able to overfit the same tiny task."""
        torch.manual_seed(0)
        model = CoupleReranker(
            hidden_dim=64, num_residual_blocks=2, couple_loss='softmax-ce',
        )
        n_events = 4
        n_couples = 30
        couple_features = torch.randn(n_events, 51, n_couples) * 0.1
        couple_features[:, :, :3] *= 5.0
        couple_labels = torch.zeros(n_events, n_couples)
        couple_labels[:, :3] = 1.0
        couple_mask = torch.ones(n_events, n_couples)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial_loss = None
        for step in range(200):
            optimizer.zero_grad()
            loss = model.compute_loss(
                couple_features, couple_labels, couple_mask,
            )['total_loss']
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f'Loss did not decrease: initial={initial_loss:.4f}, '
            f'final={final_loss:.4f}'
        )

    def test_label_smoothing_reduces_to_plain_at_zero(self):
        """Two softmax-CE forwards with ε=0 must return the same loss when
        the sampled negatives are the same (seed torch before each call)."""
        model = CoupleReranker(
            couple_loss='softmax-ce', label_smoothing=0.0,
        )
        couple_features = torch.randn(2, 51, 40)
        couple_labels = torch.zeros(2, 40)
        couple_labels[:, [0, 1]] = 1.0
        couple_mask = torch.ones(2, 40)

        torch.manual_seed(0)
        l_first = model.compute_loss(
            couple_features, couple_labels, couple_mask,
        )['total_loss']
        torch.manual_seed(0)
        l_second = model.compute_loss(
            couple_features, couple_labels, couple_mask,
        )['total_loss']
        assert abs(l_first.item() - l_second.item()) < 1e-6

    # ----------------------------------------------------------------
    # Hard-negative mining (T2.1)
    # ----------------------------------------------------------------

    def test_hardneg_fraction_invalid_rejected(self):
        with pytest.raises(ValueError):
            CoupleReranker(hardneg_fraction=1.5)
        with pytest.raises(ValueError):
            CoupleReranker(hardneg_fraction=-0.1)

    def test_hardneg_zero_fraction_matches_random(self):
        """With `hardneg_fraction=0` the sampler must be identical to the
        legacy random negative sampling (reproducibility guarantee for
        baselines)."""
        torch.manual_seed(0)
        model_a = CoupleReranker(
            couple_loss='softmax-ce', hardneg_fraction=0.0,
        )
        model_b = CoupleReranker(
            couple_loss='softmax-ce', hardneg_fraction=0.0,
        )
        model_b.load_state_dict(model_a.state_dict())

        couple_features = torch.randn(2, 51, 40)
        couple_labels = torch.zeros(2, 40)
        couple_labels[:, [0, 1]] = 1.0
        couple_mask = torch.ones(2, 40)

        torch.manual_seed(1)
        l_a = model_a.compute_loss(
            couple_features, couple_labels, couple_mask,
        )['total_loss']
        torch.manual_seed(1)
        l_b = model_b.compute_loss(
            couple_features, couple_labels, couple_mask,
        )['total_loss']
        assert abs(l_a.item() - l_b.item()) < 1e-6

    def test_hardneg_sampling_returns_expected_count(self):
        """`_sample_negative_indices` must return exactly `ranking_num_samples`
        negative indices regardless of hardneg_fraction."""
        model = CoupleReranker(
            couple_loss='softmax-ce',
            ranking_num_samples=20,
            hardneg_fraction=0.5,
        )
        # Fake per-event scores: 5 positives at high score, 50 negatives
        # at random scores.
        torch.manual_seed(0)
        event_scores = torch.randn(55)
        event_scores[:5] += 5.0  # make positives clearly higher
        positive_indices = torch.arange(5)
        negative_indices = torch.arange(5, 55)
        sampled = model._sample_negative_indices(
            event_scores, positive_indices, negative_indices,
        )
        assert sampled.shape[0] == model.ranking_num_samples
        # No positive index should sneak in
        assert not any(int(idx) < 5 for idx in sampled)

    def test_hardneg_margin_filters_out_false_negatives(self):
        """High-score negatives close to a positive (within margin) must
        be excluded. Verify by constructing a score tensor where all
        top negatives are within margin — result should contain only
        random fills (no indices from the top-scoring negatives)."""
        torch.manual_seed(0)
        model = CoupleReranker(
            couple_loss='softmax-ce',
            ranking_num_samples=10,
            hardneg_fraction=1.0,  # all slots requested as hard
            hardneg_margin=2.0,  # margin is large → no neg clears
        )
        event_scores = torch.zeros(20)
        positive_indices = torch.tensor([0])
        event_scores[0] = 3.0  # positive
        # Set all negatives to scores just below positive (within margin)
        event_scores[1:] = 2.0  # 2.0 not < 3.0 - 2.0 = 1.0, so all filtered out
        negative_indices = torch.arange(1, 20)
        sampled = model._sample_negative_indices(
            event_scores, positive_indices, negative_indices,
        )
        assert sampled.shape[0] == 10  # filled with random

    def test_label_smoothing_changes_loss_value(self):
        """ε=0.1 must produce a different loss value than ε=0.0 (same
        model, same scores). Seed torch around each call so the sampled
        negatives are identical between the two forward passes."""
        model_a = CoupleReranker(
            couple_loss='softmax-ce', label_smoothing=0.0,
        )
        model_b = CoupleReranker(
            couple_loss='softmax-ce', label_smoothing=0.1,
        )
        model_b.load_state_dict(model_a.state_dict())

        couple_features = torch.randn(2, 51, 40)
        couple_labels = torch.zeros(2, 40)
        couple_labels[:, [0, 1]] = 1.0
        couple_mask = torch.ones(2, 40)

        torch.manual_seed(0)
        l_a = model_a.compute_loss(
            couple_features, couple_labels, couple_mask,
        )['total_loss']
        torch.manual_seed(0)
        l_b = model_b.compute_loss(
            couple_features, couple_labels, couple_mask,
        )['total_loss']
        assert l_a.item() != l_b.item()


# ---------------------------------------------------------------------------
# Synthetic overfit check
# ---------------------------------------------------------------------------

class TestOverfitsTinyTask:
    def test_loss_decreases_on_overfit(self):
        """The model should be able to overfit a tiny synthetic task in
        a few hundred steps. This is the most basic learnability sanity
        check — if it can't overfit 4 events, the architecture is broken.
        """
        torch.manual_seed(0)
        model = CoupleReranker(hidden_dim=64, num_residual_blocks=2)  # smaller for speed
        # 4 events, 51 features per couple, 30 couples per event
        # GT couples are at positions [0, 1, 2] in each event with high feature norm
        n_events = 4
        n_couples = 30
        couple_features = torch.randn(n_events, 51, n_couples) * 0.1
        couple_features[:, :, :3] *= 5.0  # GT couples are easily separable
        couple_labels = torch.zeros(n_events, n_couples)
        couple_labels[:, :3] = 1.0
        couple_mask = torch.ones(n_events, n_couples)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial_loss = None
        for step in range(200):
            optimizer.zero_grad()
            loss = model.compute_loss(couple_features, couple_labels, couple_mask)['total_loss']
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f'Loss did not decrease enough: initial={initial_loss:.4f}, '
            f'final={final_loss:.4f}'
        )


# ---------------------------------------------------------------------------
# Couple-embedding modes (Batch 2): pluggable Block-1 aggregation
# ---------------------------------------------------------------------------

class TestCoupleEmbedMode:
    """Pluggable couple-embedding block: concat / infersent / symmetric /
    bilinear_lrb / projected_infersent. The model always consumes the raw
    51-dim couple feature tensor emitted by the feature builder; mode-specific
    Block-1 reconstruction happens inside the model's forward (so projector
    params can receive gradients).

    Formulas (d = 16 per-track raw dim, rest = 19 for physics + geom +
    cascade-score blocks, r = 8 bilinear rank):

        concat              : [t_i, t_j]                           → 32 + 19 = 51
        infersent           : [t_i, t_j, |t_i-t_j|, t_i ⊙ t_j]     → 64 + 19 = 83
        symmetric           : [max(t_i,t_j), mean, |t_i-t_j|]      → 48 + 19 = 67
        bilinear_lrb (r=8)  : [t_i, t_j, (Ut_i) ⊙ (Vt_j)]          → 40 + 19 = 59
        projected_infersent : [φ(t_i), φ(t_j), |Δφ|, φ(t_i)⊙φ(t_j)] → 4p + 19
          where φ(t) = LayerNorm(ReLU(Linear(d, p)(t))).
    """

    def test_default_mode_is_concat(self):
        model = CoupleReranker()
        assert model.couple_embed_mode == 'concat'

    @pytest.mark.parametrize(
        'mode,projector_dim,expected_input_dim',
        [
            ('concat', 0, 51),
            ('infersent', 0, 83),
            ('symmetric', 0, 67),
            ('bilinear_lrb', 0, 59),
            ('projected_infersent', 16, 83),
            ('projected_infersent', 32, 147),
        ],
    )
    def test_input_dim_computed_from_mode(
        self, mode, projector_dim, expected_input_dim,
    ):
        model = CoupleReranker(
            couple_embed_mode=mode,
            couple_projector_dim=projector_dim,
        )
        assert model.input_dim == expected_input_dim

    @pytest.mark.parametrize(
        'mode,projector_dim',
        [
            ('concat', 0),
            ('infersent', 0),
            ('symmetric', 0),
            ('bilinear_lrb', 0),
            ('projected_infersent', 16),
            ('projected_infersent', 32),
        ],
    )
    def test_forward_shape_per_mode(self, mode, projector_dim):
        """All modes consume the raw 51-dim couple tensor and emit (B, C)."""
        model = CoupleReranker(
            couple_embed_mode=mode,
            couple_projector_dim=projector_dim,
        )
        model.eval()
        x = torch.randn(2, 51, 5)
        scores = model(x)
        assert scores.shape == (2, 5)

    def test_symmetric_mode_is_permutation_invariant(self):
        """Swap the two track blocks — scores must match exactly."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_embed_mode='symmetric')
        model.eval()
        x = torch.randn(2, 51, 4)
        x_swapped = x.clone()
        x_swapped[:, :16, :] = x[:, 16:32, :]
        x_swapped[:, 16:32, :] = x[:, :16, :]
        s1 = model(x)
        s2 = model(x_swapped)
        assert torch.allclose(s1, s2, atol=1e-5)

    def test_concat_mode_is_not_permutation_invariant(self):
        """Regression guard: concat asymmetry must survive the refactor."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_embed_mode='concat')
        model.eval()
        x = torch.randn(2, 51, 4)
        x_swapped = x.clone()
        x_swapped[:, :16, :] = x[:, 16:32, :]
        x_swapped[:, 16:32, :] = x[:, :16, :]
        s1 = model(x)
        s2 = model(x_swapped)
        assert not torch.allclose(s1, s2, atol=1e-3)

    def test_projected_infersent_has_trainable_projector(self):
        model = CoupleReranker(
            couple_embed_mode='projected_infersent',
            couple_projector_dim=16,
        )
        assert hasattr(model, 'couple_projector')
        proj_params = list(model.couple_projector.parameters())
        assert len(proj_params) > 0
        assert all(p.requires_grad for p in proj_params)

    def test_projected_infersent_gradient_flows_through_projector(self):
        torch.manual_seed(0)
        model = CoupleReranker(
            couple_embed_mode='projected_infersent',
            couple_projector_dim=16,
        )
        x = torch.randn(2, 51, 4)
        loss = model(x).sum()
        loss.backward()
        grads = [
            p.grad for p in model.couple_projector.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_bilinear_lrb_has_trainable_interaction(self):
        model = CoupleReranker(couple_embed_mode='bilinear_lrb')
        assert hasattr(model, 'bilinear_u')
        assert hasattr(model, 'bilinear_v')
        params = (
            list(model.bilinear_u.parameters())
            + list(model.bilinear_v.parameters())
        )
        assert all(p.requires_grad for p in params)

    def test_bilinear_lrb_gradient_flows_through_interaction(self):
        torch.manual_seed(0)
        model = CoupleReranker(couple_embed_mode='bilinear_lrb')
        x = torch.randn(2, 51, 4)
        loss = model(x).sum()
        loss.backward()
        u_grad = model.bilinear_u.weight.grad
        v_grad = model.bilinear_v.weight.grad
        assert u_grad is not None and u_grad.abs().sum().item() > 0
        assert v_grad is not None and v_grad.abs().sum().item() > 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match='couple_embed_mode'):
            CoupleReranker(couple_embed_mode='gibberish')

    def test_projector_dim_required_for_projected_infersent(self):
        with pytest.raises(ValueError, match='couple_projector_dim'):
            CoupleReranker(couple_embed_mode='projected_infersent',
                           couple_projector_dim=0)

    def test_param_count_grows_with_projector_width(self):
        narrow = sum(
            p.numel() for p in CoupleReranker(
                couple_embed_mode='projected_infersent',
                couple_projector_dim=16,
            ).parameters()
        )
        wide = sum(
            p.numel() for p in CoupleReranker(
                couple_embed_mode='projected_infersent',
                couple_projector_dim=32,
            ).parameters()
        )
        assert wide > narrow

    def test_loss_works_across_modes(self):
        """Every mode must run compute_loss end-to-end without error."""
        for mode, pdim in [
            ('concat', 0),
            ('infersent', 0),
            ('symmetric', 0),
            ('bilinear_lrb', 0),
            ('projected_infersent', 16),
        ]:
            model = CoupleReranker(
                couple_embed_mode=mode,
                couple_projector_dim=pdim,
                couple_loss='softmax-ce',
                label_smoothing=0.10,
            )
            couple_features = torch.randn(2, 51, 20)
            couple_labels = torch.zeros(2, 20)
            couple_labels[:, 0] = 1.0
            couple_mask = torch.ones(2, 20)
            out = model.compute_loss(
                couple_features, couple_labels, couple_mask,
            )
            assert torch.isfinite(out['total_loss'])


# ---------------------------------------------------------------------------
# Tokenization modes (Batch 3 — H15): feature-as-token self-attention
# within each couple. No cross-couple attention.
# ---------------------------------------------------------------------------

class TestTokenizationModes:
    """FT-Transformer and per-track token modes.

    Both modes consume the raw 51-dim couple feature tensor and tokenize
    each scalar into a token of width ``d_token``. A ``[CLS]`` token is
    prepended, the sequence runs through an internal Transformer stack,
    and the ``[CLS]`` output is linearly projected to ``hidden_dim`` for
    the downstream residual stack. This replaces the Block-1 rebuild;
    ``input_dim`` in these modes equals ``hidden_dim``.
    """

    @pytest.mark.parametrize(
        'mode', ['ft_transformer', 'per_track_tokens'],
    )
    def test_forward_shape(self, mode):
        model = CoupleReranker(
            couple_embed_mode=mode,
            tokenize_d=16,
            tokenize_blocks=2,
            tokenize_heads=4,
        )
        model.eval()
        x = torch.randn(2, 51, 5)
        out = model(x)
        assert out.shape == (2, 5)

    def test_ft_transformer_has_feature_embedding(self):
        model = CoupleReranker(
            couple_embed_mode='ft_transformer',
            tokenize_d=16,
        )
        assert hasattr(model, 'token_encoder')
        feature_embed = model.token_encoder.feature_embed
        assert feature_embed.shape == (51, 16)

    def test_per_track_tokens_has_track_embedding(self):
        model = CoupleReranker(
            couple_embed_mode='per_track_tokens',
            tokenize_d=16,
        )
        assert hasattr(model.token_encoder, 'track_embed')
        track_embed = model.token_encoder.track_embed
        assert track_embed.shape == (3, 16)

    def test_ft_transformer_no_track_embedding(self):
        model = CoupleReranker(
            couple_embed_mode='ft_transformer',
            tokenize_d=16,
        )
        assert not hasattr(model.token_encoder, 'track_embed')

    @pytest.mark.parametrize(
        'mode', ['ft_transformer', 'per_track_tokens'],
    )
    def test_gradient_flows_through_feature_embed(self, mode):
        torch.manual_seed(0)
        model = CoupleReranker(
            couple_embed_mode=mode,
            tokenize_d=16,
            tokenize_blocks=2,
            tokenize_heads=4,
        )
        x = torch.randn(2, 51, 4)
        loss = model(x).sum()
        loss.backward()
        grad = model.token_encoder.feature_embed.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0

    def test_per_track_tokens_preserves_ij_asymmetry(self):
        """Swapping the per-track halves must change the score (the
        track_embed distinguishes i from j)."""
        torch.manual_seed(0)
        model = CoupleReranker(
            couple_embed_mode='per_track_tokens',
            tokenize_d=16,
            tokenize_blocks=2,
            tokenize_heads=4,
        )
        model.eval()
        x = torch.randn(2, 51, 4)
        x_swapped = x.clone()
        x_swapped[:, :16, :] = x[:, 16:32, :]
        x_swapped[:, 16:32, :] = x[:, :16, :]
        s1 = model(x)
        s2 = model(x_swapped)
        assert not torch.allclose(s1, s2, atol=1e-3)

    def test_invalid_tokenize_heads_raises(self):
        """``tokenize_heads`` must divide ``tokenize_d``."""
        with pytest.raises(
            (ValueError, AssertionError), match='divisible|heads',
        ):
            CoupleReranker(
                couple_embed_mode='ft_transformer',
                tokenize_d=17,  # 17 is prime, not divisible by 4 heads
                tokenize_heads=4,
            )

    @pytest.mark.parametrize(
        'mode', ['ft_transformer', 'per_track_tokens'],
    )
    def test_loss_runs_end_to_end(self, mode):
        model = CoupleReranker(
            couple_embed_mode=mode,
            tokenize_d=16,
            tokenize_blocks=2,
            tokenize_heads=4,
            couple_loss='softmax-ce',
            label_smoothing=0.10,
        )
        features = torch.randn(2, 51, 20)
        labels = torch.zeros(2, 20)
        labels[:, 0] = 1.0
        mask = torch.ones(2, 20)
        out = model.compute_loss(features, labels, mask)
        assert torch.isfinite(out['total_loss'])

    def test_tokenization_replaces_block1_rebuild(self):
        """In token modes, the main ``input_projection`` first-layer
        Conv1d has input_channels = hidden_dim (CLS readout), not the
        raw 51."""
        model = CoupleReranker(
            couple_embed_mode='ft_transformer',
            tokenize_d=16,
            hidden_dim=256,
        )
        first_conv = model.input_projection[0]
        assert first_conv.in_channels == 256

    def test_concat_baseline_unchanged(self):
        """Concat mode must keep the legacy 51-in-channel first-conv so
        v3 and earlier checkpoints still load."""
        model = CoupleReranker(couple_embed_mode='concat')
        assert model.input_projection[0].in_channels == 51

    @pytest.mark.parametrize(
        'mode', ['ft_transformer', 'per_track_tokens'],
    )
    def test_tokenizer_handles_large_batch_times_c(self, mode):
        """Regression: B*C > 65 535 must not hit PyTorch's flash-SDP cap.

        B=16, C=4200 → 67 200 > 65 000 chunk boundary. Verify output
        shape and no RuntimeError.
        """
        model = CoupleReranker(
            couple_embed_mode=mode,
            tokenize_d=16,
            tokenize_blocks=1,
            tokenize_heads=4,
        )
        model.eval()
        # B*C > SDPA cap forces the chunked path.
        x = torch.randn(16, 51, 4200)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (16, 4200)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# BN-calibration safety for event-context mode (Batch-3 F1 regression)
# ---------------------------------------------------------------------------

class TestBNCalibrationSafety:
    """F1 regression: `event_context='deepset_film'` adds a required
    `k2_features` forward argument. Any helper that calls the reranker
    directly (e.g. BN calibration) must thread k2_features through."""

    def test_event_context_forward_requires_k2_features(self):
        """Sanity: without k2_features, event-context mode must raise."""
        model = CoupleReranker(
            event_context='deepset_film', context_dim=32,
        )
        model.eval()
        x = torch.randn(2, 51, 5)
        with pytest.raises(RuntimeError, match='k2_features'):
            model(x)

    def test_event_context_forward_with_k2_features(self):
        """With k2_features supplied, forward must produce finite output."""
        model = CoupleReranker(
            event_context='deepset_film', context_dim=32,
        )
        model.eval()
        x = torch.randn(2, 51, 5)
        k2 = torch.randn(2, 16, 60)
        with torch.no_grad():
            out = model(x, k2_features=k2)
        assert out.shape == (2, 5)
        assert torch.isfinite(out).all()

    def test_event_context_compute_loss_threads_k2_features(self):
        """compute_loss must forward k2_features through to _encode."""
        model = CoupleReranker(
            event_context='deepset_film', context_dim=32,
            couple_loss='softmax-ce', label_smoothing=0.10,
        )
        features = torch.randn(2, 51, 20)
        labels = torch.zeros(2, 20)
        labels[:, 0] = 1.0
        mask = torch.ones(2, 20)
        k2 = torch.randn(2, 16, 60)
        out = model.compute_loss(
            features, labels, mask, k2_features=k2,
        )
        assert torch.isfinite(out['total_loss'])

    def test_non_event_context_ignores_k2_features(self):
        """Back-compat: concat / projector modes must ignore k2_features
        (passing None or a tensor — both must work)."""
        model = CoupleReranker(
            couple_embed_mode='projected_infersent',
            couple_projector_dim=32,
            event_context='none',
        )
        model.eval()
        x = torch.randn(2, 51, 5)
        # None case
        with torch.no_grad():
            out_none = model(x)
        assert out_none.shape == (2, 5)
        # With a tensor — still fine (ignored).
        k2 = torch.randn(2, 16, 60)
        with torch.no_grad():
            out_tensor = model(x, k2_features=k2)
        assert out_tensor.shape == (2, 5)
        # Results should be identical since k2_features is ignored here.
        assert torch.allclose(out_none, out_tensor, atol=1e-6)


# ---------------------------------------------------------------------------
# LambdaLoss@K=100 NDCG-Loss2++ (Batch 3 — H1)
#
#   δ_ij = |G_i − G_j| · |1/D_i − 1/D_j|
#     G = 1 for positives, 0 for negatives
#     D = log₂(rank + 1)    (rank is 1-indexed)
#   Loss = Σ_{i,j: G_i ≠ G_j, rank_i ≤ K or rank_j ≤ K}
#            δ_ij · log(1 + exp(−σ (s_i − s_j)))
# ---------------------------------------------------------------------------

class TestLambdaNDCGLoss:
    def test_constructor_accepts_loss_name(self):
        model = CoupleReranker(couple_loss='lambda_ndcg2pp')
        assert model.couple_loss == 'lambda_ndcg2pp'

    def test_invalid_loss_still_rejected(self):
        with pytest.raises(ValueError, match='couple_loss'):
            CoupleReranker(couple_loss='pairwise_nonsense')

    def test_loss_dict_shape(self):
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='lambda_ndcg2pp')
        features = torch.randn(2, 51, 30)
        labels = torch.zeros(2, 30)
        labels[:, 0] = 1.0
        labels[:, 5] = 1.0
        mask = torch.ones(2, 30)
        out = model.compute_loss(features, labels, mask)
        assert 'total_loss' in out
        assert 'ranking_loss' in out
        assert out['total_loss'].dim() == 0
        assert torch.isfinite(out['total_loss'])

    def test_gradient_flow(self):
        """Gradient must flow from the lambda loss back to model weights."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='lambda_ndcg2pp')
        features = torch.randn(2, 51, 30)
        labels = torch.zeros(2, 30)
        labels[:, 0] = 1.0
        mask = torch.ones(2, 30)
        out = model.compute_loss(features, labels, mask)
        out['total_loss'].backward()
        grad_norms = [
            p.grad.abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert any(g > 0 for g in grad_norms)

    def test_zero_when_all_positives_rank_above_negatives(self):
        """If every positive score dominates every negative by a large
        margin, the loss should be negligibly small."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='lambda_ndcg2pp')
        model.eval()  # turn off BN training
        # fabricate scores directly via mocking forward
        scores = torch.zeros(1, 10)
        scores[0, :3] = 10.0  # 3 positives have very high score
        scores[0, 3:] = -10.0  # 7 negatives have very low score
        labels = torch.zeros(1, 10)
        labels[:, :3] = 1.0
        mask = torch.ones(1, 10)
        loss = model._lambda_ndcg_loss(scores, labels, mask)
        assert loss.item() < 1e-3

    def test_nonzero_when_positives_below_negatives(self):
        """If positives are below negatives, the loss must be strictly
        positive (there is at least one swap to punish)."""
        model = CoupleReranker(couple_loss='lambda_ndcg2pp')
        scores = torch.zeros(1, 10)
        scores[0, :3] = -5.0  # positives low
        scores[0, 3:] = 5.0   # negatives high
        labels = torch.zeros(1, 10)
        labels[:, :3] = 1.0
        mask = torch.ones(1, 10)
        loss = model._lambda_ndcg_loss(scores, labels, mask)
        assert loss.item() > 0.1

    def test_K_truncation_drops_faraway_swaps(self):
        """Setting ndcg_K = 3 should make swaps involving rank > 3 pairs
        contribute zero weight — total loss is ≤ untruncated version."""
        torch.manual_seed(0)
        scores = torch.randn(1, 50)
        labels = torch.zeros(1, 50)
        labels[:, 0] = 1.0
        labels[:, 45] = 1.0  # positive way out at rank 45
        mask = torch.ones(1, 50)

        m_full = CoupleReranker(
            couple_loss='lambda_ndcg2pp', ndcg_K=50,
        )
        m_trunc = CoupleReranker(
            couple_loss='lambda_ndcg2pp', ndcg_K=3,
        )
        loss_full = m_full._lambda_ndcg_loss(scores, labels, mask).item()
        loss_trunc = m_trunc._lambda_ndcg_loss(scores, labels, mask).item()
        # Truncation CAN drop pairs from the sum; ratio should be <= 1.
        # Use non-strict inequality because pairs involving the rank-0
        # positive may still dominate.
        assert loss_trunc <= loss_full + 1e-6

    def test_ignores_padded_positions(self):
        """Masked-out positions contribute zero to the loss."""
        model = CoupleReranker(couple_loss='lambda_ndcg2pp')
        scores = torch.randn(1, 20)
        labels = torch.zeros(1, 20)
        labels[:, 0] = 1.0
        labels[:, 15] = 1.0  # this is masked out, should not contribute
        mask = torch.ones(1, 20)
        mask[:, 10:] = 0.0

        scores_masked_changed = scores.clone()
        scores_masked_changed[:, 10:] = scores[:, 10:] + 100.0

        loss_a = model._lambda_ndcg_loss(scores, labels, mask).item()
        loss_b = model._lambda_ndcg_loss(
            scores_masked_changed, labels, mask,
        ).item()
        assert abs(loss_a - loss_b) < 1e-5

    def test_ndcg_K_zero_disables_truncation(self):
        """ndcg_K=0 means 'no truncation' — all pairs contribute. Loss
        should be >= the loss with a small K where far pairs are dropped."""
        torch.manual_seed(0)
        scores = torch.zeros(1, 50)
        scores[0, 0] = 1.0  # GT at rank 1
        scores[0, 45] = 0.5  # GT at rank ~5 (high)
        scores[0, 10:40] = -1.0  # negatives spread out
        labels = torch.zeros(1, 50)
        labels[:, 0] = 1.0
        labels[:, 45] = 1.0
        mask = torch.ones(1, 50)

        m_noK = CoupleReranker(couple_loss='lambda_ndcg2pp', ndcg_K=0)
        m_K3 = CoupleReranker(couple_loss='lambda_ndcg2pp', ndcg_K=3)
        loss_noK = m_noK._lambda_ndcg_loss(scores, labels, mask).item()
        loss_K3 = m_K3._lambda_ndcg_loss(scores, labels, mask).item()
        # No-truncation loss sums over strictly more pairs ⇒ ≥ truncated.
        assert loss_noK >= loss_K3 - 1e-6

    def test_rank_swap_increases_loss(self):
        """Monotonic rank sensitivity: pushing GT from rank 5 to rank 30
        must strictly increase the loss (position-aware signal)."""
        model = CoupleReranker(couple_loss='lambda_ndcg2pp', ndcg_K=0)
        n = 50
        # Scenario A: GT at rank ~5 among 50 tracks.
        scores_a = torch.arange(n, 0, -1).float().unsqueeze(0)  # descending
        labels = torch.zeros(1, n)
        labels[:, 4] = 1.0  # single GT at index 4 (rank 5)
        mask = torch.ones(1, n)
        # Scenario B: same config but GT moved to index 29 (rank 30).
        scores_b = scores_a.clone()
        labels_b = torch.zeros(1, n)
        labels_b[:, 29] = 1.0
        loss_a = model._lambda_ndcg_loss(scores_a, labels, mask).item()
        loss_b = model._lambda_ndcg_loss(scores_b, labels_b, mask).item()
        assert loss_b > loss_a, (
            f'expected worse rank ⇒ larger loss, got a={loss_a:.4f}, '
            f'b={loss_b:.4f}'
        )


# ---------------------------------------------------------------------------
# ApproxNDCG loss (Qin et al. 2010, Batch 6 — Batch 6 position-aware fix)
# ---------------------------------------------------------------------------

class TestApproxNDCGLoss:
    def test_constructor_accepts_approx_ndcg_loss_name(self):
        model = CoupleReranker(couple_loss='approx_ndcg')
        assert model.couple_loss == 'approx_ndcg'

    def test_loss_dict_shape(self):
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='approx_ndcg')
        features = torch.randn(2, 51, 30)
        labels = torch.zeros(2, 30)
        labels[:, 0] = 1.0
        labels[:, 5] = 1.0
        mask = torch.ones(2, 30)
        out = model.compute_loss(features, labels, mask)
        assert 'total_loss' in out
        assert 'ranking_loss' in out
        assert out['total_loss'].dim() == 0
        assert torch.isfinite(out['total_loss'])

    def test_gradient_flow(self):
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='approx_ndcg')
        features = torch.randn(2, 51, 30)
        labels = torch.zeros(2, 30)
        labels[:, 0] = 1.0
        mask = torch.ones(2, 30)
        out = model.compute_loss(features, labels, mask)
        out['total_loss'].backward()
        grad_sums = [
            p.grad.abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert any(g > 0 for g in grad_sums)

    def test_perfect_ranking_gives_minimum_loss(self):
        """GT couples scored strictly highest ⇒ NDCG ≈ 1 ⇒ loss ≈ −1.

        Strict ordering (10, 9, 8) avoids tie-induced mid-ranks from the
        sigmoid-at-zero branch; with ties at the top, ApproxNDCG assigns
        approx_rank ≈ 2 for each tied position and NDCG drops slightly.
        The test checks rank recovery, not tie-breaking semantics.
        """
        model = CoupleReranker(couple_loss='approx_ndcg', ndcg_alpha=5.0)
        scores = torch.zeros(1, 10)
        scores[0, 0] = 12.0
        scores[0, 1] = 11.0
        scores[0, 2] = 10.0
        scores[0, 3:] = -10.0
        labels = torch.zeros(1, 10)
        labels[:, :3] = 1.0
        mask = torch.ones(1, 10)
        loss = model._approx_ndcg_loss(scores, labels, mask).item()
        # Perfect: NDCG ~1.0 so loss = −NDCG ~= −1.0
        assert loss < -0.99

    def test_worst_ranking_gives_higher_loss_than_perfect(self):
        model = CoupleReranker(couple_loss='approx_ndcg', ndcg_alpha=5.0)
        n = 10
        # Perfect: GT at top 3
        scores_perfect = torch.tensor(
            [[10., 9., 8.] + [-10.] * 7],
        )
        # Worst: GT at bottom 3
        scores_worst = torch.tensor(
            [[10.] * 7 + [-8., -9., -10.]],
        )
        labels_perfect = torch.zeros(1, n)
        labels_perfect[:, :3] = 1.0
        labels_worst = torch.zeros(1, n)
        labels_worst[:, -3:] = 1.0
        mask = torch.ones(1, n)
        loss_p = model._approx_ndcg_loss(
            scores_perfect, labels_perfect, mask,
        ).item()
        loss_w = model._approx_ndcg_loss(
            scores_worst, labels_worst, mask,
        ).item()
        assert loss_w > loss_p

    def test_rank_perturbation_strictly_monotonic(self):
        """Pushing a GT couple from rank 5 down to rank 30 must strictly
        increase ApproxNDCG loss (position-aware signal over full range)."""
        torch.manual_seed(0)
        n = 50
        model = CoupleReranker(couple_loss='approx_ndcg', ndcg_alpha=5.0)
        # descending scores → index = rank − 1
        scores = torch.arange(n, 0, -1).float().unsqueeze(0)
        mask = torch.ones(1, n)
        labels_rank5 = torch.zeros(1, n)
        labels_rank5[:, 4] = 1.0
        labels_rank30 = torch.zeros(1, n)
        labels_rank30[:, 29] = 1.0
        loss_5 = model._approx_ndcg_loss(scores, labels_rank5, mask).item()
        loss_30 = model._approx_ndcg_loss(scores, labels_rank30, mask).item()
        assert loss_30 > loss_5

    def test_ignores_padded_positions(self):
        """Masked positions contribute zero to the loss."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='approx_ndcg')
        scores = torch.randn(1, 20)
        labels = torch.zeros(1, 20)
        labels[:, 0] = 1.0
        labels[:, 1] = 1.0
        mask = torch.ones(1, 20)
        mask[:, 10:] = 0.0

        scores_pad_changed = scores.clone()
        scores_pad_changed[:, 10:] = scores[:, 10:] + 100.0

        loss_a = model._approx_ndcg_loss(scores, labels, mask).item()
        loss_b = model._approx_ndcg_loss(
            scores_pad_changed, labels, mask,
        ).item()
        assert abs(loss_a - loss_b) < 1e-4

    def test_loss_in_expected_range(self):
        """NDCG ∈ [0, 1] ⇒ loss = −NDCG ∈ [−1, 0]."""
        torch.manual_seed(0)
        model = CoupleReranker(couple_loss='approx_ndcg')
        scores = torch.randn(4, 30)
        labels = torch.zeros(4, 30)
        labels[:, 0] = 1.0
        labels[:, 1] = 1.0
        labels[:, 2] = 1.0
        mask = torch.ones(4, 30)
        loss = model._approx_ndcg_loss(scores, labels, mask).item()
        assert -1.0 - 1e-4 <= loss <= 0.0 + 1e-4, f'got {loss}'

    def test_ndcg_alpha_default_and_validation(self):
        model_default = CoupleReranker(couple_loss='approx_ndcg')
        assert model_default.ndcg_alpha == 5.0
        model_tuned = CoupleReranker(
            couple_loss='approx_ndcg', ndcg_alpha=10.0,
        )
        assert model_tuned.ndcg_alpha == 10.0
        with pytest.raises(ValueError, match='ndcg_alpha'):
            CoupleReranker(couple_loss='approx_ndcg', ndcg_alpha=-1.0)


# ---------------------------------------------------------------------------
# Multi-positive soft target for softmax-CE (Batch 3 — H2)
#
#   uniform target: 1/k on each GT couple, 0 elsewhere (k = # GT per event)
#   loss: cross-entropy between this target and the softmax over all valid
#         couples.
# ---------------------------------------------------------------------------

class TestMultiPositive:
    def test_default_multi_positive_is_none(self):
        model = CoupleReranker(couple_loss='softmax-ce')
        assert model.multi_positive == 'none'

    def test_invalid_multi_positive_rejected(self):
        with pytest.raises(ValueError, match='multi_positive'):
            CoupleReranker(
                couple_loss='softmax-ce', multi_positive='nonsense',
            )

    def test_multi_positive_loss_finite(self):
        torch.manual_seed(0)
        model = CoupleReranker(
            couple_loss='softmax-ce', multi_positive='uniform',
        )
        features = torch.randn(2, 51, 20)
        labels = torch.zeros(2, 20)
        labels[:, 0] = 1.0
        labels[:, 3] = 1.0  # 2 GT per event
        mask = torch.ones(2, 20)
        out = model.compute_loss(features, labels, mask)
        assert torch.isfinite(out['total_loss'])

    def test_multi_positive_reduces_to_single_when_k_eq_1(self):
        """One GT per event: multi_positive='uniform' must produce the
        same loss as the default single-positive softmax-CE (ε=0).

        Weight init and negative sampling both draw from the global RNG,
        so re-seed before every forward + loss call for a fair test.
        """
        def make_model(multi_positive):
            torch.manual_seed(0)
            return CoupleReranker(
                couple_loss='softmax-ce',
                multi_positive=multi_positive,
                label_smoothing=0.0,
            )

        features = torch.randn(2, 51, 20)
        labels = torch.zeros(2, 20)
        labels[:, 0] = 1.0  # exactly 1 GT per event
        mask = torch.ones(2, 20)

        model_single = make_model('none')
        model_multi = make_model('uniform')
        model_single.eval()
        model_multi.eval()

        torch.manual_seed(99)
        scores_single = model_single.forward(features)
        torch.manual_seed(99)
        scores_multi = model_multi.forward(features)
        # Ensure forward paths produced identical scores under the same
        # seed (sanity check; multi_positive does not affect forward).
        assert torch.allclose(scores_single, scores_multi, atol=1e-6)

        torch.manual_seed(123)
        loss_single = model_single._softmax_ce_loss(
            scores_single, labels, mask,
        )
        torch.manual_seed(123)
        loss_multi = model_multi._softmax_ce_loss(
            scores_multi, labels, mask,
        )
        assert torch.allclose(loss_single, loss_multi, atol=1e-5)

    def test_multi_positive_differs_from_single_when_k_gt_1(self):
        torch.manual_seed(0)
        model_single = CoupleReranker(
            couple_loss='softmax-ce', multi_positive='none',
            label_smoothing=0.0,
        )
        torch.manual_seed(0)
        model_multi = CoupleReranker(
            couple_loss='softmax-ce', multi_positive='uniform',
            label_smoothing=0.0,
        )
        features = torch.randn(2, 51, 20)
        labels = torch.zeros(2, 20)
        labels[:, 0] = 1.0
        labels[:, 5] = 1.0
        labels[:, 10] = 1.0  # 3 GT per event
        mask = torch.ones(2, 20)
        loss_single = model_single._softmax_ce_loss(
            model_single.forward(features), labels, mask,
        )
        loss_multi = model_multi._softmax_ce_loss(
            model_multi.forward(features), labels, mask,
        )
        assert not torch.allclose(loss_single, loss_multi, atol=1e-3)
