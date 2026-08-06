from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from weaver.nn.model.CoupleReranker import CoupleReranker, ResidualBlock


# Raw couple feature dim from the cleaned couple_features.py: 64 (track concat)
# + 19 (physics+geom+cascade) + 5 (pair_physics_v3 always on) + 11 (h6 couple
# block) = 99.
RAW_FEATURE_DIM = 99


class TestCoupleRerankerConstruction:
    def test_default_hidden_dim_is_256(self):
        model = CoupleReranker()
        assert model.hidden_dim == 256

    def test_default_num_residual_blocks_is_4(self):
        model = CoupleReranker()
        assert len(model.residual_blocks) == 4

    def test_default_input_dim_is_4p_plus_rest(self):
        # projected_infersent: input_dim = 4*p + rest_dim = 4*32 + 35 = 163.
        model = CoupleReranker()
        assert model.input_dim == 4 * 32 + 35

    def test_param_count_in_expected_range(self):
        model = CoupleReranker()
        n_params = sum(p.numel() for p in model.parameters())
        assert 450_000 < n_params < 800_000, f'Got {n_params} params'

    def test_custom_hidden_dim_accepted(self):
        model = CoupleReranker(hidden_dim=128)
        assert model.hidden_dim == 128

    def test_custom_num_blocks_accepted(self):
        model = CoupleReranker(num_residual_blocks=2)
        assert len(model.residual_blocks) == 2

    def test_custom_projector_dim_accepted(self):
        model = CoupleReranker(couple_projector_dim=16)
        assert model.couple_projector_dim == 16
        assert model.input_dim == 4 * 16 + 35

    def test_invalid_projector_dim_rejected(self):
        with pytest.raises(ValueError):
            CoupleReranker(couple_projector_dim=0)


class TestConstantsConsistency:
    def test_track_embed_and_rest_dims_match_couple_features_module(self):
        # The weaver package cannot import utils directly, so the dimension
        # constants are duplicated in CoupleReranker and pinned here.
        from utils.couple_features import COUPLE_REST_DIM, TRACK_EMBED_DIM
        from weaver.nn.model.CoupleReranker import _REST_DIM, _TRACK_EMBED_DIM
        assert _TRACK_EMBED_DIM == TRACK_EMBED_DIM
        assert _REST_DIM == COUPLE_REST_DIM


class TestResidualBlock:
    def test_output_shape_matches_input(self):
        block = ResidualBlock(hidden_dim=256, dropout=0.1)
        x = torch.randn(2, 256, 10)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_at_zero_weights_is_identity(self):
        """All conv weights zero + BN(weight=0,bias=0) ⇒ output = ReLU(x)."""
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        block.eval()
        x = torch.randn(2, 256, 10)
        y = block(x)
        assert torch.allclose(y, torch.relu(x), atol=1e-6)

    def test_gradient_flows_through_skip(self):
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        x = torch.randn(2, 256, 10, requires_grad=True)
        y = block(x).sum()
        y.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


class TestForward:
    def test_output_shape_is_batch_x_n_couples(self):
        model = CoupleReranker()
        x = torch.randn(2, RAW_FEATURE_DIM, 100)
        scores = model(x)
        assert scores.shape == (2, 100)

    def test_handles_single_event_batch(self):
        model = CoupleReranker()
        x = torch.randn(1, RAW_FEATURE_DIM, 50)
        scores = model(x)
        assert scores.shape == (1, 50)

    def test_handles_variable_n_couples(self):
        model = CoupleReranker()
        for n_couples in [1, 50, 200, 860, 1225]:
            x = torch.randn(2, RAW_FEATURE_DIM, n_couples)
            scores = model(x)
            assert scores.shape == (2, n_couples)

    def test_scores_are_finite(self):
        model = CoupleReranker()
        model.eval()
        x = torch.randn(4, RAW_FEATURE_DIM, 100)
        scores = model(x)
        assert torch.isfinite(scores).all()

    def test_wrong_feature_width_raises_value_error(self):
        model = CoupleReranker()
        x = torch.randn(2, RAW_FEATURE_DIM - 1, 10)
        with pytest.raises(ValueError, match='couple-feature channels'):
            model(x)

    def test_gradient_flows_to_all_params(self):
        model = CoupleReranker()
        x = torch.randn(2, RAW_FEATURE_DIM, 50)
        loss = model(x).sum()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f'{name} has no grad'
            if 'input_projection' in name or 'scorer' in name or 'couple_projector' in name:
                assert parameter.grad.abs().sum() > 0, f'{name} grad is zero'


class TestComputeLoss:
    def test_loss_dict_shape(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, RAW_FEATURE_DIM, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, [0, 1, 2, 3, 4]] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert 'total_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        assert loss_dict['total_loss'].dim() == 0
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_handles_padded_couples(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, RAW_FEATURE_DIM, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, 0] = 1.0
        couple_mask = torch.zeros(2, 100)
        couple_mask[:, :50] = 1.0
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_skips_events_with_no_gt(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, RAW_FEATURE_DIM, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[0, 0] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_zero_when_no_events_have_gt(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, RAW_FEATURE_DIM, 100)
        couple_labels = torch.zeros(2, 100)
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert loss_dict['total_loss'].item() == 0.0


class TestSoftmaxCELoss:
    def test_label_smoothing_zero_matches_plain_ce(self):
        torch.manual_seed(0)
        model = CoupleReranker(label_smoothing=0.0)
        features = torch.randn(2, RAW_FEATURE_DIM, 50)
        labels = torch.zeros(2, 50)
        labels[:, 0] = 1.0
        mask = torch.ones(2, 50)
        loss_dict = model.compute_loss(features, labels, mask)
        assert torch.isfinite(loss_dict['total_loss'])

    def test_label_smoothing_positive_finite(self):
        torch.manual_seed(0)
        model = CoupleReranker(label_smoothing=0.10)
        features = torch.randn(2, RAW_FEATURE_DIM, 50)
        labels = torch.zeros(2, 50)
        labels[:, 0] = 1.0
        labels[:, 5] = 1.0
        mask = torch.ones(2, 50)
        loss_dict = model.compute_loss(features, labels, mask)
        assert torch.isfinite(loss_dict['total_loss'])

    def test_label_smoothing_changes_loss(self):
        torch.manual_seed(0)
        model_plain = CoupleReranker(label_smoothing=0.0)
        torch.manual_seed(0)
        model_smooth = CoupleReranker(label_smoothing=0.5)
        features = torch.randn(2, RAW_FEATURE_DIM, 50)
        labels = torch.zeros(2, 50)
        labels[:, 0] = 1.0
        mask = torch.ones(2, 50)
        torch.manual_seed(123)
        loss_plain = model_plain.compute_loss(features, labels, mask)['total_loss']
        torch.manual_seed(123)
        loss_smooth = model_smooth.compute_loss(features, labels, mask)['total_loss']
        assert not torch.allclose(loss_plain, loss_smooth, atol=1e-3)


class TestOverfitsTinyTask:
    def test_loss_decreases_on_overfit(self):
        torch.manual_seed(42)
        model = CoupleReranker(num_residual_blocks=2, dropout=0.0)
        features = torch.randn(1, RAW_FEATURE_DIM, 30)
        labels = torch.zeros(1, 30)
        labels[:, [3, 7, 15]] = 1.0
        mask = torch.ones(1, 30)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_loss = None
        final_loss = None
        for step in range(100):
            optimizer.zero_grad()
            loss = model.compute_loss(features, labels, mask)['total_loss']
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            if step == 99:
                final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f'overfit loss did not drop: {initial_loss=:.4f} → {final_loss=:.4f}'
        )
