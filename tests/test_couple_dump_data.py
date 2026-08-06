"""Tests for the dump-based Stage-3 training path.

Covers the ``CoupleDumpDataset`` reader/collate, the ``CoupleDumpModel``
wrapper, and the dump-mode trainer smoke run. Fixture-dependent tests use the
static real-data dump ``tests/fixtures/stage3_dump_fixture.parquet``
(generated once from the production checkpoints over the local eval shard)
and skip when it is absent.
"""
from __future__ import annotations

import glob
import os
import sys

import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

_DELIVERABLES = os.path.join(os.path.dirname(__file__), '..')
_FIXTURE = os.path.join(
    _DELIVERABLES, 'tests', 'fixtures', 'stage3_dump_fixture.parquet',
)

requires_fixture = pytest.mark.skipif(
    not os.path.exists(_FIXTURE),
    reason='stage3 dump fixture not generated',
)


@pytest.fixture(scope='module')
def fixture_table():
    return pq.read_table(_FIXTURE)


@pytest.fixture(scope='module')
def dump_dataset():
    from utils.couple_dump_data import CoupleDumpDataset

    return CoupleDumpDataset([_FIXTURE])


@pytest.fixture(scope='module')
def dump_batch(dump_dataset):
    from utils.couple_dump_data import CoupleDumpDataset

    return CoupleDumpDataset.collate([dump_dataset[i] for i in range(4)])


@pytest.fixture(scope='module')
def dump_model():
    from weaver.nn.model.CoupleDumpModel import CoupleDumpModel
    from weaver.nn.model.CoupleReranker import CoupleReranker

    torch.manual_seed(0)
    couple_reranker = CoupleReranker(
        hidden_dim=32, num_residual_blocks=1, dropout=0.0,
        couple_projector_dim=8,
    )
    return CoupleDumpModel(
        couple_reranker=couple_reranker, top_k2=12,
        k_values_tracks=(30, 50, 75, 100, 200),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@requires_fixture
class TestCoupleDumpDataset:
    def test_length_matches_fixture_rows(self, dump_dataset, fixture_table):
        assert len(dump_dataset) == fixture_table.num_rows

    def test_item_shapes(self, dump_dataset):
        top_k1 = dump_dataset.top_k1
        item = dump_dataset[0]
        assert item['features'].shape == (32, top_k1)
        assert item['points'].shape == (26, top_k1)
        assert item['lorentz'].shape == (4, top_k1)
        for key in ('stage1_scores', 'stage2_scores', 'labels'):
            assert item[key].shape == (top_k1,)
            assert item[key].dtype == torch.float32
        assert item['original_indices'].shape == (top_k1,)
        assert item['original_indices'].dtype == torch.long
        cone_length = item['cone_eta'].shape[0]
        for key in ('cone_phi', 'cone_dz', 'cone_pt'):
            assert item[key].shape == (cone_length,)

    def test_labels_binary(self, dump_dataset):
        for index in range(len(dump_dataset)):
            labels = dump_dataset[index]['labels']
            assert ((labels == 0.0) | (labels == 1.0)).all()

    def test_collate_pads_cone_and_masks(self, dump_dataset, dump_batch):
        items = [dump_dataset[i] for i in range(4)]
        cone_lengths = [item['cone_eta'].shape[0] for item in items]
        max_length = max(cone_lengths)
        mask = dump_batch['cone_valid_mask']
        assert mask.shape == (4, max_length)
        assert mask.dtype == torch.bool
        for row, expected_length in enumerate(cone_lengths):
            assert int(mask[row].sum()) == expected_length
            assert (dump_batch['cone_pt'][row, expected_length:] == 0.0).all()
        assert dump_batch['cone_points'].shape == (4, 3, max_length)
        assert dump_batch['cone_lorentz'].shape == (4, 4, max_length)
        torch.testing.assert_close(
            dump_batch['cone_points'][:, 0], dump_batch['cone_eta'],
        )
        torch.testing.assert_close(
            dump_batch['cone_lorentz'][:, 0], dump_batch['cone_pt'],
        )
        assert (dump_batch['cone_lorentz'][:, 1:] == 0.0).all()

    def test_fixed_size_tensors_stacked(self, dump_dataset, dump_batch):
        top_k1 = dump_dataset.top_k1
        assert dump_batch['features'].shape == (4, 32, top_k1)
        assert dump_batch['points'].shape == (4, 26, top_k1)
        assert dump_batch['lorentz'].shape == (4, 4, top_k1)
        assert dump_batch['stage2_scores'].shape == (4, top_k1)

    def test_stage2_finite_count_matches_valid_count(self, fixture_table):
        # The dump's stage-2 scores double as the validity mask: exactly
        # min(K1, n_valid_tracks) entries are finite, where n_valid is the
        # full-event valid-track count captured by the cone columns.
        for row in fixture_table.to_pylist():
            top_k1 = len(row['k1_stage2_scores'])
            n_valid = len(row['cone_eta'])
            finite_count = sum(
                1 for score in row['k1_stage2_scores']
                if score == score and abs(score) != float('inf')
            )
            assert finite_count == min(top_k1, n_valid)
            # Labels at padded (non-finite score) positions are zeroed.
            for score, label in zip(row['k1_stage2_scores'], row['k1_labels']):
                if abs(score) == float('inf'):
                    assert label == 0


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@requires_fixture
class TestCoupleDumpModel:
    def test_forward_scores_finite_where_valid(self, dump_model, dump_batch):
        scores, filter_a_mask = dump_model(dump_batch)
        n_couples = dump_model.top_k2 * (dump_model.top_k2 - 1) // 2
        assert scores.shape == (4, n_couples)
        assert filter_a_mask.shape == (4, n_couples)
        assert filter_a_mask.dtype == torch.bool
        assert torch.isfinite(scores[filter_a_mask]).all()

    def test_couple_features_width_99(self, dump_model, dump_batch):
        couple_inputs = dump_model._build_couple_inputs(dump_batch)
        assert couple_inputs['couple_features'].shape[1] == 99
        assert torch.isfinite(couple_inputs['couple_features']).all()

    def test_compute_loss_contract(self, dump_model, dump_batch):
        loss_dict = dump_model.compute_loss(dump_batch)
        for key in ('total_loss', 'ranking_loss', '_scores', '_couple_labels',
                    '_couple_mask', '_n_gt_in_top_k1', '_n_gt_in_top_k_tracks'):
            assert key in loss_dict
        assert torch.isfinite(loss_dict['total_loss'])
        n_couples = dump_model.top_k2 * (dump_model.top_k2 - 1) // 2
        assert loss_dict['_scores'].shape == (4, n_couples)
        assert loss_dict['_n_gt_in_top_k1'].shape == (4,)
        assert loss_dict['_n_gt_in_top_k_tracks'].shape == (4, 5)

    def test_top_k2_selection_follows_stage2_order(self, dump_model, dump_batch):
        # End-to-end gather check via the cascade-score block of the couple
        # feature vector: for couple 0 = (pool position 0, pool position 1),
        # channel 80 holds s2(i) and channel 82 holds s2(j). The K2 pool is
        # selected by descending stored stage-2 score, so these must be the
        # two highest finite stage-2 scores of each event.
        couple_inputs = dump_model._build_couple_inputs(dump_batch)
        features = couple_inputs['couple_features']
        stage2_scores = dump_batch['stage2_scores']
        for event in range(stage2_scores.shape[0]):
            finite_scores = stage2_scores[event][
                torch.isfinite(stage2_scores[event])
            ]
            top_two = finite_scores.sort(descending=True).values[:2]
            torch.testing.assert_close(features[event, 80, 0], top_two[0])
            torch.testing.assert_close(features[event, 82, 0], top_two[1])

    def test_n_gt_in_top_k1_equals_finite_label_sum(self, dump_model, dump_batch):
        loss_dict = dump_model.compute_loss(dump_batch)
        valid = torch.isfinite(dump_batch['stage2_scores'])
        expected = (dump_batch['labels'] * valid.float()).sum(dim=1)
        torch.testing.assert_close(loss_dict['_n_gt_in_top_k1'], expected)

    def test_n_gt_in_top_k_tracks_semantics(self, dump_model, dump_batch):
        loss_dict = dump_model.compute_loss(dump_batch)
        n_gt_per_k = loss_dict['_n_gt_in_top_k_tracks']
        assert (n_gt_per_k[:, 1:] - n_gt_per_k[:, :-1] >= 0).all()
        assert (n_gt_per_k[:, -1] <= 3).all()
        # Recompute the K=50 column directly from the stored stage-2 order.
        stage2_scores = dump_batch['stage2_scores']
        valid = torch.isfinite(stage2_scores)
        safe_scores = stage2_scores.masked_fill(~valid, -1e9)
        order = torch.argsort(safe_scores, dim=1, descending=True)
        gt_mask = (dump_batch['labels'] > 0.5) & valid
        sorted_gt = gt_mask.gather(1, order)
        expected_at_50 = sorted_gt[:, :50].sum(dim=1)
        k_index = dump_model.k_values_tracks.index(50)
        assert torch.equal(
            n_gt_per_k[:, k_index].long(), expected_at_50.long(),
        )


# ---------------------------------------------------------------------------
# Trainer dump mode
# ---------------------------------------------------------------------------

@requires_fixture
def test_trainer_dump_mode_smoke(tmp_path):
    from train_couple_reranker import main as train_main

    experiments = str(tmp_path / 'experiments')
    train_main([
        '--train-dump', _FIXTURE,
        '--val-dump', _FIXTURE,
        '--experiments-dir', experiments,
        '--epochs', '1',
        '--batch-size', '8',
        '--num-workers', '0',
        '--device', 'cpu',
        '--couple-hidden-dim', '32',
        '--couple-num-residual-blocks', '1',
        '--couple-projector-dim', '8',
        '--bn-calibration-steps', '2',
    ])
    run_dirs = glob.glob(os.path.join(experiments, '*'))
    assert len(run_dirs) == 1
    checkpoints_dir = os.path.join(run_dirs[0], 'checkpoints')
    best_path = os.path.join(checkpoints_dir, 'best_model.pt')
    assert os.path.exists(best_path)
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    assert 'couple_reranker_state_dict' in checkpoint
    assert checkpoint['feature_layout'] == {
        'track_embed_dim': 32,
        'rest_dim': 35,
        'couple_feature_dim_total': 99,
    }
    assert os.path.exists(
        os.path.join(checkpoints_dir, 'best_model_calibrated.pt'),
    )


def test_dump_mode_requires_both_dump_paths():
    from train_couple_reranker import main as train_main

    with pytest.raises(SystemExit):
        train_main(['--train-dump', 'only_train.parquet'])
