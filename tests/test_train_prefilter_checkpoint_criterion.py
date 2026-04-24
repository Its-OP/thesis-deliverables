"""Tests for ``train_prefilter`` CLI checkpoint-criterion wiring.

Phase-0 instrumentation for the prefilter expressiveness sweep: best
checkpoint selection must be switchable from the default ``R@200`` to
``P@256`` so every run's chosen model tracks the target metric directly.
These tests pin:

1. The argparse parser exposes ``--checkpoint-criterion`` with at least
   ``recall_at_200`` and ``perfect_at_256`` as choices; default is
   ``recall_at_200`` (backcompat with the 17-experiment campaign).
2. ``K = 256`` is present in the val ``MetricsAccumulator`` ``k_values``
   tuple so ``perfect_at_256`` is actually computed at val time.
3. ``perfect_at_N`` for an arbitrary N matches a direct P@K computation
   on the same scores/labels, so the checkpoint-selection signal is
   equivalent to running the perfect-recall diagnostic offline.
"""
from __future__ import annotations

import torch


class TestCheckpointCriterionFlag:
    def test_flag_accepts_perfect_at_256(self):
        from train_prefilter import _build_argument_parser

        parser = _build_argument_parser()
        args = parser.parse_args([
            '--data-config', 'foo.yaml',
            '--data-dir', 'train/',
            '--val-data-dir', 'val/',
            '--network', 'net.py',
            '--experiments-dir', '/tmp',
            '--model-name', 'test',
            '--checkpoint-criterion', 'perfect_at_256',
        ])
        assert args.checkpoint_criterion == 'perfect_at_256'

    def test_flag_default_is_recall_at_200(self):
        from train_prefilter import _build_argument_parser

        parser = _build_argument_parser()
        args = parser.parse_args([
            '--data-config', 'foo.yaml',
            '--data-dir', 'train/',
            '--val-data-dir', 'val/',
            '--network', 'net.py',
            '--experiments-dir', '/tmp',
            '--model-name', 'test',
        ])
        assert args.checkpoint_criterion == 'recall_at_200'

    def test_flag_rejects_unsupported_criterion(self):
        import pytest

        from train_prefilter import _build_argument_parser

        parser = _build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                '--data-config', 'foo.yaml',
                '--data-dir', 'train/',
                '--val-data-dir', 'val/',
                '--network', 'net.py',
                '--experiments-dir', '/tmp',
                '--model-name', 'test',
                '--checkpoint-criterion', 'not_a_real_metric',
            ])


class TestK256InMetricsAccumulator:
    def test_val_k_values_include_256(self):
        """The training script's val MetricsAccumulator must emit
        ``perfect_at_256`` so ``--checkpoint-criterion perfect_at_256``
        has a value to compare."""
        from train_prefilter import VAL_METRICS_K_VALUES

        assert 256 in VAL_METRICS_K_VALUES, (
            f'K=256 missing from VAL_METRICS_K_VALUES '
            f'{VAL_METRICS_K_VALUES}'
        )


class TestPerfectAt256SemanticsMatchDiagnostic:
    def test_perfect_at_256_matches_manual_p_at_k(self):
        """MetricsAccumulator's ``perfect_at_256`` must agree with a
        direct "all GT in top-256" fraction — same definition as the
        offline diagnostic in ``prefilter_perfect_recall_diagnostic``.
        """
        from utils.training_utils import MetricsAccumulator

        # 3 events, 300 tracks each, 3 GT pions per event.
        # Event 0: all 3 GT within top-10 (perfect @256).
        # Event 1: 2 of 3 GT in top-256; 3rd at rank ~270 (fail @256).
        # Event 2: all 3 GT at tail (fail @256).
        num_events = 3
        num_tracks = 300
        scores = torch.full((num_events, num_tracks), -1e3, dtype=torch.float32)
        labels = torch.zeros(num_events, 1, num_tracks)
        mask = torch.ones(num_events, 1, num_tracks)

        # Event 0: GT at positions 0, 1, 2 with the three highest scores.
        scores[0, 0] = 100; scores[0, 1] = 99; scores[0, 2] = 98
        scores[0, 3:] = torch.linspace(-100, -1, num_tracks - 3)
        labels[0, 0, 0] = 1; labels[0, 0, 1] = 1; labels[0, 0, 2] = 1

        # Event 1: GT at positions 0, 1, 270. Third GT ranks beyond 256
        # because 268 other tracks (indices 2..269 except the GT) score
        # higher than position 270.
        scores[1, 0] = 100; scores[1, 1] = 99
        scores[1, 2:270] = torch.linspace(80, 10, 268)   # 268 non-GT above pos 270
        scores[1, 270] = 0                                # GT ranked ~270
        scores[1, 271:] = torch.linspace(-10, -90, num_tracks - 271)
        labels[1, 0, 0] = 1; labels[1, 0, 1] = 1; labels[1, 0, 270] = 1

        # Event 2: all 3 GT at the tail.
        scores[2, :297] = torch.linspace(100, -50, 297)
        scores[2, 297] = -60; scores[2, 298] = -70; scores[2, 299] = -80
        labels[2, 0, 297] = 1; labels[2, 0, 298] = 1; labels[2, 0, 299] = 1

        accumulator = MetricsAccumulator(k_values=(200, 256))
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()

        # Only event 0 is perfect @256 → 1/3.
        assert metrics['perfect_at_256'] == 1 / 3

        # Direct P@K reference computation as a sanity peer.
        gt_indices = [
            [int(i) for i in labels[event, 0].nonzero(as_tuple=True)[0].tolist()]
            for event in range(num_events)
        ]
        pass_count = 0
        for event in range(num_events):
            sorted_positions = torch.argsort(
                scores[event], descending=True,
            )[:256].tolist()
            if all(g in sorted_positions for g in gt_indices[event]):
                pass_count += 1
        assert pass_count / num_events == metrics['perfect_at_256']
