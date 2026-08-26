from __future__ import annotations

import argparse

from utils.training import add_common_training_args, resolve_eval_steps


def test_val_steps_flag_parses_and_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_common_training_args(parser, require_data_args=False)
    assert parser.parse_args([]).val_steps is None
    assert parser.parse_args(['--val-steps', '375']).val_steps == 375


def test_resolve_eval_steps_default_uses_divisor():
    assert resolve_eval_steps(500, 4, None) == 125
    assert resolve_eval_steps(100, 4, None) == 25
    assert resolve_eval_steps(3, 4, None) == 1


def test_resolve_eval_steps_override_wins():
    assert resolve_eval_steps(500, 4, 375) == 375
    assert resolve_eval_steps(100, 4, 1000) == 1000
    assert resolve_eval_steps(500, 4, 0) == 1
