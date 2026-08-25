from __future__ import annotations

import argparse

from utils.training import add_common_training_args


def test_fetch_step_files_flag_parses_and_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_common_training_args(parser, require_data_args=False)
    assert parser.parse_args([]).fetch_step_files is None
    assert parser.parse_args(['--fetch-step-files', '1']).fetch_step_files == 1
