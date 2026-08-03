"""Phase 2 gate test: SimpleIterDataset loads a subset parquet and yields one batch.

Verifies the utils.dataset + utils.data.{tools,fileio,config,preprocess}
import graph is wired end-to-end. Uses real parquet data (no mocks — see
feedback_no_fake_data).

Subset location is resolved in this order:
    1. env var TAU_SUBSET_DIR (absolute path to dir containing train/, val/)
    2. ../part/data/low-pt/subset (repo-local fallback during migration)

If neither exists, the test is skipped with a clear message.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from weaver.utils.dataset import SimpleIterDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = REPO_ROOT.parent / 'part' / 'data' / 'low-pt' / 'subset'
DATA_CONFIG = REPO_ROOT / 'data' / 'low-pt' / 'lowpt_tau_trackfinder.yaml'
# The frozen legacy sidecar is a complete, self-contained data config; the
# dataset tests use it because the subset fixtures carry only the legacy
# columns the live 32-channel config no longer suffices with.
FROZEN_CONFIG = (REPO_ROOT / 'data' / 'low-pt' /
                 'lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml')


def _resolve_subset_dir() -> Path | None:
    env_dir = os.environ.get('TAU_SUBSET_DIR')
    if env_dir:
        candidate = Path(env_dir)
        if candidate.is_dir():
            return candidate
    if DEFAULT_SUBSET.is_dir():
        return DEFAULT_SUBSET
    return None


@pytest.fixture(scope='module')
def subset_dir() -> Path:
    resolved = _resolve_subset_dir()
    if resolved is None:
        pytest.skip(
            f'Subset data not found. Set TAU_SUBSET_DIR or place files at {DEFAULT_SUBSET}'
        )
    return resolved


def test_data_config_yaml_exists():
    assert DATA_CONFIG.exists(), f'Missing data config: {DATA_CONFIG}'


def test_simpleiterdataset_constructs(subset_dir: Path):
    train_files = sorted(str(p) for p in (subset_dir / 'train').glob('*.parquet'))
    assert train_files, f'No train parquet files in {subset_dir}/train'

    dataset = SimpleIterDataset(
        file_dict={'all': train_files},
        data_config_file=str(FROZEN_CONFIG),
        for_training=False,
        fetch_by_files=True,
        fetch_step=1,
    )
    assert dataset.config is not None
    assert dataset.config.input_names, 'DataConfig has no input groups'


def test_simpleiterdataset_yields_batch(subset_dir: Path):
    train_files = sorted(str(p) for p in (subset_dir / 'train').glob('*.parquet'))
    dataset = SimpleIterDataset(
        file_dict={'all': train_files[:1]},
        data_config_file=str(FROZEN_CONFIG),
        for_training=False,
        fetch_by_files=True,
        fetch_step=1,
        async_load=False,
    )
    iterator = iter(dataset)
    x, y, z = next(iterator)
    assert isinstance(x, dict) and len(x) > 0
    assert isinstance(y, dict) and len(y) > 0
