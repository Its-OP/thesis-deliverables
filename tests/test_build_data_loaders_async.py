"""The dataset's async_load background thread deadlocks fork-based
DataLoader workers (thread started in the parent, lock inherited locked by
the children — observed as a silent post-load hang on two training boxes).
build_data_loaders must therefore construct every SimpleIterDataset with
async_load=False; in_memory mode gains nothing from the async thread."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import torch.utils.data

from utils import training


class _CapturingDataset(torch.utils.data.IterableDataset):
    captured_kwargs: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        _CapturingDataset.captured_kwargs.append(kwargs)
        self.config = SimpleNamespace(
            input_names=('pf_points', 'pf_features', 'pf_vectors',
                         'pf_mask', 'pf_label'))

    def __iter__(self):
        return iter(())


def test_datasets_disable_async_load(tmp_path, monkeypatch):
    (tmp_path / 'shard.parquet').touch()
    _CapturingDataset.captured_kwargs.clear()
    monkeypatch.setattr(training, 'SimpleIterDataset', _CapturingDataset)
    arguments = SimpleNamespace(
        no_in_memory=False,
        data_dir=str(tmp_path),
        val_data_dir=str(tmp_path),
        data_config='unused.yaml',
        train_fraction=0.8,
        num_workers=0,
        batch_size=4,
        steps_per_epoch=5,
    )
    training.build_data_loaders(arguments, logging.getLogger('test'))

    assert len(_CapturingDataset.captured_kwargs) == 2
    for kwargs in _CapturingDataset.captured_kwargs:
        assert kwargs.get('async_load') is False
