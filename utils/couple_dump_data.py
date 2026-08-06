"""Map-style dataset over Stage-3 input dumps written by
``scripts/python/eval_cascade_pipeline.py --dump-stage3-inputs``.

The entire dump is held in memory as per-row numpy arrays. This is
intentional: dumps are a few GB even for full train/val sets, are read once
per run, and random-access shuffling at batch 2048 needs the whole set
resident anyway.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import torch

from utils.couple_features import REQUIRED_POINT_CHANNELS, TRACK_EMBED_DIM

LORENTZ_CHANNELS = 4

_DUMP_COLUMNS = (
    'k1_features', 'k1_points', 'k1_lorentz',
    'k1_stage1_scores', 'k1_stage2_scores',
    'k1_labels', 'k1_original_indices',
    'cone_eta', 'cone_phi', 'cone_dz', 'cone_pt',
)
_CONE_KEYS = ('cone_eta', 'cone_phi', 'cone_dz', 'cone_pt')
_STACKED_KEYS = (
    'features', 'points', 'lorentz', 'stage1_scores', 'stage2_scores',
    'labels', 'original_indices',
)


def _row_tensor(row: np.ndarray, dtype: type) -> torch.Tensor:
    converted = np.asarray(row, dtype=dtype)
    # pyarrow-backed rows are read-only; torch.from_numpy warns on those.
    if not converted.flags.writeable:
        converted = converted.copy()
    return torch.from_numpy(converted)


class CoupleDumpDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_paths: list[str]):
        column_chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in _DUMP_COLUMNS
        }
        for path in parquet_paths:
            table = pq.read_table(path, columns=list(_DUMP_COLUMNS))
            for name in _DUMP_COLUMNS:
                column_chunks[name].append(
                    table.column(name).to_numpy(zero_copy_only=False),
                )
        self._columns = {
            name: np.concatenate(chunks)
            for name, chunks in column_chunks.items()
        }
        self._length = len(self._columns['k1_stage2_scores'])
        if self._length == 0:
            raise ValueError(f'No events in dump files: {parquet_paths}')

        # K1 is inferred from list lengths plus the known channel counts.
        self.top_k1 = len(self._columns['k1_stage2_scores'][0])
        k1_lengths = {
            len(row) for row in self._columns['k1_stage2_scores']
        }
        if k1_lengths != {self.top_k1}:
            raise ValueError(
                f'Inconsistent K1 across dump rows: {sorted(k1_lengths)}',
            )
        first_features = self._columns['k1_features'][0]
        if len(first_features) != TRACK_EMBED_DIM * self.top_k1:
            raise ValueError(
                f'k1_features length {len(first_features)} does not match '
                f'{TRACK_EMBED_DIM} channels x K1={self.top_k1}.',
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Returns features (32, K1), points (26, K1), lorentz (4, K1),
        stage1_scores/stage2_scores/labels (K1,) float32,
        original_indices (K1,) long, cone_eta/cone_phi/cone_dz/cone_pt
        (n_valid,) float32."""
        columns = self._columns

        def as_float(name: str) -> torch.Tensor:
            return _row_tensor(columns[name][index], np.float32)

        item = {
            'features': as_float('k1_features').view(
                TRACK_EMBED_DIM, self.top_k1),
            'points': as_float('k1_points').view(
                REQUIRED_POINT_CHANNELS, self.top_k1),
            'lorentz': as_float('k1_lorentz').view(
                LORENTZ_CHANNELS, self.top_k1),
            'stage1_scores': as_float('k1_stage1_scores'),
            'stage2_scores': as_float('k1_stage2_scores'),
            'labels': as_float('k1_labels'),
            'original_indices': _row_tensor(
                columns['k1_original_indices'][index], np.int64),
        }
        for name in _CONE_KEYS:
            item[name] = as_float(name)
        return item

    @staticmethod
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """batch: list of ``__getitem__`` dicts. Returns the fixed-size keys
        stacked to (B, ...), cone_eta/cone_phi/cone_dz/cone_pt zero-padded to
        (B, P) with P = batch-max cone length, cone_valid_mask (B, P) bool,
        cone_points (B, 3, P) and cone_lorentz (B, 4, P) with channel 0 = pT
        and channels 1..3 = 0 (the couple builder only consumes
        hypot(ch0, ch1) = pT)."""
        collated = {
            key: torch.stack([item[key] for item in batch])
            for key in _STACKED_KEYS
        }
        batch_size = len(batch)
        cone_lengths = [item['cone_eta'].shape[0] for item in batch]
        max_length = max(cone_lengths)
        for key in _CONE_KEYS:
            padded = torch.zeros(batch_size, max_length)
            for row, item in enumerate(batch):
                padded[row, :cone_lengths[row]] = item[key]
            collated[key] = padded
        cone_valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        for row, length in enumerate(cone_lengths):
            cone_valid_mask[row, :length] = True
        collated['cone_valid_mask'] = cone_valid_mask
        collated['cone_points'] = torch.stack(
            [collated['cone_eta'], collated['cone_phi'], collated['cone_dz']],
            dim=1,
        )
        cone_lorentz = torch.zeros(batch_size, LORENTZ_CHANNELS, max_length)
        cone_lorentz[:, 0, :] = collated['cone_pt']
        collated['cone_lorentz'] = cone_lorentz
        return collated
