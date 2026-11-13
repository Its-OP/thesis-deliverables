"""Composite event key shared by the ROOT->parquet converter and the baseline
extractor.

The downstream tooling joins events on the 5-key
``(event_run, event_id, event_luminosity_block, source_batch_id,
source_microbatch_id)``. ``(run, event, luminosityBlock)`` repeats inside one
file, so the two source-id columns must make the key unique:

* old merged production: the ``source_batch_id`` / ``source_microbatch_id``
  branches exist (written by merge_batches.py) and are copied;
* abcbatch production (``abcbatch{B}_2024_job{J}_ditaus_mc.root``, no such
  branches): ``source_batch_id = B`` and
  ``source_microbatch_id = J * ENTRIES_PER_JOB_LIMIT + entry_index``;
* anything else without the branches: both ids are -1 (legacy fill).

``event_source_job`` / ``event_source_entry`` carry the decoded job number and
entry index (not part of the key).
"""
from __future__ import annotations

import os
import re
import warnings
from collections import namedtuple

import numpy as np

BATCH_NUMBER_PATTERN = re.compile(r'abcbatch(\d+)')      # anywhere in the path
FALLBACK_BATCH_PATTERN = re.compile(r'batch(\d+)')       # file name only (merged production)
JOB_NUMBER_PATTERN = re.compile(r'job(\d+)')             # abcbatch ntuplizer files
FILE_NUMBER_PATTERN = re.compile(r'_(\d+)\.root$')       # CRAB outputs: miniaod_<N>.root, step1_<N>.root
ENTRIES_PER_JOB_LIMIT = 100_000
INT32_MAX = 2 ** 31 - 1
MISSING_SOURCE_ID = -1

EVENT_ID_BRANCHES = ('run', 'event', 'luminosityBlock')
SOURCE_ID_BRANCHES = ('source_batch_id', 'source_microbatch_id')
IDENTITY_COLUMNS = ('event_run', 'event_id', 'event_luminosity_block',
                    'source_batch_id', 'source_microbatch_id')
SOURCE_DETAIL_COLUMNS = ('event_source_job', 'event_source_entry')
IDENTITY_DTYPES = {
    'event_run': np.int32, 'event_id': np.int64, 'event_luminosity_block': np.int32,
    'source_batch_id': np.int32, 'source_microbatch_id': np.int32,
    'event_source_job': np.int32, 'event_source_entry': np.int32,
}

SourceIds = namedtuple('SourceIds', ['batch', 'microbatch', 'job', 'entry', 'origin'])


def batch_number_from_filename(path):
    """path: str. Returns the batch number from the first ``abcbatch<N>``
    token anywhere in the path (file name or a parent directory), else
    from a ``batch<N>`` token in the file name, else None."""
    match = (BATCH_NUMBER_PATTERN.search(os.path.normpath(path))
             or FALLBACK_BATCH_PATTERN.search(os.path.basename(path)))
    return int(match.group(1)) if match else None


def job_number_from_filename(path):
    """path: str. Returns the number of the ``job<N>`` token of the file
    name, else the trailing ``_<N>.root`` number of a CRAB output, else
    None."""
    basename = os.path.basename(path)
    match = JOB_NUMBER_PATTERN.search(basename) or FILE_NUMBER_PATTERN.search(basename)
    return int(match.group(1)) if match else None


def file_sort_key(path):
    """path: str. Returns (batch, job) for numeric file ordering; missing
    tokens count as 0."""
    return (batch_number_from_filename(path) or 0, job_number_from_filename(path) or 0)


def assert_unique_file_keys(paths):
    """paths: list of ROOT files of one run. Raises ValueError when two
    files with a job token share (batch, job): their event keys and their
    per-file output names would collide."""
    seen = {}
    for path in paths:
        job_number = job_number_from_filename(path)
        if job_number is None:
            continue
        key = (batch_number_from_filename(path) or 0, job_number)
        if key in seen:
            raise ValueError(f'{os.path.basename(seen[key])} and {os.path.basename(path)} share '
                             f'batch/job {key}: their event keys would collide')
        seen[key] = path


def encode_microbatch_id(job_number, entry_index):
    """job_number: int; entry_index: int or (N,) ints. Returns int32
    ``job_number * ENTRIES_PER_JOB_LIMIT + entry_index``."""
    entry_index = np.asarray(entry_index, dtype=np.int64)
    if np.any(entry_index < 0) or np.any(entry_index >= ENTRIES_PER_JOB_LIMIT):
        raise ValueError(f'entry index must lie in [0, {ENTRIES_PER_JOB_LIMIT}) to be packed')
    if job_number < 0 or job_number * ENTRIES_PER_JOB_LIMIT + ENTRIES_PER_JOB_LIMIT - 1 > INT32_MAX:
        raise ValueError(f'job number {job_number} cannot be packed into int32')
    encoded = (job_number * ENTRIES_PER_JOB_LIMIT + entry_index).astype(np.int32)
    return encoded if encoded.ndim else int(encoded)


def decode_microbatch_id(value):
    """value: int or (N,) packed ids. Returns (job_number, entry_index)."""
    value = np.asarray(value, dtype=np.int64)
    return (value // ENTRIES_PER_JOB_LIMIT).astype(np.int32), (value % ENTRIES_PER_JOB_LIMIT).astype(np.int32)


def _window(tree, entry_start, entry_stop):
    start = 0 if entry_start is None else int(entry_start)
    stop = tree.num_entries if entry_stop is None else int(entry_stop)
    return start, stop


def source_ids(tree, path, entry_start=None, entry_stop=None):
    """tree: uproot Events tree; path: its file path; [entry_start,
    entry_stop) window. Returns SourceIds of (N,) int32 arrays and the origin
    ('branches', 'filename' or 'missing')."""
    start, stop = _window(tree, entry_start, entry_stop)
    count = stop - start
    available = set(tree.keys())
    if all(branch in available for branch in SOURCE_ID_BRANCHES):
        data = tree.arrays(list(SOURCE_ID_BRANCHES), entry_start=start, entry_stop=stop)
        batch = np.asarray(data['source_batch_id']).astype(np.int32)
        microbatch = np.asarray(data['source_microbatch_id']).astype(np.int32)
        return SourceIds(batch, microbatch, microbatch.copy(),
                         np.full(count, MISSING_SOURCE_ID, dtype=np.int32), 'branches')
    job_number = job_number_from_filename(path)
    if job_number is None:
        missing = np.full(count, MISSING_SOURCE_ID, dtype=np.int32)
        return SourceIds(missing, missing.copy(), missing.copy(), missing.copy(), 'missing')
    batch_number = batch_number_from_filename(path)
    if batch_number is None:
        print(f'  identity: no batch token in {os.path.basename(path)}, using batch 0')
        batch_number = 0
    entry = np.arange(start, stop, dtype=np.int32)
    return SourceIds(np.full(count, batch_number, dtype=np.int32),
                     encode_microbatch_id(job_number, entry),
                     np.full(count, job_number, dtype=np.int32), entry, 'filename')


def identity_columns(tree, path, entry_start=None, entry_stop=None):
    """tree, path, window as in source_ids. Returns a dict of the five
    IDENTITY_COLUMNS plus SOURCE_DETAIL_COLUMNS as (N,) arrays in their
    final dtypes (no event mask applied)."""
    start, stop = _window(tree, entry_start, entry_stop)
    data = tree.arrays(list(EVENT_ID_BRANCHES), entry_start=start, entry_stop=stop)
    ids = source_ids(tree, path, start, stop)
    columns = {
        'event_run': np.asarray(data['run']).astype(np.int32),
        'event_id': np.asarray(data['event']).astype(np.int64),
        'event_luminosity_block': np.asarray(data['luminosityBlock']).astype(np.int32),
        'source_batch_id': ids.batch,
        'source_microbatch_id': ids.microbatch,
        'event_source_job': ids.job,
        'event_source_entry': ids.entry,
    }
    return columns


def identity_tuples(columns):
    """columns: mapping with the five IDENTITY_COLUMNS as (N,) arrays.
    Returns a list of N 5-tuples of Python ints."""
    arrays = [np.asarray(columns[name]).tolist() for name in IDENTITY_COLUMNS]
    return list(zip(*arrays))


def assert_unique_identity(columns, context=''):
    """columns: as in identity_tuples, optionally with event_source_entry.
    Raises ValueError when two rows with file-derived source ids (an
    event_source_entry >= 0, or a source_batch_id != -1 when that column is
    absent) share the 5-key; duplicates among the other rows (copied or
    legacy -1 ids) only warn."""
    tuples = identity_tuples(columns)
    if 'event_source_entry' in columns:
        strict_rows = [int(entry) >= 0 for entry in np.asarray(columns['event_source_entry']).tolist()]
    else:
        strict_rows = [key[3] != MISSING_SOURCE_ID for key in tuples]
    seen = set()
    duplicates = []
    strict_duplicates = set()
    for key, strict in zip(tuples, strict_rows):
        if key in seen:
            duplicates.append(key)
            if strict:
                strict_duplicates.add(key)
        seen.add(key)
    if not duplicates:
        return
    real = [key for key in duplicates if key in strict_duplicates]
    if real:
        raise ValueError(f'{context}: {len(real)} duplicate event keys with real source ids, '
                         f'first {real[0]}')
    warnings.warn(f'{context}: {len(duplicates)} duplicate event keys among rows with legacy '
                  f'-1 source ids, first {duplicates[0]}')
