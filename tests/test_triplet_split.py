from __future__ import annotations

import numpy as np

from utils.triplet_split import load_split, make_event_split, write_split


def test_split_is_deterministic_for_same_seed():
    a_train, a_test = make_event_split(1000, seed=0)
    b_train, b_test = make_event_split(1000, seed=0)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_test, b_test)


def test_split_differs_for_different_seed():
    _, test0 = make_event_split(1000, seed=0)
    _, test1 = make_event_split(1000, seed=1)
    assert not np.array_equal(test0, test1)


def test_split_disjoint_and_covers_all():
    train, test = make_event_split(1000, seed=0)
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted(train.tolist() + test.tolist()) == list(range(1000))


def test_split_respects_fraction():
    train, test = make_event_split(1000, frac_train=0.8, seed=0)
    assert len(train) == 800
    assert len(test) == 200


def test_split_indices_sorted():
    train, test = make_event_split(1000, seed=3)
    assert np.array_equal(train, np.sort(train))
    assert np.array_equal(test, np.sort(test))


def test_write_then_load_round_trip(tmp_path):
    path = str(tmp_path / "split.json")
    train, test = write_split(path, 500, frac_train=0.8, seed=7)
    assert np.array_equal(load_split(path, "train"), train)
    assert np.array_equal(load_split(path, "test"), test)
    assert len(train) == 400 and len(test) == 100
