import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from scripts.python.score_couples_from_dump import score_dump

_ROOT = os.path.join(os.path.dirname(__file__), '..')
_DUMP = os.path.join(_ROOT, 'data', 'dumps', 'stage3_dump_eval.parquet')
_REFERENCE = os.path.join(_ROOT, 'data', 'dumps', 'couples_k125_dump.parquet')
_CHECKPOINT = os.path.join(_ROOT, 'models', 'couple_reranker_best.pt')

_HAVE_INPUTS = all(os.path.exists(path)
                   for path in (_DUMP, _REFERENCE, _CHECKPOINT))
requires_inputs = pytest.mark.skipif(
    not _HAVE_INPUTS,
    reason='stage-3 dump, reference couples dump or checkpoint not present',
)

NUM_EVENTS = 64
TOP_K2 = 125
NUM_COUPLES = 200


@pytest.fixture(scope='module')
def scored(tmp_path_factory):
    output = str(tmp_path_factory.mktemp('scored') / 'couples.parquet')
    score_dump(
        dump_path=_DUMP,
        checkpoint_path=_CHECKPOINT,
        output_path=output,
        top_k2=TOP_K2,
        num_couples=NUM_COUPLES,
        batch_size=64,
        device='cpu',
        max_events=NUM_EVENTS,
        num_workers=0,
    )
    return pq.read_table(output)


@pytest.fixture(scope='module')
def reference():
    return pq.ParquetFile(_REFERENCE).read_row_group(0).slice(0, NUM_EVENTS)


@requires_inputs
def test_schema_matches_the_perstage_contract(scored):
    from scripts.python.eval_cascade_pipeline import OUTPUT_SCHEMA
    assert scored.schema.names == OUTPUT_SCHEMA.names
    assert scored.num_rows == NUM_EVENTS


@requires_inputs
def test_event_identity_is_carried_through_in_order(scored, reference):
    for column in ('event_run', 'event_id', 'event_luminosity_block'):
        assert scored.column(column).to_pylist() == \
            reference.column(column).to_pylist()


@requires_inputs
def test_stage1_and_stage2_columns_are_copied_from_the_dump(scored):
    source = next(pq.ParquetFile(_DUMP).iter_batches(batch_size=NUM_EVENTS))
    for column in ('stage1_sorted_indices', 'stage2_sorted_indices',
                   'stage1_scores', 'stage2_scores'):
        assert scored.column(column).to_pylist() == \
            source.column(column).to_pylist()


@requires_inputs
def test_couples_are_capped_and_score_ordered(scored):
    couples = scored.column('stage3_sorted_couples').to_pylist()
    scores = scored.column('stage3_couple_scores').to_pylist()
    for event_couples, event_scores in zip(couples, scores):
        assert len(event_couples) == len(event_scores) <= NUM_COUPLES
        assert all(len(pair) == 2 for pair in event_couples)
        assert all(pair[0] != pair[1] for pair in event_couples)
        assert event_scores == sorted(event_scores, reverse=True)


@requires_inputs
def test_ranked_list_tracks_the_live_cascade_reference(scored, reference):
    """The reference ran the live cascade at batch 64; this path replays Stage 3
    from the dump built at batch 384. Both stages carry BatchNorm with
    track_running_stats=False, so the upstream selection itself is
    batch-dependent (measured: Stage-2 top-125 overlap 0.968) and the couple
    lists agree closely rather than exactly. A bug in the top-K2 selection or
    the couple index mapping would collapse the overlap, which is what this
    guards."""
    scored_couples = scored.column('stage3_sorted_couples').to_pylist()
    reference_couples = reference.column('stage3_sorted_couples').to_pylist()
    scored_scores = scored.column('stage3_couple_scores').to_pylist()
    reference_scores = reference.column('stage3_couple_scores').to_pylist()

    overlaps = []
    score_deltas = []
    top_one_matches = 0
    for event in range(len(scored_couples)):
        mine = {tuple(sorted(pair)): score for pair, score
                in zip(scored_couples[event], scored_scores[event])}
        theirs = {tuple(sorted(pair)): score for pair, score
                  in zip(reference_couples[event], reference_scores[event])}
        shared = set(mine) & set(theirs)
        overlaps.append(len(shared) / max(len(theirs), 1))
        score_deltas.extend(abs(mine[key] - theirs[key]) for key in shared)
        top_one_matches += sorted(scored_couples[event][0]) == \
            sorted(reference_couples[event][0])

    print(f'\ncouple overlap: mean {np.mean(overlaps):.4f} '
          f'min {np.min(overlaps):.4f} | top-1 identical '
          f'{top_one_matches / len(scored_couples):.3f} | score delta mean '
          f'{np.mean(score_deltas):.4f} max {np.max(score_deltas):.4f}')
    assert float(np.mean(overlaps)) >= 0.85
    assert top_one_matches / len(scored_couples) >= 0.80
    assert float(np.mean(score_deltas)) <= 0.30
