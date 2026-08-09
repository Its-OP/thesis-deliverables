import argparse
import json
import os

import joblib
import numpy as np
import torch
from tqdm import tqdm

from scripts.python.probe_triplet_features import FEATURE_SETS
from scripts.python.build_triplet_filter_table import (
    ROLE_DEFAULTS,
    _event_candidates,
    _featurize,
    _shard_blocks,
    _worker_pool,
    _WORKER_STATE,
)
from utils.triplet_join import FEATURE_NAMES, FEATURE_NAMES_EXTENDED  # noqa: F401

K_VALUES = (1, 5, 10, 20, 50, 100)


def deduped_gt_rank(scores, is_gt, triplets):
    """Rank of the ground-truth 3-set once duplicate 3-sets are collapsed: the
    same (i, j, k) reaches the list through up to three different couples, and
    counting those separately would inflate every rank ahead of it."""
    if not is_gt.any():
        return None
    order = np.argsort(-scores, kind="stable")
    seen = set()
    rank = 0
    for index in order:
        key = tuple(sorted(triplets[index]))
        if key in seen:
            continue
        seen.add(key)
        if is_gt[index]:
            return rank
        rank += 1
    return None


def feature_columns_for_width(width):
    """The sweep trains on named subsets of the extended layout, so a model's
    own input width identifies which columns it expects. The subsets are
    prefixes of that layout, so the H6 part is the leading `width - 89`
    columns and anything past it need never be computed."""
    for name, names in FEATURE_SETS.items():
        if len(names) == width:
            indices = [FEATURE_NAMES_EXTENDED.index(entry) for entry in names]
            h6_width = max(width - len(FEATURE_NAMES), 0)
            return name, np.asarray(indices), h6_width
    raise ValueError(
        f"no feature set has {width} columns; known widths are "
        f"{sorted(len(names) for names in FEATURE_SETS.values())}")


def use_cpu_inference(model):
    """XGBoost boosters trained on the GPU keep device='cuda'. Setting the
    device back is not enough: every predict call still probes CUDA, which
    cannot initialize in a forked worker and stalls for tens of seconds each
    time. Hiding the device makes the booster take the CPU path directly —
    measured 200 events in 13 s against 9 minutes for 100."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        model.get_booster().set_param({"device": "cpu"})
        model.set_params(device="cpu")
    except AttributeError:
        pass  # sklearn estimators have no device to set
    return model


def _rank_for_event(r, couples, cols, top_c, model, h6_width, columns=None):
    candidates = _event_candidates(r, couples, cols, top_c)
    is_gt = candidates["is_gt"]
    n_candidates = len(is_gt)
    if n_candidates == 0:
        return dict(rank=None, n_candidates=0,
                    reconstructable=candidates["reconstructable"])
    features = _featurize(candidates, np.arange(n_candidates), r, cols,
                          h6_width > 0, h6_width)
    if columns is not None:
        features = features[:, columns]
    scores = model.predict_proba(features)[:, 1]
    triplets = candidates["triplets"].numpy()
    return dict(rank=deduped_gt_rank(scores, is_gt, triplets),
                n_candidates=n_candidates,
                reconstructable=candidates["reconstructable"])


def _worker_rank(r):
    state = _WORKER_STATE
    return _rank_for_event(r, state["couples"], state["cols"], state["top_c"],
                           state["model"], state["h6_width"], state["columns"])


def evaluate(dump_path, src_glob, model, *, top_c=100, max_events=None,
             workers=0):
    width = int(getattr(model, "n_features_in_", len(FEATURE_NAMES)))
    feature_set, columns, h6_width = feature_columns_for_width(width)
    use_cpu_inference(model)
    print(f'feature set {feature_set} ({width} columns), h6 block {h6_width})')
    results = []
    for couples, cols, n in _shard_blocks(dump_path, src_glob, max_events):
        if workers > 1:
            state = dict(couples=couples, cols=cols, top_c=top_c, model=model,
                         h6_width=h6_width, columns=columns)
            with _worker_pool(workers, state) as pool:
                results.extend(tqdm(
                    pool.imap(_worker_rank, range(n), chunksize=8), total=n,
                    desc=f"rank x{workers} (+{n})"))
        else:
            results.extend(
                _rank_for_event(r, couples, cols, top_c, model, h6_width, columns)
                for r in tqdm(range(n), desc=f"rank (+{n})"))

    n_events = len(results)
    ranks = [entry["rank"] for entry in results]
    found = [rank for rank in ranks if rank is not None]
    # T@K denominates by ALL events, matching the cascade's other metrics.
    metrics = {f"T@{k}": sum(1 for rank in found if rank < k) / n_events
               for k in K_VALUES}
    metrics.update(
        n_events=n_events,
        n_features=width,
        feature_set=feature_set,
        h6_width=int(h6_width),
        reconstructable=sum(1 for e in results if e["reconstructable"]) / n_events,
        gt_in_list=len(found) / n_events,
        median_rank=float(np.median(found)) if found else float("nan"),
        mean_candidates=float(np.mean([e["n_candidates"] for e in results])),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Deduplicated T@K of a third-pion filter over the full '
                    'Tier-H survivor list.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dump', default=ROLE_DEFAULTS['eval'][0])
    parser.add_argument('--src-glob', default=ROLE_DEFAULTS['eval'][1])
    parser.add_argument('--top-c', type=int, default=100)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model = joblib.load(args.model)
    metrics = evaluate(args.dump, args.src_glob, model, top_c=args.top_c,
                       max_events=args.max_events, workers=args.workers)
    print(os.path.basename(args.model))
    for key, value in metrics.items():
        print(f'  {key}: {value}')
    if args.output:
        with open(args.output, 'w') as handle:
            json.dump({'model': args.model, **metrics}, handle, indent=2)
        print(f'-> {args.output}')


if __name__ == '__main__':
    main()
