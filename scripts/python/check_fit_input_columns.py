import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq

COLUMNS = [
    'track_covariance_dxy_dxy',
    'track_covariance_dsz_dsz',
    'track_covariance_dxy_dsz',
    'track_covariance_phi_dxy',
    'track_covariance_phi_phi',
    'track_covariance_lambda_lambda',
    'track_pt_error',
    'track_norm_chi2',
    'track_dxy_significance',
    'track_dz_significance',
    'track_vertex_x',
    'track_vertex_y',
    'track_vertex_z',
    'track_pt',
    'track_eta',
    'track_phi',
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Scan the raw track columns feeding the vertex-fit layer '
                    'for non-finite or non-positive values.')
    parser.add_argument('--src-glob', required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.src_glob))
    assert paths, f'no shards match {args.src_glob}'
    stats = {name: {'min': np.inf, 'max': -np.inf, 'n_nonfinite': 0,
                    'n_nonpositive': 0, 'n_total': 0} for name in COLUMNS}
    for path in paths:
        for name in COLUMNS:
            column = pq.read_table(path, columns=[name])[name]
            values = column.combine_chunks().flatten() \
                .to_numpy(zero_copy_only=False).astype(np.float64)
            finite = np.isfinite(values)
            entry = stats[name]
            entry['n_total'] += int(values.size)
            entry['n_nonfinite'] += int(values.size - finite.sum())
            entry['n_nonpositive'] += int((values[finite] <= 0.0).sum())
            if finite.any():
                entry['min'] = min(entry['min'], float(values[finite].min()))
                entry['max'] = max(entry['max'], float(values[finite].max()))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
