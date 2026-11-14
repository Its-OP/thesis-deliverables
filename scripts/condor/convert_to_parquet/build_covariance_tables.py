"""One-off builder of the numpy covariance tables used by packed_candidate.py.

Source: CovarianceParameterization_version<N>.root of the public repository
github.com/cms-data/DataFormats-PatCandidates (the file CMSSW loads in
DataFormats/PatCandidates/src/CovarianceParameterization.cc). The output
holds, for the 15 upper-triangular elements (i <= j) of the 5x5 track
covariance in the order k = ((9 - i) * i) // 2 + j:

    pt_edges, eta_edges, hits_edges   bin edges of the three histogram axes
    reference    (15, 2, nx + 2, ny + 2, nz + 2) float32 mean values,
                 second axis 0 = tracks without pixel hits, 1 = with;
                 under- and overflow bins included, as TH3 stores them
    schema_ids   (S,) packing schema numbers
    schema_method, schema_target, schema_bits   (S, 15) ints
    schema_params  (S, 15, 2) float64 (NaN where a parameter is absent)

Needs ROOT. Usage:
    python build_covariance_tables.py CovarianceParameterization_version1.root \
        data/covariance_parameterization_version1.npz
"""
from __future__ import annotations

import sys

import numpy as np
import ROOT

PARAMETER_NAMES = ('qoverp', 'lambda', 'phi', 'dxy', 'dsz')
MAX_PARAMETERS = 2


def element_index(i, j):
    """i <= j in 0..4. Returns the position of the element in the tables."""
    return ((9 - i) * i) // 2 + j


def histogram_array(histogram):
    """histogram: TH3D. Returns (nx + 2, ny + 2, nz + 2) float64 including
    under- and overflow bins."""
    nx, ny, nz = histogram.GetNbinsX(), histogram.GetNbinsY(), histogram.GetNbinsZ()
    flat = np.frombuffer(histogram.GetArray(), dtype=np.float64, count=(nx + 2) * (ny + 2) * (nz + 2)).copy()
    return flat.reshape((nz + 2, ny + 2, nx + 2)).transpose(2, 1, 0)


def axis_edges(axis):
    """axis: TAxis. Returns the (N + 1,) bin edges."""
    return np.array([axis.GetBinLowEdge(index) for index in range(1, axis.GetNbins() + 2)])


def build(input_path, output_path):
    root_file = ROOT.TFile.Open(input_path)
    first = root_file.Get('covariance_qoverp_qoverp_pixel_parametrization')
    reference = np.zeros((15, 2) + histogram_array(first).shape, dtype=np.float32)
    for i in range(5):
        for j in range(i, 5):
            for has_pixel, tag in ((1, '_pixel_'), (0, '_noPixel_')):
                name = f'covariance_{PARAMETER_NAMES[i]}_{PARAMETER_NAMES[j]}{tag}parametrization'
                reference[element_index(i, j), has_pixel] = histogram_array(root_file.Get(name))
    schema_ids = sorted(int(key.GetName()) for key in root_file.Get('schemas').GetListOfKeys()
                        if key.GetName().isdigit())
    method = np.zeros((len(schema_ids), 15), dtype=np.int32)
    target = np.zeros((len(schema_ids), 15), dtype=np.int32)
    bits = np.zeros((len(schema_ids), 15), dtype=np.int32)
    params = np.full((len(schema_ids), 15, MAX_PARAMETERS), np.nan)
    for row, schema_id in enumerate(schema_ids):
        for i in range(5):
            for j in range(i, 5):
                folder = f'schemas/{schema_id}/{i}{j}'
                column = element_index(i, j)
                method[row, column] = root_file.Get(folder + '/method').GetVal()
                target[row, column] = root_file.Get(folder + '/target').GetVal()
                bits[row, column] = root_file.Get(folder + '/bit').GetVal()
                vector = root_file.Get(folder + '/param')
                count = vector.GetNoElements()
                if count > MAX_PARAMETERS:
                    raise ValueError(f'{folder}: {count} parameters, tables hold {MAX_PARAMETERS}')
                for position in range(count):
                    params[row, column, position] = vector[position]
    np.savez_compressed(
        output_path,
        pt_edges=axis_edges(first.GetXaxis()), eta_edges=axis_edges(first.GetYaxis()),
        hits_edges=axis_edges(first.GetZaxis()), reference=reference,
        schema_ids=np.array(schema_ids, dtype=np.int32), schema_method=method, schema_target=target,
        schema_bits=bits, schema_params=params,
        source=np.array('github.com/cms-data/DataFormats-PatCandidates ' + input_path.split('/')[-1]))
    print(f'{output_path}: reference {reference.shape}, {len(schema_ids)} schemas')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
