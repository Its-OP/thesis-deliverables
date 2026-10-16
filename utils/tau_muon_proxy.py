import numpy as np

PION_MASS = 0.13957
MUON_MASS = 0.10566
B_MASS = 5.27966
DZ_MAX = 0.2
DR_MAX = 0.8

IDENTITY_COLUMNS = ['event_run', 'event_id', 'event_luminosity_block',
                    'source_batch_id', 'source_microbatch_id']
TRACK_COLUMNS = ['track_pt', 'track_eta', 'track_phi', 'track_charge', 'track_dz',
                 'track_label_from_tau']
MUON_COLUMNS = ['muon_pt', 'muon_eta', 'muon_phi', 'muon_charge', 'muon_dz', 'muon_soft_id']
SHARD_COLUMNS = IDENTITY_COLUMNS + TRACK_COLUMNS + MUON_COLUMNS
_EVENT_KEYS = {'track_label_from_tau': 'track_label', 'muon_soft_id': 'muon_soft'}
_INTEGER_COLUMNS = {'track_label_from_tau', 'muon_soft_id'}

# Each variant drops exactly one cut so the matching feature can be measured
# without the subset selection having been made on it.
PROXY_VARIANTS = {
    'full': dict(use_dz=True, use_dr=True, use_mass=True),
    'no_dz': dict(use_dz=False, use_dr=True, use_mass=True),
    'no_dr': dict(use_dz=True, use_dr=False, use_mass=True),
    'no_mass': dict(use_dz=True, use_dr=True, use_mass=False),
}
FEATURE_NAMES = ('delta_r', 'dz_gap', 'mass', 'charge_product')


def list_parts(chunked_column):
    """chunked_column: pyarrow ChunkedArray of list type. Returns (values, offsets) numpy arrays;
    event i spans values[offsets[i]:offsets[i + 1]]."""
    array = chunked_column.combine_chunks()
    return np.asarray(array.values), np.asarray(array.offsets)


def events_from_table(table):
    """table: pyarrow Table carrying SHARD_COLUMNS. Returns one dict per event with
    'identity' (5-tuple) and numpy arrays track_pt/eta/phi/charge/dz/label, muon_pt/eta/phi/charge/dz/soft."""
    identity = np.stack([table.column(name).to_numpy() for name in IDENTITY_COLUMNS], axis=1)
    parts = {}
    for name in TRACK_COLUMNS + MUON_COLUMNS:
        values, offsets = list_parts(table.column(name))
        dtype = np.int64 if name in _INTEGER_COLUMNS else np.float64
        parts[name] = (values.astype(dtype, copy=False), offsets)
    events = []
    for row in range(table.num_rows):
        event = {'identity': tuple(int(value) for value in identity[row])}
        for name, (values, offsets) in parts.items():
            event[_EVENT_KEYS.get(name, name)] = values[offsets[row]:offsets[row + 1]]
        events.append(event)
    return events


def lorentz_components(pt, eta, phi, mass):
    """pt, eta, phi: broadcastable arrays; mass: scalar. Returns px, py, pz, energy."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(pt ** 2 * np.cosh(eta) ** 2 + mass ** 2)
    return px, py, pz, energy


def wrap_delta_phi(delta_phi):
    return (delta_phi + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta_a, phi_a, eta_b, phi_b):
    return np.sqrt((eta_a - eta_b) ** 2 + wrap_delta_phi(phi_a - phi_b) ** 2)


def gt_members(track_label):
    """track_label: (N,). Returns indices of the GT tracks."""
    return np.flatnonzero(np.asarray(track_label) > 0)


def system_from_members(event, members):
    """members: (C, m) track indices. Returns dict of (C,) arrays px, py, pz, energy, eta, phi,
    charge (sum) and dz (median over members) for each candidate system under the pion mass."""
    members = np.asarray(members, dtype=np.int64)
    px, py, pz, energy = lorentz_components(
        event['track_pt'][members], event['track_eta'][members], event['track_phi'][members], PION_MASS)
    px, py, pz, energy = (component.sum(axis=1) for component in (px, py, pz, energy))
    momentum = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    cosine = np.clip(pz / np.maximum(momentum, 1e-12), -1.0 + 1e-12, 1.0 - 1e-12)
    return dict(px=px, py=py, pz=pz, energy=energy, eta=np.arctanh(cosine), phi=np.arctan2(py, px),
                charge=event['track_charge'][members].sum(axis=1),
                dz=np.median(event['track_dz'][members], axis=1))


def muon_relations(event, system):
    """system: dict of (C,) arrays from system_from_members. Returns dict of (C, K) arrays:
    delta_r, mass (muon + system), dz_gap, charge_product (muon charge x system charge)."""
    mu_px, mu_py, mu_pz, mu_energy = lorentz_components(
        event['muon_pt'], event['muon_eta'], event['muon_phi'], MUON_MASS)
    total_energy = system['energy'][:, None] + mu_energy[None, :]
    total_px = system['px'][:, None] + mu_px[None, :]
    total_py = system['py'][:, None] + mu_py[None, :]
    total_pz = system['pz'][:, None] + mu_pz[None, :]
    mass = np.sqrt(np.maximum(total_energy ** 2 - total_px ** 2 - total_py ** 2 - total_pz ** 2, 0.0))
    return dict(
        delta_r=delta_r(event['muon_eta'][None, :], event['muon_phi'][None, :],
                        system['eta'][:, None], system['phi'][:, None]),
        mass=mass,
        dz_gap=np.abs(event['muon_dz'][None, :] - system['dz'][:, None]),
        charge_product=event['muon_charge'][None, :] * system['charge'][:, None])


def gt_relations(event):
    """Returns muon relations (1, K) to the event's GT three-track system."""
    return muon_relations(event, system_from_members(event, gt_members(event['track_label'])[None, :]))


def select_proxy_muon(relations, soft, variant='full'):
    """relations: (1, K) dict from gt_relations; soft: (K,). Returns the proxy tau-muon index or -1."""
    flags = PROXY_VARIANTS[variant]
    mask = (np.asarray(soft) == 1) & (relations['charge_product'][0] < 0)
    if flags['use_dz']:
        mask &= relations['dz_gap'][0] < DZ_MAX
    if flags['use_dr']:
        mask &= relations['delta_r'][0] < DR_MAX
    if flags['use_mass']:
        mask &= relations['mass'][0] < B_MASS
    if not mask.any():
        return -1
    ordering = relations['dz_gap'][0] if flags['use_dz'] else relations['delta_r'][0]
    return int(np.argmin(np.where(mask, ordering, np.inf)))


def policy_indices(relations, soft, policy, oracle_index=-1):
    """relations: (C, K) dict; soft: (K,). Returns (C,) muon indices per candidate, -1 where none
    qualifies. Realistic policies restrict to soft muons whose charge is not same-sign to the candidate."""
    n_candidates, n_muons = relations['delta_r'].shape
    if policy == 'oracle':
        return np.full(n_candidates, oracle_index, dtype=np.int64)
    if policy == 'min_dr':
        key = relations['delta_r']
    elif policy == 'min_dz':
        key = relations['dz_gap']
    else:
        raise ValueError(f'unknown muon policy: {policy}')
    if n_muons == 0:
        return np.full(n_candidates, -1, dtype=np.int64)
    eligible = (np.asarray(soft) == 1)[None, :] & (relations['charge_product'] <= 0)
    masked = np.where(eligible, key, np.inf)
    indices = np.argmin(masked, axis=1)
    indices[~np.isfinite(masked[np.arange(n_candidates), indices])] = -1
    return indices


def gather(relations, indices):
    """indices: (C,) muon index per candidate (-1 = none). Returns dict of (C,) feature arrays, NaN where none."""
    indices = np.asarray(indices)
    valid = indices >= 0
    rows = np.arange(len(indices))
    features = {}
    for name in FEATURE_NAMES:
        values = np.full(len(indices), np.nan)
        values[valid] = relations[name][rows[valid], indices[valid]]
        features[name] = values
    return features
