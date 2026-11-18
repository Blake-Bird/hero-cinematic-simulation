"""Native black-hole seed and accretion ledgers, with conservation replay.

Snapshot ledgers and event journals describe the same source operations. Seed
extensive quantities are already physical CGS; accretion compensation values
also retain raw native masses so restart comparisons never invert a rounded
unit conversion. A pending accretion request is not transferred mass.
"""
import math
import numpy as np
from core.foundation import KPC_CM, MSUN_G


def source_mass_change_msun(row, *, physical=False):
    """Subtract native operands before the one linear unit conversion.

    Converting two large nearly equal masses independently can erase an actual
    one-ULP donor decrement. Native equal operands still return exactly zero;
    this helper never invents a transfer or changes its acceptance tolerance.
    """
    prefix = 'physical_mass' if physical else 'mass'
    before, after = prefix+'_before_native', prefix+'_after_native'
    if before in row and after in row and 'native_mass_unit_msun' in row:
        factor = row['native_mass_unit_msun']
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError('Positive finite original BH mass conversion required')
        return (row[after]-row[before])*factor
    return row[prefix+'_after_msun']-row[prefix+'_before_msun']


SEED_FIELDS = {
    'BHSeedReservoirVersion': (1, 'uint64'),
    'BHSeedAssembledMassGRAM': (1, 'g'),
    'BHSeedInternalEnergyERG': (1, 'erg'),
    'BHSeedMixingEnergyERG': (1, 'erg'),
    'BHSeedElementMassGRAM': (11, 'g'),
    'BHSeedOrbitalAngularMomentumCGS': (3, 'g cm^2/s'),
}

# HDF5 exponents refer to native length, mass and velocity units. Values
# themselves are physical CGS, with to_cgs=1 and no remaining a/h scaling.
SEED_DIMENSIONS = {
    'uint64': (0., 0., 0.), 'g': (0., 1., 0.),
    'erg': (0., 1., 2.), 'g cm^2/s': (1., 1., 1.),
}

def read_seed_ledger(group, count):
    """Missing legacy ledger is inactive; partial or lossy state is an error."""
    present = set(SEED_FIELDS).intersection(group)
    arrays = {name: np.zeros((count, width) if width > 1 else count,
                            dtype=np.uint64 if unit == 'uint64' else np.float64)
              for name, (width, unit) in SEED_FIELDS.items()}
    if not present:
        return arrays
    if present != set(SEED_FIELDS):
        raise ValueError('Partial native BH seed assembly ledger')
    for name, (width, unit) in SEED_FIELDS.items():
        dataset = group[name]
        expected = (count, width) if width > 1 else (count,)
        dtype = np.dtype('uint64' if unit == 'uint64' else 'float64')
        if dataset.shape != expected or dataset.dtype != dtype:
            raise ValueError('Lossy or misaligned BH seed assembly ledger: ' + name)
        dimensions = dict(zip(('length_scaling', 'mass_scaling', 'velocity_scaling'), SEED_DIMENSIONS[unit]))
        if any(key not in dataset.attrs or float(dataset.attrs[key]) != value
               for key, value in {'to_cgs': 1., 'a_scaling': 0., 'h_scaling': 0., **dimensions}.items()):
            raise ValueError('Incorrect physical CGS seed ledger units: ' + name)
        value = dataset[:]
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite BH seed assembly ledger: ' + name)
        arrays[name] = value
    version = arrays['BHSeedReservoirVersion']
    if np.any(version > 1):
        raise ValueError('Unsupported BH seed assembly ledger version')
    active = version == 1
    for name, (_, unit) in SEED_FIELDS.items():
        if name == 'BHSeedReservoirVersion':
            continue
        value = arrays[name]
        if np.any(value[~active] != 0):
            raise ValueError('Inactive BH seed ledger contains unversioned state')
        if name in {'BHSeedAssembledMassGRAM', 'BHSeedInternalEnergyERG', 'BHSeedElementMassGRAM'} and np.any(value < 0):
            raise ValueError('Negative extensive BH seed ledger state')
    mass = arrays['BHSeedAssembledMassGRAM']
    if np.any(mass[active] <= 0) or np.any(arrays['BHSeedElementMassGRAM'].sum(axis=1, dtype=np.longdouble) > mass):
        raise ValueError('Invalid BH seed assembly element/mass ledger')
    return arrays


SEED_EVENT_FIELDS = {
    'seed_donor': {'source_id', 'transaction_index', 'mass_before_native', 'mass_after_native',
        'velocity_before_native', 'velocity_after_native', 'internal_energy_before_native',
        'internal_energy_after_native', 'offset_proper_cm', 'element_mass_before_g', 'element_mass_after_g'},
    'seed_assembly': {'transaction_index', 'previous_species', 'mass_before_native', 'mass_after_native',
        'physical_mass_native', 'velocity_before_native', 'velocity_after_native', 'internal_energy_before_native',
        'captured_internal_energy_native', 'mixing_energy_native', 'captured_element_mass_g',
        'captured_orbital_angular_momentum_g_cm2_s', 'carrier_element_mass_before_g', 'donor_count',
        'maximum_donor_fraction', 'aperture_proper_kpc'},
}


def validate_seed_group(rows, tolerances):
    """Replay one native seed transaction under the existing hydro tolerances.

    Input rows already passed native schema/unit ingestion. Masses below use
    Msun, momenta Msun km/s; element masses, angular momentum and energies use
    physical g, g cm^2/s and erg. No cosmological a/h factor remains in CGS.
    """
    sinks = [row for row in rows if row['kind'] == 'seed_assembly']
    donors = [row for row in rows if row['kind'] == 'seed_donor']
    if len(sinks) != 1:
        return ['INCOMPLETE_BH_SEED_ASSEMBLY_GROUP']
    sink = sinks[0]
    if len(donors) != sink['donor_count'] or len({row['particle_id'] for row in donors}) != len(donors):
        return ['INCOMPLETE_BH_SEED_DONOR_CENSUS']
    if any(row['source_id'] != sink['particle_id']
           or row['transaction_index'] != sink['transaction_index']
           or row['native_time'] != sink['native_time']
           or row['counter'] != sink['counter']
           or row['particle_id'] == sink['particle_id'] for row in donors):
        return ['INCONSISTENT_BH_SEED_TRANSACTION_IDENTITY']

    absolute = tolerances['mass_absolute_msun']
    relative = tolerances['mass_relative']
    impulse = tolerances['momentum_absolute_msun_km_s']
    issues = []
    # Delta m = (m_before_native - m_after_native) * native_mass_to_Msun.
    # Reuse that same recorded transfer in mass, momentum, angular momentum
    # and energy. Subtracting converted endpoints can erase a one-ULP loss.
    transfers = [(-source_mass_change_msun(row), row) for row in donors]
    removed = math.fsum(mass for mass, _ in transfers)
    mass_scale = max(abs(sink['mass_before_msun']+removed), abs(sink['mass_after_msun']))
    if abs(source_mass_change_msun(sink)-removed) > absolute+relative*mass_scale:
        issues.append('BH_SEED_MASS_NOT_CONSERVED')

    for _, row in transfers:
        native_loss = row['mass_before_native']-row['mass_after_native']
        if not 0 < native_loss <= sink['maximum_donor_fraction']*row['mass_before_native']:
            issues.append('BH_SEED_DONOR_BOUND_EXCEEDED')
        if (row['velocity_before_km_s'] != row['velocity_after_km_s']
                or row['internal_energy_before_cm2_s2'] != row['internal_energy_after_cm2_s2']):
            issues.append('BH_SEED_DONOR_SPECIFIC_STATE_CHANGED')
        if np.linalg.norm(row['offset_proper_cm']) >= sink['aperture_proper_kpc']*KPC_CM:
            issues.append('BH_SEED_DONOR_OUTSIDE_APERTURE')
        before = np.asarray(row['element_mass_before_g'])
        after = np.asarray(row['element_mass_after_g'])
        # A donor retains its composition: each element scales with its
        # remaining mass fraction, calculated before any unit conversion.
        expected = before*(row['mass_after_native']/row['mass_before_native'])
        if np.any(np.abs(after-expected) > MSUN_G*(absolute+relative*row['mass_before_msun'])):
            issues.append('BH_SEED_DONOR_COMPOSITION_CHANGED')

    before_p = sink['mass_before_msun']*np.asarray(sink['velocity_before_km_s'])
    after_p = sink['mass_after_msun']*np.asarray(sink['velocity_after_km_s'])
    donor_p = sum((mass*np.asarray(row['velocity_before_km_s']) for mass, row in transfers), start=np.zeros(3))
    scale = sum(np.linalg.norm(value) for value in (before_p, after_p, donor_p))
    if np.linalg.norm(before_p+donor_p-after_p) > impulse+tolerances['momentum_relative']*scale:
        issues.append('BH_SEED_MOMENTUM_NOT_CONSERVED')

    elements = np.asarray(sink['carrier_element_mass_before_g']) + sum(
        (np.asarray(row['element_mass_before_g'])-np.asarray(row['element_mass_after_g']) for row in donors),
        start=np.zeros(11))
    captured = np.asarray(sink['captured_element_mass_g'])
    if np.any(np.abs(elements-captured) > MSUN_G*(absolute+relative*sink['mass_after_msun'])):
        issues.append('BH_SEED_ELEMENTS_NOT_CONSERVED')
    if np.sum(captured, dtype=np.longdouble) > sink['mass_after_msun']*MSUN_G:
        issues.append('BH_SEED_ELEMENTS_EXCEED_MASS')

    # L = sum r x (Delta m * v), about the recorded carrier origin.
    # Msun -> g and km/s -> cm/s give g cm^2/s; r is already proper cm.
    angular = sum((np.cross(np.asarray(row['offset_proper_cm']),
                           mass*MSUN_G*np.asarray(row['velocity_before_km_s'])*1.e5)
                   for mass, row in transfers), start=np.zeros(3))
    recorded = np.asarray(sink['captured_orbital_angular_momentum_g_cm2_s'])
    radius = max((float(np.linalg.norm(row['offset_proper_cm'])) for row in donors), default=0.)
    speed = max((float(np.linalg.norm(row['velocity_before_km_s'])) for row in [sink, *donors]), default=0.)
    angular_floor = MSUN_G*1.e5*radius*(impulse+absolute*speed)
    if np.linalg.norm(angular-recorded) > angular_floor+tolerances['momentum_relative']*(np.linalg.norm(angular)+np.linalg.norm(recorded)):
        issues.append('BH_SEED_ANGULAR_MOMENTUM_NOT_CONSERVED')

    # E_before = m_carrier*(u + v^2/2) + sum Delta m*(u_donor + v_donor^2/2).
    # E_after includes carrier kinetic energy and the separately recorded
    # internal/mixing reservoirs. Only velocity squared needs the 1e10 factor;
    # specific internal energy already uses cm^2/s^2.
    before_energy = sink['mass_before_msun']*MSUN_G*(sink['internal_energy_before_cm2_s2']
        + .5*float(np.dot(sink['velocity_before_km_s'], sink['velocity_before_km_s']))*1.e10)
    before_energy += math.fsum(mass*MSUN_G*(row['internal_energy_before_cm2_s2']
        + .5*float(np.dot(row['velocity_before_km_s'], row['velocity_before_km_s']))*1.e10)
        for mass, row in transfers)
    after_energy = (.5*sink['mass_after_msun']*MSUN_G
        * float(np.dot(sink['velocity_after_km_s'], sink['velocity_after_km_s']))*1.e10
        + sink['captured_internal_energy_erg'] + sink['mixing_energy_erg'])
    thermal = max(row['internal_energy_before_cm2_s2'] for row in [sink, *donors])
    floor = MSUN_G*(absolute*(thermal+.5*speed**2*1.e10)+impulse*speed*1.e10)
    if abs(after_energy-before_energy) > floor+relative*(abs(before_energy)+abs(after_energy)):
        issues.append('BH_SEED_ENERGY_NOT_CONSERVED')
    return sorted(set(issues))


ACCRETION_FIELDS = ('BHAccretionRoundoffVersion', 'BHAccretionPendingMass', 'BHAccretionPendingCorrection',
          'BHGravityMassCorrection', 'BHPhysicalMassCorrection')
ACCRETION_RAW_FIELDS = ('bh_roundoff_native_pending_mass', 'bh_roundoff_native_pending_correction',
              'bh_roundoff_native_gravity_correction', 'bh_roundoff_native_physical_correction')
ACCRETION_STATE_KEYS = ('pending_mass', 'pending_mass_correction', 'gravity_mass_correction', 'physical_mass_correction')
ACCRETION_INTEGRAL_FIELDS = {'version', 'requested_inflow_mass_native', 'applied_inflow_mass_native', 'retained_gain_mass_native',
    'mass_before_native', 'mass_after_native', 'physical_mass_before_native', 'physical_mass_after_native',
    *(key+'_'+side+'_native' for key in ACCRETION_STATE_KEYS for side in ('before', 'after'))}
ACCRETION_MERGE_FIELDS = {'version', 'parent_ids', *(key+'_'+side+'_native' for key in ACCRETION_STATE_KEYS for side in ('before', 'after'))}


def compensated_add(value, correction, increment):
    """Replay native Kahan addition: the represented value is value-correction.

    Corrections must survive source steps and mergers. Replacing this with an
    ordinary sum changes native bookkeeping even when physical mass rounds off.
    """
    effective = increment-correction
    after = value+effective
    return after, (after-value)-effective


def event_state(row, side, index=None):
    return tuple(row[key+'_'+side+'_native'] if index is None else row[key+'_'+side+'_native'][index] for key in ACCRETION_STATE_KEYS)


def merge_values(left, right):
    pending = compensated_add(left[0], left[1], right[0])
    pending = compensated_add(*pending, -right[1])
    return (*pending, left[2]+right[2], left[3]+right[3])


def validate_roundoff_event(row):
    if type(row['version']) is not int or row['version'] != 1:
        raise ValueError('Unsupported native accretion roundoff version')
    if row['kind'] == 'accretion_merge':
        left, right = event_state(row, 'before', 0), event_state(row, 'before', 1)
        if left[0]-left[1]<0 or right[0]-right[1]<0:
            raise ValueError('Native merger has negative pending accretion demand')
        expected = merge_values(left, right)
    else:
        before = event_state(row, 'before')
        requested, applied, gain = (row[key+'_native'] for key in (
            'requested_inflow_mass', 'applied_inflow_mass', 'retained_gain_mass'))
        demand = compensated_add(before[0], before[1], requested)
        if not 0 <= applied <= demand[0]-demand[1] or not 0 <= gain <= applied:
            raise ValueError('Native accretion integral overdraws represented demand')
        pending = compensated_add(*demand, -applied)
        if applied==0:
            gravity = (row['mass_before_native'], before[2])
            physical = (row['physical_mass_before_native'], before[3])
        else:
            gravity = compensated_add(row['mass_before_native'], before[2], gain)
            physical = compensated_add(row['physical_mass_before_native'], before[3], gain)
        if gravity[0] != row['mass_after_native'] or physical[0] != row['physical_mass_after_native']:
            raise ValueError('Native accretion compensated mass sum differs from journal')
        expected = (*pending, gravity[1], physical[1])
    after = event_state(row, 'after')
    if after != expected or not all(math.isfinite(value) for value in after) or after[0]-after[1] < 0:
        raise ValueError('Native accretion integral/state arithmetic does not replay exactly')


def allocate_roundoff_state(count):
    return {name: np.zeros(count, dtype=np.uint64 if name==ACCRETION_FIELDS[0] else float) for name in (*ACCRETION_FIELDS, *ACCRETION_RAW_FIELDS)}


def read_roundoff_state(group, count, *, parameters):
    """Load Msun analysis columns and untouched code-mass columns together."""
    result = allocate_roundoff_state(count)
    present = set(ACCRETION_FIELDS) & set(group)
    if not present:
        return result  # Version-zero legacy state is explicitly unavailable.
    if present != set(ACCRETION_FIELDS):
        raise ValueError('Partial native BH accretion integral snapshot state')
    factor = parameters['unit_mass_g']/MSUN_G/(parameters['hubble_param'] if parameters['comoving'] else 1.)
    for i, name in enumerate(ACCRETION_FIELDS):
        dataset = group[name]
        if dataset.shape != (count,) or dataset.dtype != np.dtype('uint64' if i==0 else 'float64'):
            raise ValueError('Invalid native BH accretion integral storage')
        value = dataset[:]
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite native BH accretion integral storage')
        expected = {'to_cgs': 1. if i==0 else parameters['unit_mass_g'], 'a_scaling': 0.,
                    'h_scaling': 0. if i==0 or not parameters['comoving'] else -1.}
        if any(key not in dataset.attrs or float(dataset.attrs[key]) != expected_value for key, expected_value in expected.items()):
            raise ValueError('Native BH accretion integral units differ')
        result[name] = value if i==0 else value*factor
        if i: result[ACCRETION_RAW_FIELDS[i-1]] = value
    version = result[ACCRETION_FIELDS[0]]
    active = version==1
    if np.any(version>1) or any(np.any(result[key][~active] != 0) for key in ACCRETION_RAW_FIELDS):
        raise ValueError('Invalid active/legacy BH accretion integral state')
    if np.any(result[ACCRETION_RAW_FIELDS[0]][active]-result[ACCRETION_RAW_FIELDS[1]][active] < 0):
        raise ValueError('Negative native pending accretion demand')
    return result


def reconcile_roundoff_state(before, after, records):
    """Compare raw native bits, avoiding inverse conversion of small residuals."""
    def census(data):
        if ACCRETION_FIELDS[0] not in data:
            return {}
        selected = (data['species']==5) & (data['mass_msun']>0) & (data[ACCRETION_FIELDS[0]]==1)
        return {int(data['particle_ids'][i]): tuple(float(data[key][i]) for key in ACCRETION_RAW_FIELDS)
                for i in np.flatnonzero(selected)}
    expected = census(before)
    observed = census(after)
    issues = []
    count = 0
    active_ids=set(expected)|set(observed)
    transactions={(r['native_time'],r['particle_id'],r['counter']):r for r in records if r['kind']=='accretion_integral'}
    mergers={(r['native_time'],r['particle_id'],r['counter']):r for r in records if r['kind']=='merger'}
    for row in records:
        key=(row['native_time'],row['particle_id'],row['counter'])
        if row['kind']=='accretion' and row['particle_id'] in active_ids and key not in transactions:
            issues.append({'kind':'BH_ROUNDOFF_MISSING_SOURCE_TRANSACTION','particle_id':row['particle_id']})
        if row['kind']=='accretion_merge' and (key not in mergers or mergers[key]['parent_ids']!=row['parent_ids']):
            issues.append({'kind':'BH_ROUNDOFF_MERGER_LACKS_NATIVE_ANCESTRY','particle_id':row['particle_id']})
    priority = {'seed': 0, 'seed_assembly': 0, 'accretion_integral': 1, 'accretion_merge': 2}
    for row in sorted((r for r in records if r['kind'] in priority),
            key=lambda r: (r['native_time'], priority[r['kind']], r['counter'], r['particle_id'])):
        kind = row['kind']
        pid = row['particle_id']
        if kind in ('seed', 'seed_assembly'):
            if pid in expected: issues.append({'kind': 'BH_ROUNDOFF_SEED_OVERWRITES_STATE', 'particle_id': pid})
            expected[pid] = (0., 0., 0., 0.)
            continue
        validate_roundoff_event(row)
        count += 1
        parents = row['parent_ids'] if kind=='accretion_merge' else [pid]
        for i, parent in enumerate(parents):
            state = event_state(row, 'before', i if kind=='accretion_merge' else None)
            if parent not in expected or expected[parent] != state:
                issues.append({'kind': 'BH_ROUNDOFF_SOURCE_HISTORY_MISMATCH', 'particle_id': parent, 'counter': row['counter']})
        for parent in parents: expected.pop(parent, None)
        expected[pid] = event_state(row, 'after')
    # Seed rows in old journals have no roundoff ledger. Leave those legacy
    # versions unavailable unless this interval actually has active state.
    active = bool(census(before) or observed or count)
    if active:
        for pid in set(expected)|set(observed):
            if expected.get(pid) != observed.get(pid):
                issues.append({'kind': 'BH_ROUNDOFF_SNAPSHOT_STATE_MISMATCH', 'particle_id': pid})
    return {'status': 'FAIL' if issues else 'PASS' if active else 'LEGACY_STATE_UNAVAILABLE',
            'event_count': count, 'issues': issues,
            'scope': 'Exact native numerical integral/compensation states; pending demand is not accreted mass or feedback'}
