"""Compare retained native states before and after restart.

Physical observables carry explicit units and tolerances. Raw native members
retain their declared layouts and code units. Integer state and identities
compare exactly, including retired black holes and committed lifecycle events.
These readers compare saved data; they launch no simulations.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import re
from core.foundation import file_sha256, atomic_json, canonical_digest
from pathlib import Path

# Restart hydro particle state


def _layout(reference, schema):
    """Read a selected native layout from shared defaults or a standalone file.

    The file digest binds the input bytes. The selected record's canonical
    digest must still match the native snapshot header: combining files does
    not change field order, integer encoding, native units or feature guards.
    """
    path = Path(reference['path']).resolve()
    # Hash the exact bounded bytes we decode, rather than reopening the file
    # after checking a different read that could have changed in between.
    with path.open('rb') as stream:
        content = stream.read(1024**2 + 1)
    if len(content) > 1024**2 or hashlib.sha256(content).hexdigest() != reference['sha256']:
        raise ValueError('Frozen native restart layout changed or exceeds metadata limit')
    record = json.loads(content)
    if not isinstance(record, dict):
        raise ValueError('Native restart layout must be a JSON object')
    if record.get('schema') == 'native-restart-layouts-v1':
        layouts = record.get('layouts')
        if not isinstance(layouts, dict) or schema not in layouts:
            raise ValueError('Shared restart layouts omit the requested native state')
        record = layouts[schema]
        if not isinstance(record, dict):
            raise ValueError('Selected native restart layout must be a JSON object')
    body = {key: value for key, value in record.items() if key != 'layout_sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if record.get('schema') != schema or record.get('layout_sha256') != digest:
        raise ValueError('Native restart layout schema or canonical fingerprint differs')
    return record, {str(path): reference['sha256']}


def _float_difference(a, b, limits):
    """Return |b-a| and atol + rtol*max(|a|,|b|), in the input field's units.

    Tolerances are validated by the request reader. Overflow cannot establish
    agreement: inf <= inf would otherwise accept an unrepresentable contrast.
    Integer identities and counters never pass through this floating path.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        difference = np.abs(b-a)
        allowed = limits['absolute'] + limits['relative']*np.maximum(np.abs(a), np.abs(b))
    if not np.isfinite(difference).all() or not np.isfinite(allowed).all():
        raise ValueError('Native restart difference or tolerance arithmetic overflow')
    return difference, allowed


def _text(value):
    return value.decode('ascii') if isinstance(value, bytes) else str(value)


def compare_hydro_particle_state(request, sides, *, species, maximum_particles, memory_gib,
                                source_layout=None):
    from core.native import original_native_snapshot_files
    from core.data import verify_native_hdf5_set, snapshot_total_counts, native_publication_markers
    import h5py
    if len(sides) != 2:
        raise ValueError('Restart state comparison requires two retained native outputs')
    layout, sources = _layout(request['layout'], 'native-restart-hydro-particle-layout-v1')
    features = request['compiled_features']
    if not isinstance(features, list) or any(not isinstance(value, str) for value in features) or len(set(features)) != len(features):
        raise ValueError('Explicit distinct compiled-feature names required')
    fields = [row for row in layout['fields'] if row['feature'] is None or row['feature'] in features]
    names = [row['dataset'] for row in fields]
    if not fields or len(set(names)) != len(names):
        raise ValueError('Native raw-state layout requires distinct named fields')
    required_species = {row['species'] for row in fields}
    if not required_species <= set(species):
        raise ValueError('Raw native restart comparison cannot omit a compiled source species')
    tolerances = request['float_tolerances']
    float_names = {row['dataset'] for row in fields if row['memory_type'] != 'MEM_UINT64'}
    if set(tolerances) != float_names:
        raise ValueError('Every active floating native raw-state field needs its own frozen tolerance')
    for limits in tolerances.values():
        if set(limits) != {'absolute', 'relative'} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in limits.values()):
            raise ValueError('Finite nonnegative raw native-unit tolerances required')
    block = request['read_chunk_particles']
    if type(block) is not int or block < 1 or block > maximum_particles:
        raise ValueError('Explicit bounded native diagnostic read chunk required')
    retirement_index = None
    capsule_layout = None
    if 5 in required_species:
        if source_layout is None:
            raise ValueError('BH raw state requires a pinned typed capsule layout for retired identity')
        capsule_layout, pins = _layout(source_layout, 'native-restart-source-capsule-layout-v1')
        sources.update(pins)
        retirement_index = capsule_layout['integer_columns'].index('bh_retired_particle_id')
    retained = []
    identity_bytes = 0
    for side in sides:
        files = original_native_snapshot_files(side['run_directory'], side['snapshot'])
        verify_native_hdf5_set(files, require_publication=True)
        sources.update({str(path.resolve()): file_sha256(path)
                        for path in [*files, *native_publication_markers(files)]})
        with h5py.File(files[0], 'r') as handle:
            counts = snapshot_total_counts(handle['Header'].attrs)
        if sum(int(counts[kind]) for kind in species) > maximum_particles:
            raise MemoryError('Raw native restart population exceeds frozen particle bound')
        domains = {}
        for kind in required_species:
            count = int(counts[kind])
            # IDs, sorting permutation, retired identity and comparison workspaces.
            identity_bytes += count * 48
            if identity_bytes > memory_gib * 1024**3 / 4:
                raise MemoryError('Raw native identity workspace exceeds declared memory')
            ids = np.empty(count, dtype=np.uint64)
            cursor = 0
            segments = []
            for path in files:
                with h5py.File(path, 'r') as handle:
                    header = handle['Header'].attrs
                    active = _text(header.get('ResearchHydroParticleActiveFields', '')).split(';')
                    if active[-1:] == ['']:
                        active.pop()
                    if _text(header.get('ResearchHydroParticleLayoutSHA256', '')) != layout['layout_sha256'] or active != names:
                        raise ValueError('Snapshot raw-state feature set differs from frozen compiled profile')
                    group = handle.get('PartType' + str(kind))
                    if group is None:
                        continue
                    if 'ParticleIDs' not in group:
                        raise ValueError('Native raw-state group has no exact identity dataset')
                    dataset = group['ParticleIDs']
                    if dataset.ndim != 1 or dataset.dtype.kind != 'u' or dataset.dtype.itemsize != 8:
                        raise ValueError('Raw native restart identity requires uint64 storage')
                    n = len(dataset)
                    if cursor + n > count:
                        raise ValueError('Raw native state exceeds header census')
                    for row in fields:
                        if row['species'] != kind:
                            continue
                        data = group.get(row['dataset'])
                        expected = (n,) if row['components'] == 1 else (n, row['components'])
                        storage_kind = 'u' if row['memory_type'] == 'MEM_UINT64' else 'f'
                        if data is None or data.shape != expected or data.dtype.kind != storage_kind or data.dtype.itemsize != 8:
                            raise ValueError('Raw native state field absent or not exact ' + ('uint64' if storage_kind == 'u' else 'float64') + ': ' + row['dataset'])
                    for begin in range(0, n, block):
                        end = min(begin + block, n)
                        local = dataset[begin:end]
                        dead = local == 0
                        if kind == 5:
                            if _text(header.get('ResearchSourceCapsuleLayoutSHA256', '')) != capsule_layout['layout_sha256']:
                                raise ValueError('BH raw state lacks matching typed lifecycle layout')
                            lane_data = group.get('ResearchSourceStateUInt64Lanes')
                            if lane_data is None or lane_data.shape != (n, 2 * len(capsule_layout['integer_columns'])) or lane_data.dtype.kind != 'i' or lane_data.dtype.itemsize != 4:
                                raise ValueError('BH raw state lacks exact retirement identity lanes')
                            lanes = lane_data[begin:end, 2*retirement_index:2*retirement_index+2].astype(np.uint32).astype(np.uint64)
                            retired = lanes[:, 0] | (lanes[:, 1] << np.uint64(32))
                            if np.any(dead & (retired == 0)) or np.any((~dead) & (retired != 0)):
                                raise ValueError('Ambiguous BH raw-state retirement identity')
                            local = np.where(dead, retired, local)
                        elif dead.any():
                            raise ValueError('Gas raw state contains a zero identity')
                        ids[cursor+begin:cursor+end] = local
                    segments.append((path, cursor, n))
                    cursor += n
            if cursor != count:
                raise ValueError('Raw native restart identity census incomplete')
            order = np.argsort(ids)
            sorted_ids = ids[order]
            if count > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError('Raw native restart identities repeat across snapshot shards')
            domains[kind] = {'ids': sorted_ids, 'order': order, 'segments': segments, 'count': count}
        retained.append(domains)
    for kind in required_species:
        if not np.array_equal(retained[0][kind]['ids'], retained[1][kind]['ids']):
            raise ValueError('Raw native gas/BH lifecycle population differs after restart')
    rows = []
    for field in fields:
        kind = field['species']
        count = retained[0][kind]['count']
        columns = field['components']
        # Two retained values plus sorting, abs differences, thresholds and masks.
        scratch = count * columns * 80 + min(block, max(count, 1)) * columns * 16
        if identity_bytes + scratch > memory_gib * 1024**3:
            raise MemoryError('Raw native field comparison exceeds cumulative workspace')
        values = []
        for domains in retained:
            domain = domains[kind]
            integer = field['memory_type'] == 'MEM_UINT64'
            array = np.empty((count, columns), dtype=np.uint64 if integer else np.float64)
            for path, offset, n in domain['segments']:
                with h5py.File(path, 'r') as handle:
                    dataset = handle['PartType' + str(kind)][field['dataset']]
                    for begin in range(0, n, block):
                        end = min(begin + block, n)
                        array[offset+begin:offset+end] = dataset[begin:end].reshape(end-begin, columns)
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite raw native restart member: ' + field['dataset'])
            values.append(array[domain['order']])
        a, b = values
        limits = None if integer else tolerances[field['dataset']]
        if integer:
            # Ordered subtraction avoids unsigned wrap; never round through float64.
            difference = np.maximum(a, b) - np.minimum(a, b)
            allowed = 0
            failed = a != b
        else:
            difference, allowed = _float_difference(a, b, limits)
            failed = difference > allowed
        rows.append({'field': field['dataset'], 'native_member': field['member'], 'species': kind,
                     'status': 'FAIL' if failed.any() else 'PASS', 'failed_entries': int(failed.sum()),
                     'particle_count': count, 'population_measured': count > 0,
                     'maximum_absolute_difference_native': (int(difference.max()) if integer else float(difference.max())) if count else None,
                     'units': field['units'], 'tolerance': limits,
                     'comparison': 'raw named member; uint64 identity; ' + ('exact uint64 equality' if integer else 'float64 export with frozen tolerance')})
        del values, array, a, b, difference, allowed, failed
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    for path, digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Raw native restart input changed during comparison')
    return {'fields': rows, 'sources': sources, 'layout_sha256': layout['layout_sha256'],
            'compiled_features': features, 'species_particle_counts': {
                str(kind): retained[0][kind]['count'] for kind in sorted(required_species)},
            'scope': layout['scope'], 'empty_population_policy': 'zero entries do not establish physical population coverage'}


# Restart integrator capsule


def _compiled_condition(expression, features):
    """Parse the catalog's small C-preprocessor boolean grammar; never eval it."""
    pattern = r'defined\([A-Z][A-Z0-9_]*\)|[A-Z][A-Z0-9_]*|&&|\|\||!|\(|\)|1|0'
    tokens = re.findall(pattern, expression)
    if ''.join(tokens) != re.sub(r'\s+', '', expression):
        raise ValueError('Unsupported native layout preprocessor expression')
    cursor = 0

    def atom():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError('Incomplete native layout condition')
        token = tokens[cursor]
        cursor += 1
        if token == '!':
            return not atom()
        if token == '(':
            result = disjunction()
            if cursor >= len(tokens) or tokens[cursor] != ')':
                raise ValueError('Unclosed native layout condition')
            cursor += 1
            return result
        if token == '1':
            return True
        if token == '0':
            return False
        if token.startswith('defined('):
            return token[8:-1] in features
        if re.fullmatch(r'[A-Z][A-Z0-9_]*', token):
            return token in features
        raise ValueError('Unexpected native layout condition token')

    def conjunction():
        nonlocal cursor
        result = atom()
        while cursor < len(tokens) and tokens[cursor] == '&&':
            cursor += 1
            other = atom()
            result = result and other
        return result

    def disjunction():
        nonlocal cursor
        result = conjunction()
        while cursor < len(tokens) and tokens[cursor] == '||':
            cursor += 1
            other = conjunction()
            result = result or other
        return result

    value = disjunction()
    if cursor != len(tokens):
        raise ValueError('Trailing native layout condition tokens')
    return value


def compare_integrator_capsules(request, sides, *, species, maximum_particles, memory_gib,
                                source_layout):
    from core.native import original_native_snapshot_files
    from core.data import verify_native_hdf5_set, snapshot_total_counts, native_publication_markers
    import h5py
    layout, sources = _layout(request['layout'], 'native-restart-integrator-capsule-layout-v1')
    identity_layout, identity_sources = _layout(source_layout, 'native-restart-source-capsule-layout-v1')
    sources.update(identity_sources)
    features = request['compiled_features']
    if not isinstance(features, list) or any(not isinstance(value, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', value) for value in features) or len(features) != len(set(features)):
        raise ValueError('Distinct native compiled feature names required')
    floating = [row['name'] for row in layout['float_columns']
                if _compiled_condition(row['condition'], set(features))]
    integer = layout['integer_columns']
    coordinates = layout['coordinate_lanes']
    if not floating or len(coordinates) != 12 or not integer or len(set(floating + integer + coordinates)) != len(floating + integer + coordinates):
        raise ValueError('Native integrator capsule needs a distinct complete named layout')
    bits = request['coordinate_bits']
    if type(bits) is not int or bits not in {32, 64, 128}:
        raise ValueError('Frozen integer coordinate width must be 32, 64 or 128')
    block = request['read_chunk_particles']
    if type(block) is not int or block < 1 or block > maximum_particles:
        raise ValueError('Explicit bounded integrator-capsule read chunk required')
    tolerances = request['float_tolerances']
    if set(tolerances) != set(floating):
        raise ValueError('Every active native integration member needs a frozen native-unit tolerance')
    for limits in tolerances.values():
        if set(limits) != {'absolute', 'relative'} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in limits.values()):
            raise ValueError('Finite nonnegative native integration tolerances required')
    if len(sides) != 2:
        raise ValueError('Exactly two native integration snapshots required')
    retirement_index = identity_layout['integer_columns'].index('bh_retired_particle_id')
    payloads = []
    storage_lanes = len(coordinates) + 2*len(integer)
    cumulative_bytes = 0
    for side in sides:
        files = original_native_snapshot_files(side['run_directory'], side['snapshot'])
        verify_native_hdf5_set(files, require_publication=True)
        sources.update({str(path.resolve()): file_sha256(path)
                        for path in [*files, *native_publication_markers(files)]})
        with h5py.File(files[0], 'r') as handle:
            count = sum(int(snapshot_total_counts(handle['Header'].attrs)[kind]) for kind in species)
        if count > maximum_particles:
            raise MemoryError('Native integration snapshot exceeds declared population bound')
        # Includes permutation, sorted copies, bit-lane conversion and float comparisons.
        cumulative_bytes += count*(64 + 32*(len(floating) + len(integer) + len(coordinates)))
        chunk_bytes = min(block, max(count, 1))*(32*storage_lanes + 32*len(floating) + 128)
        if cumulative_bytes + chunk_bytes > memory_gib*1024**3:
            raise MemoryError('Native integrator comparison exceeds cumulative workspace')
        ids = np.empty(count, dtype=np.uint64)
        kinds = np.empty(count, dtype=np.int8)
        retired_flags = np.empty(count, dtype=bool)
        floats = np.empty((count, len(floating)), dtype=np.float64)
        raw_coordinates = np.empty((count, len(coordinates)), dtype=np.uint32)
        integers = np.empty((count, len(integer)), dtype=np.uint64)
        cursor = 0
        for path in files:
            with h5py.File(path, 'r') as handle:
                header = handle['Header'].attrs
                names = _text(header.get('ResearchIntegratorActiveFloatColumns', '')).split(';')
                if names[-1:] == ['']:
                    names.pop()
                if _text(header.get('ResearchIntegratorLayoutSHA256', '')) != layout['layout_sha256'] or names != floating or header.get('ResearchIntegratorCoordinateBits') != bits:
                    raise ValueError('Native integration layout/features/coordinate width differ from frozen request')
                if _text(header.get('ResearchSourceCapsuleLayoutSHA256', '')) != identity_layout['layout_sha256']:
                    raise ValueError('Native integration snapshot lacks its frozen lifecycle identity layout')
                for kind in species:
                    group = handle.get('PartType'+str(kind))
                    if group is None:
                        continue
                    identity = group.get('ParticleIDs')
                    if identity is None or identity.ndim != 1 or identity.dtype.kind != 'u' or identity.dtype.itemsize != 8:
                        raise ValueError('Native integrator identity must use exact uint64 storage')
                    n = len(identity)
                    for name, shape, dtype, itemsize in (
                            ('ResearchIntegratorFloat64', (n, len(floating)), 'f', 8),
                            ('ResearchIntegratorInt32Lanes', (n, storage_lanes), 'i', 4),
                            ('ResearchSourceStateUInt64Lanes', (n, 2*len(identity_layout['integer_columns'])), 'i', 4)):
                        dataset = group.get(name)
                        if dataset is None or dataset.shape != shape or dataset.dtype.kind != dtype or dataset.dtype.itemsize != itemsize:
                            raise ValueError('Native integrator diagnostic has another shape/type: '+name)
                    if cursor+n > count:
                        raise ValueError('Native integrator census exceeds snapshot header')
                    for begin in range(0, n, block):
                        end = min(begin+block, n)
                        destination = slice(cursor+begin, cursor+end)
                        local = identity[begin:end]
                        dead = local == 0
                        identity_lanes = group['ResearchSourceStateUInt64Lanes'][begin:end, 2*retirement_index:2*retirement_index+2].astype(np.uint32).astype(np.uint64)
                        retired_id = identity_lanes[:, 0] | (identity_lanes[:, 1] << np.uint64(32))
                        if np.any(dead & ((kind != 5) | (retired_id == 0))) or np.any((~dead) & (retired_id != 0)):
                            raise ValueError('Ambiguous native integrator retirement identity')
                        ids[destination] = np.where(dead, retired_id, local)
                        kinds[destination] = kind
                        retired_flags[destination] = dead
                        floats[destination] = group['ResearchIntegratorFloat64'][begin:end]
                        lanes = group['ResearchIntegratorInt32Lanes'][begin:end].astype(np.uint32)
                        raw_coordinates[destination] = lanes[:, :12]
                        pairs = lanes[:, 12:].astype(np.uint64)
                        integers[destination] = pairs[:, 0::2] | (pairs[:, 1::2] << np.uint64(32))
                        if not np.isfinite(floats[destination]).all():
                            raise ValueError('Nonfinite native particle integration state')
                        if np.any(integers[destination, integer.index('Type')] != kind):
                            raise ValueError('Native particle type disagrees with snapshot species group')
                        for axis in range(3):
                            if np.any(raw_coordinates[destination, axis*4+bits//32:(axis+1)*4] != 0):
                                raise ValueError('Nonzero unused native coordinate lanes')
                    cursor += n
        if cursor != count:
            raise ValueError('Native integration state lacks complete snapshot census')
        order = np.argsort(ids)
        ordered_ids = ids[order]
        if count > 1 and np.any(ordered_ids[1:] == ordered_ids[:-1]):
            raise ValueError('Native integration identities repeat across species/shards')
        payloads.append((ordered_ids, kinds[order], retired_flags[order], floats[order], raw_coordinates[order], integers[order]))
    left, right = payloads
    if any(not np.array_equal(left[index], right[index]) for index in range(3)):
        raise ValueError('Native resumed integration identity/species/retirement population differs')
    rows = []
    for axis in range(3):
        failed = left[4][:, axis*4:(axis+1)*4] != right[4][:, axis*4:(axis+1)*4]
        rows.append({'field': 'particle.IntPos['+str(axis)+']', 'status': 'FAIL' if failed.any() else 'PASS',
                     'failed_lanes': int(failed.sum()), 'failed_particles': int(np.any(failed, axis=1).sum()),
                     'comparison': 'exact '+str(bits)+'-bit unsigned coordinate; four low-first uint32 lanes'})
    for index, name in enumerate(integer):
        failed = left[5][:, index] != right[5][:, index]
        rows.append({'field': 'particle.'+name, 'status': 'FAIL' if failed.any() else 'PASS',
                     'failed_entries': int(failed.sum()), 'comparison': 'exact integer modulo 2^64; no floating conversion'})
    for index, name in enumerate(floating):
        a, b = left[3][:, index], right[3][:, index]
        limits = tolerances[name]
        difference, allowed = _float_difference(a, b, limits)
        failed = difference > allowed
        rows.append({'field': 'particle.'+name, 'status': 'FAIL' if failed.any() else 'PASS',
                     'failed_entries': int(failed.sum()), 'tolerance': limits,
                     'maximum_absolute_difference_native': float(difference.max()) if len(a) else None,
                     'comparison': 'raw native member exported as float64'})
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    for path, digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Native integrator diagnostic changed during comparison')
    return {'fields': rows, 'sources': sources, 'layout_sha256': layout['layout_sha256'],
            'coordinate_bits': bits, 'compiled_features': features,
            'particle_count_including_retired': len(left[0]), 'retired_bh_count': int(left[2].sum()),
            'scope': layout['scope'], 'operational_integer_note': 'GravCost and TreeID are compared exactly; differing decomposition/rearrangement can legitimately fail this strict state comparison'}


# Restart source capsule


def compare_source_capsules(request,sides,*,species,maximum_particles,memory_gib):
    from core.native import original_native_snapshot_files
    from core.data import verify_native_hdf5_set,snapshot_total_counts,exact_uint64_ids,native_publication_markers
    import h5py
    layout, sources = _layout(request['layout'], 'native-restart-source-capsule-layout-v1')
    expected = layout['layout_sha256']
    floating=layout['float_columns'];integer=layout['integer_columns']
    if not floating or not integer or len(set(floating+integer))!=len(floating+integer):raise ValueError('Distinct complete typed source layout required')
    tolerance=request['float_tolerances']
    if set(tolerance)!=set(floating):raise ValueError('Every floating source member needs an explicit native-unit restart tolerance')
    for limits in tolerance.values():
        if set(limits)!={'absolute','relative'} or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 for value in limits.values()):
            raise ValueError('Explicit finite nonnegative typed source-state tolerances required')
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    if len(sides) != 2:
        raise ValueError('Exactly two retained source-state snapshots required')
    payloads=[]
    for side in sides:
        files=original_native_snapshot_files(side['run_directory'],side['snapshot'])
        verify_native_hdf5_set(files,require_publication=True)
        sources.update({str(path.resolve()):file_sha256(path) for path in [*files,*native_publication_markers(files)]})
        with h5py.File(files[0],'r') as file:count=sum(int(snapshot_total_counts(file['Header'].attrs)[kind]) for kind in species)
        per_particle=32+8*(len(floating)+len(integer))
        if count>maximum_particles or count*per_particle*8>memory_gib*1024**3/2:
            raise MemoryError('Complete typed native source-state comparison exceeds cumulative workspace')
        ids=np.empty(count,dtype=np.uint64);kinds=np.empty(count,dtype=np.int8);retired=np.empty(count,dtype=bool)
        floats=np.empty((count,len(floating)),dtype=np.float64);integers=np.empty((count,len(integer)),dtype=np.uint64)
        cursor=0
        for path in files:
            with h5py.File(path,'r') as file:
                fingerprint=file['Header'].attrs.get('ResearchSourceCapsuleLayoutSHA256')
                if isinstance(fingerprint,bytes):fingerprint=fingerprint.decode('ascii')
                if fingerprint!=expected:raise ValueError('Native snapshot does not contain the frozen complete typed source layout')
                for kind in species:
                    group=file.get('PartType'+str(kind))
                    if group is None:continue
                    local=exact_uint64_ids(group['ParticleIDs'][:],unique=False)
                    n=len(local)
                    for name,shape,dtype in (('ResearchSourceStateFloat64',(n,len(floating)),'f'),
                                             ('ResearchSourceStateUInt64Lanes',(n,2*len(integer)),'i')):
                        if name not in group or group[name].shape!=shape or group[name].dtype.kind!=dtype or group[name].dtype.itemsize!=(8 if dtype=='f' else 4):
                            raise ValueError('Native source capsule shape or exact storage type differs from its typed layout')
                    stop=cursor+n
                    if stop>count:raise ValueError('Native source capsule exceeds complete header identity census')
                    # Read blocks bounded independently of the full retained comparison arrays.
                    for begin in range(0,n,4096):
                        end=min(begin+4096,n);destination=slice(cursor+begin,cursor+end)
                        floats[destination]=group['ResearchSourceStateFloat64'][begin:end]
                        lanes=group['ResearchSourceStateUInt64Lanes'][begin:end].astype(np.uint32).astype(np.uint64)
                        integers[destination]=lanes[:,0::2]|(lanes[:,1::2]<<np.uint64(32))
                    if not np.isfinite(floats[cursor:stop]).all():raise ValueError('Native source capsule contains nonfinite source state')
                    retirement=integers[cursor:stop,integer.index('bh_retired_particle_id')]
                    dead=local==0
                    if np.any(dead & (kind!=5)) or np.any(dead & (retirement==0)) or np.any((~dead)&(retirement!=0)):
                        raise ValueError('Native retirement must preserve original BH identity; ambiguous legacy zero-ID tombstones unsupported')
                    ids[cursor:stop]=np.where(dead,retirement,local);kinds[cursor:stop]=kind;retired[cursor:stop]=dead;cursor=stop
        if cursor!=count or len(np.unique(ids))!=count:raise ValueError('Native source capsule identity census missing or aliased')
        order=np.argsort(ids)
        payloads.append((ids[order],kinds[order],retired[order],floats[order],integers[order]))
    left,right=payloads
    if any(not np.array_equal(left[index],right[index]) for index in range(3)):
        raise ValueError('Resumed native source-state identity/species/retirement lifecycle differs')
    rows=[]
    for index,name in enumerate(integer):
        difference=left[4][:,index]!=right[4][:,index]
        rows.append({'field':'research_source_state.'+name,'status':'FAIL' if difference.any() else 'PASS',
            'failed_entries':int(difference.sum()),'comparison':'exact logical uint64; low/high storage lanes reconstructed without floating conversion'})
    for index,name in enumerate(floating):
        a,b=left[3][:,index],right[3][:,index];limits=tolerance[name]
        difference, allowed = _float_difference(a, b, limits)
        failed=difference>allowed
        rows.append({'field':'research_source_state.'+name,'status':'FAIL' if failed.any() else 'PASS','failed_entries':int(failed.sum()),
            'maximum_absolute_difference_native':float(difference.max()) if count else 0.,'tolerance':limits})
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Typed restart source capsule changed during comparison')
    return {'fields':rows,'sources':sources,'particle_count_including_retired':len(left[0]),
        'retired_bh_count':int(left[2].sum()),'layout_sha256':expected,
        'scope':'all declared typed research_source_state fields for selected native species, including retained BH retirement identities; no struct padding'}


# Restart global state


def compare_global_state(request, sides):
    from core.native import original_native_snapshot_files
    from core.data import verify_native_hdf5_set, native_publication_markers
    import h5py
    layout, sources = _layout(request['layout'], 'native-restart-global-state-layout-v1')
    features = request['compiled_features']
    if not isinstance(features, list) or len(set(features)) != len(features) or any(
            not isinstance(value, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', value) for value in features):
        raise ValueError('Distinct frozen native global-state build features required')
    fields = [row for row in layout['fields'] if _compiled_condition(row['condition'], set(features))]
    arrays = layout['arrays']
    names = [row['name'] for row in fields]
    all_names = names + [row['name'] for row in arrays]
    if not fields or len(set(all_names)) != len(all_names):
        raise ValueError('Distinct named global native state layout required')
    budget = request['maximum_attribute_bytes']
    if type(budget) is not int or budget < 1:
        raise ValueError('Explicit positive global state metadata workspace required')
    lengths = request['array_lengths']
    if set(lengths) != {row['name'] for row in arrays} or any(type(value) is not int or value < 0 for value in lengths.values()):
        raise ValueError('Every declared global native table needs a frozen exact length')
    floating = {row['name'] for row in fields+arrays if row['type'] == 'float'}
    limits = request['float_tolerances']
    if set(limits) != floating:
        raise ValueError('Every global floating scalar/table needs an explicit frozen tolerance')
    for tolerance in limits.values():
        if set(tolerance) != {'absolute', 'relative'} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in tolerance.values()):
            raise ValueError('Finite nonnegative global raw native-unit tolerances required')
    if len(sides) != 2:
        raise ValueError('Exactly two retained global-state endpoints required')
    payloads = []
    used_bytes = 0
    for side in sides:
        files = original_native_snapshot_files(side['run_directory'], side['snapshot'])
        verify_native_hdf5_set(files, require_publication=True)
        sources.update({str(path.resolve()): file_sha256(path)
                        for path in [*files, *native_publication_markers(files)]})
        reference = None
        for path in files:
            with h5py.File(path, 'r') as handle:
                header = handle['Header'].attrs
                for metadata, bound in (('ResearchGlobalLayoutSHA256', 128),
                        ('ResearchGlobalActiveFields', sum(len(name)+1 for name in names)+1)):
                    if metadata not in header:
                        raise ValueError('Missing native global-state header declaration')
                    identifier=header.get_id(metadata)
                    string=h5py.check_string_dtype(identifier.dtype)
                    if identifier.shape!=() or string is None or string.length is None or string.length>min(bound,budget//16):
                        raise MemoryError('Native global-state header declaration exceeds bounded fixed text scope')
                if _text(header.get('ResearchGlobalLayoutSHA256', '')) != layout['layout_sha256']:
                    raise ValueError('Native snapshot has another global-state layout')
                active = _text(header.get('ResearchGlobalActiveFields', '')).split(';')
                if active[-1:] == ['']:
                    active.pop()
                if active != names:
                    raise ValueError('Native global-state fields disagree with frozen build features')
                state = {}
                local_bytes = 0
                for row in fields+arrays:
                    name = 'ResearchGlobal_'+row['name']
                    expected = ()
                    count = 1
                    if row in arrays:
                        counter = name+'_Length'
                        if counter not in header:
                            raise ValueError('Global table has no explicit native length: '+name)
                        length_id = header.get_id(counter)
                        if length_id.shape != () or length_id.dtype.kind not in 'iu' or length_id.dtype.itemsize != 4:
                            raise ValueError('Global table length must retain its scalar native int32 type')
                        count = int(header[counter])
                        if count != lengths[row['name']]:
                            raise ValueError('Global table length differs from frozen native profile/schedule: '+name)
                        expected = (count,)
                        if count == 0:
                            if name in header:
                                raise ValueError('Empty global native table unexpectedly has a value attribute')
                            state[row['name']] = np.empty(0, dtype=np.float64 if row['type'] == 'float' else np.uint64)
                            continue
                    if name not in header:
                        raise ValueError('Declared global native restart field missing: '+name)
                    attribute = header.get_id(name)
                    if attribute.shape != expected:
                        raise ValueError('Native global-state field shape changed: '+name)
                    storage = int(attribute.get_storage_size())
                    local_bytes += max(storage, count*8)
                    # Numeric comparison scratch plus decoded strings and a second shard.
                    if used_bytes + local_bytes*16 > budget:
                        raise MemoryError('Native global-state attributes exceed cumulative workspace')
                    if row['type'] == 'text':
                        string = h5py.check_string_dtype(attribute.dtype)
                        if string is None or string.length is None or string.length > budget//16:
                            raise ValueError('Native global model binding needs bounded fixed-width text')
                        value = _text(header[name])
                        if '\x00' in value or any(ord(character) > 127 for character in value):
                            raise ValueError('Native model/seed binding is not exact ASCII text')
                        state[row['name']] = value
                    else:
                        dtype = attribute.dtype
                        wanted = 'f' if row['type'] == 'float' else 'u'
                        if dtype.kind != wanted or dtype.itemsize != 8:
                            raise ValueError('Global native member does not retain exact float64/uint64 storage: '+name)
                        value = np.asarray(header[name])
                        if wanted == 'f' and not np.isfinite(value).all():
                            raise ValueError('Nonfinite global native integration/source state: '+name)
                        state[row['name']] = value
                if reference is not None and any(
                        not np.array_equal(reference[name], state[name]) for name in all_names):
                    raise ValueError('Native snapshot shards disagree about declared global state')
                reference = state
        used_bytes += local_bytes*16
        payloads.append(reference)
    left, right = payloads
    rows = []
    for row in fields+arrays:
        name = row['name']
        a, b = left[name], right[name]
        if row['type'] == 'float':
            tolerance = limits[name]
            difference, allowed = _float_difference(a, b, tolerance)
            failed = difference > allowed
            rows.append({'field': 'All.'+name, 'status': 'FAIL' if failed.any() else 'PASS',
                         'failed_entries': int(failed.sum()), 'tolerance': tolerance,
                         'maximum_absolute_difference_native': float(difference.max()) if difference.size else None,
                         'comparison': 'named float64 global member/table in raw native units'})
        else:
            failed = np.not_equal(a, b)
            rows.append({'field': 'All.'+name, 'status': 'FAIL' if np.any(failed) else 'PASS',
                         'failed_entries': int(np.sum(failed)),
                         'comparison': 'exact ASCII model/seed binding' if row['type'] == 'text' else 'exact uint64 integer modulo representation'})
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    for path, digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Native global-state input changed before comparison publication')
    return {'fields': rows, 'sources': sources, 'layout_sha256': layout['layout_sha256'],
            'compiled_features': features, 'array_lengths': lengths, 'scope': layout['scope'],
            'model_binding_note': 'native CRC strings/seeds/hashes compare exactly; CRC alone is not a cryptographic external-asset identity'}


# Restart lifecycle journals


KINDS = {
    'stellar_birth': ('STARFORMATION', 'StellarBirth', 'commit_count', 'event_count'),
    'stellar_return': ('STELLAR_EVOLUTION', 'StellarReturn', 'return_commit_count', 'return_event_count'),
    'black_holes': ('BLACK_HOLES', 'BlackHole', 'bh_commit_count', 'bh_event_count'),
}
INTEGER_FIELDS = {'ssp_id', 'parent_id', 'mode', 'opportunity_counter', 'particle_id',
                  'counter', 'source_id', 'companion_id', 'parent_ids', 'previous_species',
                  'source_counter', 'species', 'sequence', 'previous_count', 'cumulative_count',
                  'previous_event_count', 'cumulative_event_count', 'event_count', 'ti_current',
                  'comoving', 'table_crc32', 'fit_reference_gm_c2'}


def _endpoint(side, requested):
    from core.native import original_native_snapshot_files
    from core.data import verify_native_hdf5_set, native_publication_markers
    import h5py
    files = original_native_snapshot_files(side['run_directory'], side['snapshot'])
    verify_native_hdf5_set(files, require_publication=True)
    sources = {str(path.resolve()): file_sha256(path)
               for path in [*files, *native_publication_markers(files)]}
    mapping = {'unit_mass_g': 'UnitMass_in_g', 'unit_time_s': 'UnitTime_in_s',
               'unit_velocity_cm_s': 'UnitVelocity_in_cm_per_s', 'unit_length_cm': 'UnitLength_in_cm',
               'hubble_param': 'HubbleParam', 'comoving': 'ComovingIntegrationOn'}
    if 'stellar_birth' in requested:
        mapping.update(factor_sn='FactorSN', seed='StellarRandomSeed')
    reference = None
    for path in files:
        with h5py.File(path, 'r') as handle:
            if 'Config' not in handle or 'Parameters' not in handle:
                raise ValueError('Native lifecycle comparison requires recorded build features and parameters')
            config = handle['Config'].attrs
            compiled = {name for name, row in KINDS.items() if row[0] in config}
            if compiled != set(requested):
                raise ValueError('Lifecycle comparison must include every compiled native journal family')
            params = handle['Parameters'].attrs
            parameters = {}
            for key, native in mapping.items():
                if native not in params or np.asarray(params[native]).shape != ():
                    raise ValueError('Missing scalar native lifecycle parameter: '+native)
                raw = params[native]
                if key == 'seed':
                    value = raw.decode('ascii') if isinstance(raw, bytes) else raw
                    if not isinstance(value, str) or not value.isdecimal() or int(value) >= 2**64:
                        raise ValueError('Lifecycle seed must retain exact uint64 decimal representation')
                    parameters[key] = str(int(value))
                elif key == 'comoving':
                    if np.asarray(raw).dtype.kind not in 'iu' or int(raw) not in (0, 1):
                        raise ValueError('Invalid native lifecycle integration mode')
                    parameters[key] = int(raw)
                else:
                    value = float(raw)
                    if not math.isfinite(value) or value <= 0 or key == 'factor_sn' and value >= 1:
                        raise ValueError('Invalid native lifecycle units/model parameter: '+key)
                    parameters[key] = value
            header = handle['Header'].attrs
            time = float(header['Time'])
            if not math.isfinite(time) or time <= 0:
                raise ValueError('Full native lifecycle history requires a positive endpoint clock')
            result = {'native_time': time, 'parameters': parameters}
            for kind in requested:
                _, prefix, commit_key, event_key = KINDS[kind]
                values = []
                for suffix in ('CommitCount', 'EventCount', 'JournalSchema'):
                    name = prefix+suffix
                    if name not in header or np.asarray(header[name]).shape != () or np.asarray(header[name]).dtype.kind != 'u' or np.asarray(header[name]).dtype.itemsize != 8:
                        raise ValueError('Lifecycle endpoint requires exact uint64 journal counters: '+name)
                    values.append(int(header[name]))
                if values[2] != 1 or values[0] > values[1]:
                    raise ValueError('Invalid native lifecycle endpoint schema/counters')
                result[commit_key], result[event_key] = values[:2]
            if reference is not None and result != reference:
                raise ValueError('Native lifecycle snapshot shards disagree about endpoint state')
            reference = result
    return reference, sources


def _read(kind, directory, endpoint, memory_bytes):
    origin = {'native_time': 0., 'parameters': endpoint['parameters']}
    _, _, commit_key, event_key = KINDS[kind]
    origin[commit_key] = origin[event_key] = 0
    if kind == 'stellar_birth':
        from core.stars import read_native_ssp_birth_journal
        arrays, commits, sources = read_native_ssp_birth_journal(directory,
            start_commit=0, stop_commit=endpoint[commit_key], start_count=0,
            stop_count=endpoint[event_key], native_time_interval=[0., endpoint['native_time']],
            expected_parameters=endpoint['parameters'], memory_bytes=memory_bytes)
        rows = [{name: values[index].item() for name, values in arrays.items()}
                for index in range(endpoint[event_key])]
    elif kind == 'stellar_return':
        from core.stars import read_native_stellar_return_journal
        rows, commits, sources = read_native_stellar_return_journal(directory,
            before=origin, after=endpoint, memory_bytes=memory_bytes)
    else:
        from core.black_holes import read_native_black_hole_journal
        rows, commits, sources = read_native_black_hole_journal(directory,
            before=origin, after=endpoint, memory_bytes=memory_bytes)
    return rows, commits, sources


def _event_key(kind, row):
    if kind == 'stellar_birth':
        return (row['ssp_id'], row['parent_id'], row['opportunity_counter'])
    if kind == 'stellar_return':
        return (row['source_id'], row['source_counter'], row['particle_id'], row['species'])
    return (row['kind'], row['particle_id'], row['counter'], row.get('source_id'), row.get('companion_id'))


def compare_lifecycle_journals(request, sides, *, memory_gib):
    kinds = request['kinds']
    if not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= set(KINDS):
        raise ValueError('Distinct complete native lifecycle journal families required')
    for key in ('maximum_events_per_family', 'maximum_commits_per_family', 'maximum_tolerance_fields'):
        if type(request[key]) is not int or request[key] < 1:
            raise ValueError('Positive bounded lifecycle comparison allocation required: '+key)
    tolerances = request['float_tolerances']
    if not isinstance(tolerances, dict) or len(tolerances) > request['maximum_tolerance_fields']:
        raise ValueError('Bounded explicit lifecycle floating field tolerances required')
    for limits in tolerances.values():
        if set(limits) != {'absolute', 'relative'} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in limits.values()):
            raise ValueError('Finite nonnegative lifecycle tolerances required in the published row units')
    endpoints = []
    sources = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for side in sides:
        endpoint, pins = _endpoint(side, kinds)
        endpoints.append(endpoint)
        sources.update(pins)
    if len(endpoints) != 2 or endpoints[0]['native_time'] != endpoints[1]['native_time'] or endpoints[0]['parameters'] != endpoints[1]['parameters']:
        raise ValueError('Native lifecycle endpoints must have the same exact actual epoch and physical model')
    summaries = {}
    used_tolerances = set()

    def compare(a, b, path, integer=False):
        summary = summaries.setdefault(path, {'field': path, 'compared_entries': 0, 'failed_entries': 0,
                                             'maximum_absolute_difference': None})
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                summary['failed_entries'] += 1
                return
            for name in sorted(a):
                compare(a[name], b[name], path+'.'+name, name in INTEGER_FIELDS)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                summary['failed_entries'] += 1
                return
            for x, y in zip(a, b):
                compare(x, y, path, integer)
            return
        summary['compared_entries'] += 1
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool) and not integer:
            if path not in tolerances:
                raise ValueError('Missing predeclared lifecycle field tolerance: '+path)
            limits = tolerances[path]
            used_tolerances.add(path)
            difference = abs(b-a)
            if not math.isfinite(difference):
                raise ValueError('Nonfinite native lifecycle comparison difference')
            summary['maximum_absolute_difference'] = max(summary['maximum_absolute_difference'] or 0., difference)
            allowed = limits['absolute'] + limits['relative']*max(abs(a), abs(b))
            if not math.isfinite(allowed):
                raise ValueError('Native lifecycle tolerance arithmetic overflow')
            summary['failed_entries'] += int(difference > allowed)
            summary['comparison'] = 'explicit tolerance in producer-published field units'
            summary['tolerance'] = limits
        else:
            summary['failed_entries'] += int(type(a) != type(b) or a != b)
            summary['comparison'] = 'exact integer/text/structural state'

    family_counts = {}
    for kind in kinds:
        _, _, commit_key, event_key = KINDS[kind]
        for endpoint in endpoints:
            if endpoint[event_key] > request['maximum_events_per_family'] or endpoint[commit_key] > request['maximum_commits_per_family']:
                raise MemoryError('Full lifecycle journal prefix exceeds frozen event/commit allocation')
        data = []
        for name, endpoint in zip(('uninterrupted', 'resumed'), endpoints):
            directory = Path(request['directories'][name][kind]).resolve()
            prefix={'stellar_birth':'ssp_birth','stellar_return':'stellar_returns','black_holes':'black_holes'}[kind]
            start=0 if kind=='stellar_birth' else 1
            metadata_bytes=0
            for sequence in range(start,endpoint[commit_key]+start):
                commit_path=directory/(prefix+'_'+format(sequence,'020d')+'.commit.json')
                if commit_path.is_symlink() or not commit_path.is_file():
                    raise ValueError('Full lifecycle comparison requires the retained inherited commit prefix')
                metadata_bytes+=commit_path.stat().st_size
                if metadata_bytes*64>memory_gib*1024**3/16:
                    raise MemoryError('Lifecycle commit descriptor workspace exceeds cumulative allocation')
            rows, commits, pins = _read(kind, directory, endpoint, int(memory_gib*1024**3/4))
            sources.update(pins)
            index = {}
            for row in rows:
                key = _event_key(kind, row)
                if key in index:
                    raise ValueError('Ambiguous repeated lifecycle operation identity; cannot invent restart alignment')
                index[key] = row
            data.append((index, commits))
        left, right = data
        compare(endpoints[0][commit_key], endpoints[1][commit_key], kind+'.commit_count', True)
        compare(endpoints[0][event_key], endpoints[1][event_key], kind+'.event_count', True)
        missing = set(left[0])-set(right[0])
        extra = set(right[0])-set(left[0])
        summaries[kind+'.event_identity_census'] = {'field': kind+'.event_identity_census',
            'compared_entries': len(set(left[0]) | set(right[0])), 'failed_entries': len(missing)+len(extra),
            'missing_events': len(missing), 'extra_events': len(extra), 'comparison': 'exact operation identities independent of rank/row order'}
        for key in left[0].keys() & right[0].keys():
            row = left[0][key]
            category = row.get('kind', kind)
            compare(row, right[0][key], kind+'.events.'+category)
        # Compare scientific commit metadata, not decomposition-specific file descriptors.
        for a, b in zip(left[1], right[1]):
            omit = {'parts', 'ranks', 'mpi_ranks'}
            compare({k: v for k, v in a.items() if k not in omit},
                    {k: v for k, v in b.items() if k not in omit}, kind+'.commits')
        family_counts[kind] = {'uninterrupted_events': len(left[0]), 'resumed_events': len(right[0]),
                               'uninterrupted_commits': len(left[1]), 'resumed_commits': len(right[1]),
                               'population_measured': bool(left[0] or right[0])}
        del data, left, right, index, rows, commits
    for path, digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Lifecycle restart journal changed before comparison publication')
    fields = [{**row, 'status': 'FAIL' if row['failed_entries'] else 'PASS'} for row in summaries.values()]
    for name in ('stars.py','black_holes.py','data.py','foundation.py'):
        path=Path(__file__).resolve().parents[1]/name
        sources[str(path)]=file_sha256(path)
    return {'fields': fields, 'sources': sources, 'families': family_counts,
            'unused_tolerance_fields': sorted(set(tolerances)-used_tolerances),
            'scope': 'complete retained committed lifecycle prefixes from zero through each actual snapshot counter, including removed particle events; uncommitted files excluded',
            'empty_population_policy': 'counter agreement with zero events does not establish physical event coverage'}


# Restart comparison


def compare_native_restart(request,destination):
    from core.observation_geometry import read_native_gravitating_particles
    from core.native import original_native_snapshot_files
    from core.data import exact_uint64_ids
    import h5py
    if request.get('schema')!='native-restart-comparison-request-v1':raise ValueError('Frozen native restart comparison request required')
    for key in ('maximum_particles','maximum_source_fields'):
        if type(request[key]) is not int or request[key]<1:raise ValueError('Explicit positive restart comparison bound required: '+key)
    memory=request['memory_gib']
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive restart comparison workspace required')
    for name in ('require_complete_source_state','require_particle_integration_state',
                 'require_lifecycle_history','require_global_state'):
        if type(request.get(name,False)) is not bool:
            raise ValueError('Restart comparison scope must be an explicit boolean: '+name)
    marker=Path(request['checkpoint_marker']['path']).resolve()
    if file_sha256(marker)!=request['checkpoint_marker']['sha256'] or marker.with_name(marker.name+'.sha256').read_text().strip()!=file_sha256(marker):
        raise ValueError('Actual checkpoint publication marker changed')
    checkpoint=json.loads(marker.read_text())
    if checkpoint.get('schema')!='native-checkpoint-publication-v1' or checkpoint.get('status')!='COMPLETE':
        raise ValueError('Restart comparison requires actual completed checkpoint publication')
    invocation=Path(request['resume_invocation']['path']).resolve()
    if file_sha256(invocation)!=request['resume_invocation']['sha256']:raise ValueError('Actual retained resume invocation changed')
    resumed=json.loads(invocation.read_text())
    marker_hash=file_sha256(marker)
    if resumed.get('returncode')!=0 or resumed.get('input_hashes',{}).get(str(marker))!=marker_hash:
        raise ValueError('Resume invocation must have completed and bind the actual interrupted checkpoint marker')
    sources={str(marker):marker_hash,str(invocation):file_sha256(invocation)}
    if request.get('uninterrupted_invocation'):
        reference=request['uninterrupted_invocation'];path=Path(reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:
            raise ValueError('Retained uninterrupted reference invocation changed')
        baseline=json.loads(path.read_text())
        if baseline.get('returncode')!=0:
            raise ValueError('Uninterrupted reference must have completed successfully')
        sources[str(path)]=reference['sha256']
    fields=request['source_fields']
    if not isinstance(fields,list) or len(fields)>request['maximum_source_fields'] or len({row['dataset'] for row in fields})!=len(fields):
        raise ValueError('Restart source fields require a bounded distinct frozen field list')
    species=request['species']
    if not isinstance(species,list) or not species or any(type(kind) is not int or kind not in range(6) for kind in species) or len(set(species))!=len(species):
        raise ValueError('Restart comparison needs distinct explicit native species 0 through 5')
    sides=[];reports=[]
    for side in ('uninterrupted','resumed'):
        reference=request[side]
        report,arrays=read_native_gravitating_particles(reference['run_directory'],reference['snapshot'],
            species=request['species'],maximum_particles=request['maximum_particles'],memory_gib=memory/4,
            namespace=request['identity_namespace'])
        sources.update(report['source_hashes'])
        order=np.argsort(arrays['particle_ids']);arrays={name:values[order] for name,values in arrays.items()}
        ids=arrays['particle_ids'];additional={};counts={};retained_source_bytes=0
        if not len(ids):raise ValueError('Empty native physical population cannot establish restart agreement')
        for file in original_native_snapshot_files(reference['run_directory'],reference['snapshot']):
            sources[str(Path(file).resolve())]=file_sha256(file)
            with h5py.File(file,'r') as handle:
                for kind in request['species']:
                    group=handle.get('PartType'+str(kind))
                    if group is None:continue
                    local=exact_uint64_ids(group['ParticleIDs'][:],unique=False)
                    positions=np.searchsorted(ids,local);keep=positions<len(ids)
                    keep[keep]=ids[positions[keep]]==local[keep]
                    for field in fields:
                        if kind not in field['species']:continue
                        name=field['dataset']
                        if name not in group or len(group[name])!=len(local):raise ValueError('Native restart source-state field absent or has another identity census')
                        dataset=group[name]
                        if dataset.dtype.kind not in 'fiu':
                            raise ValueError('Native restart source state contains unsupported storage')
                        shape=(len(ids),*dataset.shape[1:])
                        if name not in additional:
                            field_bytes=math.prod(shape)*dataset.dtype.itemsize+len(ids)
                            if retained_source_bytes+field_bytes>memory*1024**3/16:
                                raise MemoryError('Native restart source-state array exceeds frozen workspace')
                            retained_source_bytes+=field_bytes
                            additional[name]=np.zeros(shape,dtype=dataset.dtype);counts[name]=np.zeros(len(ids),dtype=bool)
                        elif additional[name].shape!=shape or additional[name].dtype!=dataset.dtype:
                            raise ValueError('Native restart source-state storage differs across species/shards')
                        entry_bytes=math.prod(dataset.shape[1:])*dataset.dtype.itemsize
                        chunk=min(4096,max(1,int(memory*1024**3/64)//max(entry_bytes,1)))
                        if entry_bytes>memory*1024**3/64:raise MemoryError('Native restart field row exceeds workspace')
                        for begin in range(0,len(local),chunk):
                            end=min(begin+chunk,len(local));selected=keep[begin:end]
                            values=dataset[begin:end][selected];index=positions[begin:end][selected]
                            if values.dtype.kind=='f' and not np.isfinite(values).all():
                                raise ValueError('Native restart source state contains nonfinite payload')
                            if counts[name][index].any():raise ValueError('Native restart source identity repeats across shards')
                            additional[name][index]=values;counts[name][index]=True
        for field in fields:
            wanted=np.isin(arrays['species'],field['species'])
            if field['dataset'] not in counts or not counts[field['dataset']][wanted].all():raise ValueError('Native restart source-state domain is incomplete')
        sides.append((arrays,additional,counts));reports.append(report)
    left,right=sides[0][0],sides[1][0]
    if not np.array_equal(left['particle_ids'],right['particle_ids']) or not np.array_equal(left['species'],right['species']):
        raise ValueError('Resumed native physical identity/species census differs from uninterrupted reference')
    epoch_keys=('actual_scale_factor','native_time_code','elapsed_time_gyr','comoving_integration')
    if not any(key in reports[0]['epoch'] for key in epoch_keys) or any(
            reports[0]['epoch'].get(key)!=reports[1]['epoch'].get(key) for key in epoch_keys):
        raise ValueError('Restart comparisons require the exact actual native physical epoch')
    capsule=None
    if request.get('require_complete_source_state',False):
        if not request.get('source_capsule') or set(request['species'])!=set(range(6)):
            raise ValueError('Complete native source-state restart scope requires the typed capsule and all native species')
    if request.get('source_capsule'):
        capsule=compare_source_capsules(request['source_capsule'],[request['uninterrupted'],request['resumed']],
            species=request['species'],maximum_particles=request['maximum_particles'],memory_gib=memory/2)
        sources.update(capsule['sources'])
    rows=list(capsule['fields']) if capsule else []
    integrator_state=None
    if request.get('require_particle_integration_state',False) and (not request.get('integrator_state') or not capsule):
        raise ValueError('Required native particle integration comparison needs both integration and typed lifecycle capsules')
    if request.get('integrator_state'):
        if not capsule:raise ValueError('Native integration comparison requires typed source/lifecycle capsule comparison')
        integrator_state=compare_integrator_capsules(request['integrator_state'],
            [request['uninterrupted'],request['resumed']],species=species,
            maximum_particles=request['maximum_particles'],memory_gib=memory/4,
            source_layout=request['source_capsule']['layout'])
        sources.update(integrator_state['sources']);rows.extend(integrator_state['fields'])
    hydro_state=None
    if request.get('hydro_particle_state'):
        hydro_state=compare_hydro_particle_state(request['hydro_particle_state'],
            [request['uninterrupted'],request['resumed']],species=request['species'],
            maximum_particles=request['maximum_particles'],memory_gib=memory/4,
            source_layout=(request.get('source_capsule') or {}).get('layout'))
        sources.update(hydro_state['sources']);rows.extend(hydro_state['fields'])
    lifecycle_journals=None
    if request.get('require_lifecycle_history',False) and not request.get('lifecycle_journals'):
        raise ValueError('Required removed-particle lifecycle history cannot be replaced by snapshot state alone')
    if request.get('lifecycle_journals'):
        lifecycle_journals=compare_lifecycle_journals(request['lifecycle_journals'],
            [request['uninterrupted'],request['resumed']],memory_gib=memory/4)
        sources.update(lifecycle_journals['sources']);rows.extend(lifecycle_journals['fields'])
    global_state=None
    if request.get('require_global_state',False) and not request.get('global_state'):
        raise ValueError('Required native global-state comparison cannot be replaced by particle snapshots alone')
    if request.get('global_state'):
        global_state=compare_global_state(request['global_state'],
            [request['uninterrupted'],request['resumed']])
        sources.update(global_state['sources']);rows.extend(global_state['fields'])
    for name,unit in (('position_proper_kpc','proper kpc'),('velocity_peculiar_kms','km / s'),('mass_msun','Msun')):
        tolerance=request['tolerances'][name]
        absolute,relative=tolerance['absolute'],tolerance['relative']
        if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 for value in (absolute,relative)):
            raise ValueError('Frozen finite nonnegative native restart tolerances required')
        difference=np.abs(right[name]-left[name])
        if name=='position_proper_kpc' and request.get('periodic_box_proper_kpc') is not None:
            box=request['periodic_box_proper_kpc']
            if isinstance(box,bool) or not math.isfinite(box) or box<=0:raise ValueError('Explicit positive proper periodic comparison box required')
            # Minimum-image displacement in the declared proper-kpc box:
            # subtract the nearest integer box multiple before taking |dx|.
            signed=right[name]-left[name];difference=np.abs(signed-box*np.rint(signed/box))
        allowed=absolute+relative*np.maximum(np.abs(left[name]),np.abs(right[name]))
        if not np.isfinite(difference).all() or not np.isfinite(allowed).all():
            raise ValueError('Physical restart difference or tolerance arithmetic overflow')
        rows.append({'field':name,'units':unit,'maximum_absolute_difference':float(difference.max()),
            'failed_entries':int(np.sum(difference>allowed)),'status':'PASS' if np.all(difference<=allowed) else 'FAIL'})
    for field in fields:
        name=field['dataset'];a,b=sides[0][1][name],sides[1][1][name]
        mask=np.isin(left['species'],field['species']);a,b=a[mask],b[mask]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Native restart source field changes shape or storage type')
        if a.dtype.kind in 'iu':passed=np.equal(a,b)
        else:
            absolute,relative=field['absolute_tolerance'],field['relative_tolerance']
            if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 for value in (absolute,relative)):
                raise ValueError('Native floating source-state comparison requires frozen nonnegative tolerances')
            difference, allowed = _float_difference(a, b, {'absolute': absolute, 'relative': relative})
            passed = difference <= allowed
        rows.append({'field':name,'status':'PASS' if passed.all() else 'FAIL','failed_entries':int((~passed).sum()),
            'comparison':'exact integer state' if a.dtype.kind in 'iu' else 'frozen floating source-state tolerance'})
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Retained restart output changed before comparison publication')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Retained restart comparisons are immutable')
    destination.mkdir(parents=True)
    measurements=destination/'measurements.json';atomic_json(measurements,{'fields':rows,'particle_count':len(left['particle_ids'])})
    result={'schema':'native-restart-comparison-v1','execution_status':'COMPLETE',
        'status':'PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL','request_hash':canonical_digest(request),
        'checkpoint_marker_sha256':marker_hash,'source_hashes':sources,'files_sha256':{str(measurements):file_sha256(measurements)},
        'fields':rows,'particle_count':len(left['particle_ids']),'science_acceptance':'DECLARED_RESTART_STATE_ONLY',
        'source_capsule':{key:value for key,value in capsule.items() if key not in {'fields','sources'}} if capsule else None,
        'hydro_particle_state':{key:value for key,value in hydro_state.items() if key not in {'fields','sources'}} if hydro_state else None,
        'integrator_state':{key:value for key,value in integrator_state.items() if key not in {'fields','sources'}} if integrator_state else None,
        'lifecycle_journals':{key:value for key,value in lifecycle_journals.items() if key not in {'fields','sources'}} if lifecycle_journals else None,
        'global_state':{key:value for key,value in global_state.items() if key not in {'fields','sources'}} if global_state else None,
        'scope':'surviving native physical state plus complete selected typed source/retirement state' if capsule else 'exact declared surviving native species/fields; full typed source/retirement scope not requested'}
    output=destination/'restart-comparison.json';atomic_json(output,result)
    return result


def restart_comparison_main():
    import argparse
    parser=argparse.ArgumentParser(description='Compare actual resumed/uninterrupted native states with frozen tolerances')
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.request.stat().st_size>16*1024**2:raise MemoryError('Restart comparison request exceeds metadata intake bound')
    print(json.dumps(compare_native_restart(json.loads(args.request.read_text()),args.output),indent=2))
