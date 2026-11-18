"""Read GADGET's recorded SUBFIND descendant/progenitor tables exactly.

FOF links are explicitly the projection of a previous FOF central's *native*
primary descendant into its next FOF host. They are FOF membership joins, not
continuous central-galaxy coalescence times or particle-transfer measurements.
"""
from pathlib import Path
import numpy as np
from .foundation import file_sha256
from .data import canonical_native_links, exact_uint64_ids
from .foundation import merge_source_bindings as _merge


def recorded_link_sources():
    base = Path(__file__).with_name('software') / 'gadget4/src'
    names = ('mergertree/io_descendant.cc', 'mergertree/io_descendant.h', 'mergertree/io_progenitors.cc',
        'mergertree/io_progenitors.h', 'mergertree/descendant.cc', 'main/run.cc')
    return {str((base / name).resolve()): file_sha256(base / name) for name in names}


def native_link_table(run, snapshot, kind, *, expected_rows, maximum_rows, memory_gib, native_sources):
    import h5py
    from .halos import _check_file_series, _group_files
    if kind not in {'desc', 'prog'} or type(snapshot) is not int or snapshot < 0:
        raise ValueError('Exact native descendant/progenitor saved table required')
    if expected_rows > maximum_rows or expected_rows * 256 > memory_gib * 1024**3 / 4:
        raise MemoryError('Native ancestry tables exceed predeclared census/workspace')
    # Select the same basename emitted by io_descendant/io_progenitors. The
    # run may contain private input copies; only the native output tree counts.
    catalogues = _group_files(Path(run), snapshot)
    if not catalogues:raise FileNotFoundError('Native ancestry has no original FOF catalogue')
    parent = catalogues[0].parent
    directory = parent.parent if parent.name == f'groups_{snapshot:03d}' else parent
    stem = f'subhalo_{kind}_{snapshot:03d}'
    single = directory / (stem + '.hdf5')
    multi = list((directory / f'groups_{snapshot:03d}').glob(stem + '.*.hdf5'))
    if single.is_file() and multi:
        raise ValueError('Duplicate single/multiple native ancestry publication')
    paths = [single] if single.is_file() else sorted(multi, key=lambda p: int(p.name.split('.')[-2]))
    if not paths:
        raise FileNotFoundError('Missing recorded native ancestry table: ' + stem)
    if multi and [int(p.name.split('.')[-2]) for p in paths] != list(range(len(paths))):
        raise ValueError('Native ancestry shard numbering has a gap')
    pins = {}
    for path in paths:
        key = str(path.resolve()); digest = file_sha256(path)
        if native_sources.get(key) != digest:
            raise ValueError('Native ancestry table is outside the completed evolution generation')
        pins[key] = digest
    _check_file_series(paths, 'Header', 'Nsubhalos_Total')
    identity = []; link = []; field = 'DescSubhaloNr' if kind == 'desc' else 'ProgSubhaloNr'
    for path in paths:
        with h5py.File(path) as f:
            h = f['Header'].attrs; count = int(h['Nsubhalos_ThisFile'])
            if int(h['Nsubhalos_Total']) != expected_rows:
                raise ValueError('Native ancestry table differs from its actual SUBFIND row census')
            if not count:
                continue
            for name in ('SubhaloNr', field):
                if name not in f.get('Subhalo', {}) or f['Subhalo'][name].shape != (count,):
                    raise ValueError('Incomplete aligned native ancestry identity/link column')
            identity.append(exact_uint64_ids(f['Subhalo/SubhaloNr'][:]))
            # Preserve signed -1 as unavailable; never cast it to a huge ID.
            values, available = canonical_native_links(f['Subhalo/' + field][:])
            link.append((values, available))
    ids = np.concatenate(identity) if identity else np.empty(0, np.uint64)
    if len(ids) != expected_rows or len(np.unique(ids)) != len(ids) or np.any(ids >= expected_rows):
        raise ValueError('Native ancestry identities do not cover the exact original row set')
    order = np.argsort(ids)
    values = np.concatenate([x[0] for x in link]) if link else np.empty(0, np.uint64)
    available = np.concatenate([x[1] for x in link]) if link else np.empty(0, bool)
    return values[order], available[order], pins


def project_native_central_links(previous, current, descendants, descendant_available, progenitors, progenitor_available):
    """Project recorded links without promoting a largest-mass candidate."""
    for fields in (previous, current):
        for key in ('SubhaloGroupNr', 'SubhaloRankInGr'):
            values = np.asarray(fields[key])
            if values.ndim != 1 or values.dtype.kind not in 'ui' or np.any(values < 0):
                raise ValueError('Native central/host map needs exact nonnegative row fields')
        if len(fields['SubhaloGroupNr']) != len(fields['SubhaloRankInGr']):
            raise ValueError('Native central/host arrays do not align')
        hosts = fields['SubhaloGroupNr'][fields['SubhaloRankInGr'] == 0]
        if len(np.unique(hosts)) != len(hosts):
            raise ValueError('More than one native rank-zero central in one FOF host')
    nprev = len(previous['SubhaloGroupNr']); nnext = len(current['SubhaloGroupNr'])
    if (descendants.shape != (nprev,) or descendant_available.shape != (nprev,)
            or progenitors.shape != (nnext,) or progenitor_available.shape != (nnext,)
            or descendant_available.dtype != np.dtype(bool) or progenitor_available.dtype != np.dtype(bool)):
        raise ValueError('Native ancestry links must align with both saved SUBFIND populations')
    if np.any(descendants[descendant_available] >= nnext) or np.any(progenitors[progenitor_available] >= nprev):
        raise ValueError('Native ancestry link endpoint is outside its actual saved catalogue')
    links = []; unresolved = []
    for source in np.flatnonzero(previous['SubhaloRankInGr'] == 0):
        host = int(previous['SubhaloGroupNr'][source])
        if not descendant_available[source]:
            unresolved.append({'source_host_row': host, 'source_central_subhalo_row': int(source), 'reason': 'NO_RECORDED_NATIVE_DESCENDANT'}); continue
        dest = int(descendants[source]); dest_host = int(current['SubhaloGroupNr'][dest])
        central = np.flatnonzero((current['SubhaloGroupNr'] == dest_host) & (current['SubhaloRankInGr'] == 0))
        native_main = (len(central) == 1 and dest == central[0] and progenitor_available[dest]
            and int(progenitors[dest]) == source)
        evidence = {'origin': 'native_ancestry', 'basis': 'GADGET MERGERTREE primary descendant of the previous FOF rank-zero central, projected into its next native FOF membership',
            'event_scope': 'FOF host membership join through recorded central-core ancestry; not continuous subhalo/galaxy coalescence',
            'source_central_subhalo_row': int(source), 'native_descendant_subhalo_row': dest,
            'native_descendant_is_next_central': bool(current['SubhaloRankInGr'][dest] == 0)}
        if native_main:
            evidence['main_progenitor'] = True
        links.append({'source_host_row': host, 'target_host_row': dest_host, 'evidence': evidence})
    # No main progenitor is manufactured if the two native choices are not
    # mutual. The original assembly estimator then retains its ambiguity.
    return links, unresolved


def build_native_fof_ancestry(run, snapshots, destination, *, native_sources, maximum_nodes, maximum_edges,
                            maximum_subhalos, maximum_particles, minimum_host_particles, memory_gib):
    from .halos import read_native_subhalo_spans
    from .halos import IdentityGraph, _read_catalogue_series
    from .zoom import read_host_catalogue
    from .data import run_epoch_contract
    if (not isinstance(snapshots, list) or not snapshots or any(type(s) is not int or s < 0 for s in snapshots)
            or snapshots != sorted(set(snapshots)) or any(b != a + 1 for a, b in zip(snapshots, snapshots[1:]))):
        raise ValueError('Native ancestry requires every consecutive actual saved snapshot; no sparse-output bridge')
    if type(minimum_host_particles) is not int or minimum_host_particles < 1:
        raise ValueError('Original native FOF support particle count required')
    graph = IdentityGraph(maximum_nodes=maximum_nodes, maximum_edges=maximum_edges)
    pins = dict(recorded_link_sources())
    if any(native_sources.get(p) != h for p, h in pins.items()):
        raise ValueError('Actual completed evolution must bind the native ancestry writer/operator sources')
    reader=Path(__file__).with_name('halos.py').resolve()
    _merge(pins,{str(reader):file_sha256(reader)})
    namespace = str(Path(run).resolve()) + ':fof:catalogue-row'
    previous = None; previous_epoch = None; unresolved = []
    for snapshot in snapshots:
        fields, _, _ = read_native_subhalo_spans(run, snapshot, native_sources=native_sources,
            maximum_subhalos=maximum_subhalos, maximum_particles=maximum_particles, memory_gib=memory_gib)
        catalogue = read_host_catalogue(run, snapshot, memory_gib=memory_gib)
        count = catalogue['group_count']
        if len(graph.nodes) + count > maximum_nodes or (len(graph.nodes) + count) * 8192 > memory_gib * 1024**3 / 3:
            raise MemoryError('Native FOF ancestry graph exceeds declared census/workspace')
        _merge(pins, {r['path']: r['sha256'] for r in catalogue['sources'] + catalogue['snapshot_sources']})
        if any(native_sources.get(p) != h for p, h in pins.items()):
            raise ValueError('Native ancestry host catalogue is not owned by this evolution')
        if np.any(fields['SubhaloGroupNr'] >= count):
            raise ValueError('Native subhalo/FOF ownership points outside the actual catalogue')
        a, box, h = catalogue['frame']; epoch = run_epoch_contract(Path(run), snapshot, a)
        if epoch['canonical_time'] != 'scale_factor' or epoch.get('cosmic_time_gyr') is None:
            raise ValueError('Native merger/centroid joins require actual cosmological proper clocks')
        if previous_epoch is not None and epoch['actual_scale_factor'] <= previous_epoch['actual_scale_factor']:
            raise ValueError('Native ancestry epochs do not increase in actual saved time')
        graph.add_epoch(snapshot, epoch)
        masses = _read_catalogue_series(catalogue['catalogues'], 'Group', 'GroupMass')
        if masses.shape != (count,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError('Native FOF ancestry requires original positive inclusive masses')
        clock = epoch.get('cosmic_time_source', {})
        if clock.get('path'):
            _merge(pins, {clock['path']: clock['sha256']})
        for row in range(count):
            graph.add_node(namespace, row, snapshot, kind='halo', attributes={
                'id': row, 'position_proper_kpc': (catalogue['positions'][row] * a / h).tolist(),
                'mass_msun': float(masses[row] * 1e10 / h), 'mass_definition': 'FOF inclusive mass',
                'measurement_quality_passed': sum(map(int, catalogue['lengths'][row])) >= minimum_host_particles,
                'native_ancestry_scope': 'Recorded previous-central descendant projected to current FOF membership; no halo row continuity assumption'})
        if previous is not None:
            left, left_snapshot = previous
            desc, present, desc_sources = native_link_table(run, left_snapshot, 'desc', expected_rows=len(left['SubhaloGroupNr']),
                maximum_rows=maximum_subhalos, memory_gib=memory_gib, native_sources=native_sources); _merge(pins, desc_sources)
            prog, present_prog, sources = native_link_table(run, snapshot, 'prog', expected_rows=len(fields['SubhaloGroupNr']),
                maximum_rows=maximum_subhalos, memory_gib=memory_gib, native_sources=native_sources); _merge(pins, sources)
            links, missing = project_native_central_links(left, fields, desc, present, prog, present_prog)
            unresolved.extend({'snapshot': left_snapshot, **r} for r in missing)
            for link in links:
                graph.add_edge((namespace, link['source_host_row'], left_snapshot), (namespace, link['target_host_row'], snapshot),
                    'descendant', evidence={**link['evidence'], 'source_hashes': {**desc_sources,**sources,**recorded_link_sources()}})
        previous = (fields, snapshot); previous_epoch = epoch
    _merge(pins, {str(Path(__file__).resolve()): file_sha256(__file__)})
    for p, h in pins.items():
        if file_sha256(p) != h:
            raise ValueError('Native ancestry input changed during graph construction')
    report = graph.publish(destination, source_hashes=pins)
    return report, unresolved
