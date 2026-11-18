"""Native catalogue ancestry to existing orbital reconstruction and event plots.

Positions keep explicitly recorded periodic images; velocities are converted
with the shared original-run clock. Native quality and cadence masks remain
mandatory. Interpolated events are candidates unless independently certified.
"""
from pathlib import Path
import numpy as np
from .foundation import file_sha256,canonical_digest
from .data import export_numeric_audit,exact_uint64_ids,numeric_audit_source_bindings,proper_position_derivative_kms
from core.stream_common import read_product
from core.research.display import export_scientific_suite


def native_orbit_inputs(graph_directory,request,*,memory_gib):
    from .halos import verify_identity_graph
    root=Path(graph_directory).resolve()
    if (root/'report.json').stat().st_size*16>memory_gib*1024**3/3:raise MemoryError('Native orbit graph exceeds allocation')
    graph=verify_identity_graph(root);sources=dict(graph['sources'])
    sources.update({str(root/n):file_sha256(root/n) for n in ('report.json','report.sha256','index.sqlite','nodes.npz')})
    binding=request['recorded_images'];path=Path(binding['path']).resolve()
    if file_sha256(path)!=binding['sha256']:raise ValueError('Recorded catalogue images changed')
    record,arrays,pins=read_product(path,memory_gib=memory_gib/3)
    if (record.get('schema')!='native-catalogue-orbit-images-v1' or record['graph_fingerprint']!=graph['fingerprint']
            or record['object_keys']!=request['object_keys'] or record['host_keys']!=request['host_keys']):
        raise ValueError('Recorded images must bind exact graph and epoch-qualified object/host keys')
    sources.update(pins);sources.update(numeric_audit_source_bindings(record))
    run=Path(request['run_directory']).resolve()
    sources.update({str(run/name):file_sha256(run/name) for name in ('config.yaml','manifest.json')})
    from .analysis import snapshot_ids
    from .halos import _snapshot_files
    saved=snapshot_ids(run)
    if len(saved)>request['maximum_saved_epochs']:raise MemoryError('Native orbit saved-source census exceeds allocation')
    for snap in saved:
        for path in _snapshot_files(run,snap):sources[str(path.resolve())]=file_sha256(path)
    for epoch in graph['epochs'].values():
        clock=epoch.get('cosmic_time_source',{})
        if clock.get('path'):
            if file_sha256(clock['path'])!=clock['sha256']:raise ValueError('Native graph proper-time source changed')
            sources[clock['path']]=clock['sha256']
    return graph,arrays['image_offsets'],sources


def catalogue_orbit_worldlines(graph,images,request,epochs,times,box_proper,hubble,*,saved_order,frame):
    """Adapt normalized catalogue graph nodes; no nearest-image or identity fit."""
    from core.stream.trajectories import Worldline
    nodes={tuple(n['key']):n for n in graph['nodes']};n=len(times)
    if images.dtype.kind not in 'iu' or images.shape!=(n,2,3):raise ValueError('Exact recorded images required for each object/host epoch')
    native_edges={(tuple(e['source']),tuple(e['target'])) for e in graph['edges']
        if e['kind']=='descendant' and e['evidence'].get('origin')=='native_ancestry'}
    lines=[];raw={}
    for index,kind in enumerate(('object','host')):
        keys=request[kind+'_keys'];positions=[];velocities=[];masses=[];radii=[];quality=[]
        if any(len(k)!=3 or not isinstance(k[0],str) or type(k[2]) is not int or k[2]<0 for k in keys):
            raise ValueError('Exact namespace/uint64/saved-epoch observation keys required')
        if len(keys)!=n or len({tuple(k) for k in keys})!=n:raise ValueError('Distinct original object/host observation keys required')
        ids=exact_uint64_ids([key[1] for key in keys],unique=False)
        namespaces={key[0] for key in keys}
        if len(namespaces)!=1:raise ValueError('Catalogue orbit changes finder namespace')
        flags=request[kind+'_valid_intervals']
        if not isinstance(flags,list) or len(flags)!=n-1 or any(type(v) is not bool for v in flags):raise ValueError('Exact boolean native orbit interval support required')
        for i,key in enumerate(keys):
            node=nodes.get(tuple(key))
            if node is None or node['kind'] not in ({'halo'} if kind=='host' else {'halo','subhalo'}):raise ValueError('Native orbit selection has wrong identity kind')
            attributes=node['attributes']
            if kind=='host' and attributes.get('radius_definition')!=request['host_radius_definition']:
                raise ValueError('Infall boundary must preserve original native host radius definition')
            positions.append(attributes['position_proper_kpc']);velocities.append(attributes['velocity_peculiar_kms'])
            masses.append(attributes['mass_msun']);radii.append(attributes.get('radius_proper_kpc',np.nan))
            quality.append(node['status']=='DETECTED' and attributes.get('measurement_quality_passed') is True)
            clock=graph['epochs'][str(key[2])]
            if clock['actual_scale_factor']!=epochs[i]['actual_scale_factor']:raise ValueError('Catalogue graph clock differs from actual native saved epoch')
        position=np.asarray(positions,float)+images[:,index]*np.asarray(box_proper)[:,None]
        peculiar=np.asarray(velocities,float)
        if position.shape!=(n,3) or peculiar.shape!=position.shape or not np.isfinite(position).all() or not np.isfinite(peculiar).all():
            raise ValueError('Finite normalized native catalogue phase space required')
        velocity=np.array([proper_position_derivative_kms(position[i],peculiar[i],hubble_kms_mpc=hubble[i],
            origin_proper_kpc=np.zeros(3),origin_velocity_peculiar_kms=np.zeros(3)) for i in range(n)])
        valid=np.array(flags,bool)&np.asarray(quality[:-1])&np.asarray(quality[1:])
        for i in range(n-1):
            if (tuple(keys[i]),tuple(keys[i+1])) not in native_edges:raise ValueError('Catalogue orbit lacks direct native ancestry evidence')
            valid[i]&=saved_order[keys[i+1][2]]==saved_order[keys[i][2]]+1
        valid&=np.diff(times)<=request['maximum_interval_myr']
        line={'identity':canonical_digest(keys),'namespace':keys[0][0],'frame':frame,'units':'proper_kpc_kms_myr',
            'times_myr':np.asarray(times).tolist(),'position_kpc':position.tolist(),'velocity_kms':velocity.tolist(),
            'mass_msun':masses,'valid_intervals':valid.tolist(),'maximum_interval_myr':request['maximum_interval_myr'],
            'interpolation_certificate':{'status':'NOT_CERTIFIED'},'original_observation_keys':keys}
        if kind=='host':line.update(halo_radius_kpc=radii,halo_radius_definition=request['host_radius_definition'])
        Worldline(line)
        lines.append(line)
        raw.update({kind+'_ids':ids,kind+'_position_proper_kpc':position,kind+'_velocity_derivative_kms':velocity,
            kind+'_valid_intervals':valid,kind+'_quality_supported':np.asarray(quality,bool)})
    return lines,raw


def measure_native_assembly_orbits(graph_directory,request,destination,*,memory_gib=1.):
    from .analysis import snapshot_ids,read_snapshot_header,group_files
    from .halos import _snapshot_files
    from .data import run_epoch_contract
    from .flagship_products import native_history_hubble_rates
    from .assembly import orbital_assembly_events
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite native orbit allocation required')
    if request.get('schema')!='flagship-native-assembly-orbits-v1':raise ValueError('Complete native assembly orbit request required')
    if (type(request['maximum_saved_epochs']) is not int or request['maximum_saved_epochs']<2
            or isinstance(request['maximum_interval_myr'],bool) or not np.isfinite(request['maximum_interval_myr']) or request['maximum_interval_myr']<=0):
        raise ValueError('Explicit saved-epoch and cadence bounds required')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous orbital event evidence')
    graph,images,sources=native_orbit_inputs(graph_directory,request,memory_gib=memory_gib)
    run=Path(request['run_directory']).resolve();saved=snapshot_ids(run)
    if len(saved)>request['maximum_saved_epochs']:raise MemoryError('Native saved-epoch census exceeds request bound')
    clocks=[]
    for snap in saved:
        header,_=read_snapshot_header(run,snap);clocks.append((float(header['Time']),snap))
    clocks.sort()
    if any(b[0]<=a[0] for a,b in zip(clocks,clocks[1:])):raise ValueError('Duplicate/inconsistent actual native saved clocks')
    order={snap:i for i,(_,snap) in enumerate(clocks)};epochs=[];times=[];boxes=[]
    if len(request['object_keys'])!=len(request['host_keys']) or len(request['object_keys'])<2:raise ValueError('Aligned native object and host observations required')
    if len(request['object_keys'])**2*256>memory_gib*1024**3/2:raise MemoryError('Bounded orbital interval-pair search exceeds allocation')
    for object_key,host_key in zip(request['object_keys'],request['host_keys']):
        snap=object_key[2]
        if snap!=host_key[2] or snap not in order:raise ValueError('Object/host must share an actual native saved snapshot')
        header,_=read_snapshot_header(run,snap);a=float(header['Time']);epoch=run_epoch_contract(run,snap,a)
        if epoch['canonical_time']!='scale_factor' or epoch.get('cosmic_time_gyr') is None:raise ValueError('Bound cosmological native proper-time clock required')
        epochs.append(epoch);times.append(epoch['cosmic_time_gyr']*1000.)
        import h5py
        files=_snapshot_files(run,snap)
        for path in [*files,*group_files(run,snap)]:
            if graph['sources'].get(str(path.resolve()))!=file_sha256(path):raise ValueError('Graph omits original selected native snapshot/catalogue source')
        with h5py.File(files[0]) as f:boxes.append(float(f['Header'].attrs['BoxSize'])*a/float(f['Parameters'].attrs['HubbleParam']))
    for snap in saved:
        for path in _snapshot_files(run,snap):sources[str(path.resolve())]=file_sha256(path)
    times=np.asarray(times);hubble=native_history_hubble_rates(run,epochs,times,memory_gib=memory_gib,sources=sources)
    frame='original-native-global-proper:'+str(run)
    lines,raw=catalogue_orbit_worldlines(graph,images,request,epochs,times,boxes,hubble,saved_order=order,frame=frame)
    events=orbital_assembly_events(*lines,request['controls'])
    branch={tuple(k) for k in request['object_keys']};mergers=[]
    for edge in graph['edges']:
        if edge['kind']!='descendant' or tuple(edge['target']) not in branch or tuple(edge['source']) in branch:continue
        if edge['evidence'].get('origin')!='native_ancestry':raise ValueError('Assembly mergers require original native ancestry evidence')
        target=tuple(edge['target']);i=request['object_keys'].index(list(target))
        mergers.append({'source':edge['source'],'descendant':edge['target'],'time_myr':float(times[i]),
            'evidence':edge['evidence'],'meaning':'native progenitor joins selected descendant at saved epoch; coalescence time unresolved'})
    arrays={**raw,'times_myr':times,'hubble_kms_mpc':hubble,
        'event_time_myr':np.array([e['time_myr'] for e in events['events']]),
        'event_kind_code':np.array([{'pericenter':0,'infall':1,'apocenter':3}[e['kind']] for e in events['events']],np.int16),
        'event_brackets_myr':np.array([e['bracket_myr'] for e in events['events']],float).reshape(-1,2),
        'merger_time_myr':np.array([e['time_myr'] for e in mergers]),'merger_kind_code':np.full(len(mergers),2,np.int16)}
    units={k:'identity' if k.endswith('_ids') else 'proper kpc' if 'position_' in k else 'km/s' if 'velocity_' in k else
        'km/s/Mpc' if k=='hubble_kms_mpc' else 'Myr' if k.endswith('_myr') else 'dimensionless' for k in arrays}
    specs={k:{'units':u,'quantity':k,'missing_policy':'original gaps/quality masks retained; interpolation candidates are not certified events'} for k,u in units.items()}
    for kind in ('object','host'):specs[kind+'_ids']['identity_namespace']=request[kind+'_keys'][0][0]
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    result={**events,'schema':'flagship-native-assembly-orbital-events-v1','request':request,'epoch':{'row_epoch_contracts':epochs},
        'native_mergers':mergers,'worldlines':lines,'source_graph_fingerprint':graph['fingerprint'],
        'coordinate_definition':'proper catalogue position derivative = native peculiar velocity + H times recorded-image proper position; fixed native box origin',
        'position_velocity_consistency':'native catalogue centre and bulk velocity conventions retained; trajectory consistency is not independently certified',
        'uncertainty_status':'UNAVAILABLE_INTERPOLATION_ERROR; saved brackets retained, refinement required'}
    return export_numeric_audit(destination,result,arrays,specs,algorithm=__file__,context={'epoch':result['epoch']},source_hashes=sources)


def export_native_assembly_event_figures(product,destination,*,memory_gib=1.):
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite native event allocation required')
    report,arrays,sources=read_product(product,memory_gib=memory_gib/3)
    if report.get('schema')!='flagship-native-assembly-orbital-events-v1':raise ValueError('Native graph-bound assembly orbital event product required')
    sources.update(numeric_audit_source_bindings(report));epoch=report['epoch']
    return export_scientific_suite({'sources':{'events':str(Path(product).resolve())},'dpi':160,
        'maximum_panels_per_figure':1,'organized_outputs':True,'figures':[{'figure_id':'native-assembly-events',
            'family':'assembly','variant':'pericenter_infall_merger_events','title':'Orbital candidates and native ancestry joins',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'events','title':'0 pericenter; 1 infall; 2 ancestry join; 3 apocenter',
                'quality_status':'INTERPOLATION_UNCERTIFIED','selection':'exact native observations; original gaps and quality flags',
                'xlabel':'Actual proper time [Myr]','ylabel':'Event kind [dimensionless]','x_units':'Myr','y_units':'dimensionless',
                'series':[{'source':'events','kind':'scatter','x':x,'y':y,'label':label,'marker_size':12}
                    for x,y,label in (('event_time_myr','event_kind_code','reconstructed orbital candidates'),
                        ('merger_time_myr','merger_kind_code','native progenitor join at saved epoch'))]}]}]},destination,base=Path(product).resolve().parent,memory_gib=memory_gib)


def native_orbit_diagnostics(simulation, destination, settings):
    """Use real native stars and calibrated measured fields; invent no sample."""
    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic
    from gala.integrate import DOPRI853Integrator
    from core.stream_resimulation import require_backend
    from core.halos import ReconstructedHostPotential, verify_halo_structure, pinned_agama_backend
    from core.stream_common import json_safe
    from core.foundation import atomic_json
    from core.workflow import verify_completed_receipt
    # Reject malformed controls and bound retained orbit arrays before native
    # integration. The workspace allowance is an estimate, not a peak-RSS claim.
    import math
    import json
    for name in ('maximum_stellar_orbits', 'steps'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError('Positive integer orbit control required: '+name)
    for name in ('step_myr', 'relative_tolerance', 'absolute_tolerance', 'memory_gib'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive orbit control required: '+name)
    snapshots = settings['snapshot_indices']
    if not isinstance(snapshots, list) or not snapshots or any(type(i) is not int or i < 0 for i in snapshots) or len(set(snapshots)) != len(snapshots):
        raise ValueError('Distinct nonnegative native snapshot indices required')
    retained_bytes = (settings['steps']+1)*settings['maximum_stellar_orbits']*160
    if retained_bytes > settings['memory_gib']*1024**3/2:
        raise MemoryError('Retained orbit-array allowance exceeds half the diagnostic allocation')
    _, gala_binding = require_backend('gala')
    _, agama_binding = pinned_agama_backend()
    state_path = simulation/'workflow.json'
    state = json.loads(state_path.read_text())
    rows = []
    scale = (1*u.km/u.s).to_value(u.kpc/u.Myr)
    destination.mkdir()
    for stage_id, entry in state['stages'].items():
        generation = Path(entry['generation']).resolve()
        if not generation.is_relative_to((simulation/'generations').resolve()): raise ValueError('Foreign workflow generation')
        receipt_path = generation/'stage-receipt.json'
        if file_sha256(receipt_path) != entry['receipt_sha256']: raise ValueError('Native receipt binding changed')
        receipt = verify_completed_receipt(receipt_path)
        if receipt['kind'] != 'zoom_recovered_structure': continue
        for relative, digest in receipt['files_sha256'].items():
            if not relative.endswith('.json'): continue
            product = generation/relative
            meta = json.loads(product.read_text())
            if meta.get('schema') != 'native-halo-structure-v1': continue
            snapshot = meta['request']['snapshot']
            if snapshot not in settings['snapshot_indices']: continue
            if meta.get('source_origin') != 'native_snapshot': raise ValueError('Only original native stars are permitted')
            if meta['field_validation_status'] != 'PASS':
                rows.append({'source': str(product), 'status': 'UNRESOLVED_NATIVE_FIELD'}); continue
            verify_halo_structure(product)
            report, arrays, pins = read_product(product, memory_gib=settings['memory_gib']/2)
            required = ['parent_stars_'+name for name in ('offsets_proper_kpc', 'velocities_peculiar_km_s', 'masses_msun', 'particle_ids')]
            if any(name not in arrays for name in required) or len(arrays[required[0]]) == 0:
                rows.append({'source': str(product), 'status': 'NO_NATIVE_STARS_IN_MEASURED_SUPPORT'}); continue
            host = ReconstructedHostPotential.from_coefficients(product.parent, report['potential_coefficients'])
            for component in report['potential_coefficients']:
                coefficient = product.parent/component['file']
                pins[str(coefficient)] = component['sha256']
            x, v, mass, ids = [np.asarray(arrays[name]) for name in required]
            ids = exact_uint64_ids(ids, allow_empty=False)
            if x.shape != (len(ids), 3) or v.shape != x.shape or mass.shape != (len(ids),):
                raise ValueError('Native stellar positions, velocities, masses and IDs must align')
            if not all(np.isfinite(value).all() for value in (x, v, mass)) or np.any(mass <= 0):
                raise ValueError('Finite native stellar states and positive masses required')
            # Reconstructed fields use the original proper host coordinates;
            # remove the same explicitly recorded peculiar bulk velocity.
            bulk = np.asarray(report['request']['bulk_velocity_peculiar_km_s'], dtype=float)
            if bulk.shape != (3,) or not np.isfinite(bulk).all():
                raise ValueError('Finite host peculiar bulk velocity required')
            v = v-bulk
            order = np.argsort(ids, kind='stable')
            order = order[np.linspace(0, len(order)-1, min(len(order), settings['maximum_stellar_orbits']), dtype=int)]
            x, v, ids = x[order], v[order], ids[order]
            # The pinned Gala interface uses N x 3 for _energy but 3 x N for
            # _gradient. AGAMA always takes N x 3. Its Phi is (km/s)^2;
            # multiplying by K^2 gives kpc^2/Myr^2, and grad(Phi) = -a.
            # E = Phi + |v|^2/2 below stays in (km/s)^2. The displayed RMS
            # gives equal weight to selected tracers; it is not mass weighted.
            class NativeFrozenPotential(gp.PotentialBase):
                def _energy(self, q, t=0.):
                    return host.evaluate(np.asarray(q))[0]*scale**2
                def _gradient(self, q, t=0.):
                    return -host.evaluate(np.asarray(q).T)[1].T*scale**2
            potential = NativeFrozenPotential(units=galactic)
            initial = gd.PhaseSpacePosition(pos=x.T*u.kpc, vel=v.T*u.km/u.s)
            orbit = gp.Hamiltonian(potential).integrate_orbit(initial,
                dt=settings['step_myr']*u.Myr, n_steps=settings['steps'],
                Integrator=DOPRI853Integrator, Integrator_kwargs={'rtol': settings['relative_tolerance'], 'atol': settings['absolute_tolerance']})
            position = orbit.xyz.to_value(u.kpc).transpose(1, 2, 0)
            velocity = orbit.v_xyz.to_value(u.km/u.s).transpose(1, 2, 0)
            time = orbit.t.to_value(u.Myr)
            energy = np.stack([host.evaluate(q)[0]+.5*np.sum(w*w, axis=1) for q, w in zip(position, velocity)])
            if not all(np.isfinite(value).all() for value in (position, velocity, time, energy)):
                raise ArithmeticError('Nonfinite native frozen-field orbit diagnostic')
            drift = energy-energy[0]
            pins.update(gala_binding['installed_source_hashes']); pins.update(agama_binding['installed_asset_hashes'])
            pins.update({str(product): digest, str(receipt_path): file_sha256(receipt_path), str(state_path): file_sha256(state_path),
                         str(Path(__file__).resolve()): file_sha256(__file__)})
            values = {'particle_ids': ids, 'elapsed_time_myr': time, 'positions_kpc': position,
                      'velocities_kms': velocity, 'specific_energy_kms2': energy,
                      'rms_specific_energy_drift_kms2': np.sqrt(np.mean(drift*drift, axis=1))}
            units = {'particle_ids': 'identity', 'elapsed_time_myr': 'Myr', 'positions_kpc': 'proper kpc',
                     'velocities_kms': 'km/s', 'specific_energy_kms2': '(km/s)^2', 'rms_specific_energy_drift_kms2': '(km/s)^2'}
            out = destination/(stage_id+'-'+str(snapshot))
            out.mkdir()
            target = out/'native-orbits.json'
            result = {'schema': 'native-star-frozen-field-gala-v1', 'source_origin': 'native_snapshot',
                'source_product': str(product), 'actual_native_particle_count': len(ids),
                'epoch': report['source_context']['epoch'], 'gala_backend': gala_binding, 'agama_backend': agama_binding,
                'settings': settings,
                'initial_velocity_convention': 'native peculiar velocity minus recorded host peculiar bulk velocity; nonexpanding frozen host diagnostic, without Hubble-flow addition',
                'tracer_weighting': 'equal weight per selected native stellar particle; not stellar-mass weighting',
                'scope': 'Conditional frozen finite-source host orbit diagnostics; not an evolving cosmological trajectory, stream detection or movie input',
                'science_acceptance': 'NOT_ACCEPTED'}
            specifications = {key: {'units': units[key], 'quantity': key} for key in values}
            specifications['particle_ids']['identity_namespace'] = report['data_contract']['identity_namespaces']['native_particle']
            export_numeric_audit(target, json_safe(result), values, specifications, algorithm=__file__,
                context={'epoch': result['epoch']}, source_hashes=pins)
            figure = {'figure_id': 'native-star-orbit-energy', 'family': 'convergence', 'variant': 'timestep',
                'title': 'Actual native stars: frozen-field Gala orbit diagnostic', 'science_status': 'DIAGNOSTIC_ONLY_NOT_CONVERGENCE_ACCEPTANCE',
                'epoch': result['epoch'], 'panels': [{'panel_id': 'energy', 'title': 'AGAMA field; original measured stellar initial states',
                'quality_status': 'CONDITIONAL_FROZEN_FIELD', 'selection': result['scope'],
                'xlabel': 'Elapsed diagnostic time [Myr]', 'ylabel': 'RMS specific-energy drift [(km/s)^2]',
                'x_units': 'Myr', 'y_units': '(km/s)^2', 'series': [{'source': 'orbits', 'kind': 'line',
                'x': 'elapsed_time_myr', 'y': 'rms_specific_energy_drift_kms2', 'label': 'Actual integration residual; no pass threshold inferred'}]}]}
            export_scientific_suite({'sources': {'orbits': str(target)}, 'figures': [figure], 'dpi': 300,
                'maximum_panels_per_figure': 1, 'organized_outputs': True}, out/'figures', base=out, memory_gib=settings['memory_gib'])
            rows.append({'source': str(product), 'status': 'NATIVE_STARS_INTEGRATED', 'output': str(target), 'particles': len(ids)})
    atomic_json(destination/'inventory.json', {'products': rows, 'synthetic_particles_created': 0,
        'science_acceptance': 'NOT_ACCEPTED', 'note': 'Absent native stars or validated fields remain unresolved; no mock fallback'})
