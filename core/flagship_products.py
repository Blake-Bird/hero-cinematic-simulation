"""Connect native stream measurements to existing science and display products.

These adapters reuse the stream estimators and immutable numeric format. They
do not identify arbitrary stellar populations as streams or certify resolution.
"""
from pathlib import Path
import json
import numpy as np

from .foundation import atomic_json, file_sha256
from core.stream_common import read_product, publish_stream_product, json_safe


def require_native_worldline_units(report):
    """Native history adapters accept physical arrays only, never encoded v."""
    expected={'particle_ids':'identity','position_kpc':'proper kpc','velocity_kms':'km / s',
              'mass_msun':'Msun','times_myr':'Myr','periodic_box_proper_kpc':'proper kpc'}
    contracts=report['numeric_data']['array_contract']['datasets']
    for name,units in expected.items():
        if name not in contracts or contracts[name]['units']!=units:
            raise ValueError('Native worldline boundary unit mismatch: '+name)
    if contracts['particle_ids'].get('identity_namespace')!=report['namespace']:
        raise ValueError('Native worldline identity namespace differs from dataset contract')


def preflight_flagship_profile(request, destination, *, output_directory):
    """Budget actual retained geometries; retain failures before any launch.

    Retained IC counts are evidence about those files, not a census of a future
    corrected generation. Native geometry planning remains mandatory at launch.
    """
    import math
    import shutil
    import h5py
    from .zoom import estimate_mesh_resources, _configuration, _music_noise_files, _frozen_noise_workspace
    from .foundation import RHO_CRIT_MSUN_H_MPC_H3
    from core.resources import initial_condition_census, estimate_run_resources, periodic_image_resource_allowance
    from .flagship_profiles import profile_input_bindings, validate_case_configuration,case_parent_bindings,validate_branch_allocation,validate_profile_resources
    bound_values, common_pins = profile_input_bindings(request)
    if request.get('schema') != 'flagship-run-profile-v1':
        raise ValueError('Explicit flagship run profile required')
    resources = request['resources']
    validate_profile_resources(resources)
    validate_branch_allocation(request,bound_values)
    probe = Path(output_directory).resolve()
    while not probe.exists(): probe = probe.parent
    free = shutil.disk_usage(probe).free
    budget = min(resources['disk_gib'] * 1024**3, free - resources['disk_margin_gib'] * 1024**3)
    if not isinstance(request.get('zoom_cases'), list) or not request['zoom_cases']:
        raise ValueError('At least one actual zoom geometry required for resource preflight')
    from . import flagship_profiles
    rows = []; failures = []; sources = {**common_pins,str(Path(__file__).resolve()):file_sha256(__file__),
        str(Path(__file__).with_name('foundation.py').resolve()):file_sha256(Path(__file__).with_name('foundation.py')),
        str(Path(flagship_profiles.__file__).resolve()):file_sha256(flagship_profiles.__file__)}; seen = set()
    common_parent=None
    for case in request['zoom_cases']:
        if case['id'] in seen: raise ValueError('Duplicate zoom profile case')
        seen.add(case['id'])
        binding = case['generation']; path = Path(binding['path']).resolve()
        if file_sha256(path) != binding['sha256']: raise ValueError('Retained generation changed: ' + case['id'])
        generation = json.loads(path.read_text()); sources[str(path)] = binding['sha256']
        parent,parent_pins=case_parent_bindings(case,generation)
        if common_parent is not None and parent!=common_parent:
            raise ValueError('Zoom cases change the frozen common parent phases')
        common_parent=parent;sources.update(parent_pins)
        config = Path(generation['source_configuration']['path'])
        if file_sha256(config) != generation['source_configuration']['sha256']:
            raise ValueError('Retained generation configuration changed')
        sources[str(config)] = file_sha256(config)
        cfg = _configuration(config); noise = _music_noise_files(cfg, config)
        validate_case_configuration(case,cfg,request,bound_values['cosmology'])
        for asset in noise.values(): sources[str(asset)] = file_sha256(asset)
        estimate = estimate_mesh_resources(generation['native_resource_plan']['native_geometry'])
        estimate['estimated_memory_bytes'] += _frozen_noise_workspace(cfg, noise)
        ic = path.parent / 'ics_gadget.hdf5'
        expected = generation['output_hashes']['ics_gadget.hdf5'] if 'output_hashes' in generation else generation['structural_audit']['ic_sha256']
        if file_sha256(ic) != expected: raise ValueError('Retained IC changed')
        sources[str(ic)] = expected
        census = initial_condition_census([ic])
        counts = {}; masses = {}
        with h5py.File(ic) as native:
            table = native['Header'].attrs['MassTable']
            for species in range(6):
                group = native.get('PartType%d' % species)
                if group is None: continue
                count = len(group['ParticleIDs']); counts[str(species)] = count
                total = 0.
                for first in range(0, count, 100000):
                    values = group['Masses'][first:first+100000] if 'Masses' in group else np.full(min(100000, count-first), table[species])
                    if not np.isfinite(values).all() or np.any(values <= 0): raise ValueError('Positive native census masses required')
                    total += float(np.sum(values))
                masses[str(species)] = total
        particle_bound = estimate['particle_count_upper_bound']
        if sum(counts.values()) > particle_bound: raise ValueError('Actual particle count exceeds geometry upper bound')
        answers_binding = request['combined_physics_answers']; answers_path = Path(answers_binding['path'])
        if file_sha256(answers_path) != answers_binding['sha256']: raise ValueError('Physics answers changed')
        answers = bound_values['combined_physics_answers']; sources[str(answers_path.resolve())] = answers_binding['sha256']
        evolution_plan = estimate_run_resources({'simulation':answers, 'black_holes':answers['black_holes']},
            resources['saved_epochs'] + resources['checkpoint_copies'], particle_counts=census['particles_by_type'])
        image_bytes=periodic_image_resource_allowance(particle_bound)
        evolution = max(evolution_plan['memory_bytes'], particle_bound * resources['evolution_bytes_per_particle']+image_bytes)
        storage = estimate['estimated_disk_bytes'] + max(evolution_plan['disk_bytes'], (particle_bound * resources['snapshot_bytes_per_particle']+image_bytes) * (resources['saved_epochs'] + resources['checkpoint_copies']))
        # Explicit provisional allowance for native MERGERTREE particle/rank,
        # matching and catalogue-link workspaces; measured RSS remains required.
        tree_memory=particle_bound*512 if answers.get('native_mergertree',False) else 0
        tree_disk=particle_bound*64*resources['saved_epochs'] if answers.get('native_mergertree',False) else 0
        evolution+=tree_memory;storage+=tree_disk
        peak = max(estimate['estimated_memory_bytes'], evolution)
        h = cfg.getfloat('cosmology', 'H0') / 100.
        volume = cfg.getfloat('setup', 'boxlength')**3
        cell_volume = volume / 2**(3*cfg.getint('setup', 'levelmax'))
        mean_mass = {'gas': RHO_CRIT_MSUN_H_MPC_H3 * cfg.getfloat('cosmology', 'Omega_b') * cell_volume / h,
                     'dm': RHO_CRIT_MSUN_H_MPC_H3 * (cfg.getfloat('cosmology', 'Omega_m')-cfg.getfloat('cosmology', 'Omega_b')) * cell_volume / h}
        rows.append({'id':case['id'], 'actual_retained_counts':counts, 'actual_total_mass_native':masses,
                     'fine_mean_particle_mass_msun':mean_mass, 'geometry_estimate':estimate,
                     'evolution_estimate':evolution_plan, 'census':census,
                     'evolution_memory_estimate_bytes':evolution, 'peak_memory_estimate_bytes':peak,
                     'native_mergertree_provisional_memory_bytes':tree_memory,'native_mergertree_provisional_disk_bytes':tree_disk,
                     'storage_estimate_bytes':storage, 'census_scope':'retained IC; regenerate and recount before evolution'})
        if peak > resources['memory_gib']*1024**3: failures.append(case['id'] + ': memory allocation exceeded')
    storage = sum(row['storage_estimate_bytes'] for row in rows)
    if storage > budget: failures.append('Campaign storage exceeds declared/available disk after reserve')
    if any(file_sha256(path) != checksum for path, checksum in sources.items()):
        raise ValueError('Resource preflight inputs changed during census')
    result = {'schema':'flagship-profile-resource-preflight-v1', 'profile':request['name'],
              'status':'FAIL' if failures else 'ESTIMATE_PASS', 'failures':failures,
              'resources':resources, 'available_disk_bytes':free, 'usable_disk_budget_bytes':budget,
              'campaign_storage_estimate_bytes':storage, 'cases':rows, 'source_hashes':sources,
              'execution_policy':'sequential zoom cases; concurrent execution requires a summed memory budget',
              'limitations':['Geometry estimates are planning assumptions, not measured peak RSS',
                             'Retained source/binary execution is historical; current native planner and handoff gate remain required'],
              'science_acceptance':'NOT_ACCEPTED'}
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Keep previous resource evidence')
    atomic_json(destination, result)
    if failures: raise MemoryError('; '.join(failures))
    return result


def catalogue_native_stream_epoch(worldlines, request, destination, *, memory_gib):
    """Connect measured native stellar histories to existing debris classification."""
    from core.stream_catalogue import trace_progenitors, build_stream_catalogue
    report, values, sources = read_product(worldlines, memory_gib=memory_gib/3)
    if report.get('schema')!='measured-native-stream-worldlines-v1' or report['request']['species']!=4:
        raise ValueError('Native type-4 stellar worldlines required for stellar catalogue')
    require_native_worldline_units(report)
    if request.get('schema')!='flagship-native-stream-catalogue-v1':
        raise ValueError('Explicit native stellar catalogue request required')
    if request['controls']['maximum_neighbor_edges']*32>memory_gib*1024**3/3:
        raise MemoryError('Stellar catalogue neighbor edge budget exceeds allocation')
    ordinal=request['epoch_index']
    if type(ordinal) is not int or not 0<=ordinal<len(report['epochs']):
        raise ValueError('Select an actual saved stellar epoch')
    center=np.asarray(request['chart_center_proper_kpc'],float)
    bulk=np.asarray(request['velocity_origin_peculiar_kms'],float)
    if center.shape!=(3,) or bulk.shape!=(3,) or not np.isfinite([center,bulk]).all():
        raise ValueError('Explicit finite stellar catalogue chart and peculiar origin required')
    x=values['position_kpc'][ordinal]-center
    if report['periodic']:
        box=float(values['periodic_box_proper_kpc'][ordinal]);fraction=request['maximum_chart_box_fraction']
        if isinstance(fraction,bool) or not 0<fraction<.5:raise ValueError('Bounded minimum-image chart required')
        x-=box*np.floor(x/box+.5)
        if len(x) and np.max(np.linalg.norm(x,axis=1))>fraction*box:
            raise ValueError('Stellar catalogue population exceeds declared spatial chart')
    sample={'position_kpc':x,'velocity_kms':values['velocity_kms'][ordinal]-bulk,
            'mass_msun':values['mass_msun'][ordinal],'particle_ids':values['particle_ids'],'namespace':report['namespace']}
    ancestry=None;calibration=None
    graph=request.get('identity_graph')
    if graph:
        path=Path(graph['path']).resolve()
        if file_sha256(path/'report.json')!=graph['sha256']:raise ValueError('Stellar ancestry graph changed')
        snapshot=report['epochs'][ordinal].get('snapshot_storage_index')
        if type(snapshot) is not int:raise ValueError('Ancestry requires actual saved snapshot label')
        ancestry=trace_progenitors(sample['particle_ids'],report['namespace'],snapshot,path,memory_gib=memory_gib/3)
        sources.update({str(path/name):file_sha256(path/name) for name in ('report.json','index.sqlite')})
    if request.get('calibration'):
        binding=request['calibration'];path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('Stellar classifier calibration changed')
        if path.stat().st_size*8>memory_gib*1024**3/3:raise MemoryError('Classifier calibration exceeds intake allocation')
        calibration=json.loads(path.read_text());calibration=calibration.get('classification_calibration',calibration)
        sources[str(path)]=binding['sha256']
    result,arrays=build_stream_catalogue(sample,request['controls'],ancestry=ancestry,calibration=calibration)
    result.update(native_worldlines=str(Path(worldlines).resolve()),native_worldlines_sha256=file_sha256(worldlines),
                  epoch=report['epochs'][ordinal],request=request,science_acceptance='NOT_ACCEPTED',
                  origin='declared proper spatial chart center',
                  periodic_wrapping='single-epoch minimum-image chart; temporal winding unavailable' if report['periodic'] else 'isolated native positions minus declared origin',
                  velocity_convention='physical peculiar km/s minus declared peculiar origin velocity',
                  limitations='Coherent debris proposals; missing ancestry/calibration remains unresolved; no inferred stellar birth or stripping history')
    units={key:('identity' if key=='particle_ids' else 'proper kpc' if key=='position_kpc' else
                'km / s' if key=='velocity_kms' else 'Msun' if key=='mass_msun' else 'dimensionless') for key in arrays}
    # Classifier feature columns have heterogeneous units; their definitions
    # remain explicitly recorded by the established catalogue feature schema.
    result['feature_columns']=list(result['features'])
    result['feature_units']='Dimensionless features; natural logs of width/(1 proper kpc), dispersion/(1 km/s), linear density/(1 Msun/proper kpc); count and shape ratios dimensionless'
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    return publish_stream_product(destination,result,arrays,units,sources=sources,
                                  namespace=report['namespace'],epoch=report['epochs'][ordinal])


def measure_native_stream_epoch(worldlines, request, destination, *, memory_gib):
    """Measure one explicitly selected wrap at an actual native saved epoch."""
    from core.stream_tracks import (
        fit_uncertainty_aware_principal_curve, project_to_arclength_transverse_frame,
        choose_bins_from_resolution_and_sampling, measure_local_observables,
        bootstrap_track,
    )
    if request.get('schema') != 'flagship-native-stream-epoch-v1':
        raise ValueError('Explicit native stream epoch request required')
    if not isinstance(request.get('selection'), dict) or not request['selection'].get('description'):
        raise ValueError('Declare the selected population and membership limitations')
    report, values, sources = read_product(worldlines, memory_gib=memory_gib / 3)
    if report.get('schema') != 'measured-native-stream-worldlines-v1':
        raise ValueError('Original native measured stream worldlines required')
    require_native_worldline_units(report)
    ordinal = request['epoch_index']
    if type(ordinal) is not int or not 0 <= ordinal < len(values['times_myr']):
        raise ValueError('Select an actual saved native epoch')
    x = values['position_kpc'][ordinal].copy()
    v = values['velocity_kms'][ordinal].copy()
    m = values['mass_msun'][ordinal].copy()
    ids = values['particle_ids']
    if ids.dtype != np.dtype('uint64') or len(np.unique(ids)) != len(ids):
        raise ValueError('Exact distinct native tracer identities required')
    if x.shape != (len(ids), 3) or v.shape != x.shape or m.shape != (len(ids),):
        raise ValueError('Native epoch must have matching positions, velocities and masses')
    if not np.isfinite(x).all() or not np.isfinite(v).all() or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError('Finite native vectors and positive tracer masses required')
    if request.get('catalogue_selection'):
        binding=request['catalogue_selection'];path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('Selected stellar catalogue changed')
        catalogue,selected,bindings=read_product(path,memory_gib=memory_gib/3)
        if (catalogue.get('schema')!='tidal-stream-catalogue-v1' or catalogue['namespace']!=report['namespace']
                or catalogue.get('native_worldlines_sha256')!=file_sha256(worldlines)
                or catalogue.get('epoch')!=report['epochs'][ordinal]):
            raise ValueError('Catalogue selection belongs to a different native population or epoch')
        candidate=binding['candidate_id']
        if type(candidate) is not int or not 0<=candidate<len(catalogue['candidates']):
            raise ValueError('Select an actual catalogue candidate')
        chosen=selected['particle_ids'][selected['candidate_id']==candidate]
        from .data import exact_id_join
        rows=exact_id_join(chosen,ids,requested_namespace=report['namespace'],available_namespace=report['namespace'])
        x,v,m,ids=x[rows],v[rows],m[rows],ids[rows]
        sources.update(bindings)
        request={**request,'selection':{**request['selection'],'catalogue_candidate':catalogue['candidates'][candidate],
                    'classification_policy':'Measured candidate morphology; classification/support remains as recorded'}}
    if len(ids) * (request['curve_controls']['ordering_neighbors'] * 192 + 2048) + request['curve_controls']['grid_points'] * 4096 > memory_gib * 1024 ** 3:
        raise MemoryError('Principal-curve ordering exceeds the declared workspace')
    # An explicit center selects a single minimum-image spatial chart. This is
    # not a temporal winding reconstruction or an inferred progenitor center.
    center = np.asarray(request['chart_center_proper_kpc'], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Declare a finite proper Cartesian chart center')
    x -= center
    if report['periodic']:
        box = float(values['periodic_box_proper_kpc'][ordinal])
        fraction = request['maximum_chart_box_fraction']
        if isinstance(fraction, bool) or not 0 < fraction < .5:
            raise ValueError('Periodic chart support must be less than half a box')
        x -= box * np.floor(x / box + .5)
        if len(x) and np.max(np.linalg.norm(x, axis=1)) > fraction * box:
            raise ValueError('Selected wrap exceeds the declared minimum-image chart')
    velocity_origin = np.asarray(request['velocity_origin_peculiar_kms'], dtype=float)
    if velocity_origin.shape != (3,) or not np.isfinite(velocity_origin).all():
        raise ValueError('Declare the peculiar velocity origin')
    v -= velocity_origin
    sample = dict(position_kpc=x, velocity_kms=v, mass_msun=m,
                  particle_ids=ids, namespace=report['namespace'])
    from core.stream_common import integer
    minimum=integer(request['curve_controls']['minimum_track_particles'],'minimum_track_particles',10)
    bin_minimum=integer(request['bin_controls']['minimum_count'],'minimum_count',3)
    if len(ids)<minimum or len(ids)//bin_minimum<2:
        unresolved={'schema':'stream-coordinates-v1','epoch':report['epochs'][ordinal],'request':request,
            'selection':request['selection'],'native_worldlines':str(Path(worldlines).resolve()),
            'native_worldlines_sha256':file_sha256(worldlines),'coordinate_kind':'intrinsic',
            'origin':'declared minimum-image spatial chart',
            'periodic_wrapping':'single-epoch spatial chart; temporal winding unavailable',
            'velocity_convention':'physical peculiar km/s minus declared peculiar origin velocity',
            'wraps':[{'wrap_id':0,'status':'UNRESOLVED','count':len(ids),
                'reason':'INSUFFICIENT_PARTICLES_FOR_TRACK_OR_TWO_BINS','required_track_particles':minimum,
                'minimum_particles_per_bin':bin_minimum}], 'science_acceptance':'NOT_ACCEPTED'}
        values={key:sample[key] for key in ('position_kpc','velocity_kms','mass_msun','particle_ids')}
        values['wrap_id']=np.zeros(len(ids),dtype=np.int64)
        units={'position_kpc':'proper kpc','velocity_kms':'km / s','mass_msun':'Msun','particle_ids':'identity','wrap_id':'dimensionless'}
        sources[str(Path(__file__).resolve())]=file_sha256(__file__)
        return publish_stream_product(destination,unresolved,values,units,sources=sources,
            namespace=report['namespace'],epoch=report['epochs'][ordinal])
    curve = fit_uncertainty_aware_principal_curve(sample, request['curve_controls'])
    coordinates = project_to_arclength_transverse_frame(sample, curve)
    edges = choose_bins_from_resolution_and_sampling(coordinates['s_kpc'], **request['bin_controls'])
    controls = request['measurement_controls']
    # This entry point measures the explicitly selected population. Probabilistic
    # membership/selection corrections belong to the existing campaign adapter.
    if 'membership_probability' in controls or 'selection_probability' in controls:
        raise ValueError('Use the stream campaign adapter for probabilistic membership corrections')
    local_report, local = measure_local_observables(sample, coordinates, edges, controls)
    covariance, uncertainty = bootstrap_track(
        sample, request['curve_controls'], curve,
        replicates=request['track_bootstrap_replicates'], seed=controls['seed'])
    measured = {
        **sample, **coordinates, **local,
        'curve_grid': curve.grid, 'curve_parameter': curve.parameter,
        'curve_arclength_kpc': curve.arclength, 'curve_position_kpc': curve.position,
        'curve_tangent': curve.tangent, 'curve_normal': curve.normal,
        'curve_binormal': curve.binormal, 'curve_support': curve.support,
        'curve_fit_variance_kpc2': curve.fit_variance_kpc2,
        'curve_track_covariance_kpc2': covariance,
    }
    measured.pop('namespace')
    prefix = 'wrap_0000_'
    arrays = {prefix + key: value for key, value in measured.items()}
    arrays.update({key: sample[key] for key in ('particle_ids', 'position_kpc', 'velocity_kms', 'mass_msun')})
    arrays['wrap_id'] = np.zeros(len(ids), dtype=np.int64)
    units = {}
    for name, value in arrays.items():
        if name.endswith('particle_ids'): unit = 'identity'
        elif name.endswith(('wrap_id', 'count', 'supported', 'support', 'candidate', 'significance', 'tangent', 'normal', 'binormal', 'window')): unit = 'dimensionless'
        elif name.endswith('msun_kpc'): unit = 'Msun / proper kpc'
        elif name.endswith('kms2'): unit = '(km / s)^2'
        elif name.endswith('kms'): unit = 'km / s'
        elif name.endswith('kpc_inverse'): unit = '1 / proper kpc'
        elif name.endswith('kpc3'): unit = 'proper kpc^3'
        elif name.endswith('kpc2'): unit = 'proper kpc^2'
        elif name.endswith('kpc') or name.endswith(('curve_grid', 'curve_parameter')): unit = 'proper kpc'
        elif name.endswith('msun'): unit = 'Msun'
        else: raise ValueError('Missing stream measurement unit declaration: ' + name)
        units[name] = unit
    epoch = report['epochs'][ordinal]
    result = {
        'schema': 'stream-coordinates-v1', 'epoch': epoch, 'request': request,
        'native_worldlines': str(Path(worldlines).resolve()),
        'native_worldlines_sha256': file_sha256(worldlines),
        'selection': request['selection'], 'origin': 'declared minimum-image spatial chart',
        'coordinate_kind': 'intrinsic',
        'velocity_convention': 'native peculiar velocity minus declared peculiar velocity origin',
        'periodic_wrapping': 'single-epoch minimum-image chart; temporal winding unavailable' if report['periodic'] else 'isolated native positions minus declared origin',
        'limitations': 'Selected single-wrap morphology; no independent stream detection, progenitor ancestry, causal encounter or convergence established',
        'wraps': [{'wrap_id': 0, 'status': 'MEASURED', 'count': len(ids),
                   'curve_fit': curve.diagnostics, 'observables': local_report,
                   'track_uncertainty': uncertainty}],
    }
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    from . import source_path
    sources[str(source_path('stream_tracks'))] = file_sha256(source_path('stream_tracks'))
    return publish_stream_product(destination, result, arrays, units, sources=sources,
                                  namespace=report['namespace'], epoch=epoch)


def native_history_hubble_rates(root, epochs, times, *, memory_gib, sources):
    """One clock-bound expansion conversion for native trajectory adapters."""
    from .cosmology import run_proper_time_clock, class_background_table, interpolate_proper_time
    from .data import isolated_epoch_contract, native_unit_factor
    from .foundation import C_SI
    times=np.asarray(times)
    if (times.dtype.kind not in 'fiu' or times.ndim!=1 or not len(times) or len(epochs)!=len(times)
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        raise ValueError('Aligned strictly increasing actual physical history times required')
    root=Path(root).resolve();manifest=root/'manifest.json'
    if manifest.stat().st_size*8>memory_gib*1024**3/4:raise MemoryError('Native clock manifest exceeds intake allocation')
    def bind_clock_source(path,digest):
        name=str(Path(path).resolve())
        if any(str(Path(bound).resolve())==name and expected!=digest for bound,expected in sources.items()):
            raise ValueError('Native expansion clock conflicts with its history source binding')
        if file_sha256(name)!=digest:raise ValueError('Native clock source changed')
        sources[name]=digest
    bind_clock_source(manifest,file_sha256(manifest))
    kinds={epoch['canonical_time'] for epoch in epochs}
    if len(kinds)!=1:raise ValueError('Native encounter history mixes clock representations')
    if kinds=={'scale_factor'}:
        native_manifest=json.loads(manifest.read_text())
        background_entry=native_manifest.get('input_artifacts',{}).get('boltzmann_background')
        if not isinstance(background_entry,dict):raise ValueError('Native run must bind its expansion source')
        background_path=Path(background_entry['path']).expanduser()
        if not background_path.is_absolute():background_path=root/background_path
        if background_path.stat().st_size*24>memory_gib*1024**3/4:
            raise MemoryError('Native expansion table and interpolation workspace exceed allocation')
        clock=run_proper_time_clock(root)
        if clock[2]['status']!='VERIFIED':raise ValueError('Hashed native expansion clock required for encounter derivatives')
        background=Path(clock[2]['path']);columns,table=class_background_table(background)
        if 'H [1/Mpc]' not in columns:raise ValueError('CLASS H [1/Mpc] definition required')
        scale=1/(1+table[:,columns['z']]);order=np.argsort(scale);scale=scale[order]
        hubble=table[order,columns['H [1/Mpc]']]*(C_SI/1000.)
        if not np.isfinite(scale).all() or np.any(scale<=0) or np.any(np.diff(scale)<=0):
            raise ValueError('Finite positive unique CLASS expansion epochs required')
        if not np.isfinite(hubble).all() or np.any(hubble<=0):raise ValueError('Positive finite measured expansion history required')
        rates=[]
        for index,epoch in enumerate(epochs):
            a=epoch['actual_scale_factor']
            native_unit_factor('catalogue_velocity_peculiar_kms',scale_factor=a,hubble_param=1.)
            if not scale[0]<=a<=scale[-1]:raise ValueError('Native expansion interpolation outside recorded CLASS support')
            t=1000*interpolate_proper_time(clock[0],clock[1],a)
            if not np.isclose(times[index],t,rtol=1e-12,atol=1e-9):raise ValueError('Native encounter clock differs from bound CLASS clock')
            rates.append(float(np.interp(a,scale,hubble)))
        bind_clock_source(background,clock[2]['sha256'])
    elif kinds=={'isolated_native_elapsed_time'}:
        unit=None
        for index,epoch in enumerate(epochs):
            declared=epoch.get('time_unit_gyr')
            if declared is None:raise ValueError('Isolated history requires a declared native time unit')
            actual=isolated_epoch_contract(epoch['native_time_code'],time_unit_gyr=declared)
            elapsed=actual['elapsed_time_gyr']
            if (not np.isfinite(elapsed) or not np.isfinite(epoch.get('elapsed_time_gyr',np.nan))
                    or not np.isclose(epoch['elapsed_time_gyr'],elapsed,rtol=1e-12,atol=1e-12)
                    or not np.isclose(times[index],1000*elapsed,rtol=1e-12,atol=1e-9)):
                raise ValueError('Isolated history times differ from the actual native elapsed-time clock')
            if unit is not None and not np.isclose(unit,declared,rtol=1e-12,atol=0):
                raise ValueError('Isolated history changes its native time unit')
            unit=declared
        rates=[0.]*len(times)
    else:raise ValueError('Unsupported encounter clock representation')
    return np.asarray(rates)


def native_encounter_frame(product, request, destination, *, memory_gib):
    """Convert recorded native positions/peculiar velocities once for Hermite intake."""
    from .data import proper_position_derivative_kms, exact_uint64_ids, apply_recorded_periodic_images, numeric_audit_source_bindings
    from .cosmology import run_proper_time_clock, class_background_table, interpolate_proper_time
    report,arrays,sources=read_product(product,memory_gib=memory_gib/4)
    def bind_sources(additional):
        for source,digest in additional.items():
            if source in sources and sources[source]!=digest:
                raise ValueError('Native frame inputs contain conflicting source bindings')
            sources[source]=digest
    bind_sources(numeric_audit_source_bindings(report))
    if report.get('schema')!='measured-native-stream-worldlines-v1' or request.get('schema')!='native-encounter-frame-v1':
        raise ValueError('Native measured history and explicit encounter frame request required')
    require_native_worldline_units(report)
    x=arrays['position_kpc'].copy();v=arrays['velocity_kms'];times=arrays['times_myr'];ids=exact_uint64_ids(arrays['particle_ids'])
    n=len(times)
    if x.shape!=(n,len(ids),3) or v.shape!=x.shape or len(report['epochs'])!=n:
        raise ValueError('Aligned native trajectory and epoch arrays required')
    if x.nbytes*12>memory_gib*1024**3:raise MemoryError('Native frame workspace exceeds allocation')
    if not isinstance(request.get('frame'),str) or not request['frame'].strip():raise ValueError('Named fixed Cartesian frame required')
    origin=np.asarray(request['origin_proper_kpc'],float);bulk=np.asarray(request['origin_velocity_peculiar_kms'],float)
    if origin.shape!=(n,3) or bulk.shape!=(n,3) or not np.isfinite([origin,bulk]).all():
        raise ValueError('Origin position/peculiar velocity required at each actual saved epoch')
    if report['periodic']:
        binding=request['recorded_images'];path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('Recorded periodic images changed')
        image_report,image_arrays,image_sources=read_product(path,memory_gib=memory_gib/4)
        images=image_arrays['image_offsets']
        if (image_report.get('schema')!='native-worldline-periodic-images-v1'
                or image_report.get('history_sha256')!=file_sha256(product)):
            raise ValueError('Recorded periodic images must bind this exact native history')
        contracts=image_report['numeric_data']['array_contract']['datasets']
        for name,unit in {'image_offsets':'dimensionless','particle_ids':'identity','times_myr':'Myr','periodic_box_proper_kpc':'proper kpc'}.items():
            if contracts.get(name,{}).get('units')!=unit:
                raise ValueError('Recorded image boundary unit mismatch: '+name)
        if contracts['particle_ids'].get('identity_namespace')!=report['namespace']:
            raise ValueError('Recorded image dataset identity namespace differs from native history')
        if image_report.get('namespace')!=report['namespace'] or images.dtype.kind not in 'iu' or images.shape!=x.shape:
            raise ValueError('Exact namespace-qualified integer periodic images required for every saved particle')
        if not np.array_equal(exact_uint64_ids(image_arrays['particle_ids']),ids) or not np.array_equal(image_arrays['times_myr'],times):
            raise ValueError('Recorded image identities/times differ from native history')
        if not np.array_equal(image_arrays['periodic_box_proper_kpc'],arrays['periodic_box_proper_kpc']):
            raise ValueError('Recorded image boxes differ from actual native history')
        support=image_arrays.get('image_supported')
        if support is None or support.dtype.kind!='b' or support.shape!=x.shape[:-1] or not support.all():
            raise ValueError('Native periodic image history contains unavailable or discontinuous counter support')
        x=apply_recorded_periodic_images(x,images,box_proper_kpc_by_epoch=arrays['periodic_box_proper_kpc'])
        bind_sources(image_sources)
        bind_sources(numeric_audit_source_bindings(image_report))
    rates=native_history_hubble_rates(report['native_run_directory'],report['epochs'],times,memory_gib=memory_gib,sources=sources)
    derivative=np.array([proper_position_derivative_kms(x[i],v[i],hubble_kms_mpc=rates[i],
        origin_proper_kpc=origin[i],origin_velocity_peculiar_kms=bulk[i]) for i in range(n)])
    output={**arrays,'position_kpc':x-origin[:,None,:],'velocity_kms':derivative,
            'origin_proper_kpc':origin,'origin_velocity_peculiar_kms':bulk,'hubble_kms_mpc':np.asarray(rates)}
    if report['periodic']:output['recorded_image_offsets']=images
    units={'particle_ids':'identity','position_kpc':'proper kpc','velocity_kms':'km / s','mass_msun':'Msun',
           'times_myr':'Myr','periodic_box_proper_kpc':'proper kpc','origin_proper_kpc':'proper kpc',
           'origin_velocity_peculiar_kms':'km / s','hubble_kms_mpc':'km / s / Mpc','recorded_image_offsets':'dimensionless'}
    if set(output)-set(units):raise ValueError('Unknown native encounter array semantics')
    result={'schema':'native-encounter-frame-history-v1','namespace':report['namespace'],'frame':request['frame'],
            'source_origin':'synthetic_control' if report.get('fixture') or report.get('source_origin')=='synthetic_control' else 'native_snapshot_and_catalogue',
            'epochs':report['epochs'],'request':request,'source_history':str(Path(product).resolve()),
            'source_history_sha256':file_sha256(product),'origin':'declared epoch-dependent frame origin; fixed Cartesian axes',
            'velocity_convention':'proper position derivative in km/s; relative peculiar velocity plus H*r',
            'periodic_wrapping':'explicit recorded periodic images applied once; no temporal winding inferred',
            'periodic_images_applied':bool(report['periodic']),
            'interpolation_acceptance':'NOT_CERTIFIED; retain gaps and request refinement', 'science_acceptance':'NOT_ACCEPTED'}
    from .foundation import canonical_digest
    result['frame_definition_sha256']=canonical_digest({'frame':request['frame'],'times_myr':times.tolist(),
        'origin_proper_kpc':origin.tolist(),'origin_velocity_peculiar_kms':bulk.tolist(),
        'hubble_kms_mpc':rates.tolist(),'axes':'fixed Cartesian native axes'})
    if request.get('worldline_selections') is not None:
        from core.stream.trajectories import worldlines_from_history
        selections=request['worldline_selections']
        if not isinstance(selections,list) or not selections or len(selections)>len(ids):
            raise ValueError('Bounded explicit native encounter worldline selections required')
        for selection in selections:
            flags=selection.get('valid_intervals')
            if not isinstance(flags,list) or len(flags)!=n-1 or any(type(value) is not bool for value in flags):
                raise ValueError('Native encounter selections require explicit saved-interval support flags')
        result['source_hashes']=sources
        result['worldlines']=worldlines_from_history(result,output,selections,frame=request['frame'],
            maximum_interval_myr=request['maximum_interval_myr'])
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Native encounter frame inputs changed')
    return publish_stream_product(destination,result,output,units,sources=sources,namespace=report['namespace'],
        epoch={'representation':'actual saved native history','epochs':report['epochs']})


def measure_native_encounters(product, request, destination, *, memory_gib):
    """Join converted native stream/perturber histories to the existing search."""
    from core.stream.trajectories import reconstruct_encounters
    stream,stream_arrays,sources=read_product(product,memory_gib=memory_gib/4)
    if request.get('schema')!='native-encounter-catalogue-v1':raise ValueError('Explicit native encounter catalogue request required')
    binding=request['perturber_history'];path=Path(binding['path']).resolve()
    if file_sha256(path)!=binding['sha256']:raise ValueError('Native perturber history changed')
    perturbers,perturber_arrays,more=read_product(path,memory_gib=memory_gib/4);sources.update(more)
    for record in (stream,perturbers):
        if record.get('schema')!='native-encounter-frame-history-v1' or not record.get('worldlines'):
            raise ValueError('Converted native histories with explicit worldline selections required')
    if stream['frame']!=perturbers['frame'] or stream['frame_definition_sha256']!=perturbers['frame_definition_sha256']:
        raise ValueError('Stream and perturbers require the same clock-bound origin trajectory and fixed Cartesian frame')
    controls=request['controls']
    if controls['maximum_segments']*1024+controls['maximum_pairs']*512>memory_gib*1024**3/2:
        raise MemoryError('Encounter search bound exceeds allocated workspace')
    for record in (stream,perturbers):
        for line in record['worldlines']:
            certificate=line.get('interpolation_certificate') or {}
            if not isinstance(certificate,dict):raise ValueError('Explicit interpolation certificate object or unavailable null required')
            sources.update(certificate.get('source_hashes',{}))
    for source,checksum in sources.items():
        if file_sha256(source)!=checksum:raise ValueError('Native encounter input/certificate changed')
    result,arrays=reconstruct_encounters(perturbers['worldlines'],stream['worldlines'],controls)
    result.update(request=request,frame=stream['frame'],stream_history_sha256=file_sha256(product),
        source_origin='synthetic_control' if any(record.get('fixture') or record.get('source_origin')=='synthetic_control' for record in (stream,perturbers)) else 'native_snapshot_and_catalogue',
        perturber_history_sha256=binding['sha256'],origin='recorded common fixed Cartesian frame',
        velocity_convention='proper separation derivative km/s; expansion conversion applied at native intake',
        periodic_wrapping='recorded periodic images already applied once at native intake',
        profiles=[{'identity':line['identity'],'namespace':line['namespace'],'profiles':line.get('profile_history'),
                   'profile_support':line.get('profile_support','UNAVAILABLE')} for line in perturbers['worldlines']])
    units={'encounter_time_myr':'Myr','impact_parameter_kpc':'proper kpc','relative_velocity_kms':'km / s',
           'sampling_error_kpc':'proper kpc','geometry_supported':'dimensionless'}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    return publish_stream_product(destination,result,arrays,units,sources=sources,namespace='encounters:'+stream['frame'],
        epoch={'representation':'reconstructed events within actual saved native intervals'})


def export_native_encounter_figures(product,destination,*,memory_gib=1.,dpi=160):
    """Registered encounter geometry/worldline exports with retained support."""
    from .data import export_numeric_audit
    from core.research.display import export_scientific_suite
    report,arrays,sources=read_product(product,memory_gib=memory_gib/3)
    output=Path(destination).resolve()
    if output.exists():raise FileExistsError('Keep prior encounter figures')
    output.mkdir(parents=True);products={};figures=[]
    science_status='SYNTHETIC_CONTROL_NOT_ACCEPTED' if report.get('source_origin')=='synthetic_control' or report.get('fixture') else 'NOT_ACCEPTED'
    if report['schema']=='stream-encounters-v1':
        values={'time':arrays['encounter_time_myr'],'impact':arrays['impact_parameter_kpc'],
            'speed':np.linalg.norm(arrays['relative_velocity_kms'],axis=1),
            'error':arrays['sampling_error_kpc'],'supported':arrays['geometry_supported']}
        units={'time':'Myr','impact':'proper kpc','speed':'km / s','error':'proper kpc','supported':'dimensionless'}
        path=output/'geometry.json'
        export_numeric_audit(path,{'schema':'native-encounter-figure-bridge-v1','events':report['events'],
            'event_branch_requests':report['event_branch_requests'],'science_acceptance':'NOT_ACCEPTED'},values,
            {k:{'units':v,'quantity':'reconstructed encounter '+k,'missing_policy':'geometry_supported=false requires refinement'} for k,v in units.items()},
            algorithm=__file__,context={'epoch':{'representation':'reconstructed event time'}},source_hashes=sources)
        products['geometry']=str(path)
        panels=[]
        for x in ('time','speed'):
            panels.append({'panel_id':x,'title':'Certified support only; unresolved events retained in arrays',
                'quality_status':'PENDING_CONVERGENCE','selection':'recorded geometry support; unaccepted events remain in source catalogue',
                'xlabel':x+' ['+units[x]+']','ylabel':'impact [proper kpc]','x_units':units[x],'y_units':'proper kpc',
                'series':[{'source':'geometry','kind':'errorbar','label':'reconstructed closest approach','x':x,'y':'impact','error':'error','support':'supported'}]})
        figures.append({'figure_id':'native-impact-geometry','family':'encounters','variant':'impact_velocity_geometry',
            'title':'Native encounter reconstruction','science_status':science_status,'panels':panels})
        for index,line in enumerate(report.get('profiles',[])):
            series=[]
            for ordinal,profile in enumerate(line.get('profiles') or []):
                if profile is None:continue
                radius=np.asarray(profile['radius_kpc'],float);mass=np.asarray(profile['enclosed_mass_msun'],float)
                if (radius.ndim!=1 or not len(radius) or radius.shape!=mass.shape or not np.isfinite([radius,mass]).all()
                        or np.any(radius<=0) or np.any(np.diff(radius)<=0) or np.any(mass<0) or np.any(np.diff(mass)<0)):
                    raise ValueError('Measured increasing bound enclosed-mass profile required')
                key='profile_%04d_%04d'%(index,ordinal);path=output/(key+'.json')
                export_numeric_audit(path,{'schema':'native-perturber-profile-figure-v1','identity':line['identity'],
                    'namespace':line['namespace'],'saved_profile_ordinal':ordinal,'profile':profile,
                    'science_acceptance':'NOT_ACCEPTED'}, {'radius':radius,'mass':mass,'supported':np.ones(len(radius),bool)},
                    {'radius':{'units':'proper kpc','quantity':'native bound-member enclosed radius'},
                     'mass':{'units':'Msun','quantity':'actual bound enclosed gravitating mass'},
                     'supported':{'units':'dimensionless','quantity':'measured saved profile; convergence not established'}},
                    algorithm=__file__,context={'epoch':{'saved_profile_ordinal':ordinal}},source_hashes=sources)
                products[key]=str(path)
                series.append({'source':key,'kind':'line','label':'saved profile '+str(ordinal),'x':'radius','y':'mass','support':'supported'})
            if series:
                figures.append({'figure_id':'native-perturber-profile-'+str(index),'family':'encounters','variant':'perturber_mass_profile',
                    'title':'Native bound mass '+line['identity'],'science_status':science_status,
                    'panels':[{'panel_id':'bound','title':'Saved native bound memberships; no analytic replacement',
                        'quality_status':'PENDING_CONVERGENCE','selection':'actual gravitating member masses; radii retain declared support',
                        'xlabel':'Radius [proper kpc]','ylabel':'Enclosed bound mass [Msun]',
                        'x_units':'proper kpc','y_units':'Msun','series':series}]})
    elif report['schema']=='native-encounter-frame-history-v1':
        for index,line in enumerate(report['worldlines']):
            key='line_%04d'%index;path=output/(key+'.json');x=np.asarray(line['position_kpc']);flags=np.asarray(line['valid_intervals'],bool)
            supported=np.r_[flags,False]|np.r_[False,flags]
            export_numeric_audit(path,{'schema':'native-worldline-figure-bridge-v1','identity':line['identity'],
                'namespace':line['namespace'],'frame':line['frame'],'science_acceptance':'NOT_ACCEPTED'},
                {'position':x,'supported':supported},
                {'position':{'units':'proper kpc','quantity':'saved proper positions in declared frame'},
                 'supported':{'units':'dimensionless','quantity':'endpoint touches a recorded supported interval'}},
                algorithm=__file__,context={'epoch':{'actual_saved_times_myr':line['times_myr']}},source_hashes=sources)
            products[key]=str(path)
            # Saved points avoid visually interpolating across temporal gaps.
            figures.append({'figure_id':'native-worldline-'+str(index),'family':'encounters','variant':'worldlines_3d',
                'title':'Saved native worldline '+line['identity'],'science_status':science_status,
                'panels':[{'panel_id':'saved','title':'Saved positions; no display interpolation',
                    'quality_status':'PENDING_INTERPOLATION_CERTIFICATION','selection':'exact namespace-qualified identity; temporal gaps preserved in source',
                    'xlabel':'x [proper kpc]','ylabel':'y [proper kpc]','zlabel':'z [proper kpc]',
                    'x_units':'proper kpc','y_units':'proper kpc','z_units':'proper kpc',
                    'series':[{'source':key,'kind':'scatter3d','label':'saved epochs','x':'position','y':'position','z':'position',
                        'x_column':0,'y_column':1,'z_column':2,'support':'supported'}]}]})
    else:raise ValueError('Converted native frame history or encounter catalogue required')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':dpi,
        'maximum_panels_per_figure':2,'organized_outputs':True},output/'gallery',base=output,memory_gib=memory_gib)


def derive_native_stream_product(product, request, destination, *, memory_gib):
    """Delegate geometry/observer/action measurements to the established runner."""
    from core.stream.workflow import Inputs, execute_stream_stage
    from core.stream_common import stream_array_units
    if request.get('schema')!='flagship-derived-stream-v1' or request.get('kind') not in {'geometry','observer','actions'}:
        raise ValueError('Explicit supported geometry, observer or actions request required')
    parent,arrays,sources=read_product(product,memory_gib=memory_gib/3)
    if parent.get('schema')!='stream-coordinates-v1':raise ValueError('Measured stream coordinates required')
    maximum=request['maximum_particles']
    if type(maximum) is not int or maximum<1 or len(arrays['particle_ids'])>maximum:
        raise MemoryError('Derived stream particle bound exceeded')
    stage={'stage_id':'native-stream-derived','kind':request['kind'],'parameters':request['parameters'],
           'inputs':{**request.get('inputs',{}),'coordinates':str(Path(product).resolve()),'sample':str(Path(product).resolve())}}
    inputs=Inputs(Path(product).resolve().parent,{}, {'memory_gib':memory_gib,'maximum_particles':maximum})
    inputs.sources.update(sources)
    result,values,namespace,epoch=execute_stream_stage(stage,inputs,Path(destination).resolve().parent)
    result.update(epoch=parent.get('epoch'),request=request,source_coordinates_sha256=file_sha256(product),
                  source_coordinates=str(Path(product).resolve()),science_acceptance='NOT_ACCEPTED',
                  origin=parent.get('origin','declared host-centered proper chart'),
                  periodic_wrapping=parent.get('periodic_wrapping','explicit spatial chart; no temporal winding inferred'))
    units=stream_array_units(values)
    if request['kind']=='observer':
        for key in values:
            if key.endswith('masyr'):units[key]='mas/year'
    units.update(result.get('units',{}))
    return publish_stream_product(destination,result,values,units,sources=inputs.sources,
                                  namespace=parent['namespace'],epoch=parent.get('epoch'))


def export_registered_stream_figures(product, destination, *, memory_gib=1., dpi=160):
    """Publish separate registered graphs from existing measured wrap arrays."""
    from .data import export_numeric_audit
    from core.research.display import export_scientific_suite
    report, arrays, sources = read_product(product, memory_gib=memory_gib / 2)
    if report.get('schema') in {'stream-geometry-v1','stream-observer-v1'}:
        return export_registered_stream_projection_figures(product,destination,memory_gib=memory_gib,dpi=dpi)
    if report.get('schema') != 'stream-coordinates-v1':
        raise ValueError('Measured stream coordinate product required')
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=False)
    figures = []; products = {}
    for wrap in report['wraps']:
        if wrap['status'] != 'MEASURED': continue
        prefix = 'wrap_%04d_' % wrap['wrap_id']
        a = {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
        support = np.asarray(a['supported'], dtype=bool)
        widths = np.full((len(support), 2), np.nan)
        widths[support] = np.sqrt(np.maximum(np.linalg.eigvalsh(a['width_covariance_kpc2'][support]), 0))
        dispersion = np.sqrt(np.maximum(np.diagonal(a['velocity_covariance_kms2'], axis1=1, axis2=2), 0))
        numeric = dict(a)
        numeric['width_axes_kpc'] = widths
        numeric['velocity_dispersion_kms'] = dispersion
        specs = {}
        for key in a:
            declared=report['numeric_data']['array_contract']['datasets'][prefix+key]
            specs[key]={'units':declared['units'],'quantity':declared['quantity_definition'],
                        'missing_policy':declared['missing_policy']}
            if declared.get('identity_namespace'):specs[key]['identity_namespace']=declared['identity_namespace']
        specs['width_axes_kpc'] = {'units': 'proper kpc', 'quantity': 'transverse covariance principal RMS axes', 'missing_policy': 'unsupported bins are NaN'}
        specs['velocity_dispersion_kms'] = {'units': 'km / s', 'quantity': 'local velocity covariance diagonal RMS', 'missing_policy': 'unsupported bins retain support mask'}
        name = 'wrap-%04d' % wrap['wrap_id']; derived = output / (name + '.json')
        export_numeric_audit(derived, {'schema': 'native-stream-figure-bridge-v1', 'status': 'MEASURED',
            'selection': report.get('selection'), 'epoch': report.get('epoch'),
            'limitations': report.get('limitations'), 'source_product': str(Path(product).resolve())},
            numeric, specs, algorithm=__file__, context={'epoch': report.get('epoch')},
            source_hashes={**sources, str(Path(__file__).resolve()): file_sha256(__file__)})
        products[name] = str(derived)
        def add(variant, x, y, xunit, yunit, *, columns=(None,), mask='supported', error=None, three_d=False):
            series = []
            for column in columns:
                item = {'source': name, 'kind': 'line3d' if three_d else 'errorbar' if error else 'line',
                        'label': 'measured' if column is None else 'component %d' % column, 'x': x, 'y': y}
                if column is not None: item['y_column'] = column
                if mask: item['support'] = mask
                if error: item['error'] = error
                if three_d: item.update(x_column=0, y_column=1, z=y, z_column=2)
                series.append(item)
            supported_entries=int(np.count_nonzero(numeric[mask])) if mask else len(numeric[x])
            panel = {'panel_id': 'measured', 'title': 'Saved native stream measurements' if supported_entries else 'UNRESOLVED: no supported values; numerical inputs and masks retained',
                     'quality_status': 'PENDING_CONVERGENCE' if supported_entries else 'UNRESOLVED_SUPPORT', 'selection': 'explicit selected wrap; exported support mask',
                     'xlabel': x + ' [' + xunit + ']', 'ylabel': y + ' [' + yunit + ']',
                     'x_units': xunit, 'y_units': yunit, 'series': series}
            if three_d: panel.update(z_units=yunit, zlabel='z [' + yunit + ']')
            figures.append({'figure_id': name + '-' + variant, 'family': 'streams', 'variant': variant,
                'title': variant.replace('_', ' '), 'science_status': 'SYNTHETIC_CONTROL_NOT_ACCEPTED' if report.get('source_origin')=='synthetic_control' else 'NOT_ACCEPTED',
                'epoch': report.get('epoch'), 'panels': [panel]})
        add('intrinsic_track', 'curve_position_kpc', 'curve_position_kpc', 'proper kpc', 'proper kpc', mask='curve_support', three_d=True)
        add('linear_density', 'bin_center_kpc', 'linear_density_msun_kpc', 'proper kpc', 'Msun / proper kpc', error='density_sampling_error_msun_kpc')
        add('width', 'bin_center_kpc', 'width_axes_kpc', 'proper kpc', 'proper kpc', columns=(0, 1))
        velocity_units=specs['velocity_mean_kms']['units']
        if velocity_units not in {'km/s','km / s'}:raise ValueError('Physical peculiar-velocity units required for native stream figure')
        add('velocity', 'bin_center_kpc', 'velocity_mean_kms', 'proper kpc', velocity_units, columns=(0, 1, 2))
        add('dispersion', 'bin_center_kpc', 'velocity_dispersion_kms', 'proper kpc', 'km / s', columns=(0, 1, 2))
        add('gaps', 'bin_center_kpc', 'density_feature_significance', 'proper kpc', 'dimensionless')
        add('spurs', 'bin_center_kpc', 'transverse_mean_kpc', 'proper kpc', 'proper kpc', columns=(0, 1))
        add('density_power', 'wavenumber_kpc_inverse', 'density_power_kpc', '1 / proper kpc', 'proper kpc', mask='power_supported')
        if 'track_power_kpc3' in a:
            add('track_power','wavenumber_kpc_inverse','track_power_kpc3','1 / proper kpc','proper kpc^3',columns=(0,1),mask='track_power_supported')
        if 'bifurcation_separation_kpc' in a:
            add('bifurcations','bin_center_kpc','bifurcation_separation_kpc','proper kpc','proper kpc',mask='bifurcation_candidate_supported')
            figures[-1]['title']='Transverse two-mode bifurcation candidates'
            if np.any(a['bifurcation_candidate_supported']):
                figures[-1]['panels'][0]['title']='Local proposals; physical fork/encounter acceptance unresolved'
            elif np.any(a['supported']):
                figures[-1]['panels'][0]['title']='No transverse two-mode candidates in supported bins'
                figures[-1]['panels'][0]['quality_status']='NO_BIFURCATION_CANDIDATES'
    return export_scientific_suite({'sources': products, 'figures': figures, 'dpi': dpi,
        'maximum_panels_per_figure': 1, 'organized_outputs': True}, output / 'gallery',
        base=output, memory_gib=memory_gib)


def export_registered_stream_projection_figures(product,destination,*,memory_gib=1.,dpi=160):
    """Render supported geometry/projection arrays without inventing detections."""
    from core.research.display import export_scientific_suite
    report,arrays,_=read_product(product,memory_gib=memory_gib/2)
    figures=[]
    def add(variant,suffix,x,y,xunits,yunits,*,xcolumn=None,ycolumn=None,support=None,error=None):
        series={'source':'measured','kind':'errorbar' if error else 'line','label':'measured selected population','x':x,'y':y}
        if xcolumn is not None:series['x_column']=xcolumn
        if ycolumn is not None:series['y_column']=ycolumn
        if support:series['support']=support
        if error:series['error']=error
        figures.append({'figure_id':variant+'-'+suffix,'family':'streams','variant':variant,
            'title':variant.replace('_',' '),'science_status':'SYNTHETIC_CONTROL_NOT_ACCEPTED' if report.get('source_origin')=='synthetic_control' else 'NOT_ACCEPTED','epoch':report.get('epoch'),
            'panels':[{'panel_id':'measured','title':'Selected population geometry','quality_status':'PENDING_CONVERGENCE',
                'selection':'source-selected population and recorded viewing basis; support retained',
                'xlabel':x+' ['+xunits+']','ylabel':y+' ['+yunits+']','x_units':xunits,'y_units':yunits,'series':[series]}]})
    if report['schema']=='stream-geometry-v1':
        for view in report['views']:
            prefix='view_%03d_'%view['view_id'];s=prefix+'projected_arclength_kpc'
            add('projected_track',str(view['view_id']),prefix+'projected_kpc',prefix+'projected_kpc','proper kpc','proper kpc',xcolumn=0,ycolumn=1,support=prefix+'diagnostic_supported')
            add('curvature',str(view['view_id']),s,prefix+'curvature_kpc_inverse','proper kpc','1 / proper kpc',support=prefix+'normal_supported',error=prefix+'curvature_std_kpc_inverse')
            add('convexity',str(view['view_id']),s,prefix+'convex_probability','proper kpc','dimensionless',support=prefix+'convex_probability_supported')
    elif report['schema']=='stream-observer-v1':
        add('projected_track','observer','longitude_rad','latitude_rad','radian','radian')
    else:raise ValueError('Stream geometry or observer product required')
    output=Path(destination).resolve()
    if output.exists():raise FileExistsError('Keep previous geometry figures')
    return export_scientific_suite({'sources':{'measured':str(Path(product).resolve())},'figures':figures,
        'dpi':dpi,'maximum_panels_per_figure':1,'organized_outputs':True},output,base=Path(product).resolve().parent,memory_gib=memory_gib)


def export_stream_timeline(product, destination, *, memory_gib, maximum_saved_vertices):
    """Export all saved epochs, preserving exact IDs and evolving tracer masses."""
    report, arrays, sources = read_product(product, memory_gib=memory_gib / 2)
    x = arrays['position_kpc']; v = arrays['velocity_kms']; ids = arrays['particle_ids']
    times = arrays['times_myr']; mass = arrays['mass_msun']
    if x.ndim != 3 or x.shape[-1] != 3 or v.shape != x.shape or x.shape[:2] != (len(times), len(ids)):
        raise ValueError('Saved stream timeline dimensions differ')
    if type(maximum_saved_vertices) is not int or maximum_saved_vertices < 1 or x.shape[0] * x.shape[1] > maximum_saved_vertices:
        raise MemoryError('Saved stream display vertex budget exceeded')
    if ids.dtype != np.dtype('uint64') or len(np.unique(ids)) != len(ids):
        raise ValueError('Exact distinct uint64 stream identities required')
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('At least two strictly increasing saved physical epochs required')
    if not np.isfinite(x).all() or not np.isfinite(v).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError('Finite saved tracer data and positive masses required')
    if mass.shape == (len(ids),): mass = np.broadcast_to(mass, x.shape[:2]).copy()
    if mass.shape != x.shape[:2]: raise ValueError('Tracer masses must bind every saved epoch')
    output = Path(destination).resolve(); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'inspector.npz', position_kpc=x, velocity_kms=v,
                        particle_ids=ids, mass_msun=mass, times_myr=times)
    atomic_json(output / 'inspector.json', {'schema': 'stream-inspector-timeline-v1',
        'arrays_sha256': file_sha256(output / 'inspector.npz'),
        'product': str(Path(product).resolve()), 'product_sha256': file_sha256(product),
        'namespace': report['namespace'], 'length_units': 'proper kpc',
        'velocity_units': 'km / s', 'time_units': 'Myr', 'saved_times_myr': times.tolist(),
        'epochs': report.get('epochs'), 'coordinate_semantics': report.get('scope', report.get('origin')),
        'periodic_wrapping': report.get('periodic', False), 'source_hashes': sources,
        'identity_encoding': 'uint64 archive; decimal-string inspector IDs',
        'display_policy': 'hold each saved epoch until the next; no trajectory interpolation',
        'maximum_saved_vertices': maximum_saved_vertices, 'science_acceptance': 'NOT_ACCEPTED'})
    return json.loads((output / 'inspector.json').read_text())


def stellar_yield_audit(request, destination):
    """Report exact offending yield rows without changing their physical values."""
    from .stars import load_stellar_yield_library
    asset = request['yield_asset']
    rows = load_stellar_yield_library(asset['path'], expected_sha256=asset['sha256'])
    tolerance = request['yield_mass_closure_relative_tolerance']
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or not 0 <= tolerance <= .05:
        raise ValueError('Bounded finite yield closure tolerance required')
    measurements = []
    for row in rows:
        expected = row['mass_msun'] - row['remnant_msun']
        total = float(np.sum(row['elements_msun'])); error = abs(total - expected) / expected
        measurements.append({key: row[key] for key in ('mass_msun', 'metallicity', 'remnant_msun')} |
            {'expected_return_msun': expected, 'represented_ejecta_msun': total,
             'relative_closure_error': error, 'status': 'PASS' if error <= tolerance else 'FAIL'})
    result = {'schema': 'flagship-stellar-yield-audit-v1', 'yield_asset': asset,
              'tolerance': tolerance, 'rows': measurements,
              'status': 'PASS' if all(row['status'] == 'PASS' for row in measurements) else 'FAIL',
              'science_acceptance': 'NOT_ACCEPTED'}
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Keep previous yield audit')
    atomic_json(destination, result); return result


def export_convergence_figures(request, destination, *, request_base, memory_gib=2.):
    """Connect existing native comparison math to all six convergence variants."""
    from core.research.convergence import compare_campaign
    from .data import export_numeric_audit
    from core.research.display import export_scientific_suite
    mapping={'mass_resolution':'mass_resolution','softening':'softening','timestep':'timestep',
             'output_cadence':'output_cadence','start_epoch':'start_redshift','finder':'finder_tree_variation'}
    comparison=request['comparison']; control=comparison['control']
    if control not in mapping:raise ValueError('No registered convergence graph for this control')
    base=Path(request_base).resolve();measurements=[];sources={}
    for item in request['measurements']:
        path=Path(item['path']);path=(base/path).resolve() if not path.is_absolute() else path.resolve()
        if path.stat().st_size>memory_gib*1024**3/8:raise MemoryError('Convergence measurement exceeds input workspace')
        if file_sha256(path)!=item['sha256']:raise ValueError('Convergence measurement changed')
        measurements.append(json.loads(path.read_text()));sources[str(path)]=item['sha256']
    if sum(len(record['bins'])**2*128 for record in measurements)>memory_gib*1024**3:
        raise MemoryError('Convergence covariance exceeds workspace')
    result=compare_campaign(comparison,measurements)
    x=np.asarray(request['bin_coordinate']['values'],dtype=float)
    if x.shape!=(len(measurements[0]['bins']),) or not np.isfinite(x).all():
        raise ValueError('Explicit measured bin coordinate must match comparison bins')
    output=Path(destination).resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'comparison.json',json_safe(result))
    numeric={};specs={};series=[]
    xunit=request['bin_coordinate']['units'];yunit=result['units']
    numeric['bin_coordinate']=x
    specs['bin_coordinate']={'units':xunit,'quantity':request['bin_coordinate']['quantity'],'missing_policy':'all comparison domain bins retained'}
    for ordinal,(record,row) in enumerate(zip(measurements,result['rows'])):
        prefix='run_%04d_'%ordinal
        for name,values in [('value',record['value']),('difference',row['difference']),
                            ('error_bound',row['error_bound']),('supported',row['supported'])]:
            numeric[prefix+name]=np.asarray(values,dtype=bool if name=='supported' else float)
            specs[prefix+name]={'units':'dimensionless' if name=='supported' else yunit,
                'quantity':name+' for native convergence run '+record['run_id'],
                'missing_policy':'unsupported contrast is masked; null numeric entries remain NaN'}
        series.append({'source':'comparison','kind':'line','label':record['run_id'],
                       'x':'bin_coordinate','y':prefix+'difference','support':prefix+'supported'})
        sources.update(record['source_hashes'])
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    product=output/'measurements.json'
    export_numeric_audit(product,{'schema':'native-convergence-figure-bridge-v1','status':'MEASURED',
        'control':control,'comparison_status':result['status'],'epoch':measurements[0]['epoch'],
        'scope':result['scope'],'bins':measurements[0]['bins']},numeric,specs,algorithm=__file__,
        context={'epoch':measurements[0]['epoch']},source_hashes=sources)
    panel={'panel_id':'contrast','title':'Matched native numerical contrast',
           'quality_status':result['status'],'selection':'same physical bins, phase and observable; measured common support',
           'xlabel':request['bin_coordinate']['quantity']+' ['+xunit+']','ylabel':'Difference ['+yunit+']',
           'x_units':xunit,'y_units':yunit,'series':series}
    variant=mapping[control]
    return export_scientific_suite({'sources':{'comparison':str(product)},'dpi':160,
        'maximum_panels_per_figure':1,'organized_outputs':True,'figures':[{
            'figure_id':'native-'+variant,'family':'convergence','variant':variant,
            'title':'Native '+variant.replace('_',' '),'science_status':'NOT_ACCEPTED',
            'epoch':measurements[0]['epoch'],'panels':[panel]}]},output/'gallery',base=output,memory_gib=memory_gib)


def flagship_product_input_bindings(parameters):
    if parameters['operation']=='sidm-matched-figures':
        from .sidm_products import sidm_matched_inputs
        path=Path(parameters['product']).resolve()
        if path.stat().st_size*16>parameters['memory_gib']*1024**3/4:raise MemoryError('Matched SIDM request exceeds allocation')
        return {str(path):file_sha256(path),**sidm_matched_inputs(json.loads(path.read_text()),memory_gib=parameters['memory_gib'])[2]}
    if parameters['operation']=='sidm-history-figures':
        from .sidm_products import sidm_history_inputs
        return sidm_history_inputs(parameters['product'],run_directory=parameters['run_directory'],memory_gib=parameters['memory_gib'])[2]
    if parameters['operation']=='web-history-figures':
        from .halo_products import web_history_inputs
        return web_history_inputs(parameters['product'],memory_gib=parameters['memory_gib'])[2]
    if parameters['operation']=='web-figures':
        from .halo_products import web_inputs
        return web_inputs(parameters['product'],memory_gib=parameters['memory_gib'])[2]
    if parameters['operation']=='cloud-figures':
        from .first_stars import cloud_figure_inputs
        return cloud_figure_inputs(parameters['product'],memory_gib=parameters['memory_gib'])[2]
    if parameters['operation']=='first-star-figures':
        from .first_stars import first_star_inputs
        return first_star_inputs(parameters['product'],memory_gib=parameters['memory_gib'])[2]
    if parameters['operation']=='population-history-figures':
        from .baryon_products import population_inputs
        path=Path(parameters['product']).resolve()
        if path.stat().st_size*16>parameters['memory_gib']*1024**3/4:raise MemoryError('Population request exceeds allocation')
        return {str(path):file_sha256(path),**population_inputs(json.loads(path.read_text()),memory_gib=parameters['memory_gib'])[1]}
    if parameters['operation'] in {'sfr-history-figures','mass-return-figures','metal-history-figures'}:
        from .baryon_products import hydro_baryon_inputs
        return hydro_baryon_inputs(parameters['product'],memory_gib=parameters['memory_gib'])[2]
    if parameters['operation']=='gas-phase-figures':
        from .baryon_products import gas_phase_inputs
        path=Path(parameters['product']).resolve()
        if path.stat().st_size*16>parameters['memory_gib']*1024**3/4:raise MemoryError('Gas phase request exceeds allocation')
        return {str(path):file_sha256(path),**gas_phase_inputs(json.loads(path.read_text()),memory_gib=parameters['memory_gib'])[2]}
    if parameters['operation']=='bh-dm-response':
        from .flagship_bh_response import response_inputs
        path=Path(parameters['product']).resolve()
        if path.stat().st_size*16>parameters['memory_gib']*1024**3/4:raise MemoryError('BH response request exceeds allocation')
        sources=response_inputs(json.loads(path.read_text()),memory_gib=parameters['memory_gib'])[-1]
        return {str(path):file_sha256(path),**sources}
    if parameters['operation']=='bh-recoil-figures':
        from .flagship_bh import bh_figure_input_bindings,bh_event_input_bindings
        from .data import numeric_audit_source_bindings
        sources=bh_figure_input_bindings(parameters['product'],run_directory=parameters.get('run_directory'),memory_gib=parameters['memory_gib'])[-1]
        request=parameters['request'];sources.update(bh_event_input_bindings(request['hydro_history']['path'],memory_gib=parameters['memory_gib'])[-1])
        for name in ('hydro_history','recorded_images'):
            binding=request[name];path=Path(binding['path']).resolve()
            if file_sha256(path)!=binding['sha256']:raise ValueError('Recoil input binding changed')
            report,_,pins=read_product(path,memory_gib=parameters['memory_gib']/4);sources.update(pins);sources.update(numeric_audit_source_bindings(report))
        return sources
    if parameters['operation']=='bh-pair-figures':
        from .flagship_bh import bh_figure_input_bindings
        return bh_figure_input_bindings(parameters['product'],run_directory=parameters.get('run_directory'),memory_gib=parameters['memory_gib'])[-1]
    if parameters['operation']=='bh-merger-figures':
        from .flagship_bh import bh_event_input_bindings
        return bh_event_input_bindings(parameters['product'],memory_gib=parameters['memory_gib'])[-1]
    if parameters['operation']=='bh-figures':
        from .flagship_bh import bh_figure_input_bindings
        return bh_figure_input_bindings(parameters['product'],run_directory=parameters.get('run_directory'),memory_gib=parameters['memory_gib'])[-1]
    if parameters['operation']=='ic-flow-noise':
        from core.music.flow import flow_noise_input_bindings
        return flow_noise_input_bindings(parameters['product'],parameters['request'])
    if parameters['operation']=='ic-flow-covariance':
        from core.music.covariance import flow_covariance_input_bindings
        return flow_covariance_input_bindings(parameters['product'],parameters['request'])
    if parameters['operation']=='ic-compensated-covariance':
        from core.music.covariance import window_covariance_input_bindings
        return window_covariance_input_bindings(parameters['product'],parameters['request'])
    if parameters['operation']=='ic-flow-windows':
        from core.music.replay import transfer_replay_input_bindings
        return transfer_replay_input_bindings(parameters)
    if parameters['operation']=='ic-compensated-windows':
        from core.music.replay import transfer_replay_input_bindings
        return transfer_replay_input_bindings(parameters)
    if parameters['operation']=='ic-transfer-replay':
        from core.music.replay import transfer_replay_input_bindings
        return transfer_replay_input_bindings(parameters)
    if parameters['operation'] in {'ic-species-window-modes','ic-quadratic-assembly','ic-poisson-equations','ic-gradient-operators','ic-boundary-reference'}:
        # These consume original HDF5 ICs, not an already-exported numeric
        # audit. Freeze their actual source dependencies before cache reuse.
        bindings={str(Path(parameters['product']).resolve()):file_sha256(parameters['product'])}
        if parameters.get('configuration'):
            from .zoom import _configuration
            config=Path(parameters['configuration']).resolve();bindings[str(config)]=file_sha256(config)
            cfg=_configuration(config);transfer=Path(cfg.get('cosmology','transfer_file'))
            if not transfer.is_absolute():transfer=config.parent/transfer
            bindings[str(transfer.resolve())]=file_sha256(transfer)
        if parameters.get('source_index'):
            index=Path(parameters['source_index']).resolve();seal=index.with_name(index.name+'.sha256')
            if seal.read_text().strip()!=file_sha256(index):raise ValueError('Retained IC source index changed')
            record=json.loads(index.read_text());archive=Path(record['archive']['path'])
            if file_sha256(archive)!=record['archive']['sha256']:raise ValueError('Retained IC source archive changed')
            bindings.update({str(path):file_sha256(path) for path in (index,seal,archive)})
        return bindings
    if parameters['operation']=='analysis-figures':
        from .analysis import verify_analysis_bundle
        path=Path(parameters['product']).resolve();verify_analysis_bundle(path,verify_sources=True)
        return {str(path/name):file_sha256(path/name) for name in ('analysis_provenance.json','analysis_products.h5')}
    if parameters['operation']=='assembly-orbits':
        from .flagship_orbits import native_orbit_inputs
        return native_orbit_inputs(parameters['product'],parameters['request'],memory_gib=parameters['memory_gib'])[-1]
    if parameters['operation']=='assembly-event-figures':
        from .data import numeric_audit_source_bindings
        report,_,pins=read_product(parameters['product'],memory_gib=parameters['memory_gib'])
        return {**pins,**numeric_audit_source_bindings(report)}
    if parameters['operation'] in {'abundance','host-abundance','matched-host-abundance'}:
        from .flagship_abundance import abundance_input_bindings,matched_host_input_bindings,_provenance_sources
        path=Path(parameters['product']).resolve()
        if path.stat().st_size*8>parameters['memory_gib']*1024**3:raise MemoryError('Abundance request exceeds input allocation')
        request=json.loads(path.read_text())
        if parameters['operation']=='abundance':sources=abundance_input_bindings(request,memory_gib=parameters['memory_gib'])[-1]
        elif parameters['operation']=='matched-host-abundance':sources=matched_host_input_bindings(request,memory_gib=parameters['memory_gib'])[-1]
        else:
            from .analysis import _begin_analysis_contract
            _,contract=_begin_analysis_contract(request['run_directory'],['hmf']);sources=_provenance_sources(contract)
            run=Path(request['run_directory']).resolve()
            sources.update({str(run/name):file_sha256(run/name) for name in ('config.yaml','manifest.json')})
        return {str(path):file_sha256(path),**sources}
    if parameters['operation']=='assembly-figures':
        from .halos import verify_identity_graph
        verify_identity_graph(parameters['identity_graph_directory'])
    if parameters['operation']=='splashback-figures':
        from .analysis import verify_native_environment
        verify_native_environment(parameters['product'])
    if parameters['operation']=='structure-figures':
        from .halos import verify_halo_structure
        verify_halo_structure(parameters['product'])
    if parameters['operation'] in {'power-linear','power-evolved','power-matched','power-ensemble'}:
        from .power import power_input_bindings
        return power_input_bindings(parameters)
    from .data import verify_numeric_audit
    product=Path(parameters['product']).resolve()
    report=verify_numeric_audit(product)
    archive=product.with_name(product.name+'.data')/'arrays.npz'
    bindings={str(product):file_sha256(product),str(archive):file_sha256(archive),
            **{str(path):checksum for path,checksum in report.get('source_hashes',{}).items()}}
    request=parameters.get('request',{})
    for name in ('catalogue_selection','identity_graph','calibration','recorded_images','perturber_history'):
        binding=request.get(name)
        if not binding:continue
        path=Path(binding['path']).resolve()
        if name=='identity_graph':
            if file_sha256(path/'report.json')!=binding['sha256']:raise ValueError('Identity graph binding changed')
            bindings.update({str(path/asset):file_sha256(path/asset) for asset in ('report.json','index.sqlite')})
        else:
            if file_sha256(path)!=binding['sha256']:raise ValueError('Stellar selection/calibration changed')
            bindings[str(path)]=binding['sha256']
            if name in ('catalogue_selection','recorded_images','perturber_history'):
                extra=verify_numeric_audit(path);asset=path.with_name(path.name+'.data')/'arrays.npz'
                bindings[str(asset)]=file_sha256(asset);bindings.update(extra.get('source_hashes',{}))
    return bindings


def flagship_product_runner(parameters, context):
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Flagship product escapes its workflow generation')
    memory=parameters['memory_gib']
    if not np.isfinite(memory) or not 0<memory<=context.resources['memory_gib']:
        raise MemoryError('Flagship product exceeds workflow allocation')
    operation=parameters['operation']
    if operation=='sidm-matched-figures':
        from .sidm_products import export_native_sidm_matched
        return export_native_sidm_matched(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
    if operation=='sidm-history-figures':
        from .sidm_products import export_native_sidm_history
        return export_native_sidm_history(parameters['product'],destination,run_directory=parameters['run_directory'],memory_gib=memory)
    if operation=='web-history-figures':
        from .halo_products import export_native_web_history
        return export_native_web_history(parameters['product'],destination,memory_gib=memory)
    if operation=='web-figures':
        from .halo_products import export_native_web_figures
        return export_native_web_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='cloud-figures':
        from .first_stars import export_native_cloud_figures
        return export_native_cloud_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='first-star-figures':
        from .first_stars import export_native_first_star_figures
        return export_native_first_star_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='metal-history-figures':
        from .baryon_products import export_native_metal_history
        return export_native_metal_history(parameters['product'],destination,memory_gib=memory)
    if operation=='population-history-figures':
        from .baryon_products import export_native_population_history
        return export_native_population_history(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
    if operation=='mass-return-figures':
        from .baryon_products import export_native_mass_return
        return export_native_mass_return(parameters['product'],destination,memory_gib=memory)
    if operation=='sfr-history-figures':
        from .baryon_products import export_native_sfr_history
        return export_native_sfr_history(parameters['product'],destination,memory_gib=memory)
    if operation=='gas-phase-figures':
        from .baryon_products import export_native_gas_phase
        return export_native_gas_phase(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
    if operation=='bh-dm-response':
        from .flagship_bh_response import export_native_bh_dm_response
        return export_native_bh_dm_response(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
    if operation=='bh-recoil-figures':
        from .flagship_bh import export_native_bh_recoil_trajectory
        return export_native_bh_recoil_trajectory(parameters['product'],parameters['request'],destination,run_directory=parameters.get('run_directory'),memory_gib=memory)
    if operation=='bh-pair-figures':
        from .flagship_bh import export_native_bh_pair_separation
        return export_native_bh_pair_separation(parameters['product'],parameters['request'],destination,run_directory=parameters.get('run_directory'),memory_gib=memory)
    if operation=='bh-merger-figures':
        from .flagship_bh import export_native_bh_merger_timeline
        return export_native_bh_merger_timeline(parameters['product'],destination,memory_gib=memory)
    if operation=='bh-figures':
        from .flagship_bh import export_native_bh_figures
        return export_native_bh_figures(parameters['product'],destination,run_directory=parameters.get('run_directory'),memory_gib=memory)
    if operation=='analysis-figures':
        from core.figures import export_native_analysis_figures
        return export_native_analysis_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='assembly-orbits':
        from .flagship_orbits import measure_native_assembly_orbits
        return measure_native_assembly_orbits(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='assembly-event-figures':
        from .flagship_orbits import export_native_assembly_event_figures
        return export_native_assembly_event_figures(parameters['product'],destination,memory_gib=memory)
    if operation in {'abundance','host-abundance','matched-host-abundance'}:
        from .flagship_abundance import export_native_abundance,export_native_host_abundance,export_matched_host_abundance
        exporter={'abundance':export_native_abundance,'host-abundance':export_native_host_abundance,'matched-host-abundance':export_matched_host_abundance}[operation]
        return exporter(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
    if operation=='assembly-figures':
        from .assembly import export_native_assembly_figures
        return export_native_assembly_figures(parameters['product'],parameters['identity_graph_directory'],destination,memory_gib=memory)
    if operation=='splashback-figures':
        from .halo_products import export_native_splashback_figures
        return export_native_splashback_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='structure-figures':
        from .halo_products import export_native_structure_figures
        return export_native_structure_figures(parameters['product'],destination,memory_gib=memory)
    if operation in {'power-linear','power-evolved','power-matched','power-ensemble'}:
        from .power import export_linear_power,export_evolved_power,export_matched_power
        if operation=='power-linear':return export_linear_power(parameters['product'],destination,memory_gib=memory)
        if operation=='power-evolved':return export_evolved_power(parameters['product'],destination,memory_gib=memory)
        if operation=='power-ensemble':
            from .power import export_power_ensemble
            return export_power_ensemble(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
        return export_matched_power(json.loads(Path(parameters['product']).read_text()),destination,memory_gib=memory)
    if operation=='stream-catalogue':
        return catalogue_native_stream_epoch(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='stream-derived':
        return derive_native_stream_product(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='encounter-frame':
        return native_encounter_frame(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='encounter-images':
        from .periodic_images import publish_native_worldline_images
        return publish_native_worldline_images(parameters['product'],destination,memory_gib=memory)
    if operation=='ic-species-window-modes':
        from .zoom import measure_native_species_window_modes
        return measure_native_species_window_modes(parameters['product'],parameters['request']['modes'],destination,memory_gib=memory)
    if operation=='ic-quadratic-assembly':
        from .music.archive import audit_native_quadratic_velocity
        return audit_native_quadratic_velocity(parameters['product'],parameters['configuration'],destination,memory_gib=memory)
    if operation=='ic-poisson-equations':
        from core.music.archive import audit_music_archive_poisson
        return audit_music_archive_poisson(parameters['source_index'],parameters['configuration'],parameters['product'],destination,memory_gib=memory)
    if operation=='ic-gradient-operators':
        from core.music.replay import audit_music_archive_gradients
        return audit_music_archive_gradients(parameters['source_index'],parameters['configuration'],parameters['product'],destination,memory_gib=memory)
    if operation=='ic-boundary-reference':
        from core.music.poisson import audit_music_archive_boundary
        return audit_music_archive_boundary(parameters['source_index'],parameters['configuration'],parameters['product'],destination,memory_gib=memory)
    if operation=='ic-transfer-replay':
        from core.music.replay import audit_music_archive_transfer
        return audit_music_archive_transfer(parameters['source_index'],parameters['configuration'],parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='ic-compensated-windows':
        from core.music.windows import measure_native_compensated_windows
        return measure_native_compensated_windows(parameters['source_index'],parameters['configuration'],parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='ic-flow-noise':
        from core.music.flow import publish_flow_noise_response
        return publish_flow_noise_response(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='ic-flow-covariance':
        from core.music.covariance import publish_flow_window_covariance
        return publish_flow_window_covariance(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='ic-compensated-covariance':
        from core.music.covariance import publish_compensated_window_covariance
        return publish_compensated_window_covariance(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='ic-flow-windows':
        from core.music.windows import measure_native_flow_windows
        return measure_native_flow_windows(parameters['source_index'],parameters['configuration'],parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='encounter-catalogue':
        return measure_native_encounters(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='encounter-figures':
        return export_native_encounter_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='stream-epoch':
        return measure_native_stream_epoch(parameters['product'],parameters['request'],destination,memory_gib=memory)
    if operation=='stream-figures':
        return export_registered_stream_figures(parameters['product'],destination,memory_gib=memory)
    if operation=='stream-timeline':
        return export_stream_timeline(parameters['product'],destination,memory_gib=memory,
                                     maximum_saved_vertices=parameters['maximum_saved_vertices'])
    raise ValueError('Unknown flagship product operation')


def validate_flagship_product(parameters):
    if parameters.get('operation') not in {'sidm-matched-figures','sidm-history-figures','web-history-figures','web-figures','cloud-figures','first-star-figures','metal-history-figures','population-history-figures','mass-return-figures','sfr-history-figures','gas-phase-figures','bh-dm-response','bh-recoil-figures','bh-pair-figures','bh-merger-figures','bh-figures','analysis-figures','assembly-orbits','assembly-event-figures','abundance','host-abundance','matched-host-abundance','assembly-figures','splashback-figures','structure-figures','power-linear','power-evolved','power-matched','power-ensemble','encounter-figures','encounter-catalogue','encounter-frame','encounter-images','ic-species-window-modes','ic-quadratic-assembly','ic-poisson-equations','ic-gradient-operators','ic-boundary-reference','ic-transfer-replay','ic-compensated-windows','ic-flow-windows','ic-flow-noise','ic-flow-covariance','ic-compensated-covariance','stream-derived','stream-catalogue','stream-epoch','stream-figures','stream-timeline'}:
        raise ValueError('Explicit supported flagship product operation required')
    memory=parameters['memory_gib']
    if isinstance(memory,bool) or not np.isfinite(memory) or memory<=0:
        raise ValueError('Positive finite flagship product workspace required')
    if parameters['operation']=='ic-species-window-modes':
        from .zoom import _modes
        request=parameters.get('request',{})
        if request.get('schema')!='native-music-species-window-request-v1':
            raise ValueError('Explicit native species-window mode request required')
        _modes(request.get('modes'))
    if parameters['operation'] in ('ic-quadratic-assembly','ic-poisson-equations','ic-gradient-operators','ic-boundary-reference','ic-transfer-replay','ic-compensated-windows','ic-flow-windows') and not parameters.get('configuration'):
        raise ValueError('Original MUSIC configuration required for quadratic component units')
    if parameters['operation'] in ('ic-poisson-equations','ic-gradient-operators','ic-boundary-reference','ic-transfer-replay','ic-compensated-windows','ic-flow-windows') and not parameters.get('source_index'):
        raise ValueError('Exact retained native source archive index required')
    if parameters['operation']=='ic-flow-noise':
        from core.music.flow import validate_flow_noise_request
        validate_flow_noise_request(parameters.get('request'))
    if parameters['operation']=='ic-flow-covariance':
        from core.music.covariance import validate_flow_covariance_request
        validate_flow_covariance_request(parameters.get('request'))
    if parameters['operation']=='ic-compensated-covariance':
        from core.music.covariance import validate_window_covariance_request
        validate_window_covariance_request(parameters.get('request'))
    if parameters['operation']=='ic-flow-windows':
        from core.music.windows import validate_flow_window_request
        validate_flow_window_request(parameters.get('request'))
    if parameters['operation']=='ic-compensated-windows':
        from core.music.windows import validate_compensated_window_request
        validate_compensated_window_request(parameters.get('request'))
    if parameters['operation']=='ic-transfer-replay':
        request=parameters.get('request',{})
        if (not isinstance(request,dict) or set(request)!={'schema','mesh','noise_directory'} or request.get('schema')!='native-music-transfer-replay-request-v1'):
            raise ValueError('Explicit original native noise/mesh replay request required')
    if parameters['operation']=='assembly-orbits' and parameters.get('request',{}).get('schema')!='flagship-native-assembly-orbits-v1':
        raise ValueError('Explicit native assembly orbit request required')
    if parameters['operation']=='assembly-figures' and not parameters.get('identity_graph_directory'):
        raise ValueError('Assembly figures require the exact native identity graph directory')
    if parameters['operation']=='stream-epoch' and parameters.get('request',{}).get('schema')!='flagship-native-stream-epoch-v1':
        raise ValueError('Native stream epoch request required')
    if parameters['operation']=='stream-catalogue' and parameters.get('request',{}).get('schema')!='flagship-native-stream-catalogue-v1':
        raise ValueError('Native stellar catalogue request required')
    if parameters['operation']=='stream-derived' and parameters.get('request',{}).get('schema')!='flagship-derived-stream-v1':
        raise ValueError('Derived stream request required')
    if parameters['operation']=='encounter-frame' and parameters.get('request',{}).get('schema')!='native-encounter-frame-v1':
        raise ValueError('Native encounter frame request required')
    if parameters['operation']=='encounter-catalogue' and parameters.get('request',{}).get('schema')!='native-encounter-catalogue-v1':
        raise ValueError('Native encounter catalogue request required')
    if parameters['operation']=='stream-timeline' and (type(parameters.get('maximum_saved_vertices'))is not int or parameters['maximum_saved_vertices']<=0):
        raise ValueError('Positive exact display vertex bound required')


def flagship_products_main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    for kind in ('sidm-matched-figures','sidm-history-figures','web-history-figures','web-figures','cloud-figures','first-star-figures','metal-history-figures','population-history-figures','mass-return-figures','sfr-history-figures','gas-phase-figures','bh-dm-response','bh-recoil-figures','bh-pair-figures','bh-merger-figures','bh-figures','analysis-figures','assembly-orbits','assembly-event-figures','abundance','host-abundance','matched-host-abundance','assembly-figures','splashback-figures','structure-figures','power-linear','power-evolved','power-matched','power-ensemble','encounter-figures','encounter-catalogue','encounter-frame','encounter-images','ic-species-window-modes','ic-quadratic-assembly','ic-poisson-equations','ic-gradient-operators','ic-boundary-reference','ic-transfer-replay','ic-compensated-windows','ic-flow-windows','ic-flow-noise','ic-flow-covariance','ic-compensated-covariance','stream-derived','stream-catalogue', 'stream-epoch', 'stream-figures', 'stream-timeline', 'yield-audit', 'convergence-figures', 'profile-preflight'):
        command = commands.add_parser(kind); command.add_argument('input', type=Path)
        command.add_argument('--output', type=Path, required=True)
        if kind != 'yield-audit': command.add_argument('--memory-gib', type=float, default=2.)
        if kind in ('ic-transfer-replay','ic-compensated-windows','ic-flow-windows','ic-flow-noise','ic-flow-covariance','ic-compensated-covariance','ic-species-window-modes','bh-recoil-figures','bh-pair-figures','assembly-orbits','encounter-catalogue','encounter-frame','stream-derived','stream-epoch','stream-catalogue'): command.add_argument('--request', type=Path, required=True)
        if kind == 'stream-timeline': command.add_argument('--maximum-saved-vertices', type=int, default=2000000)
        if kind == 'profile-preflight': command.add_argument('--output-directory', type=Path, required=True)
        if kind in ('ic-quadratic-assembly','ic-poisson-equations','ic-gradient-operators','ic-boundary-reference','ic-transfer-replay','ic-compensated-windows','ic-flow-windows'): command.add_argument('--configuration',type=Path,required=True)
        if kind in ('ic-poisson-equations','ic-gradient-operators','ic-boundary-reference','ic-transfer-replay','ic-compensated-windows','ic-flow-windows'):command.add_argument('--source-index',type=Path,required=True)
        if kind == 'assembly-figures': command.add_argument('--identity-graph',type=Path,required=True)
        if kind in ('bh-figures','bh-recoil-figures','bh-pair-figures'): command.add_argument('--run-directory',type=Path)
        if kind=='sidm-history-figures':command.add_argument('--run-directory',type=Path,required=True)
    args = parser.parse_args()
    if args.kind != 'yield-audit' and (not np.isfinite(args.memory_gib) or args.memory_gib <= 0):
        parser.error('memory-gib must be finite and positive')
    if args.kind=='sidm-matched-figures':
        from .sidm_products import export_native_sidm_matched
        result=export_native_sidm_matched(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind=='sidm-history-figures':
        from .sidm_products import export_native_sidm_history
        result=export_native_sidm_history(args.input,args.output,run_directory=args.run_directory,memory_gib=args.memory_gib)
    elif args.kind=='web-history-figures':
        from .halo_products import export_native_web_history
        result=export_native_web_history(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='web-figures':
        from .halo_products import export_native_web_figures
        result=export_native_web_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='cloud-figures':
        from .first_stars import export_native_cloud_figures
        result=export_native_cloud_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='first-star-figures':
        from .first_stars import export_native_first_star_figures
        result=export_native_first_star_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='metal-history-figures':
        from .baryon_products import export_native_metal_history
        result=export_native_metal_history(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='population-history-figures':
        from .baryon_products import export_native_population_history
        result=export_native_population_history(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind=='mass-return-figures':
        from .baryon_products import export_native_mass_return
        result=export_native_mass_return(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='sfr-history-figures':
        from .baryon_products import export_native_sfr_history
        result=export_native_sfr_history(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='gas-phase-figures':
        from .baryon_products import export_native_gas_phase
        result=export_native_gas_phase(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind=='bh-dm-response':
        from .flagship_bh_response import export_native_bh_dm_response
        result=export_native_bh_dm_response(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind=='bh-recoil-figures':
        from .flagship_bh import export_native_bh_recoil_trajectory
        result=export_native_bh_recoil_trajectory(args.input,json.loads(args.request.read_text()),args.output,run_directory=args.run_directory,memory_gib=args.memory_gib)
    elif args.kind=='bh-pair-figures':
        from .flagship_bh import export_native_bh_pair_separation
        result=export_native_bh_pair_separation(args.input,json.loads(args.request.read_text()),args.output,run_directory=args.run_directory,memory_gib=args.memory_gib)
    elif args.kind=='bh-merger-figures':
        from .flagship_bh import export_native_bh_merger_timeline
        result=export_native_bh_merger_timeline(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='bh-figures':
        from .flagship_bh import export_native_bh_figures
        result=export_native_bh_figures(args.input,args.output,run_directory=args.run_directory,memory_gib=args.memory_gib)
    elif args.kind=='assembly-orbits':
        from .flagship_orbits import measure_native_assembly_orbits
        result=measure_native_assembly_orbits(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind=='assembly-event-figures':
        from .flagship_orbits import export_native_assembly_event_figures
        result=export_native_assembly_event_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind in {'abundance','host-abundance','matched-host-abundance'}:
        from .flagship_abundance import export_native_abundance,export_native_host_abundance,export_matched_host_abundance
        exporter={'abundance':export_native_abundance,'host-abundance':export_native_host_abundance,'matched-host-abundance':export_matched_host_abundance}[args.kind]
        result=exporter(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'assembly-figures':
        from .assembly import export_native_assembly_figures
        result=export_native_assembly_figures(args.input,args.identity_graph,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'structure-figures':
        from .halo_products import export_native_structure_figures
        result=export_native_structure_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'splashback-figures':
        from .halo_products import export_native_splashback_figures
        result=export_native_splashback_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind=='analysis-figures':
        from core.figures import export_native_analysis_figures
        result=export_native_analysis_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind in {'power-linear','power-evolved','power-matched','power-ensemble'}:
        from .power import export_linear_power,export_evolved_power,export_matched_power
        if args.kind=='power-linear':result=export_linear_power(args.input,args.output,memory_gib=args.memory_gib)
        elif args.kind=='power-evolved':result=export_evolved_power(args.input,args.output,memory_gib=args.memory_gib)
        elif args.kind=='power-ensemble':
            from .power import export_power_ensemble
            result=export_power_ensemble(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
        else:result=export_matched_power(json.loads(args.input.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'profile-preflight':
        result = preflight_flagship_profile(json.loads(args.input.read_text()), args.output, output_directory=args.output_directory)
    elif args.kind == 'stream-epoch':
        result = measure_native_stream_epoch(args.input, json.loads(args.request.read_text()), args.output, memory_gib=args.memory_gib)
    elif args.kind == 'stream-catalogue':
        result = catalogue_native_stream_epoch(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'stream-derived':
        result = derive_native_stream_product(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'encounter-frame':
        result = native_encounter_frame(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'encounter-images':
        from .periodic_images import publish_native_worldline_images
        result = publish_native_worldline_images(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-species-window-modes':
        from .zoom import measure_native_species_window_modes
        request=json.loads(args.request.read_text())
        validate_flagship_product({'operation':args.kind,'request':request,'memory_gib':args.memory_gib})
        result=measure_native_species_window_modes(args.input,request['modes'],args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-quadratic-assembly':
        from .music.archive import audit_native_quadratic_velocity
        result=audit_native_quadratic_velocity(args.input,args.configuration,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-poisson-equations':
        from core.music.archive import audit_music_archive_poisson
        result=audit_music_archive_poisson(args.source_index,args.configuration,args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-gradient-operators':
        from core.music.replay import audit_music_archive_gradients
        result=audit_music_archive_gradients(args.source_index,args.configuration,args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-boundary-reference':
        from core.music.poisson import audit_music_archive_boundary
        result=audit_music_archive_boundary(args.source_index,args.configuration,args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-transfer-replay':
        from core.music.replay import audit_music_archive_transfer
        result=audit_music_archive_transfer(args.source_index,args.configuration,args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-compensated-windows':
        from core.music.windows import measure_native_compensated_windows
        result=measure_native_compensated_windows(args.source_index,args.configuration,args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-flow-noise':
        from core.music.flow import publish_flow_noise_response
        result=publish_flow_noise_response(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-flow-covariance':
        from core.music.covariance import publish_flow_window_covariance
        result=publish_flow_window_covariance(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-compensated-covariance':
        from core.music.covariance import publish_compensated_window_covariance
        result=publish_compensated_window_covariance(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'ic-flow-windows':
        from core.music.windows import measure_native_flow_windows
        result=measure_native_flow_windows(args.source_index,args.configuration,args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'encounter-catalogue':
        result = measure_native_encounters(args.input,json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    elif args.kind == 'encounter-figures':
        result = export_native_encounter_figures(args.input,args.output,memory_gib=args.memory_gib)
    elif args.kind == 'stream-figures': result = export_registered_stream_figures(args.input, args.output, memory_gib=args.memory_gib)
    elif args.kind == 'stream-timeline': result = export_stream_timeline(args.input, args.output, memory_gib=args.memory_gib, maximum_saved_vertices=args.maximum_saved_vertices)
    elif args.kind == 'convergence-figures': result = export_convergence_figures(json.loads(args.input.read_text()), args.output, request_base=args.input.resolve().parent, memory_gib=args.memory_gib)
    else: result = stellar_yield_audit(json.loads(args.input.read_text()), args.output)
    print(json.dumps(json_safe(result), indent=2))
    if args.kind in ('ic-flow-noise','ic-flow-covariance','ic-compensated-covariance') and result.get('status') in ('UNRESOLVED_COVARIANCE','UNRESOLVED_SOLVE'):return 2
    return 1 if result.get('status') == 'FAIL' else 0
