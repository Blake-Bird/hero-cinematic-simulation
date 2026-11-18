"""MUSIC flow preparation, shared-noise responses and pre-evolution checks.

Preparation binds the selected physical answers to the actual native mesh.
Response publication carries density/velocity windows through the coupled
Poisson transpose onto the same noise cells for every species. The launch
check then uses the declared phase covariance and separate species/level
windows. Passing these discrete checks does not establish physical convergence
or continuum boundary accuracy. Native operators load only when requested.
"""
import datetime
import json
import math
import re
from pathlib import Path
import numpy as np
from core.foundation import file_sha256, atomic_json
from core.music.replay import _budget

_CORE_ROOT = Path(__file__).resolve().parent.parent


def validate_flow_noise_request(request):
    if (not isinstance(request,dict) or set(request)-{'response_workspace'}!={'schema','configuration','ic','mesh','noise_directory','boundary_adapter_build','solve'}
            or request['schema']!='native-music-flow-noise-request-v1'
            or any(not isinstance(request[k],str) or not request[k] for k in ('configuration','ic','mesh','noise_directory'))):
        raise ValueError('Explicit full-flow coupled-noise request required')
    if 'response_workspace' in request:
        from core.numeric import validate_response_workspace
        validate_response_workspace(request['response_workspace'])
    binding=request['boundary_adapter_build']
    solve=request['solve']
    if (not isinstance(binding,dict) or set(binding)!={'path','sha256'} or not isinstance(binding['path'],str) or not binding['path']
            or not isinstance(binding['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',binding['sha256'])):
        raise ValueError('Checksum-bound native boundary adapter build required')
    if (not isinstance(solve,dict) or set(solve)!={'restart','max_iterations'}
            or type(solve['restart']) is not int or not 2<=solve['restart']<=256
            or type(solve['max_iterations']) is not int or not 1<=solve['max_iterations']<=100000):
        raise ValueError('Explicit bounded solver iterations required; numerical tolerances are preserved')


def flow_noise_input_bindings(product,request):
    from core.data import verify_numeric_audit,numeric_audit_source_bindings
    from core.music.boundary import verify_boundary_adapter_build
    validate_flow_noise_request(request)
    product=Path(product).resolve()
    binding=request['boundary_adapter_build']
    build=Path(binding['path']).resolve()
    if file_sha256(build)!=binding['sha256']:
        raise ValueError('Native boundary build receipt checksum changed')
    binary,record=verify_boundary_adapter_build(build)
    report=verify_numeric_audit(product)
    if report.get('schema')!='native-music-full-flow-window-response-v1' or report.get('status')!='PASS':
        raise ValueError('Successful original full-flow window response required')
    pins=numeric_audit_source_bindings(report)
    for name in ('configuration','ic','mesh'):
        p=Path(request[name]).resolve()
        if pins.get(str(p))!=file_sha256(p):
            raise ValueError('Full-flow request differs from original response: '+name)
    mesh=json.loads(Path(request['mesh']).read_text())
    for row in mesh['hierarchies']['convolution']:
        p=(Path(request['noise_directory'])/f'wnoise_{row["level"]:04d}.bin').resolve()
        if pins.get(str(p))!=file_sha256(p):
            raise ValueError('Full-flow request noise differs from original response')
    for p in (product,product.with_name(product.name+'.sha256'),product.with_name(product.name+'.data')/'arrays.npz',build,binary):
        pins[str(p)]=file_sha256(p)
    for source,checksum in record['source_sha256'].items():
        if source in pins and pins[source]!=checksum:
            raise ValueError('Conflicting native boundary source binding')
        pins[source]=checksum
    return pins


def publish_flow_noise_response(product,request,destination,*,memory_gib=16.):
    """Map each measured density/velocity channel onto shared native noise.

    For A phi = B source, a potential weight w contributes
    B.T solve(A.T, w) to the source weights. Add the direct source term, then
    transpose transfer/refinement onto the actual noise caches. Gas and dark
    matter share those variables, so sum their responses before covariance.
    Density channels are dimensionless; velocity-divergence channels use
    km/s per comoving kpc/h. The saved row_units records this distinction.
    """
    import h5py
    from core.data import export_numeric_audit,verify_numeric_audit
    from core.stream_common import read_product
    from core.zoom import _configuration
    from core.music.poisson import NativePoissonCoupling
    from core.music.adjoint import potential_source_response, flow_source_noise_response
    from core.music.replay import load_native_noise
    from core.numeric import NumericArraySpool, LazyNpzArrays, preflight_response_disk
    pins=flow_noise_input_bindings(product,request)
    streamed='response_workspace' in request
    if streamed:
        report=verify_numeric_audit(product)
        data=LazyNpzArrays(product,report,memory_gib=memory_gib/3)
    else:
        report,data,_=read_product(product,memory_gib=memory_gib/3)
    cfg=_configuration(request['configuration'])
    mesh=json.loads(Path(request['mesh']).read_text())
    index=Path(report['source_archive_index']['path']).resolve()
    archive=json.loads(index.read_text())
    if file_sha256(index)!=report['source_archive_index']['sha256']:
        raise ValueError('Full-flow source archive index changed')
    physical=mesh['hierarchies']['poisson']
    levels=[row['level'] for row in physical]
    channels=report['channels']
    count=len(channels)
    if not count:
        raise ValueError('Nonempty original flow channels required')
    for field in archive['fields']:
        if [(r['level'],r['shape'],r['origin']) for r in field['grids']]!=[(r['level'],r['shape'],r['offset']) for r in physical]:
            raise ValueError('Actual full-flow source and mesh geometry differ')
    from core.music.resources import noise_response_resources
    publication_preflight=None
    resources=None
    if streamed:
        copied=[k for k in data if k.endswith('_particle_ids') or k.endswith('_selection')]
        resources=noise_response_resources(mesh,count,source_channel_bytes=data.maximum_channel_bytes(),
            metadata_bytes=sum(data.nbytes(k) for k in copied),metadata_arrays=len(copied))
        shapes=resources['cache_shapes']
        resident=resources['input_convolution_resident_bytes']
        publication_preflight=preflight_response_disk(destination,request['response_workspace']['disk_gib'],
            payload_bytes=resources['payload_upper_bound_bytes'],array_count=resources['array_count_upper_bound'])
    else:
        shapes={row['level']:tuple(n+(0 if row['level']==levels[0] else 2*mesh['convolution_margin']) for n in row['shape']) for row in mesh['hierarchies']['convolution']}
        resident=64*1024**2+sum(v.nbytes for v in data.values())*2+(32+16*count)*sum(math.prod(s) for s in shapes.values())
    _budget(memory_gib,resident+128*1024**2)
    operator_budget=memory_gib-resident/1024**3
    geometry=[{'level':row['level'],'origin':row['offset'],'interior_shape':row['shape']} for row in physical]
    table=Path(cfg.get('cosmology','transfer_file'))
    if not table.is_absolute():
        table=Path(request['configuration']).resolve().parent/table
    table=np.loadtxt(table)
    noises={level:load_native_noise(Path(request['noise_directory'])/f'wnoise_{level:04d}.bin',s,
        byte_order=archive['byte_order'],memory_gib=memory_gib/3) for level,s in shapes.items()}
    responses=None if streamed else {level:[] for level in levels}
    arrays=NumericArraySpool(Path(destination).with_name(Path(destination).name+'.rows'),disk_gib=request['response_workspace']['disk_gib'],memory_gib=memory_gib) if streamed else None
    if streamed:
        for level in levels:
            arrays.create_rows(f'response_noise_level{level}',(count,)+shapes[level])
    solves=[]
    predictions=[]
    operator_binding=None
    with NativePoissonCoupling(geometry,request['boundary_adapter_build']['path'],order=cfg.getint('poisson','laplace_order',fallback=6),memory_gib=operator_budget,**request['solve']) as operator:
        operator_binding=operator.source_binding
        for ordinal,channel in enumerate(channels):
            sources={}
            compensation={}
            channel_solves=[]
            for preparation in (0,1):
                prefix=f'channel{ordinal}_prepared{preparation}_'
                if prefix+f'potential_level{levels[0]}' not in data:
                    continue
                potential=[data[prefix+f'potential_level{level}'] for level in levels]
                weights,solve=potential_source_response(operator,potential,tolerance=min(1e-10,cfg.getfloat('poisson','accuracy')),
                    **request['solve'])
                channel_solves.append({'preparation':preparation,**solve})
                sources[preparation]={level:weight+data[prefix+f'source_level{level}'] for level,weight in zip(levels,weights)}
                if prefix+'compensated_source' in data:
                    compensation[preparation]=data[prefix+'compensated_source']
            response,metadata=flow_source_noise_response(sources,compensation,mesh,cfg,table,quantity=channel['quantity'],memory_gib=memory_gib/3)
            predictions.append(sum(float(np.sum(response[level]*noises[level])) for level in levels))
            for level in levels:
                if streamed:
                    arrays.write_row(f'response_noise_level{level}',ordinal,np.asarray(response[level],dtype='<f8'))
                else:responses[level].append(response[level])
            solves.append({'channel':ordinal,'preparations':channel_solves,'source_noise_operator':metadata})
    observed=np.full(count,np.nan)
    checks=[]
    for check in report['checks']:
        indices=check['response_channels']
        if len(indices)!=2 or any(type(i) is not int or not 0<=i<count for i in indices):
            raise ValueError('Exact paired complex flow channels required')
        for index,component in zip(indices,('real','imag')):
            channel=channels[index]
            if (channel['component']!=component or any(channel[k]!=check[k] for k in ('window_id','quantity','mode'))
                    or not np.isnan(observed[index])):raise ValueError('Full-flow response channel ancestry differs')
        values=np.asarray(check['measured'],dtype=float)
        if values.shape!=(2,) or not np.isfinite(values).all():
            raise ValueError('Actual measured full-flow channels required')
        observed[indices]=values
        reference=np.asarray([predictions[i] for i in indices])
        norm=float(np.linalg.norm(values))
        difference=float(np.linalg.norm(reference-values))
        error=difference/norm if norm else 0. if difference==0 else math.inf
        checks.append({k:check[k] for k in ('window_id','quantity','mode','response_channels')}|{'measured':values.tolist(),
            'reference':reference.tolist(),'relative_l2_error':error if math.isfinite(error) else None,'status':'PASS' if error<=1e-7 else 'FAIL'})
    if not np.isfinite(observed).all():
        raise ValueError('Missing measured full-flow channels')
    converged=all(s['status']=='PASS' for channel in solves for s in channel['preparations'])
    status='UNRESOLVED_SOLVE' if not converged else 'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'
    if not streamed:
        arrays={f'response_noise_level{level}':np.stack(rows) for level,rows in responses.items()}
    arrays.update(observed_real_channels=observed,predicted_real_channels=np.asarray(predictions))
    specs={k:{'units':'Per-channel units in row_units; response per unit native noise' if k.startswith('response') else 'Per-channel units in row_units','quantity':k} for k in arrays}
    for key in data:
        if key.endswith('_particle_ids') or key.endswith('_selection'):
            if streamed:
                data.copy_member(key,arrays)
            else:
                arrays[key]=data[key]
            specs[key]={'units':'identity' if key.endswith('_particle_ids') else 'dimensionless boolean support','quantity':key}
            if key.endswith('_particle_ids'):
                species=int(key.split('_')[0][7:])
                specs[key]['identity_namespace']=f'native-music-ic:{pins[str(Path(request["ic"]).resolve())]}:PartType{species}'
    with h5py.File(request['ic']) as f:
        shift=np.asarray(f['Header'].attrs['Music_output_coordinate_shift_box_fraction'],dtype=float).tolist()
    row_units=['dimensionless density mode' if c['quantity']=='density' else 'km/s per comoving kpc/h velocity divergence mode' for c in channels]
    for module in (
        'music/flow.py',
        'music/adjoint.py',
        'music/poisson.py',
        'music/boundary.py',
        'music/replay.py',
        'music/windows.py',
        'numeric.py',
        'music/resources.py',
        'data.py',
        'foundation.py',
    ):
        p=(_CORE_ROOT / module).resolve()
        pins[str(p)]=file_sha256(p)
    result={'schema':'native-music-full-flow-noise-response-v1','status':status,'source_response':{'path':str(Path(product).resolve()),'sha256':file_sha256(product)},
        'source_archive_index':report['source_archive_index'],'mesh':mesh,'output_shift_box_fraction':shift,'channels':channels,'row_units':row_units,
        'windows':report['windows'],'checks':checks,'coupled_solves':solves,'coupled_operator':operator_binding,
        'response_resources':resources,'mass_normalization':'Original complete species/refinement selections; full-fine compensated totals and constant coarse masses',
        'maximum_relative_operator_error':1e-7,'scope':'Full original linear density/velocity windows on actual shared hierarchical noise, including native coupled potentials, hybrid sources and compensated masses. Explicit species/refinement selections; no joint covariance or continuum accuracy acceptance.',
        'physical_boundary_accuracy':'NOT_ESTABLISHED','launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    if streamed:
        result['response_publication']={**arrays.metadata(),'preflight':publication_preflight}
    epoch=json.loads(next(iter(report['numeric_data']['array_contract']['datasets'].values()))['epoch_json'])
    return export_numeric_audit(destination,result,arrays,specs,algorithm=__file__,context={'epoch':epoch},source_hashes=pins)


def original_flow_model_bindings(request,*,levels=None):
    """Verify current adapter and the explicitly selected native noise model."""
    from core.music.boundary import verify_boundary_adapter_build
    from core.data import verify_numeric_audit,numeric_audit_source_bindings
    validate_original_flow_launch_request(request)
    pins={}
    build=Path(request['boundary_adapter_build']['path']).resolve()
    if file_sha256(build)!=request['boundary_adapter_build']['sha256']:
        raise ValueError('Native adapter preparation binding changed')
    binary,record=verify_boundary_adapter_build(build)
    pins.update(record['source_sha256'])
    pins.update({str(build):file_sha256(build),str(binary):file_sha256(binary)})
    model=Path(request['noise_model_verification']['path']).resolve()
    if file_sha256(model)!=request['noise_model_verification']['sha256']:
        raise ValueError('Native phase-model preparation binding changed')
    proof=verify_numeric_audit(model)
    if proof.get('schema') not in ('native-frozen-noise-generation-model-control-v1','native-local-seeded-noise-model-v1') or proof.get('status')!='PASS':
        raise ValueError('Verified supported native noise generation model required')
    local=proof['schema']=='native-local-seeded-noise-model-v1'
    if local!=(request['parent_conditioning']=='fixed_prefix'):
        raise ValueError('Local noise model requires fixed-prefix conditioning; full-frozen conditioning remains separate')
    if levels is not None and [row['level'] for row in proof['checks']][:len(levels)]!=list(levels):
        raise ValueError('Verified noise model does not cover the actual native hierarchy prefix; prepare its phase strategy first')
    pins.update(numeric_audit_source_bindings(proof))
    pins[str(model)]=file_sha256(model)
    pins[str(model.with_name(model.name+'.data')/'arrays.npz')]=file_sha256(model.with_name(model.name+'.data')/'arrays.npz')
    for name in (
        'music/flow.py',
        'music/windows.py',
        'music/precision.py',
        'music_binary128_reference.cc',
        'music/archive.py',
        'music/replay.py',
        'music/covariance.py',
        'music/seeded.py',
        'numeric.py',
        'music/resources.py',
        'music/poisson.py',
        'music/boundary.py',
        'data.py',
        'foundation.py',
    ):
        path=(_CORE_ROOT / name).resolve()
        pins[str(path)]=file_sha256(path)
    return pins


def prepare_original_flow_answers(request,destination):
    """Bind physical answers and separate species/level windows to this mesh.

    Resolution comes from the actual finest level, not an unrelated template.
    Box length, starting epoch and hydro settings must already agree. This
    saves inputs for the launch check; it neither runs nor approves evolution.
    """
    from core.zoom import _configuration
    from core.music.resources import rectangular_particle_census
    from core.research.capabilities import validate_native_answers
    if (not isinstance(request,dict) or set(request)-{'seeded_covariance','response_workspace'}!={'schema','answers','configuration','mesh','boundary_adapter_build','noise_model_verification','modes','parent_conditioning','solve'}
            or request['schema']!='native-flow-answer-preparation-request-v1'):
        raise ValueError('Explicit native flow answer preparation request required')
    target=Path(destination).resolve()
    if target.exists():
        raise FileExistsError('Keep previous native flow answer preparation')
    pins={}
    def bound(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('Exact native flow preparation input binding required')
        p=Path(ref['path']).resolve()
        if file_sha256(p)!=ref['sha256']:
            raise ValueError('Native flow preparation input changed: '+str(p))
        pins[str(p)]=ref['sha256']
        return p
    answers=json.loads(bound(request['answers']).read_text())
    configuration=bound(request['configuration'])
    mesh=json.loads(bound(request['mesh']).read_text())
    cfg=_configuration(configuration)
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if not low<high or not cfg.getboolean('setup','baryons'):
        raise ValueError('Original nested gas/DM hierarchy required')
    census=rectangular_particle_census(mesh,configuration)
    if census['status']!='EXACT_GEOMETRY_CENSUS':
        raise ValueError('Exact original species/refinement census required')
    populations={0:[high],1:[high],2:[row['level'] for row in census['leaf_cells_by_level'][:-1] if row['leaf_cells']]}
    for parent,child in zip(mesh['hierarchies']['poisson'][:-1],mesh['hierarchies']['poisson'][1:]):
        offset=[(o//2-p)%2**parent['level'] for o,p in zip(child['offset'],parent['offset'])]
        if any(o<4 or o+n//2+4>p for o,n,p in zip(offset,child['shape'],parent['shape'])):
            raise ValueError('Native coupled window preparation requires four parent interior cells around each child')
    launch={k:request[k] for k in ('modes','boundary_adapter_build','noise_model_verification','parent_conditioning','solve')}
    if 'seeded_covariance' in request:
        launch['seeded_covariance']=request['seeded_covariance']
    if 'response_workspace' in request:
        launch['response_workspace']=request['response_workspace']
    launch.update(schema='native-original-flow-launch-request-v1',windows=[{'id':f'species{species}-level{level}','particle_types':[species],'refinement_levels':[level]} for species,levels in populations.items() for level in levels])
    require_original_flow_coverage(launch,populations)
    pins.update(original_flow_model_bindings(launch,levels=range(low,high+1)))
    if answers.get('physics')!='hydro' or not math.isclose(float(answers['box_mpc_h']),cfg.getfloat('setup','boxlength'),rel_tol=1e-12):
        raise ValueError('Actual native configuration and supplied hydro answers disagree')
    # Resolution follows the actual pinned mesh, with the adjustment explicit
    # in the receipt. Selected targets, physics, phases and softenings persist.
    original_resolution=answers['particles_per_dimension']
    answers['particles_per_dimension']=2**high
    answers['native_flow_window_comparison']=launch
    resources={'mpi_ranks':answers['mpi_ranks'],'omp_threads':answers['omp_threads'],'memory_gib':answers['memory_limit_gb']}
    assembled=validate_native_answers(answers,resources)
    if assembled['simulation']['native_flow_window_comparison']!=launch:
        raise ValueError('Public builder dropped the frozen native window request')
    if not math.isclose(float(assembled['simulation']['start_redshift']),cfg.getfloat('setup','zstart'),rel_tol=1e-12):
        raise ValueError('Actual native start epoch differs from the supplied cosmology answer bundle')
    for name in (
        'music/flow.py',
        'music/resources.py',
        'music/boundary.py',
    ):
        p=(_CORE_ROOT / name).resolve()
        pins[str(p)]=file_sha256(p)
    p=_CORE_ROOT.parent/'commands.py'
    pins[str(p)]=file_sha256(p)
    for p,digest in pins.items():
        if file_sha256(p)!=digest:
            raise ValueError('Native flow preparation input changed before publication')
    target.mkdir(parents=True,exist_ok=False)
    atomic_json(target/'answers.json',answers)
    atomic_json(target/'launch-request.json',launch)
    result={'schema':'native-flow-answer-preparation-v1','status':'PREPARED','source_hashes':pins,
        'answers':{'path':str(target/'answers.json'),'sha256':file_sha256(target/'answers.json')},
        'launch_request':{'path':str(target/'launch-request.json'),'sha256':file_sha256(target/'launch-request.json')},
        'populations_by_level':{str(k):v for k,v in populations.items()},'particle_census':census,
        'input_answer_particles_per_dimension':original_resolution,'actual_mesh_particles_per_dimension':2**high,
        'scope':'Complete original native species/refinement windows and bound model/build settings frozen through the public answers builder. No native generation, launch, evolution or full miniature.',
        'science_acceptance':'NOT_ACCEPTED','launch_ready':False}
    atomic_json(target/'preparation.json',result)
    return result


def native_flow_public_command_bindings(parameters):
    """Include nested adapter/model dependencies in evolution cache keys."""
    if parameters.get('command')=='prepare-generated-flow':
        from core.music.staging import generated_flow_bindings
        arguments=parameters.get('arguments',[])
        if arguments.count('--request')!=1 or arguments.index('--request')+1>=len(arguments):
            raise ValueError('Generated-flow command needs one resolved request')
        path=Path(arguments[arguments.index('--request')+1]).resolve()
        return {str(path):file_sha256(path),**generated_flow_bindings(json.loads(path.read_text()))}
    if parameters.get('command')!='gadget':
        return {}
    arguments=parameters.get('arguments',[])
    if '--answers' not in arguments:
        return {}
    position=arguments.index('--answers')+1
    if position>=len(arguments) or not isinstance(arguments[position],str):
        raise ValueError('Resolved actual public evolution answers path required')
    path=Path(arguments[position]).resolve()
    answers=json.loads(path.read_text())
    request=answers.get('native_flow_window_comparison')
    if request is None:
        return {}
    return {str(path):file_sha256(path),**original_flow_model_bindings(request)}


def _response_stage_settings(request,stage):
    return ({'response_workspace':{'disk_gib':request['response_workspace'][stage+'_disk_gib']}}
        if 'response_workspace' in request else {})


def validate_original_flow_launch_request(request):
    from core.music.windows import validate_flow_window_request
    from core.music.covariance import validate_flow_covariance_request
    if (not isinstance(request,dict) or set(request)-{'seeded_covariance','response_workspace'}!={'schema','modes','windows','boundary_adapter_build','noise_model_verification','parent_conditioning','solve'}
            or request['schema']!='native-original-flow-launch-request-v1'):
        raise ValueError('Nested original MUSIC launch requires an explicit native_flow_window_comparison request')
    if 'response_workspace' in request or request['parent_conditioning']=='fixed_prefix':
        from core.numeric import validate_response_workspace
        validate_response_workspace(request.get('response_workspace'),combined=True)
    # Validate reusable request components before any output is created.
    validate_flow_window_request({'schema':'native-music-flow-window-request-v1','mesh':'.','noise_directory':'.',
        'modes':request['modes'],'quantities':['density','velocity'],'windows':request['windows'],**_response_stage_settings(request,'source')})
    validate_flow_noise_request({'schema':'native-music-flow-noise-request-v1','configuration':'.','ic':'.','mesh':'.','noise_directory':'.',
        'boundary_adapter_build':request['boundary_adapter_build'],'solve':request['solve'],**_response_stage_settings(request,'noise')})
    validate_flow_covariance_request({'schema':'native-flow-window-covariance-request-v1',
        'noise_model_verification':request['noise_model_verification'],'parent_conditioning':request['parent_conditioning'],
        **({'seeded_covariance':request['seeded_covariance']} if 'seeded_covariance' in request else {})})
    if np.linalg.matrix_rank(np.asarray(request['modes'],dtype=float))!=3:
        raise ValueError('Original native window launch modes must span all three spatial axes')


def require_original_flow_coverage(request,populations):
    """Require complete, separate species and each actual refinement level."""
    validate_original_flow_launch_request(request)
    for species,levels in populations.items():
        for level in levels:
            if not any(window['particle_types']==[species] and window['refinement_levels']==[level] for window in request['windows']):
                raise ValueError(f'Original flow gate omits a separate complete species/refinement window: {species}/{level}')


def native_original_flow_window_gate(cfg,run_dir,ic_path):
    """Check the exact delivered IC's discrete windows before evolution.

    Follow measurements -> coupled responses -> declared phase covariance ->
    familywise Gaussian blocks. Every actual species/refinement population
    needs its own complete window; a mixed total could hide a species error.
    Failure or interruption preserves a diagnostic and prevents launch.
    """
    import h5py
    from core.resources import memory_limit_gb
    from core.data import snapshot_total_counts,numeric_audit_source_bindings,verify_numeric_audit
    from core.stream_common import read_product
    from core.zoom import _configuration,_native_cell_levels,verify_generated_zoom
    from core.music.windows import measure_native_flow_windows
    from core.music.covariance import publish_flow_window_covariance, flow_gaussian_blocks
    run_dir=Path(run_dir).resolve()
    ic=Path(ic_path).resolve()
    memory=memory_limit_gb(cfg)
    request=cfg['simulation'].get('native_flow_window_comparison')
    validate_original_flow_launch_request(request)
    manifest=json.loads((run_dir/'manifest.json').read_text())
    binding=manifest.get('validation',{}).get('native_ic_handoff',{})
    if not {'report','report_sha256'}<=set(binding):
        raise ValueError('Original flow launch requires the preceding source-bound native technical handoff')
    handoff_path=Path(binding['report']).resolve()
    if file_sha256(handoff_path)!=binding['report_sha256']:
        raise ValueError('Original native technical handoff changed')
    handoff=json.loads(handoff_path.read_text())
    generation_path=Path(cfg['simulation']['native_generation_report']).resolve()
    if (handoff.get('status')!='TECHNICAL_PASS' or handoff.get('input_kind')!='zoom'
            or handoff['generation_report']!={'path':str(generation_path),'sha256':file_sha256(generation_path)}
            or handoff.get('structural_audit',{}).get('status')!='STRUCTURAL_PASS'
            or handoff.get('source_archive_binding',{}).get('actual_ic_sha256')!=file_sha256(ic)):
        raise ValueError('Exact original nested IC and current technical handoff required')
    verify_generated_zoom(generation_path)
    base=generation_path.parent
    if ic!=base/'ics_gadget.hdf5':
        raise ValueError('Original flow gate requires the exact generated IC')
    configuration=base/'music.conf'
    _configuration(configuration)
    index=handoff_path.parent/'source-hierarchy-index.json'
    populations={}
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        counts=snapshot_total_counts(h)
        box=float(h['BoxSize'])
        low=int(h['Music_levelmin'])
        high=int(h['Music_levelmax'])
        for species in (0,1,2):
            if not counts[species]:
                continue
            levels=set()
            width=f[f'PartType{species}/LagrangianCellWidths']
            for start in range(0,len(width),65536):
                levels.update(map(int,_native_cell_levels(np.asarray(width[start:start+65536]),box,low,high)))
            populations[species]=sorted(levels)
    require_original_flow_coverage(request,populations)
    fraction=cfg.get('validation',{}).get('ic_velocity_divergence',{}).get('kmax_fraction_nyquist',.25)
    modes=np.asarray(request['modes'],dtype=float)
    # Integer box modes have cell Nyquist radius 2**(level-1). Apply the
    # configured fraction at the coarsest population present, not the finest.
    if not 0<float(fraction)<=1 or np.max(np.linalg.norm(modes,axis=1))>fraction*2**(min(min(v) for v in populations.values())-1):
        raise ValueError('Original window launch modes exceed the declared coarsest-cell Nyquist support')
    directory=run_dir/'input'/('original-flow-window-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True,exist_ok=False)
    result={'schema':'native-original-flow-launch-gate-v1','status':'IN_PROGRESS','request':request,'technical_handoff':binding,
        'generation_report':{'path':str(generation_path),'sha256':file_sha256(generation_path)},
        'actual_ic':{'path':str(ic),'sha256':file_sha256(ic)},'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
    try:
        flow=measure_native_flow_windows(index,configuration,ic,{'schema':'native-music-flow-window-request-v1',
            'mesh':str(base/'planning/native-mesh.json'),'noise_directory':str(base),'modes':request['modes'],
            'quantities':['density','velocity'],'windows':request['windows'],**_response_stage_settings(request,'source')},directory/'flow-windows.json',memory_gib=memory)
        if flow['status']!='PASS':
            raise ValueError('Original full-flow assembly replay failed')
        response=publish_flow_noise_response(directory/'flow-windows.json',{'schema':'native-music-flow-noise-request-v1',
            'configuration':str(configuration),'ic':str(ic),'mesh':str(base/'planning/native-mesh.json'),'noise_directory':str(base),
            'boundary_adapter_build':request['boundary_adapter_build'],'solve':request['solve'],**_response_stage_settings(request,'noise')},directory/'flow-noise.json',memory_gib=memory)
        if response['status']!='PASS':
            raise ValueError('Original full-flow coupled response failed or solver did not converge')
        publish_flow_window_covariance(directory/'flow-noise.json',{'schema':'native-flow-window-covariance-request-v1',
            'noise_model_verification':request['noise_model_verification'],'parent_conditioning':request['parent_conditioning'],
            **({'seeded_covariance':request['seeded_covariance']} if 'seeded_covariance' in request else {})},directory/'flow-covariance.json',memory_gib=memory)
        # Test the predeclared per-window/quantity blocks with Bonferroni.
        # Cross-block redundancy need not admit an inverse of the full matrix.
        report,arrays,_=read_product(directory/'flow-covariance.json',memory_gib=memory)
        alpha=cfg.get('validation',{}).get('ic_power_spectrum',{}).get('false_rejection_probability',.001)
        check=flow_gaussian_blocks(report['channels'],arrays['centered_real_channels'],arrays['joint_covariance'],false_rejection_probability=alpha)
        if check['status']!='PASS':
            raise ValueError('Original full-flow familywise Gaussian check failed or has unresolved covariance: '+check['status'])
        for name in ('flow-windows','flow-noise','flow-covariance'):
            path=directory/(name+'.json')
            numeric_audit_source_bindings(verify_numeric_audit(path))
            result[name]={'path':str(path),'sha256':file_sha256(path)}
        result.update(status='PASS',gaussian_check=check,particle_ids_unique=True,
            particle_id_validation={'method':'Exact cross-species uint64 uniqueness in the preceding bound native structural handoff','handoff':binding},
            physics_validation_scope='Original species/refinement linear density/velocity windows against actual native discrete hierarchy, transfer and verified phase/noise model; physical convergence remains separate',
            linear_spectrum_acceptance='SCOPED_NATIVE_DISCRETE_WINDOW_CONSISTENCY',physical_boundary_accuracy='NOT_ESTABLISHED',launch_authorized=True)
    except BaseException as error:
        result['status']='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAIL'
        result['failure']={'type':type(error).__name__,'message':str(error)}
        atomic_json(directory/'launch.json',result)
        raise
    atomic_json(directory/'launch.json',result)
    return result
