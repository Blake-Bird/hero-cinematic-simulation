"""Profile input validation; resource suitability never certifies science."""

import copy
import json
import math
from core.foundation import file_sha256
from .native import native_particle_index_requirement
from pathlib import Path


def validate_profile_resources(resources):
    for name in ('machine_ram_gib','memory_gib','disk_gib','disk_margin_gib',
                 'evolution_bytes_per_particle','snapshot_bytes_per_particle'):
        value=resources[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Positive finite profile resource required: '+name)
    for name in ('mpi_ranks','omp_threads','cpu_slots','build_jobs','saved_epochs','checkpoint_copies'):
        if type(resources[name]) is not int or resources[name]<1:
            raise ValueError('Positive exact profile integer required: '+name)
    if resources['memory_gib']>resources['machine_ram_gib']:
        raise ValueError('Application allocation exceeds machine RAM')
    if resources['mpi_ranks']*resources['omp_threads']>resources['cpu_slots'] or resources['build_jobs']>resources['cpu_slots']:
        raise ValueError('MPI/thread/build settings exceed CPU allocation')


def preflight_planned_geometry(request,destination,*,output_directory):
    """Budget pinned native geometry; never present an upper bound as a census."""
    import json
    import shutil
    from .foundation import atomic_json, file_sha256, RHO_CRIT_MSUN_H_MPC_H3, canonical_digest
    from core.resources import periodic_image_resource_allowance
    from .zoom import estimate_mesh_resources,_configuration
    from core.music.resources import rectangular_particle_census
    if request.get('schema')!='flagship-planned-geometry-profile-v1':
        raise ValueError('Explicit planned geometry profile required')
    resources=request['resources'];validate_profile_resources(resources)
    required=request['required_effective_level']
    if type(required) is not int or not 1<=required<=20:raise ValueError('Explicit supported science refinement level required')
    sources={str(Path(__file__).resolve()):file_sha256(__file__),
        str(Path(__file__).with_name('foundation.py').resolve()):file_sha256(Path(__file__).with_name('foundation.py'))}
    def load(reference):
        if set(reference)!={'path','sha256'}:raise ValueError('Exact planned input path/SHA256 required')
        path=Path(reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Planned geometry profile source changed')
        sources[str(path)]=reference['sha256'];return json.loads(path.read_text())
    spectrum=load(request['cosmology']);findings=load(request['geometry_findings'])
    capacity=load(request['generation_handoff_capacity']) if request.get('generation_handoff_capacity') is not None else None
    if capacity is not None:
        if capacity.get('schema')!='native-generation-handoff-capacity-report-v1':raise ValueError('Bound native generation/handoff capacity report required')
        for asset,checksum in capacity['source_hashes'].items():
            if file_sha256(asset)!=checksum:raise ValueError('Generation/handoff capacity dependency changed')
            sources[str(Path(asset).resolve())]=checksum
    if findings.get('schema')!='flagship-production-geometry-audit-v1':
        raise ValueError('Original retained production geometry findings required')
    for asset,digest in findings.get('source_hashes',{}).items():
        path=Path(asset).resolve()
        if file_sha256(path)!=digest:raise ValueError('Production geometry lineage source changed')
        sources[str(path)]=digest
    selected=request['case_ids']
    if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected):raise ValueError('Distinct actual planned case IDs required')
    rows=[];failures=[]
    for name in selected:
        matches=[row for row in findings['rows'] if row['id']==name]
        if len(matches)!=1:raise ValueError('Planned case missing or duplicated in geometry source')
        row=matches[0];configuration=Path(row['configuration']).resolve()
        if file_sha256(configuration)!=row['configuration_sha256']:raise ValueError('Original planned configuration changed')
        sources[str(configuration)]=row['configuration_sha256']
        binary=Path(row['binary']['path']).resolve()
        if file_sha256(binary)!=row['binary']['sha256']:raise ValueError('Retained geometry planner binary changed')
        sources[str(binary)]=row['binary']['sha256']
        cfg=_configuration(configuration);level=cfg.getint('setup','levelmax');parent=cfg.getint('setup','levelmin')
        validate_case_configuration({'level':level},cfg,{'phases':{'parent_level':parent}},spectrum)
        mesh=row['native_geometry'];poisson=mesh['hierarchies']['poisson']
        if poisson[0]['level']!=parent or poisson[-1]['level']!=level:
            raise ValueError('Planned mesh hierarchy differs from its original configuration')
        estimate=estimate_mesh_resources(mesh)
        from core.music.resources import estimate_music_phase_resources
        phase=estimate_music_phase_resources(mesh,configuration,threads=resources['omp_threads'])
        sources.update(phase['source_hashes'])
        estimate['phase_generation']=phase
        estimate['phase_generation_working_set_bytes']=phase['phase_peak_working_set_bytes']
        estimate['estimated_memory_bytes']+=phase['phase_peak_working_set_bytes']
        estimate['estimated_disk_bytes']+=max(0,phase['cache_storage_bytes']-estimate['noise_cache_bytes'])
        estimate['phase_allowance_policy']='Conservative addition to bulk mesh/FFT allowances; overlapping payloads not subtracted'
        census=rectangular_particle_census(mesh,configuration)
        exact=census['status']=='EXACT_GEOMETRY_CENSUS'
        if exact:
            from core.music.resources import native_header_count_requirements
            header=native_header_count_requirements(census,configuration)
            sources.update(header['source_hashes']);estimate['header_count_contract']=header
            if header['status']!='SUPPORTED_ENCODING':failures.append(name+': native output header capacity unsupported')
        count=census['total_particle_count'] if exact else estimate['particle_count_upper_bound']
        if exact:sources.update(census['algorithm_sources'])
        image_bytes=periodic_image_resource_allowance(count)
        evolution=count*resources['evolution_bytes_per_particle']+image_bytes
        snapshots=(count*resources['snapshot_bytes_per_particle']+image_bytes)*(resources['saved_epochs']+resources['checkpoint_copies'])
        handoff=None
        if capacity is not None:
            matches=[c for c in capacity['cases'] if c['id']==name]
            if len(matches)!=1:raise ValueError('Generation/handoff capacity case missing or duplicated')
            handoff=matches[0]
            if handoff['configuration']!={'path':str(configuration),'sha256':row['configuration_sha256']}:
                raise ValueError('Capacity configuration differs from actual planned case')
            if canonical_digest(load(handoff['mesh']))!=canonical_digest(mesh):raise ValueError('Capacity hierarchy differs from actual planned case')
        generation_peak=handoff['sequential_peak_memory_bytes'] if handoff is not None else estimate['estimated_memory_bytes']
        pre_evolution_disk=handoff['retained_pre_evolution_disk_bytes'] if handoff is not None else estimate['estimated_disk_bytes']
        peak=max(generation_peak,evolution);storage=pre_evolution_disk+snapshots
        h=cfg.getfloat('cosmology','H0')/100.;cell=cfg.getfloat('setup','boxlength')**3/2**(3*level)
        mass={'gas':RHO_CRIT_MSUN_H_MPC_H3*cfg.getfloat('cosmology','Omega_b')*cell/h,
            'dm':RHO_CRIT_MSUN_H_MPC_H3*(cfg.getfloat('cosmology','Omega_m')-cfg.getfloat('cosmology','Omega_b'))*cell/h}
        if level<required:failures.append(name+': selected resolution below science requirement')
        if peak>resources['memory_gib']*1024**3:failures.append(name+': estimated memory exceeds allocation')
        rows.append({'id':name,'parent_level':parent,'effective_level':level,'fine_mean_particle_mass_msun':mass,
            'geometry_estimate':estimate,'planned_particle_census':census,'actual_particle_census':None,
            'evolution_particle_count':count,'particle_count_scope':'exact count implied by native box/Arepo geometry; no generated IC census' if exact else 'conservative native geometry upper bound; no generated IC census',
            'evolution_memory_estimate_bytes':evolution,'peak_memory_estimate_bytes':peak,'storage_estimate_bytes':storage,
            'generation_handoff_memory_estimate_bytes':generation_peak,'generation_handoff_capacity':None if handoff is None else request['generation_handoff_capacity'],
            'handoff_resource_status':'INCLUDED' if handoff is not None else 'NOT_INCLUDED; bind a generation_handoff_capacity report to size original-window validation',
            'periodic_image_memory_allowance_bytes':image_bytes,'periodic_image_bytes_per_snapshot':image_bytes,
            'phase_strategy_status':row['phase_strategy'],'native_planner_scope':row['planner_scope']})
    probe=Path(output_directory).resolve()
    while not probe.exists():probe=probe.parent
    free=shutil.disk_usage(probe).free;budget=min(resources['disk_gib']*1024**3,free-resources['disk_margin_gib']*1024**3)
    storage=sum(row['storage_estimate_bytes'] for row in rows)
    if storage>budget:failures.append('Campaign storage exceeds declared/available disk after reserve')
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Planning input changed during preflight')
    result={'schema':'flagship-planned-resource-preflight-v1','status':'FAIL' if failures else 'ESTIMATE_ONLY',
        'failures':failures,'resources':resources,'cases':rows,'source_hashes':sources,
        'available_disk_bytes':free,'usable_disk_budget_bytes':budget,'campaign_storage_estimate_bytes':storage,
        'launch_ready':False,'science_acceptance':'NOT_ACCEPTED',
        'remaining_requirements':['Supported parent/refinement phase strategy','Current native generation/handoff acceptance',
            'Generated actual species census','Calibrated native peak memory and convergence'],
        'scope':'Retained geometry estimates with explicit phase payloads; bound generation/handoff capacity includes source, coupled solver and covariance stages. Missing handoff capacity remains reported; no execution or science certification'}
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Keep previous planned resource receipt')
    atomic_json(destination,result)
    if failures:raise MemoryError('; '.join(failures))
    return result




def size_production_requirements(planning,*,headroom_fraction):
    """Turn estimates into a hardware request, never a claim of allocation."""
    if planning.get('schema')!='flagship-planned-resource-preflight-v1' or not planning.get('cases'):
        raise ValueError('Actual pinned production geometry planning required')
    if isinstance(headroom_fraction,bool) or not isinstance(headroom_fraction,(int,float)) or not math.isfinite(headroom_fraction) or headroom_fraction<0:
        raise ValueError('Explicit nonnegative finite planning headroom required')
    multiplier=1+headroom_fraction
    exact=all(row.get('planned_particle_census',{}).get('status')=='EXACT_GEOMETRY_CENSUS' for row in planning['cases'])
    indexing=[native_particle_index_requirement(row['planned_particle_census']['total_particle_count']) for row in planning['cases']] if exact else []
    minimum=max((row['minimum_mpi_ranks'] for row in indexing),default=None)
    requested=planning['resources']['mpi_ranks']
    index_status='UNKNOWN_WITHOUT_EXACT_CENSUS' if minimum is None else 'REQUESTED_RANK_COUNT_BELOW_NECESSARY_FLOOR' if requested<minimum else 'NECESSARY_PARTICLE_CAPACITY_ONLY'
    generation=max(row.get('generation_handoff_memory_estimate_bytes',row['geometry_estimate']['estimated_memory_bytes']) for row in planning['cases'])
    evolution=max(row['evolution_memory_estimate_bytes'] for row in planning['cases'])
    storage=planning['campaign_storage_estimate_bytes']
    gib=1024**3
    return {'schema':'flagship-production-hardware-requirements-v1','status':'REQUIREMENTS_PREPARED_ALLOCATION_UNVERIFIED',
        'selected_cases':[{'id':row['id'],'parent_level':row['parent_level'],'effective_level':row['effective_level'],
            'fine_mean_particle_mass_msun':row['fine_mean_particle_mass_msun'],
            'particle_count_upper_bound':row['geometry_estimate']['particle_count_upper_bound'],
            'planned_particle_census':row.get('planned_particle_census'),
            'evolution_particle_count':row.get('evolution_particle_count',row['geometry_estimate']['particle_count_upper_bound'])} for row in planning['cases']],
        'estimated_generation_working_set_gib':generation/gib,'estimated_evolution_working_set_gib':evolution/gib,
        'estimated_campaign_storage_gib':storage/gib,
        'requested_generation_memory_gib':math.ceil(generation*multiplier/gib),
        'requested_evolution_memory_gib':math.ceil(evolution*multiplier/gib),
        'requested_campaign_storage_gib':math.ceil(storage*multiplier/gib)+planning['resources']['disk_margin_gib'],
        'headroom_fraction':headroom_fraction,'headroom_scope':'Explicit provisioning assumption; not calibrated runtime overhead or a numerical acceptance tolerance',
        'execution_policy':'Sequential targets; generation and evolution capacities are separate, concurrency requires summed capacities',
        'requested_execution_controls':{k:planning['resources'][k] for k in ('mpi_ranks','omp_threads','build_jobs')},
        'local_allocation_unchanged':planning['resources'],'local_failures':planning['failures'],
        'native_particle_indexing':{'status':index_status,'minimum_mpi_ranks':minimum,'requested_mpi_ranks':requested,'cases':indexing},
        'actual_hardware_allocation':None,'launch_ready':False,'science_acceptance':'NOT_ACCEPTED',
        'source_hashes':{**planning['source_hashes'],**{path:digest for row in indexing for path,digest in row['source_hashes'].items()}},'remaining_requirements':planning['remaining_requirements'],
        'limitations':['Supported box/Arepo particle counts are exact geometry-derived counts; unsupported plugins retain upper bounds. Emitted IC census and measured runtime resources remain required',
            'Generation/handoff stages are included only for cases with a current bound capacity report; other handoff workspace remains additional',
            'Rank/node distribution and scheduler limits must be checked against actual hardware; aggregate memory does not prove per-process feasibility',
            'Wall time and peak RSS are not calibrated']}


def publish_production_requirements(request,destination,*,headroom_fraction):
    import json
    import tempfile
    from .foundation import atomic_json
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous production hardware requirements')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.production-sizing-',dir=target.parent) as temp:
        path=Path(temp)/'planning.json'
        try:preflight_planned_geometry(request,path,output_directory=target.parent)
        except MemoryError:
            if not path.is_file():raise
        planning=json.loads(path.read_text())
        result=size_production_requirements(planning,headroom_fraction=headroom_fraction)
        result['planning_receipt']=planning
        atomic_json(target,result)
    return result


def extend_refinement_seeds(seeds,refinement_level):
    """Keep frozen seeds and append disjoint existing MUSIC cube streams."""
    from .zoom import _music_seed_stream_ranges
    levels=sorted(map(int,seeds))
    if not levels or levels!=list(range(levels[0],levels[-1]+1)) or type(refinement_level) is not int or not levels[-1]<=refinement_level<=20:
        raise ValueError('Contiguous retained seeds and supported higher refinement required')
    result=dict(seeds);streams=_music_seed_stream_ranges(result)
    next_seed=max(row['last_seed'] for row in streams)+1
    for level in range(levels[-1]+1,refinement_level+1):
        if next_seed>2147483647:raise ValueError('New MUSIC seed exceeds supported signed range')
        result[str(level)]=next_seed
        streams=_music_seed_stream_ranges(result);next_seed=max(row['last_seed'] for row in streams)+1
    return result,streams


def production_bridge_bindings(reference,original,seeds):
    """Check the executed bridge against the original prefix and allocated seed."""
    import json
    from .foundation import file_sha256
    pins={};loaded={}
    if not isinstance(reference,dict) or set(reference)!={'product','build','invocation'}:
        raise ValueError('Explicit bridge product/build/invocation bindings required')
    for key,binding in reference.items():
        if set(binding)!={'path','sha256'}:raise ValueError('Exact phase bridge input binding required')
        path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('Phase bridge source changed')
        pins[str(path)]=binding['sha256'];loaded[key]=json.loads(path.read_text())
    product,build,native=loaded['product'],loaded['build'],loaded['invocation']
    if product.get('schema')!='frozen-phase-bridge-validated-control-v1' or product.get('status')!='LOW_MODE_GATE_PASSED':
        raise ValueError('Original executed native low-mode bridge gate required')
    if build.get('schema')!='native-helper-build-v1' or native.get('schema')!='native-command-invocation-v1' or native.get('status')!='RETURNED' or native.get('returncode')!=0:
        raise ValueError('Successful original native bridge build/invocation required')
    for key in ('build','invocation'):
        path=str(Path(reference[key]['path']).resolve())
        if product['source_hashes'].get(path)!=reference[key]['sha256']:
            raise ValueError('Bridge product does not bind the selected original '+key)
    for mapping in [product['source_hashes'],build['source_and_object_hashes'],native['input_hashes']]:
        for asset,digest in mapping.items():
            path=Path(asset).resolve()
            if file_sha256(path)!=digest:raise ValueError('Executed bridge dependency changed')
            pins[str(path)]=digest
    highest=max(map(int,original['phase_noise_by_level']));level=highest+1
    parent=original['phase_noise_by_level'][str(highest)]
    command=native['command'];metrics=product['metrics'];output=product['output']
    if len(command)!=7 or command[1]!=str(highest) or Path(command[2]).resolve()!=Path(parent['path']).resolve() or command[3]!=str(seeds[str(level)]) or Path(command[4]).resolve()!=Path(output['path']).resolve():
        raise ValueError('Bridge invocation changes retained parent, allocated seed or output')
    if metrics['parent_level']!=highest or metrics['bridge_level']!=level or metrics['seed']!=seeds[str(level)]:
        raise ValueError('Bridge metrics change selected hierarchy or seed')
    if file_sha256(command[0])!=build['binary_sha256'] or file_sha256(parent['path'])!=parent['sha256']:
        raise ValueError('Bridge build or retained parent binding differs')
    if file_sha256(output['path'])!=output['sha256']:raise ValueError('Bridge output changed')
    pins[str(Path(output['path']).resolve())]=output['sha256']
    return level,output,pins


def stage_production_inputs(request,destination):
    """Prepare retained-parent high-resolution inputs without generating ICs."""
    import json
    from .foundation import atomic_json,file_sha256
    from .zoom import verify_staged_zoom_inputs,stage_zoom_inputs,_configuration,apply_traced_region
    if request.get('schema')!='flagship-production-staging-request-v1':raise ValueError('Explicit production input staging request required')
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous production input staging')
    binding=request['parent_staging'];parent=Path(binding['path']).resolve()
    if file_sha256(parent/'staging.json')!=binding['sha256']:raise ValueError('Retained parent staging changed')
    original=verify_staged_zoom_inputs(parent)
    sources={str(parent/'staging.json'):binding['sha256'],str(Path(__file__).resolve()):file_sha256(__file__)}
    spectrum=Path(original['solver_manifest']['path']).resolve()
    if request.get('extended_linear_bundle'):
        reference=request['extended_linear_bundle'];candidate=Path(reference['path']).resolve()
        if file_sha256(candidate)!=reference['sha256']:raise ValueError('Extended linear bundle changed')
        baseline=json.loads(spectrum.read_text());extended=json.loads(candidate.read_text())
        a,b=baseline['class_parameters']['start'],extended['class_parameters']['start']
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in {'root','P_k_max_h/Mpc'}):
            raise ValueError('Extended transfer coverage changes the frozen physical/numerical cosmology')
        sources[str(candidate)]=reference['sha256'];spectrum=candidate
    cases=request['cases']
    if not cases or len({c['id'] for c in cases})!=len(cases):raise ValueError('Distinct production cases required')
    target.mkdir(parents=True);records=[]
    try:
        for case in cases:
            if not isinstance(case['id'],str) or not re_safe_id(case['id']):raise ValueError('Safe production case identity required')
            ref=case['traced_region'];region=Path(ref['path']).resolve()
            if file_sha256(region)!=ref['sha256']:raise ValueError('Actual traced production target changed')
            traced=json.loads(region.read_text())
            if traced['catalogue_binding']['host_fof_row']!=case['host_row']:raise ValueError('Production case changes traced host selection')
            sources[str(region)]=ref['sha256']
            seeds,ranges=extend_refinement_seeds(original['seeds_by_level'],case['level'])
            generation_base=original['parent_level'];noise_inputs=original['phase_noise_by_level'];generation_seeds=seeds
            if request.get('periodic_bridge'):
                generation_base,bridge,bridge_pins=production_bridge_bindings(request['periodic_bridge'],original,seeds)
                if generation_base>=case['level']:raise ValueError('Production refinement must exceed periodic bridge level')
                sources.update(bridge_pins)
                noise_inputs={str(generation_base):bridge}
                generation_seeds={str(level):seeds[str(level)] for level in range(generation_base,case['level']+1)}
            stage=target/case['id']/'staged'
            stage_zoom_inputs(spectrum,stage,
                box_mpc_h=original['box_mpc_h'],parent_level=generation_base,refinement_level=case['level'],
                seeds_by_level=generation_seeds,padding_cells=request['padding_cells'],poisson_accuracy=request['poisson_accuracy'],
                lpt_order=original['perturbation_order'],lpt_seed_kind=original['lpt_seed_kind'],lpt_seed=original['lpt_seed'],
                phase_noise_by_level=noise_inputs,bounded_source_settings=request.get('bounded_source_settings'),
                source_archive_max_mib=request.get('source_archive_max_mib'))
            staged=verify_staged_zoom_inputs(stage);cfg=_configuration(stage/'zoom_template.conf')
            apply_traced_region(cfg,region)
            configuration=stage.parent/'planner.conf'
            with configuration.open('x') as output:cfg.write(output)
            records.append({'id':case['id'],'host_row':case['host_row'],'parent_level':original['parent_level'],
                'generation_base_level':generation_base,'periodic_bridge':request.get('periodic_bridge'),
                'original_frozen_prefix':original['phase_noise_by_level'],
                'level':case['level'],'configuration':{'path':str(configuration),'sha256':file_sha256(configuration)},
                'staging':{'path':str(stage/'staging.json'),'sha256':file_sha256(stage/'staging.json')},
                'frozen_prefix':staged['phase_noise_by_level'],'seeds_by_level':seeds,'raw_seed_ranges':ranges,
                'native_generation':'NOT_EXECUTED','numerical_handoff':'NOT_ACCEPTED'})
        for p,digest in sources.items():
            if file_sha256(p)!=digest:raise ValueError('Production staging source changed during preparation')
        result={'schema':'flagship-production-input-staging-v1','status':'INPUTS_STAGED',
            'cases':records,'source_hashes':sources,'actual_hardware_allocation':None,'launch_ready':False,
            'scope':'Frozen-prefix plus declared disjoint refinement seeds; no production fields/ICs/evolution generated',
            'remaining_requirements':['Actual hierarchy resource planning on these configurations','Current generator binary/operator provenance',
                'Nested-window/species numerical handoff acceptance','Actual production allocation and convergence'],
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(target/'manifest.json',result);return result
    except Exception as error:
        atomic_json(target/'failure.json',{'status':'FAIL','error':str(error),'completed_cases':records});raise


def re_safe_id(value):
    import re
    return bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',value))


def plan_staged_production(request,destination):
    """Run existing native mesh-only planner for every public staged case."""
    import json
    from .foundation import atomic_json,file_sha256
    from .zoom import verify_staged_zoom_inputs,plan_native_music
    if request.get('schema')!='flagship-production-native-plan-request-v1':
        raise ValueError('Explicit staged production native planning request required')
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous production native planning')
    pins={}
    def bound(reference):
        if set(reference)!={'path','sha256'}:raise ValueError('Exact native planning path/SHA256 binding required')
        path=Path(reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Native production planning input changed')
        pins[str(path)]=reference['sha256'];return path
    source=bound(request['staging']);staging=json.loads(source.read_text())
    if staging.get('schema')!='flagship-production-input-staging-v1' or staging.get('status')!='INPUTS_STAGED':
        raise ValueError('Completed public production input staging required')
    if not staging.get('cases') or len({c['id'] for c in staging['cases']})!=len(staging['cases']):
        raise ValueError('Distinct nonempty staged production cases required')
    binary=bound(request['binary']);build_path=bound(request['build_receipt']);build=json.loads(build_path.read_text())
    if build.get('schema')!='music-current-generator-build-v1' or build.get('status')!='BUILT' or build.get('binary_sha256')!=file_sha256(binary):
        raise ValueError('Current generator build receipt must identify the selected binary')
    for mapping in [staging['source_hashes'],build['source_hashes']]:
        for path,digest in mapping.items():
            if file_sha256(path)!=digest:raise ValueError('Production staging or generator build source changed')
            pins[str(Path(path).resolve())]=digest
    limits=request['planning_limits']
    for key in ('memory_gib','disk_gib'):
        value=limits[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Positive native mesh planning allocation required')
    if type(limits['threads']) is not int or limits['threads']<1:raise ValueError('Explicit native planning thread count required')
    target.mkdir(parents=True);rows=[]
    try:
        for case in staging['cases']:
            if not re_safe_id(case['id']):raise ValueError('Safe staged production case ID required')
            stage_path=bound(case['staging']);verify_staged_zoom_inputs(stage_path.parent)
            configuration=bound(case['configuration']);out=target/case['id']
            try:
                plan_native_music(configuration,binary,out,memory_gib=limits['memory_gib'],
                    disk_gib=limits['disk_gib'],threads=limits['threads'])
            except MemoryError:
                if not (out/'resource-plan.json').is_file():raise
            plan_path=out/'resource-plan.json';plan=json.loads(plan_path.read_text())
            # Resource rejection is retained; it is never translated to a launch pass.
            rows.append({'id':case['id'],'host_row':case['host_row'],'configuration':str(configuration),
                'configuration_sha256':file_sha256(configuration),'binary':{'path':str(binary),'sha256':file_sha256(binary)},
                'native_geometry':plan['native_geometry'],'native_resource_report':{'path':str(plan_path),'sha256':file_sha256(plan_path)},
                'native_invocation':{'path':str(out/'native-invocation.json'),'sha256':file_sha256(out/'native-invocation.json')},
                'allocation_check':plan['status'],'allocation_reason':plan.get('reason'),
                'phase_strategy':'Public staged original-prefix/bridge lineage; native low-mode gate is not full IC/species handoff acceptance',
                'planner_scope':'Current-source rebuilt MUSIC native mesh-only execution; no production fields/IC/evolution generated'})
        for path,digest in pins.items():
            if file_sha256(path)!=digest:raise ValueError('Native planning dependency changed during execution')
        result={'schema':'flagship-production-geometry-audit-v1','status':'GEOMETRY_PLANNED',
            'source_hashes':pins,'rows':rows,'planning_limits':limits,'launch_ready':False,
            'resource_status':'FAIL' if any(row['allocation_check']!='PASS' for row in rows) else 'ESTIMATE_PASS',
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(target/'resource-findings.json',result);return result
    except Exception as error:
        atomic_json(target/'failure.json',{'status':'FAIL','error':str(error),'completed_cases':rows});raise


def prepare_production_evolution_settings(request,destination):
    """Materialize native answer files for requested hardware without launch."""
    import copy
    import json
    from .foundation import atomic_json,file_sha256
    from core.research.capabilities import validate_native_answers
    from commands import redshift_schedule
    if request.get('schema')!='flagship-production-evolution-settings-request-v1':
        raise ValueError('Explicit production evolution setting request required')
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous production evolution settings')
    from . import source_path
    code_paths=[Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'commands.py',
        *[source_path(name).resolve() for name in ('research.capabilities','evolution','foundation','black_holes','native')]]
    pins={str(path):file_sha256(path) for path in code_paths}
    def load(reference):
        if set(reference)!={'path','sha256'}:raise ValueError('Exact production setting source binding required')
        path=Path(reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Production evolution setting source changed')
        pins[str(path)]=reference['sha256'];return json.loads(path.read_text())
    staged=load(request['staging']);sizing=load(request['hardware_requirements'])
    template=load(request['combined_physics_answers']);parent=load(request['parent_answers'])
    if staged.get('schema')!='flagship-production-input-staging-v1' or staged.get('status')!='INPUTS_STAGED':
        raise ValueError('Actual completed production staging required')
    if sizing.get('schema')!='flagship-production-hardware-requirements-v1':raise ValueError('Prepared production hardware requirements required')
    seed_extension=request.get('seed_assembly')
    if seed_extension is not None:
        if seed_extension!={'mode':'density_exclusion_reservoir','donor_aperture':'configured_bh_kernel','maximum_donor_fraction':'configured_bh_source_fraction'}:
            raise ValueError('Explicit supported production seed assembly contract required')
        if not template.get('black_holes',{}).get('enabled') or template['black_holes']['seeding_mode']!='density_exclusion':
            raise ValueError('Reservoir extension requires the retained dense-gas seed prescription')
    choices=request['numerical_choices'];resources=request['execution_controls']
    for key in ('mpi_ranks','omp_threads','build_jobs'):
        if type(resources[key]) is not int or not 1<=resources[key]<=1024:raise ValueError('Explicit bounded native execution control required')
    if resources['mpi_ranks']*resources['omp_threads']>resources['cpu_slots']:raise ValueError('Requested production MPI/thread count exceeds proposed CPU allocation')
    minimum=sizing.get('native_particle_indexing',{}).get('minimum_mpi_ranks')
    if minimum is None or any(row.get('planned_particle_census',{}).get('status')!='EXACT_GEOMETRY_CENSUS' for row in sizing['selected_cases']):
        raise ValueError('Exact planned particle census and native rank-capacity requirement required')
    derived=max(native_particle_index_requirement(row['planned_particle_census']['total_particle_count'])['minimum_mpi_ranks'] for row in sizing['selected_cases'])
    if minimum!=derived:raise ValueError('Stored native rank floor differs from the actual exact particle count')
    if resources['mpi_ranks']<minimum:raise ValueError('Requested native MPI rank count is below necessary particle-index capacity')
    memory=sizing['requested_evolution_memory_gib'];allocation={'mpi_ranks':resources['mpi_ranks'],
        'omp_threads':resources['omp_threads'],'memory_gib':memory}
    schedule=redshift_schedule(str(choices['schedule']),float(load(request['linear_bundle'])['parameters']['z_start']))
    if choices['schedule']!=parent['schedule']:raise ValueError('This production profile must retain the explicit parent output schedule')
    if type(choices['pm_grid']) is not int or choices['pm_grid']<2:raise ValueError('Explicit production PM mesh required')
    if isinstance(choices['softening_parent_level'],bool) or type(choices['softening_parent_level']) is not int:
        raise ValueError('Exact reference softening hierarchy level required')
    if choices['softening_policy']!='fixed_fraction_of_fine_cell':raise ValueError('Explicit recorded production softening policy required')
    proposed_cases={row['id'] for row in staged['cases']}
    planned_targets={row['id'] for row in sizing['selected_cases']}
    # IDs may describe their original parent or periodic generation base. Match
    # actual target and fine level using explicit map, never a name rewrite.
    mapping=request['hardware_case_map']
    if set(mapping)!=proposed_cases or set(mapping.values())!=planned_targets:
        raise ValueError('Every staged target must bind one actual hardware sizing case')
    flow_requests={}
    if request.get('native_flow_window_comparisons') is not None:
        from core.music.flow import original_flow_model_bindings
        if set(request['native_flow_window_comparisons'])!=proposed_cases:raise ValueError('Every production case needs a frozen native window request')
        for case in staged['cases']:
            launch=load(request['native_flow_window_comparisons'][case['id']])
            pins.update(original_flow_model_bindings(launch,levels=range(case.get('generation_base_level',case['parent_level']),case['level']+1)))
            flow_requests[case['id']]=launch
    target.mkdir(parents=True);records=[]
    try:
        for case in staged['cases']:
            plan=next(row for row in sizing['selected_cases'] if row['id']==mapping[case['id']])
            if plan['effective_level']!=case['level']:raise ValueError('Sizing and evolution requested resolution differ')
            answers=copy.deepcopy(template)
            factor=2.**(choices['softening_parent_level']-case['level'])
            softening=template['softening_kpc_h']*factor
            physical=template['softening_max_physical_kpc_h']*factor
            answers.update(name=case['id']+'-production',cosmology=str(Path(request['linear_bundle']['path']).resolve()),
                schedule=choices['schedule'],particles_per_dimension=2**case['level'],pm_grid=choices['pm_grid'],
                softening_kpc_h=softening,softening_max_physical_kpc_h=physical,
                mpi_ranks=resources['mpi_ranks'],omp_threads=resources['omp_threads'],build_jobs=resources['build_jobs'],
                memory_limit_gb=memory)
            if seed_extension is not None:answers['black_holes']['seeding_mode']=seed_extension['mode']
            if flow_requests:answers['native_flow_window_comparison']=copy.deepcopy(flow_requests[case['id']])
            config=validate_native_answers(answers,allocation)
            if config['simulation']['snapshot_redshift_list']!=schedule:raise ValueError('Native output schedule changed during answer preparation')
            path=target/(case['id']+'.answers.json');atomic_json(path,answers)
            records.append({'id':case['id'],'host_row':case['host_row'],'effective_level':case['level'],
                'answers':{'path':str(path),'sha256':file_sha256(path)},'expected_saved_redshifts':schedule,
                'proper_force_resolution_status':'UNVALIDATED_CONVERGENCE_REQUIRED',
                'softening_kpc_h':softening,'softening_max_physical_kpc_h':physical,
                'allocation_status':'REQUESTED_LARGER_HARDWARE_NOT_ALLOCATED','native_execution':'NOT_EXECUTED'})
        for path,digest in pins.items():
            if file_sha256(path)!=digest:raise ValueError('Production setting input changed during preparation')
        result={'schema':'flagship-production-evolution-settings-v1','status':'SETTINGS_PREPARED',
            'cases':records,'source_hashes':pins,'numerical_choices':choices,'requested_resources':allocation,
            'seed_assembly_extension':seed_extension,'original_seed_mode':template.get('black_holes',{}).get('seeding_mode'),
            'native_flow_launch_request_status':'FROZEN' if flow_requests else 'REQUIRES_PHASE_MODEL_AND_LAUNCH_PREPARATION',
            'science_acceptance':'NOT_ACCEPTED','launch_ready':False,'actual_hardware_allocation':None,
            'scope':'Complete answer objects checked by existing native public builder; no build or evolution launched',
            'remaining_requirements':['Actual hardware allocation and native particle census','IC/species/2LPT/window handoff acceptance',
                'Generation/evolution campaign connection','Selected physics accounting and convergence acceptance']}
        atomic_json(target/'manifest.json',result);return result
    except Exception as error:
        atomic_json(target/'failure.json',{'status':'FAIL','error':str(error),'completed_cases':records});raise


def validate_profile_selection(request):
    """Validate arbitrary target/resolution lists before reading large assets."""
    if request.get('schema') != 'flagship-run-profile-v1':
        raise ValueError('Explicit flagship run profile required')
    branch = request.get('branch', request.get('name'))
    if branch not in {'miniature','local-preparation','production-science','sidm','observational','inference'}:
        raise ValueError('Explicit supported flagship profile branch required')
    cases = request.get('zoom_cases')
    if not isinstance(cases,list) or not cases:
        raise ValueError('At least one actual zoom geometry required for resource preflight')
    identities = set()
    for case in cases:
        if not isinstance(case.get('id'),str) or not case['id'] or case['id'] in identities:
            raise ValueError('Unique nonempty zoom case identity required')
        identities.add(case['id'])
        if type(case.get('host_row')) is not int or case['host_row'] < 0:
            raise ValueError('Exact nonnegative selected host row required')
        if type(case.get('level')) is not int or not 1 <= case['level'] <= 20:
            raise ValueError('Exact supported requested refinement level required')
    if branch == 'production-science':
        required = request.get('science_resolution',{}).get('required_effective_level')
        if type(required) is not int or not 1 <= required <= 20:
            raise ValueError('Explicit production effective refinement level required')
        if any(case['level'] < required for case in cases):
            raise ValueError('Production profile selects miniature resolution below its declared requirement')
    return branch


def profile_input_bindings(request):
    """Freeze common assets before census; reject mislabeled physical settings."""
    import json
    from .foundation import file_sha256
    branch=validate_profile_selection(request)
    values = {}; pins = {}
    for key in ('parent_answers','combined_physics_answers','cosmology'):
        reference=request[key]
        if set(reference) != {'path','sha256'}:
            raise ValueError('Exact profile path/SHA256 binding required: '+key)
        path=Path(reference['path']).resolve()
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Frozen profile asset changed: '+key)
        values[key]=json.loads(path.read_text());pins[str(path)]=reference['sha256']
    if request.get('native_flow_window_comparisons') is not None:
        from core.music.flow import original_flow_model_bindings
        mapping=request['native_flow_window_comparisons']
        if not isinstance(mapping,dict) or set(mapping)!={case['id'] for case in request['zoom_cases']}:
            raise ValueError('Every selected zoom case needs exactly one frozen native window request')
        values['native_flow_window_comparisons']={}
        for case in request['zoom_cases']:
            ref=mapping[case['id']];path=Path(ref['path']).resolve()
            if set(ref)!={'path','sha256'} or file_sha256(path)!=ref['sha256']:raise ValueError('Native profile window request changed')
            launch=json.loads(path.read_text());pins[str(path)]=ref['sha256']
            from core.music.flow import validate_original_flow_launch_request
            validate_original_flow_launch_request(launch)
            declared_levels=[level for window in launch['windows'] for level in (window['refinement_levels'] or [])]
            if not declared_levels or max(declared_levels)!=case['level']:
                raise ValueError('Frozen window request differs from the selected actual zoom resolution')
            pins.update(original_flow_model_bindings(launch,levels=range(min(declared_levels),case['level']+1)))
            values['native_flow_window_comparisons'][case['id']]=launch
    parent=values['parent_answers'];hydro=values['combined_physics_answers'];spectrum=values['cosmology']
    cosmos=Path(request['cosmology']['path']).resolve()
    for answers in (parent,hydro):
        if Path(answers['cosmology']).resolve()!=cosmos:
            raise ValueError('Profile answers select a different cosmology bundle')
        if answers['seed'] != request['phases']['parent_seed']:
            raise ValueError('Profile answers change the frozen parent phase seed')
    if type(request['phases']['parent_level']) is not int:
        raise ValueError('Exact parent hierarchy level required')
    expected={
        'gas':hydro['physics']=='hydro',
        'cooling':hydro['physics']=='hydro' and hydro.get('hydro_model') in {'cooling','star_formation'},
        'star_formation':hydro.get('hydro_model')=='star_formation',
        'black_holes':hydro.get('black_holes',{}).get('enabled',False),
    }
    for name,value in expected.items():
        if type(request['physics'].get(name)) is not bool or request['physics'][name] != value:
            raise ValueError('Profile physics label differs from selected native answers: '+name)
    # Explicitly retain selected return model status; this check never changes it.
    if 'delayed_returns' not in request['physics']:
        raise ValueError('Explicit prompt/delayed stellar return selection required')
    if branch=='sidm':
        from .sidm_physics import sidm_native_parameters, sidm_physical_model
        selected=hydro.get('sidm',{})
        if selected.get('enabled') is not True:
            raise ValueError('SIDM profile requires selected enabled native SIDM settings; collisionless answers cannot stand in for this branch')
        sidm_native_parameters(selected,physics=hydro['physics'])
        _,asset=sidm_physical_model(selected)
        pins[str(Path(asset['path']).resolve())]=asset['sha256']
    if branch in {'observational','inference'}:
        _,extra=analysis_campaign_bindings(branch,request.get('branch_campaign'),request['resources'])
        pins.update(extra)
    return values,pins


def analysis_campaign_bindings(branch,reference,allocation):
    """Validate existing analysis campaigns and pin their explicit file inputs."""
    import json
    from .foundation import file_sha256
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError(branch+' profile requires an exact branch_campaign path/SHA256 binding')
    path=Path(reference['path']).resolve()
    if path.stat().st_size>16*1024**2:raise MemoryError('Analysis campaign metadata exceeds profile intake bound')
    if file_sha256(path)!=reference['sha256']:raise ValueError('Analysis branch campaign changed')
    campaign=json.loads(path.read_text())
    if branch=='observational':
        from .cosmography import validate_campaign_request
        validate_campaign_request(campaign)
        if not any(s['kind'] in {'distance_ladder','bao','bao_compressed','time_delay','time_delay_reduced','lens_recovery'} for s in campaign['stages']):
            raise ValueError('Observational branch must select an actual measurement or likelihood stage')
    elif branch=='inference':
        if campaign.get('schema')=='cosmography-campaign-v1':
            from .cosmography import validate_campaign_request
            validate_campaign_request(campaign)
            from .bao import validate_bao_inference_settings
            for stage in campaign['stages']:
                if stage['kind']=='bao_compressed_posterior':validate_bao_inference_settings(stage['parameters'],allocation['memory_gib'])
        else:
            from core.stream.workflow import validate_stream_campaign_request
            validate_stream_campaign_request(campaign)
        if not any(s['kind'].startswith('sbi_') or s['kind'] in {'posterior','bao_compressed_posterior'} for s in campaign['stages']):
            raise ValueError('Inference branch must select actual inference stages')
    else:raise ValueError('Analysis campaign requires observational or inference branch')
    memory=campaign['resources']['memory_gib']
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or not 0<memory<=allocation['memory_gib']:
        raise MemoryError('Analysis campaign exceeds the profile memory allocation')
    pins={str(path):reference['sha256']}
    def bind(value,dependencies):
        if isinstance(value,dict):
            if 'stage' in value:
                if set(value)-{'stage','artifact'} or value['stage'] not in dependencies:
                    raise ValueError('Analysis input stage must be an explicit preceding dependency')
                artifact=Path(value.get('artifact','product.json'))
                if artifact.is_absolute() or '..' in artifact.parts:raise ValueError('Analysis stage input escapes its generation')
                return
            for nested in value.values():bind(nested,dependencies)
        elif isinstance(value,list):
            for nested in value:bind(nested,dependencies)
        elif isinstance(value,str):
            asset=Path(value);asset=asset if asset.is_absolute() else path.parent/asset;asset=asset.resolve()
            if not asset.is_file():raise FileNotFoundError('Required analysis branch input missing: '+str(asset))
            pins[str(asset)]=file_sha256(asset)
        elif value is not None and not isinstance(value,(int,float,bool)):
            raise ValueError('Unsupported analysis input reference')
    for stage in campaign['stages']:bind(stage.get('inputs',{}),stage.get('depends_on',[]))
    return campaign,pins


def validate_branch_allocation(request,values):
    """Apply native SIDM per-rank limits to the actual requested allocation."""
    if validate_profile_selection(request)!='sidm':return
    from .sidm_physics import sidm_native_parameters
    resources=request['resources'];answers=values['combined_physics_answers']
    sidm_native_parameters(answers['sidm'],physics=answers['physics'],mpi_ranks=resources['mpi_ranks'],
        max_mem_rank_mib=resources['memory_gib']*1024/resources['mpi_ranks'])


def validate_case_configuration(case, configuration, request, spectrum):
    """Bind displayed selection and physical mass resolution to the real config."""
    if configuration.getint('setup','levelmax') != case['level']:
        raise ValueError('Zoom case refinement label differs from its actual configuration')
    if configuration.getint('setup','levelmin') != request['phases']['parent_level']:
        raise ValueError('Zoom case changes the selected parent hierarchy level')
    parameters=spectrum['parameters']
    comparisons=[('H0',parameters['H0']),('Omega_b',parameters['Omega_b']),
        ('Omega_m',parameters['Omega_b']+parameters['Omega_cdm']),('A_s',parameters['A_s']),('n_s',parameters['n_s'])]
    for name,expected in comparisons:
        if not math.isclose(configuration.getfloat('cosmology',name),expected,rel_tol=1e-12,abs_tol=0):
            raise ValueError('Zoom case changes frozen cosmology: '+name)
    if not math.isclose(configuration.getfloat('setup','zstart'),parameters['z_start'],rel_tol=1e-12):
        raise ValueError('Zoom case start epoch differs from its spectrum')


def case_parent_bindings(case, generation):
    """Verify actual traced host selection and parent assets, not case labels."""
    import json
    from .foundation import file_sha256
    reference=generation['traced_region'];path=Path(reference['path']).resolve()
    if file_sha256(path)!=reference['sha256']:
        raise ValueError('Zoom traced host selection changed')
    traced=json.loads(path.read_text())
    binding=traced.get('catalogue_binding',{})
    if binding.get('schema')!='darkmattersim-catalogue-target-v1' or binding.get('host_fof_row')!=case['host_row']:
        raise ValueError('Requested host differs from actual traced catalogue target')
    parent=generation['parent_phase_inputs']
    if not parent or parent.get(traced['initial_ic']['path'])!=traced['initial_ic']['sha256']:
        raise ValueError('Traced host and generation do not share the same frozen parent IC')
    pins={str(path):reference['sha256']}
    for asset,digest in parent.items():
        actual=Path(asset).resolve()
        if file_sha256(actual)!=digest:raise ValueError('Frozen parent IC/phase input changed')
        pins[str(actual)]=digest
    return parent,pins


def freeze_profile(request, destination):
    """Publish selected settings and a current resource receipt, without launch."""
    import copy
    import json
    import os
    import tempfile
    import uuid
    from .foundation import atomic_json,file_sha256
    from .flagship_products import preflight_flagship_profile
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous frozen profile settings')
    target.parent.mkdir(parents=True,exist_ok=True)
    # Validation and estimation happen before publishing selected settings.
    with tempfile.TemporaryDirectory(prefix='.profile-freeze-',dir=target.parent) as temp:
        working=Path(temp)
        try:
            budget=preflight_flagship_profile(request,working/'resources.json',output_directory=target.parent)
        except Exception as error:
            failure=target.with_name(target.name+'.failure-'+uuid.uuid4().hex+'.json')
            report=json.loads((working/'resources.json').read_text()) if (working/'resources.json').is_file() else None
            atomic_json(failure,{'schema':'flagship-profile-freeze-failure-v1','status':'FAIL',
                'error_type':type(error).__name__,'error':str(error),'requested_profile':request,
                'resource_preflight':report,'science_acceptance':'NOT_ACCEPTED'})
            error.args=(str(error)+'; retained evidence: '+str(failure),)
            raise
        values,pins=profile_input_bindings(request)
        frozen=copy.deepcopy(request)
        resources=request['resources']
        for key in ('parent_answers','combined_physics_answers'):
            answers=copy.deepcopy(values[key])
            answers.update(mpi_ranks=resources['mpi_ranks'],omp_threads=resources['omp_threads'],
                build_jobs=resources['build_jobs'],memory_limit_gb=resources['memory_gib'])
            path=working/(key+'.json');atomic_json(path,answers)
            frozen[key]={'path':str(target/path.name),'sha256':file_sha256(path)}
        if 'native_flow_window_comparisons' in values:
            from core.research.capabilities import validate_native_answers
            frozen['case_native_answers']={}
            for ordinal,case in enumerate(request['zoom_cases']):
                answers=copy.deepcopy(values['combined_physics_answers'])
                answers.update(particles_per_dimension=2**case['level'],mpi_ranks=resources['mpi_ranks'],omp_threads=resources['omp_threads'],build_jobs=resources['build_jobs'],memory_limit_gb=resources['memory_gib'])
                answers['native_flow_window_comparison']=values['native_flow_window_comparisons'][case['id']]
                validate_native_answers(answers,resources)
                path=working/f'case-{ordinal:04d}.answers.json';atomic_json(path,answers)
                frozen['case_native_answers'][case['id']]={'path':str(target/path.name),'sha256':file_sha256(path)}
        atomic_json(working/'profile.json',frozen)
        branch=validate_profile_selection(request)
        if branch in {'observational','inference'}:
            import sys
            root=Path(__file__).resolve().parents[1]
            campaign,_=analysis_campaign_bindings(branch,request['branch_campaign'],resources)
            atomic_json(working/'branch-recipe.json',{'schema':'flagship-analysis-branch-recipe-v1',
                'argv':[sys.executable,str(root/'commands.py'),'cosmography' if campaign['schema']=='cosmography-campaign-v1' else 'streams',
                    str(Path(request['branch_campaign']['path']).resolve()),'--output',str(target/'campaign')],
                'input_hashes':pins,'status':'PREPARED_PRODUCER_INTAKE_REQUIRED','science_acceptance':'NOT_ACCEPTED'})
        copied={p.name:file_sha256(p) for p in working.iterdir() if p.is_file()}
        manifest={'schema':'flagship-frozen-profile-v1','status':'SETTINGS_FROZEN',
            'branch':validate_profile_selection(request),'profile':'profile.json','files_sha256':copied,
            'original_input_hashes':{**budget['source_hashes'],**pins},
            'execution_policy':budget['execution_policy'],
            'campaign_launch_status':request.get('launch_status','NOT_MATERIALIZED'),
            'science_acceptance':'NOT_ACCEPTED',
            'scope':'Frozen settings and resource estimate; executable campaign and numerical handoff acceptance are separate requirements'}
        atomic_json(working/'manifest.json',manifest)
        for p,digest in manifest['original_input_hashes'].items():
            if file_sha256(p)!=digest:raise ValueError('Profile inputs changed during freeze')
        # Rename only the retained publication folder; the scratch directory is cleaned.
        publish=working/'publication';publish.mkdir()
        for p in list(working.iterdir()):
            if p.is_file():p.rename(publish/p.name)
        if target.exists():raise FileExistsError('Frozen profile destination appeared during preparation')
        os.rename(publish,target)
    return manifest


def prepare_matched_sidm_profiles(request, settings_path, destination):
    """Prepare paired CDM/SIDM settings; the selected operator is explicit."""
    import copy
    import json
    import os
    import tempfile
    from .foundation import atomic_json,canonical_digest,file_sha256
    from .sidm_physics import sidm_native_parameters, sidm_physical_model
    target=Path(destination).resolve();settings_path=Path(settings_path).resolve()
    if target.exists():raise FileExistsError('Keep previous matched SIDM profile preparation')
    settings=json.loads(settings_path.read_text());settings_hash=file_sha256(settings_path)
    reference=settings['physical_model'];model=Path(reference['path'])
    if not model.is_absolute():model=settings_path.parent/model
    settings['physical_model']={'path':str(model.resolve()),'sha256':reference['sha256']}
    baseline=copy.deepcopy(request);baseline['branch']='local-preparation'
    values,pins=profile_input_bindings(baseline)
    if values['combined_physics_answers'].get('sidm',{}).get('enabled',False):
        raise ValueError('Matched CDM baseline must use collisionless native answers')
    sidm_native_parameters(settings,physics=values['combined_physics_answers']['physics'],
        mpi_ranks=request['resources']['mpi_ranks'],
        max_mem_rank_mib=request['resources']['memory_gib']*1024/request['resources']['mpi_ranks'])
    _,asset=sidm_physical_model(settings);pins[str(model.resolve())]=asset['sha256'];pins[str(settings_path)]=settings_hash
    namespace='flagship-sidm:'+canonical_digest({'settings':settings,'cases':request['zoom_cases']})[:24]
    settings['event_identity_namespace']=namespace
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.matched-profile-',dir=target.parent) as temp:
        work=Path(temp);publication=work/'publication';publication.mkdir()
        for label in ('cdm','sidm'):
            answers=copy.deepcopy(values['combined_physics_answers'])
            # SIDM forces global gravity stepping. Freeze the same explicit
            # numerical choice in its collisionless physical control.
            answers['force_equal_timesteps']=True
            answers['native_mergertree']=True
            if label=='sidm':answers['sidm']=settings
            answer_path=publication/(label+'-physics.answers.json');atomic_json(answer_path,answers)
            selected=copy.deepcopy(request);selected['name']=request['name']+'-'+label
            selected['branch']='sidm' if label=='sidm' else 'local-preparation'
            selected['combined_physics_answers']={'path':str(target/answer_path.name),'sha256':file_sha256(answer_path)}
            selected['matched_branch']={'role':label,'paired_role':'cdm' if label=='sidm' else 'sidm',
                'same_parent_phases':True,'same_baryons':True,'physical_model_varies':label=='sidm'}
            atomic_json(publication/(label+'-profile.json'),selected)
        # Validate both profiles against temporary retained assets before publication.
        from .flagship_products import preflight_flagship_profile
        receipts={}
        for label in ('cdm','sidm'):
            selected=json.loads((publication/(label+'-profile.json')).read_text())
            selected['combined_physics_answers']['path']=str(publication/(label+'-physics.answers.json'))
            receipt=preflight_flagship_profile(selected,work/(label+'-resources.json'),output_directory=target.parent)
            receipts[label]={'status':receipt['status'],'case_counts':[r['actual_retained_counts'] for r in receipt['cases']]}
        if receipts['cdm']['case_counts']!=receipts['sidm']['case_counts']:
            raise ValueError('Matched profiles have different actual retained particle census')
        for p,digest in pins.items():
            if file_sha256(p)!=digest:raise ValueError('Matched profile source changed during preparation')
        manifest={'schema':'flagship-matched-sidm-profiles-v1','status':'SETTINGS_PREPARED',
            'files_sha256':{p.name:file_sha256(p) for p in publication.iterdir()},'source_hashes':pins,
            'resource_preflight':receipts,'event_identity_namespace':namespace,
            'matching':'Identical source parent, phases, selected cases, baryon settings and explicit global timestep scheme; only native SIDM interaction settings differ',
            'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED',
            'remaining_requirements':['Current native IC/handoff acceptance','Complete executable evolution/analysis campaign recipes','Native matched SIDM diagnostic and convergence acceptance']}
        atomic_json(publication/'manifest.json',manifest)
        if target.exists():raise FileExistsError('Matched profile destination appeared')
        os.rename(publication,target)
    return manifest


def freeze_matched_profiles(manifest_path, destination):
    """Recheck and freeze both published branches at stable final paths."""
    import copy
    import json
    from .foundation import atomic_json,file_sha256
    source=Path(manifest_path).resolve();base=source.parent;target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous paired profile freeze')
    manifest=json.loads(source.read_text())
    if manifest.get('schema')!='flagship-matched-sidm-profiles-v1' or manifest.get('status')!='SETTINGS_PREPARED':
        raise ValueError('Published prepared matched SIDM settings required')
    bindings={str(source):file_sha256(source),**manifest['source_hashes']}
    for name,digest in manifest['files_sha256'].items():
        path=(base/name).resolve()
        if not path.is_relative_to(base) or file_sha256(path)!=digest:
            raise ValueError('Published matched profile asset changed or escapes bundle')
        bindings[str(path)]=digest
    profiles={label:json.loads((base/(label+'-profile.json')).read_text()) for label in ('cdm','sidm')}
    a,b=profiles['cdm'],profiles['sidm']
    for field in ('resources','zoom_cases','parent_answers','cosmology','phases','physics'):
        if a[field]!=b[field]:raise ValueError('Matched profile changes shared '+field)
    av,ap=profile_input_bindings(a);bv,bp=profile_input_bindings(b)
    cdm=copy.deepcopy(av['combined_physics_answers']);sidm=copy.deepcopy(bv['combined_physics_answers'])
    selected=sidm.pop('sidm',None)
    # Explicit disabled SIDM is equivalent to the collisionless default.
    disabled=cdm.pop('sidm',{'enabled':False})
    if disabled.get('enabled') is not False or selected is None or selected.get('enabled') is not True or cdm!=sidm:
        raise ValueError('Matched physics differs beyond the selected native SIDM operator')
    bindings.update(ap);bindings.update(bp)
    for p,digest in bindings.items():
        if file_sha256(p)!=digest:raise ValueError('Matched branch input changed before freeze')
    target.mkdir(parents=True)
    completed={}
    try:
        for label in ('cdm','sidm'):
            freeze_profile(profiles[label],target/label)
            publication=target/label/'manifest.json'
            completed[label]={'path':str(publication),'sha256':file_sha256(publication)}
        for p,digest in bindings.items():
            if file_sha256(p)!=digest:raise ValueError('Matched input changed during branch freeze')
        result={'schema':'flagship-frozen-matched-profiles-v1','status':'SETTINGS_FROZEN',
            'source_hashes':bindings,'branches':completed,'native_execution':'NOT_EXECUTED',
            'science_acceptance':'NOT_ACCEPTED','campaign_launch_status':'CAMPAIGN_RECIPES_AND_NATIVE_GATES_PENDING'}
        atomic_json(target/'manifest.json',result)
        return result
    except Exception as error:
        atomic_json(target/'failure.json',{'schema':'flagship-matched-freeze-failure-v1','status':'FAIL',
            'error':str(error),'completed_branches':completed,'source_hashes':bindings})
        raise


# Flagship connected profile


def connected_launch_profile(request,*,request_base,overrides=None):
    from core.workflow import safe_id,positive_number,compile_requested_product_dag
    from core.flagship_parent_preparation import validate_parent_preparation
    from core.flagship_zoom_preparation import validate_zoom_preparation
    from core.flagship_runtime_display import validate_scene_options
    base=Path(request_base).resolve();value=copy.deepcopy(request);pins={}
    required={'schema','name','kind','resources','followup_limits','build_engines','class','generator','parent','zoom','history','measurements','display'}
    if set(value)!=required or value['schema']!='flagship-connected-profile-request-v1' or value['kind'] not in {'miniature','local-preparation'}:raise ValueError('Complete connected miniature/local preparation profile required')
    safe_id(value['name']);resources={**value['resources'],**(overrides or {})}
    for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','build_jobs','maximum_stage_count'):
        positive_number(resources[key],key,key in {'mpi_ranks','omp_threads','build_jobs','maximum_stage_count'})
    def bind_assets(node):
        if isinstance(node,list):return [bind_assets(item) for item in node]
        if isinstance(node,dict):
            if set(node)=={'path','sha256'}:
                path=Path(node['path']).expanduser();path=(base/path).resolve() if not path.is_absolute() else path.resolve()
                if file_sha256(path)!=node['sha256']:raise ValueError('Connected profile asset changed: '+str(path))
                if str(path) in pins and pins[str(path)] != node['sha256']:
                    raise ValueError('Connected profile assets declare conflicting source versions')
                pins[str(path)]=node['sha256'];return {'path':str(path),'sha256':node['sha256']}
            return {key:bind_assets(item) for key,item in node.items()}
        return node
    value=bind_assets(value)
    def check_bounds(node):
        if isinstance(node,list):
            for item in node:check_bounds(item)
        elif isinstance(node,dict):
            for key,item in node.items():
                if key in {'memory_gib','audit_memory_gib','disk_gib','copy_disk_gib'}:
                    positive_number(item,key)
                    budget='memory_gib' if key in {'memory_gib','audit_memory_gib'} else 'disk_gib'
                    if item>resources[budget]:raise MemoryError('Connected stage '+key+' exceeds the selected launch allocation')
                check_bounds(item)
    check_bounds({key:item for key,item in value.items() if key not in {'resources','parent','history'}})
    check_bounds(value['parent']['execution'])
    check_bounds({key:item for key,item in value['history'].items() if key!='combined_answers'})
    solver=value['class']
    if set(solver)!={'configuration','binary','wall_seconds'}:raise ValueError('Explicit actual CLASS configuration, binary and wall bound required')
    positive_number(solver['wall_seconds'],'CLASS wall seconds')
    configuration = Path(solver['configuration']['path'])
    if configuration.stat().st_size > min(16*1024**2, resources['memory_gib']*1024**3/8):
        raise MemoryError('Connected CLASS configuration exceeds metadata allocation')
    parameters=json.loads(configuration.read_text())
    from core.linear import validate
    validate(parameters);safe_id(parameters['name'])
    def stage_ref(sid,name):return {'stage':sid,'artifact':name}
    def product(sid,name):return {'path':stage_ref(sid,name),'sha256':{**stage_ref(sid,name),'load':'sha256'}}
    linear=product('cosmology','class/'+parameters['name']+'/spectrum.json');linear_receipt=stage_ref('cosmology','stage-receipt.json')
    display=value['display']
    if set(display)!={'memory_gib','disk_gib','maximum_metadata_files','maximum_figures','scene'}:raise ValueError('Complete bounded native display settings required')
    validate_scene_options(display['scene'])
    display_stage={'stage_id':'prepare_display','kind':'flagship_runtime_display','parameters':{
        'schema':'flagship-runtime-display-request-v1','prefix':'display','cases':{'runtime_measurement_cases':True},**display},
        'outputs':{'display_preparation':'display-preparation.json'}}
    measurements=value['measurements']
    if set(measurements)!={'memory_gib','maximum_metadata_files','options'}:raise ValueError('Complete native measurement selections required')
    measurement_stage={'stage_id':'prepare_measurements','kind':'flagship_runtime_measurements','parameters':{
        'schema':'flagship-runtime-measurements-request-v1','prefix':'measure','cases':{'runtime_history_cases':True},
        'linear_bundle':linear,'linear_receipt':linear_receipt,**measurements,'continuation':display_stage},
        'outputs':{'measurement_preparation':'measurement-preparation.json'}}
    history=value['history']
    if set(history)!={'combined_answers','numerical_choices','flow_preparation','execution','case_staging','template'}:raise ValueError('Complete native history and flow preparation settings required')
    def answers(values):
        result=copy.deepcopy(values)
        for key,resource in [('mpi_ranks','mpi_ranks'),('omp_threads','omp_threads'),('build_jobs','build_jobs'),('memory_limit_gb','memory_gib')]:result[key]=resources[resource]
        return result
    history_stage={'stage_id':'prepare_history','kind':'flagship_runtime_history','parameters':{
        'schema':'flagship-runtime-history-request-v1','kind':value['kind'],'zoom_preparation':{'zoom_preparation_reference':True},
        'zoom_preparation_receipt':{'zoom_preparation_receipt':True},'cases':{'zoom_cases':True},
        'parent_answers':{'parent_output':'answers_reference'},'combined_answers':answers(history['combined_answers']),
        **{key:history[key] for key in ('numerical_choices','flow_preparation','execution','case_staging')},
        'generator':value['generator'],'history_template':history['template'],'continuation':measurement_stage},
        'outputs':{'history_preparation':'runtime-history-preparation.json','history_manifest':'history/manifest.json',
            'native_campaign':'derived-native-campaign.json'}}
    zoom=value['zoom']
    if set(zoom)!={'prefix','snapshot','selection_rules','explicit_host_rows','staging','execution'}:raise ValueError('Complete configurable target/refinement selection required')
    zoom_stage={'stage_id':'prepare_zooms','kind':'flagship_zoom_preparation','parameters':{
        'schema':'flagship-zoom-preparation-v1',**zoom,'native_run_directory':{'parent_output':'run'},
        'native_stage_receipt':{'parent_output':'run_receipt'},'generation_report':{'parent_output':'generation_report'},
        'generation_stage_receipt':{'parent_output':'generation_receipt'},'parent_noise':{'parent_output':'noise'},
        'generator':value['generator'],'continuation':history_stage},'outputs':{'zoom_preparation':'zoom-preparation.json'}}
    validate_zoom_preparation(zoom_stage['parameters'])
    parent=value['parent']
    if set(parent)!={'prefix','answers','staging','execution'}:raise ValueError('Complete native parent settings required')
    low=parent['staging']['parent_level'];frozen=zoom['staging'].get('frozen_phase_prefix',{})
    if str(low) not in frozen or parent['staging'].get('frozen_parent_noise')!=frozen[str(low)]:raise ValueError('Connected parent and zooms require the same declared established frozen parent phase input')
    if history['case_staging']!={key:zoom['staging'][key] for key in ('source_archive_max_mib','bounded_source_settings')}:raise ValueError('Connected history cannot change actual zoom source-retention/correction settings')
    if history['execution']['generation_wall_hours']!=zoom['execution']['generation_wall_hours'] or history['execution']['audit_memory_gib']!=zoom['execution']['audit_memory_gib']:raise ValueError('Completed zoom reuse requires identical generator execution controls')
    parent_stage={'stage_id':'prepare_parent','kind':'flagship_parent_preparation','parameters':{
        'schema':'flagship-parent-preparation-v1','prefix':parent['prefix'],'linear_bundle':linear,'linear_stage_receipt':linear_receipt,
        'answers':answers(parent['answers']),'staging':parent['staging'],'execution':parent['execution'],
        'generator':value['generator'],'continuation':zoom_stage},
        'outputs':{'parent_preparation':'parent-preparation.json','parent_answers':'answers.json'}}
    validate_parent_preparation(parent_stage['parameters'])
    allocation={key:{'launch_resource':key} for key in resources}
    common={'schema':'research-workflow-v1','physics_spec':{},'capabilities':[],'resources':allocation,
        'followup_limits':copy.deepcopy(value['followup_limits'])}
    initial={'stage_id':'cosmology','kind':'public_command','parameters':{'command':'class','wall_seconds':solver['wall_seconds'],
        'arguments':['--config',{'input':solver['configuration']['path']},'--binary',{'input':solver['binary']['path']},'--output','{generation}/class']},
        'outputs':{'cosmology':'class/'+parameters['name']}}
    native={**copy.deepcopy(common),'stages':[initial,parent_stage],'requested_products':['parent_preparation']}
    workflows={'test':copy.deepcopy(native),'run':copy.deepcopy(native)}
    for operation in ('postprocess','render'):
        row={'stage_id':'retained_'+operation,'kind':'flagship_retained_operation','parameters':{'operation':operation,
            'source_workflow':{'launch_source_workflow':True},'memory_gib':resources['memory_gib'],'maximum_stages':resources['maximum_stage_count']},
            'outputs':{'retained_'+operation:'retained-operation.json'}}
        workflows[operation]={**copy.deepcopy(common),'stages':[row],'requested_products':['retained_'+operation]}
    for workflow in workflows.values():
        # Compilation checks initial producer contracts. Native followups retain
        # their original runtime preflights after actual upstream files exist.
        compile_requested_product_dag({**workflow,'resources':resources},request_base=base)
        for row in workflow['stages']:row['input_files']=[{'path':name,'sha256':sha} for name,sha in pins.items()]
    return {'schema':'flagship-launch-profile-v1','name':value['name'],'kind':value['kind'],'resources':resources,
        'build_engines':value['build_engines'],'workflows':workflows,'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()]}
