"""Prepare and inspect representative native pilot campaigns.

Pilot recipes use the ordinary native evolution and scientific producers.
Population counts come from saved native IDs; coverage and artifact reports
retain their measured conditions. Preparation does not execute the campaign
or establish numerical convergence or physical accuracy.
"""
from __future__ import annotations

import json
import math
import uuid
import zipfile
from copy import deepcopy
from core.foundation import atomic_json, file_sha256, canonical_digest
from core.native import native_run_lock
from core.foundation import merge_source_bindings as _merge
from core.research.errors import _retained_json
from core.workflow import execute_workflow, verify_completed_receipt, path_at
from pathlib import Path, PurePosixPath

# Pilot campaign

FAMILIES = {'collisionless_parent_zoom_stream', 'hydro_stellar_bh', 'sidm_parent_zoom', 'observer_recovery'}
REQUIRED_COVERAGE = {
    'collisionless_parent_zoom_stream': {'parent_evolution', 'zoom_evolution', 'live_stream', 'contamination', 'force_convergence'},
    'hydro_stellar_bh': {'native_galaxy', 'stellar_birth', 'stellar_mass_return', 'black_hole_evolution'},
    'sidm_parent_zoom': {'parent_evolution', 'zoom_evolution', 'sidm_scattering', 'sidm_region_support'},
    'observer_recovery': {'local_observer', 'distance_ladder', 'time_delay', 'bao', 'cosmic_chronometer', 'blinded_recovery'}}


def verify_pilot_coverage(reference, family, *, expected_workflow_hash):
    path = Path(reference['path']).resolve()
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Pilot coverage receipt changed')
    proof = json.loads(path.read_text())
    if proof.get('schema') != 'native-pilot-coverage-v1' or proof.get('family') != family:
        raise ValueError('Pilot coverage schema/family mismatch')
    if proof.get('status') != 'PASS' or proof.get('execution_status') != 'COMPLETE':
        raise ValueError('Actual executed passing pilot coverage required')
    if proof.get('workflow_hash') != expected_workflow_hash:
        raise ValueError('Coverage belongs to another pilot workflow')
    if proof.get('synthetic_fixture') is not False:
        raise ValueError('Representative pilots require evolved native populations')
    rows = proof['coverage']
    covered = set()
    for row in rows:
        label = row['capability']
        if label in covered or label not in REQUIRED_COVERAGE[family]:
            raise ValueError('Pilot coverage must contain each exact required physical capability once')
        if row.get('status') != 'PASS' or not row.get('measurement_hashes'):
            raise ValueError('Pilot capability lacks retained measured evidence')
        if row.get('origin') not in {'native_evolved_population', 'dedicated_physical_control', 'blinded_observational_control'}:
            raise ValueError('Manufactured analysis inputs cannot establish representative pilot coverage')
        for source, expected in row['measurement_hashes'].items():
            p = Path(source)
            if not p.is_absolute():
                p = path.parent/p
            if file_sha256(p) != expected:
                raise ValueError('Pilot capability measurement changed')
        covered.add(label)
    missing = REQUIRED_COVERAGE[family]-covered
    if missing:
        raise ValueError('Pilot family missing physical coverage: '+str(sorted(missing)))
    # Population availability is measured, never inferred from configured switches.
    population = proof.get('native_population_counts', {})
    if family == 'hydro_stellar_bh' and any(type(population.get(q)) is not int or population[q] <= 0 for q in ('stars', 'black_holes')):
        raise ValueError('Hydro pilot must contain actual evolved stars and black holes')
    if family == 'collisionless_parent_zoom_stream' and proof.get('live_stripped_particle_count', 0) <= 0:
        raise ValueError('Stream pilot needs an actually evolved stripped stream')
    if family != 'observer_recovery':
        history = verify_population_evidence(proof,family,receipt_base=path.parent)
        if family == 'collisionless_parent_zoom_stream':
            verify_live_stream_evidence(proof,history,receipt_base=path.parent)
    return proof


def _pilot_execution_signature(request):
    # Evidence arrives after evolution and can be refreshed on resume. The
    # physics request and its actual implementation must stay unchanged.
    execution = {key:value for key,value in request.items() if key != 'pilots'}
    execution['pilots'] = [{key:value for key,value in pilot.items()
        if key not in {'coverage_receipt','visual_inspection_receipt'}} for pilot in request['pilots']]
    source = Path(__file__).resolve()
    return canonical_digest({'execution_request':execution,
        'sources':{str(source):file_sha256(source)}})


def run_pilot_family(request, destination, *, request_base, resume=False, dry_run=False):
    if request.get('schema') != 'representative-pilot-family-v1':
        raise ValueError('Expected representative-pilot-family-v1')
    pilots = request['pilots']
    names = [q['family'] for q in pilots]
    if set(names) != FAMILIES or len(names) != len(FAMILIES):
        raise ValueError('All four representative pilot families are mandatory')
    base = Path(request_base).resolve()
    prepared = []
    for pilot in pilots:
        workflow_path = path_at(base, pilot['workflow'])
        if file_sha256(workflow_path) != pilot['workflow_sha256']:
            raise ValueError('Pilot workflow specification changed')
        specification = json.loads(workflow_path.read_text())
        if pilot.get('coverage_request'):
            coverage_path=path_at(base,pilot['coverage_request']['path'])
            if coverage_path.stat().st_size>16*1024**2 or file_sha256(coverage_path)!=pilot['coverage_request']['sha256']:
                raise ValueError('Frozen pilot coverage request changed or exceeds intake bound')
            coverage_request=json.loads(coverage_path.read_text())
            if coverage_request.get('family')!=pilot['family']:raise ValueError('Coverage request names another physical pilot family')
            validate_pilot_coverage_request(coverage_request,specification)
        if pilot.get('inspection_request'):
            inspection_path=path_at(base,pilot['inspection_request']['path'])
            if inspection_path.stat().st_size>16*1024**2 or file_sha256(inspection_path)!=pilot['inspection_request']['sha256']:
                raise ValueError('Frozen pilot science/artifact inspection request changed or exceeds intake bound')
            inspection_request=json.loads(inspection_path.read_text())
            if inspection_request['memory_gib']>specification['resources']['memory_gib']:
                raise MemoryError('Pilot artifact inspection exceeds frozen workflow memory')
            validate_pilot_inspection_request(inspection_request,specification)
        plan = execute_workflow(specification, Path(destination)/pilot['family'],
            request_base=workflow_path.parent, dry_run=True)
        prepared.append((pilot, workflow_path, specification, plan))
    if dry_run:
        return {'schema': 'pilot-family-plan-v1', 'pilots': [{'family': p['family'], 'plan': plan} for p, _, _, plan in prepared]}
    destination = Path(destination).resolve()
    with native_run_lock(destination, operation='representative physical pilot family'):
        destination.mkdir(parents=True, exist_ok=True)
        signature = _pilot_execution_signature(request)
        state_path = destination/'pilots.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if not resume:
                raise FileExistsError('Existing pilot family requires --resume')
            if previous.get('request_signature') != signature:
                raise ValueError('Pilot inputs or intake code changed; choose a new destination')
            atomic_json(destination/'history'/(uuid.uuid4().hex+'.json'),previous)
        rows = []
        state = {'schema':'pilot-family-result-v1','request_signature':signature,'pilots':rows,
            'execution_status':'RUNNING','active_family':None,'status':'UNRESOLVED',
            'science_acceptance':'PILOT_CONTROLS_ONLY'}
        atomic_json(state_path,state)
        try:
            for pilot,path,specification,_ in prepared:
                family = pilot['family']
                state['active_family'] = family
                atomic_json(state_path,state)
                workflow = execute_workflow(specification,destination/family,request_base=path.parent,resume=resume)
                if workflow['execution_status'] != 'COMPLETE':
                    rows.append({'family':family,'workflow':workflow,'status':'UNRESOLVED',
                        'coverage':None,'reason':'Native pilot workflow has unfinished required execution'})
                    atomic_json(state_path,state)
                    raise ValueError('Pilot workflow has unfinished required execution: '+family)
                status,evidence = 'UNRESOLVED',None
                reason = 'Retained pilot coverage and visual inspection receipts absent'
                visual_evidence = None
                visual_ref = None
                try:
                    if pilot.get('coverage_request'):
                        coverage_request=json.loads(path_at(base,pilot['coverage_request']['path']).read_text())
                        if file_sha256(path_at(base,pilot['coverage_request']['path']))!=pilot['coverage_request']['sha256']:
                            raise ValueError('Frozen pilot coverage production request changed')
                        ref=publish_native_pilot_coverage(coverage_request,workflow,
                            destination/family/'coverage-generations'/uuid.uuid4().hex,request_base=base,
                            workflow_specification=specification)
                    else:
                        ref = dict(pilot['coverage_receipt'])
                    ref['path'] = str(path_at(base,ref['path']))
                    if pilot.get('inspection_request'):
                        inspection_path=path_at(base,pilot['inspection_request']['path'])
                        if file_sha256(inspection_path)!=pilot['inspection_request']['sha256']:
                            raise ValueError('Frozen pilot science/artifact inspection request changed')
                        visual_ref=publish_pilot_inspection(json.loads(inspection_path.read_text()),workflow,
                            destination/family/'inspection-generations'/uuid.uuid4().hex,
                            specification=specification,request_base=base)
                    else:
                        visual_ref = pilot['visual_inspection_receipt']
                    evidence = verify_pilot_coverage(ref,family,expected_workflow_hash=canonical_digest(workflow))
                    visual_path = path_at(base,visual_ref['path'])
                    if file_sha256(visual_path) != visual_ref['sha256']:
                        raise ValueError('Pilot visual inspection changed')
                    visual = json.loads(visual_path.read_text())
                    visual_evidence = verify_inspected_artifacts(visual,workflow_hash=canonical_digest(workflow),base=visual_path.parent)
                    status,reason = 'PASS',None
                except (OSError,KeyError,ValueError) as error:
                    reason = str(error)
                rows.append({'family':family,'workflow':workflow,'status':status,'coverage':evidence,
                    'visual_evidence':visual_evidence,'visual_inspection_receipt':visual_ref,'reason':reason})
                atomic_json(state_path,state)
            state.update(execution_status='COMPLETE',active_family=None,
                status='PASS' if len(rows) == len(FAMILIES) and all(q['status'] == 'PASS' for q in rows) else 'UNRESOLVED')
            atomic_json(state_path,state)
            return state
        except BaseException as error:
            state.update(execution_status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
                failure={'type':type(error).__name__,'message':str(error)})
            atomic_json(state_path,state)
            raise


def pilots_main():
    import argparse
    parser = argparse.ArgumentParser(description='Execute the four representative physical pilot families')
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = run_pilot_family(json.loads(args.campaign.read_text()), args.output,
                             request_base=args.campaign.parent, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if args.dry_run or result['status'] == 'PASS' else 1


# Pilot coverage


def validate_pilot_coverage_request(request,workflow_specification):
    """Check frozen coverage routes before evolving the pilot, without running it."""
    if request.get('schema')!='native-pilot-coverage-request-v1' or request.get('family') not in REQUIRED_COVERAGE:
        raise ValueError('Frozen native pilot coverage request required')
    bound=request['maximum_capabilities']
    if type(bound) is not int or bound<1:raise ValueError('Explicit positive pilot capability count bound required')
    rows=request['capabilities']
    if not isinstance(rows,list) or len(rows)>bound or len(rows)!=len(REQUIRED_COVERAGE[request['family']]) or {row['capability'] for row in rows}!=REQUIRED_COVERAGE[request['family']]:
        raise ValueError('Every required pilot capability needs one actual product contract')
    stages={row['stage_id']:row for row in workflow_specification['stages']}
    resources=request['resources']
    if isinstance(resources['memory_gib'],bool) or not isinstance(resources['memory_gib'],(int,float)) or not math.isfinite(resources['memory_gib']) or resources['memory_gib']<=0:
        raise ValueError('Pilot coverage must retain positive finite intake memory before native evolution')
    if type(resources['maximum_snapshots']) is not int or resources['maximum_snapshots']<2:
        raise ValueError('Pilot coverage requires an explicit saved-history count bound')
    producers={product:row for row in workflow_specification['stages'] for product in row['outputs']}
    for row in rows:
        reference=row['measurement']
        if set(reference)=={'stage','product'}:
            if reference['stage'] not in stages or producers.get(reference['product'],{}).get('stage_id')!=reference['stage']:
                raise ValueError('Pilot capability refers to an absent or different actual product producer')
        elif set(reference)!={'receipt','product'}:raise ValueError('Pilot coverage needs actual product references')
        if row['capability'] == 'native_galaxy' and row.get('product_schema') != 'native-pilot-galaxy-measurement-v1':
            raise ValueError('Representative hydro pilot requires actual bound native galaxy measurement, not only a stellar census')
        ancestry = row.get('native_stage_ids')
        if ancestry is not None and (not isinstance(ancestry,list) or not ancestry or len(set(ancestry)) != len(ancestry)
                or not set(ancestry) <= set(request.get('native_evolution_stages',[]))):
            raise ValueError('Pilot capability native ancestry must name distinct declared actual evolution stages')
        conditions=row['acceptance_conditions']
        if not isinstance(conditions,list) or not conditions or not any(condition['operator'] in {'minimum','maximum','all_true','nonempty'} for condition in conditions):
            raise ValueError('Pilot capability requires measured support/quantity conditions, not only a status label')
        for condition in conditions:
            if not isinstance(condition.get('path'),list) or not condition['path'] or condition['operator'] not in {'exact','minimum','maximum','all_true','nonempty'}:
                raise ValueError('Malformed frozen pilot acceptance condition')
    for identity in request.get('native_evolution_stages',[]):
        row=stages[identity]
        if not (row['kind']=='public_command' and row['parameters'].get('command')=='gadget' or row['kind']=='isolated_physical_collision'):
            raise ValueError('Pilot population ancestry requires actual native evolution producers')
    if request['family']!='observer_recovery':
        if not request.get('native_evolution_stages'):raise ValueError('Physical representative pilot requires native evolution ancestry')
        sets=request['native_snapshot_sets']
        if not 2<=len(sets)<=request['resources']['maximum_snapshots']:
            raise ValueError('Pilot coverage needs a bounded native saved history')
        for row in sets:
            if type(row.get('snapshot')) is not int or row['snapshot']<0:
                raise ValueError('Pilot coverage snapshot selectors must retain exact nonnegative native indices')
            if 'native_stage' in row:
                if set(row)!={'native_stage','run_product','snapshot'} or row['native_stage'] not in request['native_evolution_stages'] or row['run_product'] not in stages[row['native_stage']]['outputs']:
                    raise ValueError('Pilot snapshot selector must name a declared native run directory producer')


def _at(record,path):
    if not isinstance(path,list) or not path:raise ValueError('Pilot acceptance condition requires an explicit product field path')
    value=record
    for key in path:
        if isinstance(value,dict) and isinstance(key,str):value=value[key]
        elif isinstance(value,list) and type(key) is int:value=value[key]
        else:raise ValueError('Pilot acceptance path does not identify a retained measured field')
    return value


def _condition(record,condition):
    value=_at(record,condition['path']);operator=condition['operator']
    if operator=='nonempty':return isinstance(value,(list,dict,str)) and bool(value)
    if operator=='exact':return type(value) is type(condition['value']) and value==condition['value']
    if operator=='all_true':return isinstance(value,list) and bool(value) and all(type(row) is bool and row for row in value)
    if operator in {'minimum','maximum'}:
        limit=condition['value']
        if any(isinstance(row,bool) or not isinstance(row,(int,float)) or not math.isfinite(row) for row in (value,limit)):
            raise ValueError('Pilot numerical acceptance condition needs finite actual values and frozen limits')
        return value>=limit if operator=='minimum' else value<=limit
    raise ValueError('Unsupported explicit pilot acceptance condition')


def publish_native_pilot_coverage(request,workflow,destination,*,request_base,workflow_specification):
    from core.workflow import path_at,verify_completed_receipt
    from core.stream_common import read_product
    import numpy as np
    if request.get('schema')!='native-pilot-coverage-request-v1' or request['family'] not in REQUIRED_COVERAGE:
        raise ValueError('Frozen representative native pilot coverage request required')
    if workflow.get('execution_status')!='COMPLETE':raise ValueError('Pilot coverage cannot be produced from unfinished workflow execution')
    if workflow.get('specification_hash')!=canonical_digest(workflow_specification):
        raise ValueError('Pilot coverage workflow differs from the frozen executed specification')
    validate_pilot_coverage_request(request,workflow_specification)
    declared_stages={row['stage_id']:row for row in workflow_specification['stages']}
    resources=request['resources'];memory=resources['memory_gib']*1024**3
    if isinstance(resources['memory_gib'],bool) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Pilot coverage requires explicit positive memory')
    contracts=request['capabilities']
    if not isinstance(contracts,list) or len(contracts)>request['maximum_capabilities'] or {row['capability'] for row in contracts}!=REQUIRED_COVERAGE[request['family']] or len(contracts)!=len(REQUIRED_COVERAGE[request['family']]):
        raise ValueError('Every mandatory physical pilot capability requires exactly one frozen product contract')
    def resolve(reference):
        if set(reference)=={'stage','product'}:
            stage=workflow['stages'][reference['stage']]
            return {'receipt':str(Path(stage['generation'])/'stage-receipt.json'),'product':reference['product']}
        if set(reference)=={'receipt','product'}:return reference
        raise ValueError('Pilot coverage products require their exact executed stage and declared product')
    native_sources={};native_receipts=[];native_by_stage={}
    for reference in request.get('native_evolution_stages',[]):
        stage=workflow['stages'][reference];path=Path(stage['generation'])/'stage-receipt.json'
        receipt=verify_completed_receipt(path)
        # A completed analysis stage is not a substitute for native evolution.
        declared=declared_stages[reference]
        if receipt['kind']!=declared['kind'] or not (declared['kind']=='public_command' and declared['parameters'].get('command')=='gadget' or declared['kind']=='isolated_physical_collision'):
            raise ValueError('Pilot native ancestry must refer to actual public GADGET evolution')
        if declared['kind']=='isolated_physical_collision':
            result_path=Path(receipt['generation'])/'collision-evolution.json'
            if receipt['files_sha256'].get('collision-evolution.json')!=file_sha256(result_path):
                raise ValueError('Dedicated physical control lacks its actual sealed native evolution record')
            result=json.loads(result_path.read_text())
            if result.get('schema')!='isolated-physical-collision-evolution-v1' or result.get('execution_status')!='COMPLETE' or not result.get('snapshots'):
                raise ValueError('Prepared analytic ICs cannot establish actually evolved dedicated physical coverage')
        native_receipts.append(str(path.resolve()))
        bound = {str(path.resolve()):file_sha256(path),
            **{str(Path(receipt['generation'])/name):digest for name,digest in receipt['files_sha256'].items()}}
        native_by_stage[reference] = bound
        _merge(native_sources,bound)
    covered=[];sources=dict(native_sources);failures=[]
    for contract in contracts:
        reference=resolve(contract['measurement'])
        product,bindings=_retained_json(reference,request_base,int(memory))
        _merge(sources,bindings)
        origin=contract['origin']
        if origin not in {'native_evolved_population','dedicated_physical_control','blinded_observational_control'}:
            raise ValueError('Pilot capability has no permitted physical evidence origin')
        if origin!='blinded_observational_control' and (not native_sources or not any(
                product.get('source_hashes',{}).get(path)==digest for path,digest in native_sources.items())):
            raise ValueError('Pilot physical capability has no retained native evolution ancestry')
        if product.get('synthetic_fixture') is True:raise ValueError('Synthetic fixtures cannot establish representative pilot coverage')
        if product.get('schema')!=contract['product_schema']:raise ValueError('Pilot coverage product differs from its frozen producer contract')
        for native_stage in contract.get('native_stage_ids',[]):
            if not any(product['source_hashes'].get(path) == digest for path,digest in native_by_stage[native_stage].items()):
                raise ValueError('Pilot capability lacks its specific frozen actual native evolution ancestry: '+native_stage)
        conditions=contract['acceptance_conditions']
        if not isinstance(conditions,list) or not conditions:raise ValueError('Pilot capability needs concrete measured acceptance conditions')
        decisions=[{'condition':row,'passed':bool(_condition(product,row))} for row in conditions]
        passed=all(row['passed'] for row in decisions)
        if contract['capability'] == 'native_galaxy':
            count = product.get('resolved_galaxy_count')
            if type(count) is not int or count <= 0:
                passed = False
                decisions.append({'condition':{'path':['resolved_galaxy_count'],'operator':'minimum','value':1},'passed':False})
        covered.append({'capability':contract['capability'],'origin':origin,'status':'PASS' if passed else 'FAIL',
            'measurement_hashes':bindings,'product_schema':product['schema'],'acceptance_conditions':decisions})
        if not passed:failures.append({'capability':contract['capability'],'reason':'frozen measured product conditions failed'})
    proof={'schema':'native-pilot-coverage-v1','family':request['family'],'workflow_hash':canonical_digest(workflow),
        'coverage':covered,'resources':resources,'synthetic_fixture':False,'execution_status':'COMPLETE',
        'request_hash':canonical_digest(request),'source_hashes':sources,'native_evolution_receipts':native_receipts,
        'status':'PASS' if not failures else 'FAIL','failures':failures,'science_acceptance':'PILOT_CONTROLS_ONLY'}
    if request['family']!='observer_recovery':
        sets=[]
        for row in request['native_snapshot_sets']:
            files=[]
            references=row.get('files')
            if 'native_stage' in row:
                from core.native import original_native_snapshot_files
                stage=verify_completed_receipt(Path(workflow['stages'][row['native_stage']]['generation'])/'stage-receipt.json')
                product=stage['products'][row['run_product']]
                if product['kind']!='directory':raise ValueError('Pilot snapshot source is not an actual native run directory')
                root=Path(stage['generation'])/product['path']
                references=[{'path':str(path.resolve()),'sha256':file_sha256(path)} for path in original_native_snapshot_files(root,row['snapshot'])]
            if not references:raise ValueError('Pilot native snapshot has no actual delivered shard files')
            for reference in references:
                path=path_at(request_base,reference['path'])
                if native_sources.get(str(path))!=reference['sha256']:
                    raise ValueError('Pilot population snapshot is not an actual retained native evolution payload')
                files.append({'path':str(path),'sha256':reference['sha256']})
            sets.append({'snapshot':row['snapshot'],'files':files})
        history=native_population_history(sets,memory_gib=resources['memory_gib'],maximum_snapshots=resources['maximum_snapshots'],base=request_base)
        proof.update(native_snapshot_sets=sets,native_population_counts=history['latest_counts'])
        _merge(sources,history['source_hashes'])
        required=('gas','stars','black_holes') if request['family']=='hydro_stellar_bh' else ('dm',)
        if any(history['latest_counts'][kind]<=0 for kind in required):
            proof['status']='FAIL';failures.append({'reason':'actual evolved native population missing','required_species':list(required)})
        if request['family']=='collisionless_parent_zoom_stream':
            ref=resolve(request['live_stream_measurement'])
            receipt=verify_completed_receipt(path_at(request_base,ref['receipt']))
            path=Path(receipt['generation'])/receipt['products'][ref['product']]['path']
            report,arrays,bindings=read_product(path,memory_gib=resources['memory_gib'])
            bound=np.asarray(arrays['bound']);before=np.asarray(arrays['release_before_myr'],float);after=np.asarray(arrays['release_after_myr'],float)
            if bound.dtype.kind!='b' or bound.ndim!=2:raise ValueError('Live pilot requires actual bound-state history')
            count=int(((~bound[-1])&np.isfinite(before)&np.isfinite(after)&(before<after)).sum())
            proof.update(live_stream_product={'path':str(path),'sha256':file_sha256(path)},live_stripped_particle_count=count)
            _merge(sources,bindings)
            if count:
                verify_live_stream_evidence(proof,history,receipt_base=request_base)
            else:
                proof['status']='FAIL';failures.append({'reason':'no actually evolved stripped live stream particles'})
    source = Path(__file__).resolve()
    sources[str(source)] = file_sha256(source)
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Pilot native coverage evidence changed before publication')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Pilot coverage generations are immutable')
    destination.mkdir(parents=True)
    output=destination/'pilot-coverage.json';atomic_json(output,proof)
    return {'path':str(output),'sha256':file_sha256(output),'status':proof['status']}


# Pilot evidence

SPECIES = {'gas': 0, 'dm': 1, 'coarse_dm': 2, 'stars': 4, 'black_holes': 5}


def validate_native_galaxy_measurement(parameters):
    """Bound actual SUBFIND intake before evaluating a native galaxy pilot."""
    request = parameters['request']
    if request.get('schema') != 'native-pilot-galaxy-measurement-request-v1':
        raise ValueError('Native measured galaxy pilot request required')
    for name in ('minimum_star_particles','minimum_dm_particles','maximum_galaxies'):
        if type(request[name]) is not int or request[name] < 1:
            raise ValueError('Positive exact native galaxy support bound required: '+name)
    if type(request['snapshot']) is not int or request['snapshot'] < 0:
        raise ValueError('Native galaxy pilot requires its exact saved snapshot index')
    memory = request['memory_gib']
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Explicit positive native galaxy measurement memory required')
    return request


def galaxy_measurement_input_bindings(parameters):
    from core.halos import _group_files,_snapshot_files
    from core.data import native_publication_markers
    from core.workflow import verify_completed_receipt
    import h5py
    import numpy as np
    request = validate_native_galaxy_measurement(parameters)
    root = Path(parameters['native_run_directory']).resolve()
    receipt_path = Path(parameters['native_stage_receipt']).resolve()
    receipt = verify_completed_receipt(receipt_path)
    owner = Path(receipt['generation']).resolve()
    if receipt['kind'] != 'public_command' or not root.is_relative_to(owner):
        raise ValueError('Galaxy pilot requires an actual completed owned native evolution directory')
    files = [*_snapshot_files(root,request['snapshot']),*_group_files(root,request['snapshot'])]
    if not files:
        raise ValueError('Native galaxy pilot requires evolved snapshots and actual bound SUBFIND membership')
    raw_bytes = 0
    count = 0
    for path in files:
        name = path.resolve().relative_to(owner).as_posix()
        if receipt['files_sha256'].get(name) != file_sha256(path):
            raise ValueError('Galaxy pilot snapshot/catalogue is outside the actual sealed native evolution')
        with h5py.File(path,'r') as source:
            def measure(name,obj):
                nonlocal raw_bytes
                if isinstance(obj,h5py.Dataset):
                    if obj.dtype.kind == 'O':
                        raise ValueError('Native galaxy input contains unbounded variable-length data')
                    raw_bytes += int(np.prod(obj.shape,dtype=object))*obj.dtype.itemsize
            source.visititems(measure)
            if 'Subhalo/SubhaloLenType' in source:
                count += len(source['Subhalo/SubhaloLenType'])
    if count > request['maximum_galaxies'] or raw_bytes*8+32*1024**2 > request['memory_gib']*1024**3:
        raise MemoryError('Actual native galaxy/catalogue intake exceeds its frozen allocation; no particle selection is reduced')
    sources = {str(receipt_path):file_sha256(receipt_path)}
    _merge(sources,receipt['source_hashes'])
    _merge(sources,receipt['input_hashes'])
    for path in [*files,*native_publication_markers(files),root/'manifest.json',root/'config.yaml']:
        sources[str(path.resolve())] = file_sha256(path)
    return sources


def publish_native_galaxy_measurement(parameters,destination):
    """Measure galaxies from actual evolved bound stellar subhalos.

    The catalogue is produced by the established native SUBFIND adapter. A
    star-particle census alone never supplies a galaxy count or resolution.
    """
    from core.halos import build_galaxy_catalogue,verify_galaxy_catalogue
    from core.native import native_run_lock
    import h5py
    import numpy as np
    request = validate_native_galaxy_measurement(parameters)
    sources = galaxy_measurement_input_bindings(parameters)
    root = Path(parameters['native_run_directory']).resolve()
    destination = Path(destination).resolve()
    with native_run_lock(destination,operation='publish actual native pilot galaxies'):
        if destination.exists():
            raise FileExistsError('Native pilot galaxy measurements are immutable')
        destination.mkdir(parents=True)
        catalogue = destination/'galaxies.hdf5'
        build_galaxy_catalogue(root,request['snapshot'],minimum_star_particles=request['minimum_star_particles'],
            minimum_dm_particles=request['minimum_dm_particles'],output_path=catalogue)
        result = verify_galaxy_catalogue(catalogue,root)
        sources[str(catalogue)] = file_sha256(catalogue)
        with h5py.File(catalogue,'r') as source:
            provenance = json.loads(source.attrs['data_contract_json'])
            for row in provenance['input_hashes']+provenance['algorithm_sources']:
                _merge(sources,{str(Path(row['path']).resolve()):row['sha256']})
            resolved = np.asarray(source['Resolved'][:])
            if resolved.dtype.kind != 'b' or len(resolved) > request['maximum_galaxies']:
                raise ValueError('Native galaxy resolution requires actual bounded boolean support flags')
            count = int(np.count_nonzero(resolved))
            star_counts = np.asarray(source['NStars'][:])
            dm_counts = np.asarray(source['NDM'][:])
            if np.any(resolved & ((star_counts < request['minimum_star_particles']) | (dm_counts < request['minimum_dm_particles']))):
                raise ValueError('Native resolved galaxy catalogue contradicts frozen particle support')
        helper = Path(__file__).resolve()
        sources[str(helper)] = file_sha256(helper)
        for path,digest in sources.items():
            if file_sha256(path) != digest:
                raise ValueError('Native galaxy source changed during measured pilot publication')
        report = {'schema':'native-pilot-galaxy-measurement-v1','execution_status':'COMPLETE',
            'measurement_origin':'native_evolved_population','synthetic_fixture':False,
            'request_hash':canonical_digest(request),'snapshot':request['snapshot'],
            'galaxy_count':result['galaxy_count'],'resolved_galaxy_count':count,
            'minimum_star_particles':request['minimum_star_particles'],'minimum_dm_particles':request['minimum_dm_particles'],
            'catalogue':{'path':str(catalogue),'sha256':file_sha256(catalogue)},
            'source_hashes':sources,'science_acceptance':'PILOT_MEASURED_SUPPORT_ONLY',
            'selection':'actual luminous SUBFIND stellar membership; measured native particle-count resolution floors'}
        output = destination/'galaxy-measurement.json'
        atomic_json(output,report)
        return report


def native_galaxy_measurement_runner(parameters,context):
    destination = Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Native galaxy measurement escapes its owned workflow generation')
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Native galaxy measurement exceeds declared workflow memory')
    report = publish_native_galaxy_measurement(parameters,destination)
    return {'measurements':{key:report[key] for key in ('galaxy_count','resolved_galaxy_count')}}


def native_population_history(snapshot_sets, *, memory_gib, maximum_snapshots, base):
    """Count saved live representatives on their actual native time axis.

    Cosmological Header/Time is dimensionless a. In an isolated run it is
    code time, converted to Gyr using UnitLength/UnitVelocity. Zero-ID BH
    retirement slots remain in the storage census but are not live objects.
    """
    import h5py
    import numpy as np
    from core.data import verify_native_hdf5_set, epoch_contract, isolated_epoch_contract, native_time_unit_seconds
    from core.foundation import SECONDS_PER_YEAR
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Explicit finite native intake memory required')
    if type(maximum_snapshots) is not int or maximum_snapshots < 2:
        raise ValueError('At least two allowed native history epochs required')
    if not 2 <= len(snapshot_sets) <= maximum_snapshots:
        raise ValueError('Evolved pilot populations require at least two bounded saved epochs')
    records, seen = [], set()
    previous_time, history_units = None, None
    chunk_rows = max(1,min(100000,int(memory_gib*1024**3/64)))
    for epoch in snapshot_sets:
        snapshot = epoch['snapshot']
        if type(snapshot) is not int or snapshot < 0 or snapshot in seen:
            raise ValueError('Distinct explicit saved native snapshot indices required')
        seen.add(snapshot)
        references = epoch['files']
        if not references:
            raise ValueError('Native pilot snapshot needs a complete shard set')
        paths = [path_at(base,ref['path']) for ref in references]
        if len(set(paths)) != len(paths):
            raise ValueError('Repeated native pilot snapshot shard')
        for path, ref in zip(paths,references):
            if file_sha256(path) != ref['sha256']:
                raise ValueError('Native pilot snapshot shard changed')
        verify_native_hdf5_set(paths,require_publication=True)
        counts = {kind:0 for kind in SPECIES}
        stored_counts = dict(counts)
        native_time, snapshot_units = None, None
        for path in paths:
            with h5py.File(path,'r') as source:
                current = float(source['Header'].attrs['Time'])
                parameters = source['Parameters'].attrs
                mode = np.asarray(parameters['ComovingIntegrationOn'])
                if mode.shape != () or mode.dtype.kind not in 'iu' or int(mode) not in (0,1):
                    raise ValueError('Pilot history requires the recorded native integration mode')
                comoving = bool(mode)
                time_unit_gyr = None if comoving else native_time_unit_seconds(parameters)/(1e9*SECONDS_PER_YEAR)
                current_units = (comoving,time_unit_gyr)
                if not math.isfinite(current) or current < 0 or (comoving and current == 0):
                    raise ValueError('Pilot history requires positive a or nonnegative isolated time')
                if native_time is not None and (not math.isclose(native_time,current,rel_tol=0,abs_tol=1e-12) or snapshot_units != current_units):
                    raise ValueError('Native pilot shards disagree about physical clocks or time units')
                native_time, snapshot_units = current,current_units
                header = np.asarray(source['Header'].attrs['NumPart_ThisFile'])
                if header.shape != (6,) or header.dtype.kind not in 'ui':
                    raise ValueError('Native pilot census requires exact integer species counts')
                for kind, particle_type in SPECIES.items():
                    group = source.get('PartType'+str(particle_type))
                    measured = len(group['ParticleIDs']) if group is not None else 0
                    if measured != int(header[particle_type]):
                        raise ValueError('Native pilot census differs from delivered shard header')
                    stored_counts[kind] += measured
                    if kind == 'black_holes' and group is not None:
                        identity = group['ParticleIDs']
                        if identity.ndim != 1 or identity.dtype.kind != 'u' or identity.dtype.itemsize != 8:
                            raise ValueError('Pilot BH identities require exact native uint64 storage')
                        measured = sum(int(np.count_nonzero(identity[start:start+chunk_rows]))
                            for start in range(0,len(identity),chunk_rows))
                    counts[kind] += measured
        if history_units is not None and history_units != snapshot_units:
            raise ValueError('Pilot history cannot change integration mode or isolated time units')
        if previous_time is not None and native_time <= previous_time:
            raise ValueError('Native pilot population histories must follow increasing actual epochs')
        previous_time, history_units = native_time,snapshot_units
        comoving,time_unit_gyr = snapshot_units
        clock = epoch_contract(native_time) if comoving else isolated_epoch_contract(native_time,time_unit_gyr=time_unit_gyr,snapshot=snapshot)
        row = {'snapshot':snapshot,'epoch_contract':clock,'counts':counts,'stored_counts':stored_counts,
            'source_hashes':{str(path):ref['sha256'] for path,ref in zip(paths,references)}}
        if comoving:
            row['scale_factor'] = native_time
        else:
            row.update(native_time_code=native_time,elapsed_time_gyr=clock['elapsed_time_gyr'])
        records.append(row)
    return {'schema':'native-pilot-population-history-v1','epochs':records,
        'latest_counts':records[-1]['counts'],
        'source_hashes':{path:checksum for row in records for path,checksum in row['source_hashes'].items()},
        'count_semantics':'live native identities; zero-ID BH retirement slots excluded; stars represent SSP populations',
        'science_acceptance':'POPULATION_PRESENCE_ONLY'}


def verify_population_evidence(proof, family, *, receipt_base):
    history = native_population_history(proof['native_snapshot_sets'], memory_gib=proof['resources']['memory_gib'],
        maximum_snapshots=proof['resources']['maximum_snapshots'], base=receipt_base)
    if proof.get('native_population_counts') != history['latest_counts']:
        raise ValueError('Pilot receipt population counts differ from current evolved native outputs')
    if family == 'hydro_stellar_bh':
        for kind in ('gas', 'stars', 'black_holes'):
            if history['latest_counts'][kind] <= 0:
                raise ValueError('Hydro pilot lacks actual evolved '+kind)
    if family in {'collisionless_parent_zoom_stream', 'sidm_parent_zoom'}:
        if history['latest_counts']['dm'] <= 0:
            raise ValueError('Parent/zoom pilot has no delivered fine DM population')
    return history


def verify_inspected_artifacts(inspection, *, workflow_hash, base):
    from core.workflow import path_at
    if inspection.get('status') != 'PASS' or inspection.get('workflow_hash') != workflow_hash:
        raise ValueError('Visual inspection belongs to another or failed pilot')
    if not inspection.get('reviewer') or not inspection.get('inspection_method'):
        raise ValueError('Visual inspection must record who/what examined the artifacts')
    if inspection.get('schema')=='pilot-science-artifact-inspection-v1':
        decisions=inspection.get('decisions',[])
        required={'science','drawn_array','image','scene'}
        if inspection.get('execution_status')!='COMPLETE' or not decisions or not required<={row.get('category') for row in decisions} or any(row.get('passed') is not True for row in decisions):
            raise ValueError('Automated pilot inspection lacks passing measured decisions in every requested artifact category')
        if not inspection.get('source_hashes'):
            raise ValueError('Automated pilot inspection lacks current source bindings')
        for source,digest in inspection['source_hashes'].items():
            if file_sha256(path_at(base,source))!=digest:
                raise ValueError('Pilot inspected science/artifact source changed')
    retained = []
    for category in ('inspected_figures', 'fresh_loaded_scenes'):
        references = inspection.get(category)
        if not references:
            raise ValueError('Missing visual coverage: '+category)
        for ref in references:
            path = path_at(base, ref['path'])
            if file_sha256(path) != ref['sha256']:
                raise ValueError('Inspected figure/scene changed after inspection')
            row = {'category': category, 'path': str(path), 'sha256': ref['sha256']}
            if category == 'fresh_loaded_scenes':
                evidence_path = path_at(base, ref['fresh_load_receipt']['path'])
                if file_sha256(evidence_path) != ref['fresh_load_receipt']['sha256']:
                    raise ValueError('Fresh scene inspection receipt changed')
                receipt = json.loads(evidence_path.read_text())
                if receipt.get('schema') != 'blender-fresh-load-v1' or receipt.get('status') != 'PASS' or receipt.get('blend_sha256') != ref['sha256']:
                    raise ValueError('Scene inspection has no actual matching passing fresh-load evidence')
                row['fresh_load_receipt_sha256'] = ref['fresh_load_receipt']['sha256']
            retained.append(row)
    return {'status': 'PASS', 'reviewer': inspection['reviewer'], 'inspection_method': inspection['inspection_method'],
            'artifacts': retained, 'workflow_hash': workflow_hash}


def verify_live_stream_evidence(proof, history, *, receipt_base):
    """Recount actual released tracers and join progenitor ancestry to native IDs."""
    import numpy as np
    import h5py
    from core.stream_common import read_product
    from core.data import exact_uint64_ids
    from core.workflow import path_at
    reference = proof['live_stream_product']
    path = path_at(receipt_base,reference['path'])
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Representative live stream product changed')
    report,arrays,bindings = read_product(path,memory_gib=proof['resources']['memory_gib'])
    if report.get('schema') != 'live-stream-resimulation-v1' or (report.get('initialization') or {}).get('model') != 'actual measured progenitor; no equilibrium assumption':
        raise ValueError('Representative live stream must evolve an actual measured progenitor')
    if not report.get('parent_particle_namespace'):
        raise ValueError('Representative stream requires exact native parent ancestry')
    ids = exact_uint64_ids(arrays['particle_ids'])
    parents = exact_uint64_ids(arrays['parent_particle_ids'])
    times = np.asarray(arrays['times_myr'],float)
    bound = np.asarray(arrays['bound'])
    if not len(ids) or len(np.unique(ids)) != len(ids) or parents.shape != ids.shape or len(np.unique(parents)) != len(parents):
        raise ValueError('Live stream needs unique exact tracer and progenitor identities')
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Live stream needs distinct actual evolved output times')
    if bound.dtype.kind != 'b' or bound.shape != (len(times),len(ids)):
        raise ValueError('Live stream requires exact per-epoch bound state masks')
    before,after = (np.asarray(arrays[key],float) for key in ('release_before_myr','release_after_myr'))
    if before.shape != ids.shape or after.shape != ids.shape:
        raise ValueError('Measured release brackets must match exact tracers')
    bracket = np.isfinite(before)&np.isfinite(after)
    if np.any(before[bracket] >= after[bracket]) or np.any(after[bracket] > times[-1]):
        raise ValueError('Live stream release brackets contradict evolved output coverage')
    stripped = (~bound[-1])&bracket
    count = int(stripped.sum())
    declared = proof.get('live_stripped_particle_count')
    if type(declared) is not int or declared != count or count <= 0:
        raise ValueError('Pilot stripped count differs from actually evolved unbound tracers')
    # A resimulation namespace may allocate new IDs, but its recorded parent IDs
    # must be found in actual delivered native species, not an analysis fixture.
    budget = proof['resources']['memory_gib']*1024**3
    if len(parents)*40 > budget/2:
        raise MemoryError('Exact live progenitor ancestry workspace exceeds pilot intake memory')
    chunk_rows = max(1,min(100000,int(budget//64)))
    ordered = np.sort(parents)
    seen = np.zeros(len(ordered),bool)
    for native in history['source_hashes']:
        with h5py.File(native,'r') as source:
            for species in ('PartType1','PartType4'):
                group = source.get(species)
                if group is None:
                    continue
                dataset = group['ParticleIDs']
                for start in range(0,len(dataset),chunk_rows):
                    values = exact_uint64_ids(dataset[start:start+chunk_rows])
                    indices = np.searchsorted(ordered,values)
                    valid = indices < len(ordered)
                    valid[valid] = ordered[indices[valid]] == values[valid]
                    seen[indices[valid]] = True
    if not seen.all():
        raise ValueError('Live progenitor ancestry is absent from retained evolved native history')
    from core.data import numeric_audit_source_bindings
    sources = numeric_audit_source_bindings(report)
    joined = {}
    for group in (sources,history['source_hashes'],bindings):
        for source,expected in group.items():
            if source in joined and joined[source] != expected:
                raise ValueError('Contradictory live stream and native ancestry source bindings')
            joined[source] = expected
    for source,expected in joined.items():
        if file_sha256(source) != expected:
            raise ValueError('Live-stream/native ancestry evidence changed during intake')
    return {'live_stripped_particle_count':count,'parent_native_particle_count':len(parents),
        'measured_output_times_myr':times.tolist(),'source_hashes':bindings,
        'physical_scope':'self-gravitating measured progenitor in evolved prescribed host; no host backreaction claim'}


# Pilot inspection


def validate_pilot_inspection_request(request,specification):
    if request.get('schema')!='pilot-science-artifact-inspection-request-v1':
        raise ValueError('Frozen pilot science/artifact inspection request required')
    for name in ('maximum_science_checks','maximum_figures','maximum_scenes','maximum_image_pixels'):
        if type(request[name]) is not int or request[name]<1:raise ValueError('Positive exact inspection bound required: '+name)
    memory=request['memory_gib']
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Explicit positive pilot inspection memory required')
    stages={row['stage_id']:row for row in specification['stages']}
    def artifact(reference):
        if set(reference)!={'stage','artifact'} or reference['stage'] not in stages:
            raise ValueError('Inspection artifact must name its actual declared workflow producer')
        path=PurePosixPath(reference['artifact'])
        if path.is_absolute() or '..' in path.parts or '\\' in str(path) or not path.parts:
            raise ValueError('Inspection artifact must stay inside its producer generation')
    for key,bound in (('science_checks','maximum_science_checks'),('figures','maximum_figures'),('scenes','maximum_scenes')):
        rows=request[key]
        if not isinstance(rows,list) or not 1<=len(rows)<=request[bound]:raise ValueError('Bounded nonempty inspection category required: '+key)
    for row in request['science_checks']:
        reference=row['measurement']
        if set(reference)!={'stage','product'} or reference['stage'] not in stages or reference['product'] not in stages[reference['stage']]['outputs']:
            raise ValueError('Science inspection must use an actual declared measured product')
        conditions=row['conditions']
        if not isinstance(conditions,list) or not conditions or not any(item['operator'] in {'minimum','maximum','nonempty','all_true'} for item in conditions):
            raise ValueError('Science inspection needs measured quantity/support conditions')
        for item in conditions:
            if not isinstance(item.get('path'),list) or not item['path'] or item['operator'] not in {'exact','minimum','maximum','nonempty','all_true'}:
                raise ValueError('Malformed science inspection condition')
    for row in request['figures']:
        artifact(row['image']);artifact(row['sidecar'])
        if not isinstance(row['required_plotted_arrays'],list) or not row['required_plotted_arrays']:
            raise ValueError('Figure inspection must identify its actual drawn scientific arrays')
        for array in row['required_plotted_arrays']:
            if not isinstance(array['name'],str) or not array['name'] or not isinstance(array['units'],str) or not array['units'] or array['units'].startswith('UNDECLARED'):
                raise ValueError('Drawn arrays require exact names and physical units')
            if type(array['minimum_supported_values']) is not int or array['minimum_supported_values']<1:
                raise ValueError('Positive drawn-array support count required')
            fraction=array['maximum_invalid_fraction']
            if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not math.isfinite(fraction) or not 0<=fraction<=1:
                raise ValueError('Finite drawn-array invalid fraction required')
        for field in ('minimum_width_pixels','minimum_height_pixels'):
            if type(row[field]) is not int or row[field]<1:raise ValueError('Positive frozen image size threshold required')
        limit=row['minimum_rgb_standard_deviation']
        if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<=limit<=127.5:
            raise ValueError('Finite image-variation threshold in RGB byte units required')
    for row in request['scenes']:
        artifact(row['scene']);artifact(row['fresh_load'])
        if type(row['minimum_saved_epochs']) is not int or row['minimum_saved_epochs']<2:
            raise ValueError('Scene inspection requires an explicit measured-history epoch minimum')


def publish_pilot_inspection(request,workflow,destination,*,specification,request_base):
    from core.workflow import verify_completed_receipt
    from core.research.errors import _retained_json
    from core.figures import verify_figure_export
    import numpy as np
    from PIL import Image
    validate_pilot_inspection_request(request,specification)
    if workflow.get('execution_status')!='COMPLETE' or workflow.get('specification_hash')!=canonical_digest(specification):
        raise ValueError('Inspection requires the actual completed frozen pilot workflow')
    memory=int(request['memory_gib']*1024**3);sources={};receipts={}
    def receipt(stage):
        if stage not in receipts:
            path=Path(workflow['stages'][stage]['generation'])/'stage-receipt.json'
            record=verify_completed_receipt(path)
            sources[str(path.resolve())]=file_sha256(path)
            _merge(sources,record['source_hashes'])
            _merge(sources,record['input_hashes'])
            receipts[stage]=record
        return receipts[stage]
    def artifact(reference):
        record=receipt(reference['stage']);root=Path(record['generation']).resolve()
        path=(root/reference['artifact']).resolve()
        if not path.is_relative_to(root) or record['files_sha256'].get(path.relative_to(root).as_posix())!=file_sha256(path):
            raise ValueError('Inspected artifact lacks current completed producer ownership')
        owned=False
        for product in record['products'].values():
            product_path=(root/product['path']).resolve()
            if path==product_path or product['kind']=='directory' and path.is_relative_to(product_path):owned=True
        if not owned:raise ValueError('Inspected artifact is outside declared producer products')
        sources[str(path)]=file_sha256(path)
        return path
    decisions=[];measurement_bindings={}
    for index,row in enumerate(request['science_checks']):
        record=receipt(row['measurement']['stage']);path=Path(record['generation'])/'stage-receipt.json'
        product,bindings=_retained_json({'receipt':str(path),'product':row['measurement']['product']},request_base,memory)
        if product.get('schema')!=row['product_schema'] or product.get('synthetic_fixture') is True:
            raise ValueError('Inspection science product differs from its frozen physical producer contract')
        _merge(sources,bindings)
        measured_root=Path(record['generation']).resolve()
        # Code fingerprints and a shared configuration hash cannot establish
        # a figure's connection to a measured scientific artifact.
        _merge(measurement_bindings,{name:digest for name,digest in bindings.items()
            if (Path(name).resolve().is_relative_to(measured_root) and Path(name).name!='stage-receipt.json')
            or Path(name).suffix.lower() in {'.h5','.hdf5','.npz'}})
        for condition in row['conditions']:
            decisions.append({'category':'science','index':index,'condition':condition,'passed':bool(_condition(product,condition))})
    figures=[]
    def retained_digests(value):
        if isinstance(value,dict):
            return set().union(*(retained_digests(item) for item in value.values())) if value else set()
        if isinstance(value,list):return set().union(*(retained_digests(item) for item in value)) if value else set()
        return {value} if isinstance(value,str) and len(value)==64 else set()
    for index,row in enumerate(request['figures']):
        image=artifact(row['image']);sidecar=artifact(row['sidecar'])
        if sidecar.stat().st_size>min(memory//8,16*1024**2):raise MemoryError('Figure metadata exceeds inspection memory')
        raw_export=json.loads(sidecar.read_text())
        if raw_export.get('schema')!='darkmattersim-figure-v1' or len(raw_export.get('artifacts',[]))>4096:
            raise ValueError('Bounded actual scientific figure export required')
        producer=receipt(row['sidecar']['stage']);producer_root=Path(producer['generation']).resolve()
        for entry in raw_export['artifacts']:
            asset=(sidecar.parent/entry['path']).resolve()
            if not asset.is_relative_to(producer_root) or producer['files_sha256'].get(asset.relative_to(producer_root).as_posix())!=file_sha256(asset):
                raise ValueError('Drawn-array inspection requires current native pilot figure asset ownership')
            if asset.suffix=='.npz':
                with zipfile.ZipFile(asset) as archive:
                    if len(archive.infolist())>4096 or sum(item.file_size for item in archive.infolist())>memory//2:
                        raise MemoryError('Figure export array decoding exceeds declared inspection memory')
        export=verify_figure_export(sidecar)
        if not any(entry['sha256']==sources[str(image)] and (sidecar.parent/entry['path']).resolve()==image for entry in export['artifacts']):
            raise ValueError('Inspected image is not retained in its figure export')
        if not retained_digests(export.get('scientific_metadata',{}))&set(measurement_bindings.values()):
            raise ValueError('Figure scientific metadata has no inspected measured-product source binding')
        data=(sidecar.parent/export['plot_data']).resolve()
        record=receipt(row['sidecar']['stage']);root=Path(record['generation']).resolve()
        for entry in export['artifacts']:
            owned=(sidecar.parent/entry['path']).resolve()
            if not owned.is_relative_to(root) or record['files_sha256'].get(owned.relative_to(root).as_posix())!=file_sha256(owned):
                raise ValueError('Figure export asset is outside completed pilot producer ownership')
            sources[str(owned)]=file_sha256(owned)
        seal=sidecar.with_name(sidecar.name+'.sha256')
        if not seal.is_relative_to(root) or record['files_sha256'].get(seal.relative_to(root).as_posix())!=file_sha256(seal):
            raise ValueError('Figure export seal is outside completed producer ownership')
        sources[str(seal)]=file_sha256(seal)
        with zipfile.ZipFile(data) as archive:
            if sum(item.file_size for item in archive.infolist())>memory//2:
                raise MemoryError('Drawn scientific arrays exceed inspection allocation')
        contract=export.get('plot_data_contract',{}).get('datasets',{})
        with np.load(data,allow_pickle=False) as arrays:
            for expected in row['required_plotted_arrays']:
                name=expected['name'];values=np.asarray(arrays[name])
                if values.dtype.kind not in 'biuf' or values.size<1:raise ValueError('Drawn scientific arrays must be real numeric')
                metadata=contract.get(name,{})
                units=metadata.get('units')
                # A boolean support array contributes only its True entries.
                # Floating measurements contribute their finite entries.
                count = int(np.count_nonzero(values) if values.dtype.kind == 'b' else np.isfinite(values).sum())
                invalid = 1-count/values.size
                passed=units==expected['units'] and count>=expected['minimum_supported_values'] and invalid<=expected['maximum_invalid_fraction']
                decisions.append({'category':'drawn_array','figure':index,'array':name,'units':units,
                    'supported_values':count,'invalid_fraction':invalid,'passed':bool(passed)})
        with Image.open(image) as decoded:
            width,height=decoded.size
            if width*height>request['maximum_image_pixels'] or width*height*40>memory:
                raise MemoryError('Image decoding exceeds the frozen pilot inspection allocation')
            rgb=np.asarray(decoded.convert('RGB'),dtype=np.float64)
            deviation=float(np.std(rgb,axis=(0,1)).max())
        passed=width>=row['minimum_width_pixels'] and height>=row['minimum_height_pixels'] and deviation>=row['minimum_rgb_standard_deviation']
        decisions.append({'category':'image','figure':index,'width':width,'height':height,
                          'rgb_standard_deviation':deviation,'passed':bool(passed)})
        figures.append({'path':str(image),'sha256':sources[str(image)],'figure_export':{'path':str(sidecar),'sha256':sources[str(sidecar)]}})
    scenes=[]
    for index,row in enumerate(request['scenes']):
        scene=artifact(row['scene']);fresh=artifact(row['fresh_load'])
        if fresh.stat().st_size>min(memory//8,16*1024**2):raise MemoryError('Fresh scene receipt exceeds intake allocation')
        result=json.loads(fresh.read_text())
        if result.get('schema')!='blender-fresh-load-v1' or result.get('execution_status')!='COMPLETE' or result.get('blend_sha256')!=sources[str(scene)]:
            raise ValueError('Scene inspection requires its actual matching completed fresh-load result')
        epochs=result.get('epochs',[]);assets=result.get('assets',[]);timeline=result.get('timeline_controls',[])
        passed=result.get('status')=='PASS' and result.get('inspector_registered') is True and len(epochs)>=row['minimum_saved_epochs'] and bool(assets) and bool(timeline) and all(item.get('status')=='PASS' for item in epochs+assets+timeline)
        decisions.append({'category':'scene','scene':index,'saved_epoch_count':len(epochs),'asset_count':len(assets),
                          'timeline_control_count':len(timeline),'passed':bool(passed)})
        scenes.append({'path':str(scene),'sha256':sources[str(scene)],'fresh_load_receipt':{'path':str(fresh),'sha256':sources[str(fresh)]}})
    for path in (Path(__file__).resolve(), Path(__file__).resolve().parents[1]/'figures.py'):
        sources[str(path)] = file_sha256(path)
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Inspected pilot source changed before publication')
    report={'schema':'pilot-science-artifact-inspection-v1','workflow_hash':canonical_digest(workflow),
        'request_hash':canonical_digest(request),'reviewer':'automated frozen measured-artifact inspector',
        'inspection_method':'measured science conditions, sealed drawn arrays/image support, actual fresh-load identities/assets/timeline',
        'decisions':decisions,'status':'PASS' if all(row['passed'] for row in decisions) else 'FAIL',
        'execution_status':'COMPLETE','inspected_figures':figures,'fresh_loaded_scenes':scenes,'source_hashes':sources,
        'limits':['No human visual review is asserted','Rendered pixels do not establish scientific accuracy','Acceptance is confined to the frozen measurable conditions']}
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Pilot inspection generations are immutable')
    destination.mkdir(parents=True)
    output=destination/'pilot-inspection.json';atomic_json(output,report)
    return {'path':str(output),'sha256':file_sha256(output),'status':report['status']}


# Pilot recipes


def compile_native_pilot_recipe(request,destination,*,request_base):
    from core.workflow import path_at,safe_id,default_registry
    family=request['family']
    if family not in FAMILIES:raise ValueError('Unknown representative physical pilot family')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Native pilot recipe publications are immutable')
    resources=deepcopy(request['resources'])
    native=request['native_runs'];science=deepcopy(request['science_stages'])
    if type(request['maximum_native_runs']) is not int or request['maximum_native_runs']<1 or not isinstance(native,list) or not native or len(native)>request['maximum_native_runs']:
        raise ValueError('Pilot requires an explicitly bounded actual native run family')
    destination.mkdir(parents=True)
    stages=[];products={};native_roles={};assets={}
    for asset in request.get('assets',[]):
        path=path_at(request_base,asset['path'])
        if file_sha256(path)!=asset['sha256']:raise ValueError('Frozen physical pilot input changed')
        assets[str(path)]=asset['sha256']
    for row in native:
        role=safe_id(row['role']);sid=safe_id('native_'+role);product=safe_id(role+'_evolution')
        if role in native_roles:raise ValueError('Repeated native pilot role')
        if 'answers_reference' in row:
            if 'answers' in row:raise ValueError('Select one complete inline or pinned native answer specification')
            reference=row['answers_reference'];path=path_at(request_base,reference['path'])
            if path.stat().st_size>16*1024**2 or file_sha256(path)!=reference['sha256']:
                raise ValueError('Pilot complete native answers changed or exceed bounded metadata intake')
            answers=json.loads(path.read_text());assets[str(path)]=reference['sha256']
        else:
            answers=deepcopy(row['answers'])
        if family=='hydro_stellar_bh' and answers.get('physics')!='hydro':raise ValueError('Hydro pilot cannot replace evolved gas/stars with DM-only inputs')
        if family=='sidm_parent_zoom' and not answers.get('sidm',{}).get('enabled',False):raise ValueError('SIDM pilot native runs require actual enabled SIDM settings')
        if family=='collisionless_parent_zoom_stream' and answers.get('sidm',{}).get('enabled',False):raise ValueError('Collisionless pilot cannot silently enable scattering')
        if answers['mpi_ranks']!=resources['mpi_ranks'] or answers['omp_threads']!=resources['omp_threads'] or answers['memory_limit_gb']!=resources['memory_gib']:
            raise ValueError('Pilot native resource settings must agree with the frozen workflow allocation')
        answer_path=destination/(role+'.answers.json');atomic_json(answer_path,answers)
        arguments=['--answers',{'input':str(answer_path)},'--run-dir','{generation}/run']
        row_assets=[]
        for key,flag in (('existing_ic','--existing-ic'),('native_generation_report','--native-generation-report')):
            if key in row:
                asset=row[key]
                if 'pilot_product' in asset:
                    arguments += [flag,deepcopy(asset)]
                    continue
                path=path_at(request_base,asset['path'])
                if file_sha256(path)!=asset['sha256']:raise ValueError('Native pilot physical handoff changed: '+key)
                row_assets.append({'path':str(path),'sha256':asset['sha256']});arguments+=[flag,{'input':str(path)}]
        stage={'stage_id':sid,'kind':'public_command','parameters':{'command':'gadget','arguments':arguments,
            'wall_seconds':row['wall_seconds']},'outputs':{product:'run'},'input_files':row_assets,
            'product_contracts':{product:{'reader':'native_snapshot_series','snapshots':row['snapshots']}}}
        stages.append(stage);native_roles[role]={'stage':sid,'product':product};products[product]=sid
    if family in {'collisionless_parent_zoom_stream','sidm_parent_zoom'} and not {'parent','zoom'}<=set(native_roles):
        raise ValueError('Parent/zoom pilots require both actual native trajectories')
    for row in science:
        safe_id(row['stage_id'])
        for product in row['outputs']:
            safe_id(product)
            if product in products:raise ValueError('Pilot science product has multiple producers')
            products[product]=row['stage_id']
    templates=request.get('physical_requests',{})
    request_ids={name:safe_id('physical_request_'+name) for name in templates}
    def resolve(value):
        if isinstance(value,dict):
            if 'pilot_product' in value:
                if not set(value)<={'pilot_product','artifact','load'}:raise ValueError('Malformed actual pilot product reference')
                product=value['pilot_product']
                if product not in products:raise ValueError('Pilot physical request consumes an absent measured product')
                if 'artifact' not in value:raise ValueError('Pilot physical product reference must identify the actual delivered artifact')
                return {'stage':products[product],**{key:item for key,item in value.items() if key!='pilot_product'}}
            if set(value)=={'pilot_request'}:
                name=value['pilot_request']
                if name not in request_ids:raise ValueError('Pilot physical request file has no concrete materializer')
                return {'stage':request_ids[name],'artifact':'request.json'}
            return {key:resolve(item) for key,item in value.items()}
        if isinstance(value,list):return [resolve(item) for item in value]
        return value
    stages=resolve(stages)
    for name,template in templates.items():
        stages.append({'stage_id':request_ids[name],'kind':'materialize_request',
            'parameters':{'request':resolve(deepcopy(template))},'outputs':{safe_id(name+'_request'):'request.json'}})
    stages+=resolve(science)
    ids=[row['stage_id'] for row in stages]
    if len(set(ids))!=len(ids) or len(ids)>resources['maximum_stage_count']:raise ValueError('Pilot stage identities or allocation are invalid')
    registry=default_registry()
    for row in stages:
        definition=registry.get(row['kind'])
        if any(key not in row.get('parameters',{}) for key in definition.required_parameters):
            raise ValueError('Pilot producer lacks its concrete required parameters: '+row['stage_id'])
    requested=request['requested_products']
    if not isinstance(requested,list) or not requested or any(product not in products for product in requested):
        raise ValueError('Pilot must request its actual science and visual products')
    workflow={'schema':'research-workflow-v1','resources':resources,'followup_limits':deepcopy(request['followup_limits']),
        'physics_spec':deepcopy(request['physics_spec']),'requested_products':requested,'stages':stages}
    coverage=deepcopy(request['coverage']);coverage.update(schema='native-pilot-coverage-request-v1',family=family)
    coverage['native_evolution_stages']=[row['stage'] for row in native_roles.values()]+[row['stage_id'] for row in stages if row['kind']=='isolated_physical_collision']
    for row in coverage['capabilities']:
        reference=row['measurement']
        if set(reference)=={'product'}:
            product=reference['product'];row['measurement']={'stage':products[product],'product':product}
        if 'native_roles' in row:
            roles=row.pop('native_roles')
            if not isinstance(roles,list) or not roles or any(role not in native_roles for role in roles):
                raise ValueError('Pilot capability needs its actual declared native physical roles')
            row['native_stage_ids']=[native_roles[role]['stage'] for role in roles]
    for row in coverage.get('native_snapshot_sets',[]):
        if 'native_role' in row:
            native_role=native_roles[row.pop('native_role')]
            row.update(native_stage=native_role['stage'],run_product=native_role['product'])
    if coverage.get('live_stream_measurement') and set(coverage['live_stream_measurement'])=={'product'}:
        product=coverage['live_stream_measurement']['product'];coverage['live_stream_measurement']={'stage':products[product],'product':product}
    validate_pilot_coverage_request(coverage,workflow)
    required_products=[row['measurement']['product'] for row in coverage['capabilities'] if 'stage' in row['measurement']]
    required_products += [row['product'] for row in native_roles.values()]
    required_products += [product for stage in stages if stage['kind']=='isolated_physical_collision' for product in stage['outputs']]
    if coverage.get('live_stream_measurement') and 'stage' in coverage['live_stream_measurement']:
        required_products.append(coverage['live_stream_measurement']['product'])
    inspection=None
    if request.get('inspection'):
        inspection=deepcopy(request['inspection'])
        inspection['schema']='pilot-science-artifact-inspection-request-v1'
        for row in inspection['science_checks']:
            reference=row['measurement']
            if set(reference)=={'product'}:
                product=reference['product'];row['measurement']={'stage':products[product],'product':product}
            required_products.append(row['measurement']['product'])
        for category,fields in (('figures',('image','sidecar')),('scenes',('scene','fresh_load'))):
            for row in inspection[category]:
                for field in fields:
                    reference=row[field]
                    if 'product' in reference:
                        if set(reference)!={'product','artifact'}:raise ValueError('Pilot inspection artifact shorthand requires exact product and artifact')
                        product=reference['product'];row[field]={'stage':products[product],'artifact':reference['artifact']}
                        required_products.append(product)
                    else:
                        producer=next(stage for stage in stages if stage['stage_id']==reference['stage'])
                        required_products.extend(producer['outputs'])
        validate_pilot_inspection_request(inspection,workflow)
    workflow['requested_products']=list(dict.fromkeys([*requested,*required_products]))
    atomic_json(destination/'workflow.json',workflow);atomic_json(destination/'coverage-request.json',coverage)
    report={'family':family,'workflow':str(destination/'workflow.json'),'workflow_sha256':file_sha256(destination/'workflow.json'),
        'coverage_request':{'path':str(destination/'coverage-request.json'),'sha256':file_sha256(destination/'coverage-request.json')},
        'physics_input_sources':assets,'execution_status':'NOT_EXECUTED','request_hash':canonical_digest(request)}
    if inspection is not None:
        atomic_json(destination/'inspection-request.json',inspection)
        report['inspection_request']={'path':str(destination/'inspection-request.json'),'sha256':file_sha256(destination/'inspection-request.json')}
    atomic_json(destination/'recipe.json',report)
    return report


def compile_representative_pilot_family(request,destination,*,request_base):
    if request.get('schema')!='native-pilot-recipes-request-v1':raise ValueError('Complete native pilot recipe request required')
    recipes=request['pilots']
    if not isinstance(recipes,list) or len(recipes)!=4 or {row['family'] for row in recipes}!=FAMILIES:
        raise ValueError('All four distinct representative physical pilot recipes are required')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Pilot family recipe publications are immutable')
    rows=[compile_native_pilot_recipe(row,destination/row['family'],request_base=request_base) for row in recipes]
    family={'schema':'representative-pilot-family-v1','pilots':rows}
    atomic_json(destination/'pilot-family.json',family)
    return {'schema':'compiled-native-pilot-family-v1','campaign':str(destination/'pilot-family.json'),
        'campaign_sha256':file_sha256(destination/'pilot-family.json'),'pilots':rows,
        'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED'}


def pilot_recipes_main():
    import argparse
    parser=argparse.ArgumentParser(description='Compile connected native physical pilot workflows and frozen coverage requests')
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.request.stat().st_size>16*1024**2:raise MemoryError('Native pilot recipe request exceeds bounded intake')
    print(json.dumps(compile_representative_pilot_family(json.loads(args.request.read_text()),args.output,
        request_base=args.request.resolve().parent),indent=2))
