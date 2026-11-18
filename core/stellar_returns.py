"""Delayed stellar-return input support, native state and campaign stages.

Model compilation still lives beside stellar yield/IMF calculations in stars.
Here, preparation checks the requested inputs; history replay connects exact
native counters and cumulative mass/element/energy ledgers to committed source
operations. Passing bookkeeping does not establish stellar-model validity.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
from core.foundation import MSUN_G, atomic_json, file_sha256


# Snapshot units follow the native model: code masses -> Msun, extensive
# energies -> erg. Comoving mass and energy carry h^-1, with no a factor.
# Source time stays as scale factor for cosmological runs and native elapsed
# time otherwise; it is not silently relabelled as stellar age.
RETURN_FIELDS={'SSPCumulativeReturnedMass':('mass',1),'SSPCumulativeReturnedElements':('mass',11),
        'SSPCumulativeInjectedEnergy':('energy',1),'SSPLastSourceNativeTime':('time',1),
        'SSPSourceEventCounter':('uint64',1),'SSPSourceModelCRC32':('uint64',1)}


def allocate_stellar_return_state(total):
    """Allocate aligned census columns; zero placeholders remain unavailable."""
    result={'stellar_return_state_available':np.zeros(total,dtype=bool)}
    for name,(kind,width) in RETURN_FIELDS.items():
        result[name]=np.zeros((total,width) if width>1 else total,dtype=np.uint64 if kind=='uint64' else float)
    return result


def read_stellar_return_state(group,*,parameters,native_time,required):
    """Read exact counters and convert only extensive physical quantities.

    M_Msun = M_code * UnitMass_g / (Msun_g * h) in comoving runs;
    E_erg = E_code * UnitMass_g * UnitVelocity_cm_s^2 / h. Use h=1
    for noncomoving runs. Counters/model IDs remain original uint64 values.
    """
    present=set(RETURN_FIELDS)&set(group)
    if not present and not required:
        return None
    if present!=set(RETURN_FIELDS):
        raise ValueError('Complete native cumulative stellar return state required')
    n=len(group['ParticleIDs'])
    comoving=bool(parameters['comoving'])
    unit_mass=parameters['unit_mass_g']
    unit_energy=unit_mass*parameters['unit_velocity_cm_s']**2
    mass_factor=unit_mass/MSUN_G/(parameters['hubble_param'] if comoving else 1.)
    energy_factor=unit_energy/(parameters['hubble_param'] if comoving else 1.)
    result={}
    for name,(kind,width) in RETURN_FIELDS.items():
        dataset=group[name]
        shape=(n,width) if width>1 else (n,)
        if dataset.shape!=shape or dataset.dtype!=(np.dtype('uint64') if kind=='uint64' else np.dtype('float64')):
            raise ValueError('Misaligned or lossy cumulative stellar source state: '+name)
        cgs=unit_mass if kind=='mass' else unit_energy if kind=='energy' else parameters['unit_time_s'] if kind=='time' and not comoving else 1.
        h_scale=-1. if comoving and kind in ('mass','energy') else 0.
        if any(key not in dataset.attrs or float(dataset.attrs[key])!=expected for key,expected in
               {'to_cgs':cgs,'a_scaling':0.,'h_scaling':h_scale}.items()):
            raise ValueError('Cumulative stellar source units disagree with native parameters: '+name)
        values=dataset[:]
        if not np.isfinite(values).all() or np.any(values<0):
            raise ValueError('Invalid negative/nonfinite cumulative stellar source state')
        # Convert only dimensional extensive fields. Counters and clocks
        # already have their required representation; avoid a multiplied copy
        # that would immediately be discarded for each uint64 field.
        if kind=='mass':
            result[name]=values*mass_factor
        elif kind=='energy':
            result[name]=values*energy_factor
        else:
            result[name]=values
    if np.any(result['SSPSourceModelCRC32']>2**32-1):
        raise ValueError('Native stellar source model CRC exceeds uint32 range')
    if np.any(result['SSPLastSourceNativeTime']>native_time):
        raise ValueError('Native cumulative stellar source clock is ahead of its saved epoch')
    if 'SSPBirthNativeTime' in group and np.any(result['SSPLastSourceNativeTime']<group['SSPBirthNativeTime'][:]):
        raise ValueError('Native cumulative stellar source clock precedes birth')
    return result


def reconcile_stellar_return_state(before,after,records,*,before_meta,after_meta,tolerances):
    """Compare original uint64 counters and extensive cumulative increments.

    Mass/element checks retain the hydro mass tolerance. The energy floor is
    derived from that same mass floor and actual source specific energies.
    No new configurable tolerance is introduced here.
    """
    required=before_meta.get('return_schema',0)==1 or after_meta.get('return_schema',0)==1 or bool(records)
    if not required:
        return {'status':'NOT_APPLICABLE','mode':'PROMPT_ONLY','sources':[],'issues':[]}
    if before_meta.get('return_schema')!=1 or after_meta.get('return_schema')!=1:
        raise ValueError('Delayed stellar source schema changes between actual saved epochs')
    def slots(census):
        ids=np.asarray(census['particle_ids'])
        species=np.asarray(census['species'])
        if ids.dtype!=np.dtype('uint64') or ids.ndim!=1 or species.shape!=ids.shape or len(np.unique(ids))!=len(ids):
            raise ValueError('Exact distinct native census identities required for stellar source replay')
        selected=np.flatnonzero(species==4)
        available=np.asarray(census.get('stellar_return_state_available',[]))
        if available.shape!=ids.shape or available.dtype.kind!='b' or not available[selected].all():
            raise ValueError('Delayed native SSP has no observed cumulative source state')
        for name,(kind,width) in RETURN_FIELDS.items():
            values=np.asarray(census.get(name,[]))
            shape=(len(ids),width) if width>1 else ids.shape
            if values.shape!=shape or values.dtype!=(np.dtype('uint64') if kind=='uint64' else np.dtype('float64')) or not np.isfinite(values[selected]).all() or np.any(values[selected]<0):
                raise ValueError('Exact typed native cumulative source census required: '+name)
        return {int(ids[index]):int(index) for index in selected}
    left,right=slots(before),slots(after)
    donors={}
    for row in records:
        if row['species']!=4:
            continue
        pid=row['particle_id']
        counter=row['source_counter']
        if type(pid) is not int or type(counter) is not int or not 0<=pid<2**64 or not 0<counter<2**64 or row['source_id']!=pid:
            raise ValueError('Exact per-source native stellar operation identity required')
        donors.setdefault(pid,[]).append(row)
    issues=[]
    summaries=[]
    model_crc=after_meta.get('parameters',{}).get('stellar_model_crc32')
    if type(model_crc) is not int or not 0<model_crc<2**32:
        raise ValueError('Delayed source audit requires the actual frozen native stellar model CRC')
    def issue(code,pid,**details):
        issues.append({'kind':code,'particle_id':pid,**details})
    for pid in sorted(set(left)|set(right)|set(donors)):
        if pid not in right:
            issue('RETURN_SOURCE_DISAPPEARED_FROM_CUMULATIVE_CENSUS',pid)
            continue
        i=left.get(pid)
        j=right[pid]
        rows=sorted(donors.get(pid,[]),key=lambda row:row['source_counter'])
        start=int(before['SSPSourceEventCounter'][i]) if i is not None else 0
        stop=int(after['SSPSourceEventCounter'][j])
        counter_values=[row['source_counter'] for row in rows]
        if stop<start or stop-start!=len(rows) or any(counter!=start+ordinal+1 for ordinal,counter in enumerate(counter_values)):
            issue('RETURN_SOURCE_COUNTER_DOES_NOT_MATCH_COMMITTED_OPERATIONS',pid,before=start,after=stop,committed=len(rows))
        if i is None:
            birth=float(after['SSPBirthNativeTime'][j])
            if not before_meta['native_time']<=birth<=after_meta['native_time']:
                issue('NEW_RETURN_SOURCE_HAS_NO_BIRTH_IN_SAVED_INTERVAL',pid)
        else:
            clock_before=float(before['SSPLastSourceNativeTime'][i])
            clock_after=float(after['SSPLastSourceNativeTime'][j])
            if clock_after<clock_before:
                issue('RETURN_SOURCE_CLOCK_REGRESSED',pid)
        old_model=int(before['SSPSourceModelCRC32'][i]) if i is not None else 0
        new_model=int(after['SSPSourceModelCRC32'][j])
        if old_model and old_model!=new_model or rows and not new_model:
            issue('RETURN_SOURCE_MODEL_CHANGED_OR_UNBOUND',pid)
        if new_model and new_model!=model_crc:
            issue('RETURN_SOURCE_MODEL_DIFFERS_FROM_NATIVE_PARAMETERS',pid)
        clock=float(after['SSPLastSourceNativeTime'][j])
        if clock>after_meta['native_time'] or rows and clock<max(row['native_time'] for row in rows):
            issue('RETURN_SOURCE_CLOCK_DOES_NOT_COVER_COMMITTED_OPERATIONS',pid)
        increments={}
        for name in ('SSPCumulativeReturnedMass','SSPCumulativeReturnedElements','SSPCumulativeInjectedEnergy'):
            final=np.asarray(after[name][j])
            initial=np.asarray(before[name][i]) if i is not None else np.zeros_like(final)
            delta=final-initial
            if not np.isfinite(delta).all() or np.any(delta<0):
                issue('RETURN_CUMULATIVE_SOURCE_REGRESSED',pid,field=name)
            increments[name]=delta
        expected_mass=sum(row['mass_before_msun']-row['mass_after_msun'] for row in rows)
        expected_elements=sum((np.asarray(row['elements_after_msun'],float) for row in rows),start=np.zeros(11))
        expected_energy=sum(row['injected_energy_erg'] for row in rows)
        for name,expected in (('SSPCumulativeReturnedMass',expected_mass),('SSPCumulativeReturnedElements',expected_elements)):
            measured=increments[name]
            allowed=tolerances['mass_absolute_msun']+tolerances['mass_relative']*np.maximum(abs(measured),abs(expected))
            if np.any(abs(measured-expected)>allowed):
                issue('RETURN_CUMULATIVE_LEDGER_MISMATCH',pid,field=name)
        measured=float(increments['SSPCumulativeInjectedEnergy'])
        specific=expected_energy/expected_mass if expected_mass>0 else 0.
        allowed=tolerances['mass_absolute_msun']*specific+tolerances['mass_relative']*max(abs(measured),abs(expected_energy))
        if abs(measured-expected_energy)>allowed:
            issue('RETURN_CUMULATIVE_ENERGY_MISMATCH',pid)
        mass=float(increments['SSPCumulativeReturnedMass'])
        elements=increments['SSPCumulativeReturnedElements']
        if abs(float(elements.sum())-mass)>tolerances['mass_absolute_msun']+tolerances['mass_relative']*max(abs(mass),abs(float(elements.sum()))):
            issue('RETURN_CUMULATIVE_ELEMENTS_DO_NOT_CLOSE',pid)
        if 'SSPInitialLockedMass' in after:
            locked=float(after['SSPInitialLockedMass'][j])
            current=float(after['mass_msun'][j])
            returned=float(after['SSPCumulativeReturnedMass'][j])
            if abs(locked-current-returned)>tolerances['mass_absolute_msun']+tolerances['mass_relative']*max(locked,current+returned):
                issue('RETURN_RETAINED_SSP_MASS_NOT_CLOSED',pid)
        summaries.append({'particle_id':pid,'counter_before':start,'counter_after':stop,'committed_source_operations':len(rows),
            'returned_mass_increment_msun':mass,'injected_energy_increment_erg':measured,'model_crc32':new_model})
    return {'status':'FAIL' if issues else 'PASS','mode':'DELAYED_RETURN','sources':summaries,'issues':issues,
        'scope':'Exact native cumulative SSP source fields versus committed interval operations; zero-event clocks do not imply physical population coverage'}


def stellar_return_input_preflight(request,destination):
    """Report yield closure and requested IMF/age support without modifying data.

    Total ejecta should sum to initial mass minus remnant mass, in Msun.
    Every supplied table row is retained, including failed rows. This check
    cannot certify SNIa reservoirs, deposition, or native evolution.
    """
    from .stars import load_stellar_yield_library,stellar_imf_density
    if request.get('schema')!='stellar-return-model-v1':
        raise ValueError('Explicit delayed stellar return model required')
    destination=Path(destination).resolve()
    if not destination.parent.is_dir() or destination.exists():
        raise FileExistsError('Preserve previous stellar input preflight; select a new file in an existing directory')
    tolerance=request['yield_mass_closure_relative_tolerance']
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or not 0<=tolerance<=.05:
        raise ValueError('Frozen finite stellar closure tolerance required')
    pins={}
    for name in ('yield_asset','snia_asset'):
        ref=request[name]
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:
            raise ValueError('Pinned stellar preflight input changed: '+name)
        pins[str(path)]=ref['sha256']
    source=request['yield_asset']
    rows=load_stellar_yield_library(source['path'],expected_sha256=source['sha256'])
    ages=np.asarray(request['age_grid_years'],dtype=float)
    if ages.ndim!=1 or len(ages)<3 or ages[0]!=0 or not np.isfinite(ages).all() or np.any(np.diff(ages)<=0):
        raise ValueError('Finite increasing declared stellar model ages required')
    stellar_imf_density(np.array([1.]),request['imf'])
    minimum=request['imf']['minimum_mass_msun']
    maximum=request['imf']['maximum_mass_msun']
    cutoff=request['core_collapse_minimum_mass_msun']
    if type(cutoff) not in (int,float) or not math.isfinite(cutoff) or cutoff<=0:
        raise ValueError('Explicit stellar channel boundary required')
    failures=[]
    measurements=[]
    support=[]
    for row in rows:
        # Total ejecta are extensive masses, not signed net production yields.
        expected=row['mass_msun']-row['remnant_msun']
        ejecta=float(row['elements_msun'].sum())
        residual=ejecta-expected
        relative=abs(residual)/expected
        measurement={name:row[name] for name in ('mass_msun','metallicity','remnant_msun','lifetime_years')}
        measurement.update(expected_return_msun=expected,represented_ejecta_msun=ejecta,
            signed_closure_residual_msun=residual,relative_closure_error=relative,
            elements_msun=row['elements_msun'].tolist(),status='PASS' if relative<=tolerance else 'FAIL')
        measurements.append(measurement)
        if relative>tolerance:
            failures.append({'code':'TOTAL_EJECTA_MASS_CLOSURE','mass_msun':row['mass_msun'],
                'metallicity':row['metallicity'],'relative_error':relative,'tolerance':tolerance,
                'required_input':'Source-supported total ejecta/remnant interpretation or corrected table; no normalization or row removal'})
    for metallicity in sorted({row['metallicity'] for row in rows}):
        grid=sorted((row for row in rows if row['metallicity']==metallicity),key=lambda row:row['mass_msun'])
        low=grid[0]
        high=grid[-1]
        local=[]
        if maximum>high['mass_msun']:
            local.append('IMF_UPPER_MASS_OUTSIDE_YIELD_SUPPORT')
        if minimum<low['mass_msun'] and ages[-1]>=low['lifetime_years']:
            local.append('LOW_MASS_LIFETIME_YIELD_SUPPORT_MISSING')
        if not max(minimum,low['mass_msun'])<cutoff<maximum:
            local.append('CHANNEL_BOUNDARY_OUTSIDE_IMF_YIELD_SUPPORT')
        if np.any(np.diff([row['lifetime_years'] for row in grid])>=0):
            local.append('NONMONOTONIC_STELLAR_LIFETIME')
        support.append({'metallicity':metallicity,'yield_mass_range_msun':[low['mass_msun'],high['mass_msun']],
            'lifetime_range_years':[high['lifetime_years'],low['lifetime_years']],
            'requested_imf_mass_range_msun':[minimum,maximum],'requested_maximum_age_years':float(ages[-1]),
            'unsupported_requirements':local,'status':'FAIL' if local else 'PASS'})
        failures.extend({'code':code,'metallicity':metallicity,
            'required_input':'Complete source-supported stellar lifetime/yield coverage; preserve requested IMF and ages'} for code in local)
    for path,digest in pins.items():
        if file_sha256(path)!=digest:
            raise ValueError('Stellar preflight input changed during audit')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    pins[str(Path(__file__).with_name('stars.py').resolve())]=file_sha256(Path(__file__).with_name('stars.py'))
    result={'schema':'stellar-return-input-preflight-v1','status':'FAIL' if failures else 'PASS',
        'readiness':'INPUTS_REJECTED' if failures else 'INPUT_SUPPORT_CHECKED_NOT_COMPILED',
        'request':request,'source_hashes':pins,'element_order':['H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe'],
        'yield_rows':measurements,'yield_row_count':len(rows),'failed_yield_row_count':sum(row['status']=='FAIL' for row in measurements),
        'metallicity_support':support,'failures':failures,'normalization_applied':False,
        'scope':'Full source-row closure and requested IMF/age support. SNIa reservoir, prompt subtraction, deposition, energy and restart accounting remain mandatory compiler/native checks.',
        'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(destination,result)
    return result


def validate_stellar_model_stage(parameters):
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Positive native stellar model preparation memory required')


def stellar_model_runner(parameters,context):
    """Check frozen yield support, then compile the existing source model."""
    from .stars import compile_stellar_return_model,read_stellar_return_model
    validate_stellar_model_stage(parameters)
    memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:
        raise MemoryError('Stellar model preparation exceeds campaign allocation')
    path=Path(parameters['request_path']).resolve()
    if file_sha256(path)!=parameters['request_sha256']:
        raise ValueError('Frozen stellar source request changed')
    if path.stat().st_size*8>memory*1024**3/16:
        raise MemoryError('Stellar model request exceeds bounded intake')
    request=json.loads(path.read_text())
    for name in ('yield_asset','snia_asset'):
        if Path(request[name]['path']).stat().st_size*8>memory*1024**3/16:
            raise MemoryError('Stellar source asset exceeds bounded model preparation intake')
    # Size both the cumulative channels and all metallicity/age population
    # grids, together with Python compilation/publication temporaries.
    if len(request['age_grid_years'])*64*3*13*8*16+request['mass_quadrature_points']*2048>memory*1024**3:
        raise MemoryError('Declared native stellar table preparation exceeds memory allocation')
    audit=stellar_return_input_preflight(request,context.generation/'input-support.json')
    if audit['status']!='PASS':
        raise ValueError('Selected stellar source inputs failed closure/support; input-support.json retains every unmet prerequisite; delayed branch is not ready')
    destination=context.generation/'model.json'
    compile_stellar_return_model(request,destination)
    result=read_stellar_return_model(destination)[0]
    if file_sha256(path)!=parameters['request_sha256']:
        raise ValueError('Stellar source request changed during model compilation')
    atomic_json(context.generation/'model-binding.json',{'schema':'native-stellar-model-binding-v1',
        'model_manifest':str(destination),'model_sha256':file_sha256(destination),'table':result['table'],
        'mode':'DELAYED_RETURN','compiled_only':True,'science_acceptance':'NOT_ACCEPTED'})
    return {'status':'COMPILED_MODEL','science_acceptance':'NOT_ACCEPTED'}


def validate_stellar_branch_stage(parameters):
    if parameters['mode'] not in ('prompt_only','delayed_returns'):
        raise ValueError('Explicit prompt_only or delayed_returns stellar mode required')
    required={'kernel_radius_proper_kpc','source_fraction','maximum_count','maximum_subcycles'}
    if parameters['mode']=='delayed_returns':
        if not parameters.get('model_manifest') or not isinstance(parameters.get('source_controls'),dict) or set(parameters['source_controls'])!=required:
            raise ValueError('Delayed stellar branch requires an actual compiled model and complete native source controls')
    elif parameters.get('model_manifest') is not None or parameters.get('source_controls') is not None:
        raise ValueError('Prompt-only branch cannot silently carry a delayed source model or controls')


def stellar_branch_answers_runner(parameters,context):
    """Select prompt-only or delayed returns while preserving the baseline."""
    from core.research.capabilities import validate_native_answers
    from .evolution import gadget_config_text
    from .stars import read_stellar_return_model
    validate_stellar_branch_stage(parameters)
    path=Path(parameters['answers_path']).resolve()
    if path.stat().st_size>min(16*1024**2,context.resources['memory_gib']*1024**3/16):
        raise MemoryError('Native stellar branch answers exceed bounded intake')
    digest=file_sha256(path)
    if parameters.get('answers_sha256') is not None and digest!=parameters['answers_sha256']:
        raise ValueError('Frozen native stellar branch baseline changed')
    original=json.loads(path.read_text())
    answers=copy.deepcopy(original)
    if original.get('stellar_evolution',{}).get('enabled',False):
        raise ValueError('Stellar branch preparation requires the original explicit prompt-only baseline')
    model=None
    pins={str(path):digest}
    if parameters['mode']=='prompt_only':
        answers['stellar_evolution']={'enabled':False}
    else:
        model_path=Path(parameters['model_manifest']).resolve()
        model_digest=file_sha256(model_path)
        model=read_stellar_return_model(model_path)[0]
        for source,checksum in model['source_hashes'].items():
            if file_sha256(source)!=checksum:
                raise ValueError('Compiled stellar model input/source changed')
            pins[source]=checksum
        pins[str(model_path)]=model_digest
        pins[model['table']['path']]=model['table']['sha256']
        answers['stellar_evolution']={'enabled':True,'model_manifest':str(model_path),'model_sha256':model_digest,
            **parameters['source_controls']}
    cfg=validate_native_answers(answers,context.resources)
    flags=gadget_config_text(cfg).splitlines()
    if ('STELLAR_EVOLUTION' in flags)!=(parameters['mode']=='delayed_returns'):
        raise ValueError('Explicit stellar branch differs from selected native compiled physics')
    if answers.get('physics')!='hydro' or 'STARFORMATION' not in flags:
        raise ValueError('Native stellar branches require the retained hydro/star-formation baseline')
    for source,checksum in pins.items():
        if file_sha256(source)!=checksum:
            raise ValueError('Native stellar branch input changed before publication')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    atomic_json(context.generation/'answers.json',answers)
    result={'schema':'native-stellar-branch-preparation-v1','mode':parameters['mode'],'source_hashes':pins,
        'answers':{'path':str(context.generation/'answers.json'),'sha256':file_sha256(context.generation/'answers.json')},
        'compiled_features':[line.split('=',1)[0] for line in flags],
        'changes':{'stellar_evolution':answers['stellar_evolution']},
        'model':None if model is None else {'schema':model['schema'],'table':model['table'],
            'prompt_return_fraction':model['request']['prompt_return_fraction']},
        'scope':'Explicit source branch answers; all other baseline answers retained. No evolution or physical acceptance established.',
        'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(context.generation/'branch.json',result)
    return result


def validate_stellar_history_stage(parameters):
    from .stars import validate_native_hydro_tolerances
    snapshots=parameters['snapshots']
    if not isinstance(snapshots,list) or len(snapshots)<2 or any(type(value) is not int or value<0 for value in snapshots) or len(set(snapshots))!=len(snapshots):
        raise ValueError('At least two distinct actual native stellar history snapshots required')
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Positive native stellar history allocation required')
    validate_native_hydro_tolerances(parameters['tolerances'])


def stellar_history_runner(parameters,context):
    """Execute and independently replay the existing hydro/source audit."""
    from .stars import audit_native_hydro_history,verify_native_hydro_history
    validate_stellar_history_stage(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:
        raise MemoryError('Native source history exceeds stage allocation')
    request={'run_directory':str(Path(parameters['run_dir']).resolve()),
        **{key:copy.deepcopy(parameters[key]) for key in ('snapshots','memory_gib','tolerances')}}
    destination=context.generation/'hydro-history.json'
    measured=audit_native_hydro_history(request,destination)
    replay=verify_native_hydro_history(destination)
    if replay!=measured:
        raise ValueError('Native stellar/hydro history changed during independent replay')
    if measured['status']!='PASS':
        raise ValueError('Native stellar/hydro accounting failed; hydro-history.json retains actual ledgers and source-state mismatches')
    return {'status':'PASS','science_acceptance':'NOT_ACCEPTED',
        'scope':'Native journal/cumulative state accounting; physical model validity and resolution remain separate'}
