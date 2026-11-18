"""Stars: consolidated scientific implementation."""
from __future__ import annotations


def validate_ssp_birth_arrays(arrays, *, factor_sn, actual_native_time, comoving):
    """Audit native SH03 gross/retained birth mass without inventing legacy SSPs."""
    import numpy as np
    from core.data import exact_uint64_ids
    required={'ParticleIDs','SSPParentGasID','SSPInitialMass','SSPInitialLockedMass',
        'SSPPromptReturnedMass','SSPCurrentMass','SSPBirthScaleFactor','SSPBirthNativeTime',
        'SSPBirthMetallicity','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter'}
    if set(arrays)!=required: raise ValueError('Declare every native SSP birth field exactly')
    if (isinstance(factor_sn,bool) or not np.isfinite(factor_sn) or not 0<factor_sn<1
            or not np.isfinite(actual_native_time) or actual_native_time<0 or type(comoving) is not bool
            or (comoving and actual_native_time<=0)):
        raise ValueError('Explicit SH03 beta, actual time and integration mode required')
    out={name:np.asarray(value) for name,value in arrays.items()}
    n=len(out['ParticleIDs'])
    if any(a.shape!=(n,) for a in out.values()):raise ValueError('SSP fields must align exactly')
    for name in ('ParticleIDs','SSPParentGasID','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter'):
        if out[name].dtype!=np.dtype('uint64'):raise ValueError('SSP identities/counters require native uint64')
    exact_uint64_ids(out['ParticleIDs'])
    if len(np.unique(out['ParticleIDs']))!=n:raise ValueError('Duplicate native SSP identity')
    # Current gravity mass is measured even when legacy birth state is absent.
    current_all=out['SSPCurrentMass']
    if current_all.dtype.kind!='f' or not np.isfinite(current_all).all() or np.any(current_all<0):
        raise ValueError('Invalid measured SSP current mass, including legacy particles')
    supported=out['SSPBirthStateVersion']==1
    if np.any(out['SSPBirthStateVersion']>1):raise ValueError('Unsupported SSP birth-state version')
    f={k:v[supported] for k,v in out.items()}
    names=required-{'ParticleIDs','SSPParentGasID','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter'}
    if any(not np.isfinite(f[k]).all() for k in names):raise ValueError('Nonfinite supported SSP birth state')
    gross,locked,prompt,current=(f[k] for k in ('SSPInitialMass','SSPInitialLockedMass','SSPPromptReturnedMass','SSPCurrentMass'))
    tolerance=128*np.finfo(float).eps*np.maximum(gross,1e-300)
    if (np.any(gross<=0) or np.any(locked<=0) or np.any(prompt<0) or np.any(current<0)
            or np.any(current>locked+tolerance)
            or np.any(abs(gross-locked-prompt)>tolerance)
            or np.any(abs(locked-(1-factor_sn)*gross)>tolerance)):
        raise ValueError('Inconsistent SH03 initial/current/prompt mass accounting')
    birth=f['SSPBirthNativeTime']; a=f['SSPBirthScaleFactor']; z=f['SSPBirthMetallicity']
    if (np.any(birth<0) or np.any(birth>actual_native_time)
            or np.any((z<0)|(z>1)) or np.any(~np.isin(f['SSPBirthMode'],[1,2]))
            or (comoving and (np.any(a<=0) or np.any(a!=birth)))
            or (not comoving and np.any(a!=0))):
        raise ValueError('Inconsistent SSP birth epoch, metallicity or mode')
    converted=f['SSPBirthMode']==1
    if np.any(f['ParticleIDs'][converted]!=f['SSPParentGasID'][converted]):
        raise ValueError('Whole-gas conversion must preserve exact parent identity')
    if np.any(f['ParticleIDs'][~converted]==f['SSPParentGasID'][~converted]):
        raise ValueError('Spawned SSP must have its independently allocated identity')
    return {'supported':supported,'supported_count':int(supported.sum()),
        'legacy_birth_unavailable_count':int((~supported).sum()),
        'mass_semantics':'InitialMass is effective gross SH03 formation mass; InitialLockedMass is retained birth gravity mass; PromptReturnedMass is already-recycled model bookkeeping and must not be deposited again',
        'science_acceptance':'NOT_ACCEPTED'}


def audit_native_ssp_birth_state(request, destination):
    """Public, source-bound audit of actual native SSP snapshot fields."""
    import json
    import h5py
    import numpy as np
    from pathlib import Path
    from core.foundation import file_sha256 as digest, MSUN_G, product_provenance
    from core.data import (verify_native_hdf5_set, snapshot_id_shards, native_publication_markers,
        run_epoch_contract, export_numeric_audit, coordinate_contract)
    from halos import _snapshot_files
    if set(request)!={'run_directory','snapshot','memory_gib'} or type(request['snapshot']) is not int or request['snapshot']<0:
        raise ValueError('Declare run_directory, exact snapshot index and memory_gib')
    budget=request['memory_gib']
    if isinstance(budget,bool) or not np.isfinite(budget) or budget<=0:raise ValueError('Positive finite SSP audit memory required')
    root=Path(request['run_directory']).resolve(); files=_snapshot_files(root,request['snapshot'])
    verify_native_hdf5_set(files,require_publication=True)
    sources={str(p.resolve()):digest(p) for p in files+native_publication_markers(files)}
    for p in (Path(__file__),root/'config.yaml',root/'manifest.json'):
        if p.is_file():sources[str(p.resolve())]=digest(p)
    algorithm=Path(__file__).resolve().parents[1]/'configs/hydro_audit.algorithms.json'
    sources[str(algorithm)]=digest(algorithm)
    records,n=snapshot_id_shards(files,4)
    if n*512+16*1024**2>budget*1024**3:raise MemoryError('SSP state and uniqueness work exceed declared memory')
    with h5py.File(files[0],'r') as f:
        params=f['Parameters'].attrs; time=float(f['Header'].attrs['Time'])
        if params.get('ComovingIntegrationOn')!=1:raise ValueError('This SSP audit requires the cosmological proper-time contract')
        h=float(params['HubbleParam']); mass_unit=float(params['UnitMass_in_g'])
        beta=float(params['FactorSN'])
    mass_factor=mass_unit/MSUN_G/h
    if not np.isfinite([h,mass_unit,mass_factor]).all() or min(h,mass_unit,mass_factor)<=0:raise ValueError('Explicit native mass unit and h required')
    epoch=run_epoch_contract(root,request['snapshot'],time)
    clock=epoch.get('cosmic_time_source',{})
    # The epoch contract carries its hashed clock source; additionally bind its
    # file when available rather than trusting an unsealed timestamp.
    if isinstance(clock,dict) and clock.get('path'):
        p=Path(clock['path']);sources[str(p.resolve())]=digest(p)
    integer={'ParticleIDs','SSPParentGasID','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter'}
    floating={'SSPInitialMass','SSPInitialLockedMass','SSPPromptReturnedMass','SSPCurrentMass',
        'SSPBirthScaleFactor','SSPBirthNativeTime','SSPBirthMetallicity'}
    arrays={k:np.empty(n,dtype=np.uint64 if k in integer else np.float64) for k in integer|floating}
    for record in records:
        if not record['count']:continue
        with h5py.File(record['path'],'r') as f:
            p=f['Parameters'].attrs
            if any(float(p[k])!=v for k,v in [('HubbleParam',h),('UnitMass_in_g',mass_unit),('FactorSN',beta)]) or p.get('ComovingIntegrationOn')!=1:
                raise ValueError('SSP shard physical parameters differ')
            group=f['PartType4']; selection=slice(record['start'],record['stop'])
            for k in arrays:
                if k not in group or group[k].shape!=(record['count'],):raise ValueError('Missing or misaligned native SSP field: '+k)
                if k in integer and group[k].dtype!=np.dtype('uint64'):raise ValueError('Lossy native SSP integer field: '+k)
                if k!='ParticleIDs':
                    expected={'to_cgs':mass_unit if k.endswith('Mass') else 1.,
                              'a_scaling':0.,'h_scaling':-1. if k.endswith('Mass') else 0.}
                    if any(key not in group[k].attrs or float(group[k].attrs[key])!=value for key,value in expected.items()):
                        raise ValueError('Inconsistent declared native SSP units: '+k)
                arrays[k][selection]=group[k][:]
            native_mass=np.asarray(group['Masses'][:] if 'Masses' in group else
                np.full(record['count'],f['Header'].attrs['MassTable'][4]))
            if not np.isfinite(native_mass).all() or np.any(native_mass<0):
                raise ValueError('Invalid SSP native gravity mass')
            precision=np.finfo(native_mass.dtype).eps if native_mass.dtype.kind=='f' else np.finfo(float).eps
            if np.any(abs(arrays['SSPCurrentMass'][selection]-native_mass)>4*precision*np.maximum(abs(native_mass),1e-300)):
                raise ValueError('SSP current mass disagrees with native gravity mass')
    audit=validate_ssp_birth_arrays(arrays,factor_sn=beta,actual_native_time=time,comoving=True)
    arrays['birth_supported']=audit.pop('supported')
    for k in floating:
        if k.endswith('Mass'):arrays[k]*=mass_factor
    specs={k:{'units':'Msun' if k.endswith('Mass') else 'dimensionless scale factor' if k in {'SSPBirthScaleFactor','SSPBirthNativeTime'}
        else 'metal mass fraction' if k=='SSPBirthMetallicity' else 'dimensionless',
        'quantity':'native SH03 SSP state: '+k,
        'missing_policy':'birth_supported=false means unavailable legacy birth; retained zeros are not inferred births'} for k in arrays}
    provenance=product_provenance(root,algorithm=__file__,selection='Complete native type-4 SSP birth-state census',
        source_files=files,coordinates=coordinate_contract(position_units='proper kpc',
        origin='no spatial quantities in this SSP state census',periodic_wrapping='not applicable',
        velocity_convention='not applicable'),snapshot=request['snapshot'])
    for k in ('ParticleIDs','SSPParentGasID'):
        specs[k]['identity_namespace']=provenance['identity_namespaces']['native_particle']
    for artifact in provenance['algorithm_sources']:
        sources[artifact['path']]=artifact['sha256']
    for artifact in [*provenance['initial_conditions'],provenance['executable'],provenance['parameter_file'],provenance['compile_options'],*provenance['boltzmann_inputs'].values()]:
        if artifact.get('status')=='VERIFIED':sources[artifact['path']]=artifact['sha256']
    report={'schema':'native-ssp-birth-audit-v1','status':'MEASURED','request':request,'epoch':epoch,
        'particle_count':n,**audit,'source_hashes':sources,'provenance':provenance,
        'source_origin':'synthetic_control' if (root/'fixture_manifest.json').is_file() or
            (root/'manifest.json').is_file() and json.loads((root/'manifest.json').read_text()).get('fixture') else 'native_snapshot'}
    for path,checksum in sources.items():
        if digest(path)!=checksum:raise RuntimeError('SSP source changed during audit: '+path)
    return export_numeric_audit(destination,report,arrays,specs,
        algorithm=algorithm,
        context={'epoch':epoch,'producer':'native SSP snapshot birth-state audit','provenance':provenance},source_hashes=sources)


def verify_native_ssp_birth_audit(path):
    from core.data import verify_numeric_audit
    from core.foundation import file_sha256 as digest, verify_product_provenance
    report=verify_numeric_audit(path)
    if report.get('schema')!='native-ssp-birth-audit-v1':raise ValueError('Not a native SSP birth audit')
    for source,checksum in report['source_hashes'].items():
        if digest(source)!=checksum:raise ValueError('Native SSP audit source changed: '+source)
    verify_product_provenance(report['request']['run_directory'],report['provenance'])
    return report

SSP_BIRTH_EVENT_COLUMNS = ('ssp_id','parent_id','mode','opportunity_counter',
    'birth_time','scale_factor','metallicity','gross_mass','locked_mass','prompt_mass',
    'gas_mass_before','gas_mass_after','gas_v_before_x','gas_v_before_y','gas_v_before_z',
    'gas_v_after_x','gas_v_after_y','gas_v_after_z','ssp_v_x','ssp_v_y','ssp_v_z')


def read_native_ssp_birth_journal(directory, *, start_commit, stop_commit, start_count, stop_count,
                                  native_time_interval, expected_parameters, memory_bytes):
    """Read only the interval certified by two complete snapshot counter sets.

    A branch may inherit a positive counter prefix without copying old journal
    chunks. Every commit inside the selected interval is mandatory. Orphaned
    rank/staging files do not count as committed events.
    """
    import csv,json,re,zlib
    from pathlib import Path
    import numpy as np
    from core.foundation import file_sha256 as digest
    uint=lambda x:type(x) is int and 0<=x<2**64
    if (not all(uint(x) for x in (start_commit,stop_commit,start_count,stop_count))
            or stop_commit<start_commit or stop_count<start_count
            or stop_commit-start_commit>stop_count-start_count):
        raise ValueError('Inconsistent authoritative SSP journal counter interval')
    if type(memory_bytes) is not int or memory_bytes<1:
        raise ValueError('Declare positive SSP journal workspace')
    model_keys={'unit_mass_g','unit_time_s','unit_velocity_cm_s','unit_length_cm','hubble_param','factor_sn','comoving','seed'}
    if not isinstance(expected_parameters,dict) or set(expected_parameters)!=model_keys:
        raise ValueError('Declare the complete native SSP model and units')
    count=stop_count-start_count
    if count*4096+16*1024**2>memory_bytes:raise MemoryError('SSP event arrays exceed declared workspace')
    bounds=np.asarray(native_time_interval,dtype=float)
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]<0 or bounds[1]<=bounds[0]:
        raise ValueError('SSP interval requires increasing actual native times')
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('Duplicate SSP commit JSON key: '+key)
            out[key]=value
        return out
    directory=Path(directory).resolve();sources={};rows=[];commits=[];previous=start_count;fingerprint=None;last_tick=None
    keys={'schema','sequence','previous_event_count','event_count','cumulative_event_count','ti_current',
        'native_time','comoving','seed','source_fingerprint','unit_mass_g','unit_time_s','unit_velocity_cm_s',
        'unit_length_cm','hubble_param','factor_sn','velocity_convention','ranks'}
    for sequence in range(start_commit,stop_commit):
        path=directory/f'ssp_birth_{sequence:020d}.commit.json'
        if path.is_symlink() or not path.is_file():raise ValueError('Missing or nonregular SSP commit: '+str(path))
        if path.stat().st_size>memory_bytes//16:raise MemoryError('SSP commit metadata exceeds workspace')
        checksum=digest(path);commit=json.loads(path.read_text(),object_pairs_hook=unique)
        if set(commit)!=keys or commit['schema']!='native-ssp-birth-journal-v1':raise ValueError('Unsupported SSP journal schema')
        if (not all(uint(commit[k]) for k in ('sequence','previous_event_count','event_count','cumulative_event_count','ti_current'))
                or commit['sequence']!=sequence or commit['previous_event_count']!=previous
                or commit['event_count']<=0 or commit['cumulative_event_count']!=previous+commit['event_count']
                or commit['cumulative_event_count']>stop_count):
            raise ValueError('SSP commit count chain is incomplete or inconsistent')
        time=commit['native_time']
        if isinstance(time,bool) or not isinstance(time,(int,float)) or not np.isfinite(time) or not bounds[0]<=time<=bounds[1]:
            raise ValueError('SSP commit lies outside the actual epoch interval')
        if last_tick is not None and commit['ti_current']<=last_tick:raise ValueError('SSP commits do not follow increasing native ticks')
        last_tick=commit['ti_current']
        fp=commit['source_fingerprint']
        if not isinstance(fp,str) or not re.fullmatch('[0-9a-f]{64}',fp):raise ValueError('Missing native source fingerprint')
        if fingerprint is not None and fp!=fingerprint:raise ValueError('SSP interval mixes source builds')
        fingerprint=fp
        if any(isinstance(commit[k],bool) or not isinstance(commit[k],(int,float)) or not np.isfinite(commit[k]) or commit[k]<=0
               for k in model_keys-{'comoving','seed'}) or commit['factor_sn']>=1:
            raise ValueError('Nonfinite or invalid SSP journal model/units')
        if (type(commit['comoving']) is not int or commit['comoving'] not in (0,1)
                or not isinstance(commit['seed'],str) or not re.fullmatch('[0-9]+',commit['seed'])
                or int(commit['seed'])>=2**64):raise ValueError('Invalid exact SSP model/seed metadata')
        if commit['comoving'] and time<=0:raise ValueError('Comoving SSP event requires a positive scale factor')
        for key,value in expected_parameters.items():
            if key not in commit or commit[key]!=value or isinstance(commit[key],bool):
                raise ValueError('SSP journal differs from snapshot physical parameter: '+key)
        if commit['velocity_convention']!='native canonical P.Vel; peculiar=P.Vel/a when comoving':
            raise ValueError('Unknown native SSP velocity convention')
        ranks=commit['ranks']
        if not isinstance(ranks,list) or not ranks:raise ValueError('SSP commit requires the complete rank census')
        local_total=0
        for rank,record in enumerate(ranks):
            if (not isinstance(record,dict) or set(record)!={'rank','count','bytes','crc32','name'}
                    or type(record['rank']) is not int or record['rank']!=rank
                    or not all(uint(record[k]) for k in ('count','bytes','crc32')) or record['crc32']>=2**32):
                raise ValueError('Invalid SSP rank descriptor')
            local_total+=record['count']
            if record['count']==0:
                if record['name'] is not None or record['bytes']!=0 or record['crc32']!=0:raise ValueError('Invalid empty SSP rank descriptor')
                continue
            name=f'ssp_birth_{sequence:020d}.rank_{rank:06d}.of_{len(ranks):06d}.tsv'
            if record['name']!=name:raise ValueError('SSP rank filename does not match its commit')
            chunk=directory/name
            if chunk.is_symlink() or not chunk.is_file() or chunk.stat().st_size!=record['bytes']:
                raise ValueError('Missing or changed committed SSP rank chunk')
            # Streaming rows have bounded length. No whole-chunk JSON/body copy.
            crc=0
            with chunk.open('rb') as f:
                while block:=f.read(65536):crc=zlib.crc32(block,crc)
            if crc!=record['crc32']:raise ValueError('SSP rank CRC mismatch')
            chunk_hash=digest(chunk);local_rows=0
            with chunk.open('r',newline='') as f:
                header=f.readline(4096)
                if header.rstrip('\n').split('\t')!=list(SSP_BIRTH_EVENT_COLUMNS):raise ValueError('Unexpected SSP rank columns')
                while line:=f.readline(4096):
                    if not line.endswith('\n'):raise ValueError('Oversized or incomplete SSP event row')
                    cells=next(csv.reader([line],delimiter='\t'))
                    if len(cells)!=len(SSP_BIRTH_EVENT_COLUMNS):raise ValueError('Misaligned SSP event row')
                    if any(not re.fullmatch('[0-9]+',x) or int(x)>=2**64 for x in cells[:4]):
                        raise ValueError('Lossy SSP event identity/counter')
                    integers=[int(x) for x in cells[:4]];values=np.asarray(cells[4:],dtype=float)
                    if not np.isfinite(values).all():raise ValueError('Nonfinite SSP source event')
                    row=dict(zip(SSP_BIRTH_EVENT_COLUMNS,[*integers,*values.tolist()]))
                    if (row['mode'] not in (1,2) or row['opportunity_counter']==0
                            or row['birth_time']!=time or row['scale_factor']!=(time if commit['comoving'] else 0.)
                            or not 0<=row['metallicity']<=1 or min(row['gross_mass'],row['locked_mass'],row['gas_mass_before'])<=0
                            or min(row['prompt_mass'],row['gas_mass_after'])<0
                            or (row['mode']==1 and (row['ssp_id']!=row['parent_id'] or row['gas_mass_after']!=0))
                            or (row['mode']==2 and (row['ssp_id']==row['parent_id'] or row['gas_mass_after']<=0))):
                        raise ValueError('Inconsistent native SSP birth event state')
                    tolerance=128*np.finfo(float).eps*row['gross_mass']
                    if (abs(row['gross_mass']-row['locked_mass']-row['prompt_mass'])>tolerance
                            or abs(row['locked_mass']-(1-commit['factor_sn'])*row['gross_mass'])>tolerance):
                        raise ValueError('Inconsistent SH03 event mass bookkeeping')
                    if len(rows)>=count or local_rows>=record['count']:raise ValueError('SSP journal has undeclared extra events')
                    rows.append(row);local_rows+=1
            if local_rows!=record['count'] or digest(chunk)!=chunk_hash:raise ValueError('SSP rank row count or source changed')
            sources[str(chunk)]=chunk_hash
        if local_total!=commit['event_count'] or digest(path)!=checksum:raise ValueError('SSP commit rank census or source changed')
        sources[str(path)]=checksum;previous=commit['cumulative_event_count'];commits.append(commit)
    if previous!=stop_count or len(rows)!=count:raise ValueError('Missing SSP birth history in selected interval')
    if len({r['ssp_id'] for r in rows})!=len(rows):raise ValueError('SSP identity born more than once')
    arrays={name:np.asarray([row[name] for row in rows],dtype=np.uint64 if i<4 else np.float64)
            for i,name in enumerate(SSP_BIRTH_EVENT_COLUMNS)}
    return arrays,commits,sources


def native_ssp_event_physical_arrays(arrays,parameters):
    """Convert each recorded source row at its own epoch, preserving integers."""
    from core.foundation import MSUN_G
    from core.data import native_physical_unit_scales
    converted={k:v.copy() for k,v in arrays.items()}
    scales=native_physical_unit_scales(unit_mass_g=parameters['unit_mass_g'],unit_length_cm=parameters['unit_length_cm'],
        unit_velocity_cm_s=parameters['unit_velocity_cm_s'],comoving=parameters['comoving'],hubble_param=parameters['hubble_param'])
    mass_factor=scales['mass_g']/MSUN_G
    velocity_factor=parameters['unit_velocity_cm_s']/1.e5
    if parameters['comoving']:velocity_factor=velocity_factor/arrays['scale_factor']
    converted['source_mass_residual_msun']=(arrays['gas_mass_before']-arrays['gas_mass_after']-arrays['locked_mass'])*mass_factor
    for k in 'xyz':
        converted['source_momentum_residual_'+k]=(arrays['gas_mass_before']*arrays['gas_v_before_'+k]-
            arrays['gas_mass_after']*arrays['gas_v_after_'+k]-arrays['locked_mass']*arrays['ssp_v_'+k])*mass_factor*velocity_factor
    for name in ('gross_mass','locked_mass','prompt_mass','gas_mass_before','gas_mass_after'):converted[name]*=mass_factor
    for name in SSP_BIRTH_EVENT_COLUMNS[12:]:converted[name]*=velocity_factor
    return converted


def audit_native_ssp_birth_events(request,destination):
    """Publish source events for two actual outputs, with measured residuals.

    Completeness is tied to snapshot counters. This does not yet reconcile the
    full six-species census or accept the evolved thermodynamic/SF campaign.
    """
    import json
    from pathlib import Path
    import numpy as np,h5py
    from halos import _snapshot_files
    from core.data import verify_native_hdf5_set,export_numeric_audit,run_epoch_contract,isolated_epoch_contract,coordinate_contract
    from core.foundation import file_sha256 as digest, MSUN_G, product_provenance
    if not isinstance(request,dict) or set(request)!={'run_directory','snapshots','memory_gib'}:
        raise ValueError('Declare run directory, two snapshot indices and event workspace')
    selected=request['snapshots']
    if (not isinstance(selected,list) or len(selected)!=2 or selected[0]==selected[1]
            or any(type(x) is not int or x<0 for x in selected)):
        raise ValueError('Select two distinct native snapshot indices')
    memory=request['memory_gib']
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not np.isfinite(memory) or memory<=0:
        raise ValueError('Declare positive SSP event memory_gib')
    root=Path(request['run_directory']).resolve();sources={};headers=[];files=[]
    names=('StellarBirthEventCount','StellarBirthCommitCount','StellarBirthJournalSchema')
    parameter_names={'unit_mass_g':'UnitMass_in_g','unit_time_s':'UnitTime_in_s',
        'unit_velocity_cm_s':'UnitVelocity_in_cm_per_s','unit_length_cm':'UnitLength_in_cm',
        'hubble_param':'HubbleParam','factor_sn':'FactorSN','comoving':'ComovingIntegrationOn','seed':'StellarRandomSeed'}
    reference=None
    for index in selected:
        paths=_snapshot_files(root,index);verify_native_hdf5_set(paths,require_publication=True)
        if not paths:raise ValueError('Missing complete native SSP epoch')
        current=None
        for path in paths:
            checksum=digest(path)
            with h5py.File(path,'r') as f:
                h=f['Header'].attrs;p=f['Parameters'].attrs
                if any(name not in h or np.asarray(h[name]).shape!=() or np.asarray(h[name]).dtype!=np.dtype('uint64') for name in names):
                    raise ValueError('Native SSP output lacks exact journal counters')
                counts=tuple(int(h[name]) for name in names)
                time=float(h['Time'])
                if counts[2]!=1 or counts[1]>counts[0]:raise ValueError('Unsupported SSP snapshot journal counters')
                from core.data import native_time_unit_seconds
                parameters={key:p[name] for key,name in parameter_names.items() if key!='unit_time_s'}
                parameters['unit_time_s']=native_time_unit_seconds(p)
                seed=parameters['seed']
                if isinstance(seed,bytes):seed=seed.decode('ascii')
                if not isinstance(seed,str) or not seed.isdecimal() or int(seed)>=2**64:raise ValueError('Exact native SSP seed required')
                parameters['seed']=str(int(seed))
                mode=parameters['comoving']
                if np.asarray(mode).dtype.kind not in 'iu' or int(mode) not in (0,1):raise ValueError('Explicit native integration mode required')
                parameters['comoving']=int(mode)
                if not np.isfinite(time) or time<0 or (int(mode) and time<=0):raise ValueError('Invalid actual SSP snapshot clock')
                for name in set(parameters)-{'seed','comoving'}:
                    parameters[name]=float(parameters[name])
                    if not np.isfinite(parameters[name]) or parameters[name]<=0:raise ValueError('Invalid native SSP parameter: '+name)
                if parameters['factor_sn']>=1:raise ValueError('Invalid SH03 recycling fraction')
                if current is not None and current!=(counts,time):raise ValueError('SSP snapshot shards disagree about journal history')
                current=(counts,time)
                if reference is not None and parameters!=reference:raise ValueError('SSP event interval changes physical units/model/seed')
                reference=parameters
            if digest(path)!=checksum:raise ValueError('SSP snapshot changed during event audit')
            sources[str(Path(path).resolve())]=checksum;files.append(path)
        headers.append(current)
    arrays,commits,hashes=read_native_ssp_birth_journal(root/'data',start_commit=headers[0][0][1],stop_commit=headers[1][0][1],
        start_count=headers[0][0][0],stop_count=headers[1][0][0],native_time_interval=[h[1] for h in headers],
        expected_parameters=reference,memory_bytes=int(memory*1024**3))
    sources.update(hashes)
    arrays=native_ssp_event_physical_arrays(arrays,reference)
    epoch=run_epoch_contract(root,selected[-1],headers[-1][1]) if reference['comoving'] else isolated_epoch_contract(headers[-1][1],snapshot=selected[-1])
    coordinates=coordinate_contract(position_units='proper kpc',origin='no positions in source event journal',
        periodic_wrapping='not applicable',velocity_convention='peculiar km/s from recorded canonical native velocities')
    provenance=product_provenance(root,algorithm=__file__,selection='Complete SSP birth event interval certified by native snapshot counters',
        source_files=files,coordinates=coordinates,snapshot=selected[-1])
    namespace=provenance['identity_namespaces']['native_particle']
    specs={name:{'units':'Msun km/s' if name.startswith('source_momentum_') else 'Msun' if 'mass' in name or name.endswith('_msun')
        else 'km/s' if '_v_' in name else 'native integration time' if name=='birth_time'
        else 'metal mass fraction' if name=='metallicity' else 'dimensionless',
        'quantity':name.replace('_',' ')} for name in arrays}
    for name in ('ssp_id','parent_id'):specs[name]['identity_namespace']=namespace
    event_epoch_axis={'axis':0,'time_dataset':'birth_time',
        'time_convention':'scale_factor' if reference['comoving'] else 'isolated_native_elapsed_time',
        'audit_scope_epoch_role':'upper interval boundary; event rows are measured at birth_time'}
    for spec in specs.values():spec['annotations']={'event_epoch_axis':event_epoch_axis}
    algorithm=Path(__file__).resolve().parents[1]/'configs/hydro_audit.algorithms.json'
    sources[str(algorithm)]=digest(algorithm);sources[str(Path(__file__).resolve())]=digest(__file__)
    for artifact in provenance['algorithm_sources']:
        sources[artifact['path']]=artifact['sha256']
    report={'schema':'native-ssp-birth-events-audit-v1','status':'MEASURED','request':request,'epoch':epoch,
        'provenance':provenance,'event_count':len(arrays['ssp_id']),'commit_count':len(commits),
        'counter_interval':{'start':list(headers[0][0][:2]),'stop':list(headers[1][0][:2])},
        'actual_native_time_interval':[h[1] for h in headers],'native_build_fingerprints':sorted({c['source_fingerprint'] for c in commits}),
        'source_hashes':sources,'physical_reconciliation':'PENDING complete gas/star/BH census and frozen dimensional conservation tolerances',
        'event_epoch_axis':event_epoch_axis,
        'prompt_recycling':'Virtual SH03 bookkeeping; not an additional native gas deposition',
        'zero_gas_after_velocity':'Zero vector denotes no remaining gas for whole conversion; momentum contribution is zero',
        'source_origin':'synthetic_control' if (root/'fixture_manifest.json').is_file() or
            (root/'manifest.json').is_file() and json.loads((root/'manifest.json').read_text()).get('fixture') else 'native_snapshot_and_operator_journal'}
    for source,checksum in sources.items():
        if digest(source)!=checksum:raise ValueError('Native SSP event source changed during audit')
    return export_numeric_audit(destination,report,arrays,specs,algorithm=algorithm,
        context={'epoch':epoch,'provenance':provenance,'coordinates':coordinates},source_hashes=sources)


def verify_native_ssp_birth_events(path):
    from core.data import verify_numeric_audit
    from core.foundation import file_sha256 as digest, verify_product_provenance
    report=verify_numeric_audit(path)
    if report.get('schema')!='native-ssp-birth-events-audit-v1':raise ValueError('Not a native SSP event product')
    for source,checksum in report['source_hashes'].items():
        if digest(source)!=checksum:raise ValueError('Native SSP event source changed: '+source)
    verify_product_provenance(report['request']['run_directory'],report['provenance'])
    return report

def validate_native_hydro_tolerances(tolerances):
    """Freeze dimensional floors before reading or evolving scientific data."""
    import numpy as np
    required={'mass_relative','mass_absolute_msun','momentum_relative','momentum_absolute_msun_km_s'}
    if not isinstance(tolerances,dict) or set(tolerances)!=required:
        raise ValueError('Freeze complete dimensional mass and momentum tolerances')
    for name,value in tolerances.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0 or ('absolute' in name and value==0):
            raise ValueError('Invalid frozen hydro tolerance: '+name)
    return dict(tolerances)


def reconcile_native_hydro_species(before,after,events,tolerances,source_records=(),bh_records=()):
    """Replay native birth operations and account for every live union ID.

    Later source operators must supply their own event ledgers; unexplained
    stellar return, BH growth/mergers or disappearance cannot pass this audit.
    Velocities between outputs are not compared: gravity/hydro act there.
    Momentum conservation is measured at each recorded instantaneous source.
    """
    import numpy as np
    from core.data import exact_uint64_ids
    tolerances=validate_native_hydro_tolerances(tolerances)
    def census(data):
        if np.asarray(data['particle_ids']).dtype!=np.dtype('uint64'):
            raise ValueError('Native hydro identities must already be uint64')
        ids=exact_uint64_ids(data['particle_ids']);types=np.asarray(data['species']);mass=np.asarray(data['mass_msun'])
        if (ids.dtype!=np.dtype('uint64') or types.shape!=ids.shape or types.dtype.kind not in 'iu'
                or mass.shape!=ids.shape or mass.dtype.kind!='f' or not np.isfinite(mass).all()
                or np.any((types<0)|(types>5)) or np.any(mass<0) or len(np.unique(ids))!=len(ids)):
            raise ValueError('Complete hydro census requires unique exact IDs, six species and finite masses')
        return {int(pid):(int(kind),float(m)) for pid,kind,m in zip(ids,types,mass)}
    initial=census(before);observed=census(after);expected=dict(initial);issues=[]
    if not set(SSP_BIRTH_EVENT_COLUMNS)<=set(events):raise ValueError('Incomplete native source event columns')
    n=len(events['ssp_id'])
    if any(np.asarray(v).shape!=(n,) for v in events.values()):raise ValueError('Source event columns must align')
    for name in ('ssp_id','parent_id','mode','opportunity_counter'):
        if events[name].dtype!=np.dtype('uint64'):raise ValueError('Native source events require exact uint64 IDs/counters')
    for name in SSP_BIRTH_EVENT_COLUMNS[4:]:
        if np.asarray(events[name]).dtype.kind!='f' or not np.isfinite(events[name]).all():
            raise ValueError('Source measurements require finite floating values: '+name)
    if (np.any(events['birth_time']<0) or np.any(events['scale_factor']<0)
            or np.any((events['metallicity']<0)|(events['metallicity']>1))
            or np.any(events['gross_mass']<=0) or np.any(events['prompt_mass']<0)):
        raise ValueError('Invalid native source birth state')
    if len(np.unique(events['ssp_id']))!=n:raise ValueError('Duplicate source birth identity')
    def close(x,y):
        return abs(x-y)<=tolerances['mass_absolute_msun']+tolerances['mass_relative']*max(abs(x),abs(y))
    opportunities={}
    operations=[(float(time),0,0,0,'birth',int(i)) for i,time in enumerate(events['birth_time'])]
    operations.extend((float(row['native_time']),1,int(row['source_id']),int(row['source_counter']),'return',i)
        for i,row in enumerate(source_records))
    operations.extend((float(row['native_time']),
        2 if row['kind'] in ('seed_donor','seed_assembly','seed') else 4 if row['kind']=='merger' else 3,
        int(row.get('transaction_index',row.get('source_id',row['particle_id']))),
        1 if row['kind']=='seed_assembly' else 0 if row['kind']=='seed_donor' else int(row['counter']),
        'bh',i) for i,row in enumerate(bh_records))
    bh_groups={}
    seed_groups={}
    source_groups={}
    for _,_,_,_,operation,i in sorted(operations):
        if operation=='bh':
            row=bh_records[i];pid=row['particle_id'];kind=row['kind']
            def bh_issue(reason):issues.append({'kind':reason,'particle_id':pid,'bh_event_row':i})
            if kind in ('seed_donor','seed_assembly'):
                pre=row['mass_before_msun'];post=row['mass_after_msun']
                if pid not in expected or expected[pid][0]!=0:
                    bh_issue('BH_SEED_PARTICLE_NOT_LIVE_GAS');continue
                if not close(expected[pid][1],pre):bh_issue('BH_SEED_MASS_DOES_NOT_MATCH_CENSUS_HISTORY')
                expected[pid]=(0 if kind=='seed_donor' else 5,post)
                seed_groups.setdefault((row['native_time'],row.get('source_id',pid),row['counter']),[]).append(row)
            elif kind=='seed':
                if pid not in expected or expected[pid][0]!=row['previous_species']:
                    bh_issue('BH_SEED_PARENT_NOT_LIVE_EXPECTED_SPECIES');continue
                if not close(expected[pid][1],row['gravity_mass_msun']):bh_issue('BH_SEED_GRAVITY_MASS_MISMATCH')
                if not 0<row['physical_mass_msun']<=row['gravity_mass_msun']:raise ValueError('BH physical seed mass exceeds its composite gravity carrier')
                expected[pid]=(5,row['gravity_mass_msun'])
            elif kind in ('gas_accretion','gas_feedback','accretion'):
                species=0 if kind in ('gas_accretion','gas_feedback') else 5;pre=row['mass_before_msun'];post=row['mass_after_msun']
                if pre<=0 or post<=0:raise ValueError('BH source must leave positive live particle masses')
                if pid not in expected or expected[pid][0]!=species:bh_issue('BH_SOURCE_PARTICLE_NOT_LIVE_EXPECTED_SPECIES');continue
                if not close(expected[pid][1],pre):bh_issue('BH_SOURCE_MASS_DOES_NOT_MATCH_CENSUS_HISTORY')
                expected[pid]=(species,post)
                bh_groups.setdefault((row['native_time'],row.get('source_id',pid),row['counter']),[]).append(row)
            elif kind=='accretion_integral':
                if pid not in expected or expected[pid][0]!=5:bh_issue('BH_INTEGRAL_CARRIER_NOT_LIVE')
                bh_groups.setdefault((row['native_time'],pid,row['counter']),[]).append(row)
            elif kind=='accretion_merge':
                if any(parent not in expected or expected[parent][0]!=5 for parent in row['parent_ids']):
                    bh_issue('BH_INTEGRAL_MERGER_PARENT_NOT_LIVE')
            elif kind=='merger':
                parents=row['parent_ids'];pre=row['gravity_mass_before_msun'];post=row['gravity_mass_after_msun'];rad=row['radiated_mass_msun']
                if any(parent not in expected or expected[parent][0]!=5 for parent in parents):bh_issue('BH_MERGER_PARENT_NOT_LIVE');continue
                if any(not close(expected[parent][1],m) for parent,m in zip(parents,pre)):bh_issue('BH_MERGER_PARENT_MASS_MISMATCH')
                if not close(sum(pre),post+rad):bh_issue('BH_MERGER_MASS_RADIATION_NOT_CLOSED')
                if not close(sum(row['physical_mass_before_msun']),row['physical_mass_after_msun']+rad):bh_issue('BH_MERGER_PHYSICAL_MASS_NOT_CLOSED')
                vb=np.asarray(row['velocity_before_km_s']);va=np.asarray(row['velocity_after_km_s']);rp=np.asarray(row['radiated_momentum_msun_km_s'])
                residual=sum(pre)*vb-post*va-rp;scale=sum(pre)*np.linalg.norm(vb)+post*np.linalg.norm(va)+np.linalg.norm(rp)
                if np.linalg.norm(residual)>tolerances['momentum_absolute_msun_km_s']+tolerances['momentum_relative']*scale:bh_issue('BH_MERGER_MOMENTUM_RADIATION_NOT_CLOSED')
                for parent in parents:del expected[parent]
                expected[pid]=(5,post)
            elif kind not in ('friction','binary_capture','binary_hardening'):raise ValueError('Unknown native BH replay operation')
            continue
        if operation=='return':
            row=source_records[i];pid=row['particle_id'];species=row['species'];pre=row['mass_before_msun'];post=row['mass_after_msun']
            if type(pid) is not int or not 0<=pid<2**64 or species not in (0,4) or not np.isfinite([pre,post]).all() or pre<=0 or post<=0:
                raise ValueError('Invalid native stellar return identity/masses')
            if pid not in expected or expected[pid][0]!=species:
                issues.append({'kind':'RETURN_PARTICLE_NOT_LIVE_EXPECTED_SPECIES','particle_id':pid,'source_row':i})
                continue
            if not close(expected[pid][1],pre):
                issues.append({'kind':'RETURN_MASS_DOES_NOT_MATCH_CENSUS_HISTORY','particle_id':pid,'source_row':i})
            expected[pid]=(species,post)
            source_groups.setdefault((row['native_time'],row['source_id'],row['source_counter']),[]).append(row)
            continue
        child=int(events['ssp_id'][i]);parent=int(events['parent_id'][i]);mode=int(events['mode'][i]);counter=int(events['opportunity_counter'][i])
        def issue(reason):issues.append({'kind':reason,'particle_id':child,'parent_id':parent,'event_row':int(i)})
        if mode not in (1,2) or counter<=opportunities.get(parent,0):raise ValueError('Invalid source mode or repeated gas opportunity')
        opportunities[parent]=counter
        masses=[float(events[k][i]) for k in ('gas_mass_before','gas_mass_after','locked_mass')]
        if not np.isfinite(masses).all() or masses[0]<=0 or masses[1]<0 or masses[2]<=0:raise ValueError('Invalid native source masses')
        pre,post,locked=masses
        if not close(float(events['gross_mass'][i]),locked+float(events['prompt_mass'][i])):
            issue('SOURCE_GROSS_PROMPT_BOOKKEEPING_MISMATCH')
        if parent not in expected or expected[parent][0]!=0:
            issue('BIRTH_PARENT_NOT_LIVE_GAS');continue
        if not close(expected[parent][1],pre):issue('SOURCE_PARENT_MASS_DOES_NOT_MATCH_CENSUS_HISTORY')
        if not close(pre,post+locked):issue('SOURCE_MASS_NOT_CONSERVED')
        velocity=np.array([[events[prefix+k][i] for k in 'xyz'] for prefix in ('gas_v_before_','gas_v_after_','ssp_v_')],dtype=float)
        if not np.isfinite(velocity).all():raise ValueError('Nonfinite native source velocity')
        momentum=np.asarray(masses)[:,None]*velocity
        residual=np.linalg.norm(momentum[0]-momentum[1]-momentum[2])
        floor=tolerances['momentum_absolute_msun_km_s']
        scale=sum(np.linalg.norm(p) for p in momentum)
        if residual>floor+tolerances['momentum_relative']*scale:issue('SOURCE_MOMENTUM_NOT_CONSERVED')
        if mode==1:
            if child!=parent or post!=0:raise ValueError('Whole-gas birth must preserve ID and leave no gas')
            expected[parent]=(4,locked)
        else:
            if child==parent or post<=0:raise ValueError('Partial spawn needs a new ID and remaining gas')
            if child in expected:issue('SPAWN_ID_ALREADY_EXISTS');continue
            expected[parent]=(0,post);expected[child]=(4,locked)
    from core.black_hole_ledger import validate_seed_group
    for key,rows in sorted(seed_groups.items()):
        for reason in validate_seed_group(rows,tolerances):
            issues.append({'kind':reason,'particle_id':key[1],'native_time':key[0]})
    for key,rows in sorted(source_groups.items()):
        donors=[row for row in rows if row['species']==4];recipients=[row for row in rows if row['species']==0]
        if len(donors)!=1 or not recipients or donors[0]['particle_id']!=key[1]:
            issues.append({'kind':'INCOMPLETE_RETURN_SOURCE_RECIPIENT_GROUP','particle_id':key[1]});continue
        element_gain=np.zeros(11)
        for row in recipients:
            eb=np.asarray(row['elements_before_msun']);ea=np.asarray(row['elements_after_msun'])
            if eb.shape!=(11,) or ea.shape!=(11,) or not np.isfinite(eb).all() or not np.isfinite(ea).all() or np.any(eb<0) or np.any(ea<0):
                raise ValueError('Invalid stellar return recipient element ledger')
            if not close(float(eb.sum()),row['mass_before_msun']) or not close(float(ea.sum()),row['mass_after_msun']):
                issues.append({'kind':'RETURN_RECIPIENT_ELEMENT_MASS_NOT_CLOSED','particle_id':row['particle_id']})
            element_gain+=ea-eb
        emitted=np.asarray(donors[0]['elements_after_msun'])
        if emitted.shape!=(11,) or not np.isfinite(emitted).all() or np.any(emitted<0):raise ValueError('Invalid SSP ejecta element ledger')
        if any(not close(float(x),float(y)) for x,y in zip(emitted,element_gain)):
            issues.append({'kind':'RETURN_ELEMENTS_NOT_CONSERVED','particle_id':key[1]})
        if not close(float(emitted.sum()),donors[0]['mass_before_msun']-donors[0]['mass_after_msun']):
            issues.append({'kind':'RETURN_SOURCE_ELEMENT_MASS_NOT_CLOSED','particle_id':key[1]})
        mass_change=sum(row['mass_after_msun']-row['mass_before_msun'] for row in rows)
        mass_scale=sum(row['mass_before_msun'] for row in rows)
        if abs(mass_change)>tolerances['mass_absolute_msun']+tolerances['mass_relative']*mass_scale:
            issues.append({'kind':'RETURN_MASS_NOT_CONSERVED','particle_id':key[1]})
        momentum=np.zeros(3);scale=0.
        for row in rows:
            vb=np.asarray(row['velocity_before_km_s']);va=np.asarray(row['velocity_after_km_s'])
            if vb.shape!=(3,) or va.shape!=(3,) or not np.isfinite(vb).all() or not np.isfinite(va).all():raise ValueError('Invalid return source velocity')
            before_momentum=row['mass_before_msun']*vb;after_momentum=row['mass_after_msun']*va
            momentum+=after_momentum-before_momentum;scale+=np.linalg.norm(before_momentum)+np.linalg.norm(after_momentum)
        if np.linalg.norm(momentum)>tolerances['momentum_absolute_msun_km_s']+tolerances['momentum_relative']*scale:
            issues.append({'kind':'RETURN_MOMENTUM_NOT_CONSERVED','particle_id':key[1]})
    import math
    from core.black_hole_ledger import source_mass_change_msun
    for key,rows in sorted(bh_groups.items()):
        sinks=[r for r in rows if r['kind']=='accretion'];gas=[r for r in rows if r['kind']=='gas_accretion']
        recipients=[r for r in rows if r['kind'] in ('gas_accretion','gas_feedback')]
        integrals=[r for r in rows if r['kind']=='accretion_integral']
        if integrals:
            from core.black_hole_ledger import validate_roundoff_event
            if len(integrals)!=1:
                issues.append({'kind':'BH_DUPLICATE_ACCRETION_INTEGRAL','particle_id':key[1]});continue
            integral=integrals[0];validate_roundoff_event(integral)
            if integral['applied_inflow_mass_native']==0:
                if sinks or recipients:issues.append({'kind':'BH_UNAPPLIED_DEMAND_HAS_PHYSICAL_TRANSFER','particle_id':key[1]})
                continue
        if len(sinks)!=1 or not gas or sinks[0]['particle_id']!=key[1]:
            issues.append({'kind':'INCOMPLETE_BH_ACCRETION_GROUP','particle_id':key[1]});continue
        sink=sinks[0];removed=-math.fsum(source_mass_change_msun(r) for r in gas)
        nondecreasing=any(source_mass_change_msun(r)>=0 for r in gas)
        inflow=sink['inflow_mass_msun'];gain=source_mass_change_msun(sink);radiated=sink['radiated_mass_msun']
        physical_gain=source_mass_change_msun(sink,physical=True)
        if integrals:
            integral=integrals[0]
            names=('mass_before_native','mass_after_native','physical_mass_before_native','physical_mass_after_native')
            if (any(integral[name]!=sink[name] for name in names)
                    or integral['applied_inflow_mass_native']!=sink['inflow_mass_native']):
                issues.append({'kind':'BH_ACCRETION_INTEGRAL_TRANSFER_MISMATCH','particle_id':key[1]})
            factor=sink['native_mass_unit_msun']
            gain=math.fsum([sink['mass_after_native']-sink['mass_before_native'],
                integral['gravity_mass_correction_before_native'],-integral['gravity_mass_correction_after_native']])*factor
            physical_gain=math.fsum([sink['physical_mass_after_native']-sink['physical_mass_before_native'],
                integral['physical_mass_correction_before_native'],-integral['physical_mass_correction_after_native']])*factor
        if nondecreasing or not close(removed,inflow) or not close(removed,gain+radiated):
            issues.append({'kind':'BH_ACCRETION_MASS_RADIATION_NOT_CLOSED','particle_id':key[1],
                'native_time':sink['native_time'],'counter':sink['counter'],
                'donor_removed_msun':removed,'declared_inflow_msun':inflow,
                'gravity_gain_msun':gain,'radiated_mass_msun':radiated,
                'donor_inflow_residual_msun':removed-inflow,'donor_gain_radiation_residual_msun':removed-gain-radiated,
                'nondecreasing_donor_present':nondecreasing,
                'tolerance':{'mass_absolute_msun':tolerances['mass_absolute_msun'],'mass_relative':tolerances['mass_relative']}})
        if not close(physical_gain,gain):
            issues.append({'kind':'BH_ACCRETION_PHYSICAL_GRAVITY_GAIN_MISMATCH','particle_id':key[1]})
        donor=sum((r['mass_before_msun']*np.asarray(r['velocity_before_km_s'])-r['mass_after_msun']*np.asarray(r['velocity_after_km_s']) for r in recipients),start=np.zeros(3))
        before_p=sink['mass_before_msun']*np.asarray(sink['velocity_before_km_s']);after_p=sink['mass_after_msun']*np.asarray(sink['velocity_after_km_s']);radiation=np.asarray(sink['radiated_momentum_msun_km_s'])
        residual=donor+before_p-after_p-radiation;scale=sum(np.linalg.norm(p) for p in (donor,before_p,after_p,radiation))
        if np.linalg.norm(residual)>tolerances['momentum_absolute_msun_km_s']+tolerances['momentum_relative']*scale:
            issues.append({'kind':'BH_ACCRETION_MOMENTUM_RADIATION_NOT_CLOSED','particle_id':key[1]})
    # Extensive kinetic+thermal source closure uses the frozen relative mass
    # tolerance. Absolute mass/impulse floors propagate to energy via the actual
    # source velocities and specific energies; no denominator clamp is used.
    def check_energy(key,rows,energy_input,advected=0.,radiation_kinetic=0.,bh=False):
        before_energy=0.;after_energy=0.;maximum_u=0.;maximum_speed=0.
        for row in rows:
            vb=np.asarray(row['velocity_before_km_s']);va=np.asarray(row['velocity_after_km_s'])
            ub=row.get('internal_energy_before_cm2_s2',0.);ua=row.get('internal_energy_after_cm2_s2',0.)
            before_energy+=row['mass_before_msun']*(ub+.5*float(np.dot(vb,vb))*1.e10)*MSUN_G
            after_energy+=row['mass_after_msun']*(ua+.5*float(np.dot(va,va))*1.e10)*MSUN_G
            maximum_u=max(maximum_u,ub,ua);maximum_speed=max(maximum_speed,float(np.linalg.norm(vb)),float(np.linalg.norm(va)))
        residual=after_energy-before_energy-energy_input+advected+radiation_kinetic
        floor=tolerances['mass_absolute_msun']*MSUN_G*(maximum_u+.5*maximum_speed**2*1.e10)+tolerances['momentum_absolute_msun_km_s']*MSUN_G*maximum_speed*1.e10
        scale=abs(before_energy)+abs(after_energy)+abs(energy_input)+abs(advected)+abs(radiation_kinetic)
        if abs(residual)>floor+tolerances['mass_relative']*scale:
            issues.append({'kind':'BH_SOURCE_ENERGY_NOT_CLOSED' if bh else 'RETURN_SOURCE_ENERGY_NOT_CLOSED','particle_id':key[1],'residual_erg':residual})
    from core.foundation import MSUN_G
    for key,rows in source_groups.items():
        donors=[r for r in rows if r['species']==4]
        if len(donors)==1:check_energy(key,rows,donors[0]['injected_energy_erg'])
    for key,rows in bh_groups.items():
        sinks=[r for r in rows if r['kind']=='accretion'];gas=[r for r in rows if r['kind'] in ('gas_accretion','gas_feedback')]
        if len(sinks)!=1 or not gas:continue
        sink=sinks[0];normal=[dict(r) for r in gas]+[dict(sink)]
        lost=sum(-.5*source_mass_change_msun(r)*float(np.dot(r['velocity_before_km_s'],r['velocity_before_km_s']))*MSUN_G*1.e10 for r in gas)
        fraction=sink['radiated_mass_msun']/sink['inflow_mass_msun']
        check_energy(key,normal,sink['thermal_energy_erg']+sink['kinetic_energy_erg'],sink['advected_internal_energy_erg'],fraction*lost,bh=True)
    missing=sorted(set(expected)-set(observed));extra=sorted(set(observed)-set(expected))
    for pid in missing:issues.append({'kind':'UNACCOUNTED_DISAPPEARANCE','particle_id':pid})
    for pid in extra:issues.append({'kind':'UNACCOUNTED_APPEARANCE','particle_id':pid})
    matched=sorted(set(expected)&set(observed));residuals=[];type_errors=[]
    for pid in matched:
        kind,mass=expected[pid];actual_kind,actual_mass=observed[pid]
        residuals.append(actual_mass-mass);type_errors.append(actual_kind!=kind)
        if actual_kind!=kind:issues.append({'kind':'UNACCOUNTED_SPECIES_CHANGE','particle_id':pid})
        if not close(mass,actual_mass):issues.append({'kind':'UNACCOUNTED_MASS_CHANGE','particle_id':pid,'expected_msun':mass,'actual_msun':actual_mass})
    # Frozen birth fields are checked against the actual final SSP observation.
    slots={int(pid):i for i,pid in enumerate(after['particle_ids'])}
    field_map={'parent_id':'SSPParentGasID','mode':'SSPBirthMode','birth_time':'SSPBirthNativeTime',
        'scale_factor':'SSPBirthScaleFactor','metallicity':'SSPBirthMetallicity',
        'gross_mass':'SSPInitialMass','locked_mass':'SSPInitialLockedMass','prompt_mass':'SSPPromptReturnedMass'}
    for i,pid in enumerate(events['ssp_id']):
        j=slots.get(int(pid))
        if j is None:continue
        if after.get('SSPBirthStateVersion',np.zeros(len(slots),dtype=np.uint64))[j]!=1:
            issues.append({'kind':'NEW_SSP_BIRTH_STATE_UNAVAILABLE','particle_id':int(pid)});continue
        for event_field,native_field in field_map.items():
            if native_field not in after:raise ValueError('Partial final SSP birth state')
            x=events[event_field][i];y=after[native_field][j]
            equal=close(float(x),float(y)) if 'Mass' in native_field else x==y
            if not equal:issues.append({'kind':'NEW_SSP_FROZEN_BIRTH_MISMATCH','particle_id':int(pid),'field':native_field})
        if after['StellarRandomCounter'][j]<events['opportunity_counter'][i]:
            issues.append({'kind':'NEW_SSP_RANDOM_COUNTER_REGRESSED','particle_id':int(pid)})
    before_slots={int(pid):i for i,pid in enumerate(before['particle_ids'])}
    for pid in sorted(set(initial)&set(observed)):
        if initial[pid][0]!=4 or observed[pid][0]!=4:continue
        a=before_slots[pid];b=slots[pid]
        for field in ('SSPBirthStateVersion',*field_map.values()):
            if field in before and field in after and before[field][a]!=after[field][b]:
                issues.append({'kind':'EXISTING_SSP_FROZEN_BIRTH_CHANGED','particle_id':pid,'field':field})
        if ('StellarRandomCounter' in before and 'StellarRandomCounter' in after
                and after['StellarRandomCounter'][b]<before['StellarRandomCounter'][a]):
            issues.append({'kind':'EXISTING_SSP_RANDOM_COUNTER_REGRESSED','particle_id':pid})
    def totals(rows):
        count=np.zeros(6,dtype=np.int64);mass=np.zeros(6)
        for kind,m in rows.values():count[kind]+=1;mass[kind]+=m
        return count,mass
    bc,bm=totals(initial);ec,em=totals(expected);ac,am=totals(observed)
    arrays={'matched_ids':np.asarray(matched,dtype=np.uint64),'missing_ids':np.asarray(missing,dtype=np.uint64),
        'extra_ids':np.asarray(extra,dtype=np.uint64),'matched_mass_residual_msun':np.asarray(residuals),
        'matched_species_mismatch':np.asarray(type_errors,dtype=bool),
        'species_counts_before':bc,'species_counts_expected':ec,'species_counts_after':ac,
        'species_mass_before_msun':bm,'species_mass_expected_msun':em,'species_mass_after_msun':am}
    return {'status':'PASS' if not issues else 'FAIL','event_count':n,'issues':issues,
        'return_row_count':len(source_records),'bh_row_count':len(bh_records),
        'accounting_scope':'All six native species; replayed SSP births, delayed returns and native BH seed/accretion/merger rows with explicit radiated mass/momentum.',
        'momentum_scope':'Instantaneous source operations only; gravitational/hydrodynamic momentum evolution between outputs is not subtracted'},arrays


def read_native_hydro_census(root,snapshot,*,memory_bytes):
    """Complete bounded native census, frozen SSP fields and gas observables."""
    from pathlib import Path
    import h5py,numpy as np
    from halos import _snapshot_files
    from core.data import (verify_native_hdf5_set,native_publication_markers,snapshot_total_counts,
        snapshot_id_shards,dataset_to_cgs_factor,run_epoch_contract,isolated_epoch_contract)
    from core.foundation import file_sha256 as digest, MSUN_G, SECONDS_PER_YEAR
    root=Path(root).resolve();files=_snapshot_files(root,snapshot);verify_native_hdf5_set(files,require_publication=True)
    sources={str(p.resolve()):digest(p) for p in files+native_publication_markers(files)}
    with h5py.File(files[0],'r') as f:total=sum(int(x) for x in snapshot_total_counts(f['Header'].attrs))
    if type(memory_bytes) is not int or memory_bytes<1 or total*2048+32*1024**2>memory_bytes:
        raise MemoryError('Complete hydro census exceeds declared workspace')
    mapping={'unit_mass_g':'UnitMass_in_g','unit_time_s':'UnitTime_in_s','unit_velocity_cm_s':'UnitVelocity_in_cm_per_s',
        'unit_length_cm':'UnitLength_in_cm','hubble_param':'HubbleParam','factor_sn':'FactorSN','comoving':'ComovingIntegrationOn','seed':'StellarRandomSeed'}
    reference=None;header=None;return_header=None;bh_header=None;gamma_reference=None
    for path in files:
        with h5py.File(path,'r') as f:
            p=f['Parameters'].attrs;h=f['Header'].attrs
            missing=sorted((set(mapping.values())-{'UnitTime_in_s'})-set(p))
            if missing:raise ValueError('Native hydro census missing required Parameters attributes: '+', '.join(missing))
            from core.data import native_time_unit_seconds
            parameters={k:p[v] for k,v in mapping.items() if k!='unit_time_s'}
            parameters['unit_time_s']=native_time_unit_seconds(p)
            seed=parameters['seed'];seed=seed.decode('ascii') if isinstance(seed,bytes) else seed
            if not isinstance(seed,str) or not seed.isdecimal() or int(seed)>=2**64:raise ValueError('Exact native hydro random seed required')
            parameters['seed']=str(int(seed));mode=parameters['comoving']
            model_crc=p.get('StellarEvolutionTableCRC32',0)
            if isinstance(model_crc,bytes):model_crc=model_crc.decode('ascii')
            if isinstance(model_crc,str):
                if not model_crc.isdecimal():raise ValueError('Exact native stellar model CRC required')
                model_crc=int(model_crc)
            if isinstance(model_crc,(bool,np.bool_)) or not isinstance(model_crc,(int,np.integer)) or not 0<=int(model_crc)<2**32:
                raise ValueError('Exact native stellar model CRC required')
            parameters['stellar_model_crc32']=int(model_crc)
            parameters['grackle_network']=int(p.get('GracklePrimordialChemistry',0))
            parameters['deuterium_hydrogen_number_ratio']=float(p.get('GrackleDeuteriumToHydrogenRatio',0.))
            seed_names=('HI','HII','HeI','HeII','HeIII','Electron','HM','H2I','H2II','DI','DII','HDI')
            parameters['grackle_seed_number_fractions']={name:float(p.get('GrackleSeed'+name,0.)) for name in seed_names}
            if any(not np.isfinite(value) or value<0 for value in parameters['grackle_seed_number_fractions'].values()):
                raise ValueError('Saved primordial seed abundance is invalid')
            if not 0<=parameters['grackle_network']<=3 or not np.isfinite(parameters['deuterium_hydrogen_number_ratio']) or parameters['deuterium_hydrogen_number_ratio']<0:
                raise ValueError('Invalid saved primordial chemical network or isotope abundance')
            if np.asarray(mode).shape!=() or np.asarray(mode).dtype.kind not in 'iu' or int(mode) not in (0,1):raise ValueError('Invalid native hydro integration mode')
            parameters['comoving']=int(mode)
            for name in set(mapping)-{'seed','comoving'}:
                parameters[name]=float(parameters[name])
                if not np.isfinite(parameters[name]) or parameters[name]<=0:raise ValueError('Invalid native hydro physical parameter')
            if parameters['factor_sn']>=1:raise ValueError('Invalid native SH03 beta')
            configuration=f.get('Config');gamma=float(configuration.attrs.get('GAMMA',5./3)) if configuration is not None else 5./3
            if not np.isfinite(gamma) or not 1<gamma<=2 or configuration is not None and 'ISOTHERM_EQS' in configuration.attrs:raise ValueError('Native hydro thermodynamics requires the recorded supported ideal gas gamma')
            if gamma_reference is not None and gamma!=gamma_reference:raise ValueError('Hydro shards disagree about the gas equation of state')
            gamma_reference=gamma
            names=('StellarBirthCommitCount','StellarBirthEventCount','StellarBirthJournalSchema')
            if any(k not in h or np.asarray(h[k]).shape!=() or np.asarray(h[k]).dtype!=np.dtype('uint64') for k in names):
                raise ValueError('Hydro history requires authoritative exact journal counters')
            current=(float(h['Time']),*(int(h[k]) for k in names))
            if not np.isfinite(current[0]) or current[0]<0 or (parameters['comoving'] and current[0]<=0) or current[3]!=1 or current[1]>current[2]:
                raise ValueError('Invalid native hydro epoch/counters')
            if reference is not None and (parameters!=reference or current!=header):raise ValueError('Hydro shards disagree about model/epoch/counters')
            reference=parameters;header=current
            rnames=('StellarReturnCommitCount','StellarReturnEventCount','StellarReturnJournalSchema')
            present=sum(k in h for k in rnames)
            if present not in (0,3):raise ValueError('Partial native stellar return header')
            if present:
                if any(np.asarray(h[k]).shape!=() or np.asarray(h[k]).dtype!=np.dtype('uint64') for k in rnames):
                    raise ValueError('Native stellar return counters must be scalar uint64')
                rcurrent=tuple(int(h[k]) for k in rnames)
                if rcurrent[2]!=1 or rcurrent[0]>rcurrent[1]:raise ValueError('Invalid native stellar return counters')
            else:rcurrent=(0,0,0)
            if return_header is not None and rcurrent!=return_header:raise ValueError('Stellar return shard counter/schema mismatch')
            return_header=rcurrent
            bnames=('BlackHoleEventCommitCount','BlackHoleEventCount','BlackHoleEventJournalSchema')
            present=sum(k in h for k in bnames)
            if present not in (0,3):raise ValueError('Partial native BH event header')
            if present:
                if any(np.asarray(h[k]).shape!=() or np.asarray(h[k]).dtype!=np.dtype('uint64') for k in bnames):raise ValueError('BH event counters must be scalar uint64')
                bcurrent=tuple(int(h[k]) for k in bnames)
                if bcurrent[2]!=1 or bcurrent[0]>bcurrent[1]:raise ValueError('Invalid BH journal schema/counters')
            else:bcurrent=(0,0,0)
            if bh_header is not None and bh_header!=bcurrent:raise ValueError('BH shard counter/schema mismatch')
            bh_header=bcurrent
    from core.data import native_physical_unit_scales
    physical_scales=native_physical_unit_scales(unit_mass_g=reference['unit_mass_g'],unit_length_cm=reference['unit_length_cm'],
        unit_velocity_cm_s=reference['unit_velocity_cm_s'],comoving=reference['comoving'],hubble_param=reference['hubble_param'])
    time,commit,count,_=header;mass_factor=physical_scales['mass_g']/MSUN_G
    mass_h_exponent=-1. if reference['comoving'] else 0.
    arrays={'particle_ids':np.empty(total,dtype=np.uint64),'species':np.empty(total,dtype=np.int8),'mass_msun':np.empty(total)}
    birth_fields=('SSPParentGasID','SSPInitialMass','SSPInitialLockedMass','SSPPromptReturnedMass','SSPCurrentMass',
        'SSPBirthScaleFactor','SSPBirthNativeTime','SSPBirthMetallicity','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter')
    integer={'SSPParentGasID','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter'}
    for field in birth_fields:arrays[field]=np.zeros(total,dtype=np.uint64 if field in integer else float)
    for field in ('SSPLiveMass','SSPRemnantMass','SSPPopulationVersion'):arrays[field]=np.zeros(total,dtype=np.uint64 if field=='SSPPopulationVersion' else float)
    from core.stellar_returns import allocate_stellar_return_state,read_stellar_return_state
    arrays.update(allocate_stellar_return_state(total))
    gas_fields={'Density':'gas_density_g_cm3','InternalEnergy':'gas_internal_energy_cm2_s2',
        'StarFormationRate':'gas_sfr_msun_per_year','ElectronAbundance':'gas_electron_abundance'}
    for field in gas_fields.values():arrays[field]=np.zeros(total);arrays[field+'_available']=np.zeros(total,dtype=bool)
    arrays['gas_element_mass_fraction']=np.zeros((total,11));arrays['gas_elements_available']=np.zeros(total,dtype=bool)
    arrays['gas_atomic_species_fraction']=np.zeros((total,6));arrays['gas_atomic_species_available']=np.zeros(total,dtype=bool)
    arrays['gas_molecular_species_fraction']=np.zeros((total,6));arrays['gas_molecular_species_available']=np.zeros(total,dtype=bool)
    arrays['gas_chemistry_state_version']=np.zeros(total,dtype=np.uint64)
    arrays['gas_gamma_pressure']=np.zeros(total);arrays['gas_gamma_sound']=np.zeros(total)
    arrays['gas_mean_molecular_weight']=np.zeros(total);arrays['gas_molecular_eos_available']=np.zeros(total,dtype=bool)
    arrays['gas_native_eos_available']=np.zeros(total,dtype=bool)
    arrays['gas_native_chemistry_temperature_kelvin']=np.zeros(total)
    arrays['gas_inverse_cooling_time_per_s']=np.zeros(total);arrays['gas_cooling_time_available']=np.zeros(total,dtype=bool)
    arrays['positions_proper_kpc']=np.zeros((total,3));arrays['velocities_peculiar_km_s']=np.zeros((total,3))
    arrays['gas_temperature_kelvin']=np.zeros(total);arrays['gas_temperature_available']=np.zeros(total,dtype=bool)
    arrays['gas_chemistry_radiated_energy_erg']=np.zeros(total);arrays['gas_chemistry_energy_available']=np.zeros(total,dtype=bool)
    bh_fields={'BH_Mass':(1,'mass'),'BHSpin':(3,'dimensionless'),'BHStateVersion':(1,'uint64'),
        'BHCompanionID':(1,'uint64'),'BHParentIDs':(2,'uint64'),'BHGeneration':(1,'uint64'),'BHEventCounter':(1,'uint64'),
        'BHBinaryCounter':(1,'uint64'),'BHBinarySemimajorAxisCM':(1,'cm'),'BHBinaryEccentricity':(1,'dimensionless'),
        'BHBinaryAxis':(3,'dimensionless'),'BHBinaryRadial':(3,'dimensionless'),'BHBinarySecondarySpin':(3,'dimensionless'),
        'BHBinaryPhase':(1,'dimensionless'),'BHBinaryGWOrbitalEnergyERG':(1,'erg'),'BHBinaryStellarScatteringEnergyERG':(1,'erg'),
        'BHFrictionHeatERG':(1,'erg'),'BHBinaryConstraintReservoirERG':(1,'erg'),'BHBinaryPreviousPhysicalMassGRAM':(2,'g'),
        'BHBinaryAngularReservoirCGS':(3,'angular'),'BHBinaryGWAngularMomentumCGS':(3,'angular'),
        'BHLastSourceNativeTime':(1,'time'),'BHLastDynamicsNativeTime':(1,'time'),'BHBinaryLastNativeTime':(1,'time')}
    for name,(width,unit) in bh_fields.items():
        arrays[name]=np.zeros((total,width) if width>1 else total,dtype=np.uint64 if unit=='uint64' else float)
    from core.black_hole_ledger import SEED_FIELDS, read_seed_ledger
    for name,(width,unit) in SEED_FIELDS.items():
        arrays[name]=np.zeros((total,width) if width>1 else total,dtype=np.uint64 if unit=='uint64' else float)
    from core.black_hole_ledger import allocate_roundoff_state,read_roundoff_state
    arrays.update(allocate_roundoff_state(total))
    arrays['bh_native_state_available']=np.zeros(total,dtype=bool)
    cursor=0
    for kind in range(6):
        records,_=snapshot_id_shards(files,kind)
        for record in records:
            n=record['count']
            if not n:continue
            selection=slice(cursor,cursor+n)
            with h5py.File(record['path'],'r') as f:
                group=f[f'PartType{kind}']
                if group['ParticleIDs'].dtype!=np.dtype('uint64'):raise ValueError('Lossy native hydro identity')
                raw=group['Masses'][:] if 'Masses' in group else np.full(n,f['Header'].attrs['MassTable'][kind])
                if not np.isfinite(raw).all() or np.any(raw<0):raise ValueError('Invalid native hydro mass')
                if 'Masses' in group:
                    attrs=group['Masses'].attrs
                    if any(k not in attrs or float(attrs[k])!=v for k,v in {'to_cgs':reference['unit_mass_g'],'a_scaling':0.,'h_scaling':mass_h_exponent}.items()):
                        raise ValueError('Hydro mass units disagree with the native model')
                for native,target,to_unit in [('Coordinates','positions_proper_kpc',3.0856775814913673e21),('Velocities','velocities_peculiar_km_s',1.e5)]:
                    dataset=group[native]
                    if dataset.shape!=(n,3) or not np.isfinite(dataset[:]).all():raise ValueError('Invalid native phase-space census')
                    a=time if reference['comoving'] else 1.
                    factor=dataset_to_cgs_factor(dict(dataset.attrs),a,reference['hubble_param'])/to_unit
                    expected=(reference['unit_length_cm'],1. if reference['comoving'] else 0.,mass_h_exponent) if native=='Coordinates' else (reference['unit_velocity_cm_s'],.5 if reference['comoving'] else 0.,0.)
                    if any(k not in dataset.attrs or not np.isclose(float(dataset.attrs[k]),v,rtol=8*np.finfo(float).eps,atol=0) for k,v in zip(('to_cgs','a_scaling','h_scaling'),expected)):raise ValueError('Native census position/velocity unit mismatch')
                    arrays[target][selection]=dataset[:]*factor
                arrays['particle_ids'][selection]=group['ParticleIDs'][:];arrays['species'][selection]=kind;arrays['mass_msun'][selection]=raw*mass_factor
                if kind==4:
                    present=set(birth_fields)&set(group)
                    if present and present!=set(birth_fields):raise ValueError('Partial native SSP birth state')
                    if present:
                        payload={'ParticleIDs':arrays['particle_ids'][selection]}
                        for name in birth_fields:
                            dataset=group[name]
                            if dataset.shape!=(n,) or (name in integer and dataset.dtype!=np.dtype('uint64')):raise ValueError('Misaligned or lossy native SSP field')
                            cgs=reference['unit_mass_g'] if name.endswith('Mass') else reference['unit_time_s'] if name=='SSPBirthNativeTime' and not reference['comoving'] else 1.
                            if any(k not in dataset.attrs or float(dataset.attrs[k])!=v for k,v in {'to_cgs':cgs,'a_scaling':0.,'h_scaling':mass_h_exponent if name.endswith('Mass') else 0.}.items()):
                                raise ValueError('Incorrect native SSP field units: '+name)
                            payload[name]=dataset[:]
                        # Native removal slots are not live SSPs. Multiple
                        # deleted rows may legitimately have the zero sentinel.
                        live=~((payload['ParticleIDs']==0)&(raw==0))
                        validate_ssp_birth_arrays({k:v[live] for k,v in payload.items()},factor_sn=reference['factor_sn'],actual_native_time=time,comoving=bool(reference['comoving']))
                        eps=np.finfo(raw.dtype).eps if raw.dtype.kind=='f' else np.finfo(float).eps
                        if np.any(abs(payload['SSPCurrentMass']-raw)>4*eps*np.maximum(abs(raw),1e-300)):
                            raise ValueError('SSP gravity/current mass mismatch')
                        for name in birth_fields:arrays[name][selection]=payload[name]*(mass_factor if name.endswith('Mass') else 1)
                    else:arrays['SSPCurrentMass'][selection]=raw*mass_factor
                if kind==4:
                    names=('SSPLiveMass','SSPRemnantMass','SSPPopulationVersion');present=sum(k in group for k in names)
                    if present not in (0,3):raise ValueError('Partial native live/remnant population state')
                    if present:
                        for name in names:
                            dataset=group[name]
                            if dataset.shape!=(n,) or name=='SSPPopulationVersion' and dataset.dtype!=np.dtype('uint64'):raise ValueError('Misaligned/lossy SSP population state')
                            cgs=1. if name=='SSPPopulationVersion' else reference['unit_mass_g']
                            if any(k not in dataset.attrs or float(dataset.attrs[k])!=v for k,v in {'to_cgs':cgs,'a_scaling':0.,'h_scaling':0. if name=='SSPPopulationVersion' else mass_h_exponent}.items()):raise ValueError('SSP population unit mismatch')
                            arrays[name][selection]=dataset[:]*(1 if name=='SSPPopulationVersion' else mass_factor)
                        valid=arrays['SSPPopulationVersion'][selection]==1
                        if np.any(arrays['SSPPopulationVersion'][selection]>1) or any(not np.isfinite(arrays[k][selection]).all() or np.any(arrays[k][selection]<0) for k in names[:2]):raise ValueError('Invalid SSP population masses/version')
                        if not np.allclose((arrays['SSPLiveMass'][selection]+arrays['SSPRemnantMass'][selection])[valid],arrays['mass_msun'][selection][valid],rtol=1.e-12,atol=0):raise ValueError('Native live/remnant mass closure failed')
                    cumulative=read_stellar_return_state(group,parameters=reference,native_time=time,required=return_header[2]==1)
                    if cumulative is not None:
                        for name,values in cumulative.items():arrays[name][selection]=values
                        arrays['stellar_return_state_available'][selection]=True
                if kind==5 and 'BH_Mass' in group:
                    present=set(bh_fields)&set(group)
                    if present!=set(bh_fields):raise ValueError('Partial native BH lifecycle/binary restart state')
                    for name,(width,unit) in bh_fields.items():
                        dataset=group[name];shape=(n,width) if width>1 else (n,)
                        if dataset.shape!=shape or unit=='uint64' and dataset.dtype!=np.dtype('uint64'):raise ValueError('Lossy or misaligned native BH state: '+name)
                        values=dataset[:]
                        if not np.isfinite(values).all():raise ValueError('Nonfinite native BH state')
                        cgs=reference['unit_mass_g'] if unit=='mass' else reference['unit_time_s'] if unit=='time' and not reference['comoving'] else 1.
                        if any(k not in dataset.attrs or float(dataset.attrs[k])!=v for k,v in {'to_cgs':cgs,'a_scaling':0.,'h_scaling':mass_h_exponent if unit=='mass' else 0.}.items()):raise ValueError('Incorrect native BH field units: '+name)
                        arrays[name][selection]=values*(mass_factor if unit=='mass' else 1)
                    state=arrays['BHStateVersion'][selection];physical=arrays['BH_Mass'][selection]
                    live=~((arrays['particle_ids'][selection]==0)&(raw==0))
                    if np.any(state>1) or np.any(physical[live]<=0) or np.any(physical[live]>arrays['mass_msun'][selection][live]):raise ValueError('Invalid physical/composite BH masses or version')
                    if np.any(np.linalg.norm(arrays['BHSpin'][selection][live],axis=1)>1):raise ValueError('Native BH spin exceeds Kerr domain')
                    arrays['bh_native_state_available'][selection]=state==1
                    for name,values in read_seed_ledger(group,n).items():
                        arrays[name][selection]=values
                    for name,values in read_roundoff_state(group,n,parameters=reference).items():
                        arrays[name][selection]=values
                if kind==0:
                    cpresent=sum(name in group for name in ('ElementMassFractions','ChemicalStateVersion'))
                    if cpresent not in (0,2):raise ValueError('Partial native eleven-element gas state')
                    if cpresent:
                        for native in ('ChemicalStateVersion','ElementMassFractions'):
                            if any(k not in group[native].attrs or float(group[native].attrs[k])!=v for k,v in {'to_cgs':1.,'a_scaling':0.,'h_scaling':0.}.items()):raise ValueError('Incorrect native chemical field units')
                        state=group['ChemicalStateVersion'][:];fractions=group['ElementMassFractions'][:]
                        if state.dtype!=np.dtype('uint64') or state.shape!=(n,) or np.any(state>1) or fractions.shape!=(n,11) or not np.isfinite(fractions).all() or np.any(fractions<0):
                            raise ValueError('Invalid native gas element state')
                        valid=state==1
                        if np.any(abs(fractions[valid].sum(axis=1)-1)>1.e-12):raise ValueError('Native gas element fractions do not normalize')
                        arrays['gas_element_mass_fraction'][selection]=fractions;arrays['gas_elements_available'][selection]=valid
                    gpresent=sum(name in group for name in ('GrackleAtomicSpeciesFractions','GrackleStateVersion','ChemistryCumulativeRadiatedEnergy'))
                    if gpresent not in (0,3):raise ValueError('Partial native Grackle gas state')
                    if gpresent:
                        for native in ('GrackleStateVersion','GrackleAtomicSpeciesFractions','ChemistryCumulativeRadiatedEnergy'):
                            energy = native == 'ChemistryCumulativeRadiatedEnergy'
                            cgs = reference['unit_mass_g'] * reference['unit_velocity_cm_s']**2 if energy else 1.
                            h_scaling = -1. if energy and reference['comoving'] else 0.
                            attrs = group[native].attrs
                            # GADGET computes the energy unit via length squared / time squared.
                            if (not np.isclose(float(attrs.get('to_cgs', np.nan)), cgs, rtol=1e-14, atol=0.)
                                    or attrs.get('a_scaling') != 0. or attrs.get('h_scaling') != h_scaling):
                                raise ValueError('Incorrect native Grackle field units')
                        state=group['GrackleStateVersion'][:];fractions=group['GrackleAtomicSpeciesFractions'][:];radiated=group['ChemistryCumulativeRadiatedEnergy'][:]
                        if state.dtype!=np.dtype('uint64') or state.shape!=(n,) or np.any(state>4) or fractions.shape!=(n,6) or not np.isfinite(fractions).all() or np.any(fractions<0) or radiated.shape!=(n,) or not np.isfinite(radiated).all():
                            raise ValueError('Invalid native Grackle atomic/energy state')
                        arrays['gas_atomic_species_fraction'][selection]=fractions
                        arrays['gas_atomic_species_available'][selection]=np.isin(state,[1,3,4])
                        arrays['gas_chemistry_state_version'][selection]=state
                        if 'GrackleInverseCoolingTime' in group:
                            field=group['GrackleInverseCoolingTime']
                            if any(k not in field.attrs or float(field.attrs[k])!=v for k,v in {'to_cgs':1.,'a_scaling':0.,'h_scaling':0.}.items()):
                                raise ValueError('Native inverse Grackle cooling-time units are not physical inverse seconds')
                            inverse=field[:]
                            if inverse.shape!=(n,) or not np.isfinite(inverse).all():raise ValueError('Native cooling-time diagnostic is invalid')
                            arrays['gas_inverse_cooling_time_per_s'][selection]=inverse
                            arrays['gas_cooling_time_available'][selection]=state>0
                        molecular_fields=('GrackleMolecularSpeciesFractions','GrackleGammaPressure','GrackleGammaSound','GrackleTemperature','GrackleMeanMolecularWeight')
                        if np.any(state>=3) and not all(name in group for name in molecular_fields):
                            raise ValueError('Molecular chemistry state is missing species or native EOS fields')
                        if all(name in group for name in molecular_fields):
                            for name in molecular_fields:
                                if any(k not in group[name].attrs or float(group[name].attrs[k])!=v for k,v in {'to_cgs':1.,'a_scaling':0.,'h_scaling':0.}.items()):
                                    raise ValueError('Molecular chemistry/EOS dataset has incorrect physical units')
                            molecular=group['GrackleMolecularSpeciesFractions'][:]
                            if molecular.shape!=(n,6) or not np.isfinite(molecular).all() or np.any(molecular<0):
                                raise ValueError('Invalid aligned H2/HD species fractions')
                            molecular_available=state>=3
                            available=(state>0)&arrays['gas_sfr_msun_per_year_available'][selection]&(arrays['gas_sfr_msun_per_year'][selection]<=0)
                            if np.any((state==3)&(reference['grackle_network']!=2)) or np.any((state==4)&(reference['grackle_network']!=3)):
                                raise ValueError('Native molecular state differs from the saved physical network')
                            arrays['gas_molecular_species_fraction'][selection]=molecular
                            arrays['gas_molecular_species_available'][selection]=molecular_available
                            for native,key in [('GrackleGammaPressure','gas_gamma_pressure'),('GrackleGammaSound','gas_gamma_sound'),
                                    ('GrackleTemperature','gas_native_chemistry_temperature_kelvin'),('GrackleMeanMolecularWeight','gas_mean_molecular_weight')]:
                                value=group[native][:]
                                if value.shape!=(n,) or not np.isfinite(value).all() or np.any(value[available]<=0):
                                    raise ValueError('Native molecular EOS fields are nonfinite or unavailable')
                                if 'Gamma' in native and np.any((value[available]<=1)|(value[available]>gamma_reference+1e-6)):
                                    raise ValueError('Native molecular EOS gamma leaves the supported ideal-mixture domain')
                                arrays[key][selection]=value
                            arrays['gas_molecular_eos_available'][selection]=molecular_available
                            arrays['gas_native_eos_available'][selection]=available
                            valid=molecular_available&arrays['gas_elements_available'][selection]
                            nuclear_h=fractions[:,0]+fractions[:,1]+molecular[:,0]+molecular[:,1]+molecular[:,2]+molecular[:,5]/3
                            nuclear_d=molecular[:,3]+molecular[:,4]+2*molecular[:,5]/3
                            nuclear_he=fractions[:,2:5].sum(axis=1)
                            elements=arrays['gas_element_mass_fraction'][selection]
                            if np.any(np.abs((nuclear_h+nuclear_d-elements[:,0])[valid])>1e-6) or np.any(np.abs((nuclear_he-elements[:,1])[valid])>1e-6):
                                raise ValueError('Actual molecular snapshot violates H/D/He nuclear conservation')
                            charge=fractions[:,1]+fractions[:,3]/4+fractions[:,4]/2+molecular[:,2]/2+molecular[:,4]/2-molecular[:,0]
                            if np.any(np.abs((fractions[:,5]-charge)[valid])>1e-6):
                                raise ValueError('Actual molecular snapshot violates species charge balance')
                        arrays['gas_chemistry_radiated_energy_erg'][selection]=radiated*physical_scales['energy_erg']
                        arrays['gas_chemistry_energy_available'][selection]=state>0
                    for native,name in gas_fields.items():
                        if native not in group:continue
                        dataset=group[native]
                        if dataset.shape!=(n,):raise ValueError('Misaligned native gas field')
                        values=dataset[:]
                        if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid native gas observable: '+native)
                        if native=='ElectronAbundance':factor=1. # pinned native COOLING: electrons per H nucleus; no unit attribute
                        else:
                            a=time if reference['comoving'] else 1.
                            factor=dataset_to_cgs_factor(dict(dataset.attrs),a,reference['hubble_param'])
                            expected={'Density':(reference['unit_mass_g']/reference['unit_length_cm']**3,-3. if reference['comoving'] else 0.,2. if reference['comoving'] else 0.),
                                'InternalEnergy':(reference['unit_velocity_cm_s']**2,0.,0.)}
                            if native in expected:
                                c,aa,hh=expected[native]
                                if not np.isclose(float(dataset.attrs['to_cgs']),c,rtol=8*np.finfo(float).eps,atol=0) or float(dataset.attrs['a_scaling'])!=aa or float(dataset.attrs['h_scaling'])!=hh:
                                    raise ValueError('Gas field units disagree with native model: '+native)
                            else:
                                if float(dataset.attrs['a_scaling'])!=0 or float(dataset.attrs['h_scaling'])!=0:raise ValueError('Native SFR has unexpected epoch/h scaling')
                                factor*=SECONDS_PER_YEAR/MSUN_G
                        arrays[name][selection]=values*factor;arrays[name+'_available'][selection]=True
                        if native=='ElectronAbundance' and gpresent:
                            # Equilibrium Grackle does not transport an atomic
                            # electron field. Its neutral input placeholder is
                            # not an observed ionization measurement.
                            arrays[name+'_available'][selection]&=state!=2
            cursor+=n
    if cursor!=total:raise ValueError('Incomplete native hydro census')
    tombstone=(arrays['particle_ids']==0)&(arrays['mass_msun']==0)&(arrays['species']==5)
    # Native monofonIC SC lattice IDs start at zero. Identity zero alone is
    # not a missingness sentinel; retirement is the complete type5/id0/mass0
    # tuple written by the native BH operator.
    if np.any((arrays['mass_msun']<=0)&~tombstone):raise ValueError('Only retired native BH carriers may have nonpositive mass')
    removed=int(tombstone.sum());arrays={k:v[~tombstone] for k,v in arrays.items()}
    if len(np.unique(arrays['particle_ids']))!=len(arrays['particle_ids']):raise ValueError('Duplicate live ID across native species/shards')
    gas=arrays['species']==0;mass=arrays['mass_msun'];summary={'live_particle_count':len(mass),'tombstone_count':removed,
        'gas_mass_msun':float(mass[gas].sum()),'fields':{},'phase_convention':'SFR>0 denotes native SH03 effective EOS; SFR=0 single-phase energy; temperature is derived only with measured composition and electrons'}
    for name in gas_fields.values():
        valid=gas&arrays[name+'_available'];weight=mass[valid].sum()
        summary['fields'][name]={'available_count':int(valid.sum()),'complete':bool(np.all(arrays[name+'_available'][gas])),
            'mass_weighted_mean':float(np.dot(mass[valid],arrays[name][valid])/weight) if weight>0 else None}
    sfr_available=gas&arrays['gas_sfr_msun_per_year_available'];forming=sfr_available&(arrays['gas_sfr_msun_per_year']>0)
    summary['instantaneous_sfr_msun_per_year']=float(arrays['gas_sfr_msun_per_year'][sfr_available].sum()) if np.all(arrays['gas_sfr_msun_per_year_available'][gas]) else None
    summary['star_forming_gas_mass_msun']=float(mass[forming].sum()) if summary['instantaneous_sfr_msun_per_year'] is not None else None
    thermal=gas&arrays['gas_elements_available']&arrays['gas_electron_abundance_available']&arrays['gas_internal_energy_cm2_s2_available']&sfr_available&~forming
    atomic_mass=np.asarray([1.,4.,12.,14.,16.,20.,24.,28.,32.,40.,56.])
    fractions=arrays['gas_element_mass_fraction'];inverse_mu=np.sum(fractions/atomic_mass,axis=1)+arrays['gas_electron_abundance']*fractions[:,0]
    if np.any(inverse_mu[thermal]<=0):raise ValueError('Invalid single-phase mean particle mass')
    arrays['gas_temperature_kelvin'][thermal]=(gamma_reference-1)*arrays['gas_internal_energy_cm2_s2'][thermal]*1.67262192595e-24/1.380649e-16/inverse_mu[thermal]
    arrays['gas_temperature_available'][thermal]=True
    molecular=gas&arrays['gas_native_eos_available']&sfr_available&~forming
    arrays['gas_cooling_time_available']&=gas&sfr_available&~forming
    arrays['gas_temperature_kelvin'][molecular]=arrays['gas_native_chemistry_temperature_kelvin'][molecular]
    arrays['gas_temperature_available'][molecular]=True
    summary['temperature_convention']='Atomic legacy: recorded constant gamma, measured U/composition/electrons. Molecular: actual native Grackle temperature, caloric gamma and molecular weight; SH03 effective EOS remains masked.'
    summary['native_eos_available_count']=int(molecular.sum())
    summary['molecular_eos_available_count']=int((gas&arrays['gas_molecular_eos_available']&sfr_available&~forming).sum())
    summary['single_phase_temperature_available_count']=int(thermal.sum())
    for phase,mask in (('effective_eos',forming),('single_phase',sfr_available&~forming)):
        valid=mask&arrays['gas_internal_energy_cm2_s2_available'];weight=mass[valid].sum()
        summary[phase+'_specific_internal_energy_cm2_s2']=float(np.dot(mass[valid],arrays['gas_internal_energy_cm2_s2'][valid])/weight) if weight>0 else None
    epoch=run_epoch_contract(root,snapshot,time) if reference['comoving'] else isolated_epoch_contract(time,
        time_unit_gyr=reference['unit_time_s']/SECONDS_PER_YEAR/1.e9,snapshot=snapshot)
    for path,checksum in sources.items():
        if digest(path)!=checksum:raise ValueError('Native hydro source changed during census')
    return arrays,{'snapshot':snapshot,'native_time':time,'commit_count':commit,'event_count':count,
        'return_commit_count':return_header[0],'return_event_count':return_header[1],'return_schema':return_header[2],
        'bh_commit_count':bh_header[0],'bh_event_count':bh_header[1],'bh_schema':bh_header[2],
        'element_order':['H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe'],
        'atomic_species_order':['HI','HII','HeI','HeII','HeIII','electron_mH_equivalent'],
        'molecular_species_order':['HM','H2I','H2II','DI','DII','HDI'],
        'chemical_state_versions':{'0':'unavailable','1':'atomic','2':'equilibrium','3':'H2','4':'H2_HD'},
        'parameters':reference,'gas_adiabatic_index':gamma_reference,'epoch':epoch,'summary':summary},sources,files


def read_native_stellar_return_journal(directory,*,before,after,memory_bytes):
    """Read immutable native return rows through complete MPI commit censuses."""
    from pathlib import Path
    import json,zlib,re
    import numpy as np
    from core.foundation import file_sha256 as digest, MSUN_G
    directory=Path(directory).resolve();rows=[];sources={};commits=[]
    start,stop=before['return_commit_count'],after['return_commit_count']
    count,target=before['return_event_count'],after['return_event_count']
    if stop<start or target<count or (stop==start and target!=count):raise ValueError('Stellar return counter chain regressed or incomplete')
    if (target-count)*4096+16*1024**2>memory_bytes:raise MemoryError('Stellar return row workspace exceeds declared limit')
    params=after['parameters'];unit_mass=params['unit_mass_g']/MSUN_G/(params['hubble_param'] if params['comoving'] else 1.)
    unit_velocity=params['unit_velocity_cm_s']/1.e5;unit_u=params['unit_velocity_cm_s']**2
    unit_energy=unit_mass*MSUN_G*unit_u
    fingerprint=None;model=None;last_time=before['native_time'];seen=set()
    def unique_object(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate stellar return commit key')
            out[key]=value
        return out
    def regular(path):
        if not path.is_file() or path.is_symlink():raise ValueError('Missing or linked stellar return source')
        sources[str(path)]=digest(path)
    for sequence in range(start+1,stop+1):
        path=directory/f'stellar_returns_{sequence:020d}.commit.json';regular(path)
        if path.stat().st_size>memory_bytes//16:raise MemoryError('Stellar return commit exceeds workspace')
        commit=json.loads(path.read_text(),object_pairs_hook=unique_object)
        for key in ('sequence','previous_count','cumulative_count','event_count','table_crc32','mpi_ranks'):
            if type(commit.get(key)) is not int or not 0<=commit[key]<2**64:raise ValueError('Exact stellar commit counters required')
        if commit.get('schema')!='native-stellar-returns-v1' or commit['sequence']!=sequence or commit['previous_count']!=count or commit['event_count']<=0 or commit['cumulative_count']!=count+commit['event_count']:
            raise ValueError('Incomplete stellar return commit chain')
        time=commit['native_time']
        if isinstance(time,bool) or not isinstance(time,(int,float)) or not np.isfinite(time) or time<last_time or time>after['native_time']:
            raise ValueError('Stellar return event outside actual epoch bracket')
        for name in ('unit_mass_g','unit_velocity_cm_s','unit_time_s','hubble_param','comoving'):
            if commit[name]!=params[name]:raise ValueError('Stellar return model units/integration mode mismatch')
        fp=commit['source_fingerprint']
        if not isinstance(fp,str) or re.fullmatch('[0-9a-f]{64}',fp) is None or fingerprint is not None and fp!=fingerprint:
            raise ValueError('Stellar return native build changed within interval')
        if commit['table_crc32']>2**32-1 or model is not None and model!=commit['table_crc32']:raise ValueError('Stellar return table changed within interval')
        if params.get('stellar_model_crc32') and commit['table_crc32']!=params['stellar_model_crc32']:
            raise ValueError('Stellar return journal table differs from frozen native model parameters')
        fingerprint=fp;model=commit['table_crc32'];ranks=commit['mpi_ranks'];parts=commit['parts']
        if not 1<=ranks<=1000000 or not isinstance(parts,list) or len(parts)!=ranks:raise ValueError('Incomplete stellar return MPI census')
        observed=0
        for rank,part in enumerate(parts):
            if set(part)!={'rank','rows','bytes','crc32','filename'} or part['rank']!=rank or any(type(part[k]) is not int or part[k]<0 for k in ('rank','rows','bytes','crc32')):
                raise ValueError('Invalid stellar return payload descriptor')
            if part['rows']==0:
                if part['filename'] is not None or part['bytes'] or part['crc32']:raise ValueError('Empty stellar rank claims a payload')
                continue
            name=f'stellar_returns_{sequence:020d}.rank_{rank:06d}.of_{ranks:06d}.tsv'
            if part['filename']!=name or not part['rows']<=part['bytes']<=part['rows']*4096:raise ValueError('Stellar return payload name/size mismatch')
            chunk=directory/name;regular(chunk);crc=0;nbytes=0;nrows=0
            with chunk.open('rb') as stream:
                while True:
                    line=stream.readline(4097)
                    if not line:break
                    if len(line)>4096 or not line.endswith(b'\n'):raise ValueError('Unbounded/truncated stellar return row')
                    nbytes+=len(line);crc=zlib.crc32(line,crc);nrows+=1
                    if nrows>part['rows']:raise ValueError('Undeclared stellar return rows')
                    tokens=line.decode('ascii').rstrip('\n').split('\t')
                    if len(tokens)!=40 or any(re.fullmatch('[0-9]+',x) is None for x in tokens[:4]):raise ValueError('Malformed/lossy stellar return row')
                    ids=[int(x) for x in tokens[:4]];values=np.array([float(x) for x in tokens[4:]])
                    if any(x>=2**64 for x in ids) or ids[2]==0 or ids[3] not in (0,4) or not np.isfinite(values).all() or values[0]!=time or values[1]<0 or values[2]<values[1]:
                        raise ValueError('Invalid stellar return identity/epoch')
                    identity=(time,ids[1],ids[2],ids[3],ids[0])
                    if identity in seen:raise ValueError('Duplicated stellar return operation/recipient')
                    seen.add(identity)
                    if values[3]<=0 or values[4]<=0 or values[11]<0 or values[12]<0 or np.any(values[13:]<0):raise ValueError('Negative stellar return mass/energy/elements')
                    row={'particle_id':ids[0],'source_id':ids[1],'source_counter':ids[2],'species':ids[3],
                        'native_time':float(time),'age_before_years':float(values[1]),'age_after_years':float(values[2]),
                        'mass_before_msun':float(values[3]*unit_mass),'mass_after_msun':float(values[4]*unit_mass),
                        'velocity_before_km_s':(values[5:8]*unit_velocity).tolist(),'velocity_after_km_s':(values[8:11]*unit_velocity).tolist(),
                        'internal_energy_before_cm2_s2':float(values[11]*unit_u),'internal_energy_after_cm2_s2':float(values[12]*unit_u),
                        'elements_before_msun':(values[13:24]*unit_mass).tolist(),'elements_after_msun':(values[24:35]*unit_mass).tolist(),
                        'injected_energy_erg':float(values[35]*unit_energy)}
                    rows.append(row)
            if (nbytes,nrows,crc&0xffffffff)!=(part['bytes'],part['rows'],part['crc32']):raise ValueError('Stellar return payload CRC/length/census mismatch')
            observed+=nrows
        if observed!=commit['event_count']:raise ValueError('Stellar return commit row census mismatch')
        count=commit['cumulative_count'];last_time=time;commits.append(commit)
    if count!=target:raise ValueError('Stellar return journal does not reach actual snapshot counter')
    for p,sha in sources.items():
        if digest(p)!=sha:raise ValueError('Stellar return source changed while reading')
    return rows,commits,sources


def audit_native_hydro_history(request,destination):
    """Public multi-epoch six-species/source reconciliation and gas history."""
    import json
    from pathlib import Path
    import h5py,numpy as np
    from halos import _snapshot_files
    from core.data import export_numeric_audit,snapshot_total_counts,coordinate_contract
    from core.foundation import file_sha256 as digest, product_provenance, MSUN_G
    required={'run_directory','snapshots','memory_gib','tolerances'}
    if not isinstance(request,dict) or set(request)!=required:raise ValueError('Declare complete native hydro-history request')
    selected=request['snapshots'];memory=request['memory_gib']
    if not isinstance(selected,list) or len(selected)<2 or len(set(selected))!=len(selected) or any(type(x) is not int or x<0 for x in selected):
        raise ValueError('Select at least two distinct snapshot storage indices in actual-time order')
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not np.isfinite(memory) or memory<=0:raise ValueError('Declare positive native hydro workspace')
    validate_native_hydro_tolerances(request['tolerances'])
    root=Path(request['run_directory']).resolve();budget=int(memory*1024**3);size=0;event_bound=0
    authority={str(p):digest(p) for p in (root/'manifest.json',root/'config.yaml') if p.is_file()}
    previous_count=None;previous_return_count=None;previous_bh_count=None
    for index in selected:
        files=_snapshot_files(root,index)
        if not files:raise ValueError('Missing native hydro snapshot')
        with h5py.File(files[0],'r') as f:
            size+=sum(int(x) for x in snapshot_total_counts(f['Header'].attrs))
            value=f['Header'].attrs.get('StellarBirthEventCount')
            if value is None or np.asarray(value).shape!=() or np.asarray(value).dtype!=np.dtype('uint64'):
                raise ValueError('Hydro history requires exact native event counters')
            count=int(value)
            if previous_count is not None:
                if count<previous_count:raise ValueError('Hydro event counter regressed')
                event_bound+=count-previous_count
            previous_count=count
            rvalue=f['Header'].attrs.get('StellarReturnEventCount',np.uint64(0))
            if np.asarray(rvalue).shape!=() or np.asarray(rvalue).dtype!=np.dtype('uint64'):
                raise ValueError('Hydro history requires exact stellar return counters')
            rcount=int(rvalue)
            if previous_return_count is not None:
                if rcount<previous_return_count:raise ValueError('Stellar return counter regressed')
                event_bound+=rcount-previous_return_count
            previous_return_count=rcount
            bvalue=f['Header'].attrs.get('BlackHoleEventCount',np.uint64(0))
            if np.asarray(bvalue).shape!=() or np.asarray(bvalue).dtype!=np.dtype('uint64'):raise ValueError('BH history requires exact event counters')
            bcount=int(bvalue)
            if previous_bh_count is not None:
                if bcount<previous_bh_count:raise ValueError('BH counter regressed')
                event_bound+=4*(bcount-previous_bh_count)
            previous_bh_count=bcount
    # Includes retained arrays, both current censuses, replay dictionaries,
    # event row parsing, sorting and publication buffers. This is a preflight
    # working-set estimate, not a measured RSS acceptance result.
    if size*4096+event_bound*16384+64*1024**2>budget:raise MemoryError('Complete hydro history/replay exceeds declared workspace')
    payload={};specs={};metadata=[];intervals=[];sources=dict(authority);input_files=[];previous=None;previous_meta=None
    for ordinal,index in enumerate(selected):
        census,meta,hashes,files=read_native_hydro_census(root,index,memory_bytes=budget)
        sources.update(hashes);input_files.extend(files);metadata.append(meta)
        if previous is not None:
            if meta['native_time']<=previous_meta['native_time'] or meta['parameters']!=previous_meta['parameters'] or meta['gas_adiabatic_index']!=previous_meta['gas_adiabatic_index']:
                raise ValueError('Hydro history must advance actual time with a fixed native model/units/seed')
            events,commits,event_sources=read_native_ssp_birth_journal(root/'data',start_commit=previous_meta['commit_count'],
                stop_commit=meta['commit_count'],start_count=previous_meta['event_count'],stop_count=meta['event_count'],
                native_time_interval=[previous_meta['native_time'],meta['native_time']],
                expected_parameters={key:meta['parameters'][key] for key in ('unit_mass_g','unit_time_s','unit_velocity_cm_s','unit_length_cm','hubble_param','factor_sn','comoving','seed')},memory_bytes=budget)
            sources.update(event_sources);events=native_ssp_event_physical_arrays(events,meta['parameters'])
            return_rows,return_commits,return_sources=read_native_stellar_return_journal(root/'data',before=previous_meta,after=meta,memory_bytes=budget)
            sources.update(return_sources)
            from black_holes import read_native_black_hole_journal,black_hole_merger_lineage
            bh_rows,bh_commits,bh_sources=read_native_black_hole_journal(root/'data',before=previous_meta,after=meta,memory_bytes=budget)
            sources.update(bh_sources)
            report,residuals=reconcile_native_hydro_species(previous,census,events,request['tolerances'],return_rows,bh_rows)
            from core.stellar_returns import reconcile_stellar_return_state
            source_state=reconcile_stellar_return_state(previous,census,return_rows,
                before_meta=previous_meta,after_meta=meta,tolerances=request['tolerances'])
            report['stellar_return_state_accounting']=source_state
            report['issues'].extend(source_state['issues'])
            if source_state['status']=='FAIL':report['status']='FAIL'
            from core.black_hole_ledger import reconcile_roundoff_state
            roundoff=reconcile_roundoff_state(previous,census,bh_rows)
            report['black_hole_accretion_integral_accounting']=roundoff
            report['issues'].extend(roundoff['issues'])
            if roundoff['status']=='FAIL':report['status']='FAIL'
            time_key='cosmic_time_gyr' if meta['parameters']['comoving'] else 'elapsed_time_gyr'
            t0=previous_meta['epoch'].get(time_key);t1=meta['epoch'].get(time_key)
            duration=None if t0 is None or t1 is None else (t1-t0)*1.e9
            if duration is not None and duration<=0:raise ValueError('Hydro proper-time interval must be positive')
            report.update(before_snapshot=previous_meta['snapshot'],after_snapshot=index,
                duration_years=duration,interval_locked_birth_sfr_msun_per_year=None if duration is None else float(events['locked_mass'].sum()/duration),
                sfr_convention='Stochastic retained birth mass per actual proper-time interval; native instantaneous SFR remains a separate observable',
                proper_clock_status='UNAVAILABLE' if duration is None else 'VERIFIED',
                native_build_fingerprints=sorted({c['source_fingerprint'] for c in commits}))
            report['black_hole_rows']=bh_rows
            report['black_hole_lineage']=black_hole_merger_lineage(bh_rows)
            report['black_hole_build_fingerprints']=sorted({c['source_fingerprint'] for c in bh_commits})
            report['stellar_return_rows']=return_rows
            report['stellar_return_build_fingerprints']=sorted({c['source_fingerprint'] for c in return_commits})
            intervals.append(report)
            for name,values in {**residuals,**{'event_'+k:v for k,v in events.items()}}.items():
                key=f'interval_{ordinal-1:03d}_{name}';payload[key]=values
                specs[key]={'units':'identity' if name.endswith('_ids') or name in {'event_ssp_id','event_parent_id'} else
                    'Msun km/s' if 'momentum' in name else 'km/s' if '_v_' in name else 'Msun' if 'mass' in name else
                    'native elapsed time' if name=='event_birth_time' and not meta['parameters']['comoving'] else 'dimensionless',
                    'quantity':name.replace('_',' '),'annotations':{'interval_snapshots':[previous_meta['snapshot'],index],
                    'source_event_epoch_dataset':f'interval_{ordinal-1:03d}_event_birth_time',
                    'native_elapsed_time_to_seconds':meta['parameters']['unit_time_s'] if not meta['parameters']['comoving'] else None}}
        for name,values in census.items():
            key=f'epoch_{ordinal:03d}_{name}';payload[key]=values
            unit=('identity' if name in {'particle_ids','SSPParentGasID','BHCompanionID','BHParentIDs'} else 'Msun' if name=='mass_msun' or name.endswith('Mass') else
                'g/cm^3' if name=='gas_density_g_cm3' else 'cm^2/s^2' if name=='gas_internal_energy_cm2_s2' else
                'Msun/yr' if name=='gas_sfr_msun_per_year' else 'proper kpc' if name=='positions_proper_kpc' else 'peculiar km/s' if name=='velocities_peculiar_km_s' else 'K' if name=='gas_temperature_kelvin' else 'erg' if name=='gas_chemistry_radiated_energy_erg' or name.endswith('ERG') else 'g cm^2/s' if name in {'BHBinaryAngularReservoirCGS','BHBinaryGWAngularMomentumCGS','BHSeedOrbitalAngularMomentumCGS'} else 'cm' if name.endswith('CM') else 'g' if name.endswith('GRAM') else 'native elapsed time' if name=='SSPBirthNativeTime' and not meta['parameters']['comoving'] else 'dimensionless')
            specs[key]={'units':unit,'quantity':name.replace('_',' '),'epoch':meta['epoch'],
                'missing_policy':'Gas values are meaningful only where the matching _available array is true; non-gas gas values are placeholders; version-zero SSP birth is unavailable'}
            from core.stellar_returns import RETURN_FIELDS as return_fields
            if name in return_fields:
                kind=return_fields[name][0]
                specs[key]['units']='Msun' if kind=='mass' else 'erg' if kind=='energy' else 'scale factor' if kind=='time' and meta['parameters']['comoving'] else 'native elapsed time' if kind=='time' else 'dimensionless'
                specs[key]['missing_policy']='Meaningful only where epoch stellar_return_state_available is true; zero placeholders do not imply measured source coverage'
            from core.black_hole_ledger import ACCRETION_FIELDS as roundoff_fields, ACCRETION_RAW_FIELDS as roundoff_raw
            if name in roundoff_fields or name in roundoff_raw:
                specs[key]['units']='dimensionless' if name==roundoff_fields[0] else 'native mass code' if name in roundoff_raw else 'Msun'
                specs[key]['missing_policy']='Version1 carries a measured source integral; version0 is legacy unavailable. Pending demand is not transferred mass.'
                if name in roundoff_raw:
                    specs[key]['annotations']={'native_mass_to_msun':meta['parameters']['unit_mass_g']/MSUN_G/(meta['parameters']['hubble_param'] if meta['parameters']['comoving'] else 1.)}
        previous=census;previous_meta=meta
    coordinates=coordinate_contract(position_units='proper kpc',origin='native box origin; measured periodic proper coordinates',
        periodic_wrapping='wrapped in native periodic box',velocity_convention='event peculiar km/s; no between-output momentum subtraction')
    provenance=product_provenance(root,algorithm=__file__,selection='Complete native six-species census and source-certified SSP birth history',
        source_files=input_files,coordinates=coordinates,snapshot=selected[-1],
        helper_sources=[Path(__file__).with_name('black_holes.py'),Path(__file__).with_name('halos.py'),
            Path(__file__).with_name('black_hole_ledger.py'),Path(__file__).with_name('stellar_returns.py')])
    namespace=provenance['identity_namespaces']['native_particle']
    for key,spec in specs.items():
        if spec['units']=='identity':spec['identity_namespace']=namespace
    algorithm=Path(__file__).resolve().parents[1]/'configs/hydro_audit.algorithms.json'
    sources[str(algorithm)]=digest(algorithm)
    for asset in provenance['algorithm_sources']:sources[asset['path']]=asset['sha256']
    for asset in [*provenance['initial_conditions'],provenance['executable'],provenance['parameter_file'],provenance['compile_options'],*provenance['boltzmann_inputs'].values()]:
        if asset.get('status')=='VERIFIED':sources[asset['path']]=asset['sha256']
    report={'schema':'native-hydro-history-v1','status':'PASS' if all(r['status']=='PASS' for r in intervals) else 'FAIL',
        'request':request,'epochs':metadata,'intervals':intervals,'source_hashes':sources,'provenance':provenance,
        'thermodynamic_scope':'Proper density, native internal/effective EOS energy and retained SH03 SFR; measured composition/electrons support single-phase ideal-gas temperature only. No multiphase temperature is inferred',
        'physics_acceptance':'NOT_ACCEPTED','missing_later_operators':'Unrecorded source changes fail reconciliation; SIDM-specific friction and non-quasi-circular binary physics remain outside this declared native profile',
        'source_origin':'synthetic_control' if (root/'fixture_manifest.json').is_file() or (root/'manifest.json').is_file() and json.loads((root/'manifest.json').read_text()).get('fixture') else 'native_snapshot_and_operator_journal'}
    for source,checksum in sources.items():
        if digest(source)!=checksum:raise ValueError('Native hydro history source changed before publication')
    return export_numeric_audit(destination,report,payload,specs,algorithm=algorithm,
        context={'epoch':metadata[-1]['epoch'],'provenance':provenance,'coordinates':coordinates},source_hashes=sources)


def verify_native_hydro_history(path):
    from pathlib import Path
    import numpy as np
    from core.data import verify_numeric_audit
    from core.foundation import file_sha256 as digest, verify_product_provenance
    report=verify_numeric_audit(path)
    if report.get('schema')!='native-hydro-history-v1':raise ValueError('Not a native hydro history')
    for source,checksum in report['source_hashes'].items():
        if digest(source)!=checksum:raise ValueError('Native hydro history source changed: '+source)
    verify_product_provenance(report['request']['run_directory'],report['provenance'])
    if len(report['epochs'])!=len(report['request']['snapshots']) or len(report['intervals'])!=len(report['epochs'])-1:
        raise ValueError('Incomplete hydro history report')
    def equal(a,b):
        if a.dtype!=b.dtype or a.shape!=b.shape:return False
        return np.array_equal(a,b,equal_nan=True) if a.dtype.kind=='f' else np.array_equal(a,b)
    # Verification reproduces the accounting decisions from the immutable
    # census/event arrays, without reinterpreting output row order as identity.
    with np.load(str(Path(path).resolve())+'.data/arrays.npz',allow_pickle=False) as arrays:
        previous=None
        for ordinal,meta in enumerate(report['epochs']):
            prefix=f'epoch_{ordinal:03d}_'
            census={name[len(prefix):]:arrays[name] for name in arrays.files if name.startswith(prefix)}
            if previous is not None:
                iprefix=f'interval_{ordinal-1:03d}_'
                eprefix=iprefix+'event_'
                events={name[len(eprefix):]:arrays[name] for name in arrays.files if name.startswith(eprefix)}
                stored=report['intervals'][ordinal-1]
                from black_holes import read_native_black_hole_journal,black_hole_merger_lineage
                root=Path(report['request']['run_directory']).resolve();memory=int(report['request']['memory_gib']*1024**3)
                actual_returns,_,_=read_native_stellar_return_journal(root/'data',before=report['epochs'][ordinal-1],after=meta,memory_bytes=memory)
                actual_bh,_,_=read_native_black_hole_journal(root/'data',before=report['epochs'][ordinal-1],after=meta,memory_bytes=memory)
                if actual_returns!=stored.get('stellar_return_rows',[]) or actual_bh!=stored.get('black_hole_rows',[]):raise ValueError('Hydro report source rows differ from committed native payloads')
                if black_hole_merger_lineage(actual_bh)!=stored.get('black_hole_lineage'):raise ValueError('Hydro report BH ancestry differs from native events')
                replay,residuals=reconcile_native_hydro_species(previous,census,events,report['request']['tolerances'],stored.get('stellar_return_rows',[]),stored.get('black_hole_rows',[]))
                from core.stellar_returns import reconcile_stellar_return_state
                source_state=reconcile_stellar_return_state(previous,census,actual_returns,
                    before_meta=report['epochs'][ordinal-1],after_meta=meta,tolerances=report['request']['tolerances'])
                replay['stellar_return_state_accounting']=source_state
                replay['issues'].extend(source_state['issues'])
                if source_state['status']=='FAIL':replay['status']='FAIL'
                for key,value in replay.items():
                    if stored.get(key)!=value:raise ValueError('Hydro accounting decision does not replay: '+key)
                for key,value in residuals.items():
                    name=iprefix+key
                    if name not in arrays or not equal(value,arrays[name]):raise ValueError('Hydro accounting residual does not replay: '+key)
                if stored['before_snapshot']!=report['epochs'][ordinal-1]['snapshot'] or stored['after_snapshot']!=meta['snapshot']:
                    raise ValueError('Hydro accounting interval/epoch mismatch')
            previous=census
    expected='PASS' if all(x['status']=='PASS' for x in report['intervals']) else 'FAIL'
    if report['status']!=expected or report.get('physics_acceptance')!='NOT_ACCEPTED':
        raise ValueError('Hydro aggregate status is inconsistent')
    return report

def load_stellar_yield_library(asset, *, expected_sha256):
    """Read pinned NuPyCEE total ejecta/lifetime data without executing upstream Python.

    The source table has one row per isotope and explicitly reports stellar
    lifetime and remnant mass. Signed net yields are not total ejecta and are
    deliberately refused here. The eleven element ordering is shared with
    native source state and Grackle's bulk-metal input.
    """
    import re
    import numpy as np
    from pathlib import Path
    from core.foundation import file_sha256 as digest
    path=Path(asset).resolve()
    if digest(path)!=expected_sha256:raise ValueError('Stellar yield asset checksum mismatch')
    elements=('H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe')
    result=[];current=None
    pattern=re.compile(r'^H Table: \(M=([^,]+),Z=([^\)]+)\)$')
    def commit():
        if current is None:return
        if set(current)!={'mass_msun','metallicity','lifetime_years','remnant_msun','elements_msun'}:
            raise ValueError('Incomplete stellar yield/lifetime record')
        m,z,t,r=(current[k] for k in ('mass_msun','metallicity','lifetime_years','remnant_msun'))
        if not np.isfinite([m,z,t,r]).all() or m<=0 or t<=0 or not 0<=z<=1 or not 0<=r<m:
            raise ValueError('Invalid stellar yield/lifetime coordinates')
        if not np.isfinite(current['elements_msun']).all() or np.any(current['elements_msun']<0):
            raise ValueError('Total ejecta require finite nonnegative element masses')
        if np.sum(current['elements_msun'])<=0:raise ValueError('Empty stellar ejecta record')
        result.append(current)
    with path.open() as stream:
        for line in stream:
            text=line.strip();match=pattern.match(text)
            if match:
                commit();current={'mass_msun':float(match[1]),'metallicity':float(match[2]),'elements_msun':np.zeros(11)}
            elif current is not None and text.startswith('H Lifetime:'):
                if 'lifetime_years' in current:raise ValueError('Duplicate stellar lifetime')
                current['lifetime_years']=float(text.split(':',1)[1])
            elif current is not None and text.startswith('H Mfinal:'):
                if 'remnant_msun' in current:raise ValueError('Duplicate stellar remnant')
                current['remnant_msun']=float(text.split(':',1)[1])
            elif current is not None and text.startswith('&'):
                tokens=[x.strip() for x in text.split('&') if x.strip()]
                if tokens and tokens[0]=='Isotopes':continue
                if len(tokens)<2:raise ValueError('Malformed stellar isotope record')
                symbol=tokens[0].split('-',1)[0]
                if symbol not in elements:raise ValueError('Unrepresented element in eleven-element total-ejecta table: '+symbol)
                value=float(tokens[1])
                if not np.isfinite(value) or value<0:raise ValueError('Signed net yields cannot be treated as total ejecta')
                current['elements_msun'][elements.index(symbol)]+=value
    commit()
    if not result or len({(r['metallicity'],r['mass_msun']) for r in result})!=len(result):
        raise ValueError('Empty or duplicate yield grid')
    if digest(path)!=expected_sha256:raise ValueError('Stellar yield asset changed while reading')
    return result


def stellar_imf_density(mass_msun, specification):
    """Mass-normalized dN/dm, with an explicitly selected IMF and limits."""
    import numpy as np
    from scipy.integrate import quad
    required={'kind','minimum_mass_msun','maximum_mass_msun'}
    if not isinstance(specification,dict) or set(specification)!=required:
        raise ValueError('Declare IMF family and normalization limits')
    low,high=(specification[k] for k in ('minimum_mass_msun','maximum_mass_msun'))
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in (low,high)) or not 0<low<high:
        raise ValueError('Invalid IMF mass range')
    kind=specification['kind']
    if kind not in {'salpeter','chabrier2003_single'}:raise ValueError('Unsupported IMF family')
    def raw(m):
        if kind=='salpeter':return m**-2.35
        # Chabrier 2003 single-star dN/dlog10(m) below 1 Msun;
        # continuous m^-2.3 dN/dm above it. Overall scale cancels.
        mu=np.log10(.079);sigma=.69
        return np.exp(-.5*((np.log10(m)-mu)/sigma)**2)/m if m<=1 else np.exp(-.5*(mu/sigma)**2)*m**-2.3
    norm=quad(lambda x: np.exp(2*x)*raw(np.exp(x)),np.log(low),np.log(high),
        epsabs=1.e-12,epsrel=1.e-10,points=[0.] if low<1<high else None,limit=200)[0]
    masses=np.asarray(mass_msun,dtype=float)
    if not np.isfinite(masses).all() or np.any(masses<=0):raise ValueError('IMF masses must be finite and positive')
    out=np.asarray([raw(float(m))/norm if low<=m<=high else 0. for m in masses.ravel()]).reshape(masses.shape)
    return out


def compile_stellar_return_model(request, destination):
    """Compile pinned yields into a flat native cumulative source table.

    Production never calls Python while stepping. This preparatory public
    stage writes one numeric .dat file and a sealed model manifest, both in
    the chosen existing directory. Every model choice is explicit; changing
    any choice produces a new model rather than silently reusing a table.
    """
    from pathlib import Path
    import json,zlib
    import numpy as np
    from core.foundation import file_sha256 as digest, atomic_text, atomic_json
    from core.native import native_run_lock
    required={'schema','yield_asset','snia_asset','imf','age_grid_years','mass_quadrature_points',
        'core_collapse_minimum_mass_msun','prompt_return_fraction','prompt_subtraction',
        'metallicity_boundary_policy','yield_mass_closure_relative_tolerance',
        'snia_delay','snia_events_per_formed_msun','snia_energy_erg','agb_specific_energy_erg_per_msun','initial_abundance_pattern'}
    if not isinstance(request,dict) or set(request)!=required or request['schema']!='stellar-return-model-v1':
        raise ValueError('Declare the complete stellar-return model')
    destination=Path(destination).resolve();table=destination.with_suffix('.dat')
    if destination.suffix!='.json' or not destination.parent.is_dir() or table==destination:
        raise ValueError('Use a .json destination in an existing directory')
    if request['prompt_subtraction']!='composition_proportional_ccsn_mass_only':
        raise ValueError('Declare the supported SH03 prompt-mass subtraction explicitly')
    if request['metallicity_boundary_policy'] not in {'reject','nearest_edge'}:
        raise ValueError('Declare out-of-table metallicity policy')
    abundance=request['initial_abundance_pattern']
    if not isinstance(abundance,dict) or set(abundance)!={'primordial_hydrogen_fraction','metal_element_fractions'}:
        raise ValueError('Declare the initial hydrogen and heavy-element pattern')
    hydrogen=abundance['primordial_hydrogen_fraction'];metals=np.asarray(abundance['metal_element_fractions'],dtype=float)
    if isinstance(hydrogen,bool) or not isinstance(hydrogen,(int,float)) or not np.isfinite(hydrogen) or not 0<hydrogen<1 or metals.shape!=(9,) or not np.isfinite(metals).all() or np.any(metals<0) or abs(metals.sum()-1)>1.e-12:
        raise ValueError('Invalid declared eleven-element abundance pattern')
    def number(name,minimum=0.,maximum=None,strict=False):
        value=request[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<minimum or strict and value==minimum or maximum is not None and value>maximum:
            raise ValueError('Invalid stellar model setting: '+name)
        return float(value)
    beta=number('prompt_return_fraction',maximum=1.,strict=True)
    if beta>=1:raise ValueError('SH03 prompt fraction must be below unity')
    cutoff=number('core_collapse_minimum_mass_msun',strict=True)
    closure=number('yield_mass_closure_relative_tolerance',maximum=.05)
    snia_norm=number('snia_events_per_formed_msun');snia_energy=number('snia_energy_erg');agb_energy=number('agb_specific_energy_erg_per_msun')
    points=request['mass_quadrature_points']
    if type(points) is not int or not 256<=points<=131072:raise ValueError('Bounded resolved mass quadrature required')
    ages=np.asarray(request['age_grid_years'],dtype=float)
    if ages.ndim!=1 or not 3<=len(ages)<=8192 or not np.isfinite(ages).all() or ages[0]!=0 or np.any(np.diff(ages)<=0):
        raise ValueError('Age grid must start at zero and increase strictly')
    assets={}
    for name in ('yield_asset','snia_asset'):
        asset=request[name]
        if not isinstance(asset,dict) or set(asset)!={'path','sha256'}:raise ValueError('Pin path and SHA256 for '+name)
        path=Path(asset['path']).expanduser().resolve()
        if digest(path)!=asset['sha256']:raise ValueError('Stellar source asset checksum mismatch: '+name)
        assets[name]={'path':str(path),'sha256':asset['sha256']}
    rows=load_stellar_yield_library(assets['yield_asset']['path'],expected_sha256=assets['yield_asset']['sha256'])
    elements=('H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe');snia=np.zeros(11)
    for line in Path(assets['snia_asset']['path']).read_text().splitlines():
        if not line.startswith('&'):continue
        tokens=[x.strip() for x in line.split('&') if x.strip()]
        if len(tokens)!=2:raise ValueError('Malformed pinned SNIa table')
        if tokens[0]=='Isotopes':continue
        symbol=tokens[0].split('-',1)[0];value=float(tokens[1])
        if not np.isfinite(value) or value<0:raise ValueError('SNIa total yields must be nonnegative')
        if symbol not in elements:
            if value!=0:raise ValueError('Unrepresented SNIa ejecta element')
        else:snia[elements.index(symbol)]+=value
    if snia.sum()<=0:raise ValueError('Empty SNIa yields')
    dtd=request['snia_delay']
    if not isinstance(dtd,dict) or set(dtd)!={'kind','minimum_years','maximum_years'} or dtd['kind']!='power_law_minus_one':
        raise ValueError('Declare a normalized bounded t^-1 SNIa delay-time distribution')
    tmin,tmax=(dtd[k] for k in ('minimum_years','maximum_years'))
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in (tmin,tmax)) or not 0<tmin<tmax:
        raise ValueError('Invalid SNIa delay window')
    ia_cdf=np.clip(np.log(np.maximum(ages,tmin)/tmin)/np.log(tmax/tmin),0.,1.)
    zs=np.array(sorted({r['metallicity'] for r in rows}));cumulative=np.zeros((len(zs),len(ages),3,13))
    diagnostics=[];population=np.zeros((len(zs),len(ages),2))
    imf=request['imf'];stellar_imf_density(np.array([1.]),imf)
    for zi,z in enumerate(zs):
        grid=sorted((r for r in rows if r['metallicity']==z),key=lambda r:r['mass_msun'])
        masses=np.array([r['mass_msun'] for r in grid]);lifetimes=np.array([r['lifetime_years'] for r in grid])
        remnants=np.array([r['remnant_msun'] for r in grid]);raw=np.array([r['elements_msun'] for r in grid])
        if np.any(np.diff(lifetimes)>=0):raise ValueError('Yield lifetimes must decrease monotonically with progenitor mass')
        returned=masses-remnants;error=np.abs(raw.sum(axis=1)-returned)/returned
        if np.any(error>closure):
            failed=[{'mass_msun':float(masses[i]),'metallicity':float(z),
                     'expected_return_msun':float(returned[i]),'represented_ejecta_msun':float(raw[i].sum()),
                     'relative_error':float(error[i])} for i in np.flatnonzero(error>closure)]
            raise ValueError('Eleven-element yield mass closure exceeds frozen tolerance: '+json.dumps(failed,sort_keys=True))
        # Keep the tabulated total ejecta exactly. Passing a closure tolerance
        # permits a declared source discrepancy; it does not authorize changing
        # all represented abundances to force exact mass closure.
        lower=max(masses[0],imf['minimum_mass_msun']);upper=imf['maximum_mass_msun']
        if upper>masses[-1] or lower>=upper or not lower<cutoff<upper:
            raise ValueError('IMF and channel range exceed yield coverage')
        if imf['minimum_mass_msun']<masses[0] and ages[-1]>=lifetimes[0]:
            raise ValueError('Requested ages require unresolved low-mass lifetime/yield data')
        # Midpoint quadrature in log mass. Channel boundary is an exact cell
        # boundary, avoiding artificial mixing of AGB and CCSN channels.
        edges=np.unique(np.r_[np.geomspace(lower,upper,points+1),cutoff])
        mass=np.sqrt(edges[:-1]*edges[1:]);weight=stellar_imf_density(mass,imf)*np.diff(edges)
        lifetime=np.exp(np.interp(np.log(mass),np.log(masses),np.log(lifetimes)))
        ejecta=np.stack([np.interp(np.log(mass),np.log(masses),raw[:,ei]) for ei in range(11)],axis=1)
        wd_remnant=np.interp(np.log(mass),np.log(masses),remnants)
        for channel,mask in ((0,mass<cutoff),(1,mass>=cutoff)):
            ordered=np.flatnonzero(mask)[np.argsort(lifetime[mask],kind='stable')]
            times=lifetime[ordered];contribution=ejecta[ordered]*weight[ordered,None]
            sums=np.vstack([np.zeros((1,11)),np.cumsum(contribution,axis=0)])
            values=sums[np.searchsorted(times,ages,side='right')]
            total=values.sum(axis=1)
            if channel==1:
                # SH03 already deposits beta * gross mass at birth. Exclude
                # that reservoir from delayed CCSN ejecta proportionally to
                # the declared total-ejecta composition, and exclude all
                # CCSN thermal energy already represented by the SH03 EOS.
                delayed=np.maximum(total-beta,0.)
                values*=np.divide(delayed,total,out=np.zeros_like(total),where=total>0)[:,None]
                total=delayed
            cumulative[zi,:,channel,0]=total;cumulative[zi,:,channel,1:12]=values
            if channel==0:cumulative[zi,:,channel,12]=agb_energy*total
        cumulative[zi,:,2,1:12]=ia_cdf[:,None]*snia_norm*snia[None,:]
        cumulative[zi,:,2,0]=cumulative[zi,:,2,1:12].sum(axis=1)
        cumulative[zi,:,2,12]=ia_cdf*snia_norm*snia_energy
        agb=mass<cutoff;ordered=np.flatnonzero(agb)[np.argsort(lifetime[agb],kind='stable')]
        remnant_cdf=np.r_[0.,np.cumsum(wd_remnant[ordered]*weight[ordered])]
        available=remnant_cdf[np.searchsorted(lifetime[ordered],ages,side='right')]
        if np.any(cumulative[zi,:,2,0]>available+1.e-12):raise ValueError('SNIa mass exceeds the evolved white-dwarf reservoir')
        if np.any(cumulative[zi,:,:,0].sum(axis=1)>=1-beta):raise ValueError('Delayed ejecta exhaust the SH03 retained SSP mass')
        order=np.argsort(lifetime,kind='stable');dead=np.r_[0.,np.cumsum(mass[order]*weight[order])]
        remnant=np.r_[0.,np.cumsum(wd_remnant[order]*weight[order])]
        ii=np.searchsorted(lifetime[order],ages,side='right')
        population[zi,:,0]=dead[ii]
        population[zi,:,1]=remnant[ii]-cumulative[zi,:,2,0]
        retained=1-beta-cumulative[zi,:,:,0].sum(axis=1)
        if np.any(population[zi,:,1]<-1.e-12) or np.any(population[zi,:,1]>retained+1.e-12):
            raise ValueError('Remnant/live partition exceeds the retained SSP budget')
        diagnostics.append({'metallicity':float(z),'maximum_yield_closure_relative_error':float(error.max()),
            'source_ejecta_preserved':True,'normalization_applied':False,
            'source_mass_closure_residual_msun':(raw.sum(axis=1)-returned).tolist(),
            'source_mass_grid_msun':masses.tolist(),
            'quadrature_cells':len(mass),'final_delayed_mass_per_gross_msun':float(cumulative[zi,-1,:,0].sum())})
    lines=[f'RESEARCH_STELLAR_TABLE_V2 {len(zs)} {len(ages)} 11 {beta:.17g} {int(request["metallicity_boundary_policy"]=="nearest_edge")}',
        ' '.join(format(x,'.17g') for x in zs),' '.join(format(x,'.17g') for x in ages),
        ' '.join(format(x,'.17g') for x in [hydrogen,1-hydrogen,*metals])]
    for zi in range(len(zs)):
        for ai in range(len(ages)):
            for channel in range(3):
                lines.append(f'{zi} {ai} {channel} '+' '.join(format(x,'.17g') for x in cumulative[zi,ai,channel]))
    for zi in range(len(zs)):
        for ai in range(len(ages)):
            lines.append(f'{zi} {ai} '+' '.join(format(x,'.17g') for x in population[zi,ai]))
    text='\n'.join(lines)+'\n';checksum=zlib.crc32(text.encode())&0xffffffff
    sources={str(Path(__file__).resolve()):digest(__file__),**{a['path']:a['sha256'] for a in assets.values()}}
    registry_path=Path(__file__).resolve().parents[1]/'configs/stellar_chemistry.upstream.json'
    registry=json.loads(registry_path.read_text());sources[str(registry_path)]=digest(registry_path)
    attribution={}
    for name,asset in assets.items():
        matches=[item for item in registry['assets'] if item['sha256']==asset['sha256']]
        attribution[name]=matches[0] if len(matches)==1 else {'status':'UNREGISTERED_EXTERNAL_DATASET','sha256':asset['sha256'],'attribution_requirement':'Retain original yield/lifetime publication and redistribution terms for this external table'}
        if len(matches)==1 and matches[0].get('license_asset'):
            license_path=Path(__file__).resolve().parents[1]/matches[0]['license_asset'];sources[str(license_path)]=matches[0]['license_sha256']

    report={'schema':'compiled-stellar-return-model-v1','request':request,'assets':assets,'dataset_attribution':attribution,'elements':list(elements),
        'population_grid':{'metallicities':zs.tolist(),'ages_years':ages.tolist(),'columns':['initial_mass_of_dead_stars_per_gross_msun','remnant_mass_after_snia_per_gross_msun'],'values':population.tolist()},
        'population_convention':'Live plus remnant equals actual retained SSP mass; gross initial formation mass and SH03 prompt reservoir remain separate. Population remnant evolution follows applied source age, not an invented post-hoc native transfer.',
        'channels':['agb','ccsn_after_prompt_subtraction','snia'],'age_interpolation':'linear proper age; finite closed grid',
        'metallicity_interpolation':'linear Z; declared edge policy; nearest edge at Z=0 is an approximation, not primordial yields',
        'table':{'path':str(table),'crc32':checksum},'diagnostics':diagnostics,'source_hashes':sources,
        'prompt_accounting':'SH03 prompt mass is never re-deposited; CCSN thermal energy remains in SH03; delayed AGB/SNIa feedback explicitly tabulated',
        'implementation_status':'COMPILED_MODEL','validation_status':'NOT_RUN','physics_acceptance':'NOT_ACCEPTED'}
    with native_run_lock(destination.parent,operation='compile-stellar-return-model'):
        if any(p.exists() for p in (destination,table,destination.with_name(destination.name+'.sha256'))):
            raise FileExistsError('Preserve existing stellar model generation')
        for p,sha in sources.items():
            if digest(p)!=sha:raise ValueError('Stellar model source changed before publication')
        atomic_text(table,text);report['table']['sha256']=digest(table)
        atomic_json(destination,report);atomic_text(destination.with_name(destination.name+'.sha256'),digest(destination)+'\n')
    return report


def read_stellar_return_model(path):
    """Verify a compiled flat model and load its native cumulative kernel."""
    from pathlib import Path
    import json,zlib
    import numpy as np
    from core.foundation import file_sha256 as digest
    path=Path(path).resolve();report=json.loads(path.read_text())
    if digest(path)!=path.with_name(path.name+'.sha256').read_text().strip() or report.get('schema')!='compiled-stellar-return-model-v1':
        raise ValueError('Invalid stellar model publication')
    table=Path(report['table']['path'])
    if table!=path.with_suffix('.dat') or table.is_symlink() or digest(table)!=report['table']['sha256']:
        raise ValueError('Stellar native table changed or relocated')
    raw=table.read_bytes()
    if zlib.crc32(raw)&0xffffffff!=report['table']['crc32']:raise ValueError('Stellar table CRC mismatch')
    lines=raw.decode('ascii').splitlines();header=lines[0].split()
    if len(header)!=6 or header[0] not in {'RESEARCH_STELLAR_TABLE_V1','RESEARCH_STELLAR_TABLE_V2'} or header[3]!='11':raise ValueError('Invalid stellar native table schema')
    nz,na=map(int,header[1:3])
    if not 1<=nz<=64 or not 3<=na<=8192 or len(lines)!=4+nz*na*(4 if header[0]=='RESEARCH_STELLAR_TABLE_V2' else 3):raise ValueError('Incomplete or excessive stellar table')
    if float(header[4])!=report['request']['prompt_return_fraction'] or int(header[5])!=int(report['request']['metallicity_boundary_policy']=='nearest_edge'):
        raise ValueError('Stellar native table/model mismatch')
    zs=np.array([float(x) for x in lines[1].split()]);ages=np.array([float(x) for x in lines[2].split()])
    values=np.zeros((nz,na,3,13))
    abundance=np.array([float(x) for x in lines[3].split()])
    declared=report['request']['initial_abundance_pattern']
    if not np.array_equal(abundance,np.asarray([declared['primordial_hydrogen_fraction'],1-declared['primordial_hydrogen_fraction'],*declared['metal_element_fractions']])):
        raise ValueError('Stellar native initial abundance pattern mismatch')
    for position,line in enumerate(lines[4:4+nz*na*3]):
        row=line.split();zi,ai,ci=map(int,row[:3]);expected=(position//(na*3),(position//3)%na,position%3)
        if len(row)!=16 or (zi,ai,ci)!=expected:raise ValueError('Stellar native table row order mismatch')
        values[zi,ai,ci]=[float(x) for x in row[3:]]
    if zs.shape!=(nz,) or ages.shape!=(na,) or not np.isfinite(zs).all() or np.any(np.diff(zs)<=0) or zs[0]<0 or zs[-1]>1 or ages[0]!=0 or not np.isfinite(ages).all() or np.any(np.diff(ages)<=0):
        raise ValueError('Invalid stellar table coordinate axes')
    if not np.isfinite(values).all() or np.any(values<0) or np.any(np.diff(values,axis=1)<-1.e-12):
        raise ValueError('Nonconservative or nonmonotonic stellar cumulative table')
    if not np.allclose(values[...,0],values[...,1:12].sum(axis=-1),rtol=1.e-12,atol=1.e-14):
        raise ValueError('Stellar table element/mass accounting mismatch')
    if header[0]=='RESEARCH_STELLAR_TABLE_V2':
        pg=report.get('population_grid',{});pvalues=np.asarray(pg.get('values'),dtype=float)
        if pvalues.shape!=(nz,na,2) or not np.isfinite(pvalues).all() or np.any(pvalues<0):raise ValueError('Invalid SSP remnant population grid')
        if pg.get('metallicities')!=zs.tolist() or pg.get('ages_years')!=ages.tolist():raise ValueError('Population age/Z coordinates differ from source table')
        for position,line in enumerate(lines[4+nz*na*3:]):
            row=line.split();zi,ai=map(int,row[:2])
            if len(row)!=4 or (zi,ai)!=(position//na,position%na) or not np.array_equal(np.asarray(row[2:],dtype=float),pvalues[zi,ai]):raise ValueError('Native/population table mismatch')
        if np.any(np.diff(pvalues[:,:,0],axis=1)<-1.e-12) or np.any(pvalues[:,0]!=0):raise ValueError('Dead-star population clock regressed')
        if np.any(pvalues[:,:,1]>1-float(header[4])-values[:,:,:,0].sum(axis=2)+1.e-12):raise ValueError('Population remnant exceeds retained source mass')
    return report,zs,ages,values


def stellar_return_interval(model, *, age_before_years,age_after_years,birth_metallicity,gross_birth_mass_msun):
    """Evaluate a physical age interval; interval differences telescope at restarts."""
    import numpy as np
    report,zs,ages,values=model
    numbers=[age_before_years,age_after_years,birth_metallicity,gross_birth_mass_msun]
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in numbers):raise ValueError('Finite stellar source coordinates required')
    if not 0<=age_before_years<=age_after_years<=ages[-1] or not 0<=birth_metallicity<=1 or gross_birth_mass_msun<=0:
        raise ValueError('Stellar interval exceeds declared physical domain')
    z=birth_metallicity
    extrapolated=z<zs[0] or z>zs[-1]
    if extrapolated and report['request']['metallicity_boundary_policy']=='reject':raise ValueError('Birth metallicity outside pinned yield grid')
    z=float(np.clip(z,zs[0],zs[-1]));right=min(max(int(np.searchsorted(zs,z,side='right')),1),len(zs)-1) if len(zs)>1 else 0
    left=max(right-1,0);fraction=(z-zs[left])/(zs[right]-zs[left]) if right!=left else 0.
    kernel=values[left]*(1-fraction)+values[right]*fraction
    def evaluate(age):
        i=min(max(int(np.searchsorted(ages,age,side='right')),1),len(ages)-1);f=(age-ages[i-1])/(ages[i]-ages[i-1])
        return kernel[i-1]*(1-f)+kernel[i]*f
    delta=(evaluate(age_after_years)-evaluate(age_before_years))*gross_birth_mass_msun
    if np.any(delta<-1.e-12*gross_birth_mass_msun):raise ValueError('Stellar source interval has negative cumulative increment')
    delta=np.maximum(delta,0.)
    return {'mass_msun':delta[:,0],'elements_msun':delta[:,1:12],'energy_erg':delta[:,12],
        'channels':report['channels'],'metallicity_edge_approximation':extrapolated,
        'prompt_mass_redeposited_msun':0.}


def stellar_population_partition(model, *, proper_age_years,birth_metallicity,gross_birth_mass_msun,current_mass_msun):
    """Mass components at the applied native source age; no gas transfer is inferred."""
    import numpy as np
    report,zs,ages,_=model
    if 'population_grid' not in report:raise ValueError('Legacy source table has no live/remnant population decomposition')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in (proper_age_years,birth_metallicity,gross_birth_mass_msun,current_mass_msun)):
        raise ValueError('Finite SSP population state required')
    if not 0<=proper_age_years<=ages[-1] or not 0<=birth_metallicity<=1 or gross_birth_mass_msun<=0 or current_mass_msun<=0:raise ValueError('Population state exceeds source model domain')
    if report['request']['metallicity_boundary_policy']=='reject' and not zs[0]<=birth_metallicity<=zs[-1]:raise ValueError('Population metallicity outside yield coverage')
    values=np.asarray(report['population_grid']['values']);perz=np.array([np.interp(proper_age_years,ages,row[:,1]) for row in values])
    remnant=float(np.interp(birth_metallicity,zs,perz))*gross_birth_mass_msun
    if remnant>current_mass_msun*(1+1.e-12):raise ValueError('SSP remnant exceeds actual current mass')
    live=max(0.,current_mass_msun-remnant);beta=report['request']['prompt_return_fraction']
    return {'initial_gross_mass_msun':gross_birth_mass_msun,'initial_locked_mass_msun':(1-beta)*gross_birth_mass_msun,
        'prompt_returned_mass_msun':beta*gross_birth_mass_msun,'current_mass_msun':current_mass_msun,
        'live_mass_msun':live,'remnant_mass_msun':remnant,'delayed_returned_mass_msun':(1-beta)*gross_birth_mass_msun-current_mass_msun,
        'population_age_years':proper_age_years,'metallicity_edge_approximation':not zs[0]<=birth_metallicity<=zs[-1]}


def deposit_stellar_ejecta(gas,source,returns,weights):
    """Conservative reference deposition used by public source-budget consumers.

    Gas state is mass, peculiar velocity, specific internal energy and eleven
    element masses. This function returns new arrays; it never mutates partial
    state on a failed budget. The native operator uses the same extensive
    mass/momentum/energy mixing, with MPI reductions for normalization.
    """
    import numpy as np
    from core.foundation import MSUN_G
    mass=np.asarray(gas['mass_msun'],dtype=float);vel=np.asarray(gas['velocity_km_s'],dtype=float)
    u=np.asarray(gas['specific_internal_energy_cm2_s2'],dtype=float);elements=np.asarray(gas['element_mass_msun'],dtype=float)
    weights=np.asarray(weights,dtype=float);n=len(mass)
    if mass.shape!=(n,) or vel.shape!=(n,3) or u.shape!=(n,) or elements.shape!=(n,11) or weights.shape!=(n,):raise ValueError('Aligned conservative gas state required')
    if any(not np.isfinite(x).all() for x in (mass,vel,u,elements,weights)) or np.any(mass<=0) or np.any(u<0) or np.any(elements<0) or np.any(weights<0):
        raise ValueError('Invalid gas deposition state')
    if not np.allclose(elements.sum(axis=1),mass,rtol=1.e-12,atol=1.e-14):raise ValueError('Gas element masses do not close')
    source_velocity=np.asarray(source['velocity_km_s'],dtype=float);current=float(source['current_mass_msun'])
    total=float(np.sum(returns['mass_msun']));ejecta=np.asarray(returns['elements_msun'],dtype=float).sum(axis=0);energy=float(np.sum(returns['energy_erg']))
    if source_velocity.shape!=(3,) or not np.isfinite(source_velocity).all() or not np.isfinite([current,total,energy]).all() or not 0<=total<current or energy<0 or ejecta.shape!=(11,) or not np.isfinite(ejecta).all() or np.any(ejecta<0):
        raise ValueError('Invalid stellar extensive source budget')
    if not np.isclose(ejecta.sum(),total,rtol=1.e-12,atol=1.e-14):raise ValueError('Stellar ejecta element masses do not close')
    norm=weights.sum()
    if total>0 or energy>0:
        if norm<=0:raise ValueError('No conservative recipient support; source clock must remain unchanged')
    fractions=weights/norm if norm>0 else weights.copy();dm=total*fractions;new_mass=mass+dm
    new_velocity=(mass[:,None]*vel+dm[:,None]*source_velocity)/new_mass[:,None]
    # Kinetic mixing is evaluated as a nonnegative relative-velocity form,
    # avoiding subtraction of large bulk kinetic energies.
    mixing=.5*mass*dm/new_mass*np.sum((vel-source_velocity)**2,axis=1)*1.e10
    new_u=(mass*u+energy/MSUN_G*fractions+mixing)/new_mass
    return {'gas':{'mass_msun':new_mass,'velocity_km_s':new_velocity,'specific_internal_energy_cm2_s2':new_u,
            'element_mass_msun':elements+fractions[:,None]*ejecta},
        'source_current_mass_msun':current-total,'source_velocity_km_s':source_velocity.copy(),
        'deposited_mass_msun':total,'deposited_energy_erg':energy,'mixing_heat_erg':float(mixing.sum()*MSUN_G)}


def stellar_evolution_native_parameters(settings, *, factor_sn):
    """Resolve the pinned native source model before a build or launch."""
    from pathlib import Path
    import numpy as np
    from core.foundation import file_sha256 as digest
    if not isinstance(settings,dict) or type(settings.get('enabled',False)) is not bool:
        raise ValueError('stellar_evolution requires a boolean enabled setting')
    if not settings.get('enabled',False):
        if set(settings)-{'enabled'}:raise ValueError('Disabled stellar sources accept only enabled=false')
        return {}
    keys={'enabled','model_manifest','model_sha256','kernel_radius_proper_kpc','source_fraction','maximum_count','maximum_subcycles'}
    if set(settings)!=keys:raise ValueError('Declare the complete native stellar source configuration')
    path=Path(settings['model_manifest']).expanduser().resolve()
    if digest(path)!=settings['model_sha256']:raise ValueError('Native stellar model manifest checksum mismatch')
    model=read_stellar_return_model(path)[0]
    if model['request']['prompt_return_fraction']!=factor_sn:raise ValueError('Native stellar model and SH03 prompt fraction differ')
    for key in ('kernel_radius_proper_kpc','source_fraction'):
        x=settings[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<=0:
            raise ValueError('Invalid native stellar setting: '+key)
    if settings['source_fraction']>.1:raise ValueError('Stellar source fraction must not exceed 0.1')
    for key,maximum in (('maximum_count',10000000),('maximum_subcycles',1000000)):
        if type(settings[key]) is not int or not 1<=settings[key]<=maximum:raise ValueError('Invalid native stellar allocation/iteration bound')
    native_path=model['table']['path']
    if len(native_path.encode())>=255 or any(c.isspace() for c in native_path):raise ValueError('Native stellar table path must fit a whitespace-free 254-byte parameter token')
    return {'StellarEvolutionTableFile':native_path,'StellarEvolutionTableCRC32':str(model['table']['crc32']),
        'StellarSourceRadiusProperKpc':settings['kernel_radius_proper_kpc'],'StellarSourceFraction':settings['source_fraction'],
        'StellarMaximumCount':settings['maximum_count'],'StellarMaximumSubcycles':settings['maximum_subcycles']}


def grackle_native_parameters(settings):
    """Pin the data and compiled library identity for native chemistry."""
    from pathlib import Path
    import math
    import zlib
    from core.foundation import file_sha256 as digest
    if not isinstance(settings,dict) or type(settings.get('enabled',False)) is not bool:raise ValueError('chemistry requires boolean enabled')
    if not settings.get('enabled',False):
        if set(settings)-{'enabled'}:raise ValueError('Disabled chemistry accepts only enabled=false')
        return {}
    required={'enabled','primordial_network','metal_cooling','uv_background','data_file','data_sha256','library_file','library_sha256','upstream_revision'}
    optional={'deuterium_hydrogen_number_ratio','initial_species_number_fractions'}
    if not required<=set(settings) or set(settings)-required-optional or settings['upstream_revision']!='af7939494ce65007887ada7b98d1813df6843346':
        raise ValueError('Declare the complete pinned Grackle 3.4.1 model and library')
    if type(settings['primordial_network']) is not int or settings['primordial_network'] not in (0,1,2,3):
        raise ValueError('Declare equilibrium, atomic, H2 or H2/HD primordial network')
    ratio=settings.get('deuterium_hydrogen_number_ratio',0.)
    if (isinstance(ratio,bool) or not isinstance(ratio,(float,int)) or not math.isfinite(ratio) or not 0<=ratio<=1e-3
            or settings['primordial_network']==3 and ratio==0
            or settings['primordial_network']!=3 and ratio!=0):
        raise ValueError('Network 3 requires an explicit positive primordial D/H number ratio; other networks require zero')
    species=('HI','HII','HeI','HeII','HeIII','HM','H2I','H2II','DI','DII','HDI')
    seed=settings.get('initial_species_number_fractions')
    if seed is None:
        if settings['primordial_network']>=2:
            raise ValueError('Molecular chemistry requires explicit initial H/He-relative number fractions and residual ionization')
        seed={name:float(name in {'HI','HeI'}) for name in species}
    if not isinstance(seed,dict) or set(seed)!=set(species) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in seed.values()):
        raise ValueError('Complete nonnegative finite primordial seed number fractions required')
    hsum=seed['HI']+seed['HII']+seed['HM']+2*(seed['H2I']+seed['H2II'])+seed['HDI']
    hesum=seed['HeI']+seed['HeII']+seed['HeIII'];dsum=seed['DI']+seed['DII']+seed['HDI']
    if abs(hsum-1)>1e-12 or abs(hesum-1)>1e-12 or abs(dsum-ratio)>1e-12:
        raise ValueError('Seed fractions normalize H nuclei to one, He nuclei to one, and D nuclei to the explicit D/H ratio')
    if settings['primordial_network']<2 and any(seed[name] for name in ('HM','H2I','H2II')):
        raise ValueError('Atomic/equilibrium networks cannot retain molecular seed species')
    if seed['HII']+seed['H2II']+seed['DII']<seed['HM']:
        raise ValueError('Primordial seed ions cannot support the declared negative-hydrogen charge')
    if type(settings['metal_cooling']) is not bool or type(settings['uv_background']) is not bool:raise ValueError('Declare Grackle cooling and UV switches explicitly')
    original_data=Path(settings['data_file']).expanduser();original_library=Path(settings['library_file']).expanduser()
    if original_data.is_symlink() or original_library.is_symlink():raise ValueError('Pin regular Grackle assets, not symbolic links')
    data=original_data.resolve();library=original_library.resolve()
    if len(str(data).encode())>=255 or any(c.isspace() for c in str(data)+str(library)):raise ValueError('Grackle asset paths must be whitespace-free native parameter/linker tokens')
    if data.is_symlink() or library.is_symlink() or digest(data)!=settings['data_sha256'] or digest(library)!=settings['library_sha256']:
        raise ValueError('Grackle data/library does not match pinned regular-file identity')
    with data.open('rb') as stream:
        if stream.read(8)!=b'\x89HDF\r\n\x1a\n':raise ValueError('Grackle cooling data must be HDF5')
        stream.seek(0);crc=0
        for chunk in iter(lambda:stream.read(1024**2),b''):crc=zlib.crc32(chunk,crc)
    return {'GrackleDataFile':str(data),'GrackleDataCRC32':str(crc&0xffffffff),
        'GracklePrimordialChemistry':settings['primordial_network'],'GrackleMetalCooling':int(settings['metal_cooling']),
        'GrackleUVBackground':int(settings['uv_background']),'GrackleDeuteriumToHydrogenRatio':float(ratio),
        'GrackleSeedElectron':0.,**{'GrackleSeed'+name:float(seed[name]) for name in species}}


# ========================================================================
# stellar_history: Trace saved stellar candidates through exact IDs and SUBFIND memberships.

import json
import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from core.data import (hdf5_scalar_attr, snapshot_id_shards, exact_uint64_ids,
                       native_unit_factor, coordinate_contract, run_epoch_contract, write_dataset_contract)
from halos import (_snapshot_files, _group_files, _read_catalogue_series,
                      _check_file_series, _unit_conversion)


def exact_ids(values):
    return exact_uint64_ids(values,allow_empty=False)


def bound_rows(indices, offsets, lengths, total):
    """Map selected stellar global indices to disjoint bound-subhalo spans."""
    offsets, lengths = np.asarray(offsets), np.asarray(lengths)
    if (offsets.shape != lengths.shape or offsets.ndim != 2 or offsets.shape[1] != 6
            or offsets.dtype.kind not in "iu" or lengths.dtype.kind not in "iu"
            or np.any(offsets < 0) or np.any(lengths < 0)):
        raise ValueError("SUBFIND needs aligned nonnegative integer six-type spans")
    spans = sorted((int(o[4]), int(o[4]) + int(n[4]), row)
                   for row, (o, n) in enumerate(zip(offsets, lengths)) if n[4])
    if any(end > total for _, end, _ in spans) or any(
            right[0] < left[1] for left, right in zip(spans, spans[1:])):
        raise ValueError("SUBFIND stellar spans overlap or exceed snapshot")
    result = np.full(len(indices), -1, dtype=np.int64)
    if spans:
        starts, stops, rows = map(np.asarray, zip(*spans))
        slots = np.searchsorted(starts, indices, side="right") - 1
        valid = slots >= 0
        valid[valid] &= indices[valid] < stops[slots[valid]]
        result[valid] = rows[slots[valid]]
    return result


def read_epoch(run_dir, snapshot, ids, chunk_size=250_000):
    """Read one epoch in bounded chunks, including stars outside all FOF hosts."""
    from analysis import validate_gadget_snapshot_set
    paths = _snapshot_files(Path(run_dir), snapshot)
    from core.foundation import file_sha256 as digest
    groups = _group_files(Path(run_dir), snapshot)
    input_hashes = {str(p.resolve()):digest(p) for p in paths+groups}
    validate_gadget_snapshot_set(paths, snapshot)
    _unit_conversion(paths)
    records, total = snapshot_id_shards(paths, 4)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    n = len(ids)
    present = np.zeros(n, dtype=bool)
    position, velocity = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    masses = np.full(n, np.nan)
    indices = np.full(n, -1, dtype=np.int64)
    with h5py.File(paths[0], "r") as source:
        hdr, params = source["Header"].attrs, source["Parameters"].attrs
        a = hdf5_scalar_attr(hdr["Time"], "Time")
        box = hdf5_scalar_attr(hdr["BoxSize"], "BoxSize")
        h = float(params["HubbleParam"])
    if not np.isfinite([a, box, h]).all() or min(a, box, h) <= 0:
        raise ValueError("Stellar histories require positive finite scale factor, box and h")
    for record in records:
        if not record["count"]:
            continue
        with h5py.File(record["path"], "r") as source:
            stars = source["PartType4"]
            if "Velocities" not in stars:
                raise ValueError("Stellar histories require saved Velocities")
            for start in range(0, record["count"], chunk_size):
                raw = np.asarray(stars["ParticleIDs"][start:start+chunk_size])
                if raw.dtype.kind not in "iu" or np.any(raw < 0):
                    raise ValueError("Invalid snapshot stellar IDs")
                raw = exact_uint64_ids(raw)
                slots = np.searchsorted(sorted_ids, raw)
                local = np.flatnonzero(slots < n)
                local = local[sorted_ids[slots[local]] == raw[local]]
                target = order[slots[local]]
                if np.any(present[target]) or len(np.unique(target)) != len(target):
                    raise ValueError("Selected stellar ID occurs more than once in snapshot")
                if not len(local):
                    continue
                rows = start + local
                position[target] = stars["Coordinates"][rows] * native_unit_factor('position_comoving_kpc',scale_factor=a,hubble_param=h)
                velocity[target] = stars["Velocities"][rows] * native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h)
                mass_table = source["Header"].attrs["MassTable"]
                masses[target] = (stars["Masses"][rows] if "Masses" in stars
                                  else np.full(len(rows), mass_table[4])) * native_unit_factor('mass_msun',scale_factor=a,hubble_param=h)
                indices[target] = record["start"] + rows
                present[target] = True
    if (not np.isfinite(position[present]).all() or not np.isfinite(velocity[present]).all()
            or not np.isfinite(masses[present]).all() or np.any(masses[present] <= 0)):
        raise ValueError("Saved selected stellar phase space/masses are invalid")
    membership = np.full(n, -2, dtype=np.int64)
    membership[present] = -1
    catalogue_available = bool(groups)
    if groups:
        _check_file_series(groups, "Header", "Nsubhalos_Total")
        _unit_conversion(paths + groups)
        for path in groups:
            with h5py.File(path, "r") as source:
                hdr, params = source["Header"].attrs, source["Parameters"].attrs
                values = (hdf5_scalar_attr(hdr["Time"], "Time"),
                          hdf5_scalar_attr(hdr["BoxSize"], "BoxSize"),
                          float(params["HubbleParam"]))
                if not np.allclose(values, (a, box, h), rtol=1e-10, atol=0):
                    raise ValueError("SUBFIND and particle snapshot epochs/box/h disagree")
        offsets = _read_catalogue_series(groups, "Subhalo", "SubhaloOffsetType")
        lengths = _read_catalogue_series(groups, "Subhalo", "SubhaloLenType")
        membership[present] = bound_rows(indices[present], offsets, lengths, total)
    else:
        membership[present] = -3
    metadata = {"snapshot": int(snapshot), "scale_factor": a, "redshift": 1/a-1,
                "h": h, "box_comoving_kpc": box/h, "catalogue_available": catalogue_available,
                "source_files": [str(p.resolve()) for p in paths + groups]}
    metadata['epoch_contract'] = run_epoch_contract(run_dir,int(snapshot),a)
    metadata['coordinate_contract'] = coordinate_contract(position_units='comoving kpc',
        velocity_convention='peculiar km/s',origin='global simulation Cartesian box origin',
        periodic_wrapping='positions wrapped to [0, box); no inferred unwrapped trajectory')
    if (sorted(input_hashes) != sorted(str(p.resolve()) for p in _snapshot_files(Path(run_dir),snapshot)+_group_files(Path(run_dir),snapshot))
            or any(digest(p)!=sha for p,sha in input_hashes.items())):
        raise RuntimeError('Stellar epoch source files changed during reading')
    metadata['source_sha256'] = input_hashes
    return position, velocity, masses, present, membership, metadata


from core.native import native_run_owner, directory_owner


@native_run_owner
@directory_owner('output_directory',default_relative='results/stellar_histories')
def trace_stellar_candidates(run_dir, *, catalogue, snapshots, candidate_ids,
                             memory_gib=1.0, output_directory=None):
    """Write exact saved histories and a measured multi-epoch diagnostic plate.

    Membership sentinels: -1 unbound, -2 absent star, -3 unavailable catalogue.
    Unbound only means outside SUBFIND bound spans, not unbound to the host.
    Periodic positions are saved wrapped. No interpolation or orbit integration
    is performed here, and no automatic 'proven tidal stream' label is emitted.
    """
    from commands import integer_value
    from core.foundation import product_provenance, publish_directory, file_sha256 as digest, verify_product_provenance
    run_dir, catalogue = Path(run_dir).resolve(), Path(catalogue).resolve()
    if not isinstance(snapshots, list) or len(snapshots) < 2:
        raise ValueError("History requires at least two explicitly selected snapshots")
    snapshots = [integer_value(s, "snapshot", 0, 2147483647) for s in snapshots]
    if len(set(snapshots)) != len(snapshots):
        raise ValueError("Snapshot selection contains duplicates")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError("Select one or more saved candidate IDs")
    candidate_ids = [integer_value(c, "candidate ID", 1, 2**64-1) for c in candidate_ids]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate selection contains duplicates")
    if isinstance(memory_gib, bool) or not isinstance(memory_gib, (int, float)) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError("memory_gib must be finite and positive")
    catalogue_hash=digest(catalogue)
    from core.data import verify_hdf5_dataset_payloads
    verify_hdf5_dataset_payloads(catalogue)
    with h5py.File(catalogue, "r") as source:
        if source["ParticleIDs"].size*64 + 64*1024**2 > memory_gib*1024**3:
            raise MemoryError("Candidate source exceeds allocation cap; increase memory_gib")
        provenance = json.loads(source.attrs["provenance_json"])
        all_ids = exact_ids(source["ParticleIDs"][:])
        labels = np.asarray(source["CandidateID"][:])
        if labels.shape != all_ids.shape or labels.dtype.kind not in "iu" or np.any(labels < 0):
            raise ValueError("CandidateID must align with exact source ParticleIDs")
        missing = set(candidate_ids) - set(map(int, np.unique(labels)))
        if missing:
            raise ValueError("Candidate IDs do not occur in catalogue: " + str(sorted(missing)))
        selection = np.isin(labels, np.asarray(candidate_ids, dtype=np.uint64))
        ids, labels = all_ids[selection], labels[selection]
    # Each epoch is streamed into HDF5; this cap covers selected arrays, sorting,
    # plotting copies, and 250k-particle read blocks. It is not an RSS guarantee.
    if len(ids)*256 + 64*1024**2 > memory_gib*1024**3:
        raise MemoryError("Selected histories exceed allocation cap; increase memory_gib")
    destination = Path(output_directory).resolve() if output_directory else run_dir / "results" / "stellar_histories"
    if destination.exists():
        raise FileExistsError("History destination already exists: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "darkmattersim-stellar-history-v1", "catalogue": str(catalogue),
               'science_acceptance':'NOT_ACCEPTED',
               "candidate_provenance": provenance, "candidate_ids": candidate_ids,
               "selected_particles": len(ids), "epochs": [], "diagnostics": [],
               "units": {"Coordinates": "wrapped comoving kpc", "Velocities": "peculiar km/s", "Masses": "Msun"},
               "membership_sentinels": {"-1": "outside all SUBFIND bound subhaloes",
                                        "-2": "star not saved at epoch", "-3": "SUBFIND unavailable"},
               "interpretation": "Exact member histories; progenitor row numbers are epoch-local. Tidal origin, encounters, stream gaps, resolution convergence and emitted light require further evidence."}
    stage=Path(tempfile.mkdtemp(prefix='.stellar-history-',dir=destination.parent))
    from core.foundation import atomic_json
    try:
        previous_a = -1
        reference = None
        with h5py.File(stage / "history.hdf5", "w") as output:
            output.create_dataset('ParticleIDs',data=ids,chunks=True,fletcher32=True)
            output.create_dataset('CandidateID',data=labels.astype(np.uint64),chunks=True,fletcher32=True)
            source_hashes = [{'path':str(catalogue),'sha256':catalogue_hash}]
            for snapshot in snapshots:
                xyz, vel, mass, present, member, meta = read_epoch(run_dir, snapshot, ids)
                a = meta["scale_factor"]
                if a <= previous_a:
                    raise ValueError("Snapshots must be selected in strictly increasing scale factor order")
                current = (meta["h"], meta["box_comoving_kpc"])
                if reference is not None and not np.allclose(current, reference, rtol=1e-10, atol=0):
                    raise ValueError("History epochs disagree about box or Hubble parameter")
                previous_a, reference = a, current
                epoch = output.create_group(f"Epochs/{snapshot:03d}")
                epoch.attrs["metadata_json"] = json.dumps(meta, allow_nan=False)
                source_hashes.extend({'path':p,'sha256':sha} for p,sha in meta['source_sha256'].items())
                for key, values in (("Coordinates", xyz), ("Velocities", vel), ("Masses", mass),
                                    ("Present", present), ("BoundSubhaloRow", member)):
                    dataset=epoch.create_dataset(key, data=values, compression="gzip", shuffle=True, fletcher32=True)
                    write_dataset_contract(dataset,units={'Coordinates':'comoving kpc','Velocities':'km/s','Masses':'Msun',
                        'Present':'dimensionless','BoundSubhaloRow':'epoch-local SUBFIND row'}[key],
                        coordinates=meta['coordinate_contract'] if key in ('Coordinates','Velocities') else None,
                        epoch=meta['epoch_contract'],provenance={'source_sha256':meta['source_sha256']},
                        quantity={'Coordinates':'saved global simulation positions','Velocities':'saved physical peculiar velocities',
                            'Masses':'saved star-population dynamical mass','Present':'selected exact ID occurs at epoch',
                            'BoundSubhaloRow':'native epoch-local bound membership; no persistent ancestry'}[key],
                        missing_policy='NaN for absent particles; membership sentinels declared in provenance')
                summary["epochs"].append(meta)
                for candidate in candidate_ids:
                    selected = labels == candidate
                    available = selected & present
                    bound = member[available & (member >= 0)]
                    rows, counts = np.unique(bound, return_counts=True)
                    dominant = int(rows[np.argmax(counts)]) if len(rows) else None
                    number = int(counts.max()) if len(counts) else 0
                    summary["diagnostics"].append({"snapshot": snapshot, "candidate_id": candidate,
                        "selected_count": int(selected.sum()), "present_count": int(available.sum()),
                        "outside_bound_subhaloes_count": int(np.sum(selected & (member == -1))),
                        "dominant_bound_subhalo_row": dominant, "dominant_bound_count": number,
                        "dominant_fraction_of_selected": number/int(selected.sum()) if meta["catalogue_available"] else None})
            # A wrong-run source or an incomplete final epoch must not silently
            # produce plausible-looking histories of unrelated integer IDs.
            source_snapshot = integer_value(provenance["snapshot"], "source snapshot", 0, 2147483647)
            if source_snapshot not in snapshots:
                raise ValueError("History must include the candidate catalogue's source snapshot")
            source_epoch = output[f"Epochs/{source_snapshot:03d}"]
            source_meta = json.loads(source_epoch.attrs["metadata_json"])
            if not np.allclose((source_meta["scale_factor"], source_meta["h"]),
                               (provenance["scale_factor"], provenance["h"]), rtol=1e-10, atol=0):
                raise ValueError("Candidate source epoch/h does not match selected run")
            if not np.all(source_epoch["Present"][:]):
                raise ValueError("Candidate IDs are missing from their source snapshot")
            expected_xyz = np.asarray(provenance["host_center_comoving_kpc_h"]) / provenance["h"]
            with h5py.File(catalogue, "r") as source:
                expected_xyz = (source["Coordinates"][:][selection] / provenance["scale_factor"] + expected_xyz) % reference[1]
            actual_xyz = source_epoch["Coordinates"][:]
            delta = (actual_xyz - expected_xyz + reference[1]/2) % reference[1] - reference[1]/2
            if not np.allclose(delta, 0, atol=1e-5, rtol=0):
                raise ValueError("Candidate positions do not match the selected source run")
            from stars import add_stream_diagnostics, plot_stream_diagnostics
            diagnostics = add_stream_diagnostics(output, summary, run_dir, memory_gib=memory_gib)
            summary["stream_diagnostics_file"] = 'stars.json'
            from evolution import history_cadence_support
            summary['event_cadence'] = history_cadence_support(summary['epochs'])
            (stage / 'stars.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+"\n")
            output.create_dataset("ProvenanceJSON", data=json.dumps(summary, allow_nan=False),
                                  dtype=h5py.string_dtype("utf-8"))
            # Keep the old compact attribute for existing readers; large epoch
            # lists use the dataset to avoid HDF5 attribute size failures.
            encoded = json.dumps(summary, allow_nan=False)
            if len(encoded.encode("utf-8")) < 60000:
                output.attrs["provenance_json"] = encoded
            summary['data_contract'] = product_provenance(run_dir,algorithm=__file__,
                selection='Exact selected stellar candidate IDs across declared saved epochs',
                source_files=source_hashes,coordinates=meta['coordinate_contract'],
                helper_sources=[Path(__file__).with_name('evolution.py')])
            output.attrs['data_contract_json'] = json.dumps(summary['data_contract'],allow_nan=False)
            for name,kind in (('ParticleIDs','star_population'),('CandidateID','progenitor')):
                write_dataset_contract(output[name],units='identity',identity_namespace=summary['data_contract']['identity_namespaces'][kind])
        summary['history_sha256'] = digest(stage/'history.hdf5')
        (stage / "provenance.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (stage/'provenance.sha256').write_text(digest(stage/'provenance.json')+'\n')
        history_figure(stage, summary)
        plot_stream_diagnostics(stage, summary, diagnostics)
        verify_product_provenance(run_dir,summary['data_contract'])
        publish_directory(stage, destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return {"directory": str(destination), "catalogue": str(destination / "history.hdf5"),
            "figure": str(destination / 'stellar_history.png'), "selected_particles": len(ids),
            "stream_diagnostics": str(destination / 'stars.json'),
            "stream_figure": str(destination / 'stream_diagnostics.png')}


def history_figure(directory, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.data import verify_hdf5_dataset_payloads
    verify_hdf5_dataset_payloads(Path(directory)/'history.hdf5')
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    with h5py.File(Path(directory) / "history.hdf5", "r") as source:
        labels = source["CandidateID"][:]
        for candidate in summary["candidate_ids"]:
            diagnostics = [d for d in summary["diagnostics"] if d["candidate_id"] == candidate]
            a = [m["scale_factor"] for m in summary["epochs"]]
            axes[0, 0].plot(a, [d["present_count"] / d["selected_count"] for d in diagnostics], "o-", label=f"Candidate {candidate}")
            fractions = [d["dominant_fraction_of_selected"] if d["dominant_fraction_of_selected"] is not None else np.nan for d in diagnostics]
            axes[0, 1].plot(a, fractions, "o-")
            for meta in (summary["epochs"][0], summary["epochs"][-1]):
                epoch = source[f"Epochs/{meta['snapshot']:03d}"]
                selected = (labels == candidate) & epoch["Present"][:]
                xyz = epoch["Coordinates"][:][selected]
                if not len(xyz):
                    continue
                # Epoch-local periodic frame, centered on the first selected
                # star. Covariance extent is ambiguous if width exceeds L/2.
                box = meta["box_comoving_kpc"]
                xyz = ((xyz - xyz[0] + box/2) % box - box/2) * meta["scale_factor"]
                xyz -= xyz.mean(axis=0)
                axis = axes[1, 0] if meta is summary["epochs"][0] else axes[1, 1]
                axis.scatter(xyz[:, 0], xyz[:, 1], s=3, alpha=.5, rasterized=True)
    axes[0, 0].set(title="Saved fraction of selected stars", xlabel="Scale factor a", ylabel="Fraction")
    axes[0, 1].set(title="Largest bound-subhalo overlap at each epoch", xlabel="Scale factor a", ylabel="Fraction of selected stars")
    axes[0, 0].legend()
    for axis in axes[0]:
        axis.set_ylim(-.03, 1.03)
        axis.grid(alpha=.2)
    for axis, label in zip(axes[1], ("First selected output", "Last selected output")):
        axis.set(title=label + " • epoch-local particle frame", xlabel="Relative proper x [kpc]", ylabel="Relative proper y [kpc]")
        axis.set_aspect("equal", adjustable="datalim")
    prefix = "SYNTHETIC FIXTURE • " if summary["candidate_provenance"].get("fixture") else ""
    figure.suptitle(prefix + "Exact stellar histories • saved outputs only • tidal origin requires further evidence")
    from figures import export_figure
    export_figure(figure,Path(directory)/'stellar_history',formats=('png','pdf'),dpi=200,close=False,
        metadata={'science_acceptance':'NOT_ACCEPTED','source_sha256':summary['history_sha256'],
            'epochs':[epoch['epoch_contract'] for epoch in summary['epochs']],
            'data_contract':summary['data_contract'],'axes_units':[['scale factor','fraction'],['proper kpc','proper kpc']]})
    plt.close(figure)


def stellar_ownership_queries(graph_directory, *, star_namespace, exact_star_ids, snapshots):
    """Read explicit bound/unbound/missing status from the shared exact graph."""
    from halos import IdentityQueries
    queries=IdentityQueries(graph_directory);ids=exact_ids(exact_star_ids)
    return [{'snapshot':snapshot,'particle_id':int(pid),**queries.bound_owner((star_namespace,int(pid),snapshot))}
            for snapshot in snapshots for pid in ids]

# ========================================================================
# stream_diagnostics: Saved-epoch candidate morphology and exact membership transitions.
from scipy.spatial import cKDTree


def candidate_morphology(xyz, velocity, mass, box, a):
    if len(xyz) < 3:
        return {"status": "UNAVAILABLE", "reason": "fewer than three saved members"}
    if (not np.isfinite(xyz).all() or not np.isfinite(velocity).all()
            or not np.isfinite(mass).all() or np.any(mass <= 0)):
        raise ValueError("Candidate phase space/masses are invalid")
    delta = (xyz - xyz[0] + box/2) % box - box/2
    if np.any(np.ptp(delta, axis=0) >= box/2):
        return {"status": "UNAVAILABLE", "reason": "candidate spans an ambiguous periodic frame"}
    weights = mass / mass.sum()
    center_offset = weights @ delta
    offsets = (delta - center_offset)*a
    tensor = (offsets.T * weights) @ offsets
    values, axes = np.linalg.eigh(tensor)
    order = np.argsort(values)[::-1]
    values, axes = np.maximum(values[order], 0), axes[:, order]
    mean_velocity = weights @ velocity
    dv = velocity - mean_velocity
    covariance = (dv.T * weights) @ dv
    return {"status": "COMPLETE", "member_count": len(xyz), "mass_msun": float(mass.sum()),
            "center_comoving_kpc": ((xyz[0]+center_offset) % box).tolist(),
            "mean_velocity_km_s": mean_velocity.tolist(),
            "principal_axes_columns": axes.tolist(), "rms_axes_proper_kpc": np.sqrt(values).tolist(),
            "b_over_a": float(np.sqrt(values[1]/values[0])) if values[0] > 0 else None,
            "c_over_a": float(np.sqrt(values[2]/values[0])) if values[0] > 0 else None,
            "velocity_covariance_km_s2": covariance.tolist(),
            "velocity_rms_km_s": float(np.sqrt(np.trace(covariance))),
            "definition": "mass-weighted covariance about candidate centroid; RMS axes are not full length or physical stream width"}


def membership_transitions(left, right):
    """Use common particles, never equality of epoch-local subhalo rows."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Membership arrays must align")
    valid = (left >= -1) & (right >= -1)
    return {"comparable": valid, "bound_to_outside": valid & (left >= 0) & (right == -1),
            "outside_to_bound": valid & (left == -1) & (right >= 0),
            "bound_at_both": valid & (left >= 0) & (right >= 0)}


def saved_subhalo_proximity(run_dir, snapshot, xyz, occupied_rows, box, a, h):
    """Nearest saved centre to a selected member; no flyby/gap interpretation."""
    from halos import _group_files, _read_catalogue_series
    paths = _group_files(Path(run_dir), snapshot)
    if not paths:
        return {"status": "UNAVAILABLE", "reason": "SUBFIND unavailable"}
    pos = _read_catalogue_series(paths, "Subhalo", "SubhaloPos", required=False)
    mass = _read_catalogue_series(paths, "Subhalo", "SubhaloMass", required=False)
    if pos is None or mass is None:
        return {"status": "UNAVAILABLE", "reason": "saved subhalo centres/masses unavailable"}
    if pos.shape != (len(mass), 3) or not np.isfinite(pos).all() or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("Invalid subhalo proximity input")
    if not len(xyz):
        return {"status": "UNAVAILABLE", "reason": "no saved candidate members"}
    properties = {}
    for name in ("SubhaloVel", "SubhaloHalfmassRad", "SubhaloGroupNr"):
        properties[name] = _read_catalogue_series(paths, "Subhalo", name, required=False)
    progenitors = []
    for row in occupied_rows:
        if row < 0 or row >= len(mass):
            raise ValueError("Bound membership row exceeds subhalo properties")
        item = {"subhalo_row": int(row), "bound_mass_msun": float(mass[row])*1e10/h,
                "center_comoving_kpc": ((pos[row]/h) % box).tolist()}
        velocity = properties["SubhaloVel"]
        item["velocity_km_s"] = velocity[row].tolist() if velocity is not None else None
        radius = properties["SubhaloHalfmassRad"]
        item["half_mass_radius_proper_kpc"] = float(radius[row])*a/h if radius is not None else None
        hosts = properties["SubhaloGroupNr"]
        item["host_fof_row"] = int(hosts[row]) if hosts is not None else None
        progenitors.append(item)
    tree = cKDTree(xyz % box, boxsize=box)
    distance, nearest = tree.query((pos/h) % box)
    rows = np.flatnonzero(~np.isin(np.arange(len(mass)), occupied_rows))
    rows = rows[np.argsort(distance[rows], kind="stable")[:10]]
    return {"status": "COMPLETE", "subhalos_considered": len(mass),
            "bound_subhalo_properties": progenitors,
            "excluded_rows_containing_selected_stars": list(map(int, occupied_rows)),
            "closest_other_subhalos": [{"subhalo_row": int(row),
                "bound_mass_msun": float(mass[row])*1e10/h,
                "nearest_member_index": int(nearest[row]),
                "distance_proper_kpc": float(distance[row])*a} for row in rows],
            "interpretation": "Closest ten saved centres to candidate particles. Includes luminous and dark objects. Centre distance is not impact parameter; no between-output flyby is inferred."}


def add_stream_diagnostics(output, summary, run_dir, *, memory_gib):
    ids, labels = output["ParticleIDs"][:], output["CandidateID"][:]
    diagnostics = {"schema": "darkmattersim-saved-stream-diagnostics-v1", "candidates": [], "intervals": [],
                   "interpretation": "Candidates require tidal-origin and convergence evidence. Overlap links use only selected stars, not complete subhalo membership. Transition times are bounded by adjacent saved epochs."}
    previous = None
    for meta in summary["epochs"]:
        # Bound catalogue arrays and the periodic tree share the history cap.
        # The summary/JSON headroom includes all candidate/interval records.
        from halos import _group_files
        catalogue_rows = 0
        for path in _group_files(Path(run_dir), meta["snapshot"]):
            with h5py.File(path, "r") as source:
                catalogue_rows += int(source["Header"].attrs.get("Nsubhalos_ThisFile", 0))
        headroom = 64*1024**2 + len(summary["epochs"])*len(summary["candidate_ids"])*16384
        if len(ids)*384 + catalogue_rows*160 + headroom > memory_gib*1024**3:
            raise MemoryError("Saved stream diagnostics exceed history allocation cap; increase memory_gib")
        epoch = output[f"Epochs/{meta['snapshot']:03d}"]
        xyz, vel, mass = (epoch[name][:] for name in ("Coordinates", "Velocities", "Masses"))
        present, member = epoch["Present"][:], epoch["BoundSubhaloRow"][:]
        for candidate in summary["candidate_ids"]:
            selected = labels == candidate
            available = selected & present
            occupied, counts = np.unique(member[available & (member >= 0)], return_counts=True)
            diagnostics["candidates"].append({"snapshot": meta["snapshot"], "candidate_id": candidate,
                "bound_member_overlaps": [{"subhalo_row": int(row), "selected_member_count": int(n)}
                                           for row, n in zip(occupied, counts)],
                "morphology": candidate_morphology(xyz[available], vel[available], mass[available],
                                                   meta["box_comoving_kpc"], meta["scale_factor"]),
                "proximity": saved_subhalo_proximity(run_dir, meta["snapshot"], xyz[available], occupied,
                    meta["box_comoving_kpc"], meta["scale_factor"], meta["h"])})
        if previous is not None:
            old_meta, old_member = previous
            masks = membership_transitions(old_member, member)
            group = output.create_group(f"Transitions/{old_meta['snapshot']:03d}_{meta['snapshot']:03d}")
            for name, mask in masks.items():
                dataset=group.create_dataset(name,data=mask,compression='gzip',fletcher32=True)
                from core.data import write_dataset_contract
                write_dataset_contract(dataset,units='boolean transition mask',
                    quantity='selected exact stellar ID membership transition: '+name,
                    epoch={'left':old_meta['epoch_contract'],'right':meta['epoch_contract']},
                    ancestry={'link_semantics':'selected-star overlap only; no complete persistent progenitor ancestry'},
                    provenance={'left_source_sha256':old_meta['source_sha256'],'right_source_sha256':meta['source_sha256']})
            for candidate in summary["candidate_ids"]:
                selected = labels == candidate
                both = selected & masks["bound_at_both"]
                pairs, counts = np.unique(np.column_stack((old_member[both], member[both])), axis=0, return_counts=True)
                diagnostics["intervals"].append({"candidate_id": candidate,
                    "left_snapshot": old_meta["snapshot"], "right_snapshot": meta["snapshot"],
                    "scale_factor_interval": [old_meta["scale_factor"], meta["scale_factor"]],
                    "counts": {name: int(np.sum(selected & mask)) for name, mask in masks.items()},
                    "selected_star_overlap_links": [{"left_subhalo_row": int(pair[0]),
                        "right_subhalo_row": int(pair[1]), "shared_selected_stars": int(n)} for pair, n in zip(pairs, counts)]})
        previous = meta, member
    output.create_dataset("StreamDiagnosticsJSON", data=json.dumps(diagnostics, allow_nan=False),
                          dtype=h5py.string_dtype("utf-8"))
    return diagnostics


def plot_stream_diagnostics(directory, summary, diagnostics):
    """Show measured RMS axes and saved-interval membership transitions."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    epochs = {m["snapshot"]: m for m in summary["epochs"]}
    for candidate in summary["candidate_ids"]:
        rows = [d for d in diagnostics["candidates"] if d["candidate_id"] == candidate]
        a = [epochs[d["snapshot"]]["scale_factor"] for d in rows]
        for index, title in ((0, "major RMS axis"), (1, "intermediate RMS axis"), (2, "minor RMS axis")):
            values = [d["morphology"].get("rms_axes_proper_kpc", [np.nan]*3)[index] for d in rows]
            axes[0, 0].plot(a, values, "o-", label=f"{candidate}: {title}")
        dispersion = [d["morphology"].get("velocity_rms_km_s", np.nan) for d in rows]
        axes[0, 1].plot(a, dispersion, "o-", label=f"Candidate {candidate}")
        intervals = [d for d in diagnostics["intervals"] if d["candidate_id"] == candidate]
        midpoint = [np.mean(d["scale_factor_interval"]) for d in intervals]
        halfwidth = [np.diff(d["scale_factor_interval"])[0]/2 for d in intervals]
        fraction = [d["counts"]["bound_to_outside"]/d["counts"]["comparable"]
                    if d["counts"]["comparable"] else np.nan for d in intervals]
        axes[1, 0].errorbar(midpoint, fraction, xerr=halfwidth, fmt="o", label=f"Candidate {candidate}")
        closest = [d["proximity"].get("closest_other_subhalos", []) for d in rows]
        axes[1, 1].plot(a, [r[0]["distance_proper_kpc"] if r else np.nan for r in closest], "o-",
                        label=f"Candidate {candidate}")
    for axis, title, ylabel in (
        (axes[0, 0], "Candidate morphology about mass centroid", "RMS axis [proper kpc]"),
        (axes[0, 1], "Mass-weighted velocity scatter", "3D RMS [km/s]"),
        (axes[1, 0], "Bound to outside SUBFIND between saved epochs", "Fraction of comparable selected stars"),
        (axes[1, 1], "Closest other saved subhalo centre", "Distance to nearest member [proper kpc]"),
    ):
        axis.set(title=title, xlabel="Scale factor a", ylabel=ylabel)
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=7)
    axes[1, 0].set_ylim(0, 1)
    fixture = summary["candidate_provenance"].get("fixture")
    prefix = "SYNTHETIC FIXTURE • " if fixture else ""
    figure.suptitle(prefix + "Saved stellar candidates: no confirmed tidal classification or inferred flyby")
    from figures import export_figure
    export_figure(figure,Path(directory)/'stream_diagnostics',formats=('png','pdf'),dpi=180,close=False,
        metadata={'science_acceptance':'NOT_ACCEPTED','source_sha256':summary['history_sha256'],
            'epochs':[epoch['epoch_contract'] for epoch in summary['epochs']],
            'data_contract':summary['data_contract'],'x_units':'scale factor',
            'y_units':['proper kpc','km/s','fraction','proper kpc']})
    plt.close(figure)

# ========================================================================
# observer_coordinates: Instantaneous local astrometry in a declared right-handed observer frame.
from dataclasses import dataclass

from core.foundation import KPC_CM, SECONDS_PER_YEAR

MAS_PER_RADIAN = 180. / np.pi * 3600. * 1000.
PROPER_MOTION_FACTOR = 1.e5 / KPC_CM * SECONDS_PER_YEAR * MAS_PER_RADIAN
OBSERVABLE_COLUMNS = ('longitude_rad', 'latitude_rad', 'distance_kpc',
                      'radial_velocity_kms', 'mu_longitude_star_masyr', 'mu_latitude_masyr')


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


@dataclass(frozen=True)
class ObserverFrame:
    """Rotation rows are observer x/y/z axes expressed in simulation coordinates."""
    position_kpc: object
    velocity_kms: object
    rotation: object

    def __post_init__(self):
        for name, shape in (('position_kpc', (3,)), ('velocity_kms', (3,)), ('rotation', (3, 3))):
            value = _array(getattr(self, name), shape, name).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if (not np.allclose(self.rotation @ self.rotation.T, np.eye(3), rtol=0., atol=1.e-12)
                or not np.isclose(np.linalg.det(self.rotation), 1., rtol=0., atol=1.e-12)):
            raise ValueError('rotation must be right-handed and orthonormal')

    def metadata(self):
        return {'position_kpc': self.position_kpc.tolist(),
                'velocity_kms': self.velocity_kms.tolist(), 'rotation': self.rotation.tolist(),
                'position_convention': 'proper_kpc', 'velocity_convention': 'peculiar_km_s',
                'epoch_scope': 'instantaneous_local_geometry',
                'proper_motion_year_seconds': SECONDS_PER_YEAR,
                'columns': list(OBSERVABLE_COLUMNS),
                'pole_convention': 'longitude_zero_at_exact_poles; tangent_axes_follow_longitude',
                'omitted_effects': ['Hubble_flow', 'light_cone', 'aberration', 'lensing',
                                   'cosmological_redshift']}


def _basis(longitude, latitude):
    cl, sl = np.cos(longitude), np.sin(longitude)
    cb, sb = np.cos(latitude), np.sin(latitude)
    direction = np.column_stack((cb * cl, cb * sl, sb))
    east = np.column_stack((-sl, cl, np.zeros_like(cl)))
    north = np.column_stack((-sb * cl, -sb * sl, cb))
    return direction, east, north


def to_observer(position_kpc, velocity_kms, frame):
    """Return N×6 exact geometric observables in OBSERVABLE_COLUMNS order."""
    position = np.asarray(position_kpc, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError('position_kpc must have shape (N, 3)')
    position = _array(position, position.shape, 'position_kpc')
    velocity = _array(velocity_kms, position.shape, 'velocity_kms')
    relative = (position - frame.position_kpc) @ frame.rotation.T
    speed = (velocity - frame.velocity_kms) @ frame.rotation.T
    distance = np.hypot(np.hypot(relative[:, 0], relative[:, 1]), relative[:, 2])
    if np.any(distance <= 0.) or not np.all(np.isfinite(distance)):
        raise ValueError('observer-coincident or overflowing distances cannot be projected')
    transverse = np.hypot(relative[:, 0], relative[:, 1])
    longitude = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2. * np.pi)
    # Floating-point wrap can round a tiny negative longitude to exactly 2π.
    longitude = np.where(longitude >= 2. * np.pi, 0., longitude)
    longitude = np.where(transverse == 0., 0., longitude)
    latitude = np.arctan2(relative[:, 2], transverse)
    direction, east, north = _basis(longitude, latitude)
    result = np.column_stack((longitude, latitude, distance,
                            np.sum(speed * direction, axis=1),
                            np.sum(speed * east, axis=1) * PROPER_MOTION_FACTOR / distance,
                            np.sum(speed * north, axis=1) * PROPER_MOTION_FACTOR / distance))
    return _array(result, result.shape, 'projected observables')


def from_observer(observables, frame):
    """Invert complete finite geometric data; NaNs and invalid distances fail."""
    data = np.asarray(observables, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError('observables must have shape (N, 6)')
    data = _array(data, data.shape, 'observables')
    lon, lat, distance, vr, mul, mub = data.T
    if (np.any(distance <= 0.) or np.any(abs(lat) > np.pi / 2.)
            or np.any(lon < 0.) or np.any(lon >= 2. * np.pi)):
        raise ValueError('invalid longitude, latitude or distance')
    direction, east, north = _basis(lon, lat)
    position = distance[:, None] * direction
    velocity = (vr[:, None] * direction
                + (distance * mul / PROPER_MOTION_FACTOR)[:, None] * east
                + (distance * mub / PROPER_MOTION_FACTOR)[:, None] * north)
    x, v = position @ frame.rotation + frame.position_kpc, velocity @ frame.rotation + frame.velocity_kms
    return _array(x, x.shape, 'reconstructed positions'), _array(v, v.shape, 'reconstructed velocities')


def native_local_coordinates(coordinates, velocities, *, scale_factor, hubble_param,
                             box_comoving_kpc_h, observer_comoving_kpc_h):
    """Native GADGET x[kpc/h], v[sqrt(a) km/s] to observer-centred local values.

    Minimum-image displacement chooses the periodic image nearest the observer.
    Exactly half-box separations are ambiguous and fail. Returned positions are
    observer-centred proper kpc: use frame.position_kpc=[0,0,0]. Velocities still
    require the observer's peculiar velocity subtraction in to_observer.
    """
    a, h, box = float(scale_factor), float(hubble_param), float(box_comoving_kpc_h)
    if not all(np.isfinite(v) and v > 0. for v in (a, h, box)):
        raise ValueError('scale factor, h and box must be finite and positive')
    x = np.asarray(coordinates, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('coordinates must have shape (N, 3)')
    x = _array(x, x.shape, 'coordinates')
    v = _array(velocities, x.shape, 'velocities')
    observer = _array(observer_comoving_kpc_h, (3,), 'observer_comoving_kpc_h')
    offset = np.mod(x - observer + box / 2., box) - box / 2.
    if np.any(np.isclose(abs(offset), box / 2., rtol=0., atol=8. * np.spacing(box))):
        raise ValueError('half-box separation has ambiguous periodic image')
    from core.data import native_unit_factor
    x = offset * native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h)
    v = v * native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h)
    return _array(x, x.shape, 'proper offsets'), _array(v, v.shape, 'peculiar velocities')


@native_run_owner
@directory_owner('output_directory')
def export_stellar_observer_epoch(run_dir, snapshot, particle_ids, *, frame,
                                 observer_comoving_kpc, output_directory, memory_gib=1.):
    """Export exact saved stellar IDs, including explicit absent-star rows.

    The observer position is supplied separately in wrapped comoving kpc with
    no h factor; frame.position_kpc must be zero for these relative coordinates.
    No measurement errors or selection are applied to this truth catalogue.
    """
    import hashlib
    import json
    from pathlib import Path
    import tempfile

    from stars import exact_ids, read_epoch, _snapshot_files, _group_files
    from core.data import native_unit_factor, coordinate_contract
    from core.foundation import product_provenance, publish_directory, verify_product_provenance

    if isinstance(snapshot, (bool, np.bool_)) or not isinstance(snapshot, (int, np.integer)) or snapshot < 0:
        raise ValueError('snapshot must be a nonnegative integer')
    if isinstance(memory_gib, (bool, np.bool_)) or not np.isfinite(memory_gib) or memory_gib <= 0.:
        raise ValueError('memory_gib must be positive and finite')
    if isinstance(particle_ids, (list, tuple)):
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               or not 0 <= int(i) < 2**64 for i in particle_ids):
            raise ValueError('requested IDs must be exact uint64 integers')
        particle_ids = np.asarray(particle_ids, dtype=np.uint64)
    ids = exact_ids(particle_ids)
    if ids.size * 512 + 64 * 1024**2 > memory_gib * 1024**3:
        raise MemoryError('selected observer catalogue exceeds workspace allocation')
    observer = _array(observer_comoving_kpc, (3,), 'observer_comoving_kpc')
    if np.any(frame.position_kpc != 0.):
        raise ValueError('stellar export requires an observer-centred frame position of zero')
    run_dir, destination = Path(run_dir).resolve(), Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f'observer destination already exists: {destination}')

    def sources():
        return sorted(str(Path(p).resolve()) for p in
                      _snapshot_files(run_dir, snapshot) + _group_files(run_dir, snapshot))

    def hashes(paths):
        result = {}
        for path in paths:
            digest = hashlib.sha256()
            with open(path, 'rb') as source:
                for block in iter(lambda: source.read(1024**2), b''):
                    digest.update(block)
            result[path] = digest.hexdigest()
        return result

    paths = sources()
    if not paths:
        raise ValueError('no saved snapshot sources')
    before = hashes(paths)
    x, v, mass, present, membership, epoch = read_epoch(run_dir, int(snapshot), ids)
    a, h = epoch['scale_factor'], epoch['h']
    relative, peculiar = native_local_coordinates(x[present] * h, v[present] / np.sqrt(a),
                          scale_factor=a, hubble_param=h,
                          box_comoving_kpc_h=epoch['box_comoving_kpc'] * h,
                          observer_comoving_kpc_h=observer * h)
    observables = np.full((len(ids), 6), np.nan)
    observables[present] = to_observer(relative, peculiar, frame)
    restored_x, restored_v = from_observer(observables[present], frame)
    error_x = np.linalg.norm(restored_x - relative, axis=1)
    error_v = np.linalg.norm(restored_v - peculiar, axis=1)
    bound = 32. * np.finfo(float).eps
    if (np.any(error_x > bound * np.linalg.norm(relative, axis=1))
            or np.any(error_v > bound * (np.linalg.norm(peculiar - frame.velocity_kms, axis=1)
                                      + np.linalg.norm(frame.velocity_kms)))):
        raise RuntimeError('saved stellar observer round trip exceeds floating-point bound')
    if sources() != paths or hashes(paths) != before:
        raise RuntimeError('saved stellar sources changed during observer export')
    report = {'schema': 'darkmattersim-stellar-observer-v1', 'status': 'TRUTH_CATALOGUE',
              'science_acceptance': 'NOT_ACCEPTED', 'epoch': epoch, 'frame': frame.metadata(),
              'observer_comoving_kpc': observer.tolist(), 'source_sha256': before,
              'selected_count': len(ids), 'present_count': int(present.sum()),
              'max_roundtrip_position_kpc': float(error_x.max()) if len(error_x) else None,
              'max_roundtrip_velocity_kms': float(error_v.max()) if len(error_v) else None,
              'missing_rows': 'Present=false; observables and saved quantities remain NaN',
              'selection': 'exact requested IDs; no survey selection',
              'membership': 'SUBFIND epoch-local row; -1 outside bound spans, -2 absent, -3 unavailable',
              'measurement_errors': 'not applied'}
    coordinates = coordinate_contract(position_units='proper kpc',velocity_convention='observer-relative peculiar km/s; no Hubble flow',
        origin='declared synthetic observer',periodic_wrapping='nearest periodic image of observer; exact half-box separation rejected',
        observer_frame=frame.metadata())
    report['data_contract'] = product_provenance(run_dir,algorithm=__file__,selection=report['selection'],
        source_files=[{'path':p,'sha256':sha} for p,sha in before.items()],coordinates=coordinates,snapshot=int(snapshot))
    report['dataset_units'] = {'particle_ids':'uint64 identity','saved_positions_comoving_kpc':'comoving kpc',
        'saved_velocities_peculiar_kms':'km/s','mass_msun':'Msun','observables':list(OBSERVABLE_COLUMNS),
        'observable_units':['rad','rad','proper kpc','km/s','mas/year','mas/year']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.observer-',dir=destination.parent))
    from core.foundation import atomic_json
    try:
        np.savez_compressed(stage / 'truth.npz', particle_ids=ids, present=present,
                            saved_positions_comoving_kpc=x, saved_velocities_peculiar_kms=v,
                            mass_msun=mass, bound_subhalo_row=membership, observables=observables)
        report['catalogue_sha256'] = hashes([str(stage / 'truth.npz')])[str(stage / 'truth.npz')]
        from core.data import numpy_archive_contract
        report['array_contract']=numpy_archive_contract(stage/'truth.npz',{
            'particle_ids':{'units':'identity','identity_namespace':report['data_contract']['identity_namespaces']['star_population'],
                'quantity':'exact requested saved star-population ID; absent rows retained','missing_policy':'none'},
            'present':{'units':'boolean availability mask','quantity':'requested ID occurs in saved stellar namespace','missing_policy':'none'},
            'saved_positions_comoving_kpc':{'units':'comoving kpc','quantity':'saved global simulation positions',
                'coordinates':coordinate_contract(position_units='comoving kpc',origin='global simulation Cartesian box origin',
                    periodic_wrapping='native fundamental box',velocity_convention='saved physical peculiar km/s'),
                'missing_policy':'NaN for absent rows'},
            'saved_velocities_peculiar_kms':{'units':'km/s','quantity':'saved physical peculiar velocities','missing_policy':'NaN for absent rows'},
            'mass_msun':{'units':'Msun','quantity':'saved star-population dynamical mass','missing_policy':'NaN for absent rows'},
            'bound_subhalo_row':{'units':'epoch-local SUBFIND row','quantity':'exact bound membership from native offset spans',
                'missing_policy':'-1 outside bound spans; -2 absent; -3 unavailable'},
            'observables':{'units':'heterogeneous observer columns','quantity':'synthetic observer coordinates from exact saved particles',
                'missing_policy':'NaN for absent rows','annotations':{'columns':list(OBSERVABLE_COLUMNS),
                    'column_units':['rad','rad','proper kpc','km/s','mas/year','mas/year']}}},
            epoch=epoch['epoch_contract'],coordinates=coordinates,provenance={'report':'report.json'},memory_bytes=int(memory_gib*1024**3))
        report['implementation_sha256'] = hashes([__file__])[__file__]
        (stage / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        from core.foundation import file_sha256 as digest
        (stage/'report.sha256').write_text(digest(stage/'report.json')+'\n')
        verify_product_provenance(run_dir,report['data_contract'])
        publish_directory(stage,destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return report

# ========================================================================
# observation_model: Correlated Gaussian astrometry in one fixed, explicitly declared sky chart.

MEASUREMENT_COLUMNS = ('gnomonic_east', 'gnomonic_north', 'parallax_mas',
                       'radial_velocity_kms', 'mu_longitude_star_masyr', 'mu_latitude_masyr')


def _chart(center):
    center = _array(center, (2,), 'chart center longitude/latitude')
    if not 0. <= center[0] < 2. * np.pi or abs(center[1]) > np.pi / 2.:
        raise ValueError('invalid chart center')
    return tuple(v[0] for v in _basis(center[:1], center[1:]))


def to_measurements(observables, *, chart_center):
    """Convert complete geometric truth into the fixed measurement chart.

    Proper motions retain the longitude-star/latitude convention at the
    object's true position; their errors are declared in those component units.
    No linear small-angle sky approximation is made.
    """
    data = np.asarray(observables, dtype=float)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError('observables must have shape (N, 6)')
    data = _array(data, data.shape, 'observables')
    if (np.any(data[:, 2] <= 0.) or np.any(abs(data[:, 1]) > np.pi / 2.)
            or np.any(data[:, 0] < 0.) or np.any(data[:, 0] >= 2. * np.pi)):
        raise ValueError('invalid geometric observables')
    anchor, east, north = _chart(chart_center)
    direction = _basis(data[:, 0], data[:, 1])[0]
    denominator = direction @ anchor
    # At 90 degrees the chart diverges; roundoff must not select an image.
    if np.any(denominator <= 32. * np.finfo(float).eps):
        raise ValueError('object lies outside the open chart hemisphere or on its boundary')
    result = data.copy()
    result[:, 0] = direction @ east / denominator
    result[:, 1] = direction @ north / denominator
    result[:, 2] = 1. / data[:, 2]
    return _array(result, result.shape, 'measurement truth')


def from_measurements(measurements, *, chart_center):
    """Invert complete data with positive parallax; negative measurements fail.

    Use negative/noisy parallaxes directly in the likelihood, not this inverse.
    """
    data = np.asarray(measurements, dtype=float)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError('measurements must have shape (N, 6)')
    data = _array(data, data.shape, 'measurements')
    if np.any(data[:, 2] <= 0.):
        raise ValueError('positive parallax required for a geometric inverse')
    anchor, east, north = _chart(chart_center)
    # Scale before forming the vector to avoid overflow at large finite u/v.
    scale = np.maximum(1., np.max(abs(data[:, :2]), axis=1))
    direction = (anchor[None, :] / scale[:, None]
                 + (data[:, 0] / scale)[:, None] * east
                 + (data[:, 1] / scale)[:, None] * north)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    result = data.copy()
    result[:, 0] = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2. * np.pi)
    result[:, 0] = np.where(result[:, 0] >= 2. * np.pi, 0., result[:, 0])
    result[:, 1] = np.arctan2(direction[:, 2], np.hypot(direction[:, 0], direction[:, 1]))
    result[:, 2] = 1. / data[:, 2]
    return _array(result, result.shape, 'inverse measurements')


def _covariances(covariance, n, dimension):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (dimension, dimension):
        covariance = np.broadcast_to(covariance, (n, dimension, dimension))
    if covariance.shape != (n, dimension, dimension) or not np.isfinite(covariance).all():
        raise ValueError('covariance must be finite (D,D) or (N,D,D)')
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    if np.any(diagonal <= 0.):
        raise ValueError('covariance diagonal must be positive')
    scale = np.sqrt(diagonal)
    correlation = covariance / scale[:, :, None] / scale[:, None, :]
    if not np.allclose(correlation, correlation.transpose(0, 2, 1), rtol=0., atol=1.e-12):
        raise ValueError('covariance must be symmetric')
    # Remove only tolerated roundoff asymmetry, never repair eigenvalues.
    correlation = .5 * (correlation + correlation.transpose(0, 2, 1))
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError('covariance must be positive definite; no regularization applied') from error
    return scale, correlation


def gaussian_log_likelihood(data, prediction, covariance, measured):
    """Per-object normalized Gaussian marginal over measured components.

    Missing values must be NaN and measured must be a boolean N×D mask.
    Select the covariance submatrix, not a submatrix of the precision matrix.
    Completely missing rows contribute log(1)=0. Prediction stays complete.
    This assumes missingness is ignorable given the declared experiment;
    informative missingness needs its own response probability.
    """
    data, prediction = np.asarray(data, dtype=float), np.asarray(prediction, dtype=float)
    measured = np.asarray(measured)
    if (data.ndim != 2 or data.shape[1] == 0 or prediction.shape != data.shape
            or not np.isfinite(prediction).all() or measured.shape != data.shape or measured.dtype != bool):
        raise ValueError('data/prediction/mask require matching N×D shapes and complete finite prediction')
    if not np.isfinite(data[measured]).all() or not np.isnan(data[~measured]).all():
        raise ValueError('measured values must be finite and missing values must be NaN')
    scale, correlation = _covariances(covariance, *data.shape)
    result = np.zeros(len(data))
    for i, mask in enumerate(measured):
        if not np.any(mask):
            continue
        factor = np.linalg.cholesky(correlation[i][np.ix_(mask, mask)])
        residual = (data[i, mask] - prediction[i, mask]) / scale[i, mask]
        whitened = np.linalg.solve(factor, residual)
        logdet = 2. * (np.log(np.diag(factor)).sum() + np.log(scale[i, mask]).sum())
        result[i] = -.5 * (whitened @ whitened + logdet + mask.sum() * np.log(2. * np.pi))
    return result


def gaussian_mock(prediction, covariance, measured, *, seed):
    """Reproducible draws with the same covariance and missing-data convention.

    Negative parallaxes and outliers remain; no clipping or repeated draws.
    Missingness mask is supplied explicitly, not inferred from noisy values.
    """
    prediction = np.asarray(prediction, dtype=float)
    measured = np.asarray(measured)
    if (prediction.ndim != 2 or prediction.shape[1] == 0 or not np.isfinite(prediction).all()
            or measured.shape != prediction.shape or measured.dtype != bool):
        raise ValueError('complete finite prediction and boolean N×D mask required')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    scale, correlation = _covariances(covariance, *prediction.shape)
    normal = np.random.default_rng(int(seed)).normal(size=prediction.shape)
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(correlation), normal) * scale
    result = prediction + noise
    result[~measured] = np.nan
    return result


def selected_mixture_log_likelihood(member_logpdf, contaminant_logpdf, *, member_fraction,
                                    selection_probability, member_efficiency, contaminant_efficiency):
    """Exact selected-population mixture with a common observed-data selection.

    PDFs must be normalized over identical measured coordinates, before
    selection. Efficiencies must be independently evaluated integrals of s(y)
    against those PDFs, including the same missing-data experiment. They are
    not guessed from selected counts. Fraction is BEFORE selection. This is
    conditional on inclusion, not a Poisson abundance likelihood. For latent
    truth-dependent selection, integrate selection inside the component PDFs
    instead; this common s(y) formula is inapplicable.
    """
    member, contaminant = np.broadcast_arrays(np.asarray(member_logpdf, dtype=float),
                                              np.asarray(contaminant_logpdf, dtype=float))
    if (np.any(np.isnan(member)) or np.any(np.isnan(contaminant))
            or np.any(np.isposinf(member)) or np.any(np.isposinf(contaminant))):
        raise ValueError('component log densities must be finite or negative infinity')
    probabilities = np.broadcast_arrays(member, np.asarray(member_fraction, dtype=float),
                        np.asarray(selection_probability, dtype=float),
                        np.asarray(member_efficiency, dtype=float), np.asarray(contaminant_efficiency, dtype=float))
    _, fraction, selection, em, ec = probabilities
    if any(np.any(~np.isfinite(x)) or np.any((x < 0.) | (x > 1.)) for x in (fraction, selection, em, ec)):
        raise ValueError('fractions, selection and efficiencies must be probabilities')
    normalization = fraction * em + (1. - fraction) * ec
    if np.any(normalization <= 0.):
        raise ValueError('selected mixture has zero inclusion probability')
    with np.errstate(divide='ignore'):
        result = (np.logaddexp(np.log(fraction) + member, np.log1p(-fraction) + contaminant)
                  + np.log(selection) - np.log(normalization))
    return result


def observed_box_selection(data, bounds):
    """Deterministic inclusive cuts on measured chart values, before masking.

    bounds is D pairs of [lower, upper]; None is an unlimited endpoint.
    No cuts means [[None,None], ...]. This is one explicit selection model.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('selection requires complete finite measurement draws')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != data.shape[1]:
        raise ValueError('selection bounds must contain one pair per measurement column')
    selected = np.ones(len(data), dtype=bool)
    cut_columns = np.zeros(data.shape[1], dtype=bool)
    for j, pair in enumerate(bounds):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('each selection bound must be a lower/upper pair')
        lower, upper = pair
        for endpoint in pair:
            if endpoint is not None and (isinstance(endpoint, (bool, np.bool_))
                    or not isinstance(endpoint, (float, int, np.floating, np.integer)) or not np.isfinite(endpoint)):
                raise ValueError('selection endpoints must be finite numbers or None')
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError('selection lower endpoint must be below upper endpoint')
        if lower is not None:
            selected &= data[:, j] >= lower
        if upper is not None:
            selected &= data[:, j] <= upper
        cut_columns[j] = lower is not None or upper is not None
    return selected, cut_columns


@directory_owner('truth_directory')
@directory_owner('output_directory')
def export_mock_observations(truth_directory, *, chart_center, covariance, measured,
                             selection_bounds, seed, output_directory, memory_gib=1.):
    """Apply the declared error/selection experiment to an immutable stellar truth export.

    Observed data and private mock truth are separate NPZ files. This is not an
    inference run; response integrals and population/dynamical models remain
    required. All selection cuts must use components returned as measured.
    """
    import hashlib
    import json
    from pathlib import Path
    import tempfile
    from stars import exact_ids

    truth_directory, destination = Path(truth_directory).resolve(), Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f'observation destination already exists: {destination}')
    if isinstance(memory_gib, bool) or not isinstance(memory_gib, (int, float)) or not np.isfinite(memory_gib) or memory_gib <= 0.:
        raise ValueError('memory_gib must be finite and positive')
    def sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        return digest.hexdigest()
    source_report, source_arrays = truth_directory / 'report.json', truth_directory / 'truth.npz'
    report_hash, array_hash = sha(source_report), sha(source_arrays)
    report = json.loads(source_report.read_text())
    if report.get('schema') != 'darkmattersim-stellar-observer-v1' or report.get('status') != 'TRUTH_CATALOGUE':
        raise ValueError('source must be an exported stellar observer truth catalogue')
    if array_hash != report.get('catalogue_sha256'):
        raise ValueError('source truth catalogue hash disagrees with its report')
    from core.data import verify_numpy_archive_contract,numpy_archive_contract
    if report.get('array_contract') is not None:
        if not (truth_directory/'report.sha256').is_file() or (truth_directory/'report.sha256').read_text().strip()!=report_hash:
            raise ValueError('source observer report checksum mismatch')
        verify_numpy_archive_contract(source_arrays,report['array_contract'],memory_bytes=int(memory_gib*1024**3))
    # Bound decompressed workspace before loading arrays from a compressed file.
    import zipfile
    with zipfile.ZipFile(source_arrays) as archive:
        size = sum(item.file_size for item in archive.infolist())
    if size * 8 + 64 * 1024**2 > memory_gib * 1024**3:
        raise MemoryError('observation catalogue exceeds workspace allocation')
    with np.load(source_arrays, allow_pickle=False) as source:
        ids = exact_ids(source['particle_ids'])
        present = source['present']
        data = source['observables']
        if present.dtype != bool or present.shape != (len(ids),) or data.shape != (len(ids), 6):
            raise ValueError('source truth IDs, presence and observables must align')
        if (not np.isfinite(data[present]).all() or not np.isnan(data[~present]).all()
                or int(present.sum()) != report.get('present_count') or len(ids) != report.get('selected_count')):
            raise ValueError('source truth counts or missing rows disagree')
        ids, truth = ids[present], data[present]
    if not len(ids):
        raise ValueError('no present saved stars; cannot fabricate a mock observed population')
    prediction = to_measurements(truth, chart_center=chart_center)
    mask = np.asarray(measured)
    if mask.shape == (6,):
        mask = np.broadcast_to(mask, prediction.shape)
    if mask.shape != prediction.shape or mask.dtype != bool:
        raise ValueError('measured mask must be boolean (6,) or (N_present,6)')
    draw = gaussian_mock(prediction, covariance, np.ones(prediction.shape, dtype=bool), seed=seed)
    selected, cuts = observed_box_selection(draw, selection_bounds)
    if np.any(~mask[:, cuts]):
        raise ValueError('selection on a missing component needs a latent-selection likelihood')
    # Preserve the exact supplied floating-point values. Reconstructing a
    # covariance from normalized correlations changes last bits and prevents
    # bitwise replay from the saved product, despite identical seed and model.
    cov = np.broadcast_to(np.asarray(covariance, dtype=float), (len(ids), 6, 6)).copy()
    observed = draw[selected].copy()
    observed[~mask[selected]] = np.nan
    if sha(source_report) != report_hash or sha(source_arrays) != array_hash:
        raise RuntimeError('source truth changed during mock observation export')
    metadata = {'schema': 'darkmattersim-mock-observation-v1', 'status': 'MOCK_OBSERVATIONS',
                'science_acceptance': 'NOT_ACCEPTED', 'source_directory': str(truth_directory),
                'source_report_sha256': report_hash, 'source_catalogue_sha256': array_hash,
                'frame': report['frame'], 'epoch': report['epoch'],
                'chart_center_longitude_latitude_rad': np.asarray(chart_center).tolist(),
                'measurement_columns': list(MEASUREMENT_COLUMNS),
                'density_measure': 'Lebesgue measure in fixed gnomonic/parallax/velocity/motion coordinates',
                'proper_motion_errors': 'Gaussian component errors in longitude-star/latitude convention',
                'noise': 'full correlated Gaussian, no clipping or resampling', 'seed': int(seed),
                'present_parent_count': len(ids), 'observed_count': int(selected.sum()),
                'selection': {'kind': 'observed_box', 'bounds': selection_bounds,
                              'efficiency_status': 'NOT_CALIBRATED'},
                'missingness': 'explicit mask; ignorable assumption requires experiment validation',
                'contamination': 'not injected; population mixture requires declared contaminant source/model',
                'private_truth': 'mock_truth.npz is diagnostic-only; do not use as inference data'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.mock-observation-',dir=destination.parent))
    from core.foundation import atomic_json, publish_directory
    try:
        np.savez_compressed(stage / 'observed.npz', particle_ids=ids[selected],
                            measurements=observed, measured=mask[selected], covariance=cov[selected])
        np.savez_compressed(stage / 'mock_truth.npz', particle_ids=ids, prediction=prediction,
                            unmasked_draw=draw, selected=selected, measured=mask, covariance=cov)
        metadata['product_sha256'] = {p.name: sha(p) for p in (stage/'observed.npz', stage/'mock_truth.npz')}
        metadata['implementation_sha256'] = sha(Path(__file__))
        namespace=report.get('data_contract',{}).get('identity_namespaces',{}).get('star_population',
            'unverified-observer-source:'+array_hash+':star_population')
        columns=list(MEASUREMENT_COLUMNS)
        units=['dimensionless','dimensionless','mas','km/s','mas/year','mas/year']
        vector={'units':'heterogeneous measurement columns','quantity':'declared Gaussian astrometric chart measurement',
            'missing_policy':'unmeasured components are NaN in observed catalogue; private unmasked draws retain all components',
            'annotations':{'columns':columns,'column_units':units}}
        covariance_spec={'units':'heterogeneous covariance column products','quantity':'exact declared correlated Gaussian covariance',
            'missing_policy':'none','annotations':{'columns':columns,'column_unit_products':[[first+' * '+second for second in units] for first in units]}}
        identity={'units':'identity','identity_namespace':namespace,'quantity':'exact source star-population identity','missing_policy':'none'}
        measured_spec={'units':'boolean measurement mask','quantity':'declared observed components','missing_policy':'false denotes unmeasured'}
        common={'epoch':report['epoch'].get('epoch_contract',report['epoch']),
            'coordinates':report.get('data_contract',{}).get('coordinates'),
            'provenance':{'report':'report.json','source_report_sha256':report_hash,'source_catalogue_sha256':array_hash},
            'memory_bytes':int(memory_gib*1024**3)}
        metadata['array_contracts']={
            'observed.npz':numpy_archive_contract(stage/'observed.npz',{'particle_ids':identity,'measurements':vector,
                'measured':measured_spec,'covariance':covariance_spec},**common),
            'mock_truth.npz':numpy_archive_contract(stage/'mock_truth.npz',{'particle_ids':identity,'prediction':vector,
                'unmasked_draw':vector,'selected':{'units':'boolean selection mask','quantity':'observed-space selection event','missing_policy':'none'},
                'measured':measured_spec,'covariance':covariance_spec},**common)}
        (stage / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (stage/'report.sha256').write_text(sha(stage/'report.json')+'\n')
        if sha(source_report)!=report_hash or sha(source_arrays)!=array_hash:
            raise RuntimeError('source truth changed before mock publication')
        publish_directory(stage,destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return metadata


@directory_owner('mock_directory')
@directory_owner('output_directory')
def plot_mock_response(mock_directory, output_directory):
    """Diagnostic-only pre-selection noise calibration, not posterior inference.

    Uses private mock truth. Pre-selection draws prevent survey truncation from
    being mistaken for a failed Gaussian error kernel. Selected data still need
    their selection-aware likelihood. Every diagnostic output is hashed.
    """
    import hashlib
    import json
    from pathlib import Path
    import tempfile
    from matplotlib import pyplot as plt
    from scipy.stats import chi2
    from figures import apply_plot_style

    mock_directory, destination = Path(mock_directory).resolve(), Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f'response figure destination already exists: {destination}')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report_path = mock_directory / 'report.json'
    report_hash = sha(report_path)
    metadata = json.loads(report_path.read_text())
    if metadata.get('schema') != 'darkmattersim-mock-observation-v1':
        raise ValueError('response figures require a mock observation export')
    private = mock_directory / 'mock_truth.npz'
    private_hash = sha(private)
    if private_hash != metadata['product_sha256'].get(private.name):
        raise ValueError('private mock truth checksum disagrees with report')
    if metadata.get('array_contracts') is not None:
        from core.data import verify_numpy_archive_contract
        if not (mock_directory/'report.sha256').is_file() or (mock_directory/'report.sha256').read_text().strip()!=report_hash:
            raise ValueError('mock observation report checksum mismatch')
        verify_numpy_archive_contract(private,metadata['array_contracts'][private.name])
    with np.load(private, allow_pickle=False) as source:
        prediction, draw, covariance = source['prediction'], source['unmasked_draw'], source['covariance']
        selected = source['selected']
    if prediction.shape != draw.shape or prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError('private mock prediction/draw arrays must align')
    scale, correlation = _covariances(covariance, *prediction.shape)
    standardized = (draw - prediction) / scale
    whitened = np.linalg.solve(np.linalg.cholesky(correlation), standardized[:, :, None])[:, :, 0]
    q = np.sum(whitened**2, axis=1)
    if not np.isfinite(q).all() or selected.shape != (len(q),) or selected.dtype != bool:
        raise ValueError('invalid private response diagnostics')
    apply_plot_style({'figure_layout': 'presentation', 'figure_formats': ['png', 'pdf'], 'figure_dpi': 180})
    figure, axes = plt.subplots(1, 3, figsize=(12., 3.65), constrained_layout=True)
    labels = ['east chart', 'north chart', 'parallax', 'radial velocity', r'$\mu_{\ell *}$', r'$\mu_b$']
    bins = np.linspace(-4.5, 4.5, 38)
    for j, label in enumerate(labels):
        # Full-parent normalization: outlying tails are not renormalized away.
        counts, edges = np.histogram(standardized[:, j], bins=bins)
        axes[0].stairs(counts / len(q) / np.diff(edges), edges, label=label, linewidth=1.)
    x = np.linspace(-4.5, 4.5, 200)
    axes[0].plot(x, np.exp(-.5*x*x)/np.sqrt(2*np.pi), color='black', linestyle='--', label='unit Gaussian')
    axes[0].set(xlabel=r'marginal residual / $\sigma$', ylabel='density', title='Error components')
    axes[0].legend(fontsize=7., loc='upper right')
    limit = max(chi2.ppf(.999, 6), float(np.quantile(q, .995)))
    edges = np.linspace(0., limit, 36)
    counts, edges = np.histogram(q, bins=edges)
    axes[1].stairs(counts / len(q) / np.diff(edges), edges, label='correlated mock')
    x = np.linspace(0., limit, 300)
    axes[1].plot(x, chi2.pdf(x, 6), color='black', linestyle='--', label=r'$\chi^2_6$')
    axes[1].set(xlabel=r'$r^T C^{-1}r$', ylabel='density', title='Joint covariance check')
    axes[1].legend(fontsize=8.)
    probability = np.sort(chi2.cdf(q, 6))
    axes[2].plot(probability, (np.arange(len(q))+.5)/len(q), label='all pre-selection draws')
    axes[2].plot([0.,1.], [0.,1.], 'k--', label='calibrated reference')
    axes[2].set(xlabel=r'$\chi^2_6$ cumulative probability', ylabel='empirical fraction',
                title='Probability calibration', xlim=(0.,1.), ylim=(0.,1.))
    axes[2].legend(fontsize=8.)
    figure.suptitle(f'Mock response only · pre-selection N={len(q):,} · retained {selected.sum():,}', fontsize=11.)
    report = {'schema': 'darkmattersim-mock-response-figure-v1', 'status': 'DIAGNOSTIC_ONLY',
              'science_acceptance': 'NOT_ACCEPTED', 'source_report_sha256': report_hash,
              'private_truth_sha256': private_hash, 'parent_count': len(q),
              'retained_count': int(selected.sum()), 'mean_joint_chi2': float(q.mean()),
              'reference_degrees_of_freedom': 6, 'uses_private_truth': True,
              'implementation_sha256': sha(Path(__file__)),
              'scope': 'unconditional Gaussian error-kernel calibration; no stream/selection recovery claim'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage=Path(tempfile.mkdtemp(prefix='.mock-response-',dir=destination.parent))
        from core.foundation import atomic_json, publish_directory
        try:
            from figures import export_figure
            export_figure(figure,stage/'response.png',formats=['png','pdf'],dpi=180,close=False,
                metadata={'source_report_sha256':report_hash,'science_acceptance':'NOT_ACCEPTED',
                    'epoch':metadata.get('epoch'),'scope':'private mock truth pre-selection error-kernel diagnostics',
                    'axes_units':[{'x':'dimensionless standardized residual','y':'probability per standardized residual'},
                        {'x':'dimensionless squared Mahalanobis residual','y':'probability per squared Mahalanobis residual'},
                        {'x':'dimensionless expected quantile','y':'dimensionless measured quantile'}]})
            if sha(private) != private_hash or sha(report_path) != report_hash:
                raise RuntimeError('mock sources changed during response figure generation')
            report['product_sha256'] = {p.name: sha(p) for p in (stage/'response.png', stage/'response.pdf')}
            (stage/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            publish_directory(stage,destination)
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'failure':{'type':type(exc).__name__,'message':str(exc)}})
            raise
    finally:
        plt.close(figure)
    return report


def hydro_comparison_design(base_config, *, variants, resolution_levels):
    """Prepare matched DMO/hydro/hydro+BH configurations without launching jobs.

    The caller supplies one phase realization through the ordinary IC contract.
    DMO retains the full gravitating matter density through the existing DMO IC
    route; formed SSPs and BHs are enabled only in their physical variants.
    """
    import copy
    if not isinstance(variants,list) or not variants or len(set(variants))!=len(variants) or any(v not in {'dmo','hydro','hydro_bh'} for v in variants):raise ValueError('Choose distinct supported physics variants')
    if not isinstance(resolution_levels,list) or len(set(resolution_levels))!=len(resolution_levels) or any(type(n) is not int or not 8<=n<=8192 for n in resolution_levels):raise ValueError('Declare bounded distinct IC grid resolutions')
    designs=[]
    for variant in variants:
        for level in sorted(resolution_levels):
            cfg=copy.deepcopy(base_config);cfg['simulation']['physics']='dm' if variant=='dmo' else 'hydro'
            cfg['simulation']['particles_per_dimension']=level
            if variant=='dmo':
                cfg['hydro']={'cooling':False,'star_formation':False};cfg['stellar_evolution']={'enabled':False};cfg['chemistry']={'enabled':False}
            if variant!='hydro_bh':cfg['black_holes']={'enabled':False}
            elif not cfg.get('black_holes',{}).get('enabled',False):raise ValueError('Hydro+BH design requires an explicit enabled physical BH model in the base configuration')
            designs.append({'variant':variant,'resolution':level,'config':cfg})
    return {'schema':'matched-hydro-comparison-design-v1','configurations':designs,
        'execution_status':'NOT_RUN','matching_requirement':'Use the same pinned initial phase/noise realization, cosmology and seed; refine the same Lagrangian volume; compare actual common epochs',
        'physics_acceptance':'NOT_ACCEPTED'}


def sod_riemann_reference(positions,time, *, left,right,discontinuity,gamma):
    """Exact ideal-gas shock tube (Toro pressure iteration), no native execution.

    State triples are density, velocity and pressure in one declared consistent
    unit system. Output includes both shocks/rarefactions and the contact.
    """
    import numpy as np
    from scipy.optimize import brentq
    x=np.asarray(positions,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite([time,discontinuity,gamma]).all() or time<0 or gamma<=1:raise ValueError('Declare finite shock-tube coordinates, t>=0 and gamma>1')
    states=[]
    for value in (left,right):
        if len(value)!=3 or not np.isfinite(value).all() or value[0]<=0 or value[2]<=0:raise ValueError('Ideal-gas Riemann states require positive density/pressure')
        rho,u,p=map(float,value);states.append((rho,u,p,math.sqrt(gamma*p/rho)))
    l,r=states
    if r[1]-l[1]>=2*(l[3]+r[3])/(gamma-1):raise ValueError('Vacuum Riemann solutions require a separate native comparison profile')
    def wave(p,state):
        rho,u,p0,cs=state
        if p>p0:return (p-p0)*math.sqrt(2/((gamma+1)*rho)/(p+(gamma-1)/(gamma+1)*p0))
        return 2*cs/(gamma-1)*((p/p0)**((gamma-1)/(2*gamma))-1)
    def pressure(p):return wave(p,l)+wave(p,r)+r[1]-l[1]
    hi=max(l[2],r[2])
    while pressure(hi)<0:
        hi*=2
        if not math.isfinite(hi):raise ValueError('Riemann pressure bracket overflow')
    pstar=brentq(pressure,np.nextafter(0.,1.),hi,xtol=np.nextafter(0.,1.),rtol=8*np.finfo(float).eps)
    ustar=.5*(l[1]+r[1]+wave(pstar,r)-wave(pstar,l));rho=np.empty(len(x));velocity=np.empty(len(x));p=np.empty(len(x))
    if time==0:
        for mask,state in ((x<discontinuity,l),(x>=discontinuity,r)):rho[mask]=state[0];velocity[mask]=state[1];p[mask]=state[2]
    else:
        xi=(x-discontinuity)/time
        for i,speed in enumerate(xi):
            side=-1 if speed<=ustar else 1;state=l if side==-1 else r;rh,u,p0,cs=state
            if pstar>p0:
                shock=u+side*cs*math.sqrt((gamma+1)/(2*gamma)*pstar/p0+(gamma-1)/(2*gamma))
                untouched=speed<shock if side==-1 else speed>shock
                if untouched:rho[i],velocity[i],p[i]=rh,u,p0
                else:rho[i]=rh*(pstar/p0+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*pstar/p0+1);velocity[i]=ustar;p[i]=pstar
            else:
                cstar=cs*(pstar/p0)**((gamma-1)/(2*gamma));head=u+side*cs;tail=ustar+side*cstar
                if (side==-1 and speed<head) or (side==1 and speed>head):rho[i],velocity[i],p[i]=rh,u,p0
                elif (side==-1 and speed>tail) or (side==1 and speed<tail):rho[i]=rh*(pstar/p0)**(1/gamma);velocity[i]=ustar;p[i]=pstar
                else:
                    uf=2/(gamma+1)*(-side*cs+(gamma-1)/2*u+speed)
                    cf=2/(gamma+1)*(cs+side*(gamma-1)/2*(speed-u))
                    rho[i]=rh*(cf/cs)**(2/(gamma-1));velocity[i]=uf;p[i]=p0*(cf/cs)**(2*gamma/(gamma-1))
    return {'density':rho,'velocity':velocity,'pressure':p,'specific_internal_energy':p/((gamma-1)*rho),
        'star_pressure':pstar,'contact_velocity':ustar,'equation':'Ideal Euler Riemann solution, declared gamma and input units'}


def measure_hydro_comparison(request,destination):
    """Compare measured native radial gas/SSP/DM/BH states at common epochs.

    Every variant, seed, resolution and common initial-phase binding is explicit.
    Missing thermal/SSP populations stay unavailable; no formed stars are filled.
    This producer reads snapshots only when invoked later by the user.
    """
    import json,itertools
    from pathlib import Path
    import numpy as np,h5py
    from core.foundation import file_sha256 as digest, MSUN_G, SECONDS_PER_YEAR
    from halos import _snapshot_files
    from core.data import export_numeric_audit,verify_native_hdf5_set,native_publication_markers,dataset_to_cgs_factor
    required={'runs','radial_edges_proper_kpc','memory_gib','maximum_actual_epoch_difference','reference_label','relative_error_limits'}
    if not isinstance(request,dict) or set(request)!=required:raise ValueError('Declare complete matched hydro comparison request')
    edges=np.asarray(request['radial_edges_proper_kpc'],dtype=float);memory=request['memory_gib'];epoch_error=request['maximum_actual_epoch_difference']
    if edges.ndim!=1 or len(edges)<3 or edges[0]<=0 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):raise ValueError('Declare increasing positive proper radial edges')
    if type(memory) not in (int,float) or not np.isfinite(memory) or memory<=0 or type(epoch_error) not in (int,float) or not np.isfinite(epoch_error) or epoch_error<0:raise ValueError('Declare finite memory and actual-epoch tolerance')
    metric_names=['dm_mass_msun','gas_mass_msun','stellar_mass_msun','bh_gravity_mass_msun','gas_sfr_msun_year','gas_thermal_energy_erg']
    limits=request['relative_error_limits']
    if set(limits)!=set(metric_names) or any(type(v) not in (int,float) or not np.isfinite(v) or v<0 for v in limits.values()):raise ValueError('Freeze relative tolerances for every measured metric')
    runs=request['runs']
    if not isinstance(runs,list) or len(runs)<2:raise ValueError('Supply measured reference and comparison runs')
    payload={};specs={};records=[];sources={str(Path(__file__).resolve()):digest(__file__)};phase_by_seed={};labels=set()
    for ordinal,run in enumerate(runs):
        keys={'label','variant','seed','resolution','run_directory','snapshot','host_center_proper_kpc','phase_asset','phase_sha256'}
        if not isinstance(run,dict) or set(run)!=keys or not isinstance(run['label'],str) or not run['label'] or run['label'] in labels or run['variant'] not in {'dmo','hydro','hydro_bh'}:raise ValueError('Explicit unique variant labels and source bindings required')
        labels.add(run['label'])
        if any(type(run[k]) is not int or run[k]<0 for k in ('seed','resolution','snapshot')) or run['resolution']==0:raise ValueError('Exact seed, resolution and snapshot indices required')
        phase=Path(run['phase_asset']).resolve()
        if digest(phase)!=run['phase_sha256']:raise ValueError('Matched initial-phase source changed')
        binding=run['phase_sha256']
        if run['seed'] in phase_by_seed and phase_by_seed[run['seed']]!=binding:raise ValueError('Same-seed runs do not share the declared initial phase realization')
        phase_by_seed[run['seed']]=binding;sources[str(phase)]=binding;root=Path(run['run_directory']).resolve();center=np.asarray(run['host_center_proper_kpc'],dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('Declare a finite measured host center')
        files=_snapshot_files(root,run['snapshot']);verify_native_hdf5_set(files,require_publication=True)
        for path in files+native_publication_markers(files):sources[str(path.resolve())]=digest(path)
        values=np.zeros((len(edges)-1,len(metric_names)));available=np.ones_like(values,dtype=bool);actual=None;model=None
        for path in files:
            with h5py.File(path,'r') as f:
                params=f['Parameters'].attrs;time=float(f['Header'].attrs['Time']);h=float(params['HubbleParam']);mode=int(params['ComovingIntegrationOn']);a=time if mode else 1.
                stamp=(time,h,mode,float(f['Header'].attrs['BoxSize']),float(params['UnitLength_in_cm']),float(params['UnitMass_in_g']),float(params['UnitVelocity_in_cm_per_s']))
                if actual is not None and actual!=stamp:raise ValueError('Comparison snapshot shards disagree about actual epoch/model units')
                actual=stamp
                current_model={k:float(params[k]) for k in ('Omega0','OmegaLambda','HubbleParam') if k in params}
                if model is not None and model!=current_model:raise ValueError('Comparison shard cosmology mismatch')
                from core.data import native_box_proper_kpc,native_physical_unit_scales
                model=current_model;box=native_box_proper_kpc(f['Header'].attrs,params)
                scales=native_physical_unit_scales(unit_mass_g=stamp[5],unit_length_cm=stamp[4],unit_velocity_cm_s=stamp[6],comoving=mode,hubble_param=h)
                if edges[-1]>=box/2:raise ValueError('Comparison aperture exceeds unique proper periodic region')
                for kind,column in ((1,0),(0,1),(4,2),(5,3)):
                    group=f.get(f'PartType{kind}')
                    if group is None:continue
                    n=len(group['ParticleIDs'])
                    if n*256+32*1024**2>memory*1024**3:raise MemoryError('Comparison shard exceeds declared workspace')
                    xyz=group['Coordinates'][:]*dataset_to_cgs_factor(dict(group['Coordinates'].attrs),a,h)/3.0856775814913673e21
                    mass=group['Masses'][:]*dataset_to_cgs_factor(dict(group['Masses'].attrs),a,h)/MSUN_G if 'Masses' in group else np.full(n,f['Header'].attrs['MassTable'][kind])*scales['mass_g']/MSUN_G
                    if not np.isfinite(xyz).all() or not np.isfinite(mass).all() or np.any(mass<0):raise ValueError('Invalid native comparison mass/coordinates')
                    offsets=(xyz-center+box/2)%box-box/2;radii=np.linalg.norm(offsets,axis=1);bins=np.searchsorted(edges,radii,side='right')-1;valid=(bins>=0)&(bins<len(edges)-1)&(mass>0)
                    values[:,column]+=np.bincount(bins[valid],weights=mass[valid],minlength=len(edges)-1)
                    if kind==0:
                        for field,col,convert in [('StarFormationRate',4,SECONDS_PER_YEAR/MSUN_G),('InternalEnergy',5,MSUN_G)]:
                            if field not in group:available[:,col]=False;continue
                            raw=group[field][:]*dataset_to_cgs_factor(dict(group[field].attrs),a,h)*convert
                            if field=='InternalEnergy':raw*=mass
                            if not np.isfinite(raw).all() or np.any(raw<0):raise ValueError('Invalid gas comparison observable')
                            values[:,col]+=np.bincount(bins[valid],weights=raw[valid],minlength=len(edges)-1)
        if actual is None:raise ValueError('Missing comparison snapshot')
        if run['variant']=='dmo':available[:,1:]=False
        prefix=f'run_{ordinal:03d}_';payload[prefix+'measurements']=values;payload[prefix+'available']=available
        specs[prefix+'measurements']={'units':'mixed explicitly ordered columns','quantity':'radial native component and gas extensive metrics','annotations':{'columns':metric_names,'radial_edges_proper_kpc':edges.tolist()}}
        specs[prefix+'available']={'units':'dimensionless','quantity':'measured metric availability'}
        records.append({**run,'actual_native_time':actual[0],'comoving':bool(actual[2]),'cosmology':model,'payload_prefix':prefix})
    refs=[(i,r) for i,r in enumerate(records) if r['label']==request['reference_label']]
    if len(refs)!=1:raise ValueError('Select one measured reference label')
    ri,ref=refs[0];comparisons=[]
    for i,row in enumerate(records):
        if i==ri:continue
        if row['seed']!=ref['seed']:continue # Independent seeds measure variance, never paired cancellation.
        if row['cosmology']!=ref['cosmology'] or row['comoving']!=ref['comoving'] or abs(row['actual_native_time']-ref['actual_native_time'])>epoch_error:raise ValueError('Paired comparison differs in cosmology or common actual epoch')
        left=payload[ref['payload_prefix']+'measurements'];right=payload[row['payload_prefix']+'measurements'];valid=payload[ref['payload_prefix']+'available']&payload[row['payload_prefix']+'available']
        scale=np.maximum(abs(left),abs(right));supported=valid&(scale>0);delta=right-left;relative=np.divide(abs(delta),scale,out=np.zeros_like(scale),where=scale>0)
        passed=supported&(relative<=np.array([limits[k] for k in metric_names])[None,:])
        prefix=f'comparison_{i:03d}_'
        for name,array in [('difference',delta),('relative_difference',relative),('supported',supported),('within_declared_limit',passed)]:
            payload[prefix+name]=array;specs[prefix+name]={'units':'mixed metric columns' if name=='difference' else 'dimensionless','quantity':name,'annotations':{'columns':metric_names}}
        comparisons.append({'reference':ref['label'],'comparison':row['label'],'supported_cells':int(supported.sum()),'cells_outside_limit':int((supported&~passed).sum()),'role':'resolution' if row['variant']==ref['variant'] else 'physics_response','physical_acceptance':'NOT_ACCEPTED'})
    if not comparisons:raise ValueError('No same-seed common-epoch comparison was supplied')
    # Independent realizations are stratified by physics and resolution;
    # they measure sample variance rather than masquerading as paired errors.
    strata={}
    for row in records:strata.setdefault((row['variant'],row['resolution']),[]).append(row)
    cells=(len(edges)-1)*len(metric_names)
    if len(strata)*cells*cells*32+len(payload)*cells*32+64*1024**2>memory*1024**3:raise MemoryError('Full measured ensemble covariance exceeds declared workspace')
    ensembles=[]
    for ordinal,(key,rows) in enumerate(sorted(strata.items())):
        if len({row['seed'] for row in rows})!=len(rows):raise ValueError('Duplicate seed in a physics/resolution ensemble')
        if any(row['cosmology']!=ref['cosmology'] or row['comoving']!=ref['comoving'] or abs(row['actual_native_time']-ref['actual_native_time'])>epoch_error for row in rows):raise ValueError('Independent ensemble lacks a matched common actual epoch/model')
        values=np.stack([payload[row['payload_prefix']+'measurements'].reshape(-1) for row in rows])
        valid=np.stack([payload[row['payload_prefix']+'available'].reshape(-1) for row in rows])
        count=valid.sum(axis=0);mean=np.divide(np.sum(np.where(valid,values,0),axis=0),count,out=np.zeros(cells),where=count>0)
        covariance=np.zeros((cells,cells));pair_count=valid.astype(np.int64).T@valid.astype(np.int64)
        for left in range(cells):
            for right in range(left,cells):
                use=valid[:,left]&valid[:,right];n=int(use.sum())
                if n>1:
                    dx=values[use,left]-values[use,left].mean();dy=values[use,right]-values[use,right].mean()
                    covariance[left,right]=covariance[right,left]=np.dot(dx,dy)/(n-1)
        prefix=f'ensemble_{ordinal:03d}_'
        for name,array in [('mean',mean),('sample_covariance',covariance),('pair_count',pair_count),('covariance_available',pair_count>1),('mean_available',count>0)]:
            payload[prefix+name]=array;specs[prefix+name]={'units':'mixed native metric columns' if name=='mean' else 'products of mixed native metric columns' if name=='sample_covariance' else 'dimensionless','quantity':name,'annotations':{'flattening':'radial bin then explicitly ordered metric column','columns':metric_names}}
        ensembles.append({'variant':key[0],'resolution_level':key[1],'seeds':[row['seed'] for row in rows],
            'payload_prefix':prefix,'uncertainty_scope':'Unbiased sample covariance of independent measured runs; pairwise missingness can yield a non-positive-semidefinite matrix, never invert it as a likelihood without an explicit supported estimator'})
    report={'schema':'measured-hydro-comparison-v1','request':request,'runs':records,'comparisons':comparisons,'ensembles':ensembles,'source_hashes':sources,
        'interpretation':'Aperture mass/energy differences, with zero/absent denominators masked. Multiple seeds and resolution levels are retained; this product does not establish statistical or numerical convergence by itself.',
        'physics_acceptance':'NOT_ACCEPTED'}
    for path,checksum in sources.items():
        if digest(path)!=checksum:raise ValueError('Hydro comparison input changed before publication')
    return export_numeric_audit(destination,report,payload,specs,algorithm=__file__,context={'selection':'native matched hydro radial metrics','columns':metric_names},source_hashes=sources)


def sh03_equilibrium_reference(*,density,threshold_density,maximum_star_formation_time,cooling_time,
    cold_specific_energy,supernova_specific_energy,prompt_fraction,evaporation_factor):
    """Stable SH03 cloud fraction and effective energy in declared native units.

    The physical cooling time is an input measured from the selected cooling
    backend, not an invented universal cooling curve. Net retained SFR differs
    from the associated gross formation rate by the declared prompt fraction.
    """
    values=[density,threshold_density,maximum_star_formation_time,cooling_time,cold_specific_energy,
        supernova_specific_energy,prompt_fraction,evaporation_factor]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in values) or not 0<prompt_fraction<1 or density<threshold_density:
        raise ValueError('Declare a supported positive SH03 equilibrium state above the physical threshold')
    tstar=maximum_star_formation_time*math.sqrt(threshold_density/density)
    evaporation=evaporation_factor*(density/threshold_density)**(-.8)
    hot=supernova_specific_energy/(1+evaporation)+cold_specific_energy
    budget=prompt_fraction*supernova_specific_energy-(1-prompt_fraction)*cold_specific_energy
    if budget<=0:raise ValueError('SH03 feedback/evaporation parameters have no supported cloud equilibrium')
    y=tstar/cooling_time*hot/budget
    # Algebraically equal to 1+1/(2y)-sqrt(1/y+1/(4y^2)),
    # without catastrophic cancellation in a cooling-dominated small-y state.
    cloud=2*y/(1+2*y+math.sqrt(1+4*y))
    return {'cloud_mass_fraction':cloud,'hot_mass_fraction':1-cloud,'hot_specific_energy':hot,
        'effective_specific_energy':hot*(1-cloud)+cold_specific_energy*cloud,
        'star_formation_time':tstar,'retained_sfr_per_gas_mass':(1-prompt_fraction)*cloud/tstar,
        'gross_sfr_per_gas_mass':cloud/tstar,'evaporation_factor':evaporation,
        'equation_source':'Springel & Hernquist 2003; native sfr_eos.cc equilibrium equations'}
