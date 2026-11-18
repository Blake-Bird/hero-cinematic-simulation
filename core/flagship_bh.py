"""Registered BH histories from the existing native BH analysis producer."""
from pathlib import Path
import json
import numpy as np
from .foundation import file_sha256,verify_product_provenance
from .data import export_numeric_audit,verify_hdf5_dataset_payloads,verify_dataset_semantics,exact_uint64_ids
from core.research.display import export_scientific_suite


def bh_figure_input_bindings(directory,*,run_directory=None,memory_gib=1.):
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite BH figure allocation required')
    directory=Path(directory).resolve();provenance=directory/'provenance.json';history=directory/'history.hdf5'
    if provenance.stat().st_size*20>memory_gib*1024**3/2:raise MemoryError('Native BH metadata exceeds allocation')
    if file_sha256(provenance)!=(directory/'provenance.sha256').read_text().strip():raise ValueError('Native BH provenance changed')
    record=json.loads(provenance.read_text());contract=record['data_contract']
    if record.get('schema')!='darkmattersim-native-bh-history-v1' or file_sha256(history)!=record['history_sha256']:
        raise ValueError('Native BH history integrity changed')
    if run_directory is None:
        candidates=[p for p in Path(contract['parameter_file']['path']).resolve().parents
            if (p/'config.yaml').is_file() and (p/'manifest.json').is_file()
            and file_sha256(p/'config.yaml')==contract['config_sha256'] and file_sha256(p/'manifest.json')==contract['run_manifest_sha256']]
        if len(candidates)!=1:raise ValueError('Supply the exact native BH run directory')
        run=candidates[0]
    else:run=Path(run_directory).resolve()
    verify_product_provenance(run,contract);verify_hdf5_dataset_payloads(history)
    sources={str(provenance):file_sha256(provenance),str(history):file_sha256(history),
        str(run/'config.yaml'):file_sha256(run/'config.yaml'),str(run/'manifest.json'):file_sha256(run/'manifest.json')}
    rows=(contract['input_hashes']+contract['initial_conditions']+list(contract['boltzmann_inputs'].values())
        +[contract['executable'],contract['parameter_file'],contract['compile_options']]+contract['algorithm_sources'])
    sources.update({r['path']:r['sha256'] for r in rows if r.get('status')!='UNAVAILABLE'})
    for epoch in record['epochs']:
        for r in epoch['source_sha256']:
            if file_sha256(r['path'])!=r['sha256']:raise ValueError('Native BH source snapshot changed')
            sources[r['path']]=r['sha256']
    return record,history,run,sources


def export_native_bh_figures(directory,destination,*,run_directory=None,memory_gib=1.):
    import h5py
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native BH figures')
    record,history,run,sources=bh_figure_input_bindings(directory,run_directory=run_directory,memory_gib=memory_gib)
    namespace=record['data_contract']['identity_namespaces']['black_hole'];epochs=record['epochs']
    scales=np.array([e['scale_factor'] for e in epochs],float)
    if not len(scales) or not np.isfinite(scales).all() or np.any(scales<=0) or np.any(np.diff(scales)<=0):raise ValueError('Actual BH epochs must strictly increase')
    saved=[e['snapshot'] for e in epochs]
    if any(type(s) is not int or s<0 for s in saved) or len(set(saved))!=len(saved):raise ValueError('Exact unique native BH saved epochs required')
    summary={}
    for row in record['history']:
        exact_uint64_ids([row['particle_id']]);key=(row['snapshot'],row['particle_id'])
        if key in summary:raise ValueError('Duplicate exact BH summary observation')
        summary[key]=row
    tracks={};observed=set()
    with h5py.File(history) as f:
        count=sum(len(g['ParticleIDs']) for g in f['Epochs'].values())
        grid=len(scales)*len({identifier for _,identifier in summary})
        if count*4096+len(summary)*2048+grid*256>memory_gib*1024**3/2:raise MemoryError('Native BH track workspace exceeds allocation')
        for ordinal,epoch in enumerate(epochs):
            g=f['Epochs/%03d'%epoch['snapshot']];meta=json.loads(g.attrs['metadata_json'])
            if meta!=epoch or epoch['epoch_contract']['actual_scale_factor']!=scales[ordinal]:raise ValueError('BH metadata differs from original actual epoch')
            ids=exact_uint64_ids(g['ParticleIDs'][:]);values={}
            for key,unit in (('ParticleIDs','identity'),('BH_Mass','Msun'),('BH_FeedbackEnergy','erg'),('BH_InflowRate','Msun/year')):
                d=g[key];verify_dataset_semantics(d)
                if d.attrs['units']!=unit or json.loads(d.attrs['epoch_json'])!=epoch['epoch_contract']:raise ValueError('Native BH units or epoch differ: '+key)
                if key=='ParticleIDs' and d.attrs['identity_namespace']!=namespace:raise ValueError('Native BH identity namespace differs')
                if d.shape!=ids.shape:raise ValueError('Native BH field rows do not align')
                values[key]=d[:]
            for i,identifier in enumerate(ids):
                key=(epoch['snapshot'],int(identifier));row=summary.get(key)
                if row is None or row['scale_factor']!=scales[ordinal]:raise ValueError('BH summary loses original exact epoch-qualified identity')
                for field,name in (('BH_Mass','subgrid_mass_msun'),('BH_FeedbackEnergy','coupled_energy_erg'),('BH_InflowRate','inflow_msun_year')):
                    if values[field][i]!=row[name]:raise ValueError('BH summary differs from original native field: '+field)
                ratio=row['mean_inflow_over_saved_mass_eddington_cap']
                vector=np.array([values['BH_Mass'][i],values['BH_FeedbackEnergy'][i],ratio],float)
                if not np.isfinite(vector).all() or vector[0]<=0 or np.any(vector[1:]<0):raise ValueError('Finite positive BH mass and nonnegative native energy/accretion ratio required')
                if int(identifier) not in tracks:tracks[int(identifier)]=np.full((len(scales),3),np.nan)
                tracks[int(identifier)][ordinal]=vector;observed.add(key)
    if observed!=set(summary):raise ValueError('BH summary has observations absent from native history')
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out.mkdir(parents=True);products={};figures=[]
    epoch_contract={'row_epoch_contracts':[e['epoch_contract'] for e in epochs],'canonical_time':'actual saved scale factor'}
    for identifier,values in sorted(tracks.items()):
        name='bh-'+str(identifier);product=out/(name+'.json');products[name]=str(product)
        arrays={'a':scales,'mass':values[:,0],'energy':values[:,1],'eddington_ratio':values[:,2],
            'present':np.isfinite(values).all(axis=1),'particle_ids':np.full(len(scales),identifier,dtype=np.uint64)}
        units={'a':'dimensionless','mass':'Msun','energy':'erg','eddington_ratio':'dimensionless','present':'dimensionless','particle_ids':'identity'}
        specs={key:{'units':unit,'quantity':key,'missing_policy':'missing native epochs remain NaN/false; no interpolation or merger inference'} for key,unit in units.items()}
        specs['particle_ids']['identity_namespace']=namespace
        export_numeric_audit(product,{'schema':'native-bh-figure-bridge-v2','epoch':epoch_contract,'particle_id':identifier,
            'native_run':str(run),'event_cadence':record.get('event_cadence'),
            'eddington_definition':'existing producer mean source-step inflow / Eddington cap evaluated at saved BH mass; not instantaneous luminosity ratio',
            'native_parameters':record['parameters'],'science_acceptance':'NOT_ACCEPTED'},arrays,specs,
            algorithm=__file__,context={'epoch':epoch_contract},source_hashes=sources)
        for variant,y,title in (('bh_growth','mass','Subgrid BH mass'),('feedback_energy','energy','Deposited feedback energy'),
            ('eddington_ratio','eddington_ratio','Mean inflow / saved-mass Eddington cap')):
            figures.append({'figure_id':name+'-'+variant,'family':'black_holes','variant':variant,'title':title+'; ID '+str(identifier),
                'science_status':'NOT_ACCEPTED','epoch':epoch_contract,'panels':[{'panel_id':'history','title':'Original saved BH observations',
                    'quality_status':'PENDING_CONVERGENCE','selection':'exact native ID; all original epochs with missingness retained',
                    'xlabel':'Actual scale factor a','ylabel':title+' ['+units[y]+']','x_units':'dimensionless','y_units':units[y],
                    'series':[{'source':name,'kind':kind,'x':'a','y':y,'support':'present','label':label,'marker_size':8}
                        for kind,label in (('line','native history; gaps retained'),('scatter','saved samples'))]}]})
    for source,expected in sources.items():
        if file_sha256(source)!=expected:raise ValueError('Native BH input changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


def bh_event_input_bindings(product,*,memory_gib):
    from .stars import verify_native_hydro_history
    from core.stream_common import read_product
    from .data import numeric_audit_source_bindings
    path=Path(product).resolve()
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite BH event allocation required')
    if path.stat().st_size*20>memory_gib*1024**3/3:raise MemoryError('Native source journal report exceeds allocation')
    report,_,pins=read_product(path,memory_gib=memory_gib/3)
    # The existing verifier replays the full native accounting/counter chain.
    # Respect its declared allocation before invoking that independent intake.
    if report.get('schema')!='native-hydro-history-v1' or report['request']['memory_gib']>memory_gib:
        raise MemoryError('Native hydro replay requires its recorded workspace allocation')
    if report!=verify_native_hydro_history(path):raise ValueError('Native hydro journal changed during verification')
    return report,{**pins,**numeric_audit_source_bindings(report)}


def native_merger_timeline_arrays(report,convert_time):
    """Preserve original merger rows, source-step brackets and exact parent IDs."""
    rows=[];times=[];brackets=[];parents=[];survivors=[]
    for interval in report['intervals']:
        for row in interval['black_hole_rows']:
            if row['kind']!='merger':continue
            ids=exact_uint64_ids(row['parent_ids'],allow_empty=False)
            if len(ids)!=2 or int(ids[0])!=row['particle_id']:raise ValueError('Native merger survivor/parent identities differ')
            begin,end=row['native_step_begin'],row['native_time']
            if not np.isfinite([begin,end]).all() or begin>end:raise ValueError('Native merger source-step bracket is invalid')
            bracket=[convert_time(begin),convert_time(end)]
            if not np.isfinite(bracket).all() or bracket[0]>bracket[1]:raise ValueError('Merger proper-time bracket is invalid')
            times.append(bracket[1]);brackets.append(bracket);parents.append(ids);survivors.append(row['particle_id']);rows.append(row)
    order=np.argsort(times,kind='stable')
    arrays={'applied_time_myr':np.asarray(times,float)[order],
        'source_step_bracket_myr':np.asarray(brackets,float).reshape(-1,2)[order],
        'parent_ids':np.asarray(parents,dtype=np.uint64).reshape(-1,2)[order],
        'survivor_ids':exact_uint64_ids(survivors,unique=False)[order],
        'event_number':np.arange(len(rows),dtype=np.int64)}
    return [rows[i] for i in order],arrays


def export_native_bh_merger_timeline(product,destination,*,memory_gib=1.):
    from .cosmology import run_proper_time_clock,interpolate_proper_time
    from .foundation import SECONDS_PER_YEAR
    report,sources=bh_event_input_bindings(product,memory_gib=memory_gib)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native BH timeline')
    metadata=report['epochs'];parameters=metadata[0]['parameters']
    run=report['request']['run_directory']
    if parameters['comoving']:
        clock=run_proper_time_clock(run)
        if clock[2]['status']!='VERIFIED':raise ValueError('Native merger timeline requires a bound proper-time clock')
        sources[clock[2]['path']]=clock[2]['sha256']
        convert=lambda value:1000*interpolate_proper_time(clock[0],clock[1],value)
    else:
        convert=lambda value:value*parameters['unit_time_s']/SECONDS_PER_YEAR/1.e6
    rows,arrays=native_merger_timeline_arrays(report,convert)
    epoch={'canonical_time':'source-step applied proper time','saved_epoch_contracts':[e['epoch'] for e in metadata]}
    namespace=report['provenance']['identity_namespaces']['black_hole']
    units={k:'Myr' if k.endswith('_myr') else 'identity' if k.endswith('_ids') else 'dimensionless' for k in arrays}
    definitions={k:{'units':u,'quantity':k,'missing_policy':'complete native journal census; zero mergers retained explicitly'} for k,u in units.items()}
    for key in ('parent_ids','survivor_ids'):definitions[key]['identity_namespace']=namespace
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out.mkdir(parents=True);target=out/'native-merger-timeline.json'
    export_numeric_audit(target,{'schema':'native-bh-merger-timeline-v1','epoch':epoch,'original_native_rows':rows,
        'hydro_history':str(Path(product).resolve()),'native_lineages':[i['black_hole_lineage'] for i in report['intervals']],
        'native_accounting_status':report.get('status','UNDECLARED'),
        'status':'NATIVE_MERGERS_RECORDED' if rows else 'NO_NATIVE_MERGERS_IN_COMPLETE_JOURNAL',
        'event_time_definition':'operator applied at native source-step end; continuous occurrence remains within retained proper-time bracket',
        'science_acceptance':'NOT_ACCEPTED'},arrays,definitions,algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    return export_scientific_suite({'sources':{'events':str(target)},'dpi':160,'maximum_panels_per_figure':1,'organized_outputs':True,
        'figures':[{'figure_id':'native-bh-merger-timeline','family':'black_holes','variant':'merger_timeline','title':'Committed native BH mergers',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'mergers','title':'Applied operator epochs; brackets retained in numerical product',
                'quality_status':'EVENT_TIME_BRACKETED','selection':'original validated native merger operations only',
                'xlabel':'Applied proper time [Myr]','ylabel':'Event ordinal [dimensionless]','x_units':'Myr','y_units':'dimensionless',
                'series':[{'source':'events','kind':'scatter','x':'applied_time_myr','y':'event_number','label':'native mergers','marker_size':12}]}]}]},
        out/'gallery',base=out,memory_gib=memory_gib)


def export_native_bh_pair_separation(directory,request,destination,*,run_directory=None,memory_gib=1.):
    """Saved-epoch minimum-image separation for explicitly selected native IDs.

    This measures spatial separation, not binary binding, merger causality or
    temporal winding. Missing either identity leaves the epoch unresolved.
    """
    import h5py
    from .data import hdf5_scalar_attr
    ids=exact_uint64_ids(request['particle_ids'],allow_empty=False)
    if request.get('schema')!='native-bh-pair-separation-v1' or len(ids)!=2:
        raise ValueError('Two distinct exact native BH identities required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native BH pair figures')
    record,history,run,sources=bh_figure_input_bindings(directory,run_directory=run_directory,memory_gib=memory_gib)
    namespace=record['data_contract']['identity_namespaces']['black_hole']
    if request['namespace']!=namespace:raise ValueError('BH pair namespace differs from native history')
    options=record['data_contract']['compile_options']
    if options.get('status')=='UNAVAILABLE':raise ValueError('Native periodic build evidence required for BH pair geometry')
    flags={line.split('=',1)[0].strip() for line in Path(options['path']).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')}
    if 'PERIODIC' not in flags:raise ValueError('BH pair minimum-image geometry requires actual PERIODIC build')
    n=len(record['epochs']);position=np.full((n,2,3),np.nan);scales=np.empty(n);boxes=np.empty(n)
    if n*2048>memory_gib*1024**3/3:raise MemoryError('BH pair output exceeds allocation')
    with h5py.File(history) as f:
        for ordinal,epoch in enumerate(record['epochs']):
            scales[ordinal]=epoch['scale_factor'];g=f['Epochs/%03d'%epoch['snapshot']]
            for name,unit in (('ParticleIDs','identity'),('Coordinates','proper kpc')):
                verify_dataset_semantics(g[name])
                if g[name].attrs['units']!=unit or json.loads(g[name].attrs['epoch_json'])!=epoch['epoch_contract']:
                    raise ValueError('Native BH pair position/identity units or epoch differ')
            if g['ParticleIDs'].attrs['identity_namespace']!=namespace:raise ValueError('BH pair identity namespace changed')
            if g['ParticleIDs'].size*256>memory_gib*1024**3/3:raise MemoryError('Native pair epoch intake exceeds allocation')
            available=exact_uint64_ids(g['ParticleIDs'][:]);xyz=g['Coordinates'][:]
            if xyz.shape!=(len(available),3) or not np.isfinite(xyz).all():raise ValueError('Finite aligned native BH coordinates required')
            for j,identifier in enumerate(ids):
                at=np.flatnonzero(available==identifier)
                if len(at):position[ordinal,j]=xyz[at[0]]
            frame=[]
            for source in epoch['source_sha256']:
                with h5py.File(source['path']) as native:
                    params=native['Parameters'].attrs
                    if int(hdf5_scalar_attr(params['ComovingIntegrationOn'],'ComovingIntegrationOn'))!=1:
                        raise ValueError('This pair route requires the producer cosmological proper-coordinate convention')
                    frame.append(hdf5_scalar_attr(native['Header'].attrs['BoxSize'],'BoxSize')*scales[ordinal]/float(params['HubbleParam']))
            if not frame or not np.isfinite(frame).all() or min(frame)<=0 or not np.allclose(frame,frame[0],rtol=1e-12,atol=0):
                raise ValueError('Actual snapshot shards disagree on proper periodic box')
            boxes[ordinal]=frame[0]
    if not np.isfinite(scales).all() or np.any(scales<=0) or np.any(np.diff(scales)<=0):raise ValueError('BH pair actual epochs must strictly increase')
    difference=position[:,1]-position[:,0]
    difference=(difference+boxes[:,None]/2)%boxes[:,None]-boxes[:,None]/2
    support=np.isfinite(position).all(axis=(1,2));separation=np.linalg.norm(difference,axis=1)
    arrays={'scale_factor':scales,'separation_proper_kpc':separation,'supported':support,
        'selected_particle_ids':ids,'saved_position_proper_kpc':position,'proper_periodic_box_kpc':boxes}
    units={'scale_factor':'dimensionless','separation_proper_kpc':'proper kpc','supported':'dimensionless',
        'selected_particle_ids':'identity','saved_position_proper_kpc':'proper kpc','proper_periodic_box_kpc':'proper kpc'}
    specs={k:{'units':u,'quantity':k,'missing_policy':'NaN and false support when either exact native BH ID is absent'} for k,u in units.items()}
    specs['selected_particle_ids']['identity_namespace']=namespace
    epoch={'row_epoch_contracts':[e['epoch_contract'] for e in record['epochs']]}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out.mkdir(parents=True);target=out/'pair-separation.json'
    export_numeric_audit(target,{'schema':'native-bh-pair-separation-product-v1','epoch':epoch,'request':request,
        'native_run':str(run),'selection_scope':'explicit exact native IDs, no nearest-neighbor pair matching',
        'geometry':'per-epoch minimum-image distance in original periodic proper box; no temporal unwrapping',
        'binary_status':'NOT_INFERRED_FROM_SPATIAL_SEPARATION','science_acceptance':'NOT_ACCEPTED'},arrays,specs,
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Native BH pair source changed')
    return export_scientific_suite({'sources':{'pair':str(target)},'dpi':160,'maximum_panels_per_figure':1,'organized_outputs':True,
        'figures':[{'figure_id':'native-bh-pair-separation','family':'black_holes','variant':'pair_separation','title':'Selected native BH separation',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'pair','title':'Saved-epoch minimum-image distance',
                'quality_status':'BINARY_BINDING_NOT_INFERRED','selection':namespace+'; exact IDs '+str(ids.tolist()),
                'xlabel':'Actual scale factor a','ylabel':'Separation [proper kpc]','x_units':'dimensionless','y_units':'proper kpc',
                'series':[{'source':'pair','kind':kind,'x':'scale_factor','y':'separation_proper_kpc','support':'supported','label':label}
                    for kind,label in (('line','saved separation; gaps retained'),('scatter','native observations'))]}]}]},out/'gallery',base=out,memory_gib=memory_gib)


def recoil_saved_arrays(epochs,positions,present,images,event):
    """Actual saved remnant positions after a recorded merger, without interpolation."""
    n=len(epochs);position=np.asarray(positions,float);mask=np.asarray(present);offset=np.asarray(images)
    if position.shape!=(n,3) or mask.shape!=(n,) or mask.dtype.kind!='b' or offset.shape!=(n,3) or offset.dtype.kind not in 'iu':
        raise ValueError('Aligned saved positions, boolean presence and recorded integer images required')
    times=np.array([e['time_myr'] for e in epochs],float);boxes=np.array([e['box_proper_kpc'] for e in epochs],float)
    native=np.array([e['native_time'] for e in epochs],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.isfinite(boxes).all() or np.any(boxes<=0):
        raise ValueError('Ordered actual proper clocks and positive native boxes required')
    supported=mask&np.isfinite(position).all(axis=1)
    # A later merger changes remnant ancestry; caller restricts this original
    # selected event's displayed support before the next native merger.
    supported&=False if event is None else native>=event['native_time']
    unwrapped=position+offset*boxes[:,None]
    return {'times_myr':times,'position_proper_kpc':unwrapped,'supported':supported,
        'recorded_image_offsets':offset,'proper_periodic_box_kpc':boxes}


def export_native_bh_recoil_trajectory(directory,request,destination,*,run_directory=None,memory_gib=1.):
    import h5py
    from core.stream_common import read_product
    from .data import numeric_audit_source_bindings,hdf5_scalar_attr
    from .flagship_products import native_history_hubble_rates
    if request.get('schema')!='native-bh-recoil-trajectory-v1':raise ValueError('Explicit native recoil trajectory request required')
    identifier=request['survivor_id'];counter=request['event_counter']
    exact_uint64_ids([identifier])
    if type(counter) is not int or not 0<=counter<2**64:raise ValueError('Exact native merger event counter required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep prior recoil trajectory evidence')
    record,history,run,sources=bh_figure_input_bindings(directory,run_directory=run_directory,memory_gib=memory_gib)
    namespace=record['data_contract']['identity_namespaces']['black_hole']
    if request['namespace']!=namespace:raise ValueError('Recoil survivor namespace differs')
    binding=request['hydro_history'];product=Path(binding['path']).resolve()
    if file_sha256(product)!=binding['sha256']:raise ValueError('Native recoil journal binding changed')
    journal,pins=bh_event_input_bindings(product,memory_gib=memory_gib);sources.update(pins)
    if Path(journal['request']['run_directory']).resolve()!=run or journal['provenance']['identity_namespaces']['black_hole']!=namespace:
        raise ValueError('Recoil history and native event journal belong to different runs/lineages')
    mergers=[row for interval in journal['intervals'] for row in interval['black_hole_rows'] if row['kind']=='merger']
    selected=[row for row in mergers if row['particle_id']==identifier and row['counter']==counter]
    if len(selected)>1:raise ValueError('Ambiguous native recoil merger operation')
    event=selected[0] if selected else None
    image=request['recorded_images'];path=Path(image['path']).resolve()
    if file_sha256(path)!=image['sha256']:raise ValueError('Recorded recoil images changed')
    img,data,pins=read_product(path,memory_gib=memory_gib/4);sources.update(pins);sources.update(numeric_audit_source_bindings(img))
    if img['numeric_data']['array_contract']['datasets']['image_offsets']['units']!='dimensionless':
        raise ValueError('Recorded recoil image offsets require dimensionless integer units')
    snapshots=[e['snapshot'] for e in record['epochs']]
    if (img.get('schema')!='native-bh-recoil-images-v1' or img['history_sha256']!=file_sha256(history)
            or img['particle_id']!=identifier or img['namespace']!=namespace or img['snapshots']!=snapshots):
        raise ValueError('Recoil images must bind original history, exact ID/namespace and all saved epochs')
    n=len(snapshots);positions=np.full((n,3),np.nan);present=np.zeros(n,bool);epochs=[]
    with h5py.File(history) as f:
        for i,epoch in enumerate(record['epochs']):
            group=f['Epochs/%03d'%epoch['snapshot']]
            for key,unit in (('ParticleIDs','identity'),('Coordinates','proper kpc')):
                verify_dataset_semantics(group[key])
                if group[key].attrs['units']!=unit or json.loads(group[key].attrs['epoch_json'])!=epoch['epoch_contract']:
                    raise ValueError('Native recoil position/identity units or epoch differ')
            if group['ParticleIDs'].size*256>memory_gib*1024**3/4:raise MemoryError('Recoil native identity intake exceeds allocation')
            ids=exact_uint64_ids(group['ParticleIDs'][:]);at=np.flatnonzero(ids==identifier)
            if len(at):positions[i]=group['Coordinates'][int(at[0])];present[i]=True
            clock=epoch['epoch_contract'];time=clock.get('cosmic_time_gyr')
            if clock['canonical_time']!='scale_factor' or time is None:raise ValueError('Bound cosmological recoil proper-time clock required')
            source=epoch['source_sha256'][0]
            with h5py.File(source['path']) as native:
                box=hdf5_scalar_attr(native['Header'].attrs['BoxSize'],'BoxSize')*epoch['scale_factor']/float(native['Parameters'].attrs['HubbleParam'])
            epochs.append({'time_myr':time*1000.,'native_time':epoch['scale_factor'],'box_proper_kpc':box})
    # Validate recorded times against the existing original-run clock adapter;
    # positions need no added velocity or speculative event extrapolation.
    native_history_hubble_rates(run,[e['epoch_contract'] for e in record['epochs']],np.array([e['time_myr'] for e in epochs]),memory_gib=memory_gib,sources=sources)
    arrays=recoil_saved_arrays(epochs,positions,present,data['image_offsets'],event)
    if 'image_supported' in data:
        image_support=np.asarray(data['image_supported'])
        if image_support.shape!=(len(epochs),) or image_support.dtype.kind!='b':
            raise ValueError('Exact per-epoch native recoil image support required')
        arrays['supported']&=image_support
    if event is not None:
        later=[row['native_time'] for row in mergers if identifier in row['parent_ids'] and row['native_time']>event['native_time']]
        if later:arrays['supported']&=np.array([e['native_time'] for e in epochs])<min(later)
    units={key:'Myr' if key=='times_myr' else 'proper kpc' if key in ('position_proper_kpc','proper_periodic_box_kpc') else 'dimensionless' for key in arrays}
    epoch={'row_epoch_contracts':[e['epoch_contract'] for e in record['epochs']]};sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out.mkdir(parents=True);target=out/'recoil-trajectory.json'
    export_numeric_audit(target,{'schema':'native-bh-recoil-trajectory-product-v1','epoch':epoch,'request':request,
        'original_native_merger':event,'status':'NATIVE_EVENT_AND_SAVED_TRAJECTORY' if event and arrays['supported'].any() else 'NO_SUPPORTED_RECORDED_RECOIL_TRAJECTORY',
        'native_accounting_status':journal['status'],'trajectory_scope':'original remnant saved positions after the selected native merger until later ancestry change; not an inferred recoil-only causal displacement',
        'interpolation':'NONE; recorded images applied once; missing epochs retained','science_acceptance':'NOT_ACCEPTED'},arrays,
        {key:{'units':unit,'quantity':key,'missing_policy':'unsupported native event/ancestry/epoch masked'} for key,unit in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    return export_scientific_suite({'sources':{'recoil':str(target)},'dpi':160,'maximum_panels_per_figure':1,'organized_outputs':True,
        'figures':[{'figure_id':'native-bh-recoil-trajectory','family':'black_holes','variant':'recoil_trajectory','title':'Saved remnant trajectory after recorded merger',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'trajectory','projection':'3d','title':'Original saved positions; no interpolation',
                'quality_status':'CAUSAL_RECOIL_RESPONSE_UNRESOLVED','selection':'exact native survivor/event counter; recorded images and original missingness',
                'xlabel':'x [proper kpc]','ylabel':'y [proper kpc]','zlabel':'z [proper kpc]',
                'x_units':'proper kpc','y_units':'proper kpc','z_units':'proper kpc','series':[{
                    'source':'recoil','kind':'scatter3d','x':'position_proper_kpc','x_column':0,'y':'position_proper_kpc','y_column':1,
                    'z':'position_proper_kpc','z_column':2,'support':'supported','label':'native saved remnant'}]}]}]},out/'gallery',base=out,memory_gib=memory_gib)
