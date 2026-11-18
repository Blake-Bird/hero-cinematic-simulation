"""Native stellar-mass and Vmax abundances with declared catalogue selections.

Uses existing native exporters, source verifiers and Garwood interval estimator.
A complete uniform periodic parent is required for a whole-box number density.
"""
from pathlib import Path
import json
import zipfile
import numpy as np
from .foundation import file_sha256, atomic_json, verify_product_provenance
from core.resources import initial_condition_census
from .data import export_numeric_audit,exact_uint64_ids,exact_id_join,verify_dataset_semantics
from core.research.display import export_scientific_suite


def logarithmic_abundance(values,selected,edges,volume_mpc3):
    from .analysis import poisson_rate_interval
    values=np.asarray(values,float);selected=np.asarray(selected);edges=np.asarray(edges,float)
    if (values.ndim!=1 or selected.dtype.kind!='b' or selected.shape!=values.shape
            or edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all()
            or np.any(edges<=0) or np.any(np.diff(edges)<=0)
            or isinstance(volume_mpc3,bool) or not np.isfinite(volume_mpc3) or volume_mpc3<=0):
        raise ValueError('Aligned measured values, boolean selection, positive logarithmic edges and volume required')
    support=selected&np.isfinite(values)&(values>0)
    counts=np.histogram(values[support],bins=edges)[0]
    widths=np.diff(np.log10(edges));density=counts/(volume_mpc3*widths)
    lower,upper=poisson_rate_interval(counts,volume_mpc3,widths)
    return {'bin_center':np.sqrt(edges[:-1]*edges[1:]),'bin_edges':edges,'counts':counts,
        'number_density':density,'poisson_lower':lower,'poisson_upper':upper,
        'bin_supported':np.ones(len(counts),bool),'object_selected':support,'native_selection':selected}, {
        'selected_count':int(support.sum()),'in_bin_count':int(counts.sum()),
        'below_range_count':int(np.sum(support&(values<edges[0]))),
        'above_range_count':int(np.sum(support&(values>edges[-1]))),
        'interval_definition':'68.27% central Garwood Poisson; zero-count upper limits retained',
        'uncertainty_scope':'counting intervals only; excludes cosmic variance, completeness and numerical convergence'}


def _pin(binding):
    path=Path(binding['path']).resolve()
    if file_sha256(path)!=binding['sha256']:raise ValueError('Native abundance input changed: '+str(path))
    return path


def _uniform_parent(run,cfg,contract,memory_gib):
    """Bind the normalization volume to actual periodic uniform parent ICs."""
    import h5py
    compile_options=contract['compile_options']
    if compile_options.get('status')=='UNAVAILABLE':raise ValueError('Periodic build evidence is required for abundance volume')
    flags={line.split('=',1)[0].strip() for line in Path(compile_options['path']).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')}
    if 'PERIODIC' not in flags:raise ValueError('Whole-box abundance requires an actual periodic build')
    records=contract['initial_conditions']
    if not records or any(r.get('status')=='UNAVAILABLE' for r in records):raise ValueError('Actual parent IC lineage required for abundance volume')
    paths=[Path(r['path']).resolve() for r in records];census=initial_condition_census(paths)
    counts=census['particles_by_type'];side=cfg['simulation']['particles_per_dimension']
    if type(side) is not int or side<1 or counts[1]!=side**3 or counts[2] or counts[3]:
        raise ValueError('Global abundance requires a complete uniform parent; zoom/coarse populations cannot use whole-box volume')
    mass=None;variable=False;mass_min=np.inf;mass_max=0.
    step=max(1,min(100000,int(memory_gib*1024**3/128)))
    for path in paths:
        with h5py.File(path) as source:
            group=source.get('PartType1')
            if group is None:continue
            table=float(source['Header'].attrs['MassTable'][1])
            if 'Masses' not in group:
                if not np.isfinite(table) or table<=0:raise ValueError('Positive uniform native DM mass required')
                if mass is not None and table!=mass:variable=True
                mass_min=min(mass_min,table);mass_max=max(mass_max,table)
                mass=table;continue
            for start in range(0,len(group['Masses']),step):
                values=group['Masses'][start:start+step]
                if not len(values):continue
                if mass is None:mass=float(values[0])
                if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Positive finite native DM masses required')
                mass_min=min(mass_min,float(values.min()));mass_max=max(mass_max,float(values.max()))
                variable|=not np.all(values==mass)
    h=float(cfg['cosmology']['H0'])/100.;box=float(cfg['simulation']['box_mpc_h'])
    if not np.isfinite([h,box]).all() or min(h,box)<=0:raise ValueError('Positive native box and h required')
    lattice=None
    if variable:
        # monofonIC can carry physical species-density perturbations in its
        # particle masses. Verify its original unigrid lattice rather than
        # equating every varying mass with a multiresolution zoom.
        from .evolution import _monofonic_lagrangian_statistics
        if len(paths)!=1 or min(side**3,1000000)*256+8*1024**2>memory_gib*1024**3:
            raise MemoryError('Variable-mass parent lattice verification exceeds supported allocation/layout')
        lattice=_monofonic_lagrangian_statistics(paths,cfg,box*1000.)
        if lattice.get('status')!='MEASURED':raise ValueError('Variable DM mass requires verified monofonIC simple-cubic full-parent layout')
        with h5py.File(paths[0]) as source:
            ids=source['PartType1/ParticleIDs']
            for start in range(0,len(ids),step):
                stop=min(start+step,len(ids))
                if not np.array_equal(ids[start:stop],np.arange(start,stop,dtype=np.uint64)):
                    raise ValueError('Variable-mass parent lacks complete exact original SC lattice identities')
    return (box/h)**3,{'definition':'complete uniform periodic parent volume (box_mpc_h/h)^3; not a zoom aperture',
        'h':h,'box_mpc_h':box,'census':census,'native_dm_particle_mass':None if variable else mass,
        'native_dm_mass_range':([mass_min,mass_max] if variable else [mass,mass]),
        'variable_mass_parent_lattice':lattice,'mass_normalization_applied':False}


def abundance_input_bindings(request,*,memory_gib):
    import h5py
    from .halos import verify_subhalo_export,verify_galaxy_catalogue
    from .analysis import load_cfg
    run=Path(request['run_directory']).resolve();cfg=load_cfg(run)
    report_path=_pin(request['subhalo_report']);root=report_path.parent
    if report_path.name!='report.json':raise ValueError('Sealed native subhalo report required')
    report=verify_subhalo_export(root);verify_product_provenance(run,report['data_contract'])
    galaxy=_pin(request['galaxy_catalogue']);verify_galaxy_catalogue(galaxy,run)
    with h5py.File(galaxy) as f:galaxy_contract=json.loads(f.attrs['data_contract_json'])
    volume,normalization=_uniform_parent(run,cfg,report['data_contract'],memory_gib)
    sources={str(run/name):file_sha256(run/name) for name in ('config.yaml','manifest.json')}
    sources.update({str(root/name):file_sha256(root/name) for name in ('report.json','report.sha256','columns.npz')})
    sources[str(galaxy)]=file_sha256(galaxy)
    for contract in (report['data_contract'],galaxy_contract):
        records=(contract['input_hashes']+contract['initial_conditions']+list(contract['boltzmann_inputs'].values())
            +[contract['executable'],contract['parameter_file'],contract['compile_options']]+contract['algorithm_sources'])
        for record in records:
            if record.get('status')!='UNAVAILABLE':sources[record['path']]=record['sha256']
    return report,galaxy,volume,normalization,sources


def export_native_abundance(request,destination,*,memory_gib=1.):
    import h5py
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive abundance workspace required')
    if request.get('schema')!='flagship-native-abundance-v1':raise ValueError('Complete native abundance request required')
    for key in ('minimum_dm_particles','minimum_star_particles'):
        if type(request[key]) is not int or request[key]<1:raise ValueError('Positive exact native selection counts required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native abundance products')
    root=Path(request['subhalo_report']['path']).resolve().parent
    with zipfile.ZipFile(root/'columns.npz') as z:
        if sum(i.file_size for i in z.infolist())*4>memory_gib*1024**3/2:raise MemoryError('Native subhalo catalogue exceeds abundance allocation')
    report,galaxy,volume,normalization,sources=abundance_input_bindings(request,memory_gib=memory_gib)
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    with np.load(root/'columns.npz',allow_pickle=False) as f:columns={k:f[k] for k in f.files}
    specs=report['array_contract']['datasets'];epoch=report['epoch_contract'];h=normalization['h']
    if not np.isclose(report['h'],h,rtol=1e-12,atol=0):raise ValueError('Native catalogue h differs from run volume')
    for key,unit in (('subhalo_row','identity'),('vmax_km_s','km/s'),('n_dm','particle count')):
        if specs[key]['units']!=unit:raise ValueError('Native subhalo abundance units differ: '+key)
    ids=exact_uint64_ids(columns['subhalo_row']);vmax=columns['vmax_km_s'];dm=columns['n_dm']
    if vmax.shape!=ids.shape or dm.shape!=ids.shape or dm.dtype.kind not in 'iu':raise ValueError('Exact native subhalo values/counts must align')
    available=columns.get('vmax_km_s_available',np.ones(len(ids),bool))
    if available.dtype.kind!='b' or available.shape!=ids.shape:raise ValueError('Exact native Vmax availability mask required')
    vmax_selection=available&(dm>=request['minimum_dm_particles'])
    with h5py.File(galaxy) as f:
        names=('GalaxyID','StellarMass','Resolved','NStars','NDM')
        if any(name not in f for name in names):raise ValueError('Complete native galaxy abundance columns required')
        if sum(f[name].size*f[name].dtype.itemsize for name in names)*16>memory_gib*1024**3/2:raise MemoryError('Native galaxy columns exceed abundance allocation')
        expected_units=('identity','Msun/h','boolean selection mask','particle count','particle count')
        for name,unit in zip(names,expected_units):
            verify_dataset_semantics(f[name])
            if f[name].attrs['units']!=unit:raise ValueError('Native galaxy abundance units differ: '+name)
            if json.loads(f[name].attrs['epoch_json'])!=epoch:raise ValueError('Native galaxy and subhalo epochs differ')
        gid=exact_uint64_ids(f['GalaxyID'][:]);stellar=f['StellarMass'][:]/h
        resolved=f['Resolved'][:];stars=f['NStars'][:];gdm=f['NDM'][:]
        if any(a.shape!=gid.shape for a in (stellar,resolved,stars,gdm)) or resolved.dtype.kind!='b' or stars.dtype.kind not in 'iu' or gdm.dtype.kind not in 'iu':
            raise ValueError('Aligned native galaxy masses and exact measured count/quality masks required')
        if not np.isclose(float(f.attrs['box_size_mpc_h']),normalization['box_mpc_h'],rtol=1e-12,atol=0):raise ValueError('Galaxy box differs from abundance normalization')
        galaxy_namespace=f['GalaxyID'].attrs['identity_namespace']
    namespace=specs['subhalo_row']['identity_namespace']
    exact_id_join(gid,ids,requested_namespace=galaxy_namespace,available_namespace=namespace)
    stellar_selection=resolved&(stars>=request['minimum_star_particles'])&(gdm>=request['minimum_dm_particles'])
    out.mkdir(parents=True);products={};figures=[]
    for variant,value,selected,object_ids,edges,unit in (
        ('stellar_mass_function',stellar,stellar_selection,gid,request['stellar_mass_edges_msun'],'Msun'),
        ('vmax_function',vmax,vmax_selection,ids,request['vmax_edges_km_s'],'km/s')):
        arrays,statistics=logarithmic_abundance(value,selected,edges,volume)
        arrays['object_ids']=object_ids;arrays['object_value']=value
        units={k:unit if k in ('bin_center','bin_edges','object_value') else 'identity' if k=='object_ids' else
            'count / comoving Mpc^3 / dex' if k in ('number_density','poisson_lower','poisson_upper') else 'dimensionless' for k in arrays}
        target=out/(variant+'.json');products[variant]=str(target)
        definitions={k:{'units':u,'quantity':k,'missing_policy':'native unsupported objects excluded by explicit mask; zero-count bins retained'} for k,u in units.items()}
        definitions['object_ids']['identity_namespace']=namespace
        export_numeric_audit(target,{'schema':'flagship-native-abundance-product-v1','variant':variant,'epoch':epoch,
            'request':request,'normalization':normalization,'counting_statistics':statistics,
            'population':'native luminous SUBFIND galaxies satisfying original Resolved mask and declared counts' if variant=='stellar_mass_function' else 'all native SUBFIND rows including centrals, with Vmax availability and declared DM count cut',
            'mass_definition':'native bound SUBFIND stellar mass / h; not aperture stellar mass' if variant=='stellar_mass_function' else 'native SUBFIND maximum circular speed',
            'selection_scope':'catalogue count selection only; no physical completeness claim','science_acceptance':'NOT_ACCEPTED'},
            arrays,definitions,algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
        figures.append({'figure_id':'native-'+variant,'family':'abundance','variant':variant,'title':variant.replace('_',' '),
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'abundance','title':'Native counts and Garwood interval endpoints',
                'quality_status':'PENDING_COMPLETENESS_AND_SAMPLE_VARIANCE','selection':json.dumps(request,sort_keys=True),
                'xlabel':variant+' ['+unit+']','ylabel':'Number density [count / comoving Mpc^3 / dex]',
                'x_units':unit,'y_units':'count / comoving Mpc^3 / dex','series':[
                    {'source':variant,'kind':'line','x':'bin_center','y':key,'support':'bin_supported','label':label}
                    for key,label in (('number_density','selected counts'),('poisson_lower','Garwood lower'),('poisson_upper','Garwood upper'))]}]})
    for source,expected in sources.items():
        if file_sha256(source)!=expected:raise ValueError('Native abundance input changed during export')
    atomic_json(out/'products.json',{'schema':'flagship-native-abundance-index-v1','products':products})
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


def _provenance_sources(contract):
    rows=(contract['input_hashes']+contract['initial_conditions']+list(contract['boltzmann_inputs'].values())
        +[contract['executable'],contract['parameter_file'],contract['compile_options']]+contract['algorithm_sources'])
    return {r['path']:r['sha256'] for r in rows if r.get('status')!='UNAVAILABLE'}


def export_native_host_abundance(request,destination,*,memory_gib=1.):
    """Reuse native host selection and rare-halo counting, including empty sets."""
    import h5py
    from .analysis import (load_cfg,group_files,halo_masses,halo_particle_counts,apply_hmf_particle_cut,
        hmf_resolution_selection_report,rare_halo_table,read_snapshot_header,poisson_rate_interval,
        _begin_analysis_contract,_verify_analysis_inputs,hmf_mass_edges,mass_to_msun_h,group_field,units_to_mpc_h)
    from .data import run_epoch_contract
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive host abundance workspace required')
    if request.get('schema')!='flagship-native-host-abundance-v1':raise ValueError('Native host abundance request required')
    snap=request['snapshot'];minimum_z=request['rare_minimum_redshift']
    if type(snap) is not int or snap<0 or isinstance(minimum_z,bool) or not np.isfinite(minimum_z) or minimum_z<0:
        raise ValueError('Exact saved snapshot and finite nonnegative high-redshift boundary required')
    run=Path(request['run_directory']).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native host abundance')
    cfg=load_cfg(run);paths=group_files(run,snap)
    if not paths:raise FileNotFoundError('No native group catalogue for requested snapshot')
    count=0
    for path in paths:
        with h5py.File(path) as f:count+=int(f['Header'].attrs['Ngroups_ThisFile'])
    if count*1024>memory_gib*1024**3/2:raise MemoryError('Complete native host census exceeds allocation')
    inputs,contract=_begin_analysis_contract(run,['hmf'])
    volume,normalization=_uniform_parent(run,cfg,contract,memory_gib)
    raw=halo_masses(run,snap);counts=halo_particle_counts(run,snap)
    empty=count==0
    if empty:
        # GADGET omits the Group dataset entirely for a verified zero-group
        # publication. This is an empty measured census, not a missing
        # populated mass column and not an invented halo measurement.
        raw=np.empty(0,float);counts=np.empty(0,np.int64)
    if raw is None:raise ValueError('Configured native halo mass definition is unavailable')
    minimum=cfg['analysis']['hmf_min_particles']
    selected,mask,cut=apply_hmf_particle_cut(raw,counts,minimum)
    if selected is None:raise ValueError('Native halo particle-count selection unavailable: '+cut['reason'])
    if len(raw)!=count:raise ValueError('Native mass census does not close declared group counts')
    mass=mass_to_msun_h(raw)/normalization['h'];valid=mask&np.isfinite(mass)&(mass>0)
    selection=hmf_resolution_selection_report(raw,counts,mask,minimum)
    if counts is not None and (counts.dtype.kind not in 'iu' or counts.shape!=mass.shape or np.any(counts<0)):
        raise ValueError('Native host particle counts must be exact aligned nonnegative integers')
    header,_=read_snapshot_header(run,snap);a=float(header['Time']);epoch=run_epoch_contract(run,snap,a);z=1/a-1
    rare=rare_halo_table(cfg,selected);thresholds=cfg['analysis']['rare_halo_thresholds_msun_h']
    native_threshold=np.asarray(thresholds,float)
    if native_threshold.ndim!=1 or not len(native_threshold) or not np.isfinite(native_threshold).all() or np.any(native_threshold<=0) or len(np.unique(native_threshold))!=len(native_threshold):
        raise ValueError('Distinct positive source-configured rare-halo thresholds required')
    order=np.argsort(native_threshold);threshold=native_threshold[order]/normalization['h']
    rare_counts=np.array([rare[str(thresholds[i])] for i in order],np.int64)
    lower,upper=poisson_rate_interval(rare_counts,volume,np.ones(len(threshold)))
    source=_provenance_sources(contract);source.update({str(run/n):file_sha256(run/n) for n in ('config.yaml','manifest.json')})
    source[str(Path(__file__).resolve())]=file_sha256(__file__)
    positions=group_field(run,snap,['GroupPos'])
    if empty:positions=np.empty((0,3),float)
    if positions is not None and positions.shape!=(len(mass),3):raise ValueError('Native host positions do not align with host mass census')
    arrays={'host_mass':mass,'host_ids':np.arange(len(mass),dtype=np.uint64),'host_selected':valid,
        'host_particle_count':np.full(len(mass),np.nan) if counts is None else counts,
        'host_count_available':np.full(len(mass),counts is not None,bool),
        'host_position_comoving_mpc':np.full((len(mass),3),np.nan) if positions is None else units_to_mpc_h(positions,normalization['box_mpc_h'])/normalization['h'],
        'selection_cut_mass_axis':hmf_mass_edges(cfg)[[0,-1]]/normalization['h'],
        'declared_particle_cut':np.full(2,minimum,dtype=np.int64),
        'threshold_mass':threshold,'rare_counts':rare_counts,'rare_number_density':rare_counts/volume,
        'rare_lower':lower,'rare_upper':upper,'rare_supported':np.full(len(threshold),z>=minimum_z,bool)}
    units={k:'comoving Mpc' if k=='host_position_comoving_mpc' else 'Msun' if k in ('host_mass','selection_cut_mass_axis','threshold_mass') else
        'count / comoving Mpc^3' if k in ('rare_number_density','rare_lower','rare_upper') else
        'particle count' if k in ('host_particle_count','declared_particle_cut') else 'identity' if k=='host_ids' else 'dimensionless' for k in arrays}
    definitions={k:{'units':u,'quantity':k,'missing_policy':'actual empty host census retained; undeclared counts/high-redshift scope explicitly masked'} for k,u in units.items()}
    definitions['host_ids']['identity_namespace']=contract['identity_namespaces']['halo']
    definitions['declared_particle_cut']['quantity']='configured catalogue GroupLen cut; not a measured convergence limit'
    _verify_analysis_inputs(run,inputs,contract)
    out.mkdir(parents=True);product=out/'host-abundance.json'
    export_numeric_audit(product,{'schema':'flagship-native-host-abundance-product-v1','epoch':epoch,
        'request':request,'native_run':str(run),'data_contract':contract,'normalization':normalization,
        'mass_definition':cfg['analysis']['hmf_mass_definition'],'selection':selection,'particle_cut':cut,
        'native_catalogue_status':'MEASURED_EMPTY_COMPLETE_CENSUS' if empty else 'MEASURED_POPULATED_CENSUS',
        'rare_native_table':rare,'actual_redshift':z,'rare_high_redshift_status':'IN_SCOPE' if z>=minimum_z else 'OUTSIDE_DECLARED_HIGH_REDSHIFT_SCOPE',
        'uncertainty_scope':'Garwood counting intervals only; nested cumulative thresholds are correlated, not independent; sample variance and completeness unmeasured',
        'science_acceptance':'NOT_ACCEPTED'},arrays,definitions,algorithm=__file__,context={'epoch':epoch},source_hashes=source)
    figures=[]
    for variant,title,xunit,yunit,series in (
        ('reliability_limits','Host catalogue selection','Msun','particle count',[
            {'x':'host_mass','y':'host_particle_count','support':'host_count_available','label':'native GroupLen','kind':'scatter'},
            {'x':'selection_cut_mass_axis','y':'declared_particle_cut','label':'configured particle cut','kind':'line'}]),
        ('rare_high_z_abundance','Rare host abundance','Msun','count / comoving Mpc^3',[
            {'x':'threshold_mass','y':key,'support':'rare_supported','label':label,'kind':'line'}
            for key,label in (('rare_number_density','selected cumulative counts'),('rare_lower','Garwood lower'),('rare_upper','Garwood upper'))])):
        figures.append({'figure_id':'native-'+variant,'family':'abundance','variant':variant,'title':title,
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'native','title':title,
                'quality_status':'SELECTION_ONLY' if variant=='reliability_limits' else 'COUNTING_ONLY','selection':json.dumps(selection,sort_keys=True),
                'xlabel':'Native '+cfg['analysis']['hmf_mass_definition']+' mass [Msun]',
                'ylabel':('Group particles' if variant=='reliability_limits' else 'n(>M)')+' ['+yunit+']',
                'x_units':xunit,'y_units':yunit,'series':[{'source':'hosts',**s} for s in series]}]})
    for path,checksum in source.items():
        if file_sha256(path)!=checksum:raise ValueError('Native host abundance source changed during export')
    return export_scientific_suite({'sources':{'hosts':str(product)},'figures':figures,'dpi':160,
        'maximum_panels_per_figure':1,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


def matched_host_input_bindings(request,*,memory_gib):
    from core.stream_common import read_product
    from .data import numeric_audit_source_bindings
    from .analysis import load_cfg
    products=[];sources={}
    for name in ('left','right'):
        path=_pin(request[name]);report,arrays,pins=read_product(path,memory_gib=memory_gib/4)
        if report.get('schema')!='flagship-native-host-abundance-product-v1':raise ValueError('Native source-bound host census required for HMF comparison')
        sources.update(pins);sources.update(numeric_audit_source_bindings(report))
        run=Path(report['native_run']);verify_product_provenance(run,report['data_contract'])
        cfg=load_cfg(run);volume,normalization=_uniform_parent(run,cfg,report['data_contract'],memory_gib/2)
        if report['normalization']!=normalization:raise ValueError('Host census normalization changed')
        products.append((report,arrays,cfg,volume))
    return products,sources


def export_matched_host_abundance(request,destination,*,memory_gib=1.):
    from .analysis import _common_hmf_bin_comparison,compare_initial_ic_phases,poisson_rate_ratio_interval
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive matched HMF workspace required')
    if request.get('schema')!='flagship-matched-host-abundance-v1':raise ValueError('Explicit matched host abundance request required')
    bins=request['bins'];interval=np.asarray(request['mass_range_msun'],float)
    if type(bins) is not int or not 2<=bins<=4096 or interval.shape!=(2,) or not np.isfinite(interval).all() or not 0<interval[0]<interval[1]:
        raise ValueError('Declared bounded shared HMF bins and positive physical mass interval required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous matched HMF evidence')
    loaded,sources=matched_host_input_bindings(request,memory_gib=memory_gib)
    (ra,a,ca,va),(rb,b,cb,vb)=loaded
    if (ra['epoch']['actual_scale_factor']!=rb['epoch']['actual_scale_factor'] or ra['mass_definition']!=rb['mass_definition']
            or ra['particle_cut']['minimum_particles']!=rb['particle_cut']['minimum_particles']):
        raise ValueError('Matched HMF requires identical actual epochs, mass definition and native particle cut')
    h=ra['normalization']['h']
    if h!=rb['normalization']['h'] or va!=vb:raise ValueError('This paired native HMF route requires equal h and periodic physical volumes')
    mesh=min(*(int(c['simulation']['particles_per_dimension']) for c in (ca,cb)),
        *(int(c.get('validation',{}).get('ic_phase',{}).get('max_mesh_size',128)) for c in (ca,cb)))
    if mesh**3*512+bins*bins*128>memory_gib*1024**3/2:raise MemoryError('Native phase/jackknife workspace exceeds matched HMF allocation')
    phase=compare_initial_ic_phases(ra['native_run'],rb['native_run'],ca,cb)
    if phase['status']!='PASS':raise ValueError('Actual native IC phases did not pass matched HMF comparison')
    groups=[]
    for report,values,_,_ in loaded:
        specs=report['numeric_data']['array_contract']['datasets']
        for key,unit in (('host_mass','Msun'),('host_position_comoving_mpc','comoving Mpc'),('host_selected','dimensionless')):
            if specs[key]['units']!=unit:raise ValueError('Matched HMF source units differ: '+key)
        mass=values['host_mass'];mask=values['host_selected'];positions=values['host_position_comoving_mpc']
        if mask.dtype.kind!='b' or mask.shape!=mass.shape or mass.ndim!=1 or positions.shape!=(len(mass),3):raise ValueError('Aligned native host masses, exact selection and positions required')
        groups.append({'host_halo_mass_unfiltered_msun_h':mass*h,'host_halo_in_hmf':mask,
            'host_halo_positions_all_mpc_h':positions*h})
    native=_common_hmf_bin_comparison(*groups,va*h**3,vb*h**3,bins,mass_range=(interval*h).tolist(),phase_agreement=phase)
    if native is None or native['status']!='COMPLETE':
        edges=np.geomspace(*interval,bins+1);counts=[np.histogram(values['host_mass'][values['host_selected']],bins=edges)[0] for _,values,_,_ in loaded]
        lower,upper=poisson_rate_ratio_interval(counts[0],counts[1],va*np.diff(np.log10(edges)),vb*np.diff(np.log10(edges)))
        native={'status':'NO_SUPPORTED_NATIVE_RATIO','reason':None if native is None else native.get('reason'),
            'paired_jackknife_status':'UNAVAILABLE_EMPTY_SELECTED_CENSUS','mass_centers_msun_h':np.sqrt(edges[:-1]*edges[1:])*h,
            'counts_a':counts[0],'counts_b':counts[1],'ratio_b_over_a':np.full(bins,np.nan),
            'ratio_poisson_68_lower':lower,'ratio_poisson_68_upper':upper,
            'ratio_paired_jackknife_68_lower':np.full(bins,np.nan),'ratio_paired_jackknife_68_upper':np.full(bins,np.nan),
            'log_ratio_paired_jackknife_covariance':np.full((bins,bins),np.nan)}
    arrays={k:np.asarray(v) for k,v in native.items() if isinstance(v,(np.ndarray,list))}
    arrays['mass']=arrays.pop('mass_centers_msun_h')/h
    if 'mass_edges_msun_h' in arrays:arrays['mass_edges']=arrays.pop('mass_edges_msun_h')/h
    for key in ('dn_dlogM_a','dn_dlogM_b'):
        if key in arrays:arrays[key]*=h**3
    arrays['supported']=(arrays['counts_a']>0)&(arrays['counts_b']>0)&np.isfinite(arrays['ratio_b_over_a'])
    arrays['paired_supported']=arrays['supported']&np.isfinite(arrays['ratio_paired_jackknife_68_lower'])&np.isfinite(arrays['ratio_paired_jackknife_68_upper'])
    units={k:'Msun' if k in ('mass','mass_edges') else 'count / comoving Mpc^3 / dex' if k in ('dn_dlogM_a','dn_dlogM_b') else 'dimensionless' for k in arrays}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Matched HMF source changed during estimation')
    out.mkdir(parents=True);product=out/'matched-hmf.json'
    export_numeric_audit(product,{'schema':'flagship-matched-hmf-v1','epoch':ra['epoch'],'request':request,
        'phase_measurement':phase,'estimator_report':{k:v for k,v in native.items() if not isinstance(v,(np.ndarray,list))},
        'mass_definition':ra['mass_definition'],'uncertainty_scope':'existing conditional independent-Poisson intervals and paired spatial jackknife retained separately; not additive or full cosmic covariance',
        'science_acceptance':'NOT_ACCEPTED'},arrays,
        {k:{'units':u,'quantity':k,'missing_policy':'undefined ratios/unavailable covariance retained as NaN with explicit support'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':ra['epoch']},source_hashes=sources)
    return export_scientific_suite({'sources':{'paired':str(product)},'dpi':160,'maximum_panels_per_figure':1,'organized_outputs':True,
        'figures':[{'figure_id':'native-hmf-ratio','family':'abundance','variant':'hmf_ratio','title':'Matched native host abundance ratio',
            'science_status':'NOT_ACCEPTED','epoch':ra['epoch'],'panels':[{'panel_id':'ratio','title':'Shared native mass bins',
                'quality_status':'PENDING_COMPLETENESS_AND_CONVERGENCE','selection':ra['mass_definition']+'; identical particle cuts and actual epoch',
                'xlabel':'Mass [Msun]','ylabel':'B/A [dimensionless]','x_units':'Msun','y_units':'dimensionless','series':[
                    {'source':'paired','kind':'line','x':'mass','y':key,'support':'paired_supported' if 'paired' in key else 'supported','label':label}
                    for key,label in (('ratio_b_over_a','B/A'),('ratio_poisson_68_lower','conditional Poisson lower'),('ratio_poisson_68_upper','conditional Poisson upper'),
                        ('ratio_paired_jackknife_68_lower','paired jackknife lower'),('ratio_paired_jackknife_68_upper','paired jackknife upper'))]}]}]},out/'gallery',base=out,memory_gib=memory_gib)
