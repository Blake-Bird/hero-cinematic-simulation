"""Matched BH-branch density/shape figures from existing native structure products.

Exact membership counterparts and actual IC phase comparison are mandatory.
Supplied joint measurement covariance binds both original products; unavailable
native bins are retained and cannot become a claimed measured response.
"""
from pathlib import Path
import json
import numpy as np
from .foundation import file_sha256,canonical_digest
from .data import export_numeric_audit,numeric_audit_source_bindings,exact_uint64_ids
from core.stream_common import read_product
from core.research.display import export_scientific_suite


def response_pair_inputs(request,*,memory_gib):
    from .halos import verify_halo_structure,match_by_exact_membership_and_phase_space
    from .analysis import load_cfg,compare_initial_ic_phases,_initial_condition_paths
    sources={};loaded=[]
    for name in ('left','right','matching'):
        binding=request[name];path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('BH response input binding changed: '+name)
        if path.stat().st_size*16>memory_gib*1024**3/4:raise MemoryError('BH response metadata exceeds allocation')
        report,arrays,pins=read_product(path,memory_gib=memory_gib/4);sources.update(pins);sources.update(numeric_audit_source_bindings(report))
        if name in ('left','right') and report!=verify_halo_structure(path):raise ValueError('Native response structure changed')
        loaded.append((report,arrays))
    (a,aa),(b,bb),(match,_)=loaded
    if match.get('schema')!='finder-comparison-v1':raise ValueError('Exact native membership comparison required')
    left,right=match['catalogues']
    replay=match_by_exact_membership_and_phase_space(left,right,match['request'])
    if replay['candidates']!=match['candidates']:raise ValueError('Native counterpart matching does not replay')
    pair=[p for p in match['candidates'] if p['left_id']==a['request']['host_row'] and p['right_id']==b['request']['host_row']]
    if len(pair)!=1 or pair[0]['status']!='RECIPROCAL_UNIQUE_MATCH':raise ValueError('BH response requires reciprocal unique exact-membership counterparts')
    for structure,catalogue in ((a,left),(b,right)):
        if structure['source_context']['epoch']['actual_scale_factor']!=catalogue['epoch']['actual_scale_factor']:raise ValueError('Response counterpart epoch differs')
        objects=[obj for obj in catalogue['objects'] if obj['id']==structure['request']['host_row']]
        context=structure['source_context']
        center=np.asarray(context['center_comoving_kpc_h'])*context['scale_factor']/context['h']
        if len(objects)!=1 or not np.allclose(objects[0]['position_proper_kpc'],center,rtol=1e-12,atol=1e-12):
            raise ValueError('Response structure centre does not bind selected native counterpart')
        for path,checksum in structure['source_context']['source_hashes'].items():
            if Path(path).suffix=='.hdf5' and catalogue['source_hashes'].get(path)!=checksum:
                raise ValueError('Counterpart matching does not bind original structure snapshot/catalogue sources')
    if a['source_context']['epoch']['actual_scale_factor']!=b['source_context']['epoch']['actual_scale_factor']:raise ValueError('BH response requires identical actual epochs')
    if a['request']['radial_edges_proper_kpc']!=b['request']['radial_edges_proper_kpc']:raise ValueError('BH response requires common original proper shell edges')
    component=request['component']
    if (a['request']['components'][component]!=b['request']['components'][component]
            or a['request']['components'][component]['species']!=[1]
            or a['request']['shape_controls']!=b['request']['shape_controls']):raise ValueError('DM response selections/shape estimators differ or include another species')
    configs=[load_cfg(s['request']['run_directory']) for s in (a,b)]
    for key in ('cosmology','simulation','hydro','stellar_evolution','chemistry','sidm'):
        if configs[0].get(key)!=configs[1].get(key):raise ValueError('BH-only response changes another configuration branch: '+key)
    enabled=[bool(c.get('black_holes',{}).get('enabled',False)) for c in configs]
    if enabled.count(True)!=1:raise ValueError('Matched BH response needs one enabled and one disabled BH branch')
    mesh=min(*(int(c['simulation']['particles_per_dimension']) for c in configs),
        *(int(c.get('validation',{}).get('ic_phase',{}).get('max_mesh_size',128)) for c in configs))
    if mesh**3*512>memory_gib*1024**3/2:raise MemoryError('BH native phase comparison exceeds allocation')
    for structure in (a,b):
        run=Path(structure['request']['run_directory']).resolve()
        for path in [run/'config.yaml',run/'manifest.json',*_initial_condition_paths(run)]:sources[str(path.resolve())]=file_sha256(path)
    phase=compare_initial_ic_phases(a['request']['run_directory'],b['request']['run_directory'],*configs)
    if phase['status']!='PASS':raise ValueError('Actual native phase comparison failed BH response intake')
    return a,aa,b,bb,pair[0],phase,sources


def response_inputs(request,*,memory_gib):
    a,aa,b,bb,pair,phase,sources=response_pair_inputs(request,memory_gib=memory_gib)
    binding=request['joint_covariance'];path=Path(binding['path']).resolve()
    if file_sha256(path)!=binding['sha256']:raise ValueError('BH response covariance input binding changed')
    cov,ca,pins=read_product(path,memory_gib=memory_gib/4)
    sources.update(pins);sources.update(numeric_audit_source_bindings(cov))
    if cov.get('schema')!='native-bh-response-joint-covariance-v1' or cov.get('left_sha256')!=request['left']['sha256'] or cov.get('right_sha256')!=request['right']['sha256']:
        raise ValueError('Measured joint covariance must bind the exact ordered native structures')
    return a,aa,b,bb,cov,ca,pair,phase,sources


def structure_response_measurements(a,aa,b,bb,component,cov,ca):
    """Use existing common-observable difference/covariance propagation."""
    from .matched import compare_common_observables
    # Logical module name is supplied by the consolidated source layout.
    radius=np.asarray(aa['profile_'+component+'_radius_proper_kpc']);n=len(radius)
    if radius.shape!=(n,) or not np.array_equal(radius,bb['profile_'+component+'_radius_proper_kpc']):raise ValueError('Native response radius arrays differ')
    datasets=cov['numeric_data']['array_contract']['datasets'];definitions={};matching={};pair={'left_row':0,'right_row':0,'left_identity':'source-bound-left-host','right_identity':'source-bound-right-host','paired_measurement_covariance':{}}
    objects=[{'observables':{}},{'observables':{}}];unresolved={}
    for name,family,unit in (('density','density_shape','Msun/(proper kpc)^3'),('shape','density_shape','dimensionless')):
        values=[];masks=[]
        for record,raw in ((a,aa),(b,bb)):
            if name=='density':
                key='profile_'+component+'_density_msun_kpc3'
                if record['numeric_data']['array_contract']['datasets'][key]['units']!=unit:raise ValueError('Native DM density units differ')
                value=np.asarray(raw[key]);mask=np.asarray(raw['profile_'+component+'_shell_count'])>0
            else:
                shells=record['density_shapes'][component]['shells']
                if len(shells)!=n:raise ValueError('Native DM shape shells differ from original radius bins')
                value=np.array([[s.get(k,np.nan) if s.get(k) is not None else np.nan for k in ('b_over_a','c_over_a')] for s in shells])
                mask=np.array([s['status']=='MEASURED' for s in shells])
            values.append(value.ravel());masks.append(mask&np.isfinite(value).reshape(n,-1).all(axis=1))
        definition={'family':family,'units':unit,'definition_hash':canonical_digest({'component':a['request']['components'][component],
            'edges':a['request']['radial_edges_proper_kpc'],'shape_controls':a['request']['shape_controls'],'observable':name})}
        definitions[name]=definition
        size=len(values[0]);matrix=ca[name+'_joint_covariance']
        expected='('+unit+')^2'
        if datasets[name+'_joint_covariance']['units']!=expected or matrix.shape!=(2*size,2*size):raise ValueError('Measured DM joint covariance units/axes differ')
        if cov.get('measurement_support',{}).get(name) is False:
            unresolved[name]={'status':'NO_SUPPORTED_MEASUREMENT_COVARIANCE','definition':definition,
                'reason':'Original paired estimator bootstrap did not meet its frozen complete-vector support fraction'}
            continue
        if not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T,rtol=1e-12,atol=0):raise ValueError('Finite symmetric joint DM covariance required')
        tolerance=128*np.finfo(float).eps*max(np.max(abs(matrix)),np.finfo(float).tiny)*len(matrix)
        if np.linalg.eigvalsh(matrix).min()<-tolerance:raise ValueError('Joint DM response covariance is not PSD')
        matching[name]=np.zeros((size,size))
        pair['paired_measurement_covariance'][name]=matrix[:size,size:]
        for i in (0,1):objects[i]['observables'][name]={'value':values[i],'supported':masks[i],
            'units':unit,'definition_hash':definition['definition_hash'],'covariance':matrix[:size,:size] if i==0 else matrix[size:,size:]}
    comparison,arrays=compare_common_observables([objects[0]],[objects[1]],{'matches':[pair]},
        {name:value for name,value in definitions.items() if name not in unresolved},
        left_reliable=[True],right_reliable=[True],matching_covariance=matching)
    results=comparison['observables'];results.update(unresolved);units=comparison['units']
    # Single matched host supplies no object-sampling covariance estimate.
    # Preserve this and display only the measured joint measurement error.
    arrays['radius']=radius;units['radius']='proper kpc'
    for name,size in (('density',n),('shape',n*2)):
        measured=results[name]['status']=='MEASURED'
        delta=arrays.get(name+'__mean_delta',np.full(size,np.nan))
        matrix=arrays.get(name+'__measurement_covariance',np.full((size,size),np.nan))
        arrays[name+'_delta']=delta.reshape(n,-1);arrays[name+'_error']=np.sqrt(np.maximum(0,np.diag(matrix))).reshape(n,-1)
        arrays[name+'_supported']=np.full(n,measured,bool)
        units[name+'_delta']=units[name+'_error']=definitions[name]['units'];units[name+'_supported']='dimensionless'
    return results,arrays,units


def export_native_bh_dm_response(request,destination,*,memory_gib=1.):
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Finite positive BH response allocation required')
    if request.get('schema')!='native-bh-matched-dm-response-v1':raise ValueError('Complete native matched DM response request required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous matched BH response')
    a,aa,b,bb,cov,ca,pair,phase,sources=response_inputs(request,memory_gib=memory_gib)
    bins=len(aa['profile_'+request['component']+'_radius_proper_kpc'])
    if (4*bins)**2*128>memory_gib*1024**3/2:raise MemoryError('DM response covariance propagation exceeds allocation')
    results,arrays,units=structure_response_measurements(a,aa,b,bb,request['component'],cov,ca)
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Native DM response source changed during comparison')
    out.mkdir(parents=True);target=out/'matched-dm-response.json';epoch=a['source_context']['epoch']
    export_numeric_audit(target,{'schema':'native-bh-matched-dm-response-product-v1','epoch':epoch,'request':request,
        'counterpart':pair,'phase_measurement':phase,'measurement_results':results,
        'uncertainty_scope':'supplied measured full joint structure covariance; single host has no object-sampling covariance; matching covariance fixed to zero as a conditioning assumption on the chosen exact counterpart, not a measured systematic',
        'causal_scope':'BH-only configured branch contrast with exact membership counterpart and actual phases; physical attribution/convergence unaccepted',
        'science_acceptance':'NOT_ACCEPTED'},arrays,{k:{'units':u,'quantity':k,'missing_policy':'unsupported original bins invalidate the declared complete response vector; unresolved arrays retained'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    figures=[]
    for name,variant,count in (('density','dm_density_response',1),('shape','dm_shape_response',2)):
        unit=units[name+'_delta']
        figures.append({'figure_id':'native-'+variant,'family':'black_holes','variant':variant,'title':'Matched DM '+name+' response B - A',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'response','title':'Exact counterpart; original common radial bins',
                'quality_status':results[name]['status'],'selection':request['component']+'; native original estimator and full covariance',
                'xlabel':'Radius [proper kpc]','ylabel':'B - A ['+unit+']','x_units':'proper kpc','y_units':unit,'series':[
                    {'source':'response','kind':'errorbar','x':'radius','y':name+'_delta','y_column':j,'error':name+'_error','error_column':j,
                        'support':name+'_supported','label':name if count==1 else ('b/a' if j==0 else 'c/a')} for j in range(count)]}]})
    return export_scientific_suite({'sources':{'response':str(target)},'figures':figures,'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


def paired_structure_bootstrap(left, right, *, edges, shape_controls,
                               support_radius, realizations, seed,
                               minimum_success_fraction, memory_gib):
    """Estimate conditional particle-sampling covariance with paired ID draws.

    One multinomial draw on the union of native IDs supplies shared
    multiplicities in A/B. One-sided members remain one-sided. The joint
    covariance gives Var(B-A) = Var(A)+Var(B)-Cov(A,B)-Cov(B,A). This is not
    volume variance, dynamical convergence, or physical-model uncertainty.
    """
    from .halos import radial_density_shapes
    if type(realizations) is not int or realizations<32 or type(seed) is not int or not 0<=seed<2**64:
        raise ValueError('At least32 declared paired bootstrap draws and exact seed required')
    if type(minimum_success_fraction) not in (int,float) or not np.isfinite(minimum_success_fraction) or not 0<minimum_success_fraction<=1:
        raise ValueError('Explicit finite bootstrap success fraction required')
    if type(memory_gib) not in (int,float) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive finite paired bootstrap workspace required')
    bins=np.asarray(edges,float)
    if bins.ndim!=1 or len(bins)<3 or not np.isfinite(bins).all() or bins[0]<0 or np.any(np.diff(bins)<=0):
        raise ValueError('Original increasing proper shell edges required')
    if not np.isfinite(support_radius) or support_radius<bins[-1]:
        raise ValueError('Original complete structure support must cover all shells')
    samples=[]
    for sample in (left,right):
        ids=exact_uint64_ids(sample['particle_ids'])
        x=np.asarray(sample['offsets_proper_kpc'],float)
        mass=np.asarray(sample['masses_msun'],float)
        if (not len(ids) or x.shape!=(len(ids),3) or mass.shape!=ids.shape
                or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass<=0)
                or np.any(np.linalg.norm(x,axis=1)>support_radius)):
            raise ValueError('Complete original selected native particle samples required')
        samples.append((ids,x,mass))
    total=sum(len(s[0]) for s in samples)
    n=len(bins)-1
    if total*512+realizations*n*128+(4*n)**2*64>memory_gib*1024**3:
        raise MemoryError('Paired native estimator bootstrap exceeds declared allocation')
    union=np.union1d(samples[0][0],samples[1][0])
    locations=[np.searchsorted(union,s[0]) for s in samples]
    rng=np.random.default_rng(seed)
    volume=4*np.pi/3*np.diff(bins**3)
    controls=dict(shape_controls,bootstrap_realizations=0)
    values={'density':np.full((realizations,2*n),np.nan),'shape':np.full((realizations,4*n),np.nan)}
    for draw in range(realizations):
        # Resample identities, not array rows. One-sided native members remain
        # in the union and contribute only to their original branch.
        multiplicity=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union))
        for side,(_,x,mass) in enumerate(samples):
            indices=np.repeat(np.arange(len(x)),multiplicity[locations[side]])
            xx=x[indices]
            mm=mass[indices]
            radius=np.linalg.norm(xx,axis=1)
            values['density'][draw,side*n:(side+1)*n]=np.histogram(radius,bins=bins,weights=mm)[0]/volume
            shape=radial_density_shapes(xx,mm,bins,support_radius_kpc=support_radius,**controls)
            ratios=np.array([[row.get(key,np.nan) if row.get(key) is not None else np.nan for key in ('b_over_a','c_over_a')]
                if row['status']=='MEASURED' else [np.nan,np.nan] for row in shape['shells']])
            values['shape'][draw,side*2*n:(side+1)*2*n]=ratios.ravel()
    arrays={}
    support={}
    counts={}
    for name,raw in values.items():
        complete=np.isfinite(raw).all(axis=1)
        count=int(complete.sum())
        counts[name]=count
        support[name]=count>=2 and count/realizations>=minimum_success_fraction
        # Covariance of the estimator's bootstrap draws; never divide again by
        # the number of draws (which controls Monte Carlo precision only).
        arrays[name+'_joint_covariance']=np.cov(raw[complete],rowvar=False) if support[name] else np.full((raw.shape[1],raw.shape[1]),np.nan)
        arrays[name+'_bootstrap_vectors']=raw
        arrays[name+'_bootstrap_complete']=complete
    return arrays,{'measurement_support':support,'complete_bootstrap_count':counts,
        'realizations':realizations,'seed':seed,'minimum_success_fraction':minimum_success_fraction,
        'particle_union_count':len(union),'left_particle_count':len(samples[0][0]),'right_particle_count':len(samples[1][0]),
        'sampling_definition':'Multinomial N draws on exact native particle-ID union; shared multiplicities per identity in A/B; original masses, coordinates and membership retained. Sample covariance of complete estimator draws, not covariance/N.',
        'scope':'Conditional particle-sampling covariance of these saved fixed-centre estimators; not volume covariance, dynamical convergence or causal uncertainty.'}



def covariance_input_bindings(parameters):
    path=Path(parameters['request_path']).resolve()
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not np.isfinite(memory) or memory<=0:
        raise ValueError('Positive finite native BH covariance workspace required')
    if path.stat().st_size*24>memory*1024**3/8:
        raise MemoryError('Native BH covariance request metadata exceeds allocation')
    if file_sha256(path)!=parameters['request_sha256']:
        raise ValueError('Native BH covariance request changed')
    request=json.loads(path.read_text())
    if set(request)!={'schema','pair','bootstrap'} or request['schema']!='native-bh-response-covariance-request-v1':
        raise ValueError('Complete native BH covariance request required')
    if set(request['bootstrap'])!={'realizations','seed','minimum_success_fraction'}:
        raise ValueError('Explicit paired bootstrap count/seed/support controls required')
    loaded=response_pair_inputs(request['pair'],memory_gib=memory/2)
    from .analysis import _initial_condition_paths
    a,_,b,*_=loaded
    initial=[sorted(file_sha256(p) for p in _initial_condition_paths(report['request']['run_directory'])) for report in (a,b)]
    if not initial[0] or initial[0]!=initial[1]:
        raise ValueError('Shared-identity BH bootstrap requires byte-identical original native IC sets')
    return request,loaded,{str(path):file_sha256(path),**loaded[-1]}



def native_bh_covariance_runner(parameters,context):
    if parameters['memory_gib']>context.resources['memory_gib']:
        raise MemoryError('Native BH covariance exceeds workflow allocation')
    request,loaded,pins=covariance_input_bindings(parameters)
    a,aa,b,bb,pair,phase,_=loaded
    component=request['pair']['component']
    samples=[]
    for report,raw in ((a,aa),(b,bb)):
        sample={name:raw['parent_'+component+'_'+name] for name in ('particle_ids','offsets_proper_kpc','masses_msun')}
        specs=report['numeric_data']['array_contract']['datasets']
        for name,unit in (('particle_ids','identity'),('offsets_proper_kpc','proper kpc'),('masses_msun','Msun')):
            if specs['parent_'+component+'_'+name]['units']!=unit:
                raise ValueError('Native paired bootstrap particle units differ')
        samples.append(sample)
    if a['request']['support_radius_proper_kpc']!=b['request']['support_radius_proper_kpc']:
        raise ValueError('Native bootstrap source supports differ')
    arrays,summary=paired_structure_bootstrap(*samples,edges=a['request']['radial_edges_proper_kpc'],
        shape_controls=a['request']['shape_controls'],support_radius=a['request']['support_radius_proper_kpc'],
        **request['bootstrap'],memory_gib=parameters['memory_gib']/2)
    epoch=a['source_context']['epoch']
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    specs={}
    for key in arrays:
        unit='(Msun/(proper kpc)^3)^2' if key=='density_joint_covariance' else '(dimensionless)^2' if key=='shape_joint_covariance' else 'Msun/(proper kpc)^3' if key=='density_bootstrap_vectors' else 'dimensionless'
        specs[key]={'units':unit,'quantity':key,'missing_policy':'Unresolved covariance is NaN, accompanied by measurement_support=false; original failed bootstrap vectors retained'}
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native paired bootstrap source changed')
    return export_numeric_audit(context.generation/'joint-covariance.json',{
        'schema':'native-bh-response-joint-covariance-v1','epoch':epoch,
        'left_sha256':request['pair']['left']['sha256'],'right_sha256':request['pair']['right']['sha256'],
        'request':request,'counterpart':pair,'phase_measurement':phase,**summary,'science_acceptance':'NOT_ACCEPTED'},
        arrays,specs,algorithm=__file__,context={'epoch':epoch},source_hashes=pins)

