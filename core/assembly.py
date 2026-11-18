"""Native halo/galaxy assembly, orbital events, workflow stages and figures.

Histories follow exact native ancestry and retain gaps in saved observations.
Masses use Msun, times Myr, radii proper kpc and velocities km/s; scale factor
and d ln(M)/d ln(a) are dimensionless. Event and transition estimates remain
conditional on the saved cadence, measurement masks and supplied fit errors.
"""
import numpy as np
from .stream_common import finite, increasing, positive, integer, digest
from .stream.trajectories import Worldline, find_closest_approach_bracket


def reconstruct_quality_masked_main_branch(graph,target,request):
    """Follow native progenitors backward, then return the saved time order.

    A missing detection is a gap, not evidence of disruption or survival.
    Main-progenitor transfer masses are in Msun; competing transfers closer
    than the supplied margin leave the branch explicitly ambiguous.
    """
    nodes={tuple(row['key']):row for row in graph['nodes']};key=tuple(target)
    if key not in nodes:raise KeyError('Missing exact target observation')
    incoming={}
    for edge in graph['edges']:
        if edge['kind']!='descendant':continue
        if edge['evidence'].get('origin')!='native_ancestry':raise ValueError('Main branch requires actual native ancestry, not phase-space matching')
        source=tuple(edge['source']);destination=tuple(edge['target'])
        if source[2]>=destination[2]:raise ValueError('Descendant edges must advance saved epoch')
        incoming.setdefault(destination,[]).append(edge)
    rows=[];seen=set();ambiguities=[]
    while key in nodes:
        if key in seen:raise ValueError('Cycle in native assembly branch')
        seen.add(key);node=nodes[key];attributes=node['attributes']
        rows.append({'key':list(key),'status':node['status'],'attributes':attributes})
        parents=incoming.get(key,[])
        if not parents:break
        declared=[e for e in parents if e['evidence'].get('main_progenitor') is True]
        if len(declared)>1:raise ValueError('Multiple native main progenitors for one descendant')
        if declared:chosen=declared[0]
        else:
            contributions=[e['evidence'].get('bound_mass_transfer_msun') for e in parents]
            if any(q is None or not np.isfinite(q) or q<0 for q in contributions):
                ambiguities.append({'target':list(key),'reason':'no_native_main_progenitor_or_resolved_transfer'});break
            order=np.argsort(contributions)[::-1]
            if len(order)>1 and contributions[order[0]]-contributions[order[1]]<=request['main_branch_transfer_margin_msun']:
                ambiguities.append({'target':list(key),'reason':'ambiguous_main_progenitor_transfer'});break
            chosen=parents[order[0]]
        key=tuple(chosen['source'])
        if len(rows)>=request['maximum_branch_epochs']:raise MemoryError('Main branch exceeds declared history length')
    rows.reverse();times=[];scale=[];mass=[];vmax=[];valid=[];stars=[];sfr=[];spin=[];host=[]
    for row in rows:
        attributes=row['attributes'];epoch=str(row['key'][2])
        times.append(request['epoch_time_myr'][epoch]);scale.append(request['epoch_scale_factor'][epoch])
        mass.append(attributes.get('mass_msun',np.nan));vmax.append(attributes.get('vmax_kms',np.nan))
        stars.append(attributes.get('stellar_mass_msun',np.nan));sfr.append(attributes.get('sfr_msun_year',np.nan))
        spin.append(attributes.get('angular_momentum_kpc_kms',[np.nan]*3));host.append(attributes.get('host_identity'))
        valid.append(row['status']=='DETECTED' and attributes.get('measurement_quality_passed',False))
    time=increasing(times);a=increasing(scale,'scale_factor');m=np.asarray(mass,float);quality=np.asarray(valid,bool)&np.isfinite(m)&(m>0)
    interval=quality[:-1]&quality[1:]&(np.diff(time)<=request['maximum_interval_myr'])
    # A missing graph epoch is not interpolated as measured halo survival.
    storage=np.array([r['key'][2] for r in rows]);interval&=np.diff(storage)==1
    report={'schema':'halo-main-branch-v1','target':target,'native_branch':rows,'ambiguities':ambiguities,
        'host_identity_history':host,'branch_identity_history':[row['key'] for row in rows],'source_graph_fingerprint':graph['fingerprint'],'science_acceptance':'NOT_ACCEPTED'}
    mergers=[];lifecycle=[]
    branch_keys={tuple(row['key']) for row in rows}
    for edge in graph['edges']:
        if edge['kind']=='descendant' and tuple(edge['target']) in branch_keys and tuple(edge['source']) not in branch_keys:
            mergers.append({'source':edge['source'],'descendant':edge['target'],'evidence':edge['evidence'],'time_myr':request['epoch_time_myr'][str(edge['target'][2])]})
    for i,row in enumerate(rows):
        if i and row['key'][2]-rows[i-1]['key'][2]>1:lifecycle.append({'kind':'reappearance_after_missing_epochs','bracket_myr':[times[i-1],times[i]],'identity':row['key']})
        evidence=row['attributes'].get('disruption_evidence')
        if evidence and evidence.get('supported') and evidence.get('source_hashes'):lifecycle.append({'kind':'supported_disruption','identity':row['key'],'time_myr':times[i],'evidence':evidence})
    report.update(native_mergers=mergers,lifecycle_events=lifecycle)
    arrays={'times_myr':time,'scale_factor':a,'mass_msun':m,'vmax_kms':np.asarray(vmax,float),
        'stellar_mass_msun':np.asarray(stars,float),'sfr_msun_year':np.asarray(sfr,float),
        'angular_momentum_kpc_kms':np.asarray(spin,float),'history_supported':quality,'interval_supported':interval,'storage_epoch':storage}
    for field,width in [('concentration',1),('spin_dimensionless',1),('density_axis_ratios',2),('potential_axis_ratios',2),('stellar_morphology_tensor_kpc2',9)]:
        shape=(len(rows),) if width==1 else (len(rows),width)
        values=np.asarray([np.asarray(row['attributes'].get(field,np.nan if width==1 else [np.nan]*width)).reshape(-1) if width>1 else row['attributes'].get(field,np.nan) for row in rows],float)
        if values.shape!=shape:raise ValueError('Assembly native attribute shape differs: '+field)
        arrays[field]=values;arrays[field+'_supported']=quality&np.isfinite(values).reshape(len(rows),-1).all(axis=1)
    tensor=arrays['stellar_morphology_tensor_kpc2'].reshape(len(rows),3,3);shape_support=arrays['stellar_morphology_tensor_kpc2_supported'].copy()
    shape_b=np.full(len(rows),np.nan);shape_c=np.full(len(rows),np.nan)
    for i in np.flatnonzero(shape_support):
        if not np.allclose(tensor[i],tensor[i].T):shape_support[i]=False;continue
        eigen=np.linalg.eigvalsh(tensor[i])[::-1]
        if eigen[-1]<0 or eigen[0]<=0:shape_support[i]=False;continue
        shape_b[i]=np.sqrt(eigen[1]/eigen[0]);shape_c[i]=np.sqrt(eigen[2]/eigen[0])
    arrays.update(stellar_morphology_b_over_a=shape_b,stellar_morphology_c_over_a=shape_c,stellar_morphology_supported=shape_support)
    return report,arrays



def measure_assembly(history,request,*,host_identity_history=None):
    """Measure mass growth, half-mass crossings, spin changes and stellar state.

    Growth is the finite-interval d ln(M)/d ln(a), not dM/dt. Half-mass
    times use linear mass interpolation only within supported saved brackets;
    their brackets express cadence, not a statistical timing error. Specific
    SFR is SFR [Msun/year] / stellar mass [Msun], hence units 1/year.
    """
    t=increasing(history['times_myr']);a=increasing(history['scale_factor'],'scale factor');mass=np.asarray(history['mass_msun'],float)
    valid=np.asarray(history['history_supported'],bool);interval=np.asarray(history['interval_supported'],bool)
    good=np.flatnonzero(valid)
    if not len(good):
        # Keep the output vocabulary and shapes when native quality is absent;
        # NaN values and false masks represent unavailable measurements.
        return {'status':'NO_SUPPORTED_ASSEMBLY','science_acceptance':'NOT_ACCEPTED'}, {
            'log_mass_accretion_rate':np.full(len(t)-1,np.nan),
            'spin_flip_angle_rad':np.full(len(t)-1,np.nan),
            'specific_sfr_year_inverse':np.full(len(t),np.nan),
            'quenched_supported':np.zeros(len(t),bool),
            'stripping_candidate':np.zeros(len(t),bool)}
    peak=int(good[np.argmax(mass[good])]);last=int(good[-1]);reference=mass[last] if request['formation_mass_reference']=='final' else mass[peak]
    crossings=[]
    for i in np.flatnonzero(interval):
        if mass[i]<reference/2<=mass[i+1]:
            fraction=(reference/2-mass[i])/(mass[i+1]-mass[i])
            crossings.append({'bracket_myr':[float(t[i]),float(t[i+1])],'time_myr':float(t[i]+fraction*(t[i+1]-t[i])),
                'scale_factor':float(a[i]+fraction*(a[i+1]-a[i])),'definition':'piecewise measured mass crossing; bracket is cadence uncertainty'})
    rate=np.full(len(t)-1,np.nan);indices=np.flatnonzero(interval)
    rate[indices]=np.log(mass[indices+1]/mass[indices])/np.log(a[indices+1]/a[indices])
    spin=np.asarray(history['angular_momentum_kpc_kms'],float);norm=np.linalg.norm(spin,axis=1);spin_valid=valid&np.isfinite(spin).all(axis=1)&(norm>request['minimum_spin_magnitude_kpc_kms'])
    angle=np.full(len(t)-1,np.nan);eligible=interval&spin_valid[:-1]&spin_valid[1:]
    cosine=np.divide(np.sum(spin[:-1]*spin[1:],axis=1),norm[:-1]*norm[1:],out=np.zeros(len(t)-1),where=(norm[:-1]*norm[1:])>0)
    angle[eligible]=np.arccos(np.clip(cosine[eligible],-1,1))
    host_changes=[]
    if host_identity_history is not None:
        for i in np.flatnonzero(interval):
            if host_identity_history[i]!=host_identity_history[i+1]:host_changes.append({'before':host_identity_history[i],'after':host_identity_history[i+1],'bracket_myr':t[i:i+2].tolist()})
    peakv=np.asarray(history['vmax_kms'],float);velocity_support=valid&np.isfinite(peakv)
    stars=np.asarray(history['stellar_mass_msun'],float);sfr=np.asarray(history['sfr_msun_year'],float)
    stellar_crossings=[];stellar_support=valid&np.isfinite(stars)&(stars>=0)
    stellar_good=np.flatnonzero(stellar_support)
    if len(stellar_good) and stars[stellar_good[-1]]>0:
        threshold=stars[stellar_good[-1]]/2
        for i in np.flatnonzero(interval&stellar_support[:-1]&stellar_support[1:]):
            if stars[i]<threshold<=stars[i+1]:stellar_crossings.append({'time_myr':float(t[i]+(threshold-stars[i])/(stars[i+1]-stars[i])*(t[i+1]-t[i])),'bracket_myr':t[i:i+2].tolist()})
    # Zero SFR is a measured quiescent state. Negative or unavailable SFR,
    # invalid stellar mass and failed native quality cannot establish quenching.
    ssfr_supported=stellar_support&(stars>0)&np.isfinite(sfr)&(sfr>=0)
    ssfr=np.divide(sfr,stars,out=np.full_like(stars,np.nan),where=ssfr_supported)
    quenched=valid&np.isfinite(ssfr)&(ssfr<request['quenching_ssfr_year_inverse'])
    stripping=valid&(mass<request['stripping_peak_mass_fraction']*mass[peak])&(np.arange(len(mass))>peak)
    report={'schema':'halo-assembly-events-v1','status':'MEASURED','half_mass_crossings':crossings,
        'formation_time_myr':crossings[0]['time_myr'] if crossings else None,'formation_mass_reference':request['formation_mass_reference'],
        'formation_status':'CADENCE_BRACKETED' if crossings else 'UNRESOLVED_OR_BEFORE_FIRST_OBSERVATION',
        'peak_mass_msun':float(mass[peak]),'peak_mass_time_myr':float(t[peak]),
        'peak_vmax_kms':float(np.max(peakv[velocity_support])) if velocity_support.any() else None,
        'stellar_half_mass_crossings':stellar_crossings,'stellar_assembly_definition':'saved native stellar mass; ex-situ/in-situ decomposition requires birth ancestry','host_changes':host_changes,'spin_flip_intervals':np.flatnonzero(angle>request['spin_flip_angle_rad']).tolist(),
        'disruption_claim':'missing detection alone does not prove physical disruption', 'science_acceptance':'NOT_ACCEPTED'}
    return report,{'log_mass_accretion_rate':rate,'spin_flip_angle_rad':angle,'specific_sfr_year_inverse':ssfr,
        'quenched_supported':quenched,'stripping_candidate':stripping}



def fit_supported_fast_slow_accretion_transition(history,request):
    """Compare conditional fits of ln(M) versus ln(a) on consecutive epochs.

    The hinge basis keeps each broken line continuous. BIC = chi^2 + k ln(N)
    counts fitted coefficients and knot locations; supplied log-mass errors
    define chi^2. A better two-knot fit prevents claiming a single transition.
    """
    a=np.asarray(history['scale_factor']);m=np.asarray(history['mass_msun']);valid=np.asarray(history['history_supported'],bool)
    interval=np.asarray(history['interval_supported'],bool)
    selected=np.flatnonzero(valid)
    if valid.sum()<request['minimum_epochs'] or np.any(np.diff(selected)!=1) or not interval[selected[0]:selected[-1]].all():
        return {'status':'UNRESOLVED_SAMPLING','transition':None}
    x=np.log(a[valid]);y=np.log(m[valid]);sigma=finite(request['log_mass_error'],m.shape)[valid]
    if np.any(sigma<=0):raise ValueError('Positive mass-history uncertainty required')
    def fit(knots):
        design=np.column_stack((np.ones(len(x)),x,*[np.maximum(x-k,0.) for k in knots]))
        coefficients=np.linalg.lstsq(design/sigma[:,None],y/sigma,rcond=None)[0]
        chi2=float(np.sum(((y-design@coefficients)/sigma)**2));bic=chi2+(len(coefficients)+len(knots))*np.log(len(x))
        return bic,coefficients,chi2
    null=fit([]);minimum=integer(request['minimum_segment_epochs'],'minimum_segment_epochs',2)
    knots=x[minimum:len(x)-minimum]
    if len(knots)>request['maximum_candidate_knots']:knots=knots[np.linspace(0,len(knots)-1,request['maximum_candidate_knots']).astype(int)]
    single=[(fit([k]),k) for k in knots]
    if not single:return {'status':'UNRESOLVED_SAMPLING','transition':None}
    best,knot=min(single,key=lambda q:q[0][0]);two=[]
    for i,left in enumerate(knots):
        for right in knots[i+1:]:
            if np.sum((x>=left)&(x<=right))>=minimum:two.append((fit([left,right]),left,right))
    if two and min(two,key=lambda q:q[0][0])[0][0]<best[0]-request['minimum_bic_improvement']:
        return {'status':'MULTIPHASE_HISTORY','transition':None,'best_two_transition_bic':min(q[0][0] for q in two),'single_transition_bic':best[0]}
    before=best[1][1];after=before+best[1][2]
    accepted=null[0]-best[0]>=request['minimum_bic_improvement'] and before-after>=request['minimum_slope_drop'] and best[2]/max(1,len(x)-3)<=request['maximum_reduced_chi2']
    return {'status':'SUPPORTED_FAST_SLOW_MODEL' if accepted else 'NO_SUPPORTED_UNIVERSAL_TRANSITION',
        'transition':{'scale_factor':float(np.exp(knot)),'early_slope':float(before),'late_slope':float(after)} if accepted else None,
        'null_bic':null[0],'broken_line_bic':best[0],'chi2':best[2],'controls':request,
        'interpretation':'conditional piecewise-log-mass fit; threshold and epoch coverage are recorded'}



def orbital_assembly_events(object_record,host_record,request):
    from core.stream.trajectories import polynomial_extrema, orbital_interpolation_contract
    body=Worldline(object_record);host=Worldline(host_record)
    if body.frame!=host.frame:raise ValueError('Host/object orbital histories need common inertial frame')
    include_apocenters=request.get('include_apocenters',False)
    if type(include_apocenters) is not bool:raise ValueError('Explicit boolean apocenter selection required')
    events=[]
    for i in np.flatnonzero(body.valid):
        for j in np.flatnonzero(host.valid):
            bracket=find_closest_approach_bracket((body,int(i),host,int(j)))
            if bracket is None:continue
            for extremum in polynomial_extrema(bracket['distance2'],0.,bracket['stop']-bracket['start']):
                kind={'minimum':'pericenter','maximum':'apocenter','stationary':'stationary'}[extremum['kind']]
                if kind=='stationary' or (kind=='apocenter' and not include_apocenters):continue
                local=extremum['coordinate']
                t=bracket['start']+local;radius=np.sqrt(max(0,bracket['distance2'](local)))
                if radius<=request['maximum_orbital_radius_kpc']:events.append({'kind':kind,'time_myr':float(t),'radius_kpc':float(radius),'bracket_myr':[bracket['start'],bracket['stop']]})
    # Every root of r(t)^2-Rhost(t)^2 is solved within common Hermite/host-radius pieces.
    from numpy.polynomial import Polynomial
    from core.stream.trajectories import real_roots
    radius=finite(host_record['halo_radius_kpc'],host.times.shape)
    if np.any(radius<=0):raise ValueError('Positive actual evolving halo radius required')
    for i in np.flatnonzero(body.valid):
        for j in np.flatnonzero(host.valid):
            bracket=find_closest_approach_bracket((body,int(i),host,int(j)))
            if bracket is None:continue
            start,stop=bracket['start'],bracket['stop']
            slope=(radius[j+1]-radius[j])/(host.times[j+1]-host.times[j])
            evolving=Polynomial([radius[j]+slope*(start-host.times[j]),slope])
            crossing=bracket['distance2']-evolving*evolving
            for local in real_roots(crossing,0.,stop-start):
                if crossing.deriv()(local)<0:
                    events.append({'kind':'infall','time_myr':float(start+local),'bracket_myr':[start,stop],'radius_definition':host_record['halo_radius_definition']})
    # Shared knot minima are genuine extrema only with opposite derivative signs.
    for t in sorted(set(body.times[1:-1])|set(host.times[1:-1])):
        epsilon=positive(request['boundary_extremum_step_myr'],'boundary extremum step')
        try:
            left=np.linalg.norm(body.state(t-epsilon)[0]-host.state(t-epsilon)[0]);center=np.linalg.norm(body.state(t)[0]-host.state(t)[0]);right=np.linalg.norm(body.state(t+epsilon)[0]-host.state(t+epsilon)[0])
        except ValueError:continue
        if center<min(left,right) and center<=request['maximum_orbital_radius_kpc']:events.append({'kind':'pericenter','time_myr':float(t),'radius_kpc':float(center),'bracket_myr':[float(t-epsilon),float(t+epsilon)]})
        if include_apocenters and center>max(left,right) and center<=request['maximum_orbital_radius_kpc']:events.append({'kind':'apocenter','time_myr':float(t),'radius_kpc':float(center),'bracket_myr':[float(t-epsilon),float(t+epsilon)]})
        host_radius=np.interp([t-epsilon,t,t+epsilon],host.times,radius)
        if left>host_radius[0] and right<host_radius[2] and abs(center-host_radius[1])<=request.get('boundary_radius_tolerance_kpc',1e-8):
            events.append({'kind':'infall','time_myr':float(t),'bracket_myr':[float(t-epsilon),float(t+epsilon)],'radius_definition':host_record['halo_radius_definition']})
    unique=[]
    for event in sorted(events,key=lambda q:q['time_myr']):
        if not any(e['kind']==event['kind'] and abs(e['time_myr']-event['time_myr'])<=request['event_time_deduplication_myr'] for e in unique):unique.append(event)
    return {'schema':'halo-orbital-assembly-events-v1','events':unique,
        'event_method':'shared normalized polynomial extrema on relative vector Hermite distance squared',
        'include_apocenters':include_apocenters,
        'algorithm_contract':orbital_interpolation_contract(),
        'uncertainty_scope':'saved trajectory brackets; interpolation alone supplies no event-error certificate',
        'trajectory_certified':bool(body.certificate_verified and host.certificate_verified),'science_acceptance':'NOT_ACCEPTED'}



ASSEMBLY_KINDS={'assembly_history','assembly_orbits','first_star_environments','primordial_clouds'}



def _graph(inputs,refs):
    if 'native_graph_directory' in refs:
        from .halos import verify_identity_graph
        root=inputs.path(refs['native_graph_directory']);report=verify_identity_graph(root)
        for name in ('report.json','index.sqlite'):inputs.sources[str(root/name)]=digest(root/name)
        return report
    return inputs.json(refs['identity_graph'])



def execute_assembly_stage(stage,inputs,working):
    kind=stage['kind'];p=stage['parameters'];refs=stage['inputs']
    if kind=='assembly_history':
        graph=_graph(inputs,refs);report,arrays=reconstruct_quality_masked_main_branch(graph,p['target'],p['branch'])
        events,values=measure_assembly(arrays,p['events'],host_identity_history=report['host_identity_history'])
        report['assembly_events']=events;report['accretion_transition']=fit_supported_fast_slow_accretion_transition(arrays,p['accretion_fit'])
        arrays.update(values)
    elif kind=='assembly_orbits':
        report=orbital_assembly_events(inputs.json(refs['object_worldline']),inputs.json(refs['host_worldline']),p['controls'])
        arrays={'event_time_myr':np.array([row['time_myr'] for row in report['events']])}
    elif kind=='first_star_environments':
        from .first_stars import identify_cooling_events,join_births_and_assembly,primordial_cloud_support,link_cooling_births_and_bound_systems
        assembly_report,assembly=inputs.product(refs['assembly']);graph=_graph(inputs,refs)
        if 'native_gas_request' in refs:
            from .stellar_population import gas_epoch,native_births
            request=inputs.json(refs['native_gas_request']);root=inputs.path(request['run_directory']);epochs=[]
            for row in request['epochs']:
                epoch,sources=gas_epoch(root,row['snapshot'],memory_bytes=int(inputs.resources['memory_gib']*1024**3),namespace=request['gas_namespace'],host_identity=row['host_identity'],resolution_fields=inputs.json(row['measured_resolution_fields']))
                epochs.append(epoch);inputs.sources.update(sources)
            births=[]
            for before,after in zip(epochs[:-1],epochs[1:]):
                rows,sources=native_births(root,before['native_metadata'],after['native_metadata'],journal_directory=inputs.path(request['birth_journal_directory']),namespace=request['stellar_namespace'],epoch_by_id=request['birth_epoch_by_id'],memory_bytes=int(inputs.resources['memory_gib']*1024**3))
                births.extend(rows);inputs.sources.update(sources)
        else:epochs=inputs.json(refs['gas_epochs']);births=inputs.json(refs['native_births'])
        cooling=identify_cooling_events(epochs,p['cooling']);joined=join_births_and_assembly(graph,births,assembly,{**p['join'],'branch_identity_history':assembly_report['branch_identity_history']})
        support=primordial_cloud_support(inputs.json(refs['capabilities']),inputs.json(refs['streaming_ics']),p['cooling']['resolution'])
        linked=link_cooling_births_and_bound_systems(cooling,joined,graph,assembly_report,p['bound_systems'])
        report={'schema':'first-star-environments-v1',**cooling,**joined,**linked,'primordial_support':support,'science_acceptance':'NOT_ACCEPTED'}
        arrays={'cooling_event_time_myr':np.array([q['event_time_myr'] for q in cooling['cooling_events']]),
            'birth_time_myr':np.array([q['applied_time_myr'] for q in joined['native_births']])}
    elif kind=='primordial_clouds':
        from .first_stars import cloud_morphology,primordial_cloud_support
        support=primordial_cloud_support(inputs.json(refs['capabilities']),inputs.json(refs['streaming_ics']),p['request']['resolution'])
        gas=inputs.json(refs['gas'])
        report,arrays=cloud_morphology(gas,p['request'],support=support)
    else:raise ValueError('Unknown assembly stage')
    namespace=gas.get('namespace','assembly:'+stage['stage_id']) if kind=='primordial_clouds' else 'assembly:'+stage['stage_id']
    return report,arrays,namespace,report.get('epoch') if kind=='primordial_clouds' else None



def export_native_assembly_figures(product,graph_directory,destination,*,memory_gib=1.):
    # Figure dependencies are needed only when exporting, so reading native
    # ancestry does not import the display/campaign layer.
    from pathlib import Path
    from .foundation import file_sha256, atomic_json
    from .data import export_numeric_audit, exact_uint64_ids
    from .stream_common import read_product
    from .research.display import export_scientific_suite
    from .halos import verify_identity_graph
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive assembly workspace required')
    path=Path(product).resolve();root=Path(graph_directory).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous assembly exports')
    if max(path.stat().st_size,(root/'report.json').stat().st_size)*16>memory_gib*1024**3/3:
        raise MemoryError('Native ancestry metadata exceeds allocation')
    report,raw,sources=read_product(path,memory_gib=memory_gib/3)
    if report.get('schema')!='halo-main-branch-v1':raise ValueError('Existing native assembly stage product required')
    graph=verify_identity_graph(root)
    if report['source_graph_fingerprint']!=graph['fingerprint']:raise ValueError('Assembly native graph fingerprint differs')
    for name in ('report.json','index.sqlite'):
        source=str(root/name)
        if report.get('source_hashes',{}).get(source)!=file_sha256(source):raise ValueError('Assembly must bind the exact native graph files')
    sources.update(report['source_hashes']);sources.update(graph['sources'])
    for name in ('report.json','report.sha256','index.sqlite','nodes.npz'):sources[str(root/name)]=file_sha256(root/name)
    nodes={tuple(n['key']):n for n in graph['nodes']};rows=report['native_branch'];n=len(rows)
    if not n:raise ValueError('Assembly product must retain original native branch observations')
    specs=report['numeric_data']['array_contract']['datasets']
    expected={'times_myr':'Myr','scale_factor':'dimensionless','mass_msun':'Msun',
        'log_mass_accretion_rate':'dimensionless','spin_dimensionless':'dimensionless',
        'concentration':'dimensionless','stellar_mass_msun':'Msun','specific_sfr_year_inverse':'1 / year',
        'density_axis_ratios':'dimensionless','potential_axis_ratios':'dimensionless',
        'stellar_morphology_b_over_a':'dimensionless','stellar_morphology_c_over_a':'dimensionless'}
    for key,unit in expected.items():
        if key not in raw or specs[key]['units']!=unit:raise ValueError('Native assembly units/measurement missing: '+key)
    time=raw['times_myr'];a=raw['scale_factor'];quality=raw['history_supported'];interval=raw['interval_supported']
    if (time.shape!=(n,) or a.shape!=(n,) or quality.shape!=(n,) or quality.dtype.kind!='b'
            or interval.shape!=(n-1,) or interval.dtype.kind!='b' or not np.isfinite(time).all()
            or not np.isfinite(a).all() or np.any(a<=0) or np.any(np.diff(time)<=0) or np.any(np.diff(a)<=0)):
        raise ValueError('Actual ordered assembly clocks and exact quality/cadence masks required')
    for i,row in enumerate(rows):
        key=tuple(row['key']);node=nodes.get(key)
        if node is None or row['attributes']!=node['attributes'] or row['status']!=node['status']:
            raise ValueError('Assembly observations differ from native graph')
        clock=graph['epochs'][str(key[2])]
        if clock['canonical_time']!='scale_factor':raise ValueError('Scale-factor assembly figures require cosmological native clocks')
        actual=clock.get('cosmic_time_gyr')
        if actual is None or not np.isclose(time[i],actual*1000,rtol=1e-12,atol=1e-9) or a[i]!=clock.get('actual_scale_factor'):
            raise ValueError('Assembly physical clock differs from actual native graph epoch')
        source=clock.get('cosmic_time_source',{})
        if source.get('path'):
            if file_sha256(source['path'])!=source['sha256']:raise ValueError('Assembly proper-time source changed')
            sources[source['path']]=source['sha256']
    arrays={key:np.asarray(value) for key,value in raw.items()}
    if len({tuple(row['key']) for row in rows})!=n:raise ValueError('Duplicate epoch-qualified assembly identities')
    arrays['object_ids']=exact_uint64_ids([row['key'][1] for row in rows],unique=False)
    arrays['storage_epochs']=exact_uint64_ids([row['key'][2] for row in rows],unique=True)
    namespace=rows[0]['key'][0]
    if any(row['key'][0]!=namespace for row in rows):raise ValueError('Assembly changes object namespace')
    arrays['interval_scale_factor']=(a[:-1]+a[1:])/2
    arrays['mass_supported']=quality&np.isfinite(raw['mass_msun'])&(raw['mass_msun']>0)
    arrays['accretion_supported']=interval&np.isfinite(raw['log_mass_accretion_rate'])
    arrays['stellar_supported']=quality&np.isfinite(raw['stellar_mass_msun'])&(raw['stellar_mass_msun']>=0)
    arrays['ssfr_supported']=quality&np.isfinite(raw['specific_sfr_year_inverse'])
    units={key:specs[key]['units'] for key in raw}
    units.update({key:'identity' if key=='object_ids' else 'storage index' if key=='storage_epochs' else 'dimensionless' for key in arrays if key not in units})
    # Insert an explicitly unsupported display row at every unavailable
    # interval. Original numerical arrays and both endpoints remain intact.
    display_rows=[]
    for i in range(n):
        if i and not interval[i-1]:display_rows.append(-1)
        display_rows.append(i)
    for key in ('scale_factor','mass_msun','mass_supported','spin_dimensionless','spin_dimensionless_supported',
                'concentration','concentration_supported','density_axis_ratios','density_axis_ratios_supported',
                'potential_axis_ratios','potential_axis_ratios_supported','stellar_mass_msun','stellar_supported',
                'specific_sfr_year_inverse','ssfr_supported','stellar_morphology_b_over_a',
                'stellar_morphology_c_over_a','stellar_morphology_supported'):
        value=arrays[key]
        if value.shape[0]!=n:raise ValueError('Native assembly node measurement shape differs: '+key)
        is_mask=key.endswith('supported')
        if is_mask and (value.ndim!=1 or value.dtype.kind!='b'):raise ValueError('Exact native assembly support mask required: '+key)
        view=np.full((len(display_rows),)+value.shape[1:],False if is_mask else np.nan,dtype=bool if is_mask else float)
        for i,row in enumerate(display_rows):
            if row>=0:view[i]=value[row]
        arrays['display_'+key]=view;units['display_'+key]=units[key]
    # Preserve exact measured branch keys separately from floating plot axes.
    epoch={'canonical_time':'scale_factor','row_epoch_contracts':[graph['epochs'][str(row['key'][2])] for row in rows]}
    target=out/'assembly.json'
    for source,expected_hash in sources.items():
        if file_sha256(source)!=expected_hash:raise ValueError('Native assembly input changed')
    out.mkdir(parents=True)
    export_numeric_audit(target,{'schema':'flagship-native-assembly-figures-v1','epoch':epoch,
        'native_assembly':str(path),'native_branch_keys':[row['key'] for row in rows],
        'assembly_events':report.get('assembly_events'),'accretion_transition':report.get('accretion_transition'),
        'science_acceptance':'NOT_ACCEPTED','sampling_scope':'original native quality and interval masks; no interpolation across missing observations',
        'uncertainty_scope':'saved cadence/quality retained; transition errors conditional on original supplied log-mass errors; no added independent uncertainty'},
        arrays,{key:{'units':unit,'quantity':key,'identity_namespace':namespace} if key=='object_ids' else
                {'units':unit,'quantity':key,'missing_policy':'original native values and support masks retained'} for key,unit in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    def series(y,mask,*,x='scale_factor',column=None,label=None):
        if x=='scale_factor':x='display_scale_factor';y='display_'+y;mask='display_'+mask
        record={'source':'assembly','kind':'line','x':x,'y':y,'support':mask,'label':label or y}
        if column is not None:record['y_column']=column
        return record
    def panel(pid,title,unit,lines):
        return {'panel_id':pid,'title':title,'quality_status':'PENDING_RESOLUTION_AND_TREE_CONVERGENCE',
            'selection':'exact native main branch; original quality/cadence masks','xlabel':'Actual scale factor a',
            'ylabel':title+' ['+unit+']','x_units':'dimensionless','y_units':unit,'series':lines}
    plots=[('mass_scale_factor',[panel('mass','Native branch mass','Msun',[series('mass_msun','mass_supported')])]),
        ('log_mass_accretion',[panel('rate','Interval d ln M / d ln a','dimensionless',[
            series('log_mass_accretion_rate','accretion_supported',x='interval_scale_factor')])]),
        ('spin_history',[panel('spin','Native dimensionless spin','dimensionless',[series('spin_dimensionless','spin_dimensionless_supported')])]),
        ('concentration_history',[panel('concentration','Native concentration','dimensionless',[series('concentration','concentration_supported')])]),
        ('shape_history',[panel('shape','Native density/potential axis ratios','dimensionless',[
            series(key,key+'_supported',column=j,label=key+' '+('b/a' if j==0 else 'c/a'))
            for key in ('density_axis_ratios','potential_axis_ratios') for j in (0,1)])]),
        ('stellar_assembly',[panel('stellar','Native stellar mass','Msun',[series('stellar_mass_msun','stellar_supported')])]),
        ('quenching_morphology',[panel('ssfr','Native specific star formation rate','1 / year',[series('specific_sfr_year_inverse','ssfr_supported')]),
            panel('morphology','Native stellar tensor axis ratios','dimensionless',[
                series('stellar_morphology_'+key,'stellar_morphology_supported') for key in ('b_over_a','c_over_a')])])]
    transition=report.get('accretion_transition',{})
    fit_panel=panel('transition','Native mass and conditional transition','Msun',[series('mass_msun','mass_supported')])
    fit_panel['quality_status']=transition.get('status','UNAVAILABLE_TRANSITION_FIT')
    if transition.get('transition'):
        fit_panel['events']=[{'x':transition['transition']['scale_factor'],'label':'conditional fast/slow transition'}]
    plots.append(('fast_slow_transition',[fit_panel]))
    figures=[{'figure_id':'native-'+variant,'family':'assembly','variant':variant,'title':variant.replace('_',' '),
        'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels} for variant,panels in plots]
    atomic_json(out/'products.json',{'schema':'flagship-native-assembly-index-v1','product':str(target),
        'pending':[{'variant':'pericenter_infall_merger_events','reason':'requires separately produced native orbital events'}]})
    return export_scientific_suite({'sources':{'assembly':str(target)},'figures':figures,'dpi':160,
        'maximum_panels_per_figure':2,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)

