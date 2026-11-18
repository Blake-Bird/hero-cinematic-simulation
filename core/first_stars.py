"""Native first-cooling/birth analysis and resolved-cloud products.

Cooling uses recorded radiative losses, not temperature decline alone. Jeans
support uses proper density, temperature and measured smoothing; effective-EOS
or missing chemistry cannot establish primordial fragmentation or an IMF.
Births retain exact gas-parent/host ancestry and saved proper-time brackets.
"""
from pathlib import Path
import re
import numpy as np
from . import source_path
from .foundation import G_CGS, MSUN_G, KPC_CM, SECONDS_PER_YEAR, file_sha256
from .data import exact_uint64_ids, export_numeric_audit, numeric_audit_source_bindings
from .stream_common import finite, positive, integer, read_product


BOLTZMANN_ERG_K=1.380649e-16



HYDROGEN_MASS_G=1.67262192595e-24



def primordial_cloud_support(capabilities,streaming,request):
    """Evaluate whether the recorded physics permits primordial cloud measurements.
    
    capabilities supplies enabled process names or source-bound native boolean
    switches. streaming declares its IC mode, proper-km/s convention, phase
    provenance and reference scale factor. request is retained as the requested
    resolution policy. Return a support decision with missing processes and
    specific rejection reasons. Invalid native source bindings raise ValueError.
    The calculation reads hashes and metadata and writes no files."""
    if 'actual_native_physics' in capabilities:
        from .foundation import file_sha256
        sources=capabilities.get('source_hashes',{})
        if not sources or any(file_sha256(path)!=sha for path,sha in sources.items()):
            raise ValueError('Native primordial capability report lost its actual evolved source bindings')
        actual=capabilities['actual_native_physics']
        if not isinstance(actual,dict) or any(type(value)is not bool for value in actual.values()):
            raise ValueError('Native primordial capability switches must retain exact measured booleans')
        capabilities={'enabled':[name for name,value in actual.items() if value],'source_hashes':sources}
    required={'non_equilibrium_primordial_chemistry','h2_cooling','hd_cooling','resolved_single_phase_temperature'}
    missing=sorted(required-set(capabilities['enabled']))
    reasons=[]
    if missing:reasons.append('missing_physics:'+','.join(missing))
    if streaming['enabled']:
        if streaming['velocity_units']!='proper_kms' or streaming['ic_mode']!='coherent_baryon_dm_relative_velocity':reasons.append('unsupported_streaming_ic_convention')
        if not streaming['phase_preserved'] or not streaming['source_hashes']:reasons.append('unproven_streaming_ic_provenance')
        if not np.isfinite(streaming['reference_scale_factor']) or streaming['reference_scale_factor']<=0:reasons.append('invalid_streaming_reference_epoch')
    return {'supported':not reasons,'missing_physics':missing,'reasons':reasons,
        'streaming':streaming,'requested_cloud_resolution':request,'imf_inference':'not inferred from unresolved SSP particles'}



def gas_resolution_diagnostics(gas,request):
    """Evaluate thermal Jeans support in the original native gas census.

    c_s^2 = gamma*k_B*T/(mu*m_H), lambda_J = c_s*sqrt(pi/(G*rho)),
    M_J = (pi/6)*rho*lambda_J^3. Intermediate units are cgs; outputs use
    km/s, proper kpc and Msun. Resolution compares M_J with neighbor-kernel
    mass and lambda_J with measured smoothing, not particle separation.
    """
    rho=np.asarray(gas['gas_density_g_cm3'],float);temperature=np.asarray(gas['gas_temperature_kelvin'],float)
    mass=finite(gas['mass_msun'],rho.shape);mu=np.asarray(gas['mean_molecular_weight'],float)
    if rho.ndim!=1 or temperature.shape!=rho.shape or mu.shape!=rho.shape:
        raise ValueError('Aligned one-dimensional gas thermodynamic census required')
    if np.any(mass<=0):raise ValueError('Positive native gas masses required for Jeans support')
    gamma=np.full(rho.shape,positive(request['adiabatic_index'],'adiabatic_index'))
    native=np.asarray(gas.get('gas_native_eos_available',np.zeros(rho.shape,dtype=bool)))
    if native.dtype.kind!='b' or native.shape!=rho.shape:raise ValueError('Native EOS support must be an exact aligned boolean mask')
    if native.any():
        measured=np.asarray(gas['gas_gamma_sound'],float)
        if measured.shape!=rho.shape or not np.isfinite(measured[native]).all() or np.any(measured[native]<=1):
            raise ValueError('Actual native molecular sound-speed gamma is invalid')
        gamma[native]=measured[native]
    flags=[]
    for name in ('gas_temperature_available','gas_density_g_cm3_available'):
        mask=np.asarray(gas[name])
        if mask.dtype.kind!='b' or mask.shape!=rho.shape:
            raise ValueError('Exact aligned gas thermodynamic availability masks required')
        flags.append(mask)
    valid=flags[0]&flags[1]&np.isfinite(rho)&np.isfinite(temperature)&np.isfinite(mu)&(rho>0)&(temperature>0)&(mu>0)
    sound=np.full_like(rho,np.nan);length=np.full_like(rho,np.nan);jeans=np.full_like(rho,np.nan)
    sound[valid]=np.sqrt(gamma[valid]*BOLTZMANN_ERG_K*temperature[valid]/(mu[valid]*HYDROGEN_MASS_G))
    length[valid]=sound[valid]*np.sqrt(np.pi/(G_CGS*rho[valid]))
    jeans[valid]=np.pi/6*rho[valid]*length[valid]**3/MSUN_G
    smoothing=finite(gas['smoothing_length_proper_kpc'],rho.shape)
    if np.any(smoothing<=0):raise ValueError('Positive measured proper gas smoothing lengths required')
    kernel_mass=mass*integer(request['kernel_neighbors'],'kernel_neighbors')
    # Overflowed intermediate estimates cannot establish resolved physics.
    valid &= np.isfinite(sound)&np.isfinite(length)&np.isfinite(jeans)
    resolved=valid&(jeans>=request['minimum_jeans_kernel_masses']*kernel_mass)&(length/KPC_CM>=request['minimum_jeans_smoothing_lengths']*smoothing)
    return {'sound_speed_kms':sound/1e5,'jeans_length_kpc':length/KPC_CM,'jeans_mass_msun':jeans,
        'thermodynamic_supported':valid,'cloud_resolution_supported':resolved}



def identify_cooling_events(epochs,request):
    """Track exact gas IDs and integrated native radiative losses between epochs."""
    events=[];unsupported=[]
    for before,after in zip(epochs[:-1],epochs[1:]):
        dt=float(after['time_myr']-before['time_myr'])
        if not 0<dt<=request['maximum_interval_myr']:
            unsupported.append({'interval_myr':[before['time_myr'],after['time_myr']],'reason':'missing_or_sparse_gas_cadence'});continue
        if before['gas_namespace']!=after['gas_namespace']:raise ValueError('Cooling history mixes gas identity namespaces')
        a=before['gas'];b=after['gas'];ids_a=exact_uint64_ids(a['particle_ids'],allow_empty=True);ids_b=exact_uint64_ids(b['particle_ids'],allow_empty=True)
        if len(set(map(int,ids_a)))!=len(ids_a) or len(set(map(int,ids_b)))!=len(ids_b):raise ValueError('Gas census has duplicate exact IDs')
        lookup={int(identifier):i for i,identifier in enumerate(ids_a)}
        resolution=gas_resolution_diagnostics(b,request['resolution'])
        for j,identifier in enumerate(ids_b):
            i=lookup.get(int(identifier))
            if i is None:continue
            available=a['gas_chemistry_energy_available'][i] and b['gas_chemistry_energy_available'][j]
            if not available or not resolution['thermodynamic_supported'][j] or not b['gas_internal_energy_cm2_s2_available'][j]:continue
            loss=b['gas_chemistry_radiated_energy_erg'][j]-a['gas_chemistry_radiated_energy_erg'][i]
            if loss<0:raise ValueError('Native gas cooling cumulative energy regressed along exact gas identity')
            if loss==0:continue
            energy=b['mass_msun'][j]*MSUN_G*b['gas_internal_energy_cm2_s2'][j]
            cooling_time=energy/(loss/(dt*1e6*SECONDS_PER_YEAR))
            freefall=np.sqrt(3*np.pi/(32*G_CGS*b['gas_density_g_cm3'][j]))
            if cooling_time>request['maximum_cooling_to_freefall_ratio']*freefall:continue
            events.append({'kind':'resolved_cooling' if resolution['cloud_resolution_supported'][j] else 'unresolved_cooling',
                'gas_id':str(int(identifier)),'namespace':after['gas_namespace'],'host_identity':after['host_key_by_gas_id'].get(str(int(identifier))),'event_time_myr':float(after['time_myr']),
                'interval_myr':[before['time_myr'],after['time_myr']],'cooling_time_myr':float(cooling_time/(1e6*SECONDS_PER_YEAR)),
                'freefall_time_myr':float(freefall/(1e6*SECONDS_PER_YEAR)),'radiated_energy_erg':float(loss),
                'cloud_resolution_supported':bool(resolution['cloud_resolution_supported'][j]),'source_hashes':after['source_hashes']})
    return {'cooling_events':events,'unsupported_intervals':unsupported,
        'definition':'actual cumulative chemistry energy loss along exact gas identities; not temperature decline from adiabatic expansion'}



def join_births_and_assembly(graph,births,assembly,request):
    """Native gas->SSP ancestry and actual host ownership join at birth epoch."""
    edges=graph['edges'];birth_parent={};owners={}
    for edge in edges:
        src=tuple(edge['source']);dst=tuple(edge['target'])
        if edge['kind']=='birth_parent':
            if edge['evidence'].get('origin')!='native_ancestry':raise ValueError('First-star birth requires native ancestry evidence')
            if dst in birth_parent and birth_parent[dst]!=src:raise ValueError('Ambiguous native gas birth parent')
            birth_parent[dst]=src
        elif edge['kind']=='bound_owner':owners.setdefault(src,[]).append(dst)
    times=np.asarray(assembly['times_myr']);mass=np.asarray(assembly['mass_msun']);valid=np.asarray(assembly['history_supported'],bool)
    joined=[];rejected=[]
    for birth in births:
        key=tuple(birth['stellar_key']);parent=birth_parent.get(key);host=owners.get(key,[])
        if parent is None or len(host)!=1:
            rejected.append({'birth':birth,'reason':'missing_native_gas_parent_or_unique_bound_host'});continue
        t=float(birth['applied_time_myr'])
        if not request['searched_interval_myr'][0]<=t<=request['searched_interval_myr'][1]:continue
        k=int(np.searchsorted(times,t,side='right')-1)
        if k<0 or k>=len(times)-1 or not valid[k:k+2].all() or not assembly['interval_supported'][k]:
            rejected.append({'birth':birth,'reason':'unsupported_host_assembly_bracket'});continue
        branch=request['branch_identity_history']
        if tuple(host[0]) not in {tuple(branch[k]),tuple(branch[k+1])}:
            rejected.append({'birth':birth,'reason':'birth_bound_host_differs_from_target_assembly'});continue
        host_mass=float(np.interp(t,times[k:k+2],mass[k:k+2]))
        joined.append({**birth,'gas_parent_key':list(parent),'bound_host_key':list(host[0]),'host_mass_msun':host_mass,
            'assembly_bracket_myr':times[k:k+2].tolist(),'birth_scope':'native SSP birth; individual stellar IMF unresolved'})
    first=min(joined,key=lambda q:q['applied_time_myr']) if joined else None
    return {'native_births':joined,'rejected_births':rejected,'first_supported_birth':first,
        'complete_search_interval_myr':request['searched_interval_myr'],
        'first_definition':'earliest supported native birth within the recorded search interval'}



def cloud_morphology(gas,request,*,support):
    """Measure connected gas candidates and their mass-weighted spatial moments.
    
    gas supplies aligned proper-kpc positions, solar masses, density, temperature,
    composition, smoothing lengths and availability masks at one saved epoch.
    request fixes Jeans support, density threshold, link radius, particle/edge
    budgets and minimum shape eigenvalue. support is the primordial physics gate.
    
    Filter the original gas census by Jeans resolution and density, construct a
    cKDTree neighbor graph, and measure each sufficiently populated component.
    For a periodic box, unwrap the local component around a measured member and
    reject ambiguous half-box extents. Covariance eigenvectors give principal
    axes and square roots of eigenvalues give RMS lengths in proper kpc.
    
    Return (report, arrays). The report contains status, cloud centroids, masses,
    axes and shape masks. Arrays retain particle selection, Jeans diagnostics,
    selected coordinates, masses and exact IDs when supplied. Unsupported physics
    returns an empty array mapping. An insufficient selection returns diagnostics
    with an unresolved status. This function writes no files."""
    metadata={'schema':'primordial-cloud-morphology-v1','epoch':gas.get('epoch'),
        'particle_namespace':gas.get('namespace'),'science_acceptance':'NOT_ACCEPTED'}
    if not support['supported']:return {**metadata,'status':'UNSUPPORTED_PRIMORDIAL_PHYSICS','support':support},{}
    diagnostics=gas_resolution_diagnostics(gas,request['resolution']);use=diagnostics['cloud_resolution_supported'].copy()
    rho=np.asarray(gas['gas_density_g_cm3']);use&=rho>=request['minimum_cloud_density_g_cm3']
    x=finite(gas['positions_proper_kpc']);m=finite(gas['mass_msun'],(len(x),))
    if use.sum()<request['minimum_cloud_particles']:return {**metadata,'status':'UNRESOLVED_CLOUD_SELECTION','selected_count':int(use.sum())},diagnostics
    points=x[use];weights=m[use]
    if len(points)>request['maximum_cloud_particles']:raise MemoryError('Cloud particle workspace exceeded')
    box=gas.get('periodic_box_proper_kpc');geometry_rejected=[]
    if box is not None:
        box=np.asarray(box,float)
        if box.ndim==0:box=np.repeat(box,3)
        if box.shape!=(3,) or not np.isfinite(box).all() or np.any(box<=0):raise ValueError('Positive native proper periodic cloud box required')
        if np.any(points<0) or np.any(points>=box):raise ValueError('Native periodic cloud points must retain wrapped box convention')
    from scipy.spatial import cKDTree
    tree=cKDTree(points,boxsize=box);groups=[];seen=set()
    counts=tree.query_ball_point(points,request['linking_length_kpc'],return_length=True)
    if counts.sum()>request['maximum_neighbor_edges']:raise MemoryError('Cloud graph edge workspace exceeded')
    neighbors=tree.query_ball_point(points,request['linking_length_kpc'])
    for root in range(len(points)):
        if root in seen:continue
        stack=[root];members=[];seen.add(root)
        while stack:
            index=stack.pop();members.append(index)
            for neighbor in neighbors[index]:
                if neighbor not in seen:seen.add(neighbor);stack.append(neighbor)
        if len(members)<request['minimum_cloud_particles']:continue
        q=points[members];w=weights[members]
        if box is not None:
            # One compact saved-time image around an actual member. This is a
            # spatial periodic selection, never an inferred time winding.
            delta=q-q[0];q=q[0]+delta-box*np.floor(delta/box+.5)
            if np.any(np.ptp(q,axis=0)>=box/2):
                geometry_rejected.append({'particle_count':len(members),'reason':'cloud_spans_ambiguous_periodic_image'});continue
        center=np.average(q,axis=0,weights=w);d=q-center
        if box is not None:center=np.mod(center,box)
        covariance=np.einsum('n,ni,nj->ij',w,d,d)/w.sum();eigenvectors=np.linalg.eigh(covariance)
        values=eigenvectors[0][::-1]
        groups.append({'center_kpc':center,'mass_msun':float(w.sum()),'particle_count':len(members),
            'axes_rms_kpc':np.sqrt(np.maximum(values,0)),'principal_axes':eigenvectors[1][:,::-1],
            'shape_supported':bool(values[0]>0 and values[-1]>request['minimum_axis_variance_kpc2']),
            'classification':'resolved gas cloud; physical fragmentation requires independent refinement campaign'})
    identity={}
    if 'particle_ids' in gas:
        ids=exact_uint64_ids(gas['particle_ids'],allow_empty=True)
        if ids.shape!=use.shape:raise ValueError('Native cloud identities must align with original gas census')
        if not isinstance(gas.get('namespace'),str) or not gas['namespace']:raise ValueError('Native cloud particle namespace required')
        identity['cloud_particle_ids']=ids[use]
    return {**metadata,'status':'MEASURED','clouds':groups,'support':support,'geometry_rejected_clouds':geometry_rejected,
        'periodic_geometry':'compact actual saved-time minimum-image groups; ambiguous half-box groups rejected; no temporal winding' if box is not None else 'nonperiodic original Cartesian gas',
        'science_acceptance':'NOT_ACCEPTED','stellar_imf_claim':False}, {**diagnostics,**identity,'cloud_selection':use,'cloud_position_kpc':points,'cloud_mass_msun':weights}



def link_cooling_births_and_bound_systems(cooling,joined,graph,assembly_report,request):
    """Exact parent-gas links, host-epoch identity and resolved native stellar owners."""
    links=[];branch={tuple(k) for k in assembly_report['branch_identity_history']}
    for birth in joined['native_births']:
        parent=tuple(birth['gas_parent_key']);time=birth['applied_time_myr']
        candidates=[event for event in cooling['cooling_events'] if event['namespace']==parent[0] and int(event['gas_id'])==int(parent[1])
            and event['cloud_resolution_supported'] and event['interval_myr'][0]<=time and event['host_identity'] is not None and tuple(event['host_identity']) in branch]
        links.append({'stellar_key':birth['stellar_key'],'gas_parent_key':list(parent),'birth_time_myr':time,
            'cooling_evidence':min(candidates,key=lambda q:q['interval_myr'][0]) if candidates else None,
            'status':'NATIVE_PARENT_COOLING_LINKED' if candidates else 'NO_SUPPORTED_PARENT_COOLING_EVENT',
            'time_order_note':'cooling is interval-bracketed; overlapping cooling/birth brackets do not establish exact within-step ordering'})
    nodes={tuple(row['key']):row for row in graph['nodes']};systems={}
    for edge in graph['edges']:
        if edge['kind']!='bound_owner' or tuple(edge['target']) not in branch:continue
        key=tuple(edge['source']);node=nodes[key]
        if node['kind'] not in {'ssp','star'} or node['status'] not in {'DETECTED','UNBOUND'}:continue
        owner=tuple(edge['target']);systems.setdefault(owner,[]).append(node)
    bound=[];time_map=request['epoch_time_myr']
    for owner,stars in systems.items():
        measured=[s for s in stars if s['attributes'].get('mass_msun') is not None and s['attributes']['mass_msun']>0]
        if len(measured)<request['minimum_bound_stellar_particles']:continue
        bound.append({'owner_key':list(owner),'time_myr':time_map[str(owner[2])],'bound_stellar_particles':len(measured),
            'bound_stellar_mass_msun':float(sum(s['attributes']['mass_msun'] for s in measured)),
            'definition':'native gravitationally bound stellar membership at saved epoch; no individual fragmentation inference'})
    bound.sort(key=lambda q:q['time_myr'])
    return {'cooling_birth_links':links,'resolved_bound_stellar_systems':bound,'first_supported_bound_system':bound[0] if bound else None}



def first_star_inputs(product,*,memory_gib):
    """Load a sealed first-star environment product within the memory budget.
    
    product is its JSON report path. Verify its numeric archive, source hashes,
    first-star schema and original estimator source binding. Return the report,
    loaded arrays and combined source checksum mapping. Invalid or changed
    products raise before figure construction. This intake writes no files."""
    report,arrays,pins=read_product(product,memory_gib=memory_gib/2)
    if report.get('schema')!='first-star-environments-v1':raise ValueError('Original first-star environment product required')
    sources={**pins,**numeric_audit_source_bindings(report)}
    owner=source_path('first_stars').resolve()
    if sources.get(str(owner))!=file_sha256(owner):raise ValueError('First-star estimator source binding missing')
    return report,arrays,sources



def _identity(value):
    if isinstance(value,bool) or not (type(value) is int or isinstance(value,str) and re.fullmatch('[0-9]+',value)):
        raise ValueError('Exact native event identity required')
    return exact_uint64_ids([int(value)],unique=False)[0]



def first_star_timeline_arrays(report,arrays):
    """Validate event identities and saved times, then construct timeline arrays.
    
    report is the original first-star environment report and arrays is its loaded
    numeric payload. Require Myr time units, exact equality between event-row and
    array times, valid cooling brackets and consistent birth/assembly brackets.
    Check that the reported first birth is the earliest retained joined birth.
    
    Return (display_arrays, identity_namespaces). Cooling and birth ordinates are
    categorical zero and one. uint64 gas/stellar IDs, interval endpoints and
    resolved/unresolved cooling masks remain aligned with the original events.
    The function writes no files and estimates no event rates."""
    declarations=report['numeric_data']['array_contract']['datasets']
    cooling=report['cooling_events'];births=report['native_births']
    for key,rows,timekey in [('cooling_event_time_myr',cooling,'event_time_myr'),('birth_time_myr',births,'applied_time_myr')]:
        if declarations[key]['units']!='Myr':raise ValueError('First-star saved time unit differs')
        actual=np.asarray(arrays[key]);expected=np.asarray([row[timekey] for row in rows],float)
        if actual.shape!=expected.shape or not np.array_equal(actual,expected) or not np.isfinite(actual).all():
            raise ValueError('First-star saved time arrays differ from original event records')
    interval=np.asarray(report['complete_search_interval_myr'],float)
    if interval.shape!=(2,) or not np.isfinite(interval).all() or interval[0]>=interval[1]:raise ValueError('Declared first-star search interval required')
    resolved=[];ids=[];brackets=[];namespaces=set()
    for row in cooling:
        flag=row['cloud_resolution_supported']
        if type(flag) is not bool or row['kind']!=('resolved_cooling' if flag else 'unresolved_cooling'):
            raise ValueError('Original cooling resolution classification differs')
        bracket=np.asarray(row['interval_myr'],float);time=row['event_time_myr']
        if bracket.shape!=(2,) or not np.isfinite(bracket).all() or bracket[0]>=bracket[1] or time!=bracket[1]:
            raise ValueError('Original cooling saved-interval bracket differs')
        if not isinstance(row['namespace'],str) or not row['namespace']:raise ValueError('Cooling identity namespace required')
        namespaces.add(row['namespace']);ids.append(_identity(row['gas_id']));brackets.append(bracket);resolved.append(flag)
    if len(namespaces)>1:raise ValueError('Cooling timeline mixes native namespaces')
    stellar_ids=[];stellar_namespaces=set()
    for row in births:
        key=row['stellar_key'];parent=row['gas_parent_key'];host=row['bound_host_key']
        if len(key)!=3 or len(parent)!=3 or len(host)!=3:raise ValueError('Original native birth ancestry keys required')
        stellar_ids.append(_identity(key[1]));stellar_namespaces.add(key[0]);_identity(parent[1])
        bracket=np.asarray(row['assembly_bracket_myr'],float);time=row['applied_time_myr']
        if bracket.shape!=(2,) or not np.isfinite(bracket).all() or bracket[0]>=bracket[1] or not bracket[0]<=time<=bracket[1] or not interval[0]<=time<=interval[1]:
            raise ValueError('Native birth host assembly bracket differs')
    if len(stellar_namespaces)>1:raise ValueError('Native birth timeline mixes stellar namespaces')
    first=report['first_supported_birth']
    if first!=(min(births,key=lambda row:row['applied_time_myr']) if births else None):raise ValueError('First supported birth differs from joined native census')
    return ({'cooling_time_myr':np.asarray(arrays['cooling_event_time_myr']),
        'cooling_bracket_myr':np.asarray(brackets,float).reshape(-1,2),'gas_ids':np.asarray(ids,np.uint64),
        'cooling_ordinate':np.zeros(len(cooling)),'resolved_cooling':np.asarray(resolved,bool),
        'unresolved_cooling':~np.asarray(resolved,bool),'birth_time_myr':np.asarray(arrays['birth_time_myr']),
        'birth_ordinate':np.ones(len(births)),'stellar_ids':np.asarray(stellar_ids,np.uint64)},
        {'gas_ids':next(iter(namespaces),'unavailable:empty-native-cooling-census'),
         'stellar_ids':next(iter(stellar_namespaces),'unavailable:empty-native-birth-census')})



def export_native_first_star_figures(product,destination,*,memory_gib=2.):
    """Publish the registered cooling/birth timeline from an original measurement.
    
    Load product with first_star_inputs and validate it with
    first_star_timeline_arrays. Create a fresh destination containing the sealed
    numeric bridge and an organized gallery of PNG, PDF, SVG, plotted NPZ arrays
    and figure metadata. Cooling times use saved interval ends in Myr and births
    use their recorded applied times. Resolved and unresolved cooling appear as
    separate series. Empty event populations remain empty.
    
    Return the scientific-suite gallery result. Existing destinations, stale
    source bindings or inconsistent event arrays raise before successful export."""
    from .research.display import export_scientific_suite
    report,arrays,sources=first_star_inputs(product,memory_gib=memory_gib)
    values,namespaces=first_star_timeline_arrays(report,arrays);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous first-star figures')
    epoch={'canonical_time':'original recorded proper time','searched_interval_myr':report['complete_search_interval_myr']}
    specs={k:{'units':'Myr' if k.endswith('_myr') else 'identity' if k.endswith('_ids') else 'dimensionless',
        'quantity':k,'missing_policy':'original joined event census; unresolved cooling separately masked'} for k in values}
    for key,namespace in namespaces.items():specs[key]['identity_namespace']=namespace
    sources[str(Path(__file__).resolve())]=file_sha256(__file__);out.mkdir(parents=True);target=out/'first-cooling-birth.json'
    export_numeric_audit(target,{'schema':'native-first-cooling-birth-figure-v1','epoch':epoch,
        'original_measurement':report,'science_acceptance':'NOT_ACCEPTED',
        'event_definition':'cooling recorded at saved interval end; bracket retained; births use applied native proper time and actual ancestry/host joins',
        'ordinate_definition':'0=cooling; 1=joined native SSP birth; categorical display, no rate or IMF inference',
        'selection':'all original cooling candidates, distinguishing resolution; all supported native birth/assembly joins; rejected joins retained in original report'},
        values,specs,algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('First-star sources changed during export')
    return export_scientific_suite({'sources':{'events':str(target)},'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True,'figures':[{'figure_id':'native-first-cooling-birth','family':'baryons','variant':'first_cooling_birth',
            'title':('Synthetic control: ' if report.get('source_origin')=='synthetic_control' else '')+'Recorded cooling and joined native SSP births',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{
                'panel_id':'events','title':'Original proper-time brackets and ancestry joins retained','quality_status':'CONDITIONAL_NATIVE_SUPPORT',
                'selection':'original candidates; unresolved cooling distinguished; no unsupported birth joins invented',
                'xlabel':'Recorded proper time [Myr]','ylabel':'Event category: 0 cooling, 1 SSP birth [dimensionless]',
                'x_units':'Myr','y_units':'dimensionless','series':[
                    {'source':'events','kind':'scatter','x':'cooling_time_myr','y':'cooling_ordinate','support':'resolved_cooling','label':'resolved cooling','marker_size':12},
                    {'source':'events','kind':'scatter','x':'cooling_time_myr','y':'cooling_ordinate','support':'unresolved_cooling','label':'unresolved cooling candidate','marker_size':12},
                    {'source':'events','kind':'scatter','x':'birth_time_myr','y':'birth_ordinate','label':'joined native SSP births','marker_size':12}]}]}]},
        out/'gallery',base=out,memory_gib=memory_gib)



def cloud_figure_inputs(product,*,memory_gib):
    """Load an original cloud product and verify its estimator provenance.
    
    Accept measured, unsupported-physics and unresolved-selection statuses. Read
    product within memory_gib and verify the numeric archive plus the current
    first_stars estimator checksum. Return (report, arrays, source_hashes).
    Malformed schemas, statuses or changed sources raise ValueError."""
    report,arrays,pins=read_product(product,memory_gib=memory_gib/2)
    status=report.get('status')
    if status not in {'MEASURED','UNSUPPORTED_PRIMORDIAL_PHYSICS','UNRESOLVED_CLOUD_SELECTION'}:
        raise ValueError('Original cloud measurement status required')
    if report.get('schema')!='primordial-cloud-morphology-v1' and report.get('request',{}).get('kind')!='primordial_clouds':
        raise ValueError('Original primordial-cloud producer required')
    sources={**pins,**numeric_audit_source_bindings(report)};owner=source_path('first_stars').resolve()
    if sources.get(str(owner))!=file_sha256(owner):raise ValueError('Cloud estimator source binding missing')
    return report,arrays,sources



def cloud_display_arrays(report,arrays):
    """Extract proper Cartesian cloud positions and preserve original shape support.
    
    report and arrays come from cloud_figure_inputs. For measured products,
    validate units, finite 3D points, cloud centroids, RMS axes, principal axes,
    boolean shape masks and optional exact particle-ID namespaces. Return separate
    gas_x/y/z and center_x/y/z arrays, supported/unresolved centroid masks and
    original uint64 IDs when available. Unsupported products return empty geometry.
    A product that claims measured points despite unsupported status raises.
    No coordinate conversion, fitting or file output occurs here."""
    measured=report['status']=='MEASURED';points=np.empty((0,3));centers=[];shape=[]
    if measured:
        if report.get('support',{}).get('supported') is not True:
            raise ValueError('Measured cloud product lacks original primordial physics support')
        declarations=report['numeric_data']['array_contract']['datasets']
        if declarations['cloud_position_kpc']['units']!='proper kpc':raise ValueError('Original cloud position units differ')
        points=np.asarray(arrays['cloud_position_kpc'])
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Original cloud gas coordinates invalid')
        for row in report['clouds']:
            center=np.asarray(row['center_kpc']);axes=np.asarray(row['axes_rms_kpc']);vectors=np.asarray(row['principal_axes'])
            if center.shape!=(3,) or axes.shape!=(3,) or vectors.shape!=(3,3) or not np.isfinite(center).all() or not np.isfinite(axes).all() or not np.isfinite(vectors).all() or np.any(axes<0) or type(row['shape_supported']) is not bool:
                raise ValueError('Original cloud morphology records invalid')
            centers.append(center);shape.append(row['shape_supported'])
    elif report.get('clouds') or 'cloud_position_kpc' in arrays and len(arrays['cloud_position_kpc']):
        raise ValueError('Unsupported cloud product claims measured clouds')
    centers=np.asarray(centers,float).reshape(-1,3);values={}
    for i,axis in enumerate('xyz'):
        values['gas_'+axis]=points[:,i];values['center_'+axis]=centers[:,i]
    values['shape_supported']=np.asarray(shape,bool);values['shape_unresolved']=~np.asarray(shape,bool)
    if measured and 'cloud_particle_ids' in arrays:
        declarations=report['numeric_data']['array_contract']['datasets'];namespace=report.get('particle_namespace')
        ids=exact_uint64_ids(arrays['cloud_particle_ids'],allow_empty=True)
        if len(ids)!=len(points) or not isinstance(namespace,str) or not namespace or declarations['cloud_particle_ids']['units']!='identity' or declarations['cloud_particle_ids'].get('identity_namespace')!=namespace:
            raise ValueError('Original cloud particle identity contract differs')
        values['cloud_particle_ids']=ids
    return values



def export_native_cloud_figures(product,destination,*,memory_gib=2.):
    """Publish the registered 3D cloud figure and its numerical bridge.
    
    product supplies the original measurement and saved epoch. destination must
    be new. After cloud_display_arrays validates geometry, write a sealed bridge
    and PNG, PDF, SVG, plotted arrays and figure metadata through the scientific
    suite. Gas positions and centroid positions use proper kpc. Original shape
    support controls the centroid series, and absent particle identity remains
    explicit in the metadata. Return the gallery result."""
    from .research.display import export_scientific_suite
    report,arrays,sources=cloud_figure_inputs(product,memory_gib=memory_gib);values=cloud_display_arrays(report,arrays)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous cloud figures')
    # Keep the original saved epoch when present. An absent epoch remains
    # unknown, rather than acquiring a fabricated time or snapshot identity.
    epoch=report.get('epoch');sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out.mkdir(parents=True);target=out/'cloud-morphology.json'
    export_numeric_audit(target,{'schema':'native-cloud-morphology-figure-v1','epoch':epoch,
        'original_measurement':report,'measurement_status':report['status'],'science_acceptance':'NOT_ACCEPTED',
        'selection':'original resolved gas selection and connected cloud candidates; original shape masks retained',
        'coordinates':'original proper Cartesian points; no wrapping, frame conversion or inferred periodic images',
        'identity_support':'original exact particle IDs and namespace retained' if 'cloud_particle_ids' in values else 'no measured per-point identity in this original product; display positions do not certify particle/scene identity',
        'uncertainty':'original conditional gas-resolution/shape status; fragmentation and IMF remain unaccepted'},values,
        {k:{'units':'identity' if k=='cloud_particle_ids' else 'dimensionless' if k.startswith('shape_') else 'proper kpc','quantity':k,
            'missing_policy':'empty unsupported products remain empty; shape classification preserved',
            **({'identity_namespace':report['particle_namespace']} if k=='cloud_particle_ids' else {})} for k in values},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Cloud sources changed during export')
    return export_scientific_suite({'sources':{'clouds':str(target)},'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True,'figures':[{'figure_id':'native-primordial-cloud-morphology','family':'baryons',
            'variant':'primordial_cloud_morphology',
            'title':('Synthetic control: ' if report.get('source_origin')=='synthetic_control' else '')+'Original resolved gas/cloud candidates','science_status':'NOT_ACCEPTED',
            'epoch':epoch,'panels':[{'panel_id':'clouds','projection':'3d','title':'Original physical selection; no fragmentation inference',
                'quality_status':report['status'],'selection':'original resolved gas points and cloud centroids; unsupported products empty',
                'xlabel':'x [proper kpc]','ylabel':'y [proper kpc]','zlabel':'z [proper kpc]',
                'x_units':'proper kpc','y_units':'proper kpc','z_units':'proper kpc','series':[
                    {'source':'clouds','kind':'scatter3d','x':'gas_x','y':'gas_y','z':'gas_z','label':'resolved selected gas','marker_size':2},
                    {'source':'clouds','kind':'scatter3d','x':'center_x','y':'center_y','z':'center_z','support':'shape_supported','label':'shape-supported cloud centroid','marker_size':18},
                    {'source':'clouds','kind':'scatter3d','x':'center_x','y':'center_y','z':'center_z','support':'shape_unresolved','label':'shape-unresolved candidate centroid','marker_size':18}]}]}]},
        out/'gallery',base=out,memory_gib=memory_gib)

