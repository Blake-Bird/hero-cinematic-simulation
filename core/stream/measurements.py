"""Native stream measurements, stellar context, support spectra and figures.

Selections keep exact IDs, saved clocks and measured support. Stellar birth
fields are already in physical units; saved stripping labels are identities,
not elapsed times. Transverse track power uses proper-kpc coordinates and has
units kpc^3; its noise is conditional on the fitted curve, not total fit error.
Unsupported detections, bins and optional host/observer inputs remain explicit.
"""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
from ..foundation import atomic_json, file_sha256


VARIANTS=('stream_progenitor_3d','intrinsic_track','projected_track','curvature','convexity',
          'linear_density','width','velocity','dispersion','gaps','spurs','bifurcations',
          'action_angle','density_power','track_power')



def validate_stream_measurement_template(template):
    required={'schema','coordinates','geometry','observer','actions','export_figures','figure_dpi'}
    if not isinstance(template,dict) or set(template)!=required or template['schema']!='native-catalogue-stream-measurements-v1':
        raise ValueError('Complete frozen native catalogue stream measurement template required')
    required_coordinates={'wrap_controls','curve_controls','bin_controls','measurement_controls','track_bootstrap_replicates'}
    if not isinstance(template['coordinates'],dict) or set(template['coordinates'])!=required_coordinates:
        raise ValueError('Explicit wrap, track, bin, uncertainty and observable controls required')
    if template['coordinates']['measurement_controls'].get('coordinate_kind','intrinsic')!='intrinsic':
        raise ValueError('Native catalogue measurements require the observed intrinsic host chart')
    if type(template['export_figures']) is not bool or type(template['figure_dpi']) is not int or not 32<=template['figure_dpi']<=2400:
        raise ValueError('Explicit bounded registered figure export settings required')
    for name in ('geometry','observer','actions'):
        if template[name] is not None and not isinstance(template[name],dict):raise ValueError('Declared native '+name+' settings or explicit unavailable null required')
    if template['geometry'] is not None and set(template['geometry'])!={'geometry'}:
        raise ValueError('Native geometry settings must declare the viewing ensemble and measured/conditional center uncertainty')
    if template['observer'] is not None:
        if set(template['observer'])!={'frame'}:raise ValueError('Native observer settings require one explicit frame in the source host chart')
        from ..stars import ObserverFrame
        ObserverFrame(**template['observer']['frame'])
    if template['actions'] is not None and set(template['actions'])!={'potential','potential_domain','controls'}:
        raise ValueError('Native stream actions require complete measured potential/domain/integration settings')
    if template['actions'] is not None and template['actions'].get('potential',{}).get('kind') not in {'coefficients','time_dependent_host'}:
        raise ValueError('Native stream actions require a source-bound measured host potential, not an assumed analytic field')
    return template



def validate_catalogue_stream_measurements(parameters):
    for key in ('maximum_particles','maximum_candidates'):
        if type(parameters[key]) is not int or parameters[key]<1:raise ValueError('Positive exact native stream '+key+' required')
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<.1:raise ValueError('Bounded native stream measurement memory required')



def candidate_selection_support(report,arrays,candidate):
    """Retain calibrated classification and exact observed ancestry separately."""
    cid=candidate['candidate_id']
    if type(cid) is not int or cid<0:raise ValueError('Exact nonnegative catalogue candidate identity required')
    labels=np.asarray(arrays['candidate_id'])
    ids=np.asarray(arrays['particle_ids'])
    if labels.shape!=ids.shape or labels.dtype.kind not in 'iu' or ids.dtype!=np.dtype('uint64'):
        raise ValueError('Native stream membership labels and exact uint64 identities required')
    selected=labels==cid
    if int(selected.sum())!=candidate['particle_count']:raise ValueError('Native stream candidate count differs from exact selection')
    decision=candidate['classification'];reasons=[]
    if candidate['resolution_supported'] is not True:reasons.append('Candidate fails the recorded native particle-resolution support')
    if decision.get('status')!='CALIBRATED_DOMAIN' or decision.get('classification')!='thin_stream':
        reasons.append('Candidate classification is '+str(decision.get('status'))+'/'+str(decision.get('classification')))
    wanted=set(map(int,ids[selected]));ancestry={}
    for row in report['ancestry']:
        pid=row['particle_id']
        if type(pid) is not int or not 0<=pid<2**64 or pid in ancestry:raise ValueError('Distinct exact native stellar ancestry records required')
        ancestry[pid]=row
    known=sum(ancestry.get(pid,{}).get('status')=='KNOWN' for pid in wanted)
    stripped=sum(ancestry.get(pid,{}).get('status')=='KNOWN' and ancestry.get(pid,{}).get('bound_to_unbound_bracket') is not None for pid in wanted)
    # Requiring observed ancestry preserves ambiguity rather than assigning
    # unknown particles to a progenitor using spatial proximity.
    if known!=len(wanted):reasons.append('Selected stellar ancestry is missing or ambiguous for one or more exact IDs')
    if stripped==0:reasons.append('No selected exact stellar ID has an observed bound-to-unbound bracket')
    return selected,{'status':'SUPPORTED_SELECTION' if not reasons else 'UNRESOLVED','reasons':reasons,
        'particle_count':len(wanted),'known_ancestry_count':known,'stripped_history_count':stripped,
        'classification':copy.deepcopy(decision),'population_acceptance':'NOT_ACCEPTED',
        'stripping_time_definition':'Saved bound-to-unbound bracket; no interpolated stripping time or long-time winding inferred'}



def catalogue_stream_measurements_runner(parameters,context):
    from core.stream_common import read_product, publish_stream_product
    from core.stream.workflow import Inputs, execute_stream_stage
    from core.stream_common import stream_array_units
    from ..data import numeric_audit_source_bindings
    from ..apertures import native_catalogue_apertures_runner
    from ..flagship_products import export_registered_stream_figures
    validate_catalogue_stream_measurements(parameters);memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Native stream measurement exceeds workflow memory')
    template_path=Path(parameters['template_path']).resolve();catalogue_path=Path(parameters['catalogue_product']).resolve()
    if template_path.stat().st_size>min(16*1024**2,memory*1024**3/32):raise MemoryError('Frozen stream measurement controls exceed bounded intake')
    if file_sha256(template_path)!=parameters['template_sha256']:raise ValueError('Frozen stream measurement controls changed')
    template=validate_stream_measurement_template(json.loads(template_path.read_text()))
    if catalogue_path.stat().st_size*8>memory*1024**3/8:raise MemoryError('Native stellar catalogue metadata exceeds bounded intake')
    report,arrays,pins=read_product(catalogue_path,memory_gib=memory/8)
    if report.get('schema')!='tidal-stream-catalogue-v1':raise ValueError('Actual native tidal stellar catalogue required')
    def merge(additional):
        for path,digest in additional.items():
            if path in pins and pins[path]!=digest:raise ValueError('Conflicting native stream measurement source bindings')
            pins[path]=digest
    merge(numeric_audit_source_bindings(report));merge({str(template_path):parameters['template_sha256']})
    if len(arrays['particle_ids'])>parameters['maximum_particles']:raise MemoryError('Native stellar catalogue exceeds declared stream census')
    if len(report['candidates'])>parameters['maximum_candidates']:raise MemoryError('Native candidate catalogue exceeds frozen selection allocation')
    # Reuse the native census/position/mass/frame intake, including its route
    # for flagship worldline catalogues. Do not trust self-described coordinates.
    intake=context.generation/'native-intake';intake.mkdir()
    native_intake=native_catalogue_apertures_runner({'run_dir':parameters['run_dir'],'catalogue_product':str(catalogue_path),
        'maximum_apertures':parameters['maximum_candidates']+parameters['maximum_particles']+2,'memory_gib':memory/2},
        SimpleNamespace(generation=intake,resources=context.resources))
    birth=None;birth_binding=None
    if parameters.get('hydro_history') is not None:
        birth_path=context.generation/'native-birth-context.json'
        birth_report,birth=publish_native_stream_birth_context(parameters['hydro_history'],birth_path,
            run_dir=parameters['run_dir'],snapshot=native_intake['snapshot'],particle_ids=arrays['particle_ids'],
            namespace=report['namespace'],memory_gib=memory)
        birth_binding={'path':str(birth_path),'sha256':file_sha256(birth_path)}
        merge({str(Path(parameters['hydro_history']).resolve()):file_sha256(parameters['hydro_history'])})
    results=[];seen=set()
    for candidate in report['candidates']:
        cid=candidate['candidate_id']
        if cid in seen:raise ValueError('Native stream candidate is duplicated')
        seen.add(cid);mask,selection=candidate_selection_support(report,arrays,candidate)
        selection['birth_context']=({'product':birth_binding,'observed_count':int(np.count_nonzero(birth['birth_context_available'][mask])),
            'unavailable_count':int(np.count_nonzero(~birth['birth_context_available'][mask]))} if birth is not None else
            {'status':'UNAVAILABLE','reason':'No source-certified native hydro/birth history supplied; no ages, metallicities or parent identities imputed'})
        record={'candidate_id':cid,'selection':selection,'products':{},'variants':{name:{'status':'UNRESOLVED','reason':'Selection lacks physical support'} for name in VARIANTS}}
        folder=context.generation/('candidate-'+str(cid));folder.mkdir()
        context_path=folder/'selection-context.json'
        publish_native_stellar_selection_context(report,arrays,mask,context_path,sources=pins,birth_binding=birth_binding,birth=birth)
        context_binding={'path':str(context_path),'sha256':file_sha256(context_path)}
        record['products']['selection_context']=context_binding;selection['selection_context']=context_binding
        if selection['status']!='SUPPORTED_SELECTION':results.append(record);continue
        progenitor_path=folder/'progenitor-view.json'
        publish_native_stream_progenitor_view(report,arrays,mask,progenitor_path,sources=pins,epoch=report.get('epoch'))
        record['products']['progenitor_view']={'path':str(progenitor_path),'sha256':file_sha256(progenitor_path)}
        record['variants']['stream_progenitor_3d']={'status':'MEASURED','product':record['products']['progenitor_view'],
            'support':'Actual selected debris and current bound progenitor membership; an absent progenitor core stays unobserved'}
        if template['export_figures']:
            export_stream_support_figures(progenitor_path,folder/'progenitor-figures',memory_gib=memory/2,dpi=template['figure_dpi'])
        coordinates=copy.deepcopy(template['coordinates']);coordinates['candidate_id']=cid
        # Native probabilistic arrays remain aligned to the exact source row
        # selection. They are not inferred from a classifier's class posterior.
        for name in ('membership_probability','selection_probability'):
            if name in arrays:
                values=np.asarray(arrays[name])
                if values.shape!=mask.shape:raise ValueError('Native membership/selection probabilities are misaligned')
                coordinates['measurement_controls'][name]=values[mask].tolist()
        if 'unwrapped_phase_rad' in arrays:
            phase=np.asarray(arrays['unwrapped_phase_rad'])
            support=np.asarray(arrays.get('unwrapped_phase_supported',[]))
            if phase.shape!=mask.shape or support.shape!=mask.shape or support.dtype.kind!='b' or not support[mask].all():
                raise ValueError('Actual recorded orbital phase support is incomplete for selected wrap')
            coordinates['unwrapped_phase']=phase[mask].tolist()
        else:coordinates['unwrapped_phase']=None
        stage={'stage_id':'native-catalogue-coordinates','kind':'coordinates','parameters':coordinates,'inputs':{'sample':str(catalogue_path)}}
        inputs=Inputs(catalogue_path.parent,{}, {'memory_gib':memory/2,'maximum_particles':parameters['maximum_particles']});inputs.sources.update(pins)
        measured,values,namespace,epoch=execute_stream_stage(stage,inputs,folder)
        merge(inputs.sources)
        measured.update(selection=selection,catalogue_candidate=copy.deepcopy(candidate),request=stage,
            source_origin='synthetic_control' if report.get('fixture') or report.get('source_origin')=='synthetic_control' else 'native_snapshot_and_catalogue',
            source_catalogue={'path':str(catalogue_path),'sha256':file_sha256(catalogue_path)},
            origin='source-bound native host-centered proper Cartesian catalogue chart',
            periodic_wrapping='single-epoch native spatial chart; only supplied supported orbital phases establish wrap turn labels',
            source_context=report.get('source_context'),source_hashes=dict(pins),science_acceptance='NOT_ACCEPTED')
        path=folder/'coordinates.json';publish_stream_product(path,measured,values,stream_array_units(values),sources=pins,namespace=namespace,epoch=epoch)
        record['products']['coordinates']={'path':str(path),'sha256':file_sha256(path)}
        wraps=[row for row in measured['wraps'] if row['status']=='MEASURED']
        for name in ('intrinsic_track','linear_density','width','velocity','dispersion','gaps','spurs','bifurcations','density_power','track_power'):
            support_key='curve_support' if name=='intrinsic_track' else 'power_supported' if name=='density_power' else 'track_power_supported' if name=='track_power' else 'supported'
            count=sum(int(np.count_nonzero(values['wrap_%04d_'%wrap['wrap_id']+support_key])) for wrap in wraps)
            record['variants'][name]={'status':'MEASURED' if count else 'UNRESOLVED','supported_entries':count,'product':record['products']['coordinates'],
                'support':'Per-wrap/bin and power support masks remain authoritative; morphology does not imply a causal encounter'}
        if any(record['variants'][name]['status']=='MEASURED' for name in ('intrinsic_track','linear_density')) and template['export_figures']:
            gallery=export_registered_stream_figures(path,folder/'coordinate-figures',memory_gib=memory/2,dpi=template['figure_dpi'])
            record['products']['coordinate_gallery']={'path':str(folder/'coordinate-figures/gallery'),'figure_count':len(gallery['figures'])}
        # Each optional derived measurement uses the same established runner.
        # Missing measured host fields/observer settings remain explicit inputs.
        for kind in ('geometry','observer','actions'):
            settings=template[kind]
            names={'geometry':('projected_track','curvature','convexity'),'observer':('projected_track',),'actions':('action_angle',)}[kind]
            if settings is None:
                for name in names:
                    summary=record['variants'][name]
                    summary.setdefault('unavailable_inputs',[]).append(kind)
                    if summary['status']!='MEASURED' and not summary.get('measurements'):
                        summary['reason']='Explicit source-bound '+kind+' inputs are unavailable'
                continue
            for wrap in wraps:
                wid=wrap['wrap_id'];derived=copy.deepcopy(settings);derived['wrap_id']=wid
                if kind=='geometry' and int(np.count_nonzero(values['wrap_%04d_supported'%wid]))<2:
                    record.setdefault('unresolved_measurements',[]).append({'kind':kind,'wrap_id':wid,
                        'reason':'Geometry requires at least two measured supported local width bins; original support masks retained'})
                    continue
                job={'stage_id':'native-catalogue-'+kind,'kind':kind,'parameters':derived,'inputs':{'coordinates':str(path),'sample':str(path)}}
                derived_inputs=Inputs(path.parent,{}, {'memory_gib':memory/2,'maximum_particles':parameters['maximum_particles']});derived_inputs.sources.update(pins)
                try:result,payload,ns,clock=execute_stream_stage(job,derived_inputs,folder)
                except (ValueError,ArithmeticError) as error:
                    record.setdefault('unresolved_measurements',[]).append({'kind':kind,'wrap_id':wid,'reason':str(error)})
                    continue
                merge(derived_inputs.sources)
                result.update(epoch=epoch,source_coordinates={'path':str(path),'sha256':file_sha256(path)},selection=selection,
                    source_origin=measured['source_origin'],
                    request=job,source_hashes=dict(pins),science_acceptance='NOT_ACCEPTED')
                derived_path=folder/(kind+'-wrap-'+str(wid)+'.json');units=stream_array_units(payload)
                if kind=='observer':
                    for key in units:
                        if key.endswith('masyr'):units[key]='mas/year'
                publish_stream_product(derived_path,result,payload,units,sources=pins,namespace=ns,epoch=epoch)
                binding={'path':str(derived_path),'sha256':file_sha256(derived_path)}
                record['products'][kind+'_'+str(wid)]=binding
                counts={}
                for name in names:
                    if kind=='actions':count=int(np.count_nonzero(payload['action_supported']))
                    elif kind=='observer':count=int(np.isfinite(payload['longitude_rad']).sum())
                    else:
                        mask_name='normal_supported' if name=='curvature' else 'diagnostic_supported'
                        count=sum(int(np.count_nonzero(payload['view_%03d_'%view['view_id']+mask_name])) for view in result['views'])
                    counts[name]=count
                    # Keep every wrap/view producer. A later unsupported wrap
                    # must not erase an earlier supported measurement, and the
                    # observer and geometry routes may both supply projections.
                    summary=record['variants'][name]
                    measurements=summary.setdefault('measurements',[])
                    measurements.append({'kind':kind,'wrap_id':wid,
                        'supported_entries':count,'product':binding})
                    total=sum(row['supported_entries'] for row in measurements)
                    representative=next((row for row in measurements if row['supported_entries']),measurements[0])
                    summary.pop('reason',None)
                    summary.update(status='MEASURED' if total else 'UNRESOLVED',supported_entries=total,
                        product=representative['product'],
                        support='Original masks retained for every measurement; count sums supported entries across products, not distinct particles')
                supported=max(counts.values(),default=0)
                if supported and template['export_figures'] and kind in ('geometry','observer'):
                    export_registered_stream_figures(derived_path,folder/(kind+'-figures-wrap-'+str(wid)),memory_gib=memory/2,dpi=template['figure_dpi'])
                elif supported and template['export_figures'] and kind=='actions':
                    export_stream_support_figures(derived_path,folder/('actions-figures-wrap-'+str(wid)),memory_gib=memory/2,dpi=template['figure_dpi'])
        results.append(record)
        del values,inputs
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native stream source changed during measurements')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    result={'schema':'native-catalogue-stream-measurement-result-v1','candidates':results,'source_hashes':pins,
        'status':'MEASURED_WITH_SUPPORT_MASKS' if any('coordinates' in row['products'] for row in results) else 'UNRESOLVED',
        'variant_inventory':list(VARIANTS),'science_acceptance':'NOT_ACCEPTED',
        'scope':'Actual native catalogue selections; existing per-wrap numerical/uncertainty producers and registered figures. Missing detection, ancestry, observer or host-potential support remains explicit.'}
    atomic_json(context.generation/'stream-measurements.json',result)
    return result



def recovered_stream_measurements_runner(parameters,context):
    """Consume the actual per-epoch catalogues emitted after host recovery."""
    validate_catalogue_stream_measurements(parameters)
    path=Path(parameters['aperture_history']).resolve();memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Recovered stream history exceeds campaign allocation')
    if path.stat().st_size*8>memory*1024**3/16:raise MemoryError('Recovered stellar history metadata exceeds bounded intake')
    history=json.loads(path.read_text())
    if history.get('schema')!='native-recovered-science-aperture-history-v1':raise ValueError('Actual recovered native stellar aperture history required')
    snapshots=parameters['snapshots']
    if not isinstance(snapshots,list) or not snapshots or any(type(x) is not int or x<0 for x in snapshots) or len(set(snapshots))!=len(snapshots):
        raise ValueError('Distinct exact native stream measurement saved epochs required')
    pins={str(path):file_sha256(path),**history['source_hashes']}
    rows={row['snapshot']:row for row in history['records']}
    if len(rows)!=len(history['records']) or not set(snapshots)<=set(rows):raise ValueError('Recovered stream history has duplicate or missing saved epochs')
    results=[]
    for snapshot in snapshots:
        row=rows[snapshot];binding=row['catalogue']
        if binding is None:
            results.append({'snapshot':snapshot,'status':'UNRESOLVED','reason':'Recovered host has no measurable native stellar catalogue',
                'variants':{name:{'status':'UNRESOLVED','reason':'No native stellar selection at this actual saved epoch'} for name in VARIANTS}})
            continue
        product=Path(binding['path']).resolve()
        if file_sha256(product)!=binding['sha256']:raise ValueError('Recovered native stellar catalogue changed')
        pins[str(product)]=binding['sha256']
        folder=context.generation/('snapshot-'+str(snapshot));folder.mkdir()
        result=catalogue_stream_measurements_runner({**{key:parameters[key] for key in ('run_dir','template_path','template_sha256','memory_gib','maximum_particles','maximum_candidates')},
            **({'hydro_history':parameters['hydro_history']} if parameters.get('hydro_history') is not None else {}),
            'catalogue_product':str(product)},SimpleNamespace(generation=folder,resources=context.resources))
        result_path=folder/'stream-measurements.json'
        results.append({'snapshot':snapshot,'status':result['status'],'product':{'path':str(result_path),'sha256':file_sha256(result_path)}})
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Recovered stream measurement source changed')
    template=Path(parameters['template_path']).resolve()
    if file_sha256(template)!=parameters['template_sha256']:raise ValueError('Frozen stream history template changed')
    pins[str(template)]=parameters['template_sha256'];pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    result={'schema':'native-recovered-stream-measurement-history-v1','records':results,'variant_inventory':list(VARIANTS),
        'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED',
        'status':'MEASURED_WITH_SUPPORT_MASKS' if any(row['status']=='MEASURED_WITH_SUPPORT_MASKS' for row in results) else 'UNRESOLVED'}
    atomic_json(context.generation/'stream-history.json',result)
    return result



FIELDS={'SSPParentGasID':('identity',True),'SSPBirthStateVersion':('dimensionless',True),
        'SSPBirthMode':('dimensionless',True),'StellarRandomCounter':('dimensionless',True),
        'SSPInitialMass':('Msun',False),'SSPInitialLockedMass':('Msun',False),
        'SSPPromptReturnedMass':('Msun',False),'SSPCurrentMass':('Msun',False),
        'SSPBirthScaleFactor':('dimensionless',False),'SSPBirthNativeTime':(None,False),
        'SSPBirthMetallicity':('dimensionless',False)}



def publish_native_stream_birth_context(history_path,destination,*,run_dir,snapshot,
                                        particle_ids,namespace,memory_gib):
    from ..stars import verify_native_hydro_history
    from ..data import numeric_audit_source_bindings,exact_uint64_ids,export_numeric_audit
    if type(memory_gib) not in (int,float) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive native birth-context memory allocation required')
    path=Path(history_path).resolve();memory=memory_gib*1024**3
    if path.stat().st_size*8>memory/16:raise MemoryError('Native birth-context metadata exceeds bounded intake')
    ids=exact_uint64_ids(particle_ids)
    if len(np.unique(ids))!=len(ids):raise ValueError('Distinct exact selected native stellar IDs required')
    # Verify archive sizes before the existing independent journal/census replay.
    archive=path.with_name(path.name+'.data')/'arrays.npz'
    with zipfile.ZipFile(archive) as container:
        if sum(row.file_size for row in container.infolist())*3>memory:
            raise MemoryError('Native source-history arrays exceed birth-context intake allocation')
    report=verify_native_hydro_history(path)
    if report['status']!='PASS':raise ValueError('Native birth context requires passing source-certified hydro accounting')
    if Path(report['request']['run_directory']).resolve()!=Path(run_dir).resolve():raise ValueError('Native birth history belongs to another evolution')
    if report['provenance']['identity_namespaces']['native_particle']!=namespace:
        raise ValueError('Native stream and birth history use different exact particle namespaces')
    if type(snapshot) is not int or snapshot<0:raise ValueError('Exact native birth-context saved epoch required')
    matches=[index for index,row in enumerate(report['epochs']) if row['snapshot']==snapshot]
    if len(matches)!=1:raise ValueError('Native birth history has missing or duplicate selected epoch')
    ordinal=matches[0];meta=report['epochs'][ordinal];prefix='epoch_%03d_'%ordinal
    declarations=report['numeric_data']['array_contract']['datasets'];payload={'particle_ids':ids}
    with np.load(archive,allow_pickle=False) as arrays:
        available=arrays[prefix+'particle_ids'];species=arrays[prefix+'species']
        if available.dtype!=np.dtype('uint64') or available.ndim!=1 or species.shape!=available.shape:
            raise ValueError('Exact native birth-census identities/species required')
        if len(np.unique(available))!=len(available):raise ValueError('Native birth census duplicates an exact identity')
        order=np.argsort(available);at=np.searchsorted(available[order],ids)
        if np.any(at>=len(order)) or not np.array_equal(available[order[at]],ids):
            raise ValueError('Selected native stellar ID is missing from its actual saved birth census')
        selected=order[at]
        if not np.all(species[selected]==4):raise ValueError('Native stream birth selection contains a non-stellar species')
        for name,(unit,integer) in FIELDS.items():
            field=prefix+name
            if field not in arrays:raise ValueError('Incomplete source-certified SSP birth fields: '+name)
            values=arrays[field]
            if values.shape!=available.shape or values.dtype!=(np.dtype('uint64') if integer else np.dtype('float64')):
                raise ValueError('Misaligned or lossy source-certified birth field: '+name)
            expected=unit if unit is not None else 'dimensionless' if meta['parameters']['comoving'] else 'native elapsed time'
            contract=declarations.get(field,{})
            if contract.get('units')!=expected:raise ValueError('Native birth field units differ from physical census: '+name)
            if integer and unit=='identity' and contract.get('identity_namespace')!=namespace:
                raise ValueError('Native SSP parent namespace differs from its particle lineage')
            payload[name]=values[selected]
    version=payload['SSPBirthStateVersion']
    if np.any(version>1):raise ValueError('Unknown native SSP birth state version')
    supported=version==1
    payload['birth_context_available']=supported
    for name,(_,integer) in FIELDS.items():
        if not integer and (not np.isfinite(payload[name][supported]).all() or np.any(payload[name][supported]<0)):
            raise ValueError('Invalid observed native SSP birth quantities')
    if np.any(payload['SSPBirthNativeTime'][supported]>meta['native_time']):raise ValueError('Native SSP birth lies after the selected saved epoch')
    # Stripping brackets remain in the catalogue ancestry sidecar; a birth
    # clock never substitutes for a measured bound-to-unbound transition.
    pins=numeric_audit_source_bindings(report)
    pins.update({str(path):file_sha256(path),str(archive):file_sha256(archive),str(Path(__file__).resolve()):file_sha256(__file__)})
    units={name:(unit if unit is not None else 'dimensionless' if meta['parameters']['comoving'] else 'native elapsed time') for name,(unit,_) in FIELDS.items()}
    units.update(particle_ids='identity',birth_context_available='dimensionless')
    result={'schema':'native-stream-birth-context-v1','snapshot':snapshot,'epoch':meta['epoch'],
        'namespace':namespace,'source_history':{'path':str(path),'sha256':file_sha256(path)},
        'birth_state_support':{'observed_count':int(supported.sum()),'unavailable_count':int((~supported).sum())},
        'scope':'Original source-certified physical birth fields joined by exact native identity at this saved epoch; no second unit conversion, age/metallicity imputation or inferred stripping time',
        'missing_policy':'Version-zero source birth fields stay unavailable with an explicit mask; their placeholders are not measurements',
        'science_acceptance':'NOT_ACCEPTED'}
    result['source_hashes']=pins
    specs={name:{'units':unit,'quantity':'Source-certified native SSP birth '+name,
        'missing_policy':result['missing_policy'],**({'identity_namespace':namespace} if unit=='identity' else {})} for name,unit in units.items()}
    return export_numeric_audit(destination,result,payload,specs,algorithm=Path(__file__).resolve(),
        context={'epoch':meta['epoch'],'provenance':{'source_history':str(path),'identity_namespace':namespace}},source_hashes=pins),payload



def publish_native_stellar_selection_context(report,arrays,mask,destination,*,sources,birth_binding=None,birth=None):
    """Keep birth provenance and observed stripping brackets beside exact IDs."""
    from ..data import export_numeric_audit
    ids=np.asarray(arrays['particle_ids'])[mask];count=len(ids)
    if ids.dtype!=np.dtype('uint64') or ids.ndim!=1 or len(np.unique(ids))!=len(ids):raise ValueError('Distinct exact native selection identities required')
    lookup={row['particle_id']:row for row in report['ancestry']}
    if len(lookup)!=len(report['ancestry']):raise ValueError('Selected stellar ancestry has duplicate identities')
    if any(type(pid) is not int or not 0<=pid<2**64 for pid in lookup):raise ValueError('Exact native ancestry identities required')
    first=np.zeros(count,dtype=np.int64);last=np.zeros(count,dtype=np.int64);supported=np.zeros(count,dtype=bool)
    known=np.zeros(count,dtype=bool)
    for index,pid in enumerate(ids):
        row=lookup.get(int(pid),{});known[index]=row.get('status')=='KNOWN';bracket=row.get('bound_to_unbound_bracket')
        if bracket is None:continue
        if not isinstance(bracket,(list,tuple)) or len(bracket)!=2 or any(type(label) is not int or not 0<=label<2**63 for label in bracket):
            raise ValueError('Observed stripping brackets need exact native storage labels')
        first[index],last[index]=bracket;supported[index]=known[index]
    payload={'particle_ids':ids,'progenitor_label':np.asarray(arrays['progenitor_label'])[mask],
        'current_bound':np.asarray(arrays['bound'])[mask],'binding_known':np.asarray(arrays['binding_known'])[mask],
        'ancestry_known':known,'stripping_before_snapshot':first,'stripping_after_snapshot':last,'stripping_bracket_available':supported}
    if birth is not None:
        if not np.array_equal(birth['particle_ids'],arrays['particle_ids']):raise ValueError('Native birth context is not in exact catalogue row order')
        payload['birth_context_available']=birth['birth_context_available'][mask]
    else:payload['birth_context_available']=np.zeros(count,dtype=bool)
    identities=[]
    for label,identity in enumerate(report['progenitor_keys']):
        if not isinstance(identity,(list,tuple)) or len(identity)!=2 or not isinstance(identity[0],str) or type(identity[1]) is not int or not 0<=identity[1]<2**64:
            raise ValueError('Exact namespace-qualified native progenitor identity required')
        identities.append({'label':label,'namespace':identity[0],'identity_decimal':str(identity[1])})
    namespace=report['namespace'];pins=dict(sources);pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    if birth_binding is not None:pins[birth_binding['path']]=birth_binding['sha256']
    result={'schema':'native-stellar-selection-context-v1','namespace':namespace,'epoch':report.get('epoch'),
        'progenitor_identity_catalogue':identities,'native_birth_context':birth_binding,
        'stripping_clock':'Original saved snapshot-label bracket; labels are not elapsed time and need not increase numerically',
        'support':'Unknown ancestry, absent birth context and unavailable stripping brackets retain false masks; placeholders are not inferred observations',
        'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'}
    specs={key:{'units':'identity' if key=='particle_ids' else 'dimensionless','quantity':key,
        'missing_policy':result['support'],**({'identity_namespace':namespace} if key=='particle_ids' else {})} for key in payload}
    return export_numeric_audit(destination,result,payload,specs,algorithm=Path(__file__).resolve(),
        context={'epoch':report.get('epoch'),'source_frame':'exact source catalogue row selection'},source_hashes=pins)



def measured_track_support_products(report,arrays,controls):
    """Use measured bin centroids and the declared Hann/support window.

    Track power is the raw two-component transverse-bin-centroid spectrum
    about the fitted principal curve. The noise estimate is conditional on
    that curve; it is published separately and never subtracted or clipped.
    """
    center=np.asarray(arrays['bin_center_kpc']);edges=np.asarray(arrays['bin_edges_kpc'])
    n=len(center);width=np.diff(edges);offset=np.asarray(arrays['transverse_mean_kpc'])
    covariance=np.asarray(arrays['width_covariance_kpc2']);neff=np.asarray(arrays['effective_count'])
    support=np.asarray(arrays['supported'])
    if center.shape!=(n,) or width.shape!=(n,) or offset.shape!=(n,2) or covariance.shape!=(n,2,2) or neff.shape!=(n,) or support.shape!=(n,) or support.dtype.kind!='b':
        raise ValueError('Aligned measured transverse-bin support required')
    if n<2 or not np.isfinite(center).all() or not np.isfinite(width).all() or np.any(width<=0):
        raise ValueError('Increasing finite measured arclength bins required')
    valid=support&np.isfinite(offset).all(axis=1)&np.isfinite(covariance).all(axis=(1,2))&(neff>0)
    window=np.hanning(n)*valid;normalization=float(np.dot(window*window,width))
    modes=np.asarray(arrays['wavenumber_kpc_inverse'])
    if modes.ndim!=1 or not np.isfinite(modes).all() or np.any(modes<0):raise ValueError('Actual physical inverse-length modes required')
    power=np.full((len(modes),2),np.nan);noise=np.full_like(power,np.nan)
    variance=np.zeros((n,2));variance[valid]=np.diagonal(covariance[valid],axis1=1,axis2=2)/neff[valid,None]
    if np.any(variance<0):raise ValueError('Transverse-bin mean variance must be nonnegative')
    if normalization>0:
        values=np.zeros_like(offset);values[valid]=offset[valid]
        kernel=np.exp(-1j*modes[:,None]*center[None,:])
        power=np.abs(kernel@(values*(window*width)[:,None]))**2/normalization
        noise[:]=np.sum((window*width)[:,None]**2*variance,axis=0)/normalization
    minimum=controls.get('minimum_power_supported_bins',n)
    if type(minimum) is not int or minimum<2:raise ValueError('Exact positive minimum track-spectrum bin support required')
    power_support=np.full(len(modes),normalization>0 and int(valid.sum())>=minimum,dtype=bool)
    separation=np.full(n,np.nan);fork_support=np.zeros(n,dtype=bool);seen=set()
    for fork in report['forks']:
        index=fork['bin'];centers=np.asarray(fork['centers_kpc'])
        if type(index) is not int or not 0<=index<n or index in seen or centers.shape!=(2,) or not np.isfinite(centers).all():
            raise ValueError('Distinct original measured transverse two-mode proposals required')
        seen.add(index);separation[index]=abs(float(centers[1]-centers[0]));fork_support[index]=bool(support[index])
    output={'track_power_kpc3':power,'track_noise_power_kpc3':noise,'track_power_supported':power_support,
        'track_spectral_support_window':window,'track_bin_support':valid,
        'bifurcation_separation_kpc':separation,'bifurcation_candidate_supported':fork_support}
    definitions={'track_power':'Raw Fourier power of transverse bin-centroid offsets around fitted curve; Hann window and actual support; no mean/baseline/noise fit or subtraction',
        'track_power_units':'proper kpc^3; each transverse component separately',
        'track_power_noise':'Conditional bin-centroid variance width covariance/effective count, propagated through the same recorded window; not total track-fit uncertainty',
        'bifurcations':'Original local two-mode proposals; candidate separation is not a confirmed fork or encounter',
        'track_power_normalization_kpc':normalization,'track_power_minimum_supported_bins':minimum}
    return output,definitions



def publish_native_stream_progenitor_view(report,arrays,mask,destination,*,sources,epoch):
    """Keep selected debris and currently observed bound progenitor members."""
    from core.stream_common import publish_stream_product
    labels=np.asarray(arrays['progenitor_label']);bound=np.asarray(arrays['bound']);known=np.asarray(arrays['binding_known'])
    if labels.shape!=mask.shape or bound.shape!=mask.shape or known.shape!=mask.shape or labels.dtype.kind not in 'iu' or bound.dtype.kind!='b' or known.dtype.kind!='b':
        raise ValueError('Exact native ancestry and bound-membership masks required')
    selected=set(labels[mask].tolist())-{-1}
    progenitor=np.isin(labels,list(selected))&bound&known
    payload={'stream_position_kpc':arrays['position_kpc'][mask],'stream_particle_ids':arrays['particle_ids'][mask],
        'stream_supported':np.ones(int(mask.sum()),dtype=bool),
        'progenitor_position_kpc':arrays['position_kpc'][progenitor],'progenitor_particle_ids':arrays['particle_ids'][progenitor],
        'progenitor_supported':np.ones(int(progenitor.sum()),dtype=bool)}
    result={'schema':'native-stream-progenitor-view-v1','namespace':report['namespace'],'epoch':epoch,
        'source_context':report.get('source_context'),
        'source_origin':'synthetic_control' if report.get('fixture') or report.get('source_origin')=='synthetic_control' else 'native_snapshot_and_catalogue',
        'selection':'Selected observed debris plus current known bound stellar membership of its recorded progenitor identities',
        'progenitor_keys':[report['progenitor_keys'][label] for label in sorted(selected)],
        'current_bound_progenitor_status':'MEASURED_MEMBERSHIP' if progenitor.any() else 'NOT_OBSERVED',
        'origin':('synthetic control host-centered proper Cartesian chart'
                  if report.get('fixture') or report.get('source_origin') == 'synthetic_control'
                  else 'original native stellar host-centered proper Cartesian catalogue chart'),
        'periodic_wrapping':'single-epoch spatial chart; no long-time winding inferred','science_acceptance':'NOT_ACCEPTED'}
    units={key:'identity' if key.endswith('particle_ids') else 'proper kpc' if key.endswith('kpc') else 'dimensionless' for key in payload}
    return publish_stream_product(destination,result,payload,units,sources=sources,namespace=report['namespace'],epoch=epoch)



def export_stream_support_figures(product,destination,*,memory_gib,dpi=160):
    """Registered native progenitor/stream and supported action-angle views."""
    from core.stream_common import read_product
    from core.research.display import export_scientific_suite
    from pathlib import Path
    report,arrays,_=read_product(product,memory_gib=memory_gib/2)
    if report['schema']=='native-stream-progenitor-view-v1':
        chart = ('Saved synthetic control chart' if report.get('source_origin') == 'synthetic_control'
                 else 'Saved native host chart')
        series=[]
        for name in ('stream','progenitor'):
            if len(arrays[name+'_particle_ids'])==0:continue
            field=name+'_position_kpc'
            series.append({'source':'native','kind':'scatter3d','label':name+' observed members','x':field,'y':field,'z':field,
                'x_column':0,'y_column':1,'z_column':2,'support':name+'_supported'})
        figures=[{'figure_id':'native-stream-progenitor','family':'streams','variant':'stream_progenitor_3d',
            'title':'Selected debris and observed bound progenitor membership','science_status':'NOT_ACCEPTED','epoch':report['epoch'],
            'panels':[{'panel_id':'native','title':f'{chart}; absent progenitor members remain unobserved',
                'quality_status':report['current_bound_progenitor_status'],'selection':report['selection'],
                'xlabel':'x [proper kpc]','ylabel':'y [proper kpc]','zlabel':'z [proper kpc]',
                'x_units':'proper kpc','y_units':'proper kpc','z_units':'proper kpc','series':series}]}]
    elif report['schema']=='stream-action-angle-v1':
        figures=[]
        for column,name in enumerate(report['action_order']):
            figures.append({'figure_id':'native-action-angle-'+name,'family':'streams','variant':'action_angle',
                'title':name+' and its measured angle','science_status':'NOT_ACCEPTED','epoch':report['epoch'],
                'panels':[{'panel_id':'measured','title':'Supported action estimator; independent orbit/convergence acceptance required',
                    'quality_status':'PENDING_INDEPENDENT_ACTION_VALIDATION','selection':report.get('selection'),
                    'xlabel':'angle [radian]','ylabel':'action [proper kpc km/s]','x_units':'radian','y_units':'proper kpc km/s',
                    'series':[{'source':'native','kind':'scatter','label':name,'x':'angles_rad','y':'actions_kpc_kms',
                        'x_column':column,'y_column':column,'support':'action_supported'}]}]})
    else:raise ValueError('Measured native progenitor or action-angle product required')
    if report.get('source_origin')=='synthetic_control':
        for figure in figures:figure['science_status']='SYNTHETIC_CONTROL_NOT_ACCEPTED'
    return export_scientific_suite({'sources':{'native':str(Path(product).resolve())},'figures':figures,
        'dpi':dpi,'maximum_panels_per_figure':1,'organized_outputs':True},destination,
        base=Path(product).resolve().parent,memory_gib=memory_gib)

