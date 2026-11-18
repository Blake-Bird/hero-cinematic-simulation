"""Connected public workflow for blocks 16–30, with durable scientific products.

python commands.py streams REQUEST.json --output DIRECTORY

Stages consume native stars or previous sealed products. Declared memory,
particle, stage and integration limits apply before numerical allocations.
Execution success remains separate from calibration and physical acceptance.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import tempfile
import numpy as np
from core.stream_common import finite, positive, integer, phase_space, read_product, publish_stream_product, json_safe, digest
from core.foundation import atomic_json, publish_directory, merge_source_bindings
from core.native import native_run_lock

from core.stream_common import stream_array_units
from core.stream.experiments import EXPERIMENT_KINDS

_CORE_ROOT = Path(__file__).resolve().parent.parent

KINDS={'native_nfw_host','catalogue','classifier_calibration','coordinates','geometry','live','spray','live_spray_comparison','native_reference_state',
       'actions','action_history','observer','lyapunov','independent_spray','streamamax_reference','independent_actions','spray_calibration'} | EXPERIMENT_KINDS


class Inputs:
    def __init__(self,base,published,resources):
        self.base=Path(base)
        self.published=published
        self.resources=resources
        self.sources={}

    def path(self,reference):
        if isinstance(reference,dict) and 'stage' in reference:
            root=self.published[reference['stage']]
            path=root/reference.get('artifact','product.json')
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Stage reference escapes its generation')
        else:
            path=Path(reference)
            if not path.is_absolute():
                path=self.base/path
        path=path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            self.sources[str(path)]=digest(path)
        return path

    def json(self,reference):
        path=self.path(reference)
        if path.stat().st_size*4>self.resources['memory_gib']*1024**3:
            raise MemoryError('Serialized input exceeds stage budget')
        if isinstance(reference,dict) and 'stage' in reference and path.name=='product.json':
            report,_,sources=read_product(path,memory_gib=self.resources['memory_gib'])
            self.sources.update(sources)
            return report
        return json.loads(path.read_text())

    def product(self,reference):
        report,arrays,sources=read_product(self.path(reference),memory_gib=self.resources['memory_gib'])
        self.sources.update(sources)
        return report,arrays

    def sample(self,reference,*,candidate_id=None,wrap_id=None,epoch_index=-1):
        """Select aligned proper-kpc, km/s and solar-mass rows at one epoch."""
        report,arrays=self.product(reference)
        prefix='' if wrap_id is None else 'wrap_%04d_'%wrap_id
        x=arrays[prefix+'position_kpc']
        v=arrays[prefix+'velocity_kms']
        m=arrays[prefix+'mass_msun']
        ids=arrays[prefix+'particle_ids']
        history=x.ndim==3
        if history:
            epoch_count=len(x)
            if v.shape!=x.shape or m.ndim==2 and m.shape!=x.shape[:2]:
                raise ValueError('Saved phase-space histories must share epochs and particles')
            # Native histories may evolve masses; live reference models may
            # instead store one constant mass vector shared by all epochs.
            x=x[epoch_index]
            v=v[epoch_index]
            if m.ndim==2:
                m=m[epoch_index]
            if 'epochs' in report:
                if len(report['epochs'])!=epoch_count:
                    raise ValueError('Saved epoch metadata must align with the trajectory')
                report={**report,'epoch':report['epochs'][epoch_index]}
        use=np.ones(len(x),bool)
        if candidate_id is not None:
            labels=arrays['candidate_id']
            if history and labels.ndim==2:
                if labels.shape!=(epoch_count,len(x)):
                    raise ValueError('Saved candidate labels must share epochs and particles')
                labels=labels[epoch_index]
            if labels.shape!=(len(x),):
                raise ValueError('Candidate labels must align with the selected epoch')
            use=labels==candidate_id
        if len(x)>self.resources['maximum_particles']:
            raise MemoryError('Input particle census exceeds request budget')
        sample={'position_kpc':x[use],'velocity_kms':v[use],'mass_msun':m[use],
                'particle_ids':ids[use],'namespace':report['namespace']}
        phase_space(sample)
        return sample,report,arrays,use

    def host(self,record):
        from core.gravity_fields import TimeDependentHost, SatelliteWorldline
        definition=self.json(record)
        frame=definition['frame']
        fields=[]
        if 'source_hashes' not in frame:
            frame['source_hashes']={}
        for row in definition['fields']:
            fields.append({**row,'product':str(self.path(row['product']))})
        host=TimeDependentHost.from_products(frame,fields,memory_gib=self.resources['memory_gib'])
        self.sources.update(host.sources)
        if definition.get('satellites'):
            if definition.get('satellite_accounting')!='excluded_from_field':
                raise ValueError('Avoid double-counting actual satellite gravity')
            host.satellites=tuple(SatelliteWorldline(row) for row in definition['satellites'])
            host.satellite_accounting='excluded_from_field'
        return host




def _saved_curve(arrays,prefix):
    from core.stream_tracks import PrincipalCurve
    from scipy.interpolate import UnivariateSpline
    grid=arrays[prefix+'curve_grid']
    position=arrays[prefix+'curve_position_kpc']
    splines=tuple(UnivariateSpline(grid,position[:,k],k=3,s=0,ext=2) for k in range(3))
    return PrincipalCurve(splines,arrays[prefix+'curve_parameter'],arrays[prefix+'curve_arclength_kpc'],grid,position,
        arrays[prefix+'curve_tangent'],arrays[prefix+'curve_normal'],arrays[prefix+'curve_binormal'],
        arrays[prefix+'curve_support'],arrays[prefix+'curve_fit_variance_kpc2'],{'reloaded_track':True})


def _potential(inputs,record):
    from core.action_angle import AnalyticSphericalPotential, SphericalFieldAdapter
    if record['kind']=='analytic':
        return AnalyticSphericalPotential(**record['parameters']),None
    if record['kind']=='time_dependent_host':
        return inputs.host(record['host']),None
    if record['kind']=='coefficients':
        from core.halos import ReconstructedHostPotential,pinned_agama_backend
        path=inputs.path(record['product'])
        report,arrays,sources=read_product(path,memory_gib=inputs.resources['memory_gib'])
        inputs.sources.update(sources)
        if report['field_validation_status']!='PASS':
            raise ValueError('Actions require a calibrated reconstructed field')
        model=ReconstructedHostPotential.from_coefficients(path.parent,report['potential_coefficients'])
        inputs.sources.update(report['source_hashes'])
        inputs.sources.update({str((path.parent/row['file']).resolve()):row['sha256'] for row in report['potential_coefficients']})
        # Composite native coefficients in one host coordinate frame. Arbitrary
        # disk rotations require a real rotated native potential adapter.
        if any(not np.allclose(r,np.eye(3),atol=1e-12) for _,_,r in model.components):
            if record['symmetry']!='spherical':
                raise ValueError('Axisymmetric actions require coefficients exported in the common symmetry frame')
        agama,binding=pinned_agama_backend()
        pot=agama.Potential(*[p for _,p,_ in model.components])
        inputs.sources.update(binding['installed_asset_hashes'])
        if record['symmetry']=='spherical':
            return SphericalFieldAdapter(model,radial_domain_kpc=record['radial_domain_kpc'],escape_energy=record['escape_energy_kms2']),pot
        class Field:
            def evaluate(self,x,t=None):
                if np.any(np.linalg.norm(x,axis=1)>report['request']['support_radius_proper_kpc']):
                    raise ValueError('Orbit exceeds calibrated coefficient support')
                return model.evaluate(x)
        return Field(),pot
    raise ValueError('Unsupported declared action potential')


def _bind_reference_sources(inputs, *reports):
    """Recheck upstream native and product source bindings, including dataset provenance."""
    from core.data import numeric_audit_source_bindings
    for report in reports:
        merge_source_bindings(inputs.sources,numeric_audit_source_bindings(report))


def _require_native_reference_units(native, frame):
    """Verify the physical-unit boundary before deriving any reference state."""
    from core.flagship_products import require_native_worldline_units
    require_native_worldline_units(native)
    contracts=frame['numeric_data']['array_contract']['datasets']
    required={'particle_ids':'identity','position_kpc':'proper kpc','velocity_kms':'km / s',
              'mass_msun':'Msun','times_myr':'Myr','origin_proper_kpc':'proper kpc',
              'origin_velocity_peculiar_kms':'km / s','hubble_kms_mpc':'km / s / Mpc'}
    for name,unit in required.items():
        if contracts.get(name,{}).get('units')!=unit:
            raise ValueError('Converted native frame boundary unit mismatch: '+name)
    if contracts['particle_ids'].get('identity_namespace')!=native['namespace']:
        raise ValueError('Converted frame identity namespace differs from native history')


def _native_reference_host(inputs, refs, state):
    if state is None:
        raise ValueError('A native host reference requires the matching native stellar reference state')
    host,host_arrays=inputs.product(refs['host_reference'])
    if (host.get('schema')!='native-nfw-host-reference-v1' or host.get('status')!='PASS'
            or host.get('frame_definition_sha256')!=state['frame_definition_sha256']
            or host.get('frame_history_sha256')!=state['frame_history_sha256']
            or host.get('native_history_sha256')!=state['native_history_sha256']
            or host.get('epoch')!=state['epoch']):
        raise ValueError('Host must pass withheld force controls and bind the identical native state, frame and epoch')
    _bind_reference_sources(inputs,host,state)
    radius=np.linalg.norm(np.asarray(state['position_kpc'],float))
    if not host['fit_domain_proper_kpc'][0]<=radius<=host['fit_domain_proper_kpc'][1]:
        raise ValueError('Native progenitor is outside the host fit domain')
    return host,host_arrays


def _check_backend_host_force(host, source_arrays, backend_acceleration, working, backend):
    """Check the actual adapter force against the same frozen direct samples."""
    from core.stream_common import finite
    queries=finite(source_arrays['query_positions_kpc'],name='withheld native host positions')
    direct=finite(source_arrays['direct_acceleration_kms2_per_kpc'],queries.shape,'native direct acceleration')
    allowed=finite(source_arrays['force_allowed_kms2_per_kpc'],(len(queries),),'native host force budget')
    actual=finite(backend_acceleration,queries.shape,backend+' acceleration')
    if queries.ndim!=2 or queries.shape[1]!=3 or len(queries)<3 or np.any(allowed<=0):
        raise ValueError('Invalid native host force-validation samples or budget')
    expected=positive(host['force_relative_tolerance'],'host force tolerance')*np.maximum(
        np.linalg.norm(direct,axis=1),positive(host['force_floor_km2_s2_kpc'],'host force floor'))
    if not np.allclose(allowed,expected,rtol=1e-13,atol=0.):
        raise ValueError('Host force budget is inconsistent with its frozen tolerances')
    errors=np.linalg.norm(actual-direct,axis=1)
    if not np.isfinite(errors).all():
        raise ValueError('Backend force residual is not finite')
    worst=float(np.max(errors/allowed))
    summary={'backend':backend,'status':'PASS' if np.all(errors<=allowed) else 'FAIL',
             'maximum_force_budget_fraction':worst,'sample_count':len(queries),
             'scope':'Identical withheld finite-source locations and frozen NFW budget; no orbit-wide or periodic-force validation'}
    if summary['status']!='PASS':
        # The surrounding campaign retains this failed temporary generation.
        np.savez_compressed(working/'backend-force-failure.npz',query_positions_kpc=queries,
            direct_acceleration_kms2_per_kpc=direct,backend_acceleration_kms2_per_kpc=actual,
            error_kms2_per_kpc=errors,allowed_kms2_per_kpc=allowed)
        atomic_json(working/'backend-force-failure.json',summary)
        raise ValueError(backend+' force exceeded the frozen withheld budget; inspect backend-force-failure files')
    return summary,{'backend_query_positions_kpc':queries,
        'backend_direct_acceleration_kms2_per_kpc':direct,
        'backend_acceleration_kms2_per_kpc':actual,'backend_force_error_kms2_per_kpc':errors,
        'backend_force_allowed_kms2_per_kpc':allowed}


def _record_backend_force(report, arrays, summary, force_arrays):
    report['backend_force_check']=summary
    arrays.update(force_arrays)
    report.setdefault('units',{}).update({name:('proper kpc' if name=='backend_query_positions_kpc'
        else '(km/s)^2 / proper kpc') for name in force_arrays})


def _record_host_support(report, arrays, host):
    lo,hi=host['fit_domain_proper_kpc']
    for name in ('position_kpc','progenitor_position_kpc'):
        if name in arrays:
            radii=np.linalg.norm(arrays[name],axis=1)
            arrays[name.replace('position_kpc','within_host_fit_domain')]=(radii>=lo)&(radii<=hi)
    report['native_host_reference']=host
    report['host_domain_scope']='Output position masks only; continuous orbit and release-time force accuracy are not established'


def execute_stream_stage(stage,inputs,working):
    if stage['kind'] in EXPERIMENT_KINDS:
        from core.stream.experiments import execute_experiment_stage
        return execute_experiment_stage(stage,inputs,working)
    kind=stage['kind']
    p=stage.get('parameters',{})
    refs=stage.get('inputs',{})
    namespace=p.get('namespace','stream:'+stage['stage_id'])
    epoch=None
    if kind=='native_nfw_host':
        from core.stream.host_reference import native_nfw_reference
        host_path=inputs.path(refs['host_structure'])
        frame_path=inputs.path(refs['frame_history'])
        native_path=inputs.path(refs['native_history'])
        host,host_arrays=inputs.product(refs['host_structure'])
        frame,frame_arrays=inputs.product(refs['frame_history'])
        native,_=inputs.product(refs['native_history'])
        _require_native_reference_units(native,frame)
        _bind_reference_sources(inputs,host,frame,native)
        report,arrays=native_nfw_reference(host,host_arrays,frame,frame_arrays,native,
            index=p['epoch_index'],structure_sha256=digest(host_path),frame_sha256=digest(frame_path),
            native_sha256=digest(native_path),force_relative_tolerance=p['force_relative_tolerance'],
            force_floor_km2_s2_kpc=p['force_floor_km2_s2_kpc'])
        report['units']={'query_positions_kpc':'proper kpc',
            'direct_acceleration_kms2_per_kpc':'(km/s)^2 / proper kpc',
            'model_acceleration_kms2_per_kpc':'(km/s)^2 / proper kpc',
            'force_error_kms2_per_kpc':'(km/s)^2 / proper kpc',
            'force_allowed_kms2_per_kpc':'(km/s)^2 / proper kpc'}
        epoch=report['epoch']
    elif kind=='native_reference_state':
        # Consume the pre-existing frame conversion rather than guessing periodic
        # images or treating peculiar velocity as a proper position derivative.
        frame_path=inputs.path(refs['frame_history'])
        frame, history=inputs.product(refs['frame_history'])
        native_path=inputs.path(refs['native_history'])
        native, original=inputs.product(refs['native_history'])
        _require_native_reference_units(native,frame)
        _bind_reference_sources(inputs,frame,native)
        if (frame.get('schema')!='native-encounter-frame-history-v1'
                or frame.get('source_origin')!='native_snapshot_and_catalogue'
                or native.get('schema')!='measured-native-stream-worldlines-v1'
                or native.get('execution_status')!='COMPLETE'
                or native.get('request',{}).get('species')!=4
                or frame.get('source_history_sha256')!=digest(native_path)
                or frame.get('namespace')!=native.get('namespace')
                or frame.get('epochs')!=native.get('epochs')):
            raise ValueError('Reference state requires the exact original native stellar history and its converted frame')
        if not isinstance(p.get('host_frame'),str) or p['host_frame']!=frame.get('frame'):
            raise ValueError('Reference host model must explicitly use the converted native frame')
        if frame.get('velocity_convention')!='proper position derivative in km/s; relative peculiar velocity plus H*r':
            raise ValueError('Reference requires the proper position derivative in the fixed native frame')
        index=p['epoch_index']
        if type(index) is not int or index<0 or index>=len(frame['epochs']):
            raise ValueError('Select an explicit nonnegative saved epoch index')
        for key in ('particle_ids','times_myr','mass_msun'):
            if not np.array_equal(history[key],original[key]):
                raise ValueError('Converted history changed original stellar '+key)
        ids=history['particle_ids']; x=history['position_kpc'][index]; v=history['velocity_kms'][index]
        mass=history['mass_msun'][index]
        sample={'position_kpc':x,'velocity_kms':v,'mass_msun':mass,'particle_ids':ids,'namespace':native['namespace']}
        x,v,mass,ids=phase_space(sample)
        minimum=integer(p['minimum_stars'],'minimum_stars',32)
        if len(ids)<minimum or len(ids)>inputs.resources['maximum_particles']:
            raise ValueError('Native stellar cohort is outside the declared resolution or particle limit')
        weight=mass/mass.max(); center=np.average(x,axis=0,weights=weight); bulk=np.average(v,axis=0,weights=weight)
        radius=np.linalg.norm(x-center,axis=1); order=np.argsort(radius)
        half=radius[order[np.searchsorted(np.cumsum(weight[order]),weight.sum()/2)]]
        if not np.isfinite(half) or half<=0:raise ValueError('Native stellar half-mass radius is unresolved')
        total=float(mass.sum())
        if not np.isfinite(total) or total<=0:raise ValueError('Native stellar mass is invalid')
        epoch=frame['epochs'][index]; namespace=native['namespace']
        report={'schema':'native-stellar-reference-state-v1','frame':frame['frame'],
            'frame_definition_sha256':frame['frame_definition_sha256'],'epoch':epoch,
            'time_myr':float(history['times_myr'][index]),'native_history_sha256':digest(native_path),
            'frame_history_sha256':digest(frame_path),'native_particle_namespace':namespace,
            'stellar_particle_count':len(ids),'mass_msun':total,'position_kpc':center.tolist(),
            'velocity_kms':bulk.tolist(),'half_mass_radius_kpc':float(half),
            'plummer_scale_kpc':float(half*np.sqrt(2**(2/3)-1)),
            'definition':'mass-weighted centroid of the entire explicitly selected native stellar cohort; Plummer scale matches its 3D half-mass radius',
            'limitations':'Selection and binding are not inferred; stellar-only mass excludes any dark matter or gas; no equilibrium, host-fit or force-residual acceptance is implied. A time-dependent frame origin is frozen for the static control; its acceleration is not modeled.',
            'science_acceptance':'NOT_ACCEPTED'}
        arrays={'particle_ids':ids}
    elif kind=='catalogue':
        from core.halos import _stellar_host_particles
        from core.data import run_epoch_contract
        from core.stream_catalogue import trace_progenitors,build_stream_catalogue
        from core.stars import read_epoch
        root=inputs.path(refs['run_directory'])
        census_memory=min(inputs.resources['memory_gib'],inputs.resources['maximum_particles']*2048/1024**3)
        x,v,m,ids,context=_stellar_host_particles(root,p['snapshot'],p['host_row'],census_memory)
        if len(x)>inputs.resources['maximum_particles']:
            raise MemoryError('Native stellar census exceeds stage bound')
        namespace=p['particle_namespace']
        bulk=finite(p['host_bulk_velocity_kms'],(3,))
        v=v-bulk
        sample={'position_kpc':x,'velocity_kms':v,'mass_msun':m,'particle_ids':ids,'namespace':namespace}
        ancestry=None
        if refs.get('identity_graph'):
            ancestry=trace_progenitors(ids,namespace,p['snapshot'],inputs.path(refs['identity_graph']),memory_gib=inputs.resources['memory_gib'])
        _,_,_,present,membership,binding_context=read_epoch(root,p['snapshot'],ids)
        if not present.all():
            raise ValueError('Native stellar identity changed between catalogue and binding intake')
        if ancestry is None:
            ancestry={'labels':np.full(len(ids),-1,np.int64),'records':[],'keys':[]}
        ancestry['current_bound']=membership>=0
        ancestry['binding_known']=membership>=-1
        calibration=inputs.json(refs['calibration']) if refs.get('calibration') else None
        if calibration and 'classification_calibration' in calibration:
            calibration=calibration['classification_calibration']
        report,arrays=build_stream_catalogue(sample,p['controls'],ancestry=ancestry,calibration=calibration)
        inputs.sources.update({r['path']:r['sha256'] for r in context['source_sha256']})
        if refs.get('identity_graph'):
            graph=inputs.path(refs['identity_graph'])
            inputs.sources.update({str(graph/'report.json'):digest(graph/'report.json'),str(graph/'index.sqlite'):digest(graph/'index.sqlite')})
        epoch=run_epoch_contract(root,p['snapshot'],context['scale_factor'])
        report.update({'source_context':context,'epoch':epoch,'host_bulk_velocity_kms':bulk})
    elif kind=='classifier_calibration':
        from core.stream_catalogue import fit_classifier_controls
        control=inputs.json(refs['controls'])
        record=fit_classifier_controls(**control)
        report={'schema':'stream-classifier-calibration-v1','classification_calibration':record}
        arrays={'class_mean':np.asarray(record['mean']),'class_covariance':np.asarray(record['covariance'])}
    elif kind=='coordinates':
        from core.stream_tracks import (separate_by_ancestry_and_phase_space,fit_uncertainty_aware_principal_curve,project_to_arclength_transverse_frame,choose_bins_from_resolution_and_sampling,measure_local_observables,bootstrap_track)
        sample,parent,raw,use=inputs.sample(refs['sample'],candidate_id=p.get('candidate_id'),epoch_index=p.get('epoch_index',-1))
        namespace=sample['namespace']
        epoch=parent.get('epoch')
        phase=finite(p['unwrapped_phase'],(len(sample['mass_msun']),)) if p.get('unwrapped_phase') is not None else None
        ancestry=raw.get('progenitor_label')
        ancestry=ancestry[use] if ancestry is not None else None
        groups,wrap_labels=separate_by_ancestry_and_phase_space(sample,progenitor_label=ancestry,unwrapped_phase=phase,**p['wrap_controls'])
        arrays={}
        reports=[]
        for group in groups:
            wid=group['wrap_id']
            rows=group['rows']
            prefix='wrap_%04d_'%wid
            sub={key:np.asarray(sample[key])[rows] for key in ('position_kpc','velocity_kms','mass_msun','particle_ids')}
            sub['namespace']=namespace
            if len(rows)<p['curve_controls']['minimum_track_particles']:
                reports.append({'wrap_id':wid,'status':'INSUFFICIENT_SAMPLING','count':len(rows)})
                continue
            try:
                measurement=dict(p['measurement_controls'])
                for key in ('membership_probability','selection_probability'):
                    if key in measurement:
                        measurement[key]=np.asarray(measurement[key])[rows]
                probability=np.asarray(measurement.get('membership_probability',np.ones(len(rows))))
                selection_probability=np.asarray(measurement.get('selection_probability',np.ones(len(rows))))
                if not np.isfinite(probability).all() or not np.isfinite(selection_probability).all() or np.any((probability<0)|(probability>1)|(selection_probability<=0)|(selection_probability>1)):
                    raise ValueError('Invalid track membership/selection probabilities')
                eligible=probability>0
                fitted_sample={key:sub[key][eligible] for key in ('position_kpc','velocity_kms','mass_msun','particle_ids')}
                fitted_sample['namespace']=namespace
                fitted_sample['mass_msun']=fitted_sample['mass_msun']*probability[eligible]/selection_probability[eligible]
                curve=fit_uncertainty_aware_principal_curve(fitted_sample,p['curve_controls'])
                coordinates=project_to_arclength_transverse_frame(sub,curve)
                edges=choose_bins_from_resolution_and_sampling(coordinates['s_kpc'],**p['bin_controls'])
                measurements,local=measure_local_observables(sub,coordinates,edges,measurement)
                bootstrap_sample={key:sub[key][eligible] for key in ('position_kpc','velocity_kms','mass_msun','particle_ids')}
                bootstrap_sample['namespace']=namespace
                bootstrap_sample['mass_msun']/=selection_probability[eligible]
                covariance,boot=bootstrap_track(bootstrap_sample,p['curve_controls'],curve,replicates=p['track_bootstrap_replicates'],seed=measurement['seed'],membership_probability=probability[eligible])
                values={**{k:sub[k] for k in ('position_kpc','velocity_kms','mass_msun','particle_ids')},**coordinates,**local,
                    'curve_grid':curve.grid,'curve_parameter':curve.parameter,'curve_arclength_kpc':curve.arclength,
                    'curve_position_kpc':curve.position,'curve_tangent':curve.tangent,'curve_normal':curve.normal,
                    'curve_binormal':curve.binormal,'curve_support':curve.support,'curve_fit_variance_kpc2':curve.fit_variance_kpc2,
                    'curve_track_covariance_kpc2':covariance}
                arrays.update({prefix+k:v for k,v in values.items()})
                reports.append({k:v for k,v in group.items() if k!='rows'} | {'status':'MEASURED','curve_fit':curve.diagnostics,'observables':measurements,'track_uncertainty':boot})
            except (ValueError,ArithmeticError) as exc:
                reports.append({'wrap_id':wid,'status':'UNRESOLVED','reason':str(exc),'count':len(rows)})
        arrays.update({'particle_ids':sample['particle_ids'],'wrap_id':wrap_labels,'position_kpc':sample['position_kpc'],'velocity_kms':sample['velocity_kms'],'mass_msun':sample['mass_msun']})
        report={'schema':'stream-coordinates-v1','wraps':reports,'epoch':epoch,'parent_reference':refs['sample'],
                'coordinate_kind':p['measurement_controls'].get('coordinate_kind','intrinsic')}
    elif kind=='geometry':
        from core.stream_tracks import measure_viewing_ensemble,match_shape_profile
        parent,raw=inputs.product(refs['coordinates'])
        namespace=parent['namespace']
        epoch=parent.get('epoch')
        prefix='wrap_%04d_'%p['wrap_id']
        curve=_saved_curve(raw,prefix)
        centers=raw[prefix+'bin_center_kpc']
        wcov=raw[prefix+'width_covariance_kpc2']
        valid=raw[prefix+'supported']&np.isfinite(wcov).all(axis=(1,2))
        if valid.sum()<2:
            raise ValueError('Geometry requires resolved local width bins')
        # Embed the measured 2D transverse tensor into actual host 3D axes.
        c2=np.array([[np.interp(curve.arclength,centers[valid],wcov[valid,i,j]) for j in range(2)] for i in range(2)]).transpose(2,0,1)
        basis=np.stack([curve.normal,curve.binormal],axis=1)
        spatial=np.einsum('nji,njk,nkl->nil',basis,c2,basis)
        # Interpolated tensors may be shown across gaps, but missing width bins
        # cannot establish geometry support on either side of the gap.
        curve.support &= np.interp(curve.arclength,centers,valid.astype(float),left=0.,right=0.)>.999
        tcov=raw[prefix+'curve_track_covariance_kpc2']
        report,arrays=measure_viewing_ensemble(curve,width_covariance_kpc2=spatial,track_covariance_kpc2=tcov if np.isfinite(tcov).all() else None,**p['geometry'])
        if refs.get('shape_profile'):
            report=match_shape_profile(report,inputs.json(refs['shape_profile']),definition=p['shape_definition'])
    elif kind=='live':
        from core.stream_resimulation import initialize_equilibrium_model,initialize_from_measured_progenitor,evolve_self_gravitating_progenitor
        host=inputs.host(refs['host'])
        namespace=p['tracer_namespace']
        if p['initialization']['kind']=='measured':
            sample,parent,_,_=inputs.sample(refs['progenitor'],candidate_id=p.get('candidate_id'))
            sample=initialize_from_measured_progenitor(sample,{**p['initialization']['parameters'],'tracer_namespace':namespace})
        elif p['initialization']['kind']=='plummer':
            sample=initialize_equilibrium_model(p['initialization']['parameters'],namespace=namespace,seed=p['seed'])
        else:raise ValueError('Select a measured progenitor or explicit equilibrium model')
        controls={**p['controls'],'memory_gib':inputs.resources['memory_gib'],'maximum_particles':inputs.resources['maximum_particles']}
        if sample['namespace']==sample.get('parent_namespace'):
            raise ValueError('Resimulation tracers must have a separate namespace')
        import hashlib
        binding=json_safe({'parameters':p,'host_sources':host.sources,'namespace':namespace,
            'initial_state_sha256':hashlib.sha256(b''.join(np.asarray(sample[k]).tobytes() for k in ('position_kpc','velocity_kms','mass_msun','particle_ids','parent_particle_ids') if k in sample)).hexdigest(),
            'code_hashes':{key:value for key,value in inputs.sources.items() if key.endswith('.py')}})
        resume=inputs.path(refs['resume']) if refs.get('resume') else None
        checkpoint=Path(p['checkpoint_directory'])
        checkpoint=checkpoint if checkpoint.is_absolute() else inputs.base/checkpoint
        with native_run_lock(checkpoint,operation='live-stream checkpoint writer'):
            report,arrays=evolve_self_gravitating_progenitor(sample,host,p['output_times_myr'],controls,checkpoint_directory=checkpoint,resume=resume,binding=binding)
    elif kind=='spray':
        from core.stream_resimulation import calibrated_spray_adapter
        host=inputs.host(refs['host'])
        calibration=inputs.json(refs['calibration'])
        calibration=calibration.get('spray_calibration',calibration)
        report,arrays=calibrated_spray_adapter(p,host,calibration=calibration)
        namespace=p['tracer_namespace']
    elif kind=='spray_calibration':
        from core.stream_resimulation import calibrate_spray_from_live
        host=inputs.host(refs['host'])
        controls=[]
        for reference in refs['live_controls']:
            lr,la=inputs.product(reference)
            controls.append((lr,la,digest(inputs.path(reference))))
        comparisons=[]
        for reference in refs.get('heldout_comparisons',[]):
            comparison=inputs.json(reference)
            comparisons.append(comparison)
        calibration=calibrate_spray_from_live(controls,host,heldout_comparisons=comparisons,**p)
        report={'schema':'stream-spray-calibration-v1','spray_calibration':calibration}
        arrays={'release_mean':calibration['release_mean'],'release_covariance':calibration['release_covariance']}
    elif kind=='live_spray_comparison':
        from core.stream_resimulation import compare_live_and_spray
        lr,live=inputs.product(refs['live'])
        sr,spray=inputs.product(refs['spray'])
        report,arrays=compare_live_and_spray(live,spray,**p)
        report.update({'schema':'stream-live-spray-comparison-v1','live_namespace':lr['namespace'],'spray_namespace':sr['namespace'],
            'live_source_sha256':digest(inputs.path(refs['live'])),'comparison_id':stage['stage_id']})
    elif kind in {'actions','action_history','lyapunov','independent_actions'}:
        from core.action_angle import measure_actions, action_history, classify_potential_domain, finite_time_lyapunov, measure_action_distribution
        sample,parent,_,_=inputs.sample(refs['sample'],candidate_id=p.get('candidate_id'),wrap_id=p.get('wrap_id'),epoch_index=p.get('epoch_index',-1))
        namespace=sample['namespace']
        epoch=parent.get('epoch')
        potential,agama_potential=_potential(inputs,p['potential'])
        if p['potential']['kind']=='coefficients':
            potential.evaluate(sample['position_kpc'])
        if kind=='independent_actions':
            from core.action_angle import independent_action_comparison
            report,arrays=independent_action_comparison(sample,potential,p['controls'],agama_potential=agama_potential)
            inputs.sources.update(report['backend']['installed_asset_hashes'])
        elif kind=='actions':
            report,arrays=measure_actions(sample,potential,p['potential_domain'],p['controls'],agama_potential=agama_potential)
            report['action_distribution']=measure_action_distribution(arrays['actions_kpc_kms'],weights=sample['mass_msun'])
            if report.get('backend'):
                inputs.sources.update(report['backend']['installed_asset_hashes'])
        elif kind=='action_history':
            if len(sample['particle_ids'])*len(p['times_myr'])*1024>inputs.resources['memory_gib']*1024**3:
                raise MemoryError('Orbit/action history exceeds conservative stage workspace')
            report,arrays=action_history(sample,potential,p['times_myr'],p['controls'],domain=classify_potential_domain(p['potential_domain']),agama_potential=agama_potential)
        else:
            row=p['particle_row']
            value=finite_time_lyapunov({'position_kpc':sample['position_kpc'][row],'velocity_kms':sample['velocity_kms'][row]},potential,p['times_myr'],p['controls'])
            report={'schema':'stream-orbit-lyapunov-v1','meaning':value.pop('meaning'),'particle_id':int(sample['particle_ids'][row])}
            arrays=value
        arrays.update({'particle_ids':sample['particle_ids'],'position_kpc':sample['position_kpc'],'velocity_kms':sample['velocity_kms'],'mass_msun':sample['mass_msun']})
    elif kind=='observer':
        from core.stars import ObserverFrame,to_observer
        sample,parent,_,_=inputs.sample(refs['sample'],candidate_id=p.get('candidate_id'),wrap_id=p.get('wrap_id'),epoch_index=p.get('epoch_index',-1))
        namespace=sample['namespace']
        epoch=parent.get('epoch')
        frame=ObserverFrame(**p['frame'])
        observed=to_observer(sample['position_kpc'],sample['velocity_kms'],frame)
        columns=('longitude_rad','latitude_rad','distance_kpc','radial_velocity_kms','mu_longitude_star_masyr','mu_latitude_masyr')
        report={'schema':'stream-observer-v1','frame':frame.metadata(),'intrinsic_parent':refs['sample'],'coordinate_kind':'observed'}
        arrays={**{key:observed[:,i] for i,key in enumerate(columns)},'particle_ids':sample['particle_ids'],'position_kpc':sample['position_kpc'],'velocity_kms':sample['velocity_kms'],'mass_msun':sample['mass_msun']}
    elif kind=='independent_spray':
        from core.stream_resimulation import gala_spray_adapter
        import gala.potential as gp
        from gala.units import galactic
        from core.stream_resimulation import require_backend
        _,binding=require_backend('gala')
        inputs.sources.update(binding['installed_source_hashes'])
        controls=dict(p['controls'])
        state=None
        if 'reference_state' in refs:
            state,_=inputs.product(refs['reference_state'])
            _bind_reference_sources(inputs,state)
            if state.get('schema')!='native-stellar-reference-state-v1' or p.get('host_frame')!=state['frame']:
                raise ValueError('Gala host frame must match a native stellar reference state')
            if p.get('tracer_namespace')==state['native_particle_namespace']:
                raise ValueError('Gala tracers require a namespace distinct from native stars')
            if float(finite(controls['step_myr'],(),'Gala step Myr'))>=0:
                raise ValueError('Native final-epoch state requires a negative Gala timestep')
            if any(k in controls for k in ('mass_msun','scale_kpc','position_kpc','velocity_kms')):
                raise ValueError('Do not override measured native reference state')
            controls.update(mass_msun=state['mass_msun'],scale_kpc=state['plummer_scale_kpc'],
                position_kpc=state['position_kpc'],velocity_kms=state['velocity_kms'])
            epoch=state['epoch']
        host=None
        if 'host_reference' in refs:
            if 'gala_potential' in p:
                raise ValueError('Do not override the source-bound native host model')
            host,host_arrays=_native_reference_host(inputs,refs,state)
            potential=gp.NFWPotential(m=host['scale_mass_msun'],r_s=host['scale_radius_proper_kpc'],units=galactic)
            import astropy.units as u
            query=host_arrays['query_positions_kpc']
            acceleration=potential.acceleration(query.T*u.kpc).to_value((u.km/u.s)**2/u.kpc).T
            force_summary,force_arrays=_check_backend_host_force(host,host_arrays,acceleration,working,'Gala')
        else:
            definitions=p['gala_potential']
            cls=getattr(gp,definitions['type'])
            potential=cls(units=galactic,**definitions['parameters'])
        report,arrays=gala_spray_adapter(controls,potential,backend_binding=binding)
        if host is not None:
            _record_host_support(report,arrays,host)
            _record_backend_force(report,arrays,force_summary,force_arrays)
        if state is not None:
            report['native_reference_state']=state
            report['time_reference']='Gala integration and release times are relative to the native final-epoch state; no absolute cosmic-time conversion applied'
            if host is None:report['host_frame_limitation']='Matching declared frame names does not verify the analytical host origin, axes, parameters or force residuals'
        namespace=p['tracer_namespace']
        n=len(arrays['position_kpc'])
        tracer_mass=positive(p['total_tracer_mass_msun'],'total_tracer_mass_msun')
        arrays['particle_ids']=np.arange(1,n+1,dtype=np.uint64)
        arrays['mass_msun']=np.full(n,tracer_mass/n) if n else np.empty(0)
        report['empty_release_selection']=n==0
    elif kind=='streamamax_reference':
        from core.stream_resimulation import streamamax_reference
        controls=dict(p['controls']); state=None
        if 'reference_state' in refs:
            state,_=inputs.product(refs['reference_state'])
            _bind_reference_sources(inputs,state)
            if state.get('schema')!='native-stellar-reference-state-v1' or p.get('host_frame')!=state['frame']:
                raise ValueError('StreaMAX host frame must match a native stellar reference state')
            if p.get('tracer_namespace')==state['native_particle_namespace']:
                raise ValueError('StreaMAX tracers require a namespace distinct from native stars')
            if any(k in controls for k in ('xv_f','params_sat','m_f_sat')) or controls.get('type_sat')!='Plummer':
                raise ValueError('Native reference requires a Plummer satellite without manual state or parameters')
            logmass=float(np.log10(state['mass_msun']))
            controls.update(xv_f=state['position_kpc']+state['velocity_kms'],
                params_sat={'logM':logmass,'Rs':state['plummer_scale_kpc'],
                    'x_origin':state['position_kpc'][0],'y_origin':state['position_kpc'][1],
                    'z_origin':state['position_kpc'][2]},m_f_sat=logmass)
            epoch=state['epoch']
        host=None
        if 'host_reference' in refs:
            if 'type_host' in controls or 'params_host' in controls:
                raise ValueError('Do not override the source-bound native host model')
            host,host_arrays=_native_reference_host(inputs,refs,state)
            controls['type_host']='NFW'
            controls['params_host']={'logM':float(np.log10(host['scale_mass_msun'])),
                'Rs':host['scale_radius_proper_kpc'],'a':1.,'b':1.,'c':1.,
                'x_origin':0.,'y_origin':0.,'z_origin':0.,'dirx':0.,'diry':0.,'dirz':1.}
            # Match upstream's parameter preparation and float32 inputs; its
            # epsilon and JAX differentiation are part of the actual model.
            from core.stream_resimulation import require_backend
            _,backend_binding=require_backend('StreaMAX')
            inputs.sources.update(backend_binding['installed_source_hashes'])
            import importlib
            import jax
            import jax.numpy as jnp
            import astropy.units as u
            potentials=importlib.import_module('StreaMAX.potentials')
            upstream_utils=importlib.import_module('StreaMAX.utils')
            params=upstream_utils.prepare_params(controls['params_host'])
            query=jnp.asarray(host_arrays['query_positions_kpc'],dtype=jnp.float32)
            acceleration=jax.vmap(lambda q:potentials.NFW_acceleration(q[0],q[1],q[2],params))(query)
            acceleration=(np.asarray(acceleration)*u.kpc/u.Gyr**2).to_value((u.km/u.s)**2/u.kpc)
            force_summary,force_arrays=_check_backend_host_force(host,host_arrays,acceleration,working,'StreaMAX')
        report,arrays=streamamax_reference(controls)
        if host is not None:
            _record_host_support(report,arrays,host)
            _record_backend_force(report,arrays,force_summary,force_arrays)
        if state is not None:
            report['native_reference_state']=state
            report['satellite_mass_history']='constant measured stellar mass; no inferred native mass loss'
            report['time_reference']='StreaMAX times start at zero and end at the native final-epoch state; no absolute cosmic-time conversion applied'
            if host is None:report['host_frame_limitation']='Matching declared frame names does not verify the analytical host origin, axes, parameters or force residuals'
        inputs.sources.update(report['backend']['installed_source_hashes'])
        namespace=p['tracer_namespace']
        n=len(arrays['position_kpc'])
        tracer_mass=positive(p['total_tracer_mass_msun'],'total_tracer_mass_msun')
        arrays['particle_ids']=np.arange(1,n+1,dtype=np.uint64)
        arrays['mass_msun']=np.full(n,tracer_mass/n) if n else np.empty(0)
    else:raise ValueError('Unknown stream stage')
    return report,arrays,namespace,epoch


def validate_stream_campaign_request(request):
    """Check stage order and references before any producer starts work."""
    if request.get('schema')!='stream-campaign-v1':
        raise ValueError('Expected stream-campaign-v1')
    resources=request['resources']
    positive(resources['memory_gib'],'memory_gib')
    integer(resources['maximum_particles'],'maximum_particles',32)
    stages=request['stages']
    integer(resources['maximum_stages'],'maximum_stages')
    if not stages or len(stages)>resources['maximum_stages']:
        raise ValueError('Stage count exceeds campaign bound')
    known=[]
    def references(value):
        if isinstance(value,dict):
            if 'stage' in value:
                yield value['stage']
            else:
                for item in value.values():
                    yield from references(item)
        elif isinstance(value,list):
            for item in value:
                yield from references(item)
    for stage in stages:
        name=stage['stage_id']
        if not name.replace('-','_').isidentifier() or name in known or stage['kind'] not in KINDS:
            raise ValueError('Distinct safe stage IDs and known kinds required')
        if any(dep not in known for dep in stage.get('depends_on',[])):
            raise ValueError('Stage dependencies must precede consumers')
        # Potential definitions can contain product references in parameters,
        # as well as the stage's ordinary input list.
        if any(source not in known for source in references([stage.get('inputs',{}),stage.get('parameters',{})])):
            raise ValueError('Stage input references must name an earlier producer')
        known.append(name)
    return request


def run_stream_campaign(request,base,destination,*,request_path=None):
    request=validate_stream_campaign_request(request)
    resources=request['resources']
    stages=request['stages']
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    published={}
    results=[]
    code_paths = [(_CORE_ROOT/name) for name in
        ('streams.py','stream_common.py','action_angle.py','stream_tracks.py','stream_resimulation.py','gravity_fields.py','stream_catalogue.py','observations.py','observation_data.py','stellar_observations.py','bao.py','observation_geometry.py','first_stars.py','stellar_population.py','matched.py','assembly.py','foundation.py','data.py','halos.py','stars.py','cosmology.py')]
    code_paths.extend(sorted((_CORE_ROOT/'stream').glob('*.py')))
    code_paths.extend(sorted((_CORE_ROOT/'research').glob('*.py')))
    codes={str(path.resolve()):digest(path) for path in code_paths}
    with native_run_lock(destination):
        for stage in stages:
            inputs=Inputs(base,published,resources)
            inputs.sources.update(codes)
            if request_path:
                inputs.sources[str(Path(request_path).resolve())]=digest(request_path)
            final=destination/stage['stage_id']
            # Reuse only after immutable payload intake and exact source/request binding.
            if final.exists():
                receipt=json.loads((final/'stage.json').read_text())
                if receipt['request']!=stage or receipt['code_hashes']!=codes or receipt.get('resources')!=resources:
                    raise ValueError('Existing stage is from different code/request/resources')
                for path,expected in receipt['source_hashes'].items():
                    if digest(path)!=expected:
                        raise ValueError('Existing stage inputs changed')
                read_product(final/'product.json',memory_gib=resources['memory_gib'])
                published[stage['stage_id']]=final
                results.append(receipt)
                continue
            working=Path(tempfile.mkdtemp(prefix='.'+stage['stage_id']+'-',dir=destination))
            try:
                report,arrays,namespace,epoch=execute_stream_stage(stage,inputs,working)
                if sum(np.asarray(a).nbytes for a in arrays.values())*3>resources['memory_gib']*1024**3:
                    raise MemoryError('Stage output exceeds declared publication workspace')
                report.update({'stage_id':stage['stage_id'],'request':stage,'resources':resources,'execution_status':'COMPLETE','source_hashes':inputs.sources})
                units=stream_array_units(arrays)
                units.update(report.get('units',{}))
                publish_stream_product(working/'product.json',report,arrays,units,sources=inputs.sources,namespace=namespace,epoch=epoch)
                receipt={'schema':'stream-stage-receipt-v1','stage_id':stage['stage_id'],'execution_status':'COMPLETE',
                    'request':stage,'resources':resources,'code_hashes':codes,'source_hashes':inputs.sources,'product_sha256':digest(working/'product.json'),'science_acceptance':'NOT_ACCEPTED'}
                from hashlib import sha256
                receipt['fingerprint']=sha256(json.dumps(json_safe({'request':stage,'resources':resources,'code':codes,'sources':inputs.sources}),sort_keys=True,separators=(',',':')).encode()).hexdigest()
                atomic_json(working/'stage.json',receipt)
                publish_directory(working,final)
                published[stage['stage_id']]=final
                results.append(receipt)
            except BaseException as exc:
                atomic_json(working/'failure.json',{'stage_id':stage['stage_id'],'status':'FAIL','reason':str(exc)})
                raise
        result={'schema':'stream-campaign-result-v1','experiment_id':request['experiment_id'],'stages':results,'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(destination/'campaign.json',result)
    return result


def streams_main():
    parser=argparse.ArgumentParser(description='Native stream catalogue, live stripping, local geometry and action/angle stages (blocks 16–30).')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    request=args.request.resolve()
    result=run_stream_campaign(json.loads(request.read_text()),request.parent,args.output,request_path=request)
    print(json.dumps(json_safe(result),indent=2))
