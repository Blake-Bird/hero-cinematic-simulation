"""Content-bound, capability-aware product DAG with durable stage generations.

Generations keep permanent paths so absolute native provenance survives receipt
publication. Only COMPLETE sealed receipts are reusable. Execution completion
never implies numerical or scientific acceptance.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from . import source_path
import importlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from .foundation import atomic_json, file_sha256, canonical_digest, fsync_directory
from core.native import native_run_lock
from core.research.capabilities import CapabilityRegistry, expand_all_products, require_supported_combination

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class StageDefinition:
    kind: str
    runner: object
    code_paths: tuple[Path, ...]
    capabilities: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    validator: object | None = None

class StageRegistry:
    def __init__(self):
        self.stages = {}

    def register(self, definition):
        if definition.kind in self.stages:
            raise ValueError('Duplicate workflow stage kind: ' + definition.kind)
        self.stages[definition.kind] = definition
        return definition

    def get(self, kind):
        if kind not in self.stages:
            raise ValueError('No public workflow runner registered: ' + kind)
        return self.stages[kind]

    def source_hashes(self, kind):
        paths = [*self.get(kind).code_paths, ROOT/'core/__init__.py', ROOT/'configs/core_source_layout.json',
                 ROOT/'core/resources.py', ROOT/'core/native.py', ROOT/'core/stream_common.py', ROOT/'core/action_angle.py', ROOT/'core/stream_tracks.py', ROOT/'core/stream_resimulation.py', ROOT/'core/gravity_fields.py', ROOT/'core/stream_catalogue.py', ROOT/'core/research/capabilities.py',
                 ROOT/'core/stream_resolution.py']
        # Research helpers are moving out of the historical bundle. Include
        # their actual source bytes so old stage receipts cannot be reused
        # after an extracted calculation or input policy changes.
        paths.extend(sorted((ROOT/'core/research').rglob('*.py')))
        if ROOT/'core/observations.py' in paths:
            paths.extend((ROOT/'core/observation_geometry.py', ROOT/'core/observation_data.py', ROOT/'core/stellar_observations.py', ROOT/'core/bao.py', ROOT/'core/cosmography.py', ROOT/'core/expansion_products.py', ROOT/'core/matched.py'))
        if ROOT/'core/sidm.py' in paths:
            paths.extend((ROOT/'core/sidm_physics.py', ROOT/'core/sidm_measurements.py'))
        if ROOT/'core/streams.py' in paths:paths.append(ROOT/'core/stream/measurements.py')
        if ROOT/'core/stars.py' in paths or ROOT/'core/black_holes.py' in paths:paths.append(ROOT/'core/black_hole_ledger.py')
        return {str(p.resolve()): file_sha256(p) for p in dict.fromkeys(paths)}


def safe_id(value):
    if not isinstance(value, str) or not value or len(value) > 128 or not value.replace('-', '_').isidentifier():
        raise ValueError('Safe explicit stage/product identifier required')
    return value


def positive_number(value, name, integer=False):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite resource required: ' + name)
    if integer and type(value) is not int:
        raise ValueError('Integer resource required: ' + name)
    return value


def path_at(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()


def tree_hashes(root):
    root = Path(root).resolve()
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Stage products must package assets as regular files: ' + str(p))
        if p.is_file() and p.name not in {'stage-receipt.json', 'stage-receipt.sha256'} and '.native-run-locks' not in p.parts:
            result[str(p.relative_to(root))] = file_sha256(p)
    return result


def _input_references(value):
    if isinstance(value, dict):
        if set(value) == {'input'} or set(value) == {'json_input'}:
            yield value.get('input', value.get('json_input'))
        else:
            for item in value.values():
                yield from _input_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _input_references(item)


def _stage_references(value):
    if isinstance(value, dict):
        if 'stage' in value and set(value) <= {'stage','artifact','load','field'}:
            if value.get('load') not in {None,'json','sha256'}:
                raise ValueError('Unknown dependent product loading mode')
            yield value['stage']
        else:
            for item in value.values():
                yield from _stage_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _stage_references(item)


def _reference_records(value):
    if isinstance(value, dict):
        if 'stage' in value and set(value) <= {'stage', 'artifact', 'load','field'}:
            yield value
        else:
            for item in value.values():
                yield from _reference_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _reference_records(item)


def _check_product_references(row, stages):
    """Reject references to undeclared artifacts before any upstream evolution."""
    for reference in _reference_records([row.get('parameters', {}), row.get('inputs', {})]):
        if 'field' in reference:
            selector=reference['field']
            if (reference.get('load')!='json' or not isinstance(selector,list) or not 1<=len(selector)<=64
                    or any(not (isinstance(key,str) and key or type(key)is int and key>=0) for key in selector)):
                raise ValueError('Dependent JSON field selectors require bounded nonempty exact key/index paths')
        producer = stages.get(reference['stage'])
        if producer is None:
            raise ValueError('Dependent artifact has no selected producer: '+str(reference['stage']))
        artifact = reference.get('artifact', 'stage-receipt.json')
        if not isinstance(artifact, str) or not artifact or Path(artifact).is_absolute() or '..' in Path(artifact).parts:
            raise ValueError('Dependent artifact reference escapes its producer generation')
        if artifact != 'stage-receipt.json' and not any(artifact == path or artifact.startswith(path.rstrip('/')+'/') for path in producer['outputs'].values()):
            raise ValueError('Dependent artifact is absent from declared producer outputs: '+reference['stage']+'/'+artifact)


STANDARD_PRODUCT_CATALOGUE={
    'sidm_reference_build':{'kind':'sidm_reference_build','output':'reference-build/reference-build.json','directory_output':True,'native':False,
        'include_in_all':False},
    'sidm_control_population':{'kind':'sidm_control_population','output':'sidm-control-population.json','native':False,
        'include_in_all':False},
    'sidm_canonical_population':{'kind':'sidm_canonical_population','output':'sidm-canonical-population.json','native':False,
        'include_in_all':False,'request_defaults':{'population':'sidm_control_population','native_clock':'sidm_control_native_clock'}},
    'sidm_homogeneous_run':{'kind':'sidm_homogeneous_run','output':'native-control/native-control.json','directory_output':True,'native':False,
        'include_in_all':False,'request_defaults':{'population':'sidm_control_population'}},
    'sidm_canonical_run':{'kind':'sidm_homogeneous_run','output':'native-canonical-control/native-control.json','directory_output':True,'native':False,
        'include_in_all':False,'request_defaults':{'population':'sidm_canonical_population'}},
    'sidm_control_history':{'kind':'sidm_control_history','output':'sidm-control-history.json','native':False,
        'include_in_all':False,'extras':['native_stage_receipt','parameter_file','build_configuration','invocation_receipt']},
    'sidm_canonical_history':{'kind':'sidm_control_history','output':'sidm-canonical-history.json','native':False,
        'include_in_all':False,'request_defaults':{'population':'sidm_canonical_population'},
        'extras':['native_stage_receipt','parameter_file','build_configuration','invocation_receipt']},
    'sidm_control_comparison':{'kind':'sidm_control_comparison','output':'sidm-control-comparison.json','native':False,
        'include_in_all':False},
    'sidm_control_clock_mapping':{'kind':'sidm_control_clock_mapping','output':'sidm-canonical-clock-mapping.json','native':False,
        'include_in_all':False,'request_defaults':{'history':'sidm_control_history'}},
    'sidm_control_native_clock':{'kind':'sidm_control_native_clock','output':'native-canonical-clock/clock.json','native':False,'directory_output':True,
        'include_in_all':False,'request_defaults':{'mapping':'sidm_control_clock_mapping'}},
    'sidm_mapped_campaign_plan':{'kind':'sidm_mapped_campaign_plan','output':'mapped-design/followup-plan.json','directory_output':True,'native':False,
        'include_in_all':False},
    'sidm_control_ensemble':{'kind':'sidm_control_ensemble','output':'sidm-control-ensemble.json','native':False,
        'include_in_all':False},
    'sidm_control_refinement_measurement':{'kind':'sidm_control_refinement_measurement','output':'sidm-refinement.json','native':False,
        'include_in_all':False},
    'sidm_control_expectation':{'kind':'sidm_control_expectation','output':'sidm-control-expectation.json','native':False,
        'include_in_all':False,'request_defaults':{'history':'sidm_control_history'}},
    'native_evolution':{'kind':'public_command','output':'run','native':False},
    'halo_membership':{'kind':'sidm_membership','output':'membership.json','native':True},
    'sidm_history':{'kind':'sidm_history','output':'history.json','native':True},
    'sidm_satellite':{'kind':'sidm_satellite','output':'satellite.json','native':True},
    'sidm_matched':{'kind':'sidm_matched','output':'matched.json','native':False},
    'sidm_observations':{'kind':'sidm_observations','output':'observations.json','native':False},
    'sidm_derived':{'kind':'sidm_derived','output':'derived.json','native':False},
    'physical_target_support':{'kind':'physical_target_support','output':'support.json','native':True,
        'request_defaults':{'center_product':'halo_membership'}},
    'physical_target_model_control':{'kind':'physical_target_model_control','output':'model-control.json','native':False},
    'physical_target_measurement':{'kind':'physical_target_measurement','output':'measurement.json','native':False,
        'request_defaults':{'support_product':'physical_target_support'}},
    'physical_target_budget':{'kind':'physical_target_budget','output':'budget.json','native':False,
        'extras':['target','memory_gib'],'parameter_defaults':{'primary_support':'physical_target_support'}},
    'physical_target_convergence':{'kind':'physical_target_convergence','output':'convergence.json','native':False,
        'extras':['fingerprint','comparisons','memory_gib']},
    'native_forward_vector':{'kind':'native_forward_vector','output':'vector.json','native':True},
    'joint_response_training':{'kind':'sidm_joint_response','output':'response.json','native':False},
    'joint_response_calibration':{'kind':'sidm_joint_calibration','output':'calibration.json','native':False,
        'request_defaults':{'training_response':'joint_response_training'}},
    'calibrated_joint_response':{'kind':'sidm_calibrated_response','output':'calibrated-response.json','native':False,
        'request_defaults':{'training_response':'joint_response_training','calibrations':['joint_response_calibration']}},
    'sidm_reference_cdf':{'kind':'sidm_reference_cdf','output':'sidm-reference-cdf.json','native':False},
    'sidm_native_events':{'kind':'sidm_native_events','output':'sidm-native-events.json','native':True,'receipt_only':True},
    'sidm_angular_control':{'kind':'sidm_angular_control','output':'sidm-angular-control.json','native':False,
        'request_defaults':{'events':'sidm_native_events','reference_cdf':'sidm_reference_cdf'}},
    'twofluid_relative':{'kind':'twofluid_relative','output':'relative-modes.json','native':False,'extras':['initial_fields']},
    'twofluid_nonlinear_reference':{'kind':'twofluid_nonlinear_reference','output':'nonlinear-reference.json','native':False,
        'parameter_defaults':{'initial_fields':'twofluid_eulerian_seed'},'include_in_all':False},
    'twofluid_eulerian_seed':{'kind':'twofluid_eulerian_seed','output':'eulerian-seed.json','native':True,'include_in_all':False},
    'twofluid_nonlinear_refinement':{'kind':'twofluid_nonlinear_refinement','output':'nonlinear-refinement.json','native':False,'include_in_all':False},
    'twofluid_projection_refinement':{'kind':'twofluid_projection_refinement','output':'projection-refinement.json','native':False,'include_in_all':False},
    'twofluid_refinement_campaign':{'kind':'twofluid_refinement_campaign','output':'nonlinear-campaign/design.json','native':False,'directory_output':True,'include_in_all':False},
    'dm_wake_force':{'kind':'dm_wake_force','output':'dm-wake-force.json','native':True,'include_in_all':False,'species':['dm']},
    'dm_wake_kernel_comparison':{'kind':'dm_wake_kernel_comparison','output':'dm-wake-kernel-comparison.json','native':False,'include_in_all':False},
    'dm_wake_contrast':{'kind':'dm_wake_contrast','output':'dm-wake-contrast.json','native':False,'include_in_all':False,'species':['dm']},
    'dm_wake_ensemble':{'kind':'dm_wake_ensemble','output':'dm-wake-ensemble.json','native':False,'include_in_all':False,'species':['dm']},
    'dm_wake_response':{'kind':'dm_wake_response','output':'dm-wake-response.json','native':False,'include_in_all':False,'species':['dm']},
    'dm_wake_state_response':{'kind':'dm_wake_state_response','output':'dm-wake-state-response.json','native':False,'include_in_all':False,'species':['dm']},
    'dm_wake_context':{'kind':'dm_wake_context','output':'dm-wake-context.json','native':False,'include_in_all':False,'species':['dm']},
    'dm_wake_state_table':{'kind':'dm_wake_state_table','output':'dm-wake-state-table.json','native':False,'include_in_all':False,'species':['dm'],
        'request_defaults':{'response':'dm_wake_state_response'}},
    'dm_wake_native_table':{'kind':'dm_wake_native_table','output':'dm-wake-native-table.json','native':False,'include_in_all':False,'species':['dm'],
        'request_defaults':{'response':'dm_wake_response'}},
    'native_stream_worldlines':{'kind':'native_stream_worldlines','output':'native-stream-worldlines.json','native':True,'include_in_all':False},
    'native_live_counterfactual_campaign':{'kind':'native_counterfactual_campaign','output':'native-campaign/followup-plan.json','native':False,'directory_output':True,'include_in_all':False},
    'native_stream_comparison':{'kind':'native_stream_comparison','output':'native-stream-comparison.json','native':False,'include_in_all':False},
    'native_late_threeway':{'kind':'native_late_threeway','output':'native-late-threeway.json','native':False,'include_in_all':False},
    'twofluid_native_comparison':{'kind':'twofluid_native_comparison','output':'twofluid-native-comparison.json','native':False,'include_in_all':False},
    'twofluid_native_family':{'kind':'twofluid_native_family','output':'twofluid-native-family.json','native':False,'include_in_all':False},
    'native_counterfactual_result':{'kind':'native_counterfactual_result','output':'native-counterfactual-result.json','native':False,'include_in_all':False},
    'native_counterfactual_ic':{'kind':'native_counterfactual_ic','output':'counterfactual-ic/counterfactual-ic.json','native':True,'directory_output':True,'include_in_all':False},
    'native_counterfactual_run':{'kind':'native_counterfactual_run','output':'counterfactual-run/counterfactual-run.json','native':True,'directory_output':True,'include_in_all':False,
        'request_defaults':{'intervention_ic':'native_counterfactual_ic'}},
    'twofluid_particle_projection':{'kind':'twofluid_particle_projection','output':'projected-particles.json','native':False,
        'request_defaults':{'relative_fields':'twofluid_relative'}},
    'twofluid_nonlinear_projection':{'kind':'twofluid_particle_projection','output':'projected-material.json','native':False,'include_in_all':False,
        'request_defaults':{'relative_fields':'twofluid_nonlinear_reference'}},
    'twofluid_nonlinear_native_ic':{'kind':'twofluid_native_ic','output':'native-ic/twofluid-generation.json','native':False,'directory_output':True,'include_in_all':False,
        'request_defaults':{'particle_projection':'twofluid_nonlinear_projection'}},
    'twofluid_nonlinear_window_comparison':{'kind':'twofluid_window_comparison','output':'material-window.json','native':False,'include_in_all':False,
        'request_defaults':{'native_generation':'twofluid_nonlinear_native_ic'}},
    'twofluid_native_ic':{'kind':'twofluid_native_ic','output':'native-ic/twofluid-generation.json','native':False,'directory_output':True,
        'request_defaults':{'particle_projection':'twofluid_particle_projection'}},
    'twofluid_window_comparison':{'kind':'twofluid_window_comparison','output':'material-window.json','native':False,
        'request_defaults':{'native_generation':'twofluid_native_ic'}},
    'native_galaxy_measurement':{'kind':'native_pilot_galaxy_measurement','output':'galaxies/galaxy-measurement.json','native':True,'directory_output':True},
}


def standard_product_recipes(spec,*,request_base,registry=None):
    """Turn scientific options into concrete existing registered producers.

    Named instances can select one catalogue producer without writing stage
    definitions. Scientific requests, assets, physical limits and measurements
    remain explicit. References select whole products, paths or actual JSON
    fields; every referenced product becomes a prerequisite automatically.
    """
    from copy import deepcopy
    registry=registry or default_registry();base=Path(request_base).resolve()
    options=spec['science'].get('product_options',{});custom=deepcopy(spec['science'].get('product_recipes',{}))
    declarations={};sources={};current_assets=set();memory=spec['resources']['memory_gib']*1024**3
    for product,option in options.items():
        safe_id(product);producer=option.get('producer',product)
        if producer not in STANDARD_PRODUCT_CATALOGUE:raise ValueError('Unknown standard scientific producer: '+str(producer))
        descriptor=STANDARD_PRODUCT_CATALOGUE[producer];registry.get(descriptor['kind'])
        declarations[product]=(option,descriptor)
    catalogue=deepcopy(custom)
    for product,(_,descriptor) in declarations.items():
        catalogue[product]={'stage_id':product,'kind':descriptor['kind'],'outputs':{product:descriptor['output']}}
    def pinned(reference,*,load=False):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Standard producer source requires an exact path and SHA256')
        path=path_at(base,reference['path'])
        if not path.is_file() or path.is_symlink() or file_sha256(path)!=reference['sha256']:raise ValueError('Standard scientific input changed: '+str(path))
        sources[str(path)]=reference['sha256']
        current_assets.add(str(path))
        if load:
            if path.stat().st_size*8>memory:raise MemoryError('Standard scientific request exceeds metadata workspace')
            return json.loads(path.read_text())
        return str(path)
    def reference(product,mode='product',field=None,relative_path=None):
        safe_id(product)
        if product not in catalogue:raise ValueError('Scientific prerequisite has no supplied producer options/recipe: '+str(product))
        row=catalogue[product];sid=row.get('stage_id',product);output=row['outputs'].get(product)
        if output is None:raise ValueError('Scientific recipe omits its named concrete output: '+product)
        if relative_path is not None:
            if not isinstance(relative_path,str) or not relative_path:raise ValueError('Scientific child artifact path must be a nonempty string')
            child=Path(relative_path)
            if child.is_absolute() or '..' in child.parts or mode!='path':
                raise ValueError('Scientific child artifact references require a safe relative path and path mode')
            output=(Path(output)/child).as_posix()
        if mode=='receipt':return {'stage':sid,'artifact':'stage-receipt.json'}
        path={'stage':sid,'artifact':output}
        if mode=='path':return path
        if mode in {'json','field'}:
            value={**path,'load':'json'}
            if mode=='field':value['field']=field
            return value
        if mode!='product':raise ValueError('Unknown scientific product reference mode')
        return {'path':path,'sha256':{**path,'load':'sha256'}}
    def expand(value,dependencies,pin_local=True):
        if isinstance(value,dict):
            if 'science_product' in value:
                if set(value)-{'science_product','as','field','relative_path'}:raise ValueError('Unknown high-level scientific reference field')
                product=value['science_product'];dependencies.add(product)
                mode=value.get('as','field' if 'field' in value else 'product')
                if 'field' in value and mode!='field':raise ValueError('Scientific field selection requires field reference mode')
                return reference(product,mode,value.get('field'),value.get('relative_path'))
            if pin_local and set(value)=={'path','sha256'} and isinstance(value['path'],str) and isinstance(value['sha256'],str):
                return {'path':pinned(value),'sha256':value['sha256']}
            return {key:expand(item,dependencies,pin_local) for key,item in value.items()}
        if isinstance(value,list):return [expand(item,dependencies,pin_local) for item in value]
        return deepcopy(value)
    for product,(option,descriptor) in declarations.items():
        unknown=set(option)-{'producer','request','request_reference','native_product','native_source','answers','initial_conditions',
            'wall_seconds','snapshots','include_in_all','species',*descriptor.get('extras',[]),*descriptor.get('parameter_defaults',{})}
        if unknown:raise ValueError('Unknown scientific producer option fields: '+', '.join(sorted(unknown)))
        dependencies=set();row=catalogue[product];current_assets.clear()
        if descriptor['kind']=='public_command':
            answers=pinned(option['answers']);parameters={'command':'gadget','arguments':['--answers',{'input':answers},'--run-dir','{generation}/run'],
                'wall_seconds':positive_number(option['wall_seconds'],'native wall_seconds')}
            ic=option.get('initial_conditions')
            if ic:
                primary=pinned(ic['primary']);files=[pinned(ref) for ref in ic['files']]
                if primary not in files or len(set(files))!=len(files):raise ValueError('Complete standard native IC shard set must include its primary once')
                parameters['arguments']+=['--existing-ic',{'input':primary},'--native-generation-report',{'input':pinned(ic['native_generation_report'])}]
            row['product_contracts']={product:{'reader':'native_snapshot_series','snapshots':option['snapshots']}}
        else:
            if ('request' in option)==('request_reference' in option):raise ValueError('Select one explicit inline or pinned standard scientific request')
            request=deepcopy(option['request']) if 'request' in option else pinned(option['request_reference'],load=True)
            if not isinstance(request,dict):raise ValueError('Standard scientific request must be an object')
            if option.get('producer',product)=='joint_response_training' and request.get('schema')!='sidm-joint-response-training-request-v2':
                raise ValueError('Standard joint-response training uses the explicit uncalibrated training request schema')
            if descriptor['kind'] not in {'physical_target_budget','physical_target_convergence'}:
                request.setdefault('memory_gib',spec['resources']['memory_gib'])
            for key,default in descriptor.get('request_defaults',{}).items():
                if key not in request:
                    request[key]=[{'science_product':name} for name in default] if isinstance(default,list) else {'science_product':default}
            parameters={'request':expand(request,dependencies),'request_base':str(base)}
            definition=registry.get(descriptor['kind'])
            if 'destination' in definition.required_parameters:
                output=descriptor['output'];parameters['destination']='{generation}/'+(str(Path(output).parent) if descriptor.get('directory_output') else output)
            for key in descriptor.get('extras',[]):
                if key=='memory_gib':parameters[key]=option.get(key,spec['resources']['memory_gib'])
                else:
                    if key not in option:raise ValueError('Standard producer requires explicit '+product+'/'+key)
                    parameters[key]=expand(option[key],dependencies)
            for key,default in descriptor.get('parameter_defaults',{}).items():
                parameters[key]=expand(option.get(key,{'science_product':default}),dependencies)
            if descriptor['native']:
                if option.get('native_source'):
                    if 'native_product' in option:raise ValueError('Select a completed native source or an automatic native producer')
                    native=option['native_source']
                    if not isinstance(native,dict) or set(native)!={'stage_receipt','run_directory'}:raise ValueError('Completed native source requires its pinned stage receipt and owned run directory')
                    parameters['native_stage_receipt']=pinned(native['stage_receipt'])
                    if not descriptor.get('receipt_only'):parameters['native_run_directory']=str(path_at(base,native['run_directory']))
                else:
                    native=option.get('native_product','native_evolution');dependencies.add(native)
                    parameters['native_stage_receipt']=reference(native,'receipt')
                    if not descriptor.get('receipt_only'):parameters['native_run_directory']=reference(native,'path')
        include=option.get('include_in_all',descriptor.get('include_in_all',True));species=option.get('species',['sidm'] if descriptor['kind'] in {'sidm_native_events','sidm_reference_cdf','sidm_angular_control'} else ['dm'])
        if type(include)is not bool or not isinstance(species,list) or len(set(species))!=len(species) or any(name not in {'dm','gas','stars','black_holes','sidm'} for name in species):
            raise ValueError('Standard product all-expansion needs exact include boolean and known distinct physical species')
        if descriptor['kind']=='physical_target_support':
            required=['gas'] if request.get('kind')=='first_stars' else ['dm','black_holes']
            species=list(dict.fromkeys([*species,*required]))
        if descriptor['kind']=='native_stream_worldlines':
            required={0:'gas',1:'dm',2:'dm',3:'dm',4:'stars'}.get(request.get('species'))
            if required is not None:species=list(dict.fromkeys([*species,required]))
        if descriptor['kind'] in {'twofluid_relative','twofluid_nonlinear_reference','twofluid_eulerian_seed','twofluid_particle_projection','twofluid_native_ic','twofluid_window_comparison'}:species=list(dict.fromkeys([*species,'gas']))
        if descriptor['kind']=='native_pilot_galaxy_measurement':species=list(dict.fromkeys([*species,'stars']))
        row.update(parameters=parameters,requires=sorted(dependencies),include_in_all=include,species=species)
        row['assets']=[{'path':path,'sha256':sources[path]} for path in sorted(current_assets)]
        row['standard_producer']=option.get('producer',product)
    for product in custom:
        dependencies=set(custom[product].get('requires',[]))
        catalogue[product]=expand(custom[product],dependencies,False)
        catalogue[product]['requires']=sorted(dependencies)
    return catalogue,sources


def compile_run_spec_v2(spec, *, registry=None, request_base=None):
    """Expand scientific products and their prerequisites into ordinary owned stages."""
    from copy import deepcopy
    from .foundation import validate_run_spec_v2
    canonical = validate_run_spec_v2(spec)
    registry = registry or default_registry()
    base = Path(request_base or '.').resolve()
    bundle = canonical['cosmology_bundle']
    bundle_path = path_at(base, bundle['path'])
    if file_sha256(bundle_path) != bundle['sha256']:
        raise ValueError('Version-2 specification linear bundle changed')
    catalogue,catalogue_sources = standard_product_recipes(canonical,request_base=base,registry=registry)
    selected = canonical['science']['products']
    if 'all' in selected:
        if selected != ['all']:
            raise ValueError('Version-2 all expansion must be selected explicitly on its own')
        if not canonical.get('native_config'):
            raise ValueError('Version-2 all expansion requires explicit native_config to establish available physical species')
        from core.research.capabilities import available_species
        species = available_species(canonical.get('native_config', {}) | {'dark_matter': canonical['dark_matter']})
        selected = [name for name, row in catalogue.items() if row.get('include_in_all', True)
            and set(row.get('species', [])) <= species]
    rows, order, active = {}, [], set()
    def substitute(value):
        if isinstance(value, dict):
            if set(value) == {'run_spec'}:
                keys = value['run_spec']
                if not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys):
                    raise ValueError('Exact canonical run-spec parameter path required')
                out = canonical
                for key in keys:
                    if not isinstance(out, dict) or key not in out:
                        raise ValueError('Canonical run-spec recipe parameter is missing: ' + '.'.join(keys))
                    out = out[key]
                return deepcopy(out)
            return {key: substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        return deepcopy(value)
    def visit(product):
        safe_id(product)
        if product in active:
            raise ValueError('Cyclic version-2 product prerequisites: ' + product)
        if product not in catalogue:
            raise ValueError('Requested scientific prerequisite has no recipe: ' + product)
        recipe = catalogue[product]
        row = substitute({key: value for key, value in recipe.items() if key != 'requires'})
        sid = safe_id(row.get('stage_id', product))
        row['stage_id'] = sid
        if sid in rows:
            if rows[sid]['recipe_hash'] != canonical_digest(recipe):
                raise ValueError('Shared product stage has conflicting definitions: ' + sid)
            return sid
        active.add(product)
        requires = recipe.get('requires', [])
        if not isinstance(requires, list) or len(set(requires)) != len(requires):
            raise ValueError('Distinct explicit product prerequisites required')
        dependencies = [visit(name) for name in requires]
        registry.get(row['kind'])
        if product not in row.get('outputs', {}):
            raise ValueError('Product recipe does not publish its named concrete output: ' + product)
        row['depends_on'] = list(dict.fromkeys(row.get('depends_on', []) + dependencies))
        row['assets'] = [*row.get('assets', []), {'path': str(bundle_path), 'sha256': bundle['sha256']}]
        if row['kind']=='public_command' and row['parameters'].get('command')=='gadget':
            row['canonical_run_constraints']={'epochs':canonical['epochs'],'dark_matter':canonical['dark_matter'],
                'cosmology_bundle':canonical['cosmology_bundle']}
            row['parameters']['canonical_run_constraints']=row['canonical_run_constraints']
        row['recipe_hash'] = canonical_digest(recipe)
        rows[sid] = row; order.append(sid); active.remove(product)
        if len(rows) > canonical['resources']['maximum_stage_count']:
            raise ValueError('Automatic product closure exceeds the explicit stage allocation')
        return sid
    for product in selected:
        visit(product)
    resources = {key: value for key, value in canonical['resources'].items() if key != 'followup_limits'}
    physics = {'dark_matter': canonical['dark_matter'], 'cosmology_model': bundle['background_model']}
    if 'combination_specification' in canonical:
        physics['combination_specification'] = canonical['combination_specification']
    native = canonical.get('native_config', {})
    if native:
        physics.update(physics=native.get('simulation', {}).get('physics'),
            hydro=native.get('hydro', {}), black_holes=native.get('black_holes', {}))
    return {'schema': 'research-workflow-v1', 'resources': resources,
        'followup_limits': canonical['resources']['followup_limits'], 'physics_spec': physics,
        'capabilities': canonical.get('capabilities', []), 'requested_products': selected,
        'stages': [rows[sid] for sid in order], 'canonical_run_spec': canonical,
        'canonical_run_spec_sha256': canonical_digest(canonical), 'linear_bundle_sha256': bundle['sha256'],
        'standard_catalogue_source_hashes':catalogue_sources,
        'producer_policy': 'maintained registered standard producers and explicit custom recipes; automatic complete prerequisite closure; missing scientific inputs fail before evolution'}


def compile_requested_product_dag(spec, registry=None, *, request_base=None):
    registry = registry or default_registry()
    if spec.get('schema_version') == 2:
        spec = compile_run_spec_v2(spec, registry=registry, request_base=request_base)
    if spec.get('schema') != 'research-workflow-v1':
        raise ValueError('Expected research-workflow-v1')
    resources = spec['resources']
    for key in ('mpi_ranks', 'omp_threads', 'memory_gib', 'disk_gib', 'maximum_stage_count'):
        positive_number(resources[key], key, key in {'mpi_ranks', 'omp_threads', 'maximum_stage_count'})
    limits = spec['followup_limits']
    for key in ('maximum_children', 'maximum_depth'):
        if type(limits[key]) is not int or limits[key] < 0:
            raise ValueError('Nonnegative follow-up bound required: ' + key)
    require_supported_combination(spec.get('physics_spec', {}), request_base=request_base)
    stages = {}
    for row in spec.get('stages', []):
        sid = safe_id(row['stage_id'])
        if sid in stages:
            raise ValueError('Duplicate stage ID: ' + sid)
        definition = registry.get(row['kind'])
        params = row.get('parameters', {})
        for key in definition.required_parameters:
            if key not in params:
                raise ValueError('Missing ' + sid + ' parameter ' + key)
        if definition.validator and not list(_stage_references(params)) and not list(_input_references(params)):
            definition.validator(params)
        outputs = row.get('outputs', {})
        if not outputs:
            raise ValueError('Stage must declare concrete products: ' + sid)
        
        validate_product_contract_definitions(row)
        for product, filename in outputs.items():
            safe_id(product)
            p = Path(filename)
            if p.is_absolute() or '..' in p.parts or filename in {'.', ''}:
                raise ValueError('Product path escapes generation')
        deps = list(dict.fromkeys(row.get('depends_on', []) + list(_stage_references(params)) + list(_stage_references(row.get('inputs', {})))))
        depth = row.get('depth', 0)
        if type(depth) is not int or depth < 0 or depth > limits['maximum_depth']:
            raise ValueError('Explicit stage follow-up depth is outside the declared bound')
        stages[sid] = {**row, 'depends_on': deps, 'depth': depth}
    for sid, row in stages.items():
        if sid in row['depends_on'] or any(d not in stages for d in row['depends_on']):
            raise ValueError('Unknown/self dependency: ' + sid)
        _check_product_references(row, stages)
    # Product-to-stage aliases are explicit; ambiguity is never resolved by order.
    producers = {}
    for sid, row in stages.items():
        for name in row['outputs']:
            if name in producers:
                raise ValueError('Multiple producers for requested product: ' + name)
            producers[name] = {'stage': sid, 'species': row.get('species', []),
                               'include_in_all': row.get('include_in_all', True)}
    wanted = expand_all_products(spec.get('requested_products', []), producers, spec.get('physics_spec', {}))
    if not wanted:
        raise ValueError('Explicit requested products required')
    needed = set()
    def visit(sid, stack):
        if sid in stack:
            raise ValueError('Cyclic product DAG: ' + ' -> '.join((*stack, sid)))
        if sid in needed:
            return
        for dep in stages[sid]['depends_on']:
            visit(dep, (*stack, sid))
        needed.add(sid)
    for name in wanted:
        visit(producers[name]['stage'], ())
    order = []
    remaining = set(needed)
    while remaining:
        ready = sorted(s for s in remaining if set(stages[s]['depends_on']) <= set(order))
        if not ready:
            raise ValueError('Cyclic stage graph')
        order.extend(ready)
        remaining.difference_update(ready)
    if len(order) > resources['maximum_stage_count']:
        raise ValueError('Initial product DAG exceeds stage budget')
    return {'schema': 'research-plan-v1', 'requested_products': wanted,
            'stages': [stages[s] for s in order], 'resources': resources, 'followup_limits': limits}


def _json_reference_field(record,reference):
    for key in reference.get('field',[]):
        if isinstance(record,dict) and isinstance(key,str) and key in record:record=record[key]
        elif isinstance(record,list) and type(key)is int and 0<=key<len(record):record=record[key]
        else:raise ValueError('Dependent scientific JSON field is absent or changes its type')
    return record


def _static_parameters(value, context, stages, stack=()):
    """Read frozen inputs and pure materialized requests without executing stages."""
    if isinstance(value, dict):
        if set(value) == {'json_input'}:
            path = path_at(context.base, value['json_input'])
            if path.stat().st_size*8 > context.resources['memory_gib']*1024**3:
                raise MemoryError('Preflight JSON input exceeds the declared workspace')
            return json.loads(path.read_text())
        if set(value) == {'input'}:
            return str(path_at(context.base, value['input']))
        if 'stage' in value and set(value) <= {'stage', 'artifact', 'load','field'}:
            producer = stages[value['stage']]
            if value.get('load') == 'json' and producer['kind'] == 'materialize_request':
                if value['stage'] in stack:
                    raise ValueError('Cyclic materialized preflight request')
                filename = producer['parameters'].get('filename', 'request.json')
                if value.get('artifact') == filename:
                    return _json_reference_field(_static_parameters(producer['parameters']['request'], context, stages, (*stack, value['stage'])),value)
            return value
        return {key:_static_parameters(item, context, stages, stack) for key,item in value.items()}
    if isinstance(value, list):
        return [_static_parameters(item, context, stages, stack) for item in value]
    if isinstance(value, str):
        return value.replace('{request_base}', str(context.base)).replace('{python}', sys.executable).replace('{suite}', str(ROOT))
    return value


def _validate_numerical_domains(row, parameters):
    """Apply recorded parameter units/ranges at preflight and again at dispatch."""
    import math
    for contract in row.get('numerical_domains', []):
        path = contract['parameter_path']
        if not isinstance(path, list) or not path or any(not isinstance(part, str) or not part for part in path):
            raise ValueError('Numerical domain requires an exact nested parameter path')
        value = parameters
        for part in path:
            if not isinstance(value, dict) or part not in value:
                raise ValueError('Stage omits a declared numerical-domain parameter: '+'.'.join(path))
            value = value[part]
        if list(_stage_references(value)):
            raise ValueError('Numerical-domain values must be frozen before upstream evolution')
        if not isinstance(contract.get('units'), str) or not contract['units']:
            raise ValueError('Numerical-domain units required: '+'.'.join(path))
        if 'allowed' in contract:
            if not isinstance(contract['allowed'], list) or not contract['allowed'] or value not in contract['allowed']:
                raise ValueError('Stage parameter is outside its declared categorical domain: '+'.'.join(path))
        else:
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                raise ValueError('Stage numerical-domain parameter must be finite: '+'.'.join(path))
            if not any(key in contract for key in ('minimum','maximum')):
                raise ValueError('Stage numerical-domain requires a finite bound')
            for key in ('minimum','maximum'):
                if key in contract and (isinstance(contract[key], bool) or not isinstance(contract[key], (int,float)) or not math.isfinite(contract[key])):
                    raise ValueError('Stage numerical-domain bounds must be finite')
            if contract.get('minimum', value) > contract.get('maximum', value):
                raise ValueError('Stage numerical-domain interval is reversed')
            if value < contract.get('minimum', value) or value > contract.get('maximum', value):
                raise ValueError('Stage parameter is outside its declared numerical domain: '+'.'.join(path))


def resolve_values(value, context):
    if isinstance(value, dict):
        if set(value) <= {'stage', 'artifact', 'load','field'} and 'stage' in value:
            receipt = context.completed[value['stage']]
            root = Path(receipt['generation'])
            q = (root/value.get('artifact', 'stage-receipt.json')).resolve()
            if not q.is_relative_to(root) or not q.exists():
                raise ValueError('Missing or escaped dependent artifact')
            if value.get('load') == 'json':
                if not q.is_file() or q.stat().st_size*6 > context.resources['memory_gib']*1024**3:
                    raise MemoryError('Dependent JSON product exceeds declared input workspace')
                return _json_reference_field(json.loads(q.read_text()),value)
            if value.get('load') == 'sha256':
                if not q.is_file():raise ValueError('Dependent SHA256 reference must name a retained regular file')
                return file_sha256(q)
            if value.get('load') is not None:
                raise ValueError('Unknown dependent product loading mode')
            return str(q)
        if set(value) == {'input'}:
            return str(path_at(context.base, value['input']))
        if set(value) == {'json_input'}:
            path = path_at(context.base, value['json_input'])
            if path.stat().st_size*6 > context.resources['memory_gib']*1024**3:
                raise MemoryError('Serialized stage input exceeds declared workspace')
            return json.loads(path.read_text())
        return {key: resolve_values(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_values(item, context) for item in value]
    if isinstance(value, str):
        return value.replace('{generation}', str(context.generation)).replace('{request_base}', str(context.base)).replace('{python}', sys.executable).replace('{suite}', str(ROOT))
    return value


def verify_completed_receipt(path):
    path = Path(path)
    if file_sha256(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Workflow receipt seal changed')
    record = json.loads(path.read_text())
    if record.get('execution_status') != 'COMPLETE':
        raise ValueError('Incomplete generation cannot be resumed')
    generation = Path(record['generation']).resolve()
    if record.get('schema') != 'research-stage-receipt-v1' or path.resolve().parent != generation:
        raise ValueError('Stage receipt location does not match its declared immutable generation')
    for relative, expected in record['files_sha256'].items():
        p = (generation/relative).resolve()
        if not p.is_relative_to(generation) or file_sha256(p) != expected:
            raise ValueError('Stage payload missing or changed: ' + relative)
    if tree_hashes(generation) != record['files_sha256']:
        raise ValueError('Published generation gained or lost payload files')
    for p, expected in record['source_hashes'].items():
        if file_sha256(p) != expected:
            raise ValueError('Stage implementation changed: ' + p)
    for contract in record.get('product_contract_receipts',{}).values():
        for p, expected in contract['source_hashes'].items():
            if file_sha256(p) != expected:
                raise ValueError('Scientific product intake source changed: '+p)
    for p, expected in record['input_hashes'].items():
        if file_sha256(p) != expected:
            raise ValueError('Stage input changed: ' + p)
    return record

@dataclass
class WorkflowContext:
    destination: Path
    base: Path
    resources: dict
    capabilities: CapabilityRegistry
    completed: dict
    generation: Path | None = None
    resume_checkpoint: dict | None = None
    stage_id: str | None = None


def preflight_plan(plan, spec, context, registry):
    """Preflight every selected producer before evolving any native particles."""
    stages = {row['stage_id']:row for row in plan['stages']}
    for row in plan['stages']:
        definition = registry.get(row['kind'])
        parameters = _static_parameters(row.get('parameters', {}), context, stages)
        _validate_numerical_domains(row, parameters)
        if definition.validator and not list(_stage_references(parameters)):
            definition.validator(parameters)
        names = tuple(dict.fromkeys(definition.capabilities + tuple(row.get('capabilities', []))))
        context.capabilities.require(names, row.get('domain_parameters', {}), row.get('required_features', []))
        r = row.get('resources', context.resources)
        for key in ('mpi_ranks', 'omp_threads', 'memory_gib', 'disk_gib'):
            value = positive_number(r[key], key, key in {'mpi_ranks', 'omp_threads'})
            if value > context.resources[key]:
                raise ValueError('Stage requires more resources; configuration unchanged: ' + row['stage_id'] + '/' + key)
        if row['kind'] in {'command','public_command'}:
            from core.resources import validate_sampling_options
            validate_sampling_options(row.get('parameters',{}).get('resource_sampling',{}))
        if row['kind']=='public_command':
            from core.research.checkpoints import validate_evolution_checkpoint_policy
            validate_evolution_checkpoint_policy(row)
            from core.research.capabilities import preflight_public_command_stage
            from core.music.staging import preflight_generated_flow_evolution
            if not preflight_generated_flow_evolution(row,plan,context):
                preflight_public_command_stage(row,plan,context)
        if row['kind'] in {'twofluid_relative','twofluid_nonlinear_reference'}:
            from .initial_conditions import twofluid_relative_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:
                raise MemoryError('Two-fluid relative fields exceed allocated stage memory')
            if not list(_stage_references(parameters)):
                twofluid_relative_input_bindings(parameters)
        if row['kind']=='twofluid_eulerian_seed':
            from .initial_conditions import twofluid_eulerian_seed_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Eulerian seed exceeds allocated memory')
            if not list(_stage_references(parameters)):twofluid_eulerian_seed_input_bindings(parameters)
        if row['kind']=='dm_wake_force':
            from .black_holes import dm_wake_force_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Wake force exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_force_input_bindings(parameters)
        if row['kind']=='dm_wake_kernel_comparison':
            from .black_holes import dm_wake_kernel_comparison_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native kernel comparison exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_kernel_comparison_input_bindings(parameters)
        if row['kind']=='dm_wake_contrast':
            from .black_holes import dm_wake_contrast_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Wake contrast exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_contrast_input_bindings(parameters)
        if row['kind']=='dm_wake_ensemble':
            from .black_holes import dm_wake_ensemble_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Wake ensemble exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_ensemble_input_bindings(parameters)
        if row['kind']=='dm_wake_response':
            from .black_holes import dm_wake_response_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Wake response exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_response_input_bindings(parameters)
        if row['kind']=='dm_wake_state_response':
            from .black_holes import dm_wake_state_response_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('State response exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_state_response_input_bindings(parameters)
        if row['kind']=='dm_wake_state_table':
            from .black_holes import dm_wake_state_table_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('State table exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_state_table_input_bindings(parameters)
        if row['kind']=='dm_wake_context':
            from .black_holes import dm_wake_context_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native context exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_context_input_bindings(parameters)
        if row['kind']=='twofluid_nonlinear_refinement':
            from .initial_conditions import twofluid_refinement_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Nonlinear refinement exceeds allocation')
            if not list(_stage_references(parameters)):twofluid_refinement_input_bindings(parameters)
        if row['kind']=='twofluid_projection_refinement':
            from .initial_conditions import twofluid_projection_refinement_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Projection refinement exceeds allocation')
            if not list(_stage_references(parameters)):twofluid_projection_refinement_input_bindings(parameters)
        if row['kind']=='twofluid_refinement_campaign':
            from .initial_conditions import twofluid_refinement_campaign_input_bindings
            if parameters['request']['resources']['memory_gib']>r['memory_gib']:raise MemoryError('Nonlinear campaign exceeds allocated memory')
            if not list(_stage_references(parameters)):twofluid_refinement_campaign_input_bindings(parameters)
        if row['kind']=='dm_wake_native_table':
            from .black_holes import dm_wake_native_table_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native wake table exceeds allocated memory')
            if not list(_stage_references(parameters)):dm_wake_native_table_input_bindings(parameters)
        if row['kind']=='native_stream_worldlines':
            from .stream_forward_model import native_stream_worldline_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native stream measurement exceeds allocated memory')
            if not list(_stage_references(parameters)):native_stream_worldline_input_bindings(parameters)
        if row['kind']=='native_stream_comparison':
            from .stream_forward_model import native_stream_comparison_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native stream comparison exceeds allocated memory')
            if not list(_stage_references(parameters)):native_stream_comparison_input_bindings(parameters)
        if row['kind']=='native_late_threeway':
            from .stream_forward_model import native_late_threeway_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Late three-way comparison exceeds allocated memory')
            if not list(_stage_references(parameters)):native_late_threeway_input_bindings(parameters)
        if row['kind']=='twofluid_native_comparison':
            from .initial_conditions import twofluid_native_comparison_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native two-fluid comparison exceeds allocated memory')
            if not list(_stage_references(parameters)):twofluid_native_comparison_input_bindings(parameters)
        if row['kind']=='twofluid_native_family':
            from .initial_conditions import twofluid_native_family_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native grid-family measurement exceeds allocated memory')
            if not list(_stage_references(parameters)):twofluid_native_family_input_bindings(parameters)
        if row['kind']=='native_counterfactual_result':
            from .stream_forward_model import native_counterfactual_result_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native result exceeds allocated memory')
            if not list(_stage_references(parameters)):native_counterfactual_result_input_bindings(parameters)
        if row['kind']=='combined_graph_gallery':
            from core.research.display import gallery_inputs
            for key in ('memory_gib','disk_gib'):
                if parameters['request'][key]>r[key]:raise MemoryError('Combined gallery exceeds allocated '+key)
            if not list(_stage_references(parameters)):gallery_inputs(parameters['request'])
        if row['kind']=='native_counterfactual_ic':
            from .stream_forward_model import native_counterfactual_ic_input_bindings
            for key in ('memory_gib','disk_gib'):
                if parameters['request'][key]>r[key]:raise MemoryError('Native intervention exceeds allocated '+key)
            if not list(_stage_references(parameters)):native_counterfactual_ic_input_bindings(parameters)
        if row['kind']=='native_counterfactual_run':
            from .stream_forward_model import native_counterfactual_run_input_bindings
            for key in ('mpi_ranks','omp_threads'):
                if parameters['request'][key]!=r[key]:raise ValueError('Native counterfactual changes allocated '+key)
            for key in ('memory_gib','disk_gib'):
                if parameters['request'][key]>r[key]:raise MemoryError('Native counterfactual exceeds allocated '+key)
            if not list(_stage_references(parameters)):native_counterfactual_run_input_bindings(parameters)
        if row['kind']=='native_counterfactual_campaign':
            from .stream_forward_model import native_counterfactual_campaign_input_bindings
            allocation=parameters['request']['resources']
            for key in ('mpi_ranks','omp_threads'):
                if allocation[key]!=r[key]:raise ValueError('Native campaign changes allocated '+key)
            for key in ('memory_gib','disk_gib','maximum_stage_count'):
                if allocation[key]>r[key]:raise MemoryError('Native campaign exceeds allocated '+key)
            if not list(_stage_references(parameters)):native_counterfactual_campaign_input_bindings(parameters)
        if row['kind']=='sidm_mapped_campaign_plan':
            from .sidm_control_products import validate_mapped_campaign_plan_request,mapped_campaign_plan_input_bindings
            request=validate_mapped_campaign_plan_request(parameters);allocation=request['resources']
            for key in ('mpi_ranks','omp_threads'):
                if allocation[key]!=r[key]:raise ValueError('Mapped planner changes allocated '+key)
            for key in ('memory_gib','disk_gib','maximum_stage_count'):
                if allocation[key]>r[key]:raise MemoryError('Mapped planner exceeds allocated '+key)
            if not list(_stage_references(parameters)):mapped_campaign_plan_input_bindings(parameters)
        if row['kind']=='sidm_control_population':
            from .sidm_control_products import validate_control_population_request
            validate_control_population_request(parameters)
            for name in ('memory_gib','disk_gib'):
                if parameters['request'][name]>r[name]:raise MemoryError('SIDM control population exceeds allocated '+name)
        if row['kind']=='sidm_canonical_population':
            from .sidm_control_products import canonical_population_input_bindings
            for name in ('memory_gib','disk_gib'):
                if parameters['request'][name]>r[name]:raise MemoryError('Canonical population exceeds allocated '+name)
            if not list(_stage_references(parameters)):canonical_population_input_bindings(parameters)
        if row['kind']=='sidm_homogeneous_run':
            from .sidm_control_products import control_run_input_bindings
            for name in ('mpi_ranks','omp_threads'):
                if parameters['request'][name]!=r[name]:raise ValueError('Homogeneous native run changes '+name)
            for name in ('memory_gib','disk_gib'):
                if parameters['request'][name]>r[name]:raise MemoryError('Homogeneous native run exceeds '+name)
            if not list(_stage_references(parameters)):control_run_input_bindings(parameters)
        if row['kind']=='sidm_reference_build':
            from .sidm_control_products import reference_build_input_bindings
            if parameters['request']['omp_threads']!=r['omp_threads'] or parameters['request']['jobs']>r['mpi_ranks']*r['omp_threads']:
                raise ValueError('Independent compile exceeds workflow CPU allocation')
            for name in ('memory_gib','disk_gib'):
                if parameters['request'][name]>r[name]:raise MemoryError('Independent compile exceeds allocated '+name)
            if not list(_stage_references(parameters)):reference_build_input_bindings(parameters)
        if row['kind'] in {'sidm_control_history','sidm_control_comparison'}:
            from .sidm_control_measurements import control_stage_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Physical control stage exceeds allocated memory')
            if not list(_stage_references(parameters)):control_stage_input_bindings(row['kind'],parameters)
        if row['kind']=='sidm_control_clock_mapping':
            from .sidm_control_measurements import control_clock_mapping_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Canonical clock mapping exceeds allocated memory')
            if not list(_stage_references(parameters)):control_clock_mapping_input_bindings(parameters)
        if row['kind']=='sidm_control_native_clock':
            from .sidm_control_measurements import control_native_clock_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native canonical clock exceeds allocated memory')
            if not list(_stage_references(parameters)):control_native_clock_input_bindings(parameters)
        if row['kind']=='sidm_control_ensemble':
            from .sidm_control_measurements import control_ensemble_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Paired native ensemble exceeds allocation')
            if not list(_stage_references(parameters)):control_ensemble_input_bindings(parameters)
        if row['kind']=='sidm_control_refinement_measurement':
            from .sidm_control_measurements import control_refinement_measurement_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native refinement measurement exceeds allocation')
            if not list(_stage_references(parameters)):control_refinement_measurement_input_bindings(parameters)
        if row['kind']=='sidm_control_expectation':
            from .sidm_control_measurements import control_expectation_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Physical expectation stage exceeds allocation')
            if not list(_stage_references(parameters)):control_expectation_input_bindings(parameters)
        if row['kind']=='twofluid_particle_projection':
            from .initial_conditions import twofluid_particle_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Native two-fluid particle projection exceeds allocated workspace')
            if not list(_stage_references(parameters)):twofluid_particle_input_bindings(parameters)
        if row['kind']=='twofluid_native_ic':
            from .initial_conditions import twofluid_native_ic_input_bindings
            for name in ('memory_gib','disk_gib'):
                if parameters['request'][name]>r[name]:raise MemoryError('Native IC writer exceeds allocated '+name)
            if not list(_stage_references(parameters)):twofluid_native_ic_input_bindings(parameters)
        if row['kind']=='twofluid_window_comparison':
            from .initial_conditions import twofluid_window_input_bindings
            if parameters['request']['memory_gib']>r['memory_gib']:raise MemoryError('Material-window comparison exceeds allocated stage memory')
            if not list(_stage_references(parameters)):twofluid_window_input_bindings(parameters)
        if row['kind'] in {'physical_target_support','physical_target_model_control','physical_target_measurement','physical_target_budget','physical_target_convergence'}:
            from .zoom import physical_target_stage_input_bindings
            memory=parameters.get('memory_gib',parameters['request'].get('memory_gib',parameters['request'].get('refinement',{}).get('memory_gib')))
            if memory>r['memory_gib']:raise MemoryError('Physical target producer exceeds its allocated stage workspace')
            if not list(_stage_references(parameters)):
                physical_target_stage_input_bindings(row['kind'],parameters)
        if row['kind'] in {'sidm_history','sidm_matched','sidm_observations','sidm_membership','sidm_satellite','sidm_derived','native_forward_vector','sidm_joint_response','sidm_joint_calibration','sidm_calibrated_response'}:
            from .sidm_analysis import sidm_input_bindings
            if parameters['request']['memory_gib'] > r['memory_gib']:
                raise MemoryError('SIDM analysis exceeds its allocated stage workspace')
            if not list(_stage_references(parameters)):
                sidm_input_bindings(parameters)
        if row['kind'] == 'finding_repair_receipt':
            from core.research.findings import repair_input_bindings
            if parameters['memory_bytes'] > r['memory_gib']*1024**3:
                raise MemoryError('Finding repair evidence exceeds the allocated stage workspace')
            if not list(_stage_references(parameters)):
                repair_input_bindings(parameters)
        if row['kind'] == 'native_pilot_galaxy_measurement':
            from core.research.pilots import galaxy_measurement_input_bindings
            if parameters['request']['memory_gib'] > r['memory_gib']:
                raise MemoryError('Native pilot galaxy measurement exceeds its allocated stage workspace')
            if not list(_stage_references(parameters)):
                galaxy_measurement_input_bindings(parameters)
        if row['kind']=='flagship_production_admission':
            from .flagship_production_campaigns import validate_production_admission
            validate_production_admission(parameters,output_parent=context.destination)
        if row['kind'] == 'zoom_campaign':
            from core.research.zooms import preflight_zoom_campaign
            if list(_stage_references(parameters['request'])):
                raise ValueError('Zoom campaign physical request must be frozen before native evolution')
            preflight_zoom_campaign(parameters['request'], request_base=parameters.get('request_base', context.base), allocated_resources=r)
        if row['kind'] == 'native_zoom_diagnostics':
            from core.research.zooms import zoom_diagnostic_input_bindings
            if parameters['request']['memory_gib'] > r['memory_gib']:
                raise MemoryError('Native zoom history diagnostics exceed their allocated stage workspace')
            if not list(_stage_references(parameters)):
                zoom_diagnostic_input_bindings(parameters)
        if row['kind'] == 'public_command' and row.get('parameters', {}).get('command') == 'gadget':
            arguments = row['parameters'].get('arguments', [])
            if '--answers' not in arguments:
                raise ValueError('Noninteractive evolution requires explicit --answers')
            position = arguments.index('--answers')+1
            if position >= len(arguments):
                raise ValueError('Missing native answer file argument')
            reference = arguments[position]
            if isinstance(reference, dict) and set(reference) == {'input'}:
                answer_path = path_at(context.base, reference['input'])
                answers = json.loads(answer_path.read_text())
                for key, resource_key in (('mpi_ranks','mpi_ranks'), ('omp_threads','omp_threads'), ('memory_limit_gb','memory_gib')):
                    if key not in answers or answers[key] != r[resource_key]:
                        raise ValueError('Native answers must explicitly match allocated resources: '+key)
            elif not isinstance(reference, dict) or 'stage' not in reference:
                raise ValueError('Native --answers must use a hashed input or produced stage reference')
        for asset in row.get('assets', []):
            p = path_at(context.base, asset['path'])
            if file_sha256(p) != asset['sha256']:
                raise ValueError('Missing/changed stage asset: ' + str(p))
        if row['kind'] in {'isolated_control_history','sidm_angular_control','sidm_reference_cdf','sidm_native_events'}:
            if row['parameters']['request']['memory_gib'] > r['memory_gib']:
                raise MemoryError('Native SIDM control exceeds allocated stage workspace')
        if row['kind'] == 'convergence_measurement':
            from core.research.measurements import validate_measurement_request
            request = validate_measurement_request(row['parameters']['request'])
            if request['memory_gib'] > r['memory_gib']:
                raise MemoryError('Native measurement exceeds the allocated stage workspace')
        if row['kind'] == 'isolated_physical_collision':
            from core.research.controls import collision_input_bindings
            params=row['parameters'];request=params['request']
            if request['memory_gib']>r['memory_gib'] or request['mpi_ranks']!=r['mpi_ranks'] or request['omp_threads']!=r['omp_threads']:
                raise ValueError('Physical collision stage cannot alter workflow resource allocation')
            if not list(_stage_references(params)):
                collision_input_bindings(resolve_values(params,context))
        if row['kind'] == 'native_sidm_resources':
            from .sidm_campaign import sidm_resource_input_bindings
            params=row['parameters']
            if params['request']['memory_gib']>r['memory_gib']:
                raise MemoryError('Native resource telemetry exceeds stage allocation')
            if not list(_stage_references(params)):
                sidm_resource_input_bindings(resolve_values(params,context))
        if row['kind'] == 'native_sidm_zoom_support':
            from .apertures import zoom_support_input_bindings
            params=row['parameters']
            if params['request']['memory_gib']>r['memory_gib']:
                raise MemoryError('Native support workspace exceeds the declared stage allocation')
            if not list(_stage_references(params)):
                zoom_support_input_bindings(resolve_values(params,context))
        if row['kind'] == 'isolated_observable':
            from core.research.measurements import isolated_observable_input_bindings
            params = row['parameters']
            if params['memory_bytes'] > r['memory_gib']*1024**3:
                raise MemoryError('Saved-history observable exceeds allocated workspace')
            if not list(_stage_references(params)):
                isolated_observable_input_bindings(resolve_values(params,context))
        if row['kind'] == 'native_error_envelope':
            from core.research.errors import envelope_input_bindings
            params = row['parameters']
            if params['memory_bytes'] > r['memory_gib']*1024**3:
                raise MemoryError('Native envelope exceeds the allocated stage workspace')
            if not list(_stage_references(params)):
                envelope_input_bindings(resolve_values(params,context))
        if row['kind'] == 'native_control_residual':
            params = row['parameters']
            memory = positive_number(params['memory_bytes'],'residual workspace',integer=True)
            positive_number(params['maximum_components'],'residual component bound',integer=True)
            if memory > r['memory_gib']*1024**3:
                raise MemoryError('Residual limits exceed allocated memory')
            if not list(_stage_references(params)):
                from core.research.errors import residual_input_bindings
                residual_input_bindings(resolve_values(params,context))
        if row['kind'] == 'error_ensemble':
            params = row['parameters']
            memory = positive_number(params['memory_bytes'],'ensemble workspace',integer=True)
            positive_number(params['maximum_realizations'],'ensemble realization limit',integer=True)
            if params['maximum_realizations'] < 2 or memory > r['memory_gib']*1024**3:
                raise MemoryError('Ensemble limits exceed the allocated memory or omit independent realizations')
            if not list(_stage_references(params)):
                from core.research.errors import ensemble_input_bindings
                resolved = resolve_values(params,context)
                ensemble_input_bindings(resolved['realizations'],resolved['systematic_controls'],
                    request_base=resolved['request_base'],memory_bytes=memory)
        if row['kind'] == 'display_products':
            ceiling = positive_number(row['parameters']['memory_gib'],'display preflight memory')
            if ceiling > r['memory_gib']:
                raise MemoryError('Display stage exceeds allocated memory; settings unchanged')
            # A future produced request cannot be inspected before its producer
            # runs. Its explicit ceiling is still preflighted before evolution;
            # actual request/bundle bindings are captured before display execution.
            if not list(_stage_references(row['parameters']['request'])):
                from core.research.display import display_request_input_bindings
                params = resolve_values(row['parameters'],context)
                display_request_input_bindings(params['request'],
                    request_base=params['request_base'],memory_gib=ceiling)
        if row['kind'] == 'error_budget':
            from core.research.errors import preflight_refinement_routes
            preflight_refinement_routes(row.get('parameters', {}), context, registry)
        for value in _input_references(row.get('parameters', {})):
            path = path_at(context.base, value)
            if not path.is_file():
                raise FileNotFoundError('Missing stage input before evolution: '+str(path))
        for item in row.get('input_files', []):
            p = path_at(context.base, item['path'])
            if file_sha256(p) != item['sha256']:
                raise ValueError('Missing/changed stage input: ' + str(p))
    if any(row['kind'] == 'native_control_configuration' for row in plan['stages']):
        from core.research.convergence import preflight_configuration_stages
        plan['native_configuration_preflight'] = preflight_configuration_stages(plan,context)
    if spec.get('production_resource_spec'):
        from core.resources import estimate_production_resources, require_resources
        resource_plan = estimate_production_resources(spec['production_resource_spec'])
        require_resources(resource_plan, context.resources, context.destination,
                          available_memory_bytes=spec.get('available_memory_bytes'),
                          available_cpu_count=spec.get('available_cpu_count'))
        plan['production_resource_plan'] = resource_plan
    return plan


def _input_hashes(row, context):
    result = {}
    if row['kind']=='flagship_runtime_bao':
        from .flagship_analysis_profile import runtime_bao_inputs
        result.update(runtime_bao_inputs(resolve_values(row['parameters'],context))[1])
    if row['kind']=='native_branch_products':
        from .flagship_native_branches import native_branch_inputs
        result.update(native_branch_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='flagship_retained_operation':
        from .flagship_retained import retained_operation_inputs
        result.update(retained_operation_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='flagship_runtime_measurements':
        from .flagship_runtime_measurements import measurement_preparation_inputs
        result.update(measurement_preparation_inputs(resolve_values(row['parameters'],context))[1])
    if row['kind']=='flagship_runtime_display':
        from .flagship_runtime_display import display_preparation_inputs
        result.update(display_preparation_inputs(resolve_values(row['parameters'],context))[1])
    if row['kind']=='flagship_parent_preparation':
        from .flagship_parent_preparation import parent_preparation_inputs
        result.update(parent_preparation_inputs(resolve_values(row['parameters'],context)))
    if row['kind']=='flagship_zoom_preparation':
        from .flagship_zoom_preparation import zoom_preparation_inputs
        result.update(zoom_preparation_inputs(resolve_values(row['parameters'],context))[0])
    if row['kind']=='flagship_runtime_history':
        from .flagship_runtime_history import runtime_history_inputs
        result.update(runtime_history_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='native_paired_stream_responses':
        from .response_campaign import paired_response_input_bindings
        result.update(paired_response_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='recovered_native_encounters':
        from .stream.native_encounters import recovered_encounter_input_bindings
        result.update(recovered_encounter_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_encounter_refinement':
        from .stream.native_encounters import refinement_input_bindings
        result.update(refinement_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_encounter_display':
        from .stream.native_encounters import encounter_display_input_bindings
        result.update(encounter_display_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_perturber_history':
        from .stream.perturbers import perturber_history_input_bindings
        result.update(perturber_history_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_stream_responses':
        from .native_stream_responses import native_response_input_bindings
        result.update(native_response_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='matched_perturber_figures':
        from .matched_perturber_figures import matched_template_input_bindings
        result.update(matched_template_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='public_command':
        from core.music.flow import native_flow_public_command_bindings
        result.update(native_flow_public_command_bindings(resolve_values(row['parameters'],context)))
        from core.response_campaign import public_ic_command_bindings
        result.update(public_ic_command_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_recovered_sidm_pair':
        from .sidm_recovery import recovered_sidm_pair_inputs
        result.update(recovered_sidm_pair_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_recovered_sidm_satellite':
        from .sidm_recovery import satellite_inputs
        result.update(satellite_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_recovered_sidm_centroids':
        from .sidm_recovery import recovered_centroid_inputs
        result.update(recovered_centroid_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_recovered_sidm_lensing':
        from .sidm_forward import lensing_inputs
        result.update(lensing_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_recovered_sidm_stream_response':
        from .sidm_forward import stream_inputs
        result.update(stream_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_public_ic':
        from core.response_campaign import public_ic_inputs
        result.update(public_ic_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_response_target':
        from .response_campaign import response_target_inputs
        result.update(response_target_inputs(resolve_values(row['parameters'],context))[3])
    if row['kind']=='combined_graph_gallery':
        from core.research.display import gallery_inputs
        result.update(gallery_inputs(resolve_values(row['parameters'],context)['request'])[1])
    if row['kind']=='hubble_figures':
        from .expansion_products import hubble_figure_arrays
        parameters=resolve_values(row['parameters'],context)
        result.update(hubble_figure_arrays(parameters['request'],memory_gib=parameters['memory_gib'])[-1])
    if row['kind']=='inference_figures':
        from .flagship_inference import inference_figure_arrays
        parameters=resolve_values(row['parameters'],context)
        result.update(inference_figure_arrays(parameters['request'],memory_gib=parameters['memory_gib'])[-1])
    if row['kind']=='native_recovered_sidm':
        from .sidm_recovery import recovered_sidm_inputs
        result.update(recovered_sidm_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_sidm_export':
        from .sidm_products import sidm_export_inputs
        result.update(sidm_export_inputs(resolve_values(row['parameters'],context)))
    if row['kind']=='native_web_epoch':
        from .halo_campaign import native_web_epoch_inputs
        result.update(native_web_epoch_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='native_web_history':
        from .halo_campaign import native_web_history_inputs
        result.update(native_web_history_inputs(resolve_values(row['parameters'],context)))
    if row['kind']=='native_baryon_figures':
        from .native_baryon_campaign import baryon_figure_inputs
        result.update(baryon_figure_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='native_cooling_births':
        from .native_baryon_campaign import cooling_birth_inputs
        result.update(cooling_birth_inputs(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_bh_history':
        from .native_bh_campaign import bh_history_inputs
        result.update(bh_history_inputs(resolve_values(row['parameters'],context)))
    if row['kind']=='native_bh_event_figures':
        from .native_bh_campaign import bh_event_stage_inputs
        result.update(bh_event_stage_inputs(resolve_values(row['parameters'],context)))
    if row['kind']=='native_bh_matched_response':
        from .native_bh_campaign import bh_matched_stage_inputs
        result.update(bh_matched_stage_inputs(resolve_values(row['parameters'],context)))
    if row['kind']=='native_bh_response_covariance':
        from .flagship_bh_response import covariance_input_bindings
        result.update(covariance_input_bindings(resolve_values(row['parameters'],context))[2])
    if row['kind']=='native_assembly_history':
        from .halo_campaign import native_assembly_history_inputs
        result.update(native_assembly_history_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='native_assembly_measurements':
        from .halo_campaign import assembly_measurement_inputs
        result.update(assembly_measurement_inputs(resolve_values(row['parameters'],context))[2])
    if row['kind']=='flagship_product':
        from .flagship_products import flagship_product_input_bindings
        result.update(flagship_product_input_bindings(resolve_values(row['parameters'],context)))
    for dep in row['depends_on']:
        old = context.completed[dep]
        p = Path(old['generation'])/'stage-receipt.json'
        verify_completed_receipt(p)
        result[str(p)] = file_sha256(p)
        for name, expected in old['files_sha256'].items():
            result[str(Path(old['generation'])/name)] = expected
    if row['kind'] == 'native_control_configuration':
        from core.research.convergence import configuration_input_bindings
        result.update(configuration_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] in {'twofluid_relative','twofluid_nonlinear_reference'}:
        from .initial_conditions import twofluid_relative_input_bindings
        result.update(twofluid_relative_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_eulerian_seed':
        from .initial_conditions import twofluid_eulerian_seed_input_bindings
        result.update(twofluid_eulerian_seed_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_force':
        from .black_holes import dm_wake_force_input_bindings
        result.update(dm_wake_force_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_kernel_comparison':
        from .black_holes import dm_wake_kernel_comparison_input_bindings
        result.update(dm_wake_kernel_comparison_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_contrast':
        from .black_holes import dm_wake_contrast_input_bindings
        result.update(dm_wake_contrast_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_ensemble':
        from .black_holes import dm_wake_ensemble_input_bindings
        result.update(dm_wake_ensemble_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_response':
        from .black_holes import dm_wake_response_input_bindings
        result.update(dm_wake_response_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_state_response':
        from .black_holes import dm_wake_state_response_input_bindings
        result.update(dm_wake_state_response_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_state_table':
        from .black_holes import dm_wake_state_table_input_bindings
        result.update(dm_wake_state_table_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_context':
        from .black_holes import dm_wake_context_input_bindings
        result.update(dm_wake_context_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_nonlinear_refinement':
        from .initial_conditions import twofluid_refinement_input_bindings
        result.update(twofluid_refinement_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_projection_refinement':
        from .initial_conditions import twofluid_projection_refinement_input_bindings
        result.update(twofluid_projection_refinement_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_refinement_campaign':
        from .initial_conditions import twofluid_refinement_campaign_input_bindings
        result.update(twofluid_refinement_campaign_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='dm_wake_native_table':
        from .black_holes import dm_wake_native_table_input_bindings
        result.update(dm_wake_native_table_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_stream_worldlines':
        from .stream_forward_model import native_stream_worldline_input_bindings
        result.update(native_stream_worldline_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_stream_comparison':
        from .stream_forward_model import native_stream_comparison_input_bindings
        result.update(native_stream_comparison_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_late_threeway':
        from .stream_forward_model import native_late_threeway_input_bindings
        result.update(native_late_threeway_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_native_comparison':
        from .initial_conditions import twofluid_native_comparison_input_bindings
        result.update(twofluid_native_comparison_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_native_family':
        from .initial_conditions import twofluid_native_family_input_bindings
        result.update(twofluid_native_family_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_counterfactual_result':
        from .stream_forward_model import native_counterfactual_result_input_bindings
        result.update(native_counterfactual_result_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_counterfactual_ic':
        from .stream_forward_model import native_counterfactual_ic_input_bindings
        result.update(native_counterfactual_ic_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='native_counterfactual_run':
        from .stream_forward_model import native_counterfactual_run_input_bindings
        result.update(native_counterfactual_run_input_bindings(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='native_counterfactual_campaign':
        from .stream_forward_model import native_counterfactual_campaign_input_bindings
        result.update(native_counterfactual_campaign_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] in {'sidm_control_history','sidm_control_comparison'}:
        from .sidm_control_measurements import control_stage_input_bindings
        result.update(control_stage_input_bindings(row['kind'],resolve_values(row['parameters'],context)))
    if row['kind']=='sidm_control_clock_mapping':
        from .sidm_control_measurements import control_clock_mapping_input_bindings
        result.update(control_clock_mapping_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='sidm_control_native_clock':
        from .sidm_control_measurements import control_native_clock_input_bindings
        result.update(control_native_clock_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='sidm_mapped_campaign_plan':
        from .sidm_control_products import mapped_campaign_plan_input_bindings
        result.update(mapped_campaign_plan_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='sidm_homogeneous_run':
        from .sidm_control_products import control_run_input_bindings
        result.update(control_run_input_bindings(resolve_values(row['parameters'],context))[2])
    if row['kind']=='sidm_canonical_population':
        from .sidm_control_products import canonical_population_input_bindings
        result.update(canonical_population_input_bindings(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='sidm_reference_build':
        from .sidm_control_products import reference_build_input_bindings
        result.update(reference_build_input_bindings(resolve_values(row['parameters'],context))[-1])
    if row['kind']=='sidm_control_ensemble':
        from .sidm_control_measurements import control_ensemble_input_bindings
        result.update(control_ensemble_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='sidm_control_refinement_measurement':
        from .sidm_control_measurements import control_refinement_measurement_input_bindings
        result.update(control_refinement_measurement_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='sidm_control_expectation':
        from .sidm_control_measurements import control_expectation_input_bindings
        result.update(control_expectation_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_particle_projection':
        from .initial_conditions import twofluid_particle_input_bindings
        result.update(twofluid_particle_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_native_ic':
        from .initial_conditions import twofluid_native_ic_input_bindings
        result.update(twofluid_native_ic_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind']=='twofluid_window_comparison':
        from .initial_conditions import twofluid_window_input_bindings
        result.update(twofluid_window_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] in {'physical_target_support','physical_target_model_control','physical_target_measurement','physical_target_budget','physical_target_convergence'}:
        from .zoom import physical_target_stage_input_bindings
        result.update(physical_target_stage_input_bindings(row['kind'],resolve_values(row['parameters'],context)))
    if row['kind'] in {'sidm_history','sidm_matched','sidm_observations','sidm_membership','sidm_satellite','sidm_derived','native_forward_vector','sidm_joint_response','sidm_joint_calibration','sidm_calibrated_response'}:
        from .sidm_analysis import sidm_input_bindings
        result.update(sidm_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'finding_repair_receipt':
        from core.research.findings import repair_input_bindings
        result.update(repair_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_pilot_galaxy_measurement':
        from core.research.pilots import galaxy_measurement_input_bindings
        result.update(galaxy_measurement_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_zoom_diagnostics':
        from core.research.zooms import zoom_diagnostic_input_bindings
        result.update(zoom_diagnostic_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'sidm_reference_cdf':
        from .sidm_control_products import cdf_input_bindings
        result.update(cdf_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'sidm_native_events':
        from .sidm_control_products import event_input_bindings
        result.update(event_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'isolated_control_history':
        from .sidm_control_measurements import isolated_history_input_bindings
        result.update(isolated_history_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'sidm_angular_control':
        from .sidm_control_measurements import angular_control_input_bindings
        result.update(angular_control_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'convergence_measurement':
        from core.research.measurements import measurement_input_bindings
        result.update(measurement_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_convergence_figures':
        from core.research.convergence import convergence_figure_inputs
        result.update(convergence_figure_inputs(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_convergence_phase_pair':
        from core.research.convergence import convergence_phase_inputs
        result.update(convergence_phase_inputs(resolve_values(row['parameters'],context))[0])
    if row['kind'] == 'native_convergence_catalogue':
        from .native_convergence_catalogue import catalogue_profile_inputs
        result.update(catalogue_profile_inputs(resolve_values(row['parameters'],context)))
    if row['kind'] == 'isolated_physical_collision':
        from core.research.controls import collision_input_bindings
        result.update(collision_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_sidm_resources':
        from .sidm_campaign import sidm_resource_input_bindings
        result.update(sidm_resource_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_sidm_zoom_support':
        from .apertures import zoom_support_input_bindings
        result.update(zoom_support_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'isolated_observable':
        from core.research.measurements import isolated_observable_input_bindings
        result.update(isolated_observable_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_error_envelope':
        from core.research.errors import envelope_input_bindings
        result.update(envelope_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'native_control_residual':
        from core.research.errors import residual_input_bindings
        result.update(residual_input_bindings(resolve_values(row['parameters'],context)))
    if row['kind'] == 'error_ensemble':
        from core.research.errors import ensemble_input_bindings
        params = resolve_values(row['parameters'],context)
        result.update(ensemble_input_bindings(params['realizations'],params['systematic_controls'],
            request_base=params['request_base'],memory_bytes=params['memory_bytes']))
    if row['kind'] == 'display_products':
        from core.research.display import display_request_input_bindings
        params = resolve_values(row['parameters'],context)
        result.update(display_request_input_bindings(params['request'],
            request_base=params['request_base'],memory_gib=params['memory_gib']))
    for item in row.get('input_files', []) + row.get('assets', []):
        p = path_at(context.base, item['path'])
        if file_sha256(p) != item['sha256']:
            raise ValueError('Input differs from declared SHA256: ' + str(p))
        result[str(p)] = item['sha256']
    for value in _input_references(row.get('parameters', {})):
        p = path_at(context.base, value)
        result[str(p)] = file_sha256(p)
    return result


def execute_stage(row, context, registry, *, resume):
    definition = registry.get(row['kind'])
    capability_names = tuple(dict.fromkeys(definition.capabilities + tuple(row.get('capabilities', []))))
    context.capabilities.require(capability_names,row.get('domain_parameters',{}),row.get('required_features',[]))
    source_hashes = registry.source_hashes(row['kind'])
    for helper in ('workflow.py', 'foundation.py', 'resources.py', 'stream_resolution.py'):
        source = Path(__file__).with_name(helper)
        source_hashes[str(source.resolve())] = file_sha256(source)
    
    for source in reader_source_paths(row.get('product_contracts',{})):
        source_hashes[str(source)] = file_sha256(source)
    inputs = _input_hashes(row, context)
    names = capability_names
    fingerprint = canonical_digest({'stage': row, 'inputs': inputs, 'sources': source_hashes,
        'capabilities': context.capabilities.signature(names), 'resources': context.resources})
    index = context.destination/'receipts'/row['stage_id']/(fingerprint+'.json')
    if resume and index.is_file():
        pointer = json.loads(index.read_text())
        receipt_path = Path(pointer['receipt']).resolve()
        if not receipt_path.is_relative_to(context.destination/'generations'/row['stage_id']):
            raise ValueError('Resume pointer belongs to another stage or workflow directory')
        if file_sha256(receipt_path) != pointer['sha256']:
            raise ValueError('Receipt pointer target changed')
        record = verify_completed_receipt(receipt_path)
        if record['fingerprint'] != fingerprint:
            raise ValueError('Stale resume fingerprint')
        return record
    generation_parent = context.destination/'generations'/row['stage_id']
    generation = generation_parent/(fingerprint[:16]+'-'+uuid.uuid4().hex)
    context.resume_checkpoint = None
    policy = row.get('checkpoint_resume')
    if resume and policy:
        candidates = []
        for saved_path in generation_parent.glob('*/stage-receipt.json'):
            old = json.loads(saved_path.read_text())
            if old.get('fingerprint') == fingerprint and old.get('execution_status') in {'FAILED','INTERRUPTED','RUNNING'}:
                candidates.append((old.get('started_unix', 0), saved_path, old))
        for _, saved_path, old in sorted(candidates, key=lambda q: q[0], reverse=True):
            marker_path = (saved_path.parent/policy['marker']).resolve()
            if not marker_path.is_relative_to(saved_path.parent) or not marker_path.is_file():
                continue
            marker = json.loads(marker_path.read_text())
            if marker.get('status') != 'COMPLETE' or not marker.get('files_sha256'):
                continue
            for filename, expected in marker['files_sha256'].items():
                checkpoint_file = (marker_path.parent/filename).resolve()
                if not checkpoint_file.is_relative_to(saved_path.parent) or file_sha256(checkpoint_file) != expected:
                    raise ValueError('Native resume checkpoint shard missing, changed or outside generation')
            if old.get('source_hashes') != source_hashes or old.get('input_hashes') != inputs:
                raise ValueError('Native continuation source/input version changed')
            atomic_json(context.destination/'attempt-history'/(uuid.uuid4().hex+'.json'), old)
            generation = saved_path.parent
            context.resume_checkpoint = {'marker_path': str(marker_path), 'marker_sha256': file_sha256(marker_path),
                'previous_execution_status': old['execution_status'], 'policy': policy}
            break
    if context.resume_checkpoint is None:
        generation.mkdir(parents=True, exist_ok=False)
    context.generation = generation
    context.stage_id = row['stage_id']
    record = {'schema': 'research-stage-receipt-v1', 'stage_id': row['stage_id'], 'kind': row['kind'],
        'fingerprint': fingerprint, 'generation': str(generation), 'execution_status': 'RUNNING',
        'scientific_acceptance': 'UNRESOLVED', 'source_hashes': source_hashes,
        'input_hashes': inputs, 'resources': row.get('resources', context.resources),
        'started_unix': time.time(), 'products': {}, 'files_sha256': {},
        'resume_checkpoint': context.resume_checkpoint}
    atomic_json(generation/'stage-receipt.json', record)
    start = time.monotonic()
    try:
        params = resolve_values(row.get('parameters', {}), context)
        _validate_numerical_domains(row, params)
        if definition.validator:
            definition.validator(params)
        stage_context = replace(context, resources=row.get('resources', context.resources))
        result = definition.runner(params, stage_context) or {}
        for product, name in row['outputs'].items():
            p = (generation/name).resolve()
            if not p.is_relative_to(generation) or not p.exists():
                raise ValueError('Producer did not deliver declared product: ' + product)
            if p.is_dir() and not any(q.is_file() for q in p.rglob('*')):
                raise ValueError('Declared product directory is empty: ' + product)
            record['products'][product] = {'path': name, 'kind': 'directory' if p.is_dir() else 'file'}
        
        record['product_contract_receipts'] = {product:read_stage_product(generation/row['outputs'][product],descriptor,
            generation=generation,memory_gib=stage_context.resources['memory_gib'])
            for product,descriptor in row.get('product_contracts',{}).items()}
        for p, expected in {**inputs, **source_hashes}.items():
            if file_sha256(p) != expected:
                raise ValueError('Source/input changed during stage execution')
        record.update(execution_status='COMPLETE', elapsed_seconds=time.monotonic()-start,
                      findings=result.get('findings', []), remaining_requirements=result.get('remaining_requirements', []),
                      scientific_exclusions=result.get('scientific_exclusions', []),
                      validation_receipts=result.get('validation_receipts', []),
                      followups=result.get('followups', []), measurements=result.get('measurements', {}))
        record['files_sha256'] = tree_hashes(generation)
        for name in record['files_sha256']:
            with (generation/name).open('rb') as stream:
                os.fsync(stream.fileno())
        atomic_json(generation/'stage-receipt.json', record)
        from .foundation import atomic_text
        atomic_text(generation/'stage-receipt.sha256', file_sha256(generation/'stage-receipt.json')+'\n')
        fsync_directory(generation)
        atomic_json(index, {'receipt': str(generation/'stage-receipt.json'),
                            'sha256': file_sha256(generation/'stage-receipt.json')})
        return record
    except BaseException as exc:
        record.update(execution_status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED',
                      elapsed_seconds=time.monotonic()-start, failure={'type': type(exc).__name__, 'message': str(exc)})
        atomic_json(generation/'stage-receipt.json', record)
        raise


def append_budgeted_child_stages(plan, parent, children, child_count):
    limits = plan['followup_limits']
    if child_count + len(children) > limits['maximum_children']:
        raise ValueError('Follow-up child count exhausted')
    if len(plan['stages']) + len(children) > plan['resources']['maximum_stage_count']:
        raise ValueError('Workflow stage count exhausted')
    ids = {q['stage_id'] for q in plan['stages']}
    depths = {q['stage_id']: q['depth'] for q in plan['stages']}
    normalized = []
    for child in children:
        safe_id(child['stage_id'])
        if child['stage_id'] in ids:
            raise ValueError('Follow-up reuses existing stage identity')
        deps = list(dict.fromkeys(child.get('depends_on', []) + [parent['stage_id']] +
            list(_stage_references(child.get('parameters', {}))) + list(_stage_references(child.get('inputs', {})))))
        if any(dep not in ids for dep in deps):
            raise ValueError('Follow-up depends on unknown/future stage')
        depth = max([parent['depth'], *[depths[dep] for dep in deps]]) + 1
        if depth > limits['maximum_depth']:
            raise ValueError('Follow-up dependency depth exhausted')
        row = {**child, 'depends_on': deps, 'depth': depth, 'parent_stage': parent['stage_id']}
        normalized.append(row)
        ids.add(row['stage_id'])
        depths[row['stage_id']] = depth
    plan['stages'].extend(normalized)
    return child_count + len(children)


def execute_workflow(spec, destination, *, request_base, resume=False, dry_run=False, registry=None):
    if spec.get('schema_version') == 2:
        spec = compile_run_spec_v2(spec, registry=registry, request_base=request_base)
    registry = registry or default_registry()
    destination = Path(destination).resolve()
    base = Path(request_base).resolve()
    plan = compile_requested_product_dag(spec, registry, request_base=base)
    context = WorkflowContext(destination, base, plan['resources'],
        CapabilityRegistry.from_spec(spec.get('capabilities', []), base), {})
    preflight_plan(plan, spec, context, registry)
    if dry_run:
        return plan
    with native_run_lock(destination, operation='research product DAG'):
        destination.mkdir(parents=True, exist_ok=True)
        state_path = destination/'workflow.json'
        specification_hash = canonical_digest(spec)
        old = json.loads(state_path.read_text()) if state_path.exists() else None
        if old and not resume:
            raise FileExistsError('Existing workflow requires --resume')
        if old and old['specification_hash'] != specification_hash:
            # Per-stage content fingerprints decide reuse after deliberate edits.
            atomic_json(destination/'history'/(uuid.uuid4().hex+'.json'), old)
        state = {'schema': 'research-workflow-receipt-v1', 'specification_hash': specification_hash,
                 'execution_status': 'RUNNING', 'scientific_acceptance': 'UNRESOLVED',
                 'stages': {}, 'followup_count': 0, 'remaining_requirements': [], 'scientific_exclusions': []}
        atomic_json(state_path, state)
        position = 0
        try:
            while position < len(plan['stages']):
                row = plan['stages'][position]
                result = execute_stage(row, context, registry, resume=resume)
                context.completed[row['stage_id']] = result
                state['stages'][row['stage_id']] = {'generation': result['generation'], 'fingerprint': result['fingerprint'],
                    'receipt_sha256': file_sha256(Path(result['generation'])/'stage-receipt.json')}
                state['scientific_exclusions'].extend({'stage': row['stage_id'], **q} for q in result.get('scientific_exclusions', []))
                state['remaining_requirements'].extend({'stage': row['stage_id'], **q} for q in result.get('remaining_requirements', []))
                children = result.get('followups', [])
                if children:
                    previous_length = len(plan['stages'])
                    try:
                        new_count = append_budgeted_child_stages(plan, row, children, state['followup_count'])
                        # Recompile new child definitions too, before any child execution.
                        child_spec = {**spec, 'stages': plan['stages'],
                            'requested_products': [name for q in plan['stages'] for name in q['outputs']]}
                        checked = compile_requested_product_dag(child_spec, registry, request_base=base)
                        preflight_plan(checked, spec, context, registry)
                        state['followup_count'] = new_count
                    except ValueError as exc:
                        state['remaining_requirements'].append({'parent': row['stage_id'], 'reason': str(exc), 'requests': children})
                        # No unpreflighted children can execute.
                        plan['stages'] = plan['stages'][:previous_length]
                atomic_json(state_path, state)
                position += 1
            state['execution_status'] = 'INCOMPLETE' if state['remaining_requirements'] else 'COMPLETE'
            if plan.get('production_resource_plan'):
                from core.resources import publish_resource_comparison
                resource_output = destination/'resource-measurements'/(uuid.uuid4().hex)
                comparison = publish_resource_comparison(plan['production_resource_plan'],
                    [Path(record['generation'])/'stage-receipt.json' for record in context.completed.values()], resource_output)
                state['resource_comparison'] = {'path':str(resource_output/'resource-comparison.json'),
                    'sha256':file_sha256(resource_output/'resource-comparison.json'), 'plan_hash':comparison['plan_hash']}
        except BaseException as exc:
            state.update(execution_status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED',
                         failed_stage=plan['stages'][position]['stage_id'] if position < len(plan['stages']) else 'resource_publication', failure=str(exc))
            atomic_json(state_path, state)
            raise
        atomic_json(state_path, state)
        return state


def command_runner(params, context):
    """No shell, recorded argv, streamed logs, bounded wall time, owned process."""
    argv = params['argv']
    if context.resume_checkpoint is not None:
        from core.research.checkpoints import archive_resume_checkpoint
        archive_resume_checkpoint(context)
        resume_argv = context.resume_checkpoint['policy'].get('argv')
        if not resume_argv:
            raise ValueError('Native continuation needs an explicit resume argument vector')
        argv = resolve_values(resume_argv, context)
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError('Explicit nonempty argument vector required')
    wall = positive_number(params['wall_seconds'], 'wall_seconds')
    env = os.environ.copy()
    resources = context.resources
    env.update(OMP_NUM_THREADS=str(resources['omp_threads']), OPENBLAS_NUM_THREADS=str(resources['omp_threads']))
    from core.native import run_native_command
    log_path = context.generation/'execution.log'
    if context.resume_checkpoint is not None and log_path.exists():
        os.replace(log_path, context.generation/('execution.previous-'+uuid.uuid4().hex+'.log'))
    if context.resume_checkpoint is not None and (context.generation/'invocation.json').exists():
        os.replace(context.generation/'invocation.json',context.generation/('invocation.previous-'+uuid.uuid4().hex+'.json'))
    from core.resources import LocalProcessResourceSampler
    resource_path = context.generation/'local-resource-samples.json'
    if context.resume_checkpoint is not None and resource_path.exists():
        os.replace(resource_path,context.generation/('local-resource-samples.previous-'+uuid.uuid4().hex+'.json'))
    sampler = LocalProcessResourceSampler(context.generation/'local-resource-samples.json',
        **params.get('resource_sampling',{}))
    with log_path.open('wb') as log:
        run_native_command(argv, cwd=context.generation, receipt_path=context.generation/'invocation.json',
            inputs=(Path(context.resume_checkpoint['marker_path']),) if context.resume_checkpoint is not None else (), timeout=wall,
            env=env, stdout=log, stderr=subprocess.STDOUT,process_observer=sampler)
    result = json.loads((context.generation/'followups.json').read_text()) if (context.generation/'followups.json').is_file() else {}
    sampled = json.loads((context.generation/'local-resource-samples.json').read_text())
    result['measurements'] = {**result.get('measurements',{}),
        'sampled_local_peak_rss_bytes':sampled['sampled_local_peak_rss_bytes'],
        'local_resource_coverage':sampled['coverage_status'],
        'resource_samples_path':'local-resource-samples.json'}
    return result


def public_command_runner(params, context):
    from commands import COMMANDS
    command = params['command']
    if command not in COMMANDS or command in {'workflow', 'pilots', 'clean-release'}:
        raise ValueError('Unknown or recursive public command')
    arguments = list(params.get('arguments', []))
    if command == 'gadget':
        if '--answers' not in arguments:
            raise ValueError('Native evolution requires explicit --answers')
        answer_path = Path(arguments[arguments.index('--answers')+1])
        answers = json.loads(answer_path.read_text())
        if params.get('canonical_run_constraints') is not None:
            from core.research.capabilities import validate_native_answers
            validate_native_answers(answers,context.resources,params['canonical_run_constraints'])
        for key, resource_key in (('mpi_ranks','mpi_ranks'), ('omp_threads','omp_threads'), ('memory_limit_gb','memory_gib')):
            if answers.get(key) != context.resources[resource_key]:
                raise ValueError('Native evolution resource choice differs from the explicit stage allocation: '+key)
    if params.get('copy_run_input') is not None:
        import shutil
        index = params['copy_run_input']
        if type(index) is not int or not 0 <= index < len(arguments):
            raise ValueError('copy_run_input must select an existing run-directory argument')
        source = Path(arguments[index]).resolve()
        if not source.is_dir():
            raise ValueError('Public analysis/display input must be an actual native run directory')
        byte_count=0
        for path in source.rglob('*'):
            # rglob does not traverse directory symlinks, while copytree with
            # symlinks=False does. Reject that uncounted/cyclic copy route.
            if path.is_symlink() and path.is_dir():
                raise ValueError('Private run copies require explicit directories rather than directory symlinks')
            if path.is_file():byte_count+=path.stat().st_size
        maximum=params.get('copy_run_max_bytes')
        if maximum is not None:
            if type(maximum) is not int or maximum<1:
                raise ValueError('Exact positive private-run copy byte bound required')
            if byte_count>maximum:
                raise MemoryError('Private analysis run copy exceeds its explicit stage disk bound')
        free = shutil.disk_usage(context.generation).free
        if byte_count > context.resources['disk_gib']*1024**3 or byte_count > free:
            raise RuntimeError('Private analysis run copy exceeds declared/free disk; requested physics unchanged')
        target = context.generation/'private_run'
        shutil.copytree(source, target, symlinks=False, ignore=shutil.ignore_patterns('.native-run-locks'))
        arguments[index] = str(target)
        atomic_json(context.generation/'run-copy.json', {'source': str(source), 'destination': str(target),
            'copied_bytes': byte_count, 'purpose': 'public analysis mutates its private run while upstream generation remains immutable'})
    argv = [sys.executable,str(ROOT/'commands.py'),command,*arguments]
    try:
        return command_runner({'argv':argv,'wall_seconds':params['wall_seconds'],
            'resource_sampling':params.get('resource_sampling',{})},context)
    except BaseException:
        if command=='gadget' and '--run-dir' in arguments:
            from core.research.checkpoints import retain_failed_evolution_checkpoint
            retain_failed_evolution_checkpoint(context.generation,arguments)
        raise


def json_function_runner(module_name, function_name, filename):
    def runner(params, context):
        function = getattr(importlib.import_module(module_name), function_name)
        result = function(**params)
        atomic_json(context.generation/filename, result)
        return result if isinstance(result, dict) else {}
    return runner


def display_products_runner(params,context):
    from core.research.display import assemble_measured_display
    ceiling = positive_number(params['memory_gib'],'display preflight memory')
    if ceiling > context.resources['memory_gib'] or params['request']['memory_gib'] > ceiling:
        raise MemoryError('Measured display memory exceeds its frozen allocation')
    destination = Path(params['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Workflow display product destination escapes its owned generation')
    result = assemble_measured_display(params['request'],destination,request_base=params['request_base'])
    atomic_json(context.generation/'display-assembly.json',result)
    return result


def error_ensemble_runner(params,context):
    from core.research.errors import publish_correlated_ensemble
    memory = positive_number(params['memory_bytes'],'ensemble workspace',integer=True)
    if memory > context.resources['memory_gib']*1024**3:
        raise MemoryError('Ensemble workspace exceeds its frozen stage allocation')
    destination = Path(params['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Ensemble publication escapes its permanent workflow generation')
    result = publish_correlated_ensemble(**params)
    atomic_json(context.generation/'ensemble-publication.json',result)
    return result


def default_registry():
    registry = StageRegistry()
    from .flagship_native_branches import native_branch_runner
    registry.register(StageDefinition('native_branch_products',native_branch_runner,
        (Path(__file__),ROOT/'commands.py',*tuple(sorted((ROOT/'core').glob('*.py')))),
        required_parameters=('case','design','memory_gib','maximum_metadata_files')))
    from .flagship_retained import retained_operation_runner
    registry.register(StageDefinition('flagship_retained_operation',retained_operation_runner,
        (Path(__file__),ROOT/'core/flagship_retained.py',ROOT/'core/flagship_runtime_measurements.py',ROOT/'core/flagship_runtime_display.py',
         ROOT/'core/halo_campaign.py',ROOT/'core/foundation.py'),required_parameters=('operation','source_workflow','memory_gib','maximum_stages')))
    from .flagship_analysis_profile import runtime_bao_runner
    registry.register(StageDefinition('flagship_runtime_bao',runtime_bao_runner,
        (Path(__file__),ROOT/'commands.py',*tuple(sorted((ROOT/'core').glob('*.py')))),
        required_parameters=('schema','kind','linear_bundle','linear_receipt','observations','analysis','wall_seconds','resources')))
    from .flagship_runtime_measurements import measurement_preparation_runner
    registry.register(StageDefinition('flagship_runtime_measurements',measurement_preparation_runner,
        (Path(__file__),ROOT/'commands.py',ROOT/'core/figures.py',*tuple(sorted((ROOT/'core').glob('*.py')))),
        required_parameters=('schema','prefix','cases','linear_bundle','linear_receipt','memory_gib','maximum_metadata_files','options','continuation')))
    from .flagship_runtime_display import display_preparation_runner,runtime_scene_runner
    display_sources=(Path(__file__),ROOT/'core/flagship_runtime_measurements.py',ROOT/'core/flagship_runtime_display.py',ROOT/'core/flagship_runtime_history.py',
        source_path('research.convergence'),source_path('research.display'),ROOT/'core/native_scene_overlay.py',
        ROOT/'core/graph_producer_registry.py',
        ROOT/'configs/research/graph_producers.json',ROOT/'core/scene_render.py',ROOT/'core/cli.py',ROOT/'commands.py',
        ROOT/'core/blender.py',ROOT/'core/visualization.py',ROOT/'core/data.py',
        ROOT/'core/analysis.py',ROOT/'core/sidm.py',ROOT/'core/research/generator_controls.py',ROOT/'core/foundation.py',ROOT/'commands.py')
    registry.register(StageDefinition('flagship_runtime_display',display_preparation_runner,display_sources,
        required_parameters=('schema','prefix','cases','memory_gib','disk_gib','maximum_metadata_files','maximum_figures','scene')))
    registry.register(StageDefinition('flagship_runtime_scene',runtime_scene_runner,display_sources,
        required_parameters=('schema','run','native_receipt','gallery','streams','progenitors','contamination','encounter_overlay','options','memory_gib','disk_gib')))
    from .flagship_runtime_history import runtime_history_runner
    registry.register(StageDefinition('flagship_runtime_history',runtime_history_runner,
        (
            Path(__file__),
            ROOT/'core/flagship_runtime_history.py',
            ROOT/'core/flagship_zoom_preparation.py',
            ROOT/'core/flagship_campaigns.py',
            ROOT/'core/flagship_history.py',
            ROOT/'core/flagship_profiles.py',
            ROOT/'core/flagship_products.py',
            source_path('research.convergence'),
            ROOT/'core/music/staging.py',
            ROOT/'core/research/restart_campaign.py',
            ROOT/'core/halo_campaign.py',
            ROOT/'core/zoom.py',
            ROOT/'core/research/generator_controls.py',
            ROOT/'core/foundation.py',
            ROOT/'commands.py',
        ),
        required_parameters=('schema','kind','zoom_preparation','zoom_preparation_receipt','cases','parent_answers','combined_answers',
            'numerical_choices','generator','execution','flow_preparation','case_staging','history_template','continuation')))
    from .flagship_zoom_preparation import zoom_preparation_runner,validate_zoom_preparation
    registry.register(StageDefinition('flagship_zoom_preparation',zoom_preparation_runner,
        (
            Path(__file__),
            ROOT/'core/flagship_zoom_preparation.py',
            ROOT/'core/music/phases.py',
            ROOT/'core/zoom.py',
            ROOT/'core/research/generator_controls.py',
            ROOT/'core/initial_conditions.py',
            ROOT/'core/foundation.py',
            ROOT/'core/analysis.py',
            ROOT/'core/sidm.py',
            ROOT/'core/halos.py',
            ROOT/'core/data.py',
            ROOT/'commands.py',
            ROOT/'core/halo_campaign.py',
        ),
        required_parameters=('schema','prefix','native_run_directory','native_stage_receipt','generation_report','generation_stage_receipt',
            'parent_noise','snapshot','selection_rules','explicit_host_rows','staging','generator','execution','continuation'),validator=validate_zoom_preparation))
    from .flagship_parent_preparation import parent_preparation_runner,validate_parent_preparation
    registry.register(StageDefinition('flagship_parent_preparation',parent_preparation_runner,
        (Path(__file__),ROOT/'core/flagship_parent_preparation.py',ROOT/'core/zoom.py',ROOT/'core/initial_conditions.py',
         ROOT/'core/research/generator_controls.py',ROOT/'core/foundation.py',ROOT/'core/research/checkpoints.py',ROOT/'commands.py',
         ROOT/'core/halo_campaign.py'),
        required_parameters=('schema','prefix','linear_bundle','linear_stage_receipt','answers','staging','generator','execution','continuation'),
        validator=validate_parent_preparation))
    from .flagship_products import flagship_product_runner,validate_flagship_product
    registry.register(StageDefinition('flagship_product',flagship_product_runner,
        (
            Path(__file__),
            ROOT/'core/periodic_images.py',
            ROOT/'core/zoom.py',
            ROOT/'core/music/archive.py',
            ROOT/'core/music/archive.py',
            ROOT/'core/music/precision.py',
            ROOT/'core/music_binary128_reference.cc',
            ROOT/'core/music/replay.py',
            ROOT/'core/music/poisson.py',
            ROOT/'core/music/windows.py',
            ROOT/'core/music/covariance.py',
            ROOT/'core/music/seeded.py',
            ROOT/'core/music/phases.py',
            ROOT/'core/numeric.py',
            ROOT/'core/numeric.py',
            ROOT/'core/music/adjoint.py',
            ROOT/'core/music/boundary.py',
            ROOT/'core/music/resources.py',
            ROOT/'core/native_boundary_operator.cc',
            ROOT/'core/music/flow.py',
            ROOT/'core/music/staging.py',
            ROOT/'core/research/checkpoints.py',
            ROOT/'core/flagship_products.py',
            ROOT/'core/figures.py',
            ROOT/'core/stellar_returns.py',
            ROOT/'core/stellar_returns.py',
            ROOT/'core/flagship_profiles.py',
            ROOT/'core/power.py',
            ROOT/'core/halo_products.py',
            ROOT/'core/assembly.py',
            ROOT/'core/flagship_abundance.py',
            ROOT/'core/flagship_orbits.py',
            ROOT/'core/flagship_bh.py',
            ROOT/'core/flagship_bh_response.py',
            ROOT/'core/baryon_products.py',ROOT/'core/stellar_population.py',
            ROOT/'core/first_stars.py',
            ROOT/'core/halo_products.py',
            ROOT/'core/sidm_products.py',
            ROOT/'core/sidm.py',
            ROOT/'core/figures.py',
            ROOT/'core/black_holes.py',
            ROOT/'core/evolution.py',
            ROOT/'core/observations.py',
            ROOT/'core/halos.py',
            ROOT/'core/analysis.py',
            ROOT/'core/cosmology.py',
            ROOT/'core/initial_conditions.py',
            ROOT/'core/streams.py',
            ROOT/'core/research/generator_controls.py',
            ROOT/'core/data.py',
            ROOT/'core/foundation.py',
        ),
        required_parameters=('operation','product','destination','memory_gib'),validator=validate_flagship_product))
    registry.register(StageDefinition('materialize_request', materialize_request_runner, (Path(__file__),), required_parameters=('request',)))
    from .sidm_control_products import control_population_runner,validate_control_population_request
    registry.register(StageDefinition('sidm_control_population',control_population_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_control_population_request))
    from .sidm_control_products import canonical_population_runner,validate_canonical_population_request
    registry.register(StageDefinition('sidm_canonical_population',canonical_population_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_canonical_population_request))
    from .sidm_control_products import mapped_campaign_plan_runner,validate_mapped_campaign_plan_request
    registry.register(StageDefinition('sidm_mapped_campaign_plan',mapped_campaign_plan_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_mapped_campaign_plan_request))
    from .sidm_control_products import homogeneous_control_runner,validate_control_run_request
    from .sidm_control_products import reference_build_runner,validate_reference_build_request
    registry.register(StageDefinition('sidm_reference_build',reference_build_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/research/generator_controls.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_reference_build_request))
    registry.register(StageDefinition('sidm_homogeneous_run',homogeneous_control_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/research/generator_controls.py',ROOT/'core/evolution.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_control_run_request))
    from .sidm_control_measurements import (control_history_runner,control_comparison_runner,
        validate_control_history_request,validate_control_comparison_request)
    control_sources=(Path(__file__),ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'core/evolution.py')
    registry.register(StageDefinition('sidm_control_history',control_history_runner,control_sources,
        required_parameters=('request','request_base','destination','native_stage_receipt','parameter_file','build_configuration','invocation_receipt'),
        validator=validate_control_history_request))
    registry.register(StageDefinition('sidm_control_comparison',control_comparison_runner,control_sources,
        required_parameters=('request','request_base','destination'),validator=validate_control_comparison_request))
    from .sidm_control_measurements import control_clock_mapping_runner,validate_control_clock_mapping_request
    registry.register(StageDefinition('sidm_control_clock_mapping',control_clock_mapping_runner,control_sources,
        required_parameters=('request','request_base','destination'),validator=validate_control_clock_mapping_request))
    from .sidm_control_measurements import control_native_clock_runner,validate_control_native_clock_request
    registry.register(StageDefinition('sidm_control_native_clock',control_native_clock_runner,
        control_sources+(ROOT/'core/software/gadget4/src/sidm/research_canonical_clock.h',ROOT/'core/software/gadget4/src/sidm/research_physical_clock.h',ROOT/'core/software/gadget4/src/black_holes/research_asset_sha256.h'),
        required_parameters=('request','request_base','destination'),validator=validate_control_native_clock_request))
    from .sidm_control_measurements import control_ensemble_runner,validate_control_ensemble_request
    registry.register(StageDefinition('sidm_control_ensemble',control_ensemble_runner,control_sources,
        required_parameters=('request','request_base','destination'),validator=validate_control_ensemble_request))
    from .sidm_control_measurements import control_refinement_measurement_runner,validate_control_refinement_measurement_request
    registry.register(StageDefinition('sidm_control_refinement_measurement',control_refinement_measurement_runner,control_sources,
        required_parameters=('request','request_base','destination'),validator=validate_control_refinement_measurement_request))
    from .sidm_control_measurements import control_expectation_runner,validate_control_expectation_request
    registry.register(StageDefinition('sidm_control_expectation',control_expectation_runner,control_sources,
        required_parameters=('request','request_base','destination'),validator=validate_control_expectation_request))
    from .initial_conditions import twofluid_relative_runner
    from .initial_conditions import twofluid_nonlinear_runner
    from .initial_conditions import twofluid_eulerian_seed_runner,validate_twofluid_eulerian_seed_request
    from .black_holes import dm_wake_force_runner,validate_dm_wake_force_request
    from .black_holes import dm_wake_kernel_comparison_runner,validate_dm_wake_kernel_comparison_request
    registry.register(StageDefinition('dm_wake_kernel_comparison',dm_wake_kernel_comparison_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py',ROOT/'core/software/gadget4/src/black_holes/research_sidm_wake_measurement.h'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_kernel_comparison_request))
    from .black_holes import dm_wake_contrast_runner,validate_dm_wake_contrast_request
    from .black_holes import dm_wake_ensemble_runner,validate_dm_wake_ensemble_request
    from .black_holes import dm_wake_response_runner,validate_dm_wake_response_request
    from .black_holes import dm_wake_native_table_runner,validate_dm_wake_native_table_request
    from .stream_forward_model import native_stream_worldline_runner,validate_native_stream_worldline_request
    from .stream_forward_model import native_stream_comparison_runner,validate_native_stream_comparison_request
    from .stream_forward_model import native_late_threeway_runner,validate_native_late_threeway_request
    registry.register(StageDefinition('native_late_threeway',native_late_threeway_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/sidm.py',ROOT/'core/evolution.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_native_late_threeway_request))
    from .stream_forward_model import native_counterfactual_ic_runner,validate_native_counterfactual_ic_request
    from .stream_forward_model import native_counterfactual_result_runner,validate_native_counterfactual_result_request
    registry.register(StageDefinition('native_counterfactual_result',native_counterfactual_result_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/observations.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_native_counterfactual_result_request))
    from .stream_forward_model import native_counterfactual_run_runner,validate_native_counterfactual_run_request
    from .stream_forward_model import native_counterfactual_campaign_runner,validate_native_counterfactual_campaign_request
    registry.register(StageDefinition('native_counterfactual_campaign',native_counterfactual_campaign_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_native_counterfactual_campaign_request))
    registry.register(StageDefinition('native_counterfactual_run',native_counterfactual_run_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'core/evolution.py',ROOT/'core/research/generator_controls.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','native_stage_receipt','native_run_directory','destination'),validator=validate_native_counterfactual_run_request))
    registry.register(StageDefinition('native_counterfactual_ic',native_counterfactual_ic_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'core/sidm.py'),
        required_parameters=('request','native_stage_receipt','native_run_directory','destination'),validator=validate_native_counterfactual_ic_request))
    registry.register(StageDefinition('native_stream_comparison',native_stream_comparison_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_native_stream_comparison_request))
    registry.register(StageDefinition('native_stream_worldlines',native_stream_worldline_runner,
        (Path(__file__),ROOT/'core/streams.py',ROOT/'core/observations.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','native_stage_receipt','native_run_directory','destination'),validator=validate_native_stream_worldline_request))
    registry.register(StageDefinition('dm_wake_native_table',dm_wake_native_table_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py',ROOT/'core/software/gadget4/src/black_holes/research_sidm_drag_response.h'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_native_table_request))
    registry.register(StageDefinition('dm_wake_response',dm_wake_response_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_response_request))
    from .black_holes import dm_wake_state_response_runner,validate_dm_wake_state_response_request
    from .black_holes import dm_wake_state_table_runner,validate_dm_wake_state_table_request
    from .black_holes import dm_wake_context_runner,validate_dm_wake_context_request
    from .initial_conditions import twofluid_refinement_runner,validate_twofluid_refinement_request
    from .initial_conditions import twofluid_native_comparison_runner,validate_twofluid_native_comparison_request
    from .initial_conditions import twofluid_native_family_runner,validate_twofluid_native_family_request
    registry.register(StageDefinition('twofluid_native_family',twofluid_native_family_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/data.py',ROOT/'core/sidm.py',
         ROOT/'core/evolution.py',ROOT/'core/software/gadget4/src/logs/logs.cc',
         ROOT/'core/software/gadget4/src/gravtree/gwalk.cc',ROOT/'core/software/gadget4/src/gravity/gravity.cc',
         ROOT/'core/software/gadget4/src/gravtree/gravtree_build.cc',ROOT/'core/software/gadget4/src/pm/pm_periodic.cc',
         ROOT/'core/software/gadget4/src/sph/hydra.cc'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_native_family_request))
    registry.register(StageDefinition('twofluid_native_comparison',twofluid_native_comparison_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/data.py',ROOT/'core/sidm.py',
         ROOT/'core/software/gadget4/src/time_integration/driftfac.cc',ROOT/'core/software/gadget4/src/main/begrun.cc'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_native_comparison_request))
    from .initial_conditions import twofluid_projection_refinement_runner,validate_twofluid_projection_refinement_request
    registry.register(StageDefinition('twofluid_projection_refinement',twofluid_projection_refinement_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_projection_refinement_request))
    from .initial_conditions import twofluid_refinement_campaign_runner,validate_twofluid_refinement_campaign_request
    registry.register(StageDefinition('twofluid_refinement_campaign',twofluid_refinement_campaign_runner,
        (
            Path(__file__),
            ROOT/'core/initial_conditions.py',
            ROOT/'core/data.py',
            ROOT/'core/sidm.py',
            ROOT/'core/research/generator_controls.py',
            ROOT/'core/evolution.py',
            ROOT/'core/foundation.py',
            ROOT/'core/zoom.py',
            ROOT/'core/music/archive.py',
            ROOT/'core/music/precision.py',
            ROOT/'core/music_binary128_reference.cc',
            ROOT/'core/music/replay.py',
            ROOT/'core/music/poisson.py',
            ROOT/'core/music/windows.py',
            ROOT/'core/music/covariance.py',
            ROOT/'core/music/seeded.py',
            ROOT/'core/music/phases.py',
            ROOT/'core/numeric.py',
            ROOT/'core/numeric.py',
            ROOT/'core/music/adjoint.py',
            ROOT/'core/music/boundary.py',
            ROOT/'core/music/resources.py',
            ROOT/'core/native_boundary_operator.cc',
            ROOT/'core/music/flow.py',
            ROOT/'core/music/staging.py',
            ROOT/'core/research/checkpoints.py',
            ROOT/'commands.py',
        ),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_refinement_campaign_request))
    registry.register(StageDefinition('twofluid_nonlinear_refinement',twofluid_refinement_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_refinement_request))
    registry.register(StageDefinition('dm_wake_context',dm_wake_context_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py',ROOT/'core/software/gadget4/src/black_holes/research_sidm_drag_context.h'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_context_request))
    registry.register(StageDefinition('dm_wake_state_table',dm_wake_state_table_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py',ROOT/'core/software/gadget4/src/black_holes/research_sidm_drag_state_response.h'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_state_table_request))
    registry.register(StageDefinition('dm_wake_state_response',dm_wake_state_response_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_state_response_request))
    registry.register(StageDefinition('dm_wake_ensemble',dm_wake_ensemble_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_ensemble_request))
    registry.register(StageDefinition('dm_wake_contrast',dm_wake_contrast_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination'),validator=validate_dm_wake_contrast_request))
    registry.register(StageDefinition('dm_wake_force',dm_wake_force_runner,
        (Path(__file__),ROOT/'core/black_holes.py',ROOT/'core/observations.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','native_stage_receipt','native_run_directory','destination'),validator=validate_dm_wake_force_request))
    registry.register(StageDefinition('twofluid_eulerian_seed',twofluid_eulerian_seed_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/observations.py',ROOT/'core/data.py',ROOT/'core/sidm.py',
         ROOT/'core/software/gadget4/src/time_integration/driftfac.cc',ROOT/'core/software/gadget4/src/main/begrun.cc'),
        required_parameters=('request','native_stage_receipt','native_run_directory','destination'),validator=validate_twofluid_eulerian_seed_request))
    registry.register(StageDefinition('twofluid_nonlinear_reference',twofluid_nonlinear_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','request_base','destination','initial_fields')))
    registry.register(StageDefinition('twofluid_relative',twofluid_relative_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/data.py',ROOT/'core/sidm.py'),
        required_parameters=('request','initial_fields','request_base','destination')))
    from .initial_conditions import twofluid_particle_runner,validate_twofluid_particle_request
    registry.register(StageDefinition('twofluid_particle_projection',twofluid_particle_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_particle_request))
    from .initial_conditions import twofluid_native_ic_runner,validate_twofluid_native_ic_request
    registry.register(StageDefinition('twofluid_native_ic',twofluid_native_ic_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_native_ic_request))
    from .initial_conditions import twofluid_window_runner,validate_twofluid_window_request
    registry.register(StageDefinition('twofluid_window_comparison',twofluid_window_runner,
        (Path(__file__),ROOT/'core/initial_conditions.py',ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/zoom.py',ROOT/'core/foundation.py'),
        required_parameters=('request','request_base','destination'),validator=validate_twofluid_window_request))
    from .zoom import (physical_target_support_runner,physical_target_model_runner,physical_target_measurement_runner,
        physical_target_budget_runner,physical_target_convergence_runner,validate_physical_target_request,validate_target_model_control_request,
        validate_target_measurement_request,validate_target_budget_request,validate_target_convergence_request)
    target_sources=(Path(__file__),ROOT/'core/zoom.py',ROOT/'core/stars.py',ROOT/'core/observations.py',ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py')
    for kind,runner,validator,extras in (
        ('physical_target_support',physical_target_support_runner,validate_physical_target_request,('native_run_directory','native_stage_receipt')),
        ('physical_target_model_control',physical_target_model_runner,validate_target_model_control_request,()),
        ('physical_target_measurement',physical_target_measurement_runner,validate_target_measurement_request,()),
        ('physical_target_budget',physical_target_budget_runner,validate_target_budget_request,('primary_support','target','memory_gib')),
        ('physical_target_convergence',physical_target_convergence_runner,validate_target_convergence_request,('fingerprint','comparisons','memory_gib'))):
        registry.register(StageDefinition(kind,runner,target_sources,required_parameters=('request','request_base','destination',*extras),validator=validator))
    from .apertures import aperture_contamination_runner,validate_aperture_contamination
    from .stellar_returns import (stellar_model_runner,validate_stellar_model_stage,
        stellar_branch_answers_runner,validate_stellar_branch_stage,stellar_history_runner,validate_stellar_history_stage)
    stellar_sources=(Path(__file__),ROOT/'core/stellar_returns.py',
        ROOT/'core/stars.py',ROOT/'core/evolution.py',ROOT/'core/research/generator_controls.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'commands.py')
    registry.register(StageDefinition('native_stellar_model',stellar_model_runner,stellar_sources,
        required_parameters=('request_path','request_sha256','memory_gib'),validator=validate_stellar_model_stage))
    registry.register(StageDefinition('native_stellar_branch_answers',stellar_branch_answers_runner,stellar_sources,
        required_parameters=('answers_path','mode'),validator=validate_stellar_branch_stage))
    registry.register(StageDefinition('native_stellar_history',stellar_history_runner,
        (*stellar_sources,ROOT/'core/black_holes.py',ROOT/'core/halos.py',ROOT/'core/black_hole_ledger.py',ROOT/'configs/hydro_audit.algorithms.json'),
        required_parameters=('run_dir','snapshots','memory_gib','tolerances'),validator=validate_stellar_history_stage))
    from .stream.measurements import (catalogue_stream_measurements_runner,recovered_stream_measurements_runner,
        validate_catalogue_stream_measurements)
    stream_measurement_sources=(Path(__file__),ROOT/'core/stream/measurements.py',
        ROOT/'core/stellar_returns.py',ROOT/'core/black_holes.py',ROOT/'core/black_hole_ledger.py',ROOT/'configs/hydro_audit.algorithms.json',
        ROOT/'core/apertures.py',ROOT/'core/streams.py',ROOT/'core/flagship_products.py',ROOT/'core/stars.py',
        ROOT/'core/halos.py',ROOT/'core/observations.py',ROOT/'core/research/generator_controls.py',ROOT/'core/figures.py',ROOT/'core/data.py',ROOT/'core/foundation.py')
    registry.register(StageDefinition('native_catalogue_stream_measurements',catalogue_stream_measurements_runner,stream_measurement_sources,
        required_parameters=('run_dir','catalogue_product','template_path','template_sha256','memory_gib','maximum_particles','maximum_candidates'),validator=validate_catalogue_stream_measurements))
    registry.register(StageDefinition('native_recovered_stream_measurements',recovered_stream_measurements_runner,stream_measurement_sources,
        required_parameters=('run_dir','aperture_history','snapshots','template_path','template_sha256','memory_gib','maximum_particles','maximum_candidates'),validator=validate_catalogue_stream_measurements))
    from .apertures import native_catalogue_apertures_runner,validate_catalogue_apertures
    registry.register(StageDefinition('native_catalogue_apertures',native_catalogue_apertures_runner,
        (Path(__file__),ROOT/'core/apertures.py',ROOT/'core/streams.py',ROOT/'core/halos.py',ROOT/'core/observations.py',ROOT/'core/zoom.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('run_dir','catalogue_product','memory_gib','maximum_apertures'),validator=validate_catalogue_apertures))
    from .stream.native_encounters import recovered_native_encounters_runner,validate_recovered_encounters
    registry.register(StageDefinition('recovered_native_encounters',recovered_native_encounters_runner,
        (Path(__file__),ROOT/'core/stream/native_encounters.py',ROOT/'core/stream/perturbers.py',ROOT/'core/periodic_images.py',ROOT/'core/flagship_products.py',ROOT/'core/research/generator_controls.py',ROOT/'core/halos.py',ROOT/'core/observations.py',ROOT/'core/evolution.py',ROOT/'core/cosmology.py',ROOT/'core/streams.py',ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('run_dir','native_stage_receipt','stream_measurement_history','template_path','template_sha256','memory_gib'),validator=validate_recovered_encounters))
    from .response_campaign import native_paired_response_runner,validate_paired_response
    registry.register(StageDefinition('native_paired_stream_responses',native_paired_response_runner,
        (Path(__file__),ROOT/'core/response_campaign.py',ROOT/'core/native_stream_responses.py',ROOT/'core/periodic_images.py',ROOT/'core/flagship_products.py',ROOT/'core/streams.py',ROOT/'core/sidm.py',ROOT/'core/halos.py',ROOT/'core/stars.py',ROOT/'core/cosmology.py',ROOT/'core/research/generator_controls.py',ROOT/'core/figures.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('comparison','template_path','template_sha256','memory_gib'),validator=validate_paired_response))
    from .stream.native_encounters import native_encounter_refinement_runner,validate_encounter_refinement
    registry.register(StageDefinition('native_encounter_refinement',native_encounter_refinement_runner,
        (Path(__file__),ROOT/'core/stream/native_encounters.py',ROOT/'core/evolution.py',ROOT/'core/streams.py',ROOT/'core/cosmology.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('encounters','stream_history','run_dir','checkpoint_directories','ranks','omp_threads','maximum_branches','maximum_outputs','maximum_timestep','memory_gib','disk_gib','wall_hours'),validator=validate_encounter_refinement))
    from .stream.native_encounters import native_encounter_display_runner,validate_encounter_display
    registry.register(StageDefinition('native_encounter_display',native_encounter_display_runner,
        (Path(__file__),ROOT/'core/stream/native_encounters.py',ROOT/'core/research/generator_controls.py',ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('stream_history','perturber_history','encounters','memory_gib','maximum_objects','maximum_vertices','frame_stride'),validator=validate_encounter_display))
    from .matched_perturber_figures import matched_template_figures_runner,validate_matched_template_figures
    registry.register(StageDefinition('matched_perturber_figures',matched_template_figures_runner,
        (Path(__file__),ROOT/'core/matched_perturber_figures.py',ROOT/'core/streams.py',ROOT/'core/figures.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'configs/runtime/orbit_interpolation.json'),
        required_parameters=('product','memory_gib','maximum_pairs','dpi'),validator=validate_matched_template_figures))
    from .native_stream_responses import native_response_runner,validate_native_response
    registry.register(StageDefinition('native_stream_responses',native_response_runner,
        (Path(__file__),ROOT/'core/native_stream_responses.py',ROOT/'core/streams.py',ROOT/'core/sidm.py',ROOT/'core/halos.py',ROOT/'core/stars.py',ROOT/'core/cosmology.py',ROOT/'core/research/generator_controls.py',ROOT/'core/flagship_products.py',ROOT/'core/observations.py',ROOT/'core/figures.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request_path','request_sha256','memory_gib'),validator=validate_native_response))
    from .sidm_products import native_sidm_export_runner
    sidm_export_sources=(Path(__file__),ROOT/'core/sidm_products.py',ROOT/'core/halo_campaign.py',ROOT/'core/sidm.py',ROOT/'core/observations.py',ROOT/'core/halos.py',ROOT/'core/analysis.py',ROOT/'core/streams.py',ROOT/'core/research/generator_controls.py',ROOT/'core/figures.py',ROOT/'core/black_holes.py',ROOT/'core/stars.py',ROOT/'core/evolution.py',ROOT/'core/cosmology.py',ROOT/'core/data.py',ROOT/'core/foundation.py')
    registry.register(StageDefinition('native_sidm_export',native_sidm_export_runner,sidm_export_sources,
        required_parameters=('operation','product','run_directory','request','memory_gib')))
    from .sidm_recovery import native_recovered_sidm_runner
    registry.register(StageDefinition('native_recovered_sidm',native_recovered_sidm_runner,
        (*sidm_export_sources,ROOT/'core/sidm_recovery.py',ROOT/'core/sidm_campaign.py'),
        required_parameters=('native_run_directory','native_stage_receipt','recovery_path','template_path','template_sha256','matching_identity','memory_gib')))
    from .sidm_recovery import native_recovered_sidm_pair_runner
    registry.register(StageDefinition('native_recovered_sidm_pair',native_recovered_sidm_pair_runner,
        (*sidm_export_sources,ROOT/'core/sidm_recovery.py',ROOT/'core/sidm_campaign.py'),
        required_parameters=('cdm_result','sidm_result','cdm_run_directory','sidm_run_directory','template_path','template_sha256','memory_gib')))
    from .sidm_recovery import native_recovered_satellite_runner
    registry.register(StageDefinition('native_recovered_sidm_satellite',native_recovered_satellite_runner,
        (*sidm_export_sources,ROOT/'core/sidm_recovery.py'),
        required_parameters=('target_result','native_run_directory','native_stage_receipt','template_path','template_sha256','memory_gib')))
    from .sidm_recovery import native_recovered_centroid_runner
    registry.register(StageDefinition('native_recovered_sidm_centroids',native_recovered_centroid_runner,
        (*sidm_export_sources,ROOT/'core/sidm_recovery.py',ROOT/'core/native_gadget_ancestry.py'),
        required_parameters=('target_result','native_run_directory','native_stage_receipt','template_path','template_sha256','memory_gib')))
    from .sidm_forward import native_recovered_lensing_runner
    registry.register(StageDefinition('native_recovered_sidm_lensing',native_recovered_lensing_runner,
        (*sidm_export_sources,ROOT/'core/sidm_forward.py',ROOT/'core/sidm_recovery.py'),
        required_parameters=('pair_result','cdm_result','sidm_result','cdm_run_directory','sidm_run_directory','template_path','template_sha256','memory_gib')))
    from .sidm_forward import native_recovered_stream_response_runner
    registry.register(StageDefinition('native_recovered_sidm_stream_response',native_recovered_stream_response_runner,
        (*sidm_export_sources,ROOT/'core/sidm_forward.py',ROOT/'core/stream/native_encounters.py',ROOT/'core/stream/measurements.py',ROOT/'core/sidm_recovery.py'),
        required_parameters=('pair_result','cdm_stream_history','sidm_stream_history','cdm_stream_receipt','sidm_stream_receipt','cdm_run_directory','sidm_run_directory','template_path','template_sha256','memory_gib')))
    from core.response_campaign import public_ic_runner
    from .expansion_products import hubble_figure_runner,validate_hubble_figure_stage
    from .flagship_inference import inference_figure_runner,validate_inference_figure_stage
    from core.research.display import combined_gallery_runner, validate_combined_gallery_stage
    registry.register(StageDefinition('combined_graph_gallery',combined_gallery_runner,
        (Path(__file__),source_path('research.display'),ROOT/'core/graph_producer_registry.py',ROOT/'configs/research/graph_producers.json',ROOT/'core/halo_campaign.py',ROOT/'core/research/generator_controls.py',ROOT/'core/figures.py',ROOT/'core/blender.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request',),validator=validate_combined_gallery_stage))
    registry.register(StageDefinition('inference_figures',inference_figure_runner,
        (Path(__file__),ROOT/'core/flagship_inference.py',ROOT/'core/expansion_products.py',ROOT/'core/observations.py',ROOT/'core/streams.py',ROOT/'core/research/generator_controls.py',ROOT/'core/halo_campaign.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','memory_gib'),validator=validate_inference_figure_stage))
    registry.register(StageDefinition('hubble_figures',hubble_figure_runner,
        (Path(__file__),ROOT/'core/expansion_products.py',ROOT/'core/bao.py',ROOT/'core/halo_campaign.py',ROOT/'core/observations.py',ROOT/'core/streams.py',ROOT/'core/research/generator_controls.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request','memory_gib'),validator=validate_hubble_figure_stage))
    from .response_campaign import native_response_target_runner
    registry.register(StageDefinition('native_response_target',native_response_target_runner,
        (*sidm_export_sources,ROOT/'core/response_campaign.py',ROOT/'core/sidm_recovery.py',ROOT/'core/sidm_campaign.py',ROOT/'core/zoom.py'),
        required_parameters=('native_run_directory','native_stage_receipt','baseline_run_directory','baseline_stage_receipt','template_path','template_sha256','memory_gib')))
    registry.register(StageDefinition('native_public_ic',public_ic_runner,
        (Path(__file__),source_path('response_campaign'),ROOT/'core/evolution.py',ROOT/'core/analysis.py',ROOT/'core/foundation.py',ROOT/'core/data.py'),
        required_parameters=('source_run','source_receipt','phase_seed','memory_gib')))
    from .halo_campaign import native_web_epoch_runner,native_web_history_runner
    web_sources=(Path(__file__),ROOT/'core/halo_campaign.py',ROOT/'core/halo_products.py',ROOT/'core/analysis.py',ROOT/'core/halos.py',ROOT/'core/cosmology.py',ROOT/'core/observations.py',ROOT/'core/streams.py',ROOT/'core/research/generator_controls.py',ROOT/'core/figures.py',ROOT/'core/data.py',ROOT/'core/foundation.py')
    registry.register(StageDefinition('native_web_epoch',native_web_epoch_runner,web_sources,
        required_parameters=('run_directory','snapshot','halo_structure_product','filament_skeleton','design_path','design_sha256','memory_gib')))
    registry.register(StageDefinition('native_web_history',native_web_history_runner,web_sources,
        required_parameters=('identity_graph_directory','observations','maximum_interval_myr','memory_gib')))
    from .native_baryon_campaign import native_baryon_figures_runner,native_cooling_births_runner
    baryon_sources=(Path(__file__),ROOT/'core/native_baryon_campaign.py',ROOT/'core/research/generator_controls.py',ROOT/'core/figures.py',ROOT/'core/stellar_returns.py',ROOT/'core/black_hole_ledger.py',ROOT/'configs/hydro_audit.algorithms.json',ROOT/'core/halo_campaign.py',ROOT/'core/baryon_products.py',ROOT/'core/stellar_population.py',ROOT/'core/first_stars.py',ROOT/'core/flagship_bh.py',ROOT/'core/observations.py',ROOT/'core/stars.py',ROOT/'core/black_holes.py',ROOT/'core/halos.py',ROOT/'core/analysis.py',ROOT/'core/cosmology.py',ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py')
    registry.register(StageDefinition('native_baryon_figures',native_baryon_figures_runner,baryon_sources,
        required_parameters=('hydro_history','memory_gib','design_path','design_sha256')))
    registry.register(StageDefinition('native_cooling_births',native_cooling_births_runner,baryon_sources,
        required_parameters=('hydro_history','memory_gib','design_path','design_sha256','identity_graph','assembly_product')))
    from .native_bh_campaign import native_bh_history_runner,native_bh_event_figures_runner,native_bh_response_runner
    bh_sources=(Path(__file__),ROOT/'core/native_bh_campaign.py',ROOT/'core/flagship_bh.py',ROOT/'core/flagship_bh_response.py',ROOT/'core/periodic_images.py',ROOT/'core/black_holes.py',ROOT/'core/stars.py',ROOT/'core/research/generator_controls.py',ROOT/'core/halos.py',ROOT/'core/analysis.py',ROOT/'core/data.py',ROOT/'core/streams.py',ROOT/'core/foundation.py')
    registry.register(StageDefinition('native_bh_history',native_bh_history_runner,bh_sources,
        required_parameters=('run_directory','maximum_saved_epochs','memory_gib')))
    registry.register(StageDefinition('native_bh_event_figures',native_bh_event_figures_runner,bh_sources,
        required_parameters=('run_directory','history_directory','hydro_history','memory_gib','pair_ids','recoil_selection')))
    registry.register(StageDefinition('native_bh_matched_response',native_bh_response_runner,bh_sources,
        required_parameters=('pair_request_path','pair_request_sha256','joint_covariance','memory_gib')))
    from .flagship_bh_response import native_bh_covariance_runner
    registry.register(StageDefinition('native_bh_response_covariance',native_bh_covariance_runner,
        (Path(__file__),ROOT/'core/flagship_bh_response.py',ROOT/'core/halos.py',ROOT/'core/analysis.py',ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('request_path','request_sha256','memory_gib')))
    from .halo_campaign import native_assembly_history_runner
    registry.register(StageDefinition('native_assembly_history',native_assembly_history_runner,
        (Path(__file__),ROOT/'core/halo_campaign.py',ROOT/'core/assembly.py',ROOT/'core/halos.py',ROOT/'core/observations.py',ROOT/'core/research/generator_controls.py',ROOT/'core/analysis.py',ROOT/'core/data.py',ROOT/'core/streams.py',ROOT/'core/foundation.py'),
        required_parameters=('identity_graph','template_path','template_sha256','memory_gib')))
    from .halo_campaign import native_assembly_measurements_runner, validate_assembly_measurements
    registry.register(StageDefinition('native_assembly_measurements',native_assembly_measurements_runner,
        (Path(__file__),ROOT/'core/halo_campaign.py',ROOT/'core/halos.py',ROOT/'core/analysis.py',ROOT/'core/data.py',ROOT/'core/streams.py',ROOT/'core/foundation.py'),
        required_parameters=('identity_graph','template_path','template_sha256','memory_gib','maximum_nodes','maximum_edges','maximum_measurements'),validator=validate_assembly_measurements))
    from .stream.perturbers import native_perturber_history_runner,validate_perturber_history
    registry.register(StageDefinition('native_perturber_history',native_perturber_history_runner,
        (Path(__file__),ROOT/'core/stream/perturbers.py',ROOT/'core/stream/native_encounters.py',ROOT/'core/halos.py',ROOT/'core/zoom.py',ROOT/'core/observations.py',ROOT/'core/streams.py',ROOT/'core/data.py',ROOT/'core/foundation.py',ROOT/'configs/runtime/orbit_interpolation.json'),
        required_parameters=('run_dir','identity_graph','stream_frame_history','template_path','template_sha256','memory_gib','maximum_particles','maximum_nodes','maximum_tracks'),validator=validate_perturber_history))
    from .apertures import native_encounter_apertures_runner,validate_encounter_apertures
    registry.register(StageDefinition('native_encounter_apertures',native_encounter_apertures_runner,
        (Path(__file__),ROOT/'core/apertures.py',ROOT/'core/streams.py',ROOT/'core/flagship_products.py',ROOT/'core/observations.py',ROOT/'core/zoom.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('run_dir','encounter_product','stream_history','snapshot','memory_gib','maximum_apertures'),validator=validate_encounter_apertures))
    from .apertures import recovered_apertures_runner,validate_recovered_apertures
    registry.register(StageDefinition('native_recovered_apertures',recovered_apertures_runner,
        (Path(__file__),ROOT/'core/apertures.py',
         ROOT/'core/streams.py',ROOT/'core/stars.py',ROOT/'core/halos.py',ROOT/'core/observations.py',ROOT/'core/zoom.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('run_dir','recovery_path','controls_path','controls_sha256','snapshots','memory_gib','maximum_particles','maximum_apertures','maximum_coarse_mass_fraction'),validator=validate_recovered_apertures))
    from core.research.restart_campaign import campaign_restart_comparison_runner,validate_campaign_restart_comparison
    from core.research.restart_campaign import restart_exercise_runner,validate_restart_exercise
    registry.register(StageDefinition('campaign_restart_exercise',restart_exercise_runner,
        (Path(__file__),ROOT/'commands.py',ROOT/'core/research/restart_campaign.py',ROOT/'core/research/checkpoints.py',ROOT/'core/research/restarts.py',ROOT/'core/evolution.py',ROOT/'core/data.py',ROOT/'core/foundation.py',
         *tuple(sorted((ROOT/'configs/research').glob('native_restart*.layout.json'))),
         *tuple(p for p in (ROOT/'core/software/gadget4/src').rglob('*') if p.is_file()),
         *tuple(p for p in (ROOT/'core/software/gadget4/buildsystem').rglob('*.py') if p.is_file()),ROOT/'core/software/gadget4/Template-Config.sh'),
        required_parameters=('answers_path','existing_ic','generation_report','template_path','template_sha256','snapshot','pause_step','memory_gib','wall_seconds'),validator=validate_restart_exercise))
    registry.register(StageDefinition('campaign_restart_comparison',campaign_restart_comparison_runner,
        (Path(__file__),ROOT/'core/research/restart_campaign.py',ROOT/'core/research/checkpoints.py',ROOT/'core/research/restarts.py',ROOT/'core/evolution.py',ROOT/'core/data.py',ROOT/'core/foundation.py',
         *tuple(sorted((ROOT/'configs/research').glob('native_restart*.layout.json')))),
        required_parameters=('uninterrupted_run','resumed_run','snapshot','template_path','template_sha256','memory_gib'),validator=validate_campaign_restart_comparison))
    registry.register(StageDefinition('science_aperture_contamination',aperture_contamination_runner,
        (Path(__file__),ROOT/'core/apertures.py',ROOT/'core/zoom.py',ROOT/'core/halos.py',ROOT/'core/data.py',ROOT/'core/foundation.py'),
        required_parameters=('run_dir','aperture_product','memory_gib','maximum_apertures','maximum_coarse_mass_fraction'),validator=validate_aperture_contamination))
    registry.register(StageDefinition('command', command_runner, (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/foundation.py'), required_parameters=('argv', 'wall_seconds')))
    registry.register(StageDefinition('public_command', public_command_runner,
        (Path(__file__), ROOT/'commands.py', *tuple(sorted((ROOT/'core').glob('*.py')))), required_parameters=('command', 'wall_seconds')))
    from core.research.errors import execute_budget_stage
    registry.register(StageDefinition('error_budget', execute_budget_stage,
        (Path(__file__), ROOT/'core/research/generator_controls.py', ROOT/'core/research/generator_controls.py'),
        required_parameters=('request', 'measurements')))
    registry.register(StageDefinition('error_ensemble',error_ensemble_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py'),
        required_parameters=('request','realizations','systematic_controls','independence_key',
            'maximum_realizations','memory_bytes','request_base','destination')))
    from core.research.errors import native_residual_runner
    from .expansion_products import expansion_acceptance_runner
    registry.register(StageDefinition('expansion_acceptance',expansion_acceptance_runner,
        (Path(__file__),ROOT/'core/expansion_products.py',ROOT/'core/foundation.py',ROOT/'core/research/errors.py'),
        required_parameters=('request','comparison','request_base','memory_bytes')))
    from .matched import matched_hydro_measurement_runner
    registry.register(StageDefinition('matched_hydro_measurement',matched_hydro_measurement_runner,
        (Path(__file__),ROOT/'core/matched.py',ROOT/'core/observations.py',ROOT/'core/stars.py',ROOT/'core/data.py'),
        required_parameters=('request','destination')))
    from core.research.findings import repair_receipt_runner, validate_repair_parameters
    registry.register(StageDefinition('finding_repair_receipt',repair_receipt_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/data.py',ROOT/'core/research/generator_controls.py'),
        required_parameters=('finding_store','finding_id','product_id','finding_event_hash','dependency_graph_hash',
            'requirement','measurements','memory_bytes','request_base','destination'),validator=validate_repair_parameters))
    from core.research.pilots import native_galaxy_measurement_runner, validate_native_galaxy_measurement
    registry.register(StageDefinition('native_pilot_galaxy_measurement',native_galaxy_measurement_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/halos.py',ROOT/'core/data.py'),
        required_parameters=('request','native_run_directory','native_stage_receipt','destination'),validator=validate_native_galaxy_measurement))
    from .flagship_history import material_map_runner, material_recovery_runner, validate_material_map, validate_material_recovery, contamination_history_runner, validate_contamination_history
    history_sources = (Path(__file__), ROOT/'core/flagship_history.py', ROOT/'core/research/generator_controls.py', ROOT/'core/zoom.py', ROOT/'core/halos.py', ROOT/'core/data.py')
    registry.register(StageDefinition('zoom_material_map', material_map_runner, history_sources,
        required_parameters=('selected_particles','zoom_ic','generation_report','parent_level','box_comoving_kpc_h','memory_gib','maximum_tracers'), validator=validate_material_map))
    registry.register(StageDefinition('zoom_material_recovery', material_recovery_runner, history_sources,
        required_parameters=('run_dir','material_map','snapshots','rules','requested_redshifts','maximum_redshift_error','memory_gib'), validator=validate_material_recovery))
    registry.register(StageDefinition('zoom_contamination_history', contamination_history_runner, history_sources,
        required_parameters=('run_dir','material_map','recovery_path','aperture_r200c_factor','maximum_coarse_mass_fraction','memory_gib'), validator=validate_contamination_history))
    from .halo_campaign import recovered_structure_runner,validate_recovered_structure
    registry.register(StageDefinition('zoom_recovered_structure',recovered_structure_runner,
        (*history_sources,ROOT/'core/halo_campaign.py',ROOT/'core/halo_products.py',ROOT/'core/figures.py',ROOT/'core/foundation.py'),
        required_parameters=('run_dir','recovery_path','template_path','template_sha256','snapshot','memory_gib','export_figures'),validator=validate_recovered_structure))
    from core.research.zooms import native_zoom_diagnostic_runner, validate_zoom_diagnostic_request
    def validate_zoom_diagnostic_stage(parameters):
        validate_zoom_diagnostic_request(parameters['request'])
    registry.register(StageDefinition('native_zoom_diagnostics',native_zoom_diagnostic_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py',ROOT/'core/zoom.py',ROOT/'core/data.py'),
        required_parameters=('run_dir','recovery_path','request','request_base','destination'),validator=validate_zoom_diagnostic_stage))
    from core.research.controls import physical_collision_runner, validate_collision_stage
    physical_collision_sources=(Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py',
        ROOT/'core/evolution.py',ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/foundation.py',
        *tuple(path for path in (ROOT/'core/software/gadget4/src').rglob('*') if path.is_file()),
        *tuple(path for path in (ROOT/'core/software/gadget4/buildsystem').rglob('*') if path.is_file() and '__pycache__' not in path.parts),
        ROOT/'core/software/gadget4/Makefile',ROOT/'core/software/gadget4/Template-Config.sh')
    registry.register(StageDefinition('isolated_physical_collision',physical_collision_runner,physical_collision_sources,
        required_parameters=('request','base_configuration'),validator=validate_collision_stage))
    from .sidm_campaign import native_sidm_resources_runner,validate_sidm_resources
    registry.register(StageDefinition('native_sidm_resources',native_sidm_resources_runner,
        (Path(__file__),ROOT/'core/sidm_campaign.py',ROOT/'core/sidm.py',ROOT/'core/research/generator_controls.py'),
        required_parameters=('request','native_stage_receipt'),validator=validate_sidm_resources))
    from .apertures import native_zoom_support_runner, validate_zoom_support
    registry.register(StageDefinition('native_sidm_zoom_support',native_zoom_support_runner,
        (Path(__file__),ROOT/'core/apertures.py',ROOT/'core/sidm.py',ROOT/'core/data.py'),
        required_parameters=('request','native_run_directory','native_stage_receipt','destination'),validator=validate_zoom_support))
    from core.research.measurements import isolated_observable_runner, validate_isolated_observable
    registry.register(StageDefinition('isolated_observable',isolated_observable_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/sidm.py',
         ROOT/'core/research/generator_controls.py',ROOT/'core/data.py'),
        required_parameters=('observable_definition','history','memory_bytes','maximum_bins','request_base','destination'),
        validator=validate_isolated_observable))
    from core.research.errors import native_error_envelope_runner, validate_envelope_parameters
    registry.register(StageDefinition('native_error_envelope',native_error_envelope_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py',
         ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py'),
        required_parameters=('request','component','control','bound_scope','pairs',
            'memory_bytes','maximum_pairs','request_base','destination'),
        validator=validate_envelope_parameters))
    registry.register(StageDefinition('native_control_residual',native_residual_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py',
         ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py'),
        required_parameters=('request','pairs','realization_id','native_realization_receipt',
            'independence_key','memory_bytes','maximum_components','request_base','destination')))
    registry.register(StageDefinition('display_products',display_products_runner,
        (Path(__file__),source_path('research.display'),ROOT/'core/blender_scene.py',ROOT/'core/data.py'),
        required_parameters=('request','request_base','destination','memory_gib')))
    from core.research.measurements import convergence_measurement_runner, validate_measurement_request
    from core.research.convergence import convergence_figure_runner, convergence_phase_runner
    convergence_sources = (Path(__file__), source_path('research.convergence'),
        ROOT/'core/flagship_products.py', ROOT/'core/research/generator_controls.py', ROOT/'core/data.py',
        ROOT/'core/zoom.py', ROOT/'core/foundation.py', ROOT/'core/observations.py',
        ROOT/'core/halos.py', ROOT/'core/halo_campaign.py')
    registry.register(StageDefinition('native_convergence_figures', convergence_figure_runner,
        convergence_sources, required_parameters=('request', 'measurement_receipts', 'memory_gib')))
    registry.register(StageDefinition('native_convergence_phase_pair', convergence_phase_runner,
        convergence_sources, required_parameters=('control', 'baseline_generation', 'current_generation',
            'baseline_receipt', 'current_receipt', 'memory_gib')))
    from .native_convergence_catalogue import catalogue_profile_runner
    registry.register(StageDefinition('native_convergence_catalogue',catalogue_profile_runner,
        (*convergence_sources,ROOT/'core/native_convergence_catalogue.py',ROOT/'core/analysis.py',ROOT/'core/sidm.py'),
        required_parameters=('request','selection','native_run_directory','native_stage_receipt','paired_reference')))
    def validate_measurement_stage(parameters):
        validate_measurement_request(parameters['request'])
    registry.register(StageDefinition('convergence_measurement',convergence_measurement_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/observations.py',
         ROOT/'core/data.py',ROOT/'core/research/generator_controls.py',ROOT/'core/halos.py'),
        required_parameters=('request','variant_metadata','native_stage_receipt'),
        validator=validate_measurement_stage))
    from .sidm_control_measurements import isolated_history_runner, angular_control_runner, validate_isolated_history_request, validate_angular_control_stage
    def validate_isolated_history_stage(parameters):
        validate_isolated_history_request(parameters['request'])
    registry.register(StageDefinition('isolated_control_history',isolated_history_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/data.py',ROOT/'core/halos.py'),
        required_parameters=('request','native_run_directory','native_stage_receipt'),
        validator=validate_isolated_history_stage))
    registry.register(StageDefinition('sidm_angular_control',angular_control_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/data.py'),
        required_parameters=('request',),validator=validate_angular_control_stage))
    from .sidm_control_products import reference_cdf_runner, native_events_runner, validate_cdf_request, validate_events_request
    def validate_cdf_stage(parameters):
        validate_cdf_request(parameters['request'])
    def validate_events_stage(parameters):
        validate_events_request(parameters['request'])
    registry.register(StageDefinition('sidm_reference_cdf',reference_cdf_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/sidm.py',ROOT/'core/data.py'),
        required_parameters=('request',),validator=validate_cdf_stage))
    registry.register(StageDefinition('sidm_native_events',native_events_runner,
        (Path(__file__),ROOT/'core/sidm.py',ROOT/'core/sidm.py',ROOT/'core/data.py'),
        required_parameters=('request','native_stage_receipt'),validator=validate_events_stage))
    from core.research.convergence import native_configuration_runner, validate_native_configuration
    registry.register(StageDefinition('native_control_configuration',native_configuration_runner,
        (Path(__file__),ROOT/'core/research/generator_controls.py',ROOT/'core/research/generator_controls.py'),
        required_parameters=('mode','baseline_input','baseline_sha256','variant_metadata','bindings','memory_bytes'),
        validator=validate_native_configuration))
    definitions = [
        ('convergence', 'core.research.convergence', 'compare_campaign', 'convergence.json', ('request','measurements')),
        ('resources', 'core.resources', 'estimate_production_resources', 'resources.json', ('spec',)),
        ('resource_comparison', 'core.resources', 'publish_resource_comparison', 'resource-comparison-stage.json', ('plan','receipts','destination')),
        ('build_capability', 'core.research.capabilities', 'publish_native_build_capability', 'capability.json',
            ('destination','name','executable','compiled_features','profile','source_hashes','invocation_receipt','domain')),
        ('native_pilot_population', 'core.research.pilots', 'native_population_history', 'population-history.json', ('snapshot_sets','memory_gib','maximum_snapshots','base')),
        ('zoom_campaign', 'core.research.zooms', 'run_zoom_campaign', 'zoom-campaign.json', ('request','destination')),
        ('scene_package','core.blender_scene','package_display_products','display-package.json', ('manifest_path','destination')),
    ]
    for kind, module, function, filename, required in definitions:
        registry.register(StageDefinition(kind, json_function_runner(module, function, filename),
            (Path(__file__), ROOT/(module.replace('.', '/')+'.py')), required_parameters=required))
    from .flagship_production_campaigns import production_admission_runner,validate_production_admission_contract
    registry.register(StageDefinition('flagship_production_admission',production_admission_runner,
        (Path(__file__),ROOT/'core/flagship_production_campaigns.py',ROOT/'core/foundation.py',ROOT/'core/zoom.py'),
        required_parameters=('request',),validator=validate_production_admission_contract))
    from .sidm_analysis import register_sidm_stages
    register_sidm_stages(registry)
    return registry


def workflow_main():
    import argparse
    parser = argparse.ArgumentParser(description='Execute a capability-aware scientific product DAG')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--answers', type=Path)
    source.add_argument('--campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    request = args.answers or args.campaign
    result = execute_workflow(json.loads(request.read_text()), args.output, request_base=request.parent,
                              resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if args.dry_run or result['execution_status'] == 'COMPLETE' else 1


def run_spec_main():
    """Author canonical migrated specifications and automatic product plans."""
    import argparse
    from .foundation import validate_run_spec_v2,migrate_legacy_run_spec,atomic_text
    parser=argparse.ArgumentParser(description='Canonical v2 science specification and recorded v1 migration')
    parser.add_argument('mode',choices=('plan','migrate','catalogue'))
    parser.add_argument('request',type=Path,nargs='?')
    parser.add_argument('--explicit',type=Path,help='Required explicit run fields when migrating a v1 science specification')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='catalogue':
        if args.request is not None or args.explicit is not None:parser.error('catalogue publishes registered definitions without a run request')
        result={'schema':'research2-standard-science-product-catalogue-v1','producers':STANDARD_PRODUCT_CATALOGUE,
            'source':{'path':str(Path(__file__).resolve()),'sha256':file_sha256(__file__)},'scientific_inputs':'explicit product_options; no missing physics or calibration fabricated'}
    elif args.request is None:parser.error('plan and migrate require a saved run request')
    elif args.mode=='migrate':
        source=args.request.expanduser().resolve();request=json.loads(source.read_text())
        if args.explicit is None:parser.error('migration requires --explicit; no epochs/physics are guessed')
        result=migrate_legacy_run_spec(request,json.loads(args.explicit.read_text()))
    else:
        source=args.request.expanduser().resolve();request=json.loads(source.read_text())
        result=compile_run_spec_v2(validate_run_spec_v2(request),request_base=source.parent)
    if args.output.exists():raise FileExistsError('Keep previous canonical specification/plan')
    atomic_json(args.output,result)
    atomic_text(args.output.with_name(args.output.name+'.sha256'),file_sha256(args.output)+'\n')
    return 0


def materialize_request_runner(params, context):
    """Save an immutable native/analysis request after resolving DAG references."""
    value = params['request']
    if not isinstance(value, dict):
        raise ValueError('Materialized request must be a JSON object')
    path = context.generation/params.get('filename', 'request.json')
    if not path.resolve().is_relative_to(context.generation):
        raise ValueError('Request filename escapes stage generation')
    atomic_json(path, value)
    return {}


# Workflow products
READERS = {'native_snapshot_series','native_hdf5_set','numeric_audit','model_arrays',
           'stream_product','measured_display','blender_scene'}


def _numeric_audit_workspace(path, memory_gib):
    """Check expanded numeric bytes before the shared reader allocates arrays."""
    import math
    import zipfile
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Scientific product intake requires a finite positive memory allocation')
    archive = Path(path).with_name(Path(path).name+'.data')/'arrays.npz'
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        names = [row.filename for row in members]
        if len(set(names)) != len(names) or any(not name.endswith('.npy') or '/' in name or '\\' in name for name in names):
            raise ValueError('Scientific numeric product needs distinct flat numeric NPY members')
        if sum(row.file_size for row in members)*8 > memory_gib*1024**3:
            raise MemoryError('Expanded scientific arrays and reader workspace exceed the declared stage allocation')
    return archive


def validate_product_contract_definitions(stage):
    outputs = stage['outputs']
    contracts = stage.get('product_contracts', {})
    if not isinstance(contracts, dict) or not set(contracts) <= set(outputs):
        raise ValueError('Scientific contract names must refer to declared stage products')
    for name, descriptor in contracts.items():
        if descriptor.get('reader') not in READERS:
            raise ValueError('Unknown existing scientific-product reader: '+name)
        if descriptor['reader'] == 'native_snapshot_series':
            snapshots = descriptor['snapshots']
            if not snapshots or len(set(snapshots)) != len(snapshots) or any(type(q) is not int or q < 0 for q in snapshots):
                raise ValueError('Native workflow products require an explicit distinct snapshot schedule')
        if descriptor['reader'] == 'native_hdf5_set':
            files = descriptor['files']
            if not files or len(set(files)) != len(files):
                raise ValueError('Native workflow product requires its complete distinct shard list')
            for relative in files:
                p = Path(relative)
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError('Native product shard reference escapes its generation')
    return contracts


def read_stage_product(product_path, descriptor, *, generation, memory_gib):
    path = Path(product_path).resolve()
    root = Path(generation).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Scientific workflow product escapes its generation')
    reader = descriptor['reader']
    bindings = {}
    if reader == 'native_snapshot_series':
        from core.halos import snapshot_sets
        from core.data import verify_native_hdf5_set
        sets = snapshot_sets(path)
        for snapshot in descriptor['snapshots']:
            shards = sets.get(snapshot, [])
            if not shards:
                raise ValueError('Missing requested published native snapshot: '+str(snapshot))
            if any(not Path(shard).resolve().is_relative_to(root) for shard in shards):
                raise ValueError('Native snapshot-series shard escaped its workflow generation')
            verify_native_hdf5_set(shards, require_publication=True)
            bindings.update({str(Path(q).resolve()):file_sha256(q) for q in shards})
    elif reader == 'native_hdf5_set':
        from core.data import verify_native_hdf5_set
        shards = [(root/relative).resolve() for relative in descriptor['files']]
        if any(not q.is_relative_to(root) for q in shards):
            raise ValueError('Native scientific product shard escaped generation')
        verify_native_hdf5_set(shards, require_publication=True)
        bindings = {str(q):file_sha256(q) for q in shards}
    elif reader == 'numeric_audit':
        from core.data import verify_numeric_audit, numeric_audit_arrays_path
        _numeric_audit_workspace(path,memory_gib)
        report = verify_numeric_audit(path)
        archive = numeric_audit_arrays_path(path,report)
        if archive.stat().st_size > memory_gib*1024**3:
            raise MemoryError('Numeric product archive exceeds declared stage intake budget')
        bindings = {str(path):file_sha256(path),str(archive):file_sha256(archive)}
    elif reader == 'model_arrays':
        from core.data import verify_model_arrays
        verify_model_arrays(path)
        bindings = {str(p):file_sha256(p) for p in path.iterdir() if p.is_file()}
    elif reader == 'stream_product':
        from core.stream_common import read_product
        report, arrays, sources = read_product(path,memory_gib=memory_gib)
        bindings = dict(sources)
        bindings[str(path)] = file_sha256(path)
        del arrays
    elif reader == 'measured_display':
        from core.blender_scene import load_verified_display_products
        report = load_verified_display_products(path)
        bindings = {str(path):file_sha256(path)}
        bindings.update({str(path.parent/q['path']):q['sha256'] for q in report.get('assets',[])})
    elif reader == 'blender_scene':
        from core.blender import verify_scene_publication
        verify_scene_publication(path)
        bindings = {str(path):file_sha256(path)}
    else:
        raise ValueError('Unknown scientific workflow product reader')
    return {'reader':reader,'status':'INTAKE_VERIFIED','source_hashes':bindings,
            'scope':'existing structural/semantic/payload contract; convergence acceptance remains separate'}


def stage_result_from_receipt(receipt):
    """Adapt the persisted workflow result to the shared immutable StageResult."""
    from core.foundation import ProductRef, StageResult
    path = Path(receipt['generation'])/'stage-receipt.json'
    status = receipt['execution_status']
    if status == 'INTERRUPTED':
        status = 'INCOMPLETE'
    if status == 'RUNNING':
        raise ValueError('Running stage is not a final shared StageResult')
    refs = []
    if status == 'COMPLETE':
        stored = verify_completed_receipt(path)
        if stored != receipt:
            raise ValueError('Shared StageResult must come from the exact retained receipt, not edited metadata')
        digest = file_sha256(path)
        for product in sorted(receipt['products']):
            refs.append(ProductRef(receipt['stage_id']+':'+product,path,digest,1))
    return StageResult(status,tuple(refs),tuple(receipt.get('findings',[])),tuple(receipt.get('validation_receipts',[])))


def reader_source_paths(contracts):
    """Include semantic-reader implementations in a stage's code fingerprint."""
    owners = {
        'native_snapshot_series':('data','halos','analysis','cosmology'),
        'native_hdf5_set':('data',), 'numeric_audit':('data',),
        'model_arrays':('data',), 'stream_product':('stream_common','data'),
        'measured_display':('blender_scene',), 'blender_scene':('blender',)}
    names = set()
    for descriptor in contracts.values():
        names.update(owners[descriptor['reader']])
    return tuple(source_path(name).resolve() for name in sorted(names))


def load_workflow_stage_results(destination):
    """Return shared ProductRef/StageResult objects for retained stage products."""
    import json
    root = Path(destination).resolve()
    state = json.loads((root/'workflow.json').read_text())
    if state.get('schema') != 'research-workflow-receipt-v1':
        raise ValueError('Expected a retained research workflow receipt')
    results = {}
    for stage_id, reference in state['stages'].items():
        safe_id(stage_id)
        generation = Path(reference['generation']).resolve()
        if not generation.is_relative_to(root/'generations'/stage_id):
            raise ValueError('Workflow stage product pointer escaped its owner generation')
        path = generation/'stage-receipt.json'
        if file_sha256(path) != reference['receipt_sha256']:
            raise ValueError('Workflow stage receipt pointer changed')
        receipt = json.loads(path.read_text())
        if receipt['stage_id'] != stage_id or receipt['fingerprint'] != reference['fingerprint']:
            raise ValueError('Workflow product reference has another stage/configuration identity')
        results[stage_id] = stage_result_from_receipt(receipt)
    return results


def workflow_products_main():
    import argparse
    import json
    p = argparse.ArgumentParser(description='Read retained workflow products through shared science contracts')
    p.add_argument('workflow',type=Path)
    args = p.parse_args()
    result = load_workflow_stage_results(args.workflow)
    print(json.dumps({sid:{'execution_status':row.execution_status,
        'products':[{'product_id':q.product_id,'manifest_path':str(q.manifest_path),
                     'content_hash':q.content_hash,'schema_version':q.schema_version} for q in row.products],
        'science_acceptance':'READ_PRODUCT_MANIFEST_AND_CURRENT_DOMAIN_RECEIPTS'} for sid,row in result.items()},indent=2))
