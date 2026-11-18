"""Decide which workflows their declared physics and saved inputs can support.

Build support requires the recorded executable, compile options and input
fields. Saved-run selection checks actual particle species, epochs and fields.
These checks establish input availability; they do not establish accuracy or
convergence of the underlying physical model.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import math
import os
from core.foundation import file_sha256, canonical_digest

@dataclass(frozen=True)
class Capability:
    """One software feature, its physical domain and unchanged build assets."""
    name: str
    executable: Path | None = None
    executable_sha256: str | None = None
    features: frozenset[str] = frozenset()
    assets: tuple[tuple[str, str], ...] = ()
    domain: Mapping = field(default_factory=dict)
    profile_hash: str | None = None
    build_receipt: Path | None = None
    build_receipt_sha256: str | None = None

    def check(self):
        native = self.name.startswith(('gadget4', 'opengadget3', 'music', 'monofonic', 'class', 'axiclass', 'rockstar', 'velociraptor', 'blender'))
        if native and (self.executable is None or self.build_receipt is None):
            raise ValueError('Native capability requires an exact executable and retained build receipt: '+self.name)
        if self.build_receipt is not None:
            import json
            if not self.build_receipt_sha256 or file_sha256(self.build_receipt) != self.build_receipt_sha256:
                raise ValueError('Native build receipt changed: '+self.name)
            proof = json.loads(self.build_receipt.read_text())
            if proof.get('schema') != 'native-build-capability-v1' or proof.get('execution_status') != 'COMPLETE':
                raise ValueError('Executed native build capability receipt required')
            if proof.get('name') != self.name:
                raise ValueError('Native build receipt belongs to another capability')
            if proof.get('executable_sha256') != self.executable_sha256:
                raise ValueError('Capability build receipt identifies a different executable')
            if set(proof.get('compiled_features', [])) != set(self.features):
                raise ValueError('Claimed capability features differ from the exact build receipt')
            if self.profile_hash != proof.get('profile_hash') or not self.profile_hash:
                raise ValueError('Native capability requires the exact compiled profile hash')
            if canonical_digest(proof.get('profile')) != self.profile_hash:
                raise ValueError('Native build receipt profile content differs from its hash')
            if dict(proof.get('domain', {})) != dict(self.domain):
                raise ValueError('Capability domain differs from the actual retained build receipt')
            invocation = proof.get('invocation_receipt', {})
            if not invocation.get('path') or not invocation.get('sha256'):
                raise ValueError('Capability receipt omits the actual build invocation')
            invocation_path = Path(invocation['path'])
            if not invocation_path.is_absolute():
                invocation_path = self.build_receipt.parent/invocation_path
            if file_sha256(invocation_path) != invocation['sha256']:
                raise ValueError('Native build invocation receipt changed')
            actual = json.loads(invocation_path.read_text())
            if actual.get('schema') != 'native-command-invocation-v1' or actual.get('status') != 'RETURNED' or actual.get('returncode') != 0:
                raise ValueError('Capability build invocation did not complete successfully')
            if not proof.get('source_hashes'):
                raise ValueError('Native build capability source bindings missing')
            for source, expected in proof['source_hashes'].items():
                p = Path(source)
                if not p.is_absolute():
                    p = self.build_receipt.parent/p
                if file_sha256(p) != expected:
                    raise ValueError('Native capability build source changed')
        if self.executable is not None:
            if not os.access(self.executable, os.X_OK) or not self.executable_sha256 or file_sha256(self.executable) != self.executable_sha256:
                raise ValueError('Missing or changed capability executable: ' + self.name)
        for path, expected in self.assets:
            if file_sha256(path) != expected:
                raise ValueError('Missing or changed capability asset: ' + path)
        return self

    def require_domain(self, parameters):
        """Compare values in the domain's declared units; never convert implicitly."""
        for key, bounds in self.domain.items():
            if key not in parameters:
                raise ValueError('Capability domain requires parameter: ' + key)
            value = parameters[key]
            if isinstance(bounds, dict):
                if not bounds or not {'minimum', 'maximum'} & set(bounds) or set(bounds)-{'minimum', 'maximum', 'units'}:
                    raise ValueError('Explicit numeric capability-domain bounds required: '+key)
                if 'units' in bounds and (not isinstance(bounds['units'], str) or not bounds['units'].strip()):
                    raise ValueError('Capability domain units must be a nonempty string: '+key)
                for endpoint in ('minimum', 'maximum'):
                    if endpoint in bounds and (isinstance(bounds[endpoint], bool) or not isinstance(bounds[endpoint], (int, float)) or not math.isfinite(bounds[endpoint])):
                        raise ValueError('Capability bounds must be finite numbers: '+key)
                if 'minimum' in bounds and 'maximum' in bounds and bounds['minimum'] > bounds['maximum']:
                    raise ValueError('Capability domain interval is reversed: '+key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Nonfinite numerical-domain input: ' + key)
                if 'minimum' in bounds and value < bounds['minimum'] or 'maximum' in bounds and value > bounds['maximum']:
                    raise ValueError('Outside declared capability domain: ' + key)
            else:
                if not isinstance(bounds, (list, tuple)) or not bounds:
                    raise ValueError('Explicit categorical capability domain required: '+key)
                if value not in bounds:
                    raise ValueError('Unsupported categorical-domain input: ' + key)

class CapabilityRegistry:
    def __init__(self, entries=()):
        self.entries = {}
        for entry in entries:
            self.add(entry)

    def add(self, capability):
        if capability.name in self.entries:
            raise ValueError('Duplicate capability: ' + capability.name)
        self.entries[capability.name] = capability

    @classmethod
    def from_spec(cls, records, base):
        entries = []
        base = Path(base).resolve()
        for record in records:
            def path(p):
                q = Path(p).expanduser()
                return q.resolve() if q.is_absolute() else (base/q).resolve()
            entries.append(Capability(record['name'], path(record['executable']) if record.get('executable') else None,
                record.get('executable_sha256'), frozenset(record.get('features', [])),
                tuple((str(path(p['path'])), p['sha256']) for p in record.get('assets', [])),
                record.get('domain', {}), record.get('profile_hash'),
                path(record['build_receipt']) if record.get('build_receipt') else None, record.get('build_receipt_sha256')))
        return cls(entries)

    def require(self, names, parameters=None, features=()):
        for name in names:
            if name not in self.entries:
                raise ValueError('Required capability unavailable: ' + name)
            entry = self.entries[name].check()
            if parameters is not None:
                entry.require_domain(parameters)
            if not set(features) <= entry.features:
                raise ValueError('Required compiled feature missing in ' + name)

    def signature(self, names):
        return canonical_digest([{'name': self.entries[n].name,
            'binary': self.entries[n].executable_sha256,
            'features': sorted(self.entries[n].features), 'assets': self.entries[n].assets,
            'domain': dict(self.entries[n].domain), 'profile': self.entries[n].profile_hash,
            'build_receipt': self.entries[n].build_receipt_sha256}
            for n in sorted(names)])

def _enabled(value):
    """Configuration dictionaries need explicit enabled semantics."""
    if isinstance(value, Mapping):
        value = value.get('enabled', False)
    if type(value) is not bool:
        raise ValueError('Physical feature enabled settings must be booleans')
    return value


def available_species(spec):
    dm = spec.get('dark_matter', {})
    simulation = spec.get('simulation', {})
    physics = spec.get('physics', simulation.get('physics'))
    if isinstance(physics, Mapping):
        mode = physics.get('mode', physics.get('kind'))
        interaction = physics.get('interaction', dm.get('interactions', {}).get('kind'))
    else:
        mode, interaction = physics, dm.get('interactions', {}).get('kind')
    result = {'dm'}
    if mode in {'hydro', 'hydro_bh', 'hydro_stellar', 'hydro_stellar_bh', 'baryonic'}:
        result.add('gas')
    hydro = spec.get('hydro', {})
    star_flag = spec.get('star_formation', simulation.get('star_formation',
        hydro.get('star_formation', False) if isinstance(hydro, Mapping) else False))
    if _enabled(star_flag):
        if 'gas' not in result:
            raise ValueError('Stellar products require a configured gas/star-formation domain')
        result.add('stars')
    if _enabled(spec.get('black_holes', simulation.get('black_holes', False))):
        if 'gas' not in result:
            raise ValueError('Black-hole products require a configured hydro domain')
        result.add('black_holes')
    if interaction == 'elastic_sidm' or _enabled(spec.get('sidm', False)):
        result.add('sidm')
    return result

def inspect_native_feature_support(root, config, species_by_epoch, field_support, sources, snapshot_build_flags=None):
    """Bind declarations, native handoff assets and actual saved field support.

    Presence establishes input availability only. It never certifies a native
    source model, operator accuracy or scientific validity from a marker type.
    """
    import re
    from core.foundation import product_provenance, canonical_digest
    from core.data import coordinate_contract
    provenance = product_provenance(root,algorithm=__file__,selection='saved-run capability inventory',
        source_files=[{'path':path,'sha256':checksum} for path,checksum in sources.items()],
        coordinates=coordinate_contract(position_units='comoving kpc/h',
            velocity_convention='native sqrt(a) storage; physical peculiar velocity requires conversion',
            origin='native simulation origin',periodic_wrapping='saved native images'))
    assets = {name:provenance[name] for name in ('executable','compile_options','parameter_file')}
    for record in assets.values():
        if record['status']=='VERIFIED':sources[record['path']]=record['sha256']
    options = assets['compile_options'];flags = {}
    if options['status']=='VERIFIED':
        if Path(options['path']).stat().st_size>1024**2:
            raise ValueError('Native compile-option inventory exceeds bounded input size')
        text = Path(options['path']).read_text()
        for line in text.splitlines():
            line = line.split('#',1)[0].strip()
            if not line:continue
            match = re.fullmatch(r'([A-Z][A-Z0-9_]*)(?:=(.*))?',line)
            if match is None or match[1] in flags:
                raise ValueError('Malformed or repeated native build option')
            flags[match[1]]=match[2] if match[2] is not None else True
    definitions = {
        'black_holes':('BLACK_HOLES','black_holes',
            ['BH_Mass','BH_InflowRate','BH_RadiatedMass','BH_FeedbackEnergy']),
        'sidm':('RESEARCH_SIDM_REFERENCE','dm',
            ['SIDMInteractionCount','SIDMOperatorSubstep','SIDMExposure']),
        'stellar_evolution':('STELLAR_EVOLUTION','stars',
            ['SSPInitialMass','SSPCurrentMass','SSPBirthNativeTime']),
        'chemistry':('RESEARCH_GRACKLE','gas',
            ['GrackleAtomicSpeciesFractions','GrackleTemperature'])}
    snapshot_build_flags = snapshot_build_flags or {}
    support = {};unresolved = [];declared = {}
    for feature,(flag,species,required) in definitions.items():
        settings = config.get(feature,{})
        if not isinstance(settings,dict) or type(settings.get('enabled',False)) is not bool:
            raise ValueError('Physical feature requires a boolean enabled declaration: '+feature)
        enabled = settings.get('enabled',False);declared[feature]=enabled
        populated = [key for key,counts in species_by_epoch.items() if counts.get(species,0)]
        missing = {key:sorted(set(required)-set(field_support[key].get(species,[])))
                   for key in populated}
        missing = {key:names for key,names in missing.items() if names}
        observed = any(set(required)<=set(field_support[key].get(species,[])) for key in populated)
        reasons = []
        if enabled and any(row['status']!='VERIFIED' for row in assets.values()):
            reasons.append('Source-bound native executable, compile options and parameter file are required')
        if enabled and options['status']=='VERIFIED' and flag not in flags:
            reasons.append('Declared feature absent from native compile options: '+flag)
        if enabled and missing:reasons.append('Saved source fields missing at populated epochs: '+str(missing))
        missing_native = [key for key in species_by_epoch if flag not in snapshot_build_flags.get(key,{})]
        if enabled and missing_native:
            reasons.append('Actual snapshot native Config omits required flag at epochs: '+str(missing_native))
        if not enabled and (flag in flags or observed or any(flag in row for row in snapshot_build_flags.values())):
            reasons.append('Disabled configuration contradicts native build or complete source-field inventory')
        status = 'NOT_DECLARED' if not enabled and not reasons else 'UNRESOLVED' if reasons else 'INPUT_BINDINGS_PRESENT'
        support[feature] = {'status':status,'required_build_flag':flag,'build_flag_present':flag in flags,
            'populated_epochs':populated,'missing_fields_by_epoch':missing,'complete_source_fields_observed':observed,
            'model_definition_sha256':canonical_digest(settings),'missing_snapshot_build_flag_epochs':missing_native,'reasons':reasons}
        unresolved.extend(feature+': '+reason for reason in reasons)
    return {'declared':declared,'build_flags':flags,'snapshot_build_flags':snapshot_build_flags,'handoff_assets':assets,'support':support,
        'acceptance':'NOT_ESTABLISHED','support_scope':'Input bindings only; native model/operator/control-domain acceptance is separate'},unresolved


def inspect_run_capabilities(run_directory):
    """Inventory cosmological snapshots and their declared native features.

The periodic-box reader validates code units and the dimensionless scale
factor a before z = 1/a - 1 is used. Isolated native elapsed time belongs to
the separate isolated-domain readers and is never interpreted as a here.
"""
    import json
    import h5py
    import yaml
    from core.analysis import snapshot_ids, read_snapshot_header, validate_gadget_snapshot_set, group_files
    root = Path(run_directory).expanduser().resolve()
    cfg_path = root/'config.yaml'
    if not cfg_path.is_file():
        raise ValueError('Saved-run configuration is missing: '+str(cfg_path))
    config_hash = file_sha256(cfg_path)
    config = yaml.safe_load(cfg_path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Run configuration must be a mapping')
    epochs, fields, species_by_epoch, catalogues, sources = [], {}, {}, {}, {str(cfg_path): config_hash}
    snapshot_build_flags = {}
    species_names = {0: 'gas', 1: 'dm', 2: 'coarse_dm', 3: 'other', 4: 'stars', 5: 'black_holes'}
    for snap in snapshot_ids(root):
        _header, files = read_snapshot_header(root, snap)
        metadata = validate_gadget_snapshot_set(files, snap)
        counts = {name: 0 for name in species_names.values()}
        field_sets = {name: [] for name in species_names.values()}
        epoch_flags = None
        for path in files:
            before = file_sha256(path)
            with h5py.File(path, 'r') as source:
                import numpy as np
                native_config = source.get('Config')
                flags = {}
                if native_config is not None:
                    if len(native_config.attrs)>4096:
                        raise ValueError('Snapshot native Config exceeds bounded flag inventory')
                    for name,value in native_config.attrs.items():
                        if np.asarray(value).ndim!=0:
                            raise ValueError('Snapshot native build flags must be scalar')
                        value = value.item() if isinstance(value,np.generic) else value
                        if isinstance(value,bytes):value = value.decode('ascii')
                        if value=='':value = True
                        if not isinstance(value,(str,bool,int,float)) or isinstance(value,float) and not np.isfinite(value):
                            raise ValueError('Malformed snapshot native build flag')
                        flags[name]=value
                if epoch_flags is not None and flags!=epoch_flags:
                    raise ValueError('Snapshot shards disagree on actual native build flags')
                epoch_flags = flags
                for ptype, name in species_names.items():
                    group = source.get('PartType'+str(ptype))
                    count = int(source['Header'].attrs['NumPart_ThisFile'][ptype])
                    counts[name] += count
                    if count and group is not None:
                        field_sets[name].append(set(group.keys()))
            if file_sha256(path) != before:
                raise RuntimeError('Snapshot changed during capability inspection')
            sources[str(Path(path).resolve())] = before
        key = str(snap)
        snapshot_build_flags[key] = epoch_flags or {}
        species_by_epoch[key] = counts
        fields[key] = {name: sorted(set.intersection(*sets)) if sets else [] for name, sets in field_sets.items()}
        epochs.append({'snapshot': snap, 'scale_factor': metadata['scale_factor'],
                       'redshift': metadata['redshift']})
        catalogues[key] = bool(group_files(root, snap))
    if file_sha256(cfg_path) != config_hash:
        raise RuntimeError('Run configuration changed during capability inspection')
    manifest_path = root/'manifest.json'
    manifest_hash = file_sha256(manifest_path) if manifest_path.is_file() else None
    manifest = json.loads(manifest_path.read_text()) if manifest_hash is not None else {}
    if manifest_hash is not None:
        if file_sha256(manifest_path)!=manifest_hash:
            raise RuntimeError('Run manifest changed during capability inspection')
        sources[str(manifest_path)] = manifest_hash
    features, unresolved = inspect_native_feature_support(root,config,species_by_epoch,fields,sources,snapshot_build_flags)
    from core.foundation import run_namespace_binding
    ownership = run_namespace_binding(root)
    if ownership['manifest_sha256'] != manifest_hash:
        raise RuntimeError('Run ownership changed during capability inspection')
    if ownership['status'] == 'UNDECLARED':
        unresolved.append('Persistent run ownership is undeclared; transitive single-run product ownership is unresolved')
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:
            raise RuntimeError('Run capability source changed during complete inspection: '+path)
    return {'schema': 'run-capabilities-v1', 'run_directory': str(root),
        'run_id': manifest.get('run_id', root.name),
        'run_namespace': ownership['run_namespace'], 'run_ownership':ownership, 'source_config_hash': config_hash,
        'snapshot_epochs': epochs, 'species_by_epoch': species_by_epoch, 'field_support': fields,
        'native_features': features,
        'catalogue_support': catalogues,
        'zoom_support': {'declared': bool(config.get('zoom', {}).get('enabled', False)
            or config.get('simulation', {}).get('zoom', False)),
            'coarse_particles_present': any(row['coarse_dm'] for row in species_by_epoch.values())},
        'identity_support': 'REQUIRES_BOUND_GRAPH', 'journal_support': 'REQUIRES_JOURNAL_INTAKE',
        'calibration_assets': {}, 'unresolved_reasons': unresolved, 'source_hashes': sources}


def classify_product(product, inventory):
    """Select supported saved-run products without launching new evolution.

An absent species makes a product inapplicable to this run. Missing requested
epochs, fields or calibration assets leave the product waiting for inputs.
Field requirements apply to the requested epochs, where that species exists.
"""
    prerequisites = product.get('prerequisite_ids',[])
    if (not isinstance(prerequisites,list) or any(not isinstance(value,str) or not value.strip() for value in prerequisites)
            or len(set(prerequisites))!=len(prerequisites)):
        raise ValueError('Product prerequisites must be distinct nonempty IDs')
    def decision(status,reason):
        return {'status':status,'reason':reason,'prerequisite_ids':list(prerequisites)}
    scope = product.get('scope', 'single_run')
    if scope not in {'single_run','validation_only','excluded_multirun'}:
        raise ValueError('Unknown product scope: '+str(scope))
    if scope=='excluded_multirun':
        return decision('EXCLUDED_MULTIRUN','Requires excluded multi-run science')
    if scope=='validation_only':
        return decision('VALIDATION_ONLY','Retained external validation control; excluded from science-run analysis')
    if product.get('launches_science_evolution',False):
        return decision('EXCLUDED_SCIENCE_EVOLUTION','Automatic saved-run analysis cannot launch science evolution')
    if product.get('conditional_input_absent', False):
        return decision('NOT_APPLICABLE', 'No inference experiment or supplied posterior was requested')
    minimum_epochs = product.get('minimum_epochs',1)
    if type(minimum_epochs) is not int or minimum_epochs<1:
        raise ValueError('Product minimum_epochs must be a positive integer')
    epochs = inventory['snapshot_epochs']
    present = {name for row in inventory['species_by_epoch'].values() for name, count in row.items() if count}
    required = set(product.get('required_species', []))
    if not required <= present:
        return decision('NOT_APPLICABLE', 'Absent saved species: '+', '.join(sorted(required-present)))
    if product.get('requires_zoom') and not (inventory['zoom_support']['declared'] or inventory['zoom_support']['coarse_particles_present']):
        return decision('NOT_APPLICABLE', 'Run has no declared zoom or coarse support')
    requested_epochs = product.get('requested_epochs')
    if 'requested_epochs' in product:
        if (not isinstance(requested_epochs,list) or any(type(value) is not int or value<0 for value in requested_epochs)
                or len(set(requested_epochs))!=len(requested_epochs)):
            raise ValueError('Requested epoch selection must contain distinct nonnegative snapshot IDs')
        if not requested_epochs or not set(requested_epochs)<={row['snapshot'] for row in epochs}:
            return decision('MISSING_INPUT','Requested saved epochs are unavailable: '+str(requested_epochs))
    if (len(requested_epochs) if requested_epochs is not None else len(epochs)) < minimum_epochs:
        return decision('MISSING_INPUT', 'Insufficient saved epochs for this temporal product')
    for feature in product.get('required_features', []):
        if not inventory['native_features']['declared'].get(feature, False):
            return decision('MISSING_INPUT', 'Required physical source declaration: '+feature)
        support = inventory['native_features'].get('support',{}).get(feature,{})
        if support.get('status')!='INPUT_BINDINGS_PRESENT':
            return decision('MISSING_INPUT','Unresolved native build/data support for '+feature+': '+
                '; '.join(support.get('reasons',['Source-bound feature inventory is missing'])))

    for asset in product.get('required_assets', []):
        binding = inventory.get('calibration_assets', {}).get(asset)
        if not isinstance(binding, dict) or binding.get('status') != 'ACCEPTED' or not binding.get('sha256'):
            return decision('MISSING_INPUT', 'Accepted source-bound calibration asset required: '+asset)
    if product.get('requires_catalogues'):
        catalogue_epochs = product.get('catalogue_epochs', requested_epochs if requested_epochs is not None
                                       else [row['snapshot'] for row in epochs])
        if (not isinstance(catalogue_epochs,list) or any(type(value) is not int or value<0 for value in catalogue_epochs)
                or len(set(catalogue_epochs))!=len(catalogue_epochs)):
            raise ValueError('Catalogue epoch selection must contain distinct nonnegative snapshot IDs')
        available = {row['snapshot'] for row in epochs}
        if not catalogue_epochs or any(value not in available or not inventory['catalogue_support'].get(str(value),False)
                                      for value in catalogue_epochs):
            return decision('MISSING_INPUT', 'Complete native catalogues required at selected epochs: '+str(catalogue_epochs))
        if any(not inventory['species_by_epoch'].get(str(value),{}).get(species,0)
               for value in catalogue_epochs for species in required):
            return decision('MISSING_INPUT','Selected catalogue epochs lack required source species')
    if product.get('requires_complete_volume') and (inventory['zoom_support']['declared'] or inventory['zoom_support']['coarse_particles_present']):
        return decision('MISSING_INPUT', 'Zoom requires an explicit valid volume-selection estimator')
    for species, names in product.get('required_fields', {}).items():
        selected_keys = None if requested_epochs is None else {str(snapshot) for snapshot in requested_epochs}
        for key, counts in inventory['species_by_epoch'].items():
            if selected_keys is not None and key not in selected_keys:
                continue
            if counts.get(species, 0) and not set(names) <= set(inventory['field_support'][key].get(species, [])):
                return decision('MISSING_INPUT', 'Missing '+species+' fields at snapshot '+key)
    if not product.get('inputs_ready', True):
        return decision('MISSING_INPUT', product.get('input_reason', 'Explicit product request required'))
    return decision('APPLICABLE', 'Saved content meets declared input prerequisites')


def resolve_applicable_products(requested, inventory, catalogue):
    if isinstance(requested, str):
        requested = requested.split(',')
    requested = list(dict.fromkeys(requested))
    if 'all' in requested and requested != ['all']:
        raise ValueError('Select all alone or an explicit product list')
    unknown = set(requested)-set(catalogue)-{'all'}
    if unknown:
        raise ValueError('Unknown analysis products: '+', '.join(sorted(unknown)))
    decisions = {name: classify_product(row, inventory) for name, row in catalogue.items()}
    candidates = list(catalogue) if requested == ['all'] else requested
    return {'schema': 'applicable-products-v1', 'requested': requested, 'decisions': decisions,
        'selected': [name for name in candidates if decisions[name]['status'] == 'APPLICABLE'],
        'missing_inputs': {name: decisions[name]['reason'] for name in candidates if decisions[name]['status'] == 'MISSING_INPUT'},
        'explicit_unavailable': {} if requested == ['all'] else {name: decisions[name] for name in candidates
            if decisions[name]['status'] != 'APPLICABLE'}, 'science_acceptance': 'NOT_ESTABLISHED'}


def expand_all_products(requested, products, spec):
    """Expand all only into producers whose requested physical species exist."""
    requested = list(requested)
    if len(set(requested)) != len(requested):
        raise ValueError('Repeated requested product')
    species = available_species(spec)
    result = []
    for product in requested:
        if product == 'all':
            result.extend(name for name, rule in sorted(products.items())
                          if set(rule.get('species', [])) <= species and rule.get('include_in_all', True))
        elif product in products:
            if not set(products[product].get('species', [])) <= species:
                raise ValueError('Requested product requires unavailable physical species: ' + product)
            result.append(product)
        else:
            raise ValueError('No registered producer for product: ' + product)
    return list(dict.fromkeys(result))

def require_supported_combination(spec, request_base=None):
    """Require measured support before combining a new background/linear model with SIDM."""
    dm = spec.get('dark_matter', {})
    physics = spec.get('physics', {})
    kind = dm.get('interactions', {}).get('kind', physics.get('interaction', 'collisionless') if isinstance(physics, Mapping) else 'collisionless')
    legacy_sidm = spec.get('sidm')
    if isinstance(legacy_sidm, Mapping) and legacy_sidm.get('enabled') is True:
        if 'kind' in dm.get('interactions', {}) and kind != 'elastic_sidm':
            raise ValueError('Conflicting nonlinear interaction declarations')
        kind = 'elastic_sidm'
    background = spec.get('cosmology_model', spec.get('cosmology_bundle', {}).get('background_model', spec.get('model', 'lcdm')))
    linear = dm.get('linear_model', 'cdm')
    if background not in {'lcdm', 'ede'} or kind not in {'collisionless', 'elastic_sidm'} or linear not in {'cdm', 'wdm'}:
        raise ValueError('Unsupported dark-matter model')
    if kind == 'elastic_sidm' and (background != 'lcdm' or linear != 'cdm'):
        combination = spec.get('combination_specification')
        if not combination or not combination.get('receipt'):
            raise ValueError('EDE/WDM plus SIDM requires an explicit combined-physics receipt')
        path = Path(combination['receipt']).expanduser()
        if not path.is_absolute():
            path = Path(request_base or '.')/path
        if file_sha256(path) != combination['sha256']:
            raise ValueError('Combined-physics receipt changed')
        import json
        proof = json.loads(path.read_text())
        expected = {'background': background, 'linear': linear, 'interaction': kind}
        if proof.get('status') != 'PASS' or proof.get('execution_status') != 'COMPLETE' or proof.get('combination') != expected:
            raise ValueError('Combined-physics campaign does not cover this combination')
        model_hash = spec.get('combination_model_sha256')
        if model_hash is not None and proof.get('model_specification_sha256') != model_hash:
            raise ValueError('Combined-physics campaign uses a different requested cosmology/linear/interaction model')
        for label in ('source_hashes', 'input_hashes', 'files_sha256'):
            if not proof.get(label):
                raise ValueError('Combined-physics campaign lacks current measured '+label)
            for name, digest in proof[label].items():
                source = Path(name)
                if not source.is_absolute():
                    source = path.parent/source
                if file_sha256(source) != digest:
                    raise ValueError('Combined-physics campaign has stale retained evidence')


def publish_native_build_capability(destination, *, name, executable, compiled_features,
                                    profile, source_hashes, invocation_receipt, domain, assets=()):
    """Bind an actual successful compile invocation to current source/profile bytes.

    This is build capability publication, not native science validation. The
    compile invocation must be recorded by the existing owned-process runner.
    """
    import json
    from core.foundation import atomic_json
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Native build capability receipts cannot overwrite history')
    binary = Path(executable).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK) or not source_hashes:
        raise ValueError('Executable and exact build sources required')
    invocation_path = Path(invocation_receipt).resolve()
    invocation = json.loads(invocation_path.read_text())
    if invocation.get('schema') != 'native-command-invocation-v1' or invocation.get('status') != 'RETURNED' or invocation.get('returncode') != 0:
        raise ValueError('Build capability requires a returned successful native compile invocation')
    for bindings in (source_hashes, invocation['input_hashes']):
        for source, expected in bindings.items():
            if file_sha256(source) != expected:
                raise ValueError('Native build source/input changed before capability publication')
    features = list(compiled_features)
    if len(set(features)) != len(features) or any(not isinstance(q, str) or not q for q in features):
        raise ValueError('Explicit distinct compiled features required')
    profile_hash = canonical_digest(profile)
    proof = {'schema': 'native-build-capability-v1', 'name': name, 'execution_status': 'COMPLETE',
        'executable': str(binary), 'executable_sha256': file_sha256(binary),
        'compiled_features': sorted(features), 'profile_hash': profile_hash, 'profile': profile,
        'source_hashes': source_hashes, 'invocation_receipt': {'path': str(invocation_path), 'sha256': file_sha256(invocation_path)},
        'domain': domain, 'science_acceptance': 'NOT_ESTABLISHED_BY_BUILD'}
    atomic_json(destination, proof)
    return {'name': name, 'executable': str(binary), 'executable_sha256': proof['executable_sha256'],
        'features': sorted(features), 'profile_hash': profile_hash, 'domain': domain,
        'assets': list(assets), 'build_receipt': str(destination), 'build_receipt_sha256': file_sha256(destination)}


# Public native requests: allocated resources, physical settings and input paths.
RECURSIVE_COMMANDS={'workflow','pilots','clean-release'}
RESOURCE_FIELDS={'mpi_ranks':'mpi_ranks','omp_threads':'omp_threads','memory_limit_gb':'memory_gib'}


def validate_public_command(parameters):
    """Check the selected public route and its frozen argument references."""
    if 'public_route_trace' in parameters:
        raise ValueError('Public route tracing has been removed; omit public_route_trace')
    from commands import COMMANDS
    command=parameters['command']
    if not isinstance(command,str) or command not in COMMANDS or command in RECURSIVE_COMMANDS:
        raise ValueError('Unknown or recursive public command before native evolution: '+str(command))
    wall=parameters['wall_seconds']
    if isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<=0:
        raise ValueError('Public native command requires an explicit finite positive wall limit')
    arguments=parameters.get('arguments',[])
    if not isinstance(arguments,list):raise ValueError('Public command arguments must be an explicit argument vector')
    for argument in arguments:
        if isinstance(argument,str):
            if not argument:raise ValueError('Empty public command argument')
        elif isinstance(argument,dict):
            if set(argument)!={'input'} and not ('stage' in argument and set(argument)<={'stage','artifact'}):
                raise ValueError('Public command argv references must produce paths, not arbitrary loaded JSON values')
        else:raise ValueError('Public command arguments require strings or actual file/product references')


def validate_native_answers(answers,resources,canonical_constraints=None):
    """Build and check native settings without launching a solver or simulation."""
    from commands import build_config
    if not isinstance(answers,dict):raise ValueError('Native public answers must be a complete JSON object')
    for field,resource in RESOURCE_FIELDS.items():
        if field not in answers or answers[field]!=resources[resource]:
            raise ValueError('Native generated answers must retain explicitly allocated '+resource)
    # The ordinary public builder supplies the complete physical/numerical,
    # transfer-asset, source-model and option-domain checks. It does not launch
    # an evolution or build a native executable.
    config=build_config(answers)[0]
    if canonical_constraints is not None:
        epoch=canonical_constraints['epochs'];simulation=config['simulation']
        actual=simulation.get('snapshot_redshift_list')
        if simulation['start_redshift']!=epoch['z_start'] or simulation['end_redshift']!=epoch['z_end'] or actual!=epoch['backbone_redshifts']:
            raise ValueError('Native answers conflict with authoritative version-2 start/end/output epochs')
        bundle=canonical_constraints['cosmology_bundle']
        if file_sha256(config['spectrum']['file'])!=bundle['sha256']:
            raise ValueError('Native answers use a different exact linear solver bundle')
        nonlinear='elastic_sidm' if config.get('sidm',{}).get('enabled') is True else 'collisionless'
        linear='wdm' if config['cosmology']['model']=='wdm' or config.get('wdm',{}).get('enabled') is True else 'cdm'
        requested=canonical_constraints['dark_matter']
        if nonlinear!=requested['interactions']['kind'] or linear!=requested['linear_model']:
            raise ValueError('Native answers conflict with canonical linear/nonlinear dark matter physics')
    return config


def _preflight_json(path,memory_bytes):
    path=Path(path).resolve()
    if not path.is_file() or path.stat().st_size*8>memory_bytes:
        raise MemoryError('Native generated-answer source is missing or exceeds preflight workspace')
    import json
    return json.loads(path.read_text())


def preflight_public_command_stage(row,plan,context):
    """Resolve Gadget or zoom inputs before the workflow launches native work."""
    from core.workflow import path_at,_stage_references
    validate_public_command(row['parameters'])
    parameters=row['parameters']
    if parameters['command']=='zoom':
        from core.zoom import validate_native_music_regression,require_mixed_lpt_validation,verify_staged_zoom_inputs
        import configparser
        arguments=parameters.get('arguments',[])
        def source(value):
            if not isinstance(value,dict) or set(value)!={'input'}:
                raise ValueError('Native zoom preflight requires frozen configuration/binary/control inputs')
            return path_at(context.base,value['input'])
        def option(flag):
            if arguments.count(flag)!=1 or arguments.index(flag)+1>=len(arguments):
                raise ValueError('Native zoom needs exactly one '+flag+' input')
            return source(arguments[arguments.index(flag)+1])
        if not arguments:raise ValueError('Native zoom requires a configuration input')
        configuration=source(arguments[0]);binary=option('--binary');proof=validate_native_music_regression(binary,option('--native-validation'))
        cfg=configparser.ConfigParser(interpolation=None);cfg.optionxform=str;cfg.read(configuration)
        if cfg.getboolean('setup','use_2LPT',fallback=False):require_mixed_lpt_validation(proof,binary)
        if (configuration.parent/'staging.json').is_file():verify_staged_zoom_inputs(configuration.parent)
        return
    if parameters['command']!='gadget':return
    arguments=parameters.get('arguments',[])
    if arguments.count('--answers') != 1 or arguments.index('--answers')+1>=len(arguments):
        raise ValueError('Noninteractive evolution needs exactly one --answers input')
    reference=arguments[arguments.index('--answers')+1]
    resources=row.get('resources',context.resources);memory=resources['memory_gib']*1024**3
    if isinstance(reference,dict) and set(reference)=={'input'}:
        validate_native_answers(_preflight_json(path_at(context.base,reference['input']),memory),resources,row.get('canonical_run_constraints'))
        return
    if not isinstance(reference,dict) or 'stage' not in reference:
        raise ValueError('Native answers must be a hashed input or actual produced stage artifact')
    stages={stage['stage_id']:stage for stage in plan['stages']}
    producer=stages[reference['stage']]
    if producer['kind']=='native_control_configuration':
        # Dedicated control materializers are rendered and checked in memory
        # by preflight_configuration_stages after this common stage pass.
        return
    if producer['kind']!='materialize_request' or reference.get('artifact')!='request.json':
        raise ValueError('Generated native answers need a registered, inspectable physical request materializer')
    answers=producer['parameters']['request']
    if isinstance(answers,dict) and set(answers)=={'json_input'}:
        answers=_preflight_json(path_at(context.base,answers['json_input']),memory)
    if not isinstance(answers,dict) or list(_stage_references(answers)):
        raise ValueError('Native resolution, resource and physics answers must be frozen before particle evolution; whole future JSON substitution is unsupported')
    validate_native_answers(answers,resources,row.get('canonical_run_constraints'))
