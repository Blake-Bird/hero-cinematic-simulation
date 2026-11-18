"""Assemble measured displays, graph galleries and native scene attachments.

The source arrays keep their measured units, identities, epochs and support.
Only scene adapters convert proper kpc to comoving Mpc/h, using h/(1000*a).
Periodic copies show the same physical selection; they add no physical mass.
"""
from __future__ import annotations

import copy
import html
import itertools
import json
import math
import numpy as np
import os
import shutil
import zipfile
from core.blender_scene import KINDS, load_verified_display_products
from core.foundation import atomic_json, atomic_text, file_sha256, canonical_digest
from core.native import native_run_lock
from core.foundation import merge_source_bindings as _merge
from core.stream_common import read_product, digest, json_safe
from pathlib import Path
from urllib.parse import quote


# Measured arrays and lifecycle records


def resolve_measured_display_source(source,*,request_base,memory_bytes):
    """Resolve actual upstream products after their workflow generation exists.

    A produced scientific product is identified by its sealed receipt and
    declared product, so callers never predict a future artifact hash. The
    usual direct path/SHA256 intake remains available for retained inputs.
    """
    from core.workflow import path_at,verify_completed_receipt
    if 'measurement' not in source:
        if 'path' not in source or 'sha256' not in source:
            raise ValueError('Measured display source needs a retained path/hash or exact executed workflow product')
        return source,{}
    if 'path' in source or 'sha256' in source:
        raise ValueError('Select one exact direct or executed-product measured display reference')
    reference = source['measurement']
    if not isinstance(reference,dict) or set(reference) != {'receipt','product'}:
        raise ValueError('Produced display measurements require their exact receipt and declared product')
    receipt_path = path_at(request_base,reference['receipt'])
    if receipt_path.stat().st_size*8 > memory_bytes:
        raise MemoryError('Measured display producer receipt exceeds bounded input memory')
    receipt = verify_completed_receipt(receipt_path)
    product = receipt['products'][reference['product']]
    if product['kind'] != 'file':
        raise ValueError('Produced display source must identify its exact scientific report file')
    generation = Path(receipt['generation']).resolve()
    path = (generation/product['path']).resolve()
    digest = file_sha256(path)
    if not path.is_relative_to(generation) or receipt['files_sha256'].get(path.relative_to(generation).as_posix()) != digest:
        raise ValueError('Produced display report is outside its exact sealed scientific generation')
    bindings = {str(receipt_path):file_sha256(receipt_path)}
    _merge(bindings, receipt['source_hashes'])
    _merge(bindings, receipt['input_hashes'])
    _merge(bindings, {str(generation/name):expected for name,expected in receipt['files_sha256'].items()})
    return {key:value for key,value in source.items() if key != 'measurement'} | {'path':str(path),'sha256':digest},bindings


def measured_epoch_series_annotation(epochs, *, axis):
    """Producer annotation for a retained history, including epoch-invariant IDs.

    Attach the returned value at specifications[name]['annotations']['epoch_series']
    when publishing the numeric archive. ``axis=None`` explicitly declares that
    this array applies unchanged throughout the actual history. It must never
    be used to conceal an identity roster that changes between saved epochs.
    """
    if not isinstance(epochs,list) or len(epochs) < 2:
        raise ValueError('Measured epoch series requires at least two actual saved clocks')
    if axis is not None and (type(axis) is not int or axis < 0):
        raise ValueError('Producer time axis must be a nonnegative integer or explicit None')
    clocks = {'scale_factor':'scale_factor','isolated_native_elapsed_time':'native_time_code'}
    representation = None
    previous = None
    for epoch in epochs:
        if not isinstance(epoch,dict):
            raise ValueError('Each measured epoch needs its producer canonical clock contract')
        current = epoch.get('canonical_time')
        clock = clocks.get(current)
        if clock is None or (representation is not None and current != representation):
            raise ValueError('Measured history cannot change its canonical clock representation')
        value = epoch.get(clock)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Measured history clocks must be finite actual native values')
        if (clock == 'scale_factor' and value <= 0) or (previous is not None and value <= previous):
            raise ValueError('Measured history clocks must increase strictly in retained order')
        representation,previous = current,value
    # Copy JSON-compatible contracts so subsequent producer edits cannot alter
    # the annotation by retaining references to the original clock objects.
    return {'axis':axis,'epochs':json.loads(json.dumps(epochs,allow_nan=False))}


def _field(arrays,contract,descriptor,epoch,frame):
    if isinstance(descriptor,str):
        descriptor = {'array':descriptor}
    if not isinstance(descriptor,dict) or not set(descriptor) <= {'array','epoch_index','epoch_axis'}:
        raise ValueError('Display fields need exact array names and saved epoch selectors')
    name = descriptor['array']
    metadata = contract['datasets'][name]
    if metadata.get('epoch_status') != 'PRODUCER_DECLARED' or metadata.get('quantity_definition_status') != 'PRODUCER_DECLARED':
        raise ValueError('Display fields require producer-declared physical epoch and quantity')
    declared_epoch = json.loads(metadata['epoch_json'])
    values = arrays[name]
    series = metadata.get('annotations',{}).get('epoch_series')
    if 'epoch_axis' in descriptor and 'epoch_index' not in descriptor:
        raise ValueError('Display epoch axis requires a saved epoch index')
    if 'epoch_index' in descriptor:
        index = descriptor['epoch_index']
        if type(index) is not int or index < 0:
            raise ValueError('Saved display epoch index must be a nonnegative integer')
        if series is not None:
            if not isinstance(series,dict) or set(series) != {'axis','epochs'} or not isinstance(series['epochs'],list):
                raise ValueError('Producer epoch-series annotation must retain its axis and actual clocks')
            epochs,axis = series['epochs'],series['axis']
            if 'epoch_axis' in descriptor and descriptor['epoch_axis'] != axis:
                raise ValueError('Display selector changes the sealed producer epoch axis')
        elif isinstance(declared_epoch,list):
            epochs,axis = declared_epoch,descriptor.get('epoch_axis',0)
        else:
            raise ValueError('Saved-history display selection requires actual producer epoch-series metadata')
        measured_epoch_series_annotation(epochs,axis=axis)
        if index >= len(epochs):
            raise ValueError('Display saved epoch index exceeds the measured history')
        if axis is not None:
            if type(axis) is not int or not 0 <= axis < values.ndim or values.shape[axis] != len(epochs):
                raise ValueError('Array time axis does not match the retained epoch series')
            values = np.take(values,index,axis=axis)
        # axis=None is a producer-declared array constant across this measured
        # history, such as the exact particle IDs. Its identity row is intact.
        declared_epoch = epochs[index]
    elif series is not None or isinstance(declared_epoch,list):
        raise ValueError('A measured time-series field requires explicit saved epoch selection')
    if not isinstance(declared_epoch,dict) or declared_epoch.get('canonical_time') != epoch.get('canonical_time'):
        raise ValueError('Display epoch differs from the measured producer clock representation')
    clocks = {'scale_factor':'scale_factor','isolated_native_elapsed_time':'native_time_code'}
    clock = clocks.get(epoch.get('canonical_time'))
    if clock is None:
        raise ValueError('Display requires a supported canonical native clock')
    actual,requested = declared_epoch.get(clock),epoch.get(clock)
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in (actual,requested)) or not math.isclose(actual,requested,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Display requested epoch is not the actual measured product epoch')
    if clock == 'scale_factor' and actual <= 0:
        raise ValueError('Actual display scale factor must be positive')
    if clock == 'native_time_code' and declared_epoch.get('time_unit_gyr') != epoch.get('time_unit_gyr'):
        raise ValueError('Isolated display clocks use different physical time units')
    coordinates = json.loads(metadata['coordinate_conventions_json']) if 'coordinate_conventions_json' in metadata else None
    if coordinates is not None and coordinates != frame:
        raise ValueError('Display field coordinate contract differs from the sealed measured frame')
    units = metadata['units']
    if not isinstance(units,str) or not units or units.startswith('UNDECLARED'):
        raise ValueError('Display field units are undeclared')
    return values,metadata


def _lifecycle_product(source, *, request_base, destination, epoch, bindings,
                       remaining_events, memory_bytes):
    """Retain explicitly measured event rows and the producer's source bindings."""
    from core.workflow import path_at
    if not isinstance(source,dict) or set(source) != {'path','sha256'}:
        raise ValueError('Lifecycle intake requires an exact source path and SHA256')
    path = path_at(request_base,source['path'])
    if path.stat().st_size > memory_bytes:
        raise MemoryError('Lifecycle product exceeds declared structured intake memory')
    encoded = path.read_bytes()
    import hashlib
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != source['sha256']:
        raise ValueError('Measured lifecycle report changed')
    def invalid_constant(value):
        raise ValueError('Lifecycle JSON contains a nonfinite numeric constant: '+value)
    report = json.loads(encoded,parse_constant=invalid_constant)
    json.dumps(report,allow_nan=False)
    if report.get('schema') != 'measured-lifecycle-events-v1':
        raise ValueError('Expected measured-lifecycle-events-v1 producer report')
    if report.get('epoch_contract') != epoch:
        raise ValueError('Lifecycle events do not belong to the exact saved native epoch')
    provenance = report.get('source_hashes')
    if not isinstance(provenance,dict) or not provenance:
        raise ValueError('Lifecycle producer must retain actual scientific source hashes')
    for original,expected in provenance.items():
        native = path_at(path.parent,original)
        if not isinstance(expected,str) or file_sha256(native) != expected:
            raise ValueError('Lifecycle scientific source changed')
        name = str(native)
        if name in bindings and bindings[name] != expected:
            raise ValueError('Lifecycle provenance conflicts with another saved product')
        bindings[name] = expected
    events = report.get('events')
    if not isinstance(events,list) or len(events) > remaining_events:
        raise MemoryError('Measured lifecycle event census exceeds its explicit bound')
    target = destination/'assets'/('lifecycle-'+digest+'.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_sha256(target) != digest:
            raise ValueError('Portable lifecycle report collides with changed bytes')
    else:
        target.write_bytes(encoded)
    reference = {'path':target.relative_to(destination).as_posix(),
        'sha256':digest,'label':path.name}
    bindings[str(path)] = digest
    bindings[str(target)] = digest
    from core.blender_scene import _identity
    retained = []
    for row in events:
        if not isinstance(row,dict) or 'source_links' in row:
            raise ValueError('Producer events require records; source links are generated from sealed intake')
        if row.get('measurement_origin') != 'retained_scientific_product':
            raise ValueError('Lifecycle report cannot supply inferred or synthetic events')
        if not isinstance(row.get('quantity_definition'),str) or not row['quantity_definition'].strip():
            raise ValueError('Lifecycle events require their actual producer measurement definition')
        _identity(row)
        if row.get('event_type') not in {'birth','death','merger','missing_detection','reappearance','stripping','mass_return'}:
            raise ValueError('Unsupported measured lifecycle event type')
        if not isinstance(row.get('subjects'),list) or not row['subjects']:
            raise ValueError('Measured events require exact namespaced subjects')
        for subject in row['subjects']:
            _identity(subject)
        for parent in row.get('parents',[]):
            _identity(parent)
        retained.append({**row,'source_links':[reference]})
    return retained,len(encoded)


def display_request_input_bindings(request, *, request_base, memory_gib):
    """Fingerprint external measured inputs without loading scientific arrays.

    This is provenance/resource preflight, not numerical or science acceptance.
    Produced stage references must be resolved by the workflow before calling.
    """
    from core.workflow import path_at,positive_number
    from core.data import numeric_audit_source_bindings
    positive_number(memory_gib,'display input memory')
    if not isinstance(request,dict) or request.get('schema') != 'measured-display-build-v1':
        raise ValueError('Display workflow requires a measured-display-build-v1 request')
    requested = positive_number(request['memory_gib'],'measured display intake memory')
    if requested > memory_gib:
        raise MemoryError('Measured display request exceeds its allocated memory; settings unchanged')
    bindings = {}
    def pin(path,expected=None):
        path = Path(path).resolve()
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Display workflow source differs from the declared hash: '+str(path))
        name = str(path)
        if name in bindings and bindings[name] != digest:
            raise ValueError('Display workflow sources changed during fingerprint capture')
        bindings[name] = digest
    for saved in request['timeline']:
        for source in saved['products']:
            source,producer_bindings = resolve_measured_display_source(source,request_base=request_base,
                memory_bytes=int(memory_gib*1024**3))
            for path,digest in producer_bindings.items():
                pin(path,digest)
            path = path_at(request_base,source['path'])
            pin(path,source['sha256'])
            if path.stat().st_size*8 > memory_gib*1024**3:
                raise MemoryError('Measured display report exceeds input workspace')
            report = json.loads(path.read_text())
            directory = report['numeric_data']['directory']
            if directory != path.name+'.data':
                raise ValueError('Display numeric report requires its sealed relative sibling bundle')
            pin(path.with_name(path.name+'.sha256'))
            for member in sorted(path.with_name(directory).iterdir()):
                if member.is_symlink() or not member.is_file():
                    raise ValueError('Display numeric bundle requires regular file members')
                pin(member)
            for native,expected in numeric_audit_source_bindings(report).items():
                pin(native,expected)
        for source in saved.get('lifecycle_products',[]):
            source,producer_bindings = resolve_measured_display_source(source,request_base=request_base,
                memory_bytes=int(memory_gib*1024**3))
            for path,digest in producer_bindings.items():
                pin(path,digest)
            path = path_at(request_base,source['path'])
            pin(path,source['sha256'])
            if path.stat().st_size*8 > memory_gib*1024**3:
                raise MemoryError('Measured lifecycle report exceeds input workspace')
            report = json.loads(path.read_text())
            if report.get('schema') != 'measured-lifecycle-events-v1' or not report.get('source_hashes'):
                raise ValueError('Lifecycle workflow input requires actual producer source bindings')
            for native,expected in report['source_hashes'].items():
                pin(path_at(path.parent,native),expected)
    return bindings


def join_measured_sidm_diagnostics(records):
    """Attach diagnostics to one exact object without manufacturing identities.

    Multiple geometric representations require an explicit primary product.
    SIDM statistics retain their own support, units, arrays and source links;
    they never change the primary shape's support or create collision imagery.
    """
    from core.blender_scene import _identity
    groups = {}
    for record in records:
        groups.setdefault(_identity(record),[]).append(record)
    joined = []
    for group in groups.values():
        if len(group) == 1:
            joined.append(group[0])
            continue
        geometry = [row for row in group if row['kind'] != 'sidm']
        diagnostics = [row for row in group if row['kind'] == 'sidm']
        if not geometry:
            raise ValueError('Duplicate SIDM-only records need a producer-combined diagnostic product')
        if len({row['kind'] for row in geometry}) != len(geometry):
            raise ValueError('Repeated measured geometry kinds are ambiguous for an exact identity')
        candidates = [row for row in geometry if row.get('display_primary_geometry',False)]
        if len(candidates) > 1 or (len(geometry) > 1 and not candidates):
            raise ValueError('Multiple measured geometry layers require exactly one explicit primary product')
        original = candidates[0] if candidates else geometry[0]
        primary = dict(original)
        if any(row['position'] != primary['position'] for row in group):
            raise ValueError('Measured geometry/diagnostics change the exact object position/frame')
        primary['measured_diagnostics'] = diagnostics
        primary['measured_geometry_layers'] = [row for row in geometry if row is not original]
        references = {}
        for row in group:
            for reference in row['source_links']:
                path = reference['path']
                if path in references and references[path] != reference:
                    raise ValueError('Joined inspector source paths have conflicting measured bindings')
                references[path] = reference
        primary['source_links'] = list(references.values())
        if diagnostics:
            primary['diagnostic_representation'] = 'measured_macro_diagnostics'
        joined.append(primary)
    return joined


def assemble_measured_display(request,destination,*,request_base):
    from core.data import verify_numeric_audit,numeric_audit_arrays_path,numeric_audit_source_bindings
    from core.workflow import path_at
    if request.get('schema') != 'measured-display-build-v1':
        raise ValueError('Expected measured-display-build-v1')
    memory = request['memory_gib']
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Explicit positive measured display intake memory required')
    geometry_limit = request['maximum_geometry_vertices']
    if type(geometry_limit) is not int or geometry_limit < 1:
        raise ValueError('Explicit positive measured geometry vertex bound required')
    geometry_object_limit = request.get('maximum_geometry_objects',request['maximum_objects'])
    if type(geometry_object_limit) is not int or geometry_object_limit < 1:
        raise ValueError('Explicit positive rendered geometry object bound required')
    event_limit = request.get('maximum_lifecycle_events',0)
    if type(event_limit) is not int or event_limit < 0:
        raise ValueError('Lifecycle event bound must be an explicit nonnegative integer')
    maximum = request['maximum_objects']
    measurement_limit = request.get('maximum_measurement_records',maximum)
    if type(measurement_limit) is not int or measurement_limit < 1:
        raise ValueError('Measured source row census requires a positive integer bound')
    if type(maximum) is not int or maximum < 1 or len(request['timeline']) < 2:
        raise ValueError('Bounded object census and at least two measured epochs required')
    destination = Path(destination).resolve()
    frame = request['frame_contract']
    manifest = {'schema':'measured-display-v1','maximum_objects':maximum,
        'maximum_geometry_vertices':request['maximum_geometry_vertices'],'maximum_geometry_objects':geometry_object_limit,
        'coordinate_units':frame['position_units'],'frame_contract':frame,'timeline':[],
        'assets':[],'science_acceptance':'UNRESOLVED','assembly_request_hash':canonical_digest(request)}
    census = 0
    object_census = 0
    event_census = 0
    lifecycle_bytes = 0
    geometry_census = 0
    geometry_object_census = 0
    bindings = {}
    with native_run_lock(destination,operation='assemble sealed measured Blender products'):
        if destination.exists():
            raise FileExistsError('Measured display generations are immutable')
        destination.mkdir(parents=True)
        try:
            for saved in request['timeline']:
                epoch = {'frame':saved['frame'],'snapshot':saved['snapshot'],
                    'epoch_contract':saved['epoch_contract'],'objects':[],'lifecycle_events':[]}
                for source in saved['products']:
                    source,producer_bindings = resolve_measured_display_source(source,request_base=request_base,
                        memory_bytes=int(memory*1024**3))
                    for native,expected in producer_bindings.items():
                        if native in bindings and bindings[native] != expected:
                            raise ValueError('Measured display producer provenance conflicts with another saved source')
                        bindings[native] = expected
                    kind = source['kind']
                    if kind not in KINDS:
                        raise ValueError('Unknown measured display kind')
                    path = path_at(request_base,source['path'])
                    if file_sha256(path) != source['sha256']:
                        raise ValueError('Measured display source report changed')
                    archive = path.with_name(path.name+'.data')/'arrays.npz'
                    with zipfile.ZipFile(archive) as packed:
                        if sum(member.file_size for member in packed.infolist())*6 > memory*1024**3:
                            raise MemoryError('Measured display arrays exceed intake/reduction workspace')
                    report = verify_numeric_audit(path)
                    if report['numeric_data']['directory'] != path.name+'.data':
                        raise ValueError('Portable measured source reports require their immutable relative sibling bundle')
                    provenance = numeric_audit_source_bindings(report)
                    for native,expected in provenance.items():
                        if native in bindings and bindings[native] != expected:
                            raise ValueError('Scientific display sources changed between product intakes')
                        bindings[native] = expected
                    contract = report['numeric_data']['array_contract']
                    if type(source.get('primary_geometry',False)) is not bool:
                        raise ValueError('Primary geometry selection must be an exact boolean')
                    fields = source['fields']
                    if not {'object_id','position','geometry_supported'} <= set(fields):
                        raise ValueError('Every display mapping needs exact identity, measured position and boolean geometry support')
                    required_geometry = {'shape':{'semiaxes','orientation'},'tide':{'eigenvalues','eigenvectors'},'stream':{'points'}}
                    if not required_geometry.get(kind,set()) <= set(fields):
                        raise ValueError('Display kind lacks its measured geometry arrays')
                    if kind == 'tide' and ('glyph_scale_per_eigenvalue' not in source or isinstance(source['glyph_scale_per_eigenvalue'],bool) or not math.isfinite(source['glyph_scale_per_eigenvalue']) or source['glyph_scale_per_eigenvalue'] <= 0):
                        raise ValueError('Tidal display requires an explicit finite positive glyph scale')
                    allowed = {'object_id','position','geometry_supported','semiaxes','orientation','eigenvalues','eigenvectors',
                        'points','point_counts','density','interaction_exposure','interaction_count','core_radius'}
                    if not set(fields) <= allowed:
                        raise ValueError('Unrecognized scientific display field mapping')
                    packed = destination/'assets'/('product-'+source['sha256'])
                    packed.mkdir(parents=True,exist_ok=True)
                    references = []
                    retained = [path,path.with_name(path.name+'.sha256'),*sorted(path.with_name(path.name+'.data').iterdir())]
                    for original in retained:
                        if not original.is_file():
                            raise ValueError('Measured numeric product contains an unexpected non-file member')
                        relative = original.relative_to(path.parent)
                        target = packed/relative
                        target.parent.mkdir(parents=True,exist_ok=True)
                        digest = file_sha256(original)
                        if original == path and digest != source['sha256']:
                            raise ValueError('Measured source report changed during portable copying')
                        if not target.exists():
                            shutil.copy2(original,target)
                        if file_sha256(target) != digest:
                            raise ValueError('Measured numeric product changed while copying')
                        references.append({'path':target.relative_to(destination).as_posix(),'sha256':digest,'label':original.name,
                            'bundle':{'id':source['sha256'],'relative_path':relative.as_posix()}})
                        bindings[str(original)] = digest
                    verify_numeric_audit(packed/path.name)
                    with np.load(numeric_audit_arrays_path(path,report),allow_pickle=False) as arrays:
                        measured = {key:_field(arrays,contract,descriptor,saved['epoch_contract'],frame) for key,descriptor in fields.items()}
                        ids,identity = measured['object_id']
                        namespace = identity.get('identity_namespace')
                        if ids.ndim != 1 or ids.dtype != np.dtype('uint64') or not namespace or len(np.unique(ids)) != len(ids):
                            raise ValueError('Measured display identities must be unique canonical uint64 with the producer namespace')
                        n = len(ids)
                        census += n
                        if census*4096+sum(values.nbytes*8 for values,_ in measured.values()) > memory*1024**3:
                            raise MemoryError('Measured display structured records exceed declared intake memory')
                        if census > measurement_limit:
                            raise MemoryError('Measured display input row census exceeds the declared bound')
                        position,position_contract = measured['position']
                        if position.shape != (n,3) or not np.isfinite(position).all() or position_contract['units'] != frame['position_units'] or 'coordinate_conventions_json' not in position_contract:
                            raise ValueError('Measured display positions require finite xyz in the exact producer frame/units')
                        support,_ = measured['geometry_supported']
                        if support.dtype.kind != 'b' or support.shape != (n,):
                            raise ValueError('Measured geometry support requires exact per-object booleans')
                        for key,(values,metadata) in measured.items():
                            if values.ndim == 0 or len(values) != n:
                                raise ValueError('Display arrays do not share the exact identity row axis')
                        for key in ('semiaxes','points'):
                            if key in measured and measured[key][1]['units'] != frame['position_units']:
                                raise ValueError('Shape/stream geometry units differ from the measured display frame')
                        for key in ('orientation','eigenvectors'):
                            if key in measured and measured[key][1]['units'] != 'dimensionless':
                                raise ValueError('Measured orientation/eigenvectors must be dimensionless')
                        expected_shapes = {'semiaxes':(n,3),'orientation':(n,3,3),'eigenvalues':(n,3),
                            'eigenvectors':(n,3,3),'density':(n,),'interaction_exposure':(n,),
                            'interaction_count':(n,),'core_radius':(n,),'point_counts':(n,)}
                        for key,shape in expected_shapes.items():
                            if key in measured and measured[key][0].shape != shape:
                                raise ValueError('Measured display geometry/scalar shape mismatch: '+key)
                        if 'points' in measured:
                            points = measured['points'][0]
                            if points.ndim != 3 or points.shape[0] != n or points.shape[1] < 2 or points.shape[2] != 3:
                                raise ValueError('Measured streams need object-by-point-by-xyz geometry')
                        if 'semiaxes' in measured and np.any(measured['semiaxes'][0][support] <= 0):
                            raise ValueError('Supported shape semiaxes must be positive measured lengths')
                        for key in ('orientation','eigenvectors'):
                            if key in measured:
                                matrices = measured[key][0][support]
                                gram = np.einsum('nji,njk->nik',matrices,matrices)
                                if not np.allclose(gram,np.eye(3),atol=1e-5,rtol=0):
                                    raise ValueError('Supported measured orientations/eigenvectors must be orthonormal')
                                if key == 'orientation' and not np.allclose(np.linalg.det(matrices),1.,atol=1e-5,rtol=0):
                                    raise ValueError('Measured shape orientation must be right-handed')
                        for row,pid in enumerate(ids):
                            record = {'namespace':namespace,'object_id':str(int(pid)),'kind':kind,
                                'position':position[row].tolist(),'geometry_supported':bool(support[row]),
                                'measurement_origin':'retained_scientific_product','source_links':references,
                                'measurement_units':{key:metadata['units'] for key,(_,metadata) in measured.items()},
                                'source_array_fields':fields,'display_primary_geometry':source.get('primary_geometry',False)}
                            for key,(values,_) in measured.items():
                                if key in {'object_id','position','geometry_supported','point_counts'}:
                                    continue
                                value = values[row]
                                if key == 'points' and 'point_counts' in measured:
                                    count = measured['point_counts'][0][row]
                                    if not isinstance(count,np.integer) or not 2 <= int(count) <= len(value):
                                        raise ValueError('Measured stream point count is outside retained geometry')
                                    value = value[:int(count)]
                                if np.asarray(value).dtype.kind not in 'biuf':
                                    raise ValueError('Scientific display values must be numeric or boolean')
                                if not np.isfinite(value).all():
                                    if support[row]:
                                        raise ValueError('Supported display geometry contains unmeasured numeric values')
                                    record[key] = None
                                else:
                                    record[key] = value.tolist()
                            if kind == 'sidm':
                                record['representation'] = 'measured_macro_diagnostics'
                            for key in ('display_radius','display_line_width','glyph_scale_per_eigenvalue'):
                                if key in source:
                                    record[key] = source[key]
                            epoch['objects'].append(record)
                epoch['objects'] = join_measured_sidm_diagnostics(epoch['objects'])
                object_census += len(epoch['objects'])
                if object_census > maximum:
                    raise MemoryError('Joined exact-object census exceeds its explicit scene bound')
                for record in epoch['objects']:
                    for layer in [record,*record.get('measured_geometry_layers',[])]:
                        kind = layer['kind']
                        geometry_object_census += 1
                        geometry_census += 6 if not layer['geometry_supported'] else 195 if kind == 'shape' else len(layer['points']) if kind == 'stream' else 6
                if geometry_object_census > geometry_object_limit:
                    raise MemoryError('Measured scene geometry object census exceeds its declared bound')
                if geometry_census > geometry_limit:
                    raise MemoryError('Measured geometry exceeds the declared scene vertex bound')
                for source in saved.get('lifecycle_products',[]):
                    source,producer_bindings = resolve_measured_display_source(source,request_base=request_base,
                        memory_bytes=int(memory*1024**3))
                    for native,expected in producer_bindings.items():
                        if native in bindings and bindings[native] != expected:
                            raise ValueError('Measured lifecycle producer provenance conflicts with another saved source')
                        bindings[native] = expected
                    events,encoded_size = _lifecycle_product(source,request_base=request_base,
                        destination=destination,epoch=saved['epoch_contract'],bindings=bindings,
                        remaining_events=event_limit-event_census,memory_bytes=int(memory*1024**3/8))
                    event_census += len(events)
                    lifecycle_bytes += encoded_size
                    if lifecycle_bytes*8+census*4096 > memory*1024**3:
                        raise MemoryError('Measured objects and event records exceed intake memory')
                    epoch['lifecycle_events'].extend(events)
                manifest['timeline'].append(epoch)
            manifest['assets'] = [{key:ref[key] for key in ('path','sha256','bundle') if key in ref} for epoch in manifest['timeline']
                for record in [*epoch['objects'],*epoch['lifecycle_events']] for ref in record['source_links']]
            manifest['assets'] = list({ref['path']:ref for ref in manifest['assets']}.values())
            for path,expected in bindings.items():
                if file_sha256(path) != expected:
                    raise ValueError('Scientific display assembly input changed')
            output = destination/'display.json'
            atomic_json(output,manifest)
            atomic_text(output.with_suffix('.sha256'),file_sha256(output)+'\n')
            load_verified_display_products(output)
            receipt = {'schema':'measured-display-assembly-v1','execution_status':'COMPLETE',
                'manifest':str(output),'manifest_sha256':file_sha256(output),'source_hashes':bindings,
                'object_count':object_census,'measurement_record_count':census,'lifecycle_event_count':event_census,'geometry_object_count':geometry_object_census,'geometry_vertex_estimate':geometry_census,'science_acceptance':'NOT_ESTABLISHED'}
            atomic_json(destination/'assembly.json',receipt)
            return receipt
        except BaseException as error:
            atomic_json(destination/'failure.json',{'execution_status':'FAILED',
                'failure':{'type':type(error).__name__,'message':str(error)}})
            raise


def display_products_main():
    import argparse
    parser = argparse.ArgumentParser(description='Assemble portable Blender display inputs from measured numeric arrays')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(assemble_measured_display(json.loads(args.request.read_text()),args.output,
        request_base=args.request.resolve().parent),indent=2))


# Scientific graph galleries


def _asset(root,name):
    relative=Path(name);path=(root/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root) or not path.is_file() or (root/relative).is_symlink():raise ValueError('Original figure artifact escapes its owned gallery or is absent')
    return path


def _json(path,memory):
    if path.stat().st_size*32>memory*1024**3/8:raise MemoryError('Combined gallery metadata exceeds allocation')
    return json.loads(path.read_text())


def figure_context(metadata):
    """Carry the exact original context; never match nearby epochs or labels."""
    recipe=metadata['recipe']
    return {'catalogue_id':recipe.get('catalogue_id'),'epoch':metadata.get('epoch'),
        'selections':metadata['selections'],'declared_target':recipe.get('target',recipe.get('target_identity')),
        'source_products':metadata['source_hashes']}


def validate_combined_gallery_stage(parameters):
    request=parameters['request']
    if request.get('schema')!='combined-scientific-gallery-request-v1':raise ValueError('Combined gallery request required')
    for key in ['memory_gib','disk_gib']:
        value=request[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive combined gallery allocation required')


def combined_gallery_runner(parameters,context):
    validate_combined_gallery_stage(parameters);request=parameters['request']
    for key in ['memory_gib','disk_gib']:
        if request[key]>context.resources[key]:raise MemoryError('Combined gallery exceeds allocated '+key)
    result=combine_graph_galleries(request,context.generation/'gallery')
    verify_combined_gallery(context.generation/'gallery/gallery.json',memory_gib=request['memory_gib'])
    return {'execution_status':'COMPLETE','measurements':{'combined_gallery':{'variant_count':102,'exported_views':len(result['views']),
        'pending_variants':sum(row['measurement_status']=='PENDING_MEASURED_INPUT' for row in result['entries']),'science_acceptance':'NOT_ACCEPTED'}}}


def gallery_inputs(request):
    from core.graph_producer_registry import audit_graph_producers
    from core.figures import verify_figure_export
    from core.data import verify_numpy_archive_contract,verify_numeric_audit,numeric_audit_source_bindings
    required={'schema','galleries','scenes','memory_gib','disk_gib','maximum_figures'}
    if set(request)!=required or request['schema']!='combined-scientific-gallery-request-v1':raise ValueError('Complete source-bound combined gallery request required')
    memory=request['memory_gib'];disk=request['disk_gib'];maximum=request['maximum_figures']
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in [memory,disk]) or type(maximum) is not int or maximum<1:raise ValueError('Positive bounded gallery memory/disk/figure allocation required')
    if not isinstance(request['galleries'],list) or not isinstance(request['scenes'],list):raise ValueError('Explicit original gallery and scene reference lists required')
    registry=audit_graph_producers();pins=dict(registry['source_hashes']);files={};figures=[];seen=set();total=0
    def bind(path,sha=None):
        path=Path(path).resolve();actual=file_sha256(path)
        if sha is not None and actual!=sha:raise ValueError('Original graph/scene artifact changed')
        _merge(pins,{str(path):actual});return actual
    for reference in request['galleries']:
        if set(reference)!={'path','sha256'}:raise ValueError('Exact original gallery path/hash required')
        path=Path(reference['path']).resolve();bind(path,reference['sha256']);root=path.parent;gallery=_json(path,memory)
        if gallery.get('schema')!='scientific-figure-suite-v1':raise ValueError('Original scientific figure suite required')
        if len(figures)+len(gallery['figures'])>maximum:raise MemoryError('Combined measured figure count exceeds allocation')
        for entry in gallery['figures']:
            manifest=_asset(root,entry['manifest']);sha=bind(manifest)
            if sha in seen:continue
            seen.add(sha);preliminary=_json(manifest,memory)
            for artifact in preliminary['artifacts']:
                source=_asset(manifest.parent,artifact['path'])
                if source.suffix=='.npz':
                    with zipfile.ZipFile(source) as archive:
                        if sum(info.file_size for info in archive.infolist())*4>memory*1024**3/3:raise MemoryError('Original figure plot-data expansion exceeds allocation')
            meta=verify_figure_export(manifest);scientific=meta['scientific_metadata']
            for key in ['family','variant','science_status']:
                if scientific[key]!=entry[key]:raise ValueError('Original figure and gallery scientific identity/status differ')
            if scientific['recipe']['figure_id']!=entry['figure_id'] or scientific['selections']!=entry['selections']:raise ValueError('Original figure target/selection differs from gallery entry')
            if not any(row['family']==entry['family'] and row['variant']==entry['variant'] for row in registry['entries']):raise ValueError('Figure has no original registered producer route')
            assets={};copy_paths={manifest,manifest.with_name(manifest.name+'.sha256')}
            for asset in meta['artifacts']:
                source=_asset(manifest.parent,asset['path']);bind(source,asset['sha256']);copy_paths.add(source)
                if source.suffix[1:] in ['png','svg','pdf']:assets[source.suffix[1:]]=source
            if set(assets)!={'png','svg','pdf'}:raise ValueError('All three original figure formats required')
            for extension,source in assets.items():
                organized=_asset(root,entry[extension]);bind(organized,file_sha256(source))
            arrays=_asset(root,entry['physical_arrays']);bind(arrays,entry['physical_arrays_sha256'])
            verify_numpy_archive_contract(arrays,scientific['physical_array_contract'],memory_bytes=int(memory*1024**3/3));copy_paths.add(arrays)
            if arrays.parent!=manifest.parent:raise ValueError('Original physical arrays and figure manifest must share their numerical bundle')
            for source,digest in scientific['source_hashes'].items():
                source=Path(source).resolve();bind(source,digest)
                if source.suffix=='.json':
                    data=_json(source,memory)
                    if isinstance(data,dict) and 'numeric_data' in data:
                        verified=verify_numeric_audit(source);_merge(pins,numeric_audit_source_bindings(verified))
            # Every copied artifact retains its exact original filename, bytes,
            # plot-data contract and original figure sidecar checksum.
            for source in copy_paths:
                bind(source)
                if str(source) not in files:total+=source.stat().st_size;files[str(source)]=True
            if total>disk*1024**3*.8:raise MemoryError('Combined original gallery artifacts exceed disk allocation')
            context=figure_context(scientific)
            figures.append({'original_entry':copy.deepcopy(entry),'manifest':str(manifest),'manifest_sha256':sha,
                'assets':{key:str(value) for key,value in assets.items()},'copy_paths':[str(p) for p in sorted(copy_paths)],
                'context':context,'context_sha256':canonical_digest(context),'scene_links':[]})
    for reference in request['scenes']:
        if set(reference)!={'path','sha256'}:raise ValueError('Exact original scene publication reference required')
        from core.blender import verify_export_manifest
        path=Path(reference['path']).resolve();bind(path,reference['sha256'])
        scene=verify_export_manifest(path,schema='blender-scene-product-v1')
        if scene.get('integrity_status')!='INTEGRITY_VERIFIED' or scene.get('source_integrity_status')!='INTEGRITY_VERIFIED':raise ValueError('Sealed original scene and source publication required')
        source=Path(scene['source_manifest']['path']).resolve();bind(source,scene['source_manifest']['sha256']);record=_json(source,memory)
        # Item19 supplies this explicit binding when authoring the combined
        # scene. Missing context is not filled from a matching title/PNG alone.
        bindings=record.get('scientific_figure_bindings',[])
        for binding in bindings:
            matching=[f for f in figures if f['manifest_sha256']==binding['figure_manifest_sha256']]
            if not matching:continue
            figure=matching[0]
            if binding['context']!=figure['context'] or binding['context_sha256']!=figure['context_sha256']:raise ValueError('Scene and figure target/epoch/selection contexts differ')
            png=file_sha256(figure['assets']['png'])
            if not any(item['sha256']==png for item in record.get('figure_gallery',[])):raise ValueError('Original combined scene does not contain this exact measured figure')
            figure['scene_links'].append({'scene_manifest':str(path),'scene_manifest_sha256':reference['sha256'],'source_manifest':str(source),
                'context_sha256':figure['context_sha256'],'scope':'Exact original scene/figure context and included measured image; scene physical validation remains separate.'})
    for source,sha in pins.items():
        if file_sha256(source)!=sha:raise ValueError('Gallery source changed during validation')
    return figures,pins,registry


def combine_graph_galleries(request,destination):
    from core.figures import verify_figure_export
    figures,pins,registry=gallery_inputs(request);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve earlier combined gallery')
    out.mkdir(parents=True);views=[];inventory={}
    for figure in figures:
        folder=out/'figures'/figure['manifest_sha256'];folder.mkdir(parents=True)
        for path in figure['copy_paths']:shutil.copyfile(path,folder/Path(path).name)
        manifest=folder/Path(figure['manifest']).name;copied=verify_figure_export(manifest)
        from core.data import verify_numpy_archive_contract
        verify_numpy_archive_contract(folder/Path(figure['original_entry']['physical_arrays']).name,copied['scientific_metadata']['physical_array_contract'],memory_bytes=int(request['memory_gib']*1024**3/3))
        original=figure['original_entry'];view={'view_id':figure['manifest_sha256'],'figure_id':original['figure_id'],'family':original['family'],
            'variant':original['variant'],'science_status':original['science_status'],'original_manifest':figure['manifest'],
            'manifest':str(manifest.relative_to(out)),'manifest_sha256':figure['manifest_sha256'],
            'context':figure['context'],'context_sha256':figure['context_sha256'],'scene_links':figure['scene_links'],
            **{extension:str((folder/Path(path).name).relative_to(out)) for extension,path in figure['assets'].items()},
            'physical_arrays':str((folder/Path(original['physical_arrays']).name).relative_to(out))}
        views.append(view)
        for path in folder.iterdir():inventory[str(path.relative_to(out))]=file_sha256(path)
    entries=[]
    for row in registry['entries']:
        available=[v['view_id'] for v in views if (v['family'],v['variant'])==(row['family'],row['variant'])]
        entries.append({'family':row['family'],'variant':row['variant'],'producer_status':'IMPLEMENTED','measurement_status':'EXPORTED_VIEWS_AVAILABLE' if available else 'PENDING_MEASURED_INPUT',
            'view_ids':available,'missing_reason':None if available else 'Original producer has no supplied measured export for this target/epoch/selection; no substitute graph generated.'})
    result={'schema':'combined-scientific-gallery-v1','variant_count':102,'entries':entries,'views':views,'files_sha256':inventory,
        'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED','scope':'Exactly102 implemented producer entries; source-verified measured views preserve contexts and original masks. Available exports do not imply supported science; missing outputs remain explicit.'}
    for source,sha in pins.items():
        if file_sha256(source)!=sha:raise ValueError('Original gallery source changed during bundle copying')
    write_combined_gallery(out,result)
    inventory['index.html']=file_sha256(out/'index.html')
    atomic_json(out/'gallery.json',result);atomic_text(out/'gallery.json.sha256',file_sha256(out/'gallery.json')+'\n')
    return result


def verify_combined_gallery(path,*,memory_gib=1.,verify_sources=True):
    
    from core.figures import verify_figure_export
    from core.data import verify_numpy_archive_contract
    path=Path(path).resolve();root=path.parent
    if path.with_name(path.name+'.sha256').read_text().strip()!=file_sha256(path):raise ValueError('Combined gallery manifest changed')
    record=_json(path,memory_gib)
    if record.get('schema')!='combined-scientific-gallery-v1' or record.get('variant_count')!=102:raise ValueError('Exact102 combined gallery contract required')
    expected={(family,variant) for family,variants in scientific_figure_registry()['families'].items() for variant in variants}
    keys=[(row['family'],row['variant']) for row in record['entries']]
    if len(keys)!=102 or len(set(keys))!=102 or set(keys)!=expected:raise ValueError('Combined gallery omits or repeats a registered variant')
    for name,sha in record['files_sha256'].items():
        if file_sha256(_asset(root,name))!=sha:raise ValueError('Combined gallery measured artifact changed')
    if 'index.html' not in record['files_sha256']:raise ValueError('Combined gallery navigation is not sealed')
    views={}
    for view in record['views']:
        if view['view_id'] in views:raise ValueError('Duplicate combined gallery view identity')
        views[view['view_id']]=view;manifest=_asset(root,view['manifest'])
        if file_sha256(manifest)!=view['manifest_sha256'] or view['view_id']!=view['manifest_sha256']:raise ValueError('Original combined figure identity changed')
        # The copied artifacts have already been bounded by the preparer; also
        # bound NPZ expansion here before the original figure verifier reads it.
        preliminary=_json(manifest,memory_gib)
        for artifact in preliminary['artifacts']:
            source=_asset(manifest.parent,artifact['path'])
            if source.suffix=='.npz':
                with zipfile.ZipFile(source) as archive:
                    if sum(item.file_size for item in archive.infolist())*4>memory_gib*1024**3/3:raise MemoryError('Combined figure expansion exceeds allocation')
        original=verify_figure_export(manifest);metadata=original['scientific_metadata']
        if figure_context(metadata)!=view['context'] or canonical_digest(view['context'])!=view['context_sha256']:raise ValueError('Combined figure target/epoch/selection changed')
        if any(view[key]!=metadata[key] for key in ['family','variant','science_status']):raise ValueError('Combined scientific figure identity/status changed')
        verify_numpy_archive_contract(_asset(root,view['physical_arrays']),metadata['physical_array_contract'],memory_bytes=int(memory_gib*1024**3/3))
        expected_assets={Path(asset['path']).suffix[1:]:manifest.parent/asset['path'] for asset in original['artifacts'] if Path(asset['path']).suffix in ['.png','.svg','.pdf']}
        for key in ['png','svg','pdf']:
            if _asset(root,view[key])!=expected_assets[key].resolve():raise ValueError('Combined gallery links a different figure format')
        required={manifest,manifest.with_name(manifest.name+'.sha256'),_asset(root,view['physical_arrays']),*expected_assets.values()}
        required.update(manifest.parent/asset['path'] for asset in original['artifacts'])
        if any(str(asset.relative_to(root)) not in record['files_sha256'] for asset in required):raise ValueError('Combined gallery artifact omitted from inventory')
        if verify_sources:
            from core.blender import verify_export_manifest
            for link in view['scene_links']:
                scene_path=Path(link['scene_manifest'])
                if file_sha256(scene_path)!=link['scene_manifest_sha256']:raise ValueError('Linked scene publication changed')
                scene=verify_export_manifest(scene_path,schema='blender-scene-product-v1')
                if scene.get('integrity_status')!='INTEGRITY_VERIFIED' or scene.get('source_integrity_status')!='INTEGRITY_VERIFIED':raise ValueError('Linked scene publication is unsealed')
                source=Path(scene['source_manifest']['path'])
                if str(source)!=link['source_manifest'] or file_sha256(source)!=scene['source_manifest']['sha256']:raise ValueError('Linked scene source changed')
                source_record=_json(source,memory_gib)
                binding={'figure_manifest_sha256':view['manifest_sha256'],'context':view['context'],'context_sha256':view['context_sha256']}
                if binding not in source_record.get('scientific_figure_bindings',[]) or link['context_sha256']!=view['context_sha256']:raise ValueError('Linked scene has a different scientific context')
                if not any(item['sha256']==file_sha256(_asset(root,view['png'])) for item in source_record.get('figure_gallery',[])):raise ValueError('Linked scene omits original figure')
    for entry in record['entries']:
        wanted=[view['view_id'] for view in record['views'] if (view['family'],view['variant'])==(entry['family'],entry['variant'])]
        status='EXPORTED_VIEWS_AVAILABLE' if wanted else 'PENDING_MEASURED_INPUT'
        if entry['view_ids']!=wanted or entry['measurement_status']!=status or entry['producer_status']!='IMPLEMENTED':raise ValueError('Combined gallery changes measured/missing/producer status')
    if verify_sources:
        for name,sha in record['source_hashes'].items():
            if file_sha256(name)!=sha:raise ValueError('Combined gallery original source changed')
    return record


def write_combined_gallery(out,result):
    cards=[];views={row['view_id']:row for row in result['views']}
    escape=lambda value:html.escape(str(value),quote=True)
    for entry in result['entries']:
        title=entry['family']+'/'+entry['variant'];body=[]
        for view_id in entry['view_ids']:
            view=views[view_id]
            links=' · '.join('<a href="'+escape(view[key])+'">'+key.upper()+'</a>' for key in ['png','svg','pdf','manifest','physical_arrays'])
            scenes=''.join('<p><a href="'+escape(quote(os.path.relpath(link['scene_manifest'],out),safe='/'))+'">Exact-context scene publication</a></p>' for link in view['scene_links'])
            body.append('<section><p>'+escape(view['science_status'])+'</p><details><summary>Exact target, epoch and selection</summary><pre>'+escape(json.dumps(view['context'],indent=2))+'</pre></details><p>'+links+'</p>'+scenes+'<a href="'+escape(view['png'])+'"><img loading="lazy" src="'+escape(view['png'])+'" alt="'+escape(title)+'"></a></section>')
        if not body:body=['<p>'+escape(entry['measurement_status'])+' — '+escape(entry['missing_reason'])+'</p>']
        cards.append('<article data-search="'+escape((title+' '+entry['measurement_status']).casefold())+'"><h2>'+escape(title)+'</h2>'+''.join(body)+'</article>')
    atomic_text(Path(out)/'index.html','<!doctype html><html lang="en"><meta charset="utf-8"><title>102 scientific graph variants</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}img{max-width:100%}article{border-bottom:1px solid #bbb;padding:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}input{font:inherit;width:90%;padding:.6rem}</style><h1>102 scientific graph variants</h1><p>Measured exports retain original support and acceptance. Missing measurements are shown explicitly.</p><label>Filter family, variant or measured status <input id="filter" type="search"></label>'+''.join(cards)+r'<script>document.getElementById("filter").addEventListener("input",function(){const terms=this.value.toLowerCase().split(/\s+/).filter(Boolean);document.querySelectorAll("article[data-search]").forEach(card=>{card.hidden=!terms.every(term=>card.dataset.search.includes(term));});});</script></html>')


# Attach measured products to a native scene


def attach_scene_products(request,combined,root):
    """Author attachments during a fresh original combined-scene export."""
    import yaml
    from core.blender import verify_timeline_assets
    if set(request)-{'encounter_overlay','native_selections'}!={'schema','memory_gib','disk_gib','streams','scientific_gallery'} or request['schema']!='combined-scene-attachments-request-v1':raise ValueError('Complete combined scene attachment request required')
    for key in ['memory_gib','disk_gib']:
        value=request[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive explicit scene attachment allocation required')
    if not isinstance(request['streams'],list):raise ValueError('Explicit saved stream attachment list required')
    timeline=verify_timeline_assets(Path(root)/combined['species_manifests']['dm'])
    config=Path(combined['run_dir'])/'config.yaml'
    if file_sha256(config)!=timeline['data_contract']['config_sha256']:raise ValueError('Scene configuration differs from original timeline provenance')
    h=yaml.safe_load(config.read_text())['cosmology']['H0']/100
    if combined.get('hubble_parameter')!=h:raise ValueError('Combined scene cosmology differs from original configuration')
    sources=[];seen=set()
    for stream in request['streams']:
        if set(stream)!={'inspector','maximum_saved_vertices','marker_radius_kpc'}:raise ValueError('Complete bounded native stream attachment required')
        path=_reference(stream['inspector']);identity=stream['inspector']['sha256']
        if identity in seen:raise ValueError('Duplicate stream attachment')
        seen.add(identity);sources.extend([path,path.with_name('inspector.npz')])
    if sum(path.stat().st_size for path in sources)>request['disk_gib']*1024**3*.8:raise MemoryError('Stream attachment copies exceed disk allocation')
    combined['stream_timelines']=[prepare_stream_scene_bundle(stream['inspector'],combined,root,maximum_saved_vertices=stream['maximum_saved_vertices'],
        marker_radius_kpc=stream['marker_radius_kpc'],memory_gib=request['memory_gib']) for stream in request['streams']]
    gallery=request['scientific_gallery']
    if gallery is not None:
        if set(gallery)-{'analysis_binding'}!={'reference','view_ids'}:raise ValueError('Exact scientific gallery and selected views required')
        attach_scientific_gallery(gallery['reference'],gallery['view_ids'],combined,root,memory_gib=request['memory_gib'],disk_gib=request['disk_gib'],analysis_binding=gallery.get('analysis_binding'))
    if request.get('encounter_overlay') is not None:
        from core.native_scene_overlay import prepare_native_scene_overlay
        prepare_native_scene_overlay(request['encounter_overlay'],combined,root,memory_gib=request['memory_gib'])
    if request.get('native_selections') is not None:
        prepare_native_scene_selections(request['native_selections'],combined,root,memory_gib=request['memory_gib'])
    attachment_files=[path for name in ['stream_timelines','native_encounter_overlay','measured_display','native_selections'] for path in (Path(root)/name).rglob('*') if path.is_file()]
    if gallery is not None:
        attachment_files.extend(Path(root)/entry['path'] for entry in combined['figure_gallery'] if 'scientific_context' in entry)
    if sum(path.stat().st_size for path in attachment_files)>request['disk_gib']*1024**3*.8:raise MemoryError('Complete scene attachments exceed disk allocation')
    combined['attachments_request_sha256']=canonical_digest(request)
    combined['attachments_configuration_sha256']=file_sha256(config)
    from core import source_path
    names=['research.display','native_scene_overlay','graph_producer_registry','data','stream.measurements','blender','blender_scene','foundation']
    combined['attachment_implementation_sha256']={str(source_path(name).resolve()):file_sha256(source_path(name)) for name in names}
    atomic_json(Path(root)/'scene-attachments-request.json',request)


def _reference(reference):
    if set(reference)!={'path','sha256'}:raise ValueError('Exact scene attachment path/hash required')
    path=Path(reference['path']).resolve()
    if file_sha256(path)!=reference['sha256']:raise ValueError('Scene attachment source changed')
    return path


def prepare_stream_scene_bundle(reference,combined,root,*,maximum_saved_vertices,memory_gib,marker_radius_kpc):
    """Keep original proper coordinates; record the one display-only scale."""
    from core.stream_common import read_product
    from core.data import numeric_audit_source_bindings
    from core.blender import saved_stream_frame_schedule
    inspector=_reference(reference)
    if inspector.stat().st_size*32>memory_gib*1024**3/8:raise MemoryError('Stream inspector metadata exceeds allocation')
    record=json.loads(inspector.read_text())
    if record.get('schema')!='stream-inspector-timeline-v1':raise ValueError('Original saved stream timeline required')
    if (record.get('length_units'),record.get('velocity_units'),record.get('time_units'))!=('proper kpc','km / s','Myr'):raise ValueError('Original native stream physical units required')
    source=Path(record['product']).resolve()
    if file_sha256(source)!=record['product_sha256']:raise ValueError('Original stream product changed')
    report,arrays,pins=read_product(source,memory_gib=memory_gib/4)
    _merge(pins, numeric_audit_source_bindings(report))
    for key,unit in [('position_kpc','proper kpc'),('velocity_kms','km / s'),('times_myr','Myr'),('mass_msun','Msun'),('particle_ids','identity')]:
        if report['numeric_data']['array_contract']['datasets'][key]['units']!=unit:raise ValueError('Native stream source unit differs: '+key)
    if report.get('schema')!='measured-native-stream-worldlines-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Original native box-coordinate stream histories required')
    if Path(report['native_run_directory']).resolve()!=Path(combined['run_dir']).resolve():raise ValueError('Stream belongs to a different native scene run')
    if record['namespace']!=report['namespace'] or record['epochs']!=report['epochs']:raise ValueError('Stream inspector changes original identity or epochs')
    archive=inspector.with_name('inspector.npz')
    if file_sha256(archive)!=record['arrays_sha256']:raise ValueError('Original stream inspector arrays changed')
    import zipfile
    with zipfile.ZipFile(archive) as packed:
        if sum(row.file_size for row in packed.infolist())*4>memory_gib*1024**3/4:raise MemoryError('Stream scene attachment exceeds allocation')
    with np.load(archive,allow_pickle=False) as packed:
        for key in ['position_kpc','velocity_kms','particle_ids','mass_msun','times_myr']:
            original=arrays[key]
            if key=='mass_msun' and original.ndim==1:original=np.broadcast_to(original,arrays['position_kpc'].shape[:2])
            if not np.array_equal(packed[key],original):raise ValueError('Stream inspector differs from original native arrays')
    # Preserve saved proper-kpc trajectories. The attachment records one scale
    # per epoch: x_comoving[Mpc/h] = x_proper[kpc] * h/(1000*a).
    positions=arrays['position_kpc'];ids=arrays['particle_ids'];epochs=report['epochs']
    if type(maximum_saved_vertices)is not int or maximum_saved_vertices<1 or positions.shape!=(len(epochs),len(ids),3) or positions.shape[0]*positions.shape[1]>maximum_saved_vertices:raise MemoryError('Bounded original saved stream vertex census required')
    if ids.dtype!=np.dtype('uint64') or len(np.unique(ids))!=len(ids) or not np.isfinite(positions).all():raise ValueError('Exact identities and finite original stream positions required')
    if type(marker_radius_kpc) not in (float,int) or not math.isfinite(marker_radius_kpc) or marker_radius_kpc<=0:raise ValueError('Positive proper stream marker radius required')
    from core.blender import verify_timeline_assets
    timeline=verify_timeline_assets(Path(root)/combined['species_manifests']['dm'])
    h=combined['hubble_parameter']
    if type(h) not in (float,int) or not math.isfinite(h) or h<=0:raise ValueError('Original positive cosmological h required')
    saved=timeline['snapshots'];stride=combined['frames_per_snapshot'];end=(len(saved)-1)*stride
    mapping={row['snapshot']:(i,row) for i,row in enumerate(saved)};frames=[];scales=[]
    for index,epoch in enumerate(epochs):
        if epoch.get('canonical_time')!='scale_factor':raise ValueError('Cosmological scene requires actual cosmological stream epochs')
        snapshot=epoch['snapshot_storage_index']
        if snapshot not in mapping:raise ValueError('Stream saved epoch absent from scene schedule')
        ordinal,row=mapping[snapshot]
        if epoch['scale_factor']!=row['scale_factor'] or arrays['times_myr'][index]!=1000*row['cosmic_time_gyr']:raise ValueError('Stream and scene actual saved clocks differ')
        frames.append(1 if ordinal==0 else ordinal*stride);scales.append(h/(1000*epoch['scale_factor']))
    saved_stream_frame_schedule(arrays['times_myr'].tolist(),scene_frames=frames,scene_frame_end=end)
    _merge(pins, {str(inspector):reference['sha256'],str(archive):record['arrays_sha256']})
    if any(file_sha256(path)!=sha for path,sha in pins.items()):raise ValueError('Native stream attachment provenance changed')
    folder=Path(root)/'stream_timelines'/reference['sha256'];folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(inspector,folder/'inspector.json');shutil.copyfile(archive,folder/'inspector.npz')
    attachment={'schema':'combined-native-stream-attachment-v1','inspector_sha256':reference['sha256'],'arrays_sha256':record['arrays_sha256'],
        'native_run_directory':report['native_run_directory'],'namespace':report['namespace'],'epochs':epochs,'saved_times_myr':arrays['times_myr'].tolist(),
        'scene_frames':frames,'scene_frame_end':end,'coordinate_scales':scales,'hubble_parameter':h,'maximum_saved_vertices':maximum_saved_vertices,'marker_radius_kpc':marker_radius_kpc,
        'coordinate_meaning':'Native global box origin proper kpc multiplied once by h/(1000*a) for comoving Mpc/h display; no trajectory interpolation or periodic winding inferred.',
        'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(folder/'attachment.json',attachment)
    return {'path':str((folder/'attachment.json').relative_to(root)),'sha256':file_sha256(folder/'attachment.json')}


def attach_scientific_gallery(reference,view_ids,combined,root,*,memory_gib,disk_gib,analysis_binding=None):
    """Copy exact measured images and their original scientific context bindings."""
    path=_reference(reference);record=verify_combined_gallery(path,memory_gib=memory_gib)
    if not isinstance(view_ids,list) or len(set(view_ids))!=len(view_ids):raise ValueError('Distinct explicit scientific figure views required')
    views={view['view_id']:view for view in record['views']}
    if any(identity not in views for identity in view_ids):raise ValueError('Scientific scene view is absent from original gallery')
    if sum((path.parent/views[identity]['png']).stat().st_size for identity in view_ids)>disk_gib*1024**3*.8:raise MemoryError('Scientific gallery copies exceed scene disk allocation')
    private_sources={}
    if analysis_binding is not None:
        from core.flagship_runtime_measurements import private_analysis_scene_sources
        private_sources,_,evidence=private_analysis_scene_sources(analysis_binding,combined['run_dir'],memory_gib)
        combined['scientific_gallery_analysis_lineage']=evidence
    combined.setdefault('scientific_figure_bindings',[])
    for identity in view_ids:
        view=views[identity]
        from core.data import verify_numeric_audit,numeric_audit_source_bindings
        pins=dict(view['context']['source_products'])
        for name in list(pins):
            product=Path(name)
            if product.suffix=='.json':
                original=json.loads(product.read_text())
                if isinstance(original,dict) and 'numeric_data' in original:
                    _merge(pins, numeric_audit_source_bindings(verify_numeric_audit(product)))
        run=Path(combined['run_dir']).resolve()
        if not any(Path(name).resolve().is_relative_to(run) or private_sources.get(str(Path(name).resolve()))==sha for name,sha in pins.items()):raise ValueError('Scientific figure has no original provenance in this native scene run')
        epoch=view['context']['epoch']
        if isinstance(epoch,dict):
            scale=epoch.get('actual_scale_factor',epoch.get('scale_factor'))
            if scale is not None:
                from core.blender import verify_timeline_assets
                timeline=verify_timeline_assets(Path(root)/combined['species_manifests']['dm'])
                if scale not in [row['scale_factor'] for row in timeline['snapshots']]:raise ValueError('Scientific figure epoch absent from scene saved schedule')
        source=path.parent/view['png'];target=Path(root)/'figures'/(identity+'.png');target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,target)
        combined['figure_gallery'].append({'path':str(target.relative_to(root)),'title':view['family']+'/'+view['variant'],'sha256':file_sha256(target),
            'scientific_context':view['context'],'scientific_context_sha256':view['context_sha256']})
        combined['scientific_figure_bindings'].append({'figure_manifest_sha256':view['manifest_sha256'],'context':view['context'],'context_sha256':view['context_sha256']})
    combined['scientific_gallery_source']={'path':str(path),'sha256':reference['sha256'],'selected_view_ids':view_ids}


# Progenitor memberships and periodic support apertures


def prepare_native_scene_selections(request,combined,root,*,memory_gib):
    from core.stream_common import read_product
    from core.data import numeric_audit_source_bindings
    from core.blender import verify_timeline_assets
    required={'progenitor_views','contamination','maximum_points','maximum_apertures','marker_radius_kpc'}
    if set(request)!=required:raise ValueError('Complete native membership/aperture scene request required')
    for name in ['maximum_points','maximum_apertures']:
        if type(request[name]) is not int or request[name]<1:raise ValueError('Explicit positive native selection display bound required')
    radius=request['marker_radius_kpc']
    if type(radius) not in (int,float) or not np.isfinite(radius) or radius<=0:raise ValueError('Positive symbolic stellar marker radius required')
    timeline=verify_timeline_assets(Path(root)/combined['species_manifests']['dm']);saved=timeline['snapshots'];stride=combined['frames_per_snapshot'];end=(len(saved)-1)*stride
    lookup={row['snapshot']:(i,row) for i,row in enumerate(saved)};run=Path(combined['run_dir']).resolve();h=combined['hubble_parameter'];pins={};points=0
    folder=Path(root)/'native_selections';folder.mkdir(exist_ok=False);overlays=[];apertures=[];unresolved=[];rendered_apertures=0
    def sources(values):
        if not values or not any(Path(path).resolve().is_relative_to(run) for path in values):raise ValueError('Native display selection lacks this scene run provenance')
        for path,sha in values.items():
            if path in pins and pins[path]!=sha or file_sha256(path)!=sha:raise ValueError('Native scene selection source changed')
            pins[path]=sha
    def clock(snapshot,a):
        if type(snapshot) is not int or snapshot<0 or type(a) not in (int,float) or not np.isfinite(a) or a<=0:raise ValueError('Exact native snapshot label and positive physical scale factor required')
        if snapshot not in lookup:raise ValueError('Native selection saved snapshot absent from scene')
        ordinal,row=lookup[snapshot]
        if row['scale_factor']!=a:raise ValueError('Native selection actual saved clock differs')
        first=1 if ordinal==0 else ordinal*stride;last=(ordinal+1)*stride-1 if ordinal+1<len(saved) else end
        return first,last
    for reference in request['progenitor_views']:
        path=_reference(reference);report,arrays,bindings=read_product(path,memory_gib=memory_gib/4)
        if report.get('schema')!='native-stream-progenitor-view-v1':raise ValueError('Original measured progenitor membership view required')
        _merge(bindings, numeric_audit_source_bindings(report))
        sources(bindings)
        context=report.get('source_context')
        if not isinstance(context,dict):raise ValueError('Original native host chart context required; regenerate the progenitor view')
        snapshot=context['snapshot'];a=context['scale_factor'];first,last=clock(snapshot,a)
        if context['h']!=h:raise ValueError('Progenitor source chart changes cosmological h')
        center=np.asarray(context['host_center_comoving_kpc_h'],float)/1000
        box=float(context['box_comoving_kpc_h'])/1000
        if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('Original measured native host center required')
        if not np.isfinite(box) or box<=0:raise ValueError('Original positive native periodic box required')
        owned=folder/reference['sha256'];owned.mkdir();originals=[path,path.with_name(path.name+'.sha256')]
        bundle=path.with_name(path.name+'.data');shutil.copytree(bundle,owned/bundle.name)
        for source in originals:shutil.copyfile(source,owned/source.name)
        for kind in ['progenitor','stream']:
            xyz=arrays[kind+'_position_kpc'];ids=arrays[kind+'_particle_ids'];support=arrays[kind+'_supported'];points+=len(ids)
            for name,unit in [(kind+'_position_kpc','proper kpc'),(kind+'_particle_ids','identity')]:
                if report['numeric_data']['array_contract']['datasets'][name]['units']!=unit:raise ValueError('Native progenitor overlay units differ')
            if xyz.shape!=(len(ids),3) or ids.dtype!=np.dtype('uint64') or len(np.unique(ids))!=len(ids) or not np.isfinite(xyz).all() or support.dtype.kind!='b' or support.shape!=(len(ids),) or not support.all():raise ValueError('Original finite supported exact native membership required')
            if points>request['maximum_points']:raise MemoryError('Native progenitor display point budget exceeded')
            if points*1024>memory_gib*1024**3*.5:raise MemoryError('Native stellar display instance workspace estimate exceeds allocation')
            if not len(ids):unresolved.append({'source':str(path),'kind':kind,'status':'NOT_OBSERVED','epoch':report['epoch']});continue
            # xyz is the measured host-centered proper position in kpc.
            # h/(1000*a) converts it to comoving Mpc/h; center has those units.
            # Integer box images wrap this display once, preserving source xyz.
            scale = h/(1000*a)
            global_xyz = xyz*scale + center
            images=-np.floor(global_xyz/box)
            if not np.isfinite(images).all() or np.any(np.abs(images)>2**52):raise ValueError('Native display image exceeds exact integer range')
            images=images.astype(np.int64);display_xyz=xyz+images*(box/scale)
            target=owned/kind;target.mkdir();np.savez_compressed(target/'inspector.npz',position_kpc=display_xyz,particle_ids=ids,source_position_kpc=xyz,display_image_offsets=images)
            metadata={'schema':'stream-inspector-v1','product':'../'+path.name,'product_sha256':reference['sha256'],'arrays_sha256':file_sha256(target/'inspector.npz'),
                'namespace':report['namespace'],'tracks':[],'science_acceptance':'NOT_ACCEPTED','role':'current bound progenitor member stars' if kind=='progenitor' else 'selected observed stream debris',
                'periodic_display_policy':'Recorded host chart transformed once and reduced to the original periodic box at this one saved epoch; original source positions and exact integer display images retained; no temporal winding inferred'}
            atomic_json(target/'inspector.json',metadata)
            overlays.append({'path':str((target/'inspector.json').relative_to(root)),'sha256':file_sha256(target/'inspector.json'),'arrays_sha256':metadata['arrays_sha256'],
                'first_frame':first,'last_frame':last,'coordinate_scale':h/(1000*a),'origin_comoving_mpc_h':center.tolist(),'marker_radius_kpc':radius,
                'maximum_points':len(ids),'role':metadata['role'],'epoch':report['epoch'],'namespace':report['namespace'],'progenitor_keys':report['progenitor_keys']})
    for reference in request['contamination']:
        path=_reference(reference)
        if path.stat().st_size*32>memory_gib*1024**3/8:raise MemoryError('Contamination scene metadata exceeds allocation')
        report=json.loads(path.read_text());sources(report['source_hashes']);pins[str(path)]=reference['sha256']
        if report.get('schema') not in ['native-zoom-contamination-history-v1','native-science-aperture-contamination-v1']:raise ValueError('Original native contamination measurements required')
        for ordinal,row in enumerate(report['records']):
            measured=row.get('native_contamination')
            if measured is None:unresolved.append({'source':str(path),'status':row['status'],'selection':row.get('id',row.get('host_row'))});continue
            first,last=clock(measured['snapshot'],measured['scale_factor'])
            if measured['h']!=h:raise ValueError('Contamination aperture changes original native h')
            for source in measured['source_sha256']:
                sources({source['path']:source['sha256']})
            box=float(measured['box_comoving_kpc_h'])/1000
            center=np.asarray(measured['center_comoving_kpc_h'],float)/1000;radius=float(measured['radius_proper_kpc'])*h/(1000*measured['scale_factor'])
            if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:raise ValueError('Measured finite native aperture required')
            if not np.isfinite(box) or not 0<radius<box/2:raise ValueError('Native support aperture must fit within half its recorded periodic box')
            # A sphere crossing a box face needs a neighboring display image.
            # These copies represent one aperture; their masses must not add.
            center=center%box;choices=[[0]+([1] if value-radius<0 else [])+([-1] if value+radius>box else []) for value in center]
            images=[list(image) for image in itertools.product(*choices)];rendered_apertures+=len(images)
            apertures.append({'source_sha256':reference['sha256'],'source_record':ordinal,'selection':row.get('selection_identity',row.get('host_row')),
                'center_comoving_mpc_h':center.tolist(),'radius_comoving_mpc_h':radius,'box_comoving_mpc_h':box,'periodic_display_images':images,'first_frame':first,'last_frame':last,
                'measurement':measured,'maximum_coarse_mass_fraction':report['maximum_coarse_mass_fraction'],'status':row['status']})
            if rendered_apertures>request['maximum_apertures']:raise MemoryError('Native contamination aperture/periodic-copy display budget exceeded')
            if points*1024+rendered_apertures*195*256>memory_gib*1024**3*.5:raise MemoryError('Native selection/periodic-aperture display workspace estimate exceeds allocation')
    for path,sha in pins.items():
        if file_sha256(path)!=sha:raise ValueError('Native selection source changed during scene preparation')
    result={'schema':'native-scene-selections-v1','progenitor_overlays':overlays,'contamination_apertures':apertures,'unresolved':unresolved,'source_hashes':pins,
        'rendered_periodic_aperture_spheres':rendered_apertures,
        'scope':'Highlights retain exact original stellar memberships; glyphs are duplicate display selections, not additional physical particles. Spheres are measured support apertures, not inferred stream tubes. Neighboring periodic sphere copies show the same aperture at crossed box faces; never sum their repeated masses.','science_acceptance':'NOT_ACCEPTED'}
    atomic_json(folder/'selections.json',result)
    combined['native_selections']={'path':str((folder/'selections.json').relative_to(root)),'sha256':file_sha256(folder/'selections.json')}
    return result


# Scientific figures

FAMILIES={
 'cosmology':('input_power','evolved_power','power_ratios','transfer_comparison','matched_residuals'),
 'abundance':('differential_hmf','cumulative_hmf','hmf_ratio','reliability_limits','stellar_mass_function','vmax_function','rare_high_z_abundance'),
 'structure':('density_profile','circular_velocity','velocity_dispersion','anisotropy','concentration_mass','radial_axis_ratios','triaxiality','axis_orientation_twist','density_potential_shape','splashback_slope_radius_depth_width'),
 'assembly':('mass_scale_factor','log_mass_accretion','fast_slow_transition','spin_history','concentration_history','shape_history','pericenter_infall_merger_events','stellar_assembly','quenching_morphology'),
 'web':('filament_map','spin_alignment','shape_alignment','web_evolution'),
 'streams':('stream_progenitor_3d','intrinsic_track','projected_track','curvature','convexity','linear_density','width','velocity','dispersion','gaps','spurs','bifurcations','action_angle','density_power','track_power'),
 'encounters':('worldlines_3d','impact_velocity_geometry','perturber_mass_profile','before_after_controls','gap_growth','action_diffusion','bh_extended_subhalo_comparison'),
 'black_holes':('bh_growth','eddington_ratio','pair_separation','merger_timeline','recoil_trajectory','feedback_energy','dm_density_response','dm_shape_response'),
 'baryons':('gas_phase_diagram','star_formation_history','mass_return','metals_history','luminosity_history','color_history','first_cooling_birth','primordial_cloud_morphology'),
 'sidm':('sidm_core_evolution','collapse_phase','scattering_rates','sidm_cdm_controls',
         'sidm_density_dispersion_anisotropy','sidm_shape_baryon_misalignment',
         'sidm_heat_flux','sidm_kernel_support','sidm_satellite_survival',
         'sidm_merger_centroid_offsets','sidm_stream_profile_response',
         'sidm_lensing_h0_response','sidm_baryonic_degeneracy'),
 'hubble':('hubble_method_comparison','distance_ladder','bao_cmb_background','time_delay_posteriors'),
 'inference':('actual_posterior','posterior_predictive','sbc_ranks','coverage','ood_support','graph_baseline_gain'),
 'convergence':('mass_resolution','softening','timestep','output_cadence','start_redshift','finder_tree_variation')}


def scientific_figure_registry():
    """Catalogue names only; registration does not certify measurement producers."""
    return {'schema':'scientific-figure-registry-v1',
        'families':{family:list(variants) for family,variants in FAMILIES.items()},
        'variant_count':sum(map(len,FAMILIES.values())),
        'missing_inputs':'PENDING_MEASURED_INPUT',
        'producer':'core.research.display.export_scientific_suite',
        'scope':'Registered rendering variants; producer connectivity and scientific acceptance require separate evidence'}


def reconcile_scientific_figure_registry(record):
    """Report missing/extra names and bad counts without promoting acceptance."""
    expected=scientific_figure_registry()
    if not isinstance(record,dict) or record.get('schema')!=expected['schema']:
        raise ValueError('Scientific figure registry schema required')
    families=record.get('families')
    if not isinstance(families,dict) or any(not isinstance(names,list) or
            any(not isinstance(name,str) for name in names) for names in families.values()):
        raise ValueError('Figure registry requires family lists of variant names')
    actual={(family,name) for family,names in families.items() for name in names}
    required={(family,name) for family,names in expected['families'].items() for name in names}
    duplicates=sorted(family for family,names in families.items() if len(names)!=len(set(names)))
    missing=sorted(required-actual);extra=sorted(actual-required)
    count=sum(map(len,families.values()))
    counts_match=type(record.get('variant_count')) is int and record['variant_count']==count==expected['variant_count']
    metadata_match=all(record.get(key)==expected[key] for key in ('producer','missing_inputs'))
    unknown_families=sorted(set(families)-set(expected['families']))
    return {'status':'CURRENT' if not (missing or extra or duplicates or unknown_families) and counts_match and metadata_match else 'STALE_OR_INVALID',
        'missing_variants':[{'family':family,'variant':name} for family,name in missing],
        'extra_variants':[{'family':family,'variant':name} for family,name in extra],
        'duplicate_families':duplicates,'unknown_families':unknown_families,
        'declared_count':record.get('variant_count'),'actual_count':count,
        'expected_count':expected['variant_count'],'metadata_match':metadata_match,
        'science_acceptance':'NOT_ESTABLISHED_BY_REGISTRATION'}


def _resolve_array(series,name,cache):
    source=series['source'];report,arrays,provenance=cache[source]
    key=series[name];value=np.asarray(arrays[key])
    epoch=series.get(name+'_epoch_index',series.get('epoch_index') if value.ndim>1 else None)
    if epoch is not None:
        if type(epoch) is not int or value.ndim == 0 or not 0 <= epoch < len(value):
            raise ValueError('Figure epoch index must select a retained array row')
        value = value[epoch]
    if series.get(name+'_column') is not None:value=value[...,series[name+'_column']]
    conversion=series.get('unit_conversion',{}).get(name)
    if conversion is not None:
        factor=float(conversion['factor']);offset=float(conversion.get('offset',0.))
        if not np.isfinite([factor,offset]).all() or factor==0:raise ValueError('Finite nonzero declared unit conversion required')
        # A standard deviation changes by the positive unit scale, not by
        # the offset used to transform a measured coordinate or temperature.
        if name == 'error' and (factor <= 0 or offset != 0):
            raise ValueError('Uncertainty conversion needs a positive scale and zero offset')
        value=value*factor+offset
    return value


def _panel(panel,axis,cache,physical_arrays,units):
    import textwrap
    title=panel['title']+' — '+panel['quality_status'].replace('_',' ')
    axis.set_title(textwrap.fill(title,width=76),fontsize=9)
    axis.set_xlabel(textwrap.fill(panel['xlabel'],width=64),fontsize=9)
    axis.set_ylabel(textwrap.fill(panel['ylabel'],width=54),fontsize=9)
    axis._physical_plot_units={'x':panel['x_units'],'y':panel['y_units'],'z':panel.get('z_units','dimensionless'),'color':panel.get('color_units','dimensionless')}
    drawn=0;counts=[]
    for j,series in enumerate(panel['series']):
        source=series['source'];report,arrays,provenance=cache[source]
        if series['kind'] in {'histogram','histogram2d','quantile_band','heatmap','contour','vector','tensor','sky','graph','correlation','pca'}:
            from core.figures import draw_measured_representation
            bindings=series.get('array_bindings',{})
            if not bindings:
                raise ValueError('Measured representation requires explicit source-array bindings')
            declarations=report['numeric_data']['array_contract']['datasets']
            supplied={};source_units={}
            for role,array_name in bindings.items():
                if array_name not in declarations:
                    raise ValueError('Representation array lacks a numeric contract: '+array_name)
                supplied[role]=_resolve_array({**series,role:array_name},role,cache)
                declared=declarations[array_name]['units']
                expected=series.get('binding_units',{}).get(role)
                conversion=series.get('unit_conversion',{}).get(role)
                if conversion is not None and (conversion.get('from_units')!=declared or conversion.get('to_units')!=expected):
                    raise ValueError('Representation conversion must bind source and target units: '+role)
                source_units[role]=expected if conversion is not None else declared
                if expected is None or expected!=source_units[role]:
                    raise ValueError('Representation must declare matching units for '+role)
            if series['kind']=='sky' and any(source_units.get(role)!='rad' for role in ('x','y')):
                raise ValueError('Sky input coordinates must declare rad units')
            rendered=draw_measured_representation(axis,series['kind'],supplied,
                {**series.get('options',{}),'label':series['label']})
            output_units=series.get('output_units',{})
            if set(output_units)!=set(rendered) or any(not isinstance(unit,str) or not unit for unit in output_units.values()):
                raise ValueError('Every derived representation array requires explicit output units')
            prefix=panel['panel_id']+'__series_%03d_'%j
            for role,value in rendered.items():
                physical_arrays[prefix+role]=value
                units[prefix+role]=output_units[role]
            support=rendered.get('sample_support',rendered.get('support'))
            counts.append({'label':series['label'],'supported':int(np.count_nonzero(support)) if support is not None else None,
                'masked':int(np.count_nonzero(~support)) if support is not None else None,
                'representation':series['kind']})
            drawn+=1
            continue
        if series['kind']=='posterior':
            if report.get('status')!='INFERRED' and not report.get('chain_written'):raise ValueError('Posterior panel requires actual persisted inference samples')
            if 'posterior_samples' not in arrays or not len(arrays['posterior_samples']):raise ValueError('No real posterior samples to draw')
        for coordinate in ('x','y','z','color','error'):
            if coordinate not in series:continue
            declared=report['numeric_data']['array_contract']['datasets'][series[coordinate]]['units']
            expected=panel[('y' if coordinate=='error' else coordinate)+'_units']
            conversion=series.get('unit_conversion',{}).get(coordinate)
            if conversion is not None and (conversion.get('from_units')!=declared or conversion.get('to_units')!=expected):raise ValueError('Unit conversion must identify both source and target physical units')
            if declared!=expected and conversion is None:raise ValueError('Figure unit recipe differs from sealed source array: '+series[coordinate])
        x=_resolve_array(series,'x',cache);y=_resolve_array(series,'y',cache);key=panel['panel_id']+'__series_%03d_'%j
        if x.shape!=y.shape:raise ValueError('Panel x/y array shape differs')
        mask=np.isfinite(x)&np.isfinite(y)
        if series.get('support'):
            supported = _resolve_array(series,'support',cache)
            if supported.dtype.kind != 'b' or supported.shape != mask.shape:
                raise ValueError('Physical support needs exact booleans matching the figure arrays')
            physical_arrays[key+'source_supported'] = supported.copy()
            units[key+'source_supported'] = 'dimensionless'
            mask &= supported
        kind=series['kind'];label=series['label'];extras={}
        for coordinate in ({'errorbar':('error',),'line3d':('z',),'scatter3d':('z',),'phase':('color',)}.get(kind,())):
            value=_resolve_array(series,coordinate,cache)
            if value.shape!=x.shape:raise ValueError('Additional figure array shape differs')
            if coordinate=='error' and np.any(value[mask]<0):raise ValueError('Measured errors must be nonnegative')
            mask&=np.isfinite(value);extras[coordinate]=value
            physical_arrays[key+coordinate]=value.copy();units[key+coordinate]=panel[('y' if coordinate=='error' else coordinate)+'_units']
        # Save the plotted domain explicitly: finite, physically supported
        # points with positive coordinates on each logarithmic axis.
        if panel.get('xscale') == 'log':
            mask &= x > 0
        if panel.get('yscale') == 'log':
            mask &= y > 0
        physical_arrays[key+'x']=x.copy();physical_arrays[key+'y']=y.copy();physical_arrays[key+'supported']=mask.copy()
        units[key+'x']=panel['x_units'];units[key+'y']=panel['y_units'];units[key+'supported']='dimensionless'
        counts.append({'label':label,'supported':int(mask.sum()),'masked':int((~mask).sum())})
        if not mask.any():continue
        if kind in {'line','posterior'}:axis.plot(np.where(mask,x,np.nan),np.where(mask,y,np.nan),label=label)
        elif kind=='scatter':axis.scatter(x[mask],y[mask],s=series.get('marker_size',4),label=label)
        elif kind=='errorbar':axis.errorbar(x[mask],y[mask],yerr=extras['error'][mask],fmt='.',label=label)
        elif kind in {'line3d','scatter3d'}:
            z=extras['z']
            if kind=='line3d':axis.plot(np.where(mask,x,np.nan),np.where(mask,y,np.nan),np.where(mask,z,np.nan),label=label)
            else:axis.scatter(x[mask],y[mask],z[mask],s=series.get('marker_size',4),label=label)
            axis.set_zlabel(panel['zlabel'])
        elif kind=='phase':
            artist=axis.scatter(x[mask],y[mask],c=extras['color'][mask],s=series.get('marker_size',4));axis.figure.colorbar(artist,ax=axis,label=panel['color_label'])
        else:raise ValueError('Unknown measured figure series kind: '+kind)
        drawn+=1
    for event in panel.get('events',[]):
        axis.axvline(event['x'],linestyle=':',alpha=.5,label=event['label'])
    for limit in panel.get('reliability_limits',[]):axis.axvline(limit['x'],linestyle='--',label=limit['label'])
    if panel.get('xscale'):axis.set_xscale(panel['xscale'])
    if panel.get('yscale'):axis.set_yscale(panel['yscale'])
    for coordinate in ('x','y','z'):
        if coordinate+'lim' in panel:
            limits=np.asarray(panel[coordinate+'lim'],float)
            if limits.shape!=(2,) or not np.isfinite(limits).all() or limits[0]>=limits[1]:
                raise ValueError('Explicit physical figure limits must be finite and increasing')
            setter=getattr(axis,'set_'+coordinate+'lim',None)
            if setter is None:raise ValueError('Figure limit requires a matching coordinate axis')
            setter(*limits)
    if not drawn:
        label=axis.text2D if hasattr(axis,'text2D') else axis.text
        label(.5,.5,'NO SUPPORTED MEASUREMENTS',transform=axis.transAxes,ha='center')
    elif panel.get('legend',True):axis.legend(fontsize=7)
    return counts


def export_scientific_suite(request,destination,*,base,memory_gib):
    from matplotlib import pyplot as plt
    from core.figures import export_figure
    from core.data import numpy_archive_contract
    if type(memory_gib) not in (int,float) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Scientific figures need a finite positive memory allocation')
    records = request['figures']
    names = [record['figure_id'] for record in records]
    if len(set(names)) != len(names):
        raise ValueError('Scientific figure IDs must be distinct within a suite')
    organized = request.get('organized_outputs',False)
    if type(organized) is not bool:
        raise ValueError('Organized figure output selection must be a boolean')
    base=Path(base);destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True);cache={};entries=[];covered=set()
    retained_bytes = 0
    for key,path in request['sources'].items():
        path=Path(path);path=path if path.is_absolute() else base/path
        archive = path.with_name(path.name+'.data')/'arrays.npz'
        with zipfile.ZipFile(archive) as packed:
            expanded = sum(member.file_size for member in packed.infolist())
        # Keep room for retained inputs, masks, transformed arrays and plotting.
        if 4*(retained_bytes+expanded) > memory_gib*1024**3:
            raise MemoryError('Combined figure sources exceed the declared workspace')
        cache[key]=read_product(path,memory_gib=memory_gib-retained_bytes*4/1024**3)
        retained_bytes += sum(value.nbytes for value in cache[key][1].values())
    for record in request['figures']:
        family=record['family'];variant=record['variant']
        if family not in FAMILIES or variant not in FAMILIES[family]:raise ValueError('Unknown required scientific figure family/variant')
        name=record['figure_id']
        if not name.replace('-','_').isidentifier():raise ValueError('Safe figure identifier required')
        numeric_directory=destination/'products'/'scientific_figures'/name if organized else destination
        numeric_directory.mkdir(parents=True,exist_ok=True)
        if (numeric_directory/(name+'.physical_arrays.npz')).exists() or (numeric_directory/(name+'.figure.json')).exists():
            raise FileExistsError('Scientific figure products already exist: '+name)
        panels=record['panels'];count=len(panels)
        if not 1<=count<=request['maximum_panels_per_figure']:raise ValueError('Explicit bounded panel count required')
        fig=plt.figure(figsize=(8,3.2*count),layout='constrained');physical={};units={};selections=[]
        for i,panel in enumerate(panels):
            projection=panel.get('projection') or ('3d' if any(s['kind'].endswith('3d') for s in panel['series']) else None)
            axis=fig.add_subplot(count,1,i+1,projection=projection)
            selections.append({'panel':panel['panel_id'],'counts':_panel(panel,axis,cache,physical,units),'selection':panel['selection'],'quality_status':panel['quality_status']})
        import textwrap
        fig.suptitle(textwrap.fill(record['title']+' — '+record['science_status'],width=86),fontsize=11)
        array_path=numeric_directory/(name+'.physical_arrays.npz');np.savez_compressed(array_path,**physical)
        contract=numpy_archive_contract(array_path,{key:{'units':units[key],'quantity':key,'missing_policy':'unsupported/nonfinite measurements remain explicitly masked'} for key in physical},epoch=record.get('epoch'),provenance={'sources':{source:cache[source][2] for source in request['sources']},'recipe':record})
        metadata={'family':family,'variant':variant,'recipe':record,'selections':selections,'unit_source':'explicit physical source-array recipe','source_hashes':{path:checksum for source in cache.values() for path,checksum in source[2].items()},'physical_array_contract':contract,'science_status':record['science_status'],'epoch':record.get('epoch')}
        exported=export_figure(fig,numeric_directory/name,metadata=json_safe(metadata),formats=('png','pdf','svg'),dpi=request['dpi'])
        format_paths={}
        if organized:
            from core.foundation import atomic_copy
            for source in exported:
                output=destination/'figures'/source.suffix[1:]/family/source.name
                output.parent.mkdir(parents=True,exist_ok=True)
                atomic_copy(source,output)
                format_paths[source.suffix[1:]]=str(output.relative_to(destination))
        else:
            format_paths={source.suffix[1:]:str(source.relative_to(destination)) for source in exported}
        entries.append({'figure_id':name,'catalogue_id':record.get('catalogue_id'),'family':family,'variant':variant,'science_status':record['science_status'],
            'manifest':str((numeric_directory/(name+'.figure.json')).relative_to(destination)),
            **format_paths,'physical_arrays':str(array_path.relative_to(destination)),
            'physical_arrays_sha256':digest(array_path),'selections':selections});covered.add((family,variant))
    missing=[{'family':family,'variant':variant,'status':'PENDING_MEASURED_INPUT'} for family,variants in FAMILIES.items() for variant in variants if (family,variant) not in covered]
    for observable in request.get('required_convergence_observables',[]):
        for variant in FAMILIES['convergence']:
            if not any(row['family']=='convergence' and row['variant']==variant and observable in next(q for q in request['figures'] if q['figure_id']==row['figure_id']).get('observables',[]) for row in entries):
                missing.append({'family':'convergence','variant':variant,'observable':observable,'status':'PENDING_MEASURED_INPUT'})
    receipt={'schema':'scientific-figure-suite-v1','figures':entries,'pending_measurements':missing,'science_acceptance':'NOT_ACCEPTED','code_sha256':digest(__file__)}
    atomic_json(destination/'gallery.json',json_safe(receipt));write_manifest_gallery(destination,receipt)
    return receipt


def write_manifest_gallery(destination,record):
    destination=Path(destination);cards=[]
    for row in record['figures']:
        text=html.escape(row['figure_id']);status=html.escape(row['science_status'])
        search=' '.join(str(row.get(key,'')) for key in ('figure_id','catalogue_id','family','variant','science_status'))
        links=' · '.join('<a href="'+html.escape(row[key],quote=True)+'">'+key.upper()+'</a>' for key in ('pdf','png','svg') if row.get(key))
        cards.append('<article data-search="'+html.escape(search.casefold(),quote=True)+'"><h2>'+text+'</h2><p>'+status+'</p>'+links+' · <a href="'+html.escape(row['manifest'],quote=True)+'">manifest</a> · <a href="'+html.escape(row['physical_arrays'],quote=True)+'">physical arrays</a><img loading="lazy" src="'+html.escape(row['png'],quote=True)+'" alt="'+text+'"></article>')
    pending=''.join('<li>'+html.escape(q['family']+'/'+q['variant'])+' — pending measured inputs</li>' for q in record['pending_measurements'])
    atomic_text(destination/'index.html','<!doctype html><html lang="en"><meta charset="utf-8"><title>Scientific figure gallery</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;background:#16181e;color:#eee}a{color:#9fd6ff}img{max-width:100%}article{padding:1rem;border-bottom:1px solid #555}input{font:inherit;padding:.6rem;width:90%}</style><h1>Scientific figures</h1><label>Filter by catalogue ID, family or status <input id="filter" type="search"></label>'+''.join(cards)+'<h2>Awaiting measurements</h2><ul>'+pending+'</ul><script>document.getElementById("filter").addEventListener("input",function(){const terms=this.value.toLowerCase().split(/\\s+/).filter(Boolean);document.querySelectorAll("article[data-search]").forEach(card=>{card.hidden=!terms.every(term=>card.dataset.search.includes(term));});});</script></html>')


def scientific_figures_main():
    import argparse
    parser=argparse.ArgumentParser(description='Render every registered scientific figure family from sealed measured arrays.')
    parser.add_argument('request',nargs='?');parser.add_argument('--output');parser.add_argument('--memory-gib',type=float)
    parser.add_argument('--export-registry',type=Path,help='Write the current variant catalogue without rendering or claiming measurement coverage.')
    parser.add_argument('--organized',action='store_true',help='Separate PDF/PNG/SVG by family and retain numeric products under products/scientific_figures.')
    args=parser.parse_args()
    if args.export_registry is not None:
        if args.request is not None or args.output is not None or args.memory_gib is not None or args.organized:
            parser.error('--export-registry must be used alone')
        atomic_json(args.export_registry,scientific_figure_registry())
        return 0
    if args.request is None or args.output is None or args.memory_gib is None:
        parser.error('Rendering requires request, --output and --memory-gib')
    path=Path(args.request).resolve()
    request=json.loads(path.read_text())
    if args.organized:request['organized_outputs']=True
    export_scientific_suite(request,args.output,base=path.parent,memory_gib=args.memory_gib)
    return 0
