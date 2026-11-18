"""Standard-library Blender runtime embedded in .blend; measured epochs only."""
from __future__ import annotations
import json
import math
import bisect
import uuid

_EPOCH_CACHE = {}
from pathlib import Path

COLORS = {'halo': (0.1, 0.4, 1., 1.), 'subhalo': (0.3, 0.7, 1., 1.),
    'galaxy': (1., 0.7, 0.3, 1.), 'black_hole': (0.6, 0.1, 0.8, 1.),
    'stream': (0.8, 0.8, 0.3, 1.), 'shape': (0.3, 1., 0.6, 1.),
    'tide': (1., 0.3, 0.3, 1.), 'event': (1., 0.2, 0.6, 1.), 'sidm': (0.2, 1., 1., 1.)}


def object_key(record):
    return record['namespace']+':'+str(record['object_id'])


def _collection(bpy, kind):
    scene = bpy.context.scene
    root = bpy.data.collections[scene['research_display_root_collection']]
    for collection in root.children:
        if collection.get('research_display_kind') == kind:
            return collection
    collection = bpy.data.collections.new('Measured '+kind)
    collection['research_display_kind'] = kind
    root.children.link(collection)
    return collection


def _material(bpy, kind):
    name = 'Measured_'+kind
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = COLORS[kind]
    return material


def _mesh(bpy, name, record):
    position = record.get('position', [0., 0., 0.])
    if len(position) != 3 or not all(math.isfinite(q) for q in position):
        raise ValueError('Measured display position must be finite xyz')
    raw_radius = record.get('display_radius',0.01)
    if isinstance(raw_radius,bool):
        raise ValueError('Display marker radius must be numeric, not a boolean')
    radius = float(raw_radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Positive display marker radius required')
    supported = record.get('geometry_supported', record.get('supported', True))
    if type(supported) is not bool:
        raise ValueError('Measured geometry support must be an exact boolean')
    display_kind = record['kind'] if supported else 'unsupported_marker'
    if display_kind == 'tide':
        eigenvalues, eigenvectors = record['eigenvalues'], record['eigenvectors']
        if len(eigenvalues) != 3 or len(eigenvectors) != 3 or not all(math.isfinite(q) for q in eigenvalues):
            raise ValueError('Measured tidal eigenvalues/eigenvectors required')
        scale = record['glyph_scale_per_eigenvalue']
        if isinstance(scale,bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Explicit positive tidal glyph display scale required')
        vertices = [(0., 0., 0.)]
        for value, direction in zip(eigenvalues, eigenvectors):
            if len(direction) != 3 or not all(math.isfinite(q) for q in direction):
                raise ValueError('Finite measured tidal eigenvector required')
            vertices.append(tuple(scale*value*q for q in direction))
        edges = [(0,1), (0,2), (0,3)]
    elif display_kind == 'stream':
        points = record['points']
        if len(points) < 2 or any(len(p) != 3 or not all(math.isfinite(q) for q in p) for p in points):
            raise ValueError('Stream needs measured finite polyline points')
        vertices = [tuple(p[i]-position[i] for i in range(3)) for p in points]
        edges = [(i, i+1) for i in range(len(points)-1)]
    else:
        vertices = [(radius,0,0),(-radius,0,0),(0,radius,0),(0,-radius,0),(0,0,radius),(0,0,-radius)]
        edges = [(0,2),(2,1),(1,3),(3,0),(0,4),(1,4),(2,4),(3,4),(0,5),(1,5),(2,5),(3,5)]
    if display_kind in {'stream','tide','shape'}:
        curves = []
        if display_kind == 'shape':
            axes, matrix = record['semiaxes'], record['orientation']
            if len(axes) != 3 or any(q <= 0 or not math.isfinite(q) for q in axes):
                raise ValueError('Positive measured shape semiaxes required')
            from mathutils import Matrix
            orientation = Matrix(matrix)
            gram = orientation.transposed()@orientation
            if abs(orientation.determinant()-1.) > 1e-5 or any(abs(gram[i][j]-(1. if i == j else 0.)) > 1e-5 for i in range(3) for j in range(3)):
                raise ValueError('Measured orientation must be a right-handed orthonormal basis')
            for first,second in ((0,1),(0,2),(1,2)):
                hoop = []
                for j in range(65):
                    angle = 2*math.pi*j/64
                    point = [0.,0.,0.]
                    point[first] = axes[first]*math.cos(angle)
                    point[second] = axes[second]*math.sin(angle)
                    hoop.append(tuple(sum(matrix[i][k]*point[k] for k in range(3)) for i in range(3)))
                curves.append(hoop)
        elif display_kind == 'stream':
            curves.append(vertices)
        else:
            curves.extend([vertices[first],vertices[second]] for first,second in edges)
        width = record.get('display_line_width',radius*.1)
        if isinstance(width,bool) or not math.isfinite(width) or width <= 0:
            raise ValueError('Positive finite display curve width required')
        geometry = bpy.data.curves.new(name,'CURVE')
        geometry.dimensions = '3D'
        geometry.bevel_depth = width
        geometry.bevel_resolution = 2
        geometry.resolution_u = 1
        for points in curves:
            spline = geometry.splines.new('POLY')
            spline.points.add(len(points)-1)
            for point,coordinate in zip(spline.points,points):
                point.co = (*coordinate,1.)
        obj = bpy.data.objects.new(name,geometry)
        obj['research_curve_width_semantics'] = 'display thickness; no physical stream width inferred'
    else:
        geometry = bpy.data.meshes.new(name)
        # Solid octahedron makes symbolic markers visible in actual renders.
        faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),
                 (2,0,5),(1,2,5),(3,1,5),(0,3,5)]
        geometry.from_pydata(vertices,[],faces)
        geometry.update()
        obj = bpy.data.objects.new(name,geometry)
    _collection(bpy,record['kind']).objects.link(obj)
    obj.location = position
    geometry.materials.append(_material(bpy,record['kind']))
    obj['research_geometry_supported'] = supported
    obj['research_geometry_representation'] = display_kind
    obj['research_display_owner'] = bpy.context.scene['research_display_owner']
    obj['research_identity'] = object_key(record)
    obj['research_kind'] = record['kind']
    obj['research_measurement_json'] = json.dumps(record, sort_keys=True)
    obj['research_representation'] = 'saved measured state; display geometry is symbolic unless measured points/axes are supplied'
    return obj


def apply_epoch(bpy, scene):
    serialized = scene['research_display_manifest']
    key = scene.as_pointer()
    cache = _EPOCH_CACHE.get(key)
    if cache is None or cache['serialized'] != serialized:
        manifest = json.loads(serialized)
        epochs = manifest['timeline']
        cache = {'serialized':serialized,'epochs':epochs,'frames':[q['frame'] for q in epochs]}
        _EPOCH_CACHE[key] = cache
    epochs = cache['epochs']
    # Floor to the most recent saved epoch; no lifecycle interpolation.
    index = max(0,bisect.bisect_right(cache['frames'],scene.frame_current)-1)
    selected = epochs[index]
    owner = scene.get('research_display_owner')
    for obj in scene.objects:
        if 'research_epoch_index' not in obj or obj.get('research_display_owner') != owner:
            continue
        visible = int(obj['research_epoch_index']) == index and epochs[0]['frame'] <= scene.frame_current <= epochs[-1]['frame']
        obj.hide_viewport = not visible
        obj.hide_render = not visible
    scene['research_current_epoch_index'] = index
    if scene.get('research_follow_exact_identity',True) and bpy.context.scene == scene:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.get('research_display_owner') == owner:
            identity = active.get('research_identity')
            layer_role = active.get('research_geometry_layer','primary')
            target = next((q for q in scene.objects if q.get('research_display_owner') == owner and
                q.get('research_identity') == identity and q.get('research_epoch_index') == index and
                q.get('research_geometry_layer','primary') == layer_role),None)
            if target is not None and target != active:
                active.select_set(False)
                target.select_set(True)
                bpy.context.view_layer.objects.active = target
    scene['research_current_epoch'] = json.dumps(selected['epoch_contract'])
    scene['research_current_snapshot'] = str(selected['snapshot'])
    scene['research_intermediate_frames'] = 'hold last measured state; no physical substeps inferred'


def frame_change(scene, depsgraph=None):
    import bpy
    if 'research_display_manifest' in scene:
        apply_epoch(bpy, scene)


def load_post(dummy):
    import bpy
    _EPOCH_CACHE.clear()
    for scene in bpy.data.scenes:
        if 'research_display_manifest' in scene:
            apply_epoch(bpy, scene)
    register_handlers(bpy)


def register_handlers(bpy):
    from bpy.app.handlers import persistent
    for group, callback in ((bpy.app.handlers.frame_change_post, frame_change), (bpy.app.handlers.load_post, load_post)):
        for old in list(group):
            if getattr(old, '_research_display_handler_tag', None) == callback.__name__ or (getattr(old, '__name__', '') == callback.__name__ and getattr(old, '__module__', '') in {__name__, 'research_display_runtime'}):
                group.remove(old)
        callback._research_display_handler_tag = callback.__name__
        group.append(persistent(callback))


def install_runtime(bpy, manifest, base, *, lifecycle=None, preserve_scene_clock=False):
    scene = bpy.context.scene
    original_clock=(scene.frame_start,scene.frame_end,scene.frame_current)
    if preserve_scene_clock and (manifest['timeline'][0]['frame']<scene.frame_start or manifest['timeline'][-1]['frame']>scene.frame_end):
        raise ValueError('Measured display epochs exceed the combined scene clock')
    if 'research_display_manifest' in scene:
        raise ValueError('Scene already contains a measured display generation')
    maximum = manifest['maximum_objects']
    total = sum(len(q['objects']) for q in manifest['timeline'])
    if type(maximum) is not int or maximum < 1 or total > maximum:
        raise MemoryError('Measured scene object census exceeds declared display bound')
    geometry_objects = sum(1+len(record.get('measured_geometry_layers',[]))
        for epoch in manifest['timeline'] for record in epoch['objects'])
    geometry_object_limit = manifest.get('maximum_geometry_objects',maximum)
    if type(geometry_object_limit) is not int or geometry_object_limit < 1 or geometry_objects > geometry_object_limit:
        raise MemoryError('Measured geometry objects exceed the declared scene bound')
    vertex_limit = manifest.get('maximum_geometry_vertices')
    if vertex_limit is not None:
        if type(vertex_limit) is not int or vertex_limit < 1:
            raise ValueError('Declared geometry vertex budget must be a positive integer')
        estimated = 0
        for epoch in manifest['timeline']:
            for record in epoch['objects']:
                for layer in [record,*record.get('measured_geometry_layers',[])]:
                    supported = layer.get('geometry_supported',layer.get('supported',True))
                    if type(supported) is not bool:
                        raise ValueError('Measured geometry support must be an exact boolean')
                    kind = layer['kind'] if supported else 'unsupported_marker'
                    estimated += len(layer['points']) if kind == 'stream' else 195 if kind == 'shape' else 6
        if estimated > vertex_limit:
            raise MemoryError('Measured scene geometry exceeds its declared vertex budget')
    root = bpy.data.collections.new('Measured Research '+scene.name)
    scene.collection.children.link(root)
    scene['research_display_root_collection'] = root.name
    scene['research_display_owner'] = uuid.uuid4().hex
    scene['research_follow_exact_identity'] = True
    scene['research_display_manifest'] = json.dumps(manifest, sort_keys=True)
    scene['research_lifecycle_index'] = json.dumps(lifecycle or {},sort_keys=True)
    scene['research_display_base'] = str(Path(base).resolve())
    scene['research_coordinate_units'] = manifest['coordinate_units']
    for index, epoch in enumerate(manifest['timeline']):
        for record in epoch['objects']:
            obj = _mesh(bpy, 'Measured '+object_key(record)+' epoch '+str(index), record)
            obj['research_geometry_layer'] = 'primary'
            obj['research_epoch_index'] = index
            obj['research_epoch_contract'] = json.dumps(epoch['epoch_contract'])
            obj['research_source_links'] = json.dumps(record['source_links'])
            for layer in record.get('measured_geometry_layers',[]):
                child = _mesh(bpy,'Measured '+object_key(record)+' '+layer['kind']+' epoch '+str(index),layer)
                # Local geometry is already centred on the same measured point.
                # The primary object has translation only; no world-frame fitting.
                child.parent = obj
                child.location = (0.,0.,0.)
                child['research_geometry_layer'] = 'layer:'+layer['kind']
                child['research_epoch_index'] = index
                child['research_epoch_contract'] = json.dumps(epoch['epoch_contract'])
                child['research_source_links'] = json.dumps(record['source_links'])
                child['research_measurement_json'] = json.dumps(record,sort_keys=True)
        marker = scene.timeline_markers.new('Snapshot '+str(epoch['snapshot']), frame=epoch['frame'])
    if not preserve_scene_clock:
        scene.frame_start = manifest['timeline'][0]['frame']
        scene.frame_end = manifest['timeline'][-1]['frame']
    source = Path(__file__).read_text()
    text = bpy.data.texts.get('research_display_runtime.py') or bpy.data.texts.new('research_display_runtime.py')
    text.clear()
    text.write(source+'\n\nregister_handlers(__import__("bpy"))\n')
    text.use_module = True
    register_handlers(bpy)
    scene.frame_set(original_clock[2] if preserve_scene_clock else scene.frame_start)
    apply_epoch(bpy, scene)
