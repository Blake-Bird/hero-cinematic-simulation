"""Blender: consolidated physical calculations and scene interfaces."""
from __future__ import annotations

import sys as _entry_sys
if __name__ == '__main__':
    _entry_sys.modules.setdefault('blender', _entry_sys.modules[__name__])

# ========================================================================
# blender

# ========================================================================
# blender_scene: Cinematic sampled-particle scene, executed by Blender's bundled Python.

# Workflow and units
# Runs inside Blender; builds scenes from particle PLYs and JSON export manifests.
# One unit is comoving Mpc/h for cosmological scenes, or proper kpc for stellar-host scenes. Glyph radii/emission strengths are display parameters, not halo boundaries, particle softening or luminosities.
# Measured gas log10(T/K) and stellar age attributes select explicit materials. All periodic copies are cropped to [0,L)^3. Blender is the renderer; CLASS/GADGET/halo finders supply the science.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import sys
import os
import json as blender_json
import math
from pathlib import Path
runtime_root = Path(__file__).resolve().parent
if not (runtime_root / 'core/__init__.py').is_file():
    runtime_root = runtime_root.parent
sys.path.insert(0, str(runtime_root))
from core.foundation import file_sha256, verify_numpy_container
from core.resources import cgroup_available_memory


def disable_cycles_denoising(scene):
    """Keep rendering compatible with builds lacking OpenImageDenoise."""
    if hasattr(scene.cycles, 'use_denoising'):
        scene.cycles.use_denoising = False
    for layer in scene.view_layers:
        settings = getattr(layer, 'cycles', None)
        if settings is not None and hasattr(settings, 'use_denoising'):
            settings.use_denoising = False
    # A saved compositor can request the same unavailable backend independently
    # of Cycles' scene and view-layer switches. Mute those nodes in memory;
    # the original .blend stays unchanged when rendering a saved scene.
    pending = [getattr(scene, 'node_tree', None)]
    seen = set()
    while pending:
        tree = pending.pop()
        if tree is None or id(tree) in seen:
            continue
        seen.add(id(tree))
        for node in tree.nodes:
            if getattr(node, 'bl_idname', None) == 'CompositorNodeDenoise':
                node.mute = True
            nested = getattr(node, 'node_tree', None)
            if nested is not None:
                pending.append(nested)


def set_output_format(scene, file_format):
    """Blender 5 separates media type from the dynamically available formats."""
    settings = scene.render.image_settings
    if hasattr(settings, 'media_type'):
        settings.media_type = ('VIDEO' if file_format == 'FFMPEG' else
                               'MULTI_LAYER_IMAGE' if file_format == 'OPEN_EXR_MULTILAYER' else 'IMAGE')
    settings.file_format = file_format


def configure_glare(glare):
    """Use Blender 5 menu sockets, retaining older enum-property support."""
    kind = glare.inputs.get('Type')
    quality = glare.inputs.get('Quality')
    if kind is not None:
        kind.default_value = 'Fog Glow'
    else:
        glare.glare_type = 'FOG_GLOW'
    if quality is not None:
        quality.default_value = 'High'
    else:
        glare.quality = 'HIGH'


def action_fcurves(animation):
    """Curves for this datablock's action slot, across legacy and layered APIs."""
    action = getattr(animation, 'action', None)
    if action is None:
        return ()
    slot = getattr(animation, 'action_slot', None)
    if getattr(action, 'is_action_layered', False):
        curves = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', ()):
                    if slot is None or bag.slot_handle == slot.handle:
                        curves.extend(bag.fcurves)
        return curves
    return getattr(action, 'fcurves', ())


def read_particle_mesh(bpy, path):
    """Read our exact float PLY schema without importer reordering or attribute loss."""
    import numpy as np
    path = Path(path)
    count = verify_particle_ply_payload(path)
    names = []
    with path.open('rb') as stream:
        while True:
            line = stream.readline().strip()
            if line.startswith(b'property '):
                names.append(line.split()[2].decode('ascii'))
            if line == b'end_header':
                break
        if names[:3] != ['x', 'y', 'z'] or len(set(names)) != len(names):
            raise ValueError('Particle PLY requires unique properties beginning with x,y,z')
        values = np.fromfile(stream, dtype='<f4').reshape(count, len(names))
    if not np.isfinite(values).all():
        raise ValueError('Particle PLY contains nonfinite coordinates or attributes')
    mesh = bpy.data.meshes.new(path.stem)
    mesh.vertices.add(count)
    mesh.vertices.foreach_set('co', np.ascontiguousarray(values[:, :3]).ravel())
    for index, name in enumerate(names[3:], 3):
        attribute = mesh.attributes.new(name=name, type='FLOAT', domain='POINT')
        attribute.data.foreach_set('value', np.ascontiguousarray(values[:, index]))
    mesh.update()
    obj = bpy.data.objects.new(path.stem, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _manifest_asset_path(manifest_path, asset_path):
    """Resolve an asset stored as either a portable relative path or legacy absolute path."""
    path = Path(asset_path).expanduser()
    if not path.is_absolute():
        path = Path(manifest_path).expanduser().resolve().parent / path
    return path.resolve()


def verify_timeline_assets(manifest_path):
    """Verify sealed portable exports before Blender reads any geometry."""
    manifest_path=Path(manifest_path).resolve()
    manifest=blender_json.loads(manifest_path.read_text())
    if 'data_contract' not in manifest:
        return manifest  # Historical unsealed exports retain their legacy reader.
    receipt=manifest_path.with_name(manifest_path.name+'.sha256')
    if not receipt.is_file() or receipt.read_text().strip() != file_sha256(manifest_path):
        raise ValueError('Blender timeline manifest checksum mismatch')
    if manifest['data_contract'].get('schema') != 'darkmattersim-data-contract-v1':
        raise ValueError('Unsupported Blender data contract')
    seen=set()
    for item in manifest['assets']:
        path=_manifest_asset_path(manifest_path,item['path'])
        if not path.is_relative_to(manifest_path.parent) or path in seen:
            raise ValueError('Invalid or duplicate Blender asset path')
        seen.add(path)
        if not path.is_file() or path.stat().st_size != item['size_bytes'] or file_sha256(path) != item['sha256']:
            raise ValueError('Blender timeline asset missing or changed: '+item['path'])
        if path.suffix == '.ply':
            verify_particle_ply_payload(path)
    for row in manifest['snapshots']:
        epoch=row['epoch_contract']
        if row['scale_factor'] != epoch['scale_factor'] or row['redshift'] != epoch['redshift']:
            raise ValueError('Blender epoch metadata disagrees with canonical epoch')
    for interval in manifest.get('intervals',[]):
        if not interval.get('identity_contract'):continue  # Explicit legacy numeric sidecar support.
        contract=interval['identity_contract']
        verify_numpy_container(_manifest_asset_path(manifest_path,interval['identity_asset']),contract)
        for record in contract['datasets'].values():
            if record.get('dtype')!='uint64' or record.get('identity_namespace')!=manifest['identity_namespace']:
                raise ValueError('Blender interval identity contract disagrees with timeline namespace/dtype')
    return manifest


def verify_particle_ply_payload(path):
    """Require the complete float-only binary point payload used by our exporter."""
    path=Path(path)
    count=None;properties=0;binary=False
    with path.open('rb') as source:
        if source.readline().strip() != b'ply':
            raise ValueError('Invalid Blender PLY magic')
        for _ in range(256):
            line=source.readline().strip()
            if line == b'format binary_little_endian 1.0':
                binary=True
            elif line.startswith(b'element vertex '):
                count=int(line.split()[2])
            elif line.startswith(b'property '):
                if len(line.split()) != 3 or line.split()[1] != b'float':
                    raise ValueError('Unsupported Blender PLY property')
                properties+=1
            elif line == b'end_header':
                break
            elif line.startswith(b'element '):
                raise ValueError('Unsupported Blender PLY element')
            elif not line:
                raise ValueError('Incomplete Blender PLY header')
        else:
            raise ValueError('Blender PLY header exceeds limit')
        offset=source.tell()
    if not binary or count is None or count < 0 or properties < 3 or path.stat().st_size != offset+count*properties*4:
        raise ValueError('Incomplete or invalid Blender PLY payload: '+str(path))
    return count


def verify_single_scene_assets(manifest_path):
    """Verify a sealed portable single-snapshot bundle before constructing a scene."""
    manifest_path=Path(manifest_path).resolve()
    receipt=manifest_path.with_name(manifest_path.name+'.sha256')
    if not receipt.is_file() or receipt.read_text().strip()!=file_sha256(manifest_path):
        raise ValueError('Blender single-scene manifest checksum mismatch')
    manifest=blender_json.loads(manifest_path.read_text())
    if manifest.get('schema')!='blender-single-publication-v1':
        raise ValueError('Unsupported Blender single-scene contract')
    seen=set()
    for item in manifest['assets']:
        path=_manifest_asset_path(manifest_path,item['path'])
        if not path.is_relative_to(manifest_path.parent) or path in seen:
            raise ValueError('Invalid or duplicate Blender single-scene asset path')
        seen.add(path)
        if not path.is_file() or path.stat().st_size!=item['size_bytes'] or file_sha256(path)!=item['sha256']:
            raise ValueError('Blender single-scene asset missing or changed: '+item['path'])
        if path.suffix=='.ply':verify_particle_ply_payload(path)
    return manifest


def _ply_vertex_count(ply_path):
    """Read the point count from a binary PLY header without loading its payload."""
    with open(ply_path, "rb") as source:
        for _ in range(256):
            line = source.readline()
            if not line:
                break
            if line.startswith(b"element vertex "):
                try:
                    return int(line.split()[2])
                except (IndexError, ValueError) as exc:
                    raise ValueError("PLY has an invalid vertex count: " + str(ply_path)) from exc
            if line.strip() == b"end_header":
                break
    raise ValueError("PLY header has no vertex count: " + str(ply_path))


def _available_memory_bytes():
    """Return available host RAM, respecting common Linux cgroup limits."""
    available = None
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                available = int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError):
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess
            import re
            # vm_stat exposes file-backed pages separately from anonymous pages.
            # A lower bound on inactive file cache subtracts *all* active, wired
            # and speculative pages, rather than treating all inactive RAM as free.
            # Kernel free_count already contains speculative pages.
            output = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=3)
            page = re.search(r'page size of (\d+) bytes', output)
            counters = {name: int(value) for name,value in
                        re.findall(r'^([^:\n]+):\s+(\d+)\.', output, re.MULTILINE)}
            if page and all(k in counters for k in ('Pages free','Pages active','Pages inactive','Pages wired down','File-backed pages')):
                free = counters['Pages free'] + counters.get('Pages speculative',0)
                inactive_file = min(counters['Pages inactive'], max(0,
                    counters['File-backed pages']-counters['Pages active']-
                    counters['Pages wired down']-counters.get('Pages speculative',0)-free))
                available = (free+inactive_file)*int(page.group(1))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    else:
        try:
            with open("/proc/meminfo", "r", encoding="ascii") as source:
                for line in source:
                    if line.startswith("MemAvailable:"):
                        available = int(float(line.split()[1]) * 1024)
                        break
        except (OSError, ValueError, IndexError):
            pass
        if available is None:
            try:
                available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (AttributeError, OSError, ValueError):
                pass
    if os.name == "posix":
        for limit_path, current_path, stat_path, key in (
            ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.stat', 'inactive_file'),
            ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes',
             '/sys/fs/cgroup/memory/memory.stat', 'total_inactive_file'),
        ):
            headroom = cgroup_available_memory(limit_path, current_path, stat_path, inactive_key=key)
            if headroom is not None:
                available = headroom if available is None else min(available, headroom)
                break
    return available if available is not None and available > 0 else None


def _validate_particle_memory(particle_count, max_memory_fraction):
    """Stop a full-resolution Blender import before it can exhaust host memory."""
    count = int(particle_count)
    fraction = float(max_memory_fraction)
    if count < 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("Full-resolution Blender particle count or memory fraction is invalid")
    coordinate_bytes = count * 3 * 4
    estimate = coordinate_bytes * 17 + 1024 * 1024**2
    available = _available_memory_bytes()
    allowed = None if available is None else int(available * fraction)
    if allowed is None or estimate > allowed:
        raise MemoryError(
            "Full-resolution Blender import is estimated to need %d bytes; "
            "the configured budget is %s bytes from %s available. "
            "Use sampled/adaptive export or raise blender.max_memory_fraction."
            % (estimate, "unknown" if allowed is None else allowed,
               "unknown" if available is None else available)
        )


def _socket(node, name):
    socket = node.inputs.get(name)
    if socket is None:
        raise RuntimeError("Blender node %s is missing its %s input" % (node.name, name))
    return socket


def _blender_emission_material(bpy, name, color, strength=2.5):
    """Create an art-directed emission material for data-colored particles."""
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = float(strength)
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _binned_point_instances(bpy, nodes, links, points, attribute_name, bins,
                            box_size, radius, material_prefix, emission_strength=2.5):
    """Assign explicit per-particle materials from a measured scalar field.

    Geometry nodes filter the imported point attribute into physical ranges,
    then instance one fixed-color material per range. This avoids relying on
    shader interpolation of PLY attributes across Blender instances.
    """
    field = nodes.new("GeometryNodeInputNamedAttribute")
    field.data_type = "FLOAT"
    field.inputs["Name"].default_value = attribute_name
    join = nodes.new("GeometryNodeJoinGeometry")
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = max(float(box_size) * float(radius), 1.0e-5)
    subdivisions = ico.inputs.get(" subdivisions") or ico.inputs.get("Subdivisions")
    if subdivisions is not None:
        subdivisions.default_value = 1
    for low, high, label, color in bins:
        tests = []
        if low is not None:
            compare = nodes.new("FunctionNodeCompare")
            compare.data_type = "FLOAT"
            compare.operation = "GREATER_EQUAL"
            compare.inputs["B"].default_value = float(low)
            links.new(field.outputs["Attribute"], compare.inputs["A"])
            tests.append(compare.outputs["Result"])
        if high is not None:
            compare = nodes.new("FunctionNodeCompare")
            compare.data_type = "FLOAT"
            compare.operation = "LESS_THAN"
            compare.inputs["B"].default_value = float(high)
            links.new(field.outputs["Attribute"], compare.inputs["A"])
            tests.append(compare.outputs["Result"])
        if not tests:
            selected = None
        elif len(tests) == 1:
            selected = tests[0]
        else:
            conjunction = nodes.new("FunctionNodeBooleanMath")
            conjunction.operation = "AND"
            links.new(tests[0], conjunction.inputs[0])
            links.new(tests[1], conjunction.inputs[1])
            selected = conjunction.outputs["Boolean"]
        separate = nodes.new("GeometryNodeSeparateGeometry")
        separate.domain = "POINT"
        links.new(points, separate.inputs["Geometry"])
        if selected is not None:
            links.new(selected, separate.inputs["Selection"])
        set_material = nodes.new("GeometryNodeSetMaterial")
        set_material.inputs["Material"].default_value = _blender_emission_material(
            bpy, "DMSim %s %s" % (material_prefix, label), color, strength=emission_strength,
        )
        links.new(ico.outputs["Mesh"], set_material.inputs["Geometry"])
        instance = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(separate.outputs["Selection"], instance.inputs["Points"])
        links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
        links.new(instance.outputs["Instances"], join.inputs["Geometry"])
    return join.outputs["Geometry"]


def _make_geometry_nodes(obj, species, box_size, *, crop_periodic=True):
    """Create glyphs/material bins; optionally select the periodic fundamental box.

    box_size is the display scale in the caller's scene units. Cosmological
    exports use Mpc/h; isolated mergers use kpc and must disable periodic crops.
    """
    import bpy

    group = bpy.data.node_groups.new("DarkMatterSim cinematic geometry", "GeometryNodeTree")
    if hasattr(group, "interface"):
        group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:
        group.inputs.new("NodeSocketGeometry", "Geometry")
        group.outputs.new("NodeSocketGeometry", "Geometry")
    nodes, links = group.nodes, group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    if crop_periodic:
        # Periodic trajectory exports carry translated copies at crossed faces.
        # Select exactly one copy in the fundamental [0,L)^3 box at every frame;
        # otherwise off-box copies would appear as duplicate physical particles.
        position = nodes.new("GeometryNodeInputPosition")
        xyz = nodes.new("ShaderNodeSeparateXYZ")
        links.new(position.outputs["Position"], xyz.inputs["Vector"])
        inside = None
        for axis in ("X", "Y", "Z"):
            for operation, bound in (("GREATER_EQUAL", 0.), ("LESS_THAN", float(box_size))):
                compare = nodes.new("FunctionNodeCompare")
                compare.data_type = "FLOAT"
                compare.operation = operation
                compare.inputs["B"].default_value = bound
                links.new(xyz.outputs[axis], compare.inputs["A"])
                result = compare.outputs["Result"]
                if inside is None:
                    inside = result
                else:
                    both = nodes.new("FunctionNodeBooleanMath")
                    both.operation = "AND"
                    links.new(inside, both.inputs[0])
                    links.new(result, both.inputs[1])
                    inside = both.outputs["Boolean"]
        crop = nodes.new("GeometryNodeSeparateGeometry")
        crop.domain = "POINT"
        links.new(input_node.outputs["Geometry"], crop.inputs["Geometry"])
        links.new(inside, crop.inputs["Selection"])
        links.new(crop.outputs["Selection"], to_points.inputs["Mesh"])
    else:
        links.new(input_node.outputs["Geometry"], to_points.inputs["Mesh"])
    if species == "gas":
        # Keep the measured SPH positions as particles in the hero scene.
        # A whole-box points-to-volume conversion overlaps thousands of
        # smoothing kernels and saturates the image, hiding the temperature
        # field. The exported smoothing radii and physical densities remain on
        # the mesh for inspection; the size here is only a display glyph.
        if obj.data.attributes.get("temperature_log10") is not None:
            thermal = _binned_point_instances(
                bpy, nodes, links, to_points.outputs["Points"], "temperature_log10",
                [(None, -1.e20, "unknown_or_effective_eos", (.35, .35, .38, 1)),
                 (-1.e20, 4.25, "cold_magenta", (0.62, 0.025, 0.88, 1)),
                 (4.25, 5.25, "cool_blue", (0.12, 0.22, 1.0, 1)),
                 (5.25, 6.25, "warm_cyan", (0.02, 0.95, 1.0, 1)),
                 (6.25, 7.0, "temperature_bin_6_25_7_0", (0.25, 1.0, 0.35, 1)),
                 (7.0, 7.7, "temperature_bin_7_0_7_7", (1.0, 0.78, 0.12, 1)),
                 (7.7, None, "hot_white", (1.0, 1.0, 0.92, 1))],
                box_size, 0.00065, "gas temperature %s" % obj.name,
            )
            links.new(thermal, output_node.inputs["Geometry"])
        else:
            ico = nodes.new("GeometryNodeMeshIcoSphere")
            ico.inputs["Radius"].default_value = max(box_size * 0.00065, 1.0e-5)
            instance = nodes.new("GeometryNodeInstanceOnPoints")
            links.new(to_points.outputs["Points"], instance.inputs["Points"])
            set_material = nodes.new("GeometryNodeSetMaterial")
            set_material.inputs["Material"].default_value = obj.data.materials[0]
            links.new(ico.outputs["Mesh"], set_material.inputs["Geometry"])
            links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
            links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])
    elif species == "black_holes" and obj.data.attributes.get("black_hole_mass_weight") is not None:
        markers = _binned_point_instances(
            bpy, nodes, links, to_points.outputs["Points"], "black_hole_mass_weight",
            [(None, 0., "unknown_particle_mass", (.35, .35, .38, 1.)),
             (0., .5, "lower_relative_particle_mass", (1., .20, .05, 1.)),
             (.5, None, "higher_relative_mass", (1., .75, .25, 1.))],
            box_size, .0008, "Newtonian BH display %s" % obj.name,
        )
        links.new(markers, output_node.inputs["Geometry"])
    elif species == "stars" and obj.data.attributes.get("stellar_candidate_color_bin") is not None:
        palette = blender_json.loads(obj["stellar_candidate_palette_linear_rgba"])
        bins = [(i-.5, i+.5, "category_%02d" % i, color) for i, color in enumerate(palette)]
        instances = _binned_point_instances(
            bpy, nodes, links, to_points.outputs["Points"], "stellar_candidate_color_bin",
            bins, box_size, obj.get("stellar_candidate_glyph_radius_fraction", .0018),
            "stellar candidate %s" % obj.name, emission_strength=1.,
        )
        links.new(instances, output_node.inputs["Geometry"])
    elif species == "stars" and obj.data.attributes.get("stellar_age_gyr") is not None:
        star_instances = _binned_point_instances(
            bpy, nodes, links, to_points.outputs["Points"], "stellar_age_gyr",
            [(None, 0., "age_unknown_grey", (.35, .35, .38, 1)),
             (0., 1., "age_under_1_gyr_cyan", (.1, .82, 1., 1)),
             (1., 5., "age_1_to_5_gyr_white", (1., .93, .75, 1)),
             (5., None, "age_over_5_gyr_gold", (1., .48, .12, 1))],
            box_size, 0.00055, "stellar age %s" % obj.name,
        )
        links.new(star_instances, output_node.inputs["Geometry"])
    else:
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        ico.inputs["Radius"].default_value = max(
            box_size * (0.00055 if species == "stars" else 0.00035), 1.0e-4,
        )
        subdivisions = ico.inputs.get(" subdivisions") or ico.inputs.get("Subdivisions")
        if subdivisions is not None:
            subdivisions.default_value = 1
        instance = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(to_points.outputs["Points"], instance.inputs["Points"])
        particle_material = nodes.new("GeometryNodeSetMaterial")
        particle_material.inputs["Material"].default_value = obj.data.materials[0]
        links.new(ico.outputs["Mesh"], particle_material.inputs["Geometry"])
        links.new(particle_material.outputs["Geometry"], instance.inputs["Instance"])
        links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])
    modifier = obj.modifiers.new("Particle rendering", "NODES")
    modifier.node_group = group


def _make_material(species):
    """Create a visible species display shader; scalar fields select materials in geometry nodes."""
    import bpy
    colours = {"dm": (.12, .32, 1., 1.), "gas": (.05, .8, .9, 1.),
               "stars": (1., .65, .22, 1.), "black_holes": (1., .3, .05, 1.)}
    if species not in colours:
        raise ValueError("Unknown particle species: " + str(species))
    return _blender_emission_material(bpy, "DMSim " + species + " display", colours[species], strength=1.2)


def _add_lighting_and_camera(scene, center, box_size, frame_count, orbit_camera=True):
    import bpy

    world = bpy.data.worlds.new("Deep space") if bpy.data.worlds.get("Deep space") is None else bpy.data.worlds["Deep space"]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.003, 0.006, 0.018, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22
    empty = bpy.data.objects.new("Camera orbit", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = center
    camera_data = bpy.data.cameras.new("Cosmic web camera")
    camera = bpy.data.objects.new("Cosmic web camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.parent = empty
    camera.location = (box_size * 1.55, -box_size * 1.75, box_size * 1.2)
    camera.data.lens = 42
    camera.data.clip_start = max(float(box_size) * 1.0e-5, 1.0e-5)
    camera.data.clip_end = max(float(box_size) * 20.0, 1.0)
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = empty
    camera.data.dof.aperture_fstop = 8.0
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = empty
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    scene.camera = camera
    if orbit_camera:
        empty.rotation_euler[2] = 0.0
        empty.keyframe_insert(data_path="rotation_euler", index=2, frame=1)
        empty.rotation_euler[2] = 6.28318530718
        empty.keyframe_insert(data_path="rotation_euler", index=2, frame=frame_count)
        if empty.animation_data and empty.animation_data.action:
            for curve in getattr(empty.animation_data.action, "fcurves", []):
                for keyframe in curve.keyframe_points:
                    keyframe.interpolation = "LINEAR"
                curve.modifiers.new("CYCLES")
    for name, offset, power, color, size in [
        ("Key light", (1.8, -1.4, 1.8), 2600, (0.45, 0.72, 1.0), 0.9),
        ("Rim light", (-1.4, 1.6, 0.8), 1800, (0.22, 0.92, 0.88), 1.2),
    ]:
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = power
        light_data.color = color
        light_data.shape = "DISK"
        light_data.size = box_size * size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.scene.collection.objects.link(light)
        light.location = tuple(center[index] + offset[index] * box_size for index in range(3))
        target = light.constraints.new("TRACK_TO")
        target.target = empty
        target.track_axis = "TRACK_NEGATIVE_Z"
        target.up_axis = "UP_Y"
    return camera


def _import_particle_mesh(ply_path, species, box_size, index, *, crop_periodic=True,
                          candidate_palette=None, candidate_glyph_fraction=.0018):
    import bpy

    ply_path = Path(ply_path).expanduser().resolve()
    if not ply_path.is_file():
        raise FileNotFoundError("Timeline particle sample is missing: " + str(ply_path))
    obj = read_particle_mesh(bpy, ply_path)
    if obj is None or len(obj.data.vertices) == 0:
        raise RuntimeError("Blender imported an empty timeline snapshot: " + str(ply_path))
    obj.name = "DarkMatterSim_%s_output_%04d" % (species, int(index))
    obj.data.name = obj.name + "_mesh"
    if candidate_palette is not None:
        obj["stellar_candidate_palette_linear_rgba"] = blender_json.dumps(candidate_palette)
        obj["stellar_candidate_glyph_radius_fraction"] = candidate_glyph_fraction
    obj.data.materials.append(_make_material(species))
    _make_geometry_nodes(obj, species, box_size, crop_periodic=crop_periodic)
    return obj


def _key_visibility(obj, first_frame, last_frame, frame_end):
    """Hold one saved snapshot visible throughout its exact output interval."""
    for field in ("hide_render", "hide_viewport"):
        if int(first_frame) == 1:
            setattr(obj, field, False)
            obj.keyframe_insert(data_path=field, frame=1)
        else:
            setattr(obj, field, True)
            obj.keyframe_insert(data_path=field, frame=1)
            setattr(obj, field, False)
            obj.keyframe_insert(data_path=field, frame=int(first_frame))
        if int(last_frame) < int(frame_end):
            setattr(obj, field, True)
            obj.keyframe_insert(data_path=field, frame=int(last_frame) + 1)
    animation = obj.animation_data
    if animation and animation.action:
        for curve in action_fcurves(animation):
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "CONSTANT"


def _make_timeline_label(camera, snapshot, box_size, species, index,
                         first_frame, last_frame, frame_end):
    import bpy

    label = bpy.data.curves.new("Snapshot label %04d" % int(index), "FONT")
    label.body = (
        "%s  |  snapshot %d  |  z = %.4g  |  a = %.5f  |  L = %.4g Mpc/h"
        % (species.upper(), int(snapshot["snapshot"]), float(snapshot["redshift"]),
           float(snapshot["scale_factor"]), float(box_size))
    )
    label.size = 1.0
    label.extrude = 0.0
    obj = bpy.data.objects.new("Snapshot label %04d" % int(index), label)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = camera
    obj.location = (-0.40 * box_size, 0.32 * box_size, -0.95 * box_size)
    obj.scale = (0.007 * box_size,) * 3
    material = bpy.data.materials.new("Timeline label %04d" % int(index))
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.88, 0.94, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.4
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    label.materials.append(material)
    _key_visibility(obj, first_frame, last_frame, frame_end)


def _add_interval_shape_key(start_obj, target_obj, first_frame, last_frame,
                            keep_target_object=False):
    """Animate two ID-aligned PLY meshes along their periodic-unwrapped paths."""
    import bpy

    start_vertices = start_obj.data.vertices
    target_vertices = target_obj.data.vertices
    if len(start_vertices) != len(target_vertices):
        raise RuntimeError(
            "Blender timeline endpoint PLYs have different vertex counts; refusing an unaligned interpolation"
        )
    start_coordinates = [vertex.co.copy() for vertex in start_vertices]
    target_coordinates = [vertex.co.copy() for vertex in target_vertices]
    if not start_coordinates:
        raise RuntimeError("Cannot create a shape key for an empty matched-particle interval")

    def assign_motion(obj, basis_coordinates, motion_coordinates):
        for vertex, coordinate in zip(obj.data.vertices, basis_coordinates):
            vertex.co = coordinate
        obj.shape_key_add(name="Saved output positions")
        motion = obj.shape_key_add(name="Interpolated next output")
        for point, coordinate in zip(motion.data, motion_coordinates):
            point.co = coordinate
        motion.value = 0.0
        motion.keyframe_insert(data_path="value", frame=int(first_frame))
        motion.value = 1.0
        motion.keyframe_insert(data_path="value", frame=int(last_frame))
        shape_keys = obj.data.shape_keys
        if shape_keys and shape_keys.animation_data and shape_keys.animation_data.action:
            for curve in getattr(shape_keys.animation_data.action, "fcurves", []):
                for keyframe in curve.keyframe_points:
                    keyframe.interpolation = "LINEAR"

    assign_motion(start_obj, start_coordinates, target_coordinates)
    if keep_target_object:
        # Retain the target PLY's gas smoothing lengths and density weights, but
        # give it the same interpolated positions as the start endpoint object.
        assign_motion(target_obj, start_coordinates, target_coordinates)
        return target_obj
    unused_mesh = target_obj.data
    unused_materials = list(unused_mesh.materials)
    unused_groups = [modifier.node_group for modifier in target_obj.modifiers
                     if modifier.type == "NODES" and modifier.node_group is not None]
    bpy.data.objects.remove(target_obj, do_unlink=True)
    if unused_mesh.users == 0:
        bpy.data.meshes.remove(unused_mesh)
    for material in unused_materials:
        if material is not None and material.users == 0:
            bpy.data.materials.remove(material)
    for group in unused_groups:
        if group.users == 0:
            bpy.data.node_groups.remove(group)
    return None


def _make_interval_label(camera, interval, box_size, species, first_frame,
                         last_frame, frame_end):
    """Label interpolation and the exact endpoint/event counts honestly."""
    import bpy

    label = bpy.data.curves.new("Timeline interval %04d" % int(interval["index"]), "FONT")
    time_start = interval.get("cosmic_time_start_gyr")
    time_end = interval.get("cosmic_time_end_gyr")
    time_label = (
        "t %.4g -> %.4g Gyr" % (float(time_start), float(time_end))
        if time_start is not None and time_end is not None else "proper time unavailable"
    )
    label.body = (
        "%s  |  outputs %d -> %d  |  z %.4g -> %.4g\n"
        "a %.5f -> %.5f  |  %s\n"
        "ID-matched: %d  |  endpoint losses/gains: %d / %d  |  periodic images: %d\n"
        "L = %.4g Mpc/h"
        % (species.upper(), int(interval["snapshot_start"]), int(interval["snapshot_end"]),
           float(interval["redshift_start"]), float(interval["redshift_end"]),
           float(interval["scale_factor_start"]), float(interval["scale_factor_end"]), time_label,
           int(interval["matched_particle_count"]), int(interval["left_only_event_count"]),
           int(interval["right_only_event_count"]), int(interval.get("periodic_image_count", 0)), float(box_size))
    )
    label.size = 1.0
    label.extrude = 0.0
    obj = bpy.data.objects.new("Timeline interval %04d label" % int(interval["index"]), label)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = camera
    obj.location = (-0.40 * box_size, 0.32 * box_size, -0.95 * box_size)
    obj.scale = (0.007 * box_size,) * 3
    material = bpy.data.materials.new("Timeline interval %04d label material" % int(interval["index"]))
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.88, 0.94, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.4
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    label.materials.append(material)
    _key_visibility(obj, first_frame, last_frame, frame_end)


def _build_exact_snapshot_timeline(manifest, manifest_path, movie_path):
    """Render a schema-v1 exact-snapshot timeline for explicit legacy mode."""
    import bpy

    snapshots = manifest.get("snapshots")
    manifest_path = Path(manifest_path).expanduser().resolve()
    species = manifest.get("species")
    box_size = float(manifest.get("box_mpc_h", 0.0))
    fps = int(manifest.get("fps", 24))
    frames_per_snapshot = int(manifest.get("frames_per_snapshot", 12))
    render_quality = str(manifest.get("render_quality", "preview")).lower()
    if (not isinstance(snapshots, list) or not snapshots or species not in {"dm", "gas", "stars", "black_holes"}
            or box_size <= 0 or fps <= 0 or frames_per_snapshot <= 0
            or render_quality not in {"preview", "high"}):
        raise ValueError("Legacy timeline manifest has invalid snapshots or render settings")
    sampling = manifest.get("sampling", {})
    if sampling.get("particle_mode") == "full":
        entries_with_particles = [entry for entry in snapshots if not entry.get("empty_species")]
        if any("particle_count" not in entry for entry in entries_with_particles):
            raise ValueError("Full-resolution timeline manifest is missing particle counts")
        particle_count = sum(int(entry["particle_count"]) for entry in entries_with_particles)
        memory_fraction = float(sampling.get("memory_estimate", {}).get("max_memory_fraction", 0.70))
        _validate_particle_memory(particle_count, memory_fraction)
        for entry in entries_with_particles:
            ply_path = _manifest_asset_path(manifest_path, entry["ply"])
            if _ply_vertex_count(ply_path) != int(entry["particle_count"]):
                raise ValueError("Full-resolution timeline PLY count does not match its manifest")
    for entry in snapshots:
        for key in ("snapshot", "scale_factor", "redshift", "ply"):
            if key not in entry:
                raise ValueError("Legacy timeline snapshot entry is missing " + key)
        if not entry.get("empty_species") and not entry.get("ply"):
            raise ValueError("Legacy timeline snapshot with particles must reference a PLY sample")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene['dmsim_data_contract_json']=blender_json.dumps(manifest.get('data_contract'),sort_keys=True)
    scene['dmsim_epochs_json']=blender_json.dumps([item.get('epoch_contract') for item in manifest['snapshots']],sort_keys=True)
    scene['dmsim_manifest_sha256']=file_sha256(manifest_path)
    scene['dmsim_science_acceptance']='NOT_ACCEPTED'
    frame_end = len(snapshots) * frames_per_snapshot
    camera = _add_lighting_and_camera(
        scene, (box_size / 2.0,) * 3, box_size, frame_end,
        orbit_camera=bool(manifest.get("camera_orbit", True)),
    )
    for index, entry in enumerate(snapshots):
        first_frame = index * frames_per_snapshot + 1
        last_frame = (index + 1) * frames_per_snapshot
        if not entry.get("empty_species"):
            obj = _import_particle_mesh(
                str(_manifest_asset_path(manifest_path, entry["ply"])), species, box_size, index,
            )
            _key_visibility(obj, first_frame, last_frame, frame_end)
        _make_timeline_label(camera, entry, box_size, species, index,
                             first_frame, last_frame, frame_end)

    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    scene.cycles.samples = 20 if render_quality == "preview" else 64
    scene.render.resolution_x = 1280 if render_quality == "preview" else 1920
    scene.render.resolution_y = 720 if render_quality == "preview" else 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    set_output_format(scene, "FFMPEG")
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    movie_path = Path(movie_path).expanduser().resolve()
    scene.render.filepath = "//" + movie_path.with_suffix("").name
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.color_mode = "RGB"
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    scene.frame_set(1)
    blend_path = movie_path.with_name(manifest_path.stem + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if bool(manifest.get("skip_render", False)):
        return blend_path
    bpy.ops.render.render(animation=True)
    ranged_output = movie_path.with_name(
        "%s%04d-%04d%s" % (movie_path.stem, scene.frame_start, scene.frame_end, movie_path.suffix)
    )
    if ranged_output.is_file():
        os.replace(ranged_output, movie_path)
    if not movie_path.is_file():
        raise RuntimeError("Blender completed without creating the requested timeline movie: " + str(movie_path))
    return movie_path


def _build_timeline_scene(manifest_path, movie_path):
    """Build a Blender scene with matched-particle interpolation between outputs."""
    import bpy

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = verify_timeline_assets(manifest_path)
    snapshots = manifest.get("snapshots")
    intervals = manifest.get("intervals")
    if intervals is None:
        return _build_exact_snapshot_timeline(manifest, manifest_path, movie_path)
    species = manifest.get("species")
    box_size = float(manifest.get("box_mpc_h", 0.0))
    fps = int(manifest.get("fps", 24))
    frames_per_snapshot = int(manifest.get("frames_per_snapshot", 12))
    render_quality = str(manifest.get("render_quality", "preview")).lower()
    if (not isinstance(snapshots, list) or not snapshots or not isinstance(intervals, list)
            or not intervals or species not in {"dm", "gas", "stars", "black_holes"}
            or box_size <= 0 or fps <= 0 or frames_per_snapshot <= 0
            or render_quality not in {"preview", "high"}):
        raise ValueError("Timeline manifest has invalid snapshots, species, box size, or frame settings")
    expected_intervals = len(snapshots) - 1
    if len(intervals) != expected_intervals:
        raise ValueError("Timeline manifest must contain one interval between each adjacent saved output")
    for interval in intervals:
        for key in ("index", "snapshot_start", "snapshot_end", "frame_start", "frame_end",
                    "matched_particle_count", "left_only_event_count", "right_only_event_count"):
            if key not in interval:
                raise ValueError("Timeline interval is missing " + key)
        for key in ("matched_start_ply", "matched_end_ply"):
            if bool(interval.get(key)) != (int(interval["matched_particle_count"]) > 0):
                raise ValueError("Timeline interval matched PLY fields do not match its particle count")
        if int(interval["frame_end"]) <= int(interval["frame_start"]):
            raise ValueError("Timeline interpolation intervals require at least two frames")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene['dmsim_data_contract_json']=blender_json.dumps(manifest.get('data_contract'),sort_keys=True)
    scene['dmsim_epochs_json']=blender_json.dumps([item.get('epoch_contract') for item in manifest['snapshots']],sort_keys=True)
    scene['dmsim_manifest_sha256']=file_sha256(manifest_path)
    scene['dmsim_science_acceptance']='NOT_ACCEPTED'
    frame_end = int(manifest.get("frame_end", len(intervals) * frames_per_snapshot))
    if frame_end != len(intervals) * frames_per_snapshot:
        raise ValueError("Timeline frame_end does not agree with interval frame settings")
    center = (box_size / 2.0,) * 3
    camera = _add_lighting_and_camera(
        scene, center, box_size, frame_end,
        orbit_camera=bool(manifest.get("camera_orbit", False)),
    )
    for interval in intervals:
        index = int(interval["index"])
        first_frame = int(interval["frame_start"])
        last_frame = int(interval["frame_end"])
        if int(interval["matched_particle_count"]):
            start_obj = _import_particle_mesh(
                str(_manifest_asset_path(manifest_path, interval["matched_start_ply"])),
                species, box_size, index * 3,
            )
            target_obj = _import_particle_mesh(
                str(_manifest_asset_path(manifest_path, interval["matched_end_ply"])),
                species, box_size, index * 3 + 1,
            )
            _add_interval_shape_key(
                start_obj, target_obj, first_frame, last_frame,
                keep_target_object=(index == len(intervals) - 1),
            )
            _key_visibility(start_obj, first_frame, last_frame - 1, frame_end)
            if index == len(intervals) - 1:
                _key_visibility(target_obj, last_frame, last_frame, frame_end)
        left_path = interval.get("left_only_ply")
        if left_path:
            left_obj = _import_particle_mesh(
                str(_manifest_asset_path(manifest_path, left_path)), species, box_size, index * 3 + 2,
            )
            _key_visibility(left_obj, first_frame, max(first_frame, last_frame - 1), frame_end)
        right_path = interval.get("right_only_ply")
        if right_path and index == len(intervals) - 1:
            right_obj = _import_particle_mesh(
                str(_manifest_asset_path(manifest_path, right_path)), species, box_size, index * 3 + 3,
            )
            _key_visibility(right_obj, last_frame, last_frame, frame_end)
        _make_interval_label(camera, interval, box_size, species, first_frame, last_frame - 1, frame_end)
        snapshot_entry = snapshots[index] if index < len(snapshots) else snapshots[-1]
        overlay_groups = snapshot_entry.get("halo_overlays", {}) if isinstance(snapshot_entry, dict) else {}
        if isinstance(overlay_groups, dict):
            marker_factors = {"halos": 0.0030, "subhalos": 0.0018, "rockstar": 0.0040, "galaxies": 0.0050}
            colors = {
                "halos": (0.20, 0.52, 0.88),
                "subhalos": (0.40, 0.75, 0.95),
                "rockstar": (0.85, 0.40, 0.20),
                "galaxies": (1.0, 0.78, 0.34),
            }
            for kind in ("halos", "subhalos", "rockstar", "galaxies"):
                overlay = overlay_groups.get(kind, {})
                items = overlay.get("items", []) if isinstance(overlay, dict) else []
                if not isinstance(items, list) or not items:
                    continue
                factor = marker_factors[kind]
                color = colors[kind]
                _import_catalogue_overlay(
                    items, str(manifest.get("model", "unlabeled")), kind, index, box_size, 0.0, color,
                    marker_radius_factor=factor,
                    first_frame=first_frame, last_frame=last_frame, frame_end=frame_end,
                )
                _import_catalogue_overlay(
                    items, str(manifest.get("model", "unlabeled")), kind, index, box_size, 0.0, color,
                    rare_only=True, marker_radius_factor=factor * 1.8,
                    first_frame=first_frame, last_frame=last_frame, frame_end=frame_end,
                )
                _import_catalogue_overlay(
                    items, str(manifest.get("model", "unlabeled")), kind, index, box_size, 0.0, color,
                    r200_only=True,
                    first_frame=first_frame, last_frame=last_frame, frame_end=frame_end,
                )
            if manifest.get("show_halo_ids_and_masses", True):
                label_limit = int(manifest.get("maximum_halo_labels_per_model_output", 8))
                for kind in ("galaxies", "rockstar", "halos", "subhalos"):
                    overlay = overlay_groups.get(kind, {})
                    items = overlay.get("items", []) if isinstance(overlay, dict) else []
                    if isinstance(items, list) and items:
                        _add_halo_metadata_labels(
                            items, str(manifest.get("model", "unlabeled")), kind, index, box_size, 0.0,
                            first_frame, last_frame, frame_end, camera, colors[kind], label_limit,
                        )
                        break

    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    scene.cycles.samples = 20 if render_quality == "preview" else 64
    scene.render.resolution_x = 1280 if render_quality == "preview" else 1920
    scene.render.resolution_y = 720 if render_quality == "preview" else 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    set_output_format(scene, "FFMPEG")
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    movie_path = Path(movie_path).expanduser().resolve()
    scene.render.filepath = "//" + movie_path.with_suffix("").name
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.color_mode = "RGB"
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    scene.frame_set(1)
    blend_path = movie_path.with_name(manifest_path.stem + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not bool(manifest.get("skip_render", False)):
        bpy.ops.render.render(animation=True)
        ranged_output = movie_path.with_name(
            "%s%04d-%04d%s" % (movie_path.stem, scene.frame_start, scene.frame_end, movie_path.suffix)
        )
        if ranged_output.is_file():
            os.replace(ranged_output, movie_path)
        if not movie_path.is_file():
            raise RuntimeError("Blender completed without creating the requested timeline movie: " + str(movie_path))
        return movie_path
    return blend_path


def _build_scene(ply_path, movie_path, box_size, species="dm", fps=24, frame_count=240,
                render_quality="preview", particle_count=None, max_memory_fraction=0.70, *, source_record=None):
    """Build a Blender scene from saved exported particle assets; rendering changes display only."""
    import bpy

    ply_path = Path(ply_path).expanduser().resolve()
    movie_path = Path(movie_path).expanduser().resolve()
    box_size = float(box_size)
    render_quality = str(render_quality).lower()
    if species not in {"dm", "gas", "stars", "black_holes"}:
        raise ValueError("species must be dm, gas, or stars")
    if (not ply_path.is_file() or not math.isfinite(box_size) or box_size <= 0 or int(frame_count) < 2
            or render_quality not in {"preview", "high"}):
        raise ValueError("PLY path, box size, or animation frame count is invalid")
    if particle_count is not None:
        _validate_particle_memory(particle_count, max_memory_fraction)
        if _ply_vertex_count(ply_path) != int(particle_count):
            raise ValueError("Full-resolution PLY vertex count does not match its exported particle count")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = read_particle_mesh(bpy, ply_path)
    if obj is None or len(obj.data.vertices) == 0:
        raise RuntimeError("The particle PLY imported without vertices")
    obj.name = "DarkMatterSim_%s_%s" % (species, "full" if particle_count is not None else "sample")
    obj.data.materials.append(_make_material(species))
    _make_geometry_nodes(obj, species, box_size)
    center = (box_size / 2.0, box_size / 2.0, box_size / 2.0)
    _add_lighting_and_camera(bpy.context.scene, center, box_size, int(frame_count))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = int(frame_count)
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    scene.cycles.samples = 20 if render_quality == "preview" else 64
    scene.render.resolution_x = 1280 if render_quality == "preview" else 1920
    scene.render.resolution_y = 720 if render_quality == "preview" else 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    set_output_format(scene, "FFMPEG")
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = "//" + movie_path.with_suffix("").name
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.color_mode = "RGB"
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    scene["dmsim_science_acceptance"] = "NOT_ACCEPTED"
    scene["dmsim_source_contract_json"] = blender_json.dumps(source_record, sort_keys=True)
    scene["dmsim_display_options_json"] = blender_json.dumps({"box_size": box_size, "species": species,
        "fps": fps, "frame_count": frame_count, "render_quality": render_quality}, sort_keys=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(movie_path.with_suffix(".blend")))
    bpy.ops.render.render(animation=True)
    ranged_output = movie_path.with_name(
        "%s%04d-%04d%s" % (movie_path.stem, scene.frame_start, scene.frame_end, movie_path.suffix)
    )
    if ranged_output.is_file():
        os.replace(ranged_output, movie_path)
    if not movie_path.is_file():
        raise RuntimeError("Blender completed without creating the requested movie: " + str(movie_path))
    return movie_path


def _overlay_material(name, color, alpha=1.0):
    import bpy

    color_key = "_".join("%.3f" % float(value) for value in color[:3])
    material_name = "DarkMatterSim overlay %s a%.3f" % (color_key, float(alpha))
    material = bpy.data.materials.get(material_name)
    if material is not None:
        return material
    material = bpy.data.materials.new(material_name)
    material.diffuse_color = (*color[:3], float(alpha))
    material.use_nodes = True
    shader = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is not None:
        _socket(shader, "Base Color").default_value = (*color[:3], float(alpha))
        if shader.inputs.get("Alpha") is not None:
            _socket(shader, "Alpha").default_value = float(alpha)
        if shader.inputs.get("Emission Color") is not None:
            _socket(shader, "Emission Color").default_value = tuple(float(value) * 0.22 for value in color[:3]) + (1.0,)
        if shader.inputs.get("Emission Strength") is not None:
            _socket(shader, "Emission Strength").default_value = 0.35
        _socket(shader, "Roughness").default_value = 0.44
    return material


def _make_catalogue_geometry_nodes(obj, box_size, radius_attribute=None, wireframe=False,
                                   marker_radius_factor=0.0022):
    import bpy

    group_name = "DarkMatterSim catalogue instancing %s %s %.5f" % (
        str(radius_attribute or "fixed"), "wire" if wireframe else "solid", float(marker_radius_factor),
    )
    group = bpy.data.node_groups.get(group_name)
    if group is not None:
        modifier = obj.modifiers.new("Instanced catalogue markers", "NODES")
        modifier.node_group = group
        return
    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    if hasattr(group, "interface"):
        group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:
        group.inputs.new("NodeSocketGeometry", "Geometry")
        group.outputs.new("NodeSocketGeometry", "Geometry")
    nodes, links = group.nodes, group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    links.new(input_node.outputs["Geometry"], to_points.inputs["Mesh"])
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 1.0
    if ico.inputs.get(" subdivisions") is not None:
        ico.inputs[" subdivisions"].default_value = 2
    elif ico.inputs.get("Subdivisions") is not None:
        ico.inputs["Subdivisions"].default_value = 2
    sphere_geometry = ico.outputs["Mesh"]
    if wireframe:
        # Blender 4.5 no longer exposes GeometryNodeWireframe. Convert each
        # icosphere edge to a curve and give it a narrow circular tube. The
        # instance Scale below still uses each halo's measured R200c.
        edge_curves = nodes.new("GeometryNodeMeshToCurve")
        links.new(sphere_geometry, edge_curves.inputs["Mesh"])
        tube_profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
        tube_profile.inputs["Radius"].default_value = 0.006
        tube_profile.inputs["Resolution"].default_value = 4
        curve_mesh = nodes.new("GeometryNodeCurveToMesh")
        links.new(edge_curves.outputs["Curve"], curve_mesh.inputs["Curve"])
        links.new(tube_profile.outputs["Curve"], curve_mesh.inputs["Profile Curve"])
        sphere_geometry = curve_mesh.outputs["Mesh"]
    catalogue_material = nodes.new("GeometryNodeSetMaterial")
    catalogue_material.inputs["Material"].default_value = obj.data.materials[0]
    links.new(sphere_geometry, catalogue_material.inputs["Geometry"])
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(to_points.outputs["Points"], instance.inputs["Points"])
    links.new(catalogue_material.outputs["Geometry"], instance.inputs["Instance"])
    if radius_attribute:
        radius = nodes.new("GeometryNodeInputNamedAttribute")
        radius.data_type = "FLOAT"
        radius.inputs["Name"].default_value = str(radius_attribute)
        combine = nodes.new("ShaderNodeCombineXYZ")
        for axis in ("X", "Y", "Z"):
            links.new(radius.outputs["Attribute"], combine.inputs[axis])
        links.new(combine.outputs["Vector"], instance.inputs["Scale"])
    else:
        default_radius = max(float(box_size) * float(marker_radius_factor), 1.0e-4)
        instance.inputs["Scale"].default_value = (default_radius,) * 3
    links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])
    modifier = obj.modifiers.new("Instanced catalogue markers", "NODES")
    modifier.node_group = group


def _import_catalogue_overlay(items, model, kind, frame_index, box_size, offset, color,
                              *, rare_only=False, r200_only=False, marker_radius_factor=0.0022,
                              first_frame=1, last_frame=1, frame_end=1):
    import bpy

    if r200_only:
        rows = [item for item in items if item.get("r200c_mpc_h") is not None
                and float(item["r200c_mpc_h"]) > 0]
    elif rare_only:
        rows = [item for item in items if bool(item.get("rare", False))]
    else:
        rows = list(items)
    if not rows:
        return None
    positions = []
    radii = []
    aligned = []
    for item in rows:
        position = item.get("position_mpc_h")
        if not isinstance(position, list) or len(position) != 3:
            continue
        xyz = [float(value) for value in position]
        if not all(math.isfinite(value) for value in xyz):
            continue
        positions.append((xyz[0] + float(offset), xyz[1], xyz[2]))
        aligned.append(item)
        if r200_only:
            radii.append(float(item["r200c_mpc_h"]))
    if not positions:
        return None
    mesh = bpy.data.meshes.new("%s %s frame %04d catalogue mesh" % (model.upper(), kind, int(frame_index)))
    mesh.from_pydata(positions, [], [])
    mesh.update()
    if r200_only:
        radius_data = mesh.attributes.new(name="r200c_mpc_h", type="FLOAT", domain="POINT")
        for index, value in enumerate(radii):
            radius_data.data[index].value = value
    if len(aligned) == len(positions):
        for attribute_name, values in (
            ("mass_msun_h", [float(item.get("mass_msun_h", 0.0)) for item in aligned]),
            ("rare", [bool(item.get("rare", False)) for item in aligned]),
        ):
            attribute_type = "BOOLEAN" if attribute_name == "rare" else "FLOAT"
            attribute = mesh.attributes.new(name=attribute_name, type=attribute_type, domain="POINT")
            for index, value in enumerate(values):
                attribute.data[index].value = value
    object_name = "DarkMatterSim_%s_%s_%s_%04d" % (
        model, kind, "r200c" if r200_only else "rare" if rare_only else "hosts", int(frame_index),
    )
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    color = (1.0, 0.67, 0.18) if rare_only else color
    alpha = 0.27 if r200_only else 1.0
    obj.data.materials.append(_overlay_material(object_name + " material", color, alpha))
    _make_catalogue_geometry_nodes(
        obj, box_size, "r200c_mpc_h" if r200_only else None,
        wireframe=r200_only, marker_radius_factor=marker_radius_factor,
    )
    _key_visibility(obj, first_frame, last_frame, frame_end)
    obj["halo_ids_json"] = blender_json.dumps([int(item.get("id", -1)) for item in aligned])
    obj["halo_masses_msun_h_json"] = blender_json.dumps([float(item.get("mass_msun_h", 0.0)) for item in aligned])
    obj["mass_definition"] = str(aligned[0].get("mass_definition", "unknown"))
    obj["catalogue_kind"] = str(kind)
    return obj


def _add_halo_metadata_labels(items, model, kind, frame_index, box_size, offset,
                              first_frame, last_frame, frame_end, camera, color, maximum=8):
    import bpy

    for item in items[:max(0, int(maximum))]:
        position = item.get("position_mpc_h")
        if not isinstance(position, list) or len(position) != 3:
            continue
        try:
            xyz = [float(value) for value in position]
            halo_id = int(item.get("id", -1))
            mass = float(item.get("mass_msun_h", 0.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in xyz) or not math.isfinite(mass):
            continue
        if kind == "galaxies":
            text = "Galaxy #%d\nM★ = %.3g M_sun/h" % (halo_id, mass)
            if item.get("formation_redshift") is not None and math.isfinite(float(item["formation_redshift"])):
                text += "\nz₅₀,★ = %.2f" % float(item["formation_redshift"])
        else:
            text = "%s #%d\nM%s = %.3g M_sun/h" % (
                str(kind).upper(), halo_id, str(item.get("mass_definition", "")), mass,
            )
            if item.get("subhalo_count") is not None:
                text += "\nNsub = %d" % int(item["subhalo_count"])
        if item.get("detection_redshift") is not None and math.isfinite(float(item["detection_redshift"])):
            text += "\nDetected z = %.4g" % float(item["detection_redshift"])
        curve = bpy.data.curves.new(
            "Halo metadata label %s %s %d %04d" % (model.upper(), kind, halo_id, int(frame_index)), "FONT",
        )
        curve.body = text
        curve.size = max(float(box_size) * 0.010, 1.0e-4)
        curve.extrude = 0.0
        label = bpy.data.objects.new(
            "DarkMatterSim_%s_%s_label_%d_%04d" % (model, kind, halo_id, int(frame_index)), curve,
        )
        bpy.context.scene.collection.objects.link(label)
        label.location = (xyz[0] + float(offset), xyz[1], xyz[2] + float(box_size) * 0.008)
        target = label.constraints.new("TRACK_TO")
        target.target = camera
        target.track_axis = "TRACK_NEGATIVE_Z"
        target.up_axis = "UP_Y"
        label.data.materials.append(_overlay_material(
            label.name + " material", (1.0, 0.67, 0.18) if item.get("rare", False) else color, 1.0,
        ))
        label["halo_id"] = halo_id
        label["id_source"] = str(item.get("id_source", "catalogue_row"))
        label["mass_msun_h"] = mass
        label["mass_definition"] = str(item.get("mass_definition", "unknown"))
        _key_visibility(label, first_frame, last_frame, frame_end)


def verify_stellar_candidate_assets(manifest_path):
    """Verify current portable stellar assets using only the standard library."""
    manifest_path=Path(manifest_path).expanduser().resolve()
    manifest=blender_json.loads(manifest_path.read_text())
    if manifest.get('schema')!='darkmattersim-stellar-candidate-blender-v1':
        raise ValueError('Unsupported stellar candidate Blender manifest')
    if manifest.get('provenance',{}).get('data_contract') is not None:
        seal=manifest_path.with_suffix('.json.sha256')
        if not seal.is_file() or seal.read_text().strip()!=file_sha256(manifest_path):
            raise ValueError('Stellar candidate Blender manifest checksum mismatch')
        if not manifest.get('assets'):raise ValueError('Stellar candidate manifest lacks sealed assets')
        seen=set()
        for item in manifest['assets']:
            path=_manifest_asset_path(manifest_path,item['path'])
            if not path.is_relative_to(manifest_path.parent) or path in seen:
                raise ValueError('Invalid or duplicate stellar candidate asset path')
            seen.add(path)
            if not path.is_file() or path.stat().st_size!=item['size_bytes'] or file_sha256(path)!=item['sha256']:
                raise ValueError('Stellar candidate asset missing or changed: '+item['path'])
            if path.suffix=='.ply':verify_particle_ply_payload(path)
    return manifest


def _build_stellar_candidate_scene(manifest_path, blend_path, render=False, *, public_directory=None):
    """Build editable measured stellar morphology in proper host-centered kpc.

    The exported point order is unchanged; negative coordinates must remain
    visible. Candidate colours encode exact catalogue membership, not stellar
    spectra, binding to the full host potential or cross-epoch trajectories.
    Native IDs remain uint64 in the referenced, checksummed source catalogue.
    """
    import bpy

    manifest_path = Path(manifest_path).expanduser().resolve()
    blend_path = Path(blend_path).expanduser().resolve()
    manifest=verify_stellar_candidate_assets(manifest_path)
    if manifest.get("length_unit") != "proper host-centered kpc":
        raise ValueError("Stellar candidate scene requires explicit physical kpc units")
    count = manifest["particle_count"]
    extent = float(manifest["scene_extent_kpc"])
    if isinstance(count, bool) or int(count) != count or count < 1 or not math.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid stellar point count or physical display extent")
    _validate_particle_memory(count, .70)
    paths = {}
    for key in ("ply", "catalogue"):
        path = _manifest_asset_path(manifest_path, manifest[key])
        if file_sha256(path) != manifest[key + "_sha256"]:
            raise ValueError("Stellar asset no longer matches its export: " + str(path))
        paths[key] = path
    if _ply_vertex_count(paths["ply"]) != count:
        raise ValueError("Stellar PLY vertex count disagrees with catalogue export")
    palette = manifest["palette_srgb_rgba"]
    glyph_fraction = float(manifest["glyph_radius_extent_fraction"])
    if not math.isfinite(glyph_fraction) or not 0 < glyph_fraction <= .02:
        raise ValueError("Stellar display radius must be in (0,0.02] times the scene extent")
    if (not isinstance(palette, list) or len(palette) != 11
            or any(not isinstance(color, list) or len(color) != 4 for color in palette)
            or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for color in palette for x in color)):
        raise ValueError("Stellar display requires eleven finite RGBA categories")
    # Figure palettes use sRGB; Blender emission inputs use linear colour.
    linear = [[x/12.92 if x <= .04045 else ((x+.055)/1.055)**2.4 for x in color[:3]]
              + [color[3]] for color in palette]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = _import_particle_mesh(paths["ply"], "stars", extent, 0,
                                crop_periodic=False, candidate_palette=linear,
                                candidate_glyph_fraction=glyph_fraction)
    if len(obj.data.vertices) != count or obj.data.attributes.get("stellar_candidate_color_bin") is None:
        raise ValueError("Blender did not preserve the exported stellar vertices/categories")
    obj["length_unit"] = manifest["length_unit"]
    obj["exact_particle_ids"] = "catalogue ParticleIDs[row]; mesh vertex index is row"
    obj["catalogue_path"] = "//" + os.path.relpath(paths["catalogue"], public_directory or blend_path.parent)
    obj["catalogue_sha256"] = manifest["catalogue_sha256"]
    scene = bpy.context.scene
    scene.unit_settings.system = "NONE"
    scene["length_unit"] = manifest["length_unit"]
    scene["snapshot_provenance_json"] = blender_json.dumps(manifest["provenance"])
    scene["display_glyph_radius_kpc"] = extent * glyph_fraction
    scene["color_categories"] = manifest["color_categories"]
    camera = _add_lighting_and_camera(scene, (0., 0., 0.), extent, 1, orbit_camera=False)
    camera.data.dof.use_dof = False
    scene.frame_start = scene.frame_end = 1
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    scene.cycles.samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.render.resolution_percentage = 100
    set_output_format(scene, "PNG")
    scene.view_settings.view_transform = "Standard"
    label = bpy.data.curves.new("Stellar candidate provenance", "FONT")
    source = manifest["provenance"]
    label.body = (f"STELLAR PHASE-SPACE CANDIDATES | snapshot {source['snapshot']} | host {source['host_fof_row']}\n"
                  f"{count:,} saved stars | {manifest['candidate_count']} candidates | proper kpc\n"
                  "Grey: unselected; colours: membership, cycling after 10.\n"
                  "Point radius/emission are display choices; saved morphology is unchanged.")
    text = bpy.data.objects.new("Stellar provenance label", label)
    scene.collection.objects.link(text)
    text.parent = camera
    text.location = (-.40*extent, .20*extent, -extent)
    text.scale = (.008*extent,)*3
    label.materials.append(_blender_emission_material(bpy, "Stellar provenance text", (.8,.9,1.,1.), strength=1.))
    # Fit to the camera rather than letting long counts/labels spill past it.
    bpy.context.view_layer.update()
    if text.dimensions.x > .78*extent:
        text.scale *= .78*extent/text.dimensions.x
    bpy.data.texts.new("STELLAR_EXPORT_PROVENANCE.json").write(blender_json.dumps(manifest, indent=2))
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = "//" + blend_path.with_suffix(".png").name
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if render:
        bpy.ops.render.render(write_still=True)
    return blend_path


def build_scene(ply_path, movie_path, box_size, species="dm", fps=24, frame_count=240,
                render_quality="preview", particle_count=None, max_memory_fraction=0.70, *, manifest_path=None):
    """Publish a single-snapshot editable scene and movie with retained inputs.

    Raw historical PLY inputs carry no scientific identity or clock. They remain
    explicitly legacy display products; a nearby sealed export supplies the
    actual central scientific contract when available.
    """
    import uuid
    from core.native import native_run_lock
    from core.foundation import atomic_json, atomic_text
    from blender import publish_scene_product
    ply_path, movie_path = Path(ply_path).expanduser().resolve(), Path(movie_path).expanduser().resolve()
    if movie_path.suffix.lower() != '.mp4':
        raise ValueError('Single-scene movie destination must have an .mp4 suffix')
    count = verify_particle_ply_payload(ply_path)
    if particle_count is not None and count != particle_count:
        raise ValueError('Single-scene vertex count disagrees with requested particle count')
    candidate = Path(manifest_path).expanduser().resolve() if manifest_path is not None else ply_path.parent / 'blender_single_contract.json'
    with native_run_lock(movie_path.with_suffix('.blend'), operation='single-snapshot Blender publication'):
        if candidate.is_file() or manifest_path is not None:
            def verify(path):
                record = verify_single_scene_assets(path)
                if not any(_manifest_asset_path(path, item['path']) == ply_path for item in record['assets']):
                    raise ValueError('Requested PLY is absent from the sealed single-scene contract')
                record['integrity_status'] = 'SEALED'
                return record
            source_path = candidate
        else:
            # Retain a checksum-bound legacy receipt; do not invent IDs, units or epochs.
            evidence = movie_path.parent / '.scene_inputs' / uuid.uuid4().hex
            evidence.mkdir(parents=True)
            source_path = evidence / 'legacy_ply.json'
            atomic_json(source_path, {'schema': 'legacy-ply-scene-input-v1',
                'integrity_status': 'LEGACY_UNSEALED', 'science_acceptance': 'NOT_ACCEPTED',
                'ply': str(ply_path), 'ply_sha256': file_sha256(ply_path), 'vertex_count': count,
                'scientific_contract': None, 'display_options': {'box_size': box_size, 'species': species,
                    'fps': fps, 'frame_count': frame_count, 'render_quality': render_quality}})
            atomic_text(source_path.with_name(source_path.name + '.sha256'), file_sha256(source_path) + '\n')
            def verify(path):
                seal = path.with_name(path.name + '.sha256')
                if seal.read_text().strip() != file_sha256(path):
                    raise ValueError('Legacy PLY input receipt changed')
                record = blender_json.loads(path.read_text())
                if file_sha256(record['ply']) != record['ply_sha256'] or verify_particle_ply_payload(record['ply']) != record['vertex_count']:
                    raise ValueError('Legacy PLY input changed')
                return record
        source = verify(source_path)
        def builder(source_path, staged_blend, render):
            return _build_scene(ply_path, staged_blend.with_suffix('.mp4'), box_size, species,
                fps, frame_count, render_quality, particle_count, max_memory_fraction, source_record=source)
        publish_scene_product(source_path, movie_path.with_suffix('.blend'), builder,
            input_schema=source.get('schema'), algorithm=__file__, render=True,
            manifest_verifier=verify, render_artifacts=[movie_path.name])
    return movie_path


def _scene_source_integrity(reader, path, *, sealed):
    """Adapt existing semantic readers to the common durable scene publisher."""
    manifest = reader(path)
    manifest['integrity_status'] = 'SEALED' if sealed(manifest) else 'LEGACY_UNSEALED'
    return manifest


def build_timeline_scene(manifest_path, movie_path):
    """Publish an editable timeline and requested movie as one retained generation."""
    from blender import publish_scene_product
    manifest_path, movie_path = Path(manifest_path).expanduser().resolve(), Path(movie_path).expanduser().resolve()
    if movie_path.suffix.lower() != '.mp4':
        raise ValueError('Timeline movie destination must have an .mp4 suffix')
    verify = lambda path: _scene_source_integrity(verify_timeline_assets, path,
        sealed=lambda record: 'data_contract' in record)
    source = verify(manifest_path)
    render = not bool(source.get('skip_render', False))
    blend_path = movie_path.with_name(manifest_path.stem + '.blend')
    def builder(source_path, staged_blend, requested_render):
        _build_timeline_scene(source_path, staged_blend.parent / movie_path.name)
    publish_scene_product(manifest_path, blend_path, builder,
        input_schema=source.get('schema'), algorithm=__file__, render=render,
        manifest_verifier=verify, render_artifacts=[movie_path.name])
    return movie_path if render else blend_path


def build_stellar_candidate_scene(manifest_path, blend_path, render=False):
    """Publish the measured stellar scene without changing its scientific inputs."""
    from blender import publish_scene_product
    blend_path = Path(blend_path).expanduser().resolve()
    verify = lambda path: _scene_source_integrity(verify_stellar_candidate_assets, path,
        sealed=lambda record: record.get('provenance', {}).get('data_contract') is not None)
    def builder(source_path, staged_blend, requested_render):
        return _build_stellar_candidate_scene(source_path, staged_blend, requested_render,
            public_directory=blend_path.parent)
    return publish_scene_product(manifest_path, blend_path, builder,
        input_schema='darkmattersim-stellar-candidate-blender-v1', algorithm=__file__,
        render=render, manifest_verifier=verify)


def blender_main(argv=None):
    """Parse command arguments and dispatch this module's documented workflow."""
    args = list(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if argv is not None:
        args = list(argv)
    if len(args) == 3 and args[0] == "--timeline":
        build_timeline_scene(args[1], args[2])
        return
    if len(args) in {3, 4} and args[0] == "--stellar-candidates":
        if len(args) == 4 and args[3] != "--render":
            raise SystemExit("Optional stellar scene argument is --render")
        build_stellar_candidate_scene(args[1], args[2], render=len(args)==4)
        return
    if len(args) not in {6, 7, 9}:
        raise SystemExit(
            "usage: blender_scene.py <particles.ply> <movie.mp4> <box_mpc_h> <species> "
            "<fps> <frames> [preview|high [particle_count max_memory_fraction]]"
        )
    quality = args[6] if len(args) == 7 else "preview"
    particle_count = int(args[7]) if len(args) == 9 else None
    memory_fraction = float(args[8]) if len(args) == 9 else 0.70
    if len(args) == 9:
        quality = args[6]
    build_scene(
        args[0], args[1], float(args[2]), args[3], int(args[4]), int(args[5]),
        quality, particle_count, memory_fraction,
    )


def _text(bpy, scene, name, body, location, size, material):
    data = bpy.data.curves.new(name, "FONT")
    data.body = body
    # Blender clamps FONT.size to a minimum; use object scale so sub-kpc
    # cameras still receive correctly sized screen annotations.
    data.size = 1.0
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.scale = (size, size, size)
    data.materials.append(material)
    return obj


def _emission(bpy, name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return material, emission


def _focus_saved_region(
    bpy, scene, overview, collection, reference, hud, caption, material
):
    """Fit a static camera to real final-epoch mesh bounds, keeping HUD scale."""
    from mathutils import Vector

    original_frame = overview.frame_current
    overview.frame_set(overview.frame_end)
    vertices = (
        obj.matrix_world @ vertex.co
        for obj in collection.objects
        if obj.type == "MESH" and not obj.hide_render
        for vertex in obj.data.vertices
    )
    lower, upper = [float("inf")] * 3, [float("-inf")] * 3
    for vertex in vertices:
        for axis in range(3):
            lower[axis] = min(lower[axis], vertex[axis])
            upper[axis] = max(upper[axis], vertex[axis])
    if lower[0] == float("inf"):
        overview.frame_set(original_frame)
        return
    box = float(reference["box_mpc_h"])
    span = max(max(upper[i] - lower[i] for i in range(3)) * 1.25, box * 0.004)
    factor = span / box
    center = Vector([(lower[i] + upper[i]) / 2 for i in range(3)])
    old = overview.camera
    camera = bpy.data.objects.new(scene.name + " Focus camera", old.data.copy())
    scene.collection.objects.link(camera)
    camera.matrix_world = old.matrix_world.copy()
    camera.location = (
        center + (old.matrix_world.translation - Vector([box / 2] * 3)) * factor
    )
    camera.data.clip_start = span * 1e-5
    camera.data.clip_end = max(box * 10, span * 10)
    scene.camera = camera
    for annotation in (hud, caption):
        if annotation is None:
            continue
        scene.collection.objects.unlink(annotation)
        clone = annotation.copy()
        scene.collection.objects.link(clone)
        clone.parent = camera
        clone.location = annotation.location * factor
        clone.scale = annotation.scale * factor
    scene["focused_region_span_comoving_mpc_h"] = span
    scene["focus_note"] = (
        "Fixed camera fitted to final saved particle bounds; particle glyph sizes are display choices."
    )
    distance = span * 0.2
    note = _text(
        bpy,
        scene,
        "Focused frame disclosure",
        f"Final-region camera reference: {span:.4g} comoving Mpc/h\nParticle glyphs visually enlarged; no resolved BH horizon",
        (0, 0, 0),
        distance * 0.006,
        material,
    )
    note.parent = camera
    note.location = (-distance * 0.24, -distance * 0.09, -distance)
    overview.frame_set(original_frame)


def build_sections(bpy, overview, collections, manifest, reference, base, *, output_base=None):
    """Add independently selectable species views and one scene per real figure."""
    overview.name = "01 | Universe timeline"
    white, _ = _emission(bpy, "Figure captions", (0.84, 0.91, 1, 1))
    label = manifest.get("validation_fixture")
    caption = None
    if label:
        distance = float(reference["box_mpc_h"]) * 0.2
        caption = _text(
            bpy,
            overview,
            "Input data origin",
            "SYNTHETIC VALIDATION FIXTURE",
            (0, 0, 0),
            distance * 0.008,
            white,
        )
        caption.parent = overview.camera
        caption.location = (-distance * 0.24, -distance * 0.12, -distance)
    sections = []
    for number, (species, title) in enumerate(
        [
            ("gas", "Gas and thermal feedback"),
            ("stars", "Stellar particles"),
            ("black_holes", "Black hole particles"),
            ("halos", "Halo catalogues"),
        ],
        2,
    ):
        collection = collections[species]
        section = bpy.data.scenes.new(f"{number:02d} | {title}")
        section.collection.children.link(collection)
        if species=='halos' and collections.get('dm') is not None:
            # The native inspection shells intentionally stay hidden from
            # renders. Actual DM particles supply the halo/subhalo morphology.
            section.collection.children.link(collections['dm'])
        section.collection.children.link(collections["rig"])
        hud = bpy.data.objects.get("Scientific_HUD_Overlay")
        if hud is not None:
            section.collection.objects.link(hud)
        if label:
            section.collection.objects.link(caption)
        section.camera = overview.camera
        section.world = overview.world
        section.render.engine = "CYCLES"
        disable_cycles_denoising(section)
        section.cycles.samples = 24
        section.render.resolution_x, section.render.resolution_y = 1280, 720
        section.render.fps = overview.render.fps
        section.view_settings.view_transform = "AgX"
        section.view_settings.exposure = overview.view_settings.exposure
        section.frame_start, section.frame_end = (
            overview.frame_start,
            overview.frame_end,
        )
        # The same HUD handler can report this section's own frame/epoch.
        for key in overview.keys():
            if key.startswith("dmsim_"):
                section[key] = overview[key]
        available = bool(collection.objects)
        section["section_status"] = "AVAILABLE" if available else "NO_SAVED_DATA"
        section["section_note"] = (
            "Comoving Mpc/h. Exact particle IDs; intermediate display frames interpolate saved positions."
            if available
            else manifest.get("unavailable_species", {}).get(
                species, "No saved catalogue objects"
            )
        )
        if available and species in {"gas", "black_holes"}:
            _focus_saved_region(
                bpy, section, overview, collection, reference, hud, caption, white
            )
        if not available:
            camera = section.camera
            distance = float(reference["box_mpc_h"]) * 0.2
            message = _text(
                bpy,
                section,
                "Unavailable data",
                title + "\nNO SAVED DATA\n" + section["section_note"],
                (0, 0, 0),
                distance * 0.022,
                white,
            )
            message.parent = camera
            message.location = (-distance * 0.18, 0, -distance)
        sections.append(
            {
                "scene": section.name,
                "status": section["section_status"],
                "note": section["section_note"],
            }
        )
    for index, entry in enumerate(manifest.get("figure_gallery", []), 6):
        path = (base / entry["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Measured figure missing: {path}")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Measured figure changed since export: {path}")
        scene = bpy.data.scenes.new(f"{index:02d} | {entry['title'][:48]}")
        scene.render.engine = "CYCLES"
        disable_cycles_denoising(scene)
        scene.cycles.samples = 1
        scene.render.resolution_x, scene.render.resolution_y = 1600, 1100
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        scene.world = bpy.data.worlds.new(scene.name + " World")
        scene.world.color = (0.005, 0.008, 0.016)
        image = bpy.data.images.load(str(path), check_existing=True)
        image.pack()
        aspect = image.size[0] / image.size[1]
        width = min(14.0, 7.6 * aspect)
        height = width / aspect
        mesh = bpy.data.meshes.new(scene.name + " Panel")
        mesh.from_pydata(
            [
                (-width / 2, -height / 2, 0),
                (width / 2, -height / 2, 0),
                (width / 2, height / 2, 0),
                (-width / 2, height / 2, 0),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        uv = mesh.uv_layers.new()
        for loop, coords in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            loop.uv = coords
        panel = bpy.data.objects.new(scene.name + " Figure", mesh)
        scene.collection.objects.link(panel)
        material, emission = _emission(bpy, scene.name + " Image", (1, 1, 1, 1))
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
        mesh.materials.append(material)
        camera_data = bpy.data.cameras.new(scene.name + " Camera")
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = 16
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        scene.collection.objects.link(camera)
        camera.location = (0, 0, 10)
        scene.camera = camera
        _text(
            bpy,
            scene,
            "Figure title",
            entry["title"].replace("_", " "),
            (-7, 4.7, 0),
            0.3,
            white,
        )
        _text(
            bpy,
            scene,
            "Figure source",
            entry.get("source", entry["path"]),
            (-7, -4.65, 0),
            0.17,
            white,
        )
        if label:
            _text(
                bpy,
                scene,
                "Input data origin",
                "SYNTHETIC VALIDATION FIXTURE",
                (-7, 4.25, 0),
                0.22,
                white,
            )
        scene["source_figure"] = str(path)
        scene["source_sha256"] = entry["sha256"]
        if 'scientific_context' in entry:
            scene['scientific_context_json']=blender_json.dumps(entry['scientific_context'],sort_keys=True)
            scene['scientific_context_sha256']=entry['scientific_context_sha256']
        scene["section_note"] = (
            "Measured analysis figure; source image packed without relabeling its physical axes."
        )
        sections.append({"scene": scene.name, "status": "AVAILABLE", "source": entry})
    overview["research_sections"] = blender_json.dumps(sections)
    from core.foundation import atomic_json
    atomic_json(Path(output_base or base) / 'blender.blender_json', sections)
    return sections


def render_sections(bpy, base):
    """Render all sections at the final saved epoch, with deterministic names."""
    base = Path(base)
    destination = base / "section_previews"
    destination.mkdir(exist_ok=True)
    records = []
    for index, scene in enumerate(bpy.data.scenes):
        scene.frame_set(scene.frame_end)
        set_output_format(scene, "PNG")
        scene.render.filepath = str(destination / f"section_{index + 1:02d}.png")
        bpy.ops.render.render(write_still=True, scene=scene.name)
        records.append(
            {
                "scene": scene.name,
                "image": Path(scene.render.filepath).relative_to(base).as_posix(),
                "status": scene.get("section_status", "AVAILABLE"),
            }
        )
    from core.foundation import atomic_json
    for record in records:
        record["sha256"] = file_sha256(base / record["image"])
        record["science_acceptance"] = "NOT_ACCEPTED"
    atomic_json(destination / "index.json", records)
    return records


def verify_export_manifest(filename, *, schema):
    """Blender-safe integrity reader for sealed exported actual-data products.

    Numeric semantics are authored by the shared dataset contract. This reader
    verifies their sealed bytes without requiring scientific Python in Blender.
    """
    filename = Path(filename).resolve()
    before = file_sha256(filename)
    record = blender_json.loads(filename.read_text())
    if record.get('schema') != schema:
        raise ValueError('Unsupported export manifest schema')
    if not record.get('publication_contract'):
        # Legacy scene imports stay identifiable; no acceptance inferred.
        return {**record, 'integrity_status': 'LEGACY_UNSEALED'}
    seal = filename.with_name(filename.name + '.sha256')
    if not seal.is_file() or seal.read_text().strip() != before:
        raise ValueError('Export manifest completion seal missing or changed')
    root = filename.parent
    names = set()
    for artifact in record['artifacts']:
        relative = Path(artifact['path'])
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Export artifact escapes the product directory')
        if artifact['path'] in names:
            raise ValueError('Duplicate export artifact')
        names.add(artifact['path'])
        if not path.is_file() or file_sha256(path) != artifact['sha256']:
            raise ValueError('Export artifact missing or changed: ' + artifact['path'])
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and
              p not in (filename, seal) and not any(v.startswith('.') for v in p.relative_to(root).parts)}
    if actual != names:
        raise ValueError('Export artifact inventory changed')
    for output in record.get('outputs', []):
        for product in output.get('numeric_products', {}).values():
            directory = root / product['directory']
            if not directory.resolve().is_relative_to(root):
                raise ValueError('Export numeric product escapes its directory')
            contract_path = directory / 'contract.json'
            if file_sha256(contract_path) != (directory / 'contract.sha256').read_text().strip():
                raise ValueError('Export numeric contract seal mismatch')
            contract = blender_json.loads(contract_path.read_text())
            if contract != product['array_contract'] or file_sha256(directory / 'arrays.npz') != contract['sha256']:
                raise ValueError('Export numeric product contract mismatch')
    if file_sha256(filename) != before:
        raise ValueError('Export manifest changed while reading')
    return {**record, 'integrity_status': 'INTEGRITY_VERIFIED'}


def publish_scene_product(manifest_path, blend_path, builder, *, input_schema, algorithm, render=False,
                          manifest_verifier=None, render_artifacts=None):
    """Retain a complete scene generation before exposing its public aliases."""
    import tempfile
    import uuid
    from core.native import native_run_lock
    from core.foundation import atomic_copy, atomic_json, atomic_text, publish_directory
    manifest_path, blend_path = Path(manifest_path).resolve(), Path(blend_path).resolve()
    pointer = blend_path.with_name(blend_path.name + '.publication.json')
    verify_source = manifest_verifier or (lambda path: verify_export_manifest(path, schema=input_schema))
    expected_renders = list(render_artifacts) if render_artifacts is not None else [blend_path.with_suffix('.png').name]
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in expected_renders):
        raise ValueError('Scene render artifact escapes its product')
    if blend_path.suffix != '.blend': raise ValueError('Scene destination must have a .blend suffix')
    with native_run_lock(blend_path, operation='standalone Blender scene publication'):
        if blend_path.exists() or pointer.exists() or (render and any((blend_path.parent / name).exists() for name in expected_renders)):
            raise FileExistsError('Keep completed/pending scene evidence; choose a fresh destination')
        source = verify_source(manifest_path)
        source_hash = file_sha256(manifest_path)
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.scene-stage-', dir=blend_path.parent))
        archive = blend_path.parent / '.scene_history' / uuid.uuid4().hex
        try:
            staged_blend = stage / blend_path.name
            builder(manifest_path, staged_blend, render)
            if not staged_blend.is_file():
                raise FileNotFoundError('Blender did not save the staged scene')
            if render and any(not (stage / name).is_file() for name in expected_renders):
                raise FileNotFoundError('Blender did not save the requested render products')
            if (verify_source(manifest_path) != source or
                    file_sha256(manifest_path) != source_hash):
                raise ValueError('Scene source changed during construction')
            aliases = [p.relative_to(stage).as_posix() for p in sorted(stage.rglob('*')) if p.is_file()]
            atomic_copy(manifest_path, stage / 'source_manifest.json')
            artifacts = [{'path': p.relative_to(stage).as_posix(), 'sha256': file_sha256(p)}
                         for p in sorted(stage.rglob('*')) if p.is_file()]
            record = {'schema': 'blender-scene-product-v1', 'publication_contract': 'blender-scene-publication-v1',
                'science_acceptance': 'NOT_ACCEPTED', 'source_manifest': {'path': str(manifest_path), 'sha256': source_hash},
                'source_integrity_status': source['integrity_status'],
                'algorithm_sha256': file_sha256(algorithm), 'publisher_sha256': file_sha256(__file__),
                'public_blend_name': blend_path.name, 'public_aliases': aliases,
                'render_artifacts': expected_renders if render else [], 'render_requested': bool(render), 'artifacts': artifacts}
            atomic_json(stage / 'scene_manifest.json', record)
            atomic_text(stage / 'scene_manifest.json.sha256', file_sha256(stage / 'scene_manifest.json') + '\n')
            verify_export_manifest(stage / 'scene_manifest.json', schema=record['schema'])
            archive.parent.mkdir(exist_ok=True)
            publish_directory(stage, archive)
            pending = {'schema': 'blender-scene-publication-v1', 'state': 'PENDING',
                'archive': archive.relative_to(blend_path.parent).as_posix(),
                'manifest_sha256': file_sha256(archive / 'scene_manifest.json'),
                'aliases': aliases, 'science_acceptance': 'NOT_ACCEPTED'}
            atomic_json(pointer, pending)
            recover_scene_publication(blend_path, archive)
            return blend_path
        except BaseException as exc:
            failure = stage / 'failure.json' if stage.exists() else archive.with_name(archive.name + '.failure.json')
            atomic_json(failure, {'status': 'FAIL', 'science_acceptance': 'NOT_ACCEPTED',
                'failure': {'type': type(exc).__name__, 'message': str(exc)}})
            raise


def _scene_publication_plan(blend_path):
    blend_path = Path(blend_path).resolve()
    pointer = blend_path.with_name(blend_path.name + '.publication.json')
    before = pointer.read_bytes()
    publication = blender_json.loads(before)
    if publication.get('schema') != 'blender-scene-publication-v1' or publication.get('state') not in ('PENDING', 'COMPLETE'):
        raise ValueError('Scene publication state is invalid')
    relative = Path(publication['archive'])
    archive = blend_path.parent / relative
    if relative.is_absolute() or '..' in relative.parts or not archive.resolve().is_relative_to(blend_path.parent / '.scene_history'):
        raise ValueError('Scene archive escapes its history directory')
    manifest = archive / 'scene_manifest.json'
    if file_sha256(manifest) != publication['manifest_sha256']:
        raise ValueError('Scene archive manifest changed')
    record = verify_export_manifest(manifest, schema='blender-scene-product-v1')
    artifacts = {r['path']: r['sha256'] for r in record['artifacts']}
    aliases = publication['aliases']
    expected_aliases = record['public_aliases']
    if any(name not in expected_aliases for name in record['render_artifacts']):
        raise ValueError('Scene render artifacts are missing from its sealed aliases')
    if record.get('public_blend_name') != blend_path.name or aliases != expected_aliases:
        raise ValueError('Scene alias inventory does not match its sealed generation')
    if not aliases or len(aliases) != len(set(aliases)) or blend_path.name not in aliases:
        raise ValueError('Scene public alias inventory is invalid')
    for name in aliases:
        relative_alias = Path(name)
        if (relative_alias.is_absolute() or '..' in relative_alias.parts or
                any(part.startswith('.') for part in relative_alias.parts) or name not in artifacts):
            raise ValueError('Scene public alias escapes its product')
    return blend_path, pointer, before, publication, archive, artifacts


def verify_scene_publication(blend_path):
    """Reject pending scene writes, corrupt archives and changed public aliases."""
    blend_path = Path(blend_path).resolve()
    pointer = blend_path.with_name(blend_path.name + '.publication.json')
    before = pointer.read_bytes()
    portable = blender_json.loads(before)
    if portable.get('schema') == 'blender-portable-scene-publication-v1':
        if portable.get('state') != 'COMPLETE' or portable.get('public_blend_name') != blend_path.name:
            raise ValueError('Portable scene publication is incomplete or names another scene')
        artifacts = portable.get('artifacts', {})
        aliases = portable.get('aliases', [])
        if not aliases or blend_path.name not in aliases or set(aliases) != set(artifacts):
            raise ValueError('Portable scene artifact inventory is invalid')
        for name in aliases:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or any(p.startswith('.') for p in relative.parts):
                raise ValueError('Portable scene artifact escapes its publication')
            destination = blend_path.parent / relative
            if destination.is_symlink() or file_sha256(destination) != artifacts[name]:
                raise ValueError('Portable scene artifact changed: ' + name)
        if pointer.read_bytes() != before:
            raise ValueError('Portable scene publication changed while reading')
        return portable
    blend_path, pointer, before, publication, archive, artifacts = _scene_publication_plan(blend_path)
    if publication['state'] != 'COMPLETE': raise ValueError('Scene publication is incomplete')
    for name in publication['aliases']:
        destination = blend_path.parent / name
        if destination.is_symlink() or file_sha256(destination) != artifacts[name]:
            raise ValueError('Scene public alias changed: ' + name)
    if pointer.read_bytes() != before:
        raise ValueError('Scene publication changed while reading')
    return publication


def recover_scene_publication(blend_path, generation=None):
    """Replay a sealed pending scene without rebuilding or rendering."""
    from core.native import native_run_lock
    from core.foundation import atomic_copy, atomic_json
    with native_run_lock(blend_path, operation='standalone scene publication recovery'):
        blend_path, pointer, before, publication, archive, artifacts = _scene_publication_plan(blend_path)
        if generation is not None and Path(generation).resolve() != archive.resolve():
            raise ValueError('Selected scene generation differs from the pending publication')
        if publication['state'] == 'COMPLETE': return verify_scene_publication(blend_path)
        for name in publication['aliases']:
            destination = blend_path.parent / name
            if destination.is_symlink() or (destination.exists() and
                    (not destination.is_file() or file_sha256(destination) != artifacts[name])):
                raise ValueError('Scene alias changed outside its publication')
        if pointer.read_bytes() != before:
            raise ValueError('Scene publication changed before replay')
        for name in publication['aliases']:
            destination = blend_path.parent / name
            if not destination.exists(): atomic_copy(archive / name, destination)
        atomic_json(pointer, {**publication, 'state': 'COMPLETE'})
        return verify_scene_publication(blend_path)


def verify_combined_scene_manifest(filename):
    """Verify portable species timelines and measured figures as one scene input."""
    from blender import verify_timeline_assets
    filename = Path(filename).resolve()
    before = file_sha256(filename)
    record = blender_json.loads(filename.read_text())
    if record.get('schema_version') != 1 or 'dm' not in record.get('species_manifests', {}):
        raise ValueError('Combined scene requires a supported DM timeline manifest')
    sealed = 'species_manifest_sha256' in record
    seal = filename.with_name(filename.name + '.sha256')
    if sealed and (not seal.is_file() or seal.read_text().strip() != before):
        raise ValueError('Combined scene completion seal missing or changed')
    for species, name in record['species_manifests'].items():
        path = filename.parent / name
        if not path.resolve().is_relative_to(filename.parent):
            raise ValueError('Combined timeline escapes its product directory')
        if sealed and file_sha256(path) != record['species_manifest_sha256'][species]:
            raise ValueError('Combined species timeline changed: ' + species)
        verify_timeline_assets(path)
    for figure in record.get('figure_gallery', []):
        path = filename.parent / figure['path']
        if not path.resolve().is_relative_to(filename.parent) or file_sha256(path) != figure['sha256']:
            raise ValueError('Combined scene measured figure missing or changed')
    for attachment in record.get('stream_timelines',[]):
        verify_combined_stream_attachment(filename.parent,record,attachment)
    for key in ['measured_display','local_impact_views','native_selections']:
        if key in record:
            relative=Path(record[key]['path']);path=(filename.parent/relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(filename.parent) or file_sha256(path)!=record[key]['sha256']:raise ValueError('Combined measured overlay changed or escaped')
    for binding in record.get('scientific_figure_bindings',[]):
        from core.foundation import canonical_digest
        if canonical_digest(binding['context'])!=binding['context_sha256']:raise ValueError('Combined scene scientific context changed')
        if not any(figure.get('scientific_context')==binding['context'] and figure.get('scientific_context_sha256')==binding['context_sha256'] for figure in record.get('figure_gallery',[])):raise ValueError('Combined scene scientific binding has no included measured image')
    shapes=record.get('radial_density_shape_profiles',[])
    if not shapes and record.get('radial_density_shapes'):
        shapes=[record['radial_density_shapes']]
    for shape in shapes:
        path=filename.parent/shape['report']
        if not path.resolve().is_relative_to(filename.parent) or file_sha256(path)!=shape['sha256']:
            raise ValueError('Combined scene radial shape report changed')
        parent=path.parent/'parent.npz'
        if 'parent_sha256' in shape and file_sha256(parent)!=shape['parent_sha256']:
            raise ValueError('Combined scene radial shape parent changed')
        profile=blender_json.loads(path.read_text())
        if profile.get('array_contract') is not None:
            from core.foundation import verify_numpy_container
            verify_numpy_container(parent,profile['array_contract'])
    if file_sha256(filename) != before:
        raise ValueError('Combined scene input changed while reading')
    return {**record, 'integrity_status': 'INTEGRITY_VERIFIED' if sealed else 'LEGACY_UNSEALED'}


def verify_combined_stream_attachment(root,combined,reference):
    """Portable runtime check of the original saved-clock/unit attachment."""
    def owned(name):
        relative=Path(name);path=(root/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root) or not path.is_file():raise ValueError('Stream attachment escapes combined export')
        return path
    path=owned(reference['path'])
    if file_sha256(path)!=reference['sha256']:raise ValueError('Combined stream attachment changed')
    record=blender_json.loads(path.read_text());inspector=path.parent/'inspector.json';archive=path.parent/'inspector.npz'
    if record.get('schema')!='combined-native-stream-attachment-v1' or file_sha256(inspector)!=record['inspector_sha256'] or file_sha256(archive)!=record['arrays_sha256']:raise ValueError('Original stream inspector bundle changed')
    original=blender_json.loads(inspector.read_text())
    if original['arrays_sha256']!=record['arrays_sha256'] or original['namespace']!=record['namespace'] or original['epochs']!=record['epochs'] or original['saved_times_myr']!=record['saved_times_myr']:raise ValueError('Stream attachment changes original saved state')
    if record['native_run_directory']!=combined['run_dir'] or record['hubble_parameter']!=combined['hubble_parameter']:raise ValueError('Stream attachment changes native run or cosmology')
    timeline=verify_timeline_assets(owned(combined['species_manifests']['dm']));stride=combined['frames_per_snapshot'];saved=timeline['snapshots'];end=(len(saved)-1)*stride
    lookup={row['snapshot']:(ordinal,row) for ordinal,row in enumerate(saved)};frames=[];scales=[]
    for epoch,time in zip(record['epochs'],record['saved_times_myr'],strict=True):
        if epoch.get('canonical_time')!='scale_factor':raise ValueError('Cosmological stream clock required')
        ordinal,row=lookup[epoch['snapshot_storage_index']]
        if epoch['scale_factor']!=row['scale_factor'] or time!=1000*row['cosmic_time_gyr']:raise ValueError('Combined stream saved clock differs')
        frames.append(1 if ordinal==0 else ordinal*stride);scales.append(combined['hubble_parameter']/(1000*row['scale_factor']))
    if frames!=record['scene_frames'] or scales!=record['coordinate_scales'] or end!=record['scene_frame_end']:raise ValueError('Combined stream display mapping changed')
    saved_stream_frame_schedule(record['saved_times_myr'],scene_frames=frames,scene_frame_end=end)
    return record


def build_local_impact_views(bpy,record):
    """Static local-chart inspection of actual reconstructed event positions."""
    from mathutils import Vector
    if record.get('schema')!='native-scene-local-impacts-v1':raise ValueError('Original local reconstructed impact view required')
    scenes=[]
    for view in record['views']:
        scene=bpy.data.scenes.new('Impact | '+view['event_id']);scene.render.engine='CYCLES';disable_cycles_denoising(scene);scene.cycles.samples=8
        scene.render.resolution_x=1280;scene.render.resolution_y=720
        points=[Vector(point) for point in view['positions_proper_kpc']];center=(points[0]+points[1])/2
        radius=float(view['marker_radius_kpc']);span=max((points[0]-points[1]).length,10*radius)
        if not math.isfinite(radius) or radius<=0:raise ValueError('Positive symbolic impact marker radius required')
        material,_=_emission(bpy,'Reconstructed supported impact' if view['geometry_supported'] else 'Uncertified impact candidate',
            (1.,.3,.1,1.) if view['geometry_supported'] else (.5,.5,.5,1.))
        for role,position,identity in zip(['stream','perturber'],points,[view['stream_identity'],view['perturber_identity']],strict=True):
            mesh=bpy.data.meshes.new(role+' reconstructed marker');mesh.from_pydata([(radius,0,0),(-radius,0,0),(0,radius,0),(0,-radius,0),(0,0,radius),(0,0,-radius)],[],
                [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)])
            obj=bpy.data.objects.new(role+' at reconstructed event',mesh);scene.collection.objects.link(obj);obj.location=position;mesh.materials.append(material)
            obj['source_identity_json']=blender_json.dumps(identity);obj['event_time_myr']=view['time_myr'];obj['coordinate_units']='proper kpc'
            obj['position_meaning']=view['position_meaning'];obj['geometry_supported']=view['geometry_supported']
        data=bpy.data.cameras.new('Local reconstructed event camera');data.type='ORTHO';data.ortho_scale=2.5*span
        camera=bpy.data.objects.new(data.name,data);scene.collection.objects.link(camera);camera.location=center+Vector((0,-3*span,2*span))
        camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler();data.clip_start=span*1e-5;data.clip_end=span*20;scene.camera=camera
        scene['local_impact_context_json']=blender_json.dumps(view,sort_keys=True);scene['section_status']='RECONSTRUCTED_SUPPORTED' if view['geometry_supported'] else 'RECONSTRUCTED_UNCERTIFIED'
        caption=_text(bpy,scene,'Original reconstructed event disclosure',
            'RECONSTRUCTED EVENT — t=%.9g Myr\nLocal proper kpc; not a saved snapshot\nGeometry support: %s'%(view['time_myr'],view['geometry_supported']),
            (0,0,0),span*.05,material)
        caption.parent=camera;caption.location=(-span,span*.55,-span)
        scenes.append(scene.name)
    return scenes


def load_native_scene_selections(bpy,manifest_path,*,scene_frame_end):
    """Source-bound stellar highlights and measured contamination spheres."""
    path=Path(manifest_path).resolve();record=blender_json.loads(path.read_text());scene=bpy.context.scene
    if record.get('schema')!='native-scene-selections-v1' or scene.frame_end!=scene_frame_end:raise ValueError('Original native selection scene clock required')
    root=path.parent
    for row in record['progenitor_overlays']:
        relative=Path(row['path'])
        # Paths were authored relative to the combined export root. The whole
        # native_selections directory is copied together into the saved scene.
        if relative.parts[0]!='native_selections' or '..' in relative.parts:raise ValueError('Progenitor selection escapes its portable package')
        inspector=root/Path(*relative.parts[1:])
        if file_sha256(inspector)!=row['sha256'] or file_sha256(inspector.with_name('inspector.npz'))!=row['arrays_sha256']:raise ValueError('Original progenitor selection changed')
        collection=load_stream_inspector_assets(bpy,inspector,maximum_points=row['maximum_points'],marker_radius_kpc=row['marker_radius_kpc'])
        for obj in collection.objects:
            obj.scale=(row['coordinate_scale'],)*3;obj.location=row['origin_comoving_mpc_h']
            key_object_visibility(obj,row['first_frame'],row['last_frame'],scene_frame_end)
            archive='//'+str(relative.with_name('inspector.npz'))
            obj['exact_identity_archive']=archive
            original=blender_json.loads(inspector.read_text())
            obj['source_product']='//'+str(relative.parent.parent/Path(original['product']).name)
            obj['native_overlay_metadata_json']=blender_json.dumps({'role':row['role'],'namespace':row['namespace'],'epoch':row['epoch'],
                'progenitor_keys':row['progenitor_keys'],'units':'global comoving Mpc/h display; source host-relative proper kpc',
                'exact_identity_archive':archive,'source_sha256':row['sha256'],'particle_count':row['maximum_points'],
                'periodic_display_policy':original.get('periodic_display_policy')},sort_keys=True)
    collection=bpy.data.collections.new('Measured contamination support apertures');scene.collection.children.link(collection)
    for row in record['contamination_apertures']:
        radius=row['radius_comoving_mpc_h'];data=bpy.data.curves.new('Actual native support aperture','CURVE');data.dimensions='3D';data.bevel_depth=radius*.002;data.bevel_resolution=1
        for first,second in [(0,1),(0,2),(1,2)]:
            spline=data.splines.new('POLY');spline.points.add(64)
            for index,point in enumerate(spline.points):
                angle=2*math.pi*index/64;xyz=[0.,0.,0.];xyz[first]=radius*math.cos(angle);xyz[second]=radius*math.sin(angle);point.co=(*xyz,1.)
        color=(.2,1.,.2,1.) if row['status']=='PASS' else (1.,.1,.1,1.) if row['status']=='FAIL' else (.5,.5,.5,1.)
        material,_=_emission(bpy,'Contamination '+row['status'],color);data.materials.append(material)
        for image in row['periodic_display_images']:
            obj=bpy.data.objects.new('Contamination aperture '+str(row['selection'])+' image '+str(image),data);collection.objects.link(obj)
            obj.location=tuple(center+offset*row['box_comoving_mpc_h'] for center,offset in zip(row['center_comoving_mpc_h'],image,strict=True))
            key_object_visibility(obj,row['first_frame'],row['last_frame'],scene_frame_end)
            obj['native_overlay_metadata_json']=blender_json.dumps({'role':'measured spherical contamination support aperture','units':'comoving Mpc/h',**row,'display_periodic_image':image},sort_keys=True)
    scene['native_selection_scope']=record['scope'];scene['native_selection_unresolved_json']=blender_json.dumps(record['unresolved'],sort_keys=True)
    _native_host_focus_scenes(bpy,scene,record)
    return collection


def _native_host_focus_scenes(bpy,overview,record):
    """Frame real late-time recovered support; never invent a host or trajectory.

    These are camera views of the same native meshes and clock. The final
    measured support sets a static framing region, not a followed worldline.
    """
    from mathutils import Vector
    rows=[row for row in record['contamination_apertures'] if type(row['selection']) is int]
    if not rows:
        overview['native_focus_status']='NO_MEASURED_RECOVERED_HOST_SUPPORT'
        return
    latest=max(row['first_frame'] for row in rows)
    reference=max((row for row in rows if row['first_frame']==latest),key=lambda row:row['radius_comoving_mpc_h'])
    center=Vector(reference['center_comoving_mpc_h']);radius=reference['radius_comoving_mpc_h']
    def focused_collection(original,fraction):
        result=bpy.data.collections.new(original.name+' native focus')
        for obj in original.objects:
            if obj.type=='FONT' and obj.parent is overview.camera:continue
            particle=any(mod.type=='NODES' and mod.name=='Particle rendering' for mod in obj.modifiers)
            clone=obj.copy() if particle else obj
            if particle:
                for mod in clone.modifiers:
                    if mod.type=='NODES' and mod.name=='Particle rendering':
                        mod.node_group=mod.node_group.copy()
                        for node in mod.node_group.nodes:
                            if node.bl_idname=='GeometryNodeMeshIcoSphere':
                                node.inputs['Radius'].default_value*=fraction
                clone['native_focus_display_glyph_fraction']=fraction
            result.objects.link(clone)
        for child in original.children:result.children.link(focused_collection(child,fraction))
        return result
    for name,factor in [('Native | Host close-up',1.5),('Native | Inner halo close-up',.4)]:
        scene=overview.copy();scene.name=name
        fraction=radius*factor/reference['box_comoving_mpc_h']
        for child in list(scene.collection.children):
            scene.collection.children.unlink(child)
            scene.collection.children.link(focused_collection(child,fraction))
        camera=bpy.data.objects.new(name+' camera',overview.camera.data.copy())
        scene.collection.objects.link(camera)
        direction=Vector((1.2,-1.6,1.0)).normalized();span=radius*factor
        camera.location=center+direction*span*3
        camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.type='PERSP';camera.data.lens=50.
        camera.data.clip_start=max(1e-8,span*1e-5);camera.data.clip_end=reference['box_comoving_mpc_h']*10
        scene.camera=camera
        # The wide-view HUD has a different camera parent and angular scale.
        # Keep its metadata in the scene and omit those off-camera text objects.
        for obj in list(scene.collection.objects):
            if obj.type=='FONT' and obj.parent is overview.camera:scene.collection.objects.unlink(obj)
        scene['native_focus_reference_json']=blender_json.dumps(reference,sort_keys=True)
        scene['native_focus_status']='MEASURED_LATE_EPOCH_SUPPORT'
        scene['native_focus_scope']='Static late-epoch aperture framing of original native particles; not a measured camera or reconstructed host worldline'
        scene['display_radius_fraction']=factor
        scene['glyph_scope']='Display glyphs scale with view span; original shared native positions, IDs, ages, temperatures and shape keys are unchanged'


def verify_image_publication(output, *, _allow_pending=False):
    """Stdlib image consumer; verify sealed canonical numeric inputs and pixels."""
    output = Path(output).resolve()
    pointer = output.with_name(output.name + '.publication.json')
    before = pointer.read_bytes()
    publication = blender_json.loads(before)
    states = ('PENDING', 'COMPLETE') if _allow_pending else ('COMPLETE',)
    if publication.get('schema') != 'scientific-image-publication-v1' or publication.get('state') not in states:
        raise ValueError('Image publication is incomplete')
    archive_name = publication['archive']
    if Path(archive_name).name != archive_name or archive_name != output.name + '.product':
        raise ValueError('Image archive location is invalid')
    archive = output.parent / archive_name
    if not archive.resolve().is_relative_to(output.parent): raise ValueError('Image archive escapes its root')
    if publication['image'] != output.name or publication['metadata'] != output.with_suffix('.json').name:
        raise ValueError('Image public aliases do not match their publication')
    metadata = archive / publication['metadata']
    if file_sha256(metadata) != publication['metadata_sha256']:
        raise ValueError('Image metadata generation changed')
    record = verify_export_manifest(metadata, schema='scientific-image-product-v1')
    if publication['state'] == 'COMPLETE' and (file_sha256(output) != record['output_sha256'] or
            file_sha256(output.with_suffix('.json')) != publication['metadata_sha256']):
        raise ValueError('Image public aliases changed')
    product = record['numeric_data']
    relative = Path(product['directory'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Image numeric data escapes its product')
    directory = archive / relative
    contract_path = directory / 'contract.json'
    if file_sha256(contract_path) != (directory / 'contract.sha256').read_text().strip():
        raise ValueError('Image numeric contract seal mismatch')
    contract = blender_json.loads(contract_path.read_text())
    if contract != product['array_contract'] or file_sha256(directory / 'arrays.npz') != contract['sha256']:
        raise ValueError('Image numeric archive changed')
    if pointer.read_bytes() != before: raise ValueError('Image publication changed while reading')
    return {**record, 'integrity_status': 'INTEGRITY_VERIFIED'}


def recover_image_publication(output, generation=None):
    """Replay retained image aliases without recalculating the model or pixels."""
    from core.native import native_run_lock
    from core.foundation import atomic_copy, atomic_json
    output = Path(output).resolve()
    with native_run_lock(output, operation='scientific image publication recovery'):
        record = verify_image_publication(output, _allow_pending=True)
        pointer = output.with_name(output.name + '.publication.json')
        before = pointer.read_bytes()
        publication = blender_json.loads(before)
        archive = output.parent / publication['archive']
        if generation is not None and Path(generation).resolve() != archive.resolve():
            raise ValueError('Image replay generation differs from its publication')
        if publication['state'] == 'COMPLETE': return verify_image_publication(output)
        aliases = [(output, record['output_sha256']),
                   (output.with_suffix('.json'), publication['metadata_sha256'])]
        for destination, expected in aliases:
            if destination.is_symlink() or (destination.exists() and
                    (not destination.is_file() or file_sha256(destination) != expected)):
                raise ValueError('Image alias changed outside its publication')
        if pointer.read_bytes() != before: raise ValueError('Image publication changed before replay')
        for destination, expected in aliases:
            if not destination.exists(): atomic_copy(archive / destination.name, destination)
        atomic_json(pointer, {**publication, 'state': 'COMPLETE'})
        return verify_image_publication(output)

# ========================================================================
# build_blender_universe

# Workflow and units
# Assembles exported particle layers, independent catalogue detections, HUD and perspective/dome cameras.
# One scene unit is one comoving Mpc/h; gas colours encode log10(T/K) and star colours encode saved ages.
# Linear display poses are not simulation time steps. Catalogue apertures never replace nonspherical particle distributions, and no nearest-neighbour halo tracks are invented.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
import json as build_blender_universe_json
from typing import List, Optional


# Source-tree execution and portable exported scenes share these imports.
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1] / "core")]


# -----------------------------------------------------------------------------
# Blender Scene Collection Helpers
# -----------------------------------------------------------------------------

def get_or_create_collection(bpy, parent_col, name: str):
    """Create or retrieve a structured Blender scene collection."""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent_col.children.link(col)
    return col


# -----------------------------------------------------------------------------
# Mesh & Shape Key Importers (Self-contained)
# -----------------------------------------------------------------------------

def import_particle_mesh(bpy, ply_path: Path, species: str, box_size: float, index: int):
    """Import a binary PLY particle snapshot into Blender."""
    ply_path = Path(ply_path).expanduser().resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"Particle sample PLY missing: {ply_path}")

    obj = read_particle_mesh(bpy, ply_path)
    if obj is None or len(obj.data.vertices) == 0:
        raise RuntimeError(f"PLY imported with zero vertices: {ply_path}")

    obj.name = f"DarkMatterSim_{species}_output_{index:04d}"
    obj.data.name = f"{obj.name}_mesh"
    return obj


def add_interval_shape_key(bpy, start_obj, target_obj, first_frame: int, last_frame: int, keep_target: bool = False):
    """Animate ID-aligned particle meshes along their periodic-unwrapped paths."""
    start_verts = start_obj.data.vertices
    target_verts = target_obj.data.vertices
    if len(start_verts) != len(target_verts):
        raise RuntimeError("Endpoint PLYs have different vertex counts; cannot interpolate")

    start_coords = [v.co.copy() for v in start_verts]
    target_coords = [v.co.copy() for v in target_verts]
    if not start_coords:
        return None

    def assign_motion(obj, basis_coords, motion_coords):
        for v, c in zip(obj.data.vertices, basis_coords):
            v.co = c
        obj.shape_key_add(name="Basis", from_mix=False)
        motion = obj.shape_key_add(name="Motion", from_mix=False)
        for point, c in zip(motion.data, motion_coords):
            point.co = c
        motion.value = 0.0
        motion.keyframe_insert(data_path="value", frame=int(first_frame))
        motion.value = 1.0
        motion.keyframe_insert(data_path="value", frame=int(last_frame))

        if obj.data.shape_keys and obj.data.shape_keys.animation_data and obj.data.shape_keys.animation_data.action:
            for curve in action_fcurves(obj.data.shape_keys.animation_data):
                for kf in curve.keyframe_points:
                    kf.interpolation = "LINEAR"

    assign_motion(start_obj, start_coords, target_coords)

    if keep_target:
        assign_motion(target_obj, start_coords, target_coords)
        return target_obj

    mesh_to_delete = target_obj.data
    bpy.data.objects.remove(target_obj, do_unlink=True)
    if mesh_to_delete.users == 0:
        bpy.data.meshes.remove(mesh_to_delete)
    return None


def key_object_visibility(obj, first_frame: int, last_frame: int, total_frames: int):
    """Control object visibility strictly over its active lifespan."""
    for prop in ("hide_render", "hide_viewport"):
        if first_frame == 1:
            setattr(obj, prop, False)
            obj.keyframe_insert(data_path=prop, frame=1)
        else:
            setattr(obj, prop, True)
            obj.keyframe_insert(data_path=prop, frame=1)
            setattr(obj, prop, False)
            obj.keyframe_insert(data_path=prop, frame=int(first_frame))

        if last_frame < total_frames:
            setattr(obj, prop, True)
            obj.keyframe_insert(data_path=prop, frame=int(last_frame) + 1)

        if obj.animation_data and obj.animation_data.action:
            for curve in action_fcurves(obj.animation_data):
                if curve.data_path == prop:
                    for kf in curve.keyframe_points:
                        kf.interpolation = "CONSTANT"


# -----------------------------------------------------------------------------
# Geometry Nodes & Physically Calibrated Shaders
# -----------------------------------------------------------------------------

def setup_particle_geometry_nodes(bpy, obj, species: str, box_size: float, *, crop_periodic=True):
    """Render measured particle coordinates with the shared explicit-bin shader.

    Glyph radii use the caller's scene units (Mpc/h for cosmology, kpc for
    isolated mergers). The crop applies only to a periodic cosmological box.
    Per-point attributes select material bins before instancing, avoiding
    unsupported shader attribute lookup on instanced point geometry.
    """
    from blender import _make_geometry_nodes, _blender_emission_material as _emission_material
    palette = {"dm": (0.12, 0.42, 1.0, 1.0), "gas": (0.18, 0.70, 0.82, 1.0),
               "stars": (1.0, 0.82, 0.50, 1.0), "black_holes": (1.0, 0.28, 0.08, 1.0)}
    obj.modifiers.clear()
    obj.data.materials.clear()
    obj.data.materials.append(_emission_material(bpy, "Particle " + species, palette[species], 1.2))
    _make_geometry_nodes(obj, species, box_size, crop_periodic=crop_periodic)


# -----------------------------------------------------------------------------
# Halo-Radius Shells, Starlight Cores, & Catalogue Track Guides
# -----------------------------------------------------------------------------


def make_selectable_halo_tracks(bpy, target_col, timelines, box: float, frames_per_snap: int, total_frame_end: int) -> int:
    """Create catalogue inspection markers at saved epochs, without inferred tracks.

    Catalogue row numbers are local to each output and cannot establish halo
    identity across time. Each marker therefore has one detection and one saved
    epoch. R200c wire spheres show the SO measurement aperture, not a spherical
    halo shape. They are excluded from rendered images; the actual particle
    distribution supplies halo morphology, tidal debris and the cosmic web.
    """
    import bmesh
    mesh = bpy.data.meshes.new("SO_aperture_inspection_wire")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=1.)
    bm.to_mesh(mesh)
    bm.free()
    count = 0
    for index, snapshot in enumerate(timelines["dm"]["snapshots"]):
        frame = 1 if index == 0 else index*int(frames_per_snap)
        for kind, layer in snapshot.get("halo_overlays", {}).items():
            for item in layer.get("items", []):
                position = tuple(float(v) for v in item["position_mpc_h"])
                if len(position) != 3 or not all(math.isfinite(v) for v in position):
                    raise ValueError("Catalogue marker has invalid comoving coordinates")
                radius = item.get("r200c_mpc_h")
                radius_source = "R200c measurement aperture"
                if radius is None and kind == "subhalos":
                    radius = item.get("half_mass_radius_mpc_h")
                    radius_source = "SUBFIND bound half-mass radius"
                measured = radius is not None and math.isfinite(float(radius)) and float(radius) > 0
                identifier = str(item.get("id", "unknown"))
                label = f"{kind}:snap{snapshot['snapshot']}:id{identifier}"
                obj = bpy.data.objects.new(label, mesh if measured else None)
                target_col.objects.link(obj)
                obj.location = position
                obj.display_type = "WIRE"
                obj.hide_render = True
                if measured:
                    obj.scale = (float(radius),)*3
                else:
                    obj.empty_display_type = "PLAIN_AXES"
                    obj.empty_display_size = box*.002
                obj["dmsim_halo_id"] = identifier
                obj["dmsim_track_id"] = label
                obj["dmsim_track_label"] = kind + " · saved catalogue detection"
                obj["dmsim_finder"] = str(item.get("finder", kind))
                obj["dmsim_mass_msun_h"] = float(item["mass_msun_h"])
                obj["dmsim_mass_definition"] = str(item.get("mass_definition", "unspecified"))
                obj["dmsim_radius_source"] = radius_source if measured else "display axes; no measured radius"
                obj["dmsim_host_id"] = str(item.get("host_id", "unavailable"))
                obj["dmsim_parent_id"] = str(item.get("parent_id", "unavailable"))
                obj["dmsim_detection_json"] = build_blender_universe_json.dumps(item, separators=(",", ":"))
                obj["dmsim_merger_history_json"] = build_blender_universe_json.dumps(item.get("merger_history", {}), separators=(",", ":"))
                row = {"snapshot": int(snapshot["snapshot"]), "frame": frame,
                       "redshift": float(snapshot["redshift"]), "scale_factor": float(snapshot["scale_factor"]),
                       "position_mpc_h": position, "mass_msun_h": float(item["mass_msun_h"]),
                       "r200c_mpc_h": item.get("r200c_mpc_h"),
                       "half_mass_radius_mpc_h": item.get("half_mass_radius_mpc_h"), "detection_ids": [item]}
                obj["dmsim_history_json"] = build_blender_universe_json.dumps([row], separators=(",", ":"))
                # Only display the catalogue at its exact stored epoch. The
                # render flag remains True: halos are visualized by particles.
                for f, hidden in ((1, frame != 1), (frame, False), (frame+1, True)):
                    obj.hide_viewport = hidden
                    obj.keyframe_insert(data_path="hide_viewport", frame=f)
                if obj.animation_data and obj.animation_data.action:
                    for curve in action_fcurves(obj.animation_data):
                        for key in curve.keyframe_points:
                            key.interpolation = "CONSTANT"
                count += 1
                shape = item.get('radial_density_shapes')
                if shape is not None:
                    obj['dmsim_density_profile_json']=build_blender_universe_json.dumps(shape)
                    from mathutils import Matrix
                    if (shape['units'] != 'comoving Mpc/h' or shape['snapshot'] != snapshot['snapshot']
                            or shape['scale_factor'] != snapshot['scale_factor']
                            or shape['epoch_local_host_row'] != item['id'] or kind != 'halos'):
                        raise ValueError('Radial density shape has a mismatched native identity/epoch/unit')
                    for shell_index, shell in enumerate(shape['shells']):
                        if shell['status'] != 'MEASURED':
                            continue
                        ellipsoid = bpy.data.objects.new(label+f':density_shell{shell_index}',mesh)
                        target_col.objects.link(ellipsoid)
                        ellipsoid.location = position
                        ellipsoid.scale = shell['fitted_semiaxes_comoving_mpc_h']
                        ellipsoid.rotation_mode = 'QUATERNION'
                        ellipsoid.rotation_quaternion = Matrix(shell['axes_columns']).to_quaternion()
                        ellipsoid.display_type = 'WIRE'
                        ellipsoid.hide_render = True
                        for metadata_key in obj.keys():
                            ellipsoid[metadata_key]=obj[metadata_key]
                        ellipsoid['dmsim_halo_id'] = identifier
                        ellipsoid['dmsim_density_shape_json'] = build_blender_universe_json.dumps(shell)
                        ellipsoid['dmsim_shape_scope'] = 'Measured density moment; sampling-unresolved axes remain flagged; not potential shape'
                        ellipsoid['dmsim_science_acceptance'] = shape['science_acceptance']
                        for f, hidden in ((1, frame != 1), (frame, False), (frame+1, True)):
                            ellipsoid.hide_viewport = hidden
                            ellipsoid.keyframe_insert(data_path='hide_viewport',frame=f)
                        if ellipsoid.animation_data and ellipsoid.animation_data.action:
                            for curve in action_fcurves(ellipsoid.animation_data):
                                for key in curve.keyframe_points:
                                    key.interpolation = 'CONSTANT'
    return count


# -----------------------------------------------------------------------------
# Deep Space Lighting, World, & Post-Processing Compositor
# -----------------------------------------------------------------------------

def setup_world_and_compositor(scene, box_size: float):
    """Setup pitch-black cosmic void lighting and filmic bloom/glare."""
    import bpy

    # 1. Pitch-black cosmic void
    world = scene.world or bpy.data.worlds.new("Cosmic_Deep_Space")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.0002, 0.0004, 0.001, 1.0)
        bg.inputs["Strength"].default_value = 0.03

    # 2. Key and rim lights
    center = (box_size / 2.0, box_size / 2.0, box_size / 2.0)
    for name, offset, color, energy in [
        ("Cosmic_Key_Light", (1.8, -1.5, 1.8), (1.0, 0.94, 0.82), 2800.0),
        ("Cosmic_Rim_Light", (-1.7, 1.7, 1.4), (0.15, 0.55, 1.0), 3200.0),
    ]:
        light = bpy.data.objects.get(name)
        if not light:
            ldata = bpy.data.lights.new(name, "AREA")
            ldata.color = color
            ldata.energy = energy
            ldata.size = box_size * 1.5
            light = bpy.data.objects.new(name, ldata)
            scene.collection.objects.link(light)
        light.location = tuple(c + o * box_size for c, o in zip(center, offset))

    # Restrained glow supports point visibility without obscuring structures.
    modern = hasattr(scene, 'compositing_node_group')
    if modern:
        c_tree = bpy.data.node_groups.new('Cosmic_Compositor', 'CompositorNodeTree')
        c_tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        scene.compositing_node_group = c_tree
    else:
        scene.use_nodes = True
        c_tree = scene.node_tree
    c_nodes, c_links = c_tree.nodes, c_tree.links
    c_nodes.clear()

    rlayers = c_nodes.new("CompositorNodeRLayers")
    glare = c_nodes.new("CompositorNodeGlare")
    configure_glare(glare)
    # Blender 4.5+ uses sockets; Ubuntu 24.04's Blender 4.0 uses RNA properties.
    threshold = glare.inputs.get("Threshold")
    size = glare.inputs.get("Size")
    if threshold is not None:
        threshold.default_value = 4.0
    else:
        glare.threshold = 4.0
    if size is not None:
        size.default_value = 0.125
    else:
        glare.size = 6

    comp_out = c_nodes.new("NodeGroupOutput" if modern else "CompositorNodeComposite")
    c_links.new(rlayers.outputs["Image"], glare.inputs["Image"])
    c_links.new(glare.outputs["Image"], comp_out.inputs["Image"])


# -----------------------------------------------------------------------------
# Dynamic On-Screen Scientific HUD (Live Frame-by-Frame Update)
# -----------------------------------------------------------------------------

def setup_dynamic_hud(bpy, scene, camera, reference, box: float, frames_per_snap: int, total_frames: int):
    """Create an animated screen-space scientific HUD with live frame change handler."""
    snapshots = reference["snapshots"]

    hud_data = bpy.data.curves.new("Scientific_HUD_Curve", type="FONT")
    hud_data.size = box * 0.016
    hud_data.align_x = "LEFT"
    hud_data.align_y = "TOP"

    hud_obj = bpy.data.objects.new("Scientific_HUD_Overlay", hud_data)
    scene.collection.objects.link(hud_obj)
    hud_obj.parent = camera
    distance = box * 0.2
    width = 2 * distance * math.tan(camera.data.angle_x / 2)
    height = width * scene.render.resolution_y / scene.render.resolution_x
    hud_obj.location = (-width * 0.46, height * 0.44, -distance)
    hud_data.size = width * 0.010

    hud_obj.rotation_euler = (0, 0, 0)

    # Neutral scientific annotation material
    mat = bpy.data.materials.new("Mat_Scientific_Annotation")
    mat.use_nodes = True
    out = mat.node_tree.nodes.get("Material Output")
    emit = mat.node_tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.84, 0.81, 0.74, 1.0)
    emit.inputs["Strength"].default_value = 2.0
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    hud_obj.data.materials.append(mat)

    scene["dmsim_hud_epochs_json"] = build_blender_universe_json.dumps([
        {"frame": 1 if i == 0 else i*int(frames_per_snap),
         "snapshot": int(item["snapshot"]), "scale_factor": float(item["scale_factor"])}
        for i, item in enumerate(snapshots)
    ])
    scene["dmsim_box_mpc_h"] = float(box)
    from blender import update_scientific_hud
    update_scientific_hud(scene)
    return hud_obj


# -----------------------------------------------------------------------------
# Cameras: 65mm perspective Hero + 180° Equidistant Planetarium Fulldome
# -----------------------------------------------------------------------------

def configure_saved_universe_view(bpy, scene):
    """Open a measured epoch with a framed box, redshift markers and inspector."""
    import json
    from mathutils import Vector
    epochs = json.loads(scene.get('dmsim_hud_epochs_json', '[]'))
    if not epochs or not scene.get('dmsim_box_mpc_h') or scene.camera is None:
        return
    box = float(scene['dmsim_box_mpc_h'])
    for marker in list(scene.timeline_markers):
        if marker.name.startswith('Saved '):
            scene.timeline_markers.remove(marker)
    for epoch in epochs:
        redshift = max(0., 1/float(epoch['scale_factor'])-1)
        scene.timeline_markers.new(f'Saved {epoch["snapshot"]:03d} · z={redshift:.4f}', frame=int(epoch['frame']))
    initial = min(epochs, key=lambda row: abs(1/float(row['scale_factor'])-1-14.4))
    scene.frame_set(int(initial['frame']))
    camera = scene.camera
    camera.data.type = 'ORTHO'
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)
    camera.data.ortho_scale = box*1.70*max(1., aspect)
    camera.location = (box*1.8, -box*2., box*1.25)
    for material in bpy.data.materials:
        if material.name.startswith('Particle dm') and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == 'EMISSION':
                    node.inputs['Color'].default_value = (.10, .65, .89, 1.)
                    node.inputs['Strength'].default_value = 2.8
    for group in bpy.data.node_groups:
        if group.name.startswith('DarkMatterSim cinematic geometry'):
            for node in group.nodes:
                if node.bl_idname == 'GeometryNodeMeshIcoSphere':
                    node.inputs['Radius'].default_value = box*.00065
    if camera.parent is not None:
        camera.parent.location = (box/2,)*3
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            space.region_3d.view_perspective = 'CAMERA'
            space.region_3d.view_location = Vector((box/2,)*3)
            space.region_3d.view_distance = box*2
            space.shading.type = 'MATERIAL'
            space.overlay.show_floor = False
            space.overlay.show_axis_x = False
            space.overlay.show_axis_y = False
            space.show_region_ui = True
    candidates = [obj for obj in scene.objects
                  if f':snap{initial["snapshot"]}:' in obj.name and 'dmsim_halo_id' in obj
                  and not obj.hide_viewport]
    if candidates:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        selected = max(candidates, key=lambda obj: float(obj.get('dmsim_mass_msun_h', 0)))
        selected.select_set(True)
        bpy.context.view_layer.objects.active = selected
    scene['dmsim_initial_snapshot'] = int(initial['snapshot'])
    scene['dmsim_view_help'] = 'Timeline markers select saved redshifts. N panel > DarkMatterSim shows the selected halo. Numpad 0 switches camera and orbit views.'


def setup_cameras(bpy, scene, col_rig, box_size: float, target_cluster_pos: Optional[List[float]] = None, *, orbit=False):
    """Configure default 65mm perspective Hero Perspective Camera & 180° Equidistant Dome Preset."""
    orbit_empty = bpy.data.objects.new("Hero_Camera_Orbit_Target", None)
    col_rig.objects.link(orbit_empty)
    center = (box_size / 2.0, box_size / 2.0, box_size / 2.0)
    orbit_empty.location = center

    # 1. 65mm perspective Hero Camera
    cam_data = bpy.data.cameras.new("Cinematic_Hero_Camera_Data")
    cam_data.lens = 65.0
    cam_data.dof.use_dof = False
    cam_data.dof.focus_object = orbit_empty
    cam_data.dof.aperture_fstop = 3.5
    cam_data.clip_end = box_size * 10.0
    cam_data.clip_start = box_size * 1.0e-5

    camera = bpy.data.objects.new("Cinematic_Hero_Camera", cam_data)
    col_rig.objects.link(camera)
    camera.parent = orbit_empty
    # Fit the entire box's bounding sphere in the narrower image dimension.
    # A 65mm lens at the former distance cropped outer structures.
    half_horizontal = math.atan(cam_data.sensor_width / (2.0 * cam_data.lens))
    half_vertical = math.atan(math.tan(half_horizontal) * scene.render.resolution_y / scene.render.resolution_x)
    distance = .95 * box_size / math.sin(min(half_horizontal, half_vertical))
    direction = (1.35, -1.55, 1.1)
    norm = math.sqrt(sum(v*v for v in direction))
    camera.location = tuple(v * distance / norm for v in direction)

    track_con = camera.constraints.new("TRACK_TO")
    track_con.target = orbit_empty
    track_con.track_axis = "TRACK_NEGATIVE_Z"
    track_con.up_axis = "UP_Y"

    if orbit and target_cluster_pos is not None:
        orbit_empty.location = center
        orbit_empty.keyframe_insert(data_path="location", frame=1)
        # Drift smoothly into cluster core
        drift_target = tuple(box_size / 2 + 0.35 * (float(v) - box_size / 2) for v in target_cluster_pos)
        orbit_empty.location = drift_target
        orbit_empty.keyframe_insert(data_path="location", frame=scene.frame_end)

    # Smooth camera rotation
    if orbit:
        orbit_empty.rotation_euler = (0, 0, 0)
        orbit_empty.keyframe_insert(data_path="rotation_euler", index=2, frame=1)
        orbit_empty.rotation_euler = (0, 0, math.pi * 0.45)
        orbit_empty.keyframe_insert(data_path="rotation_euler", index=2, frame=scene.frame_end)

    # 2. Equidistant Planetarium 180° Fisheye Preset
    dome_cam_data = bpy.data.cameras.new("Fulldome_Fisheye_Data")
    dome_cam_data.type = "PANO"
    dome_cam_data.clip_start=box_size*1.0e-5
    dome_cam_data.clip_end=box_size*10.0
    if hasattr(dome_cam_data, "panorama_type"):
        dome_cam_data.panorama_type = "FISHEYE_EQUIDISTANT"
    elif hasattr(dome_cam_data, "cycles"):
        dome_cam_data.cycles.panorama_type = "FISHEYE_EQUIDISTANT"
    if hasattr(dome_cam_data, "fisheye_fov"):
        dome_cam_data.fisheye_fov = math.radians(180.0)
    elif hasattr(dome_cam_data, "cycles") and hasattr(dome_cam_data.cycles, "fisheye_fov"):
        dome_cam_data.cycles.fisheye_fov = math.radians(180.0)

    dome_cam = bpy.data.objects.new("Fulldome_Camera", dome_cam_data)
    col_rig.objects.link(dome_cam)
    dome_cam.location = center
    # Blender cameras look along local -Z. pi+tilt points to the rotated
    # zenith (0,sin(tilt),cos(tilt)); the former 67-degree pose looked down.
    dome_cam.rotation_euler = (math.pi + math.radians(23.0), 0.0, 0.0)

    # Set DEFAULT ACTIVE CAMERA to the 65mm perspective Camera!
    scene.camera = camera
    return camera, dome_cam


# -----------------------------------------------------------------------------
# Embedded Viewport Add-On Code (Directly installed and registered)
# -----------------------------------------------------------------------------


def install_and_register_inspector(bpy):
    """Embed the inspector in this scene and register it for this Blender process."""
    # Embed the inspector section only. The rest of this module prepares and
    # builds scenes; embedding it would require __file__ and reload every
    # builder when a saved .blend is opened.
    module_source = Path(__file__).with_name('blender.py').read_text(encoding="utf-8")
    marker = "# halo_inspector_addon\n"
    start = module_source.index(marker) + len(marker)
    end = module_source.index("\n# ========================================================================\n# native_radial_shape_inspector", start)
    addon_source = module_source[start:end]

    text_block = bpy.data.texts.get("dmsim_inspector.py") or bpy.data.texts.new("dmsim_inspector.py")
    text_block.clear()
    text_block.write(addon_source+'\n\nregister()\n')
    text_block.use_module = True

    namespace = {"__name__": "dmsim_scene_inspector"}
    exec(compile(addon_source, "dmsim_scene_inspector", "exec", dont_inherit=True), namespace)
    namespace["register"]()


# -----------------------------------------------------------------------------
# Metal GPU Hardware Acceleration Setup
# -----------------------------------------------------------------------------

def configure_render_device(bpy, scene, requested='AUTO'):
    """Select an available Cycles backend, recording AUTO CPU fallback explicitly."""
    requested = str(requested).upper()
    supported = ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL')
    if requested not in ('AUTO', 'CPU', *supported):
        raise ValueError('Unknown Cycles render device: '+requested)
    scene.cycles.device = 'CPU'
    record = {'requested': requested, 'backend': 'CPU', 'devices': [], 'discovery_errors': []}
    if requested == 'CPU':
        scene['research2_render_device'] = blender_json.dumps(record)
        return record
    try:
        cprefs = bpy.context.preferences.addons["cycles"].preferences
        available_types = [t[0] for t in cprefs.get_device_types(bpy.context)]
        choices = supported if requested == 'AUTO' else (requested,)
        for backend in choices:
            if backend not in available_types:
                continue
            try:
                cprefs.compute_device_type = backend
                cprefs.get_devices()
                selected = [device for device in cprefs.devices if device.type == backend]
                if not selected:
                    continue
                for device in cprefs.devices:
                    device.use = device in selected
                scene.cycles.device = 'GPU'
                record.update(backend=backend, devices=[device.name for device in selected])
                break
            except (AttributeError, RuntimeError, ValueError) as exc:
                record['discovery_errors'].append({'backend': backend, 'message': str(exc)})
    except (AttributeError, KeyError, RuntimeError) as exc:
        record['discovery_errors'].append({'message': str(exc)})
    if requested != 'AUTO' and record['backend'] == 'CPU':
        raise RuntimeError('Requested Cycles device unavailable: '+requested)
    scene['research2_render_device'] = blender_json.dumps(record)
    return record


def setup_metal_gpu_acceleration(bpy, scene):
    """Compatibility entrypoint for portable automatic Cycles device selection."""
    return configure_render_device(bpy, scene)


# -----------------------------------------------------------------------------
# Main Scene Builder & Movie Exporter
# -----------------------------------------------------------------------------

def _build_blender_universe_build_to_file(manifest_path: Path, blend_path: Path, render: bool = False, tier: str = "research", frames_per_snap: int = 30, preview: bool = False) -> Path:
    """Build the complete, unified research & cinematic Blender 3D universe."""
    import bpy

    manifest_path = manifest_path.expanduser().resolve()
    from blender import verify_combined_scene_manifest
    combined = verify_combined_scene_manifest(manifest_path)
    base = manifest_path.parent

    from core.foundation import file_sha256
    if 'species_manifest_sha256' in combined:
        receipt=manifest_path.with_name(manifest_path.name+'.sha256')
        if not receipt.is_file() or receipt.read_text().strip() != file_sha256(manifest_path):
            raise ValueError('Combined Blender manifest checksum mismatch')
        for species,name in combined['species_manifests'].items():
            path=(base/name).resolve()
            if not path.is_relative_to(base) or file_sha256(path) != combined['species_manifest_sha256'][species]:
                raise ValueError('Combined Blender species manifest changed: '+species)
    from blender import verify_timeline_assets
    timelines = {
        species: verify_timeline_assets(base / name)
        for species, name in combined["species_manifests"].items()
    }
    if "dm" not in timelines:
        raise ValueError("Combined Blender scene requires a dark-matter timeline manifest")

    reference = timelines["dm"]
    box = float(reference["box_mpc_h"])
    fps = int(reference.get("fps", 24))
    interpolated = bool(reference.get("intervals"))
    num_intervals = len(reference["snapshots"]) - 1
    if num_intervals < 1 or frames_per_snap < 2:
        raise ValueError("An animated scene needs at least two saved outputs and two frames per interval")
    if not math.isfinite(box) or box <= 0 or fps <= 0:
        raise ValueError("Timeline box and frame rate must be positive")
    epochs = reference.get("snapshots", [])
    if len(epochs) != num_intervals + 1:
        raise ValueError("Every interval must connect two recorded saved epochs")
    # Species are independent particle selections, but their coordinates and
    # epochs must describe the same simulation box and saved-output schedule.
    schedule = [(item["snapshot"], float(item["scale_factor"])) for item in epochs]
    if any(not math.isfinite(a) or a <= 0 for _, a in schedule) or any(
        schedule[i+1][1] <= schedule[i][1] for i in range(num_intervals)
    ):
        raise ValueError("Saved scale factors must be finite, positive and strictly increasing")
    for species, timeline in timelines.items():
        if not math.isclose(float(timeline["box_mpc_h"]), box, rel_tol=1e-12):
            raise ValueError("Species timelines must share one comoving box")
        if [(item["snapshot"], float(item["scale_factor"]))
                for item in timeline.get("snapshots", [])] != schedule:
            raise ValueError("Species timelines must share the saved-output schedule")
        if bool(timeline.get("intervals")) != interpolated:
            raise ValueError("Species timelines must share exact-output or interpolated display mode")
        if interpolated and [item["index"] for item in timeline.get("intervals", [])] != list(range(num_intervals)):
            raise ValueError("Timeline intervals must be contiguous and chronologically ordered")
    total_frame_end = num_intervals * int(frames_per_snap)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene['dmsim_data_contracts_json']=build_blender_universe_json.dumps({kind:timeline.get('data_contract') for kind,timeline in timelines.items()},sort_keys=True)
    scene['dmsim_epochs_json']=build_blender_universe_json.dumps([item.get('epoch_contract') for item in reference['snapshots']],sort_keys=True)
    scene['dmsim_source_manifests_json']=build_blender_universe_json.dumps(combined['species_manifests'],sort_keys=True)
    scene['dmsim_manifest_sha256']=file_sha256(manifest_path)
    scene['dmsim_science_acceptance']='NOT_ACCEPTED'
    scene.frame_start, scene.frame_end = 1, total_frame_end

    # Render engine & tier configuration
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    if tier == "viewport":
        scene.cycles.samples = 16
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    elif tier == "cinematic":
        scene.cycles.samples = 32
        scene.render.resolution_x, scene.render.resolution_y = 2560, 1440
    else:  # research default
        scene.cycles.samples = 24
        scene.render.resolution_x, scene.render.resolution_y = 1920, 1080

    scene.render.fps = fps
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 1.5

    # Enable Apple Silicon Metal GPU Acceleration!
    setup_metal_gpu_acceleration(bpy, scene)

    # -------------------------------------------------------------------------
    # 5 Structured Collections
    # -------------------------------------------------------------------------
    col_dm = get_or_create_collection(bpy, scene.collection, "1_Dark_Matter_Structure")
    col_gas = get_or_create_collection(bpy, scene.collection, "2_Hydro_Gas_Thermal")
    col_stars = get_or_create_collection(bpy, scene.collection, "3_Stars_and_Galaxies")
    col_halos = get_or_create_collection(bpy, scene.collection, "4_Halos_and_Subhalos")
    col_rig = get_or_create_collection(bpy, scene.collection, "5_Cinematic_Rig")
    col_black_holes = get_or_create_collection(bpy, scene.collection, "6_Black_Holes_Newtonian_Particles")

    # -------------------------------------------------------------------------
    # Lighting, World, & Compositor Setup
    # -------------------------------------------------------------------------
    setup_world_and_compositor(scene, box)

    # -------------------------------------------------------------------------
    # Target Most Massive Cluster for Hero Camera Tracking
    # -------------------------------------------------------------------------
    target_cluster_pos = None
    last_overlay = reference["snapshots"][-1].get("halo_overlays", {})
    hosts = last_overlay.get("halos", {}).get("items", [])
    if hosts:
        largest = max(hosts, key=lambda row: float(row.get("mass_msun_h", 0)))
        target_cluster_pos = largest.get("position_mpc_h")

    # -------------------------------------------------------------------------
    # Perspective inspection camera and equidistant 180-degree dome view
    # -------------------------------------------------------------------------
    camera, dome_cam = setup_cameras(bpy, scene, col_rig, box, target_cluster_pos,
                                   orbit=bool(reference.get("camera_orbit", False)))

    # -------------------------------------------------------------------------
    # Dynamic On-Screen Scientific HUD (Live Frame Handler)
    # -------------------------------------------------------------------------
    setup_dynamic_hud(bpy, scene, camera, reference, box, frames_per_snap, total_frame_end)

    # -------------------------------------------------------------------------
    # Import Particle Timeline Meshes with Point Cloud Geometry Nodes
    # -------------------------------------------------------------------------
    species_col_map = {
        "dm": col_dm,
        "gas": col_gas,
        "stars": col_stars,
        "black_holes": col_black_holes,
    }
    black_hole_timeline = timelines.get("black_holes")
    if black_hole_timeline is not None:
        scene["black_hole_particles_note"] = (
            "Imported from GADGET PartType5; markers are visually enlarged, "
            "Newtonian simulation particles, not event horizons or GR rays."
        )
        scene["black_hole_mass_normalization"] = build_blender_universe_json.dumps(
            black_hole_timeline.get("black_hole_mass_normalization"), sort_keys=True,
        )

    for species, timeline in timelines.items():
        target_col = species_col_map.get(species, scene.collection)
        if not interpolated:
            for index, snapshot in enumerate(timeline["snapshots"]):
                if not snapshot.get("ply"):
                    continue
                first = 1 if index == 0 else index * frames_per_snap
                last = (index + 1) * frames_per_snap - 1 if index < num_intervals else total_frame_end
                obj = import_particle_mesh(bpy, (base / combined["species_manifests"][species]).parent / snapshot["ply"], species, box, index)
                key_object_visibility(obj, first, last, total_frame_end)
                setup_particle_geometry_nodes(bpy, obj, species, box)
                if obj.name in scene.collection.objects:
                    scene.collection.objects.unlink(obj)
                target_col.objects.link(obj)
            continue
        if len(timeline["intervals"]) != num_intervals:
            raise ValueError("Particle timelines must share the same saved epoch schedule")
        for interval in timeline["intervals"]:
            index = int(interval["index"])
            first = 1 if index == 0 else index * frames_per_snap
            last = (index + 1) * frames_per_snap

            if int(interval["matched_particle_count"]) > 0:
                start = import_particle_mesh(
                    bpy, (base / combined["species_manifests"][species]).parent / interval["matched_start_ply"], species, box, index * 10
                )
                end = import_particle_mesh(
                    bpy, (base / combined["species_manifests"][species]).parent / interval["matched_end_ply"], species, box, index * 10 + 1
                )
                add_interval_shape_key(bpy, start, end, first, last, keep_target=index == num_intervals - 1)
                key_object_visibility(start, first, last - 1, total_frame_end)

                setup_particle_geometry_nodes(bpy, start, species, box)
                if index == num_intervals - 1:
                    setup_particle_geometry_nodes(bpy, end, species, box)

                if start.name in scene.collection.objects:
                    scene.collection.objects.unlink(start)
                target_col.objects.link(start)

                if index == num_intervals - 1:
                    key_object_visibility(end, last, last, total_frame_end)
                    if end.name in scene.collection.objects:
                        scene.collection.objects.unlink(end)
                    target_col.objects.link(end)

            for side in ("left_only", "right_only"):
                name = interval.get(side + "_ply")
                if not name:
                    continue
                obj = import_particle_mesh(
                    bpy, (base / combined["species_manifests"][species]).parent / name, species, box, index * 10 + (2 if side == "left_only" else 3)
                )
                # Interior right-only particles belong to the next interval's
                # saved left epoch; importing both would count them twice.
                if side == "right_only" and index != num_intervals - 1:
                    bpy.data.objects.remove(obj, do_unlink=True)
                    continue
                shown = first if side == "left_only" else last
                end_shown = max(first, last - 1) if side == "left_only" else last
                key_object_visibility(obj, shown, end_shown, total_frame_end)
                setup_particle_geometry_nodes(bpy, obj, species, box)
                if obj.name in scene.collection.objects:
                    scene.collection.objects.unlink(obj)
                target_col.objects.link(obj)

    # -------------------------------------------------------------------------
    # Selectable display shells at catalogue radii (Collection 4)
    # -------------------------------------------------------------------------
    selectable_count = make_selectable_halo_tracks(
        bpy, col_halos, timelines, box, frames_per_snap, total_frame_end,
    )
    scene["dmsim_selectable_halo_tracks"] = selectable_count

    # -------------------------------------------------------------------------
    # Install & Register Halo Inspector Add-On
    # -------------------------------------------------------------------------
    install_and_register_inspector(bpy)

    # -------------------------------------------------------------------------
    # AUTOMATICALLY CONFIGURE 3D VIEWPORT TO RENDERED SHADING
    # -------------------------------------------------------------------------
    for ws in bpy.data.workspaces:
        for scr in ws.screens:
            for area in scr.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "RENDERED"
                            space.shading.use_scene_lights = True
                            space.shading.use_scene_world = True

    scene["dmsim_display_semantics"] = ("Measured particle positions show halo shapes and debris. Catalogue apertures are viewport inspection aids. DM emission and glyph sizes are display choices. Stellar colours encode age, not spectra. "
        +("Between saved outputs, particle positions are display-interpolated and colours retain the left saved epoch. " if interpolated else "Particle states hold their saved output values. ")
        +"Membership, encounter-worldline and contamination highlights hold their actual saved states. Reconstructed impact views use their own labelled event clocks and local coordinates.")
    scene.frame_set(1)
    blend_path = Path(blend_path).resolve()
    output_base = blend_path.parent
    scene["build_options_json"] = build_blender_universe_json.dumps({"tier":tier,"frames_per_snapshot":frames_per_snap,"movie":render,"preview":preview}, sort_keys=True)

    # -------------------------------------------------------------------------
    # Export Broadcast-Quality MP4 Movie
    # -------------------------------------------------------------------------
    movie_path = output_base / "blender_universe_movie.mp4"
    set_output_format(scene, "FFMPEG")
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = "//" + movie_path.name
    scene["dmsim_motion_interpolated"] = interpolated
    from blender import build_sections
    build_sections(bpy, scene, {"dm": col_dm, "gas": col_gas, "stars": col_stars,
                   "black_holes": col_black_holes, "halos": col_halos,
                   "rig": col_rig}, combined, reference, base, output_base=output_base)
    for attachment in combined.get('stream_timelines',[]):
        record=verify_combined_stream_attachment(base,combined,attachment)
        import shutil
        source=(base/attachment['path']).parent;packed=output_base/'stream_timelines'/source.name
        packed.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,packed)
        stream_collection=load_stream_timeline_assets(bpy,packed/'inspector.json',maximum_saved_vertices=record['maximum_saved_vertices'],marker_radius_kpc=record['marker_radius_kpc'],
            scene_frames=record['scene_frames'],scene_frame_end=record['scene_frame_end'],coordinate_scales=record['coordinate_scales'])
        relative='//stream_timelines/'+source.name+'/'
        stream_collection['stream_inspector_manifest']=relative+'inspector.json'
        for obj in stream_collection.objects:obj['exact_identity_archive']=relative+'inspector.npz'
        scene['stream_inspector_manifest']=relative+'inspector.json'
    scene['combined_stream_timelines_json']=build_blender_universe_json.dumps(combined.get('stream_timelines',[]),sort_keys=True)
    if combined.get('local_impact_views'):
        impact_reference=combined['local_impact_views'];path=(base/impact_reference['path']).resolve()
        if file_sha256(path)!=impact_reference['sha256']:raise ValueError('Local reconstructed impact view changed')
        import shutil
        shutil.copyfile(path,output_base/'local_impact_views.json')
        scene['local_impact_scenes_json']=build_blender_universe_json.dumps(build_local_impact_views(bpy,build_blender_universe_json.loads(path.read_text())))
    if combined.get('native_selections'):
        reference_selection=combined['native_selections'];source=(base/reference_selection['path']).resolve()
        if file_sha256(source)!=reference_selection['sha256']:raise ValueError('Native membership/aperture scene bundle changed')
        import shutil
        shutil.copytree(source.parent,output_base/'native_selections')
        load_native_scene_selections(bpy,output_base/'native_selections/selections.json',scene_frame_end=total_frame_end)
    if combined.get('measured_display'):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from core.blender_scene import load_verified_display_products,measured_lifecycle_index
        from core.display_runtime import install_runtime
        from core.research_inspector import install_inspector
        import shutil
        display_source = (base/combined['measured_display']['path']).resolve()
        if file_sha256(display_source) != combined['measured_display']['sha256']:
            raise ValueError('Measured inspector supplement changed')
        display=load_verified_display_products(display_source)
        if display['coordinate_units']!='comoving Mpc/h' or display['frame_contract']['origin']!='global simulation Cartesian box origin':raise ValueError('Combined scene requires measured display in its global comoving Mpc/h frame')
        for epoch in display['timeline']:
            index=next((i for i,row in enumerate(reference['snapshots']) if row['snapshot']==epoch['snapshot']),None)
            if index is None or epoch['epoch_contract'].get('scale_factor')!=reference['snapshots'][index]['scale_factor'] or epoch['frame']!=(1 if index==0 else index*frames_per_snap):raise ValueError('Measured display differs from combined saved epoch/frame mapping')
        packed = output_base/'measured_display'
        shutil.copytree(display_source.parent, packed)
        install_runtime(bpy,display,packed,lifecycle=measured_lifecycle_index(display),preserve_scene_clock=True)
        install_inspector(bpy)
        scene['research_display_base'] = '//measured_display'
    configure_saved_universe_view(bpy, scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    if render:
        print(f"\nRendering particle visualization MP4 sequence ({total_frame_end} frames at {fps} fps)...")
        print(f"Destination: {movie_path}")
        bpy.ops.render.render(animation=True)
        if not movie_path.is_file():
            ranged = movie_path.with_name('%s%04d-%04d%s' % (movie_path.stem, scene.frame_start, scene.frame_end, movie_path.suffix))
            if ranged.is_file(): ranged.rename(movie_path)
        if not movie_path.is_file(): raise RuntimeError('Blender did not write the requested movie')

    if preview:
        scene.frame_set(scene.frame_end)
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
        scene.cycles.samples = 32
        set_output_format(scene, "PNG")
        scene.render.filepath = str(output_base / 'blender_preview.png')
        bpy.ops.render.render(write_still=True)
        from blender import render_sections
        render_sections(bpy, output_base)


    print("\nBuilt Blender particle visualization:")
    print(f"  Blend file : {blend_path}")
    print(f"  Movie file : {movie_path} (frames: {total_frame_end})")
    print(f"  Tracks     : {selectable_count} selectable halos/galaxies (independent catalogue detections)")
    print("  Camera     : 65mm Perspective + 180° Fisheye (Dome)")
    print("  Layers     : 5 particle/catalogue collections + camera rig")
    print("  Shading    : Automatically configured to RENDERED mode (Cycles AgX Metal GPU)")
    return blend_path


def build_blender_universe_build(manifest_path: Path, render: bool = False, tier: str = 'research', frames_per_snap: int = 30, preview: bool = False) -> Path:
    """Publish a retained complete combined scene and optional render products."""
    from blender import publish_scene_product, verify_combined_scene_manifest
    manifest_path = Path(manifest_path).resolve()
    blend_path = manifest_path.parent / 'blender_universe.blend'
    def builder(source, destination, requested_render):
        return _build_blender_universe_build_to_file(source, destination, render=render, tier=tier,
                              frames_per_snap=frames_per_snap, preview=preview)
    renders = (['blender_universe_movie.mp4'] if render else []) + (['blender_preview.png'] if preview else [])
    return publish_scene_product(manifest_path, blend_path, builder,
        input_schema='combined-scene-v1', algorithm=__file__, render=render or preview,
        manifest_verifier=verify_combined_scene_manifest, render_artifacts=renders)


def build_blender_universe_main() -> None:
    """Parse command arguments and dispatch this module's documented workflow."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tier", choices=["viewport", "research", "cinematic"], default="research")
    parser.add_argument("--frames-per-snapshot", type=int, default=30)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    build_blender_universe_build(args.manifest, render=args.render, tier=args.tier, frames_per_snap=args.frames_per_snapshot)
import json as build_gadget_merger_scene_json


def _build_gadget_merger_scene_emission_material(bpy, name: str, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _add_camera_text(bpy, collection, body, location, rotation, size, material, name):
    curve = bpy.data.curves.new(name + "_Font", type="FONT")
    curve.body = body
    curve.size = size
    curve.space_character = 1.05
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(material)
    return obj


def _key_exact_frame_visibility(obj, visible_frame: int, frame_count: int) -> None:
    keyframes = [(1, visible_frame == 1)]
    if visible_frame > 1:
        keyframes.append((visible_frame, True))
    if visible_frame < frame_count:
        keyframes.append((visible_frame + 1, False))
    for frame, visible in keyframes:
        obj.hide_render = not visible
        obj.hide_viewport = not visible
        obj.keyframe_insert(data_path="hide_render", frame=frame)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    if obj.animation_data and obj.animation_data.action:
        for curve in action_fcurves(obj.animation_data):
            for key in curve.keyframe_points:
                key.interpolation = "CONSTANT"


def _scale_bar_length(max_view_kpc: float) -> float:
    target = max_view_kpc / 5.0
    exponent = math.floor(math.log10(target))
    base = 10.0 ** exponent
    options = [base, 2.0 * base, 5.0 * base]
    return max((value for value in options if value <= target), default=base)


def _add_scale_bar(bpy, collection, start, rotation, length_kpc, line_width, text_size, material):
    from mathutils import Vector

    origin = Vector(start)
    horizontal = rotation.to_quaternion() @ Vector((1.0, 0.0, 0.0))
    vertical = rotation.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    curve = bpy.data.curves.new("Scale_Bar_Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = line_width
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*origin, 1.0)
    end = origin + horizontal * length_kpc
    spline.points[1].co = (*end, 1.0)
    obj = bpy.data.objects.new("Physical_Scale_Bar", curve)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    _add_camera_text(
        bpy, collection, "%g kpc" % length_kpc,
        origin + horizontal * (0.02 * length_kpc) + vertical * (1.5 * line_width),
        rotation, text_size, material, "Physical_Scale_Label",
    )


def _add_hud(bpy, scene, collection, camera, view_width, view_height, manifest):
    from mathutils import Vector

    rotation = camera.rotation_euler
    quaternion = rotation.to_quaternion()
    forward = quaternion @ Vector((0.0, 0.0, -1.0))
    right = quaternion @ Vector((1.0, 0.0, 0.0))
    up = quaternion @ Vector((0.0, 1.0, 0.0))
    bounds_min = Vector(manifest["scene_bounds_kpc"]["minimum"])
    bounds_max = Vector(manifest["scene_bounds_kpc"]["maximum"])
    center = (bounds_min + bounds_max) * 0.5
    distance = (camera.location - center).length
    hud_center = camera.location + forward * max(0.1, distance - view_height * 0.9)
    title_origin = hud_center - right * (0.46 * view_width) + up * (0.43 * view_height)
    white = _build_gadget_merger_scene_emission_material(bpy, "Merger_HUD_White", (0.84, 0.91, 0.98))
    cyan = _build_gadget_merger_scene_emission_material(bpy, "Merger_HUD_Cyan", (0.24, 0.80, 0.95))
    _add_camera_text(
        bpy, collection, "ISOLATED GADGET MERGER  /  SAVED OUTPUTS",
        title_origin, rotation, 0.027 * view_height, cyan, "Merger_Scene_Title",
    )
    _add_camera_text(
        bpy, collection,
        "Relative coordinates in kpc  |  Zero is the initial median origin  |  Discrete snapshots; playback rate is not physical",
        title_origin - up * (0.045 * view_height), rotation,
        0.016 * view_height, white, "Merger_Scene_Method_Note",
    )
    species_lines = [
        "DM: blue   Stars: display marker   Gas: display marker   BH: Newtonian PartType5 marker",
        "Collection visibility is independent; use the timeline to step through saved GADGET outputs.",
    ]
    for index, body in enumerate(species_lines):
        _add_camera_text(
            bpy, collection, body,
            hud_center - right * (0.46 * view_width) + up * (0.30 * view_height - index * 0.035 * view_height),
            rotation, 0.014 * view_height, white, "Merger_Legend_%02d" % index,
        )
    bar_length = _scale_bar_length(view_height)
    bar_start = hud_center - right * (0.45 * view_width) - up * (0.31 * view_height)
    _add_scale_bar(
        bpy, collection, bar_start, rotation, bar_length,
        0.0015 * view_height, 0.014 * view_height, white,
    )

    for index, output in enumerate(manifest["outputs"], start=1):
        if output.get("time_gyr") is None:
            time_label = "t = %.7g GADGET code time" % float(output["time_code"])
        else:
            time_label = "t = %.5g Gyr" % float(output["time_gyr"])
        counts = output.get("sampled_counts", {})
        label = "Snapshot %03d  |  %s  |  DM %s  |  stars %s" % (
            int(output["snapshot"]), time_label,
            format(int(counts.get("dm", 0)), ","), format(int(counts.get("stars", 0)), ","),
        )
        label_obj = _add_camera_text(
            bpy, collection, label,
            hud_center - right * (0.46 * view_width) - up * (0.42 * view_height),
            rotation, 0.023 * view_height, cyan, "Snapshot_Label_%03d" % index,
        )
        _key_exact_frame_visibility(label_obj, index, len(manifest["outputs"]))
        scene.timeline_markers.new("Snapshot %03d" % int(output["snapshot"]), frame=index)


def _build_gadget_merger_scene_build_to_file(manifest_path: Path, blend_path: Path, render: bool = False) -> Path:
    """Create a saved-output particle viewer scene."""
    import bpy
    from mathutils import Vector

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from blender import import_particle_mesh, setup_particle_geometry_nodes

    manifest_path = manifest_path.expanduser().resolve()
    blend_path = blend_path.expanduser().resolve()
    core = Path(__file__).resolve().parents[1] / "core"
    if str(core) not in sys.path: sys.path.insert(0, str(core))
    from blender import verify_export_manifest
    manifest = verify_export_manifest(manifest_path, schema="darkmattersim-isolated-gadget-merger-v1")
    if manifest.get("schema") != "darkmattersim-isolated-gadget-merger-v1":
        raise ValueError("Unsupported merger manifest schema")
    outputs = manifest.get("outputs", [])
    if len(outputs) < 2:
        raise ValueError("The merger scene requires at least two saved outputs")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    scene.cycles.samples = 16
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    set_output_format(scene, "PNG")
    scene.view_settings.view_transform = "AgX"
    scene['data_integrity_status'] = manifest['integrity_status']
    scene['science_acceptance'] = 'NOT_ACCEPTED'
    scene['source_manifest'] = str(manifest_path)
    scene.frame_start = 1
    scene.frame_end = len(outputs)
    scene.render.fps = 12
    scene.world = bpy.data.worlds.new("GADGET_Merger_World")
    scene.world.color = (0.001, 0.002, 0.006)

    bounds = manifest["scene_bounds_kpc"]
    minimum = Vector(bounds["minimum"])
    maximum = Vector(bounds["maximum"])
    center = (minimum + maximum) * 0.5
    extent = max(float((maximum - minimum).length), 1.0)
    max_axis_extent = max(float(value) for value in maximum - minimum)
    max_axis_extent = max(max_axis_extent, 1.0)
    aspect = scene.render.resolution_x / scene.render.resolution_y
    view_height = extent * 1.10
    view_width = view_height * aspect

    camera_data = bpy.data.cameras.new("Saved_Output_Orbit_Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = view_width
    camera_data.clip_start = max(extent * 1.0e-5, 1.0e-4)
    camera_data.clip_end = extent * 10.0
    camera = bpy.data.objects.new("Saved_Output_Orbit_Camera", camera_data)
    scene.collection.objects.link(camera)
    direction = Vector((1.0, -1.8, 1.1)).normalized()
    camera.location = center + direction * (extent * 2.0)
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera

    collection_names = {
        "dm": "01_Dark_Matter_Halos",
        "stars": "02_Stellar_Disks_And_Tidal_Debris",
        "gas": "03_Gas_Particles",
        "black_holes": "04_Black_Hole_Particle_Markers",
    }
    collections = {}
    for species, name in collection_names.items():
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
        collections[species] = collection

    imported = 0
    for frame_index, output in enumerate(outputs, start=1):
        for species in collection_names:
            relative_path = output.get("particle_files", {}).get(species)
            if not relative_path:
                continue
            particle_path = manifest_path.parent / relative_path
            obj = import_particle_mesh(bpy, particle_path, species, max_axis_extent, frame_index)
            # Coordinates are translated physical kpc and may be negative.
            # Preserve all saved particles; the cosmological [0,L) crop does
            # not describe this isolated merger.
            setup_particle_geometry_nodes(bpy, obj, species, max_axis_extent, crop_periodic=False)
            for source_collection in list(obj.users_collection):
                source_collection.objects.unlink(obj)
            collections[species].objects.link(obj)
            obj["gadget_snapshot"] = int(output["snapshot"])
            obj["particle_species"] = species
            obj["coordinate_unit"] = "kpc"
            obj['epoch_contract_json'] = build_gadget_merger_scene_json.dumps(output.get('epoch'), sort_keys=True)
            obj['numeric_product_json'] = build_gadget_merger_scene_json.dumps(output.get('numeric_products', {}).get(species), sort_keys=True)
            obj['identity_domain'] = manifest.get('identity_domain', 'UNDECLARED')
            _key_exact_frame_visibility(obj, frame_index, len(outputs))
            imported += 1
    if imported == 0:
        raise ValueError("No particle PLY files were present in the merger manifest")

    overlay_collection = bpy.data.collections.new("05_Scale_And_Scientific_Labels")
    scene.collection.children.link(overlay_collection)
    _add_hud(bpy, scene, overlay_collection, camera, view_width, view_height, manifest)

    scene["simulation_source"] = str(manifest.get("source_directory", ""))
    scene["simulation_model"] = "GADGET saved-output particle data; Newtonian simulation"
    scene["coordinate_unit"] = "kpc"
    scene["coordinate_translation_origin_kpc"] = build_gadget_merger_scene_json.dumps(
        manifest["coordinate_transform"]["translation_origin_kpc"],
    )
    scene["animation_policy"] = "Discrete saved outputs only; no interpolated trajectories"
    scene["stream_membership"] = "Not classified; stellar debris is displayed from actual stellar particle positions"
    scene["subhalo_perturbations"] = "No subhalo finder applied by this scene importer"
    scene["particle_sampling"] = build_gadget_merger_scene_json.dumps(manifest.get("particle_sampling", {}), sort_keys=True)
    scene["scientific_limits"] = build_gadget_merger_scene_json.dumps(manifest.get("scientific_limits", []))
    scene.frame_set(len(outputs))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if render:
        scene.render.filepath = str(blend_path.with_suffix(".png"))
        bpy.ops.render.render(write_still=True)
    return blend_path


def build_gadget_merger_scene_build(manifest_path: Path, blend_path: Path, render: bool = False) -> Path:
    """Build and durably publish a sealed standalone actual-data scene."""
    core = Path(__file__).resolve().parents[1] / 'core'
    if str(core) not in sys.path: sys.path.insert(0, str(core))
    from blender import publish_scene_product
    return publish_scene_product(manifest_path, blend_path, _build_gadget_merger_scene_build_to_file,
        input_schema='darkmattersim-isolated-gadget-merger-v1', algorithm=__file__, render=render)


def build_gadget_merger_scene_main() -> None:
    """Parse command arguments and dispatch this module's documented workflow."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("blend", type=Path)
    parser.add_argument("--render", action="store_true", help="render the final saved snapshot to a PNG")
    args = parser.parse_args(argv)
    print("Saved GADGET merger Blender scene: " + str(build_gadget_merger_scene_build(args.manifest, args.blend, args.render)))
import hashlib
import json as build_lensing_scene_json


def _build_lensing_scene_emission_material(bpy, name, color, strength=1.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = float(strength)
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _add_text(bpy, collection, body, location, size, material, name):
    curve = bpy.data.curves.new(name + "_Font", type="FONT")
    curve.body = body
    curve.size = size
    curve.space_character = 1.05
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material)
    return obj


def _add_panel(bpy, collection, x, y, width, height, z, material, name):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(
        [(x, y, z), (x + width, y, z), (x + width, y + height, z), (x, y + height, z)],
        [], [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _build_lensing_scene_build_to_file(image_path: Path, metadata_path: Path, blend_path: Path, render: bool = False) -> Path:
    """Create and save a camera-ready, labeled, editable Blender scene."""
    import bpy

    image_path = image_path.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    blend_path = blend_path.expanduser().resolve()
    if not image_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Provide the ray-traced image and its matching JSON metadata")
    metadata = verify_lensing_input(image_path, metadata_path)
    if metadata.get("model") != "Schwarzschild vacuum null geodesics; non-rotating isolated black hole":
        raise ValueError("Metadata does not identify the supported Schwarzschild ray-tracing model")

    expected_hash = metadata.get("output_sha256")
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if not isinstance(expected_hash, str) or digest != expected_hash:
        raise ValueError("The lensing image does not match the ray-trace metadata hash")
    dimensions = metadata.get("image_size_px")
    if (not isinstance(dimensions, list) or len(dimensions) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 128 for value in dimensions)):
        raise ValueError("Ray-trace metadata needs two integer image dimensions >=128 pixels")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    disable_cycles_denoising(scene)
    scene.cycles.samples = 8
    scene.render.resolution_x, scene.render.resolution_y = metadata["image_size_px"]
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    set_output_format(scene, "PNG")
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Schwarzschild_Lensing_World")
    scene.world.color = (0.0, 0.0, 0.0)

    collection = bpy.data.collections.new("Schwarzschild Ray Trace")
    scene.collection.children.link(collection)
    picture_width, picture_height = map(int, metadata["image_size_px"])
    aspect = picture_width / picture_height
    camera_data = bpy.data.cameras.new("GR_Lensing_Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0 * aspect
    camera = bpy.data.objects.new("GR_Lensing_Camera", camera_data)
    collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    scene.camera = camera

    image = bpy.data.images.load(str(image_path), check_existing=False)
    if list(image.size) != dimensions:
        raise ValueError("Lensing image dimensions disagree with the ray-trace metadata")
    image.name = "Schwarzschild_Geodesic_Render"
    image.pack()  # Keep the calculated pixels portable inside the saved .blend.
    plate_mesh = bpy.data.meshes.new("Lensed_Sky_Plate_Mesh")
    plate_mesh.from_pydata(
        [(-aspect, -1.0, 0.0), (aspect, -1.0, 0.0),
         (aspect, 1.0, 0.0), (-aspect, 1.0, 0.0)],
        [], [(0, 1, 2, 3)],
    )
    plate_mesh.uv_layers.new(name="UVMap")
    uv = plate_mesh.uv_layers.active.data
    for loop_index, coord in zip((0, 1, 2, 3), ((0, 0), (1, 0), (1, 1), (0, 1))):
        uv[loop_index].uv = coord
    plate_mesh.update()
    plate = bpy.data.objects.new("Lensed_Celestial_Panorama", plate_mesh)
    collection.objects.link(plate)
    material = bpy.data.materials.new("Ray_Traced_Panorama_Emission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Linear"
    material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plate.data.materials.append(material)

    text_white = _build_lensing_scene_emission_material(bpy, "HUD_White", (0.82, 0.91, 0.97), 1.0)
    text_cyan = _build_lensing_scene_emission_material(bpy, "HUD_Cyan", (0.18, 0.78, 0.92), 1.0)
    text_amber = _build_lensing_scene_emission_material(bpy, "HUD_Amber", (1.0, 0.61, 0.18), 1.0)
    panel_material = _build_lensing_scene_emission_material(bpy, "HUD_Panel", (0.012, 0.026, 0.044), 1.0)

    # A shallow full-width footer keeps the measured scale readable without
    # placing a fake disk, horizon glow, or decorative lens geometry in the view.
    footer_height = 0.34
    _add_panel(bpy, collection, -aspect, -1.0, 2.0 * aspect, footer_height,
               0.04, panel_material, "Physics_And_Provenance_Panel")
    margin = min(0.055, 0.04*2.0*aspect)
    left = -aspect + margin
    _add_text(bpy, collection, "SCHWARZSCHILD  /  NULL-GEODESIC RAY TRACE",
              (left, -0.755, 0.08), 0.068, text_cyan, "Scene_Title")

    mass = float(metadata["black_hole_mass_solar"])
    rs_km = float(metadata["schwarzschild_radius_km"])
    observer_rs = float(metadata["observer_radius_r_s"])
    critical = float(metadata["critical_impact_parameter_r_s"])
    shadow_diameter = float(metadata["shadow_diameter_at_observer_deg"])
    line_one = "M = %.5g M_sun   |   r_s = %.5g km   |   observer R = %.5g r_s" % (
        mass, rs_km, observer_rs,
    )
    line_two = "photon sphere = 1.5 r_s   |   critical impact b = %.6g r_s   |   shadow = %.5g deg" % (
        critical, shadow_diameter,
    )
    line_three = "Black pixels mark captured rays; the horizon is not a luminous surface."
    line_four = "Isolated Schwarzschild metric; no GADGET matter field, spin, plasma, or emission model."
    _add_text(bpy, collection, line_one, (left, -0.84, 0.08), 0.035, text_white, "Measured_Scale")
    _add_text(bpy, collection, line_two, (left, -0.895, 0.08), 0.035, text_amber, "Photon_Sphere_And_Shadow")
    _add_text(bpy, collection, line_three, (left, -0.945, 0.08), 0.026, text_white, "Horizon_Interpretation")
    _add_text(bpy, collection, line_four, (left, -0.985, 0.08), 0.022, text_white, "Model_Limits")

    # Fit annotations to the actual aspect ratio rather than clipping long
    # labels for square or portrait ray-trace images. Font size remains editable.
    bpy.context.view_layer.update()
    available_width = 2.0*aspect - 2.0*margin
    for obj in collection.objects:
        if obj.type == "FONT" and obj.dimensions.x > available_width:
            obj.data.size *= available_width/obj.dimensions.x

    scene["relativity_model"] = metadata["model"]
    # Store the complete calculation record, not just a path to another machine.
    scene["raytrace_metadata"] = build_lensing_scene_json.dumps(metadata, sort_keys=True)
    scene["raytrace_metadata_json"] = str(metadata_path)
    scene["source_panorama_json"] = build_lensing_scene_json.dumps(metadata.get("source_panorama", {}), sort_keys=True)
    scene["event_horizon_surface_drawn"] = False
    scene["gadget_spacetime_coupling"] = False
    scene['science_acceptance'] = 'NOT_ACCEPTED'
    scene['source_integrity_status'] = metadata['integrity_status']
    scene['numeric_product_json'] = build_lensing_scene_json.dumps(metadata.get('numeric_data'), sort_keys=True)
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if render:
        scene.render.filepath = str(blend_path.with_name(blend_path.stem + "_preview.png"))
        bpy.ops.render.render(write_still=True)
    return blend_path


def verify_lensing_input(image_path, metadata_path):
    """Read the shared sealed image product; identify historical unsealed inputs."""
    core = Path(__file__).resolve().parents[1] / 'core'
    if str(core) not in sys.path: sys.path.insert(0, str(core))
    from core.foundation import file_sha256
    from blender import verify_image_publication
    image_path, metadata_path = Path(image_path).resolve(), Path(metadata_path).resolve()
    before = file_sha256(metadata_path)
    metadata = build_lensing_scene_json.loads(metadata_path.read_text())
    if metadata.get('publication_contract'):
        if metadata_path != image_path.with_suffix('.json'):
            raise ValueError('Lensing metadata must be the published image sibling')
        metadata = verify_image_publication(image_path)
    else:
        if file_sha256(image_path) != metadata.get('output_sha256'):
            raise ValueError('Legacy lensing image does not match its metadata')
        metadata['integrity_status'] = 'LEGACY_UNSEALED'
    if metadata.get('model') != 'Schwarzschild vacuum null geodesics; non-rotating isolated black hole':
        raise ValueError('Unsupported lensing model')
    if file_sha256(metadata_path) != before:
        raise ValueError('Lensing metadata changed while reading')
    return metadata


def build_lensing_scene_build(image_path: Path, metadata_path: Path, blend_path: Path, render: bool = False) -> Path:
    """Publish the actual ray-traced scene through the shared retained generation."""
    core = Path(__file__).resolve().parents[1] / 'core'
    if str(core) not in sys.path: sys.path.insert(0, str(core))
    from blender import publish_scene_product
    image_path, metadata_path, blend_path = map(lambda p: Path(p).resolve(), (image_path, metadata_path, blend_path))
    def verifier(path): return verify_lensing_input(image_path, path)
    def builder(source, destination, requested_render):
        return _build_lensing_scene_build_to_file(image_path, source, destination, requested_render)
    return publish_scene_product(metadata_path, blend_path, builder,
        input_schema='scientific-image-product-v1', algorithm=__file__, render=render,
        manifest_verifier=verifier, render_artifacts=[blend_path.stem + '_preview.png'])


def build_lensing_scene_main() -> None:
    """Parse command arguments and dispatch this module's documented workflow."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("blend", type=Path)
    parser.add_argument("--render", action="store_true", help="also render a labeled PNG preview")
    args = parser.parse_args(argv)
    print("Saved Blender scene: " + str(build_lensing_scene_build(args.image, args.metadata, args.blend, args.render)))

# ========================================================================
# halo_inspector_addon
try:
    import bpy
except ModuleNotFoundError:
    bpy = None

if bpy is not None:
    """Blender inspector and persistent epoch labels for measured catalogue detections.

    Each selectable marker belongs to one saved snapshot. A wire sphere is an SO
    measurement aperture, not an inferred halo shape. Particle geometry shows the
    actual morphology. Masses are Msun/h; coordinates and radii are comoving Mpc/h.
    This file is embedded unchanged in exported scenes, avoiding a second add-on.
    """

    # Workflow and units
    # Embedded Blender inspector for per-epoch catalogue provenance and camera focusing.
    # Properties use their exported units, including comoving Mpc/h and Msun/h. HUD epochs report saved scale factors.
    # The inspector reads scene metadata; it performs no new halo detection or force calculation. A persistent handler updates labels after .blend reload.
    # Bundled source patches: core/patches. Runtime choices are saved with each run.
    import json as halo_inspector_addon_json

    import bpy
    from bpy.app.handlers import persistent

    bl_info = {
        "name": "DarkMatterSim Halo Inspector", "author": "DarkMatterSim",
        "version": (2, 0, 0), "blender": (4, 0, 0),
        "location": "View3D > Sidebar > DarkMatterSim",
        "description": "Inspect saved halo detections and scientific scene epochs",
        "category": "3D View",
    }


    @persistent
    def update_scientific_hud(scene, *unused):
        """Label the saved or interpolated pose without pretending it is a timestep.

        The nearest saved output supplies the reported redshift. Interpolated
        positions and colours held from the left epoch are explicitly identified.
        The JSON schedule survives saving/reopening the .blend scene.
        """
        hud = bpy.data.objects.get("Scientific_HUD_Overlay")
        if hud is None or not scene.get("dmsim_hud_epochs_json"):
            return
        epochs = halo_inspector_addon_json.loads(scene["dmsim_hud_epochs_json"])
        frame = int(scene.frame_current)
        interpolated = bool(scene.get("dmsim_motion_interpolated", True))
        nearest = (min(epochs, key=lambda row: abs(row["frame"]-frame)) if interpolated else
                   max((row for row in epochs if row["frame"] <= frame), key=lambda row: row["frame"]))
        a = float(nearest["scale_factor"])
        exact = frame == int(nearest["frame"])
        state = "SAVED PARTICLE OUTPUT" if exact or not interpolated else "INTERPOLATED DISPLAY POSE"
        hud.data.body = (
            f"{state}  |  {'nearest' if interpolated else 'held'} snapshot {nearest['snapshot']}\n"
            f"saved z = {1./a-1.:.3f}  |  saved a = {a:.5f}  |  box = {scene['dmsim_box_mpc_h']:g} Mpc/h\n"
            "DM: display colour  |  gas: measured temperature bins  |  stars: age proxy\n"
            + ("Comoving coordinates; intermediate colours retain the left saved epoch" if interpolated else
             "Comoving coordinates; each saved output is held until the next output")
        )


    class DMSIM_OT_focus_halo(bpy.types.Operator):
        """Frame the selected catalogue marker in the 3D viewport."""
        bl_idname = "dmsim.focus_halo"
        bl_label = "Focus Selected Detection"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            if not context.active_object or "dmsim_halo_id" not in context.active_object:
                self.report({"WARNING"}, "Select a catalogue marker first")
                return {"CANCELLED"}
            bpy.ops.view3d.view_selected("INVOKE_DEFAULT")
            return {"FINISHED"}


    class DMSIM_PT_halo_inspector(bpy.types.Panel):
        """Display one exact catalogue row; never link local row IDs across outputs."""
        bl_label = "Catalogue Detection"
        bl_idname = "DMSIM_PT_halo_inspector"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "DarkMatterSim"

        @classmethod
        def poll(cls, context):
            """Show the panel whenever a selectable object is active."""
            return context.active_object is not None

        def draw(self, context):
            """Report the stored mass convention, aperture and detection epoch."""
            layout, obj = self.layout, context.active_object
            if "dmsim_halo_id" not in obj:
                layout.label(text="Select a catalogue wire aperture or axes marker", icon="INFO")
                return
            layout.operator("dmsim.focus_halo", icon="VIEWZOOM")
            layout.operator("dmsim.select_host_subhalos", icon="OUTLINER_OB_GROUP_INSTANCE")
            for key, title in (("dmsim_finder", "Finder"), ("dmsim_halo_id", "Catalogue ID"),
                               ("dmsim_mass_definition", "Mass definition"),
                               ("dmsim_radius_source", "Aperture"), ("dmsim_host_id", "Host ID")):
                layout.label(text=title + ": " + str(obj.get(key, "unavailable")))
            if obj.get('dmsim_mass_msun_h') is not None:
                layout.label(text="Mass: %.5g Msun/h" % float(obj["dmsim_mass_msun_h"]))
            detection = halo_inspector_addon_json.loads(obj.get("dmsim_detection_json", "{}"))
            for key, title in (("particle_count", "Particles"), ("subhalo_count", "Subhalos"),
                               ("central", "Central"), ("parent_id", "Parent")):
                if detection.get(key) is not None:
                    layout.label(text=f"{title}: {detection[key]}")
            if detection.get("velocity_km_s") is not None:
                layout.label(text="v [km/s]: " + ", ".join(f"{v:.4g}" for v in detection["velocity_km_s"]))
            for key, title, unit in (
                ("half_mass_radius_mpc_h", "Half-mass radius", "comoving Mpc/h"),
                ("vmax_radius_mpc_h", "Vmax radius", "comoving Mpc/h"),
                ("vmax_km_s", "Vmax", "km/s"),
                ("velocity_dispersion_km_s", "Velocity dispersion", "km/s"),
            ):
                if detection.get(key) is not None:
                    layout.label(text=f"{title}: {detection[key]:.5g} {unit}")
            types = ("Gas", "Fine DM", "Coarse matter", "Type 3", "Stars", "BH gravity")
            masses, counts = detection.get("mass_by_type_msun_h"), detection.get("particle_count_by_type")
            if masses is not None or counts is not None:
                for i, name in enumerate(types):
                    text = name
                    if counts is not None:
                        text += f": N={counts[i]}"
                    if masses is not None:
                        text += f"; M={masses[i]:.4g} Msun/h"
                    layout.label(text=text)
            if detection.get("formation_redshift") is not None:
                layout.label(text="Stellar formation proxy z: %.5g" % detection["formation_redshift"])
            for key, value in (detection.get("aperture") or {}).items():
                if value is not None:
                    layout.label(text=f"{key}: {value:.5g}")
            tree = halo_inspector_addon_json.loads(obj.get("dmsim_merger_history_json", "{}"))
            if tree.get("track"):
                layout.label(text="Main progenitor history (Mvir)")
                if tree.get("formation_redshift") is not None:
                    layout.label(text="Half-final-mass z: %.5g" % tree["formation_redshift"])
                for row in tree["track"]:
                    layout.label(text="z=%.3g; M=%.4g Msun/h" % (row["redshift"], row["mass_msun_h"]))
                for event in tree.get("merger_events", []):
                    layout.label(text="Merger by saved z=%.3g; ratio=%.3g" % (event["redshift"], event["mass_ratio"]))
            else:
                layout.label(text="Formation/merger history unavailable for this detection")
            for row in halo_inspector_addon_json.loads(obj.get("dmsim_history_json", "[]")):
                layout.label(text="Saved snapshot %s; z=%.5g" % (row["snapshot"], row["redshift"]))
                if row.get("r200c_mpc_h") is not None:
                    layout.label(text="R200c: %.5g comoving Mpc/h" % row["r200c_mpc_h"])
            profile=halo_inspector_addon_json.loads(obj.get('dmsim_density_profile_json','{}'))
            selected_shell=halo_inspector_addon_json.loads(obj.get('dmsim_density_shape_json','{}'))
            shells=[selected_shell] if selected_shell else profile.get('shells',[])
            if shells:
                layout.label(text='Radial density moments; proper kpc / Msun')
                layout.label(text='Simulation convergence: NOT_EVALUATED')
                for shell in shells:
                    layout.label(text="r=%.4g–%.4g kpc: %s" % (shell['lower_radius_kpc'],shell['upper_radius_kpc'],shell['status']))
                    layout.label(text=f"N={shell.get('particle_count')}; M={shell.get('mass_msun')} Msun")
                    if shell['status']=='MEASURED':
                        layout.label(text="b/a=%.4g; c/a=%.4g" % (shell['b_over_a'],shell['c_over_a']))
                    layout.label(text='Sampling: '+shell.get('sampling_uncertainty',{}).get('status','UNAVAILABLE'))
                    layout.label(text='Axes sampling resolved: '+str(shell.get('orientation_sampling_resolved',[False]*3)))
                layout.label(text='Failed bins retained; sampling is not resolution convergence')
            layout.label(text="Particle positions show halo morphology.")
            layout.label(text="Apertures do not infer spherical halo shapes or orbits.")


    class DMSIM_OT_select_host_subhalos(bpy.types.Operator):
        """Select catalogue detections belonging to this host at this same epoch."""
        bl_idname = "dmsim.select_host_subhalos"
        bl_label = "Select this host's subhalos"

        def execute(self, context):
            active = context.active_object
            if active is None:
                return {"CANCELLED"}
            detection = halo_inspector_addon_json.loads(active.get("dmsim_detection_json", "{}"))
            host, snapshot = detection.get("host_id"), detection.get("snapshot")
            if host is None or snapshot is None or detection.get("finder") not in {"GADGET-4 FOF", "GADGET-4 SUBFIND", "galaxies"}:
                self.report({"WARNING"}, "Select a saved GADGET host/subhalo detection")
                return {"CANCELLED"}
            for obj in context.selected_objects:
                obj.select_set(False)
            selected = []
            for obj in context.view_layer.objects:
                item = halo_inspector_addon_json.loads(obj.get("dmsim_detection_json", "{}"))
                if (item.get("subhalo") and item.get("host_id") == host
                        and item.get("snapshot") == snapshot and obj.visible_get()):
                    obj.select_set(True)
                    selected.append(obj)
            if not selected:
                self.report({"WARNING"}, "No visible subhalos at this saved epoch; select its timeline frame")
                return {"CANCELLED"}
            context.view_layer.objects.active = selected[0]
            return {"FINISHED"}


    class DMSIM_PT_native_selection(bpy.types.Panel):
        bl_label='Native stellar selection / contamination'
        bl_idname='DMSIM_PT_native_selection'
        bl_space_type='VIEW_3D'
        bl_region_type='UI'
        bl_category='Research'
        def draw(self,context):
            obj=context.active_object
            if obj is None or 'native_overlay_metadata_json' not in obj:
                self.layout.label(text='Select a native membership highlight or support aperture')
                return
            record=halo_inspector_addon_json.loads(obj['native_overlay_metadata_json'])
            for key in ['role','units','namespace','particle_count','status','selection','exact_identity_archive','source_sha256']:
                if key in record:self.layout.label(text=key+': '+str(record[key]))
            epoch=record.get('epoch',record.get('measurement',{}))
            for key in ['snapshot_storage_index','snapshot','scale_factor','coarse_mass_fraction','coarse_mass_inside_msun','total_mass_inside_msun']:
                if key in epoch:self.layout.label(text=key+': '+str(epoch[key]))
            if obj.get('exact_identity_preview'):self.layout.label(text='Exact IDs: '+obj['exact_identity_preview'])

    _CLASSES = (DMSIM_OT_focus_halo, DMSIM_OT_select_host_subhalos, DMSIM_PT_halo_inspector,DMSIM_PT_native_selection)


    def register():
        """Register the single inspector and one scene-reload-safe HUD handler."""
        for cls in _CLASSES:
            previous = getattr(bpy.types, cls.__name__, None)
            if previous is not None:
                bpy.utils.unregister_class(previous)
            bpy.utils.register_class(cls)
        for handler in list(bpy.app.handlers.frame_change_pre):
            if getattr(handler, "__name__", "") == "update_scientific_hud":
                bpy.app.handlers.frame_change_pre.remove(handler)
        bpy.app.handlers.frame_change_pre.append(update_scientific_hud)
        update_scientific_hud(bpy.context.scene)


    def unregister():
        """Remove the inspector and its frame handler without touching scene data."""
        for handler in list(bpy.app.handlers.frame_change_pre):
            if getattr(handler, "__name__", "") == "update_scientific_hud":
                bpy.app.handlers.frame_change_pre.remove(handler)
        for cls in reversed(_CLASSES):
            registered = getattr(bpy.types, cls.__name__, None)
            if registered is not None:
                bpy.utils.unregister_class(registered)


    # Registration is explicit in the selected scene builder; importing this
    # consolidated module never registers a second copy of the inspector.

# ========================================================================
# native_radial_shape_inspector
def native_radial_shape_inspector_main():
    """Fresh Blender load and radial inspector verification for the Block 02 analytic control."""
    import bpy,json,sys
    from pathlib import Path
    from types import SimpleNamespace
    scene_path=Path(bpy.data.filepath);sys.path.insert(0,str(scene_path.parent))
    profiles=[obj for obj in bpy.data.objects if obj.get('dmsim_density_profile_json') and not obj.get('dmsim_density_shape_json')]
    shells=[obj for obj in bpy.data.objects if obj.get('dmsim_density_shape_json')]
    assert len(profiles)==2,(len(profiles),len(shells))
    assert len(shells)==8,len(shells)
    for obj in profiles:
     record=json.loads(obj['dmsim_density_profile_json'])
     assert record['snapshot'] in (0,1)
     assert record['scale_factor']==(.5 if record['snapshot']==0 else 1.)
     assert record['science_acceptance']=='NOT_ACCEPTED'
     assert len(record['shells'])==4
    for obj in shells:
     record=json.loads(obj['dmsim_density_shape_json'])
     assert record['status']=='MEASURED'
     assert abs(record['b_over_a']-.7)<1.e-4
     assert abs(record['c_over_a']-.45)<1.e-4
     assert record['simulation_convergence']['status']=='NOT_EVALUATED'
     assert 'dmsim_mass_msun_h' in obj
    # Exercise the real inspector draw logic for every host and density shell.
    import blender as halo_inspector_addon
    labels=[]
    class Layout:
     def label(self,**kwargs):labels.append(kwargs['text'])
     def operator(self,*args,**kwargs):return None
    for obj in profiles+shells:
     halo_inspector_addon.DMSIM_PT_halo_inspector.draw(SimpleNamespace(layout=Layout()),SimpleNamespace(active_object=obj))
    assert any('Sampling:' in label for label in labels)
    assert any('Simulation convergence: NOT_EVALUATED' in label for label in labels)
    proof=scene_path.parents[4]
    result={'status':'PASS','scene':str(scene_path),'profile_objects':len(profiles),'density_shell_objects':len(shells),
            'inspector_draws':len(profiles)+len(shells),'scale_factors':[.5,1.],'axis_ratio_tolerance':1.e-4,
            'science_acceptance':'NOT_ACCEPTED','scope':'fresh-loaded analytic native-format control; no cosmological simulation acceptance'}
    (proof/'fresh_load.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

# ========================================================================
# native_catalogue_inspector
import json as native_catalogue_inspector_json


def native_catalogue_inspector_main():
    import bpy
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "5_blender"), str(root / "core")]
    from blender import make_selectable_halo_tracks
    import blender as halo_inspector_addon
    arguments = sys.argv[sys.argv.index("--")+1:]
    input_path, output_path = map(Path, arguments)
    overlays = native_catalogue_inspector_json.loads(input_path.read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    collection = bpy.data.collections.new("Synthetic catalogue inspection fixture")
    bpy.context.scene.collection.children.link(collection)
    epoch = {"snapshot": 0, "redshift": 1., "scale_factor": .5, "halo_overlays": overlays}
    count = make_selectable_halo_tracks(bpy, collection, {"dm": {"snapshots": [epoch]}}, .1, 12, 12)
    assert count == len(overlays["subhalos"]["items"])
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    for obj in collection.objects:
        detection = native_catalogue_inspector_json.loads(obj["dmsim_detection_json"])
        assert obj["dmsim_radius_source"] == "SUBFIND bound half-mass radius"
        assert abs(obj.scale.x-detection["half_mass_radius_mpc_h"]) < 1e-8
        assert obj.hide_render, "Catalogue measurement apertures must preserve render semantics"
        assert "mass_by_type_msun_h" in detection
    halo_inspector_addon.register()
    active = next(iter(collection.objects))
    active.select_set(True)
    bpy.context.view_layer.objects.active = active
    assert bpy.ops.dmsim.select_host_subhalos() == {"FINISHED"}
    assert len(bpy.context.selected_objects) == len(overlays["subhalos"]["items"])
    bpy.context.scene["fixture"] = "Synthetic catalogue layout, not evolved science data"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    output_path.with_suffix(".json").write_text(native_catalogue_inspector_json.dumps({"status": "PASS", "fixture": "synthetic",
        "markers": count, "checks": ["bound half-mass apertures", "species metadata", "same-host same-epoch selection", "inspector registration", "blend save"]}, indent=2)+"\n")
    print("NATIVE_CATALOGUE_INSPECTOR_PASS")

SCENE_COMMANDS = {
    'blender': blender_main,
    'build-blender-universe': build_blender_universe_main,
    'build-gadget-merger-scene': build_gadget_merger_scene_main,
    'build-lensing-scene': build_lensing_scene_main,
    'native-catalogue-inspector': native_catalogue_inspector_main,
    'native-radial-shape-inspector': native_radial_shape_inspector_main,
}

def main(argv=None):
    import sys
    args=list(sys.argv[sys.argv.index('--')+1:] if argv is None and '--' in sys.argv else sys.argv[1:] if argv is None else argv)
    name=args.pop(0) if args and args[0] in SCENE_COMMANDS else 'blender'
    if name not in SCENE_COMMANDS:raise SystemExit('Select a Blender scene command: '+', '.join(SCENE_COMMANDS))
    previous=sys.argv
    try:
        sys.argv=[str(Path(__file__).resolve()),'--',*args]
        return SCENE_COMMANDS[name]()
    finally:sys.argv=previous

def load_stream_inspector_assets(bpy, manifest_path, *, maximum_points, marker_radius_kpc,
                                 velocity_scale_kpc_per_kms=None, maximum_velocity_vectors=None):
    """Load measured stream points/tracks with exact identity and source links.

    Call inside Blender: core.blender.load_stream_inspector_assets(bpy, path,...).
    uint64 identities remain decimal strings/sidecar records, never float fields.
    One Blender unit represents one proper kpc.
    """
    import numpy as np
    import json
    from core.foundation import file_sha256 as digest
    manifest_path=Path(manifest_path).resolve();record=json.loads(manifest_path.read_text())
    if record.get('schema')=='stream-inspector-timeline-v1':
        if velocity_scale_kpc_per_kms is not None or maximum_velocity_vectors is not None:
            raise ValueError('Velocity vectors require a single-epoch inspector asset')
        return load_stream_timeline_assets(bpy,manifest_path,maximum_saved_vertices=maximum_points,
                                          marker_radius_kpc=marker_radius_kpc)
    arrays=manifest_path.with_name('inspector.npz')
    if record['schema']!='stream-inspector-v1' or digest(arrays)!=record['arrays_sha256']:
        raise ValueError('Stream inspector asset binding mismatch')
    packaged=record.get('packaged_product')
    if packaged is not None:
        relative=Path(packaged)
        source_product=(manifest_path.parent/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not source_product.is_relative_to(manifest_path.parent):
            raise ValueError('Packaged stream source metadata must stay inside the inspector bundle')
    else:
        source_product=Path(record['product'])
        if not source_product.is_absolute():source_product=(manifest_path.parent/source_product).resolve()
    if digest(source_product)!=record['product_sha256']:raise ValueError('Stream inspector source product changed')
    if (type(maximum_points) is not int or maximum_points<=0 or
            not math.isfinite(marker_radius_kpc) or marker_radius_kpc<=0):
        raise ValueError('Explicit inspector point/radius bounds required')
    with np.load(arrays,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
    xyz=data['position_kpc']
    if len(xyz)>maximum_points:raise MemoryError('Stream inspector particle budget exceeded')
    vectors=None
    if velocity_scale_kpc_per_kms is not None or maximum_velocity_vectors is not None:
        if (type(maximum_velocity_vectors) is not int or not 1<=maximum_velocity_vectors<=5000 or
                type(velocity_scale_kpc_per_kms) not in (int,float) or
                not np.isfinite(velocity_scale_kpc_per_kms) or velocity_scale_kpc_per_kms<=0):
            raise ValueError('Velocity display requires a positive kpc/(km/s) scale and a vector budget of 1–5000')
        if 'velocity_kms' not in data:
            raise ValueError('This inspector does not contain saved velocities')
        velocity=np.asarray(data['velocity_kms']);ids=np.asarray(data['particle_ids'])
        if (xyz.ndim!=2 or xyz.shape[1]!=3 or velocity.shape!=xyz.shape or ids.shape!=(len(xyz),)
                or ids.dtype!=np.dtype('uint64') or len(np.unique(ids))!=len(ids)
                or not np.isfinite(xyz).all() or not np.isfinite(velocity).all()):
            raise ValueError('Finite aligned positions, velocities and unique exact uint64 identities are required')
        # A deterministic identity order avoids a random display choice. The
        # selected IDs are stored as decimal strings, never floating values.
        rows=np.argsort(ids,kind='stable')[:maximum_velocity_vectors]
        vectors=(rows,velocity)
    track_segments=[]
    for name in record['tracks']:
        points=data[name]
        support_name=record.get('track_support',{}).get(name)
        if support_name is None or support_name not in data:
            raise ValueError('Stream track requires its saved support mask; re-export the inspector')
        support=data[support_name]
        if points.ndim!=2 or points.shape[1]!=3 or support.dtype.kind!='b' or support.shape!=(len(points),):
            raise ValueError('Stream track support must align with its proper-kpc coordinates')
        if not np.isfinite(points[support]).all():
            raise ValueError('Supported stream track coordinates must be finite')
        # Keep each contiguous supported run separate. Joining points across
        # an unsupported bin would draw a physical-looking path without data.
        rows=np.flatnonzero(support)
        for group in np.split(rows,np.flatnonzero(np.diff(rows)>1)+1):
            if len(group)>=2:
                track_segments.append((name,points[group]))
    collection=bpy.data.collections.new('Measured stream '+record['namespace']);bpy.context.scene.collection.children.link(collection)
    mesh=bpy.data.meshes.new('Stream sample proper kpc');mesh.from_pydata(xyz.tolist(),[],[]);mesh.update()
    origin=record.get('particle_origin','UNSPECIFIED')
    obj=bpy.data.objects.new('Stream sample - '+origin,mesh);collection.objects.link(obj)
    obj['stream_namespace']=record['namespace'];obj['source_product']=record['product'];obj['science_acceptance']=record['science_acceptance']
    obj['particle_origin']=origin
    obj['exact_identity_archive']=str(arrays);obj['coordinate_units']='proper kpc';obj['marker_radius_kpc']=marker_radius_kpc
    ids=data['particle_ids'];obj['particle_count']=len(ids)
    obj['exact_identity_preview']=','.join(str(int(value)) for value in ids[:16])
    # Render measured particles as bounded geometric instances rather than an
    # unrenderable vertex-only mesh. Archive row indices retain exact-ID lookup.
    node_group=bpy.data.node_groups.new('Measured stream tracer instances','GeometryNodeTree')
    if hasattr(node_group,'interface'):
        node_group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    else:
        node_group.inputs.new('NodeSocketGeometry','Geometry');node_group.outputs.new('NodeSocketGeometry','Geometry')
    nodes=node_group.nodes;links=node_group.links
    input_node=nodes.new('NodeGroupInput');output_node=nodes.new('NodeGroupOutput')
    points=nodes.new('GeometryNodeMeshToPoints');sphere=nodes.new('GeometryNodeMeshIcoSphere');instances=nodes.new('GeometryNodeInstanceOnPoints')
    sphere.inputs['Radius'].default_value=marker_radius_kpc;sphere.inputs['Subdivisions'].default_value=1
    links.new(input_node.outputs['Geometry'],points.inputs['Mesh']);links.new(points.outputs['Points'],instances.inputs['Points'])
    if record.get('role'):
        color=(.1,.8,1.,1.) if record['role']=='current bound progenitor member stars' else (1.,.6,.1,1.)
        material,_=_emission(bpy,'Native selection display '+record['role'],color)
        assign=nodes.new('GeometryNodeSetMaterial');assign.inputs['Material'].default_value=material
        links.new(sphere.outputs['Mesh'],assign.inputs['Geometry']);links.new(assign.outputs['Geometry'],instances.inputs['Instance'])
    else:links.new(sphere.outputs['Mesh'],instances.inputs['Instance'])
    links.new(instances.outputs['Instances'],output_node.inputs['Geometry'])
    modifier=obj.modifiers.new('Measured tracer markers','NODES');modifier.node_group=node_group
    for name in ('candidate_id','wrap_id','bound'):
        if name in data:
            attribute=mesh.attributes.new(name=name,type='INT',domain='POINT')
            attribute.data.foreach_set('value',np.asarray(data[name],dtype=np.int32))
    for ordinal,(name,points) in enumerate(track_segments):
        label=name+' supported segment '+str(ordinal)
        curve=bpy.data.curves.new(label,'CURVE');curve.dimensions='3D';curve.bevel_depth=marker_radius_kpc
        spline=curve.splines.new('POLY');spline.points.add(len(points)-1)
        for point,coordinate in zip(spline.points,points):point.co=(*coordinate,1.)
        track=bpy.data.objects.new(label,curve);collection.objects.link(track)
        track['source_product']=record['product'];track['particle_origin']=origin
        track['fit_semantics']='principal curve; only contiguous saved supported points'
    if vectors is not None:
        rows,velocity=vectors
        curve=bpy.data.curves.new('Saved velocity vectors','CURVE');curve.dimensions='3D'
        curve.bevel_depth=marker_radius_kpc*.25;curve.bevel_resolution=1
        for row in rows:
            end=xyz[row]+velocity[row]*velocity_scale_kpc_per_kms
            if not np.isfinite(end).all():raise ValueError('Scaled velocity vector exceeds numeric range')
            spline=curve.splines.new('POLY');spline.points.add(1)
            spline.points[0].co=(*xyz[row],1.);spline.points[1].co=(*end,1.)
        material,_=_emission(bpy,'Saved velocity vectors display',(.35,.9,1.,1.))
        curve.materials.append(material)
        vector_obj=bpy.data.objects.new('Saved velocity vectors',curve);collection.objects.link(vector_obj)
        vector_obj['stream_namespace']=record['namespace'];vector_obj['particle_origin']=origin
        vector_obj['selected_particle_ids_decimal']=','.join(str(int(data['particle_ids'][row])) for row in rows)
        vector_obj['velocity_units']='km/s';vector_obj['position_units']='proper kpc'
        vector_obj['velocity_scale_kpc_per_kms']=float(velocity_scale_kpc_per_kms)
        vector_obj['display_semantics']='Saved velocity components times declared display scale; lines are not trajectories'
        vector_obj['source_product_sha256']=record['product_sha256']
    bpy.context.scene['stream_inspector_manifest']=str(manifest_path)
    return collection


def stream_inspector_main():
    import argparse
    import bpy
    parser=argparse.ArgumentParser(description='Load measured stream inspector points and principal tracks.')
    parser.add_argument('manifest',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-points',type=int,required=True);parser.add_argument('--marker-radius-kpc',type=float,required=True)
    parser.add_argument('--velocity-scale-kpc-per-kms',type=float,help='Optional explicit velocity-line display scale')
    parser.add_argument('--maximum-velocity-vectors',type=int,help='Optional deterministic exact-ID vector budget (1–5000)')
    arguments=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args=parser.parse_args(arguments)
    load_stream_inspector_assets(bpy,args.manifest,maximum_points=args.maximum_points,marker_radius_kpc=args.marker_radius_kpc,
        velocity_scale_kpc_per_kms=args.velocity_scale_kpc_per_kms,maximum_velocity_vectors=args.maximum_velocity_vectors)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

def saved_stream_frame_schedule(saved_times_myr, *, scene_frames=None, scene_frame_end=None):
    """Map original saved epochs to explicit display frames without time fitting."""
    times=list(saved_times_myr)
    if len(times)<2 or any(type(t) not in (int,float) or not math.isfinite(t) for t in times) or any(right<=left for left,right in zip(times,times[1:])):
        raise ValueError('Strictly increasing original saved stream times required')
    if scene_frames is None:
        if scene_frame_end is not None:raise ValueError('Combined scene end requires explicit saved frame bindings')
        return list(range(1,len(times)+1)),len(times)
    frames=list(scene_frames)
    if len(frames)!=len(times) or any(type(frame) is not int or frame<1 for frame in frames) or any(right<=left for left,right in zip(frames,frames[1:])):
        raise ValueError('Exact increasing integer frame for each original saved epoch required')
    if type(scene_frame_end) is not int or scene_frame_end<frames[-1]:raise ValueError('Combined scene end must include every saved stream epoch')
    return frames,scene_frame_end


def load_stream_timeline_assets(bpy, manifest_path, *, maximum_saved_vertices, marker_radius_kpc, scene_frames=None, scene_frame_end=None,coordinate_scales=None):
    """Display actual saved stream epochs on a physical-time-indexed timeline.

    Each epoch is held until the next saved time. Visibility uses constant
    keyframes; no new native state or trajectory interpolation is implied.
    """
    import numpy as np
    import zipfile
    import json
    from core.foundation import file_sha256 as digest
    path=Path(manifest_path).resolve();record=json.loads(path.read_text())
    archive=path.with_name('inspector.npz')
    if record.get('schema')!='stream-inspector-timeline-v1' or digest(archive)!=record['arrays_sha256']:
        raise ValueError('Saved stream timeline binding changed')
    if type(maximum_saved_vertices)is not int or maximum_saved_vertices<1:
        raise ValueError('Explicit positive saved-vertex budget required')
    if not np.isfinite(marker_radius_kpc) or marker_radius_kpc<=0:
        raise ValueError('Finite positive marker radius required')
    with zipfile.ZipFile(archive) as packed:
        if sum(item.file_size for item in packed.infolist())>maximum_saved_vertices*160+1048576:
            raise MemoryError('Stream timeline archive exceeds display allocation')
    with np.load(archive,allow_pickle=False) as packed:
        xyz=packed['position_kpc'];ids=packed['particle_ids'];times=packed['times_myr']
    if xyz.ndim!=3 or xyz.shape[2]!=3 or xyz.shape[:2]!=(len(times),len(ids)) or xyz.shape[0]*xyz.shape[1]>maximum_saved_vertices:
        raise ValueError('Stream timeline dimensions or vertex allocation differ')
    if ids.dtype!=np.dtype('uint64') or len(np.unique(ids))!=len(ids) or not np.isfinite(xyz).all():
        raise ValueError('Finite timeline vectors and unique exact identities required')
    if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times.tolist()!=record['saved_times_myr']:
        raise ValueError('Stream timeline lacks its saved physical clock')
    if record.get('length_units')!='proper kpc' or record.get('time_units')!='Myr':
        raise ValueError('Stream timeline coordinate units differ')
    saved_frames,frame_end=saved_stream_frame_schedule(times.tolist(),scene_frames=scene_frames,scene_frame_end=scene_frame_end)
    scales=[1.]*len(times) if coordinate_scales is None else list(coordinate_scales)
    if len(scales)!=len(times) or any(type(scale) not in (float,int) or not math.isfinite(scale) or scale<=0 for scale in scales):raise ValueError('Positive explicit coordinate scale per saved stream epoch required')
    scene=bpy.context.scene
    if scene_frames is not None and scene.frame_end!=frame_end:raise ValueError('Stream frame binding differs from current combined scene end')
    # The exported arrays are sufficient to move a display bundle. The original
    # scientific product remains a provenance link, rather than a load dependency.
    collection=bpy.data.collections.new('Saved stream '+record['namespace'])
    scene.collection.children.link(collection)
    if scene_frames is None:scene.frame_start=1;scene.frame_end=frame_end
    scene['stream_saved_times_myr']=times.tolist()
    scene['stream_saved_frames']=saved_frames
    scene['stream_timeline_policy']='Explicit saved frame shows original epoch; hold until next saved frame; no interpolation'
    scene['stream_coordinate_semantics']=str(record.get('coordinate_semantics'))
    scene['stream_inspector_manifest']=str(path)
    group=bpy.data.node_groups.new('Saved stream marker instances','GeometryNodeTree')
    if hasattr(group,'interface'):
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    else:
        group.inputs.new('NodeSocketGeometry','Geometry');group.outputs.new('NodeSocketGeometry','Geometry')
    nodes=group.nodes;links=group.links
    inlet=nodes.new('NodeGroupInput');outlet=nodes.new('NodeGroupOutput')
    points=nodes.new('GeometryNodeMeshToPoints');sphere=nodes.new('GeometryNodeMeshIcoSphere')
    sphere.inputs['Radius'].default_value=marker_radius_kpc;sphere.inputs['Subdivisions'].default_value=1
    instances=nodes.new('GeometryNodeInstanceOnPoints')
    links.new(inlet.outputs['Geometry'],points.inputs['Mesh']);links.new(points.outputs['Points'],instances.inputs['Points'])
    links.new(sphere.outputs['Mesh'],instances.inputs['Instance']);links.new(instances.outputs['Instances'],outlet.inputs['Geometry'])
    for ordinal,positions in enumerate(xyz):
        mesh=bpy.data.meshes.new('Saved stream epoch %04d'%ordinal)
        mesh.from_pydata(positions.tolist(),[],[]);mesh.update()
        obj=bpy.data.objects.new('Stream t=%.9g Myr'%times[ordinal],mesh);collection.objects.link(obj)
        obj.scale=(scales[ordinal],)*3
        modifier=obj.modifiers.new('Physical tracer markers','NODES');modifier.node_group=group
        obj['stream_namespace']=record['namespace'];obj['exact_identity_archive']=str(archive)
        obj['exact_identity_preview']=','.join(str(int(value)) for value in ids[:16])
        obj['saved_epoch_index']=ordinal;obj['saved_time_myr']=float(times[ordinal])
        obj['particle_count']=len(ids);obj['source_product_sha256']=record['product_sha256']
        obj['science_acceptance']=record['science_acceptance'];obj['coordinate_units']='proper kpc'
        obj['display_coordinate_scale']=scales[ordinal]
        obj['display_coordinate_units']='comoving Mpc/h' if coordinate_scales is not None else 'proper kpc'
        first=saved_frames[ordinal];last=saved_frames[ordinal+1]-1 if ordinal+1<len(saved_frames) else frame_end
        key_object_visibility(obj,first,last,frame_end)
    if scene_frames is None:scene.frame_set(1)
    return collection


SCENE_COMMANDS['stream-inspector']=stream_inspector_main
if __name__=='__main__':main()


def export_stream_inspector(product_path,destination,*,memory_gib,epoch_index=-1):
    """Retain uint64 identities separately from Blender float attributes."""
    import numpy as np
    from core.stream_common import read_product, json_safe, digest
    from core.foundation import atomic_copy, atomic_json
    report,arrays,sources=read_product(product_path,memory_gib=memory_gib)
    output=Path(destination)
    output.mkdir(parents=True,exist_ok=False)
    tracks=[]
    track_support={}
    epoch=report.get('epoch')
    if arrays['position_kpc'].ndim==3:
        if 'epochs' in report:
            if len(report['epochs'])!=len(arrays['position_kpc']):
                raise ValueError('Inspector epoch records must align with the saved trajectory')
            epoch=report['epochs'][epoch_index]
        x=arrays['position_kpc'][epoch_index]
        v=arrays['velocity_kms'][epoch_index]
        mass=arrays['mass_msun'][epoch_index] if arrays['mass_msun'].ndim==2 else arrays['mass_msun']
    else:
        x=arrays['position_kpc']
        v=arrays['velocity_kms']
        mass=arrays['mass_msun']
    values={'position_kpc':x,'velocity_kms':v,'particle_ids':arrays['particle_ids'],'mass_msun':mass}
    for key in ('candidate_id','wrap_id','bound'):
        if key in arrays:
            values[key]=arrays[key][epoch_index] if arrays[key].ndim==2 else arrays[key]
    for key in arrays:
        if key.endswith('curve_position_kpc'):
            support_name=key.removesuffix('curve_position_kpc')+'curve_support'
            points=np.asarray(arrays[key]);support=np.asarray(arrays[support_name])
            if points.ndim!=2 or points.shape[1]!=3 or support.dtype.kind!='b' or support.shape!=(len(points),):
                raise ValueError('Stream principal curve needs an aligned explicit Boolean support mask')
            if not np.isfinite(points[support]).all():
                raise ValueError('Supported principal-curve coordinates must be finite')
            values[key]=points
            values[support_name]=support
            tracks.append(key)
            track_support[key]=support_name
    np.savez_compressed(output/'inspector.npz',**values)
    # Bundle the exact scientific report bytes needed to identify this display.
    # The selected display arrays are in inspector.npz; this metadata copy is
    # not a replacement for the full original numeric audit and source files.
    source_path=Path(product_path).resolve()
    expected=sources[str(source_path)]
    if atomic_copy(source_path,output/'source_product.json')!=expected:
        raise ValueError('Stream product changed after verified intake')
    schema=report.get('schema')
    origin={'stream-gala-reference-v1':'Gala model tracers',
            'streamamax-reference-v1':'StreaMAX model tracers',
            'stream-spray-v1':'Research2 model spray tracers',
            'live-stream-resimulation-v1':'resimulated progenitor particles',
            'tidal-stream-catalogue-v1':'native stellar catalogue particles'}.get(schema,'UNSPECIFIED')
    if schema=='measured-native-stream-worldlines-v1':
        origin=('native GADGET stellar particles' if report.get('request',{}).get('species')==4
                else 'native GADGET particles; species in source report')
    manifest={'schema':'stream-inspector-v1','arrays_sha256':digest(output/'inspector.npz'),'product':str(Path(product_path).resolve()),
        'product_sha256':digest(product_path),'namespace':report['namespace'],'epoch':epoch,'tracks':tracks,'track_support':track_support,
        'packaged_product':'source_product.json','particle_origin':origin,
        'length_units':'proper kpc','velocity_units':'km/s','identity_encoding':'uint64 archive and decimal-string inspector IDs',
        'source_hashes':sources,'science_acceptance':'NOT_ACCEPTED','classification':report.get('candidates'),'wraps':report.get('wraps')}
    if manifest['product_sha256']!=expected:
        raise ValueError('Stream product changed before inspector publication')
    if any(digest(path)!=checksum for path,checksum in sources.items()):
        raise ValueError('Stream product or arrays changed during inspector export')
    atomic_json(output/'inspector.json',json_safe(manifest))
    return manifest
