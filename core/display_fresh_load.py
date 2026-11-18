"""Executed inside a freshly loaded Blender process only when requested."""
import hashlib
import json
from pathlib import Path
import sys
import bpy


def main():
    output = Path(sys.argv[sys.argv.index('--')+1]).resolve()
    scene = bpy.context.scene
    manifest = json.loads(scene['research_display_manifest'])
    # Run the packed handler source explicitly: proof does not assume auto-run.
    text = bpy.data.texts['research_display_runtime.py']
    namespace = {'__name__': 'research_display_runtime'}
    exec(compile(text.as_string(), text.name, 'exec'), namespace)
    inspector = bpy.data.texts['research_object_inspector.py']
    exec(compile(inspector.as_string(),inspector.name,'exec'),{'__name__':'research_object_inspector'})
    rows = []
    timeline_controls = []
    owner = scene.get('research_display_owner')
    def visible_identities():
        return sorted({obj['research_identity'] for obj in scene.objects
            if 'research_epoch_index' in obj and obj.get('research_display_owner') == owner and not obj.hide_viewport and not obj.hide_render})
    def visible_geometry():
        return sorted((obj['research_identity'],obj.get('research_geometry_layer','primary'),obj['research_kind'])
            for obj in scene.objects if 'research_epoch_index' in obj and obj.get('research_display_owner') == owner
            and not obj.hide_viewport and not obj.hide_render)
    def expected_geometry(epoch):
        result = []
        for record in epoch['objects']:
            identity = record['namespace']+':'+str(record['object_id'])
            result.append((identity,'primary',record['kind']))
            result.extend((identity,'layer:'+layer['kind'],layer['kind']) for layer in record.get('measured_geometry_layers',[]))
        return sorted(result)
    def timeline_control(frame,expected,semantics,geometry):
        scene.frame_set(frame)
        observed = visible_identities()
        rendered = visible_geometry()
        timeline_controls.append({'frame':frame,'semantics':semantics,'expected':expected,'observed':observed,
            'expected_geometry':geometry,'observed_geometry':rendered,
            'status':'PASS' if observed == expected and rendered == geometry else 'FAIL'})
    for index, epoch in enumerate(manifest['timeline']):
        scene.frame_set(epoch['frame'])
        expected = sorted(q['namespace']+':'+str(q['object_id']) for q in epoch['objects'])
        observed = visible_identities()
        rows.append({'frame': epoch['frame'], 'snapshot': epoch['snapshot'], 'expected': expected,
                     'observed': observed,'expected_geometry':expected_geometry(epoch),'observed_geometry':visible_geometry(),
                     'status': 'PASS' if observed == expected and visible_geometry() == expected_geometry(epoch) else 'FAIL'})
        if index+1 < len(manifest['timeline']) and manifest['timeline'][index+1]['frame'] > epoch['frame']+1:
            timeline_control(manifest['timeline'][index+1]['frame']-1,expected,'hold last measured state',expected_geometry(epoch))
    timeline_control(manifest['timeline'][0]['frame']-1,[],'before measured coverage',[])
    timeline_control(manifest['timeline'][-1]['frame']+1,[],'after measured coverage',[])
    scene.frame_set(manifest['timeline'][0]['frame'])
    base = Path(bpy.path.abspath(scene['research_display_base'])).resolve()
    asset_rows = []
    for asset in manifest['assets']:
        path = (base/asset['path']).resolve()
        current = None
        if path.is_relative_to(base) and path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(chunk)
            current = digest.hexdigest()
        asset_rows.append({'path': asset['path'], 'status': 'PASS' if current == asset['sha256'] else 'FAIL'})
    blend = Path(bpy.data.filepath)
    inspector_registered = hasattr(bpy.types,'RESEARCH_PT_measured') and hasattr(bpy.types,'RESEARCH_OT_open_measured_source')
    result = {'schema': 'blender-fresh-load-v1', 'execution_status': 'COMPLETE',
        'status': 'PASS' if inspector_registered and all(q['status'] == 'PASS' for q in rows+asset_rows+timeline_controls) else 'FAIL',
        'blend_sha256': hashlib.sha256(blend.read_bytes()).hexdigest(), 'epochs': rows,
        'assets': asset_rows,'timeline_controls':timeline_controls,
        'inspector_registered':inspector_registered, 'endpoint_frames': [manifest['timeline'][0]['frame'], manifest['timeline'][-1]['frame']]}
    output.write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] != 'PASS':
        raise RuntimeError('Fresh-load timeline control failed')

if __name__ == '__main__':
    main()
