"""Render a sealed saved scene with explicit display and allocation settings."""
from pathlib import Path
import json
import math


def validate_render_request(request):
    required={'schema','scene','scene_name','projection','width','height','samples','frame_start','frame_end','fps','device','memory_gib','disk_gib'}
    if set(request)!=required or request['schema']!='saved-scene-render-request-v1':raise ValueError('Complete saved scene render request required')
    for name in ['width','height','samples','frame_start','frame_end','fps']:
        if type(request[name]) is not int or request[name]<1:raise ValueError('Positive exact render '+name+' required')
    for name in ['memory_gib','disk_gib']:
        value=request[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive explicit render allocation required')
    if request['frame_end']<request['frame_start']:raise ValueError('Ordered render frames required')
    if request['projection'] not in ['active','perspective','dome']:raise ValueError('Explicit supported render projection required')
    if request['projection']=='dome' and request['width']!=request['height']:raise ValueError('Equidistant dome output requires a square raster')
    if request['device'] not in ['AUTO','CPU','CUDA','OPTIX','HIP','ONEAPI','METAL']:raise ValueError('Supported explicit render device required')
    if request['scene_name'] is not None and not isinstance(request['scene_name'],str):raise ValueError('Exact saved scene name or explicit active selection required')
    if set(request['scene'])!={'path','sha256'}:raise ValueError('Exact sealed saved scene path/hash required')
    pixels=request['width']*request['height'];frames=request['frame_end']-request['frame_start']+1
    # Provisional conservative raster/workspace bounds, separate from actual
    # native renderer RSS and scene geometry. Original scene remains unchanged.
    if pixels*128>request['memory_gib']*1024**3*.5:raise MemoryError('Render raster workspace exceeds allocation')
    if pixels*8*frames>request['disk_gib']*1024**3*.8:raise MemoryError('Conservative PNG sequence storage exceeds allocation')


def render_saved_scene(request,destination, *, resume=False):
    """Render a sealed scene, optionally continuing a verified frame checkpoint.

    An unrecorded frame is never overwritten: it may be the result of an
    interrupted Blender write. Preserve it elsewhere before attempting recovery.
    """
    import bpy
    import uuid
    from .foundation import atomic_json,atomic_text,file_sha256
    from .blender import verify_scene_publication,configure_render_device,disable_cycles_denoising
    validate_render_request(request)
    source=Path(request['scene']['path']).resolve();out=Path(destination).resolve()
    if file_sha256(source)!=request['scene']['sha256']:raise ValueError('Saved scene differs from requested original bytes')
    publication=verify_scene_publication(source)
    progress_path=out/'progress.json'
    runtime={'blender_version':bpy.app.version_string,'renderer_sha256':file_sha256(Path(__file__))}
    if resume:
        if not progress_path.is_file():raise FileNotFoundError('Resume requires an existing progress.json checkpoint')
        progress=json.loads(progress_path.read_text())
        if (progress.get('schema')!='saved-scene-render-progress-v1' or
                progress.get('request')!=request or progress.get('source_publication')!=publication or
                progress.get('runtime')!=runtime):
            raise ValueError('Resume requires the same request, sealed scene, Blender version and renderer source')
        if progress.get('status')=='COMPLETE':raise ValueError('Render already completed; preserve its receipt')
        if (out/'render.json').exists():raise FileExistsError('Render receipt already exists; inspect it before recovery')
        artifacts=progress.get('artifacts')
        if not isinstance(artifacts,list):raise ValueError('Invalid render checkpoint artifacts')
        expected=list(range(request['frame_start'],request['frame_start']+len(artifacts)))
        if [item.get('frame') for item in artifacts]!=expected or len(artifacts)>request['frame_end']-request['frame_start']+1:
            raise ValueError('Render checkpoint does not contain a contiguous frame prefix')
        for item in artifacts:
            path=out/('frame_%08d.png'%item['frame'])
            if item.get('path')!=path.name or not path.is_file() or file_sha256(path)!=item.get('sha256'):
                raise ValueError('Recorded render frame is missing or has changed')
    else:
        if out.exists():raise FileExistsError('Preserve earlier render output; select a fresh directory')
        out.mkdir(parents=True)
        artifacts=[]
        progress={'schema':'saved-scene-render-progress-v1','status':'RUNNING','request':request,
                  'source_publication':publication,'runtime':runtime,'artifacts':artifacts}
        atomic_json(progress_path,progress)
    try:
        bpy.ops.wm.open_mainfile(filepath=str(source),use_scripts=False)
        # Run only the application's named packed runtimes from the already
        # verified scene. This does not depend on user auto-run preferences.
        for name in ['dmsim_inspector.py','research_display_runtime.py','research_object_inspector.py']:
            text=bpy.data.texts.get(name)
            if text is not None:
                namespace={'__name__':name.removesuffix('.py')}
                exec(compile(text.as_string(),name,'exec'),namespace)
                if callable(namespace.get('register')):namespace['register']()
        scene=bpy.data.scenes.get(request['scene_name']) if request['scene_name'] is not None else bpy.context.scene
        if scene is None:raise ValueError('Requested scene is absent from saved file')
        if request['frame_start']<scene.frame_start or request['frame_end']>scene.frame_end:raise ValueError('Render request extends beyond saved scene coverage')
        if request['projection']=='perspective':
            choices=[obj for obj in scene.objects if obj.type=='CAMERA' and obj.data.type=='PERSP']
            hero=next((obj for obj in choices if obj.name=='Cinematic_Hero_Camera'),None)
            if hero is None and len(choices)!=1:raise ValueError('Saved scene has no unambiguous perspective camera')
            scene.camera=hero or choices[0]
        elif request['projection']=='dome':
            choices=[obj for obj in scene.objects if obj.type=='CAMERA' and obj.data.type=='PANO' and obj.name=='Fulldome_Camera']
            if len(choices)!=1:raise ValueError('Saved scene lacks its original equidistant dome camera')
            scene.camera=choices[0]
        if scene.camera is None:raise ValueError('Saved scene has no selected camera')
        scene.render.engine='CYCLES';device=configure_render_device(bpy,scene,request['device'])
        scene.render.resolution_x=request['width'];scene.render.resolution_y=request['height'];scene.render.resolution_percentage=100
        scene.cycles.samples=request['samples'];scene.render.fps=request['fps'];scene.render.fps_base=1.;scene.render.image_settings.file_format='PNG'
        scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.005
        # Some Blender builds have no OpenImageDenoise support.  Do not force
        # a backend that may only fail once Cycles starts rendering.
        disable_cycles_denoising(scene)
        scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='16';scene.render.use_file_extension=True
        if resume and progress.get('device') is not None and progress['device']!=device:
            raise ValueError('Render device differs from checkpoint; use a fresh output')
        progress['device']=device;progress['status']='RUNNING';atomic_json(progress_path,progress)
        written_bytes=sum((out/item['path']).stat().st_size for item in artifacts)
        if written_bytes>request['disk_gib']*1024**3*.8:raise MemoryError('Existing render outputs exceed allocation')
        for frame in range(request['frame_start']+len(artifacts),request['frame_end']+1):
            scene.frame_set(frame);path=out/('frame_%08d.png'%frame);scene.render.filepath=str(path)
            if path.exists():raise FileExistsError('Unrecorded render frame exists; preserve it elsewhere before resuming: '+str(path))
            bpy.ops.render.render(write_still=True,scene=scene.name)
            if not path.is_file():raise FileNotFoundError('Blender did not publish requested frame')
            artifacts.append({'frame':frame,'path':path.name,'sha256':file_sha256(path)})
            written_bytes+=path.stat().st_size
            atomic_json(progress_path,progress)
            if written_bytes>request['disk_gib']*1024**3*.8:raise MemoryError('Actual render outputs exceed allocation')
        if file_sha256(source)!=request['scene']['sha256'] or verify_scene_publication(source)!=publication:raise ValueError('Source scene publication changed during rendering')
        receipt={'schema':'saved-scene-render-v1','execution_status':'COMPLETE','request':request,'source_publication':publication,'scene_name':scene.name,
            'camera':scene.camera.name,'camera_type':scene.camera.data.type,'device':device,'runtime':runtime,'artifacts':artifacts,'science_acceptance':'NOT_ACCEPTED',
            'display_quality':{'png_channel_bits':16,'adaptive_threshold':.005,'denoising':False,'maximum_samples':request['samples']},
            'scope':'Original saved geometry, clocks and identities rendered as a PNG animation sequence; projection/resolution/sampling are display settings. Raster estimates do not certify scene RSS or resolved science.'}
        atomic_json(out/'render.json',receipt);atomic_text(out/'render.json.sha256',file_sha256(out/'render.json')+'\n')
        progress['status']='COMPLETE';atomic_json(progress_path,progress)
        return receipt
    except BaseException as error:
        failure={'status':'FAIL','science_acceptance':'NOT_ACCEPTED','type':type(error).__name__,'message':str(error)}
        atomic_json(out/('failure_'+uuid.uuid4().hex+'.json'),failure)
        if not (out/'failure.json').exists():atomic_json(out/'failure.json',failure)
        progress['status']='FAILED';atomic_json(progress_path,progress)
        raise



def prepare_render_main():
    """Render sealed Blender scenes with explicit projection, resolution and frames."""
    import argparse
    parser = argparse.ArgumentParser(description=prepare_render_main.__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--projection', choices=['active', 'perspective', 'dome'], required=True)
    parser.add_argument('--scene-name')
    for name in ['width', 'height', 'samples', 'frame-start', 'frame-end', 'fps']:
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--memory-gib', required=True, type=float)
    parser.add_argument('--disk-gib', required=True, type=float)
    parser.add_argument('--device', default='AUTO', choices=['AUTO', 'CPU', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'])
    parser.add_argument('--wall-seconds', required=True, type=int)
    parser.add_argument('--blender', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    from core.foundation import atomic_json, file_sha256
    from core.native import run_native_command
    source = args.scene.resolve()
    out = args.output.resolve()
    request = {'schema': 'saved-scene-render-request-v1', 'scene': {'path': str(source), 'sha256': file_sha256(source)}, **{key: getattr(args, key) for key in ['projection', 'scene_name', 'width', 'height', 'samples', 'frame_start', 'frame_end', 'fps', 'device', 'memory_gib', 'disk_gib']}}
    validate_render_request(request)
    from core.blender import verify_scene_publication
    verify_scene_publication(source)
    if args.wall_seconds < 1:
        raise ValueError('Positive render wall-time limit required')
    if out.exists():
        raise FileExistsError('Preserve previous render preparation')
    from core.visualization import _find_blender_executable
    binary = str(args.blender.resolve()) if args.blender else _find_blender_executable()
    if not binary or not Path(binary).is_file():
        raise FileNotFoundError('Blender executable required for prepared rendering')
    out.mkdir(parents=True)
    atomic_json(out / 'request.json', request)
    script = Path(__file__).resolve()
    command = [binary, '--background', '--python-exit-code', '1', '--python', str(script), '--', str(out / 'request.json'), '--output', str(out / 'frames')]
    atomic_json(out / 'launch.json', {'command': command, 'wall_seconds': args.wall_seconds, 'status': 'PREPARED', 'source_sha256': {str(script): file_sha256(script)}})
    if not args.prepare_only:
        with (out / 'render.log').open('wb') as stream:
            run_native_command(command, cwd=out, receipt_path=out / 'invocation.json', inputs=[source, script, out / 'request.json'], timeout=args.wall_seconds, stdout=stream, stderr=stream)
    print(json.dumps({'status': 'PREPARED' if args.prepare_only else 'COMPLETE', 'output': str(out), 'frames': str(out / 'frames')}))

def main():
    import argparse,sys
    arguments=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('request',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resume',action='store_true',help='Continue a verified incomplete frame checkpoint')
    args=parser.parse_args(arguments);render_saved_scene(json.loads(args.request.read_text()),args.output,resume=args.resume)


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from core.scene_render import main
    main()
