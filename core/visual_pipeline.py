"""Saved-run visualization dispatch with independent, explicitly recorded failures."""
from pathlib import Path

VISUAL_PRODUCTS = {
    'movie': 'Dark-matter cosmic-web movie from saved outputs',
    'gas_movie': 'Gas density movie from saved outputs',
    'temperature_movie': 'Measured gas temperature movie from saved outputs',
    'stars_movie': 'Stellar density movie from saved outputs',
    'halo_movie': 'Measured halo growth movie',
    'xdmf': 'Portable particle and field XDMF/HDF5 export',
    'planetarium': 'Domemaster timeline, numeric rasters and MP4',
    'blender_movie': 'Interactive Blender scene, previews and animated MP4',
}


def export_planetarium_timeline(run_dir, cfg):
    """Publish a fresh finite sequence with exact source frame and movie hashes."""
    import uuid
    import imageio.v2 as imageio
    from core.foundation import atomic_json, file_sha256
    from visualization import export_domemaster, snapshot_ids
    run_dir = Path(run_dir)
    v = cfg['visualization']
    options = dict(v.get('planetarium', {}))
    resolution = options.get('resolution', 2048)
    budget = options.get('max_particles', v.get('blender_sequence_max_points', 300000))
    fps = v.get('movie_fps', 24)
    if (type(resolution) is not int or not 16 <= resolution <= 8192 or
            type(budget) is not int or budget < 1 or type(fps) is not int or fps < 1):
        raise ValueError('Planetarium resolution, particle budget and fps must be valid positive integers')
    snapshots = snapshot_ids(run_dir)
    if len(snapshots) < 2:
        raise ValueError('Planetarium timeline requires at least two saved outputs')
    base = run_dir/'results/planetarium';base.mkdir(parents=True, exist_ok=True)
    # export_domemaster registers paths relative to the run. Keep its fresh
    # directory stable rather than moving it after registration.
    destination = base/('timeline_'+uuid.uuid4().hex);destination.mkdir()
    records=[]
    try:
        for index, snap in enumerate(snapshots):
            path=destination/f'frame_{index:05d}.png'
            result=export_domemaster(run_dir, snap, path, field=options.get('field','dm'),
                resolution=resolution, max_particles=budget,
                observer_mpc_h=options.get('observer_mpc_h'), tilt=options.get('tilt',27.))
            records.append({'snapshot':snap,'path':path.name,'sha256':file_sha256(path),
                            'source':result['source']})
            atomic_json(destination/'progress.json',{'status':'RUNNING','frames':records})
        movie=destination/'planetarium.mp4'
        writer=imageio.get_writer(movie, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        try:
            for row in records:
                frame=destination/row['path']
                if file_sha256(frame)!=row['sha256']:raise ValueError('Planetarium frame changed')
                writer.append_data(imageio.imread(frame))
        finally:
            writer.close()
        if not movie.is_file() or movie.stat().st_size == 0:raise RuntimeError('Planetarium encoder produced no movie')
        report={'status':'COMPLETE','schema':'planetarium-timeline-v1','science_acceptance':'NOT_ACCEPTED',
            'frames':records,'fps':fps,'options':options,'resolution':resolution,'max_particles':budget,
            'time_representation':'Equal frame spacing at saved outputs; source records retain physical epochs',
            'movie':{'path':movie.name,'sha256':file_sha256(movie),'size_bytes':movie.stat().st_size}}
        atomic_json(destination/'timeline.json',report)
        atomic_json(destination/'progress.json',{'status':'COMPLETE','frame_count':len(records)})
        return str(destination/'timeline.json')
    except BaseException as exc:
        atomic_json(destination/'progress.json',{'status':'FAILED','frames':records,
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


def execute_visual_products(run_dir, selected, cfg, status, checkpoint, *, radial_shapes=None):
    """Finish independent exports even if another renderer fails; never label failures complete."""
    import visualization as visual
    from commands import export, build_scene
    v=cfg['visualization']

    def portable(value):
        if isinstance(value, Path):return str(value)
        if isinstance(value, (list,tuple)):return [portable(item) for item in value]
        if isinstance(value, dict):return {key:portable(item) for key,item in value.items()}
        return value

    def perform(name, operation):
        checkpoint(name)
        try:
            result=operation()
            if result is None or result == []:
                raise RuntimeError('No requested output was produced')
            status['artifacts'][name]=portable(result)
        except Exception as exc:
            status['unavailable'][name]=f'{type(exc).__name__}: {exc}'
            status.setdefault('visual_failures',{})[name]={'type':type(exc).__name__,'message':str(exc)}
        checkpoint(name+'_finished')

    if 'maps' in selected:
        def maps():
            import json
            paths=[str(p) for p in visual.make_maps(run_dir,v['map_fields'],v['map_projections'])]
            # Retain successful density maps while surfacing an independently
            # failed morphology/story renderer in the parent workflow record.
            for folder in {Path(p).parent for p in paths}:
                scale=folder/'scale.json'
                if not scale.is_file():continue
                story=json.loads(scale.read_text()).get('story_figures')
                if isinstance(story,dict) and story.get('status') != 'COMPLETE':
                    status['unavailable']['map_story_figures']=story.get('reason') or 'Map story renderer unavailable'
            return paths
        perform('maps',maps)
    for name,field in [('movie','dm'),('gas_movie','gas'),('temperature_movie','gas_temperature'),('stars_movie','stars')]:
        if name in selected:
            def movie(field=field):
                result=visual.make_movie(run_dir,field=field)
                return str(result) if result is not None else None
            perform(name,movie)
    if 'halo_movie' in selected:
        def halo_movie():
            directory=status['artifacts'].get('analysis')
            return visual.make_halo_growth_movie(run_dir,analysis_path=Path(directory)/'analysis_products.h5' if directory else None)
        perform('halo_movie',halo_movie)
    if 'xdmf' in selected:
        perform('xdmf',lambda:visual.make_xdmf(run_dir))
    if 'planetarium' in selected:
        perform('planetarium',lambda:export_planetarium_timeline(run_dir,cfg))
    if 'blender' in selected or 'blender_movie' in selected:
        def blender():
            manifest=export(run_dir,int(v['blender_sequence_max_points']),v['blender_render_quality'],
                            int(v['blender_frames_per_snapshot']),radial_shapes=radial_shapes,
                            analysis_directory=status['artifacts'].get('analysis'))
            status['artifacts']['blender']=str(manifest)
            build_scene(manifest,preview=True,render='blender_movie' in selected)
            status['artifacts']['blender_scene']=str(manifest.parent/'blender_universe.blend')
            status['artifacts']['blender_preview']=str(manifest.parent/'blender_preview.png')
            if 'blender_movie' in selected:
                status['artifacts']['blender_movie']=str(manifest.parent/'blender_universe_movie.mp4')
            return str(manifest)
        perform('blender_movie' if 'blender_movie' in selected else 'blender',blender)
