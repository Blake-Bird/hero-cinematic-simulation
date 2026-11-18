#!/usr/bin/env python3
"""Publish browser-ready copies of the useful retained N128 films with provenance."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256


def publish(source, destination, title, model, redshifts):
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (destination.name + '.mp4')
    if output.exists() and (destination / 'film_source.json').exists():
        return
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                                  '-show_streams', '-of', 'json', str(source)]))['streams']
    stream, = streams
    encoding = ['-c:v', 'copy'] if (stream['codec_name'], stream['pix_fmt']) == ('h264', 'yuv420p') else [
        '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p']
    temp = output.with_suffix('.staged.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-map', '0:v:0',
                    *encoding, '-movflags', '+faststart', str(temp)], check=True)
    temp.replace(output)
    sidecars = {}
    for p in sorted(source.parent.glob('*.json')):
        if p.name.startswith('frame_') or p.name.endswith('.publication.json'):
            continue
        folder = destination / 'Retained_source_records'
        folder.mkdir(exist_ok=True)
        shutil.copy2(p, folder / p.name)
        sidecars[p.name] = file_sha256(p)
    atomic_json(destination / 'film_source.json', {
        'title': title, 'model': model, 'redshifts': redshifts,
        'generation': 'Retained simulation export, published in a browser-ready container',
        'source': str(source), 'source_sha256': file_sha256(source),
        'retained_source_records': sidecars, 'published_sha256': file_sha256(output)})
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    entries = [
        ('movies/launcher_density_*/density.mp4', 'Projected density timeline'),
        ('movies/launcher_density_*/density_slow.mp4', 'Projected density timeline slow'),
        ('movies/halo_growth_*/halo_growth_mergers.mp4', 'Halo growth and merger markers'),
        ('planetarium/timeline_*/planetarium.mp4', 'Planetarium domemaster'),
        ('planetarium/timeline_*/planetarium_slow.mp4', 'Planetarium domemaster slow'),
        ('3d/universe_*/blender_cinematic.mp4', 'Blender cinematic'),
        ('3d/universe_*/blender_universe_movie.mp4', 'Blender universe timeline')]
    for model in ('EDE', 'LCDM'):
        run = root.parent / f'dmo3_N128_L5_fast128_{model.lower()}'
        movies = root / '04_Movies' / model
        for pattern, title in entries:
            source, = (run / 'results').glob(pattern)
            output = movies / f'{model}_z_199_to_000_{title.replace(" ", "_")}'
            publish(source, output, title, model, 'Saved timeline, z=199 to z=0. Inspect retained source records for displayed frames.')
        if model == 'EDE':
            for folder in sorted((run / 'analysis/05_Movies_and_3D/Readable_flights').iterdir()):
                if not folder.is_dir():
                    continue
                source, = folder.rglob('cosmic_flythrough.mp4')
                redshift = '014p400' if '14.4' in folder.name else '000p000'
                title = 'Filament tour' if 'Filament' in folder.name else 'Halo approach'
                publish(source, movies / f'EDE_z_{redshift}_{title.replace(" ", "_")}', title, model,
                        f'Fixed saved epoch z={14.4 if redshift == "014p400" else 0}.')

    # Browser dome views are rebuilt from verified numeric rasters; retain native exports.
    from scripts.rebuild_paired_dome_views import rebuild
    rebuild(root)


if __name__ == '__main__':
    main()
