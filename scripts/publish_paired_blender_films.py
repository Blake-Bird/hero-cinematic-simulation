#!/usr/bin/env python3
"""Edit the exact Blender timeline renders into narrated-by-label MP4s."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    args = parser.parse_args()
    root = args.atlas.resolve()
    directory = root / 'Source/Blender_film_frames'
    models = ('EDE', 'LCDM')
    frames = {m: json.loads((directory/m/'saved_frames.json').read_text()) for m in models}
    if [e['snapshot'] for e in frames['EDE']] != [e['snapshot'] for e in frames['LCDM']]:
        raise ValueError('The Blender films use different saved epochs')
    font_dir = Path(matplotlib.get_data_path())/'fonts/ttf'
    large = ImageFont.truetype(str(font_dir/'DejaVuSerif.ttf'), 44)
    medium = ImageFont.truetype(str(font_dir/'DejaVuSans.ttf'), 24)
    small = ImageFont.truetype(str(font_dir/'DejaVuSans.ttf'), 19)
    for group, selection in [('EDE', ('EDE',)), ('LCDM', ('LCDM',)), ('Comparisons', models)]:
        prefix = group if group != 'Comparisons' else 'EDE_vs_LCDM'
        name = f'{prefix}_z_199_to_000_Blender_saved_particle_cinematic'
        out = root/'04_Movies'/group/name
        out.mkdir(parents=True, exist_ok=True)
        film = out/(name+'.mp4')
        temp = out/('.'+film.name)
        records = []
        with imageio.get_writer(temp, fps=24, codec='libx264', quality=9, pixelformat='yuv420p',
                                ffmpeg_params=['-movflags','+faststart']) as writer:
            for index, epoch in enumerate(frames['EDE']):
                snap, z = epoch['snapshot'], 1/epoch['scale_factor'] - 1
                pixels = Image.new('RGB', (1536, 896), '#0b1923')
                draw = ImageDraw.Draw(pixels)
                draw.text((38, 22), 'The dark matter sky', font=large, fill='#f5e7cb')
                draw.text((40, 79), f'Saved output {snap:03d}  /  z = {z:.3f}  /  7.37354 comoving Mpc per side', font=medium, fill='#adc6d4')
                refs = {}
                for m in selection:
                    offset = (40 if m == 'EDE' else 794) if len(selection) == 2 else 408
                    path = directory/m/f'snapshot_{snap:03d}.png'
                    with Image.open(path) as source:
                        pixels.paste(source.convert('RGB').resize((704,704), Image.Resampling.LANCZOS), (offset,137))
                    draw.text((offset, 108), m if m == 'EDE' else 'ΛCDM', font=medium, fill='#eddfc7')
                    refs[m] = {'file': str(path.relative_to(root)), 'sha256': file_sha256(path),
                               'blender_frame': frames[m][index]['frame']}
                draw.text((40, 858), 'Blender render · sampled native dark matter · fixed display palette · camera moves slowly across saved outputs',
                          font=small, fill='#adc6d4')
                arr = np.asarray(pixels)
                hold = 26 if snap in (21,26,30,35,41,54,69,85) else 6
                for _ in range(hold):
                    writer.append_data(arr)
                if snap == 21:
                    pixels.save(out/'Poster.png')
                records.append({'snapshot': snap, 'redshift': z, 'hold_frames': hold, 'sources': refs})
        temp.replace(film)
        atomic_json(out/'film_source.json', {'title': 'The dark matter sky · Blender cinematic', 'model': group,
                    'redshifts': 'All 86 saved epochs, z=199 to z=0', 'frames': records,
                    'source_scene': {m: str(next((root/'05_Blender_and_3D'/m).glob('*all_saved*/blender_universe.blend')).relative_to(root)) for m in selection},
                    'frame_recipe_sha256': file_sha256(Path(__file__).with_name('render_paired_scene_frames.py')),
                    'semantics': 'Original scene sampled particle geometry at exact saved frames. Artificial glyph and emission palette.',
                    'published_sha256': file_sha256(film)})
        print(film, flush=True)


if __name__ == '__main__':
    main()
