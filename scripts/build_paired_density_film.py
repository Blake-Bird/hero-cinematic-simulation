#!/usr/bin/env python3
"""Create a common-exposure saved-epoch comparison from the paired measurements."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    data = root / '06_Tables_and_Data/Measurements'
    catalogues = [{row['summary']['snapshot']: row for row in json.loads(
        (data / model / 'measurement_manifest.json').read_text())['epochs']} for model in ('EDE', 'LCDM')]
    if catalogues[0].keys() != catalogues[1].keys():
        raise ValueError('Paired timeline snapshot sets differ')
    out = root / '04_Movies/Comparisons/EDE_vs_LCDM_z_199_to_000_all_saved_epoch_density'
    out.mkdir(parents=True, exist_ok=True)
    film = out / (out.name + '.mp4')
    stage = out / ('.' + film.name)
    fonts = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    title = ImageFont.truetype(str(fonts / 'DejaVuSerif.ttf'), 36)
    label = ImageFont.truetype(str(fonts / 'DejaVuSans.ttf'), 20)
    small = ImageFont.truetype(str(fonts / 'DejaVuSans.ttf'), 15)
    cmap = matplotlib.colormaps['magma']
    entries = []
    with imageio.get_writer(stage, fps=24, codec='libx264', quality=9, pixelformat='yuv420p',
                            ffmpeg_params=['-movflags', '+faststart']) as writer:
        for i, snap in enumerate(sorted(catalogues[0])):
            rows = [m[snap] for m in catalogues]
            a, b = [row['summary'] for row in rows]
            if not np.isclose(a['redshift'], b['redshift'], rtol=1e-7, atol=1e-7):
                raise ValueError(f'Redshift differs at snapshot {snap}')
            frame = Image.new('RGB', (1408, 832), '#0b1823')
            draw = ImageDraw.Draw(frame)
            draw.text((38, 24), 'Two universes through cosmic time', font=title, fill='#f4ead6')
            draw.text((38, 76), f'z = {max(a["redshift"], 0):.3f}   /   saved output {snap:03d}   /   XY dark-matter surface density',
                      font=label, fill='#acccd8')
            inputs = []
            for x, model, row in zip((38, 730), ('EDE', 'LCDM'), rows):
                asset = data / model / row['numeric_asset']
                if file_sha256(asset) != row['numeric_sha256']:
                    raise ValueError(f'Changed measurement {asset}')
                with np.load(asset, allow_pickle=False) as arrays:
                    field = gaussian_filter(arrays['dm_surface_density_msun_mpc2_xy'], 2., mode='wrap')
                intensity = np.clip((np.log10(np.maximum(field, 10**10.6))-10.6)/2.3, 0, 1)
                raster = Image.fromarray(np.flipud(cmap(intensity, bytes=True)[..., :3]))
                frame.paste(raster.resize((640, 640), Image.Resampling.LANCZOS), (x, 147))
                draw.text((x, 112), 'EDE' if model == 'EDE' else 'ΛCDM', font=label, fill='#f4ead6')
                inputs.append({'path': str(asset.relative_to(root)), 'sha256': row['numeric_sha256']})
            draw.text((38, 797), f'{a["box_comoving_mpc"]:.5f} comoving Mpc per side  ·  shared exposure  ·  saved epochs only',
                      font=small, fill='#acccd8')
            ramp = cmap(np.linspace(0, 1, 190), bytes=True)[:, :3]
            frame.paste(Image.fromarray(np.broadcast_to(ramp, (12, 190, 3)).copy()), (927, 800))
            draw.text((878, 797), '10.6', font=small, fill='#acccd8')
            draw.text((1127, 797), '12.9 log₁₀ Msun/Mpc²', font=small, fill='#acccd8')
            pixels = np.asarray(frame)
            hold = 24 if snap in (21, 26, 30, 35, 41, 54, 69, 85) else 6
            for _ in range(hold):
                writer.append_data(pixels)
            entries.append({'snapshot': snap, 'redshift': a['redshift'], 'frame_count': hold, 'inputs': inputs})
            if snap == 21:
                frame.save(out / 'Poster.png')
    stage.replace(film)
    atomic_json(out / 'film_source.json', {'title': 'Two universes through cosmic time', 'model': 'Comparisons',
        'redshifts': 'All 86 saved epochs, z=199 to z=0', 'fps': 24, 'frames': entries,
        'smoothing': 'Periodic Gaussian, sigma two raster pixels in both models',
        'log10_surface_density_msun_mpc2_range': [10.6, 12.9], 'published_sha256': file_sha256(film)})
    print(film, flush=True)


if __name__ == '__main__':
    main()
