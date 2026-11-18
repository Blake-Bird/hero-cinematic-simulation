#!/usr/bin/env python3
"""Render the original readable camera routes for either paired model."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import group_field, halo_masses
from core.cosmic_flythrough import make_cosmic_flythrough
from core.filament_flythrough import make_filament_flythrough


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    source = root / 'Source/Native_workspaces' / f'{model}_movies'
    for snap, redshift in ((21, '014p400'), (85, '000p000')):
        host = int(np.argmax(halo_masses(source, snap)))
        target = np.asarray(group_field(source, snap, ['GroupPos']))[host] / 1000
        params = {'snapshot': snap, 'host_row': host, 'frames': 240, 'fps': 24,
                  'max_particles': 250000, 'memory_gib': 3., 'width': 1280, 'height': 720}
        for title, build in [('Halo_approach', make_cosmic_flythrough),
                             ('Filament_tour', make_filament_flythrough)]:
            output = root / '04_Movies' / model / f'{model}_z_{redshift}_{title}'
            if output.exists():
                continue
            route = {'start': (target + np.array([1.6, -1.4, .9])).tolist()} if title == 'Halo_approach' else {}
            try:
                result = build(source, output, **params, **route)
                print(f'COMPLETE {result}', flush=True)
            except Exception as error:
                print(f'FAILED {model} {redshift} {title}: {error}', flush=True)


if __name__ == '__main__':
    main()
