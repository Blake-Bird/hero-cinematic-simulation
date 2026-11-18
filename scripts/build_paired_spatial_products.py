#!/usr/bin/env python3
"""Measure the spatial explorers for each retained dark-matter run."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.allsky_map import make_allsky_map
from core.analysis import halo_masses
from core.foundation import atomic_json
from core.halo_anatomy import make_halo_anatomy
from core.velocity_flow import make_velocity_flow
from core.void_explorer import make_void_explorer
from core.void_particle_history import make_void_particle_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    workspace = root / 'Source/Native_workspaces' / f'{model}_history'
    folder = root / ('02_EDE' if model == 'EDE' else '03_LCDM') / 'Spatial_explorers'
    reports = {}

    def produce(destination, builder, **options):
        key = str(destination.relative_to(root))
        try:
            result = builder(workspace, destination=destination, memory_gib=3., **options) if not destination.exists() else destination
            reports[key] = {'status': 'PRESENT', 'path': str(result)}
        except Exception as error:
            reports[key] = {'status': 'FAILED', 'error': str(error)}
        print(key, reports[key], flush=True)
        atomic_json(root / '07_Status_and_Provenance' / f'{model}_spatial_products.json', reports)

    for snap, z in ((21, '014p400'), (85, '000p000')):
        epoch = folder / f'{model}_z_{z}'
        host = int(np.argmax(halo_masses(workspace, snap)))
        for projection in ('xy', 'xz', 'yz'):
            produce(epoch / f'Peculiar_velocity_{projection.upper()}', make_velocity_flow,
                    snapshot=snap, projection=projection, grid=48)
        produce(epoch / 'Dark_matter_all_sky', make_allsky_map, snapshot=snap)
        produce(epoch / 'Most_massive_FOF_halo_anatomy', make_halo_anatomy, snapshot=snap, host_row=host)
    void = root / '04_Movies' / model / f'{model}_z_199_to_000_void_explorer'
    produce(void, make_void_explorer, frames=240, fps=24, max_particles=250000)
    if void.is_dir():
        produce(folder / f'{model}_z_199_to_000_void_particle_history', make_void_particle_history,
                void_bundle=void, radius_mpc_h=.4)


if __name__ == '__main__':
    main()
