"""Build a static teaching plate from the retained MUSIC → GADGET zoom control."""
import argparse
import hashlib
import json
from pathlib import Path


def build_zoom_reference(snapshot, output):
    """Verify saved counts, retain the plotted coordinates and render three scales."""
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = Path(snapshot)
    output = Path(output)
    types = [('PartType2', 'Coarse boundary DM', '#748caf'),
             ('PartType0', 'Refined gas', '#63c5e5'),
             ('PartType1', 'Refined DM', '#f2c47d')]
    with h5py.File(source) as file:
        header = file['Header'].attrs
        scale = float(header['Time'])
        box = float(header['BoxSize']) / 1000
        counts = header['NumPart_Total']
        arrays = {}
        for key, _, _ in types:
            position = file[key]['Coordinates'][:]
            assert position.shape == (int(counts[int(key[-1])]), 3)
            assert np.isfinite(position).all()
            assert (file[key]['Coordinates'].attrs['a_scaling'], file[key]['Coordinates'].attrs['h_scaling']) == (1., -1.)
            arrays[key] = position / 1000  # Native comoving kpc/h → comoving Mpc/h.
    limits = [(0., box), (3.5, 6.3), (4.4, 5.2)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained', facecolor='#0d1520')
    for axis, (low, high), title in zip(axes, limits, ['Parent box', 'Refined region', 'Interior lattice']):
        axis.set_facecolor('#132131')
        for key, label, color in types:
            p = arrays[key]
            axis.scatter(p[:, 0], p[:, 1], s=5, c=color, alpha=.57, linewidths=0, label=label)
        axis.set(xlim=(low, high), ylim=(low, high), aspect='equal',
                 xlabel='x [comoving Mpc/h]', ylabel='y [comoving Mpc/h]', title=title)
        axis.tick_params(colors='#b6c7dc')
        for item in [axis.title, axis.xaxis.label, axis.yaxis.label]:
            item.set_color('#e1ebf3')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'MUSIC → GADGET | zero-transfer control | saved z = {1 / scale - 1:.3g}', color='white')
    fig.savefig(output / 'native_zoom.png', dpi=170)
    plt.close(fig)
    np.savez_compressed(output / 'native_zoom.npz', **arrays)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        'source_sha256': digest(source),
        'scale_factor': scale,
        'box_comoving_mpc_h': box,
        'coordinate_units': 'comoving Mpc/h',
        'counts': {key: len(value) for key, value in arrays.items()},
        'views': limits,
        'artifacts_sha256': {name: digest(output / name) for name in ['native_zoom.png', 'native_zoom.npz']},
        'scope': 'MUSIC zero-transfer normalization control evolved by native GADGET. The lattice shows refinement geometry.',
    }
    (output / 'native_zoom.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_zoom_reference(args.snapshot, args.output), indent=2))
