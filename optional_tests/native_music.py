"""Run the native MUSIC transfer-control suite and expose its nested particle load.

Requires the setup-built MUSIC executable. Calls the production control CLI in
an owned subprocess; never builds dependencies. Ten delivered realizations and
seven expected rejection attempts retain their original configurations, logs
and outputs. Transfers are analytic controls, not fresh Boltzmann predictions.
The nested reference has 10,753 particles rather than the quick fixture's 32.
"""
from pathlib import Path
import json
import os
import subprocess
import sys


def calculate(output):
    """Check normalization/conservation and plot original uniform/nested outputs."""
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.native import native_engine_directory, run_native_command
    from core.foundation import file_sha256
    from core.research.generator_controls import CASES, NESTED_CASES, PERIODIC_CASES, REJECTIONS
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    binary = native_engine_directory('music2')/'build/MUSIC'
    if not binary.is_file():
        raise ImportError('Build MUSIC with scripts/setup.sh --engines music2.')
    controls = output/'controls'
    with (output/'generation.log').open('w') as log:
        run_native_command([sys.executable, '-B', str(root/'commands.py'), 'native-music-transfer',
            str(controls), '--binary', str(binary), '--prepare', '--nested'],
            cwd=output, inputs=[binary, root/'commands.py', root/'core/research/generator_controls.py'],
            env={**os.environ, 'OMP_NUM_THREADS':'2'}, receipt_path=output/'generation-invocation.json',
            timeout=150, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads((controls/'native-transfer-suite.json').read_text())
    assert report['status'] == report['nested']['status'] == report['periodic']['status'] == 'PASS'
    assert report['execution']['binary_sha256'] == file_sha256(binary)
    assert set(report['execution']['rejections']) == set(REJECTIONS)
    assert all(row['exit_code'] != 0 for row in report['execution']['rejections'].values())
    counts = {}
    for case in CASES + NESTED_CASES + PERIODIC_CASES:
        assert all(row['status'] == 'PASS' for row in report['conservation'][case].values())
        with h5py.File(controls/case/'zoom_fixture.hdf5') as handle:
            counts[case] = {key:len(handle[key]['Coordinates']) for key in handle if key.startswith('PartType')}
    arrays = {}
    with h5py.File(controls/'dms-reference/zoom_fixture.hdf5') as handle:
        box = float(handle['Header'].attrs['BoxSize'])
        for species in (0, 1, 2):
            group = handle[f'PartType{species}']
            for field in ('Coordinates', 'Velocities', 'ParticleIDs', 'LagrangianCoordinates', 'LagrangianCellWidths'):
                arrays[f'{field}_{species}'] = group[field][:]
            assert np.isfinite(arrays[f'Coordinates_{species}']).all()
            assert np.isfinite(arrays[f'Velocities_{species}']).all()
    np.savez_compressed(output/'arrays.npz', **arrays)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for species, label in [(2, 'coarse matter'), (1, 'fine DM'), (0, 'gas')]:
        points = arrays[f'Coordinates_{species}']
        for axis in axes:
            axis.scatter(points[:, 0], points[:, 1], s=2, alpha=.4, label=label)
    axes[0].set(xlim=(0, box), ylim=(0, box), title='Nested MUSIC particle load')
    axes[1].set(xlim=(.35*box, .65*box), ylim=(.35*box, .65*box), title='Same original particles · central zoom')
    for axis in axes:
        axis.set(xlabel='x [comoving kpc/h]', ylabel='y [comoving kpc/h]'); axis.set_aspect('equal')
    axes[0].legend(markerscale=3)
    fig.savefig(output/'nested.png', dpi=140); plt.close(fig)
    residuals = {name:{species:row['absolute_velocity_displacement_relative_residual']
        for species,row in report[name]['species'].items()} for name in ('uniform','nested')}
    return dict(realizations=10, expected_rejections=7, counts=counts,
        reference_particles=sum(counts['dms-reference'].values()),
        transfer_origin='analytic synthetic normalization controls; no Boltzmann solver execution',
        report='controls/native-transfer-suite.json', binary_sha256=file_sha256(binary),
        velocity_displacement_relative_residuals=residuals,
        units={'position':'comoving kpc/h','velocity':'stored GADGET velocity; peculiar = stored × sqrt(a)',
               'lagrangian_width':'comoving kpc/h','redshift':99},
        implementation='core/research/generator_controls.py:native_music_transfer_main → native MUSIC → verify/verify_nested/verify_periodic + validate_ic_realization',
        scope='Actual uniform/nested/periodic two-fluid 1LPT normalization and conservation controls, plus seven expected input rejections. Original q, exact IDs, velocity doubling, continuity units, mass, momentum and barycenter checked. Not a target-halo zoom, physical transfer-spectrum acceptance or evolution/convergence claim.')
