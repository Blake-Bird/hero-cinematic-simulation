"""Compile and run the optional 32-cell atomic cooling control against Grackle.

Requires an installed shared Grackle 3.4.1 library discoverable through
pkg-config (set PKG_CONFIG_PATH to its lib/pkgconfig directory) and a C compiler.
Builds only this tiny test executable, never Grackle. All files stay under the
new lab output. The independent fixed-density cells are not simulation particles.
"""
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess


def calculate(output):
    """Record native cooling arrays, conservation residuals and their cgs units."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.foundation import file_sha256
    from core.native import run_native_command
    output = Path(output).resolve()
    pkg = shutil.which('pkg-config')
    compiler = shutil.which(os.environ.get('CC', 'cc'))
    if not pkg or not compiler:
        raise ImportError('Grackle control requires pkg-config and a C compiler (CC or cc).')
    version = subprocess.run([pkg, '--modversion', 'grackle'], capture_output=True, text=True, timeout=10)
    if version.returncode:
        raise ImportError('Expose installed shared Grackle through PKG_CONFIG_PATH=/prefix/lib/pkgconfig.')
    if version.stdout.strip() != '3.4.1':
        raise ValueError('This control requires the repository-pinned Grackle 3.4.1.')
    def query(*arguments):
        return subprocess.check_output([pkg, *arguments, 'grackle'], text=True, timeout=10).strip()
    libdir = Path(query('--variable=libdir')).resolve()
    includedir = Path(query('--variable=includedir')).resolve()
    library = next((path.resolve() for path in [libdir/'libgrackle.dylib', libdir/'libgrackle.so'] if path.is_file()), None)
    if library is None:
        raise ImportError('Grackle control requires an installed shared library, not a static-only build.')
    source = Path(__file__).with_suffix('.c')
    binary = output/'cooling-control'
    headers = sorted(includedir.glob('*.h'))
    with (output/'compile.log').open('w') as log:
        run_native_command([compiler, '-std=c99', str(source), '-o', str(binary),
            *shlex.split(query('--cflags')), *shlex.split(query('--libs')), '-lm', '-Wl,-rpath,'+str(libdir)],
            cwd=output, inputs=[source, library, *headers], receipt_path=output/'compile-invocation.json',
            timeout=60, stdout=log, stderr=subprocess.STDOUT)
    csv = output/'cooling.csv'
    with (output/'native.log').open('w') as log:
        run_native_command([str(binary), str(csv)], cwd=output, inputs=[binary, library],
            receipt_path=output/'native-invocation.json', timeout=60,
            stdout=log, stderr=subprocess.STDOUT)
    rows = np.genfromtxt(csv, delimiter=',', names=True)
    assert rows.shape == (64,) and all(np.isfinite(rows[key]).all() for key in rows.dtype.names)
    before, after = rows[:32], rows[32:]
    np.testing.assert_array_equal(before['cell'], np.arange(32))
    np.testing.assert_array_equal(after['cell'], before['cell'])
    np.testing.assert_array_equal(before['epoch'], np.zeros(32))
    np.testing.assert_array_equal(after['epoch'], np.ones(32))
    np.testing.assert_array_equal(before['density_g_cm3'], after['density_g_cm3'])
    assert (after['time_s'] == 1e11).all() and (before['time_s'] == 0).all()
    for key in ['specific_energy_erg_g', 'temperature_K']:
        assert (rows[key] > 0).all(), key
    # Grackle returns energy / signed energy-rate, negative for net cooling.
    # Its own evolution helper uses abs(cooling_time) for a timestep magnitude.
    assert (rows['cooling_time_s'] < 0).all()
    hydrogen = rows['HI_mass_fraction'] + rows['HII_mass_fraction']
    helium = sum(rows[key] for key in ['HeI_mass_fraction', 'HeII_mass_fraction', 'HeIII_mass_fraction'])
    h_error = float(np.max(abs(hydrogen - .76)))
    he_error = float(np.max(abs(helium - .24)))
    assert h_error < 1e-10 and he_error < 1e-10
    for key in rows.dtype.names:
        if 'fraction' in key: assert (rows[key] >= 0).all(), key
    energy_change = after['specific_energy_erg_g'] / before['specific_energy_erg_g'] - 1
    assert (energy_change < 0).all() and np.max(abs(energy_change)) > 1e-6
    arrays = {key: rows[key].reshape(2, 32) for key in rows.dtype.names}
    np.savez_compressed(output/'arrays.npz', **arrays)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for row, label in [(before, 'initial'), (after, 'after 1e11 s')]:
        axes[0].loglog(row['density_g_cm3'], row['temperature_K'], label=label)
        axes[1].loglog(row['density_g_cm3'], row['HII_mass_fraction'], label=label)
    axes[0].set(ylabel='Temperature [K]', title='Atomic H/He cooling')
    axes[1].set(ylabel='HII mass / total gas mass', title='Ionization response')
    for axis in axes:
        axis.set_xlabel('Gas density [g/cm³]'); axis.legend()
    fig.savefig(output/'cooling.png', dpi=140); plt.close(fig)
    return dict(cells=32, epochs=2, elapsed_seconds=1e11, network='atomic H/He (primordial_chemistry=1)',
        version=version.stdout.strip(), library_sha256=file_sha256(library),
        maximum_hydrogen_mass_fraction_residual=h_error, maximum_helium_mass_fraction_residual=he_error,
        conservation_absolute_tolerance=1e-10, fractional_energy_change_range=[float(energy_change.min()),float(energy_change.max())],
        units={'density':'g/cm³', 'specific_energy':'erg/g', 'temperature':'K', 'time':'s',
               'cooling_time':'s; signed energy / energy-rate, negative for net cooling',
               'species':'species mass / total gas mass; electron field uses equivalent hydrogen mass'},
        implementation='optional_tests/native_grackle.c → installed Grackle local_initialize_chemistry_data/local_solve_chemistry/local_calculate_temperature/local_calculate_cooling_time',
        scope='Actual native library evolution of 32 independent fixed-density atomic H/He cells. No UV, metals, molecules, dust or lookup tables. Nuclear mass fractions and cooling response checked. Separate from the production GADGET bridge; no particle dynamics, convergence or full-network acceptance claimed.')
