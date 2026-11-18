"""Generate and inspect a tiny two-species IC through the production drivers.

Called by native_checks in an owned, time-limited subprocess. Input is a fresh
external directory and seed. Outputs include fresh CLASS transfers/background,
the staged monofonIC configuration, original logs/HDF5, arrays and an XY plot.
The 4³ grid has 64 particles per species, 128 total; this tests execution and
structure, not resolved cosmological statistics or phase accuracy.
"""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generate(output, seed):
    """Run CLASS → staged species transfers → monofonIC; inspect original data."""
    import h5py
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.cosmology import run_boltzmann
    from core.evolution import ensure_run_layout, generate_hydro_ic_monofonic
    from core.foundation import file_sha256
    output = Path(output).resolve()
    cfg = yaml.safe_load((ROOT/'core/config.yaml').read_text())
    cfg['cosmology'].update(model='lcdm', H0=67.4, omega_b=.0224,
        omega_cdm=.12, A_s=2.1e-9, n_s=.965, tau_reio=.054,
        T_cmb=2.7255, N_ur=3.046, m_ncdm_eV=0.)
    cfg['simulation'].update(box_mpc_h=20., particles_per_dimension=4,
        start_redshift=49., end_redshift=48., random_seed=seed,
        lpt_order_hydro=1, physics='hydro')
    cfg['hardware'].update(cpu_threads=2, omp_threads=2, mpi_ranks=1,
        build_jobs=2, memory_limit_gb=2.)
    cfg['hydro'].update(courant_factor=.1, desired_neighbors=16, neighbor_deviation=2)
    ensure_run_layout(output)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg))
    linear = run_boltzmann(cfg, output, 'lcdm', 49., kmax_hmpc=10.)
    (output/'linear.json').write_text(json.dumps(linear, indent=2))
    path = generate_hydro_ic_monofonic(cfg, output, 'lcdm', cosmo=linear)
    arrays = {}; counts = {}; masses = {}; all_ids = []
    with h5py.File(path) as handle:
        header = handle['Header'].attrs
        assert np.isclose(float(header['BoxSize']), 20000.)
        assert np.isclose(float(header['Time']), .02)
        for species in (0, 1):
            group = handle['PartType'+str(species)]
            position = group['Coordinates'][:]; velocity = group['Velocities'][:]
            ids = group['ParticleIDs'][:]
            assert position.shape == velocity.shape == (64, 3)
            assert ids.shape == (64,) and ids.dtype.kind in 'ui'
            assert np.isfinite(position).all() and np.isfinite(velocity).all()
            assert ((position >= 0) & (position < 20000.)).all()
            mass = group['Masses'][:] if 'Masses' in group else np.full(64, header['MassTable'][species])
            assert mass.shape == (64,) and np.isfinite(mass).all() and (mass > 0).all()
            counts[str(species)] = len(ids); masses[str(species)] = float(mass.sum())
            all_ids.append(ids)
            arrays.update({f'position_{species}_kpc_h':position,
                f'velocity_{species}_gadget':velocity, f'ids_{species}':ids,
                f'mass_{species}_1e10_msun_h':mass})
    assert len(np.unique(np.concatenate(all_ids))) == 128
    expected_fraction = .0224 / (.0224 + .12)
    fraction = masses['0'] / sum(masses.values())
    assert abs(fraction - expected_fraction) < 1e-6
    np.savez_compressed(output/'arrays.npz', **arrays)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for species, label in [(0, 'gas: 64'), (1, 'dark matter: 64')]:
        points = arrays[f'position_{species}_kpc_h']
        ax.scatter(points[:, 0], points[:, 1], s=18, alpha=.65, label=label)
    ax.set(xlabel='x [comoving kpc/h]', ylabel='y [comoving kpc/h]',
        title='Native monofonIC 4³ per species · z=49', xlim=(0, 20000), ylim=(0, 20000))
    ax.legend(); ax.set_aspect('equal'); fig.savefig(output/'positions.png', dpi=140); plt.close(fig)
    result=dict(particles=128, particles_per_dimension=4, counts=counts,
        box_kpc_h=20000., scale_factor=.02, gas_mass_fraction=fraction,
        expected_gas_mass_fraction=expected_fraction, unique_ids=128,
        hdf5=path.relative_to(output).as_posix(), hdf5_sha256=file_sha256(path),
        units={'position':'comoving kpc/h', 'mass':'1e10 Msun/h',
               'velocity':'native GADGET stored velocity; peculiar velocity = stored × sqrt(a)'},
        implementation='core/cosmology.py:run_boltzmann → core/evolution.py:generate_hydro_ic_monofonic → native monofonIC → rescale_monofonic_ic',
        scope='Actual fresh CLASS plus two-species 1LPT IC generation. Counts, finite fields, periodic bounds, unique IDs and cosmological gas mass fraction checked. No time evolution, phase/amplitude acceptance or physical convergence claimed.')
    (output/'inspection.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.seed)))
