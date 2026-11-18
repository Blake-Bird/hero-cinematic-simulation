"""Fresh 32-DM-particle catalogue controls for installed native halo finders.

The two saved epochs are synthetic repeated samples, not evolved snapshots.
Each worker writes a complete explicit-unit HDF5 fixture, invokes its production
driver, reads the actual catalogue and retains arrays, logs and a diagnostic.
An empty catalogue is an observed result, not a fabricated detection or a
claim of resolved halo accuracy. Rockstar needs eight available CPU slots.
"""
from pathlib import Path
import json


def fixture(output, seed, particles=32):
    """Write two explicit-unit epochs with 32 or 64 samples and exact IDs."""
    import h5py
    import numpy as np
    import yaml
    from core.foundation import KPC_CM, MSUN_G
    from core.evolution import ensure_run_layout
    if type(particles) is not int or particles not in (32, 64):
        raise ValueError('Fixed native fixture supports 32 or 64 samples')
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'core/config.yaml').read_text())
    cfg['cosmology'].update(model='lcdm', H0=70., omega_b=.0245, omega_cdm=.1225)
    # Supplied snapshots have an explicit 32-sample load, not a generated grid.
    cfg['simulation'].update(physics='dm', box_mpc_h=.1, particles_per_dimension=None,
        softening_kpc_h=.1, softening_max_physical_kpc_h=.1)
    cfg['hardware'].update(cpu_threads=8, mpi_ranks=1, omp_threads=1, build_jobs=2, memory_limit_gb=2.)
    cfg['halos'].update(min_particles=8, rockstar_overlap_length_mpc_h=.01,
        rockstar_wall_seconds=90., rockstar_exact_membership=True)
    cfg['analysis'].update(hmf_mass_definition='m200c', hmf_min_particles=8)
    ensure_run_layout(output)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg))
    (output/'manifest.json').write_text(json.dumps(dict(synthetic_fixture=True,
        purpose=f'{particles}-DM execution control; repeated epochs, no evolution, no halo truth')))
    rng = np.random.default_rng(seed)
    position = rng.normal(50., 12., (particles, 3)) % 100.
    velocity = rng.normal(0., 30., (particles, 3))
    ids = np.arange(1, particles+1, dtype=np.uint64)
    counts = np.array([0, particles, 0, 0, 0, 0], dtype=np.uint64)
    mass_table = np.array([0., .005, 0., 0., 0., 0.])
    for snap, scale in enumerate((.5, 1.)):
        with h5py.File(output/'data'/f'snapshot_{snap:03}.hdf5', 'w') as handle:
            handle.attrs.update(synthetic_fixture=True, purpose='finder execution control; no dynamics')
            handle.create_group('Header').attrs.update(Time=scale, Redshift=1/scale-1,
                BoxSize=100., MassTable=mass_table, NumFilesPerSnapshot=1,
                NumPart_Total=counts, NumPart_ThisFile=counts,
                NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32),
                HubbleParam=.7, Omega0=.3, OmegaLambda=.7)
            handle.create_group('Parameters').attrs.update(UnitLength_in_cm=KPC_CM,
                UnitMass_in_g=1e10*MSUN_G, UnitVelocity_in_cm_per_s=1e5,
                HubbleParam=.7, Omega0=.3, OmegaLambda=.7, OmegaBaryon=.05,
                ComovingIntegrationOn=1)
            group = handle.create_group('PartType1')
            for name, values, unit, aa, hh in [('Coordinates',position,KPC_CM,1.,-1.),
                    ('Velocities',velocity,1e5,.5,0.)]:
                dataset = group.create_dataset(name, data=values)
                dataset.attrs.update(to_cgs=unit, a_scaling=aa, h_scaling=hh)
            group['ParticleIDs'] = ids
    return cfg, position, velocity, ids


def calculate(name, output, seed):
    """Execute the selected finder and inspect actual object/membership outputs."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.native import native_engine_directory
    from core.foundation import file_sha256
    from core.halos import (run_rockstar, prepare_rockstar_host_catalogues,
        rockstar_catalogue, run_velociraptor, read_velociraptor_catalogue)
    from core.analysis import rockstar_host_table
    from core.resources import cpu_thread_budget
    output = Path(output).resolve()
    directory = native_engine_directory(name)
    binary = directory/('rockstar' if name == 'rockstar' else 'build/stf')
    if not binary.is_file():
        raise ImportError('Build the selected finder with scripts/setup.sh --engines '+name)
    if name == 'rockstar' and cpu_thread_budget() < 8:
        raise ImportError('Periodic Rockstar requires at least eight available CPU slots.')
    if name == 'rockstar' and not (directory/'util/find_parents').is_file():
        raise ImportError('Build the Rockstar find_parents utility for labelled catalogue intake.')
    cfg, position, velocity, ids = fixture(output, seed)
    namespace = 'optional-finder-32:seed-'+str(seed)
    if name == 'rockstar':
        status = run_rockstar(output, cfg)
        if status['status'] == 'SKIPPED': raise ImportError(status['reason'])
        assert status['status'] == 'COMPLETE'
        labelled = prepare_rockstar_host_catalogues(output, cfg)
        assert labelled['status'] == 'COMPLETE', labelled
        counts = []
        catalogue = None
        for snapshot in (0, 1):
            table = rockstar_host_table(output, snapshot)
            assert table is not None
            identity = table['_canonical_identity_links']['id']
            assert identity['available'].all()
            selected = identity['values']
            counts.append(len(selected))
            if len(selected):
                current = rockstar_catalogue(output, snapshot, particle_namespace=namespace,
                    memory_gib=.25, selected_ids=selected, mass_definition='m200c')
            else:
                current = dict(finder='rockstar', snapshot=snapshot, objects=[],
                    status='EMPTY', scope='Original labelled native catalogue has zero rows.')
            (output/f'catalogue-{snapshot:03}.json').write_text(json.dumps(current, indent=2, allow_nan=False))
            if snapshot == 1: catalogue = current
        from core.data import rockstar_directory
        raw_report = (rockstar_directory(output)/'rockstar_status.json').relative_to(output).as_posix()
        definition = 'ROCKSTAR m200c / h in Msun; native assigned membership, no bound-status claim'
    elif name == 'velociraptor':
        controls = dict(Minimum_size=8, Nsearch_physical=8, Nsearch_velocity=8,
            Cell_fraction=.1, Halo_3D_linking_length=.2,
            Substructure_physical_linking_length=.1, Search_for_substructure=0, Unbind_flag=0)
        run_velociraptor(output, output/'velociraptor', snapshot=1,
            particle_namespace=namespace, controls=controls, memory_gib=.25, threads=2, wall_hours=.025)
        catalogue = read_velociraptor_catalogue(output/'velociraptor')
        counts = [len(catalogue['objects'])]
        raw_report = 'velociraptor/report.json'
        definition = 'Native selected DM mass in Msun; unbinding disabled, not m200c'
    else:
        raise ValueError('Unknown fixed native finder')
    objects = catalogue['objects']
    for row in objects:
        assert row['mass_msun'] >= 0 and np.isfinite(row['mass_msun'])
        if row['members'] is not None: assert set(row['members']).issubset(set(map(int, ids)))
    (output/'catalogue.json').write_text(json.dumps(catalogue, indent=2, allow_nan=False))
    centers = np.array([row['position_proper_kpc'] for row in objects], dtype=float).reshape(-1, 3)
    mass = np.array([row['mass_msun'] for row in objects], dtype=float)
    np.savez_compressed(output/'arrays.npz', input_position_comoving_kpc_h=position,
        input_velocity_gadget=velocity, input_ids=ids, object_center_proper_kpc=centers,
        object_mass_msun=mass, object_count=np.asarray(counts, dtype=np.int64))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(position[:, 0]/.7, position[:, 1]/.7, s=18, label='32 input DM particles')
    if len(centers): axes[0].scatter(centers[:, 0], centers[:, 1], marker='x', s=60, label='native object centers')
    axes[0].set(xlabel='x [proper kpc, a=1]', ylabel='y [proper kpc, a=1]', title=name+' · actual input/catalogue')
    axes[0].legend(); axes[0].set_aspect('equal')
    axes[1].bar(np.arange(len(counts)), counts)
    axes[1].set(xlabel='saved catalogue index', ylabel='Native object count',
        title='Empty is an observed result' if not any(counts) else 'Native catalogue rows')
    axes[1].set_xticks(np.arange(len(counts))); axes[1].set_ylim(bottom=0, top=max(1, max(counts)*1.2))
    fig.savefig(output/'catalogue.png', dpi=140); plt.close(fig)
    return dict(backend=name, input_particles=32, saved_particle_epochs=2,
        catalogue_snapshots=[0,1] if name == 'rockstar' else [1], object_counts=counts,
        mass_definition=definition, native_report=raw_report, binary_sha256=file_sha256(binary),
        units={'input_position':'comoving kpc/h','object_center':'proper kpc','object_mass':'Msun'},
        implementation='core/halos.py:'+('run_rockstar → prepare_rockstar_host_catalogues → rockstar_host_table/rockstar_catalogue' if name == 'rockstar' else 'run_velociraptor → read_velociraptor_catalogue'),
        scope='Actual native finder execution and catalogue intake on 32 synthetic DM particles. Two repeated fixture epochs, no dynamics. Empty output remains empty; no resolved halo accuracy, finder equivalence, matching or merger-tree claim.')
