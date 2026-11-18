"""Run a fixed independent OpenGadget3 elastic-scattering execution control.

Set RESEARCH2_OPENGADGET3_BUILD to a completed native reference build directory
containing build-capability.json and source/Config.research.sh. The production
runner verifies that capability and its real compile/source hashes. This worker
never builds or installs dependencies. All fresh inputs, outputs and diagnostics
remain in its external output directory. The 100 cm²/g law is a deliberately
high-rate toy control, not a physical cross-section inference.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys


def calculate(output, seed):
    """Evolve 32 physical beam particles; check exact IDs, mass, energy and momentum."""
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.foundation import file_sha256, KPC_CM, MSUN_G
    from core.sidm_control_products import publish_control_population, homogeneous_control_runner
    setting = os.environ.get('RESEARCH2_OPENGADGET3_BUILD')
    if not setting:
        raise ImportError('Set RESEARCH2_OPENGADGET3_BUILD to an existing native rare/directed reference build; see docs/software.md.')
    build = Path(setting).expanduser().resolve()
    capability = build/'build-capability.json'
    configuration = build/'source/Config.research.sh'
    if not capability.is_file() or not configuration.is_file():
        raise ImportError('Reference build needs build-capability.json and source/Config.research.sh.')
    compiled = json.loads(capability.read_text())
    executable = Path(compiled['executable'])
    if not executable.is_file():
        raise ImportError('Reference executable is absent; build OpenGadget3 for this machine.')
    with executable.open('rb') as stream: magic = stream.read(4)
    if sys.platform == 'darwin' and magic == b'\x7fELF':
        raise ImportError('The configured OpenGadget3 executable is Linux ELF; run it in its Linux environment.')
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    def ref(path): return dict(path=str(path), sha256=file_sha256(path))
    population_request = json.loads((root/'configs/research/sidm_uniform_beam_population.example.json').read_text())
    population_request['request'].update(seed=seed, position_sampling='uniform_random',
        box_proper_kpc=1., macro_mass_msun=1e8, maximum_particles=32,
        identity_namespace='optional-independent-sidm:seed-'+str(seed))
    for component, vx in zip(population_request['request']['components'], (60., -40.)):
        component.update(particle_count=16, bulk_velocity_kms=[vx,0.,0.])
    population = output/'population.json'
    publish_control_population(population_request, population)
    model = json.loads((root/'configs/research/sidm_homogeneous_isotropic.model.json').read_text())
    model['cross_section_cm2_per_g'] = 100.
    physical_model = output/'high-rate-model.json'; physical_model.write_text(json.dumps(model, indent=2))
    base = output/'base.param'
    base.write_text('TimeBetSnapshot 0.0001\nPartAllocFactor 2.5\nPeriodicBoundariesOn 1\n'
        'DesNumNgb_pt1 16\nMaxNumNgbDeviation_pt1 2\nMaxMemSize 1024\n')
    request = json.loads((root/'configs/research/sidm_homogeneous_opengadget3_run.example.json').read_text())['request']
    request.update(population=ref(population), base_configuration=ref(base),
        reference_build=dict(capability=ref(capability), configuration=ref(configuration)),
        output_times_native=[0.,.0001,.0002], maximum_timestep_native=1e-5,
        maximum_particles=32, maximum_snapshots=3, maximum_snapshot_bytes=32*1024**2,
        mpi_ranks=1, omp_threads=1, memory_gib=2., disk_gib=1., wall_seconds=120.)
    operator_seed = seed % (2**31)
    request['sidm'].update(physical_model=ref(physical_model), operator_seed=operator_seed,
        kernel_support_proper_kpc=.2, coarse_kernel_support_proper_kpc=.24,
        maximum_kernel_coarse_support_fraction=1., maximum_omitted_scattering_upper_bound=1.)
    (output/'request.json').write_text(json.dumps(request, indent=2))
    result = homogeneous_control_runner(dict(request=request, request_base=str(root),
        destination=str(output/'run')), SimpleNamespace(generation=output,
            resources=dict(mpi_ranks=1, omp_threads=1, memory_gib=2., disk_gib=1.)))
    (output/'native-run.json').write_text(json.dumps(result, indent=2))
    assert result['execution_status'] == 'COMPLETE' and result['engine'] == 'opengadget3'
    positions = []; velocities = []; masses = []; identities = []; times = []
    units = population_request['request']['native_units']
    for row in result['snapshots']:
        assert len(row['files']) == 1
        with h5py.File(row['files'][0]) as handle:
            group = handle['PartType1']; ids = group['ParticleIDs'][:]
            assert ids.dtype.kind in 'ui' and len(ids) == len(np.unique(ids)) == 32
            order = np.argsort(ids); identities.append(ids[order])
            positions.append(group['Coordinates'][:][order] * units['length_cm']/units['hubble_parameter']/KPC_CM)
            velocities.append(group['Velocities'][:][order] * units['velocity_cm_s']/1e5)
            mass = group['Masses'][:] if 'Masses' in group else np.full(32, handle['Header'].attrs['MassTable'][1])
            masses.append(mass[order]*units['mass_g']/units['hubble_parameter']/MSUN_G)
            times.append(float(handle['Header'].attrs['Time']))
    positions, velocities, masses, identities, times = map(np.asarray, (positions, velocities, masses, identities, times))
    assert velocities.shape == positions.shape == (3,32,3)
    assert np.isfinite(positions).all() and np.isfinite(velocities).all()
    np.testing.assert_allclose(times, [0.,.0001,.0002], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(identities, np.broadcast_to(identities[0], identities.shape))
    np.testing.assert_allclose(masses, 1e8, rtol=1e-12, atol=0)
    momentum = np.sum(masses[:,:,None]*velocities, axis=1)
    energy = .5*np.sum(masses*np.sum(velocities**2, axis=2), axis=1)
    reference = np.sum(masses[0]*np.linalg.norm(velocities[0], axis=1))
    momentum_residual = np.linalg.norm(momentum-momentum[0], axis=1)/reference
    energy_residual = abs(energy/energy[0]-1)
    assert np.max(momentum_residual) < 1e-10 and np.max(energy_residual) < 1e-10
    changed = np.count_nonzero(np.linalg.norm(velocities-velocities[0], axis=2)>1e-9, axis=1)
    assert changed[-1] > 0
    # Production _control_elapsed_seconds uses L/(V*h) for static native time.
    seconds = times * units['length_cm']/units['velocity_cm_s']/units['hubble_parameter']
    np.savez_compressed(output/'arrays.npz', particle_ids=identities,
        position_proper_kpc=positions, velocity_peculiar_kms=velocities, mass_msun=masses,
        native_time_code=times, elapsed_proper_seconds=seconds, momentum_msun_kms=momentum,
        kinetic_energy_msun_kms2=energy, momentum_relative_residual=momentum_residual,
        energy_relative_residual=energy_residual, changed_velocity_particles=changed)
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for index,label in [(0,'initial'),(2,'final')]:
        axes[0].scatter(np.arange(32),velocities[index,:,0],s=18,label=label)
    axes[0].set(xlabel='Exact-ID sorted particle row',ylabel='vx [km/s]',title='Independent native scattering')
    axes[0].legend()
    axes[1].plot(times,energy_residual,marker='.',label='kinetic energy')
    axes[1].plot(times,momentum_residual,marker='.',label='momentum')
    axes[1].set(xlabel='Native time code',ylabel='Relative residual',title='Measured conservation')
    axes[1].legend(); fig.savefig(output/'scattering.png',dpi=140); plt.close(fig)
    artifacts = {path.relative_to(output).as_posix():file_sha256(path)
        for path in sorted(output.rglob('*')) if path.is_file() and '.native-run-locks' not in path.parts}
    return dict(engine='OpenGadget3', input_particles=32, epochs=3, operator_seed=operator_seed,
        population_seed=seed, cross_section_cm2_per_g=100., gravity=False,
        changed_velocity_particles=changed.tolist(), maximum_energy_relative_residual=float(energy_residual.max()),
        maximum_momentum_relative_residual=float(momentum_residual.max()), conservation_tolerance=1e-10,
        binary_sha256=file_sha256(executable), native_report='native-run.json', artifact_sha256=artifacts,
        units={'position':'proper kpc','velocity':'km/s','mass':'Msun','momentum':'Msun km/s',
            'kinetic_energy':'Msun (km/s)^2','elapsed_time':'s = native time × L/(V h)'},
        implementation='core/sidm_control_products:publish_control_population → homogeneous_control_runner → pinned OpenGadget3 rare/directed native solver → exact-ID physical snapshot census',
        scope='Actual independent elastic scattering and mass/momentum/kinetic-energy checks for 32 high-rate synthetic beam particles. Changed particles are not an event count. No Gadget4 comparison, calibrated scattering rate, cross-section inference or convergence claimed.')
