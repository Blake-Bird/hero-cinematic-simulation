"""Bounded native gas cooling, stellar birth and black-hole source controls.

These use the production input generator, build/parameter driver and GADGET
solver. A 4³ gas grid means 64 particles, not 32. Stars start with no stellar
particles; the BH case adds one explicitly seeded sink. The fixed recipes are
execution controls, not resolved galaxies or convergence studies. Compilation
may be cold; all builds, logs, snapshots and arrays remain outside the source.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKS = ('gas', 'stars', 'black-holes')


def calculate(name, output, seed):
    """Evolve three actual epochs, retain exact fields and check the named response."""
    import h5py
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.foundation import KPC_CM, MSUN_G, G_CGS, SECONDS_PER_YEAR, file_sha256
    from core.data import dataset_to_cgs_factor
    from core.evolution import ensure_run_layout, build_gadget, gadget_param_text
    from core.research.controls import prepare_native_source_control
    for tool in ('mpirun', 'mpicxx', 'make'):
        if not shutil.which(tool): raise ImportError('Native physics controls require '+tool+' from the setup toolchain.')
    output = Path(output).resolve(); ensure_run_layout(output)
    cfg = yaml.safe_load((ROOT/'core/config.yaml').read_text())
    cfg['cosmology'].update(model='lcdm', H0=70., omega_b=.0245, omega_cdm=.1225,
        omega_radiation=0., T_cmb=2.7255, N_ur=3.046)
    cfg['hardware'].update(cpu_threads=2, build_jobs=2, mpi_ranks=1, omp_threads=1, memory_limit_gb=2.)
    cfg['halos'].update(fof=False, subfind=False)
    cfg['simulation'].update(physics='hydro', ic_engine_hydro='monofonic', particles_per_dimension=4,random_seed=seed,
        box_mpc_h=.001, pm_grid=16, start_redshift=1., end_redshift=0., snapshot_redshift_list=[1.,0.],
        softening_kpc_h=.01, softening_max_physical_kpc_h=.01, max_size_timestep_ln_a=.0001,
        min_size_timestep_ln_a=0., err_tol_theta=.3, err_tol_force_acc=.001, err_tol_int_accuracy=.01,
        num_files_per_snapshot=1, restart_every_seconds=120., time_limit_cpu_seconds=120.)
    cfg['hydro'].update(courant_factor=.1, desired_neighbors=16, neighbor_deviation=2,
        cooling=True, star_formation=name=='stars', uv_background=False,stellar_random_seed=seed)
    kind = {'gas':'cooling_box','stars':'sh03_equilibrium','black-holes':'bh_accretion'}[name]
    comoving = name == 'black-holes'
    start = .5 if comoving else .002 if name == 'stars' else 0.
    times = [start, start+.01, start+.02] if name == 'stars' else [0.,.0001,.0002] if name == 'gas' else [.5,.5000001,.5000002]
    timestep = .0001 if name == 'gas' else .0002 if name == 'stars' else 1e-7
    parameters = dict(grid_size=4, box_length=1., gas_density=.001 if name=='gas' else .1,
        gas_internal_energy=10000. if name=='gas' else 100., seed=seed)
    if comoving:
        # All gravitating matter in this analytic source box is gas or its
        # seeded BH, so declare the corresponding all-baryon toy background.
        cfg['cosmology'].update(omega_b=.147,omega_cdm=0.)
        bh_mass = 10.*.7/1e10
        gravity_internal = G_CGS*1e10*MSUN_G/(KPC_CM*1e10)
        parameters.update(gas_density=.3*3*.1**2/(8*np.pi*gravity_internal)-bh_mass,
            bh_masses=[bh_mass], bh_physical_masses=[bh_mass],
            bh_positions=[[.5,.5,.5]], bh_velocities=[[0.,0.,0.]])
        cfg['black_holes']=dict(enabled=True, radiative_efficiency=.1, feedback_coupling=.05,
            bondi_boost=1., eddington_factor=1., kernel_radius_proper_kpc=.002,
            seed_mass_msun=10., source_fraction=.05, maximum_count=4,
            seeding_mode='ic_particles', seed_density_hydrogen_cm3=0., seed_exclusion_proper_kpc=0.)
    request = dict(kind=kind, parameters=parameters, native_units=dict(length_cm=KPC_CM,
        mass_g=1e10*MSUN_G, velocity_cm_s=1e5, time=start, hubble_param=.7 if comoving else 1.,
        comoving=comoving), chemistry=False, black_holes=comoving, memory_gib=2., base_config=cfg)
    (output/'request.json').write_text(json.dumps(dict(source_request=request,
        output_times=times, maximum_timestep=timestep,
        gas_coordinate_compression=.001 if comoving else 1.),indent=2))
    prepared = prepare_native_source_control(request, output/'input/control')
    initial = output/'input/control.ic.hdf5'
    if comoving:
        # Keep the original sealed grid unchanged. Publish a distinct analytic
        # compressed IC, with its input hash and declared transform retained.
        compressed = output/'input/compressed.ic.hdf5'; shutil.copyfile(initial,compressed)
        with h5py.File(compressed,'r+') as handle:
            handle['PartType0/Coordinates'][:] = .5 + (handle['PartType0/Coordinates'][:]- .5)*.001
        (output/'input/compression.json').write_text(json.dumps(dict(
            source_ic_sha256=file_sha256(initial), output_ic_sha256=file_sha256(compressed),
            scale=.001, center=[.5,.5,.5], scope='Explicit analytic initial-condition transform, not evolved compression'),indent=2))
        initial = compressed
    binary = build_gadget(cfg, output, control_profile=prepared['build_profile'])
    text = gadget_param_text(cfg,output,initial)
    values={line.split()[0]:' '.join(line.split()[1:]) for line in text.splitlines() if line.strip() and not line.lstrip().startswith('%')}
    values.update(TimeBegin=start, TimeMax=times[-1], TimeOfFirstSnapshot=start,
        TimeBetStatistics=timestep, MaxSizeTimestep=timestep, MinSizeTimestep=0.,
        ComovingIntegrationOn=int(comoving), BoxSize=1.)
    values.pop('ActivePartFracForPMinsteadOfEwald',None)
    if not comoving: values.update(HubbleParam=1.,Omega0=0.,OmegaLambda=0.,OmegaBaryon=0.)
    parameter_file=output/'input/param.txt'; schedule=output/'input/output_times.txt'
    parameter_file.write_text(''.join(f'{key} {value}\n' for key,value in values.items()))
    schedule.write_text(''.join(f'{time:.17g} 1\n' for time in times))
    (output/'config.yaml').write_text(yaml.safe_dump(cfg))
    manifest=dict(fixture=True,scope=__doc__,source_control_inputs=prepared,
        gadget_handoff=dict(initial_conditions=[dict(path=str(initial),sha256=file_sha256(initial))],
            compiled_options_file=dict(path=str(output/'input/Config.sh'),sha256=file_sha256(output/'input/Config.sh')),
            parameter_file=dict(path=str(parameter_file),sha256=file_sha256(parameter_file)),
            executable=dict(path=str(binary),sha256=file_sha256(binary))))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from core.native import run_native_command
    with (output/'native.log').open('w') as log:
        run_native_command(['mpirun','-np','1',str(binary),str(parameter_file),'0'],
            cwd=output/'input',receipt_path=output/'invocation.json',
            inputs=[initial,parameter_file,schedule,output/'input/Config.sh'],
            timeout=120,env={**os.environ,'OMP_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
    files=sorted((output/'data').glob('snapshot_*.hdf5')); assert len(files)==3
    counts=[]; masses=[]; mean_energy=[]; bh_mass=[]; inflow=[]; arrays={}
    for epoch,path in enumerate(files):
        with h5py.File(path) as handle:
            actual=float(handle['Header'].attrs['Time']); assert abs(actual-times[epoch])<1e-12
            a=actual if comoving else 1.; h=.7 if comoving else 1.
            census=np.zeros(6,dtype=np.int64); mass=np.zeros(6); all_ids=[]
            for species in (0,4,5):
                group=handle.get(f'PartType{species}')
                if group is None: continue
                ids=group['ParticleIDs'][:]; all_ids.append(ids); census[species]=len(ids)
                assert ids.dtype.kind in 'ui'
                for field in ('Coordinates','Velocities','Masses'):
                    assert np.isfinite(group[field][:]).all(),field
                factor=dataset_to_cgs_factor(group['Masses'].attrs,a,h)/MSUN_G
                mass[species]=np.sum(group['Masses'][:]*factor)
                arrays[f'particle_ids_{epoch}_{species}']=ids
                if species==4:
                    for field in ('SSPBirthNativeTime','SSPParentGasID','SSPBirthMode'):
                        arrays[f'{field}_{epoch}']=group[field][:]
            joined=np.concatenate(all_ids); assert len(np.unique(joined))==len(joined)
            gas=handle['PartType0']; u=gas['InternalEnergy'][:]*dataset_to_cgs_factor(gas['InternalEnergy'].attrs,a,h)
            assert np.isfinite(u).all() and (u>0).all()
            mean_energy.append(float(np.average(u,weights=gas['Masses'][:])))
            arrays[f'gas_specific_energy_erg_g_{epoch}']=u
            if comoving:
                group=handle['PartType5']
                bh_mass.append(float(np.sum(group['BH_Mass'][:]*dataset_to_cgs_factor(group['BH_Mass'].attrs,a,h)/MSUN_G)))
                inflow.append(float(np.sum(group['BH_InflowRate'][:]*dataset_to_cgs_factor(group['BH_InflowRate'].attrs,a,h)/MSUN_G*SECONDS_PER_YEAR)))
            counts.append(census);masses.append(mass)
    counts=np.asarray(counts);masses=np.asarray(masses);mean_energy=np.asarray(mean_energy)
    assert counts[0,0]==64 and counts[0,4]==0 and counts[0,5]==int(comoving)
    if name=='gas':
        assert (counts[:,0]==64).all() and mean_energy[-1]<mean_energy[0]*(1-1e-8)
        np.testing.assert_allclose(masses.sum(axis=1),masses[0].sum(),rtol=1e-10,atol=0)
    elif name=='stars':
        assert counts[-1,4]>0, 'No native stellar births in the declared control'
        from core.stars import audit_native_hydro_history
        history=audit_native_hydro_history(dict(run_directory=str(output),snapshots=[0,1,2],memory_gib=1.,
            tolerances=dict(mass_relative=1e-8,mass_absolute_msun=1e-6,
                momentum_relative=1e-8,momentum_absolute_msun_km_s=1e-6)),output/'history.json')
        assert history['status']=='PASS',history['status']
        assert (arrays['SSPBirthMode_2']==1).all(), 'Expected native SH03 births, not seeded SSPs'
    else:
        assert (counts[:,5]==1).all() and bh_mass[-1]>bh_mass[0], 'BH physical mass did not grow'
        assert np.isfinite(inflow).all() and max(inflow)>0
    if name!='stars':
        for species in (0,5) if comoving else (0,):
            reference=np.sort(arrays[f'particle_ids_0_{species}'])
            for epoch in (1,2):np.testing.assert_array_equal(np.sort(arrays[f'particle_ids_{epoch}_{species}']),reference)
    arrays.update(native_time=np.asarray(times),particle_counts=counts,mass_by_type_msun=masses,
        gas_mass_weighted_specific_energy_erg_g=mean_energy,bh_physical_mass_msun=np.asarray(bh_mass),
        bh_inflow_msun_per_year=np.asarray(inflow))
    np.savez_compressed(output/'arrays.npz',**arrays)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot(times,mean_energy,marker='.')
    axes[0].set(xlabel='Scale factor a' if comoving else 'Native physical time code',ylabel='Gas mean specific energy [erg/g]')
    if name=='black-holes':
        axes[1].plot(times,np.asarray(bh_mass)-bh_mass[0],marker='.')
        axes[1].set(ylabel='BH physical mass gain [Msun]')
    else:
        axes[1].plot(times,counts[:,0],marker='.',label='gas')
        axes[1].plot(times,counts[:,4],marker='.',label='stars');axes[1].legend()
        axes[1].set(ylabel='Actual particle count')
    axes[1].set_xlabel('Scale factor a' if comoving else 'Native physical time code')
    fig.suptitle('Native '+name+' control · original measurements')
    fig.savefig(output/'physics.png',dpi=140);plt.close(fig)
    return dict(control=name,seed=seed,stellar_random_seed=seed,initial_particles=int(counts[0].sum()),final_particles=int(counts[-1].sum()),
        particle_counts=counts.tolist(),native_times=times,comoving=comoving,
        gas_mean_specific_energy_erg_g=mean_energy.tolist(),bh_physical_mass_msun=bh_mass,bh_inflow_msun_per_year=inflow,
        binary_sha256=file_sha256(binary),
        units={'mass':'Msun','gas_specific_energy':'erg/g; effective EOS energy in the star-forming run',
            'BH_inflow':'Msun/year','native_time':'scale factor a' if comoving else 'L/V code time, with L=kpc and V=km/s'},
        implementation='core/research/controls.py:prepare_native_source_control → core/evolution.py:build_gadget/gadget_param_text → actual native GADGET → original snapshot census',
        scope='Fixed 64-gas execution control; stars start absent and BH is explicitly seeded. No delayed stellar returns, resolved galaxy/BH accuracy or convergence claim. BH accretion is checked for positive physical growth/inflow, not full source-budget reconciliation.')


def main():
    """Run one fixed physics control with a cold-build budget and retained errors."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('backend',choices=CHECKS)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'core/research/controls.py',ROOT/'core/evolution.py',ROOT/'core/stars.py',
        ROOT/'core/data.py',ROOT/'core/native.py',ROOT/'core/foundation.py']
    pins={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    def timeout(signum,frame): raise TimeoutError('Native physics control exceeded 900 seconds including compilation')
    def cancel(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,cancel);signal.alarm(900)
    try:record=dict(status='PASS',backend=args.backend,result=calculate(args.backend,args.output,args.seed))
    except ImportError as error:record=dict(status='UNAVAILABLE',backend=args.backend,reason=str(error))
    except Exception as error:record=dict(status='FAIL',backend=args.backend,reason=str(error),traceback=traceback.format_exc())
    finally:signal.alarm(0)
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in pins.items()):
        record=dict(status='FAIL',backend=args.backend,reason='A physics-worker source changed during execution')
    record['source_sha256']=pins
    (args.output/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in record.items() if k!='result'}),flush=True)
    return int(record['status']=='FAIL')


if __name__=='__main__':raise SystemExit(main())
