"""Native control build profiles and physical input states, without execution.

This module prepares requested source-control inputs. Solver execution and
measured comparison remain separate; input publication never certifies physics.
"""
from __future__ import annotations

import json
import math
import os
import h5py
import numpy as np
from pathlib import Path
from copy import deepcopy
from core.foundation import atomic_json, atomic_text, canonical_digest, file_sha256, file_sha256 as digest, G_CGS, KPC_CM, MSUN_G, SECONDS_PER_YEAR
from core.native import run_native_command


import copy
import shutil
import subprocess
from core.native import native_run_lock, run_native_command
from core.data import (verify_native_hdf5_set, verify_native_checkpoints,
                       export_numeric_audit, verify_numeric_audit,
                       epoch_contract, isolated_epoch_contract, CANONICAL_SCHEMA)

CONTROL_MODES = ('analytic', 'isolated_tree', 'isolated_treepm', 'periodic_treepm')


def control_config_text(profile):
    if isinstance(profile,dict) and profile.get('mode') in {'homogeneous_sidm_physical','homogeneous_sidm_canonical'}:
        if set(profile)!={'mode','sidm_enabled'} or type(profile['sidm_enabled']) is not bool:
            raise ValueError('Explicit gravity-free periodic SIDM control build profile required')
        options=['RESEARCH_NUMERICAL_CONTROL','RESEARCH_RESTART_STATE_CAPSULE','PERIODIC','OUTPUT_EXACT_SYNCHRONIZED',
            'MULTIPOLE_ORDER=2','NSOFTCLASSES=6','POSITIONS_IN_64BIT','DOUBLEPRECISION=1','OUTPUT_IN_DOUBLEPRECISION',
            'IDS_64BIT','FORCE_EQUAL_TIMESTEPS','PRESERVE_SHMEM_BINARY_INVARIANCE']
        if profile['sidm_enabled']:options.append('RESEARCH_SIDM_REFERENCE')
        if profile['mode']=='homogeneous_sidm_canonical':
            if not profile['sidm_enabled']:raise ValueError('Canonical transport requires its mapped SIDM source operator')
            options.append('RESEARCH_SIDM_CANONICAL_CONTROL')
        return '\n'.join(options)+'\n'
    if isinstance(profile,dict) and profile.get('mode')=='isolated_sidm_physical':
        if set(profile)!={'mode','multipole_order','sidm_enabled'} or type(profile['multipole_order']) is not int or not 1<=profile['multipole_order']<=5 or type(profile['sidm_enabled']) is not bool:
            raise ValueError('Explicit isolated physical collision build profile required')
        options=['RESEARCH_NUMERICAL_CONTROL','RESEARCH_RESTART_STATE_CAPSULE','SELFGRAVITY','MULTIPOLE_ORDER='+str(profile['multipole_order']),
            'NSOFTCLASSES=6','POSITIONS_IN_64BIT','DOUBLEPRECISION=1','OUTPUT_IN_DOUBLEPRECISION',
            'IDS_64BIT','OUTPUT_ACCELERATION','EVALPOTENTIAL','OUTPUT_POTENTIAL','FORCE_EQUAL_TIMESTEPS',
            'PRESERVE_SHMEM_BINARY_INVARIANCE']
        if profile['sidm_enabled']:options.append('RESEARCH_SIDM_REFERENCE')
        return '\n'.join(options)+'\n'
    if isinstance(profile,dict) and set(profile) in ({"source_kind","chemistry","black_holes"},{"source_kind","chemistry","black_holes","adiabatic_index"}):
        return source_control_build_options(profile["source_kind"],chemistry=profile["chemistry"],black_holes=profile["black_holes"],adiabatic_index=profile.get("adiabatic_index",5./3))
    if not isinstance(profile,dict) or set(profile)!={'mode','pm_grid','multipole_order','diagnostic_variant'}:
        raise ValueError('Malformed native control build profile')
    if profile['mode'] not in CONTROL_MODES:raise ValueError('Unsupported native solver control mode')
    pm=profile['pm_grid'];order=profile['multipole_order']
    if type(pm)!=int or pm<16 or pm>256 or pm&(pm-1) or type(order)!=int or not 1<=order<=5:
        raise ValueError('Invalid native control grid/multipole order')
    if type(profile['diagnostic_variant']) is not bool:raise ValueError('Diagnostic variant must be Boolean')
    options=['RESEARCH_NUMERICAL_CONTROL','SELFGRAVITY','MULTIPOLE_ORDER='+str(order),'POSITIONS_IN_64BIT','DOUBLEPRECISION=1',
             'OUTPUT_IN_DOUBLEPRECISION','IDS_64BIT','OUTPUT_ACCELERATION','EVALPOTENTIAL',
             'OUTPUT_POTENTIAL','FORCE_EQUAL_TIMESTEPS','PRESERVE_SHMEM_BINARY_INVARIANCE']
    if profile['mode']=='analytic':
        options.remove('SELFGRAVITY'); options+=['EXTERNALGRAVITY','EXTERNALGRAVITY_CONTROL']
    else:
        options+=['FORCETEST=1.0','FORCETEST_FIXEDPARTICLESET']
    if profile['mode'] in ('isolated_treepm','periodic_treepm'):
        options+=['PMGRID='+str(pm),'DOUBLEPRECISION_FFTW','ASMTH=1.25','TREEPM_NOTIMESPLIT']
    if profile['mode']=='periodic_treepm':options+=['PERIODIC']
    if profile['diagnostic_variant']:options+=['OUTPUT_TIMESTEP']
    # Both profiles contain no SIDM operator. Adding output diagnostics changes
    # binary bytes while preserving the collisionless equations being tested.
    return '\n'.join(options)+'\n'


def source_control_build_options(kind,*,chemistry=False,black_holes=False,adiabatic_index=5./3):
    """Native source-control build definitions; callers explicitly execute later."""
    if kind not in {'shock_tube','cooling_box','sh03_equilibrium','closed_ssp','bh_accretion','bh_dynamics'} or type(chemistry) is not bool or type(black_holes) is not bool:raise ValueError('Declare a supported native source control profile')
    if chemistry and kind not in {'cooling_box','closed_ssp'}:raise ValueError('Selected chemistry controls need a single-phase gas source profile')
    if type(adiabatic_index) not in (int,float) or not math.isfinite(adiabatic_index) or not 1<adiabatic_index<=2:raise ValueError('Declare a supported ideal gas adiabatic index')
    flags=['GAMMA='+format(adiabatic_index,'.17g'),'POSITIONS_IN_64BIT','IDS_64BIT','DOUBLEPRECISION=1','OUTPUT_IN_DOUBLEPRECISION','FORCE_EQUAL_TIMESTEPS',
        'OUTPUT_PRESSURE','OUTPUT_TIMESTEP','OUTPUT_POTENTIAL','OUTPUT_ACCELERATION','EVALPOTENTIAL',
        'PRESERVE_SHMEM_BINARY_INVARIANCE']
    if kind!='shock_tube':flags+=['COOLING']
    if kind in {'sh03_equilibrium','closed_ssp'} or chemistry:flags+=['STARFORMATION']
    if kind=='closed_ssp' or chemistry:flags+=['STELLAR_EVOLUTION']
    if chemistry:flags+=['RESEARCH_GRACKLE']
    if black_holes or kind.startswith('bh_'):flags+=['BLACK_HOLES']
    if kind in {'sh03_equilibrium','closed_ssp','bh_accretion','bh_dynamics'}:flags+=['SELFGRAVITY','PERIODIC']
    return '\n'.join(flags)+'\n'


def source_control_particles(kind,parameters,*,memory_gib):
    """Deterministic physical input arrays for native hydro/source controls.

    This returns IC state only. It neither evolves particles nor certifies the
    physical result. The writer supplies the exact native unit/time contract.
    Lengths, densities, specific energies and masses use those declared native
    code units. A gas cell has mass rho*(box_length/grid_size)^3.
    """
    if kind not in {'shock_tube','cooling_box','sh03_equilibrium','closed_ssp','bh_accretion','bh_dynamics'}:raise ValueError('Unknown native source control')
    if not isinstance(parameters,dict):raise ValueError('Declare explicit control parameters')
    required={'grid_size','box_length','gas_density','gas_internal_energy','seed'}
    extra=({'left_state','right_state','gamma'} if kind=='shock_tube' else
        {'ssp_mass','ssp_birth_time','ssp_birth_metallicity','ssp_prompt_fraction'} if kind=='closed_ssp' else
        {'bh_masses','bh_physical_masses','bh_positions','bh_velocities','stellar_background_mass','stellar_background_velocity_dispersion'} if kind=='bh_dynamics' else
        {'bh_masses','bh_physical_masses','bh_positions','bh_velocities'} if kind=='bh_accretion' else set())
    if not required<=set(parameters) or set(parameters)-required-extra:raise ValueError('Declare complete bounded source fixture state')
    n=parameters['grid_size'];box=parameters['box_length'];seed=parameters['seed']
    if type(n) is not int or not 4<=n<=128 or type(seed) is not int or not 0<=seed<2**64:raise ValueError('Bounded grid and exact seed required')
    for key in ('box_length','gas_density','gas_internal_energy'):
        value=parameters[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive finite native fixture '+key+' required')
    if isinstance(memory_gib, bool) or not isinstance(memory_gib, (int, float)) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('memory_gib must be finite and positive')
    if n**3*2048+64*1024**2>memory_gib*1024**3:raise MemoryError('Native source fixture exceeds declared workspace')
    q=(np.stack(np.meshgrid(*[np.arange(n)]*3,indexing='ij'),axis=-1).reshape(-1,3)+.5)*box/n
    volume=(box/n)**3;rho=np.full(len(q),parameters['gas_density']);u=np.full(len(q),parameters['gas_internal_energy']);v=np.zeros_like(q)
    if kind=='shock_tube':
        from core.stars import sod_riemann_reference
        for key in ('left_state','right_state','gamma'):
            if key not in parameters:raise ValueError('Sod control requires both physical states and gamma')
        initial=sod_riemann_reference(q[:,0],0.,left=parameters['left_state'],right=parameters['right_state'],discontinuity=box/2,gamma=parameters['gamma'])
        rho=initial['density'];v[:,0]=initial['velocity'];u=initial['specific_internal_energy']
    ids=np.arange(len(q),dtype=np.uint64)+np.uint64(2**60+1)
    species={0:{'Coordinates':q,'Velocities':v,'Masses':rho*volume,'InternalEnergy':u,'ParticleIDs':ids}}
    if kind=='closed_ssp':
        from core.stars import validate_ssp_birth_arrays
        for key in ('ssp_mass','ssp_birth_time','ssp_birth_metallicity','ssp_prompt_fraction'):
            if key not in parameters:raise ValueError('Closed SSP requires immutable birth state and declared prompt fraction')
        m=parameters['ssp_mass'];birth=parameters['ssp_birth_time'];z=parameters['ssp_birth_metallicity'];beta=parameters['ssp_prompt_fraction']
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (m, birth, z, beta))
                or not 0 < beta < 1 or m <= 0 or birth < 0 or not 0 <= z <= 1):
            raise ValueError('Finite physical SSP mass, birth time, metallicity and prompt fraction required')
        sid=np.array([2**60+len(q)+1],dtype=np.uint64)
        # m is the locked mass after prompt recycling: initial formed mass is
        # m/(1-beta), and prompt return is beta*m/(1-beta). Their difference
        # is exactly the intended locked component before delayed evolution.
        state={'ParticleIDs':sid,'SSPParentGasID':np.array([0],dtype=np.uint64),'SSPBirthStateVersion':np.ones(1,dtype=np.uint64),
            'SSPBirthMode':np.full(1,2,dtype=np.uint64),'StellarRandomCounter':np.ones(1,dtype=np.uint64),
            'SSPInitialMass':np.array([m/(1-beta)]),'SSPInitialLockedMass':np.array([m]),'SSPPromptReturnedMass':np.array([m*beta/(1-beta)]),
            'SSPCurrentMass':np.array([m]),'SSPBirthScaleFactor':np.zeros(1),'SSPBirthNativeTime':np.array([birth]),'SSPBirthMetallicity':np.array([z])}
        validate_ssp_birth_arrays(state,factor_sn=beta,actual_native_time=birth,comoving=False)
        species[4]={**state,'Coordinates':np.full((1,3),box/2),'Velocities':np.zeros((1,3)),'Masses':np.array([m])}
    if kind.startswith('bh_'):
        masses=np.asarray(parameters.get('bh_masses'),dtype=float);physical=np.asarray(parameters.get('bh_physical_masses'),dtype=float)
        x=np.asarray(parameters.get('bh_positions'),dtype=float);vel=np.asarray(parameters.get('bh_velocities'),dtype=float)
        if masses.ndim!=1 or not 1<=len(masses)<=16 or physical.shape!=masses.shape or x.shape!=(len(masses),3) or vel.shape!=x.shape or any(not np.isfinite(a).all() for a in (masses,physical,x,vel)) or np.any(masses<=0) or np.any(physical<=0) or np.any(physical>masses) or np.any((x<0)|(x>=box)):raise ValueError('Declare finite BH composite/physical masses and actual phase-space state')
        species[5]={'Coordinates':x,'Velocities':vel,'Masses':masses,'BH_Mass':physical,'ParticleIDs':np.arange(len(masses),dtype=np.uint64)+np.uint64(2**60+len(q)+1)}
    if kind=='bh_dynamics':
        mass=parameters.get('stellar_background_mass');sigma=parameters.get('stellar_background_velocity_dispersion')
        if type(mass) not in (int,float) or not math.isfinite(mass) or mass<=0 or type(sigma) not in (int,float) or not math.isfinite(sigma) or sigma<0:raise ValueError('BH dynamics requires explicit stellar background total mass and one-dimensional dispersion')
        # Counter-keyed independent normal velocities, centered and normalized
        # to the declared measured sample covariance trace. No global RNG state.
        rng=np.random.Generator(np.random.Philox(seed));sv=rng.standard_normal(q.shape);sv-=sv.mean(axis=0)
        sv*=sigma/(np.sqrt(np.mean(sv*sv)) if sigma else 1.)
        species[4]={'Coordinates':q.copy(),'Velocities':sv,'Masses':np.full(len(q),mass/len(q)),
            'ParticleIDs':np.arange(len(q),dtype=np.uint64)+np.uint64(2**60+len(q)+len(masses)+1)}
    return {'species':species,'box_length':box,'native_seed':seed,'kind':kind,
        'interpretation':'Analytic input state only; equilibrium, formation and source evolution require later native execution',
        'boundary_scope':'Shock tube is measured before waves reach domain boundaries; other control profiles declare their native periodic/isolated frame'}


def prepare_native_source_control(request,destination_prefix):
    """Write flat IC/config/profile inputs in an existing directory, never run."""
    from core.foundation import atomic_json, atomic_text
    keys={'kind','parameters','native_units','chemistry','black_holes','memory_gib','base_config'}
    if not isinstance(request,dict) or set(request)!=keys:raise ValueError('Declare the complete source-control input request')
    units=request['native_units']
    if set(units)!={'length_cm','mass_g','velocity_cm_s','time','hubble_param','comoving'} or type(units['comoving']) is not bool:raise ValueError('Declare complete native control units and clock')
    if any(type(units[k]) not in (int,float) or not math.isfinite(units[k]) or units[k]<=0 for k in ('length_cm','mass_g','velocity_cm_s','hubble_param')) or type(units['time']) not in (int,float) or not math.isfinite(units['time']) or units['time']<0:raise ValueError('Native source-control units must be finite positive, with t>=0')
    if request['kind']=='closed_ssp' and units['comoving']:raise ValueError('Closed SSP fixture explicitly uses an isolated proper-time clock')
    if units['comoving'] and units['time']<=0:raise ValueError('A comoving native source-control epoch must be positive')
    if request['kind'].startswith('bh_') and not units['comoving']:raise ValueError('Current BH source profile requires a declared comoving control clock')
    if not units['comoving'] and units['hubble_param']!=1:raise ValueError('Isolated native source controls require h=1')
    prefix=Path(destination_prefix).resolve()
    if not prefix.parent.is_dir():raise FileNotFoundError('Keep source-control files in an existing directory')
    ic=prefix.with_name(prefix.name+'.ic.hdf5');profile=prefix.with_name(prefix.name+'.Config.sh');cfg=prefix.with_name(prefix.name+'.config.json');manifest=prefix.with_name(prefix.name+'.manifest.json')
    if any(p.exists() for p in (ic,profile,cfg,manifest)):raise FileExistsError('Preserve earlier source-control inputs')
    fixture=source_control_particles(request['kind'],request['parameters'],memory_gib=request['memory_gib'])
    gamma=request['parameters'].get('gamma',5./3)
    flags=source_control_build_options(request['kind'],chemistry=request['chemistry'],black_holes=request['black_holes'],adiabatic_index=gamma)
    with h5py.File(ic,'x') as f:
        counts=np.array([len(fixture['species'].get(i,{}).get('Masses',[])) for i in range(6)],dtype=np.uint64)
        f.create_group('Header').attrs.update({'Time':units['time'],'Redshift':1/units['time']-1 if units['comoving'] else 0.,'BoxSize':fixture['box_length'],
            'MassTable':np.zeros(6),'NumFilesPerSnapshot':1,'NumPart_ThisFile':counts,'NumPart_Total':counts})
        f.create_group('Parameters').attrs.update({'UnitLength_in_cm':units['length_cm'],'UnitMass_in_g':units['mass_g'],'UnitVelocity_in_cm_per_s':units['velocity_cm_s'],'HubbleParam':units['hubble_param'],'ComovingIntegrationOn':int(units['comoving'])})
        for kind,fields in fixture['species'].items():
            group=f.create_group(f'PartType{kind}')
            for name,array in fields.items():
                dataset=group.create_dataset(name,data=array,fletcher32=True)
                cgs=units['length_cm'] if name=='Coordinates' else units['mass_g'] if name=='Masses' or name.endswith('Mass') or name=='BH_Mass' else units['velocity_cm_s'] if name=='Velocities' else units['velocity_cm_s']**2 if name=='InternalEnergy' else units['length_cm']/units['velocity_cm_s'] if name=='SSPBirthNativeTime' and not units['comoving'] else 1.
                # A physical value is raw * to_cgs * a^a_scaling * h^h_scaling.
                # Isolated coordinates/velocities are already proper native
                # quantities; cosmological a/h exponents must not leak into them.
                a_scaling = (1. if name == 'Coordinates' else .5 if name == 'Velocities' else 0.) if units['comoving'] else 0.
                mass_field = name == 'Masses' or name.endswith('Mass') or name == 'BH_Mass'
                h_scaling = -1. if units['comoving'] and (name == 'Coordinates' or mass_field) else 0.
                dataset.attrs.update({'to_cgs':cgs,'a_scaling':a_scaling,'h_scaling':h_scaling})
        f.attrs['source_control_kind']=request['kind'];f.attrs['physics_acceptance']='NOT_ACCEPTED'
    with ic.open('rb') as handle:os.fsync(handle.fileno())
    atomic_text(profile,flags);atomic_json(cfg,request['base_config'])
    output={'schema':'native-source-control-inputs-v1','request':request,'files':[{'path':str(p),'sha256':digest(p)} for p in (ic,profile,cfg)],
        'execution_status':'NOT_RUN','physics_acceptance':'NOT_ACCEPTED',
        'build_profile':{'source_kind':request['kind'],'chemistry':request['chemistry'],'black_holes':request['black_holes'],'adiabatic_index':gamma},
        'execution_requirement':'Use build_gadget(control_profile=build_profile) and the exact IC/units/source model in the normal owned native run; compare returned snapshots with independent reference functions'}
    atomic_json(manifest,output);atomic_text(manifest.with_name(manifest.name+'.sha256'),digest(manifest)+'\n')
    return output


# Physical Plummer collision ICs and optional native isolated SIDM evolution.
GYR_S=1e9*SECONDS_PER_YEAR


def validate_collision_control(request):
    if request.get('schema')!='physical-plummer-collision-control-v1':raise ValueError('Physical Plummer control request required')
    for field in ('phase_seed','first_particle_id','maximum_particles','chunk_particles','multipole_order','maximum_outputs'):
        if type(request[field]) is not int or request[field]<0:raise ValueError('Exact nonnegative control integer required: '+field)
    if not 1<=request['chunk_particles']<=request['maximum_particles'] or not 1<=request['multipole_order']<=5:
        raise ValueError('Invalid frozen control allocation or gravity multipole order')
    if request['phase_seed']>=2**128 or request['first_particle_id']<1:
        raise ValueError('Control seed or identity outside supported exact domain')
    for field in ('memory_gib','box_side_proper_kpc','initial_time_gyr','maximum_timestep_gyr','softening_proper_kpc','wall_seconds'):
        value=request[field]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<0 if field=='initial_time_gyr' else value<=0):
            raise ValueError('Explicit finite physical control quantity required: '+field)
    units=request['native_units']
    if set(units)!={'length_cm','mass_g','velocity_cm_per_s'} or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0 for value in units.values()):
        raise ValueError('Explicit positive native physical units required')
    times=request['output_times_gyr']
    if not isinstance(times,list) or not 2<=len(times)<=request['maximum_outputs'] or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in times) or any(right<=left for left,right in zip(times,times[1:])) or times[0]!=request['initial_time_gyr']:
        raise ValueError('Bounded increasing saved physical clocks beginning at the exact initial clock required')
    if request['maximum_timestep_gyr']>times[-1]-times[0]:raise ValueError('Source timestep exceeds the requested physical evolution interval')
    halos=request['halos']
    if not isinstance(halos,list) or not 1<=len(halos)<=2:raise ValueError('One or two physical Plummer populations required')
    total=0;masses=[]
    for halo in halos:
        if type(halo['particles']) is not int or halo['particles']<2:raise ValueError('Physical halo requires at least two macro particles')
        for field in ('mass_msun','scale_proper_kpc'):
            if isinstance(halo[field],bool) or not isinstance(halo[field],(int,float)) or not math.isfinite(halo[field]) or halo[field]<=0:
                raise ValueError('Positive physical Plummer mass and radius required')
        for field in ('center_proper_kpc','bulk_velocity_kms'):
            value=halo[field]
            if not isinstance(value,list) or len(value)!=3 or any(isinstance(item,bool) or not isinstance(item,(int,float)) or not math.isfinite(item) for item in value):
                raise ValueError('Explicit finite Cartesian Plummer state required')
        total+=halo['particles'];masses.append(halo['mass_msun']/halo['particles'])
    if total>request['maximum_particles'] or request['first_particle_id']+total>=2**63:
        raise ValueError('Physical control exceeds particle or native reserved-bit identity allocation')
    if request['chunk_particles']*512>request['memory_gib']*1024**3:
        raise MemoryError('Physical phase-space sampler exceeds frozen workspace')
    if total>=2**32:raise ValueError('Single-file physical control particle census exceeds uint32 header capacity')
    if type(request['mpi_ranks']) is not int or not 1<=request['mpi_ranks']<=65536:
        raise ValueError('Explicit native MPI rank allocation required')
    tolerance=request['output_time_absolute_tolerance_native']
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Positive frozen native output-time quantization tolerance required')
    if type(request['sidm'].get('enabled')) is not bool:
        raise ValueError('Physical control SIDM state must be explicit')
    if request['sidm'].get('enabled',False):
        from core.sidm_physics import sidm_native_parameters
        sidm_native_parameters(request['sidm'],physics='dm',mpi_ranks=request['mpi_ranks'],max_mem_rank_mib=request['memory_gib']*1024/request['mpi_ranks'])
        tolerance=request['sidm']['macro_mass_relative_tolerance']
        if max(masses)-min(masses)>tolerance*max(masses):
            raise ValueError('Native SIDM collision requires equal macro masses; change explicit masses/counts')
    if type(request['omp_threads']) is not int or request['omp_threads']!=1:
        raise ValueError('Physical control source currently requires one native worker thread per rank')
    return total


def prepare_plummer_collision(request,destination):
    import numpy as np
    import h5py
    total=validate_collision_control(request)
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Physical control inputs are immutable')
    destination.mkdir(parents=True)
    atomic_json(destination/'request.json',request)
    units=request['native_units'];time_unit=units['length_cm']/units['velocity_cm_per_s']
    if not math.isfinite(time_unit) or time_unit<=0:raise ValueError('Native control time unit outside finite domain')
    box=request['box_side_proper_kpc'];rng=np.random.Generator(np.random.Philox(request['phase_seed']))
    state=[];cursor=0;output=destination/'collision-ic.hdf5'
    # Chunked draws keep the requested physical particle count fixed. Failure
    # to fit the explicit box is preserved, never repaired by clipping/reseeding.
    with h5py.File(output,'x') as file:
        counts=np.array([0,total,0,0,0,0],dtype=np.uint64)
        header=file.create_group('Header')
        header.attrs.update(Time=request['initial_time_gyr']*GYR_S/time_unit,Redshift=0.,
            BoxSize=box*KPC_CM/units['length_cm'],MassTable=np.zeros(6),NumFilesPerSnapshot=1,
            NumPart_ThisFile=counts.astype(np.uint32),NumPart_Total=(counts&np.uint64(2**32-1)).astype(np.uint32),
            NumPart_Total_HighWord=(counts>>np.uint64(32)).astype(np.uint32),HubbleParam=1.,Omega0=0.,OmegaLambda=0.,
            Flag_DoublePrecision=1,Flag_Sfr=0,Flag_Cooling=0,Flag_StellarAge=0,Flag_Metals=0,Flag_Feedback=0)
        particles=file.create_group('PartType1')
        chunk=min(request['chunk_particles'],total)
        for name in ('Coordinates','Velocities'):
            particles.create_dataset(name,(total,3),dtype='f8',chunks=(chunk,3))
        particles.create_dataset('Masses',(total,),dtype='f8',chunks=(chunk,))
        particles.create_dataset('ParticleIDs',(total,),dtype='u8',chunks=(chunk,))
        for number,halo in enumerate(request['halos']):
            first=cursor;position_sum=np.zeros(3);velocity_sum=np.zeros(3);maximum_radius=0.
            for start in range(0,halo['particles'],request['chunk_particles']):
                n=min(request['chunk_particles'],halo['particles']-start)
                radial=(rng.integers(0,2**52,size=n,dtype=np.uint64).astype(np.float64)+.5)/2**52
                radius=halo['scale_proper_kpc']/np.sqrt(np.expm1(-2*np.log(radial)/3))
                cosine=2*rng.random(n)-1;azimuth=2*np.pi*rng.random(n)
                sine=np.sqrt(1-cosine*cosine)
                direction=np.column_stack((sine*np.cos(azimuth),sine*np.sin(azimuth),cosine))
                position=np.asarray(halo['center_proper_kpc'])+radius[:,None]*direction
                # f(E) proportional to (-E)^(7/2); at fixed r, v^2/v_escape^2
                # has Beta(3/2,9/2), including the v^2 phase-space Jacobian.
                escape=np.sqrt(2*G_CGS*halo['mass_msun']*MSUN_G/(KPC_CM*np.hypot(radius,halo['scale_proper_kpc'])))/1.e5
                speed=escape*np.sqrt(rng.beta(1.5,4.5,n))
                cosine=2*rng.random(n)-1;azimuth=2*np.pi*rng.random(n);sine=np.sqrt(1-cosine*cosine)
                velocity=np.asarray(halo['bulk_velocity_kms'])+speed[:,None]*np.column_stack((sine*np.cos(azimuth),sine*np.sin(azimuth),cosine))
                if not np.isfinite(position).all() or not np.isfinite(velocity).all() or np.any(position<0) or np.any(position>=box):
                    raise ValueError('Physical Plummer sample leaves the explicit native box; no truncation or resolution change applied')
                stop=cursor+n
                particles['Coordinates'][cursor:stop]=position*KPC_CM/units['length_cm']
                particles['Velocities'][cursor:stop]=velocity*1.e5/units['velocity_cm_per_s']
                particles['Masses'][cursor:stop]=halo['mass_msun']/halo['particles']*MSUN_G/units['mass_g']
                particles['ParticleIDs'][cursor:stop]=np.arange(cursor,stop,dtype=np.uint64)+np.uint64(request['first_particle_id'])
                position_sum+=position.sum(axis=0);velocity_sum+=velocity.sum(axis=0);maximum_radius=max(maximum_radius,float(radius.max()));cursor=stop
            state.append({'population':number,'first_particle_id':request['first_particle_id']+first,
                'particle_count':halo['particles'],'mass_msun':halo['mass_msun'],'macro_mass_msun':halo['mass_msun']/halo['particles'],
                'observed_center_proper_kpc':(position_sum/halo['particles']).tolist(),
                'observed_bulk_velocity_kms':(velocity_sum/halo['particles']).tolist(),'maximum_sampled_radius_proper_kpc':maximum_radius})
        file.flush()
    with output.open('rb') as stream:os.fsync(stream.fileno())
    report={'schema':'physical-plummer-collision-ic-v1','execution_status':'IC_PREPARED','synthetic_fixture':False,
        'population_origin':'analytic physical Plummer phase-space initial states; not evolved population evidence',
        'request_hash':canonical_digest(request),'populations':state,'native_units':units,'native_hubble_parameter':1.,
        'initial_time_gyr':request['initial_time_gyr'],'ic':{'path':str(output),'sha256':file_sha256(output)},
        'sampling':'Philox; untruncated Plummer inverse mass CDF; isotropic Beta(3/2,9/2) conditional squared escape-speed fraction',
        'finite_population_policy':'sample center and momentum fluctuations retained; no velocity rescaling or recentering',
        'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)},'science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination/'collision-ic.json',report)
    return report


def collision_input_bindings(parameters):
    validate_collision_control(parameters['request'])
    reference=parameters['base_configuration'];path=Path(reference['path']).resolve()
    if path.stat().st_size>1024**2 or file_sha256(path)!=reference['sha256']:
        raise ValueError('Frozen physical control base configuration changed or exceeds intake limit')
    sources={str(path):reference['sha256']}
    if parameters['request']['sidm']['enabled']:
        model=parameters['request']['sidm']['physical_model'];asset=Path(model['path']).resolve()
        if file_sha256(asset)!=model['sha256']:raise ValueError('Physical collision angular model changed')
        sources[str(asset)]=model['sha256']
    return sources


def validate_collision_stage(parameters):
    validate_collision_control(parameters['request'])
    reference=parameters['base_configuration']
    if set(reference)!={'path','sha256'}:raise ValueError('Physical control base configuration requires an exact source pin')


def physical_collision_runner(parameters,context):
    import yaml
    from core.evolution import build_gadget, gadget_param_text
    from core.data import verify_native_hdf5_set
    request=parameters['request'];total=validate_collision_control(request)
    collision_input_bindings(parameters)
    if request['memory_gib']>context.resources['memory_gib'] or request['mpi_ranks']!=context.resources['mpi_ranks'] or request['omp_threads']!=context.resources['omp_threads']:
        raise ValueError('Physical collision control cannot change its workflow resource allocation')
    reference=parameters['base_configuration'];path=Path(reference['path']).resolve()
    if file_sha256(path)!=reference['sha256']:raise ValueError('Physical control base configuration changed')
    cfg=yaml.safe_load(path.read_text());cfg=deepcopy(cfg)
    cfg['hardware'].update(mpi_ranks=request['mpi_ranks'],omp_threads=1,memory_limit_gb=request['memory_gib'],
        cpu_threads=request['mpi_ranks'],build_jobs=request['mpi_ranks'])
    cfg['simulation'].update(physics='dm',num_files_per_snapshot=1,softening_kpc_h=request['softening_proper_kpc'],
        softening_max_physical_kpc_h=request['softening_proper_kpc'],restart_every_seconds=request['wall_seconds'],
        time_limit_cpu_seconds=request['wall_seconds'])
    cfg['halos'].update(fof=False,subfind=False);cfg['sidm']=deepcopy(request['sidm'])
    if cfg['cosmology'].get('model')!='lcdm' or cfg['cosmology'].get('gadget_background_table'):
        raise ValueError('Isolated physical control base must not retain a cosmological background table/model')
    for component in ('black_holes','stellar_evolution','chemistry'):
        if cfg.get(component,{}).get('enabled',False):
            raise ValueError('DM-only isolated collision cannot retain enabled '+component)
    cfg['cosmology']['H0']=100.
    root=context.generation/'collision';root.mkdir();(root/'input').mkdir();(root/'data').mkdir()
    prepared=prepare_plummer_collision(request,root/'physical-input')
    profile={'mode':'isolated_sidm_physical','multipole_order':request['multipole_order'],'sidm_enabled':request['sidm']['enabled']}
    binary=build_gadget(cfg,root,control_profile=profile)
    text=gadget_param_text(cfg,root,Path(prepared['ic']['path']))
    values={line.split()[0]:' '.join(line.split()[1:]) for line in text.splitlines() if line.strip() and not line.lstrip().startswith('%')}
    units=request['native_units'];time_unit=units['length_cm']/units['velocity_cm_per_s'];times=[value*GYR_S/time_unit for value in request['output_times_gyr']]
    values.update(ComovingIntegrationOn=0,HubbleParam=1.,Omega0=0.,OmegaLambda=0.,OmegaBaryon=0.,
        TimeBegin=times[0],TimeMax=times[-1],TimeOfFirstSnapshot=times[0],TimeBetStatistics=request['maximum_timestep_gyr']*GYR_S/time_unit,
        MaxSizeTimestep=request['maximum_timestep_gyr']*GYR_S/time_unit,MinSizeTimestep=0.,
        UnitLength_in_cm=units['length_cm'],UnitMass_in_g=units['mass_g'],UnitVelocity_in_cm_per_s=units['velocity_cm_per_s'],
        GravityConstantInternal=G_CGS*units['mass_g']/(units['length_cm']*units['velocity_cm_per_s']**2),NativeControlPauseStep=0)
    values['SofteningClassOfPartType1']=1
    values['SofteningComovingClass1']=request['softening_proper_kpc']*KPC_CM/units['length_cm']
    values['SofteningMaxPhysClass1']=values['SofteningComovingClass1']
    values.pop('ActivePartFracForPMinsteadOfEwald',None)
    atomic_text(root/'input/param.txt','\n'.join(str(key)+' '+str(value) for key,value in values.items())+'\n')
    atomic_text(root/'input/output_times.txt','\n'.join(format(value,'.17g')+' 1' for value in times)+'\n')
    atomic_json(root/'config.json',cfg)
    inputs=[Path(prepared['ic']['path']),root/'input/param.txt',root/'input/output_times.txt',root/'input/Config.sh',path]
    if request['sidm']['enabled']:inputs.append(Path(request['sidm']['physical_model']['path']))
    env=os.environ.copy();env['OMP_NUM_THREADS']='1'
    with (root/'native.log').open('wb') as log:
        run_native_command(['mpirun','-np',str(request['mpi_ranks']),str(binary),str(root/'input/param.txt'),'0'],
            cwd=root/'input',receipt_path=root/'native-invocation.json',inputs=inputs,stdout=log,stderr=log,timeout=request['wall_seconds'],env=env)
    snapshots=sorted((root/'data').glob('snapshot_*.hdf5'))
    if len(snapshots)!=len(times):raise ValueError('Physical collision control did not produce every requested saved epoch')
    import h5py
    sources={**collision_input_bindings(parameters),str(Path(binary).resolve()):file_sha256(binary),
        str((root/'native-invocation.json').resolve()):file_sha256(root/'native-invocation.json'),
        str(Path(prepared['ic']['path'])):prepared['ic']['sha256'],**prepared['source_hashes']};saved=[]
    for snapshot,expected in zip(snapshots,times):
        verify_native_hdf5_set([snapshot],require_publication=True)
        with h5py.File(snapshot,'r') as file:
            actual=float(file['Header'].attrs['Time'])
            if int(file['Parameters'].attrs['ComovingIntegrationOn'])!=0 or abs(actual-expected)>request['output_time_absolute_tolerance_native']:
                raise ValueError('Physical control native output differs from frozen isolated clock')
            if len(file['PartType1/ParticleIDs'])!=total:raise ValueError('Physical control native population census changed')
        sources[str(snapshot.resolve())]=file_sha256(snapshot)
        saved.append({'path':str(snapshot.resolve()),'sha256':sources[str(snapshot.resolve())],'native_time_code':actual})
    report={'schema':'isolated-physical-collision-evolution-v1','execution_status':'COMPLETE','status':'EVOLVED_ONLY',
        'request_hash':canonical_digest(request),'initial_conditions':prepared,'snapshots':saved,'source_hashes':sources,
        'native_invocation':str(root/'native-invocation.json'),'science_acceptance':'NOT_ESTABLISHED',
        'coverage':'actual evolved dedicated physical control; rare-event counts and physical acceptance require downstream measurements'}
    atomic_json(context.generation/'collision-evolution.json',report)
    return {'measurements':{'native_evolved_particles':total,'saved_snapshots':len(saved)},'scientific_exclusions':[]}


# Explicitly invoked numerical controls; no demonstration/test imports.
ROOT = Path(__file__).resolve().parents[2]
POTENTIALS = {'plummer': 1, 'hernquist': 2, 'nfw': 3, 'harmonic': 4}

def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return float(value)


def potential_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) != {'kind', 'gm', 'scale', 'axis_y', 'axis_z'}:
        raise ValueError('Declare kind, gm, scale, axis_y and axis_z')
    if parameters['kind'] not in POTENTIALS:
        raise ValueError('Unsupported analytic potential')
    return {**parameters, **{k: positive(parameters[k], k) for k in ('gm', 'scale', 'axis_y', 'axis_z')}}


def analytic_field(position, parameters):
    p = potential_parameters(parameters)
    x = np.asarray(position, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError('Finite Cartesian positions required')
    gm, b, kind = p['gm'], p['scale'], p['kind']
    r = np.linalg.norm(x, axis=-1)
    if kind == 'harmonic':
        w2 = gm/b**3 * np.array([1., p['axis_y']**2, p['axis_z']**2])
        return .5*np.sum(x*x*w2, axis=-1), -x*w2, np.broadcast_to(-np.diag(w2), x.shape[:-1]+(3,3)).copy()
    if kind != 'plummer' and np.any(r == 0):
        raise ValueError('Central cusp excluded from analytic reference domain')
    if kind == 'plummer':
        d = np.sqrt(r*r+b*b); phi = -gm/d
        f = -gm/d**3; radial = 3*gm/d**5
    elif kind == 'hernquist':
        phi = -gm/(r+b); f = -gm/(r*(r+b)**2)
        radial = gm*(1/(r**3*(r+b)**2)+2/(r**2*(r+b)**3))
    else:
        u = r/b
        # The enclosed mass is integral_0^u t/(1+t)^2 dt. For small u,
        # integrate on [0,1] after t=u*s; unlike subtraction or long-double
        # libm, this remains stable across platforms and is independent of
        # the native solver's small-radius polynomial.
        m=np.log1p(u)-u/(1+u)
        small=u<.1
        if np.any(small):
            nodes,weights=np.polynomial.legendre.leggauss(16)
            s=(nodes+1)/2;weights=weights/2
            values=u[small]
            m=np.array(m,copy=True)
            m[small]=values**2*np.sum(weights*s/(1+values[...,None]*s)**2,axis=-1)
        phi = -gm*np.log1p(u)/r; f = -gm*m/r**3
        dm = r/(r+b)**2
        radial = gm*(3*m/r**5-dm/r**4)
    a = f[...,None]*x
    t = f[...,None,None]*np.eye(3)+radial[...,None,None]*x[..., :,None]*x[...,None,:]
    return phi, a, t


def analytic_orbits(position, velocity, times, parameters):
    x, v, times = map(lambda z: np.asarray(z, dtype=np.float64), (position, velocity, times))
    if x.ndim != 2 or x.shape[1] != 3 or v.shape != x.shape or not np.all(np.isfinite([x,v])):
        raise ValueError('Orbit initial states require finite matching (N,3) arrays')
    if times.ndim != 1 or len(times)<2 or np.any(~np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError('Orbit times must increase strictly')
    from scipy.integrate import solve_ivp
    n = len(x)
    def rhs(_, state):
        return np.r_[state[3*n:], analytic_field(state[:3*n].reshape(n,3), parameters)[1].ravel()]
    solution = solve_ivp(rhs, (times[0],times[-1]), np.r_[x.ravel(),v.ravel()],
                         method='DOP853', t_eval=times, rtol=2e-12, atol=2e-13)
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError('Independent orbit reference failed: '+solution.message)
    return solution.y[:3*n].T.reshape(-1,n,3), solution.y[3*n:].T.reshape(-1,n,3)


def stencil_fixture(parameters, step):
    """Central plus/minus probes; IDs and axes remain explicit through reordering."""
    positive(step,'tidal stencil step')
    centers = np.array([[.7,.2,.3],[-.4,.8,.2],[1.3,-.4,.6],[-1.1,-.7,.9]])*parameters['scale']
    rows=[]
    for c in centers:
        rows.append(c)
        for axis in np.eye(3):
            rows.extend([c+step*axis,c-step*axis])
    x=np.asarray(rows)
    v=np.zeros_like(x)
    if parameters['kind']=='harmonic':
        v = np.cross(x,[.1,.2,.3])
    else:
        a=analytic_field(x,parameters)[1]
        tangent=np.cross([0.,0.,1.],x)
        v=tangent/np.linalg.norm(tangent,axis=1)[:,None]*np.sqrt(np.linalg.norm(a,axis=1)*np.linalg.norm(x,axis=1))[:,None]
    return x,v


def stencil_tides(acceleration, step):
    a=np.asarray(acceleration,dtype=np.float64)
    if a.shape != (28,3) or np.any(~np.isfinite(a)):
        raise ValueError('Tidal stencil requires four groups of seven exact-ID probes')
    a=a.reshape(4,7,3)
    return np.stack([(a[:,1+2*j]-a[:,2+2*j])/(2*step) for j in range(3)],axis=-1)


def spline_direct_field(position, mass, force_softening, *, gravitational_constant=1., block_size=128):
    """Independent O(N^2) spline force reference matching native h=2.8*epsilon.

Self terms are removed. This is an isolated discrete reference, not a claim
that a finite particle sample reproduces a smooth halo equilibrium.
"""
    x=np.asarray(position,dtype=np.float64);m=np.asarray(mass,dtype=np.float64)
    h=positive(force_softening,'force softening');g=positive(gravitational_constant,'G')
    if x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(m)) or np.any(m<=0):
        raise ValueError('Invalid isolated particle census')
    if type(block_size)!=int or block_size<1:raise ValueError('Positive integer force block size required')
    out=np.empty_like(x)
    for start in range(0,len(x),block_size):
        delta=x[None,:,:]-x[start:start+block_size,None,:]
        r=np.linalg.norm(delta,axis=-1);u=r/h;fac=np.zeros_like(r)
        inner=u<.5;outer=(u>=.5)&(u<1);far=u>=1
        fac[inner]=(32/3+u[inner]**2*(32*u[inner]-192/5))/h**3
        fac[outer]=(64/3-48*u[outer]+192/5*u[outer]**2-32/3*u[outer]**3-1/(15*u[outer]**3))/h**3
        fac[far]=1/r[far]**3
        out[start:start+block_size]=g*np.sum(delta*fac[...,None]*m[None,:,None],axis=1)
    return out


def residual_summary(residual, reference, absolute_floor):
    r,y=map(lambda z:np.asarray(z,dtype=np.float64),(residual,reference))
    floor=positive(absolute_floor,'dimensional residual floor')
    if r.shape!=y.shape or not r.size or np.any(~np.isfinite(r)) or np.any(~np.isfinite(y)):
        raise ValueError('Residual/reference arrays must match and be finite/nonempty')
    # Vector/tensor Frobenius norms, not component-relative divisions by zero.
    axes=tuple(range(1,r.ndim))
    norm=np.sqrt(np.sum(r*r,axis=axes)) if axes else np.abs(r)
    ref=np.sqrt(np.sum(y*y,axis=axes)) if axes else np.abs(y)
    relative=norm/np.maximum(ref,floor)
    worst=int(np.argmax(relative))
    return {'absolute_max':float(norm.max()),'absolute_rms':float(np.sqrt(np.mean(norm**2))),
            'relative_max':float(relative.max()),'relative_p50':float(np.quantile(relative,.5)),
            'relative_p95':float(np.quantile(relative,.95)),'relative_p99':float(np.quantile(relative,.99)),
            'worst_index':worst,'worst_residual':r[worst].tolist(),'reference_floor':floor}


def isolated_invariants(position, velocity, parameters):
    x,v=map(np.asarray,(position,velocity))
    return .5*np.sum(v*v,axis=-1)+analytic_field(x,parameters)[0],np.cross(x,v)


def cosmological_energy_balance(scale_factors, kinetic, potential):
    """Layzer-Irvine integral: Delta(K+W)+integral(2K+W)dln(a).

K uses physical peculiar velocity; W uses physical peculiar potential with a
consistent periodic gauge. Angular momentum and isolated energy are not gates.
"""
    a,k,w=map(lambda z:np.asarray(z,dtype=float),(scale_factors,kinetic,potential))
    if a.ndim!=1 or len(a)<3 or a.shape!=k.shape or a.shape!=w.shape or np.any(~np.isfinite([a,k,w])) or np.any(a<=0) or np.any(np.diff(a)<=0):
        raise ValueError('At least three increasing finite cosmological epochs required')
    integral=np.r_[0.,np.cumsum(.5*((2*k+w)[1:]+(2*k+w)[:-1])*np.diff(np.log(a)))]
    return (k+w)-(k[0]+w[0])+integral


def validate_control_request(request):
    keys={'schema_version','resources','cases','comparisons'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema_version']!=1 or type(request['schema_version'])!=int:
        raise ValueError('Native numerical control request schema_version=1 required')
    r=request['resources']
    if not isinstance(r,dict) or set(r)!={'cpu_slots','build_jobs','memory_gib','wall_seconds','disk_margin_gib'}:
        raise ValueError('Declare native control resource limits')
    for key in ('cpu_slots','build_jobs'):
        if type(r[key])!=int or r[key]<1:raise ValueError('Positive integer '+key+' required')
    if r['build_jobs']>r['cpu_slots']:raise ValueError('Build jobs exceed CPU allocation')
    for key in ('memory_gib','wall_seconds','disk_margin_gib'):positive(r[key],key)
    cases=request['cases']
    if not isinstance(cases,list) or not 1<=len(cases)<=64:raise ValueError('Declare 1..64 control cases')
    ids=set()
    for case in cases:
        expected={'id','mode','potential','mpi_ranks','profile','times','timestep','softening','opening_angle',
                  'force_accuracy','integration_accuracy','stencil_step','particles','seed','restart','tolerances'}
        if not isinstance(case,dict) or set(case)!=expected:raise ValueError('Malformed control case')
        name=case['id']
        if not isinstance(name,str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name) or name in ids:
            raise ValueError('Control IDs must be unique safe names')
        ids.add(name)
        if case['mode'] not in CONTROL_MODES or case['profile']['mode']!=case['mode']:raise ValueError('Control mode/profile mismatch')
        control_config_text(case['profile']);potential_parameters(case['potential'])
        if type(case['mpi_ranks'])!=int or not 1<=case['mpi_ranks']<=r['cpu_slots']:raise ValueError('MPI ranks exceed declared CPU slots')
        if .85*r['memory_gib']*1024/case['mpi_ranks']<512:raise ValueError('Insufficient native per-rank memory')
        if type(case['particles'])!=int or not 28<=case['particles']<=4096:raise ValueError('Bounded control particle count required')
        if type(case['seed'])!=int or not 0<=case['seed']<2**64:raise ValueError('Exact uint64 fixture seed required')
        if type(case['restart']) is not bool:raise ValueError('Restart control must be Boolean')
        for key in ('timestep','softening','opening_angle','force_accuracy','integration_accuracy','stencil_step'):positive(case[key],key)
        ts=case['times']
        if not isinstance(ts,list) or not 3<=len(ts)<=64 or any(type(t) not in (int,float) or not math.isfinite(t) for t in ts) or ts[0]<0 or any(b<=a for a,b in zip(ts,ts[1:])):
            raise ValueError('Declare at least three increasing actual-clock targets')
        if case['mode']=='periodic_treepm' and (ts[0]<=0 or ts[-1]>1):raise ValueError('Cosmological controls require scale factors in (0,1]')
        if case['mode']=='analytic' and case['particles']!=28:raise ValueError('Analytic stencil controls require 28 exact-ID probes')
        tol=case['tolerances']
        needed={'force_relative','tide_relative','orbit_relative','energy_relative','angular_relative','cosmological_balance_relative','reference_floor'}
        if not isinstance(tol,dict) or set(tol)!=needed:raise ValueError('Declare all control tolerances before execution')
        for key in needed:positive(tol[key],key)
    comparisons=request['comparisons']
    if not isinstance(comparisons,list):raise ValueError('Declare control comparison list')
    for c in comparisons:
        if not isinstance(c,dict) or set(c)!={'left','right','kind','relative_tolerance','absolute_floor'} or c['left'] not in ids or c['right'] not in ids or c['left']==c['right'] or c['kind'] not in ('rank','restart','sidm_disabled','settings'):
            raise ValueError('Invalid native variant comparison')
        positive(c['relative_tolerance'],'variant tolerance');positive(c['absolute_floor'],'variant dimensional floor')
        left=next(x for x in cases if x['id']==c['left']);right=next(x for x in cases if x['id']==c['right'])
        allowed={'id'}|({'mpi_ranks'} if c['kind']=='rank' else {'restart'} if c['kind']=='restart' else {'profile'} if c['kind']=='sidm_disabled' else {'timestep','opening_angle','force_accuracy','integration_accuracy','profile'})
        for key in set(left)-allowed:
            if left[key]!=right[key]:raise ValueError('Unrelated input differs in '+c['kind']+' comparison: '+key)
        if c['kind']=='sidm_disabled':
            lp,rp=left['profile'].copy(),right['profile'].copy()
            lv,rv=lp.pop('diagnostic_variant'),rp.pop('diagnostic_variant')
            if lp!=rp or lv==rv:raise ValueError('SIDM-disabled equivalence requires distinct diagnostic-only builds')
    return copy.deepcopy(request)


def fixture(case):
    if case['mode']=='analytic':
        x,v=stencil_fixture(case['potential'],case['stencil_step']);mass=np.full(len(x),1e-12);box=32.
    elif case['mode']=='periodic_treepm':
        n=round(case['particles']**(1/3))
        if n**3!=case['particles']:raise ValueError('Periodic fixture requires a cubic particle count')
        box=8.;q=(np.stack(np.meshgrid(*[np.arange(n)]*3,indexing='ij'),-1).reshape(-1,3)+.5)*box/n
        x=q.copy();x[:,0]-=.001*box/(2*np.pi)*np.sin(2*np.pi*q[:,0]/box)
        v=np.zeros_like(x);mass=np.full(len(x),.3*3*.1**2/(8*np.pi)*box**3/len(x))
    else:
        rng=np.random.Generator(np.random.Philox(case['seed']));n=case['particles']
        u=(np.arange(n)+.5)/n;b=case['potential']['scale']
        if case['potential']['kind']=='plummer':r=b/np.sqrt(u**(-2/3)-1)
        elif case['potential']['kind']=='hernquist':r=b*np.sqrt(u)/(1-np.sqrt(u))
        elif case['potential']['kind']=='nfw':
            from scipy.optimize import brentq
            cutoff=8.;norm=np.log1p(cutoff)-cutoff/(1+cutoff)
            r=b*np.array([brentq(lambda z:np.log1p(z)-z/(1+z)-f*norm,1e-10,cutoff) for f in u])
        else:r=b*np.cbrt(u)*2
        direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
        x=direction*r[:,None];x-=x.mean(axis=0);v=.05*np.cross([0.,0.,1.],x)/np.maximum(np.linalg.norm(x,axis=1)[:,None],1e-8);v-=v.mean(axis=0);mass=np.full(n,case['potential']['gm']/n);box=max(32.,float(np.max(np.abs(x)))*4)
    # Use IDs above 2^53 but below the native reserved membership bit.
    ids=np.arange(len(x),dtype=np.uint64)+np.uint64(2**60+123)
    return x,v,mass,ids,box


def _configuration(request,case,box):
    import yaml
    cfg=yaml.safe_load((ROOT/'core/config.yaml').read_text());r=request['resources']
    cfg['hardware'].update(cpu_threads=r['cpu_slots'],build_jobs=r['build_jobs'],mpi_ranks=case['mpi_ranks'],omp_threads=1,memory_limit_gb=r['memory_gib'])
    cfg['numerical_control']=copy.deepcopy(case)
    cfg['halos'].update(fof=False,subfind=False)
    cfg['cosmology'].update(model='lcdm',H0=70.,omega_b=.05*.7**2,omega_cdm=.25*.7**2,omega_radiation=0.)
    cfg['simulation'].update(physics='dm',asmth=1.25,particles_per_dimension=8,pm_grid=case['profile']['pm_grid'],multipole_order=case['profile']['multipole_order'],box_mpc_h=box/1000,
        start_redshift=1.,end_redshift=0.,snapshot_redshift_list=[1.,0.],softening_kpc_h=case['softening'],softening_max_physical_kpc_h=case['softening'],
        max_size_timestep_ln_a=case['timestep'],min_size_timestep_ln_a=0.,err_tol_theta=case['opening_angle'],err_tol_force_acc=case['force_accuracy'],err_tol_int_accuracy=case['integration_accuracy'],
        num_files_per_snapshot=1,restart_every_seconds=600,time_limit_cpu_seconds=r['wall_seconds'])
    return cfg


def _write_inputs(root,request,case):
    import yaml
    from core.evolution import gadget_param_text
    root.mkdir();(root/'input').mkdir();(root/'data').mkdir()
    x,v,m,ids,box=fixture(case);cfg=_configuration(request,case,box)
    atomic_text(root/'config.yaml',yaml.safe_dump(cfg))
    counts=np.array([0,len(x),0,0,0,0],dtype=np.uint64)
    with h5py.File(root/'input/control.hdf5','w') as f:
        f.create_group('Header').attrs.update(Time=case['times'][0],Redshift=(1/case['times'][0]-1 if case['mode']=='periodic_treepm' else 0.),BoxSize=box,MassTable=np.zeros(6),NumFilesPerSnapshot=1,NumPart_ThisFile=counts,NumPart_Total=counts)
        p=f.create_group('PartType1');p['Coordinates']=x;p['Velocities']=v;p['Masses']=m;p['ParticleIDs']=ids
    # Retain native unit and registered parameter conventions; override only
    # explicit analytic-control clock/solver values, never a production request.
    text=gadget_param_text(cfg,root,root/'input/control.hdf5')
    values={line.split()[0]:' '.join(line.split()[1:]) for line in text.splitlines() if line.strip() and not line.lstrip().startswith('%')}
    values.update(TimeBegin=case['times'][0],TimeMax=case['times'][-1],TimeOfFirstSnapshot=case['times'][0],TimeBetStatistics=case['timestep'],NativeControlPauseStep=3 if case['restart'] else 0,ComovingIntegrationOn=int(case['mode']=='periodic_treepm'),GravityConstantInternal=1.,OmegaLambda=.7 if case['mode']=='periodic_treepm' else 0.,Omega0=.3 if case['mode']=='periodic_treepm' else 0.,TypeOfOpeningCriterion=0)
    if case['mode'] in ('analytic','isolated_tree'):
        values.pop('ActivePartFracForPMinsteadOfEwald',None)
    if case['mode']=='analytic':
        p=case['potential'];values.update(ControlPotential=POTENTIALS[p['kind']],ControlGM=p['gm'],ControlScale=p['scale'],ControlAxisY=p['axis_y'],ControlAxisZ=p['axis_z'])
    atomic_text(root/'input/param.txt','\n'.join(str(k)+' '+str(v) for k,v in values.items())+'\n')
    atomic_text(root/'input/output_times.txt','\n'.join(format(t,'.17g')+' 1' for t in case['times'])+'\n')
    cache=root.parent/'ewald-cache'
    for table in cache.glob('ewald_table_*.dat'):
        if table.with_name(table.name+'.sha256').read_text().strip()!=digest(table):raise ValueError('Native Ewald cache checksum changed')
        shutil.copy2(table,root/'input'/table.name)
    return cfg


def _invoke(root,binary,request,case,flag,tag):
    env=os.environ.copy();env['OMP_NUM_THREADS']='1'
    with (root/(tag+'.log')).open('w') as log:
        run_native_command(['mpirun','-np',str(case['mpi_ranks']),str(binary),str(root/'input/param.txt'),str(flag)],cwd=root/'input',receipt_path=root/(tag+'-invocation.json'),inputs=[root/'input/control.hdf5',root/'input/Config.sh',root/'input/build_signature.json',root/'input/build_receipt.json',root/'input/output_times.txt'],stdout=log,stderr=subprocess.STDOUT,timeout=request['resources']['wall_seconds'],env=env)


def _load_snapshots(root,case,*,memory_bytes):
    files=sorted((root/'data').glob('snapshot_*.hdf5'))
    if len(files)!=len(case['times']):raise ValueError('Native control did not produce every requested output')
    rows=[];times=[];sources={}
    for file,requested in zip(files,case['times']):
        verify_native_hdf5_set([file],require_publication=True)
        with h5py.File(file,'r') as f:
            a=float(f['Header'].attrs['Time'])
            if abs(a-requested)>max(1e-9,1e-7*abs(requested)):raise ValueError('Native output clock differs beyond declared quantization bound')
            if int(f['Parameters'].attrs['ComovingIntegrationOn'])!=int(case['mode']=='periodic_treepm'):raise ValueError('Native frame differs from control')
            p=f['PartType1'];ids=p['ParticleIDs'][:]
            if ids.dtype.kind!='u' or ids.dtype.itemsize!=8 or len(np.unique(ids))!=len(ids):raise ValueError('Native IDs must be unique uint64')
            from core.data import read_species_phase_space
            epoch = epoch_contract(a) if case['mode']=='periodic_treepm' else isolated_epoch_contract(a)
            physical = read_species_phase_space(p,epoch=epoch,memory_bytes=memory_bytes)
            order=np.argsort(ids);x=p['Coordinates'][:][order];v=p['Velocities'][:][order];ac=p['Acceleration'][:][order]
            # GADGET saved cosmological v = peculiar/sqrt(a); acceleration
            # already includes a^-2 (snap_io::io_func_accel).
            if case['mode']=='periodic_treepm':v=v*np.sqrt(a)
            masses=p['Masses'][:][order] if 'Masses' in p else np.full(len(ids),f['Header'].attrs['MassTable'][1])
            phi=p['Potential'][:][order]
            row={'ids':ids[order],'position':x,'velocity':v,'acceleration':ac,'mass':masses,'potential':phi}
            row.update(canonical_position_proper_kpc=physical['position_proper_kpc'][order],
                canonical_velocity_peculiar_kms=physical['velocity_peculiar_kms'][order],
                canonical_mass_msun=physical['mass_msun'][order])
            if any(np.any(~np.isfinite(value)) for key,value in row.items() if key!='ids'):raise ValueError('Nonfinite native control output')
            rows.append(row);times.append(a)
        for item in file.parent.glob(file.name+'*'):
            if item.is_file():sources[str(item.resolve())]=digest(item)
    expected=fixture(case)[3]
    if any(not np.array_equal(row['ids'],expected) for row in rows):raise ValueError('Native identity inventory changed')
    return np.array(times),rows,sources


def _force_test(root,case):
    file=root/'data/forcetest.txt'
    # Never parse identity columns as floats. Only the measured vector columns
    # enter the numerical distribution; exact uint64 IDs stay in separate arrays.
    ids=[];vectors=[]
    for line in file.read_text().splitlines():
        cols=line.split()
        if len(cols)!=16:raise ValueError('Unsupported native forcetest column schema')
        signed=int(cols[2]);ids.append(signed%(2**64));vectors.append([float(v) for v in cols[3:]])
    data=np.array(vectors);ids=np.asarray(ids,dtype=np.uint64)
    if not len(data) or np.any(~np.isfinite(data)):raise ValueError('Native direct force diagnostics are missing/nonfinite')
    # format: type,bin,id,time,x,y,z,dist, direct[3], native[3], potentials[2]
    direct=data[:,5:8];native=data[:,8:11]
    return ids,data[:,0],native,direct,file


def _analyze_case(root,request,case):
    times,rows,sources=_load_snapshots(root,case,memory_bytes=int(request['resources']['memory_gib']*1024**3))
    arrays={'actual_times':times,'particle_ids':rows[0]['ids'],**{k:np.stack([r[k] for r in rows]) for k in ('position','velocity','acceleration','mass','potential')}}
    for key in ('canonical_position_proper_kpc','canonical_velocity_peculiar_kms','canonical_mass_msun'):
        arrays[key] = np.asarray([row[key] for row in rows])
    t=case['tolerances'];floor=t['reference_floor'];findings=[];metrics={}
    def gate(name,residual,reference,key):
        metrics[name]=residual_summary(residual,reference,floor)
        worst=metrics[name]['worst_index'];n=len(rows[0]['ids'])
        if name=='native_direct_force':
            metrics[name].update(worst_particle_id=int(arrays['force_test_ids'][worst]),worst_actual_time=float(arrays['force_test_times'][worst]))
        elif name=='tide':
            metrics[name].update(worst_particle_id=int(rows[0]['ids'][7*worst]),worst_actual_time=float(times[0]))
        elif len(residual)==len(times)*n:
            metrics[name].update(worst_particle_id=int(rows[0]['ids'][worst%n]),worst_actual_time=float(times[worst//n]))
        elif len(residual)==n:
            metrics[name].update(worst_particle_id=int(rows[0]['ids'][worst]),worst_actual_time=float(times[0]))
        elif len(residual)==len(times):
            metrics[name]['worst_actual_time']=float(times[worst])
        findings.append({'metric':name,'threshold':t[key],'status':'PASS' if metrics[name]['relative_max']<=t[key] else 'FAIL'})
        arrays[name+'_residual']=residual;arrays[name+'_reference']=reference
    if case['mode']=='analytic':
        x,v=arrays['position'],arrays['velocity'];phi,acc,tides=analytic_field(x,case['potential'])
        gate('force',(arrays['acceleration']-acc).reshape(-1,3),acc.reshape(-1,3),'force_relative')
        measured_tide=stencil_tides(arrays['acceleration'][0],case['stencil_step'])
        gate('tide',measured_tide-tides[0,::7],tides[0,::7],'tide_relative')
        refx,refv=analytic_orbits(x[0],v[0],times,case['potential'])
        gate('orbit_position',(x-refx).reshape(-1,3),refx.reshape(-1,3),'orbit_relative')
        gate('orbit_velocity',(v-refv).reshape(-1,3),refv.reshape(-1,3),'orbit_relative')
        energy,angular=isolated_invariants(x,v,case['potential']);arrays['isolated_energy']=energy;arrays['angular_momentum']=angular
        gate('energy',(energy-energy[0]).ravel(),np.broadcast_to(np.abs(energy[0]),energy.shape).ravel(),'energy_relative')
        if case['potential']['kind']!='harmonic':
            gate('angular',(angular-angular[0]).reshape(-1,3),np.broadcast_to(angular[0],angular.shape).reshape(-1,3),'angular_relative')
        else:
            # Noncentral field exerts torque; compare the change to independent
            # orbits instead of asserting angular-momentum conservation.
            reference_angular=np.cross(refx,refv)
            gate('angular_reference',(angular-reference_angular).reshape(-1,3),reference_angular.reshape(-1,3),'angular_relative')
    else:
        ids,epoch,native,direct,path=_force_test(root,case)
        arrays.update(force_test_ids=ids,force_test_times=epoch)
        gate('native_direct_force',native-direct,direct,'force_relative');sources[str(path.resolve())]=digest(path)
        if case['mode']=='isolated_tree':
            independent=spline_direct_field(rows[0]['position'],rows[0]['mass'],2.8*case['softening'])
            gate('independent_discrete_force',rows[0]['acceleration']-independent,independent,'force_relative')
        if case['mode']=='periodic_treepm':
            # The initial plane-wave lattice has no transverse field. Its
            # continuum first-order force is a separate analytic control; the
            # exact finite-particle periodic reference is native Ewald summation.
            q=(np.stack(np.meshgrid(*[np.arange(round(case['particles']**(1/3)))]*3,indexing='ij'),-1).reshape(-1,3)+.5)*8/round(case['particles']**(1/3))
            plane=np.zeros_like(q);plane[:,0]=.0045*(rows[0]['position'][:,0]-q[:,0])/times[0]**2
            # Finite particle/PM sampling is distinct from native solver error.
            # Retain the continuum residual as a diagnostic, not a force gate.
            residual=rows[0]['acceleration']-plane
            metrics['planar_linear_force']=residual_summary(residual,plane,floor)
            arrays.update(planar_linear_force_residual=residual,planar_linear_force_reference=plane)
            findings.append({'metric':'planar_linear_force','status':'DIAGNOSTIC_ONLY','definition':'first-order continuum plane versus finite lattice/PM sampling; not an integration or direct-force accuracy gate'})
            transverse=np.column_stack((native[:,1:],direct[:,1:]))
            gate('periodic_plane_symmetry',transverse,np.zeros_like(transverse),'force_relative')
            k=.5*np.sum(arrays['mass']*np.sum(arrays['velocity']**2,axis=-1),axis=1)
            w=.5*np.sum(arrays['mass']*arrays['potential'],axis=1)/times
            balance=cosmological_energy_balance(times,k,w);arrays.update(peculiar_kinetic_energy=k,peculiar_potential_energy=w)
            gate('layzer_irvine_balance',balance,np.full_like(balance,max(abs(k[0])+abs(w[0]),floor)),'cosmological_balance_relative')
        else:
            k=.5*np.sum(arrays['mass']*np.sum(arrays['velocity']**2,axis=-1),axis=1)
            w=.5*np.sum(arrays['mass']*arrays['potential'],axis=1)
            e=k+w;l=np.sum(arrays['mass'][...,None]*np.cross(arrays['position'],arrays['velocity']),axis=1)
            gate('isolated_system_energy',e-e[0],np.full_like(e,abs(e[0])),'energy_relative')
            gate('isolated_system_angular',l-l[0],np.broadcast_to(l[0],l.shape),'angular_relative')
    checkpoint=verify_native_checkpoints(root/'data/restartfiles',case['mpi_ranks'],require_receipts=True)
    # Retain actual executable/build/parameter/state/log hashes, not just
    # declarations supplied by a caller.
    for path in root.rglob('*'):
        if path.is_file() and path.name not in ('analysis.json','analysis.json.sha256') and not path.name.startswith('.'):
            sources[str(path.resolve())]=digest(path)
    status='FAIL' if any(f['status']=='FAIL' for f in findings) else 'PASS'
    report={'schema':'native-numerical-control-case-v1','status':status,'science_acceptance':'NOT_ACCEPTED','case':case,'settings':case,'metrics':metrics,'findings':findings,
            'restart_integrity':checkpoint,'interaction_model':'collisionless; SIDM operator disabled/absent',
            'tidal_definition':'da_i/dx_j = -potential Hessian; native central differences at initial probes',
            'clock':'scale_factor' if case['mode']=='periodic_treepm' else 'native_time_L_over_V'}
    specs={key:{'units':_units(key),'annotations':{'meaning':key.replace('_',' ')},**({'identity_namespace':'native_particle_id'} if 'ids' in key else {})} for key in arrays}
    context={'frame':'cosmological_comoving_position_physical_peculiar_velocity' if case['mode']=='periodic_treepm' else 'isolated_proper_native',
             'units':'existing GADGET UnitLength/UnitVelocity/UnitMass; G=1 is explicit control parameter',
             'clock':report['clock'],'requested_times':case['times']}
    epoch_rows=[epoch_contract(a,requested_scale_factor=b) if case['mode']=='periodic_treepm' else {**isolated_epoch_contract(a,snapshot=i),'requested_native_time_code':b} for i,(a,b) in enumerate(zip(times,case['times']))]
    context['epoch']={'schema':CANONICAL_SCHEMA,'representation':'actual per-output control epochs','axis':0,'rows':epoch_rows}
    context['epochs']=epoch_rows
    return export_numeric_audit(root/'analysis.json',report,arrays,specs,algorithm=__file__,context=context,source_hashes=sources)


def _units(key):
    if key=='canonical_position_proper_kpc':return 'proper kpc'
    if key=='canonical_velocity_peculiar_kms':return 'physical km/s; peculiar in comoving runs, inertial in isolated runs'
    if key=='canonical_mass_msun':return 'Msun'
    if 'ids' in key:return 'uint64 particle identity'
    if key in ('actual_times','force_test_times'):return 'native time L/V for isolated controls; scale factor for periodic controls'
    if 'tide' in key:return 'native V^2/L^2'
    if 'acceleration' in key or 'force' in key or 'symmetry' in key:return 'native V^2/L'
    if 'velocity' in key:return 'native V'
    if 'position' in key:return 'native L'
    if key.startswith('isolated_system_angular'):return 'native M L V'
    if 'angular' in key:return 'native specific L V'
    if key=='mass':return 'native M'
    if key=='potential':return 'native specific V^2'
    if key.startswith(('isolated_system_energy','layzer_irvine_balance','peculiar_kinetic_energy','peculiar_potential_energy')):return 'native M V^2'
    if 'energy' in key:return 'native specific V^2'
    raise ValueError('Unspecified control quantity units: '+key)


def _comparison(left,right,comparison):
    lm=verify_numeric_audit(left);rm=verify_numeric_audit(right)
    def payload(path,manifest):
        # Numeric audit uses one .npz file adjacent to its report.
        return np.load(Path(path).with_name(Path(path).name+'.data')/'arrays.npz',allow_pickle=False)
    with payload(left,lm) as l,payload(right,rm) as r:
        if not np.array_equal(l['particle_ids'],r['particle_ids']):raise ValueError('Variant exact-ID mismatch')
        if not np.allclose(l['actual_times'],r['actual_times'],rtol=0,atol=1e-9):raise ValueError('Variant actual epochs differ')
        summaries={key:residual_summary((r[key]-l[key]).reshape(-1,3),l[key].reshape(-1,3),comparison['absolute_floor']) for key in ('position','velocity','acceleration')}
    return {**comparison,'metrics':summaries,'status':'PASS' if all(m['relative_max']<=comparison['relative_tolerance'] for m in summaries.values()) else 'FAIL',
            'policy':'exact uint64 identity + common actual clock; tolerance comparison, whole trajectory bitwise identity not required'}


def run_control_campaign(request,destination,*,resume=False):
    request=validate_control_request(request);destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    tracked = [Path(__file__), *[ROOT/name for name in (
        'core/research/generator_controls.py', 'core/evolution.py', 'core/data.py',
        'core/foundation.py', 'core/native.py', 'core/resources.py', 'commands.py',
    )]]
    native=ROOT/'core/software/gadget4'
    from core.native import native_build_input_paths
    native_inputs = native_build_input_paths(native)
    tracked += native_inputs
    tracked += list((ROOT/'core/patches').glob('gadget4-*.patch'))
    code={str(p.resolve()):digest(p) for p in tracked}
    fingerprint=canonical_digest({'request':request,'code':code})
    with native_run_lock(destination,operation='native numerical-control campaign'):
        if destination.exists():
            if not resume:raise FileExistsError('Retain previous native controls; select a fresh destination')
            report=verify_control_campaign(destination/'campaign.json')
            if report['fingerprint']!=fingerprint:raise ValueError('Resume request/code changed')
            return report
        estimate=sum(c['particles']*len(c['times'])*256+32*1024**2 for c in request['cases'])
        if shutil.disk_usage(destination.parent).free<estimate+request['resources']['disk_margin_gib']*1024**3:raise OSError('Insufficient control disk budget')
        destination.mkdir();atomic_json(destination/'request.json',request)
        atomic_json(destination/'execution.json',{'status':'RUNNING','fingerprint':fingerprint,'science_acceptance':'NOT_ACCEPTED'})
        try:
            from core.evolution import build_gadget
            cases=[]
            for case in request['cases']:
                root=destination/case['id'];cfg=_write_inputs(root,request,case)
                binary=build_gadget(cfg,root,control_profile=case['profile'])
                _invoke(root,binary,request,case,0,'native-initial')
                if case['restart']:
                    checkpoint=verify_native_checkpoints(root/'data/restartfiles',case['mpi_ranks'],require_receipts=True)
                    if not case['times'][0]<checkpoint['clock_value']<case['times'][-1]:raise ValueError('Restart control did not pause an evolved interior state')
                    atomic_json(root/'interrupted-checkpoint.json',checkpoint)
                    shutil.copytree(root/'data/restartfiles',root/'interrupted-restart')
                    _invoke(root,binary,request,case,1,'native-resume')
                for table in (root/'input').glob('ewald_table_*.dat'):
                    cache=destination/'ewald-cache';cache.mkdir(exist_ok=True)
                    target=cache/table.name
                    if target.exists() and digest(target)!=digest(table):raise ValueError('Native Ewald table changed between matched cases')
                    if not target.exists():
                        shutil.copy2(table,target);atomic_text(target.with_name(target.name+'.sha256'),digest(target)+'\n')
                report=_analyze_case(root,request,case);cases.append({'id':case['id'],'status':report['status'],'report':str(root/'analysis.json'),'sha256':digest(root/'analysis.json'),'binary_sha256':digest(binary)})
            byid={r['id']:r for r in cases}
            comparisons=[_comparison(byid[c['left']]['report'],byid[c['right']]['report'],c) for c in request['comparisons']]
            sources=dict(code)
            # Bind transitive numeric contracts and every native output/build,
            # including actual binary bytes. Verification never trusts PASS alone.
            for p in destination.rglob('*'):
                if p.is_file() and p.name!='execution.json' and not p.name.startswith('.'):
                    sources[str(p.resolve())]=digest(p)
            report={'schema':'native-numerical-control-campaign-v1','status':'PASS' if all(r['status']=='PASS' for r in cases+comparisons) else 'FAIL',
                    'execution_status':'COMPLETE','science_acceptance':'NOT_ACCEPTED','fingerprint':fingerprint,'request':request,'cases':cases,'comparisons':comparisons,'source_hashes':sources,
                    'source_inventory_version':2,'native_build_input_hashes':{str(p.resolve()):code[str(p.resolve())] for p in native_inputs},
                    'limitations':['SIDM-disabled equivalence covers collisionless diagnostic builds; an implemented SIDM-enabled zero-cross-section comparison belongs to Block 47.',
                    'Analytic external-force tests validate native integration/output against independent references; selfgravity tree/PM tests are separate finite-particle controls.',
                    'Numerical tolerances cover this recorded control matrix, not arbitrary production resolution or cosmologies.']}
            for source,checksum in sources.items():
                if digest(source)!=checksum:raise ValueError('Native control input changed before publication: '+source)
            if native_build_input_paths(native)!=native_inputs:
                raise ValueError('Native build input inventory changed during campaign')
            atomic_json(destination/'campaign.json',report);atomic_text(destination/'campaign.json.sha256',digest(destination/'campaign.json')+'\n')
            atomic_json(destination/'execution.json',{'status':'COMPLETE','campaign_sha256':digest(destination/'campaign.json')})
            return report
        except BaseException as exc:
            atomic_json(destination/'execution.json',{'status':'FAILED','fingerprint':fingerprint,'failure':str(exc),'science_acceptance':'NOT_ACCEPTED'})
            raise


def verify_control_campaign(path):
    path=Path(path).resolve()
    if path.with_name(path.name+'.sha256').read_text().strip()!=digest(path):raise ValueError('Native control receipt seal changed')
    report=json.loads(path.read_text())
    if report.get('schema')!='native-numerical-control-campaign-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Incomplete native control campaign')
    validate_control_request(report['request'])
    from core.native import native_build_input_paths
    current = {str(p.resolve()) for p in native_build_input_paths(ROOT/'core/software/gadget4')}
    binding = report.get('native_build_input_hashes',{})
    if report.get('source_inventory_version')!=2 or set(binding)!=current:
        raise ValueError('Native build input inventory changed or receipt lacks complete source closure')
    for source,checksum in binding.items():
        if report['source_hashes'].get(source)!=checksum or digest(source)!=checksum:
            raise ValueError('Native build input changed: '+source)
    for source,checksum in report['source_hashes'].items():
        if not Path(source).is_file() or digest(source)!=checksum:raise ValueError('Native control source/artifact changed: '+source)
    for row in report['cases']:
        numeric=verify_numeric_audit(row['report'])
        if row['status']!=numeric['status'] or digest(row['report'])!=row['sha256']:raise ValueError('Case receipt mismatch')
    return report
