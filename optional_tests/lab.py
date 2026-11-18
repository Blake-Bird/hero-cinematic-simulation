"""Run bounded, portable teaching experiments and preserve their numerical proof.

Usage: python optional_tests/lab.py run --particles 32 --output NEW_DIRECTORY
       python optional_tests/lab.py serve

Synthetic HDF5 fixtures are labeled and never certified as native evolution.
Core numerical functions perform the measurements. Each step records its
inputs, equations, implementation, assertions, artifacts and elapsed time.
"""
from pathlib import Path
import argparse
import hashlib
import html
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True  # Normal lab launches keep source-tree caches out.
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from optional_tests.catalogue import EQUATIONS,sha,build_catalogue


def write_json(path,value):
    """Atomically publish finite JSON; reject accidental NaN/Infinity claims."""
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def implementation(function):
    """Bind a function to the physical source file, line and current source hash."""
    path=Path(inspect.getsourcefile(function)).resolve()
    return {'function':function.__module__+'.'+function.__name__,'path':path.relative_to(ROOT).as_posix(),
            'line':function.__code__.co_firstlineno,'sha256':sha(path)}


class LabRun:
    """One immutable output directory with an append-only sequence of test receipts."""
    def __init__(self,output,particles=32,seed=42,profile='quick'):
        if type(particles) is not int or not 32<=particles<=8192:raise ValueError('Particle count must be in [32,8192]')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Seed must fit uint32')
        if profile not in {'quick','visual','native','libraries','combined','engines','physics','complete','figures'}:raise ValueError('Unknown fixed profile')
        self.output=Path(output).expanduser().resolve()
        if self.output.is_relative_to(ROOT):raise ValueError('Keep generated lab output outside the source tree')
        self.output.mkdir(parents=True,exist_ok=False)
        self.started=time.monotonic();self.state={'schema':'research2-teaching-run-v1','status':'RUNNING',
          'particles':particles,'seed':seed,'profile':profile,'origin':'SYNTHETIC_CONTROL',
          'scope':'Synthetic numerical and native execution controls; not resolved astrophysical systems or general scientific validation',
          'steps':[],'artifacts':[],'software':[]}
        from optional_tests.software_checks import CHECKS
        from optional_tests.native_checks import CHECKS as NATIVE_CHECKS
        from optional_tests.native_physics import CHECKS as PHYSICS_CHECKS
        from optional_tests.advanced_controls import CHECKS as FIGURE_CHECKS
        self.state['total_steps']=7+int(profile in {'native','combined','engines','complete'})+(len(CHECKS) if profile in {'libraries','combined','engines','complete'} else 0)+(len(NATIVE_CHECKS) if profile in {'engines','complete'} else 0)+(len(PHYSICS_CHECKS) if profile in {'physics','complete'} else 0)+(len(FIGURE_CHECKS) if profile in {'figures','complete'} else 0)
        self.save()

    def save(self):
        """Publish progress for the UI without exposing a half-written JSON file."""
        self.state['elapsed_seconds']=round(time.monotonic()-self.started,3)
        write_json(self.output/'run.json',self.state)

    def step(self,key,title,software,equations,functions,work):
        """Execute one bounded measurement; retain a failing traceback and continue."""
        row={'id':key,'title':title,'status':'RUNNING','software':software,'equations':equations,
             'implementation':[implementation(f) for f in functions]}
        self.state['steps'].append(row);self.save();started=time.monotonic()
        try:
            row['result']=work();row['status']='PASS'
        except ImportError as error:
            row.update(status='UNAVAILABLE',reason=str(error))
        except Exception as error:
            row.update(status='FAIL',reason=str(error),traceback=traceback.format_exc())
        row['seconds']=round(time.monotonic()-started,3);self.save()
        return row.get('result')

    def artifact(self,path,kind,caption):
        """Record an actual artifact and its bytes; this does not infer a test pass."""
        path=Path(path)
        self.state['artifacts'].append({'path':path.relative_to(self.output).as_posix(),
          'sha256':sha(path),'bytes':path.stat().st_size,'kind':kind,'caption':caption})

    def finish(self):
        """Finish with failures and missing dependencies visible, then write an offline report."""
        states={r['status'] for r in self.state['steps']}
        self.state['status']='FAILED' if 'FAIL' in states else 'COMPLETE_WITH_UNAVAILABLE' if 'UNAVAILABLE' in states else 'COMPLETE'
        self.save();write_report(self.output,self.state)
        return self.state


def make_fixture(run):
    """Generate exactly N synthetic proper-coordinate particles and a declared HDF5 file."""
    import h5py
    import numpy as np
    from core.foundation import KPC_CM,MSUN_G
    n=run.state['particles'];rng=np.random.default_rng(run.state['seed']);box=100.
    # A truncated Plummer-shaped sample. This is a seeded geometry, not evolution.
    u=rng.uniform(.002,.97,n);radius=4./np.sqrt(u**(-2/3)-1)
    direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    xyz=box/2+direction*radius[:,None];xyz=np.mod(xyz,box)
    velocity=rng.normal(0.,30.,(n,3));mass=np.full(n,1e10/n)
    species=np.ones(n,dtype=np.int64);species[:max(1,n//5)]=0
    species[max(1,n//5):max(2,n//3)]=4;species[-1]=5
    ids=np.arange(n,dtype=np.uint64)+np.uint64(2**60+1)
    counts=np.bincount(species,minlength=6).astype(np.uint64)
    path=run.output/'synthetic_snapshot.hdf5'
    with h5py.File(path,'x') as f:
        f.attrs.update(source_origin='synthetic_control',science_acceptance='NOT_ACCEPTED',generator='optional_tests.lab.make_fixture')
        f.create_group('Header').attrs.update(Time=0.,Redshift=0.,BoxSize=box,HubbleParam=1.,
          NumPart_ThisFile=counts,NumPart_Total=counts,NumPart_Total_HighWord=np.zeros(6,dtype=np.uint32),
          NumFilesPerSnapshot=1,MassTable=np.zeros(6))
        f.create_group('Parameters').attrs.update(ComovingIntegrationOn=0,HubbleParam=1.,
          UnitLength_in_cm=KPC_CM,UnitMass_in_g=1e10*MSUN_G,UnitVelocity_in_cm_per_s=1e5)
        for kind in [0,1,4,5]:
            selected=species==kind;g=f.create_group('PartType'+str(kind))
            for key,value,cgs in [('Coordinates',xyz[selected],KPC_CM),('Velocities',velocity[selected],1e5),
              ('Masses',mass[selected]/1e10,1e10*MSUN_G),('ParticleIDs',ids[selected],1.)]:
                d=g.create_dataset(key,data=value,fletcher32=True);d.attrs.update(to_cgs=cgs,a_scaling=0.,h_scaling=0.)
    run.fixture={'position':xyz,'velocity':velocity,'mass':mass,'species':species,'ids':ids,'box':box}
    run.artifact(path,'data','Synthetic GADGET-shaped isolated snapshot; no native publication claim.')
    write_json(run.output/'particles.json',{'box_proper_kpc':box,'origin':'synthetic_control',
       'positions':xyz.tolist(),'species':species.tolist(),'masses_msun':mass.tolist(),
       'particle_ids':[str(i) for i in ids]})
    run.artifact(run.output/'particles.json','data','Exact IDs remain decimal strings in browser data.')
    assert int(counts.sum())==n and len(set(ids.tolist()))==n
    return {'particles_by_type':counts.tolist(),'total_mass_msun':float(mass.sum()),
      'box_proper_kpc':box,'input_kind':'Synthetic file with explicit units; not produced by GADGET'}


def check_units(run):
    """Read every species through the production unit adapter and compare original physical values."""
    import h5py
    import numpy as np
    from core.data import read_species_phase_space,isolated_epoch_contract
    found=[]
    with h5py.File(run.output/'synthetic_snapshot.hdf5') as f:
        for kind in [0,1,4,5]:
            state=read_species_phase_space(f['PartType'+str(kind)],epoch=isolated_epoch_contract(0.),memory_bytes=32*1024**2)
            mask=run.fixture['species']==kind
            np.testing.assert_array_equal(state['particle_ids'],run.fixture['ids'][mask])
            np.testing.assert_allclose(state['position_proper_kpc'],run.fixture['position'][mask],rtol=1e-12)
            np.testing.assert_allclose(state['mass_msun'],run.fixture['mass'][mask],rtol=1e-12)
            np.testing.assert_allclose(state['velocity_peculiar_kms'],run.fixture['velocity'][mask],rtol=1e-12)
            found.extend(state['particle_ids'].tolist())
    return {'round_tripped_particles':len(found),'id_width_bits':64,'clock':'isolated elapsed time; not scale factor',
      'assertions':['Exact IDs preserved above 2^60','All species position/mass/velocity units round-trip']}


def plot_setup():
    """Use the noninteractive renderer; keep every label legible on the dark figure background."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor':'#111c2b','axes.facecolor':'#111c2b','savefig.facecolor':'#111c2b',
      'text.color':'#e9eef6','axes.labelcolor':'#cad5e4','xtick.color':'#91a5bb','ytick.color':'#91a5bb',
      'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','axes.edgecolor':'#41536b','grid.color':'#26384d','font.size':10,'axes.titleweight':'bold'})
    return plt


def measure_density(run):
    """Compute shell mass/volume and a mass-conserving surface-density map; save the actual bins."""
    import numpy as np
    from core.analysis import ngp_density
    plt=plot_setup();f=run.fixture;x=f['position'];m=f['mass'];box=f['box']
    delta=(x-box/2+box/2)%box-box/2;r=np.linalg.norm(delta,axis=1)
    edges=np.geomspace(.2,box*.86,15);edges=np.r_[0.,edges]
    counts,_=np.histogram(r,edges);shell_mass,_=np.histogram(r,edges,weights=m)
    volume=4*np.pi/3*np.diff(edges**3);density=shell_mass/volume
    center=(edges[:-1]+edges[1:])/2;positive=density>0
    mesh=ngp_density(x,box,16);assert int(mesh.sum())==len(x)
    image,xe,ye=np.histogram2d(x[:,0],x[:,1],bins=64,range=[[0,box],[0,box]],weights=m)
    area=(box/64)**2;surface=image/area
    np.testing.assert_allclose(shell_mass.sum(),m.sum(),rtol=1e-12)
    np.testing.assert_allclose(surface.sum()*area,m.sum(),rtol=1e-12)
    np.savez_compressed(run.output/'density_arrays.npz',radius_proper_kpc=center,radial_edges_proper_kpc=edges,
      density_msun_kpc3=density,shell_count=counts,shell_mass_msun=shell_mass,shell_volume_kpc3=volume,
      supported=positive,surface_density_msun_kpc2=surface,x_edges_proper_kpc=xe,y_edges_proper_kpc=ye)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    im=axes[0].imshow(np.log10(1+surface.T),origin='lower',extent=[0,box,0,box],cmap='magma')
    fig.colorbar(im,ax=axes[0],label='log₁₀(1 + Σ/[Msun kpc⁻²])')
    axes[0].set(xlabel='x [proper kpc]',ylabel='y [proper kpc]',title='Mass on the sky plane')
    axes[1].loglog(center[positive],density[positive],'o-',color='#64d9c0',ms=4)
    axes[1].set(xlabel='Shell midpoint [proper kpc]',ylabel='Shell density [Msun / proper kpc³]',title='Density is measured mass / volume')
    axes[1].grid(alpha=.5);fig.suptitle(f'SYNTHETIC CONTROL · {len(x):,} particles · seed {run.state["seed"]}',fontsize=11)
    fig.savefig(run.output/'density.png',dpi=150);plt.close(fig)
    run.artifact(run.output/'density.png','image','Measured projection and shell density. Zero-density log bins remain masked.')
    run.artifact(run.output/'density_arrays.npz','arrays','Shell masses, volumes, counts, density, masks and projection.')
    return {'shells':len(center),'occupied_shells':int(positive.sum()),'ngp_count_sum':int(mesh.sum()),
      'mass_closure_relative':float(abs(shell_mass.sum()-m.sum())/m.sum()),'units':'Msun / proper kpc³',
      'algorithm_note':'Shell histogram is transparent lab code; NGP mesh uses core.analysis.ngp_density.'}


def measure_power(run):
    """Call the production diagnostic FFT estimator and expose its normalization and limits."""
    import numpy as np
    from core.analysis import fft_power_spectrum
    x=run.fixture['position'];box=run.fixture['box'];k,p=fft_power_spectrum(x,box,grid=16)
    assert len(k)>0 and np.all(np.isfinite(p)) and np.all(p>=0) and np.all(np.diff(k)>0)
    plt=plot_setup();fig,ax=plt.subplots(figsize=(8,4.2),constrained_layout=True)
    mask=p>0;ax.loglog(k[mask],p[mask],'o-',color='#e8ba70',ms=3)
    ax.set(xlabel='k [1 / proper kpc]',ylabel='P(k) [proper kpc³]',title='NGP diagnostic power · sampling effects included')
    ax.grid(alpha=.5);fig.savefig(run.output/'power.png',dpi=150);plt.close(fig)
    np.savez_compressed(run.output/'power_arrays.npz',k_inverse_proper_kpc=k,power_proper_kpc3=p)
    run.artifact(run.output/'power.png','image','Measured FFT power. This is not the native folded GADGET estimator.')
    run.artifact(run.output/'power_arrays.npz','arrays','The exact measured k and power arrays, with physical unit names.')
    return {'bins':len(k),'mesh':16,'units':{'k':'1/proper kpc','power':'proper kpc³'},
      'shot_noise_subtracted':False,'assignment':'NGP','convergence':'NOT_ESTABLISHED'}


def measure_zoom(run):
    """Measure three nested projected views of the same exact synthetic particles.

    This is a spatial viewing zoom, not a refined MUSIC initial condition or
    an increase in particle resolution. Cropped mass is checked independently.
    """
    import numpy as np
    from matplotlib.patches import Rectangle
    plt=plot_setup();f=run.fixture;delta=(f['position']-f['box']/2+f['box']/2)%f['box']-f['box']/2
    extents=[50.,10.,2.5];images=[];counts=[];selected_mass=[];saved={}
    for index,radius in enumerate(extents):
        mask=(np.abs(delta[:,0])<=radius)&(np.abs(delta[:,1])<=radius)
        mass,xe,ye=np.histogram2d(delta[:,0],delta[:,1],bins=48,range=[[-radius,radius]]*2,weights=f['mass'])
        area=(2*radius/48)**2;surface=mass/area
        np.testing.assert_allclose(surface.sum()*area,f['mass'][mask].sum(),rtol=1e-12,atol=1e-6)
        images.append(surface);counts.append(int(mask.sum()));selected_mass.append(float(f['mass'][mask].sum()))
        saved.update({f'view_{index}_surface_density_msun_kpc2':surface,f'view_{index}_selected':mask,
                      f'view_{index}_x_edges_proper_kpc':xe,f'view_{index}_y_edges_proper_kpc':ye})
    upper=max(float(np.log10(1+image).max()) for image in images)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for index,(radius,image,ax) in enumerate(zip(extents,images,axes)):
        im=ax.imshow(np.log10(1+image.T),origin='lower',extent=[-radius,radius]*2,cmap='magma',vmin=0,vmax=upper)
        if index<2:
            next_radius=extents[index+1]
            ax.add_patch(Rectangle((-next_radius,-next_radius),2*next_radius,2*next_radius,fill=False,edgecolor='#7de0c5',lw=1.2))
        ax.set(xlabel='x − center [proper kpc]',ylabel='y − center [proper kpc]',
               title=f'{2*radius:g} proper kpc across · {counts[index]} particles')
    fig.colorbar(im,ax=axes,label='log10(1 + surface density / [Msun kpc^-2])',shrink=.8)
    fig.suptitle('One synthetic snapshot, three measured views · geometric zoom, no added particles',fontsize=11)
    fig.savefig(run.output/'zoom.png',dpi=160);plt.close(fig)
    np.savez_compressed(run.output/'zoom_arrays.npz',**saved)
    run.artifact(run.output/'zoom.png','image','Nested views measured from the same synthetic HDF5 particles. No native IC refinement is implied.')
    run.artifact(run.output/'zoom_arrays.npz','arrays','Exact projected selections, surface densities and pixel-edge coordinates for each zoom.')
    return {'full_widths_proper_kpc':[2*r for r in extents],'selected_particle_counts':counts,
            'selected_mass_msun':selected_mass,'new_particles_created':0,
            'scope':'Projected viewing zoom only; MUSIC refinement is a separate native workflow.'}


def measure_orbits(run):
    """Integrate exactly N tracers in a physical isochrone field and check finite-set energy drift."""
    import numpy as np
    from core.action_angle import AnalyticSphericalPotential
    from core.stream_forward_model import integrate_stream
    n=run.state['particles'];rng=np.random.default_rng(run.state['seed']+1)
    # All tracers follow a curved arc. The field is fixed, no star formation implied.
    theta=np.linspace(-.22,.22,n);r=8.+rng.normal(0,.025,n)
    x=np.c_[r*np.cos(theta),r*np.sin(theta),rng.normal(0,.02,n)]
    v=np.c_[-125*np.sin(theta),125*np.cos(theta),np.zeros(n)]
    sample={'position_kpc':x,'velocity_kms':v,'mass_msun':np.full(n,1e4/n),
      'particle_ids':np.arange(n,dtype=np.uint64),'namespace':'optional-lab:isochrone-tracers'}
    field=AnalyticSphericalPotential('isochrone',mass_msun=1e11,scale_kpc=5.)
    controls={'maximum_particles':8192,'integration_chunk_rows':min(n,128),'memory_gib':1.,
      'maximum_step_myr':1.,'relative_tolerance':1e-7,'position_tolerance_kpc':1e-8,
      'velocity_tolerance_kms':1e-6,'maximum_force_calls':100000}
    times=np.linspace(0.,20.,9);report,arrays=integrate_stream(sample,field,times,controls)
    assert arrays['position_kpc'].shape==(9,n,3) and np.isfinite(arrays['position_kpc']).all()
    energy=arrays['specific_energy_kms2'];drift=float(np.max(np.abs(energy-energy[0])/np.maximum(np.abs(energy[0]),1.)))
    assert drift<1e-5,drift
    np.savez_compressed(run.output/'orbit_arrays.npz',**arrays)
    plt=plot_setup();fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for i,t in enumerate(times):
        xx=arrays['position_kpc'][i];ax.scatter(xx[:,0],xx[:,1],s=2 if n>128 else 10,alpha=.7,c=np.full(n,t),vmin=0,vmax=20,cmap='viridis')
    ax.set(xlabel='x [proper kpc]',ylabel='y [proper kpc]',title='A physical tracer arc through a fixed isochrone field');ax.set_aspect('equal');ax.grid(alpha=.3)
    fig.savefig(run.output/'orbits.png',dpi=150);plt.close(fig)
    run.artifact(run.output/'orbits.png','image','Nine actual integration outputs, colored by elapsed Myr. No movie or interpolated frames.')
    run.artifact(run.output/'orbit_arrays.npz','arrays','Actual integration positions, velocities, energies, torques, masses and exact IDs.')
    return {'particles':n,'saved_times_myr':times.tolist(),'maximum_relative_energy_drift':drift,
      'force_calls':report['force_calls'],'self_gravity':False,'potential':'isochrone: 1e11 Msun, scale 5 proper kpc'}


def measure_covariance(run):
    """Verify the production mass-weighted population covariance against its finite-set definition."""
    import numpy as np
    from core.stream_common import weighted_covariance
    v=run.fixture['velocity'];m=run.fixture['mass'];mean,cov,neff=weighted_covariance(v,m)
    p=m/m.sum();manual=np.einsum('i,ij,ik->jk',p,v-mean,v-mean)
    np.testing.assert_allclose(cov,manual,rtol=1e-12,atol=1e-12)
    assert np.linalg.eigvalsh(cov).min()>-1e-10
    np.savez_compressed(run.output/'velocity_arrays.npz',mean_velocity_kms=mean,population_covariance_kms2=cov)
    run.artifact(run.output/'velocity_arrays.npz','arrays','Mass-weighted finite-population mean and covariance.')
    return {'mean_velocity_kms':mean.tolist(),'covariance_kms2':cov.tolist(),'effective_particles':float(neff),
      'meaning':'Finite-population covariance, not covariance of the mean'}


def native_control(run):
    """Execute one real 32-particle self-gravitating Plummer control; cold builds may take minutes."""
    from core.research.controls import run_control_campaign
    from core.data import verify_numeric_audit
    import numpy as np
    request=json.loads((ROOT/'configs/native_numerical_controls.json').read_text())
    case=next(c for c in request['cases'] if c['id']=='plummer-tree');case.update(particles=32,times=[0.,.02,.04])
    request['cases']=[case];request['comparisons']=[]
    request['resources'].update(cpu_slots=2,build_jobs=2,memory_gib=2.,wall_seconds=120.)
    write_json(run.output/'native-request.json',request)
    report=run_control_campaign(request,run.output/'native-control')
    assert report['status']=='PASS',report['status']
    run.artifact(run.output/'native-control/campaign.json','receipt','Actual native GADGET campaign and numerical checks.')
    analysis_path=Path(report['cases'][0]['report'])
    measured=verify_numeric_audit(analysis_path)
    run.artifact(analysis_path,'receipt','Measured native force and conservation diagnostics, thresholds and original unit contracts.')
    arrays_path=analysis_path.parent/measured['numeric_data']['directory']/measured['numeric_data']['array_contract']['path']
    run.artifact(arrays_path,'data','Original native diagnostic arrays; inspect the analysis receipt for units and selections.')
    with np.load(arrays_path,allow_pickle=False) as arrays:
        actual_times=arrays['actual_times'].tolist()
    checks=[{'metric':row['metric'],'maximum_relative_error':measured['metrics'][row['metric']]['relative_max'],
             'threshold':row['threshold'],'status':row['status']} for row in measured['findings']]
    run.state['software'].extend([
        {'name':'GADGET-4','module':'GADGET-4','available':True,'execution':'CALCULATION_PASS',
         'receipt':'native-control/campaign.json'},
        {'name':'MPI launcher','module':'MPI','available':True,'execution':'USED_IN_NATIVE_CONTROL',
         'receipt':'native-control/campaign.json'},
    ])
    return {'status':report['status'],'particles':32,'case_count':len(report['cases']),
       'actual_output_times':actual_times,'clock':measured['clock'],'measured_checks':checks,
       'reading':'Relative residuals are dimensionless and use the declared reference floors in the original analysis. These are isolated control units, not cosmological epochs or observational constraints.',
       'scope':'One isolated self-gravity control; not every native engine'}


def inspect_software(run):
    """Record dependency availability separately from computational execution."""
    for name,module in [('NumPy','numpy'),('SciPy','scipy'),('Matplotlib','matplotlib'),('HDF5 / h5py','h5py'),
      ('Gala','gala'),('Agama','agama'),('FSPS','fsps'),('emcee','emcee'),('dynesty','dynesty'),
      ('SBI','sbi'),('PyVista','pyvista'),('lenstronomy','lenstronomy'),('pycorr','pycorr'),('pyrecon','pyrecon')]:
        try:available=importlib.util.find_spec(module) is not None
        except (ImportError,ValueError):available=False
        run.state['software'].append({'name':name,'module':module,'available':available,
          'execution':'USED_IN_STEPS' if module in {'numpy','scipy','matplotlib','h5py'} else 'NOT_EXECUTED_BY_THIS_PROFILE'})


def library_control(run,backend):
    """Run one fixed library calculation in its own child; retain exits, logs and arrays."""
    from optional_tests.software_checks import CHECKS
    if backend not in CHECKS:raise ValueError('Unknown fixed library calculation')
    output=run.output/'libraries'/backend;output.parent.mkdir(exist_ok=True)
    log=run.output/'libraries'/(backend+'.log')
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MPLBACKEND':'Agg'}
    try:
        with log.open('w') as stream:
            process=subprocess.run([sys.executable,str(HERE/'software_checks.py'),backend,'--output',str(output),
                '--seed',str(run.state['seed'])],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    finally:
        if log.is_file():run.artifact(log,'log','Actual '+backend+' process output.')
    receipt=output/'receipt.json'
    if not receipt.is_file():raise RuntimeError(backend+' stopped without a receipt; exit '+str(process.returncode)+'; inspect '+log.name)
    record=json.loads(receipt.read_text());run.artifact(receipt,'receipt','Fresh '+backend+' calculation receipt.')
    module={'streamax':'StreaMAX'}.get(backend,backend)
    entry=next((r for r in run.state['software'] if r['module']==module),None)
    if entry is None:
        entry={'name':module,'module':module,'available':record['status']!='UNAVAILABLE'};run.state['software'].append(entry)
    entry['execution']='CALCULATION_'+record['status'];entry['receipt']=receipt.relative_to(run.output).as_posix()
    if backend=='pycorr':
        run.state['software'].append({'name':'Corrfunc','module':'Corrfunc','available':record['status']!='UNAVAILABLE',
            'execution':'CALCULATION_'+record['status'],'receipt':entry['receipt']})
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path==receipt:continue
        kind='image' if path.suffix=='.png' else 'log' if 'training-logs' in path.parts else 'data'
        run.artifact(path,kind,'Actual '+backend+' output; original units and scope in its receipt.')
    if record['status']=='UNAVAILABLE':raise ImportError(record['reason'])
    if record['status']!='PASS' or process.returncode:
        raise RuntimeError(record.get('reason','Backend process failed')+'; inspect '+receipt.name)
    return record['result']


def native_package_control(run,backend):
    """Run a shipped native-package calculation with retained outputs and bounds."""
    from optional_tests.native_checks import CHECKS
    if backend not in CHECKS:raise ValueError('Unknown fixed native check')
    output=run.output/'engines'/backend;output.parent.mkdir(exist_ok=True)
    log=output.parent/(backend+'.log')
    try:
        with log.open('w') as stream:
            process=subprocess.run([sys.executable,str(HERE/'native_checks.py'),backend,'--output',str(output),
                '--seed',str(run.state['seed'])],cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},
                stdout=stream,stderr=subprocess.STDOUT,timeout=210)
    finally:
        if log.is_file():run.artifact(log,'log','Actual '+backend+' native-worker output.')
    receipt=output/'receipt.json'
    if not receipt.is_file():raise RuntimeError('Native worker stopped without a receipt; inspect '+log.name)
    record=json.loads(receipt.read_text());run.artifact(receipt,'receipt','Fresh '+backend+' native-package receipt.')
    run.state['software'].append({'name':{'class':'CLASS','axiclass':'AxiCLASS','blender':'Blender','monofonic':'monofonIC','grackle':'Grackle','music2':'MUSIC','rockstar':'Rockstar','velociraptor':'VELOCIraptor','consistent-trees':'Consistent Trees','disperse':'DisPerSE','opengadget3':'OpenGadget3'}[backend],
        'module':backend,'available':record['status']!='UNAVAILABLE','execution':'CALCULATION_'+record['status'],
        'receipt':receipt.relative_to(run.output).as_posix()})
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path==receipt or '.native-run-locks' in path.parts:continue
        kind='image' if path.suffix=='.png' else 'scene' if path.suffix=='.blend' else 'log' if path.suffix=='.log' else 'data'
        run.artifact(path,kind,'Actual '+backend+' native output; inspect its retained units and scope.')
    if record['status']=='UNAVAILABLE':raise ImportError(record['reason'])
    if record['status']!='PASS' or process.returncode:raise RuntimeError(record.get('reason','Native worker failed'))
    return record['result']


def physics_control(run,backend):
    """Run a bounded native source control, including any cold compilation."""
    from optional_tests.native_physics import CHECKS
    if backend not in CHECKS:raise ValueError('Unknown fixed physics control')
    output=run.output/'physics'/backend;output.parent.mkdir(exist_ok=True)
    log=output.parent/(backend+'.log')
    try:
        with log.open('w') as stream:
            process=subprocess.run([sys.executable,str(HERE/'native_physics.py'),backend,'--output',str(output),
                '--seed',str(run.state['seed'])],cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                stdout=stream,stderr=subprocess.STDOUT,timeout=930)
    finally:
        if log.is_file():run.artifact(log,'log','Actual native '+backend+' physics worker output.')
    receipt=output/'receipt.json'
    if not receipt.is_file():raise RuntimeError('Physics worker stopped without a receipt; inspect '+log.name)
    record=json.loads(receipt.read_text())
    run.artifact(receipt,'receipt','Native '+backend+' source-control receipt.')
    run.state['software'].append(dict(name='GADGET-4 '+backend+' control',module='GADGET-4:'+backend,
        available=record['status']!='UNAVAILABLE',execution='CALCULATION_'+record['status'],
        receipt=receipt.relative_to(run.output).as_posix()))
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path==receipt or '.native-run-locks' in path.parts:continue
        kind='image' if path.suffix=='.png' else 'log' if path.suffix=='.log' else 'data'
        run.artifact(path,kind,'Actual native '+backend+' output; inspect original units and scope.')
    if record['status']=='UNAVAILABLE':raise ImportError(record['reason'])
    if record['status']!='PASS' or process.returncode:raise RuntimeError(record.get('reason','Physics worker failed'))
    return record['result']


def figure_control(run,control):
    """Regenerate a self-contained known-input example in an isolated worker."""
    from optional_tests.advanced_controls import CHECKS
    if control not in CHECKS:raise ValueError('Unknown known-input figure control')
    output=run.output/'figure-controls'/control;output.parent.mkdir(exist_ok=True)
    log=output.parent/(control+'.log')
    try:
        with log.open('w') as stream:
            process=subprocess.run([sys.executable,str(HERE/'advanced_controls.py'),control,
                '--output',str(output),'--seed',str(run.state['seed']),'--memory-gib','4'],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                stdout=stream,stderr=subprocess.STDOUT,timeout=930)
    finally:
        if log.is_file():run.artifact(log,'log','Known-input '+control+' worker output.')
    receipt=output/'receipt.json'
    if not receipt.is_file():raise RuntimeError('Figure worker stopped without a receipt; inspect '+log.name)
    record=json.loads(receipt.read_text());run.artifact(receipt,'receipt','Known-input figure-control receipt.')
    run.state['software'].append(dict(name=('Research2 analytic flyby' if control=='orbits' else 'AGAMA + production '+control+' figures'),module='advanced:'+control,
        available=record['status']!='UNAVAILABLE',execution='CALCULATION_'+record['status'],
        receipt=receipt.relative_to(run.output).as_posix()))
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path==receipt or '.native-run-locks' in path.parts:continue
        kind='image' if path.suffix=='.png' else 'log' if path.suffix=='.log' else 'data'
        run.artifact(path,kind,'Original known-input '+control+' output, with units and synthetic scope retained.')
    if record['status']=='UNAVAILABLE':raise ImportError(record['reason'])
    if record['status']!='PASS' or process.returncode:raise RuntimeError(record.get('reason','Figure worker failed'))
    return record['result']


def write_report(output,state):
    """Write an offline human report that needs no web server or notebook extensions."""
    css=(HERE/'app/styles.css').read_text() if (HERE/'app/styles.css').exists() else 'body{font:16px system-ui;margin:4em;max-width:1000px}'
    rows=''.join('<article class="receipt"><small>'+html.escape(r['status'])+'</small><h2>'+html.escape(r['title'])+'</h2><pre>'+html.escape(json.dumps(r.get('result',{'reason':r.get('reason')}),indent=2))+'</pre><p>'+html.escape(', '.join(p['function'] for p in r['implementation']))+'</p></article>' for r in state['steps'])
    pictures=''.join('<figure><img style="max-width:100%" src="'+html.escape(a['path'],quote=True)+'"><figcaption>'+html.escape(a['caption'])+'</figcaption></figure>' for a in state['artifacts'] if a['kind']=='image')
    (Path(output)/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Research2 optional lab</title><style>'+css+'</style><main class="offline"><p class="eyebrow">RESEARCH2 / RETAINED TEACHING RUN</p><h1>'+html.escape(state['status'])+'</h1><p>'+html.escape(state['scope'])+'</p><p>'+str(state['particles'])+' synthetic particles · '+str(state['elapsed_seconds'])+' seconds</p>'+rows+pictures+'</main>')


def run_lab(output,particles=32,seed=42,profile='quick'):
    """Run real numerical exercises with per-step receipts and explicit synthetic provenance."""
    run=LabRun(output,particles,seed,profile)
    from optional_tests.execution import PythonCalls
    observed=PythonCalls(ROOT)
    observed.__enter__()
    try:
        import numpy
        from core.data import read_species_phase_space
        from core.analysis import ngp_density,fft_power_spectrum
        from core.stream_forward_model import integrate_stream
        from core.stream_common import weighted_covariance
        inspect_software(run)
        run.step('fixture','Generate a declared synthetic HDF5 fixture',['NumPy','h5py'],['units'],[make_fixture],lambda:make_fixture(run))
        if hasattr(run,'fixture'):
            run.step('units','Round-trip exact IDs and physical units',['h5py','Research2'],['units'],[read_species_phase_space],lambda:check_units(run))
            run.step('density','Measure shell density and surface mass',['NumPy','Matplotlib','Research2'],['density','circular'],[ngp_density,measure_density],lambda:measure_density(run))
            run.step('power','Measure diagnostic FFT power',['SciPy FFT','Research2'],['power'],[fft_power_spectrum],lambda:measure_power(run))
            run.step('zoom','Measure nested views of the same particles',['NumPy','Matplotlib'],['density','units'],[measure_zoom],lambda:measure_zoom(run))
            run.step('orbits','Integrate a physical stream tracer arc',['SciPy DOP853','Research2'],['orbit'],[integrate_stream],lambda:measure_orbits(run))
            run.step('covariance','Measure the velocity covariance',['NumPy','Research2'],['covariance'],[weighted_covariance],lambda:measure_covariance(run))
        if profile in {'native','combined','engines','complete'}:
            from core.research.controls import run_control_campaign
            run.step('native','Run GADGET with 32 particles',['GADGET-4','MPI'],['orbit'],[native_control,run_control_campaign],lambda:native_control(run))
        if profile in {'libraries','combined','engines','complete'}:
            from optional_tests.software_checks import CHECKS,calculate
            for backend in CHECKS:
                run.step('library-'+backend,'Calculate with '+backend,[backend],[],[library_control,calculate],
                    lambda backend=backend:library_control(run,backend))
        if profile in {'engines','complete'}:
            from optional_tests.native_checks import CHECKS,calculate
            for backend in CHECKS:
                run.step('engine-'+backend,'Execute native '+backend,[backend],['linear_bundle'] if backend in {'class','axiclass'} else [],[native_package_control,calculate],
                    lambda backend=backend:native_package_control(run,backend))
        if profile in {'physics','complete'}:
            from optional_tests.native_physics import CHECKS,calculate
            for backend in CHECKS:
                equation={'gas':'cooling_source','stars':'stellar_birth','black-holes':'bh_accretion'}[backend]
                run.step('physics-'+backend,'Evolve native '+backend,['GADGET-4','MPI'],[equation],
                    [physics_control,calculate],lambda backend=backend:physics_control(run,backend))
        if profile in {'figures','complete'}:
            from optional_tests.advanced_controls import CHECKS,calculate
            for control in CHECKS:
                equation={'shapes':'shape_tensor','splashback':'splashback_slope','orbits':'flyby_events'}[control]
                run.step('figure-'+control,'Regenerate known '+control+' figures',['Research2','SciPy'] if control=='orbits' else ['AGAMA','Research2'],[equation],
                    [figure_control,calculate],lambda control=control:figure_control(run,control))
    except Exception as error:
        run.state['steps'].append({'id':'environment','title':'Load required environment','status':'FAIL',
          'reason':str(error),'traceback':traceback.format_exc(),'implementation':[]})
    finally:
        observed.__exit__(None,None,None)
        run.state['python_execution']=observed.report()
        path=run.output/'python-calls.json'
        write_json(path,run.state['python_execution'])
        run.artifact(path,'receipt','Observed first-party Python calls in the runner main thread; imports and subprocesses have separate scopes.')
    return run.finish()


def main():
    """CLI for the optional lab; all output destinations must be new directories."""
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('--output',type=Path,required=True)
    run.add_argument('--particles',type=int,default=32);run.add_argument('--seed',type=int,default=42)
    run.add_argument('--profile',choices=['quick','visual','native','libraries','combined','engines','physics','complete','figures'],default='quick')
    serve=commands.add_parser('serve');serve.add_argument('--port',type=int,default=8765)
    serve.add_argument('--output-root',type=Path,default=ROOT.parent/'research2-runtime/optional-lab')
    args=parser.parse_args()
    if args.command=='serve':
        from optional_tests.server import serve
        serve(args.port,args.output_root)
    else:
        state=run_lab(args.output,args.particles,args.seed,args.profile)
        print(json.dumps({'status':state['status'],'report':str(args.output.resolve()/'report.html'),
          'steps':[{k:r[k] for k in ['id','status']} for r in state['steps']]},indent=2))
        raise SystemExit(1 if state['status']=='FAILED' else 0)

if __name__=='__main__':main()
