"""Regenerate known-shape, density-valley and flyby figures using production readers.

Run with the Research2 environment from any directory. Shapes uses 6000
particles; splashback uses 32768; orbits uses 32. These are authored HDF5 inputs, not evolved
or converged halos. No downloads, builds, movies or archives are created.
Missing dependencies report UNAVAILABLE; all generated files stay external.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import signal
import sys
import traceback

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CHECKS=('shapes','splashback','orbits')


def sha256(path):
    """Hash file bytes without importing scientific dependencies."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fixture(output,offsets,*,control,seed):
    """Write complete explicit-unit geometry, exact IDs and an authored centre.

    Offsets are proper kpc and every carrier has mass 1 Msun. The a=.5,h=.7
    adapter preserves those quantities. The catalogue aperture is synthetic,
    not a finder detection or measured R200. The clock is an analytic label.
    """
    import h5py
    import numpy as np
    from core.foundation import KPC_CM,MSUN_G,C_SI,SECONDS_PER_YEAR
    (output/'data').mkdir();(output/'input').mkdir()
    count=np.array([0,len(offsets),0,0,0,0],dtype=np.uint32)
    parameters=dict(UnitLength_in_cm=KPC_CM,UnitMass_in_g=1e10*MSUN_G,
        UnitVelocity_in_cm_per_s=1e5,HubbleParam=.7,Omega0=.3,OmegaLambda=.7,
        OmegaBaryon=.05,ComovingIntegrationOn=1,InitCondFile=str(output/'input/initial_source'))
    generation=f'optional-{control}-{seed}'
    snapshot=output/'data/snapshot_000.hdf5'
    with h5py.File(snapshot,'w') as f:
        f.attrs.update(synthetic_fixture=True,purpose='Authored geometry; no native solver evolution')
        f.create_group('Header').attrs.update(Time=.5,Redshift=1.,BoxSize=100.,MassTable=np.zeros(6),
            NumFilesPerSnapshot=1,NumPart_Total=count,NumPart_ThisFile=count,
            NumPart_Total_HighWord=np.zeros(6,dtype=np.uint32),HubbleParam=.7,
            Omega0=.3,OmegaBaryon=.05,OmegaLambda=.7,Research2Generation=generation,Research2ShardCount=1)
        f.create_group('Parameters').attrs.update(parameters);g=f.create_group('PartType1')
        for name,value,unit,aa,hh in [('Coordinates',50+offsets*.7/.5,KPC_CM,1.,-1.),
                ('Velocities',np.zeros_like(offsets),1e5,.5,0.),
                ('Masses',np.full(len(offsets),.7/1e10),1e10*MSUN_G,0.,-1.)]:
            g[name]=value;g[name].attrs.update(to_cgs=unit,a_scaling=aa,h_scaling=hh)
        g['ParticleIDs']=np.arange(1,len(offsets)+1,dtype=np.uint64)
    (output/'data/snapshot_000.publication.json').write_text(json.dumps(dict(
        schema='native-hdf5-set-publication-v1',state='COMPLETE',generation=generation,shard_count=1),indent=2))
    with h5py.File(output/'data/fof_subhalo_tab_000.hdf5','w') as f:
        f.attrs.update(synthetic_fixture=True,purpose='Authored aperture centre; no finder or R200 measurement')
        f.create_group('Header').attrs.update(Time=.5,BoxSize=100.,NumFiles=1,Ngroups_Total=1,Ngroups_ThisFile=1)
        f.create_group('Parameters').attrs.update(parameters);g=f.create_group('Group')
        g['GroupPos']=np.array([[50.,50.,50.]]);g['Group_R_Crit200']=np.array([4*.7/.5])
        g['GroupLenType']=count[None,:].astype(np.uint64);g['GroupOffsetType']=np.zeros((1,6),dtype=np.uint64)
    initial=output/'input/initial_source.hdf5';shutil.copyfile(snapshot,initial)
    a=np.linspace(.5,.500005,6);t=1+2*(a-.5);h=1000*KPC_CM/(C_SI*100*2e9*SECONDS_PER_YEAR*a)
    clock=output/'input/background.dat'
    clock.write_text('# 1:z 2:proper time [Gyr] 3:Omega_r(z) 4:H [1/Mpc]\n'+''.join(
        f'{1/x-1:.17g} {y:.17g} 0 {z:.17g}\n' for x,y,z in zip(a,t,h)))
    (output/'manifest.json').write_text(json.dumps(dict(fixture=True,scope=__doc__,
        gadget_handoff=dict(initial_conditions=[dict(path=str(initial),sha256=sha256(initial))]),
        input_artifacts=dict(boltzmann_background=dict(path=str(clock),sha256=sha256(clock),
            origin='Authored analytic clock, not CLASS execution'))),indent=2))


def structure_request(output,control,memory_gib,seed):
    """Declare the complete production request and fixed recovery controls."""
    import numpy as np
    shapes=control=='shapes'
    return dict(schema='native-halo-structure-v1',run_directory=str(output),snapshot=0,host_row=0,
        components=dict(dm=dict(species=[1],kind='spheroidal',disk_rotation=np.eye(3).tolist())),
        support_radius_proper_kpc=6. if shapes else 4.1,
        radial_edges_proper_kpc=[0.,.5,1.,2.,3.] if shapes else np.r_[0.,np.geomspace(.28,3.6,33)].tolist(),
        bulk_velocity_peculiar_km_s=[0.,0.,0.],
        shape_controls=dict(window='iterative_ellipsoidal_shell' if shapes else 'spherical_shell',
            estimator='mass_quadrupole',minimum_particles=100 if shapes else 32,tolerance=1e-5,
            maximum_iterations=150 if shapes else 50,minimum_axis_ratio=.1,
            bootstrap_realizations=32 if shapes else 0,seed=seed,
            minimum_bootstrap_success_fraction=.9 if shapes else .5,
            minimum_eigenvalue_gap=.02 if shapes else .03,maximum_orientation_error_deg=20.),
        expansion_sequence=[dict(lmax=4,mmax=4,grid_size_r=20,grid_size_z=20,
            rmin_kpc=.05,rmax_kpc=40.,zmin_kpc=.05,zmax_kpc=10.)],
        calibration_positions_proper_kpc=[[30.,0.,0.],[0.,30.,0.],[0.,0.,30.]],
        validation_positions_proper_kpc=[[25.,2.,0.],[0.,25.,2.],[2.,0.,25.]],
        field_error_budget=dict(force_relative_tolerance=.04,force_floor_km2_s2_kpc=.001,
            tidal_relative_tolerance=.05,tidal_floor_km2_s2_kpc2=.001),
        equipotential_controls=dict(reference_radii_kpc=[2.],radial_bracket_kpc=[.1,8.],directions=24,
            maximum_shape_residual=.15,minimum_eigenvalue_gap=.03),concentration=None,memory_gib=memory_gib)


def calculate(control,output,*,seed,memory_gib):
    """Generate inputs, run real producers and verify analytic recovery."""
    if control=='orbits':
        from optional_tests.orbital_control import calculate as flyby
        return flyby(output,seed=seed,memory_gib=memory_gib)
    import numpy as np
    from core.halos import pinned_agama_backend,publish_halo_structure,verify_halo_structure
    from core.halo_products import export_native_structure_figures,export_native_splashback_figures
    from core.analysis import publish_native_environment
    pinned_agama_backend()
    if control=='shapes':
        n=1200;z=1-2*(np.arange(n)+.5)/n;phi=np.arange(n)*np.pi*(3-np.sqrt(5))
        directions=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z));pieces=[]
        for i,r in enumerate((.3,.75,1.5,2.5,3.5)):
            t=.15*i;rotation=np.array([[np.cos(t),-np.sin(t),0.],[np.sin(t),np.cos(t),0.],[0.,0.,1.]])
            pieces.append((directions*r*np.array([1.,.7,.45])/(.7*.45)**(1/3))@rotation.T)
        offsets=np.concatenate(pieces)
        expected=dict(axis_ratios=[.7,.45],adjacent_z_rotation_deg=float(np.degrees(.15)),
            axis_ratio_absolute_tolerance=1e-4,twist_absolute_tolerance_deg=.02)
    else:
        from scipy.special import erf
        from scipy.integrate import cumulative_trapezoid
        n=32768;width=.2;r=np.geomspace(.2,4.,200000);x=np.log(r)
        log_density=-2*x-3*width*np.sqrt(np.pi/2)*erf(x/(np.sqrt(2)*width))
        cdf=np.r_[0.,cumulative_trapezoid(np.exp(log_density)*r*r,r)];cdf/=cdf[-1]
        radius=np.interp((np.arange(n)+.5)/n,cdf,r)
        rng=np.random.Generator(np.random.Philox(seed));radius=radius[rng.permutation(n)]
        z=1-2*(np.arange(n)+.5)/n;phi=np.arange(n)*np.pi*(3-np.sqrt(5))
        offsets=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))*radius[:,None]
        np.savez_compressed(output/'analytic-profile.npz',radius_proper_kpc=r,
            log_density_up_to_normalization=log_density,logarithmic_slope=-2-3*np.exp(-x*x/(2*width**2)))
        expected=dict(minimum_radius_kpc=1.,prominence=3.,half_prominence_width_ln_radius=float(2*np.sqrt(2*np.log(2))*width),
            radius_fractional_tolerance=.1,depth_fractional_tolerance=.2,width_fractional_tolerance=.2,
            sampling_domain_proper_kpc=[.2,4.],search_domain_proper_kpc=[.5,2.])
    np.savez_compressed(output/'authored-geometry.npz',offsets_proper_kpc=offsets,
        particle_ids=np.arange(1,len(offsets)+1,dtype=np.uint64))
    request=structure_request(output,control,memory_gib,seed)
    (output/'predeclared.json').write_text(json.dumps(dict(control=control,seed=seed,
        particles=len(offsets),expected=expected,structure=request,scope=__doc__),indent=2))
    write_fixture(output,offsets,control=control,seed=seed)
    product=output/'structure.json';publish_halo_structure(request,product);report=verify_halo_structure(product)
    if report['field_validation_status']!='PASS':raise ValueError('Independent field comparison did not pass')
    if control=='shapes':
        rows=report['density_shapes']['dm']['shells'];twists=report['density_shapes']['dm']['radial_twists']
        if not all(r['status']=='MEASURED' and all(r['orientation_sampling_resolved']) for r in rows):raise ValueError('Known orientations did not resolve')
        np.testing.assert_allclose([[r['b_over_a'],r['c_over_a']] for r in rows],np.tile([.7,.45],(4,1)),rtol=0,atol=1e-4)
        np.testing.assert_allclose([r['unsigned_axis_twist_deg'] for r in twists],np.tile([np.degrees(.15),np.degrees(.15),0.],(3,1)),rtol=0,atol=.02)
        gallery=export_native_structure_figures(product,output/'figures',memory_gib=memory_gib)
        measurements=dict(shells=len(rows),measured_twists=twists)
    else:
        environment=dict(schema='native-environment-v1',run_directory=str(output),snapshot=0,mesh_size=9,
            smoothing_comoving_kpc=[0.],web_threshold=.1,eigenvalue_gap_floor=.001,
            halo_structure_products=[str(product)],splashback_component='dm',orbital_history=None,
            slope_window_study=[5,7,9],slope_controls=dict(window_bins=7,polynomial_degree=3,minimum_effective_count=20),
            splashback_controls=dict(search_interval_proper_kpc=[.5,2.],minimum_depth=.5,bootstrap_realizations=64,
                seed=571,maximum_fractional_radius_error=.2,minimum_success_fraction=.8),
            alignment_controls=dict(randomized_realizations=32,seed=713),memory_gib=memory_gib)
        (output/'environment-request.json').write_text(json.dumps(environment,indent=2))
        publish_native_environment(environment,output/'environment.json')
        row=json.loads((output/'environment.json').read_text())['halo_measurements'][0];feature=row['splashback']
        if feature['status']!='MEASURED':raise ValueError('Known feature unresolved: '+json.dumps(feature))
        c=feature['candidates'][0]
        for key,truth,tolerance in [('radius_proper_kpc',1.,.1),('depth',3.,.2),('half_depth_width_ln_radius',expected['half_prominence_width_ln_radius'],.2)]:
            if abs(c[key]/truth-1)>=tolerance:raise ValueError('Known feature recovery failed: '+key)
        gallery=export_native_splashback_figures(output/'environment.json',output/'figures',memory_gib=memory_gib)
        measurements=dict(feature=feature,window_study=[dict(window_bins=r['window_bins'],status=r['feature']['status']) for r in row['slope_window_study']])
    return dict(control=control,particles=len(offsets),seed=seed,expected=expected,measurements=measurements,
        field_validation_status=report['field_validation_status'],figures=len(gallery['figures']),
        gallery='figures/gallery/gallery.json',native_solver_executed=False,
        scope='Authored HDF5 -> production readers, AGAMA fit/reload/direct checks, measurements and figures. No evolution or convergence claim.')


def write_report(output,record):
    """Write an offline result page with original gallery/image links."""
    images=sorted(output.glob('figures/gallery/figures/png/**/*.png'))
    pictures=''.join('<figure><img src="'+html.escape(p.relative_to(output).as_posix(),quote=True)+'"><figcaption>'+html.escape(p.stem)+'</figcaption></figure>' for p in images)
    links='<a href="receipt.json">Receipt</a>'
    if (output/'predeclared.json').is_file():links+=' · <a href="predeclared.json">Inputs and criteria</a>'
    if (output/'figures/gallery/gallery.json').is_file():links+=' · <a href="figures/gallery/gallery.json">Gallery</a>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Research2 known-input control</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:3rem auto;padding:1rem;background:#101725;color:#e7efff}a{color:#89d7ff}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:2rem 0}</style>'
        '<h1>Known-input control: '+html.escape(record['status'])+'</h1><p>Authored geometry; not evolved or converged halos.</p><p>'+links+'</p><pre>'+html.escape(json.dumps(record,indent=2))+'</pre>'+pictures)


def main():
    """Run with retained failures/unavailability and a new external output."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control',choices=CHECKS)
    parser.add_argument('--output',type=Path,required=True,help='New directory outside the source tree')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--memory-gib',type=float,default=4.,help='Estimated workspace allocation, not an RSS limit (default: 4)')
    args=parser.parse_args();output=args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):parser.error('Keep generated controls outside the source tree')
    if not 0<=args.seed<2**32:parser.error('Seed must be in 0..2^32-1')
    if not .5<=args.memory_gib<=16:parser.error('Memory allocation must be finite and between 0.5 and 16 GiB')
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('orbital_control.py'),*(ROOT/'core'/n for n in ('halos.py','data.py','foundation.py','analysis.py','halo_products.py','zoom.py','native.py','assembly.py','flagship_orbits.py','stream/trajectories.py'))]
    pins={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    def timeout(signum,frame):raise TimeoutError('Known-input control exceeded 900 seconds')
    def cancel(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,cancel);signal.alarm(900)
    try:record=dict(status='PASS',result=calculate(args.control,output,seed=args.seed,memory_gib=args.memory_gib))
    except ImportError as error:record=dict(status='UNAVAILABLE',reason=str(error))
    except Exception as error:record=dict(status='FAIL',reason=str(error),traceback=traceback.format_exc())
    finally:signal.alarm(0)
    if any(sha256(ROOT/n)!=v for n,v in pins.items()):record=dict(status='FAIL',reason='A control source changed during execution')
    record.update(control=args.control,source_sha256=pins,memory_gib=args.memory_gib)
    write_report(output,record)
    record['artifacts_sha256']={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and '.native-run-locks' not in p.parts and p.name!='receipt.json'}
    (output/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps(dict(status=record['status'],output=str(output),receipt=str(output/'receipt.json'))))
    return int(record['status']=='FAIL')


if __name__=='__main__':
    raise SystemExit(main())
