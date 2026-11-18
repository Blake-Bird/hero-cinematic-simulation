"""Small real optional-library calculations, isolated one process per backend.

No installation or downloads occur. Every calculation saves actual arrays;
short inference runs are execution exercises, not converged posteriors.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CHECKS=('agama','gala','streamax','fsps','emcee','dynesty','cloudpickle','sbi','lenstronomy','pyvista','pyrecon','pycorr')


def calculate(name, output, seed):
    """Execute a fixed bounded backend request and return measured results and scope."""
    import numpy as np
    output=Path(output);arrays={};result={};sources={}
    rng=np.random.default_rng(seed)
    module_name={'streamax':'StreaMAX'}.get(name,name)
    if importlib.util.find_spec(module_name) is None:
        raise ImportError('Optional '+module_name+' profile is not installed in this Python environment.')
    if name=='pycorr' and importlib.util.find_spec('Corrfunc') is None:
        raise ImportError('The pycorr calculation also needs its Corrfunc dependency profile.')
    if name in {'emcee','dynesty','cloudpickle'}:
        from core.stream.inference import pinned_sampler_backend
        module,proof=pinned_sampler_backend(name);sources.update(proof['source_hashes'])
        if name=='cloudpickle':
            payload=module.dumps(lambda x:x*x);answer=module.loads(payload)(3)
            assert answer==9
            arrays={'serialized_bytes':np.frombuffer(payload,dtype=np.uint8),'result':np.array([answer])}
            result={'round_trip_f3':answer,'scope':'Actual function serialization and execution, not a sampler.'}
        elif name=='emcee':
            sampler=module.EnsembleSampler(8,1,lambda x:-.5*np.dot(x,x))
            sampler.random_state=np.random.RandomState(seed).get_state()
            sampler.run_mcmc(rng.normal(size=(8,1)),10,progress=False)
            arrays={'chain':sampler.get_chain(),'log_probability':sampler.get_log_prob()}
            assert arrays['chain'].shape==(10,8,1) and np.isfinite(arrays['log_probability']).all()
            result={'chain_shape':list(arrays['chain'].shape),'scope':'Ten-step toy unit-Gaussian chain; convergence is untested.'}
        else:
            sampler=module.NestedSampler(lambda x:-.5*np.dot(x,x),lambda u:10*u-5,1,nlive=20,rstate=rng)
            sampler.run_nested(maxiter=40,print_progress=False)
            arrays={'samples':sampler.results.samples,'log_evidence':sampler.results.logz}
            assert len(arrays['samples']) and np.isfinite(arrays['log_evidence']).all()
            result={'samples':len(arrays['samples']),'scope':'Short Gaussian nested-sampling control, stopped by iteration cap; evidence accuracy is untested.'}
    elif name=='agama':
        from core.halos import pinned_agama_backend
        agama,proof=pinned_agama_backend()
        sources.update(proof['installed_asset_hashes'])
        potential=agama.Potential(type='Plummer',mass=1.,scaleRadius=1.)
        force=potential.force(np.array([[1.,0.,0.]]))
        np.testing.assert_allclose(force,[[-float(agama.G)*2**-1.5,0.,0.]],rtol=1e-10,atol=1e-12)
        arrays={'position_proper_kpc':np.array([[1.,0.,0.]]),'acceleration_km2_s2_kpc':force}
        result={'Fx_km2_s2_kpc':float(force[0,0]),'G_kpc_km2_s2_msun':float(agama.G),
            'scope':'M=1 Msun, b=1 proper kpc Plummer force compared with -G/(2 sqrt(2)); pinned adapter sets physical units.'}
    elif name=='gala':
        import gala.potential as gp
        from gala.units import galactic
        from core.stream_resimulation import gala_spray_adapter
        request=dict(mass_msun=1e4,scale_kpc=.01,step_myr=-.1,steps=8,release_every=1,
            particles_per_release=2,seed=seed,position_kpc=[8.,0.,0.],velocity_kms=[0.,200.,0.])
        proof,arrays=gala_spray_adapter(request,gp.HernquistPotential(m=1e12,c=20.,units=galactic))
        assert len(arrays['position_kpc']) and np.isfinite(arrays['position_kpc']).all()
        result={'samples':len(arrays['position_kpc']),'request':request,'scope':'Actual short spray in a static Hernquist field; not native star formation or a converged stream.'}
    elif name=='streamax':
        from core.stream_resimulation import streamamax_reference
        request=dict(input_units='proper_kpc_kms_myr',type_host='Plummer',params_host={'logM':12.,'Rs':20.,'x_origin':0.,'y_origin':0.,'z_origin':0.},
            type_sat='Plummer',params_sat={'logM':4.,'Rs':.01,'x_origin':0.,'y_origin':0.,'z_origin':0.},
            xv_f=[8.,0.,0.,0.,200.,0.],time_myr=1.,n_steps=7,n_particles=32,unroll=1,m_f_sat=4.,alpha=1.,seed=seed,type_method='Fardal2015')
        proof,arrays=streamamax_reference(request)
        assert len(arrays['position_kpc'])==32 and np.isfinite(arrays['position_kpc']).all()
        result={'samples':32,'request':request,'scope':'Actual static-host StreaMAX reference; its own release model and approximations apply.'}
    elif name=='fsps':
        from core.stellar_population import pinned_fsps
        home=ROOT/'core/software/sps/python-fsps/src/fsps/libfsps'
        if not home.is_dir():raise ImportError('FSPS source/data profile is absent; install the sps profile first.')
        os.environ['SPS_HOME']=str(home)
        request=dict(sps_home=str(home),data_source_sha256=json.loads((ROOT/'configs/research/fsps-data-source-sha256.json').read_text()),
            expected_libraries=['mist','miles','DL07'],fsps_parameters={'imf_type':2},filters=['sdss_g','sdss_r'])
        pop,proof=pinned_fsps(request);wave,spectrum=pop.get_spectrum(tage=1.,peraa=True)
        mags=pop.get_mags(tage=1.,bands=request['filters'])
        assert len(wave)==len(spectrum) and np.isfinite(spectrum).all() and np.isfinite(mags).all()
        arrays={'wavelength_angstrom':wave,'spectrum_per_angstrom':spectrum,'sdss_g_r_magnitudes':mags}
        result={'age_gyr':1.,'spectral_samples':len(wave),'sdss_g_r_magnitudes':mags.tolist(),
            'spectrum_units':'Lsun / angstrom for the SSP normalized to 1 Msun initially formed',
            'scope':'FSPS 1-Gyr SSP, selected IMF and library defaults; not measured luminosity of lab marker particles.'}
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained');select=(wave>=1000)&(wave<=30000)&(spectrum>0)
        ax.loglog(wave[select],spectrum[select]);ax.set(xlabel='Wavelength [angstrom]',ylabel='Spectrum [Lsun / angstrom]',title='FSPS 1-Gyr SSP · 1 Msun initially formed')
        fig.savefig(output/'spectrum.png',dpi=140);plt.close(fig)
    elif name=='sbi':
        from core.stream.sbi import pinned_sbi
        proof=pinned_sbi();sources.update(proof['source_hashes'])
        import torch
        from sbi.inference import SNPE
        from torch.utils.tensorboard import SummaryWriter
        from torch.distributions import Independent,Uniform
        torch.set_num_threads(2);torch.manual_seed(seed)
        prior=Independent(Uniform(torch.tensor([-2.]),torch.tensor([2.])),1)
        theta=prior.sample((32,));observed=theta+.1*torch.randn_like(theta)
        # Upstream defaults to sbi-logs beneath cwd. Keep generated training
        # events with this disposable run instead of dirtying the source tree.
        with SummaryWriter(log_dir=str(output/'training-logs')) as writer:
            model=SNPE(prior=prior,summary_writer=writer,show_progress_bars=False).append_simulations(theta,observed).train(
                max_num_epochs=2,training_batch_size=8,show_train_summary=False)
        arrays={'theta':theta.numpy(),'observed':observed.numpy()}
        assert all(torch.isfinite(p).all() for p in model.parameters())
        torch.save(model.state_dict(),output/'density-estimator.pt')
        result={'simulations':32,'network_parameters':sum(p.numel() for p in model.parameters()),
            'scope':'Actual SBI toy density-estimator training; no held-out calibration, deployment or native-stream acceptance.'}
    elif name=='lenstronomy':
        from core.observation_data import activate_pinned_backend
        _,proof=activate_pinned_backend('lenstronomy')
        from lenstronomy.LensModel.lens_model import LensModel
        lens=LensModel(['SIS']);x=np.array([1.,0.,-1.,0.]);y=np.array([0.,1.,0.,-1.])
        ax,ay=lens.alpha(x,y,[{'theta_E':1.,'center_x':0.,'center_y':0.}])
        np.testing.assert_allclose(ax,x,atol=1e-12);np.testing.assert_allclose(ay,y,atol=1e-12)
        arrays={'x_arcsec':x,'y_arcsec':y,'alpha_x_arcsec':ax,'alpha_y_arcsec':ay}
        result={'rays':4,'Einstein_radius_arcsec':1.,'scope':'SIS unit-radius deflections; no image likelihood or H0 posterior.'}
    elif name in {'pyrecon','pycorr'}:
        from core.bao import apply_recorded_bao_reconstruction,pycorr_measurement
        def catalogue(n,namespace):
            return dict(role='estimator_observations',namespace=namespace,
                selection={'kind':'periodic_uniform_synthetic_control'},units={'position_mpc':'Mpc','weight':'1'},
                columns={'object_id':list(range(n)),'position_mpc':rng.uniform(0,100,(n,3)).tolist(),'weight':np.ones(n).tolist()})
        data,random=catalogue(32,'lab-tracers'),catalogue(128,'lab-randoms')
        if name=='pyrecon':
            spec=dict(algorithm='IterativeFFTReconstruction',convention='RecSym',box_mpc=100.,box_center_mpc=[50.,50.,50.],
                growth_rate=.8,bias=2.,smoothing_mpc=10.,nmesh=8,los='z',dtype='f8')
            measured=apply_recorded_bao_reconstruction(data,random,spec)
            arrays={'initial_position_mpc':np.asarray(data['columns']['position_mpc']),
                'shifted_position_mpc':np.asarray(measured['data_positions_mpc']),
                'shifted_randoms_mpc':np.asarray(measured['shifted_randoms_mpc'])}
            result={'tracers':32,'randoms':128,'settings':spec,'backend_revision':measured['backend_revision'],
                'scope':'Actual reconstruction of uniform synthetic tracers on an 8-cubed mesh. No BAO detection, resolved displacement accuracy or cosmological fit.'}
        else:
            spec=dict(threads=2,separation_edges_mpc=[1.,15.,30.,45.],mu_edges=[-1.,0.,1.],los='z',box_mpc=100.)
            measured=pycorr_measurement(data,random,spec)
            arrays={k:np.asarray(measured[k]) for k in ['xi_smu','separation_centers_mpc','mu_centers','separation_edges_mpc','mu_edges']}
            assert np.isfinite(arrays['xi_smu']).all()
            result={'tracers':32,'randoms':128,'settings':spec,'backend_revision':measured['backend_revision'],
                'counting_revision':measured['counting_revision'],'scope':'Actual pycorr estimator with Corrfunc pair counting; uniform tiny mock, dimensionless xi, no BAO fit.'}
    elif name=='pyvista':
        import pyvista as pv
        points=rng.normal(size=(32,3));mesh=pv.PolyData(points)
        plot=pv.Plotter(off_screen=True,window_size=[480,360])
        try:
            plot.add_mesh(mesh,render_points_as_spheres=True,point_size=10)
            plot.screenshot(str(output/'points.png'))
        finally:plot.close()
        from PIL import Image
        with Image.open(output/'points.png') as im:im.verify()
        assert mesh.n_points==32
        arrays={'position_display_units':points};result={'points':32,'scope':'Actual offscreen rendering of seeded geometry; not a physical simulation.'}
    else:raise ValueError('Unknown fixed library check')
    def collect_proof(value):
        if not isinstance(value,dict):return
        for key,child in value.items():
            if key in {'source_hashes','installed_asset_hashes','installed_source_hashes'} and isinstance(child,dict):
                sources.update({str(p):h for p,h in child.items() if isinstance(h,str) and len(h)==64})
            elif isinstance(child,dict):collect_proof(child)
    if 'proof' in locals():collect_proof(proof)
    # Bind the actually loaded Python backend's entry file when available.
    module_name={'streamax':'StreaMAX'}.get(name,name)
    module=sys.modules.get(module_name)
    if module is not None and getattr(module,'__file__',None):
        from core.foundation import file_sha256
        path=Path(module.__file__).resolve();sources[str(path)]=file_sha256(path)
    np.savez_compressed(output/'arrays.npz',**arrays)
    result['source_hashes']=sources
    result['arrays']={k:{'shape':list(np.asarray(v).shape),'dtype':str(np.asarray(v).dtype)} for k,v in arrays.items()}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend',choices=CHECKS);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:record={'status':'PASS','backend':args.backend,'result':calculate(args.backend,args.output,args.seed)}
    except ImportError as error:record={'status':'UNAVAILABLE','backend':args.backend,'reason':str(error)}
    except Exception as error:record={'status':'FAIL','backend':args.backend,'reason':str(error),'traceback':traceback.format_exc()}
    (args.output/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='result'}),flush=True)
    raise SystemExit(1 if record['status']=='FAIL' else 0)


if __name__=='__main__':main()
