"""Thin-plane convergence and shear from a saved projected mass map."""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def thin_lens_from_surface(surface_msun_kpc2, pixel_kpc, lens_kpc, source_kpc, lens_source_kpc, background_msun_kpc2):
    """Periodic thin-plane contrast plus a uniform sheet; distances are proper kpc."""
    from core.foundation import G_MPC_KMS2_MSUN
    surface=np.asarray(surface_msun_kpc2,dtype=float)
    if surface.ndim!=2 or surface.shape[0]!=surface.shape[1] or surface.shape[0]<4 or not np.isfinite(surface).all() or np.any(surface<0):raise ValueError('Finite nonnegative square surface-density grid required')
    values=[pixel_kpc,lens_kpc,source_kpc,lens_source_kpc]
    if any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values) or isinstance(background_msun_kpc2,bool) or not math.isfinite(background_msun_kpc2) or background_msun_kpc2<0:raise ValueError('Positive proper distances/pixel and nonnegative background required')
    sigma_crit=299792.458**2/(4*np.pi*G_MPC_KMS2_MSUN*1000)*source_kpc/(lens_kpc*lens_source_kpc)
    kappa=(surface-background_msun_kpc2)/sigma_crit;contrast=kappa-np.mean(kappa)
    freq=2*np.pi*np.fft.fftfreq(surface.shape[0],d=pixel_kpc/lens_kpc);kx,ky=np.meshgrid(freq,freq);k2=kx*kx+ky*ky
    ft=np.fft.fft2(contrast);potential=np.divide(-2*ft,k2,out=np.zeros_like(ft),where=k2>0)
    shear1=np.fft.ifft2(-.5*(kx*kx-ky*ky)*potential).real;shear2=np.fft.ifft2(-kx*ky*potential).real
    return {'surface_density_msun_kpc2':surface,'kappa':kappa,'shear1':shear1,'shear2':shear2,
            'magnification_inverse':(1-kappa)**2-shear1**2-shear2**2,'critical_surface_density_msun_kpc2':float(sigma_crit)}


def make_lens_map(run_dir,map_bundle,destination,*,map_key,snapshot,lens_kpc,source_kpc,lens_source_kpc,background_msun_kpc2,memory_gib=1.):
    from core.analysis import read_snapshot_header,_snapshot_parameter
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    from core.native import native_run_lock
    root=Path(run_dir).expanduser().resolve();maps=Path(map_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh lens-map output directory')
    if type(snapshot) is not int or snapshot<0:raise ValueError('Exact nonnegative snapshot required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite lens-map memory required')
    with native_run_lock(root,operation='thin lens map'):
        manifest=maps/'map_contract.json';manifest_hash=file_sha256(manifest);contract=json.loads(manifest.read_text())
        if contract.get('schema')!='map-publication-v1':raise ValueError('Published structure-map bundle required')
        verify_product_provenance(root,contract['provenance']);inventory={r['path']:r['sha256'] for r in contract['artifacts']}
        def checked(name):
            if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset')
            path=maps/name
            if not path.is_file() or file_sha256(path)!=inventory[name]:raise ValueError('Map asset missing or changed: '+name)
            return path
        scale=json.loads(checked('scale.json').read_text());settings=scale['maps'].get(map_key)
        if not settings or settings.get('status')!='COMPLETE':raise ValueError('Requested measured mass map is unavailable')
        figure=map_key+'_%03d.figure.json'%snapshot;meta=json.loads(checked(figure).read_text())['scientific_metadata']
        if meta['field'] not in ('dm','gas','stars','total') or meta.get('overlay_layers')!=[]:raise ValueError('A scalar mass-surface-density map is required')
        header,files=read_snapshot_header(root,snapshot);a=float(header['Time']);h=header.get('HubbleParam')
        if h is None:h=_snapshot_parameter(files,'HubbleParam')
        h=float(h)
        if not math.isfinite(a) or a<=0 or not math.isfinite(h) or h<=0:raise ValueError('Valid saved scale factor and Hubble parameter required')
        if not math.isclose(1/a-1,float(meta['redshift']),rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Mass map and snapshot epochs differ')
        with h5py.File(checked(meta['numeric_asset']),'r') as source:
            ds=source['measured']
            if len(ds.shape)!=2 or ds.shape[0]!=ds.shape[1] or np.prod(ds.shape,dtype=np.int64)*160>memory_gib*1024**3*.7:raise MemoryError('Lens grid exceeds allocation or is not square')
            raw=np.asarray(ds[:],dtype=float)
        # Native map unit h Msun / Mpc^2 comoving; convert to proper Msun/kpc^2.
        surface=raw*h/(a*a*1e6);pixel=float(meta['box_size_mpc_h'])/raw.shape[0]*a/h*1000
        result=thin_lens_from_surface(surface,pixel,lens_kpc,source_kpc,lens_source_kpc,background_msun_kpc2)
        fig,axes=plt.subplots(1,3,figsize=(14,5.6),facecolor='#08111e');fig.subplots_adjust(left=.05,right=.97,bottom=.15,top=.79,wspace=.32)
        extent=(-raw.shape[0]*pixel/2,raw.shape[0]*pixel/2,-raw.shape[0]*pixel/2,raw.shape[0]*pixel/2)
        for ax,key,title in zip(axes,('kappa','shear1','shear2'),(r'Convergence $\kappa$',r'Shear $\gamma_1$',r'Shear $\gamma_2$')):
            data=result[key];limit=max(float(np.percentile(np.abs(data),99.5)),1e-12)
            im=ax.imshow(data,origin='lower',extent=extent,cmap='coolwarm',vmin=-limit,vmax=limit,interpolation='nearest');ax.grid(False)
            ax.set_title(title,color='white');ax.set_xlabel('proper kpc',color='#c5d6e8');ax.set_ylabel('proper kpc',color='#c5d6e8');ax.tick_params(colors='#c5d6e8',labelsize=8)
            cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.025);cb.ax.tick_params(colors='#c5d6e8',labelsize=8)
        fig.suptitle('Thin-lens matter map · z=%.4g · %s'%(meta['redshift'],meta['field']),color='white',fontsize=17,y=.96)
        slab=meta['slice'].get('thickness_mpc_h')
        geometry='full-box projection' if slab is None else 'slab %.4g comoving Mpc/h'%slab
        fig.text(.5,.89,geometry+' · supplied distances · periodic contrast + uniform sheet · not a full light cone',ha='center',color='#b8cadc',fontsize=10)
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.lens-map-',dir=out.parent))
        try:
            fig.savefig(stage/'lens_map.png',dpi=180,facecolor=fig.get_facecolor());fig.savefig(stage/'lens_map.pdf',facecolor=fig.get_facecolor())
            np.savez_compressed(stage/'lens_data.npz',**result)
            checked('scale.json');checked(figure);checked(meta['numeric_asset']);verify_product_provenance(root,contract['provenance'])
            if file_sha256(manifest)!=manifest_hash:raise ValueError('Source map contract changed')
            atomic_json(stage/'lens_contract.json',{'schema':'saved-map-thin-lens-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),
                'map_bundle':str(maps),'map_contract_sha256':manifest_hash,'map_key':map_key,'snapshot':snapshot,'redshift':meta['redshift'],
                'algorithm_sha256':file_sha256(__file__),'scale_factor':a,'hubble_parameter':h,'pixel_proper_kpc':pixel,
                'lens_distance_kpc':lens_kpc,'source_distance_kpc':source_kpc,'lens_source_distance_kpc':lens_source_kpc,
                'background_surface_density_msun_kpc2':background_msun_kpc2,'critical_surface_density_msun_kpc2':result['critical_surface_density_msun_kpc2'],
                'boundary':'Periodic density contrast with explicit uniform mass sheet; edge and resolution effects require convergence.',
                'interpretation':'Single thin lens at supplied distances; no light-cone, source population, instrument, or observational-noise model.',
                'artifacts':{name:file_sha256(stage/name) for name in ('lens_map.png','lens_map.pdf','lens_data.npz')}})
            publish_directory(stage,out)
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
        finally:plt.close(fig)
    return out/'lens_map.png'
