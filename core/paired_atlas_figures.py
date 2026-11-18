"""Readable paired figures and exact tables from physical-unit epoch caches."""
from pathlib import Path
import json
import csv
import math
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2
from PIL import Image
from core.foundation import atomic_json,atomic_text,file_sha256
from core.paired_atlas import PRIORITY,epoch_name,_write_csv

COLORS={'EDE':'#62e6da','LCDM':'#f6bb70'}


def save(fig,path,arrays=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.set_facecolor('#081521')
    for ax in fig.axes:
        ax.set_facecolor('#102639');ax.tick_params(colors='#c0d5e6');ax.xaxis.label.set_color('#c0d5e6');ax.yaxis.label.set_color('#c0d5e6');ax.title.set_color('#edf7ff')
        ax.grid(alpha=.13,color='#7a98b0')
        for spine in ax.spines.values():spine.set_color('#365369')
        legend=ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#102639');legend.get_frame().set_edgecolor('#365369')
            for t in legend.get_texts():t.set_color('#d3e3ef')
    for text in fig.findobj(matplotlib.text.Text):text.set_fontfamily('DejaVu Sans');text.set_color('#d5e5f1')
    for suffix in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+suffix),dpi=180,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    if arrays is not None:np.savez_compressed(path.with_suffix('.npz'),**arrays)


def _hist(ax,values,edges,volume,name):
    count=np.histogram(values,edges)[0];width=np.diff(np.log10(edges));center=np.sqrt(edges[1:]*edges[:-1]);positive=count>0
    y=count/volume/width
    low=np.zeros_like(y);low[positive]=.5*chi2.ppf(.158655,2*count[positive]);high=.5*chi2.ppf(.841345,2*(count+1))
    if positive.any():ax.errorbar(center[positive],y[positive],yerr=np.array([count-low,high-count])[:,positive]/volume/width[positive],fmt='o-',ms=3,lw=1.6,color=COLORS[name],label=name)
    if (~positive).any():ax.scatter(center[~positive],high[~positive]/volume/width[~positive],s=15,marker='v',color=COLORS[name],alpha=.65,label=(name+' · no selected halos') if not positive.any() else None)
    return center,count,y


def _caption(fig,text):
    fig.text(.12,.015,text,fontsize=8,color='#a9c3d8');fig.tight_layout(rect=(0,.065,1,1))


def _render_binding(records,paths):
    encoded=json.dumps(records,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    return {'schema':'paired-core-render-binding-v1',
        'records_sha256':hashlib.sha256(encoded).hexdigest(),
        'inputs':{name:{'path':str(Path(paths[name]).resolve()),'sha256':file_sha256(paths[name])} for name in ('EDE','LCDM')},
        'source_sha256':{str(path):file_sha256(path) for path in
            (Path(__file__).resolve(),Path(__file__).with_name('paired_atlas.py').resolve())}}


def _verified_core_render(records,paths,out):
    receipt=out/'core_render_receipt.json';seal=out/'core_render_receipt.json.sha256'
    try:
        if not receipt.is_file() or not seal.is_file() or receipt.is_symlink() or seal.is_symlink():
            raise ValueError('Missing sealed core render receipt; use a fresh comparison directory')
        if seal.read_text().strip()!=file_sha256(receipt):raise ValueError('Changed core render receipt')
        saved=json.loads(receipt.read_text())
        if saved['schema']!='paired-core-render-receipt-v1' or saved['binding']!=_render_binding(records,paths):
            raise ValueError('Changed paired figure inputs or renderer; use a fresh comparison directory')
        manifest=json.loads((out/'comparison_manifest.json').read_text());original=saved['manifest']
        for key,value in original.items():
            # Separate plotting scripts legitimately append their own gallery entries.
            # Those additions require their own provenance checks, outside this receipt.
            if key in ('figures','tables'):
                if not isinstance(manifest.get(key),list) or not set(value)<=set(manifest[key]):
                    raise ValueError('Changed core comparison entries')
            elif manifest.get(key)!=value:raise ValueError('Changed core comparison metadata')
        for name,sha in saved['artifacts_sha256'].items():
            path=out/name
            if Path(name).name!=name or path.is_symlink() or not path.is_file() or file_sha256(path)!=sha:
                raise ValueError('Changed or missing core comparison export: '+name)
        return manifest
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as error:
        raise ValueError('Incomplete or invalid core comparison publication') from error


def render_pair(records, paths, destination):
    out=Path(destination)
    if out.exists() and any(out.iterdir()):raise ValueError('Existing comparison output; preserve it and choose a fresh directory')
    binding=_render_binding(records,paths)
    out.mkdir(parents=True,exist_ok=True)
    data={n:np.load(paths[n],allow_pickle=False) for n in ('EDE','LCDM')};summaries={n:records[n]['summary'] for n in data}
    z=summaries['EDE']['redshift'];snap=summaries['EDE']['snapshot'];box=summaries['EDE']['box_comoving_mpc'];volume=box**3
    if not math.isclose(box,summaries['LCDM']['box_comoving_mpc'],rel_tol=1e-10):raise ValueError('Common physical box required')
    if not math.isclose(z,summaries['LCDM']['redshift'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Epoch mismatch')
    label=epoch_name('EDE_vs_LCDM',snap,z);figures=[];tables=[]
    floor=max(s['particle_mass_mean_msun']*300 for s in summaries.values())
    edges=np.geomspace(floor,max(5e13,floor*100),29)
    for variant,key,xlabel,selection in (
        ('FOF_halo_mass_function','fof_mass_msun','FOF halo mass [Msun]','common mass floor and at least 300 native FOF particles'),
        ('Subhalo_mass_function','subhalo_mass_msun','SUBFIND mass [Msun]','saved SUBFIND rows; common mass floor shown, no bound-particle convergence claim')):
        fig,(ax,ratio)=plt.subplots(2,1,figsize=(10,9),sharex=True,gridspec_kw={'height_ratios':[2,1]});rates={};counts={}
        for n,d in data.items():
            values=d[key];keep=np.isfinite(values)&(values>=floor)
            if key=='fof_mass_msun':keep&=d['fof_particle_count']>=300
            x,c,y=_hist(ax,values[keep],edges,volume,n);rates[n]=y;counts[n]=c
        use=counts['LCDM']>0;r=np.full_like(x,np.nan);r[use]=rates['EDE'][use]/rates['LCDM'][use]
        ratio.semilogx(x[use],r[use],'o-',ms=3,color='#bba9ef');ratio.axhline(1,ls=':',color='#d9e7f2')
        if not use.any():
            ratio.set_ylim(0,2);ratio.text(.5,.65,'Ratio unavailable: no resolved denominator bins',ha='center',transform=ratio.transAxes,color='#bfcedd')
        if not any(np.any(c>0) for c in counts.values()):
            upper=.5*chi2.ppf(.841345,2)/volume/np.diff(np.log10(edges))[0]
            ax.set_ylim(upper/10,upper*10);ax.text(.5,.5,'No halos above the shared selection floor\nOnly count upper limits can be displayed',ha='center',transform=ax.transAxes,color='#d5e5f1')
        ax.axvline(floor,color='#c1d3e2',ls='--',lw=1,label='Common 300-particle mass floor')
        ax.set(xscale='log',yscale='log',ylabel='dn/dlog10 M [Mpc^-3]',title=f'{variant.replace("_"," ")} · z={z:.3f}');ax.legend(fontsize=9)
        ratio.set(xlabel=xlabel,ylabel='EDE / LCDM')
        _caption(fig,'Same comoving volume and physical mass bins. '+selection+'.\nCount bars: 68% Poisson intervals; triangles: empty-bin upper limits. No sample-variance or ratio-covariance claim.')
        base=out/(label+'_'+variant);save(fig,base,{'mass_edges_msun':edges,**{n+'_counts':c for n,c in counts.items()},'ratio_EDE_over_LCDM':r});figures.append(base.with_suffix('.png').name)
        rows=[{'mass_low_msun':float(edges[i]),'mass_high_msun':float(edges[i+1]),'mass_center_msun':float(x[i]),'common_mass_floor_msun':floor,
               'EDE_count':int(counts['EDE'][i]),'LCDM_count':int(counts['LCDM'][i]),'EDE_dn_dlog10M_mpc_inverse3':float(rates['EDE'][i]),'LCDM_dn_dlog10M_mpc_inverse3':float(rates['LCDM'][i]),
               'EDE_over_LCDM':float(r[i]) if use[i] else 'UNDEFINED_ZERO_DENOMINATOR'} for i in range(len(x))]
        _write_csv(base.with_suffix('.csv'),rows);tables.append(base.with_suffix('.csv').name)
    fig,ax=plt.subplots(figsize=(10,6.5));threshold=np.geomspace(edges[0],edges[-1],160);rows=[];any_cumulative=False
    for n,d in data.items():
        m=np.sort(d['fof_mass_msun'][d['fof_particle_count']>=300]);count=len(m)-np.searchsorted(m,threshold,side='left');keep=count>0
        ax.loglog(threshold[keep],count[keep]/volume,label=n,color=COLORS[n]);any_cumulative|=bool(keep.any())
        for t,c in zip(threshold,count):rows.append({'run':n,'mass_threshold_msun':float(t),'count':int(c),'number_density_mpc_inverse3':float(c/volume)})
    if not any_cumulative:
        ax.axhline(-np.log(.05)/volume,color='#d4d6ed',ls=':',label='Zero-count 95% Poisson upper limit');ax.text(.5,.5,'No resolved halos in either run',ha='center',transform=ax.transAxes,color='#d5e5f1')
    ax.axvline(floor,color='#a9bed0',ls='--');ax.set(xlabel='FOF mass threshold [Msun]',ylabel='n(>=M) [Mpc^-3]',title=f'Cumulative host abundance · z={z:.3f}');ax.legend();_caption(fig,'At least 300 native FOF particles; vertical line is the stricter shared physical mass floor.')
    base=out/(label+'_Cumulative_halo_abundance');save(fig,base);_write_csv(base.with_suffix('.csv'),rows);figures.append(base.with_suffix('.png').name);tables.append(base.with_suffix('.csv').name)
    if all('k_mpc_inverse' in d for d in data.values()):
        fig,(ax,ratio)=plt.subplots(2,1,figsize=(10,9),sharex=True,gridspec_kw={'height_ratios':[2,1]});curves={};rows=[]
        for n,d in data.items():
            k=d['k_mpc_inverse'];p=d['power_mpc3'];e=d['power_sigma_gaussian_mpc3'];valid=np.isfinite(k)&np.isfinite(p)&(p>0)
            ax.loglog(k[valid],p[valid],color=COLORS[n],label=n,lw=1.5);curves[n]=(k[valid],p[valid])
            for i in range(len(k)):rows.append({'run':n,'k_mpc_inverse':float(k[i]),'power_mpc3':float(p[i]),'native_gaussian_sigma_mpc3':float(e[i]),'mode_count':int(d['power_mode_count'][i]),'shot_noise_mpc3':float(d['power_shot_mpc3'][i])})
        lo=max(v[0][0] for v in curves.values());hi=min(v[0][-1] for v in curves.values());k=np.geomspace(lo,hi,240)
        p={n:np.exp(np.interp(np.log(k),np.log(v[0]),np.log(v[1]))) for n,v in curves.items()};rr=p['EDE']/p['LCDM'];ratio.semilogx(k,rr,color='#bba9ef');ratio.axhline(1,ls=':',color='#d7e5f0')
        ax.set(ylabel='P(k) [Mpc^3]',title=f'Matter power · z={z:.3f}');ax.legend();ratio.set(xlabel='k [Mpc^-1]',ylabel='EDE / LCDM')
        _caption(fig,'Native shot-noise-subtracted folded spectra in common physical units. Ratio uses log interpolation only within overlapping positive bins.\nNo precision joint covariance or numerical-convergence acceptance is implied; signed bins and native errors remain in the table.')
        base=out/(label+'_Matter_power_spectrum_and_ratio');save(fig,base,{'common_k_mpc_inverse':k,'ratio_EDE_over_LCDM':rr});_write_csv(base.with_suffix('.csv'),rows);figures.append(base.with_suffix('.png').name);tables.append(base.with_suffix('.csv').name)
    for variant,key,xlabel in (('Subhalo_Vmax_distribution','subhalo_vmax_km_s','Subhalo Vmax [km/s]'),):
        values={n:d[key][np.isfinite(d[key])&(d[key]>0)] for n,d in data.items()}
        allv=np.concatenate(list(values.values()))
        if not len(allv):continue
        ve=np.geomspace(max(.01,allv.min()),allv.max()*1.01,30);fig,ax=plt.subplots(figsize=(10,6.5));rows=[]
        for n,v in values.items():
            c=np.histogram(v,ve)[0];x=np.sqrt(ve[1:]*ve[:-1]);ax.step(x,c,where='mid',color=COLORS[n],label=n)
            rows.extend({'run':n,'vmax_low_km_s':float(ve[i]),'vmax_high_km_s':float(ve[i+1]),'count':int(c[i])} for i in range(len(c)))
        ax.set(xscale='log',yscale='log',xlabel=xlabel,ylabel='Native subhalo count',title=f'Saved subhalo Vmax function · z={z:.3f}');ax.legend();_caption(fig,'All native SUBFIND rows with positive finite Vmax. This population is not a completeness-corrected galaxy sample.')
        base=out/(label+'_'+variant);save(fig,base);_write_csv(base.with_suffix('.csv'),rows);figures.append(base.with_suffix('.png').name);tables.append(base.with_suffix('.csv').name)
    fig,ax=plt.subplots(figsize=(10,6.5));rows=[]
    for n,d in data.items():
        e=d['velocity_histogram_edges_km_s'];c=d['velocity_histogram_counts'];x=(e[1:]+e[:-1])/2;y=c/c.sum()/np.diff(e);ax.plot(x,y,color=COLORS[n],label=n)
        rows.extend({'run':n,'speed_low_km_s':float(e[i]),'speed_high_km_s':float(e[i+1]),'particle_count':int(c[i]),'probability_density_per_km_s':float(y[i])} for i in range(len(c)))
    ax.set(yscale='log',xlabel='Peculiar speed [km/s]',ylabel='Probability density [(km/s)^-1]',title=f'Dark-matter speed distributions · z={z:.3f}');ax.legend();_caption(fig,'Full saved type-1 velocities converted using the native peculiar-velocity convention. Histograms retain each run\'s exact bin edges in CSV.')
    base=out/(label+'_Particle_velocity_distributions');save(fig,base);_write_csv(base.with_suffix('.csv'),rows);figures.append(base.with_suffix('.png').name);tables.append(base.with_suffix('.csv').name)
    for projection in ('xy','xz','yz'):
        fields={n:gaussian_filter(d['dm_surface_density_msun_mpc2_'+projection],2.,mode='wrap') for n,d in data.items()}
        valid=np.concatenate([np.log10(v[v>0]) for v in fields.values()]);low,high=np.percentile(valid,[1,99.7]);fig,axes=plt.subplots(1,3,figsize=(16,5.9));images=[]
        for ax,n in zip(axes[:2],('EDE','LCDM')):
            v=fields[n];im=ax.imshow(np.log10(np.maximum(v,10**low)),origin='lower',extent=(0,box,0,box),cmap='magma',vmin=low,vmax=high);ax.grid(False);ax.set_title(n);ax.set_xlabel(projection[0]+' [comoving Mpc]');ax.set_ylabel(projection[1]+' [comoving Mpc]');fig.colorbar(im,ax=ax,label='log10 surface density [Msun/Mpc^2]',fraction=.045)
        rr=np.log10(np.divide(fields['EDE'],fields['LCDM'],out=np.full_like(fields['EDE'],np.nan),where=fields['LCDM']>0));limit=max(.05,float(np.nanpercentile(np.abs(rr),99)))
        im=axes[2].imshow(rr,origin='lower',extent=(0,box,0,box),cmap='coolwarm',vmin=-limit,vmax=limit);axes[2].grid(False);axes[2].set(title='log10(EDE / LCDM)',xlabel=projection[0]+' [comoving Mpc]',ylabel=projection[1]+' [comoving Mpc]');fig.colorbar(im,ax=axes[2],fraction=.045)
        fig.suptitle(f'Equal physical volume · {projection.upper()} · z={z:.3f}',color='#eef7ff',fontsize=17)
        _caption(fig,'Full type-1 mass projected into the same physical comoving grid. Periodic Gaussian display smoothing: 2 grid cells. Shared density scale; signed ratio is not a significance map.')
        base=out/(label+'_Density_'+projection.upper()+'_and_log_ratio');save(fig,base,{'EDE_surface_density_msun_mpc2':fields['EDE'],'LCDM_surface_density_msun_mpc2':fields['LCDM'],'log10_EDE_over_LCDM':rr,'box_comoving_mpc':box});figures.append(base.with_suffix('.png').name)
    for d in data.values():d.close()
    record={'schema':'physical-paired-figures-v1','snapshot':snap,'redshift':z,'physical_box_mpc':box,'common_300_particle_mass_floor_msun':floor,
        'inputs':binding['inputs'],'figures':figures,'tables':tables,'algorithm_sha256':file_sha256(__file__),
        'scope':'One EDE and one LCDM realization with different complete parameter sets. Physical units and common selections; no independent-realization covariance or EDE-only causal attribution.'}
    if _render_binding(records,paths)!=binding:raise ValueError('Paired figure source changed during rendering')
    artifacts={path.name:file_sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    atomic_json(out/'comparison_manifest.json',record)
    receipt=out/'core_render_receipt.json'
    atomic_json(receipt,{'schema':'paired-core-render-receipt-v1','binding':binding,
        'manifest':record,'artifacts_sha256':artifacts,
        'scope':'Original render_pair outputs only. Added gallery products require their own provenance checks.'})
    atomic_text(receipt.with_name(receipt.name+'.sha256'),file_sha256(receipt)+'\n')
    return record


def render_available(base):
    base=Path(base).resolve();directories={n:base/'06_Tables_and_Data/Measurements'/n for n in ('EDE','LCDM')};records={}
    for n,d in directories.items():
        records[n]={j['summary']['snapshot']:j for p in d.glob(n+'_z_*.json') for j in [json.loads(p.read_text())]}
    made=[]
    for s in sorted(set(records['EDE'])&set(records['LCDM'])):
        r={n:records[n][s] for n in records};z=r['EDE']['summary']['redshift']
        if not any(abs(z-target)<2e-5 for target in PRIORITY):continue
        out=base/'01_Comparisons'/epoch_name('EDE_vs_LCDM',s,z)
        paths={n:directories[n]/r[n]['numeric_asset'] for n in r}
        if (out/'comparison_manifest.json').exists():made.append(_verified_core_render(r,paths,out));continue
        made.append(render_pair(r,paths,out));print('PAIRED_FIGURES',s,z,flush=True)
    atomic_json(base/'07_Status_and_Provenance/paired_figure_progress.json',{'completed_priority_epochs':len(made),'required_priority_epochs':len(PRIORITY),'comparisons':made})
    return made
