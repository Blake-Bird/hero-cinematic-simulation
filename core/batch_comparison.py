"""Compare selected-run halo mass functions without replacing individual analysis."""
from __future__ import annotations
from contextlib import ExitStack
import json
import math
import tempfile
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def compare_selected_hmf(run_a,bundle_a,run_b,bundle_b,destination,*,redshift_tolerance=1e-6):
    from core.analysis import verify_analysis_bundle
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,publish_directory
    roots=[Path(run_a).resolve(),Path(run_b).resolve()];bundles=[Path(bundle_a).resolve(),Path(bundle_b).resolve()];out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Choose a fresh selected-analysis comparison directory')
    with ExitStack() as stack:
        for root in sorted(set(roots)):stack.enter_context(native_run_lock(root,operation='selected HMF comparison'))
        settings=[yaml.safe_load((root/'config.yaml').read_text()).get('analysis',{}) for root in roots]
        definitions=[str(s.get('hmf_mass_definition','fof')).lower() for s in settings]
        if definitions[0]!=definitions[1]:raise ValueError('HMF mass definitions differ between selected runs')
        results=[]
        for root,bundle in zip(roots,bundles):
            record=verify_analysis_bundle(bundle,verify_sources=True)
            if Path(record['source_run']).resolve()!=root:raise ValueError('Analysis belongs to another run')
            summary=json.loads((bundle/'analysis_products_summary.json').read_text());epochs=[]
            for snap,item in summary.get('snapshots',{}).items():
                z=float(item['redshift']);
                if item.get('products',{}).get('hmf',{}).get('status')=='COMPLETE' and math.isfinite(z):
                    epochs.append((int(snap),z))
            results.append({'bundle':bundle,'epochs':epochs,'provenance':file_sha256(bundle/'analysis_provenance.json')})
        pairs=[(abs(a[1]-b[1]),a,b) for a in results[0]['epochs'] for b in results[1]['epochs'] if abs(a[1]-b[1])<=redshift_tolerance*max(1.,abs(a[1]),abs(b[1]))]
        if not pairs:raise ValueError('No saved epochs match within the redshift tolerance')
        _,a,b=min(pairs,key=lambda row:(max(row[1][1],row[2][1]),row[0]));selected=[a,b];curves=[]
        for row,(snap,z) in zip(results,selected):
            with h5py.File(row['bundle']/'analysis_products.h5','r') as h:
                key='snap_%03d/hmf/data'%snap
                if key not in h:raise ValueError('Selected analysis does not contain halo mass functions')
                g=h[key];status=g['status'][()];status=status.decode() if isinstance(status,bytes) else str(status)
                if status!='COMPLETE':raise ValueError('Halo mass function unavailable at matched epoch')
                x=np.asarray(g['mass_msun_h'][:],float);y=np.asarray(g['dn_dlogM'][:],float);count=np.asarray(g['counts'][:])
                if x.ndim!=1 or x.shape!=y.shape or x.shape!=count.shape:raise ValueError('HMF data are misaligned')
                good=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
                if not np.any(good):raise ValueError('No positive measured HMF bins')
                curves.append((x[good],y[good],int(np.sum(count)),str(g['dn_dlogM'].attrs.get('units','not declared'))))
        if curves[0][3]!=curves[1][3]:raise ValueError('HMF units differ')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.selected-hmf-',dir=out.parent))
        try:
            fig,ax=plt.subplots(figsize=(8,6),facecolor='#081321');ax.set_facecolor('#14283e')
            for root,(x,y,count,unit),color in zip(roots,curves,('#77e5ef','#ffbf78')):
                ax.loglog(x,y,marker='o',ms=3,color=color,label=root.name+' (%d binned halos)'%count)
            ax.set(xlabel='Halo mass [Msun/h]',ylabel='dn/dlog10 M ['+curves[0][3]+']');ax.tick_params(colors='#d5e5f2');ax.xaxis.label.set_color('#d5e5f2');ax.yaxis.label.set_color('#d5e5f2');ax.grid(alpha=.15);ax.legend(labelcolor='white',facecolor='#14283e');ax.set_title('Saved %s HMF comparison · z=%.5g / %.5g'%(definitions[0],a[1],b[1]),color='white');fig.tight_layout();fig.savefig(stage/'hmf_comparison.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
            for row in results:
                verify_analysis_bundle(row['bundle'],verify_sources=True)
                if file_sha256(row['bundle']/'analysis_provenance.json')!=row['provenance']:raise ValueError('Source analysis changed')
            atomic_json(stage/'comparison.json',{'schema':'selected-hmf-comparison-v1','science_acceptance':'NOT_ACCEPTED','runs':[str(r) for r in roots],
                'analysis_bundles':[str(r['bundle']) for r in results],'analysis_provenance_sha256':[r['provenance'] for r in results],
                'snapshots':[a[0],b[0]],'redshifts':[a[1],b[1]],'hmf_units':curves[0][3],'mass_definition':definitions[0],
                'scope':'Overlaid saved HMFs; binning, resolution cuts and mass definitions follow each source analysis. No matched phases or causal inference implied.',
                'figure_sha256':file_sha256(stage/'hmf_comparison.png')})
            publish_directory(stage,out);return out/'hmf_comparison.png'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise


def compare_selected_curves(run_a,bundle_a,run_b,bundle_b,destination,*,product,redshift_tolerance=1e-6):
    """Overlay like measured spectra or halo correlations at a matching saved epoch."""
    from core.analysis import verify_analysis_bundle
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,publish_directory
    specs={'pk':('k_h_Mpc','P_Mpc_h3','k [h/Mpc]','P(k) [(Mpc/h)³]',True),
           'correlation':('r','xi','r [comoving Mpc/h]','Halo correlation ξ(r)',False)}
    if product not in specs:raise ValueError('Unsupported selected comparison')
    roots=[Path(run_a).resolve(),Path(run_b).resolve()];bundles=[Path(bundle_a).resolve(),Path(bundle_b).resolve()];out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Choose a fresh comparison directory')
    with ExitStack() as locks:
        for root in sorted(set(roots)):locks.enter_context(native_run_lock(root,operation='selected curve comparison'))
        epochs=[];hashes=[]
        for root,bundle in zip(roots,bundles):
            record=verify_analysis_bundle(bundle,verify_sources=True)
            if Path(record['source_run']).resolve()!=root:raise ValueError('Analysis belongs to another run')
            summary=json.loads((bundle/'analysis_products_summary.json').read_text());rows=[]
            for snap,item in summary.get('snapshots',{}).items():
                status=item.get('products',{}).get(product,{}).get('status');z=float(item['redshift'])
                if status=='COMPLETE' and math.isfinite(z):rows.append((int(snap),z))
            epochs.append(rows);hashes.append(file_sha256(bundle/'analysis_provenance.json'))
        pairs=[(max(a[1],b[1]),abs(a[1]-b[1]),a,b) for a in epochs[0] for b in epochs[1] if abs(a[1]-b[1])<=redshift_tolerance*max(1.,abs(a[1]),abs(b[1]))]
        if not pairs:raise ValueError('No matched epochs with both '+product+' products')
        _,_,a,b=min(pairs);xname,yname,xlabel,ylabel,positive=specs[product];curves=[];units=[]
        for bundle,(snap,z) in zip(bundles,(a,b)):
            with h5py.File(bundle/'analysis_products.h5','r') as h:
                key='snap_%03d/%s/data'%(snap,product)
                if key not in h or any(name not in h[key] for name in (xname,yname)):raise ValueError('Measured '+product+' arrays unavailable')
                g=h[key];x=np.asarray(g[xname][:],float);y=np.asarray(g[yname][:],float)
                if x.ndim!=1 or x.shape!=y.shape:raise ValueError('Measured curve arrays are misaligned')
                mask=np.isfinite(x)&np.isfinite(y)&(x>0)
                if positive:mask &= y>0
                if not np.any(mask):raise ValueError('No displayable measured curve bins')
                units.append((str(g[xname].attrs.get('units','not declared')),str(g[yname].attrs.get('units','not declared'))));curves.append((x[mask],y[mask]))
        if units[0]!=units[1]:raise ValueError('Selected curve units differ')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.curve-comparison-',dir=out.parent))
        try:
            fig,ax=plt.subplots(figsize=(8,6),facecolor='#081321');ax.set_facecolor('#14283e')
            for root,(x,y),color in zip(roots,curves,('#77e5ef','#ffbf78')):
                (ax.loglog if positive else ax.semilogx)(x,y,marker='o',ms=3,color=color,label=root.name)
            if not positive:ax.axhline(0,color='#8ca4b8',lw=.8)
            ax.set(xlabel=xlabel,ylabel=ylabel);ax.tick_params(colors='#d5e5f2');ax.xaxis.label.set_color('#d5e5f2');ax.yaxis.label.set_color('#d5e5f2');ax.grid(alpha=.15);ax.legend(labelcolor='white',facecolor='#14283e');ax.set_title(product.upper()+' comparison · z=%.5g / %.5g'%(a[1],b[1]),color='white');fig.tight_layout();fig.savefig(stage/'comparison.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
            for bundle,expected in zip(bundles,hashes):
                verify_analysis_bundle(bundle,verify_sources=True)
                if file_sha256(bundle/'analysis_provenance.json')!=expected:raise ValueError('Source analysis changed')
            atomic_json(stage/'comparison.json',{'schema':'selected-curve-comparison-v1','science_acceptance':'NOT_ACCEPTED','product':product,'runs':[str(r) for r in roots],
                'analysis_bundles':[str(b) for b in bundles],'analysis_provenance_sha256':hashes,'snapshots':[a[0],b[0]],'redshifts':[a[1],b[1]],'units':units[0],
                'scope':'Saved curves shown independently; differing resolution, selection, binning, physics and initial phases may affect the result. No causal conclusion implied.',
                'figure_sha256':file_sha256(stage/'comparison.png')})
            publish_directory(stage,out);return out/'comparison.png'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
