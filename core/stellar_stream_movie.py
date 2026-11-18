"""Animate saved exact-ID stellar candidate histories without inferring origin."""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path
import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_stellar_stream_movie(run_dir,history_bundle,destination,*,center=None,width_kpc=None,max_particles=50000,
                              fps=24,seconds_per_snapshot=1,memory_gib=1.):
    from core.data import verify_hdf5_dataset_payloads,particle_id_hash64,exact_uint64_ids
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();history=Path(history_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh stellar movie directory')
    if type(max_particles) is not int or not 1<=max_particles<=1000000 or type(fps) is not int or not 1<=fps<=120 or type(seconds_per_snapshot) is not int or not 1<=seconds_per_snapshot<=60:raise ValueError('Bounded particle and timing controls required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<.25:raise ValueError('At least 0.25 GiB declared memory required')
    if max_particles*192+64*1024**2>memory_gib*1024**3*.75:raise MemoryError('Stellar display sample exceeds declared memory')
    with native_run_lock(root,operation='stellar candidate movie'):
        provenance_path=history/'provenance.json';provenance_hash=file_sha256(provenance_path)
        if (history/'provenance.sha256').read_text().strip()!=provenance_hash:raise ValueError('Stellar history provenance seal changed')
        provenance=json.loads(provenance_path.read_text());source=history/'history.hdf5'
        if provenance.get('schema')!='darkmattersim-stellar-history-v1' or file_sha256(source)!=provenance['history_sha256']:raise ValueError('Stellar history is missing or changed')
        verify_product_provenance(root,provenance['data_contract']);verify_hdf5_dataset_payloads(source,require_contract=False)
        with h5py.File(source,'r') as h:
            if h['ParticleIDs'].size*64+64*1024**2>memory_gib*1024**3*.75:raise MemoryError('Stellar history identity selection exceeds allocation')
            ids=exact_uint64_ids(h['ParticleIDs'][:]);labels=exact_uint64_ids(h['CandidateID'][:],unique=False)
            if ids.ndim!=1 or labels.shape!=ids.shape or len(ids)==0 or len(np.unique(ids))!=len(ids):raise ValueError('Stellar history needs distinct aligned exact identities')
            selected=np.sort(np.lexsort((ids,particle_id_hash64(ids)))[:max_particles])
            epochs=[];box=None
            for name,group in h['Epochs'].items():
                meta=json.loads(group.attrs['metadata_json']);a=float(meta['scale_factor']);b=float(meta['box_comoving_kpc'])
                if not math.isfinite(a) or a<=0 or not math.isfinite(b) or b<=0 or (box is not None and not math.isclose(box,b,rel_tol=1e-10)):raise ValueError('Saved stellar history epoch/box is invalid')
                box=b;epochs.append({'group':name,'snapshot':int(meta['snapshot']),'scale_factor':a,'redshift':1/a-1})
            epochs.sort(key=lambda x:x['scale_factor'])
            if len(epochs)<2 or any(epochs[i]['scale_factor']>=epochs[i+1]['scale_factor'] for i in range(len(epochs)-1)):raise ValueError('At least two distinct saved stellar epochs required')
            origin=np.full(3,box/2) if center is None else np.asarray(center,dtype=float)
            if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Center must be three finite comoving kpc coordinates')
            origin=np.mod(origin,box);width=box if width_kpc is None else float(width_kpc)
            if not math.isfinite(width) or not 0<width<=box:raise ValueError('Display width must be positive and at most the box')
            out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.stellar-movie-',dir=out.parent));movie=stage/'stellar_candidate_evolution.mp4'
            try:
                with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
                    for epoch in epochs:
                        g=h['Epochs'][epoch['group']];present=np.asarray(g['Present'][selected],dtype=bool);positions=np.asarray(g['Coordinates'][selected],dtype=float)
                        membership=np.asarray(g['BoundSubhaloRow'][selected]);
                        if positions.shape!=(len(selected),3) or membership.shape!=(len(selected),) or not np.isfinite(positions[present]).all():raise ValueError('Stellar saved positions or memberships are malformed')
                        delta=(positions-origin+box/2)%box-box/2
                        if np.any(present & ~np.isin(membership,[-3,-1]) & (membership<0)):
                            raise ValueError('Unexpected membership sentinel for present particle')
                        frame_name='snapshot_%03d.npz'%epoch['snapshot']
                        np.savez_compressed(stage/frame_name,particle_ids=ids[selected],candidate_ids=labels[selected],present=present,
                                            positions_comoving_kpc=positions,bound_subhalo_row=membership)
                        epoch['present_count']=int(np.count_nonzero(present));epoch['numeric_asset']=frame_name;epoch['numeric_sha256']=file_sha256(stage/frame_name)
                        fig,axes=plt.subplots(1,3,figsize=(14.4,5.44),dpi=100,facecolor='#08111e');fig.subplots_adjust(left=.05,right=.98,bottom=.21,top=.8,wspace=.25)
                        categories=[(membership>=0,'#70e3f2','Bound to a saved subhalo'),(membership==-1,'#ffcb79','Outside all saved subhalos'),(membership==-3,'#c3a9ff','Subhalo catalogue unavailable')]
                        marker_size=float(np.clip(2500/max(1,np.count_nonzero(present)),2,24))
                        for ax,(i,j) in zip(axes,((0,1),(0,2),(1,2))):
                            ax.set_facecolor('#11243a');ax.grid(False)
                            for mask,color,label in categories:
                                mask=present&mask;ax.scatter(delta[mask,i],delta[mask,j],s=marker_size,c=color,alpha=.75,linewidths=0,label=label,rasterized=True)
                            ax.set_xlim(-width/2,width/2);ax.set_ylim(-width/2,width/2);ax.set_aspect('equal')
                            ax.set_xlabel('xyz'[i]+' offset [comoving kpc]',color='#c5d6e8',fontsize=9);ax.set_ylabel('xyz'[j]+' offset [comoving kpc]',color='#c5d6e8',fontsize=9);ax.tick_params(colors='#c5d6e8',labelsize=8)
                        handles,legend_labels=axes[0].get_legend_handles_labels()
                        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.035),fontsize=8,ncol=3,
                                   facecolor='#14253a',edgecolor='#7890a6',labelcolor='#e4f0fa')
                        fig.suptitle('Stellar candidate material · z=%.4g · snapshot %d · %d/%d present'%(epoch['redshift'],epoch['snapshot'],epoch['present_count'],len(selected)),color='white',fontsize=15,y=.95)
                        fig.text(.5,.88,'Exact selected IDs · fixed periodic viewing center · saved epochs only',color='#b8cadc',ha='center',fontsize=10)
                        fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                        for _ in range(fps*seconds_per_snapshot):writer.append_data(frame)
                        plt.close(fig)
                verify_product_provenance(root,provenance['data_contract'])
                if file_sha256(source)!=provenance['history_sha256'] or file_sha256(provenance_path)!=provenance_hash:raise ValueError('Stellar history changed during export')
                atomic_json(stage/'stellar_movie_contract.json',{'schema':'stellar-candidate-movie-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),
                    'history_bundle':str(history),'history_sha256':provenance['history_sha256'],'history_provenance_sha256':provenance_hash,
                    'algorithm_sha256':file_sha256(__file__),'center_comoving_kpc':origin.tolist(),'width_comoving_kpc':width,
                    'selection':'Lowest deterministic hash of saved candidate particle IDs, fixed across epochs.',
                    'total_candidate_particles':len(ids),'displayed_particle_count':len(selected),'epochs':epochs,'fps':fps,'seconds_per_snapshot':seconds_per_snapshot,
                    'movie_sha256':file_sha256(movie),'timing_scope':'Equal display time per saved epoch; no interpolated state.',
                    'interpretation':'Exact candidate material, not automatically a tidal stream; outside SUBFIND does not mean unbound to the host.'})
                publish_directory(stage,out)
            except BaseException as error:
                atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
    return out/'stellar_candidate_evolution.mp4'
