"""Trace exact native host members through saved cosmological snapshots."""
from __future__ import annotations
import math
import tempfile
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def stable_cohort(ids, maximum):
    from core.data import exact_uint64_ids, particle_id_hash64
    ids=exact_uint64_ids(ids,allow_empty=False)
    if type(maximum) is not int or maximum<1 or len(np.unique(ids))!=len(ids):raise ValueError('Unique IDs and a positive display limit are required')
    order=np.lexsort((ids,particle_id_hash64(ids)))[:maximum]
    return np.sort(ids[order])


def matching_positions(files, selected, *, chunk_rows=200000):
    """Return exact matching IDs/coordinates in ID order, rejecting duplicates."""
    from core.data import exact_uint64_ids
    ids=[];coords=[]
    for path in files:
        with h5py.File(path,'r') as h:
            if 'PartType1' not in h:continue
            g=h['PartType1'];identity=g['ParticleIDs'];xyz=g['Coordinates']
            if identity.ndim!=1 or xyz.shape!=(len(identity),3):raise ValueError('DM identity and position fields are not aligned')
            for start in range(0,len(identity),chunk_rows):
                raw=exact_uint64_ids(identity[start:start+chunk_rows],unique=False)
                index=np.searchsorted(selected,raw);valid=index<len(selected)
                valid[valid] &= selected[index[valid]]==raw[valid]
                rows=np.flatnonzero(valid)
                if len(rows):ids.append(raw[rows]);coords.append(np.asarray(xyz[start+rows],dtype=float))
    if not ids:return np.empty(0,dtype=np.uint64),np.empty((0,3),dtype=float)
    ids=np.concatenate(ids);coords=np.concatenate(coords)
    if len(np.unique(ids))!=len(ids) or not np.isfinite(coords).all():raise ValueError('Duplicate matched IDs or invalid positions')
    order=np.argsort(ids);return ids[order],coords[order]


def make_particle_origin_movie(run_dir,destination,*,anchor_snapshot,host_row,max_particles=50000,fps=24,seconds_per_snapshot=1,memory_gib=1.):
    from core.zoom import read_host_catalogue,_run_lineage,_check_lineage
    from core.data import snapshot_id_shards,read_selected_group_ids,validate_native_shard_counts
    from core.analysis import snapshot_ids,read_snapshot_header
    from halos import _unit_conversion
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,publish_directory
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh particle-origin output directory')
    if type(anchor_snapshot) is not int or anchor_snapshot<0 or type(host_row) is not int or host_row<0:raise ValueError('Exact saved anchor snapshot and host row required')
    if type(max_particles) is not int or not 1<=max_particles<=1000000 or type(fps) is not int or not 1<=fps<=120 or type(seconds_per_snapshot) is not int or not 1<=seconds_per_snapshot<=60:raise ValueError('Bounded particle and movie controls required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<.25:raise ValueError('At least 0.25 GiB declared movie memory required')
    with native_run_lock(root,operation='particle origin movie'):
        lineage=_run_lineage(root);anchor=read_host_catalogue(root,anchor_snapshot,memory_gib=memory_gib,host_row=host_row)
        ancestry=_check_lineage(anchor,lineage);count=int(anchor['lengths'][host_row,1])
        if count<1:raise ValueError('Anchor host has no saved type-1 dark-matter members')
        if count*128+64*1024**2>memory_gib*1024**3*.8:raise MemoryError('Exact host membership exceeds declared workspace')
        shards,_=snapshot_id_shards(anchor['snapshots'],1)
        ids,_=read_selected_group_ids(anchor['offsets'],anchor['lengths'],np.array([host_row]),shards,particle_type=1,max_ids=count)
        cohort=stable_cohort(ids,max_particles)
        snapshots=snapshot_ids(root)
        if anchor_snapshot not in snapshots:raise ValueError('Anchor snapshot missing from original output series')
        sources={x['path']:x['sha256'] for x in anchor['sources']+anchor['snapshot_sources']+ancestry+lineage['initial_conditions']}
        sources[lineage['path']]=lineage['sha256'];epochs=[];anchor_box=anchor['frame'][1]
        for snap in snapshots:
            header,files=read_snapshot_header(root,snap);validate_native_shard_counts(files,'NumPart_Total');_unit_conversion(files)
            a=float(header['Time']);box=float(header['BoxSize'])
            if not math.isfinite(a) or a<=0 or not math.isclose(box,anchor_box,rel_tol=1e-10):raise ValueError('Snapshot epoch or box differs from anchor')
            for path in files:
                path=Path(path).resolve();sources[str(path)]=file_sha256(path)
                with h5py.File(path) as h:
                    prefix=h['Parameters'].attrs.get('InitCondFile');prefix=prefix.decode() if isinstance(prefix,bytes) else prefix
                    if prefix!=lineage['prefix']:raise ValueError('Snapshot does not identify the anchor native IC namespace')
                    attrs=h['Header'].attrs
                    if (not math.isclose(float(attrs['Time']),a,rel_tol=1e-12)
                        or not math.isclose(float(attrs['BoxSize']),anchor_box,rel_tol=1e-10)
                        or not math.isclose(float(h['Parameters'].attrs['HubbleParam']),lineage['h'],rel_tol=1e-12)):
                        raise ValueError('Snapshot shards disagree on epoch, box or IC Hubble parameter')
            epochs.append({'snapshot':int(snap),'scale_factor':a,'redshift':1/a-1,'files':[str(Path(p).resolve()) for p in files]})
        epochs.sort(key=lambda e:e['scale_factor'])
        if len(epochs)<2 or any(epochs[i]['scale_factor']>=epochs[i+1]['scale_factor'] for i in range(len(epochs)-1)):raise ValueError('At least two distinct saved epochs required')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.particle-origin-',dir=out.parent));movie=stage/'particle_origin.mp4'
        try:
            with h5py.File(stage/'tracked_particles.hdf5','w') as tracking,imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
                tracking.create_dataset('selected_anchor_ids',data=cohort,dtype='u8');tracking.attrs['identity_scope']='same-run anchor FOF type-1 particle IDs'
                box=anchor_box/1000.
                for epoch in epochs:
                    matched,pos=matching_positions(epoch['files'],cohort);pos=pos/1000.
                    if np.any(pos<0) or np.any(pos>=box):raise ValueError('Saved particle position lies outside declared periodic box')
                    group=tracking.create_group('snapshot_%03d'%epoch['snapshot']);group.create_dataset('particle_ids',data=matched,dtype='u8');group.create_dataset('positions_comoving_mpc_h',data=pos)
                    group.attrs['scale_factor']=epoch['scale_factor'];epoch['matched_count']=int(len(matched))
                    fig,axes=plt.subplots(1,3,figsize=(14.4,5.44),dpi=100,facecolor='#08111e')
                    fig.subplots_adjust(left=.05,right=.98,bottom=.14,top=.82,wspace=.25)
                    for ax,(i,j) in zip(axes,((0,1),(0,2),(1,2))):
                        ax.set_facecolor('#11243a');ax.scatter(pos[:,i],pos[:,j],s=1.5,c='#6ee7f2',alpha=.65,linewidths=0,rasterized=True)
                        ax.set_xlim(0,box);ax.set_ylim(0,box);ax.set_aspect('equal');ax.grid(False)
                        ax.set_xlabel('xyz'[i]+' [comoving Mpc/h]',color='#c5d6e8',fontsize=9);ax.set_ylabel('xyz'[j]+' [comoving Mpc/h]',color='#c5d6e8',fontsize=9)
                        ax.tick_params(colors='#c5d6e8',labelsize=8)
                    fig.suptitle('Origin of host %d · anchor snapshot %d · z=%.4g · %d/%d traced IDs' % (host_row,anchor_snapshot,epoch['redshift'],len(matched),len(cohort)),color='white',fontsize=15,y=.95)
                    fig.text(.5,.88,'Same fixed anchor-member sample · full periodic box · saved epochs only',color='#b8cadc',ha='center',fontsize=10)
                    fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                    for _ in range(fps*seconds_per_snapshot):writer.append_data(frame)
                    plt.close(fig)
            for path,checksum in sources.items():
                if file_sha256(path)!=checksum:raise ValueError('Particle origin source changed during export: '+path)
            atomic_json(stage/'origin_contract.json',{'schema':'anchor-particle-origin-movie-v1','science_acceptance':'NOT_ACCEPTED',
                'run':str(root),'anchor_snapshot':anchor_snapshot,'host_row':host_row,'anchor_member_count':count,'selected_count':len(cohort),
                'selection':'Lowest deterministic 64-bit hash of exact native anchor type-1 IDs; fixed across saved epochs.',
                'epochs':epochs,'source_sha256':sources,'algorithm_sha256':file_sha256(__file__),
                'movie_sha256':file_sha256(movie),'tracked_particles_sha256':file_sha256(stage/'tracked_particles.hdf5'),
                'timing_scope':'Equal display time per actual saved epoch, no interpolation or uniform physical-time claim.',
                'identity_scope':'Same-run anchor FOF material continuity; not evidence of persistent halo core or cross-run identity.'})
            publish_directory(stage,out)
            return out/'particle_origin.mp4'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
            raise
