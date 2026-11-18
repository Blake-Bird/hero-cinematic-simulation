"""Exact-ID matched host comparison with common physical map and profile grids."""
from __future__ import annotations
from contextlib import ExitStack
import html
import math
import tempfile
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def common_aperture_histogram(position,mass,center,box,extent,grid,edges):
    p=np.asarray(position,float);m=np.asarray(mass,float);c=np.asarray(center,float)
    if p.ndim!=2 or p.shape[1]!=3 or m.shape!=(len(p),) or c.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('Valid aligned particle data required')
    delta=(p-c+box/2)%box-box/2;inside=np.all(np.abs(delta)<extent,axis=1);d=delta[inside];w=m[inside]
    image=np.histogram2d(d[:,0],d[:,1],bins=grid,range=((-extent,extent),(-extent,extent)),weights=w)[0]
    radius=np.linalg.norm(d,axis=1);return image,np.histogram(radius,edges,weights=w)[0],np.histogram(radius,edges)[0]


def make_halo_fingerprint(run_a,run_b,destination,*,snapshot_a,snapshot_b,host_row,min_shared=10,min_fraction=.1,grid=128,redshift_tolerance=1e-6,memory_gib=1.):
    from core.analysis import (_cross_run_identity,compare_initial_ic_phases,_read_cross_run_halo_catalogue,_mutual_best_halo_matches,read_snapshot_header,group_files,group_ordered_snapshot_files,group_field,_initial_condition_paths)
    from core.data import coordinate_contract,validate_native_shard_counts
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,product_provenance,verify_product_provenance,publish_directory,MSUN_G
    from halos import _unit_conversion
    roots=[Path(run_a).expanduser().resolve(),Path(run_b).expanduser().resolve()];out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh halo-fingerprint output directory')
    if any(type(v) is not int or v<0 for v in (snapshot_a,snapshot_b,host_row)) or type(min_shared) is not int or min_shared<1 or type(grid) is not int or not 32<=grid<=512 or not math.isfinite(min_fraction) or not 0<min_fraction<=1 or not math.isfinite(redshift_tolerance) or redshift_tolerance<0:raise ValueError('Invalid matching or map controls')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<.5:raise ValueError('At least 0.5 GiB memory required')
    with ExitStack() as locks:
        for root in sorted(set(roots)):locks.enter_context(native_run_lock(root,operation='halo fingerprint'))
        identities=[_cross_run_identity(root) for root in roots]
        if any(x is None for x in identities):raise ValueError('Both runs need recorded seed, initial lattice and box')
        if identities[0]['random_seed']!=identities[1]['random_seed'] or identities[0]['particles_per_dimension']!=identities[1]['particles_per_dimension'] or not math.isclose(identities[0]['box_mpc_h'],identities[1]['box_mpc_h'],rel_tol=1e-12):raise ValueError('Halo matching requires equal declared seeds, particle lattices and boxes')
        phase=compare_initial_ic_phases(*roots)
        if roots[0]!=roots[1] and phase.get('status')!='PASS':raise ValueError('Cross-run fingerprint requires measured initial phase agreement; status: '+str(phase.get('status')))
        same_run=roots[0]==roots[1];max_ids=int(memory_gib*1024**3*.35/128)
        catalogs=[_read_cross_run_halo_catalogue(root,snap,'m200c',1,max_ids) for root,snap in zip(roots,(snapshot_a,snapshot_b))]
        matches=_mutual_best_halo_matches(*catalogs,min_shared,min_fraction);found=np.flatnonzero(matches['a']==host_row)
        if len(found)!=1:raise ValueError('Selected host has no unique reciprocal-best counterpart passing shared-ID thresholds')
        index=int(found[0]);rows=[host_row,int(matches['b'][index])];prepared=[]
        for root,snap,cat,row in zip(roots,(snapshot_a,snapshot_b),catalogs,rows):
            loc=np.flatnonzero(cat['group_indices']==row)
            if len(loc)!=1:raise ValueError('Matched host row missing or duplicated')
            pos=int(loc[0]);header,files=read_snapshot_header(root,snap);validate_native_shard_counts(files,'NumPart_Total');_unit_conversion(files)
            a=float(header['Time']);box=float(header['BoxSize'])/1000.;radius=float(cat['r200c_mpc_h'][pos]);center=np.asarray(cat['position_mpc_h'][pos],float)
            if not math.isfinite(a) or a<=0 or not math.isfinite(radius) or radius<=0 or not math.isclose(box,identities[0]['box_mpc_h'],rel_tol=1e-10):raise ValueError('Matched halo frame, box or R200c invalid')
            if cat['redshift'] is None or not math.isclose(float(cat['redshift']),1/a-1,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Catalogue and snapshot epochs differ')
            sub=group_field(root,snap,['GroupNsubs']);first=group_field(root,snap,['GroupFirstSub'])
            sub_count=None
            if sub is not None and first is not None:
                sub=np.asarray(sub);first=np.asarray(first)
                if sub.ndim!=1 or first.shape!=sub.shape or row>=len(sub) or sub.dtype.kind not in 'ui' or first.dtype.kind not in 'ui' or int(sub[row])<0 or (int(sub[row])>0 and int(first[row])<0):raise ValueError('Invalid subhalo count fields')
                # GroupNsubs includes the central subhalo when positive.
                sub_count=max(0,int(sub[row])-1)
            sources=list(files)+list(group_files(root,snap))+list(group_ordered_snapshot_files(root,snap))+list(_initial_condition_paths(root))
            contract=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('analysis.py')],source_files=list(dict.fromkeys(sources)),selection={'snapshot':snap,'host_row':row,'grid':grid},coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='host GroupPos',periodic_wrapping='minimum image',velocity_convention='not used'))
            prepared.append({'root':root,'snap':snap,'row':row,'files':files,'a':a,'box':box,'center':center,'radius':radius,'mass':float(cat['mass_msun_h'][pos]),'sub_count':sub_count,'contract':contract})
        x,y=prepared
        if abs(1/x['a']-1/y['a'])>redshift_tolerance*max(1.,abs(1/x['a']-1),abs(1/y['a']-1)):raise ValueError('Matched snapshots do not share the declared redshift tolerance')
        extent=min(2*x['radius'],2*y['radius'],x['box']/2*(1-1e-12));edges=np.linspace(0,extent,33)
        if extent<=0:raise ValueError('No common periodic aperture')
        for item in prepared:
            image=np.zeros((grid,grid));mass=np.zeros(32);counts=np.zeros(32)
            for path in item['files']:
                with h5py.File(path) as f:
                    hdr=f['Header'].attrs
                    if not math.isclose(float(hdr['Time']),item['a'],rel_tol=1e-12) or not math.isclose(float(hdr['BoxSize'])/1000.,item['box'],rel_tol=1e-12):raise ValueError('Snapshot shards disagree on frame')
                    if 'PartType1' not in f:continue
                    g=f['PartType1'];coords=g['Coordinates'];m=g.get('Masses');n=len(coords);fixed=float(hdr['MassTable'][1]);unit=float(f['Parameters'].attrs['UnitMass_in_g'])/MSUN_G
                    if coords.shape!=(n,3) or (m is not None and m.shape!=(n,)) or (m is None and fixed<=0):raise ValueError('Invalid particle positions or mass source')
                    for start in range(0,n,200000):
                        stop=min(start+200000,n);positions=np.asarray(coords[start:stop],float)/1000.;weights=(np.asarray(m[start:stop],float) if m is not None else np.full(stop-start,fixed))*unit
                        hist,sm,ct=common_aperture_histogram(positions,weights,item['center'],item['box'],extent,grid,edges);image+=hist;mass+=sm;counts+=ct
            item['map']=image/(2*extent/grid)**2;item['density']=mass/(4*np.pi/3*np.diff(edges**3))/item['a']**3;item['counts']=counts
        if not np.any(x['map']>0) or not np.any(y['map']>0):raise ValueError('Common halo maps contain no particles')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.halo-fingerprint-',dir=out.parent))
        try:
            maps=[item['map'] for item in prepared];positive=np.concatenate([m[m>0] for m in maps]);lo,hi=np.percentile(np.log10(positive),[1,99.9]);
            if hi<=lo:hi=lo+1
            logs=[np.ma.masked_where(m<=0,np.log10(np.maximum(m,np.finfo(float).tiny))) for m in maps]
            diff=maps[1]-maps[0];lim=float(np.percentile(abs(diff),99.5));lim=lim if lim>0 else max(float(np.max(abs(diff))),1.)
            fig,axes=plt.subplots(1,3,figsize=(15,5.3),facecolor='#081321');fig.subplots_adjust(left=.045,right=.96,bottom=.15,top=.82,wspace=.3)
            for ax,data,label in zip(axes[:2],logs,('Run A','Run B')):
                ax.set_facecolor('#000000');im=ax.imshow(data.T,origin='lower',extent=(-extent,extent,-extent,extent),vmin=lo,vmax=hi,cmap='magma');ax.set_title(label,color='white');cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.02);cb.ax.tick_params(colors='#c5d6e8',labelsize=7);cb.set_label('log10 surface density',color='#c5d6e8',fontsize=8)
            im=axes[2].imshow(diff.T,origin='lower',extent=(-extent,extent,-extent,extent),cmap='coolwarm',vmin=-lim,vmax=lim);axes[2].set_title('B − A',color='white');cb=fig.colorbar(im,ax=axes[2],fraction=.046,pad=.02);cb.ax.tick_params(colors='#c5d6e8',labelsize=7);cb.set_label('surface density difference',color='#c5d6e8',fontsize=8)
            for ax in axes:ax.set_xlabel('Δx [comoving Mpc/h]',color='#c5d6e8');ax.set_ylabel('Δy [comoving Mpc/h]',color='#c5d6e8');ax.tick_params(colors='#c5d6e8',labelsize=8);ax.grid(False)
            fig.suptitle('Matched halo fingerprint · fixed common comoving aperture',color='white',fontsize=15,y=.95);fig.text(.5,.89,'Host-centered XY projection · identical cube depth · density units h Msun/Mpc²',color='#b8cadc',ha='center',fontsize=9)
            fig.savefig(stage/'fingerprint_maps.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4.5),facecolor='#081321');ax.set_facecolor('#152a40');r=(edges[1:]+edges[:-1])/2
            for item,color,label in zip(prepared,('#ffb278','#7be6ef'),('Run A','Run B')):
                valid=item['density']>0;ax.loglog(r[valid],item['density'][valid],color=color,label=label)
            ax.set(xlabel='Comoving radius [Mpc/h]',ylabel='Proper DM density [(Msun/h)/(Mpc/h)³]');ax.tick_params(colors='#c5d6e8');ax.xaxis.label.set_color('#c5d6e8');ax.yaxis.label.set_color('#c5d6e8');ax.grid(alpha=.15);ax.legend();fig.tight_layout();fig.savefig(stage/'fingerprint_profiles.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
            np.savez_compressed(stage/'fingerprint_data.npz',map_a=maps[0],map_b=maps[1],map_difference_b_minus_a=diff,radial_edges_mpc_h=edges,proper_density_a=x['density'],proper_density_b=y['density'],radial_count_a=x['counts'],radial_count_b=y['counts'])
            for item in prepared:verify_product_provenance(item['root'],item['contract'])
            info=[{'run':str(q['root']),'snapshot':q['snap'],'host_row':q['row'],'redshift':1/q['a']-1,'m200c_msun_h':q['mass'],'r200c_mpc_h':q['radius'],'satellite_subhalos':q['sub_count']} for q in prepared]
            status='Same-run control; no model difference is implied.' if same_run else 'Cross-run comparison; measured low-k initial phase agreement passed.'
            def table(q):return ''.join('<tr><th>%s</th><td>%s</td></tr>'%(html.escape(k),html.escape(str(v))) for k,v in q.items())
            page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched halo fingerprint</title><style>body{background:#081321;color:#eef8ff;font:16px system-ui;margin:0}main{max-width:1350px;padding:28px;margin:auto}img{width:100%%}.columns{display:grid;grid-template-columns:1fr 1fr;gap:26px}p{color:#bcd0e1}table{border-collapse:collapse;width:100%%}td,th{padding:7px;border-bottom:1px solid #385067;text-align:left}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style><main><h1>Matched halo fingerprint</h1><p>%s Exact particle-ID matching: %d shared, %.3g of A and %.3g of B, reciprocal-best among eligible hosts.</p><img src="fingerprint_maps.png" alt="Host-centered maps and their difference"><div class="columns"><section><h2>Radial density</h2><img src="fingerprint_profiles.png" alt="Matched halo radial density profiles"><p>Profiles include all type-1 particles in common physical shells, including unbound or neighboring material. The maps use the same comoving cube and per-pixel area. A difference measures B minus A after centering each host; it is not a significance map.</p></section><section><h2>Run A</h2><table>%s</table><h2>Run B</h2><table>%s</table><p>Satellite counts exclude the central SUBFIND subhalo and depend on resolution. Matching and phase checks do not establish convergence or isolate a causal model effect.</p></section></div></main></html>'''%(html.escape(status),int(matches['shared'][index]),float(matches['fraction_a'][index]),float(matches['fraction_b'][index]),table(info[0]),table(info[1]))
            (stage/'halo_fingerprint.html').write_text(page,encoding='utf-8')
            atomic_json(stage/'fingerprint_contract.json',{'schema':'halo-fingerprint-v1','science_acceptance':'NOT_ACCEPTED','inputs':info,'same_run_control':same_run,'initial_phase_agreement':phase,'shared_particle_count':int(matches['shared'][index]),'fraction_a':float(matches['fraction_a'][index]),'fraction_b':float(matches['fraction_b'][index]),'match_min_shared':min_shared,'match_min_fraction':min_fraction,'common_half_extent_mpc_h':extent,'source_provenance':[q['contract'] for q in prepared],'algorithm_sha256':file_sha256(__file__),'interpretation':'Reciprocal best exact FOF DM particle-ID overlap; cross-run low-k phase agreement required. Common host-centered grid, not a statistical detection or convergence test.','artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out);return out/'halo_fingerprint.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
