"""Source-bound N128 atlas measurements in common non-h-scaled physical units.

Original native-run files stay untouched. A paired cache retains arrays, exact
sample IDs, source hashes and explicit bin/selection conventions at every epoch.
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import math
import os
import numpy as np
import h5py
import yaml
from core.foundation import atomic_json, atomic_text, file_sha256, MSUN_G

PRIORITY = (14.4, 12., 10., 8., 6., 3., 1., 0.)


def epoch_name(run_name, snapshot, redshift):
    z=max(0.,float(redshift));token=f'{z:08.3f}'.replace('.','p')
    return f'{run_name}_z_{token}_snap_{snapshot:03d}'


def physical_frame(cfg):
    h=float(cfg['cosmology']['H0'])/100
    box=float(cfg['simulation']['box_mpc_h'])/h
    if h<=0 or box<=0 or not math.isfinite(h+box):raise ValueError('Positive saved h and box required')
    return h,box


def units_to_physical(x, y, product, h):
    """Explicit conversions used by paired plots, without altering native arrays."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    if product in ('hmf','subhmf'):return x/h,y*h**3
    if product=='pk':return x*h,y/h**3
    if product=='correlation':return x/h,y
    raise ValueError('Unknown comparison product')


def _write_csv(path,rows):
    if not rows:return
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def measure_run(run, workspace, destination, name, *, map_grid=384):
    from core.analysis import snapshot_ids,read_snapshot_header,halo_masses,halo_particle_counts,group_field,subhalo_field,group_files,gadget_matter_power_spectrum
    from core.visualization import _velocity_conversion_km_s
    from core.halos import _unit_conversion
    root=Path(run).resolve();work=Path(workspace).resolve();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((root/'config.yaml').read_text());h,box=physical_frame(cfg)
    manifest_before=file_sha256(root/'manifest.json');config_before=file_sha256(root/'config.yaml')
    project=Path(__file__).resolve().parents[1]
    implementations={str(project/name):file_sha256(project/name) for name in (
        'core/paired_atlas.py','core/analysis.py','core/visualization.py','core/halos.py',
        'core/data.py','core/foundation.py','core/__init__.py','configs/core_source_layout.json')}
    rows=[];epochs=[]
    for snap in snapshot_ids(work):
        header,files=read_snapshot_header(work,snap);a=float(header['Time']);z=1/a-1
        label=epoch_name(name,snap,z);numeric=out/(label+'.npz');record=out/(label+'.json')
        source_paths={Path(p).resolve() for p in [*files,*group_files(work,snap)]}
        power_path=work/'data/powerspecs'/f'powerspec_{snap:03d}.txt'
        if power_path.exists():source_paths.add(power_path.resolve())
        binding={'schema':'paired-epoch-cache-binding-v1','run':str(root),'workspace':str(work),
            'name':name,'snapshot':snap,'scale_factor':a,'map_grid':map_grid,
            'manifest_sha256':manifest_before,'config_sha256':config_before,
            'implementation_sha256':implementations,
            'source_hashes':{str(p):file_sha256(p) for p in sorted(source_paths)}}
        seal=record.with_name(record.name+'.sha256')
        if record.exists()!=numeric.exists():
            raise ValueError('Incomplete paired cache; preserve it and choose a fresh destination: '+str(record))
        if record.exists() and numeric.exists():
            if not seal.is_file() or seal.read_text().strip()!=file_sha256(record):
                raise ValueError('Unsealed or changed paired cache metadata; choose a fresh destination: '+str(record))
            saved=json.loads(record.read_text())
            if saved.get('cache_binding')!=binding:
                raise ValueError('Paired cache inputs, implementation or map grid changed; choose a fresh destination: '+str(record))
            if saved['numeric_sha256']==file_sha256(numeric):
                rows.append(saved['summary']);epochs.append(saved);print('CACHE',name,snap,flush=True);continue
            raise ValueError('Existing paired cache changed: '+str(numeric))
        _unit_conversion(files)
        velocity_factor=_velocity_conversion_km_s(files,1,header)
        if velocity_factor is None:raise ValueError('Saved peculiar velocity convention unavailable')
        maps={p:np.zeros((map_grid,map_grid),float) for p in ('xy','xz','yz')}
        cube=np.zeros((64,64,64),float);speeds=[];cloud=[];cloud_ids=[];sources=[];particle_count=0;total_mass=0.
        for path in files:
            with h5py.File(path) as f:
                if 'PartType1' not in f:continue
                if not math.isclose(float(f['Parameters'].attrs['HubbleParam']),h,rel_tol=1e-10):raise ValueError('Config and native Hubble parameters differ')
                g=f['PartType1'];n=len(g['Coordinates']);fixed=float(f['Header'].attrs['MassTable'][1]);factor=float(f['Parameters'].attrs['UnitMass_in_g'])/MSUN_G/h
                for start in range(0,n,200000):
                    stop=min(start+200000,n);p=np.asarray(g['Coordinates'][start:stop],float)/1000/h
                    v=np.asarray(g['Velocities'][start:stop],float)*velocity_factor
                    mass=(np.asarray(g['Masses'][start:stop],float) if 'Masses' in g else np.full(stop-start,fixed))*factor
                    if np.any(mass<=0) or not np.isfinite(p).all() or not np.isfinite(v).all():raise ValueError('Nonfinite or nonpositive native particle payload')
                    p=np.mod(p,box)
                    for projection,(i,j) in {'xy':(0,1),'xz':(0,2),'yz':(1,2)}.items():
                        maps[projection]+=np.histogram2d(p[:,i],p[:,j],bins=map_grid,range=((0,box),(0,box)),weights=mass)[0]
                    cube+=np.histogramdd(p,bins=64,range=[(0,box)]*3,weights=mass)[0]
                    speeds.append(np.linalg.norm(v,axis=1));particle_count+=len(p);total_mass+=float(mass.sum())
                cloud.append(np.asarray(g['Coordinates'][::64],float)/1000/h)
                cloud_ids.append(np.asarray(g['ParticleIDs'][::64],dtype=np.uint64))
            sources.append({'path':str(Path(path).resolve()),'sha256':binding['source_hashes'][str(Path(path).resolve())]})
        def field(reader,names,shape=None):
            value=reader(work,snap,names)
            if value is None:return np.empty(shape if shape is not None else (0,),float)
            return np.asarray(value)
        raw=halo_masses(work,snap);counts=halo_particle_counts(work,snap)
        masses=np.asarray(raw if raw is not None else [],float)*1e10/h
        counts=np.asarray(counts if counts is not None else [],np.int64)
        if masses.shape!=counts.shape:raise ValueError('Host mass/particle-count mismatch')
        speed=np.concatenate(speeds) if speeds else np.empty(0)
        arrays={'fof_mass_msun':masses,'fof_particle_count':counts,
                'fof_position_comoving_mpc':field(group_field,['GroupPos'],(0,3))/1000/h,
                'fof_velocity_peculiar_km_s':field(group_field,['GroupVel'],(0,3)),
                'm200c_msun':field(group_field,['Group_M_Crit200'])*1e10/h,
                'r200c_comoving_mpc':field(group_field,['Group_R_Crit200'])/1000/h,
                'subhalo_mass_msun':field(subhalo_field,['SubhaloMass'])*1e10/h,
                'subhalo_vmax_km_s':field(subhalo_field,['SubhaloVmax']),
                'subhalo_spin_kpc_km_s':field(subhalo_field,['SubhaloSpin'],(0,3))/h,
                'sampled_position_comoving_mpc':np.vstack(cloud),'sampled_particle_ids':np.concatenate(cloud_ids),
                'density_mass_msun_64_cubed':cube,'velocity_histogram_edges_km_s':np.linspace(0,max(1,float(speed.max())),101)}
        arrays['velocity_histogram_counts']=np.histogram(speed,arrays['velocity_histogram_edges_km_s'])[0]
        for projection,m in maps.items():arrays['dm_surface_density_msun_mpc2_'+projection]=m.T/(box/map_grid)**2
        pkpath=work/'data/powerspecs'/f'powerspec_{snap:03d}.txt'
        if pkpath.exists():
            pk=gadget_matter_power_spectrum(pkpath)
            if pk:
                arrays.update(k_mpc_inverse=pk['k_h_Mpc']*h,power_mpc3=pk['P_Mpc_h3']/h**3,
                    power_sigma_gaussian_mpc3=pk['P_gaussian_1sigma_Mpc_h3']/h**3,power_mode_count=pk['mode_count'],
                    power_shot_mpc3=pk['P_shot_noise_Mpc_h3']/h**3)
            sources.append({'path':str(pkpath.resolve()),'sha256':binding['source_hashes'][str(pkpath.resolve())]})
        sources.extend({'path':str(p.resolve()),'sha256':binding['source_hashes'][str(p.resolve())]} for p in group_files(work,snap))
        np.savez_compressed(numeric,**arrays)
        summary={'run':name,'snapshot':snap,'redshift':z,'scale_factor':a,'h':h,'box_comoving_mpc':box,
            'particle_count':particle_count,'particle_mass_mean_msun':total_mass/particle_count,'total_dm_mass_msun':total_mass,
            'fof_hosts_32':int(np.sum(counts>=32)),'fof_hosts_100':int(np.sum(counts>=100)),'fof_hosts_300':int(np.sum(counts>=300)),
            'fof_hosts_1000':int(np.sum(counts>=1000)),'largest_fof_mass_msun':float(masses.max()) if len(masses) else 0.,
            'subhalo_count':len(arrays['subhalo_mass_msun']),'particle_speed_median_km_s':float(np.median(speed)),
            'particle_speed_p90_km_s':float(np.percentile(speed,90)),'particle_speed_rms_km_s':float(np.sqrt(np.mean(speed**2)))}
        for path,expected in {**binding['source_hashes'],**implementations}.items():
            if file_sha256(path)!=expected:raise ValueError('Paired measurement source changed during generation: '+path)
        saved={'schema':'physical-dmo-epoch-v1','cache_binding':binding,'run':str(root),'workspace':str(work),'numeric_asset':numeric.name,'numeric_sha256':file_sha256(numeric),
            'summary':summary,'source_files':sources,'algorithm_sha256':file_sha256(__file__),
            'units':{'mass':'Msun without h','length':'comoving Mpc without h','speed':'peculiar km/s','surface_density':'Msun/Mpc^2','power':'Mpc^3','k':'Mpc^-1'},
            'selection':'Full saved type-1 mass, velocity and projected maps. All native FOF and SUBFIND rows retained; particle floors are explicit plot selections.',
            'sampling':'3D display sample: every 64th saved row per native shard; exact uint64 IDs retained. Not a completeness sample.',
            'uncertainty':'Native power diagonal Gaussian mode errors retained. Host bins use count intervals; single-box sample variance and numerical convergence are not certified.'}
        atomic_json(record,saved);atomic_text(seal,file_sha256(record)+'\n')
        rows.append(summary);epochs.append(saved);print('MEASURED',name,snap,'z',z,flush=True)
    if file_sha256(root/'manifest.json')!=manifest_before or file_sha256(root/'config.yaml')!=config_before:raise ValueError('Original run metadata changed during cache generation')
    _write_csv(out/(name+'_all_86_epochs_physical_units.csv'),rows)
    atomic_json(out/'measurement_manifest.json',{'schema':'physical-dmo-run-v1','run':str(root),'name':name,'h':h,'box_comoving_mpc':box,
        'original_manifest_sha256':manifest_before,'original_config_sha256':config_before,'epochs':epochs,
        'priority_redshifts':list(PRIORITY),'algorithm_sha256':file_sha256(__file__)})
    return out/'measurement_manifest.json'


def measure_physical_ic_agreement(workspaces, destination, *, mesh=64):
    """Compare normalized IC fields and shared lattice IDs across different h."""
    from core.analysis import _initial_condition_paths,_deposit_cic_density_mesh
    from scipy.fft import rfftn
    roots=[Path(p).resolve() for p in workspaces];configs=[yaml.safe_load((p/'config.yaml').read_text()) for p in roots]
    frames=[physical_frame(c) for c in configs]
    if not math.isclose(frames[0][1],frames[1][1],rel_tol=1e-10):raise ValueError('ICs have different physical boxes')
    if configs[0]['simulation']['particles_per_dimension']!=configs[1]['simulation']['particles_per_dimension']:raise ValueError('IC particle lattices differ')
    groups=[_initial_condition_paths(r) for r in roots];fields=[];sources=[];ids=[];positions=[]
    for files,cfg in zip(groups,configs):
        field,box,count=_deposit_cic_density_mesh(files,mesh)
        if not math.isclose(box/1000,cfg['simulation']['box_mpc_h'],rel_tol=1e-8):raise ValueError('Native IC box and config differ')
        fields.append(rfftn(field));ii=[];pp=[]
        for p in files:
            sources.append({'path':str(p),'sha256':file_sha256(p)})
            with h5py.File(p) as f:
                if 'PartType1' not in f:continue
                g=f['PartType1'];ii.append(np.asarray(g['ParticleIDs'],np.uint64));pp.append(np.asarray(g['Coordinates'],float)/box)
        i=np.concatenate(ii);p=np.vstack(pp);order=np.argsort(i);i=i[order];p=p[order]
        if len(np.unique(i))!=len(i):raise ValueError('Duplicate native IC particle IDs')
        ids.append(i);positions.append(p)
    fft_a,fft_b=fields
    kx=np.fft.fftfreq(mesh)*mesh;kz=np.fft.rfftfreq(mesh)*mesh
    xx,yy,zz=np.meshgrid(kx,kx,kz,indexing='ij');number=np.sqrt(xx*xx+yy*yy+zz*zz)
    aa,ab=np.abs(fft_a),np.abs(fft_b)
    keep=(number>0)&(number<=mesh/4)&(aa>aa.max()*1e-10)&(ab>ab.max()*1e-10)
    cross=fft_a[keep]*np.conj(fft_b[keep]);weights=np.abs(cross);phase=np.angle(cross)
    coherence=float(abs(cross.sum())/weights.sum());rms=float(np.sqrt(np.sum(weights*phase*phase)/weights.sum()))
    common,aidx,bidx=np.intersect1d(ids[0],ids[1],assume_unique=True,return_indices=True)
    delta=(positions[1][bidx]-positions[0][aidx]+.5)%1-.5
    spacing=1/configs[0]['simulation']['particles_per_dimension'];distance=np.linalg.norm(delta,axis=1)/spacing
    report={'schema':'physical-ic-agreement-v1','runs':[str(r) for r in roots],'box_comoving_mpc':frames[0][1],'h':[r[0] for r in frames],
        'mesh':mesh,'common_low_k_modes':int(keep.sum()),'complex_phase_coherence':coherence,'weighted_rms_phase_rad':rms,
        'phase_status':'PASS' if coherence>=.99 and rms<=.05 else 'FAIL','shared_initial_ids':len(common),'initial_id_counts':[len(i) for i in ids],
        'maximum_shared_id_position_difference_particle_spacings':float(distance.max()),'median_shared_id_position_difference_particle_spacings':float(np.median(distance)),
        'shared_lattice_status':'PASS' if len(common)==len(ids[0])==len(ids[1]) and np.all(distance<.5) else 'FAIL',
        'source_files':sources,'algorithm_sha256':file_sha256(__file__),
        'scope':'Independent run namespaces retained. Low-k phases are measured from actual initial mass fields; shared IDs are checked against their normalized initial positions. Matching seed metadata alone is not proof. Different cosmological parameter sets remain a full-model contrast.'}
    atomic_json(destination,report);return report
