"""Add mass-weighted gas metallicity maps to a sealed saved-map bundle."""
from __future__ import annotations
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
import h5py
import numpy as np


def metallicity_projection(chunks,grid,box,axes,los,center,thickness):
    """Periodic CIC ratio sum(m Z)/sum(m), retaining physically zero metal fractions."""
    from core.visualization import _project_cic
    total=np.zeros((grid,grid));metal=np.zeros_like(total)
    for xyz,m,z in chunks:
        xyz=np.asarray(xyz,float);m=np.asarray(m,float);z=np.asarray(z,float)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or m.shape!=(len(xyz),) or z.shape!=m.shape or not np.isfinite(xyz).all() or not np.isfinite(m).all() or not np.isfinite(z).all() or np.any(m<=0) or np.any((z<0)|(z>1)):raise ValueError('Invalid gas coordinates, masses or metallicity mass fractions')
        keep=np.ones(len(xyz),bool)
        if thickness is not None:keep=np.abs((xyz[:,los]-center+box/2)%box-box/2)<=thickness/2
        xy=xyz[keep][:,axes];total+=_project_cic(xy,m[keep],grid,box);metal+=_project_cic(xy,m[keep]*z[keep],grid,box)
    values=np.divide(metal,total,out=np.full_like(total,np.nan),where=total>0)
    return values.T,total.T


def make_metallicity_bundle(run_dir,source_bundle,destination,*,source_key,memory_gib=1.):
    from core.analysis import read_snapshot_header
    from core.data import validate_native_shard_counts,coordinate_contract,run_epoch_contract,write_dataset_contract
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,product_provenance,verify_product_provenance,publish_directory
    from halos import _unit_conversion
    root=Path(run_dir).expanduser().resolve();bundle=Path(source_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh metallicity map directory')
    if not isinstance(source_key,str) or not re.fullmatch(r'[A-Za-z0-9_]+',source_key):raise ValueError('Valid source map series key required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(float,int)) or not math.isfinite(memory_gib) or memory_gib<.25:raise ValueError('At least 0.25 GiB memory required')
    with native_run_lock(root,operation='gas metallicity maps'):
        manifest=bundle/'map_contract.json';manifest_hash=file_sha256(manifest);contract=json.loads(manifest.read_text())
        if contract.get('schema')!='map-publication-v1':raise ValueError('Published source maps required')
        verify_product_provenance(root,contract['provenance']);inventory={r['path']:r['sha256'] for r in contract['artifacts']}
        def checked(name):
            if not isinstance(name,str) or name not in inventory or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unlisted source asset')
            path=(bundle/name).resolve()
            if not path.is_relative_to(bundle):raise ValueError('Source asset escapes bundle')
            if not path.is_file() or file_sha256(path)!=inventory[name]:raise ValueError('Source map asset missing or changed: '+name)
            return path
        scale=json.loads(checked('scale.json').read_text());setting=scale['maps'].get(source_key)
        if not setting or setting.get('status')!='COMPLETE' or setting.get('normalization')!='global_fixed':raise ValueError('A complete global-scale source map series is required')
        rows=[];source_files=[]
        for name in sorted(inventory):
            if not re.fullmatch(re.escape(source_key)+r'_\d+\.figure\.json',name):continue
            meta=json.loads(checked(name).read_text())['scientific_metadata'];snap=int(meta['snapshot'])
            if meta.get('overlay_layers')!=[] or meta['field'] not in ('gas','dm','total','stars'):raise ValueError('Use an unadorned surface-density source map series')
            source_data=checked(meta['numeric_asset'])
            with h5py.File(source_data,'r') as f:
                shape=f['log10_display'].shape
                if len(shape)!=2 or shape[0]!=shape[1] or shape[0]**2*80>memory_gib*1024**3*.5:raise MemoryError('Map geometry exceeds memory')
            header,files=read_snapshot_header(root,snap);validate_native_shard_counts(files,'NumPart_Total');_unit_conversion(files)
            a=float(header['Time']);box=float(header['BoxSize'])/1000.
            if not math.isfinite(a) or a<=0 or not math.isclose(box,float(meta['box_size_mpc_h']),rel_tol=1e-10) or not math.isclose(1/a-1,float(meta['redshift']),rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Snapshot and source map geometry/epoch differ')
            rows.append({'snapshot':snap,'metadata':meta,'files':files,'a':a,'box':box,'grid':shape[0]});source_files.extend(files)
        if not rows:raise ValueError('Source series has no saved epoch maps')
        provenance=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('analysis.py'),Path(__file__).with_name('visualization.py')],source_files=list(dict.fromkeys(source_files+[manifest])),selection={'source_key':source_key,'field':'gas_metallicity'},coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global periodic box',periodic_wrapping='periodic projected CIC',velocity_convention='not used'))
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.metallicity-maps-',dir=out.parent))
        try:
            for name in inventory:
                target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(checked(name),target)
            outputs=[];limits=[];missing=[];metal_key=None
            for row in rows:
                meta=row['metadata'];sliceinfo=meta['slice'];projection=meta['projection'];axes={'xy':(0,1),'xz':(0,2),'yz':(1,2)}[projection];los=({0,1,2}-set(axes)).pop();grid=row['grid'];box=row['box']
                if int(sliceinfo['line_of_sight_axis'])!=los:raise ValueError('Source line-of-sight axis is inconsistent')
                thickness=sliceinfo['thickness_mpc_h'];center=sliceinfo['center_mpc_h']
                has=[]
                for path in row['files']:
                    with h5py.File(path) as f:
                        hdr=f['Header'].attrs
                        if not math.isclose(float(hdr['Time']),row['a'],rel_tol=1e-12) or not math.isclose(float(hdr['BoxSize'])/1000.,box,rel_tol=1e-12):raise ValueError('Snapshot shards disagree')
                        if 'PartType0' in f and len(f['PartType0/Coordinates']):has.append('Metallicity' in f['PartType0'])
                if not has or not any(has):missing.append({'snapshot':row['snapshot'],'reason':'No saved gas metallicity'});continue
                if not all(has):raise ValueError('Metallicity missing in some populated snapshot shards')
                def chunks():
                    for path in row['files']:
                        with h5py.File(path) as f:
                            if 'PartType0' not in f:continue
                            p=f['PartType0'];xyz=p['Coordinates'];n=len(xyz)
                            if not n:continue
                            z=p['Metallicity'];m=p.get('Masses');fixed=float(f['Header'].attrs['MassTable'][0])
                            if xyz.shape!=(n,3) or z.shape!=(n,) or (m is not None and m.shape!=(n,)) or (m is None and fixed<=0):raise ValueError('Gas fields do not align')
                            units=z.attrs
                            for key,expected in [('a_scaling',0.),('h_scaling',0.),('length_scaling',0.),('mass_scaling',0.),('velocity_scaling',0.),('to_cgs',1.)]:
                                if key not in units or not math.isclose(float(units[key]),expected,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Metallicity must declare dimensionless mass-fraction unit metadata')
                            for start in range(0,n,200000):
                                stop=min(start+200000,n);yield np.asarray(xyz[start:stop],float)/1000.,np.asarray(m[start:stop],float) if m is not None else np.full(stop-start,fixed),np.asarray(z[start:stop],float)
                values,mass=metallicity_projection(chunks(),grid,box,axes,los,center,thickness);log=np.full_like(values,np.nan);valid=np.isfinite(values)&(values>0);log[valid]=np.log10(values[valid])
                if not np.any(np.isfinite(values)):missing.append({'snapshot':row['snapshot'],'reason':'No gas in the selected projection'});continue
                if np.any(valid):
                    finite=log[valid];limits.append([float(np.percentile(finite,1)),float(np.percentile(finite,99.5))])
                metal_key='gas_metallicity_'+source_key.split('_',1)[1];stem=metal_key+'_%03d'%row['snapshot'];numeric=stem+'.hdf5'
                epoch=run_epoch_contract(root,row['snapshot'],row['a']);coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global periodic box',periodic_wrapping='periodic CIC',velocity_convention='not applicable')
                with h5py.File(stage/numeric,'w') as f:
                    f.attrs['science_acceptance']='NOT_ACCEPTED';measured=f.create_dataset('measured',data=values,chunks=True,fletcher32=True);display=f.create_dataset('log10_display',data=log,chunks=True,fletcher32=True)
                    context={'bundle_manifest':'map_contract.json','snapshot_storage_index':row['snapshot'],'projection':projection,'slice':sliceinfo,'assignment':'periodic CIC; no smoothing'}
                    write_dataset_contract(measured,units='dimensionless mass fraction',epoch=epoch,coordinates=coordinates,provenance=context,quantity='CIC mass-weighted gas metallicity',missing_policy='NaN where no gas mass; zero represents zero measured metals')
                    write_dataset_contract(display,units='dimensionless log10(metal mass fraction)',epoch=epoch,coordinates=coordinates,provenance=context,quantity='log10 of positive metallicity mass fraction',missing_policy='NaN for zero metallicity or absent gas; inspect measured field')
                new=dict(meta);new.update({'field':'gas_metallicity','numeric_asset':numeric,'measured_quantity_units':'dimensionless mass fraction','density_colormap':'viridis','metallicity_definition':'CIC sum(m Z)/sum(m), with Z the saved total metal mass fraction','zero_metal_display':'masked in logarithmic display; measured zeros retained','display_status':'AVAILABLE' if np.any(valid) else 'ALL_SUPPORTED_GAS_ZERO_METALLICITY','overlay_layers':[],'overlays':{}})
                atomic_json(stage/(stem+'.figure.json'),{'scientific_metadata':new});outputs.append({'snapshot':row['snapshot'],'numeric':numeric,'figure':stem+'.figure.json'})
            if not outputs:raise ValueError('No gas metallicity maps available in selected epochs: '+json.dumps(missing))
            low=min(v[0] for v in limits) if limits else -6.;high=max(v[1] for v in limits) if limits else 0.
            if high<=low:high=low+1e-6
            scale['maps'][metal_key]={'status':'COMPLETE','normalization':'global_fixed','vmin':low,'vmax':high,'missing_snapshots':missing,'definition':'mass-weighted gas metal mass fraction'}
            atomic_json(stage/'scale.json',scale)
            if file_sha256(manifest)!=manifest_hash:raise ValueError('Source map contract changed')
            for name in inventory:checked(name)
            verify_product_provenance(root,contract['provenance']);verify_product_provenance(root,provenance)
            atomic_json(stage/'map_contract.json',{'schema':'map-publication-v1','science_acceptance':'NOT_ACCEPTED','validation_state':'VALIDATION_PENDING','provenance':provenance,'parent_bundle':str(bundle),'parent_contract_sha256':manifest_hash,'metallicity_series':metal_key,'missing_metallicity_snapshots':missing,'artifacts':[{'path':p.relative_to(stage).as_posix(),'sha256':file_sha256(p),'size_bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()]})
            publish_directory(stage,out);return {'bundle':out,'map_key':metal_key,'snapshots':outputs,'missing':missing}
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
