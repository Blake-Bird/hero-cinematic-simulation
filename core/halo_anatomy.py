"""Interactive, saved-snapshot cutaway and measured radial halo profiles."""
from __future__ import annotations
import html
import json
import math
import tempfile
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def radial_statistics(radius, velocity, mass, edges):
    """Mass, count and mass-weighted 3-D velocity dispersion by radial shell."""
    radius=np.asarray(radius,float); velocity=np.asarray(velocity,float); mass=np.asarray(mass,float)
    if radius.ndim!=1 or velocity.shape!=(len(radius),3) or mass.shape!=radius.shape or not np.isfinite(radius).all() or not np.isfinite(velocity).all() or not np.isfinite(mass).all() or np.any(mass<=0):
        raise ValueError('Invalid radial particle arrays')
    result=[np.histogram(radius,edges,weights=w)[0] for w in [mass,None,*[mass*velocity[:,i] for i in range(3)],mass*np.sum(velocity**2,axis=1)]]
    return np.asarray(result)


def make_halo_anatomy(run_dir,destination,*,snapshot,host_row,grid=96,layers=24,memory_gib=1.):
    from core.zoom import read_host_catalogue
    from core.analysis import read_snapshot_header,subhalo_field,group_field
    from core.data import validate_native_shard_counts,coordinate_contract
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,product_provenance,verify_product_provenance,publish_directory,MSUN_G
    from core.visualization import _velocity_conversion_km_s
    from halos import _unit_conversion
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh halo anatomy output directory')
    if type(snapshot) is not int or snapshot<0 or type(host_row) is not int or host_row<0 or type(grid) is not int or not 32<=grid<=256 or type(layers) is not int or not 4<=layers<=64:raise ValueError('Valid snapshot, host row, grid 32–256 and layers 4–64 required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(float,int)) or not math.isfinite(memory_gib) or memory_gib<.25 or grid*grid*layers*40>memory_gib*1024**3*.5:raise ValueError('Insufficient declared anatomy memory')
    with native_run_lock(root,operation='halo anatomy'):
        host=read_host_catalogue(root,snapshot,memory_gib=memory_gib,host_row=host_row)
        header,files=read_snapshot_header(root,snapshot);validate_native_shard_counts(files,'NumPart_Total');_unit_conversion(files)
        a,box_native,h=host['frame'];box=box_native/1000.;center=np.asarray(host['positions'][host_row],float)/1000.;r200=float(host['radii'][host_row])/1000.;extent=2*r200
        if extent>box/2:raise ValueError('2 R200c must fit within the periodic box inscribed sphere')
        if not math.isclose(float(header['Time']),a,rel_tol=1e-12) or not math.isclose(float(header['BoxSize']),box_native,rel_tol=1e-12):raise ValueError('Host and snapshot frames do not match')
        vel_factor=_velocity_conversion_km_s(files,1,header)
        if vel_factor is None:raise ValueError('Saved dark-matter velocity units are required')
        mass200=group_field(root,snapshot,['Group_M_Crit200'])
        if mass200 is None or host_row>=len(mass200) or not np.isfinite(mass200[host_row]) or mass200[host_row]<=0:raise ValueError('Selected host requires measured M200c')
        provenance=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('analysis.py'),Path(__file__).with_name('zoom.py'),Path(__file__).with_name('visualization.py')],source_files=list(files)+list(host['catalogues']),selection={'snapshot':snapshot,'host_row':host_row,'grid':grid,'layers':layers},coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='selected GroupPos',periodic_wrapping='minimum image; aperture within half box',velocity_convention='peculiar km/s'))
        cube=np.zeros((grid,grid,layers));edges=np.linspace(0,extent,33);stats=np.zeros((6,32));selected=0
        for path in files:
            with h5py.File(path) as f:
                attrs=f['Header'].attrs
                if not math.isclose(float(attrs['Time']),a,rel_tol=1e-12) or not math.isclose(float(attrs['BoxSize']),box_native,rel_tol=1e-12):raise ValueError('Snapshot shards disagree on epoch or box')
                if 'PartType1' not in f:continue
                p=f['PartType1'];xyz=p['Coordinates'];v=p['Velocities'];m=p.get('Masses');n=len(xyz)
                if xyz.shape!=(n,3) or v.shape!=(n,3) or (m is not None and m.shape!=(n,)):raise ValueError('Particle fields are not aligned')
                fixed=float(attrs['MassTable'][1]);unit=float(f['Parameters'].attrs['UnitMass_in_g'])/MSUN_G
                if m is None and (not math.isfinite(fixed) or fixed<=0):raise ValueError('Dark matter mass unavailable')
                for start in range(0,n,200000):
                    stop=min(n,start+200000);p0=np.asarray(xyz[start:stop],float)/1000.;v0=np.asarray(v[start:stop],float)*vel_factor
                    m0=(np.asarray(m[start:stop],float) if m is not None else np.full(stop-start,fixed))*unit
                    if not np.isfinite(p0).all() or not np.isfinite(v0).all() or not np.isfinite(m0).all() or np.any(m0<=0):raise ValueError('Invalid particle data')
                    delta=(p0-center+box/2)%box-box/2;inside=np.all(np.abs(delta)<extent,axis=1);d=delta[inside];vv=v0[inside];mm=m0[inside]
                    cube+=np.histogramdd(d,bins=(grid,grid,layers),range=[(-extent,extent)]*3,weights=mm)[0]
                    radius=np.linalg.norm(d,axis=1);stats+=radial_statistics(radius,vv,mm,edges);selected+=int(np.count_nonzero(radius<extent))
        if selected==0:raise ValueError('No dark-matter particles in the halo aperture')
        subpos=subhalo_field(root,snapshot,['SubhaloPos']);subgroup=subhalo_field(root,snapshot,['SubhaloGroupNr']);submass=subhalo_field(root,snapshot,['SubhaloMass'])
        subhalos=[];substatus='unavailable'
        if subpos is not None and subgroup is not None and submass is not None:
            subpos=np.asarray(subpos,float);subgroup=np.asarray(subgroup);submass=np.asarray(submass,float)
            if subpos.shape!=(len(subgroup),3) or submass.shape!=(len(subgroup),) or subgroup.dtype.kind not in 'ui' or not np.isfinite(subpos).all() or not np.isfinite(submass).all():raise ValueError('Subhalo fields are malformed')
            for row in np.flatnonzero(subgroup==host_row):
                d=(subpos[row]/1000.-center+box/2)%box-box/2
                if np.all(np.abs(d)<extent):subhalos.append({'row':int(row),'x':float(d[0]),'y':float(d[1]),'z':float(d[2]),'mass_msun_h':float(submass[row]*1e10)})
            substatus='available'
        shell_mass,count,mvx,mvy,mvz,mv2=stats
        variance=np.divide(mv2,shell_mass,out=np.zeros_like(mv2),where=shell_mass>0)-np.divide(mvx**2+mvy**2+mvz**2,shell_mass**2,out=np.zeros_like(mv2),where=shell_mass>0)
        dispersion=np.full_like(shell_mass,np.nan);good=(count>=10)&(shell_mass>0);dispersion[good]=np.sqrt(np.maximum(variance[good],0))
        density=shell_mass/(4*np.pi/3*(edges[1:]**3-edges[:-1]**3))/a**3;radius=(edges[:-1]+edges[1:])/2
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.halo-anatomy-',dir=out.parent))
        try:
            finite=cube[cube>0];low,high=np.log10(finite).min(),np.log10(finite).max()
            if high<=low:high=low+1
            palette=matplotlib.colormaps['magma']
            for k in range(layers):
                plane=cube[:,:,k].T;normalized=np.zeros_like(plane);valid=plane>0;normalized[valid]=(np.log10(plane[valid])-low)/(high-low)
                color=palette(normalized,bytes=True)[...,:3];color[~valid]=0;Image.fromarray(np.flipud(color)).resize((768,768),Image.Resampling.NEAREST).save(stage/('slice_%02d.png'%k))
            fig,axes=plt.subplots(1,2,figsize=(10,4),facecolor='#0c1a2a')
            for ax in axes:ax.set_facecolor('#152a40');ax.tick_params(colors='#d4e3f1');ax.grid(alpha=.15);ax.xaxis.label.set_color('#d4e3f1');ax.yaxis.label.set_color('#d4e3f1')
            valid=density>0;axes[0].loglog(radius[valid]/r200,density[valid],color='#ffb978');axes[0].set(xlabel='r / R200c',ylabel='Proper DM density [(Msun/h)/(Mpc/h)³]')
            axes[1].plot(radius[good]/r200,dispersion[good],color='#77e4ef');axes[1].set(xlabel='r / R200c',ylabel='3-D peculiar velocity RMS [km/s]')
            fig.tight_layout();fig.savefig(stage/'radial_profiles.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
            np.savez_compressed(stage/'anatomy_data.npz',cube_mass_msun_h=cube,radial_edges_comoving_mpc_h=edges,radial_count=count,radial_mass_msun_h=shell_mass,proper_density=density,velocity_dispersion_3d_km_s=dispersion)
            payload=json.dumps({'layers':layers,'extent':extent,'r200':r200,'subhalos':subhalos},separators=(',',':')).replace('<','\\u003c')
            label=html.escape(root.name)
            page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Halo anatomy</title><style>body{background:#081321;color:#e8f3ff;font:16px system-ui;margin:0}main{max-width:1350px;margin:auto;padding:28px}.layout{display:grid;grid-template-columns:1fr 1fr;gap:28px}.frame{position:relative;aspect-ratio:1}.frame img,.frame svg{width:100%%;height:100%%;position:absolute}img{max-width:100%%}p{color:#b9cede}input{width:100%%}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #385067;text-align:left}@media(max-width:800px){.layout{grid-template-columns:1fr}}</style><main><h1>Halo anatomy</h1><p>%s · snapshot %d · host %d · z=%.5g</p><div class="layout"><section><h2>Dark-matter cutaway</h2><div class="frame"><img id="slice" alt="Dark matter mass in the selected depth slice"><svg id="overlay" viewBox="0 0 1000 1000" aria-label="Subhalo positions"></svg></div><label for="depth">Depth slice: <span id="depth-label"></span></label><input type="range" id="depth" min="0" max="%d" value="%d"><p>Each slice has equal comoving thickness. Colors share a fixed log10 mass-per-cell scale from %.3g to %.3g [Msun/h]; empty cells are black. Cyan rings mark catalogue subhalo centers within this slice. The white circle is the projected R200c boundary.</p></section><section><h2>Radial structure</h2><img src="radial_profiles.png" alt="Dark matter density and velocity dispersion profiles"><table><tr><th>M200c</th><td>%.5g Msun/h</td></tr><tr><th>R200c</th><td>%.5g comoving Mpc/h</td></tr><tr><th>Particles within 2 R200c</th><td>%d</td></tr><tr><th>Subhalo catalogue</th><td>%s; %d centers in cutaway cube</td></tr></table><p>Profiles use all saved type-1 particles in spherical shells, including unbound and neighboring material; dispersion is the 3-D mass-weighted peculiar velocity RMS about each shell mean. Shells with fewer than 10 particles omit dispersion. The cutaway is a cube extending to 2 R200c in each direction.</p></section></div></main><script>const data=%s;const s=document.querySelector('#depth'),image=document.querySelector('#slice'),overlay=document.querySelector('#overlay'),label=document.querySelector('#depth-label');function draw(){let k=+s.value,w=2*data.extent/data.layers,z0=-data.extent+k*w,z1=z0+w;image.src='slice_'+String(k).padStart(2,'0')+'.png';label.textContent=`${z0.toPrecision(4)} to ${z1.toPrecision(4)} comoving Mpc/h`;let svg=`<circle cx="500" cy="500" r="${500*data.r200/data.extent}" fill="none" stroke="white" stroke-width="2"/>`;for(const q of data.subhalos)if(q.z>=z0&&q.z<z1){let x=500*(q.x/data.extent+1),y=500*(1-q.y/data.extent);svg+=`<circle cx="${x}" cy="${y}" r="7" fill="none" stroke="#7bf8ff" stroke-width="3"><title>Subhalo ${q.row}; mass ${q.mass_msun_h.toPrecision(4)} Msun/h</title></circle>`}overlay.innerHTML=svg}s.addEventListener('input',draw);draw()</script></html>'''%(label,snapshot,host_row,1/a-1,layers-1,layers//2,low,high,float(mass200[host_row])*1e10,r200,selected,substatus,len(subhalos),payload)
            (stage/'halo_anatomy.html').write_text(page,encoding='utf-8')
            verify_product_provenance(root,provenance)
            atomic_json(stage/'anatomy_contract.json',{'schema':'halo-anatomy-v1','science_acceptance':'NOT_ACCEPTED','provenance':provenance,'redshift':1/a-1,'host_mass_msun_h':float(mass200[host_row])*1e10,'r200c_comoving_mpc_h':r200,'subhalo_status':substatus,'subhalos':subhalos,'particles_in_2r200c':selected,'display_log10_mass_per_cell_limits':[float(low),float(high)],'artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out);return out/'halo_anatomy.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
