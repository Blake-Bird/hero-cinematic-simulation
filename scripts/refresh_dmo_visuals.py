#!/usr/bin/env python3
"""Rebuild a portable, focused science atlas without modifying native run bytes.
Run from the activated Research2 environment: python scripts/refresh_dmo_visuals.py RUN [--extras]
Native figures: analyze_selected_products(RUN/analysis_source, ..., snapshots=...).
"""
from pathlib import Path
import sys, json, html, os, csv, argparse, hashlib, traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from core.analysis import (snapshot_ids, read_snapshot_header, halo_masses,
    halo_particle_counts, group_field, hmf_mass_edges)
from core.foundation import file_sha256
import yaml

EPOCHS=[(14.4,'01_JWST_z14p4','Cosmic dawn · JWST'),(12.,'02_Key_epochs/z12','Early halo assembly'),
 (10.,'02_Key_epochs/z10','High-redshift abundance'),(8.,'02_Key_epochs/z08','Reionization-era structure'),
 (6.,'02_Key_epochs/z06','Late reionization-era structure'),(3.,'02_Key_epochs/z03','Intermediate structure growth'),
 (1.,'02_Key_epochs/z01','Late assembly'),(0.,'03_Present_day_z0','Present-day fitting-function check')]
STYLE={'figure.facecolor':'#09131f','axes.facecolor':'#101f30','axes.edgecolor':'#51667a',
 'text.color':'#e7f0fa','axes.labelcolor':'#c8d9e9','xtick.color':'#abc0d4','ytick.color':'#abc0d4',
 'grid.color':'#31465a','axes.grid':True,'grid.alpha':.3,'font.family':'DejaVu Sans','font.size':10,
 'savefig.facecolor':'#09131f','legend.facecolor':'#101f30','legend.edgecolor':'#31465a'}
CSS='''*{box-sizing:border-box}body{margin:0;background:#09131f;color:#e7f0fa;font:16px/1.6 system-ui}main{max-width:1280px;margin:auto;padding:45px 28px}h1{font-size:48px;line-height:1.1;letter-spacing:-2px}h2{margin-top:42px}p{max-width:900px;color:#b3c7db}a{color:#65d9dc;text-decoration:none}nav{display:flex;gap:20px;flex-wrap:wrap;padding:15px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:20px}.card{padding:20px;border:1px solid #2a4157;background:#101f30;border-radius:14px}.card img{width:100%;border-radius:7px}video{width:100%;max-height:560px}small{color:#a4b9cf}table{border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #31465a}.tag{color:#f5c779;text-transform:uppercase;letter-spacing:2px;font-size:12px}code{color:#a6e3ca}'''

def dump(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False))
def page(path,title,body):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style><main>'+body+'</main></html>')
def save(fig,path,arrays=None):
 fig.set_facecolor('#09131f')
 for ax in fig.axes:
  ax.set_facecolor('#101f30');ax.tick_params(colors='#b3c7db')
  for spine in ax.spines.values():spine.set_color('#51667a')
  for gridline in ax.get_xgridlines()+ax.get_ygridlines():gridline.set_color('#31465a')
  legend=ax.get_legend()
  if legend:legend.get_frame().set_facecolor('#101f30');legend.get_frame().set_edgecolor('#31465a')
 for text in fig.findobj(matplotlib.text.Text):text.set_fontfamily('DejaVu Sans');text.set_color('#dce9f7')
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight',facecolor=fig.get_facecolor());fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
 if arrays:np.savez_compressed(path.with_suffix('.npz'),**arrays)
def link(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists() and not dst.is_symlink():dst.symlink_to(os.path.relpath(src,dst.parent),target_is_directory=Path(src).is_dir())
def gallery(folder):
 cards=[]
 for p in sorted(Path(folder).glob('*.png')):
  name=p.stem.replace('_',' ');pdf=p.with_suffix('.pdf');extras=f' · <a href="{pdf.name}">PDF</a>' if pdf.exists() else ''
  cards.append(f'<article class="card"><a href="{p.name}"><img loading="lazy" src="{p.name}" alt="{html.escape(name)}"></a><h3>{html.escape(name)}</h3><a href="{p.name}">Full size</a>{extras}</article>')
 return '<div class="grid">'+''.join(cards)+'</div>'

def prepare(root):
 w=root/'analysis_source';w.mkdir(exist_ok=True)
 for n in ('data','input'):link(root/n,w/n)
 if not (w/'config.yaml').exists():(w/'config.yaml').write_bytes((root/'config.yaml').read_bytes())
 if not (w/'manifest.json').exists():
  m=json.loads((root/'manifest.json').read_text());old=str(Path(m['gadget_handoff']['parameter_file']['path']).parent.parent);changes=[]
  def walk(x):
   if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
   if isinstance(x,list):return [walk(v) for v in x]
   if isinstance(x,str) and x.startswith(old+'/'):
    p=root/x[len(old)+1:]
    if p.exists():changes.append([x,str(p)]);return str(p)
   return x
  m=walk(m);m['analysis_relocation']={'original_manifest_sha256':file_sha256(root/'manifest.json'),'scope':'Path relocation only'};dump(w/'manifest.json',m);dump(w/'relocation.json',changes)
 return w

def catalogue(root,cfg):
 rows=[];cats={};particle_mass=None
 for s in snapshot_ids(root):
  header,_=read_snapshot_header(root,s);z=1/float(header['Time'])-1;m=halo_masses(root,s);n=halo_particle_counts(root,s)
  # GroupMass is in native 1e10 Msun/h units in this GADGET run.
  m=np.asarray(m if m is not None else [],float)*1e10;n=np.asarray(n if n is not None else [],int)
  if len(n)!=len(m):raise ValueError('Halo mass/count mismatch')
  if len(m) and particle_mass is None:particle_mass=float(np.median(m[n>0]/n[n>0]))
  cats[s]=(m,n);rows.append({'snapshot':s,'redshift':z,'scale_factor':float(header['Time']),'halos_32_particles':int(np.sum(n>=32)),'halos_100_particles':int(np.sum(n>=100)),'halos_300_particles':int(np.sum(n>=300)),'maximum_fof_mass_msun_h':float(m.max()) if len(m) else 0.})
 return rows,cats,particle_mass

def count_figures(folder,row,m,n,mp,cfg):
 folder.mkdir(parents=True,exist_ok=True);h=cfg['cosmology']['H0']/100;box=cfg['simulation']['box_mpc_h'];v=box**3;z=row['redshift']
 edges=np.geomspace(mp*24,max(mp*500, m.max()*1.5 if len(m) else mp*500),19)
 fig,ax=plt.subplots(figsize=(9,6))
 arrays={'edges_msun_h':edges,'fof_mass_msun_h':m,'particle_counts':n}
 for cut,c in [(32,'#8a9eb8'),(100,'#f2ba6b'),(300,'#6ce3dc')]:
  counts=np.histogram(m[n>=cut],edges)[0];x=np.sqrt(edges[:-1]*edges[1:]);y=counts/v/np.diff(np.log10(edges));mask=counts>0
  ax.step(x[mask],y[mask],where='mid',color=c,label=f'N ≥ {cut}: {np.sum(n>=cut)} halos',lw=2);arrays[f'counts_{cut}']=counts
 ax.axvline(mp*300,color='#6ce3dc',ls='--',alpha=.8,label='Configured 300-particle floor')
 ax.set(xscale='log',yscale='log',xlabel='FOF halo mass [M☉/h]',ylabel='dn/dlog₁₀M [(Mpc/h)⁻³]',title=f'Resolution audit · z = {z:.3f}')
 ax.legend(fontsize=9);fig.text(.1,.01,'32 / 100-particle curves are exploratory; convergence is not established.',fontsize=9,color='#f2ba6b');save(fig,folder/'Halo_abundance_and_resolution',arrays)
 threshold=np.geomspace(mp*32,max(mp*500,m.max()*1.3 if len(m) else mp*500),180)
 fig,ax=plt.subplots(figsize=(9,6))
 for cut,c in [(32,'#8a9eb8'),(300,'#6ce3dc')]:
  selected=np.sort(m[n>=cut]);counts=len(selected)-np.searchsorted(selected,threshold,side='left');mask=counts>0
  ax.step(threshold[mask],counts[mask]/v,where='post',color=c,label=f'N ≥ {cut}')
 # One-sided 95% upper bound for a zero-count Poisson experiment, conditional on this box.
 upper=-np.log(.05)/v
 ax.axhline(upper,color='#f2ba6b',ls=':',label=f'Zero-count 95% Poisson limit: {upper:.3g}')
 ax.axvline(mp*300,color='#6ce3dc',ls='--');ax.set(xscale='log',yscale='log',xlabel='FOF mass threshold [M☉/h]',ylabel='n(≥M) [(Mpc/h)⁻³]',title=f'Cumulative abundance · z = {z:.3f}');ax.legend(fontsize=9);save(fig,folder/'Cumulative_abundance_and_count_limit',{'mass_threshold_msun_h':threshold,'zero_count_upper95_mpc_h_cubed':upper})
 if row['snapshot']==21:
  fb=cfg['cosmology']['omega_b']/(cfg['cosmology']['omega_b']+cfg['cosmology']['omega_cdm']);eff=np.geomspace(.01,1,100);stellar=np.array([1e7,1e8,1e9])
  fig,ax=plt.subplots(figsize=(9,6))
  for st,c in zip(stellar,['#9badeb','#6ce3dc','#f2ba6b']):ax.loglog(eff,st/(fb*eff),color=c,label=f'Illustrative stellar mass {st:.0e} M☉')
  ax.axhline(mp*300/h,color='#e3a6c7',ls='--',label='300-particle floor (physical M☉)');ax.axhline(m.max()/h,color='white',ls=':',label='Largest FOF halo at z=14.4')
  ax.set(xlabel='Assumed stellar conversion efficiency ε★',ylabel='Required host mass [M☉]',title='JWST host-budget experiment · Mhost = M★ / (fb ε★)');ax.legend(fontsize=8);fig.text(.1,.01,'Illustrative baryon-budget mapping; no galaxy occupation, duty cycle, UV prediction or observed mass is assumed.',fontsize=8);save(fig,folder/'JWST_host_mass_budget',{'efficiency':eff,'illustrative_stellar_mass_msun':stellar,'baryon_fraction':fb,'h':h})

def interactive_cloud(path,xyz,box,z):
 # Embedded data + Canvas 2D projection: portable offline, no CDN or WebGL dependency.
 xyz=np.round((xyz/box-.5)*2,5);payload=json.dumps(xyz.tolist(),separators=(',',':'))
 page(path,f'Explore dark matter · z={z:.3f}',f'<a href="index.html">← Epoch figures</a><h1>Dark matter in 3D</h1><p>z={z:.4f} · {len(xyz):,} saved particles · box {box:.3f} comoving Mpc/h. Drag to rotate; scroll to zoom. Deterministic row sampling; colors encode depth, not temperature or light.</p><canvas id="view" style="width:100%;height:75vh;background:#030914;touch-action:none"></canvas><script>const P='+payload+''';const c=document.querySelector('#view'),ctx=c.getContext('2d');let a=.6,b=.3,zoom=.8,drag=false,x=0,y=0;function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;const w=c.width,h=c.height;ctx.fillStyle='#030914';ctx.fillRect(0,0,w,h);const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);for(const p of P){let X=ca*p[0]+sa*p[2],Z=-sa*p[0]+ca*p[2],Y=cb*p[1]-sb*Z;Z=sb*p[1]+cb*Z;let k=Math.min(w,h)*.38*zoom/(1+Z*.2);ctx.fillStyle=`hsla(${185+Z*25},80%,${55-Z*15}%,.48)`;ctx.fillRect(w/2+X*k,h/2-Y*k,1.5*devicePixelRatio,1.5*devicePixelRatio)}}c.onpointerdown=e=>{drag=true;x=e.clientX;y=e.clientY;c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(drag){a+=(e.clientX-x)*.006;b+=(e.clientY-y)*.006;x=e.clientX;y=e.clientY;draw()}};c.onpointerup=()=>drag=false;c.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;draw();</script>''')

def maps_and_movie(root,out,rows,selected,cfg):
 box=cfg['simulation']['box_mpc_h'];N=384;movie_dir=out/'05_Movies_and_3D';movie_dir.mkdir(exist_ok=True);numeric=out/'06_Data_and_provenance/density_frames';numeric.mkdir(parents=True,exist_ok=True)
 writer=imageio.get_writer(movie_dir/'Cosmic_web_full_86_epoch_timeline.mp4',fps=24,codec='libx264',quality=8,macro_block_size=16)
 records=[];font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),20)
 try:
  for row in rows:
   s=row['snapshot'];header,files=read_snapshot_header(root,s);grids={pr:np.zeros((N,N)) for pr in ('xy','xz','yz') if s in selected or pr=='xy'};cloud=[];sources=[]
   for f in files:
    with h5py.File(f) as hf:
     p=hf.get('PartType1')
     if p is None:continue
     pos=np.asarray(p['Coordinates'],float)/1000
     for pr,g in grids.items():
      axes={'xy':(0,1),'xz':(0,2),'yz':(1,2)}[pr];g+=np.histogram2d(pos[:,axes[0]],pos[:,axes[1]],bins=N,range=((0,box),(0,box)))[0]
     if s in selected:cloud.append(pos[::64])
    sources.append({'path':str(f.relative_to(root)) if f.is_relative_to(root) else str(f),'sha256':file_sha256(f)})
   arr=np.log10(np.maximum(gaussian_filter(grids['xy'],.9)/(128**3/N**2),1e-3)).T
   np.savez_compressed(numeric/f'snapshot_{s:03d}.npz',log10_projected_density_over_mean=arr)
   # One identical physical display range for every epoch. No interpolation of unsaved structure.
   rgb=(matplotlib.colormaps['magma'](np.clip((arr+1.3)/3.4,0,1))*255).astype('uint8')[...,:3];im=Image.fromarray(np.flipud(rgb)).resize((1024,1024),Image.Resampling.BICUBIC)
   frame=Image.new('RGB',(1024,1120),'#09131f');frame.paste(im,(0,72));d=ImageDraw.Draw(frame);d.text((25,15),f'EDE · dark matter     z = {row["redshift"]:.4f}     a = {row["scale_factor"]:.5f}',font=font,fill='#e7f0fa');d.text((25,1070),f'Saved epoch {s:03d}/085 · {box:.3f} comoving Mpc/h · fixed color scale',font=font,fill='#b3c7db')
   for _ in range(8):writer.append_data(np.asarray(frame))
   records.append({'snapshot':s,'redshift':row['redshift'],'sources':sources,'numeric_asset':str((numeric/f'snapshot_{s:03d}.npz').relative_to(out))})
   if s in selected:
    folder=out/selected[s][1];interactive_cloud(folder/'Explore_dark_matter_3D.html',np.vstack(cloud),box,row['redshift'])
    for pr,g in grids.items():
     field=np.log10(np.maximum(gaussian_filter(g,.9)/(128**3/N**2),1e-3)).T
     fig,ax=plt.subplots(figsize=(8.5,8));im=ax.imshow(field,origin='lower',extent=(0,box,0,box),cmap='magma',vmin=-1.3,vmax=2.1);ax.grid(False);ax.set(xlabel=f'{pr[0]} [comoving Mpc/h]',ylabel=f'{pr[1]} [comoving Mpc/h]',title=f'Dark-matter cosmic web · z={row["redshift"]:.3f} · {pr.upper()}');fig.colorbar(im,ax=ax,label='log₁₀(projected number density / box mean)');save(fig,folder/f'Cosmic_web_{pr.upper()}',{'projected_counts':g,'log10_density_over_mean':field,'box_mpc_h':box})
   print('density epoch',s,flush=True)
 finally:writer.close()
 dump(movie_dir/'Cosmic_web_timeline_source.json',{'schema':'focused-dmo-density-movie-v1','frames':records,'fps':24,'repeated_frames_per_epoch':8,'duration_seconds':len(rows)/3,'time_representation':'Each saved snapshot held equally; playback is not uniform in cosmic time. No particle interpolation.','display_range_log10':[-1.3,2.1],'grid':N,'kernel':'NGP histogram then 0.9 pixel Gaussian display blur','movie_sha256':file_sha256(movie_dir/'Cosmic_web_full_86_epoch_timeline.mp4')})


def evolution(out,rows,cfg,mp):
 f=out/'04_Evolution';f.mkdir(exist_ok=True);a=np.array([r['scale_factor'] for r in rows]);z=np.array([r['redshift'] for r in rows]);v=cfg['simulation']['box_mpc_h']**3
 fig,axes=plt.subplots(1,2,figsize=(13,5.6))
 for cut,c in [(32,'#8a9eb8'),(100,'#f2ba6b'),(300,'#6ce3dc')]:axes[0].plot(a,[r[f'halos_{cut}_particles'] for r in rows],color=c,label=f'N ≥ {cut}')
 axes[0].set(xscale='log',yscale='log',xlabel='Scale factor a',ylabel='FOF host count',title='Halo emergence · all 86 saved epochs');axes[0].legend()
 maximum=np.array([r['maximum_fof_mass_msun_h'] for r in rows]);mask=maximum>0;axes[1].loglog(a[mask],maximum[mask],color='#6ce3dc');axes[1].axhline(300*mp,ls='--',color='#f2ba6b',label='300-particle floor');axes[1].axvline(1/15.4,ls=':',color='#e3a6c7',label='z=14.4');axes[1].set(xlabel='Scale factor a',ylabel='Maximum FOF mass [M☉/h]',title='Growth of the largest host');axes[1].legend();fig.tight_layout();save(fig,f/'Halo_emergence_and_maximum_mass',{'scale_factor':a,'redshift':z,'maximum_fof_mass_msun_h':maximum})
 with (out/'06_Data_and_provenance/All_86_epochs.csv').open('w') as fh:
  writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def extras(work,out,selected,cats):
 from core.velocity_flow import make_velocity_flow
 from core.halo_anatomy import make_halo_anatomy
 from core.allsky_map import make_allsky_map
 from core.cosmic_flythrough import make_cosmic_flythrough
 from core.filament_flythrough import make_filament_flythrough
 from core.void_explorer import make_void_explorer
 failures={};done={};dest=out/'05_Movies_and_3D/New_visual_experiences';dest.mkdir(parents=True,exist_ok=True)
 def run(name,op):
  if (dest/name).exists():done[name]='EXISTS';return
  try:done[name]=str(op(dest/name));print('visual complete',name,flush=True)
  except Exception as e:failures[name]={'type':type(e).__name__,'reason':str(e)};print('visual unavailable',name,str(e),flush=True)
  dump(dest/'status.json',{'completed':done,'unavailable':failures})
 for s in selected:
  run(f'Velocity_flow_z{selected[s][0]:g}',lambda p,s=s:make_velocity_flow(work,p,snapshot=s,grid=24,memory_gib=2.))
 for s in (21,41,85):
  m,n=cats[s];host=int(np.argmax(m))
  run(f'Halo_anatomy_z{selected[s][0]:g}',lambda p,s=s,host=host:make_halo_anatomy(work,p,snapshot=s,host_row=host,memory_gib=2.))
 for s in (21,85):
  m,n=cats[s];host=int(np.argmax(m));pos=np.asarray(group_field(work,s,['GroupPos']))[host]/1000;box=5.517622769503022
  run(f'All_sky_z{selected[s][0]:g}',lambda p,s=s:make_allsky_map(work,p,snapshot=s,max_particles=150000,memory_gib=2.))
  run(f'Fly_through_z{selected[s][0]:g}',lambda p,s=s,host=host,pos=pos:make_cosmic_flythrough(work,p,snapshot=s,host_row=host,start=(pos+box*.25).tolist(),frames=120,max_particles=150000,memory_gib=2.))
  run(f'Filament_fly_through_z{selected[s][0]:g}',lambda p,s=s,host=host:make_filament_flythrough(work,p,snapshot=s,host_row=host,frames=120,max_particles=100000,memory_gib=2.))
 run('Void_explorer',lambda p:make_void_explorer(work,p,grid=40,frames=64,max_particles=100000,memory_gib=2.))
 return done,failures


def native_links(work,out,selected):
 bundles=sorted((work/'results').glob('analysis_products_*/analysis_products_summary.json'))
 if not bundles:return None
 b=bundles[-1].parent;summary=json.loads(bundles[-1].read_text())
 for s,(_,directory,_) in selected.items():
  folder=out/directory/'Detailed_native_figures';folder.mkdir(parents=True,exist_ok=True)
  groups={}
  for p in b.rglob('*'):
   if not p.is_file():continue
   if f'_{s:03d}' in p.stem and p.suffix in ('.png','.pdf','.svg','.npz','.csv','.json'):
    name=p.name.lower()
    topic=('01_Halo_abundance' if any(k in name for k in ('hmf','analytic','finder')) else
     '02_Power_and_clustering' if any(k in name for k in ('power','pk_','correlation','bias')) else
     '03_Halo_structure' if any(k in name for k in ('concentration','profile','shape','spin')) else
     '04_Kinematics' if any(k in name for k in ('velocity','kinematic')) else
     '05_Spatial_and_catalogues')
    link(p,folder/topic/p.name);groups[topic]=folder/topic
  body=f'<a href="../index.html">← Epoch overview</a><h1>Detailed native analysis</h1><p>Current repository measurements and figure recipes. Missing or unsupported products remain recorded in the source summary.</p><div class="grid">'
  for topic,subfolder in sorted(groups.items()):
   title=topic[3:].replace('_',' ');page(subfolder/'index.html',title,'<a href="../index.html">← All topics</a><h1>'+title+'</h1>'+gallery(subfolder))
   body+=f'<a class="card" href="{topic}/index.html"><h3>{title}</h3><p>{len(list(subfolder.glob("*.png")))} figures · PNG, PDF and source arrays</p></a>'
  page(folder/'index.html','Detailed native figures',body+'</div>')
 link(b,out/'06_Data_and_provenance/Current_native_analysis_bundle')
 for p in (b/'paper_figures').glob('*'):
  if p.is_file():link(p,out/'04_Evolution'/p.name)
 for p in b.glob('*evolution*'):
  if p.is_file():link(p,out/'04_Evolution'/p.name)
 for hist in sorted((work/'history_workspace/results').glob('analysis_products_*/analysis_products_summary.json'))[-1:]:
  link(hist.parent,out/'06_Data_and_provenance/Full_timeline_history_bundle')
  for p in hist.parent.iterdir():
   if p.is_file() and p.suffix in ('.png','.pdf','.svg','.html','.json','.npz') and not p.name.startswith(('analysis_','figure_review','snapshot_')):
    link(p,out/'04_Evolution'/p.name)
 story=out/'04_Evolution/Halo_merger_story'
 if story.exists():
  # Present the branching tree as the merger graph. Retain the former event
  # scatter under its accurate name and preserve the original sealed bundle.
  old=out/'04_Evolution/halo_merger_histories.png'
  if old.is_symlink():link(old.resolve(),out/'04_Evolution/halo_merger_mass_ratios.png')
  for p in story.glob('halo_merger_histories*'):
   target=out/'04_Evolution'/p.name
   if target.is_symlink():target.unlink()
   if not target.exists():link(p,target)
  entry=out/'04_Evolution/merger_tree.html'
  if entry.is_symlink():entry.unlink()
  entry.write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=Halo_merger_story/index.html"><a href="Halo_merger_story/index.html">Open the linked halo tree, map and movie</a>')
 return b


def fit_assessment(root,work,out,cfg,cats,selected):
 from core.hmf_fitting_functions import _linear_inputs,binned_predictions,LABELS
 from core.analysis import hmf_mass_edges
 v=cfg['simulation']['box_mpc_h']**3;report={}
 for s in (21,85):
  m,n=cats[s];edges=hmf_mass_edges(cfg);counts=np.histogram(m[n>=300],edges)[0];width=np.diff(np.log10(edges));centers=np.sqrt(edges[1:]*edges[:-1]);z=selected[s][0];folder=out/selected[s][1]
  try:
   k,p,om,prov=_linear_inputs(work,cfg,'ede',z);fits,sigma,slope,covered=binned_predictions(k,p,om,edges,z)
  except Exception as e:report[str(s)]={'status':'UNAVAILABLE','reason':str(e)};continue
  fig,axes=plt.subplots(2,1,figsize=(9,9),sharex=True,gridspec_kw={'height_ratios':[2,1]});supported=(centers>=300*np.median(m[n>0]/n[n>0]))&covered;positive=counts>0
  low=np.where(positive,.5*chi2.ppf(.158655,2*np.maximum(counts,1)),0);high=.5*chi2.ppf(.841345,2*(counts+1));rate=counts/v/width
  axes[0].errorbar(centers[positive],rate[positive],yerr=np.array([counts-low,high-counts])[:,positive]/v/width[positive],fmt='o',color='white',label='FOF ≥300 particles; 68% Poisson intervals')
  metrics={};col=plt.cm.Set2(np.linspace(0,1,len(fits)))
  for (key,fit),c in zip(fits.items(),col):
   y=fit['dn_dlog10M'];mu=y*v*width;mask=supported&np.isfinite(y)&(y>0);axes[0].loglog(centers[mask],y[mask],color=c,label=LABELS[key],lw=1.5)
   use=mask&(counts>=20);ratio=counts[use]/mu[use];axes[1].semilogx(centers[use],ratio,'o-',ms=3,color=c)
   metrics[key]={'bins_with_at_least_20_halos':int(use.sum()),'median_measured_over_fit':float(np.median(ratio)) if len(ratio) else None,'minimum_measured_over_fit':float(ratio.min()) if len(ratio) else None,'maximum_measured_over_fit':float(ratio.max()) if len(ratio) else None,'table_range_bins':int((use&fit['table_range_mask']).sum())}
  axes[0].set(yscale='log',ylabel='dn/dlog₁₀M [(Mpc/h)⁻³]',title=f'EDE FOF abundance versus six CDM reference fits · z={z:g}');axes[0].legend(fontsize=8)
  axes[1].axhline(1,color='white',ls=':');axes[1].axhspan(.8,1.2,color='#6ce3dc',alpha=.1);axes[1].set(xlabel='FOF halo mass [M☉/h]',ylabel='Measured / reference')
  if not np.any(counts>=20):axes[1].text(.5,.5,'No bins with ≥20 resolved halos',ha='center',transform=axes[1].transAxes)
  fig.text(.12,.01,'CDM fitting functions with saved EDE linear inputs: diagnostic extrapolation, not EDE calibration.\nShaded ±20% is a visual guide; sample variance and convergence errors are not included.',fontsize=8);fig.tight_layout(rect=(0,.055,1,1));save(fig,folder/'Six_fitting_functions_and_residuals',{'counts':counts,'edges_msun_h':edges,**{key:fit['dn_dlog10M'] for key,fit in fits.items()}})
  report[str(s)]={'redshift':z,'metrics':metrics,'resolved_halos':int(np.sum(n>=300)),'linear_input_provenance':prov,'scope':'CDM reference fits evaluated with EDE P(k); no LCDM simulation, EDE calibration, finite-volume correction or convergence acceptance'}
 dump(out/'06_Data_and_provenance/Fitting_function_assessment.json',report)
 return report


def navigation(root,work,out,rows,cats,mp,cfg,selected):
 b=native_links(work,out,selected);report=fit_assessment(root,work,out,cfg,cats,selected);movie=out/'05_Movies_and_3D';legacy=movie/'Original_exports';legacy.mkdir(exist_ok=True)
 media=[]
 for p in sorted((root/'results').rglob('*')):
  if p.suffix not in ('.mp4','.blend') or any(x.startswith('.') for x in p.relative_to(root).parts):continue
  label=p.parent.name+'__'+p.name;link(p,legacy/label);media.append((label,p))
  link(p.parent,movie/'Original_source_bundles'/p.parent.name)
 for p in (work/'history_workspace/results/movies').glob('*/*.mp4'):
  link(p,movie/'New_visual_experiences/Full_timeline_halo_growth'/p.name)
  link(p.parent,movie/'New_visual_experiences/Full_timeline_halo_growth/Source_bundle')
 all_movies=movie/'All_movies';all_movies.mkdir(exist_ok=True)
 for label,p in media:
  if p.suffix=='.mp4':link(p,all_movies/label)
 for p in (movie/'New_visual_experiences').rglob('*.mp4'):
  if any(parent.is_symlink() for parent in p.parents if parent!=movie):continue
  name='__'.join(p.relative_to(movie/'New_visual_experiences').parts)
  old_flight=p.relative_to(movie/'New_visual_experiences').parts[0].startswith(('Fly_through','Filament_fly_through'))
  if (movie/'Readable_flights').exists() and old_flight:
   target=all_movies/name
   if target.is_symlink():target.unlink()
   continue
  link(p,all_movies/name)
 link(movie/'Cosmic_web_full_86_epoch_timeline.mp4',all_movies/'Cosmic_web_full_86_epoch_timeline.mp4')
 moviebody='<a href="../START_HERE.html">← Atlas home</a><h1>Movies & 3D library</h1><p>The refreshed cosmic-web timeline uses all 86 saved epochs. Original exports retain their original generation and provenance. All links are relative; move the whole run to keep them working.</p><h2>Full cosmic history</h2><video controls preload="metadata" src="Cosmic_web_full_86_epoch_timeline.mp4"></video><p>Equal hold time per snapshot; fixed density color scale. <a href="Cosmic_web_timeline_source.json">Source and timing record</a>.</p><h2>Original movies and Blender scenes</h2><div class="grid">'
 recommended='<h2>Start with the rebuilt visual stories</h2><div class="grid">'
 story=out/'04_Evolution/Halo_merger_story'
 if (story/'index.html').exists():
  recommended+='<article class="card"><img src="../04_Evolution/Halo_merger_story/Preview_085.png" alt="Labelled halos and branching merger tree"><h3>Which halo is which?</h3><p>Click a tree node to find its exact catalogue ID, saved map position, and matching film time.</p><a href="../04_Evolution/Halo_merger_story/index.html">Open the linked halo story</a></article>'
  link(story/'Halo_identity_and_merger_tree.mp4',all_movies/'Halo_identity_and_merger_tree.mp4')
 for flight in sorted((movie/'Readable_flights').glob('*')):
  if not flight.is_dir():continue
  movies=list(flight.rglob('cosmic_flythrough.mp4'))
  if not movies:continue
  film=movies[0];preview=film.parent/'preview.png';name=flight.name.replace('_',' ')
  recommended+=f'<article class="card"><h3>{name}</h3><video controls preload="metadata" poster="{preview.relative_to(movie)}" src="{film.relative_to(movie)}"></video><p>Smoothed density, sampled particles, labelled target, and camera overview. Contrast is fixed within each film and adjusted between films; brightness is not a cross-epoch measurement.</p><a href="{film.relative_to(movie)}">Download MP4</a> · <a href="{film.parent.relative_to(movie)}/flythrough_contract.json">Source and camera record</a></article>'
  link(film,all_movies/(flight.name+'.mp4'))
 recommended+='</div>'
 moviebody=moviebody.replace('<h2>Full cosmic history</h2>',recommended+'<h2>Full cosmic history</h2>',1)
 for label,p in media:
  url='Original_exports/'+label;moviebody+=f'<article class="card"><h3>{html.escape(p.stem.replace("_"," "))}</h3><small>{html.escape(p.parent.name)}</small>'
  moviebody+=(f'<video controls preload="none" src="{url}"></video>' if p.suffix=='.mp4' else '')+f'<p><a href="{url}">Open / download {p.suffix}</a> · <a href="Original_source_bundles/{p.parent.name}/">Full source bundle</a></p></article>'
 moviebody+='</div><p><a href="All_movies/">Open the folder containing every movie</a></p><h2>New visual experiences</h2><div class="grid">'
 fresh=movie/'Refreshed_Blender_timeline'
 if (fresh/'Dark_matter_full_saved_timeline.blend').exists():
  moviebody+='<article class="card"><img src="Refreshed_Blender_timeline/Preview_z0.png" alt="New saved-epoch Blender scene"><h3>Refreshed Blender timeline</h3><p>86 embedded saved epochs, exact sampled IDs, no external geometry dependencies.</p><p><a href="Refreshed_Blender_timeline/Dark_matter_full_saved_timeline.blend">Open Blender scene</a> · <a href="Refreshed_Blender_timeline/README.md">Viewing guide</a> · <a href="Refreshed_Blender_timeline/scene_source.json">Sources</a></p></article>'
 d=movie/'New_visual_experiences'
 if d.exists():
  for p in sorted(d.iterdir()):
   if not p.is_dir() or p.name.startswith('.'):continue
   if (movie/'Readable_flights').exists() and p.name.startswith(('Fly_through','Filament_fly_through')):continue
   assets=[q for q in p.rglob('*') if q.suffix in ('.html','.mp4','.png') and not q.name.startswith('slice_')]
   moviebody+='<article class="card"><h3>'+html.escape(p.name.replace('_',' '))+'</h3>'
   for q in assets:
    if any(parent.is_symlink() for parent in q.parents if parent!=movie):continue
    moviebody+=(f'<video controls preload="none" src="{q.relative_to(movie)}"></video>' if q.suffix=='.mp4' else '')+f'<p><a href="{q.relative_to(movie)}">{html.escape(q.stem.replace("_"," "))}</a></p>'
   moviebody+='</article>'
 moviebody+='</div>';page(movie/'index.html','Movies & 3D library',moviebody)
 cards=[]
 for s,(target,directory,reason) in selected.items():
  folder=out/directory;row=next(r for r in rows if r['snapshot']==s);m,n=cats[s];back=os.path.relpath(out/'START_HERE.html',folder)
  body=f'<a href="{back}">← Atlas home</a><p class="tag">EDE · saved snapshot {s:03d}</p><h1>z = {target:g}</h1><p>{reason}. Actual saved redshift {row["redshift"]:.9f}; {len(m):,} FOF hosts, {np.sum(n>=300):,} at ≥300 particles.</p><nav><a href="Explore_dark_matter_3D.html">Explore particles in 3D</a><a href="Detailed_native_figures/index.html">Detailed native figures</a></nav>'
  if s==21:body+='<p>The configured 300-particle threshold contains zero halos here. Lower-count halos are exploratory. This box cannot establish the abundance of rare luminous JWST galaxies; a zero count is not evidence against their existence. The host-budget figure is an explicit assumption experiment.</p><p>MoM-z14 is at z=14.44, close to this snapshot: <a href="https://astro.theoj.org/article/156033-a-cosmic-miracle-a-remarkably-luminous-galaxy-at-_z_-sub-spec-sub-14-44-confirmed-with-jwst">discovery paper</a>. JADES-GS-z14-0 has an ALMA redshift z=14.1793: <a href="https://arxiv.org/abs/2409.20549">line measurement</a>. Neither is an exact match to z=14.4.</p>'
  body+=gallery(folder);page(folder/'index.html',f'EDE · z={target:g}',body)
  cards.append(f'<a class="card" href="{directory}/index.html"><img src="{directory}/Cosmic_web_XY.png" alt="Cosmic web at z {target}"><h3>z = {target:g}</h3><p>{reason}</p><small>{np.sum(n>=300):,} resolved hosts</small></a>')
 story_feature='<a class="card" href="Halo_merger_story/index.html"><h2>Explore the branching halo tree</h2><p>Shared branch colors, exact IDs, labelled maps and a synchronized movie.</p></a>' if (out/'04_Evolution/Halo_merger_story/index.html').exists() else ''
 page(out/'04_Evolution/index.html','Evolution', '<a href="../START_HERE.html">← Atlas home</a><h1>Evolution across 86 epochs</h1><p>The dense saved schedule is retained for trends and movies. Detailed mass-function panels are reserved for eight selected epochs.</p>'+story_feature+gallery(out/'04_Evolution'))
 zero=report.get('85',{});st=zero.get('metrics',{}).get('sheth_tormen',{});ratio=st.get('median_measured_over_fit');finding=f'At z=0 the median measured / Sheth–Tormen ratio is {ratio:.3f} across {st["bins_with_at_least_20_halos"]} bins with ≥20 halos.' if ratio is not None else 'See the z=0 residual panels and numerical assessment for the fitting-function check.'
 body='<p class="tag">Research atlas · dark matter · early dark energy</p><h1>A small box.<br>A complete cosmic history.</h1><p>Rebuilt figures, eight carefully selected epochs, and a single library for movies and 3D exploration.</p><nav><a href="01_JWST_z14p4/index.html">JWST at z=14.4</a><a href="03_Present_day_z0/index.html">z=0 fit checks</a><a href="04_Evolution/index.html">Evolution</a><a href="05_Movies_and_3D/index.html">Movies & 3D</a><a href="06_Data_and_provenance/index.html">Data & methods</a><a href="07_Future_LCDM/README.md">Future ΛCDM comparison</a></nav>'
 body+=f'<div class="card"><h2>What this run says</h2><p>At z=14.4: 70 FOF halos, zero above the configured 300-particle threshold. {html.escape(finding)} CDM reference fits evaluated with EDE inputs remain diagnostic extrapolations.</p></div><h2>Explore the important epochs</h2><div class="grid">'+''.join(cards)+'</div>'
 body+='<h2>Keep the full journey</h2><div class="grid"><a class="card" href="04_Evolution/index.html"><h3>86-epoch evolution</h3><p>Halo emergence, mass growth and exact epoch tables.</p></a><a class="card" href="05_Movies_and_3D/index.html"><h3>The screening room</h3><p>New timeline, fly-throughs, original films and Blender scenes.</p></a></div>'
 if (out/'04_Evolution/Halo_merger_story/index.html').exists():
  body=body.replace('<h2>Explore the important epochs</h2>','<a class="card" style="display:block;margin-top:24px" href="04_Evolution/Halo_merger_story/index.html"><h2>Follow a halo through its merger tree</h2><p>Click a branch. Find it in the simulation. Watch the same labels in the film.</p></a><h2>Explore the important epochs</h2>')
 body+='<p><small>For reliable video playback and seeking, double-click Open_atlas.command in the run folder. The atlas also opens directly as local HTML.</small></p>'
 page(out/'START_HERE.html','EDE research atlas',body)
 methods=f'<a href="../START_HERE.html">← Atlas home</a><h1>Data & methods</h1><p>Box {cfg["simulation"]["box_mpc_h"]:.6f} comoving Mpc/h; 128³ particles; particle mass {mp:.5g} M☉/h; 300-particle floor {mp*300:.5g} M☉/h. FOF b=0.2. These masses are not M200c.</p><p>{html.escape(finding)}</p><p>Bin counts are compared with bin-integrated fitting functions using the saved AxiCLASS spectrum and growth background. Poisson intervals omit sample variance. No precision-fit or convergence acceptance is claimed.</p><nav><a href="All_86_epochs.csv">All epochs CSV</a><a href="Fitting_function_assessment.json">Fit residual numbers</a><a href="build_manifest.json">Build and source record</a><a href="Current_native_analysis_bundle/analysis_products_summary.json">Native product status</a><a href="../../analysis_source/relocation.json">Relocation record</a></nav>'
 methods+='<nav><a href="Product_status_and_checks.json">Product status and omissions</a><a href="Link_audit.json">Link audit</a><a href="Movie_decode_checks.json">Movie checks</a></nav><h2>Present-day reference-fit ratios</h2><table><tr><th>Reference</th><th>Median measured / fit</th><th>Bins ≥20 halos</th><th>Within tabulated range</th></tr>'
 for name,values in zero.get('metrics',{}).items():
  ratio=values['median_measured_over_fit'];methods+=f'<tr><td>{name.replace("_"," ")}</td><td>{ratio:.3f}</td><td>{values["bins_with_at_least_20_halos"]}</td><td>{values["table_range_bins"]}</td></tr>' if ratio is not None else ''
 methods+='</table><p>Out-of-range curves are extrapolations. Even in-range CDM fits are not calibrated for this EDE cosmology.</p>'
 page(out/'06_Data_and_provenance/index.html','Data & methods',methods)
 future=out/'07_Future_LCDM';future.mkdir(exist_ok=True);(future/'README.md').write_text('# Future ΛCDM comparison\n\nNo ΛCDM run is present. No EDE/ΛCDM ratio or enhancement is measured.\n\nFor a controlled comparison, match the comoving box, particle load, initial random phases, output epochs, halo finder, mass convention and particle cuts. Record the cosmological parameter differences explicitly. Separate matched-phase differences from changes in cosmological parameters; use independent seeds or larger volumes to measure sample variance.\n\nPriority epochs: z=14.4,12,10,8,6,3,1,0. Keep all 86 outputs for the movies and assembly histories. Check completeness and physical mass units before comparing halo counts.\n')
 (root/'START_HERE.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=analysis/START_HERE.html"><a href="analysis/START_HERE.html">Open the EDE research atlas</a>')
 (root/'README_ANALYSIS.md').write_text('# EDE research atlas\n\nOpen START_HERE.html or analysis/START_HERE.html.\n\n- analysis/01_JWST_z14p4: priority cosmic-dawn diagnostics\n- analysis/02_Key_epochs: z=12,10,8,6,3,1\n- analysis/03_Present_day_z0: fitting-function residuals\n- analysis/04_Evolution: all saved epochs\n- analysis/05_Movies_and_3D: original and refreshed movies and interactive views\n- analysis/06_Data_and_provenance: arrays, tables and source records\n- analysis/07_Future_LCDM: controlled-comparison plan; no ΛCDM data yet\n\nOriginal data/, input/, config.yaml, manifest.json and results/ remain authoritative and are retained. analysis_source/ contains relocated analysis metadata and symlinks to original particles and inputs. Detailed figures use the current repository; original exports are explicitly identified in the library. Move the entire run together so relative symlinks remain valid.\n\nRebuild with the activated Research2 environment: python scripts/refresh_dmo_visuals.py RUN\nNative figures: add --native. New optional experiences: add --extras. For synchronized tree/film outputs and readable flights: scripts/rebuild_halo_story.py RUN. For reliable video seeking: scripts/serve_results.py RUN. Refresh links after native analysis: add --navigation-only.\n')
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--extras',action='store_true');p.add_argument('--native',action='store_true',help='Recompute the full native figure suite at the eight priority epochs');p.add_argument('--navigation-only',action='store_true');a=p.parse_args();root=a.run.resolve();work=prepare(root);out=root/'analysis';out.mkdir(exist_ok=True);(out/'06_Data_and_provenance').mkdir(exist_ok=True);cfg=yaml.safe_load((root/'config.yaml').read_text());plt.rcParams.update(STYLE)
 rows,cats,mp=catalogue(root,cfg);selected={}
 for z,d,t in EPOCHS:
  row=min(rows,key=lambda r:abs(r['redshift']-z))
  if abs(row['redshift']-z)>.002:raise ValueError('Requested priority epoch missing')
  selected[row['snapshot']]=(z,d,t)
 if a.native:
  from core.analysis import analyze_selected_products
  analyze_selected_products(work,['hmf','finder','concentration','pk','correlation','bias','kinematics','spatial','subhmf','spin','profiles','shape','subhalos'],include_pdf_report=False,snapshots=list(selected))
 if not a.navigation_only:
  for s,(_,d,_) in selected.items():count_figures(out/d,next(r for r in rows if r['snapshot']==s),*cats[s],mp,cfg)
  evolution(out,rows,cfg,mp);maps_and_movie(root,out,rows,selected,cfg)
 if a.extras:extras(work,out,selected,cats)
 report=navigation(root,work,out,rows,cats,mp,cfg,selected)
 assets=[{'path':str(q.relative_to(out)),'size_bytes':q.stat().st_size,'sha256':file_sha256(q)} for q in out.rglob('*') if q.is_file() and not q.is_symlink() and q.name!='build_manifest.json' and not any(x.is_symlink() for x in q.parents if x!=out)]
 dump(out/'06_Data_and_provenance/build_manifest.json',{'schema':'focused-dmo-atlas-v1','original_manifest_sha256':file_sha256(root/'manifest.json'),'original_config_sha256':file_sha256(root/'config.yaml'),'script_sha256':file_sha256(__file__),'key_epochs':{str(s):v for s,v in selected.items()},'particle_mass_msun_h':mp,'snapshots':rows,'artifacts':assets,'scientific_scope':'DMO EDE diagnostics; no galaxy emission prediction, convergence acceptance or LCDM comparison','fitting_function_assessment':report})
 print('Atlas ready:',out/'START_HERE.html',flush=True)
if __name__=='__main__':main()
