"""Compare a matched pair using saved measurements; no particle evolution."""
import argparse,csv,gc,html,json,os,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from core.analysis import (compare_initial_ic_phases,_read_cross_run_halo_catalogue,
    _mutual_best_halo_matches,snapshot_ids,read_snapshot_header,poisson_rate_interval,
    gadget_matter_power_spectrum)
from core.foundation import atomic_json,file_sha256
from core.figures import apply_plot_style,export_figure
from core.resources import memory_limit_gb
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--n',type=int,choices=(32,512),required=True);a=p.parse_args()
state=Path(os.environ.get('RESEARCH2_STATE_ROOT', Path.home()/'ede-runtime')).expanduser().resolve()
root=state/'campaigns'/os.environ.get('CAMPAIGN','matched40')/f'n{a.n}'
runs=[root/'runs/lcdm',root/'runs/ede'];cfg=[yaml.safe_load((r/'config.yaml').read_text()) for r in runs]
# Physical/numerical matching is stricter than merely reusing a random seed.
keys=('particles_per_dimension','random_seed','start_redshift','snapshot_redshift_list','pm_grid','asmth','multipole_order','max_size_timestep_ln_a','err_tol_force_acc','err_tol_int_accuracy','err_tol_theta','lpt_order_dm','ic_engine_dm','physics')
for k in keys:
 if cfg[0]['simulation'][k]!=cfg[1]['simulation'][k]:raise SystemExit('Pair changes numerical control '+k)

h0 = float(cfg[0]['cosmology'].get('hubble', float(cfg[0]['cosmology'].get('H0', 67.81))/100.0))
h1 = float(cfg[1]['cosmology'].get('hubble', float(cfg[1]['cosmology'].get('H0', 74.83))/100.0))
phys_box0 = float(cfg[0]['simulation']['box_mpc_h']) / h0
phys_box1 = float(cfg[1]['simulation']['box_mpc_h']) / h1
box_equal = np.isclose(float(cfg[0]['simulation']['box_mpc_h']), float(cfg[1]['simulation']['box_mpc_h']), rtol=1e-4)
phys_box_equal = np.isclose(phys_box0, phys_box1, rtol=1e-3)
if not (box_equal or phys_box_equal):
 raise SystemExit('Pair changes numerical and physical box size')

phys_soft0 = float(cfg[0]['simulation']['softening_kpc_h']) / h0
phys_soft1 = float(cfg[1]['simulation']['softening_kpc_h']) / h1
soft_equal = np.isclose(float(cfg[0]['simulation']['softening_kpc_h']), float(cfg[1]['simulation']['softening_kpc_h']), rtol=1e-4)
phys_soft_equal = np.isclose(phys_soft0, phys_soft1, rtol=1e-3)
if not (soft_equal or phys_soft_equal):
 raise SystemExit('Pair changes physical force softening')

# Detect whether this is a published best-fit comparison or controlled experiment
cosmo_diffs = [k for k in ('H0','omega_b','omega_cdm','A_s','n_s','tau_reio') if k in cfg[0]['cosmology'] and k in cfg[1]['cosmology'] and not np.isclose(float(cfg[0]['cosmology'][k]), float(cfg[1]['cosmology'][k]), rtol=1e-4)]
if not cosmo_diffs:
 scope = 'Controlled EDE experiment: identical background/primordial parameters (H0, Omega_b, Omega_cdm, As, ns), scalar EDE field changes.'
else:
 vol_label = f"matched physical comoving volume ({phys_box0:.2f} Mpc; LCDM {cfg[0]['simulation']['box_mpc_h']} Mpc/h, EDE {cfg[1]['simulation']['box_mpc_h']:.4f} Mpc/h)" if phys_box_equal else f"volume {cfg[0]['simulation']['box_mpc_h']} Mpc/h"
 scope = f"Published best-fit JWST/EDE vs LCDM comparison (MNRAS 2024 Table 1); independently fitted cosmologies with matched initial phases, random seed ({cfg[0]['simulation']['random_seed']}), {vol_label}, and N={cfg[0]['simulation']['particles_per_dimension']}^3 particles."

for k in ('N_ur','T_cmb','N_ncdm','m_ncdm_eV'):
 if k in cfg[0]['cosmology'] and k in cfg[1]['cosmology']:
  if not np.isclose(float(cfg[0]['cosmology'][k]), float(cfg[1]['cosmology'][k]), rtol=1e-4):raise SystemExit('Pair changes neutrino/CMB control '+k)

if cfg[0]['analysis']['hmf_mass_definition'] != cfg[1]['analysis']['hmf_mass_definition']:raise SystemExit('Pair changes HMF convention hmf_mass_definition')
if cfg[0]['analysis']['hmf_mass_definition']!='fof':raise SystemExit('FOF mass definition required by the fit comparison')
if [c['cosmology']['model'] for c in cfg]!=['lcdm','ede']:raise SystemExit('Model order must be LCDM then EDE')
out=root/'comparisons'/('pair_'+uuid.uuid4().hex[:10]);out.mkdir(parents=True,exist_ok=False)
report={'status':'RUNNING','runs':[str(r) for r in runs],'source_configs':[file_sha256(r/'config.yaml') for r in runs],
 'matched_controls':list(keys),'epoch_reports':[],'science_acceptance':'NOT_ESTABLISHED',
 'uncertainty':'Individual HMF bands are central 68.27% Garwood Poisson intervals. Any ratio error is an independent-count approximation; phase-matched covariance, cosmic variance and convergence errors are not estimated.',
 'comparison_scope':scope, 'units':{'mass':'Msun/h','physical_mass':'Msun (divide Msun/h by h: h_LCDM=0.6781, h_EDE=0.7483)','box_size':'comoving Mpc' if phys_box_equal else 'comoving Mpc/h','volume':'Mpc^3' if phys_box_equal else '(Mpc/h)^3','hmf':'Mpc^-3' if phys_box_equal else '(h/Mpc)^3'}}
atomic_json(out/'report.json',report)
apply_plot_style(cfg[0]['visualization'])
links=[]
def save(fig,stem,meta):
 fig.tight_layout();export_figure(fig,out/(stem+'.png'),metadata=dict(meta,science_acceptance='NOT_ESTABLISHED'))
 links.append(stem+'.png')
def write_csv(path,rows,fields):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
try:
 phase=compare_initial_ic_phases(*runs);report['initial_phase_check']=phase;atomic_json(out/'report.json',report)
 if phase.get('status')!='PASS':raise ValueError('Measured initial phase agreement did not pass; paired halo matching is blocked')
 snapshots=[snapshot_ids(r) for r in runs]
 if snapshots[0]!=snapshots[1]:raise ValueError('Saved epoch indices differ')
 exports=[]
 for model,run in zip(('lcdm','ede'),runs):
  candidates=[x for x in (root/'exports').glob(model+'*') if (x/'metadata.json').is_file()]
  if not candidates:raise FileNotFoundError('Missing measured export for '+model)
  exp=max(candidates,key=lambda x:x.stat().st_mtime_ns);meta=json.loads((exp/'metadata.json').read_text())
  if Path(meta['source_run']).resolve()!=run.resolve() or file_sha256(exp/'hmf.csv')!=meta['measurement_sha256']:raise ValueError('Measured export provenance mismatch')
  with (exp/'hmf.csv').open(newline='') as f:exports.append(list(csv.DictReader(f)))
  report.setdefault('source_exports',[]).append({'path':str(exp),'metadata_sha256':file_sha256(exp/'metadata.json'),'table_sha256':meta['measurement_sha256']})
 # Global pair colour limits come from both runs, shared across every epoch/projection.
 map_files=[]
 for run in runs:
  workflows=sorted((run/'results').glob('workflow_*.json'));w=json.loads(workflows[-1].read_text())
  maps={}
  for img in w['artifacts'].get('maps',[]):
   path=Path(img);numeric=path.with_suffix('.hdf5')
   if numeric.is_file() and not path.stem.endswith('_filaments'):
    meta=json.loads(path.with_suffix('.json').read_text());maps[(int(meta['snapshot']),meta['projection'])]=(numeric,meta)
  map_files.append(maps)
 common_maps=sorted(set(map_files[0])&set(map_files[1]))
 scale=[]
 for pair in map_files:
  for path,meta in pair.values():
   with h5py.File(path) as f:
    x=f['log10_display'][:];x=x[np.isfinite(x)]
   if x.size:scale.extend(np.percentile(x,[2,99.5]).tolist())
 if common_maps and not scale:raise ValueError('No finite paired density maps')
 lo,hi=(min(scale),max(scale)) if scale else (0,1)
 delta_limit=0.
 for key in common_maps:
  arrays=[]
  for pair in map_files:
   with h5py.File(pair[key][0]) as f:arrays.append(f['measured'][:])
  positive=(arrays[0]>0)&(arrays[1]>0)
  if np.any(positive):delta_limit=max(delta_limit,float(np.percentile(np.abs(np.log10(arrays[1][positive]/arrays[0][positive])),99)))
 delta_limit=max(delta_limit,1e-6)
 report['paired_map_scale']={'density_log10_min':lo,'density_log10_max':hi,'difference_log10_abs_limit':delta_limit,'rule':'Bounds combined across both models and all stored maps; difference clipping uses the largest per-map 99th percentile, not a physical truncation.'}
 writer=imageio.get_writer(out/'paired_density.mp4',fps=12,codec='libx264',quality=8,macro_block_size=1) if common_maps else None
 try:
  for snap,projection in common_maps:
   arrays=[];meta=[]
   for pair in map_files:
    path,m=pair[(snap,projection)];meta.append(m)
    with h5py.File(path) as f:arrays.append(f['measured'][:])
   for key in ('redshift','projection','measured_quantity_units','map_pixel_count_per_side'):
    if meta[0][key]!=meta[1][key] and not (key=='redshift' and np.isclose(meta[0][key],meta[1][key],atol=1e-6)):raise ValueError('Density map frame differs: '+key)
   b0 = float(meta[0].get('box_size_mpc_h', 0)) / h0
   b1 = float(meta[1].get('box_size_mpc_h', 0)) / h1
   if not (np.isclose(meta[0]['box_size_mpc_h'], meta[1]['box_size_mpc_h'], rtol=1e-4) or np.isclose(b0, b1, rtol=1e-3)):
    raise ValueError('Density map physical box differs')
   for skey in ('line_of_sight_axis', 'assignment_kernel'):
    if meta[0].get('slice',{}).get(skey)!=meta[1].get('slice',{}).get(skey):raise ValueError('Density map slice differs: '+skey)
   if arrays[0].shape!=arrays[1].shape:raise ValueError('Density pixels differ')
   logs=[np.where(x>0,np.log10(np.where(x>0,x,np.nan)),np.nan) for x in arrays]
   diff=logs[1]-logs[0]
   extent=(0,phys_box0,0,phys_box0) if phys_box_equal else (0,float(cfg[0]['simulation']['box_mpc_h']),0,float(cfg[0]['simulation']['box_mpc_h']))
   x_unit = 'comoving Mpc' if phys_box_equal else 'comoving Mpc/h'
   fig,axes=plt.subplots(1,3,figsize=(12,4))
   for ax,x,title in zip(axes[:2],logs,('LCDM','EDE')):
    image=ax.imshow(x,origin='lower',extent=extent,cmap='cividis',vmin=lo,vmax=hi);ax.set_title(title);ax.set_xlabel(x_unit)
   fig.colorbar(image,ax=list(axes[:2]),label='log10 surface density',shrink=.7)
   d=axes[2].imshow(diff,origin='lower',extent=extent,cmap='coolwarm',vmin=-delta_limit,vmax=delta_limit);axes[2].set_title('log10(EDE/LCDM)');axes[2].set_xlabel(x_unit);fig.colorbar(d,ax=axes[2],shrink=.7)
   title_extra = f"matched physical box {phys_box0:.1f} Mpc" if phys_box_equal else "common scales"
   fig.suptitle(f"{projection} | z={meta[0]['redshift']:.3g} | {title_extra}")
   stem=f'density_{snap:03d}_{projection}';save(fig,stem,{'source_maps':[str(pair[(snap,projection)][0]) for pair in map_files],'scale':report['paired_map_scale']})
   np.savez_compressed(out/(stem+'.npz'),lcdm=arrays[0],ede=arrays[1],log10_ratio=diff)
   if projection=='xy':
    frame=Image.fromarray(imageio.imread(out/(stem+'.png'))).convert('RGB')
    frame.thumbnail((2400,1000));canvas=Image.new('RGB',(2400,1000),'white');canvas.paste(frame,((2400-frame.width)//2,(1000-frame.height)//2))
    for _ in range(12):writer.append_data(np.asarray(canvas))
 finally:
  if writer is not None:writer.close()
 multi_epoch_data=[]
 for snap in snapshots[0]:
  hs=[read_snapshot_header(r,snap)[0] for r in runs];scales=[float(h['Time']) for h in hs]
  if not np.isclose(*scales,rtol=0,atol=1e-8):raise ValueError('Saved scale factors differ')
  z=max(0.,1/scales[0]-1);tables=[[row for row in rows if int(row['snapshot'])==snap] for rows in exports]
  if not tables[0] or not tables[1] or len(tables[0])!=len(tables[1]):raise ValueError('HMF bin rows differ')
  edges=[np.array([float(row['mass_lower_msun_h']) for row in t]+[float(t[-1]['mass_upper_msun_h'])]) for t in tables]
  if not np.array_equal(*edges):raise ValueError('HMF bin edges differ')
  counts=[np.array([int(row['halo_count']) for row in t]) for t in tables];mass=np.sqrt(edges[0][:-1]*edges[0][1:]);widths=np.diff(np.log10(edges[0]))
  volumes=[float(cfg[0]['simulation']['box_mpc_h'])**3, float(cfg[1]['simulation']['box_mpc_h'])**3]
  phys_volumes=[volumes[0]/(h0**3), volumes[1]/(h1**3)]
  use_vols=phys_volumes if phys_box_equal else volumes
  densities=[c/(v*widths) for c,v in zip(counts,use_vols)];valid=(counts[0]>0)&(counts[1]>0);ratio=np.divide(densities[1],densities[0],out=np.full(len(mass),np.nan),where=densities[0]>0)
  ratio_error=np.full_like(mass,np.nan);ratio_error[valid]=ratio[valid]*np.sqrt(1/counts[0][valid]+1/counts[1][valid])
  multi_epoch_data.append({'snapshot':snap,'redshift':z,'mass':mass,'counts':counts,'densities':densities,'ratio':ratio,'ratio_error':ratio_error,'widths':widths,'edges':edges[0]})
  rows=[]
  for i in range(len(mass)):
   rows.append({'mass_msun_h':mass[i],'count_lcdm':counts[0][i],'count_ede':counts[1][i],'density_lcdm':densities[0][i],'density_ede':densities[1][i],'ede_over_lcdm':ratio[i],'independent_poisson_ratio_sigma':ratio_error[i],'both_counts_at_least_400':bool(counts[0][i]>=400 and counts[1][i]>=400)})
  write_csv(out/f'hmf_{snap:03d}.csv',rows,list(rows[0]))
  fig,ax=plt.subplots(2,1,figsize=(7.2,6),sharex=True)
  has_halos=any(np.any(c>0) for c in counts)
  for j,(c,y,label) in enumerate(zip(counts,densities,('LCDM','EDE'))):
   lower,upper=poisson_rate_interval(c,use_vols[j],widths);keep=c>0
   if np.any(keep):
    ax[0].errorbar(mass[keep],y[keep],yerr=np.array([y-lower,upper-y])[:,keep],fmt='o-',label=label)
   empty=(c==0)&np.isfinite(upper)&(upper>0)
   if np.any(empty):
    ax[0].scatter(mass[empty],upper[empty],marker='v',s=16,alpha=0.5,label=f'{label} 95% upper limit' if np.count_nonzero(keep)==0 else None)
  dens_unit = 'Mpc⁻³' if phys_box_equal else '(h/Mpc)³'
  if has_halos:
   ax[0].set(xscale='log',yscale='log',ylabel=f'dn/dlog10 M [{dens_unit}]',title=f'FOF b=0.2 | z={z:.3g}');ax[0].legend()
  else:
   ax[0].set(xscale='log',ylabel=f'dn/dlog10 M [{dens_unit}]',title=f'FOF b=0.2 | z={z:.3g}')
   ax[0].text(0.5,0.5,'No resolved halos at this epoch',ha='center',va='center',transform=ax[0].transAxes)
  if np.any(valid):
   ax[1].errorbar(mass[valid],ratio[valid],yerr=ratio_error[valid],fmt='o')
  else:
   ax[1].text(0.5,0.5,'No common halos',ha='center',va='center',transform=ax[1].transAxes)
  ax[1].axhline(1,color='gray');ax[1].set(xscale='log',xlabel='M [Msun/h]',ylabel='EDE / LCDM')
  fig.text(.5,.005,'Ratio errors: independent Poisson approximation; paired covariance and cosmic variance unmeasured.',ha='center',fontsize=7)
  save(fig,f'hmf_{snap:03d}',{'snapshot':snap,'redshift':z,'uncertainty':report['uncertainty']})
  spectra=[gadget_matter_power_spectrum(r/'data/powerspecs'/f'powerspec_{snap:03d}.txt') for r in runs]
  if any(x is None for x in spectra):raise ValueError('Missing native power spectrum')
  if phys_box_equal:
   k0 = spectra[0]['k_h_Mpc'] * h0
   k1 = spectra[1]['k_h_Mpc'] * h1
   if not np.allclose(k0, k1, rtol=1e-3):raise ValueError('Physical power k bins differ')
   k = k0; k_unit = '1/Mpc'
   power = [spectra[0]['P_Mpc_h3'] / (h0**3), spectra[1]['P_Mpc_h3'] / (h1**3)]
   p_unit = 'Mpc³'
  else:
   if not np.allclose(spectra[0]['k_h_Mpc'],spectra[1]['k_h_Mpc'],rtol=1e-8):raise ValueError('Power k bins differ')
   k = spectra[0]['k_h_Mpc']; k_unit = 'h/Mpc'
   power = [spectra[0]['P_Mpc_h3'], spectra[1]['P_Mpc_h3']]
   p_unit = '(Mpc/h)³'
  good=(power[0]>0)&(power[1]>0)
  fig,ax=plt.subplots(2,1,figsize=(7.2,6),sharex=True)
  for values,label in zip(power,('LCDM','EDE')):ax[0].loglog(k[values>0],values[values>0],label=label)
  ax[0].legend();ax[0].set(ylabel=f'P [{p_unit}]',title=f'Native matter power | z={z:.3g}')
  ax[1].semilogx(k[good],power[1][good]/power[0][good]);ax[1].axhline(1,color='gray');ax[1].set(xlabel=f'k [{k_unit}]',ylabel='EDE / LCDM')
  save(fig,f'power_{snap:03d}',{'snapshot':snap,'redshift':z,'ratio_covariance':'not estimated'})
  np.savez_compressed(out/f'power_{snap:03d}.npz',k=k,lcdm=power[0],ede=power[1],valid_ratio=good)
  # Exact reciprocal ID overlap, with a conservative Python-memory ceiling.
  budget=memory_limit_gb({'hardware':{'memory_limit_gb':'auto'}})
  max_ids=int(.25*budget*1024**3/128)
  cat_mins=[c['analysis'].get('catalogue_min_particles',c['halos'].get('min_particles',32)) for c in cfg]
  catalogs=[_read_cross_run_halo_catalogue(r,snap,'fof',m_cut,max_ids) for r,c,m_cut in zip(runs,cfg,cat_mins)]
  actual_max_mass=[float(np.max(c['mass_msun_h'])) if len(c['mass_msun_h']) else 0.0 for c in catalogs]
  multi_epoch_data[-1]['actual_max_mass']=actual_max_mass
  sweep_counts={f'matches_frac_{int(f*100)}':len(_mutual_best_halo_matches(*catalogs,10,f)['a']) for f in (0.1,0.2,0.5)}
  matches=_mutual_best_halo_matches(*catalogs,10,.1);matched=[]
  lookup=[{int(g):i for i,g in enumerate(c['group_indices'])} for c in catalogs]
  for j,(g0,g1) in enumerate(zip(matches['a'],matches['b'])):
   i0,i1=lookup[0][int(g0)],lookup[1][int(g1)]
   matched.append({'lcdm_group_row':int(g0),'ede_group_row':int(g1),'shared_dm_ids':int(matches['shared'][j]),'fraction_lcdm':float(matches['fraction_a'][j]),'fraction_ede':float(matches['fraction_b'][j]),'lcdm_mass_msun_h':float(catalogs[0]['mass_msun_h'][i0]),'ede_mass_msun_h':float(catalogs[1]['mass_msun_h'][i1])})
  fields=['lcdm_group_row','ede_group_row','shared_dm_ids','fraction_lcdm','fraction_ede','lcdm_mass_msun_h','ede_mass_msun_h'];write_csv(out/f'halo_matches_{snap:03d}.csv',matched,fields)
  fig,ax=plt.subplots(figsize=(6,5))
  if matched:
   x=np.array([x['lcdm_mass_msun_h'] for x in matched]);y=np.array([x['ede_mass_msun_h'] for x in matched]);ax.scatter(x,y,s=6,alpha=.5);lim=[min(x.min(),y.min()),max(x.max(),y.max())];ax.plot(lim,lim,'--',color='gray');ax.set(xscale='log',yscale='log')
  else:ax.text(.5,.5,'No resolved reciprocal matches',ha='center',transform=ax.transAxes)
  ax.set(xlabel='LCDM FOF mass [Msun/h]',ylabel='EDE FOF mass [Msun/h]',title=f'Exact-ID reciprocal host matches | z={z:.3g}');save(fig,f'matched_halos_{snap:03d}',{'snapshot':snap,'thresholds':{'minimum_shared_ids':10,'minimum_fraction_each':.1},'overlap_fraction_sweep':sweep_counts})
  report['epoch_reports'].append({'snapshot':snap,'redshift':z,'matches':len(matched),'resolved_hosts':[len(c['group_indices']) for c in catalogs],'actual_max_mass_msun_h':actual_max_mass,'overlap_fraction_sweep':sweep_counts,'hmf_bins_both_at_least_400':int(np.count_nonzero((counts[0]>=400)&(counts[1]>=400)))})
  atomic_json(out/'report.json',report);del catalogs,matches;gc.collect()
 if len(multi_epoch_data)>=2:
  z_list=[ep['redshift'] for ep in multi_epoch_data]
  m_bins=multi_epoch_data[0]['mass']
  ratio_grid=np.full((len(m_bins),len(z_list)),np.nan)
  sig_grid=np.full((len(m_bins),len(z_list)),np.nan)
  count_lcdm_grid=np.zeros((len(m_bins),len(z_list)))
  count_ede_grid=np.zeros((len(m_bins),len(z_list)))
  for j,ep in enumerate(multi_epoch_data):
   count_lcdm_grid[:,j]=ep['counts'][0]
   count_ede_grid[:,j]=ep['counts'][1]
   tot=ep['counts'][0]+ep['counts'][1]
   pos=(ep['counts'][0]>0)&(ep['counts'][1]>0)
   ratio_grid[pos,j]=np.log10(ep['ratio'][pos])
   sig_mask=tot>0
   sig_grid[sig_mask,j]=(ep['counts'][1][sig_mask]-ep['counts'][0][sig_mask])/np.sqrt(tot[sig_mask])
  fig,ax=plt.subplots(figsize=(8,6))
  im=ax.imshow(ratio_grid,origin='lower',aspect='auto',cmap='coolwarm',extent=[0,len(z_list)-1,np.log10(m_bins[0]),np.log10(m_bins[-1])],vmin=-0.3,vmax=0.8)
  ax.set_xticks(range(len(z_list)))
  ax.set_xticklabels([f'{z:.2g}' for z in z_list],rotation=45)
  ax.set(xlabel='Redshift z',ylabel='log10(M [Msun/h])',title='2D Abundance Ratio log10(EDE / LCDM)')
  fig.colorbar(im,ax=ax,label='log10(dn_EDE / dn_LCDM)')
  save(fig,'heatmap_abundance_ratio',{'type':'2d_mass_redshift_ratio'})
  fig,ax=plt.subplots(figsize=(8,6))
  im=ax.imshow(sig_grid,origin='lower',aspect='auto',cmap='magma',extent=[0,len(z_list)-1,np.log10(m_bins[0]),np.log10(m_bins[-1])],vmin=0,vmax=max(5.0,float(np.nanmax(sig_grid)) if np.any(np.isfinite(sig_grid)) else 5.0))
  ax.set_xticks(range(len(z_list)))
  ax.set_xticklabels([f'{z:.2g}' for z in z_list],rotation=45)
  ax.set(xlabel='Redshift z',ylabel='log10(M [Msun/h])',title='Statistical Separation: (N_EDE - N_LCDM) / sqrt(N_tot) [sigma]')
  fig.colorbar(im,ax=ax,label='Significance [sigma]')
  save(fig,'heatmap_significance',{'type':'statistical_separation'})
  fig,axes=plt.subplots(1,2,figsize=(12,5))
  for ax,c_grid,title in zip(axes,(count_lcdm_grid,count_ede_grid),('LCDM Halo Counts','EDE Halo Counts')):
   im=ax.imshow(np.log10(np.where(c_grid>0,c_grid,np.nan)),origin='lower',aspect='auto',cmap='viridis',extent=[0,len(z_list)-1,np.log10(m_bins[0]),np.log10(m_bins[-1])])
   ax.set_xticks(range(len(z_list)))
   ax.set_xticklabels([f'{z:.2g}' for z in z_list],rotation=45)
   ax.set(xlabel='Redshift z',ylabel='log10(M [Msun/h])',title=title)
   fig.colorbar(im,ax=ax,label='log10(N_halos)')
  save(fig,'heatmap_counts',{'type':'resolved_halo_counts'})
  matrix_rows=[]
  for i,m in enumerate(m_bins):
   row={'mass_msun_h':m}
   for j,z in enumerate(z_list):
    row[f'count_lcdm_z{z:.2f}']=int(count_lcdm_grid[i,j])
    row[f'count_ede_z{z:.2f}']=int(count_ede_grid[i,j])
    row[f'ratio_z{z:.2f}']=float(10**ratio_grid[i,j]) if np.isfinite(ratio_grid[i,j]) else 'nan'
   matrix_rows.append(row)
  write_csv(out/'hmf_multi_epoch_matrix.csv',matrix_rows,list(matrix_rows[0]))
 for ep in multi_epoch_data:
  z_val=ep['redshift']
  if abs(z_val-14.44)<0.1 or abs(z_val-0.0)<0.01:
   tag='z14.44' if abs(z_val-14.44)<0.1 else 'z0'
   c0,c1=ep['counts']
   actual_max=ep.get('actual_max_mass',[0.0,0.0])
   max_m_lcdm=actual_max[0] if actual_max[0]>0 else (float(ep['mass'][c0>0][-1]) if np.any(c0>0) else 0.0)
   max_m_ede=actual_max[1] if actual_max[1]>0 else (float(ep['mass'][c1>0][-1]) if np.any(c1>0) else 0.0)
   h_lcdm=float(cfg[0]['cosmology'].get('hubble',0.6781))
   h_ede=float(cfg[1]['cosmology'].get('hubble',0.7483))
   summary={'target_epoch':tag,'actual_redshift':float(z_val),'total_halos_lcdm':int(np.sum(c0)),'total_halos_ede':int(np.sum(c1)),'max_halo_mass_msun_h_lcdm':max_m_lcdm,'max_halo_mass_msun_h_ede':max_m_ede,'max_halo_mass_msun_physical_lcdm':max_m_lcdm/h_lcdm if max_m_lcdm else 0.0,'max_halo_mass_msun_physical_ede':max_m_ede/h_ede if max_m_ede else 0.0,'units':{'mass':'Msun/h','physical_mass':'Msun','box_size':'Mpc/h'},'total_enhancement_ratio':float(np.sum(c1)/max(1,np.sum(c0)))}
   atomic_json(out/f'scientific_summary_{tag}.json',summary)
 report['status']='COMPLETE';atomic_json(out/'report.json',report)
except BaseException as error:
 report.update(status='FAILED',failure={'type':type(error).__name__,'message':str(error)});atomic_json(out/'report.json',report);raise
finally:
 body='<h1>Matched EDE and LCDM comparison</h1><p>Status: '+html.escape(report['status'])+'</p><p>'+html.escape(report['uncertainty'])+'</p><p><a href="report.json">Report</a> · <a href="paired_density.mp4">Paired density movie</a></p>'
 body+=''.join('<p><a href="'+html.escape(name)+'">'+html.escape(name)+'</a></p>' for name in links)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'+body)
print(out/'index.html')
