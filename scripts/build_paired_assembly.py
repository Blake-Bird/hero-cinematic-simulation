#!/usr/bin/env python3
"""Measure cross-model formation times and exact matched FOF assembly histories."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import _fof_main_progenitor_tracks, group_field
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import _write_csv, epoch_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    label = epoch_name('EDE_vs_LCDM',85,0)
    out = root/'01_Comparisons'/label
    matches, = out.glob('*_Matched_FOF_host_properties.csv')
    with matches.open() as stream:
        pairs = list(csv.DictReader(stream))
    pairs.sort(key=lambda r: float(r['mass_EDE_msun'])*float(r['mass_LCDM_msun']), reverse=True)
    model_tracks, rows, formation, sources = {}, [], [], {}
    for model in ('EDE','LCDM'):
        work = root/'Source/Native_workspaces'/model
        measure = root/'06_Tables_and_Data/Measurements'/model
        manifest = measure/'measurement_manifest.json'
        data = json.loads(manifest.read_text())
        epoch_records = [(r['summary']['redshift'], f'snapshot_{r["summary"]["snapshot"]:03d}') for r in data['epochs']]
        h = data['epochs'][-1]['summary']['h']
        box = data['epochs'][-1]['summary']['box_comoving_mpc']*h
        tracks = _fof_main_progenitor_tracks(work, epoch_records, 'snapshot_085',
            [int(r['FOF_row_'+model]) for r in pairs], 'fof', 100, 32, .25, 2_100_000, box)
        tracks['mass_msun'] = tracks['mass_msun_h']/h
        tracks['a'] = 1/(1+tracks['redshift'])
        mass = tracks['mass_msun']
        ratio = np.divide(mass[:,1:], mass[:,:-1], out=np.full_like(mass[:,1:],np.nan),
                          where=np.isfinite(mass[:,:-1]) & (mass[:,:-1]>0))
        tracks['dlnM_dlna'] = np.log(ratio)/np.diff(np.log(tracks['a']))
        tracks['central_vmax'] = np.full_like(mass,np.nan)
        tracks['central_spin'] = np.full_like(mass,np.nan)
        for i, record in enumerate(data['epochs']):
            g = tracks['group_index'][:,i]
            active = np.flatnonzero(g >= 0)
            if not len(active):
                continue
            snap = record['summary']['snapshot']
            central = group_field(work,snap,['GroupFirstSub'])
            n_sub = group_field(work,snap,['GroupNsubs'])
            if central is None or n_sub is None:
                continue
            source = measure/record['numeric_asset']
            if file_sha256(source) != record['numeric_sha256']:
                raise ValueError(f'Changed source {source}')
            with np.load(source,allow_pickle=False) as arrays:
                vmax, spin = arrays['subhalo_vmax_km_s'], arrays['subhalo_spin_kpc_km_s']
            index = central[g[active]].astype(np.int64)
            supported = (n_sub[g[active]] > 0) & (index >= 0) & (index < len(vmax)) & (index < len(spin))
            tracks['central_vmax'][active[supported],i] = vmax[index[supported]]
            tracks['central_spin'][active[supported],i] = np.linalg.norm(spin[index[supported]],axis=1)
        for j in range(len(pairs)):
            for i,snap in enumerate(r['summary']['snapshot'] for r in data['epochs']):
                rows.append({'model':model,'pair_rank_by_geometric_final_mass':j+1,'snapshot':snap,
                    'redshift':tracks['redshift'][i],'scale_factor':tracks['a'][i],
                    'FOF_row':tracks['group_index'][j,i] if tracks['group_index'][j,i]>=0 else '',
                    'mass_msun':mass[j,i],'central_SUBFIND_vmax_km_s':tracks['central_vmax'][j,i],
                    'central_SUBFIND_specific_angular_momentum_kpc_km_s':tracks['central_spin'][j,i],
                    'shared_IDs_with_next_epoch':tracks['shared_particle_count'][j,i] if i<len(tracks['a'])-1 else '',
                    'dlnM_dlna_to_next_epoch':tracks['dlnM_dlna'][j,i] if i<len(tracks['a'])-1 else ''})
        model_tracks[model]=tracks
        sources[model]={'measurement_manifest':str(manifest.relative_to(root)), 'sha256':file_sha256(manifest)}
        print('TRACKED',model,len(pairs),flush=True)
    for i,pair in enumerate(pairs):
        a,b=(model_tracks[m] for m in ('EDE','LCDM'))
        za,zb=a['formation_redshift'][i],b['formation_redshift'][i]
        formation.append({'rank':i+1,'EDE_final_FOF_row':pair['FOF_row_EDE'],'LCDM_final_FOF_row':pair['FOF_row_LCDM'],
            'EDE_formation_redshift':za,'LCDM_formation_redshift':zb,'EDE_minus_LCDM_formation_redshift':za-zb,
            'EDE_status':a['formation_status'][i],'LCDM_status':b['formation_status'][i],
            'geometric_final_mass_msun':np.sqrt(float(pair['mass_EDE_msun'])*float(pair['mass_LCDM_msun']))})
    stem=out/f'{label}_Matched_FOF_assembly'
    _write_csv(stem.with_suffix('.csv'),rows)
    formation_path=stem.with_name(stem.name+'_formation.csv')
    _write_csv(formation_path,formation)
    fig,ax=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    fig.suptitle('How the matched haloes assembled · z = 0 descendants',fontsize=19)
    colors={'EDE':'#ae652d','LCDM':'#0e7476'}
    for model,t in model_tracks.items():
        a=t['a'];color=colors[model]
        for axis,y,ylabel,x in [(ax[0,0],t['mass_msun'][0],'FOF mass [Msun]',a),
             (ax[0,1],t['dlnM_dlna'][0],'Interval d ln M / d ln a',np.sqrt(a[:-1]*a[1:])),
             (ax[0,2],t['central_spin'][0],'Central specific angular momentum [kpc km/s]',a),
             (ax[1,0],t['central_vmax'][0],'Central Vmax [km/s]',a)]:
            axis.plot(x,y,'o-',ms=2.7,color=color,label=model)
            axis.set(xlabel='Scale factor a',ylabel=ylabel)
            axis.grid(alpha=.2)
    ax[0,0].set_yscale('log')
    ax[0,0].legend(frameon=False)
    za=np.asarray([r['EDE_formation_redshift'] for r in formation]);zb=np.asarray([r['LCDM_formation_redshift'] for r in formation])
    x=np.asarray([r['geometric_final_mass_msun'] for r in formation]);valid=np.isfinite(za)&np.isfinite(zb)
    ax[1,1].scatter(zb[valid],za[valid],s=20,alpha=.6,c='#16767c',edgecolors='none')
    if valid.any():
        limits=[min(za[valid].min(),zb[valid].min()),max(za[valid].max(),zb[valid].max())]
        ax[1,1].plot(limits,limits,'--',color='#536271',lw=1)
    ax[1,1].set(xlabel='ΛCDM half-final-mass redshift',ylabel='EDE half-final-mass redshift')
    ax[1,2].scatter(x[valid],(za-zb)[valid],s=20,alpha=.6,c='#16767c',edgecolors='none')
    ax[1,2].axhline(0,color='#536271',lw=1,linestyle='--')
    ax[1,2].set(xscale='log',xlabel='Geometric mean final FOF mass [Msun]',ylabel='Formation redshift shift · EDE − ΛCDM')
    fig.supxlabel(f'Curves show the most massive matched pair. Formation points use {int(valid.sum())} supported pairs of {len(pairs)} final matches.',fontsize=10)
    for suffix in ('.png','.pdf','.svg'):
        fig.savefig(stem.with_suffix(suffix),dpi=180)
    atomic_json(stem.with_suffix('.json'), {'sources':sources,'matched_catalogue_sha256':file_sha256(matches),
        'formation_pairs_supported':int(valid.sum()),'candidate_pairs':len(pairs),
        'selection':'Backward most-shared native DM IDs at adjacent saved epochs. 100-particle host floor, ≥32 shared IDs and ≥25% overlap in both groups. Central SUBFIND spin/velocity are epoch-local properties.'})
    target=out/'comparison_manifest.json';manifest=json.loads(target.read_text())
    manifest['figures']=sorted(set(manifest['figures'])|{stem.with_suffix('.png').name})
    manifest['tables']=sorted(set(manifest['tables'])|{stem.with_suffix('.csv').name,formation_path.name})
    atomic_json(target,manifest)


if __name__=='__main__':
    main()
