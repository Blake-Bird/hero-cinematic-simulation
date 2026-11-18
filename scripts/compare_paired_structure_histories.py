#!/usr/bin/env python3
"""Overlay measured radial shapes and saved catalogue histories for both models."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json,file_sha256
from core.paired_atlas import _write_csv,epoch_name


def publish(fig,rows,sources,stem,manifest):
    for suffix in ('.png','.pdf','.svg'):fig.savefig(stem.with_suffix(suffix),dpi=180)
    plt.close(fig)
    _write_csv(stem.with_suffix('.csv'),rows)
    atomic_json(stem.with_suffix('.json'),{'sources':sources})
    manifest['figures']=sorted(set(manifest['figures'])|{stem.with_suffix('.png').name})
    manifest['tables']=sorted(set(manifest['tables'])|{stem.with_suffix('.csv').name})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    out=root/'01_Comparisons'/epoch_name('EDE_vs_LCDM',85,0)
    manifest_path=out/'comparison_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    rows,sources=[],{}
    fig.suptitle('Radial dark-matter shape in the largest matched pair · z = 0',fontsize=18)
    for model,section,color in [('EDE','02_EDE','#ae652d'),('LCDM','03_LCDM','#0e7476')]:
        path,=(root/section/'Spatial_explorers').glob('*000p000/Native_structure*/native_structure.json')
        d=json.loads(path.read_text());sources[model]={'file':str(path.relative_to(root)),'sha256':file_sha256(path)}
        for shell in d['density_shapes']['dm']['shells']:
            uncertainty=shell.get('sampling_uncertainty',{}).get('axis_ratio_percentile_16_50_84')
            rows.append({'model':model,'lower_proper_kpc':shell['lower_radius_kpc'],'upper_proper_kpc':shell['upper_radius_kpc'],
                'shell_particles':shell['particle_count'],'b_over_a':shell.get('b_over_a',np.nan),
                'c_over_a':shell.get('c_over_a',np.nan),'triaxiality':shell.get('triaxiality',np.nan),
                'b_over_a_bootstrap_p16':uncertainty[0][0] if uncertainty else np.nan,
                'b_over_a_bootstrap_p84':uncertainty[2][0] if uncertainty else np.nan,
                'c_over_a_bootstrap_p16':uncertainty[0][1] if uncertainty else np.nan,
                'c_over_a_bootstrap_p84':uncertainty[2][1] if uncertainty else np.nan})
        selected=[r for r in rows if r['model']==model]
        radius=[(r['lower_proper_kpc']+r['upper_proper_kpc'])/2 for r in selected]
        for ax,key,ylabel in zip(axes,('b_over_a','c_over_a','triaxiality'),('Intermediate / major axis b/a','Minor / major axis c/a','Triaxiality T')):
            ax.plot(radius,[r[key] for r in selected],'o-',color=color,label=model,ms=3)
            if key!='triaxiality':
                ax.fill_between(radius,[r[key+'_bootstrap_p16'] for r in selected],[r[key+'_bootstrap_p84'] for r in selected],color=color,alpha=.15,linewidth=0)
            ax.set(xscale='log',xlabel='Proper shell-centre radius [kpc]',ylabel=ylabel,ylim=(0,1.05));ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.supxlabel('Full spherical shell mass-quadrupole tensors. Shading gives particle-bootstrap 16–84% intervals.',fontsize=10)
    publish(fig,rows,sources,out/'EDE_vs_LCDM_z_0000p000_snap_085_Matched_halo_radial_density_shape',manifest)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    fig.suptitle('Catalogue structure along the two largest Rockstar branches',fontsize=19)
    rows,sources=[],{}
    for model,section,color in [('EDE','02_EDE','#ae652d'),('LCDM','03_LCDM','#0e7476')]:
        path,=(root/section/'Halo_history_z_199_to_000').glob('*Rockstar_main_branch.csv')
        with path.open() as stream:data=list(csv.DictReader(stream))
        sources[model]={'file':str(path.relative_to(root)),'sha256':file_sha256(path)}
        rows.extend(data)
        a=[float(r['scale_factor']) for r in data]
        for ax,key,label in zip(axes.flat,('Mvir_msun','Rvir_over_rs','spin_bullock','Vmax_km_s','c_over_a','saved_interval_major_axis_turn_degrees'),
                ('Mvir [Msun]','Catalogue Rvir / rs','Bullock spin','Vmax [km/s]','Minor / major axis c/a','Unsigned major-axis turn [degrees]')):
            ax.plot(a,[float(r[key]) for r in data],'o-',color=color,label=model,ms=3)
            ax.set(xlabel='Scale factor a',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].set_yscale('log');axes[0,0].legend(frameon=False)
    fig.supxlabel('Per-model largest final Rockstar halo. Exact epoch-local original IDs join the tree to the catalogue. Fit covariance is unavailable.',fontsize=10)
    publish(fig,rows,sources,out/'EDE_vs_LCDM_z_001p505_to_000p000_Rockstar_branch_structure',manifest)
    atomic_json(manifest_path,manifest)


if __name__=='__main__':main()
