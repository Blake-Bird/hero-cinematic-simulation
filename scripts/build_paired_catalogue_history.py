#!/usr/bin/env python3
"""Plot catalogue structure along each measured largest-progenitor branch."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.analysis import rockstar_table
from core.foundation import atomic_json,file_sha256
from core.merger_tree_visual import branch_identity
from core.paired_atlas import _write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    combined=[]
    for model,section,h,color in [('EDE','02_EDE',.7483,'#ae652d'),('LCDM','03_LCDM',.6781,'#0e7476')]:
        workspace=root/'Source/Native_workspaces'/f'{model}_history'
        treepath=sorted(workspace.glob('results/analysis_products_*/merger_tree.json'))[-1]
        tree=json.loads(treepath.read_text())
        identity=branch_identity(tree)
        branch=sorted((n for n,b in zip(tree['nodes'],identity) if b['branch']==0),key=lambda n:n['scale_factor'])
        rows=[]
        for node in branch:
            snap=node['snapshot'];table=rockstar_table(workspace,snap)
            original=int(node.get('original_halo_id',node['id']))
            indices=np.flatnonzero(np.asarray(table['id'],dtype=np.int64)==original)
            if len(indices)!=1:
                raise ValueError(f'Cannot join {model} snapshot {snap} original halo {original}')
            i=int(indices[0]);n=int(table['np'][i])
            def value(key):
                return float(table[key][i]) if key in table else float('nan')
            rs,rvir=value('rs'),value('rvir')
            concentration=rvir/rs if n>=300 and np.isfinite(rs) and rs>0 and rvir>0 else float('nan')
            b,c=value('b_to_a'),value('c_to_a')
            shape=n>=300 and 0<c<=b<=1
            rows.append({'model':model,'snapshot':snap,'redshift':node['redshift'],'scale_factor':node['scale_factor'],
                'tree_catalogue_id':node['id'],'original_Rockstar_id':str(original),'particle_count':n,
                'Mvir_msun':value('mvir')/h,'Rvir_proper_kpc':rvir/h,'rs_proper_kpc':rs/h,
                'Rvir_over_rs':concentration,'spin_bullock':value('spin_bullock') if n>=300 else np.nan,
                'b_over_a':b if shape else np.nan,'c_over_a':c if shape else np.nan,
                'Vmax_km_s':value('vmax'),'vrms_km_s':value('vrms'),
                'major_axis_x':value('a[x]') if shape else np.nan,
                'major_axis_y':value('a[y]') if shape else np.nan,
                'major_axis_z':value('a[z]') if shape else np.nan,
                'catalogue_structure_supported':n>=300})
        for j,row in enumerate(rows):
            row['saved_interval_major_axis_turn_degrees']=np.nan
            if j==0 or row['snapshot']!=rows[j-1]['snapshot']+1:continue
            p=rows[j-1];a=np.array([p['major_axis_'+v] for v in 'xyz']);b=np.array([row['major_axis_'+v] for v in 'xyz'])
            norm=np.linalg.norm(a)*np.linalg.norm(b)
            if np.isfinite(norm) and norm>0:
                row['saved_interval_major_axis_turn_degrees']=float(np.degrees(np.arccos(np.clip(abs(a@b)/norm,0,1))))
        out=root/section/'Halo_history_z_199_to_000'
        out.mkdir(exist_ok=True)
        name=f'{model}_z_{branch[0]["redshift"]:07.3f}_to_000p000_Rockstar_main_branch'.replace('.','p')
        stem=out/name
        _write_csv(stem.with_suffix('.csv'),rows)
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        fig.suptitle(f'{model} · catalogue structure along the largest-progenitor branch',fontsize=18)
        a=np.array([r['scale_factor'] for r in rows])
        for ax,key,label in zip(axes.flat,('Mvir_msun','Rvir_over_rs','spin_bullock','Vmax_km_s','b_over_a','saved_interval_major_axis_turn_degrees'),
                ('Mvir [Msun]','Catalogue Rvir / rs','Bullock spin','Vmax [km/s]','Density axis ratios','Unsigned major-axis turn [degrees]')):
            ax.plot(a,[r[key] for r in rows],'o-',color=color,ms=3,label='b/a' if key=='b_over_a' else model)
            if key=='b_over_a':
                ax.plot(a,[r['c_over_a'] for r in rows],'s--',color='#87639f',ms=3,label='c/a')
                ax.legend(frameon=False)
            ax.set(xlabel='Scale factor a',ylabel=label);ax.grid(alpha=.2)
        axes[0,0].set_yscale('log')
        fig.supxlabel('Exact tree-to-Rockstar original-ID joins at each epoch. Structure cuts require ≥300 particles; parameter-fit errors are unavailable.',fontsize=10)
        for suffix in ('.png','.pdf','.svg'):fig.savefig(stem.with_suffix(suffix),dpi=180)
        plt.close(fig)
        atomic_json(stem.with_suffix('.json'),{'source_tree':str(treepath),'sha256':file_sha256(treepath),
            'selection':'H0 branch of exact adjacent descendant tree; Rockstar properties joined by original epoch-local ID.',
            'support':'300 particles for concentration, spin and shape; catalogue fit covariance and orientation uncertainty absent.',
            'scope':'Per-model largest final Rockstar halo. This comparison does not establish that the two final Rockstar rows form an exact matched pair.'})
        combined.extend(rows)
        print(model,len(rows),flush=True)
    folder=root/'06_Tables_and_Data'
    _write_csv(folder/'EDE_and_LCDM_z_001p505_to_000p000_main_branch_catalogue_properties.csv',combined)


if __name__=='__main__':main()
