#!/usr/bin/env python3
"""Test a conditional broken-line accretion model on saved Rockstar histories."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.assembly import fit_supported_fast_slow_accretion_transition
from core.foundation import atomic_json,file_sha256
from core.paired_atlas import _write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    for model,section in [('EDE','02_EDE'),('LCDM','03_LCDM')]:
        folder=root/section/'Halo_history_z_199_to_000'
        path,=folder.glob('*Rockstar_main_branch.csv')
        with path.open() as stream:rows=list(csv.DictReader(stream))
        a=np.array([float(r['scale_factor']) for r in rows])
        m=np.array([float(r['Mvir_msun']) for r in rows])
        count=np.array([int(r['particle_count']) for r in rows])
        snapshots=np.array([int(r['snapshot']) for r in rows])
        supported=(count>=300)&np.isfinite(m)&(m>0)
        errors=1/np.sqrt(np.maximum(count,1))
        history={'scale_factor':a,'mass_msun':m,'history_supported':supported,
                 'interval_supported':supported[:-1]&supported[1:]&(np.diff(snapshots)==1)}
        request={'minimum_epochs':8,'log_mass_error':errors.tolist(),'minimum_segment_epochs':3,
                 'maximum_candidate_knots':20,'minimum_bic_improvement':6.,
                 'minimum_slope_drop':.5,'maximum_reduced_chi2':3.}
        result=fit_supported_fast_slow_accretion_transition(history,request)
        stem=folder/f'{model}_z_001p505_to_000p000_Conditional_fast_slow_accretion_test'
        atomic_json(stem.with_suffix('.json'),{'source':str(path.relative_to(root)),'sha256':file_sha256(path),
            'controls':request,'result':result,
            'error_model':'Conditional log-mass counting scale 1/sqrt(native Rockstar Np). Correlated tree/finder/finite-volume errors absent.',
            'scope':'Conditional model diagnostic over available z≤1.505 main branch.'})
        intervals=[]
        for i in range(len(a)-1):
            intervals.append({'scale_factor_left':a[i],'scale_factor_right':a[i+1],
                'dlnM_dlna':np.log(m[i+1]/m[i])/np.log(a[i+1]/a[i]),
                'supported':bool(history['interval_supported'][i])})
        _write_csv(stem.with_suffix('.csv'),intervals)
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
        axes[0].errorbar(a[supported],m[supported],yerr=m[supported]*errors[supported],fmt='o-',color='#16747b',ms=4)
        axes[0].set(xlabel='Scale factor a',ylabel='Mvir [Msun]',yscale='log')
        axes[1].plot(np.sqrt(a[:-1]*a[1:]),[r['dlnM_dlna'] if r['supported'] else np.nan for r in intervals],'o-',color='#a56b38')
        axes[1].set(xlabel='Interval geometric mean scale factor',ylabel='d ln M / d ln a')
        transition=result.get('transition')
        if transition:
            for ax in axes:ax.axvline(transition['scale_factor'],ls='--',color='#79639d')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(model+' · '+result['status'].replace('_',' ').lower(),fontsize=16)
        fig.supxlabel('Conditional model using 1/√Np mass counting scales. The saved history and model selection status are retained.',fontsize=10)
        for suffix in ('.png','.pdf','.svg'):fig.savefig(stem.with_suffix(suffix),dpi=175)
        plt.close(fig)
        print(model,result['status'],flush=True)


if __name__=='__main__':main()
