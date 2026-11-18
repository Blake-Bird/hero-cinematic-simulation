"""Small-multiple atlas of comparable-mass halo assembly histories."""
from __future__ import annotations
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def select_comparable_histories(histories, *, mass_factor=2., max_halos=9):
    if not math.isfinite(mass_factor) or mass_factor<1 or type(max_halos) is not int or max_halos<2:
        raise ValueError('A mass factor >=1 and at least two atlas slots are required')
    candidates=[h for h in histories if math.isfinite(float(h['final_mass_msun_h'])) and h['final_mass_msun_h']>0 and len(h['track'])>=2]
    candidates.sort(key=lambda h:(-float(h['final_mass_msun_h']),int(h['final_halo_id'])))
    best=[]
    for index,halo in enumerate(candidates):
        upper=float(halo['final_mass_msun_h'])
        group=[item for item in candidates[index:] if float(item['final_mass_msun_h'])>=upper/mass_factor][:max_halos]
        if len(group)>len(best):best=group
    return best


def export_formation_atlas(result, output_directory, *, mass_factor=2., max_halos=9):
    """Plot source-linked tracks only; half-mass epochs retain interpolation scope."""
    if not result or result.get('status')!='COMPLETE':return None
    selected=select_comparable_histories(result['histories'],mass_factor=mass_factor,max_halos=max_halos)
    if len(selected)<2:return None
    scales=[np.asarray([p['scale_factor'] for p in h['track']],dtype=float) for h in selected]
    ratios=[np.asarray([p['mass_msun_h'] for p in h['track']],dtype=float)/float(h['final_mass_msun_h']) for h in selected]
    if any(not np.isfinite(a).all() or np.any(np.diff(a)<=0) or not np.isfinite(m).all() or np.any(m<=0) for a,m in zip(scales,ratios)):
        raise ValueError('Formation atlas requires ordered finite saved histories and positive masses')
    cols=min(3,len(selected));rows=math.ceil(len(selected)/cols)
    height=3.35*rows+1.3
    fig,axes=plt.subplots(rows,cols,figsize=(4.5*cols,height),squeeze=False,facecolor='#0a1322')
    fig.subplots_adjust(left=.07,right=.97,bottom=.6/height,top=1-1.15/height,wspace=.28,hspace=.5)
    xmax=max(float(a[-1]) for a in scales);xmin=min(float(a[0]) for a in scales)
    ymax=max(1.05,max(float(np.max(r)) for r in ratios)*1.05)
    for ax,h,a,m in zip(axes.flat,selected,scales,ratios):
        ax.set_facecolor('#14253a');ax.plot(a,m,color='#56d5e8',lw=2,marker='o',ms=4)
        ax.axhline(.5,color='#f4c975',ls='--',lw=.8,alpha=.85)
        z=h.get('formation_redshift')
        detail='half-mass crossing: unavailable'
        if z is not None:
            crossing=1/(1+float(z));ax.axvline(crossing,color='#f4c975',lw=.8,ls=':',alpha=.85)
            detail='interpolated half-mass z = %.3g'%float(z)
        ax.text(.04,.96,detail+'\n%d linked main-branch mergers'%len(h.get('merger_events',[])),
                transform=ax.transAxes,va='top',color='#d5e6f6',fontsize=8,
                bbox={'facecolor':'#14253a','edgecolor':'none','alpha':.9,'pad':2})
        ax.set_title('Halo %s · Mvir %.3g Msun/h'%(h['final_halo_id'],h['final_mass_msun_h']),color='white',fontsize=11)
        ax.set_xlim(xmin,xmax if xmax>xmin else xmin+1e-6);ax.set_ylim(0,ymax)
        ax.set_xlabel('scale factor a',color='#c5d6e8',fontsize=9);ax.set_ylabel('Mvir / final Mvir',color='#c5d6e8',fontsize=9)
        ax.tick_params(colors='#c5d6e8',labelsize=8);ax.grid(color='#647f99',alpha=.18)
        for spine in ax.spines.values():spine.set_color('#7a91a8')
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    fig.suptitle('Halo formation atlas',color='white',fontsize=20,y=1-.2/height)
    fig.text(.5,1-.63/height,'Comparable final masses · saved main branches · dashed line: half final mass',ha='center',color='#b8cadc',fontsize=10)
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
    try:
        for extension in ('png','pdf','svg'):fig.savefig(output/('halo_formation_atlas.'+extension),dpi=180,facecolor=fig.get_facecolor())
    finally:plt.close(fig)
    from core.foundation import atomic_json,file_sha256
    record={'schema':'halo-formation-atlas-v1','science_acceptance':'NOT_ACCEPTED','source':result.get('source'),
        'particle_floor':result.get('particle_floor'),'final_snapshot':result.get('final_snapshot'),
        'selection':'Most numerous group within a factor %.4g in final Mvir; ties prefer larger masses; capped at %d.'%(mass_factor,max_halos),
        'formation_definition':'First upward crossing of half the final Mvir, interpolated in log mass versus scale factor; unavailable if no crossing is bracketed. Lines connect saved points and are not additional measurements.',
        'halos':[{'final_halo_id':str(h['final_halo_id']),'final_mass_msun_h':h['final_mass_msun_h'],
                  'formation_redshift':h.get('formation_redshift'),'main_branch_merger_count':len(h.get('merger_events',[])),
                  'track':[{key:(str(value) if key in ('halo_id','original_halo_id') else value)
                            for key,value in point.items()} for point in h['track']]} for h in selected],
        'artifacts':{('halo_formation_atlas.'+ext):file_sha256(output/('halo_formation_atlas.'+ext)) for ext in ('png','pdf','svg')}}
    atomic_json(output/'halo_formation_atlas.json',record)
    return output/'halo_formation_atlas.png'
