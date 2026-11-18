#!/usr/bin/env python3
"""Publish readable dome views from verified saved angular mass rasters."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import file_sha256,atomic_json

def rebuild(root):
 root=Path(root).resolve();qa=root/'Source/Quality_review';cache=qa/'Dome_numeric_display';cache.mkdir(parents=True,exist_ok=True)
 fields={};records={};limits=[]
 for model,h in (('EDE',.7483),('LCDM',.6781)):
  source=next((root.parent/f'dmo3_N128_L5_fast128_{model.lower()}/results/planetarium').glob('timeline_*/timeline.json'));timeline=json.loads(source.read_text());records[model]=[];fields[model]=[]
  for row in timeline['frames']:
   path=source.parent/(row['path']+'.product')/(row['path']+'.data');contract=json.loads((path/'contract.json').read_text());data=path/'arrays.npz'
   if file_sha256(data)!=contract['sha256']:raise ValueError(f'Changed dome raster {data}')
   with np.load(data,allow_pickle=False) as arrays:raster=arrays['angular_scalar_raster']
   # Scalar mass per solid angle is sampled onto the retained fisheye raster.
   # Averaging display cells and smoothing aid visibility; neither is a new measurement.
   n=raster.shape[0];factor=n//512
   if n%512:raise ValueError('Expected 512-divisible source raster')
   field=gaussian_filter(np.nan_to_num(raster).reshape(512,factor,512,factor).mean(axis=(1,3))/h,1.,mode='constant')
   yy,xx=np.ogrid[:512,:512];mask=(xx+.5-256)**2+(yy+.5-256)**2<=256**2;field[~mask]=0
   fields[model].append(field.astype('float32'));limits.append(np.percentile(np.log10(field[mask&(field>0)]),[10,99.5]))
   records[model].append({'snapshot':row['snapshot'],'redshift':row['source']['redshift'],'source':str(data),'sha256':contract['sha256']})
  print('RASTERS_VERIFIED',model,len(fields[model]),flush=True)
 low=min(x[0] for x in limits);high=max(x[1] for x in limits)
 fontpath=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf';font=ImageFont.truetype(str(fontpath),18);small=ImageFont.truetype(str(fontpath),13)
 for model in ('EDE','LCDM'):
  for slow in (False,True):
   title='Planetarium domemaster'+(' slow' if slow else '');folder=root/'04_Movies'/model/f'{model}_z_199_to_000_{title.replace(" ","_")}';output=folder/(folder.name+'.mp4');receipt=folder/'film_source.json'
   old=json.loads(receipt.read_text())
   if old.get('display_generation')=='verified-raster-browser-v2':continue
   archive=qa/'Previous_dome_publication'/folder.name;archive.mkdir(parents=True,exist_ok=True)
   for p in (receipt,):
    if p.exists() and not (archive/p.name).exists():shutil.copy2(p,archive/p.name)
   temp=cache/(folder.name+'.mp4')
   with imageio.get_writer(temp,fps=24,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
    for i,field in enumerate(fields[model]):
     normal=np.clip((np.log10(np.maximum(field,10**low))-low)/(high-low),0,1);rgb=matplotlib.colormaps['magma'](.2+.8*normal,bytes=True)[...,:3];rgb[field<=0]=0
     frame=Image.new('RGB',(768,896),'#091722');frame.paste(Image.fromarray(rgb).resize((736,736),Image.Resampling.LANCZOS),(16,72));d=ImageDraw.Draw(frame);row=records[model][i]
     d.text((22,14),f'{model} · all-sky dark matter · saved z = {row["redshift"]:.4g}',font=font,fill='#eceddf')
     d.text((22,44),f'Snapshot {row["snapshot"]:03d} · observer-centered hemispheric dome projection',font=small,fill='#adc5ce')
     d.text((22,817),'Display smoothing: 1 pixel at 512² · shared fixed log mass-per-solid-angle palette',font=small,fill='#adc5ce')
     d.text((22,840),f'log10 display range {low:.2f}–{high:.2f} [Msun/sr] · sampled DM, no stellar light',font=small,fill='#adc5ce')
     d.text((22,863),'Discrete saved epochs · display time is not physical elapsed time',font=small,fill='#adc5ce')
     for _ in range(12 if slow else 1):writer.append_data(np.asarray(frame))
     if i==len(fields[model])//2:frame.save(output.with_suffix('.poster.jpg'))
   temp.replace(output)
   atomic_json(receipt,{'title':title+' · readable sky view','model':model,'redshifts':'Saved epochs z=199 to z=0','display_generation':'verified-raster-browser-v2','source_frames':records[model],'display_definition':'Mean rebin retained angular scalar raster to 512²; Gaussian sigma 1 display pixel; common fixed logarithmic sequential palette across both runs and all epochs. Original dome exports retained separately. Screen view contains an explanatory border.','display_log10_msun_per_sr_limits':[float(low),float(high)],'fps':24,'frames_per_saved_epoch':12 if slow else 1,'published_sha256':file_sha256(output),'algorithm_sha256':file_sha256(__file__)})
   print('DOME_PUBLISHED',output,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);rebuild(parser.parse_args().atlas)
