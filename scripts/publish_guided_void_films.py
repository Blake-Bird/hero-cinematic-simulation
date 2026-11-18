#!/usr/bin/env python3
"""Add source-bound route context to the existing z=0 void camera movies."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter
import imageio.v2 as imageio
import matplotlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import file_sha256,atomic_json
from core.cosmic_flythrough import display_limits,color_density,perspective_density_raster

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);root=parser.parse_args().atlas.resolve()
 for model in ('EDE','LCDM'):
  folder=root/'04_Movies'/model/f'{model}_z_199_to_000_void_explorer';movie=folder/'void_flythrough.mp4';contract_path=folder/'void_contract.json';contract=json.loads(contract_path.read_text())
  if contract.get('guided_display',{}).get('algorithm_sha256')==file_sha256(__file__):continue
  previous=root/'Source/Quality_review/Previous_void_publication'/model;previous.mkdir(parents=True,exist_ok=True)
  for p in (contract_path,):
   if not (previous/p.name).exists():shutil.copy2(p,previous/p.name)
  original=json.loads((previous/'void_contract.json').read_text())
  sample=folder/'movie_sample.npz'
  if file_sha256(sample)!=original['artifacts'][sample.name]:raise ValueError('Changed void camera sample')
  with np.load(sample,allow_pickle=False) as arrays:camera=arrays['camera_positions_unwrapped_mpc_h'];directions=arrays['camera_directions']
  latest=original['epochs'][-1];box=original['box_mpc_h'];target=np.array(latest['center_mpc_h']);density_path=folder/latest['volume']
  if file_sha256(density_path)!=original['artifacts'][density_path.name]:raise ValueError('Changed void density')
  field=np.load(density_path,allow_pickle=False);plane=field[:,:,latest['center_grid'][2]].T
  norm=np.clip((np.log10(np.maximum(plane,1e-6))+2)/3.5,0,1);map_image=Image.fromarray(np.flipud(matplotlib.colormaps['magma'](norm,bytes=True)[...,:3])).resize((272,272),Image.Resampling.BILINEAR)
  records=original['movie_frames'];projection=root/'Source/Quality_review/Guided_void_rasters'/model;projection.mkdir(parents=True,exist_ok=True)
  display_records=[]
  for i,row in enumerate(records):
   raster=perspective_density_raster(field,camera[i],directions[i],box,960,540,60.)
   p=projection/f'frame_{i:04d}.npy';np.save(p,raster,allow_pickle=False);display_records.append({'path':str(p.relative_to(root)),'sha256':file_sha256(p)})
  low,high=display_limits(np.load(root/r['path'],allow_pickle=False) for r in display_records)
  fontpath=str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf');title=ImageFont.truetype(fontpath,28);font=ImageFont.truetype(fontpath,17);small=ImageFont.truetype(fontpath,14)
  stage=previous/'guided.staged.mp4';fps=original['movie_fps'];mapx,mapy=992,106
  def xy(point):return (mapx+float(point[0]%box)/box*272,mapy+272-float(point[1]%box)/box*272)
  with imageio.get_writer(stage,fps=fps,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart']) as writer:
   for i,row in enumerate(records):
    raster=np.load(root/display_records[i]['path'],allow_pickle=False);rgb=np.flipud(color_density(raster,low,high));view=Image.fromarray(rgb).resize((960,540),Image.Resampling.LANCZOS)
    image=Image.new('RGB',(1280,720),'#08131f');image.paste(view,(0,94));d=ImageDraw.Draw(image)
    d.text((24,14),f'INTO THE UNDERDENSE COSMIC WEB · {model}',font=title,fill='#edf2e5');d.text((24,54),'z = 0 · snapshot 085 · spatial camera move through one frozen saved epoch',font=font,fill='#aacbd7')
    d.rectangle((976,84,1279,652),fill='#102533');d.text((992,82),'WHERE AM I?',font=font,fill='#e5efde');image.paste(map_image,(mapx,mapy));d=ImageDraw.Draw(image)
    points=[xy(p) for p in camera]
    for a,b in zip(points,points[1:]):
     if np.linalg.norm(np.subtract(a,b))<136:d.line([a,b],fill='#d8f0f4',width=2)
    tx,ty=xy(target);cx,cy=xy(camera[i]);d.ellipse((tx-8,ty-8,tx+8,ty+8),outline='#ffca72',width=3);d.line((tx-11,ty,tx+11,ty),fill='#ffca72',width=2);d.line((tx,ty-11,tx,ty+11),fill='#ffca72',width=2)
    d.ellipse((cx-5,cy-5,cx+5,cy+5),fill='#ffffff',outline='#071421',width=2)
    projected=np.asarray(directions[i][:2],float);length=np.linalg.norm(projected)
    if length>0:d.line((cx,cy,cx+18*projected[0]/length,cy-18*projected[1]/length),fill='white',width=3)
    distance=float(np.linalg.norm(target-camera[i]));info=[('Gold cross: density minimum','#ffca72'),('White dot / arrow: camera','#e3f3f3'),('White line: camera route','#e3f3f3'),('XY slice at target depth','#a9c8d3'),('Dark: underdense; bright: dense','#a9c8d3'),('','white'),(f'Distance: {distance:.3f} Mpc/h','#e3f3f3'),(f'Minimum rho / mean: {latest["minimum_density_ratio"]:.3f}','#a9c8d3'),('','white'),('This is a density minimum,','#e3f3f3'),('not a halo or a star.','#e3f3f3')]
    for n,(text,color) in enumerate(info):d.text((992,394+n*22),text,font=small,fill=color)
    d.text((24,657),'Smoothed 3-D dark-matter density · 28 samples per viewing ray · fixed density palette',font=font,fill='#aacbd7');d.text((24,686),'Purpose: locate and approach a low-density region. This camera tour does not show cosmic evolution.',font=small,fill='#aacbd7');writer.append_data(np.asarray(image))
    if i==len(records)//2:image.save(movie.with_suffix('.poster.jpg'))
  stage.replace(movie)
  receipt={'schema':'guided-void-display-v1','algorithm_sha256':file_sha256(__file__),'original_contract':str((previous/'void_contract.json').relative_to(root)),'original_contract_sha256':file_sha256(previous/'void_contract.json'),'original_movie_retained':False,'original_movie_sha256':original['artifacts'][movie.name],'numeric_camera_sample_sha256':file_sha256(sample),'density_slice_sha256':file_sha256(density_path),'projection_raster_sha256':[r['sha256'] for r in records],'display_projection':'Equal-length density-ray average through the retained periodic 48-cubed smoothed density grid; 28 samples per ray. Morphology display, not emitted light.', 'display_rasters':display_records,'perspective_display_log10_limits':[float(low),float(high)],'map_log10_density_ratio_limits':[-2.,1.5],'map_definition':'Full periodic-box XY slice at target grid z; gold cross is measured minimum, white dot and arrow show saved camera position and direction.','published_sha256':file_sha256(movie)}
  contract['guided_display']=receipt;contract['artifacts'][movie.name]=file_sha256(movie);atomic_json(contract_path,contract)
  atomic_json(folder/'film_source.json',{'title':'Guided approach to the z = 0 density minimum','model':model,'redshifts':'Frozen saved epoch z = 0','definition':'One frozen saved snapshot. The retained smoothed dark-matter density volume is shown with a perspective ray-average view. The target is a measured smoothed-density minimum, not a watershed void boundary. No physical time evolution is depicted.','guided_display':receipt,'published_sha256':file_sha256(movie)})
  print('GUIDED_VOID_PUBLISHED',model,flush=True)
if __name__=='__main__':main()
