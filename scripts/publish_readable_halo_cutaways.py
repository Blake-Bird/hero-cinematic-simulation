#!/usr/bin/env python3
"""Make sparse halo cutaways legible without changing retained measurements."""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import matplotlib
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import file_sha256,atomic_json

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);root=parser.parse_args().atlas.resolve()
 for source in root.glob('0[23]_*/Spatial_explorers/*/Most_massive_FOF_halo_anatomy/anatomy_data.npz'):
  parent=source.parent;contract=json.loads((parent/'anatomy_contract.json').read_text())
  if file_sha256(source)!=contract['artifacts'][source.name]:raise ValueError(f'Changed halo array {source}')
  with np.load(source,allow_pickle=False) as a:cube=a['cube_mass_msun_h']
  n,m,layers=cube.shape
  if n!=96 or m!=96:raise ValueError('Expected retained 96 by 96 grid')
  cube=cube.reshape(32,3,32,3,layers).sum(axis=(1,3));positive=cube[cube>0];low=float(np.log10(positive).min());high=float(np.log10(positive).max());high=max(high,low+.01)
  out=parent/'Readable_cutaway';out.mkdir(exist_ok=True)
  occupied=[]
  for k in range(layers):
   plane=cube[:,:,k].T;occupied.append(int(np.count_nonzero(plane)));norm=np.clip((np.log10(np.maximum(plane,10**low))-low)/(high-low),0,1);rgb=matplotlib.colormaps['magma'](.3+.7*norm,bytes=True)[...,:3];rgb[plane<=0]=0
   Image.fromarray(np.flipud(rgb)).resize((768,768),Image.Resampling.NEAREST).save(out/f'slice_{k:02d}.png')
  page=(parent/'halo_anatomy.html').read_text().replace('src="radial_profiles.png"','src="../radial_profiles.png"')
  page=re.sub(r'(id="depth"[^>]*value=")\d+',lambda m:m[1]+str(int(np.argmax(occupied))),page)
  page=re.sub(r'Colors share a fixed log10 mass-per-cell scale from .*?\[Msun/h\]; empty cells are black\.',f'For visibility, the retained 96² image grid is summed into 32² display cells. Colors share a fixed log10 mass-per-display-cell scale from {low:.3g} to {high:.3g} [Msun/h]; empty cells are black. The lowest positive mass has a visible color floor.',page)
  note='<p style="border:1px solid #526d7b;padding:14px">Sparse cells show discrete saved particles, not a smooth resolved density field. Starts at the most occupied depth slice. <a style="color:#7bf8ff" href="../halo_anatomy.html">Original 96² cutaway</a> · <a style="color:#7bf8ff" href="display_source.json">Display source</a></p>'
  page=page.replace('<h2>Dark-matter cutaway</h2>','<h2>Dark-matter cutaway</h2>'+note)
  page=page.replace("const data=",'const occupied='+json.dumps(occupied)+';const data=',1)
  page=page.replace('`${z0.toPrecision(4)} to ${z1.toPrecision(4)} comoving Mpc/h`','`${z0.toPrecision(4)} to ${z1.toPrecision(4)} comoving Mpc/h · ${occupied[k]} occupied display cells`')
  (out/'index.html').write_text(page)
  atomic_json(out/'display_source.json',{'source':'../anatomy_data.npz','sha256':file_sha256(source),'display':'Sum 3 by 3 input cells; depth slices unchanged; fixed logarithmic ordered palette with visible positive floor; no particle smoothing or interpolation.','grid':[32,32,layers],'occupied_cells_per_slice':occupied,'log10_mass_per_cell_limits':[low,high],'algorithm_sha256':file_sha256(__file__)})
  print('READABLE_CUTAWAY',out,flush=True)
if __name__=='__main__':main()
