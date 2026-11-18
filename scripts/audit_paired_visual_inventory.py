#!/usr/bin/env python3
from pathlib import Path
from PIL import Image,ImageDraw,ImageStat
import json,subprocess
import argparse
parser=argparse.ArgumentParser(description='Inventory published images and files; generate figure contact sheets and three frames per movie for human visual review.')
parser.add_argument('atlas',type=Path)
root=parser.parse_args().atlas.resolve();out=root/'Source/Quality_review';out.mkdir(parents=True,exist_ok=True)
images=[];issues=[];strays=[];empty=[];broken=[]
for top in root.iterdir():
    if not top.is_dir() or top.name=='Source':continue
    for p in top.rglob('*'):
        if p.is_symlink() and not p.exists():broken.append(str(p.relative_to(root)))
        if not p.is_file() or 'Visual_QA' in p.parts:continue
        if p.stat().st_size==0:empty.append(str(p.relative_to(root)))
        if p.name.startswith(('._','.DS_Store')) or p.suffix in ('.tmp','.part','.bak','.blend1') or 'staged' in p.name:strays.append(str(p.relative_to(root)))
        if p.suffix.lower()=='.png':
            with Image.open(p) as im:
                im.load();thumb=im.convert('RGB');thumb.thumbnail((128,128));std=max(ImageStat.Stat(thumb).stddev)
                if min(im.size)<100 or std<2:issues.append({'path':str(p.relative_to(root)),'size':im.size,'std':std})
            images.append(p)
def sheets(items,prefix):
    paths=[]
    for n in range(0,len(items),20):
        sheet=Image.new('RGB',(1600,1450),'#17212b');d=ImageDraw.Draw(sheet)
        for j,(p,label) in enumerate(items[n:n+20]):
            x=j%4*400;y=j//4*290
            with Image.open(p) as im:
                im=im.convert('RGB');im.thumbnail((390,248));sheet.paste(im,(x+(400-im.width)//2,y))
            d.text((x+5,y+250),label[:55],fill='white');d.text((x+5,y+265),label[55:110],fill='white')
        target=out/f'{prefix}_{n//20+1:02d}.jpg';sheet.save(target,quality=90);paths.append(str(target.relative_to(root)))
    return paths
paired=[(p,str(p.relative_to(root/'01_Comparisons'))) for p in images if p.is_relative_to(root/'01_Comparisons')]
# Representative native panels: all diagnostic families at z=0 and sparse high-z.
native=[];seen=set()
for p in images:
    if '/Saved_epochs/' not in str(p) or not any(t in p.name for t in ('snap_085','snap_021')):continue
    key=p.name.split('_snap_',1)[-1]
    if key in seen:continue
    seen.add(key);native.append((p,p.name))
film=[]
for i,p in enumerate(sorted((root/'04_Movies').rglob('*.mp4'))):
    if any(s.startswith('.') for s in p.relative_to(root).parts):continue
    duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(p)],text=True))
    for fraction in (.1,.5,.9):
        target=out/f'film_{i:02d}_{int(fraction*100):02d}.jpg'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(duration*fraction),'-i',str(p),'-frames:v','1','-vf','scale=640:-1',str(target)],check=True)
        film.append((target,f'{p.parent.name} / {int(fraction*100)}%'))
report={'image_count':len(images),'suspect_images':issues,'empty_files':empty,'stray_files':strays,'broken_symlinks':broken,'contact_sheets':sheets(paired,'comparison')+sheets(native,'native')+sheets(film,'movies'),'sampled_movie_frames':len(film)}
(root/'07_Status_and_Provenance/final_visual_inventory.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
