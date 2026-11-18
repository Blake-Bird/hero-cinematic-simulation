#!/usr/bin/env python3
"""Audit local atlas links, image decoding, MP4 frames and HTTP asset responses."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote,quote
from concurrent.futures import ThreadPoolExecutor
import argparse,json,urllib.request
from PIL import Image
import imageio.v2 as imageio


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src','poster') and v)


def audit(root,base_url=None):
    root=Path(root).resolve();atlas=root/'analysis';missing=[];targets=set();links=0
    for page in atlas.rglob('*.html'):
        if 'Previous_visual_attempts' in page.parts or any(p.is_symlink() for p in page.parents if p!=atlas):continue
        parser=Links();parser.feed(page.read_text())
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path:continue
            target=(page.parent/unquote(u.path)).absolute();links+=1
            if not target.exists():missing.append({'page':str(page.relative_to(atlas)),'link':link})
            else:targets.add(target)
    # Runtime-selected maps are embedded into the story state, rather than HTML src attributes.
    targets.update((atlas/'04_Evolution/Halo_merger_story').glob('map_*.png'))
    image_checks=[]
    for p in sorted({p.resolve() for p in targets if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')}):
        try:
            with Image.open(p) as im:im.load();size=list(im.size)
            image_checks.append({'path':str(p),'status':'DECODED','size':size})
        except Exception as e:image_checks.append({'path':str(p),'status':'FAIL','reason':str(e)})
    movies=[]
    for p in sorted({p.resolve() for p in targets if p.suffix=='.mp4'}):
        try:
            with imageio.get_reader(p) as reader:
                m=reader.get_meta_data();fps=m['fps'];duration=m['duration']
                indices=[0,int(duration*fps/2),max(0,int((duration-.15)*fps))]
                for i in indices:reader.get_data(i)
            movies.append({'path':str(p),'status':'DECODED_FIRST_MIDDLE_LAST','duration':duration,'fps':fps})
        except Exception as e:movies.append({'path':str(p),'status':'FAIL','reason':str(e)})
    http=[]
    if base_url:
        def check(p):
            relative=p.relative_to(root);url=base_url.rstrip('/')+'/'+quote(str(relative))
            try:
                with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=20) as r:
                    if p.is_file() and int(r.headers.get('Content-Length',-1))!=p.stat().st_size:raise ValueError('HTTP length differs from saved asset')
                    return {'path':str(relative),'status':r.status}
            except Exception as e:return {'path':str(relative),'status':'FAIL','reason':str(e)}
        with ThreadPoolExecutor(max_workers=8) as pool:http=list(pool.map(check,sorted(targets)))
    failures=missing+[r for r in image_checks+movies if r['status']=='FAIL']+[r for r in http if r['status']=='FAIL']
    report={'checked_local_links':links,'missing_local_links':missing,'images':image_checks,'movies':movies,'http':http,'failure_count':len(failures)}
    (atlas/'06_Data_and_provenance/Loading_and_media_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'links':links,'images':len(image_checks),'movies':len(movies),'http_assets':len(http),'failures':len(failures),'details':failures[:10]},indent=2))
    return not failures


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--url');a=p.parse_args()
    raise SystemExit(0 if audit(a.run,a.url) else 1)
