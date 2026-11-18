"""Build an offline, curated navigation page for existing scientific outputs."""
from __future__ import annotations
import html
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote


def local_href(target,page_directory):
    """URL-encode a filesystem-relative link without interpreting it as a URL."""
    return quote(os.path.relpath(target,page_directory),safe='/')


def verify_results_gallery(directory):
    """Check the page and every explicitly recorded target against saved hashes."""
    from core.foundation import file_sha256
    root=Path(directory).expanduser().resolve();manifest=root/'gallery_manifest.json'
    before=file_sha256(manifest);record=json.loads(manifest.read_text())
    if record.get('schema')!='visual-results-gallery-v1' or not isinstance(record.get('items'),list):
        raise ValueError('Unsupported results gallery manifest')
    if file_sha256(root/'index.html')!=record['index_sha256']:raise ValueError('Gallery page changed')
    checked={}
    for item in record['items']:
        if not isinstance(item,dict) or not isinstance(item.get('source_sha256'),dict):raise ValueError('Malformed gallery item')
        expected_paths={item.get(k) for k in ('path','preview','contract') if item.get(k) is not None}
        if set(item['source_sha256'])!=expected_paths:raise ValueError('Gallery item source inventory is incomplete')
        for name,digest in item['source_sha256'].items():
            path=Path(name)
            if not path.is_absolute() or not path.is_file() or file_sha256(path)!=digest:
                raise ValueError('Gallery source is missing or changed: '+str(path))
            checked[name]=digest
    if file_sha256(manifest)!=before:raise ValueError('Gallery manifest changed during verification')
    return {'status':'INTEGRITY_VERIFIED','item_count':len(record['items']),'unique_source_count':len(checked),
            'scope':'Checks recorded page and target file bytes, not linked dependencies or scientific acceptance.'}


def create_results_gallery(request,destination,*,base=None):
    from core.foundation import atomic_json,file_sha256,publish_directory
    out=Path(destination).expanduser().resolve();base=Path(base or '.').expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh results gallery directory')
    if not isinstance(request,dict) or set(request)!={'schema','title','intro','items'} or request['schema']!='visual-results-gallery-request-v1' or not isinstance(request['title'],str) or not request['title'] or not isinstance(request['intro'],str):raise ValueError('Valid gallery request required')
    items=request['items']
    if not isinstance(items,list) or not items or len(items)>500:raise ValueError('Select one to 500 explicit results')
    permitted={'title','category','description','path','preview','contract','note'};required={'title','category','description','path'}
    prepared=[]
    def resolve(value):
        if not isinstance(value,str) or not value:raise ValueError('Gallery paths must be nonempty strings')
        p=Path(value).expanduser();p=(p if p.is_absolute() else base/p).resolve()
        if not p.is_file():raise FileNotFoundError('Gallery file unavailable: '+str(p))
        return p
    for item in items:
        if not isinstance(item,dict) or not required<=set(item) or set(item)-permitted or any(not isinstance(item[k],str) or not item[k] for k in required):raise ValueError('Each gallery item needs title, category, description and path')
        if 'note' in item and not isinstance(item['note'],str):raise ValueError('Item note must be text')
        path=resolve(item['path']);preview=resolve(item['preview']) if item.get('preview') else None;contract=resolve(item['contract']) if item.get('contract') else None
        if path.suffix.lower() not in ('.html','.png','.jpg','.jpeg','.svg','.pdf','.mp4','.csv'):raise ValueError('Unsupported gallery output type')
        if preview and preview.suffix.lower() not in ('.png','.jpg','.jpeg','.svg'):raise ValueError('Preview must be an image')
        if contract:
            with contract.open(encoding='utf8') as f:json.load(f)
        sources={str(p):file_sha256(p) for p in (path,preview,contract) if p is not None}
        prepared.append({'title':item['title'],'category':item['category'],'description':item['description'],'note':item.get('note',''),'path':path,'preview':preview,'contract':contract,'source_sha256':sources})
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.results-gallery-',dir=out.parent))
    try:
        cards=[]
        for row in prepared:
            escape=html.escape;href=lambda p:escape(local_href(p,out),quote=True)
            if row['preview']:
                image='<img loading="lazy" src="%s" alt="Preview of %s">'%(href(row['preview']),escape(row['title'],quote=True))
            elif row['path'].suffix.lower() in ('.png','.jpg','.jpeg','.svg'):
                image='<img loading="lazy" src="%s" alt="%s">'%(href(row['path']),escape(row['title'],quote=True))
            elif row['path'].suffix.lower()=='.mp4':
                image='<video controls preload="metadata" src="%s" aria-label="%s"></video>'%(href(row['path']),escape(row['title'],quote=True))
            else:image='<div class="placeholder">%s</div>'%escape(row['category'])
            note='<p class="note">%s</p>'%escape(row['note']) if row['note'] else ''
            contract='<a class="contract" href="%s">Source record</a>'%href(row['contract']) if row['contract'] else ''
            cards.append('<article class="card" data-category="%s"><div class="visual">%s</div><div class="content"><span class="tag">%s</span><h2>%s</h2><p>%s</p>%s<div class="actions"><a class="button" href="%s">Open result ↗</a>%s</div></div></article>'%(escape(row['category'],quote=True),image,escape(row['category']),escape(row['title']),escape(row['description']),note,href(row['path']),contract))
        categories=sorted({row['category'] for row in prepared});options=''.join('<option value="%s">%s</option>'%(html.escape(x,quote=True),html.escape(x)) for x in categories)
        page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title><style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#07111e;color:#eef8ff;font:16px/1.5 system-ui}header{padding:52px 6vw 38px;background:radial-gradient(ellipse at 75%% 0%%,#244b62,#101c31 58%%,#091422)}main{max-width:1500px;margin:auto;padding:26px 6vw 70px}h1{font-size:clamp(30px,5vw,54px);letter-spacing:-.04em;line-height:1.08;margin:0 0 14px}header p{max-width:780px;color:#c0d4e5;font-size:18px}.eyebrow,.tag{color:#75e4e8;text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:700}.toolbar{display:flex;gap:14px;align-items:center;margin:0 0 25px;color:#bbd0e1}select{font:inherit;padding:9px 12px;border:1px solid #38526a;border-radius:8px;background:#13273a;color:white}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,310px),1fr));gap:20px}.card{background:#102033;border:1px solid #263d53;border-radius:15px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 10px 30px #0003}.visual{display:block;background:#07111e;aspect-ratio:16/9;overflow:hidden}.visual img,.visual video{width:100%%;height:100%%;object-fit:contain}.placeholder{display:grid;place-items:center;height:100%%;font-size:26px;color:#78cbd7;background:radial-gradient(circle at center,#22445d,#0a1727)}.content{padding:20px;display:flex;flex:1;flex-direction:column}.content h2{font-size:21px;line-height:1.2;margin:7px 0}.content p{color:#b8ccdd;margin:6px 0 12px}.content .note{color:#e7c992;font-size:13px}.actions{display:flex;gap:16px;align-items:center;margin-top:auto;padding-top:12px}.actions a{color:#8ce9ee;text-decoration:none}.button{background:#76dfe4;color:#071923!important;padding:8px 12px;border-radius:7px;font-weight:700}.contract{font-size:13px}[hidden]{display:none!important}</style><header><div class="eyebrow">Simulation results</div><h1>%s</h1><p>%s</p></header><main><div class="toolbar"><label for="category">Explore</label><select id="category"><option value="all">All results</option>%s</select><span id="count"></span></div><div class="grid">%s</div><p style="color:#91a8bd;margin-top:35px">This page links to existing local files. File hashes were recorded when the gallery was built; individual results retain their own scientific limitations and source records.</p></main><script>const selector=document.querySelector('#category'),cards=[...document.querySelectorAll('.card')],count=document.querySelector('#count');function show(){let n=0;for(const c of cards){c.hidden=selector.value!=='all'&&c.dataset.category!==selector.value;if(!c.hidden)n++}count.textContent=n+' results'}selector.addEventListener('change',show);show()</script></html>'''%(html.escape(request['title']),html.escape(request['title']),html.escape(request['intro']),options,''.join(cards))
        (stage/'index.html').write_text(page,encoding='utf8')
        for row in prepared:
            for path,digest in row['source_sha256'].items():
                if file_sha256(path)!=digest:raise ValueError('Gallery source changed during publication: '+path)
        serial=[{**{k:v for k,v in row.items() if k not in ('path','preview','contract')},'path':str(row['path']),'preview':str(row['preview']) if row['preview'] else None,'contract':str(row['contract']) if row['contract'] else None} for row in prepared]
        atomic_json(stage/'gallery_manifest.json',{'schema':'visual-results-gallery-v1','title':request['title'],'items':serial,'algorithm_sha256':file_sha256(__file__),'index_sha256':file_sha256(stage/'index.html'),'scope':'Navigation of explicitly selected local outputs; hashes identify files at generation time and do not establish scientific acceptance.'})
        publish_directory(stage,out);return out/'index.html'
    except BaseException as error:
        atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
