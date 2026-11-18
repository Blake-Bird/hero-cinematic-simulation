"""A portable, synchronized merger tree, labelled density map and saved-epoch film.

Reads a frozen positioned tree and recorded measured density arrays. H0/H1 aliases
are display branch identities, never replacements for exact epoch-local halo IDs.
"""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import matplotlib

from core.foundation import atomic_json, file_sha256, publish_directory
from core.merger_tree_visual import branch_identity, tree_coordinates, export_merger_tree
from core.cosmic_flythrough import density_palette


def label_positions(points, width, height, *, label_width=150, label_height=30):
    """Choose non-overlapping label rectangles with short visible leader lines."""
    occupied = []; result = []
    for x, y in points:
        candidates = [(x+18,y-35),(x+18,y+12),(x-label_width-18,y-35),
                      (x-label_width-18,y+12),(x+18,y-75),(x-label_width-18,y+45)]
        best = None
        for xx, yy in candidates:
            xx = float(np.clip(xx, 8, max(8,width-label_width-8)))
            yy = float(np.clip(yy, 8, max(8,height-label_height-8)))
            box = (xx, yy, xx+label_width, yy+label_height)
            overlap = sum(max(0,min(box[2],b[2])-max(box[0],b[0])) *
                          max(0,min(box[3],b[3])-max(box[1],b[1])) for b in occupied)
            score = overlap*100 + (xx-x)**2 + (yy-y)**2
            if best is None or score < best[0]:best=(score,box)
        occupied.append(best[1]);result.append(best[1])
    return result


def _font(size):
    return ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),size)


def tree_image(tree, identities, snapshot, *, width=740, height=720):
    """Render the same branched topology used by the browser, with an epoch cursor."""
    image=Image.new('RGB',(width,height),'#0c2030');draw=ImageDraw.Draw(image)
    x,a=tree_coordinates(tree);nodes=tree['nodes']
    xx=60+(width-120)*(x-x.min())/max(1,np.ptp(x));yy=45+(height-100)*(a-a.min())/max(.00001,np.ptp(a))
    current=[i for i,n in enumerate(nodes) if n['snapshot']==snapshot]
    current_a=nodes[current[0]]['scale_factor'] if current else -1
    for parent,child in tree['edges']:
        start=np.array([xx[parent],yy[parent]]);end=np.array([xx[child],yy[child]])
        mid=(start[1]+end[1])/2;c0=np.array([start[0],mid]);c1=np.array([end[0],mid]);t=np.linspace(0,1,30)[:,None]
        points=(1-t)**3*start+3*(1-t)**2*t*c0+3*(1-t)*t*t*c1+t**3*end
        color=identities[parent]['color'] if a[child]<=current_a+1e-9 else '#284153'
        draw.line([tuple(p) for p in points],fill=color,width=3 if identities[parent]['branch']==0 else 2)
    maximum=max(n['mass_msun_h'] for n in nodes)
    for i,n in enumerate(nodes):
        r=3+7*math.sqrt(n['mass_msun_h']/maximum);color=identities[i]['color'] if a[i]<=current_a+1e-9 else '#365264'
        draw.ellipse((xx[i]-r,yy[i]-r,xx[i]+r,yy[i]+r),fill=color)
        if i in current:draw.ellipse((xx[i]-r-4,yy[i]-r-4,xx[i]+r+4,yy[i]+r+4),outline='#fff1bf',width=2)
    for branch in sorted({v['branch'] for v in identities}):
        tips=[i for i,v in enumerate(identities) if v['branch']==branch];i=min(tips,key=lambda i:a[i])
        draw.text((xx[i]+6,yy[i]-25),identities[i]['alias'],font=_font(15),fill=identities[i]['color'],stroke_width=1,stroke_fill='#0c2030')
    if current:
        cy=yy[current[0]];draw.line((0,cy,width,cy),fill='#ffd180',width=1)
        draw.text((7,min(cy+4,height-25)),'z %.3f'%nodes[current[0]]['redshift'],font=_font(14),fill='#ffe1a7')
    return image


def export_linked_halo_story(tree_path, density_contract, density_directory, destination, *, run_dir,
                             fps=24, hold_seconds=1):
    """Publish offline HTML + MP4 sharing exact nodes, labels, colors and epochs."""
    tree_path=Path(tree_path).resolve();contract_path=Path(density_contract).resolve()
    base=Path(density_directory).resolve();root=Path(run_dir).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve the previous halo story; choose a fresh directory')
    if type(fps) is not int or not 1<=fps<=60 or type(hold_seconds) is not int or not 1<=hold_seconds<=5:
        raise ValueError('Bounded integer playback controls required')
    th=file_sha256(tree_path);ch=file_sha256(contract_path)
    tree=json.loads(tree_path.read_text());source=json.loads(contract_path.read_text())
    if tree.get('schema')!='adjacent-merger-tree-v1' or not tree.get('nodes'):raise ValueError('A saved positioned merger tree is required')
    provenance_path=tree_path.parent/'analysis_provenance.json'
    selected_path=tree_path.parent/'selected_tree_contract.json'
    if selected_path.exists():
        selected=json.loads(selected_path.read_text())
        if selected.get('artifacts',{}).get(tree_path.name)!=th:
            raise ValueError('Selected tree content differs from its sealed source record')
        provenance_path=Path(selected['analysis_bundle'])/'analysis_provenance.json'
    if not provenance_path.is_file():
        raise ValueError('The source tree needs its original analysis provenance or selected-tree contract')
    provenance=json.loads(provenance_path.read_text())
    namespace=provenance.get('data_contract',{}).get('run_namespace')
    run_namespace=json.loads((root/'manifest.json').read_text()).get('run_namespace')
    if not namespace or namespace!=run_namespace:
        raise ValueError('Tree and density run must share the recorded run namespace')
    if not selected_path.exists():
        records=[r for r in provenance.get('output_artifacts',[]) if r.get('relative_path')==tree_path.name]
        if len(records)!=1 or records[0]['sha256']!=th:
            raise ValueError('Tree differs from its original sealed measured artifact')
    identity=branch_identity(tree);nodes=tree['nodes'];box=float(tree['box_mpc_h'])
    available={r['snapshot']:r for r in source['frames']};epochs=[]
    for snap in sorted({n['snapshot'] for n in nodes}):
        if snap not in available:raise ValueError('Missing density epoch '+str(snap))
        row=available[snap];matched=[i for i,n in enumerate(nodes) if n['snapshot']==snap]
        if any(not math.isclose(nodes[i]['redshift'],row['redshift'],rel_tol=1e-7,abs_tol=1e-7) for i in matched):raise ValueError('Tree and map redshifts disagree')
        for i in matched:
            p=np.asarray(nodes[i].get('position_mpc_h'),float)
            if p.shape!=(3,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>=box):raise ValueError('Tree node lacks a valid saved position')
        path=(base/row['numeric_asset']).resolve()
        if not path.is_relative_to(base):raise ValueError('Density array escaped its recorded directory')
        epoch={'snapshot':snap,'redshift':row['redshift'],'nodes':matched,'numeric_asset':str(path),'numeric_sha256':file_sha256(path),'source_files':row['sources']}
        for entry in row['sources']:
            p=Path(entry['path']);p=p if p.is_absolute() else root/p
            if file_sha256(p)!=entry['sha256']:raise ValueError('Native source changed: '+str(p))
        epochs.append(epoch)
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.halo-story-',dir=out.parent))
    try:
        export_merger_tree(tree,stage)
        writer=imageio.get_writer(stage/'Halo_identity_and_merger_tree.mp4',fps=fps,codec='libx264',quality=9,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
        try:
            for index,row in enumerate(epochs):
                with np.load(row['numeric_asset'],allow_pickle=False) as f:field=np.asarray(f['log10_projected_density_over_mean'])
                if field.ndim!=2 or not np.isfinite(field).all():raise ValueError('Finite 2D density raster required')
                rgb=(density_palette()(np.clip((field+1.3)/3.4,0,1))[...,:3]*255).astype(np.uint8)
                sky=Image.fromarray(np.flipud(rgb)).resize((720,720),Image.Resampling.LANCZOS)
                sky.save(stage/('map_%03d.png'%row['snapshot']))
                frame=Image.new('RGB',(1600,960),'#08131f');frame.paste(sky,(40,125));frame.paste(tree_image(tree,identity,row['snapshot']),(820,125))
                draw=ImageDraw.Draw(frame);draw.text((40,26),'WHICH HALO BECOMES THIS ONE?',font=_font(33),fill='#edf8ff')
                draw.text((40,75),'Saved epoch %03d  ·  z = %.4f  ·  final tree ID %s'%(row['snapshot'],row['redshift'],nodes[0]['id']),font=_font(20),fill='#b7d0e2')
                draw.text((40,855),'XY projected dark matter · full %.3f comoving Mpc/h box'%box,font=_font(18),fill='#afcbdc')
                draw.text((820,855),'Full ancestry · highlighted row is the current saved epoch',font=_font(18),fill='#afcbdc')
                ordered=sorted(row['nodes'],key=lambda i:identity[i]['branch'])
                points=[(nodes[i]['position_mpc_h'][0]/box*720,(1-nodes[i]['position_mpc_h'][1]/box)*720) for i in ordered]
                rects=label_positions(points,720,720,label_width=170,label_height=42)
                for i,(x,y),rect in zip(ordered,points,rects):
                    c=identity[i]['color'];xx=x+40;yy=y+125;r=10 if identity[i]['branch']==0 else 7
                    draw.ellipse((xx-r,yy-r,xx+r,yy+r),outline=c,width=3)
                    lx,ly,lx1,ly1=rect;lx+=40;ly+=125;lx1+=40;ly1+=125
                    draw.line((xx,yy,lx,ly+12),fill=c,width=2);draw.rounded_rectangle((lx,ly,lx1,ly1),radius=5,fill='#08131f',outline=c,width=1)
                    draw.text((lx+7,ly+3),identity[i]['alias']+' · ID '+nodes[i]['id'],font=_font(14),fill=c)
                    draw.text((lx+7,ly+22),'Mass %.2e Msun/h'%nodes[i]['mass_msun_h'],font=_font(12),fill='#c8ddeb')
                draw.text((40,905),'H0 = largest-progenitor branch. Other colors = linked branches. IDs change at saved epochs; no missing ancestry is invented.',font=_font(17),fill='#b4cada')
                for _ in range(fps*hold_seconds):writer.append_data(np.asarray(frame))
                if index in (0,len(epochs)//2,len(epochs)-1):frame.save(stage/('Preview_%03d.png'%row['snapshot']))
        finally:writer.close()
        row_index={r['snapshot']:i for i,r in enumerate(epochs)}
        payload={'nodes':nodes,'identity':identity,'epochs':[{k:r[k] for k in ('snapshot','redshift','nodes')} for r in epochs],
                 'hold_seconds':hold_seconds,'box':box,'root_id':nodes[0]['id'],'scope':tree['scope']}
        _write_story_html(stage,tree,payload,row_index)
        for row in epochs:
            if file_sha256(row['numeric_asset'])!=row['numeric_sha256']:raise ValueError('Density array changed during rendering')
        if file_sha256(tree_path)!=th or file_sha256(contract_path)!=ch:raise ValueError('Source tree or density contract changed')
        atomic_json(stage/'story_contract.json',{'schema':'linked-halo-story-v1','source_tree':str(tree_path),'source_tree_sha256':th,
            'density_contract':str(contract_path),'density_contract_sha256':ch,'algorithm_sha256':file_sha256(__file__),
            'source_tree_provenance':str(provenance_path),'source_tree_provenance_sha256':file_sha256(provenance_path),
            'run_namespace':namespace,
            'branch_algorithm_sha256':file_sha256(Path(__file__).with_name('merger_tree_visual.py')),
            'nodes':len(nodes),'branches':len({v['branch'] for v in identity}),'epochs':epochs,'fps':fps,'hold_seconds':hold_seconds,
            'scope':'Recorded adjacent descendant links and epoch-local catalogue positions; full-box XY density. Branch alias is not a persistent halo ID. '
                    'A merger edge is a catalogue relationship, not proof of a resolved physical collision. Missing or below-floor ancestry is omitted.',
            'movie_sha256':file_sha256(stage/'Halo_identity_and_merger_tree.mp4'),
            'artifacts':{str(p.relative_to(stage)):file_sha256(p) for p in stage.rglob('*') if p.is_file()}})
        publish_directory(stage,out);return out/'index.html'
    except BaseException as error:
        atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
        raise


def _write_story_html(directory, tree, payload, row_index):
    """Inline state, ordinary local images/video, no network or fetch dependency."""
    data=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    # Reuse the original tree renderer's exact geometry and shared branch identities.
    page=(directory/'merger_tree.html').read_text()
    svg=page.split('<div class="scroll">',1)[1].split('</svg>',1)[0]+'</svg>'
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Follow a halo · tree and movie</title>
<style>*{box-sizing:border-box}body{margin:0;background:#07131f;color:#e8f4ff;font:16px/1.5 system-ui}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:clamp(28px,4vw,48px);line-height:1.1;margin:12px 0}p{color:#acc8dd}a{color:#6ee8df}nav{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}.layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,.9fr);gap:22px}.card{min-width:0;background:#102336;border:1px solid #2a465c;border-radius:13px;padding:20px}.map{position:relative;aspect-ratio:1}.map img,.map svg{position:absolute;width:100%;height:100%;inset:0}.map svg{overflow:visible}.tree{height:650px;overflow:auto;background:#0b1d2d;border-radius:8px}.tree>svg{width:100%;min-width:460px;height:auto;display:block}.tree path{fill:none}.tree circle{cursor:pointer;stroke:#e0f9ff;stroke-width:1}.branch-label{font:bold 15px system-ui}.epoch-label{fill:#9db8ce;font:12px system-ui}button,select{background:#1d3c51;color:#ecf8ff;border:1px solid #43657b;padding:9px 12px;border-radius:7px}button{cursor:pointer}input[type=range]{width:100%;accent-color:#65e8dc}video{width:100%;display:block;border-radius:8px;background:#030b11}#detail{min-height:110px;overflow-wrap:anywhere}small{color:#9eb8cc}.status{color:#ffd28a}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}@media(max-width:900px){.layout{grid-template-columns:1fr}.tree{height:500px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}</style>
<main><a href="../index.html">← Evolution atlas</a><p>HALO IDENTITY · LINKED CATALOGUE ANCESTRY</p><h1>Follow the same branches.<br>In the tree. In the simulation.</h1><p id="coverage"></p><nav><a href="Halo_identity_and_merger_tree.mp4">Download labelled film</a><a href="halo_merger_histories.png">Tree PNG</a><a href="merger_tree.json">Exact node data</a><a href="story_contract.json">Sources and scope</a></nav>
<div class="controls"><button id="play">Play saved epochs</button><button id="reset">Root halo</button><button id="zoom">Enlarge tree</button><label>Branch <select id="branch"></select></label></div><input aria-label="Saved epoch" id="epoch" type="range" min="0" value="0"><p id="epoch-label" class="status" aria-live="polite"></p>
<div class="layout"><section class="card"><h2>Where is that halo?</h2><div class="map"><img id="map" alt="Saved full-box projected dark-matter density"><svg id="overlay" viewBox="0 0 720 720" aria-label="Labelled linked halo positions"></svg></div><p>XY projection · full comoving box · rings mark catalogue centers. Color shows projected density, not emitted light.</p><div id="detail" aria-live="polite"></div></section><section class="card"><h2>A branching merger tree</h2><p>Earlier at the top, later at the bottom. Click any node to show its saved position and seek the film.</p><div class="tree">TREE_SVG</div><p>H0 follows the largest incoming progenitor. H1, H2… are branch aliases; exact catalogue IDs are epoch-local. Missing and below-floor halos are not invented.</p></section></div>
<section class="card" style="margin-top:22px"><h2>The same identities in motion</h2><video id="film" controls preload="metadata" playsinline poster="Preview_POSTER.png"><source src="Halo_identity_and_merger_tree.mp4" type="video/mp4"></video><p id="video-status">The film shows saved epochs at equal display intervals, with no interpolated halo positions. Selecting a node seeks its epoch.</p></section></main>
<script>const D=STORY_DATA;let current=D.epochs.length-1,selected=0,timer=null,pendingSeek=null;const slider=document.querySelector('#epoch'),film=document.querySelector('#film'),map=document.querySelector('#map'),overlay=document.querySelector('#overlay'),branch=document.querySelector('#branch');const ns='http://www.w3.org/2000/svg';slider.max=D.epochs.length-1;document.querySelector('#coverage').textContent=`Final catalogue ID ${D.root_id} · ${D.nodes.length} linked nodes · ${D.epochs.length} saved epochs, z=${D.epochs[0].redshift.toFixed(3)} → ${D.epochs.at(-1).redshift.toFixed(3)}. ${D.scope}`;for(const b of [...new Set(D.identity.map(x=>x.branch))].sort((a,b)=>a-b)){let o=document.createElement('option');o.value=b;o.textContent='H'+b;branch.append(o)}
const circles=[...document.querySelectorAll('.tree circle')];function make(tag,attrs,text){let e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e}function show(index,seek=true,node=null){current=Math.max(0,Math.min(D.epochs.length-1,index));const ep=D.epochs[current];slider.value=current;document.querySelector('#epoch-label').textContent=`Saved snapshot ${ep.snapshot} · z=${ep.redshift.toFixed(5)} · ${ep.nodes.length} positioned progenitors`;map.src='map_'+String(ep.snapshot).padStart(3,'0')+'.png';if(node!==null)selected=node;else{let match=ep.nodes.find(i=>D.identity[i].branch===+branch.value);selected=match===undefined?ep.nodes[0]:match}branch.value=D.identity[selected].branch;overlay.replaceChildren();const busy=[];for(const i of ep.nodes){const n=D.nodes[i],b=D.identity[i],x=n.position_mpc_h[0]/D.box*720,y=(1-n.position_mpc_h[1]/D.box)*720;const options=[[18,-35],[18,12],[-178,-35],[-178,12],[18,-75],[-178,45]];
const places=options.map(([dx,dy])=>{
 const left=Math.max(8,Math.min(552,x+dx)),top=Math.max(8,Math.min(678,y+dy));
 const overlap=busy.reduce((sum,[px,py])=>sum+Math.max(0,160-Math.abs(px-left))*Math.max(0,34-Math.abs(py-top)),0);
 return {left,top,score:overlap*100+(left-x)**2+(top-y)**2};
});
const {left:lx,top:ly}=places.reduce((a,b)=>a.score<=b.score?a:b);busy.push([lx,ly]);let g=make('g',{'data-node':i,role:'button',tabindex:0,'aria-label':`${b.alias}, catalogue ID ${n.id}`});g.append(make('circle',{cx:x,cy:y,r:i===selected?13:8,fill:'none',stroke:b.color,'stroke-width':i===selected?4:2}),make('line',{x1:x,y1:y,x2:lx,y2:ly+13,stroke:b.color,'stroke-width':1.5}),make('rect',{x:lx,y:ly,width:160,height:34,rx:5,fill:'#07131f',stroke:b.color}),make('text',{x:lx+7,y:ly+14,fill:b.color,'font-size':13,'font-family':'system-ui'},`${b.alias} · ID ${n.id}`),make('text',{x:lx+7,y:ly+28,fill:'#b9d5e6','font-size':10,'font-family':'system-ui'},`Mass ${n.mass_msun_h.toExponential(2)} Msun/h`));g.onclick=()=>show(current,true,i);g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(current,true,i)}};overlay.append(g)}circles.forEach((c,i)=>{const on=D.nodes[i].snapshot===ep.snapshot;c.style.opacity=on?'1':'.35';c.style.strokeWidth=i===selected?'5':on?'2':'1'});const n=D.nodes[selected],b=D.identity[selected];document.querySelector('#detail').textContent=`Selected ${b.alias} · exact catalogue ID ${n.id} · snapshot ${n.snapshot} · Mass ${n.mass_msun_h.toExponential(4)} Msun/h · position (${n.position_mpc_h.map(v=>v.toFixed(4)).join(', ')}) comoving Mpc/h. This ID identifies this node at this epoch.`;if(seek&&Number.isFinite(film.duration)){pendingSeek=Math.min(film.duration-.02,current*D.hold_seconds+.02);film.currentTime=pendingSeek}}circles.forEach((c,i)=>{const activate=()=>{let e=D.epochs.findIndex(e=>e.snapshot===D.nodes[i].snapshot);show(e,true,i)};c.addEventListener('click',activate);c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}})});slider.oninput=()=>show(+slider.value);branch.onchange=()=>{const candidates=D.nodes.map((n,i)=>i).filter(i=>D.identity[i].branch===+branch.value);const nearest=candidates.reduce((a,b)=>Math.abs(D.nodes[a].snapshot-D.epochs[current].snapshot)<=Math.abs(D.nodes[b].snapshot-D.epochs[current].snapshot)?a:b);show(D.epochs.findIndex(e=>e.snapshot===D.nodes[nearest].snapshot),true,nearest)};document.querySelector('#reset').onclick=()=>show(D.epochs.length-1,true,0);document.querySelector('#zoom').onclick=()=>{let s=document.querySelector('.tree>svg');s.style.width=s.style.width==='170%'?'100%':'170%'};document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play saved epochs'}else{if(current===D.epochs.length-1)show(0);timer=setInterval(()=>{if(current===D.epochs.length-1){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play saved epochs'}else show(current+1)},D.hold_seconds*1000);document.querySelector('#play').textContent='Pause epochs'}};film.addEventListener('timeupdate',()=>{if(pendingSeek!==null){if(Math.abs(film.currentTime-pendingSeek)>.25)return;pendingSeek=null}if(film.paused)return;let i=Math.min(D.epochs.length-1,Math.floor(film.currentTime/D.hold_seconds));if(i!==current)show(i,false)});film.addEventListener('seeked',()=>{if(pendingSeek!==null){if(Math.abs(film.currentTime-pendingSeek)<.25)pendingSeek=null;return}let i=Math.min(D.epochs.length-1,Math.floor(film.currentTime/D.hold_seconds));if(i!==current)show(i,false)});film.addEventListener('loadedmetadata',()=>{document.querySelector('#video-status').textContent='Film ready · '+film.duration.toFixed(1)+' seconds · node selection seeks to the matching epoch.'});film.addEventListener('error',()=>document.querySelector('#video-status').textContent='The browser could not decode this film. Use Download labelled film to open it in a video player.');map.onerror=()=>{document.querySelector('#epoch-label').textContent+=' · map failed to load'};show(current,false,0);</script></html>'''
    html=html.replace('TREE_SVG',svg).replace('STORY_DATA',data).replace('Preview_POSTER.png','Preview_%03d.png'%payload['epochs'][-1]['snapshot'])
    (directory/'index.html').write_text(html,encoding='utf-8')
