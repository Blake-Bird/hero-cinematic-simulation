"""Static, portable interface and descriptive file navigation for a paired atlas."""
from pathlib import Path
import json
import html
import os
import csv
import re
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter
import matplotlib
from core.foundation import atomic_json,atomic_text,file_sha256
from core.paired_atlas import epoch_name,PRIORITY,_write_csv

CSS='''*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;background:#09131d;color:#e9f0f4;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:#84ded6;text-decoration:none}a:hover{text-decoration:underline}header{border-bottom:1px solid #273c49;background:#0b1822}header .bar{max-width:1520px;margin:auto;padding:19px 30px;display:flex;align-items:center;gap:28px;flex-wrap:wrap}.wordmark{font:24px Georgia,serif;color:#f1e8d5}.wordmark b{color:#75dbcd;font-weight:400}nav{display:flex;gap:22px;flex-wrap:wrap}nav a{font-size:13px;color:#b9cdd8}main{max-width:1450px;margin:auto;padding:40px 30px 70px}.eyebrow{font-size:11px;letter-spacing:2.5px;color:#e6bd7d;text-transform:uppercase}h1{font:clamp(38px,5vw,70px)/1.1 Georgia,serif;letter-spacing:-1px;margin:15px 0 24px;max-width:1000px}h2{font:30px Georgia,serif;margin:28px 0 18px}h3{font-size:17px;line-height:1.4;margin:8px 0 12px}p{color:#aec2cd;max-width:1000px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:21px}.card{min-width:0;background:#10212d;border:1px solid #2c4350;border-radius:12px;overflow:hidden;padding:20px}.card img{display:block;width:100%;height:auto;max-height:330px;object-fit:contain;background:#08131f;border-radius:6px}.card p{font-size:13px;margin-bottom:5px}.hero{display:grid;grid-template-columns:1fr 1.1fr;gap:40px;align-items:center;margin-bottom:35px}.hero img{width:100%;border-radius:12px;border:1px solid #36505b}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:26px 0}.metric{border-left:2px solid #567986;padding:5px 15px;background:#0e1e29}.metric strong{display:block;font:28px Georgia,serif;color:#e9eddc}.metric span{font-size:12px;color:#96adb9}.sectiontop{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;margin-top:30px}input,select,button{font:inherit;background:#142c3a;border:1px solid #46606f;color:#e4edf2;border-radius:7px;padding:10px 13px}input{min-width:240px;max-width:100%}button{cursor:pointer}.tag{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:#7cdbce}.warm{color:#f2bb78}.links{display:flex;gap:15px;flex-wrap:wrap;margin:12px 0 4px;font-size:12px}.notice{border:1px solid #77603b;background:#292417;padding:14px 19px;color:#edc888;border-radius:9px;font-size:13px}.muted{color:#8da5b2;font-size:12px}video{display:block;width:100%;max-height:520px;border-radius:7px;background:#03090f}.tablewrap{overflow:auto;border:1px solid #2e4755;border-radius:9px}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:11px 16px;border-bottom:1px solid #2b414e}th{color:#91dcd4;background:#142b38}tbody tr:hover{background:#182d39}summary{cursor:pointer;color:#b9dbda}footer{border-top:1px solid #293f4d;padding:22px 30px;color:#7f99a7;font-size:12px}[hidden]{display:none!important}@media(max-width:850px){.hero{grid-template-columns:1fr}main{padding:28px 18px}header .bar{padding:16px 18px}nav{gap:14px}h1{font-size:43px}.grid{grid-template-columns:1fr}}'''


def rel(path,base):return os.path.relpath(Path(path),Path(base)).replace(os.sep,'/')


def link(source,target):
    source=Path(source).resolve();target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()==source:return
        target.unlink()
    if not target.exists():target.symlink_to(os.path.relpath(source,target.parent),target_is_directory=source.is_dir())


def page(path,title,body,root):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    nav=[('Overview',root/'START_HERE.html'),('Compare',root/'01_Comparisons/index.html'),('Reference fits',root/'01_Comparisons/Fitting_function_comparisons/index.html'),('EDE',root/'02_EDE/index.html'),('ΛCDM',root/'03_LCDM/index.html'),('Spatial explorers',root/'Spatial_explorers.html'),('Halo histories',root/'Halo_histories.html'),('Films',root/'04_Movies/index.html'),('Blender & 3D',root/'05_Blender_and_3D/index.html'),('Tables',root/'06_Tables_and_Data/index.html'),('Coverage',root/'07_Status_and_Provenance/index.html')]
    navigation=''.join(f'<a href="{rel(p,path.parent)}">{label}</a>' for label,p in nav)
    script="""<script>const q=document.querySelector('#search'),scope=document.querySelector('#epoch-scope');function filterCards(){const text=q?q.value.toLowerCase():'';document.querySelectorAll('[data-search]').forEach(c=>{const excluded=scope&&scope.value==='priority'&&c.dataset.priority==='false'&&!text;c.hidden=excluded||!c.dataset.search.toLowerCase().includes(text)})}if(q){q.value=new URLSearchParams(location.search).get('q')||'';q.addEventListener('input',filterCards);}if(scope)scope.addEventListener('change',filterCards);filterCards();document.querySelectorAll('video').forEach(v=>{v.addEventListener('error',()=>{let p=document.createElement('p');p.className='notice';p.textContent='This browser could not load the film. Use the MP4 download or open the atlas with its local-viewer launcher.';v.after(p)})});</script>"""
    atomic_text(path,'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style><header><div class="bar"><a class="wordmark" href="'+rel(root/'START_HERE.html',path.parent)+'">N128 <b>Observatory</b></a><nav>'+navigation+'</nav></div></header><main>'+body+'</main><footer>EDE / ΛCDM saved-particle analysis · comparison units and native figure axes are declared individually · source and selection records accompany each measurement.</footer>'+script+'</html>')


def human(name):
    value=re.sub(r'^(EDE_vs_LCDM|EDE|LCDM)_z_[0-9p]+_snap_\d+_','',str(name))
    return value.replace('_',' ')


def figure_card(p,base,caption=''):
    p=Path(p);title=human(p.stem);links=[]
    for suffix,label in (('.png','PNG'),('.pdf','PDF'),('.svg','SVG'),('.csv','Table'),('.npz','Arrays'),('.plotdata.npz','Plotted arrays'),('.figure.json','Source')):
        if p.with_suffix(suffix).exists():links.append(f'<a href="{rel(p.with_suffix(suffix),base)}">{label}</a>')
    return f'<article class="card" data-search="{html.escape(title,quote=True)}"><a href="{rel(p,base)}"><img loading="lazy" src="{rel(p,base)}" alt="{html.escape(title,quote=True)}"></a><h3>{html.escape(title)}</h3><p>{html.escape(caption)}</p><div class="links">'+''.join(links)+'</div></article>'


def _cloud_page(path,record,array,box,root):
    positions=np.round((array['sampled_position_comoving_mpc']/box-.5)*2,5).tolist();payload=json.dumps(positions,separators=(',',':'))
    z=record['summary']['redshift'];name=record['summary']['run'];body=f'<p class="eyebrow">{name} · z={z:.5f}</p><h1>Move through the saved particles.</h1><p>Drag to rotate. Scroll to zoom. {len(positions):,} deterministic sampled particles from the saved epoch; full box {box:.6f} comoving Mpc. Depth color is a display cue.</p><canvas id="cloud" style="width:100%;height:72vh;touch-action:none;background:#030a11;border:1px solid #2c4655;border-radius:12px"></canvas><script>const P='+payload+''';const c=document.querySelector('#cloud'),g=c.getContext('2d');let a=.6,b=.3,z=.8,down=false,x=0,y=0;function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;let w=c.width,h=c.height;g.fillStyle='#030a11';g.fillRect(0,0,w,h);for(const p of P){let X=Math.cos(a)*p[0]+Math.sin(a)*p[2],Z=-Math.sin(a)*p[0]+Math.cos(a)*p[2],Y=Math.cos(b)*p[1]-Math.sin(b)*Z;Z=Math.sin(b)*p[1]+Math.cos(b)*Z;let k=Math.min(w,h)*.37*z/(1+Z*.18);g.fillStyle=`hsla(${182+Z*20},75%,${62-Z*12}%,.5)`;g.fillRect(w/2+X*k,h/2-Y*k,1.4*devicePixelRatio,1.4*devicePixelRatio)}}c.onpointerdown=e=>{down=true;x=e.clientX;y=e.clientY;c.setPointerCapture(e.pointerId)};c.onpointerup=()=>down=false;c.onpointermove=e=>{if(down){a+=(e.clientX-x)*.006;b+=(e.clientY-y)*.006;x=e.clientX;y=e.clientY;draw()}};c.onwheel=e=>{e.preventDefault();z=Math.max(.2,Math.min(4,z*Math.exp(-e.deltaY*.001)));draw()};window.addEventListener('resize',draw);draw();</script>'''
    page(path,'3D particles · '+name,body,root)



def native_spatial_gallery(product, root):
    manifest = product / 'gallery/gallery.json'
    if not manifest.exists():
        return False
    record = json.loads(manifest.read_text())
    cards = []
    for item in record['figures']:
        png = manifest.parent / item['png']
        title = human(item['variant'])
        links = []
        for key, label in (('png', 'PNG'), ('pdf', 'PDF'), ('svg', 'SVG'),
                           ('physical_arrays', 'Arrays'), ('manifest', 'Source record')):
            if item.get(key) and (manifest.parent / item[key]).is_file():
                links.append(f'<a href="{rel(manifest.parent / item[key], product)}">{label}</a>')
        total = sum(count['supported'] for panel in item['selections'] for count in panel['counts'])
        support = f'{total:,} plotted values across panel series' if total else 'No values passed this selection'
        cards.append(f'<article class="card" data-search="{html.escape(title, quote=True)}">'
                     f'<a href="{rel(png, product)}"><img loading="lazy" src="{rel(png, product)}" alt="{html.escape(title, quote=True)}"></a>'
                     f'<h3>{html.escape(title)}</h3><p>{support}</p><div class="links">{"".join(links)}</div></article>')
    text = ('<p class="eyebrow">Native measured figures</p><h1>' + html.escape(human(product.name)) +
            '</h1><input id="search" aria-label="Filter figures" placeholder="Find a figure">'
            '<div class="grid" style="margin-top:22px">' + ''.join(cards) + '</div>')
    page(product/'index.html', human(product.name), text, root)
    return True


def build_model_explorers(root, home, model):
    folder = home / 'Spatial_explorers'
    folder.mkdir(exist_ok=True)
    cards = []
    for path in sorted(folder.glob('*/*/*')):
        if path.suffix not in ('.png', '.html') or path.name == 'index.html' or path.name.startswith('slice_'):
            continue
        title = human(path.parent.name)
        if path.name=='halo_anatomy.html' and (path.parent/'Readable_cutaway/index.html').exists():
            path=path.parent/'Readable_cutaway/index.html'
        epoch_parent=path.parent.parent.parent if path.parent.name=='Readable_cutaway' else path.parent.parent
        epoch = epoch_parent.name.split('_z_')[-1].replace('p', '.')
        if path.suffix == '.png':
            cards.append(figure_card(path, folder, f'{title} · z = {epoch}'))
        else:
            cards.append(f'<a class="card" href="{rel(path, folder)}"><h3>{html.escape(title)}</h3><p>Interactive · z = {epoch}</p></a>')
    for manifest in sorted(folder.glob('*/*/*/gallery/gallery.json')):
        product = manifest.parent.parent
        if native_spatial_gallery(product, root):
            epoch = product.parent.parent.name.replace('p', '.').replace('_z_', ' · z = ')
            title = 'Native halo structure' if product.name == 'Figures' else human(product.name)
            cards.append(f'<a class="card" href="{rel(product / "index.html", folder)}"><h3>{html.escape(title)}</h3><p>{html.escape(epoch)}</p></a>')
    for path in sorted(folder.glob('*/*void_particle_history.html')):
        cards.append(f'<a class="card" href="{rel(path, folder)}"><h3>Void particle history</h3><p>Exact saved particle identities through the tracked void.</p></a>')
    finder = home / 'Finder_comparison_M200c'
    if finder.exists():
        plots = ''.join(figure_card(p, finder, human(p.parent.name)) for p in sorted(finder.glob('*/*.png')))
        page(finder / 'index.html', model + ' · finder comparison',
             '<p class="eyebrow">Same M200c definition</p><h1>GADGET and Rockstar</h1><div class="grid">' + plots + '</div>', root)
        cards.append(f'<a class="card" href="{rel(finder / "index.html", folder)}"><h3>Compare the halo finders</h3><p>GADGET and Rockstar · M200c</p></a>')
    history = home / 'Halo_history_z_199_to_000'
    history.mkdir(exist_ok=True)
    workspace = root / 'Source/Native_workspaces' / (model + '_history')
    for bundle in workspace.glob('results/analysis_products_*'):
        for png in sorted(bundle.glob('*.png')):
            for asset in sorted(bundle.glob(png.stem + '.*')):
                link(asset, history / asset.name)
    history_cards = ''.join(figure_card(path, history) for path in sorted(history.glob('*.png')))
    page(history / 'index.html', model + ' · halo histories',
         '<p class="eyebrow">Resolved catalogue ancestry</p><h1>Halo histories</h1><div class="grid">' + history_cards + '</div>', root)
    page(folder / 'index.html', model + ' · spatial explorers',
         '<p class="eyebrow">Structure in three dimensions</p><h1>Spatial explorers</h1><div class="grid">' + ''.join(cards) + '</div>', root)


def build_epoch_navigation(root):
    root=Path(root).resolve();font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),17)
    model_rows={};epoch_records={};vmin,vmax=10.6,12.9
    for name,section in (('EDE','02_EDE'),('LCDM','03_LCDM')):
        manifest=root/'06_Tables_and_Data/Measurements'/name/'measurement_manifest.json'
        if not manifest.exists():continue
        run=json.loads(manifest.read_text());model_rows[name]=[];epoch_records[name]=run['epochs'];box=run['box_comoving_mpc'];home=root/section;native=root/'Source/Native_workspaces'/name/'results'
        native_bundles=sorted(native.glob('analysis_products_*/analysis_products_summary.json'));native_bundle=native_bundles[-1].parent if native_bundles else None
        for record in run['epochs']:
            s=record['summary'];snap=s['snapshot'];z=s['redshift'];label=epoch_name(name,snap,z);folder=home/'Saved_epochs'/label;folder.mkdir(parents=True,exist_ok=True)
            numeric=manifest.parent/record['numeric_asset']
            with np.load(numeric,allow_pickle=False) as arrays:
                for projection in ('xy','xz','yz'):
                    target=folder/(label+'_Dark_matter_'+projection.upper()+'.png')
                    if not target.exists():
                        field=gaussian_filter(arrays['dm_surface_density_msun_mpc2_'+projection],2.,mode='wrap');normal=np.clip((np.log10(np.maximum(field,10**vmin))-vmin)/(vmax-vmin),0,1)
                        rgb=matplotlib.colormaps['magma'](normal,bytes=True)[...,:3];im=Image.fromarray(np.flipud(rgb)).resize((768,768),Image.Resampling.LANCZOS)
                        frame=Image.new('RGB',(832,880),'#091722');frame.paste(im,(32,64));d=ImageDraw.Draw(frame);d.text((32,20),f'{name} · z={z:.5f} · snapshot {snap:03d} · {projection.upper()}',font=font,fill='#e7f2f8');d.text((32,794),f'Full box: {box:.5f} comoving Mpc · periodic smoothing',font=font,fill='#aac4d5');d.text((32,824),'Fixed log10 surface density range 10.6–12.9 [Msun/Mpc²]',font=font,fill='#aac4d5');frame.save(target)
                if not (folder/(label+'_Interactive_3D.html')).exists():_cloud_page(folder/(label+'_Interactive_3D.html'),record,arrays,box,root)
            priority=any(abs(z-target)<2e-5 for target in PRIORITY)
            if native_bundle and priority:
                for source in native_bundle.rglob('*.png'):
                    if not re.search(r'_%03d(?:\.|_)'%snap,source.name):continue
                    topic='Halo_abundance' if any(x in source.name.lower() for x in ('hmf','finder')) else 'Power_and_clustering' if any(x in source.name.lower() for x in ('pk','bias','correlation')) else 'Halo_structure_and_dynamics'
                    for suffix in ('.png','.pdf','.svg','.plotdata.npz','.figure.json'):
                        src=source.with_suffix(suffix)
                        if src.exists():link(src,folder/topic/(label+'_'+src.name))
            tables = folder / 'Tables'
            if native_bundle and priority:
                for source in native_bundle.rglob('*.csv'):
                    if re.search(r'_%03d(?:\.|_)' % snap, source.name):
                        link(source, tables / (label + '_' + source.name))
            body=f'<p class="eyebrow">{name} · saved epoch {snap:03d}</p><h1>z = {z:.5f}</h1><p>{s["fof_hosts_32"]:,} FOF hosts at ≥32 particles; {s["fof_hosts_300"]:,} at ≥300. The stored particle mass is {s["particle_mass_mean_msun"]:.5g} Msun. These selections are retained explicitly.</p><div class="links"><a href="{label}_Interactive_3D.html">Explore in 3D</a><a href="{rel(numeric,folder)}">Measured arrays</a><a href="{rel(numeric.with_suffix(".json"),folder)}">Source record</a></div><h2>Three views of the same saved box</h2><div class="grid">'
            for p in folder.glob('*.png'):body+=figure_card(p,folder)
            body+='</div>'
            if priority:
                fitpage=home/'Fitting_function_comparisons'/label/'index.html'
                body+=f'<div class="links"><a href="{rel(fitpage,folder)}">Reference fitting functions: overlays, residuals and diagnostics</a></div>'
            if priority:
                for topic in ('Halo_abundance','Power_and_clustering','Halo_structure_and_dynamics'):
                    figures=list((folder/topic).glob('*.png'))
                    if figures:body+='<h2>'+topic.replace('_',' ')+'</h2><div class="grid">'+''.join(figure_card(p,folder) for p in sorted(figures))+'</div>'
            if tables.is_dir():
                body += '<h2>Native numerical tables</h2><div class="links">'
                body += ''.join(f'<a href="{rel(p, folder)}">{html.escape(human(p.stem))}</a>' for p in sorted(tables.glob('*.csv')))
                body += '</div>'
            page(folder/'index.html',name+f' · z={z:.5f}',body,root)
            model_rows[name].append({**s,'page':str((folder/'index.html').relative_to(root)),'priority':priority})
        rows=model_rows[name];cards=''
        for r in rows:
            folder=(root/r['page']).parent;label=epoch_name(name,r['snapshot'],r['redshift']);tag='Priority epoch' if r['priority'] else 'Saved epoch'
            cards+=f'<a class="card" data-priority="{str(r["priority"]).lower()}" data-search="z {r["redshift"]:.3f} snapshot {r["snapshot"]} {tag}" href="{rel(folder/"index.html",home)}"><img loading="lazy" src="{rel(folder/(label+"_Dark_matter_XY.png"),home)}" alt="{name} density at z {r["redshift"]:.3f}"><p class="tag">{tag}</p><h3>z = {r["redshift"]:.3f} · snapshot {r["snapshot"]:03d}</h3><p>{r["fof_hosts_300"]:,} hosts at ≥300 particles</p></a>'
        body=f'<p class="eyebrow">{name} · N128 · dark matter</p><h1>Every saved epoch.<br>Details where they matter.</h1><p>All {len(rows)} saved epochs remain accessible. Priority epochs include the full native figure suite; every epoch has three density views, an interactive particle view, and source arrays.</p><div class="sectiontop"><h2>Browse the timeline</h2><select id="epoch-scope" aria-label="Epoch selection"><option value="priority">8 priority epochs</option><option value="all">All 86 saved epochs</option></select><input id="search" aria-label="Find an epoch" placeholder="Search every redshift or snapshot"></div><div class="grid">'+cards+'</div>'
        build_model_explorers(root, home, name)
        intro, timeline = body.split('<div class="sectiontop">', 1)
        navigation = '<div class="grid" style="margin:24px 0"><a class="card" href="Spatial_explorers/index.html"><h3>Spatial explorers →</h3><p>Halo cutaways, density maps, velocity fields and cosmic structure.</p></a><a class="card" href="Halo_history_z_199_to_000/index.html"><h3>Halo histories →</h3><p>Follow halo growth, assembly and catalogue properties through time.</p></a><a class="card" href="Fitting_function_comparisons/index.html"><h3>Reference fitting functions →</h3><p>All eight priority epochs, overlays and diagnostic panels.</p></a></div>'
        page(home/'index.html',name+' · all epochs',intro + navigation + '<div class="sectiontop">' + timeline,root)
    return model_rows


def build_tables(root,model_rows):
    root=Path(root);folder=root/'06_Tables_and_Data';rows=[]
    if all(n in model_rows for n in ('EDE','LCDM')):
        other={r['snapshot']:r for r in model_rows['LCDM']}
        for left in model_rows['EDE']:
            right=other[left['snapshot']]
            row={'snapshot':left['snapshot'],'redshift_EDE':left['redshift'],'redshift_LCDM':right['redshift']}
            for key in ('fof_hosts_32','fof_hosts_100','fof_hosts_300','fof_hosts_1000','largest_fof_mass_msun','subhalo_count','particle_speed_median_km_s','particle_speed_rms_km_s'):
                row[key+'_EDE']=left[key];row[key+'_LCDM']=right[key];row[key+'_EDE_over_LCDM']=left[key]/right[key] if right[key] else 'UNDEFINED_ZERO_DENOMINATOR'
            rows.append(row)
        _write_csv(folder/'EDE_vs_LCDM_All_86_epochs_comparison.csv',rows)
    header='<tr><th>Snapshot</th><th>Redshift</th><th>EDE hosts ≥300</th><th>ΛCDM hosts ≥300</th><th>EDE max mass [Msun]</th><th>ΛCDM max mass [Msun]</th></tr>'
    body='<p class="eyebrow">Numerical comparisons</p><h1>Tables you can inspect.<br>Measurements you can reuse.</h1><p>Physical masses have no h scaling. Counts retain explicit particle floors. Undefined ratios are written as undefined, never as a fabricated zero.</p><div class="links"><a href="EDE_vs_LCDM_All_86_epochs_comparison.csv">All-epoch comparison CSV</a><a href="Measurements/EDE/EDE_all_86_epochs_physical_units.csv">EDE epoch table</a><a href="Measurements/LCDM/LCDM_all_86_epochs_physical_units.csv">ΛCDM epoch table</a></div><h2>Host populations across the timeline</h2><div class="tablewrap"><table><thead>'+header+'</thead><tbody>'
    for r in rows:body+=f'<tr><td>{r["snapshot"]:03d}</td><td>{r["redshift_EDE"]:.4f}</td><td>{r["fof_hosts_300_EDE"]}</td><td>{r["fof_hosts_300_LCDM"]}</td><td>{r["largest_fof_mass_msun_EDE"]:.5g}</td><td>{r["largest_fof_mass_msun_LCDM"]:.5g}</td></tr>'
    body+='</tbody></table></div><h2>Per-figure numerical tables</h2><div class="grid">'
    for p in sorted((root/'01_Comparisons').rglob('*.csv')):body+=f'<a class="card" href="{rel(p,folder)}"><h3>{html.escape(human(p.stem))}</h3><p>{html.escape(p.parent.name)}</p><span class="tag">CSV</span></a>'
    page(folder/'index.html','Tables and numerical data',body+'</div>',root)
    return rows


def film_library(root):
    folder = root / '04_Movies'
    films = sorted(p for p in folder.rglob('*.mp4')
                   if not any(part.startswith('.') for part in p.relative_to(folder).parts)
                   and not p.name.endswith('.staged.mp4'))
    cards = []
    for path in films:
        model = path.relative_to(folder).parts[0]
        product = next(parent for parent in path.parents if parent.parent == folder / model)
        contract = product / 'film_source.json'
        record = json.loads(contract.read_text()) if contract.exists() else {}
        title = record.get('title', human(re.sub(r'^(EDE|LCDM)_z_[0-9p]+_', '', product.name)))
        if 'redshifts' in record:
            context = record['redshifts']
        elif '_z_199_to_000_' in product.name:
            context = 'Saved timeline · z = 199 to 0'
        else:
            z = float(product.name.split('_z_', 1)[-1].split('_', 1)[0].replace('p', '.'))
            context = f'Anchor redshift z = {z:g}'
        purpose = ''
        if 'void_explorer' in product.name:
            purpose = 'Approach the z = 0 dark-matter density minimum. The map marks the target, camera and route. One frozen snapshot; no time evolution.'
        elif 'Halo_approach' in product.name:
            purpose = 'Move toward the selected FOF halo in one frozen snapshot. The outlined ring and halo ID identify the target; the inset shows the camera and direction. This is a spatial approach, not a merger or time evolution.'
        elif 'Filament_tour' in product.name:
            purpose = 'Travel along a density-guided route through one frozen snapshot. The outlined ring marks the target FOF halo; the inset shows the camera and direction. This is a visual tour, not a measured particle trajectory.'
        src = html.escape(rel(path, folder), quote=True)
        story = product / 'index.html'
        explorer = product / 'void_explorer.html'
        extra = f'<a href="{rel(story, folder)}">Explore the interactive merger tree</a>' if story.exists() else ''
        if explorer.exists():
            extra += f'<a href="{rel(explorer, folder)}">Explore the void timeline</a>'
        poster = path.with_suffix('.poster.jpg')
        if not poster.exists():
            poster = product / 'Poster.png'
        preview = f'<img loading="lazy" src="{rel(poster, folder)}" alt="">' if poster.exists() else ''
        cards.append(f'<article class="card" data-search="{html.escape(model + " " + title + " " + context + " " + product.name, quote=True)}">'
                     f'{preview}<p class="tag">{model}</p><h3>{html.escape(title)}</h3><p>{html.escape(context)}</p><p>{html.escape(purpose)}</p>'
                     f'<div class="links"><button type="button" data-film="{src}" data-title="{html.escape(title, quote=True)}" data-purpose="{html.escape(purpose, quote=True)}">Load film</button>'
                     f'<a href="{src}" download>Download MP4</a>{extra}</div></article>')
    body = """
<p class="eyebrow">Saved history in motion</p><h1>The film library</h1>
<p>Choose a film to load it in the player. Search by model, redshift, or subject.</p>
<section class="card" id="screen" hidden><h2 id="film-title"></h2>
<video controls preload="metadata" playsinline aria-labelledby="film-title"></video>
<p id="film-description"></p><p role="status" id="film-status"></p></section>
<div class="sectiontop"><h2>Browse films</h2>
<input id="search" aria-label="Search films" placeholder="Model, redshift, or subject"></div>
<div class="grid" id="film-cards">""" + ''.join(cards) + """</div>
<script>
const screenPanel = document.querySelector('#screen');
const player = screenPanel.querySelector('video');
const filmStatus = document.querySelector('#film-status');
for (const button of document.querySelectorAll('[data-film]')) {
  button.addEventListener('click', () => {
    screenPanel.hidden = false;
    document.querySelector('#film-title').textContent = button.dataset.title;
    document.querySelector('#film-description').textContent = button.dataset.purpose || '';
    filmStatus.textContent = 'Loading film…';
    player.src = button.dataset.film;
    player.load();
    screenPanel.scrollIntoView({block: 'start'});
  });
}
player.addEventListener('loadedmetadata', () => {
  filmStatus.textContent = 'Ready · ' + player.duration.toFixed(1) + ' seconds';
});
player.addEventListener('error', () => {
  filmStatus.textContent = 'Playback failed. Open the atlas with the local launcher, or download the MP4.';
});
</script>"""
    page(folder / 'index.html', 'Film library', body, root)
    for model in ('EDE', 'LCDM'):
        model_folder = folder / model
        links = []
        for film in films:
            if not film.is_relative_to(model_folder):
                continue
            links.append(f'<a class="card" href="{rel(film, model_folder)}">{html.escape(human(film.parent.name))}</a>')
        page(model_folder / 'index.html', model + ' · films',
             f'<h1>{model} films</h1><p><a href="../index.html">Return to the film library and player</a></p>'
             '<div class="grid">' + ''.join(links) + '</div>', root)
    return films


def install_local_viewer(root):
    server = Path(__file__).resolve().parents[1] / 'scripts/serve_results.py'
    shutil.copy2(server, root / 'Source/serve_results.py')
    launcher = root / 'Open_atlas.command'
    atomic_text(launcher, '#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nexec python3 Source/serve_results.py . --port 0 --open\n')
    launcher.chmod(0o755)


def build_fitting_galleries(root):
    coverage_path=root/'07_Status_and_Provenance/highlighted_fitting_function_coverage.json'
    if not coverage_path.exists():return
    coverage=json.loads(coverage_path.read_text())
    gallery=root/'01_Comparisons/Fitting_function_comparisons'
    cards=[]
    for manifest_path in sorted((root/'01_Comparisons').glob('*/comparison_manifest.json'),key=lambda p:-json.loads(p.read_text())['redshift']):
        manifest=json.loads(manifest_path.read_text());snap=manifest['snapshot'];z=manifest['redshift']
        figure=next((manifest_path.parent/name for name in manifest['figures'] if name.endswith('_Fitting_function_comparison.png')),None)
        if figure is None:continue
        links=[]
        for model,section in (('EDE','02_EDE'),('LCDM','03_LCDM')):
            label=epoch_name(model,snap,z);folder=root/section/'Fitting_function_comparisons'/label
            record=coverage[model][str(snap)];table=root/record['table']
            if file_sha256(table)!=record['sha256']:raise ValueError(f'Changed fitting table: {table}')
            for source in sorted(table.parent.iterdir()):
                if source.is_file() and re.search(r'_%03d(?:\.|_)'%snap,source.name):link(source,folder/(label+'_'+source.name))
            figures=sorted(folder.glob('*.png'))
            note='<p class="notice">A 300-particle FOF selection is retained. Empty bins show counting upper limits; they do not establish agreement with a fitting function. CDM fits applied to EDE are reference extrapolations. Native panels retain their declared h-scaled axes.</p>'
            body=f'<p class="eyebrow">{model} · priority epoch {snap:03d}</p><h1>Reference fits · z = {z:g}</h1>'+note
            body+=f'<div class="links"><a href="{rel(root/section/"Saved_epochs"/label/"index.html",folder)}">Full epoch analysis</a><a href="{rel(manifest_path.parent/"index.html",folder)}">Paired comparison</a><a href="{rel(table,folder)}">Numerical fit table</a></div><div class="grid">'
            body+=''.join(figure_card(p,folder) for p in figures)+'</div>'
            page(folder/'index.html',f'{model} fitting functions · z={z:g}',body,root)
            links.append(f'<a href="{rel(folder/"index.html",gallery)}">{model}: all {len(figures)} diagnostic panels</a>')
        cards.append(f'<article class="card"><a href="{rel(figure,gallery)}"><img loading="lazy" src="{rel(figure,gallery)}" alt="Reference fitting functions at redshift {z:g}"></a><h3>z = {z:g}</h3><div class="links">'+''.join(links)+f'<a href="{rel(manifest_path.parent/"index.html",gallery)}">Paired figures and tables</a></div></article>')
    page(gallery/'index.html','Reference fitting functions · eight priority epochs','<p class="eyebrow">Abundance · residuals · ratios · calibration coverage</p><h1>Reference fitting functions.<br>Eight highlighted epochs.</h1><p>z = 14.4, 12, 10, 8, 6, 3, 1 and 0. Compare six reference prescriptions with measured FOF counts in each run. Paired overlays use physical units; model diagnostics retain native units.</p><p class="notice">High-redshift empty selections are shown as upper limits. Finite box size, particle selection and calibration limits remain explicit; counting intervals exclude box sample variance.</p><div class="grid">'+''.join(cards)+'</div>',root)
    for model,section in (('EDE','02_EDE'),('LCDM','03_LCDM')):
        home=root/section/'Fitting_function_comparisons'
        entries=sorted(home.glob('*/index.html'),reverse=True)
        body=f'<p class="eyebrow">{model} · reference diagnostics</p><h1>Fitting functions by epoch</h1><div class="grid">'
        for entry in entries:
            body+=f'<a class="card" href="{rel(entry,home)}"><h3>{html.escape(human(entry.parent.name))}</h3></a>'
        page(home/'index.html',model+' reference fitting functions',body+'</div>',root)


def build_topic_hubs(root):
    for filename,title,subfolder,description in (
        ('Spatial_explorers.html','Spatial explorers','Spatial_explorers','Halo cutaways, dark-matter maps, peculiar velocity fields and cosmic structure. Choose a model to open its full spatial collection.'),
        ('Halo_histories.html','Halo histories','Halo_history_z_199_to_000','Follow halo growth, assembly and catalogue properties through saved cosmic time. Choose a model to open its history figures.')):
        body=f'<p class="eyebrow">Direct access · both cosmological models</p><h1>{title}</h1><p>{description}</p><div class="grid">'
        for model,section in (('EDE','02_EDE'),('ΛCDM','03_LCDM')):
            target=root/section/subfolder/'index.html'
            body+=f'<a class="card" href="{rel(target,root)}"><p class="tag">{model}</p><h2>{model} {title.lower()} →</h2><p>Open the complete collection</p></a>'
        page(root/filename,title,body+'</div>',root)


def refresh_site(base):
    root=Path(base).resolve();build_fitting_galleries(root);models=build_epoch_navigation(root);build_topic_hubs(root);rows=build_tables(root,models)
    folder=root/'01_Comparisons';cards='';comparisons=[]
    for m in sorted(folder.glob('*/comparison_manifest.json'),key=lambda p:-json.loads(p.read_text())['redshift']):
        j=json.loads(m.read_text());comparisons.append(j);directory=m.parent;body=f'<p class="eyebrow">EDE / ΛCDM · common physical frame</p><h1>z = {j["redshift"]:.3f}</h1><p>Identical comoving volume: {j["physical_box_mpc"]:.6f} Mpc per side. The shared 300-particle physical mass floor is {j["common_300_particle_mass_floor_msun"]:.5g} Msun.</p><input id="search" aria-label="Find a comparison figure" placeholder="Find abundance, power, density…"><div class="grid" style="margin-top:22px">'
        if abs(j['redshift'] - 14.4) < .001:
            counts = next(row for row in rows if row['snapshot'] == j['snapshot'])
            message = (f'At z = 14.4, {counts["fof_hosts_32_EDE"]} EDE and {counts["fof_hosts_32_LCDM"]} ΛCDM FOF hosts pass 32 particles. '
                       f'At 300 particles, {counts["fof_hosts_300_EDE"]} EDE and {counts["fof_hosts_300_LCDM"]} ΛCDM hosts pass. '
                       'The lower-count sample illustrates dark-matter structure at the JWST-motivated epoch.')
            story_links = []
            for model in ('EDE', 'LCDM'):
                story = root / '04_Movies' / model / f'{model}_z_014p400_exploratory_FOF_ancestry/index.html'
                if story.exists():
                    story_links.append(f'<a href="{rel(story, directory)}">{model} exploratory halo ancestry</a>')
            body = body.replace('<input id="search"', f'<p class="notice">{message}</p><div class="links">{"".join(story_links)}</div><input id="search"')
        fit_summary = next(directory.glob('*_FOF_reference_fit_review_summary.csv'), None)
        if fit_summary:
            with fit_summary.open() as stream:
                diagnostics = [row for row in csv.DictReader(stream) if row['fit'] == 'press_schechter']
            deviations = ' and '.join(f"{float(row['median_abs_fractional_difference']):.1%} in {row['model']}" for row in diagnostics)
            note = ('Press–Schechter median absolute bin deviation is ' + deviations +
                    '. Each run contributes seven bins with at least five hosts and valid fit support.')
            body = body.replace('<input id="search"', f'<p class="notice">{note} <a href="{fit_summary.name}">Read the fit table</a>.</p><input id="search"')
        body+=''.join(figure_card(directory/name,directory) for name in j['figures'])+'</div>'
        page(directory/'index.html',f'EDE / ΛCDM · z={j["redshift"]:.3f}',body,root)
        hero=next((directory/p for p in j['figures'] if 'Density_XY' in p),None)
        cards+=f'<a class="card" href="{rel(directory/"index.html",folder)}"><img loading="lazy" src="{rel(hero,folder)}" alt="EDE and LCDM at z {j["redshift"]:.3f}"><h3>z = {j["redshift"]:.3f}</h3><p>{len(j["figures"])} paired figures · {len(j["tables"])} numerical tables</p></a>'
    context_dir = folder / 'EDE_vs_LCDM_z_0199_to_0000_cosmology_and_evolution'
    context_manifest = context_dir / 'context_manifest.json'
    context_link = ''
    if context_manifest.exists():
        context = json.loads(context_manifest.read_text())
        page(context_dir / 'index.html', 'Cosmology and saved evolution',
             '<p class="eyebrow">z = 199 to 0 · retained solver and simulation outputs</p>'
             '<h1>Cosmology and saved evolution</h1><div class="grid">' +
             ''.join(figure_card(context_dir / p, context_dir) for p in context['figures']) +
             '</div><h2>Numerical tables</h2><ul>' +
             ''.join(f'<li><a href="{p}">{html.escape(human(Path(p).stem))}</a></li>' for p in context['tables']) + '</ul>', root)
        context_link = f'<div class="links"><a href="{rel(context_dir / "index.html", folder)}">Linear input, expansion and all-epoch trends</a></div>'
    page(folder/'index.html','Compare EDE and LCDM','<p class="eyebrow">Common Mpc · common solar masses</p><h1>Compare the full<br>cosmological models.</h1><p>Same physical volume and matched saved epochs. This comparison combines the cosmological parameter differences. Ratios describe these two realizations.</p>'+context_link+'<div class="grid">'+cards+'</div>',root)
    films = film_library(root)
    install_local_viewer(root)
    blender_dir = root / '05_Blender_and_3D'
    blends = sorted(p for p in blender_dir.rglob('*.blend')
                    if not any(part.startswith('.') for part in p.relative_to(root).parts))
    preferred_blends = []
    for path in blends:
        model = path.relative_to(blender_dir).parts[0]
        preferred = path.parent / f'{model}_Explore_z_014p400_with_redshift_timeline.blend'
        if preferred.exists() and path != preferred:
            continue
        preferred_blends.append(path)
    blends = preferred_blends
    blenderbody = ('<p class="eyebrow">Interactive scenes</p><h1>Open a universe in Blender.</h1>'
                   '<p>Use the local launcher in each scene folder to keep the .blend and its source assets together. '
                   'The framed entry scene opens at z = 14.4. Timeline markers show every saved redshift. The DarkMatterSim sidebar displays the selected halo and a focus control. '
                   'Each model has its current rich scene and a framed entry view.</p><div class="grid">')
    for p in blends:
        retained = 'retained' in p.parent.name
        model = p.relative_to(blender_dir).parts[0]
        preview = p.parent / 'preview_z14p4.png'
        if not preview.exists():
            preview = p.parent / 'blender_preview.png'
        thumb = f'<img loading="lazy" src="{rel(preview, blender_dir)}" alt="{model} Blender scene preview">' if preview.exists() else ''
        launcher = p.parent / 'Open_in_Blender.command'
        launcher_link = f'<a href="{rel(launcher, blender_dir)}">Local launcher</a>' if launcher.exists() else ''
        kind = 'retained' if retained else 'rebuilt'
        provenance = root / '07_Status_and_Provenance' / f'{model}_{kind}_Blender_load.json'
        if '_with_redshift_timeline' in p.stem:
            provenance = root / '07_Status_and_Provenance' / f'{model}_framed_Blender_load.json'
        report_link = f'<a href="{rel(provenance, blender_dir)}">Fresh-load check</a>' if provenance.exists() else ''
        blenderbody += (f'<article class="card">{thumb}<p class="tag">{model}</p>'
                        f'<h3>{"Retained original scene" if retained else "Rebuilt scene"} · z = 199 to 0</h3>'
                        f'<div class="links"><a href="{rel(p, blender_dir)}">Blender file</a>{launcher_link}{report_link}</div></article>')
    for xdmf in sorted(blender_dir.glob('*/*ParaView_XDMF/timeline_type1.xdmf')):
        model = xdmf.relative_to(blender_dir).parts[0]
        report = xdmf.parent / 'xdmf_contract.json'
        blenderbody += (f'<article class="card"><p class="tag">{model} · ParaView</p>'
                        '<h3>Native 3D timeline · z = 199 to 0</h3>'
                        '<p>Open the local XDMF alongside its HDF5 inputs to inspect all saved dark-matter particles.</p>'
                        f'<div class="links"><a href="{rel(xdmf, blender_dir)}">XDMF timeline</a>'
                        f'<a href="{rel(report, blender_dir)}">Source and units</a></div></article>')
    page(blender_dir / 'index.html','Blender and 3D',blenderbody+'</div>',root)
    req=json.loads((root/'07_Status_and_Provenance/requirements.json').read_text());statusbody='<p class="eyebrow">Coverage and verification</p><h1>Every product has a status.</h1><p>The collection records every native workflow product and registered graph variant. Source checks, browser playback, scene loading and physical support are listed separately. Conditional fits and unresolved measurements retain their masks.</p><div class="tablewrap"><table><thead><tr><th>Requirement</th><th>Status</th></tr></thead><tbody>'
    for r in req['requirements']:statusbody+='<tr><td>'+html.escape(r['description'])+'</td><td>'+html.escape(r['status'])+'</td></tr>'
    statusbody+='</tbody></table></div><h2>Source records</h2><div class="grid">'
    for p in sorted((root/'07_Status_and_Provenance').glob('*')):
        if p.suffix not in ('.json', '.csv'):continue
        statusbody+=f'<a class="card" href="{p.name}">{html.escape(p.stem.replace("_"," "))}</a>'
    page(root/'07_Status_and_Provenance/index.html','Coverage and status',statusbody+'</div>',root)
    phase_path=root/'07_Status_and_Provenance/physical_ic_agreement.json';phase=json.loads(phase_path.read_text()) if phase_path.exists() else {}
    hero=next((root/'01_Comparisons'/epoch_name('EDE_vs_LCDM',j['snapshot'],j['redshift'])/f for j in comparisons if abs(j['redshift']-14.4)<.001 for f in j['figures'] if 'Density_XY' in f),None)
    body='<p class="eyebrow">N128 · EDE versus ΛCDM · research collection</p><div class="hero"><section><h1>Two universes.<br>One common frame.</h1><p>A complete comparison workspace for the saved dark-matter runs: physical-unit measurements, priority-epoch figures, all saved outputs, movies and interactive scenes.</p><div class="notice">All 86 saved epochs per model are available. Start at z = 14.4 for early hosts or z = 0 for the reference-fit checks. The coverage page records measured support and loading checks.</div></section>'
    if hero:body+=f'<a href="01_Comparisons/index.html"><img src="{rel(hero,root)}" alt="EDE and LCDM density comparison at z 14.4"></a>'
    body+='</div><div class="metrics"><div class="metric"><strong>7.37354 Mpc</strong><span>Common physical comoving box</span></div><div class="metric"><strong>'+str(sum(len(v) for v in models.values()))+'</strong><span>Saved epochs measured across both runs</span></div><div class="metric"><strong>'+str(len(comparisons))+' / 8</strong><span>Priority comparison epochs generated</span></div><div class="metric"><strong>'+('%.6f'%phase['complex_phase_coherence'] if phase else 'Pending')+'</strong><span>Measured initial phase coherence</span></div></div><h2>Choose how to explore</h2><div class="grid">'
    for title,text,path in [('Spatial explorers','Go straight to halo cutaways, velocity fields and cosmic structure in either model.','Spatial_explorers.html'),('Halo histories','Go straight to halo growth, assembly and catalogue histories in either model.','Halo_histories.html'),('Reference fitting functions','Six reference fits at all eight highlighted epochs, with residuals, ratios and counting limits.','01_Comparisons/Fitting_function_comparisons/index.html'),('Animated velocity flow','EDE, LCDM and paired views at z = 14.4 and z = 0.','04_Movies/index.html?q=flow'),('Compare the models','Shared units, common mass cuts, overlays and ratios.','01_Comparisons/index.html'),('Browse EDE','All 86 saved epochs, key-epoch science and particle views.','02_EDE/index.html'),('Browse ΛCDM','The matching LCDM timeline and native figure suite.','03_LCDM/index.html'),('Watch the films','Source-labelled visual histories and spatial tours.','04_Movies/index.html'),('Open Blender','Original interactive scenes and their source assets.','05_Blender_and_3D/index.html'),('Inspect the tables','Exact counts, masses, velocities, errors and comparison arrays.','06_Tables_and_Data/index.html')]:body+=f'<a class="card" href="{path}"><h3>{title}</h3><p>{text}</p></a>'
    body+='</div><h2>Start at the important epochs</h2><div class="grid">'+cards.replace('href="EDE_vs_LCDM','href="01_Comparisons/EDE_vs_LCDM').replace('src="EDE_vs_LCDM','src="01_Comparisons/EDE_vs_LCDM')+'</div>'
    page(root/'START_HERE.html','N128 Observatory · EDE and LCDM',body,root)
    atomic_json(root/'catalog.json',{'schema':'paired-atlas-navigation-v1','state':req['status'],'model_epochs':models,'comparisons':comparisons,'movie_files':[str(p.relative_to(root)) for p in films],'blender_files':[str(p.relative_to(root)) for p in blends]})
    (root/'README.md').write_text('# N128 EDE / ΛCDM Observatory\n\nOpen START_HERE.html, or use Open_atlas.command for a local viewer with video seeking. The coverage records identify each measured product and the inputs required for unavailable physics. Movies use H.264 and the local viewer supports byte-range seeking. For distribution, see [SHARING.md](SHARING.md); keep the whole folder together.\n\n01_Comparisons: common-unit figures and per-figure tables.\n02_EDE and 03_LCDM: all saved epochs with descriptive model/redshift/snapshot filenames.\n04_Movies: films and source contracts.\n05_Blender_and_3D: original interactive scene exports.\n06_Tables_and_Data: physical-unit caches and CSV tables.\n07_Status_and_Provenance: requirement coverage, live jobs and source checks.\nSource: current native publication data, rebuild code and verification records. Raw simulation inputs are external; see Source/RAW_INPUTS.md.\n\nThe native runs differ in h, box size in Mpc/h, particle mass and cosmological parameters. The physical comoving boxes match. Matching initial phases and shared lattice IDs were measured explicitly; matching seeds alone were not assumed sufficient.\n')
    print('SITE_REFRESHED',root/'START_HERE.html',flush=True)
