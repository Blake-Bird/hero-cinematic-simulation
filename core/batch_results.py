"""Human-readable navigation for sequential analysis and comparison outputs."""
from __future__ import annotations
import html
import csv
import io
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_batch_index(directory,report):
    """Index only paths recorded by the batch; do not fabricate absent products."""
    from core.foundation import atomic_text
    root=Path(directory).expanduser().resolve()
    def esc(value):return html.escape(str(value),quote=True)
    def link(path,label):
        p=Path(path).expanduser().resolve()
        if not p.exists():return '<span>Unavailable: '+esc(label)+'</span>'
        url=quote(os.path.relpath(p,root),safe='/')
        return '<a href="'+esc(url)+'">'+esc(label)+'</a>'
    runs=[];overview=[]
    for row in report['runs']:
        run=Path(row['run']);artifacts=row.get('artifacts',{});items=[]
        if (run/'results/index.html').is_file():items.append(link(run/'results/index.html','Run results homepage'))
        if row.get('workflow_record'):items.append(link(row['workflow_record'],'Analysis workflow record'))
        analysis=artifacts.get('analysis');measured_redshifts=[]
        if isinstance(analysis,str):
            folder=Path(analysis);items.append(link(folder/'snapshot_epochs.html','Snapshot redshifts and times'))
            summary=folder/'analysis_products_summary.json'
            if summary.is_file():
                data=json.loads(summary.read_text())
                measured_redshifts=[item['redshift'] for item in data.get('snapshots',{}).values() if 'redshift' in item]
                for name in data.get('artifacts',[]):
                    if isinstance(name,str) and Path(name).name==name and name.lower().endswith(('.png','.html','.mp4')):
                        items.append(link(folder/name,name))
        for name,value in artifacts.items():
            if name=='analysis':continue
            if isinstance(value,str):items.append(link(value,name))
            elif isinstance(value,list):
                for item in value:
                    if isinstance(item,str):items.append(link(item,name+' · '+Path(item).name))
        issues=row.get('unavailable',{})
        if row.get('error'):issues={**issues,'failure':row['error'].get('message','unknown')}
        warning=''.join('<li>'+esc(k)+': '+esc(v)+'</li>' for k,v in issues.items())
        settings=row.get('settings',{})
        def value(name):
            item=settings.get(name)
            return '—' if item is None else str(item)
        def epochs(values):
            return ', '.join(format(float(z),'.6g') for z in values) if values else '—'
        overview.append([run.name,row['status'],value('model'),value('physics'),value('H0'),
                         value('box_mpc_h'),value('particles_per_dimension'),
                         epochs(settings.get('requested_redshifts',[])),epochs(measured_redshifts)])
        runs.append('<section><h2>'+esc(run.name)+'</h2><p>Status: '+esc(row['status'])+'</p><ul>'+''.join('<li>'+x+'</li>' for x in items)+'</ul>'+(('<details><summary>Unavailable or failed items</summary><ul>'+warning+'</ul></details>') if warning else '')+'</section>')
    comparisons=[]
    for row in report.get('comparisons',[]):
        entries=[]
        for name,product in row.get('products',{}).items():
            if product.get('figure'):entries.append('<li>'+link(product['figure'],name+' comparison')+'</li>')
            elif product.get('map_bundle'):
                bundle=Path(product['map_bundle']);summary=bundle/'comparison_maps_summary.json'
                entries.append('<li>'+link(summary,'Comparison map summary')+'</li>')
                if summary.is_file():
                    data=json.loads(summary.read_text())
                    for result in data.get('maps',{}).values():
                        for image in result.get('figure_files',[]):
                            if isinstance(image,str) and Path(image).name==image:entries.append('<li>'+link(bundle/image,'Map · '+image)+'</li>')
            else:entries.append('<li>'+esc(name)+': '+esc(product.get('reason','unavailable'))+'</li>')
        if row.get('reason'):entries.append('<li>'+esc(row['reason'])+'</li>')
        comparisons.append('<section><h2>'+esc(Path(row['reference_run']).name)+' vs '+esc(Path(row['other_run']).name)+'</h2><p>Status: '+esc(row['status'])+'</p><ul>'+''.join(entries)+'</ul></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Batch simulation results</title><style>body{background:#081321;color:#eef8ff;font:16px/1.5 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:32px}p{color:#bcd0e1}a{color:#8de9ef}section{background:#14263a;border:1px solid #2d465d;border-radius:12px;padding:18px 24px;margin:16px 0}li{margin:5px 0}details{color:#f6ca92}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,420px),1fr));gap:16px}</style><main><h1>Batch simulation results</h1><p>Individual run outputs and available comparisons. Differences do not by themselves establish matched initial phases or causal effects.</p><p>Batch status: %s · <a href="batch_report.json">Detailed report</a></p><h2>Individual runs</h2><div class="grid">%s</div><h2>Comparisons</h2><div class="grid">%s</div></main></html>'''%(esc(report['status']),''.join(runs),''.join(comparisons) or '<p>No comparisons requested or available.</p>')
    headings=['Run','Status','Model','Physics','H₀ [km/s/Mpc]','Box [Mpc/h]',
              'Particles per dimension','Requested output z','Analyzed snapshot z']
    stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(headings);writer.writerows(overview)
    atomic_text(root/'run_overview.csv',stream.getvalue())
    table='<h2>Run overview</h2><p>Configured settings and redshifts read from each analysis summary. Requested epochs may differ from saved snapshots. <a href="run_overview.csv">Download CSV</a></p><div style="overflow-x:auto"><table><thead><tr>'+''.join('<th scope="col">'+esc(x)+'</th>' for x in headings)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in overview)+'</tbody></table></div>'
    page=page.replace('</style>','table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #2d465d;vertical-align:top}th{color:#8de9ef;background:#14263a}tbody tr:nth-child(even){background:#102135}td:nth-last-child(-n+2){min-width:160px}</style>')
    page=page.replace('<h2>Individual runs</h2>',table+'<h2>Individual runs</h2>')
    suites=[]
    for name,item in report.get('suite_comparisons',{}).items():
        entries=[link(item[key],label) for key,label in (('figure','PNG figure'),('pdf','PDF figure'),('data_csv','Measured curves CSV'),('record','Source record')) if item.get(key)]
        if item.get('reason'):entries.append(esc(item['reason']))
        suites.append('<section><h3>'+esc(name.upper())+'</h3><p>Status: '+esc(item['status'])+'</p><p>'+' · '.join(entries)+'</p></section>')
    if suites:
        page=page.replace('<h2>Comparisons</h2>','<h2>All selected runs together</h2><div class="grid">'+''.join(suites)+'</div><h2>Comparisons</h2>')
    atomic_text(root/'index.html',page);return root/'index.html'
