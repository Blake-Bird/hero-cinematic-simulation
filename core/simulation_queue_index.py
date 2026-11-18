"""Readable index for an interactive sequential simulation queue."""
from __future__ import annotations
import html
import os
from pathlib import Path
from urllib.parse import quote


def write_queue_index(directory,report):
    from core.foundation import atomic_text
    root=Path(directory).resolve()
    def link(path,label):
        path=Path(path).resolve()
        if not path.exists():return html.escape(label)+' (not created)'
        return '<a href="%s">%s</a>'%(html.escape(quote(os.path.relpath(path,root),safe='/'),quote=True),html.escape(label))
    rows=[]
    for row in report['runs']:
        run=Path(row['run']);links=[link(row['linear_bundle'],'Linear spectrum'),link(row['answers'],'Saved inputs'),link(run/'results/index.html','Run results')]
        if row.get('workflow_record'):links.append(link(row['workflow_record'],'Analysis status'))
        if row.get('artifacts',{}).get('analysis'):
            links.append(link(row['artifacts']['analysis'],'Scientific figures'))
        error=row.get('error',{}).get('message') or row.get('analysis_reason')
        stages=''
        if row.get('simulation_status') or row.get('analysis_status'):
            stages='<p>Simulation: %s · Analysis: %s</p>'%(html.escape(row.get('simulation_status','UNKNOWN')),html.escape(row.get('analysis_status','UNKNOWN')))
        rows.append('<section><h2>%s</h2><p>Status: %s</p>%s<p>%s</p>%s</section>'%(html.escape(run.name),html.escape(row['status']),stages, ' · '.join(links),'<p class="issue">'+html.escape(str(error))+'</p>' if error else ''))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Simulation queue</title><style>body{background:#081321;color:#eef8ff;font:16px/1.5 system-ui;margin:0}main{max-width:1000px;margin:auto;padding:32px}section{background:#14263a;border:1px solid #2d465d;border-radius:12px;padding:18px 24px;margin:16px 0}a{color:#8de9ef}.issue{color:#ffcc92}</style><main><h1>Simulation queue</h1><p>Status: %s · <a href="queue_report.json">Detailed report</a></p>%s</main></html>'''%(html.escape(report['status']),''.join(rows))
    atomic_text(root/'index.html',page);return root/'index.html'
