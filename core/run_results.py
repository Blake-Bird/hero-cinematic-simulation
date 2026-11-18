"""Manifest-driven, relocatable navigation across immutable workflow attempts."""
from __future__ import annotations
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from core.epochs import epoch_label
from core.foundation import atomic_json, atomic_text, file_sha256

MEDIA = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.mp4', '.webm', '.blend', '.xdmf', '.h5', '.hdf5', '.csv', '.npz', '.json'}


def build_run_results(run_dir, *, verify=False):
    """Index only declared artifacts, retaining each workflow's independent scope.

    Older absolute paths are rebased only from the recorded source run. Missing
    files and hash mismatches are visible issues; no newest-directory guessing.
    """
    import yaml
    root = Path(run_dir).expanduser().resolve()
    config = yaml.safe_load((root/'config.yaml').read_text()) or {}
    results = root/'results'
    results.mkdir(exist_ok=True)
    highlights = config.get('simulation', {}).get('highlight_redshifts', [])
    attempts, issues, notes = [], [], []
    inputs, referenced, claimed_export_directories = [], set(), set()
    for record_path in sorted(results.glob('workflow_*Z.json')):
        try:
            record = json.loads(record_path.read_text())
            if not isinstance(record.get('artifacts'), dict) or 'run' not in record:
                continue
            directory = record['artifacts'].get('analysis')
            if isinstance(directory, str):
                original = Path(directory)
                prior = Path(record['run'])
                if original.is_absolute() and original.is_relative_to(prior):
                    original = root/original.relative_to(prior)
                referenced.add(original.resolve() if original.is_absolute() else (root/original).resolve())
            # A standalone movie manifest is historical only when no workflow
            # already claims an artifact in that immutable export directory.
            # Follow explicit recorded artifact values; do not infer a movie
            # from directory age or from an expected filename.
            pending_paths = list(record['artifacts'].values())
            while pending_paths:
                value = pending_paths.pop()
                if isinstance(value, dict):
                    pending_paths.extend(value.values())
                elif isinstance(value, list):
                    pending_paths.extend(value)
                elif isinstance(value, str):
                    claimed = Path(value)
                    if claimed.is_absolute() and claimed.is_relative_to(Path(record['run'])):
                        claimed = root/claimed.relative_to(Path(record['run']))
                    elif not claimed.is_absolute():
                        claimed = root/claimed
                    claimed = claimed.resolve()
                    if any(claimed.is_relative_to(results/section) for section in ('movies', 'planetarium')):
                        if claimed.is_file():
                            claimed_export_directories.add(claimed.parent)
                        elif claimed.is_dir():
                            claimed_export_directories.add(claimed)
            inputs.append((record_path, record))
        except (OSError, ValueError, TypeError) as error:
            issues.append({'path': record_path.name, 'reason': str(error)})
    # Older analysis summaries are explicit publication manifests. Index them as
    # separate historical attempts, without inventing a successful workflow.
    historical = []
    for summary_path in sorted(results.glob('analysis_products_*/analysis_products_summary.json')):
        if summary_path.parent.resolve() in referenced:
            continue
        try:
            summary = json.loads(summary_path.read_text())
            if 'artifacts' not in summary:
                continue
            historical.append((summary_path, {'run': str(root), 'status': 'HISTORICAL_ANALYSIS_ONLY',
                'products': summary.get('products', []), 'created_utc': summary.get('created_utc', ''),
                'artifacts': {'analysis': str(summary_path.parent)},
                'failure': {'message': 'No associated workflow record: only this published analysis generation is indexed.'}}))
        except (OSError, ValueError, TypeError) as error:
            issues.append({'path': str(summary_path.relative_to(results)), 'reason': str(error)})
    for pattern in ('movies/*/movie_contract.json', 'movies/*/halo_growth_contract.json', 'planetarium/*/timeline.json'):
        for manifest in sorted(results.glob(pattern)):
            if manifest.parent.resolve() in claimed_export_directories:
                continue
            historical.append((manifest, {'run': str(root), 'status': 'HISTORICAL_EXPORT_ONLY',
                'artifacts': {'historical_export': str(manifest)},
                'failure': {'message': 'Standalone recorded export; not attributed to the current analysis workflow.'}}))
    inputs = historical + inputs
    for record_path, record in inputs:
        try:
            old_root = Path(record['run'])
            identifier = record_path.stem if record_path.parent == results else record_path.parent.name
            attempt = {'id': identifier, 'status': record.get('status', 'UNKNOWN'),
                       'record': record_path.relative_to(results).as_posix(), 'items': [], 'epochs': [],
                       'unavailable': record.get('unavailable', {}), 'failure': record.get('failure'),
                       'requested': record.get('requested_products', record.get('products', [])),
                       'decisions': record.get('applicability', {}).get('decisions', {}),
                       'product_states': record.get('product_states', {}), 'created_utc': record.get('created_utc', ''), 'retry': record.get('retry_commands', [])}
            seen, read = set(), set()

            def resolve(name, base):
                path = Path(name)
                if path.is_absolute():
                    if path.is_relative_to(old_root):
                        path = root/path.relative_to(old_root)
                else:
                    path = base/path
                path = path.resolve()
                if not path.is_relative_to(root):
                    raise ValueError('Artifact outside this run: '+str(path))
                return path

            def register(path, metadata=None, expected=None):
                metadata = metadata or {}
                relative = path.relative_to(results).as_posix() if path.is_relative_to(results) else os.path.relpath(path, results)
                if not path.is_file():
                    issues.append({'attempt': attempt['id'], 'path': relative, 'reason': 'Missing recorded artifact'})
                    return
                if verify and expected and file_sha256(path) != expected:
                    issues.append({'attempt': attempt['id'], 'path': relative, 'reason': 'Artifact checksum mismatch'})
                    return
                if path in seen or path.suffix.lower() not in MEDIA:
                    return
                seen.add(path)
                snap = metadata.get('snapshot')
                z = metadata.get('redshift')
                if type(snap) is not int:
                    snap = None
                if not isinstance(z, (float, int)) or not math.isfinite(z):
                    z = None
                attempt['items'].append({'path': relative, 'title': metadata.get('title') or metadata.get('product') or path.name,
                    'snapshot': snap, 'redshift': z, 'topic': str(metadata.get('product') or 'Other recorded products'),
                    'kind': 'Movies' if path.suffix.lower() in ('.mp4', '.webm') else
                            'Figures' if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.svg', '.pdf') else 'Data',
                    'sha256': expected, 'source_record': metadata.get('source_record')})

            def visit(value, base, metadata=None, depth=0):
                if depth > 24:
                    issues.append({'attempt': attempt['id'], 'reason': 'Manifest nesting limit exceeded'})
                    return
                if isinstance(value, list):
                    for entry in value:
                        visit(entry, base, metadata, depth+1)
                elif isinstance(value, dict):
                    meta = {**(metadata or {}), **value.get('scientific_metadata', {})}
                    for key in ('snapshot', 'redshift', 'product', 'title'):
                        if key in value and isinstance(value[key], (str, int, float)):
                            meta[key] = value[key]
                    if isinstance(value.get('path'), str):
                        consume(value['path'], base, meta, value.get('sha256'), depth+1)
                    for key in ('artifacts', 'files', 'movie', 'frames', 'figures', 'preview', 'directory', 'report', 'manifest', 'scene', 'output_directory', 'numeric_measurements', 'records', 'variants', 'assets'):
                        if key in value:
                            visit(value[key], base, meta, depth+1)
                elif isinstance(value, str):
                    consume(value, base, metadata, None, depth+1)

            def consume(name, base, metadata, expected, depth):
                try:
                    path = resolve(name, base)
                    if path.is_dir():
                        # Follow known publication records, never every media file in a folder.
                        found = False
                        for filename in ('analysis_products_summary.json', 'report.json', 'timeline.json', 'manifest.json', 'gallery.json', 'map_contract.json', 'paper_figure_manifest.json'):
                            if (path/filename).is_file():
                                consume(str(path/filename), path, metadata, None, depth+1)
                                found = True
                        if not found:
                            issues.append({'attempt': attempt['id'], 'path': os.path.relpath(path, results),
                                           'reason': 'Directory has no supported artifact manifest; use workflow record'})
                    elif path.suffix.lower() == '.json':
                        if path in read:
                            return
                        read.add(path)
                        if not path.is_file():
                            raise ValueError('Missing artifact manifest: '+str(path))
                        supported = path.name.endswith('.figure.json') or path.name in {
                            'analysis_products_summary.json', 'paper_figure_manifest.json', 'map_contract.json',
                            'movie_contract.json', 'halo_growth_contract.json', 'timeline.json', 'manifest.json', 'gallery.json'}
                        if not supported:
                            register(path, metadata, expected)
                            return
                        if path.stat().st_size > 32*1024**2:
                            register(path, metadata, expected)
                            notes.append({'attempt': attempt['id'], 'path': os.path.relpath(path, results),
                                          'reason': 'Large metadata linked without parsing (32 MiB navigation intake bound)'})
                            return
                        if verify and expected and file_sha256(path) != expected:
                            raise ValueError('Manifest checksum mismatch: '+str(path))
                        data = json.loads(path.read_text())
                        if path.name == 'movie_contract.json' and 'movie' not in data:
                            register(path, metadata, expected)
                            notes.append({'attempt': attempt['id'], 'path': os.path.relpath(path, results), 'reason': 'Legacy movie contract has no recorded payload path; payload name is not inferred'})
                        if data.get('schema') == 'darkmattersim-figure-v1' or path.name.endswith('.figure.json'):
                            metadata = {**(metadata or {}), 'source_record': os.path.relpath(path, results)}
                        if path.name == 'analysis_products_summary.json':
                            for row in __import__('core.snapshot_tables', fromlist=['snapshot_rows']).snapshot_rows(data.get('snapshots', {})):
                                z = row['redshift']
                                requested = row['requested_redshift']
                                # Legacy summaries have no requested epoch: use the frozen schedule's storage index.
                                schedule = config.get('simulation', {}).get('snapshot_redshift_list', [])
                                if requested is None and row['snapshot'] < len(schedule):
                                    requested = schedule[row['snapshot']]
                                row['requested_redshift'] = requested
                                row['highlight'] = requested in highlights
                                row['label'] = epoch_label(row['snapshot'], z) if z is not None else f"snap_{row['snapshot']:03d}_z_unavailable"
                                attempt['epochs'].append(row)
                        if path.name == 'analysis_products_summary.json':
                            paper = path.parent/'paper_figures'/'paper_figure_manifest.json'
                            if paper.is_file():
                                consume(str(paper), path.parent, metadata, None, depth+1)
                        visit(data, path.parent, metadata, depth+1)
                    else:
                        if path.suffix.lower() in ('.png', '.svg', '.pdf'):
                            sidecar = path.with_suffix('.figure.json')
                            if sidecar.is_file() and sidecar not in read:
                                consume(str(sidecar), path.parent, metadata, None, depth+1)
                        register(path, metadata, expected)
                except (OSError, ValueError, TypeError) as error:
                    issues.append({'attempt': attempt['id'], 'path': str(name), 'reason': str(error)})

            for key, value in record['artifacts'].items():
                visit(value, root, {'product': key})
            grouped = {}
            for item in attempt['items']:
                key = str(Path(item['path']).with_suffix('')) if item['kind'] == 'Figures' else item['path']
                if key in grouped and item['kind'] == 'Figures':
                    old = grouped[key]
                    old.setdefault('formats', [old['path']]).append(item['path'])
                    if Path(item['path']).suffix.lower() == '.png':
                        old['path'] = item['path']
                else:
                    grouped[key] = item
            attempt['items'] = list(grouped.values())
            attempts.append(attempt)
        except (OSError, ValueError, TypeError) as error:
            issues.append({'path': record_path.name, 'reason': str(error)})
    report = {'schema': 'run-results-v1', 'run_name': root.name, 'current_attempt': attempts[-1]['id'] if attempts else None,
              'highlight_redshifts': highlights, 'attempts': attempts, 'issues': issues, 'notes': notes,
              'verification': 'RECORDED_HASHES_CHECKED' if verify else 'EXISTENCE_ONLY', 'science_acceptance': 'NOT_ACCEPTED'}
    atomic_json(results/'navigation.json', report)
    _write_pages(results, report)
    return results/'index.html'


def _write_pages(results, report):
    escape = html.escape
    def link(path, label, base=results):
        target = results/path
        return '<a href="%s">%s</a>' % (escape(quote(os.path.relpath(target, base), safe='/'), quote=True), escape(str(label)))
    def page(title, body):
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>body{max-width:1200px;margin:32px auto;padding:0 22px;background:#0b1523;color:#e7eff8;font:16px/1.6 system-ui}a{color:#8ce2ec}nav,section,article{padding:16px;border:1px solid #345;margin:16px 0;border-radius:10px}img{max-width:330px;max-height:230px}video{max-width:100%}select{font:inherit}code{white-space:pre-wrap}h2{color:#8ce2ec}[hidden]{display:none!important}</style>'+body+'</html>'
    def items(rows, base=results):
        chunks = []
        topics = sorted({r['topic'] for r in rows})
        for topic in topics:
            chunks.append('<h3>'+escape(topic)+'</h3>')
            for row in rows:
                if row['topic'] != topic:
                    continue
                preview = ''
                if Path(row['path']).suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    preview = '<br><img loading="lazy" alt="'+escape(row['title'], quote=True)+'" src="'+escape(quote(os.path.relpath(results/row['path'], base), safe='/'), quote=True)+'">'
                if row['kind'] == 'Movies':
                    preview = '<br><video controls preload="metadata" src="'+escape(quote(os.path.relpath(results/row['path'], base), safe='/'), quote=True)+'"></video>'
                alternatives = ' · '.join(link(path, Path(path).suffix.lstrip('.').upper(), base) for path in row.get('formats', []))
                contract = link(row['source_record'], 'Source record', base) if row.get('source_record') else ''
                chunks.append('<article>'+link(row['path'], row['title'], base)+preview+'<br>'+alternatives+' '+contract+'</article>')
        return ''.join(chunks) or '<p>No recorded outputs in this view.</p>'
    body = '<h1>'+escape(report['run_name'])+'</h1><p>Scientific outputs retain their own validity limits. '+escape(report['verification'])+'</p>'
    body += '<label>Analysis attempt <select id="attempt">'
    for attempt in reversed(report['attempts']):
        body += '<option value="'+escape(attempt['id'], quote=True)+'">'+escape(attempt['id']+' · '+attempt['status'])+'</option>'
    body += '</select></label><nav>'+' · '.join('<a href="#'+v.lower().replace(' ', '-')+'">'+v+'</a>' for v in ('Highlights', 'By redshift', 'Evolution', 'Movies', 'Data', 'Status'))+'</nav>'
    for attempt in report['attempts']:
        body += '<div class="attempt" data-attempt="'+escape(attempt['id'], quote=True)+'">'
        body += '<p>'+link(attempt['record'], 'Workflow record')+'</p>'
        epochs = attempt['epochs'];starred = {r['snapshot'] for r in epochs if r['highlight']}
        body += '<section data-view="highlights"><h2>Highlights ★</h2>'+items([r for r in attempt['items'] if r['snapshot'] in starred])+'</section>'
        body += '<section data-view="by-redshift"><h2>By redshift</h2>'
        for epoch in epochs:
            folder = results/'navigation'/attempt['id']/epoch['label']
            folder.mkdir(parents=True, exist_ok=True)
            selected = [r for r in attempt['items'] if r['snapshot'] == epoch['snapshot']]
            content = '<h1>'+escape(epoch['label'])+(' ★' if epoch['highlight'] else '')+'</h1><p>Actual z = '+escape(str(epoch['redshift']))+'; requested z = '+escape(str(epoch['requested_redshift']))+'</p>'
            content += link('index.html', 'All results', folder)+items(selected, folder)
            movies = [r for r in attempt['items'] if r['kind'] == 'Movies']
            content += '<h2>Run movies</h2><p>See each movie’s source record for its epoch coverage.</p>'+items(movies, folder)
            atomic_text(folder/'index.html', page(epoch['label'], content))
            body += link(os.path.relpath(folder/'index.html', results), ('★ ' if epoch['highlight'] else '')+epoch['label'])+'<br>'
        body += '</section>'
        body += '<section data-view="evolution"><h2>Evolution and unassigned figures</h2><p>Figures without an explicit single-snapshot contract stay here; their epoch is never inferred from a filename.</p>'+items([r for r in attempt['items'] if r['snapshot'] is None and r['kind'] == 'Figures'])+'</section>'
        for kind in ('Movies', 'Data'):
            body += '<section data-view="'+kind.lower()+'"><h2>'+kind+'</h2>'+items([r for r in attempt['items'] if r['kind'] == kind])+'</section>'
        body += '<section data-view="status"><h2>Status: '+escape(attempt['status'])+'</h2><pre>'+escape(json.dumps({k: attempt[k] for k in ('requested', 'product_states', 'decisions', 'unavailable', 'failure', 'retry')}, indent=2))+'</pre></section></div>'
    body += '<h2>Navigation issues</h2><pre>'+escape(json.dumps(report['issues'], indent=2))+'</pre><p>'+link('navigation.json', 'Navigation manifest')+'</p>'
    body += '<h2>Metadata notes</h2><pre>'+escape(json.dumps(report.get('notes', []), indent=2))+'</pre>'
    body += '''<script>const sel=document.querySelector('#attempt');function show(){for(const a of document.querySelectorAll('.attempt'))a.hidden=a.dataset.attempt!==sel.value;for(const s of document.querySelectorAll('[data-view]'))s.id=s.closest('.attempt').hidden?'':s.dataset.view}sel.addEventListener('change',show);show()</script>'''
    atomic_text(results/'index.html', page(report['run_name'], body))


def results_main():
    """Rebuild saved-run navigation without running analysis or simulation."""
    import argparse
    parser = argparse.ArgumentParser(description=results_main.__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--verify', action='store_true', help='Check recorded artifact hashes as well as existence')
    args = parser.parse_args()
    print(build_run_results(args.run_dir, verify=args.verify))
