"""Human-readable and machine-readable tables of recorded snapshot epochs."""
from __future__ import annotations

import csv
import html
import math
from pathlib import Path


COLUMNS = [
    ('snapshot', 'Snapshot', ''),
    ('redshift', 'Actual redshift', ''),
    ('scale_factor', 'Actual scale factor', ''),
    ('cosmic_time_gyr', 'Cosmic time', 'Gyr'),
    ('requested_redshift', 'Requested redshift', ''),
    ('redshift_offset', 'Actual − requested', ''),
    ('clock_status', 'Clock status', ''),
    ('products', 'Product status', ''),
]


def snapshot_rows(snapshots):
    """Return rows using recorded epochs; never replace missing values with zero."""
    rows = []
    for snapshot, record in snapshots.items():
        epoch = record.get('epoch_contract') or {}
        scale = epoch.get('actual_scale_factor', epoch.get('scale_factor'))
        redshift = record.get('actual_redshift', record.get('redshift', epoch.get('redshift')))
        requested = record.get('requested_redshift')
        requested_a = epoch.get('requested_scale_factor')
        if requested is None and requested_a is not None:
            requested = 1 / float(requested_a) - 1
        clock = epoch.get('cosmic_time_source') or {}
        statuses = record.get('products') or {}
        row = {'snapshot':int(snapshot), 'redshift':redshift, 'scale_factor':scale,
               'cosmic_time_gyr':epoch.get('cosmic_time_gyr', record.get('actual_cosmic_time_gyr')),
               'requested_redshift':requested,
               'redshift_offset':float(redshift)-float(requested) if redshift is not None and requested is not None else None,
               'clock_status':str(clock.get('status', 'UNAVAILABLE')),
               'products':'; '.join('%s: %s' % (key, value.get('status', 'UNAVAILABLE'))
                                    for key,value in sorted(statuses.items())) or 'Not listed'}
        for key in ('redshift','scale_factor','cosmic_time_gyr','requested_redshift','redshift_offset'):
            value = row[key]
            if value is not None and not math.isfinite(float(value)):
                row[key] = None
        rows.append(row)
    rows.sort(key=lambda r: (r['scale_factor'] is None, r['scale_factor'] if r['scale_factor'] is not None else 0, r['snapshot']))
    return rows


def write_snapshot_tables(snapshots, directory):
    """Publish offline HTML and unrounded CSV from the same source rows."""
    rows = snapshot_rows(snapshots)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'snapshot_epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[c[0] for c in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    headings = ''.join('<th scope="col">%s%s</th>' % (html.escape(label), ' <span>%s</span>' % html.escape(unit) if unit else '') for _,label,unit in COLUMNS)
    def cell(value):
        if value is None:
            return '<span class="missing">Unavailable</span>'
        if isinstance(value,float):
            return html.escape(format(value, '.7g'))
        return html.escape(str(value))
    body = ''.join('<tr>' + ''.join('<td>%s</td>' % cell(row[key]) for key,_,_ in COLUMNS) + '</tr>' for row in rows)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Snapshot epochs</title>
<style>body{margin:0;background:#091322;color:#e6eef8;font:15px system-ui}main{max-width:1500px;margin:auto;padding:36px}h1{font-size:28px;margin:0 0 8px}p{color:#b3c4d9;line-height:1.5}.table{overflow:auto;border:1px solid #31455c;border-radius:12px}table{border-collapse:collapse;width:100%%;text-align:left;white-space:nowrap}th{background:#1b3048;color:#e6faff;font-size:13px}th,td{padding:14px 16px;border-bottom:1px solid #273c52}th span{color:#a4c0d5}tr:nth-child(even){background:#101f31}tr:hover{background:#20364c}.missing{color:#f5c879}a{color:#85dcf1}</style>
<main><h1>Snapshot epochs</h1><p>Recorded output epochs, ordered by scale factor. Snapshot numbers are storage labels, not physical time. Cosmic time depends on the recorded clock; unavailable values are not estimated.</p><p><a href="snapshot_epochs.csv">Download full-precision CSV</a> · %d saved snapshots</p><div class="table"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div></main></html>''' % (len(rows),headings,body)
    (directory/'snapshot_epochs.html').write_text(page, encoding='utf-8')
    return {'html':'snapshot_epochs.html','csv':'snapshot_epochs.csv','rows':len(rows)}
