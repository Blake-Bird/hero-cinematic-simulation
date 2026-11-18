"""Build and display a descendant-linked halo ancestry from saved catalogues."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np


def build_merger_tree(catalogs, final_index, *, max_nodes=20000, box_mpc_h=None):
    """Trace all resolved adjacent-output progenitors of one final catalogue row.

    Input catalogues use the validated arrays produced by main_progenitor_histories.
    IDs are serialized as strings to preserve their exact value in browsers. Links
    spanning missing outputs are not inferred. The limit is reported explicitly.
    """
    if not catalogs or max_nodes < 1:
        raise ValueError('Catalogues and a positive node limit are required')
    if box_mpc_h is not None and (not math.isfinite(box_mpc_h) or box_mpc_h<=0):
        raise ValueError('A finite positive box size is required')
    lookup = []
    for cat in catalogs[:-1]:
        grouped = {}
        for i in np.flatnonzero(cat['valid'] & (cat['desc_ids'] >= 0)):
            grouped.setdefault(int(cat['desc_ids'][i]), []).append(int(i))
        lookup.append(grouped)
    nodes, edges = [], []
    pending = [(len(catalogs)-1, int(final_index), None)]
    truncated = False
    while pending:
        if len(nodes) >= max_nodes:
            truncated = True
            break
        epoch, row, child = pending.pop()
        cat = catalogs[epoch]
        index = len(nodes)
        nodes.append({'id':str(int(cat['ids'][row])), 'snapshot':int(cat['snapshot']),
                      'redshift':float(cat['redshift']), 'scale_factor':float(cat['scale']),
                      'mass_msun_h':float(cat['mass'][row]), 'epoch_index':epoch})
        if 'original_ids' in cat:
            nodes[-1]['original_halo_id'] = str(int(cat['original_ids'][row]))
        nodes[-1]['catalogue_row'] = int(row)
        nodes[-1]['identity_scope'] = 'Exact catalogue ID at this saved snapshot; branch alias is a display aid'
        positions=cat.get('position_mpc_h')
        if positions is not None and np.isfinite(positions[row]).all():
            nodes[-1]['position_mpc_h']=[float(value) for value in positions[row]]
        velocities=cat.get('velocity_peculiar_km_s')
        if velocities is not None and np.isfinite(velocities[row]).all():
            nodes[-1]['velocity_peculiar_km_s']=[float(value) for value in velocities[row]]
        if child is not None:
            edges.append([index, child])
        if epoch:
            prev = catalogs[epoch-1]
            matches = lookup[epoch-1].get(int(cat['ids'][row]), [])
            matches = [i for i in matches if not np.isfinite(prev['desc_scale'][i]) or
                       np.isclose(prev['desc_scale'][i], cat['scale'], rtol=1e-5, atol=1e-7)]
            # Visit the largest progenitor first for stable left-to-right layout.
            matches.sort(key=lambda i: (-float(prev['mass'][i]), int(prev['ids'][i])))
            pending.extend((epoch-1, i, index) for i in reversed(matches))
    if len(nodes) == 1:
        return None
    return {'schema':'adjacent-merger-tree-v1', 'nodes':nodes, 'edges':edges, 'box_mpc_h':box_mpc_h,
            'source':catalogs[-1]['source'], 'truncated':truncated, 'max_nodes':max_nodes,
            'scope':'Resolved linked progenitors in adjacent available catalogues; absent, below-floor and unlinked halos are not shown.'}


BRANCH_COLORS = ('#67e8df', '#ffc46b', '#ef95bc', '#a7a5ff', '#8dd67c', '#79c4fa', '#f09570', '#d4cf87',
                 '#e9e5f2', '#ca86e3', '#9fd7b8', '#e3a98b')


def branch_identity(tree):
    """Stable display aliases for connected main-progenitor branches.

    H0 follows the most massive incoming progenitor at each adjacent epoch.
    Aliases identify plotted branches; exact catalogue IDs remain epoch-local.
    """
    nodes = tree['nodes']
    parents = [[] for _ in nodes]
    for parent, child in tree['edges']:
        if not (0 <= child < parent < len(nodes)):
            raise ValueError('Tree edges must point to an earlier-listed descendant')
        parents[child].append(parent)
    aliases = [None] * len(nodes)
    next_branch = 1
    pending = [(0, 0)]
    while pending:
        i, branch = pending.pop()
        if aliases[i] is not None:
            raise ValueError('A tree node has multiple descendants or a cycle')
        aliases[i] = branch
        ordered = sorted(parents[i], key=lambda j: (-nodes[j]['mass_msun_h'], nodes[j]['id']))
        for rank, parent in enumerate(ordered):
            alias = branch if rank == 0 else next_branch
            if rank:
                next_branch += 1
            pending.append((parent, alias))
    if any(value is None for value in aliases):
        raise ValueError('Tree has nodes disconnected from its selected root')
    return [{'alias': 'H%d' % value, 'branch': value,
             'color': BRANCH_COLORS[value % len(BRANCH_COLORS)],
             'node_key': '%s:%s' % (nodes[i]['snapshot'], nodes[i]['id'])}
            for i, value in enumerate(aliases)]


def tree_coordinates(tree):
    """Tip-centered cladogram with physical scale factor increasing downward."""
    nodes = tree['nodes']
    parents = [[] for _ in nodes]
    for a, b in tree['edges']:
        parents[b].append(a)
    x = np.zeros(len(nodes))
    leaf = 0
    for i in reversed(range(len(nodes))):
        if parents[i]:
            x[i] = np.mean([x[j] for j in parents[i]])
        else:
            x[i] = leaf
            leaf += 1
    a = np.array([n['scale_factor'] for n in nodes], float)
    return x, a


def plot_merger_tree(tree, directory):
    """Publication figure of actual progenitor edges, sharing aliases with HTML."""
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from core.figures import export_figure
    nodes = tree['nodes']; identity = branch_identity(tree)
    x, a = tree_coordinates(tree)
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='#08131f')
    ax.set_facecolor('#0e2030')
    for parent, child in tree['edges']:
        mid = (a[parent] + a[child]) / 2
        path = PlotPath([(x[parent], a[parent]), (x[parent], mid),
                         (x[child], mid), (x[child], a[child])],
                        [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4])
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=identity[parent]['color'],
                               lw=1.2 + 2.0 * (identity[parent]['branch'] == 0), alpha=.8))
    mass = np.array([n['mass_msun_h'] for n in nodes])
    sizes = 12 + 130 * np.sqrt(mass / mass.max())
    ax.scatter(x, a, s=sizes, c=[v['color'] for v in identity], edgecolors='#e8f8ff', linewidths=.5, zorder=3)
    for branch in sorted({v['branch'] for v in identity}):
        tips = [i for i, v in enumerate(identity) if v['branch'] == branch]
        first = min(tips, key=lambda i: a[i])
        ax.annotate(identity[first]['alias'], (x[first], a[first]), xytext=(4, -12),
                    textcoords='offset points', color=identity[first]['color'], fontsize=10, weight='bold')
    ax.scatter([x[0]], [a[0]], s=[sizes[0]+80], facecolors='none', edgecolors='#ffdc8e', linewidths=2, zorder=4)
    ax.annotate('Final H0 · ID ' + nodes[0]['id'], (x[0], a[0]), xytext=(40, -8),
                textcoords='offset points', color='#fff3d2', weight='bold',
                arrowprops={'arrowstyle':'->','color':'#ffdc8e','lw':1.3})
    ax.set_ylim(a.max() + max(.0001, np.ptp(a)*.08), a.min() - max(.0001, np.ptp(a)*.1))
    ax.set_xlim(x.min() - .7, x.max() + 1.0); ax.set_xticks([])
    ax.set_ylabel('Scale factor a · time advances downward', color='#c5d9e9')
    ax.tick_params(colors='#c5d9e9'); ax.grid(axis='y', alpha=.12, color='#829bb0')
    for spine in ax.spines.values(): spine.set_color('#365169')
    ax.set_title('The ancestry of halo ' + nodes[0]['id'], color='#f0f7ff', fontsize=20, pad=25)
    fig.text(.13, .025, 'A branching tree of saved descendant links. H0 follows the largest progenitor; H1, H2… label other branches.\n'
             'Circle size encodes the catalogue mass. Branch labels are navigation aids, not persistent catalogue IDs. No missing links are inferred.',
             color='#b9cedf', fontsize=9)
    fig.subplots_adjust(bottom=.14, top=.88)
    try:
        export_figure(fig, Path(directory)/'halo_merger_histories.png', metadata={
            'product': 'descendant-linked halo merger tree', 'root_halo_id': nodes[0]['id'],
            'root_snapshot': nodes[0]['snapshot'], 'source': tree['source'],
            'node_count': len(nodes), 'edge_count': len(tree['edges']),
            'branch_alias_scope': 'Most massive progenitor continuity; epoch-local exact IDs retained in merger_tree.json',
            'truncated': tree['truncated']})
    finally:
        plt.close(fig)


def export_merger_tree(tree, directory):
    """Export a legible branching diagram, static figure and exact-ID data."""
    if not tree:
        return None
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    nodes, edges = tree['nodes'], tree['edges']
    identity = branch_identity(tree); x, a = tree_coordinates(tree)
    width, height = 1000, 1000
    xs = 90 + 810 * (x-x.min()) / max(1, np.ptp(x))
    ys = 70 + 830 * (a-a.min()) / max(.00001, np.ptp(a))
    maximum = max(n['mass_msun_h'] for n in nodes)
    paths, circles, labels = [], [], []
    for parent, child in edges:
        mid = (ys[parent]+ys[child])/2
        paths.append('<path d="M %.3f %.3f C %.3f %.3f %.3f %.3f %.3f %.3f" stroke="%s" stroke-width="%.1f"/>' %
                     (xs[parent], ys[parent], xs[parent], mid, xs[child], mid, xs[child], ys[child],
                      identity[parent]['color'], 3 if identity[parent]['branch']==0 else 1.6))
    for i, node in enumerate(nodes):
        label = '%s · catalogue ID %s · snapshot %s · z %.5g · Mass %.5g Msun/h' % (
            identity[i]['alias'], node['id'], node['snapshot'], node['redshift'], node['mass_msun_h'])
        position = node.get('position_mpc_h'); box = tree.get('box_mpc_h')
        located = position is not None and box is not None and all(0<=v<box for v in position)
        coords = (' data-x="%.8g" data-y="%.8g"' % (100*position[0]/box,100-100*position[1]/box)) if located else ''
        circles.append('<circle cx="%.3f" cy="%.3f" r="%.2f" fill="%s" data-node="%d" tabindex="0" role="button" aria-label="%s"%s><title>%s</title></circle>' %
                       (xs[i],ys[i],4+8*math.sqrt(node['mass_msun_h']/maximum),identity[i]['color'],i,
                        html.escape(label,quote=True),coords,html.escape(label)))
    for branch in sorted({v['branch'] for v in identity}):
        i=min((i for i,v in enumerate(identity) if v['branch']==branch),key=lambda i:a[i])
        labels.append('<text x="%.2f" y="%.2f" fill="%s" class="branch-label">%s</text>' %
                      (xs[i]+9,ys[i]-13,identity[i]['color'],identity[i]['alias']))
    for epoch in sorted({n['epoch_index'] for n in nodes})[::max(1,len({n['epoch_index'] for n in nodes})//10)]:
        i=next(i for i,n in enumerate(nodes) if n['epoch_index']==epoch)
        labels.append('<text x="4" y="%.2f" class="epoch-label">z %.3g</text>' % (ys[i],nodes[i]['redshift']))
    warning = 'Node limit reached: the displayed tree is incomplete.' if tree['truncated'] else 'All reachable links passing the catalogue selection are shown.'
    payload=json.dumps({'nodes':nodes,'identity':identity},separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Halo merger tree</title>
<style>*{box-sizing:border-box}body{margin:0;background:#08131f;color:#e9f5ff;font:16px/1.5 system-ui}header{padding:28px;max-width:1400px;margin:auto}h1{font-size:36px;margin:0}p{color:#aec8dc}.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px;max-width:1400px;margin:auto;padding:0 28px 28px}.panel{padding:22px;background:#122538;border:1px solid #29465c;border-radius:12px;align-self:start;position:sticky;top:20px}.scroll{overflow:auto;max-height:82vh;background:#0d1e2d;border:1px solid #29465c;border-radius:12px}.scroll svg{display:block;width:100%%;height:auto;min-width:600px}path{fill:none;opacity:.75}circle{stroke:#d9f6ff;stroke-width:.8;cursor:pointer}circle:hover,circle:focus{stroke:#fff;stroke-width:4;outline:none}.branch-label{font:bold 15px system-ui}.epoch-label{fill:#9db7cd;font:12px system-ui}button{background:#203b50;color:white;border:1px solid #45667c;padding:8px 13px;border-radius:7px;cursor:pointer}a{color:#70e6df}.identity{font-size:26px;color:#ffc46b}dl{overflow-wrap:anywhere}dt{color:#93afc6;font-size:12px;text-transform:uppercase}dd{margin:0 0 14px}#detail{display:block;margin-bottom:15px}@media(max-width:850px){.layout{grid-template-columns:1fr}.panel{position:static}.scroll{max-height:65vh}}</style>
<header><h1>One halo. Its branching ancestry.</h1><p>Earlier → later, top to bottom · branch colors stay consistent · select a halo to inspect its exact identity</p><p>%s %s</p><p><a href="halo_merger_histories.png">PNG</a> · <a href="merger_tree.json">Exact tree data</a> · <button id="fit">Fit tree</button> <button id="zoom">Enlarge</button></p></header>
<div class="layout"><div class="scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" aria-label="Linked halo ancestry">%s%s%s</svg></div><aside class="panel"><span id="detail" aria-live="polite">Select a halo to inspect its measured properties.</span><div id="properties"></div><p>Saved XY box position (where available):</p><svg viewBox="0 0 100 100" width="160" height="160" style="background:#08131f;border:1px solid #36556c"><circle id="location" r="3" fill="#ffd37a" visibility="hidden"/></svg><p>H0 tracks the largest progenitor. H1, H2… identify connected branches. Exact halo IDs are saved separately at each epoch.</p></aside></div>
<script>const tree=%s;document.querySelectorAll('.scroll circle').forEach((c,i)=>{const show=()=>{const n=tree.nodes[i],b=tree.identity[i];document.getElementById('detail').textContent=c.getAttribute('aria-label');document.getElementById('properties').replaceChildren();for(const [label,value] of [['Branch',b.alias],['Catalogue ID',n.id],['Snapshot',n.snapshot],['Redshift',n.redshift.toPrecision(6)],['Catalogue mass [Msun/h]',n.mass_msun_h.toExponential(4)]]){let dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;document.getElementById('properties').append(dt,dd)}document.querySelectorAll('.scroll circle').forEach(x=>x.style.strokeWidth='.8');c.style.strokeWidth='4';const p=document.getElementById('location');p.setAttribute('visibility',c.dataset.x?'visible':'hidden');if(c.dataset.x){p.setAttribute('cx',c.dataset.x);p.setAttribute('cy',c.dataset.y)}};c.addEventListener('click',show);c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show()}})});document.getElementById('fit').onclick=()=>document.querySelector('.scroll svg').style.width='100%%';document.getElementById('zoom').onclick=()=>document.querySelector('.scroll svg').style.width='180%%';</script></html>''' % (html.escape(tree['scope']),html.escape(warning),''.join(paths),''.join(circles),''.join(labels),payload)
    path=out/'merger_tree.html';path.write_text(page,encoding='utf-8')
    (out/'merger_tree.json').write_text(json.dumps(tree,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot_merger_tree(tree,out)
    return path
