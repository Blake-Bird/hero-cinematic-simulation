"""Overlay selected measurements from every run at a common saved epoch."""
from __future__ import annotations

import csv
import json
import math
import tempfile
from contextlib import ExitStack
from pathlib import Path


def compare_suite(runs, bundles, destination, *, product, redshift_tolerance=1e-6):
    import h5py
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.analysis import verify_analysis_bundle
    from core.foundation import atomic_json, file_sha256, publish_directory
    from core.native import native_run_lock

    specs = {
        'hmf': ('mass_msun_h', 'dn_dlogM', 'Halo mass [Msun/h]', 'dn/dlog₁₀ M', True),
        'pk': ('k_h_Mpc', 'P_Mpc_h3', 'k [h/Mpc]', 'P(k) [(Mpc/h)³]', True),
        'correlation': ('r', 'xi', 'r [comoving Mpc/h]', 'Halo correlation ξ(r)', False),
    }
    if product not in specs:
        raise ValueError('Unsupported suite comparison product')
    roots = [Path(p).expanduser().resolve() for p in runs]
    sources = [Path(p).expanduser().resolve() for p in bundles]
    if len(roots) < 2 or len(roots) != len(sources) or len(set(roots)) != len(roots):
        raise ValueError('Supply at least two distinct runs and one analysis bundle per run')
    if not math.isfinite(redshift_tolerance) or redshift_tolerance <= 0:
        raise ValueError('Redshift tolerance must be finite and positive')
    out = Path(destination).expanduser().resolve()
    if out.exists():
        raise FileExistsError('Choose a fresh suite comparison directory')

    with ExitStack() as locks:
        for root in sorted(roots):
            locks.enter_context(native_run_lock(root, operation='suite comparison'))
        epochs = []
        hashes = []
        definitions = []
        for root, bundle in zip(roots, sources):
            record = verify_analysis_bundle(bundle, verify_sources=True)
            if Path(record['source_run']).resolve() != root:
                raise ValueError('Analysis belongs to another run: ' + str(bundle))
            summary = json.loads((bundle / 'analysis_products_summary.json').read_text())
            rows = []
            for snap, item in summary.get('snapshots', {}).items():
                z = float(item['redshift'])
                if math.isfinite(z) and item.get('products', {}).get(product, {}).get('status') == 'COMPLETE':
                    rows.append((int(snap), z))
            if not rows:
                raise ValueError(root.name + ' has no completed ' + product + ' epochs')
            epochs.append(rows)
            hashes.append(file_sha256(bundle / 'analysis_provenance.json'))
            if product == 'hmf':
                config = yaml.safe_load((root / 'config.yaml').read_text())
                definitions.append(str(config.get('analysis', {}).get('hmf_mass_definition', 'fof')).lower())
        if definitions and len(set(definitions)) != 1:
            raise ValueError('HMF mass definitions differ across selected runs')

        candidates = []
        for anchor in epochs[0]:
            selected = [anchor] + [min(rows, key=lambda row: (abs(row[1] - anchor[1]), row[0])) for rows in epochs[1:]]
            if all(abs(a[1] - b[1]) <= redshift_tolerance * max(1., abs(a[1]), abs(b[1]))
                   for a in selected for b in selected):
                candidates.append(selected)
        if not candidates:
            raise ValueError('No common completed ' + product + ' epoch across every selected run')
        selected = min(candidates, key=lambda rows: (max(row[1] for row in rows), max(row[1] for row in rows) - min(row[1] for row in rows)))
        xname, yname, xlabel, ylabel, positive = specs[product]
        curves = []
        units = []
        for root, bundle, (snap, z) in zip(roots, sources, selected):
            with h5py.File(bundle / 'analysis_products.h5', 'r') as handle:
                group = handle['snap_%03d/%s/data' % (snap, product)]
                if 'status' in group:
                    status = group['status'][()]
                    status = status.decode() if isinstance(status, bytes) else str(status)
                    if status != 'COMPLETE':
                        raise ValueError(root.name + ' has an incomplete measured product')
                x = np.asarray(group[xname][:], float)
                y = np.asarray(group[yname][:], float)
                if x.ndim != 1 or x.shape != y.shape:
                    raise ValueError('Misaligned measured arrays in ' + root.name)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
                if positive:
                    mask &= y > 0
                if not np.any(mask):
                    raise ValueError('No displayable measured bins in ' + root.name)
                curves.append((x[mask], y[mask]))
                units.append(tuple(str(group[name].attrs.get('units', 'not declared')) for name in (xname, yname)))
        if len(set(units)) != 1:
            raise ValueError('Measured units differ across selected runs')
        if product == 'hmf':
            ylabel += ' [' + units[0][1] + ']'

        out.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.suite-comparison-', dir=out.parent))
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='#081321')
            ax.set_facecolor('#14283e')
            palette = plt.get_cmap('turbo')
            for i, (root, (x, y), (snap, z)) in enumerate(zip(roots, curves, selected)):
                draw = ax.loglog if positive else ax.semilogx
                draw(x, y, color=palette(.08 + .84 * i / max(1, len(roots) - 1)),
                     linestyle=('-', '--', '-.', ':')[i % 4], marker='o', ms=3,
                     label='%d. %s · z=%.6g' % (i + 1, root.name, z))
            if not positive:
                ax.axhline(0, color='#8ca4b8', lw=.8)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.set_title(product.upper() + ' · all %d selected runs' % len(roots), color='white', pad=15)
            ax.tick_params(colors='#d5e5f2')
            ax.xaxis.label.set_color('#d5e5f2'); ax.yaxis.label.set_color('#d5e5f2')
            ax.grid(alpha=.15)
            ax.legend(labelcolor='white', facecolor='#14283e', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
            fig.text(.08, .015, 'Independent saved measurements; phases, resolution, selection and binning may differ.', color='#bcd0e1', fontsize=9)
            fig.tight_layout(rect=(0, .04, 1, 1))
            for suffix in ('png', 'pdf'):
                fig.savefig(stage / ('comparison.' + suffix), dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
            with (stage / 'curves.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['run_index', 'run_path', 'snapshot', 'redshift', xname, yname])
                for i, (root, (snap, z), (x, y)) in enumerate(zip(roots, selected, curves), 1):
                    writer.writerows((i, str(root), snap, z, float(a), float(b)) for a, b in zip(x, y))
            for bundle, digest in zip(sources, hashes):
                verify_analysis_bundle(bundle, verify_sources=True)
                if file_sha256(bundle / 'analysis_provenance.json') != digest:
                    raise ValueError('Source analysis changed during comparison')
            atomic_json(stage / 'comparison.json', {
                'schema': 'selected-suite-comparison-v1', 'science_acceptance': 'NOT_ACCEPTED',
                'product': product, 'runs': [str(p) for p in roots],
                'analysis_bundles': [str(p) for p in sources], 'analysis_provenance_sha256': hashes,
                'snapshots': [row[0] for row in selected], 'redshifts': [row[1] for row in selected],
                'redshift_tolerance': redshift_tolerance, 'units': units[0], 'hmf_mass_definitions': definitions,
                'scope': 'Every selected run at a common saved epoch; independent bins, no interpolation or inferred phase matching. CSV contains displayed finite bins.',
                'files': {name: file_sha256(stage / name) for name in ('comparison.png', 'comparison.pdf', 'curves.csv')},
            })
            publish_directory(stage, out)
            return out / 'comparison.png'
        except BaseException as error:
            atomic_json(stage / 'failure.json', {'status': 'FAIL', 'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            if fig is not None:
                plt.close(fig)
