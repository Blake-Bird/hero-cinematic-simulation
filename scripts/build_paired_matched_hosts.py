#!/usr/bin/env python3
"""Compare reciprocal FOF matches with exact initial lattice identities."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import _mutual_best_halo_matches, _read_cross_run_halo_catalogue
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import epoch_name, _write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    phase_path = root / '07_Status_and_Provenance/physical_ic_agreement.json'
    phase = json.loads(phase_path.read_text())
    if phase['phase_status'] != 'PASS' or phase['shared_lattice_status'] != 'PASS':
        raise ValueError('Exact lattice and measured initial phase verification are required')
    workspaces = [root / 'Source/Native_workspaces' / model for model in ('EDE', 'LCDM')]
    hs = [float(yaml.safe_load((path / 'config.yaml').read_text())['cosmology']['H0'])/100 for path in workspaces]
    progress = {}
    for snap, z in ((21, 14.4), (26, 12), (30, 10), (35, 8), (41, 6), (54, 3), (69, 1), (85, 0)):
        label = epoch_name('EDE_vs_LCDM', snap, z)
        out = root / '01_Comparisons' / label
        try:
            cats = [_read_cross_run_halo_catalogue(path, snap, 'fof', 300, 2_100_000) for path in workspaces]
            boxes = [float(cat['box_mpc_h']) / h for cat, h in zip(cats, hs)]
            if not np.isclose(*boxes, rtol=1.e-7):
                raise ValueError('Physical box sizes disagree')
            matches = _mutual_best_halo_matches(*cats, 32, .25)
        except ValueError as error:
            progress[label] = {'status': 'UNAVAILABLE', 'reason': str(error)}
            continue
        index = [{int(g): i for i, g in enumerate(cat['group_indices'])} for cat in cats]
        rows = []
        for g_a, g_b, shared, fa, fb, merit in zip(*(matches[k] for k in ('a', 'b', 'shared', 'fraction_a', 'fraction_b', 'merit'))):
            i, j = index[0][g_a], index[1][g_b]
            a, b = cats
            m_a, m_b = a['mass_msun_h'][i]/hs[0], b['mass_msun_h'][j]/hs[1]
            delta = b['position_mpc_h'][j]/hs[1] - a['position_mpc_h'][i]/hs[0]
            displacement = np.linalg.norm((delta + .5*boxes[0]) % boxes[0] - .5*boxes[0])
            vmax_a, vmax_b = a['vmax'][i], b['vmax'][j]
            rows.append({'FOF_row_EDE': int(g_a), 'FOF_row_LCDM': int(g_b), 'exact_shared_particle_ids': int(shared),
                         'overlap_fraction_EDE': fa, 'overlap_fraction_LCDM': fb, 'overlap_merit': merit,
                         'mass_EDE_msun': m_a, 'mass_LCDM_msun': m_b, 'mass_ratio_EDE_to_LCDM': m_a/m_b,
                         'centre_shift_comoving_mpc': displacement, 'Vmax_EDE_km_s': vmax_a, 'Vmax_LCDM_km_s': vmax_b,
                         'Vmax_ratio_EDE_to_LCDM': vmax_a/vmax_b if np.isfinite(vmax_b) and vmax_b > 0 else np.nan})
        report = {'status': 'MEASURED', 'reciprocal_matches': len(rows), 'candidate_pairs': matches['candidate_edges'],
                  'eligible_EDE_hosts': len(index[0]), 'eligible_LCDM_hosts': len(index[1]),
                  'selection': '300 particles per host; at least 32 exact shared IDs and at least 25% of each host; reciprocal best overlap merit.',
                  'ic_agreement_sha256': file_sha256(phase_path)}
        progress[label] = report
        atomic_json(out / f'{label}_Matched_hosts_support.json', report)
        if not rows:
            continue
        stem = out / f'{label}_Matched_FOF_host_properties'
        _write_csv(stem.with_suffix('.csv'), rows)
        x = np.sqrt([r['mass_EDE_msun']*r['mass_LCDM_msun'] for r in rows])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout='constrained')
        fig.suptitle(f'Matching haloes by shared initial particles · z = {z:g}\n'
                     f'{len(rows)} reciprocal pairs · ≥300 particles per host', fontsize=16)
        for ax, key, ylabel in zip(axes, ('mass_ratio_EDE_to_LCDM', 'centre_shift_comoving_mpc', 'Vmax_ratio_EDE_to_LCDM'),
                ('FOF mass ratio · EDE / ΛCDM', 'Centre shift [comoving Mpc]', 'Central Vmax ratio · EDE / ΛCDM')):
            y = np.asarray([row[key] for row in rows])
            valid = np.isfinite(y)
            ax.scatter(x[valid], y[valid], c='#116f7a', alpha=.65, s=24, edgecolors='none')
            if 'ratio' in key:
                ax.axhline(1, color='#96694d', linewidth=.8, linestyle='--')
            if not np.any(valid):
                ax.text(.5, .5, 'No supported pairs', ha='center', transform=ax.transAxes)
            ax.set(xscale='log', xlabel='Geometric mean FOF mass [Msun]', ylabel=ylabel)
            ax.grid(alpha=.18)
        for suffix in ('.png', '.pdf', '.svg'):
            fig.savefig(stem.with_suffix(suffix), dpi=175)
        plt.close(fig)
        manifest_path = out / 'comparison_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for key, suffix in (('figures', '.png'), ('tables', '.csv')):
            manifest[key] = sorted(set(manifest[key]) | {stem.with_suffix(suffix).name})
        atomic_json(manifest_path, manifest)
    atomic_json(root / '07_Status_and_Provenance/matched_fof_hosts.json', progress)
    print(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
