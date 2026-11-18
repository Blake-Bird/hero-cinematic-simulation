#!/usr/bin/env python3
"""Run the repository's native structure producer on a reciprocal z=0 pair."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256
from core.halo_products import export_native_structure_figures
from core.halos import (load_native_host_component_samples,
                        measure_mass_density_kinematics_spin, publish_halo_structure, verify_halo_structure)
from core.paired_atlas import epoch_name, _write_csv


def publish_shell_profiles(request, out, model, host, radius, native_report):
    """Publish reusable native shell tables beside the full structure product."""
    components, context = load_native_host_component_samples(
        request['run_directory'], request['snapshot'], host, request['components'],
        request['support_radius_proper_kpc'], memory_gib=request['memory_gib'])
    profile = measure_mass_density_kinematics_spin(
        components, request['radial_edges_proper_kpc'], request['bulk_velocity_peculiar_km_s'])['dm']
    covariance = profile['velocity_covariance_km2_s2']
    profile['velocity_dispersion_1d_km_s'] = np.sqrt(np.trace(covariance, axis1=1, axis2=2)/3)
    stem = out / f'{model}_z_000p000_FOF_row_{host}_density_and_kinematics'
    np.savez_compressed(stem.with_suffix('.npz'), **profile)
    keys = ('radius_proper_kpc', 'radial_outer_edge_proper_kpc', 'shell_count', 'density_msun_kpc3',
            'density_shot_noise_msun_kpc3', 'circular_velocity_spherical_km_s', 'velocity_dispersion_1d_km_s',
            'anisotropy_beta', 'bullock_spin_enclosed', 'kinematics_supported')
    _write_csv(stem.with_suffix('.csv'), [dict(zip(keys, row)) for row in zip(*(profile[k] for k in keys))])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    fig.suptitle(f'{model} · z = 0 · matched FOF row {host}\nDM profiles within a complete proper aperture', fontsize=17)
    for ax, key, label in zip(axes.flat, keys[3:4]+keys[5:8],
                ('Density [Msun/kpc³]', 'Spherical circular velocity [km/s]', '1D peculiar dispersion [km/s]', 'Anisotropy β')):
        support = profile['shell_count'] >= 100
        if key in keys[6:8]:
            support &= profile['kinematics_supported']
        x = profile['radial_outer_edge_proper_kpc'] if key == keys[5] else profile['radius_proper_kpc']
        y = np.where(support, profile[key], np.nan)
        ax.plot(x, y, 'o-', color='#16737b', markersize=3)
        if key == keys[3]:
            ax.set_yscale('log')
        ax.axvline(radius, color='#b06e40', linestyle='--', linewidth=.9, label='R200c')
        ax.set(xscale='log', xlabel='Proper radius [kpc]', ylabel=label)
        ax.grid(alpha=.22)
    axes[0, 0].legend()
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(stem.with_suffix(suffix), dpi=180)
    atomic_json(out / 'profile_contract.json', {
        'status': 'MEASURED_NATIVE_SHELLS', 'model': model, 'snapshot': 85, 'FOF_row': host,
        'source_context': context, 'minimum_shell_count_drawn': 100, 'R200c_proper_kpc': radius,
        'potential_shape_status': 'FIELD_VALIDATED' if native_report['field_validation_status']=='PASS' else 'UNRESOLVED_FIELD_BUDGET',
        'native_structure_product': 'native_structure.json',
        'field_validation_status': native_report['field_validation_status'],
        'arrays_sha256': file_sha256(stem.with_suffix('.npz')),
        'scope': 'Complete aperture DM particles around the native host centre. Circular velocity is spherical-equivalent. The native shell statistics retain particles at the origin.'})
    return stem.with_suffix('.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    base = root / '01_Comparisons' / epoch_name('EDE_vs_LCDM', 85, 0)
    matches, = base.glob('*_Matched_FOF_host_properties.csv')
    with matches.open() as stream:
        pair = max(csv.DictReader(stream), key=lambda r: float(r['mass_EDE_msun'])*float(r['mass_LCDM_msun']))
    host = int(pair[f'FOF_row_{model}'])
    native = root / 'Source/Native_workspaces' / f'{model}_structure'
    from scripts.prepare_paired_analysis import prepare
    prepare(root.parent / f'dmo3_N128_L5_fast128_{model.lower()}', native)
    data = root / '06_Tables_and_Data/Measurements' / model / (epoch_name(model, 85, 0)+'.npz')
    with np.load(data, allow_pickle=False) as arrays:
        radius = float(arrays['r200c_comoving_mpc'][host])*1000.
        bulk = arrays['fof_velocity_peculiar_km_s'][host].tolist()
    if not 5 <= radius <= 1000:
        raise ValueError(f'Invalid matched-host R200c {radius}')
    template = Path(__file__).resolve().parents[1] / 'configs/halo_structure.example.json'
    request = json.loads(template.read_text())
    support = 1.5 * radius
    request.update(run_directory=str(native), snapshot=85, host_row=host, support_radius_proper_kpc=support,
                   radial_edges_proper_kpc=[0., *np.geomspace(.025*radius, support, 15)],
                   bulk_velocity_peculiar_km_s=bulk, memory_gib=4.)
    request['shape_controls']['minimum_particles'] = 100
    request['expansion_sequence'] = [
        {'lmax': order, 'mmax': order, 'grid_size_r': grid, 'grid_size_z': grid,
         'rmin_kpc': .003*radius, 'rmax_kpc': 3*radius, 'zmin_kpc': .003*radius, 'zmax_kpc': 3*radius}
        for order, grid in ((4, 24), (6, 36), (8, 48))]
    request['calibration_positions_proper_kpc'] = (radius*np.asarray([[.25,0,0],[0,.65,0],[0,0,1.3]])).tolist()
    request['validation_positions_proper_kpc'] = (radius*np.asarray([[.22,.06,.03],[-.11,.56,.14],[.07,.09,-1.19]])).tolist()
    request['equipotential_controls'].update(reference_radii_kpc=(radius*np.array([.18,.35,.6,1.])).tolist(),
                                             radial_bracket_kpc=[.025*radius, 1.45*radius])
    model_dir = root / ('02_EDE' if model == 'EDE' else '03_LCDM')
    out = model_dir / 'Spatial_explorers' / f'{model}_z_000p000' / f'Native_structure_FOF_row_{host}'
    request_file = out / 'request.json'
    atomic_json(request_file, request)
    product = out / 'native_structure.json'
    if not product.exists():
        publish_halo_structure(request, product, request_source_hashes={
            str(request_file): file_sha256(request_file), str(matches): file_sha256(matches)})
    record = verify_halo_structure(product)
    if not (out/'Figures').exists():
        export_native_structure_figures(product, out/'Figures', memory_gib=4.)
    result = publish_shell_profiles(request, out, model, host, radius, record)
    print(result, flush=True)


if __name__ == '__main__':
    main()
