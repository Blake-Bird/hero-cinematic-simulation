"""Export supervised-learning measurements directly from full-box GADGET runs.

CLASS/formula screening products never provide target halo counts here.
Only catalogue host masses and saved snapshot headers provide measurements.
"""
from pathlib import Path

import h5py
import numpy as np
import yaml

from core.foundation import atomic_json, file_sha256
from core.hmf_campaign import write_csv


def export_gadget_measurements(run, output, snapshots=None):
    from core.analysis import (snapshot_ids, read_snapshot_header, group_files, halo_masses,
                               halo_particle_counts, apply_hmf_particle_cut, mass_to_msun_h,
                               hmf_mass_edges, redshift_from_header)
    run, output = Path(run).expanduser().resolve(), Path(output).expanduser().resolve()
    config_path = run/'config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    sim = cfg['simulation']
    if sim['physics'] not in {'dm', 'dmo'}:
        raise ValueError('Export a separate calibrated hydro dataset; this exporter measures full-box DMO runs')
    if sim.get('zoom_softening_mass_ratios') or cfg.get('zoom', {}).get('enabled'):
        raise ValueError('A selected zoom volume cannot supply full-box training counts')
    n = sim['particles_per_dimension']
    if type(n) is not int or n <= 0:
        raise ValueError('Positive integer particles_per_dimension required')
    box = float(sim['box_mpc_h'])
    if not np.isfinite(box) or box <= 0:
        raise ValueError('Positive finite full-box size required')
    seed = sim['random_seed']
    edges = hmf_mass_edges(cfg)
    minimum = cfg['analysis']['hmf_min_particles']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('A positive halo-particle resolution cut is required')
    available = snapshot_ids(run)
    selected = list(available if snapshots is None else snapshots)
    if not selected or len(set(selected)) != len(selected) or set(selected)-set(available):
        raise ValueError('Select distinct existing snapshots')
    if output.exists():
        raise FileExistsError(output)
    rows, epochs = [], []
    sources = {str(config_path): file_sha256(config_path)}
    for snap in selected:
        header, snapshot_paths = read_snapshot_header(run, snap)
        if not np.isclose(float(header['BoxSize']), box*1000, rtol=1e-8):
            raise ValueError('Saved snapshot box disagrees with configured full box')
        totals = np.array(header['NumPart_Total'], dtype=np.uint64)
        totals += np.array(header.get('NumPart_Total_HighWord', np.zeros(6)), dtype=np.uint64) << np.uint64(32)
        if len(totals) != 6 or int(totals[1]) != n**3 or np.any(totals[[0, 2, 3, 4, 5]]):
            raise ValueError('Snapshot must contain the declared full uniform DMO particle load')
        particle_mass = float(header['MassTable'][1])*1e10
        if not np.isfinite(particle_mass) or particle_mass <= 0:
            raise ValueError('Uniform DMO particle mass must be declared in MassTable')
        z = float(redshift_from_header(header))
        if not np.isfinite(z) or z < -1e-10:
            raise ValueError('Invalid saved snapshot epoch')
        z = max(0., z)
        files = group_files(run, snap)
        if not files:
            raise ValueError(f'No GADGET halo catalogue for snapshot {snap}; run the halo finder first')
        declared_totals, local_total = set(), 0
        for path in files:
            sources[str(path)] = file_sha256(path)
            with h5py.File(path, 'r') as stream:
                h = stream['Header'].attrs
                declared_totals.add(int(h['Ngroups_Total']))
                local_total += int(h['Ngroups_ThisFile'])
                if not np.isclose(float(h['Time']), float(header['Time']), rtol=1e-8):
                    raise ValueError('Catalogue and snapshot epochs disagree')
        if len(declared_totals) != 1 or local_total != next(iter(declared_totals)):
            raise ValueError('Incomplete or inconsistent GADGET catalogue shards')
        masses, particles = halo_masses(run, snap), halo_particle_counts(run, snap)
        if local_total == 0:
            masses, particles = np.empty(0), np.empty(0, dtype=int)
        elif masses is None or particles is None or len(masses) != local_total:
            raise ValueError('Host masses and particle counts must cover the complete catalogue')
        masses = np.asarray(masses, float)
        if not np.isfinite(masses).all() or np.any(masses < 0):
            raise ValueError('Invalid catalogue host masses')
        cut, _, selection = apply_hmf_particle_cut(masses, particles, minimum)
        physical_mass = mass_to_msun_h(cut)
        counts, _ = np.histogram(physical_mass[physical_mass > 0], bins=edges)
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            rows.append(dict(snapshot=int(snap), mass_lower_msun_h=float(lo), mass_upper_msun_h=float(hi),
                halo_count=int(count), volume_mpc_h3=box**3, redshift=z,
                particle_mass_msun_h=particle_mass, minimum_halo_particles=minimum,
                mass_definition=cfg['analysis']['hmf_mass_definition'], box_size_mpc_h=box,
                particles_per_dimension=n, random_seed=seed))
        epochs.append(dict(snapshot=int(snap), redshift=z, particle_mass_msun_h=particle_mass,
            snapshot_files=[{'path': str(p), 'size_bytes': p.stat().st_size} for p in snapshot_paths],
            snapshot_header={'Time': float(header['Time']), 'BoxSize': float(header['BoxSize']),
                             'NumPart_Total': [int(v) for v in totals], 'MassTable': list(map(float, header['MassTable']))},
            halo_selection=selection, measured_halos_in_mass_window=int(counts.sum())))
    # Detect changed measurements before publishing their training targets.
    for path, expected in sources.items():
        if file_sha256(path) != expected:
            raise ValueError('Measurement source changed while exporting: '+path)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'hmf.csv', rows)
    metadata = dict(schema='gadget-hmf-measurements-v1', target_kind='simulation_counts',
        finder='gadget_fof_subfind', physics=sim['physics'], config=cfg, source_run=str(run),
        particles_per_dimension=n, particle_count=n**3, random_seed=seed, epochs=epochs,
        source_sha256=sources, measurement_sha256=file_sha256(output/'hmf.csv'),
        target_origin='Host counts measured directly from GADGET FoF/SUBFIND Group fields; no fitting formula targets or inputs',
        limitations='Full-box uniform DMO only. Particle cuts do not prove resolution convergence; compare matched 512^3 and 1024^3 runs before precision training.')
    atomic_json(output/'metadata.json', metadata)
    return metadata


def export_gadget_tree(runs_root, output, snapshots=None):
    """Collect many completed runs into independent measured-data folders."""
    from core.analysis import snapshot_ids
    root, output = Path(runs_root).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = sorted({p.parent for p in root.rglob('config.yaml')
                         if (p.parent/'data').is_dir() and not (p.parent/'data').is_symlink()})
    runs = [run for run in candidates if snapshot_ids(run)]
    if not runs:
        raise ValueError('No saved GADGET runs under runs-root')
    output.mkdir(parents=True, exist_ok=False)
    sources, failed = [], []
    for index, run in enumerate(runs):
        folder = output/f'run_{index:05d}'
        try:
            export_gadget_measurements(run, folder, snapshots)
            sources.append({'path': str(folder)})
        except Exception as exc:
            failed.append({'run': str(run), 'error': str(exc)})
    report = dict(schema='gadget-hmf-collection-v1', status='PARTIAL' if failed and sources else 'FAILED' if failed else 'COMPLETE',
                  exported_runs=len(sources), failures=failed, target_origin='GADGET catalogue host counts only')
    atomic_json(output/'collection.json', report)
    atomic_json(output/'dataset-request.json', dict(sources=sources, seed=42, resolution_policy='finest_per_bin'))
    return report
