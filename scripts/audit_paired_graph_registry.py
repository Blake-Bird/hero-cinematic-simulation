#!/usr/bin/env python3
"""Inventory each registered scientific graph against the paired native data."""
import argparse
import csv
import json
from pathlib import Path


FAMILY_REQUIREMENTS = {
    'streams': ('REQUIRES_SEPARATE_TRACER_SELECTION', 'No measured stellar stream or declared resolved collisionless stream selection.'),
    'encounters': ('REQUIRES_SEPARATE_EVENT_SELECTION', 'Requires named encounters and source-bound time sequences or controlled perturber pairs.'),
    'black_holes': ('ABSENT_PHYSICS', 'Neither run contains native type-5 black holes or their feedback records.'),
    'baryons': ('ABSENT_PHYSICS', 'Dark matter only. No native gas, stellar populations, enrichment or light model.'),
    'sidm': ('ABSENT_PHYSICS', 'No SIDM scattering events, matched SIDM runs, or SIDM convergence family.'),
    'hubble': ('REQUIRES_EXTERNAL_INPUTS', 'Requires method-specific observations, calibration products or validated recovery campaigns.'),
    'inference': ('REQUIRES_EXTERNAL_INPUTS', 'No source-bound simulation ensemble or posterior/inference diagnostics.'),
    'convergence': ('REQUIRES_SEPARATE_RUNS', 'One N128 realization per full parameter set. The models are not controlled resolution or timestep variants.'),
}
VARIANT_COVERAGE = {
    'input_power': ('PUBLISHED_DESCRIPTIVE', 'Saved CLASS/AxiCLASS linear input spectra and signed density transfers.'),
    'evolved_power': ('PUBLISHED_DESCRIPTIVE', 'Native P(k) and common physical units at priority epochs; exact arrays at each saved output.'),
    'power_ratios': ('PUBLISHED_DESCRIPTIVE', 'Power ratios at priority epochs; diagonal mode uncertainty only.'),
    'transfer_comparison': ('PUBLISHED_DESCRIPTIVE', 'Native signed d_tot, d_b and d_cdm shown side by side.'),
    'matched_residuals': ('REQUIRES_JOINT_COVARIANCE', 'Matching physical boxes and initial phases are measured. Full joint covariance is unavailable.'),
    'differential_hmf': ('PUBLISHED_DESCRIPTIVE', 'Native and paired HMF at priority epochs with measured support.'),
    'cumulative_hmf': ('PUBLISHED_DESCRIPTIVE', 'Common mass thresholds, volume and counts.'),
    'hmf_ratio': ('PUBLISHED_DESCRIPTIVE', 'Physical-unit ratios, with unsupported zero-denominator bins marked.'),
    'reliability_limits': ('PARTIAL', 'Particle-count floors and reference-fit support masks. Numerical convergence requires additional runs.'),
    'stellar_mass_function': ('ABSENT_PHYSICS', 'No saved star particles.'),
    'vmax_function': ('PUBLISHED_DESCRIPTIVE', 'Matched units, measured SUBFIND Vmax and selection.'),
    'rare_high_z_abundance': ('PARTIAL', 'Descriptive high-z native counts, physical HMF and resolved cuts; rare-halo sample variance unmeasured.'),
    'density_profile': ('PUBLISHED_NATIVE', 'Native host density profiles at supported priority epochs.'),
    'velocity_dispersion': ('PUBLISHED_NATIVE', 'Native host velocity dispersion profiles at supported priority epochs.'),
    'concentration_mass': ('PUBLISHED_NATIVE', 'Native concentration plots where halo count and fits support the measurement.'),
    'radial_axis_ratios': ('PUBLISHED_DIAGNOSTIC', 'z=0 radial density-shape tensor panels for each run.'),
    'triaxiality': ('PUBLISHED_DIAGNOSTIC', 'Native halo triaxiality and supported z=0 radial measurements.'),
    'axis_orientation_twist': ('PUBLISHED_DIAGNOSTIC', 'Radial shape uncertainty and orientation panels at z=0.'),
    'mass_scale_factor': ('PUBLISHED_NATIVE', 'Native halo histories and resolved branches; early FOF exploratory view separately labelled.'),
    'log_mass_accretion': ('PARTIAL', 'Mass histories available. Additional original growth-rate products require their declared selection.'),
    'pericenter_infall_merger_events': ('PARTIAL', 'Resolved merger links and mass ratios published. Orbit/pericenter selections remain open.'),
    'stellar_assembly': ('ABSENT_PHYSICS', 'No saved stars.'),
    'quenching_morphology': ('ABSENT_PHYSICS', 'No saved stars or gas.'),
    'filament_map': ('PARTIAL', 'Measured filament camera tours and structure diagnostics, environment products still under review.'),
    'web_evolution': ('PUBLISHED_DESCRIPTIVE', 'Shared-exposure density film, all native saved epochs, and void catalogue trackers.')
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    repo = Path(__file__).resolve().parents[1]
    registry = json.loads((repo / 'configs/research/graph_producers.json').read_text())
    rows = []
    for item in registry['entries']:
        family, variant = item['family'], item['variant']
        status, reason = VARIANT_COVERAGE.get(variant, FAMILY_REQUIREMENTS.get(family, (
            'OPEN_COLLISIONLESS_MEASUREMENT', 'Requires separate host/epoch selection, estimator execution and verification.')))
        rows.append({'family': family, 'variant': variant, 'status': status, 'basis': reason,
                     'required_measurement': item['required_measurement'],
                     'producer': '; '.join(f"{p['module']}.{p['function']}" for p in item['producer_route'])})
    if len(rows) != registry['variant_count'] or len({(r['family'], r['variant']) for r in rows}) != len(rows):
        raise ValueError('Graph registry variants are incomplete or duplicated')
    folder = root / '07_Status_and_Provenance'
    with (folder / 'registered_graph_coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'registered_graph_coverage.json').write_text(json.dumps({
        'scope': 'Initial variant-by-variant atlas coverage. Descriptive analogues do not satisfy stricter original acceptance, covariance, or campaign requirements.',
        'count': len(rows), 'rows': rows}, indent=2) + '\n')
    print('GRAPH_VARIANTS', len(rows))


if __name__ == '__main__':
    main()
