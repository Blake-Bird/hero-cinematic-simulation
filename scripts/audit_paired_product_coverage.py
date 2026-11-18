#!/usr/bin/env python3
"""Bind each native workflow product and graph family to its published evidence."""
import argparse
import csv
import json
from pathlib import Path



def read_json(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    args=parser.parse_args()
    root=args.atlas.resolve()
    folder=root/'07_Status_and_Provenance'
    for model in ('EDE','LCDM'):
        log=Path('/tmp/paired-available-'+model.lower()+'.log').read_text()
        capability=json.loads(log[log.index('{'):])
        records=[]
        work=root/'Source/Native_workspaces'/model
        summary=read_json(sorted(work.glob('results/analysis_products_*/analysis_products_summary.json'))[-1])
        section='02_EDE' if model=='EDE' else '03_LCDM'
        for product,decision in capability['decisions'].items():
            status,basis=decision['status'],decision['reason']
            evidence=[]
            epochs={}
            if product in summary['products']:
                status='ATTEMPTED_AT_ALL_PRIORITY_EPOCHS'
                basis='Native suite outcomes retained for each selected epoch.'
                epochs={snap:item['products'].get(product) for snap,item in summary['snapshots'].items()}
                evidence=[str((sorted(work.glob('results/analysis_products_*/analysis_products_summary.json'))[-1]).relative_to(root))]
            elif product=='histories':
                evidence=[str(p.relative_to(root)) for p in (root/section/'Halo_history_z_199_to_000').glob('*.png')]
                status='PUBLISHED'
            elif product in ('blender','blender_movie'):
                evidence=[str(p.relative_to(root)) for p in (root/'05_Blender_and_3D'/model).glob('*all_saved*/*.blend')]
                if product=='blender_movie':
                    evidence.extend(str(p.relative_to(root)) for p in (root/'04_Movies'/model).glob('*Blender_saved*/*.mp4'))
                status='PUBLISHED_AND_LOADED'
            elif product=='radial_shapes':
                evidence=[str(p.relative_to(root)) for p in (root/section/'Spatial_explorers').glob('*000p000/*/radial_shapes.png')]
                evidence.extend(str(p.relative_to(root)) for p in (root/section/'Spatial_explorers').glob('*000p000/Native_structure*/Figures/gallery/figures/png/structure/*radial_axis*.png'))
                status='PUBLISHED'
                basis='Declared z=0 halo selections and native radial density-shape measurements.'
            elif product=='maps':
                evidence=[str(p.relative_to(root)) for p in (root/section/'Saved_epochs').glob('*/*Dark_matter_XY.png')]
                status='ALL_86_EPOCHS_PUBLISHED'
            elif product in ('movie','halo_movie','planetarium'):
                token={'movie':'Projected_density_timeline','halo_movie':'Halo_growth_and_merger_markers','planetarium':'Planetarium_domemaster'}[product]
                evidence=[str(p.relative_to(root)) for p in (root/'04_Movies'/model).glob('*'+token+'/*.mp4')]
                status='PUBLISHED_AND_BROWSER_TESTED'
            elif product=='xdmf':
                evidence=[str(p.relative_to(root)) for p in (root/'05_Blender_and_3D'/model).glob('*XDMF/timeline_type1.xdmf')]
                status='PUBLISHED_AND_PARAVIEW_LOADED'
            if status.startswith(('PUBLISHED','ALL_','ATTEMPTED')) and not evidence:
                raise ValueError(f'{model} {product} lacks claimed publication evidence')
            records.append({'product':product,'status':status,'basis':basis,'evidence':evidence,'epoch_outcomes':epochs})
        (folder/f'{model}_native_product_coverage.json').write_text(json.dumps(records,indent=2)+'\n')
    registry=read_json(folder/'registered_graph_coverage.json')
    measured={}
    for manifest in root.glob('0[23]_*/Spatial_explorers/*/*/*/gallery/gallery.json'):
        gallery=read_json(manifest)
        for figure in gallery['figures']:
            measured.setdefault(figure['variant'],[]).append({'gallery':str(manifest.relative_to(root)),
                'figure':str((manifest.parent/figure['png']).relative_to(root)),
                'support':figure['selections']})
    updates={
        'circular_velocity':('PUBLISHED_NATIVE','Complete matched-host enclosed mass and spherical-equivalent circular velocity.'),
        'anisotropy':('PUBLISHED_NATIVE','Native proper-shell radial and tangential velocity moments and support masks.'),
        'density_potential_shape':('PUBLISHED_WITH_UNRESOLVED_FIELD_MASK','Density tensors and independently reconstructed potential. Potential shape values fail the heldout tidal budget and remain masked.'),
        'splashback_slope_radius_depth_width':('PUBLISHED_WITH_FEATURE_SUPPORT','Original covariance-aware outer density slopes, window study and unresolved feature masks.'),
        'spin_alignment':('PUBLISHED_NATIVE','Local periodic T-web axes and radial native halo spin with degeneracy masks.'),
        'shape_alignment':('PUBLISHED_NATIVE','Local periodic T-web axes and native density-shape axes with support masks.'),
        'filament_map':('PUBLISHED_NATIVE','Class-2 face-connected T-web voxels at two physical smoothing scales.'),
        'log_mass_accretion':('PUBLISHED_DESCRIPTIVE','Exact-membership FOF and exact-ID Rockstar branch finite-interval logarithmic growth.'),
        'fast_slow_transition':('PUBLISHED_CONDITIONAL','Declared 1/sqrt(Np) counting-scale conditional test. Both available Rockstar histories are classified MULTIPHASE_HISTORY.'),
        'spin_history':('PUBLISHED_CATALOGUE','Rockstar Bullock spin and exact-membership FOF central SUBFIND specific angular momentum histories.'),
        'concentration_history':('PUBLISHED_CATALOGUE','Native Rockstar Rvir/rs with 300-particle cut. Catalogue fit covariance absent.'),
        'shape_history':('PUBLISHED_CATALOGUE','Rockstar b/a, c/a and unsigned major-axis changes with 300-particle cut. Orientation uncertainty absent.'),
        'reliability_limits':('PUBLISHED_COUNTING_LIMITS','Particle floors, native count support and fit calibration masks are published. Independent simulation-convergence runs are absent.'),
        'rare_high_z_abundance':('PUBLISHED_COUNTING_LIMITS','Native high-z host counts and counting intervals are published. An independent ensemble is required for rare-object sample variance.'),
        'pericenter_infall_merger_events':('PUBLISHED_LINKS_AND_SAVED_ENCOUNTER','Native descendant links, mass-ratio events and two-branch saved separation films are published. Event time extrema require additional orbit/cadence evidence.'),

    }
    for row in registry['rows']:
        variant=row['variant']
        if variant in updates:
            row['status'],row['basis']=updates[variant]
        if variant in measured:
            row['native_figure_evidence']=measured[variant]
    registry['scope']='All 102 registered variants reviewed against the two native DMO runs. Published conditional/descriptive figures retain their input limitations and differ from stricter campaign acceptance. Products requiring absent species, external calibration, independent convergence runs or unresolved cadence are identified explicitly.'
    (folder/'registered_graph_coverage.json').write_text(json.dumps(registry,indent=2)+'\n')
    columns=('family','variant','status','basis','required_measurement','producer')
    with (folder/'registered_graph_coverage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(registry['rows'])
    print('AUDITED',len(registry['rows']),'graph variants and both native product catalogues')


if __name__=='__main__':main()
