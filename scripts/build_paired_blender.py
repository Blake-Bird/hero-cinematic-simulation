#!/usr/bin/env python3
"""Build the original rich Blender scenes beside the paired atlas."""
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commands import build_scene, export, verified_analysis_figure_gallery
from core.blender import verify_timeline_assets
from core.foundation import atomic_json, atomic_text, file_sha256, verify_product_provenance


def resume_dm_timeline(workspace, bundle, destination):
    """Finish a sealed DM export whose figure attachment stopped before publication."""
    relative = Path('dm/blender_dm_timeline.json')
    timeline_path = destination / relative
    digest = file_sha256(timeline_path)
    if timeline_path.with_suffix('.json.sha256').read_text().strip() != digest:
        raise ValueError('Sealed particle timeline checksum mismatch')
    timeline = json.loads(timeline_path.read_text())
    contract = timeline['data_contract']
    if (timeline['species'], timeline['render_quality'], timeline['frames_per_snapshot'],
            contract['selection']['max_points']) != ('dm', 'high', 4, 80000):
        raise ValueError('Staged timeline selection differs from the requested scene')
    verify_product_provenance(workspace, contract)
    verify_timeline_assets(timeline_path)
    entries = verified_analysis_figure_gallery(workspace, bundle, destination)
    manifest = destination / 'blender_universe.json'
    h = float(yaml.safe_load((workspace / 'config.yaml').read_text())['cosmology']['H0']) / 100
    atomic_json(manifest, {'schema_version': 1, 'run_dir': str(workspace),
        'species_manifests': {'dm': relative.as_posix()}, 'species_manifest_sha256': {'dm': digest},
        'quality': 'high', 'frames_per_snapshot': 4, 'unavailable_species': {},
        'hubble_parameter': h, 'science_acceptance': 'NOT_ACCEPTED',
        'figure_gallery': entries, 'figure_gallery_status': {'selected_analysis_directory': str(bundle)},
        'scene_note': 'Saved particle identities joined with periodic display interpolation. Interpolated frames are display estimates.',
        'resume_note': 'Published after verifying the sealed timeline and freshly sealed selected-analysis figure gallery.'})
    atomic_text(manifest.with_suffix('.json.sha256'), file_sha256(manifest) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    parser.add_argument('--destination', type=Path, help='Resume an export at this location')
    parser.add_argument('--resume-dm', action='store_true', help='Verify and reuse the existing sealed DM timeline')
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    workspace = root / 'Source/Native_workspaces' / model
    bundles = sorted(workspace.glob('results/analysis_products_*/analysis_provenance.json'))
    if not bundles:
        parser.error('A detailed figure bundle is required')
    destination = args.destination or root / '05_Blender_and_3D' / model / f'{model}_z_199_to_000_all_saved_epochs_interactive'
    manifest = destination / 'blender_universe.json'
    if not manifest.exists() and args.resume_dm:
        manifest = resume_dm_timeline(workspace, bundles[-1].parent, destination)
    elif not manifest.exists():
        manifest = export(workspace, 80000, 'high', 4, output_directory=destination,
                          analysis_directory=bundles[-1].parent)
    print(f'EXPORTED {manifest}', flush=True)
    build_scene(manifest, preview=True, render=False)
    print(f'BUILT {manifest}', flush=True)


if __name__ == '__main__':
    main()
