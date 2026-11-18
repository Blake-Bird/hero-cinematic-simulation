#!/usr/bin/env python3
"""Retain the first-party analysis and viewing code with its checksums."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    repo=Path(__file__).resolve().parents[1]
    output=root/'Source/Rebuild_code'
    files=list(repo.glob('*.py'))
    files.extend(p for p in (repo/'core').rglob('*') if p.is_file()
                 and p.suffix in ('.py','.json') and 'software' not in p.relative_to(repo/'core').parts
                 and '__pycache__' not in p.parts)
    files.extend(p for p in (repo/'scripts').glob('*') if p.suffix in ('.py','.mjs','.sh'))
    files.extend(p for p in (repo/'configs').rglob('*') if p.is_file() and p.suffix in ('.json','.yaml','.sh'))
    files.extend(repo/'tests'/name for name in ('test_empty_catalogue_shard.py','test_central_point_potential.py',
        'test_halo_collision_movie.py','test_merger_tree_visual.py','test_merger_tree_map.py',
        'test_select_merger_tree.py','test_blender_mac_memory.py','test_blender_portable_scene.py'))
    rows=[]
    for path in sorted(set(files)):
        relative=path.relative_to(repo)
        dest=output/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        before=file_sha256(path)
        shutil.copy2(path,dest)
        if file_sha256(dest)!=before:
            raise ValueError(f'Code snapshot changed during copy {path}')
        rows.append({'path':relative.as_posix(),'sha256':before,'size_bytes':path.stat().st_size})
    atomic_json(output/'source_manifest.json',{'scope':'First-party analysis, export, navigation and checking source. Native engines and installed Python/Blender environments remain external dependencies.',
        'repository':str(repo),'files':rows})
    (output/'README.md').write_text('''# Rebuilding this collection

This source snapshot records the first-party code used to complete the N128 EDE / LCDM analysis collection. The original workspace contains the same modified files. source_manifest.json hashes every retained code and configuration file.

The analysis expects the retained original N128 simulation folders, native input records, catalogue files and Python dependencies. Blender and ParaView are installed separately. The two native parameter sets remain recorded in their original run folders.

Use the original workspace Python environment for regeneration. The paired scripts accept the analysis collection path as their first argument.

- rebuild_paired_dome_views.py publishes readable dome views from verified saved numeric rasters, with originals retained.
- publish_readable_halo_cutaways.py makes an explicitly rebinned sparse-particle display alongside original cutaways.
- audit_paired_visual_inventory.py scans files and builds contact sheets in Source/Quality_review.
- refresh_paired_atlas.py rebuilds navigation from published measurement and figure manifests.
- serve_results.py provides the local browser viewer and MP4 byte-range seeking.
- verify_paired_assets.py checks links, images, MP4 codecs and HTTP ranges.
- verify_paired_browser.mjs exercises the film player, tree controls, particle canvas and epoch selector in installed Chrome.
- verify_paired_measurements.py compares all epoch caches with native source hashes.
- build_paired_blender.py builds the original rich scene from the particle timeline and verified figures.
- build_paired_blender_entry.py runs inside Blender to publish the framed z=14.4 entry view.
- build_paired_flights.py, build_paired_halo_stories.py and build_paired_encounter.py produce the native visual routes.
- build_paired_host_structure.py and build_paired_web_fields.py measure halo structure and periodic environment products.

Native outputs keep their provenance. New scientific publications require fresh destinations. Read each script's help and the original input contracts before regenerating a measured bundle.

The central particle fix is in core/halos.py. The zero-row catalogue-shard fix is in core/analysis.py. The Blender initial-view correction is in core/blender.py. Source tests are included for the specific repaired regressions.
''')
    print('FROZEN_SOURCE_FILES',len(rows),flush=True)


if __name__=='__main__':main()
