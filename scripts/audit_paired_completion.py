#!/usr/bin/env python3
"""Check the publication requirements against current files and measured audits."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    status=root/'07_Status_and_Provenance'
    def read(name):return json.loads((status/name).read_text())
    requirements=[]
    def require(identifier,condition,evidence,detail):
        requirements.append({'id':identifier,'status':'PASS' if condition else 'FAIL',
                             'evidence':evidence,'detail':detail})
    coverage=read('registered_graph_coverage.json')
    native=[read(model+'_native_product_coverage.json') for model in ('EDE','LCDM')]
    require('inventory',len(coverage['rows'])==102 and all(len(rows)==32 for rows in native)
            and not any(row['status'].startswith('OPEN') for row in coverage['rows']),
            ['registered_graph_coverage.json','EDE_native_product_coverage.json','LCDM_native_product_coverage.json'],
            'All 102 graph variants and 32 workflow products per model have explicit outcomes. Missing species, external inference inputs and independent convergence runs retain their requirements.')
    bundles=read('final_native_bundle_audit.json')
    require('native_figures',set(bundles)=={'EDE','LCDM'} and all(row['status']=='PASS' for row in bundles.values())
            and all((status/(model+'_finder_job.json')).is_file() for model in ('EDE','LCDM')),
            ['final_native_bundle_audit.json','EDE_finder_job.json','LCDM_finder_job.json'],
            'Current native priority bundles pass source and artifact verification. Comparable M200c finder products are published separately. Per-epoch particle support remains explicit.')
    epochs=read('all_epoch_source_audit.json')
    require('all_epochs',epochs['status']=='PASS' and all(len(row['epochs'])==86 for row in epochs['models'].values()),
            ['all_epoch_source_audit.json'],'86 saved epochs per run, three maps, browser particle view and numeric arrays; all recorded native source hashes verified.')
    comparisons=list((root/'01_Comparisons').glob('*/comparison_manifest.json'))
    manifests=[json.loads(p.read_text()) for p in comparisons]
    complete=len(manifests)==8 and {m['snapshot'] for m in manifests}=={21,26,30,35,41,54,69,85}
    for path,manifest in zip(comparisons,manifests):
        complete &= all((path.parent/name).is_file() for key in ('figures','tables') for name in manifest[key])
    phase=read('physical_ic_agreement.json')
    require('physical_comparison',complete and phase['phase_status']=='PASS' and phase['shared_lattice_status']=='PASS',
            ['physical_ic_agreement.json','matched_fof_hosts.json','../01_Comparisons'],
            'Eight priority epochs have figures and numerical tables. Initial phase and lattice agreement measured. Matched FOF host, assembly, shape and reference-fit comparisons are published in declared units.')
    browser=read('browser_playback_audit.json')
    decoded=read('full_video_decode_audit.json')
    films={str(p.relative_to(root)) for p in (root/'04_Movies').rglob('*.mp4')
           if not any(part.startswith('.') for part in p.relative_to(root).parts)}
    played={'04_Movies/'+row['film'] for row in browser['films']}
    fully_decoded={row['film'] for row in decoded['films'] if row['status']=='PASS'}
    require('movies',len(films)==42 and films==played==fully_decoded and not browser['failures'],
            ['browser_playback_audit.json','full_video_decode_audit.json'],
            'All 42 public films fully decode and load/seek in Chrome. The library includes both models, saved Blender timelines, flights, paired density, labelled ancestry, exact particle origins, void, encounter and animated frozen-field velocity-flow films.')
    fit_coverage=read('highlighted_fitting_function_coverage.json')
    require('highlighted_reference_fits', all(len(fit_coverage[model])==8 for model in ('EDE','LCDM'))
            and all(len(list((root/section/'Fitting_function_comparisons').glob('*/*.png')))==80 for section in ('02_EDE','03_LCDM'))
            and len([f for f in films if 'Animated_peculiar_velocity_flow' in f])==6,
            ['highlighted_fitting_function_coverage.json','../01_Comparisons/Fitting_function_comparisons/index.html'],
            'All eight priority epochs have paired physical-unit fitting overlays and ten native diagnostic panels per model. Six flow films cover both models and paired views at z=14.4 and z=0; empty halo selections retain counting upper limits.')
    scenes=read('scene_publication_audit.json')
    loaded=[read(model+'_framed_Blender_load.json') for model in ('EDE','LCDM')]
    paraview=[read(model+'_paraview_load.json') for model in ('EDE','LCDM')]
    require('blender',scenes['status']=='PASS' and len(scenes['scenes'])==4
            and all(r['status']=='PASS' and r['box_corners_inside_camera'] and len(r['epochs'])==86
                    and len(r['markers'])==86 and r['halo_inspector_registered'] for r in loaded)
            and all(r['status']=='PASS' and len(r['time_steps'])==86 for r in paraview),
            ['scene_publication_audit.json','EDE_framed_Blender_load.json','LCDM_framed_Blender_load.json',
             'EDE_paraview_load.json','LCDM_paraview_load.json','Blender_halo_inspector_GUI.png'],
            'Both rich scenes and framed entry files pass publication checks. Both entry scenes open at z=14.4, retain 86 epoch markers, fit the box in the camera and register the inspector. ParaView loads all time steps.')
    require('file_organization',all((root/name).is_dir() for name in
            ('01_Comparisons','02_EDE','03_LCDM','04_Movies','05_Blender_and_3D','06_Tables_and_Data','07_Status_and_Provenance'))
            and (root/'Open_atlas.command').is_file() and (root/'Source/Rebuild_code/source_manifest.json').is_file(),
            ['../README.md','../Source/Rebuild_code/source_manifest.json'],
            'Separate collection, model/redshift/epoch filenames, independent movie and 3D sections, numerical tables, original source records and retained rebuild code.')
    assets=read('asset_audit.json')
    require('html',not assets['failures'] and not browser['failures']
            and len(browser['pages'])>=13 and assets['html_pages']>=400,
            ['asset_audit.json','browser_playback_audit.json','atlas_desktop.png','atlas_mobile.png'],
            'All public HTML links and linked image payloads pass. Browser checks cover video controls, seeking, both tree controls, particle zoom, priority/all-epoch selection, search and mobile overflow.')
    visual=read('final_visual_inventory.json')
    domes=[json.loads(p.read_text()) for p in (root/'04_Movies').glob('*/*Planetarium*/film_source.json')]
    require('visual_polish',not visual['broken_symlinks'] and not visual['stray_files'] and not visual['empty_files']
            and not visual['visual_review']['unresolved_visual_blockers'] and len(domes)==4
            and all(d.get('display_generation')=='verified-raster-browser-v2' and len(d['source_frames'])==86 for d in domes)
            and all('slice_' not in (root/section/'Spatial_explorers/index.html').read_text() for section in ('02_EDE','03_LCDM')),
            ['final_visual_inventory.json','../Source/Quality_review/final_dome_review.jpg'],
            'Sampled visual review covers paired figures, native panels, all 42 films and desktop/mobile layouts. Dark dome views were rebuilt from verified rasters, sparse cutaways have explicit coarse display cells, duplicate slice cards and Finder metadata were removed. No broken symlinks or empty visible products remain.')
    share=read('share_ready_audit.json')
    share_tests=read('share_cleanup_tests.json')
    require('share_ready',share['status']=='PASS' and share['symbolic_links']==0
            and share['lossless_hdf5_files_verified']==1376 and len(share['fresh_loaded_scene_files'])==4
            and (root/'SHARING.md').is_file() and share_tests['status']=='PASS' and share_tests['passed']==3,
            ['share_ready_audit.json','storage_cleanup.json','lossless_xdmf_packing.json',
             'lossless_blender_packing.json','scene_share_integrity.json','share_cleanup_tests.json','../SHARING.md'],
            'Obsolete generations and rendering caches removed. All 1,376 compressed HDF5 inputs retain bitwise-identical numeric data and both 86-epoch timelines reload. Four compressed Blender scenes reload and have direct payload integrity receipts. Ordinary files replace symlinks for portable sharing.')
    regression=read('regression_tests.json')
    require('verification',regression['status']=='PASS' and regression['passed']==12
            and all(row['status']=='PASS' for row in requirements),
            ['regression_tests.json','final_native_bundle_audit.json','all_epoch_source_audit.json'],
            'Specific catalogue, central-particle, memory and tree regressions pass. Independent artifact, browser, decoder and native-source checks cover the published collection.')
    result={'status':'PASS' if all(r['status']=='PASS' for r in requirements) else 'FAIL',
            'requirements':requirements,
            'scientific_limits':['Dark matter only. No gas, stars, black holes or calibrated JWST light predictions.',
                'No 300-particle FOF hosts at z=14.4 in either run. Lower-count ancestry and locator plots are exploratory.',
                'Potential reconstruction exceeds heldout tidal budgets. Potential-shape values remain masked.',
                'Conditional accretion fits and catalogue shape/concentration histories retain their stated uncertainty limits.',
                'Independent simulation convergence, ensemble covariance and externally calibrated inference require additional inputs.']}
    (status/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({r['id']:r['status'] for r in requirements},indent=2))
    return result['status']!='PASS'


if __name__=='__main__':raise SystemExit(main())
