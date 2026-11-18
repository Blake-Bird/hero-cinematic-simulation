"""Read retained statuses and require the requested rendering/fit outputs."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--simulation-only',action='store_true');p.add_argument('--quiet',action='store_true');p.add_argument('--allow-missing-blender',action='store_true');a=p.parse_args()
r=a.run.resolve();m=json.loads((r/'manifest.json').read_text())
if m.get('status')!='SIMULATION_COMPLETE':
 print('Simulation is checkpointed/incomplete; resubmit with RESUME=1 and the same allocation.',file=sys.stderr);raise SystemExit(75)
if a.simulation_only:
 if not a.quiet:print('SIMULATION_COMPLETE',r)
 raise SystemExit(0)
workflows=sorted((r/'results').glob('workflow_*.json'))
if not workflows:raise SystemExit('Missing workflow record')
w=json.loads(workflows[-1].read_text());art=w.get('artifacts',{});errors=[]
if w.get('failure'):errors.append(w['failure'])
import os
allow_no_blender = a.allow_missing_blender or os.environ.get('ALLOW_MISSING_BLENDER') == '1' or os.environ.get('BLENDER_OPTIONAL') == '1'
for k, v in w.get('visual_failures', {}).items():
    if k in ('blender', 'blender_scene', 'blender_movie') and allow_no_blender:
        continue
    if k == 'halo_movie':
        continue
    errors.append(v)
req_keys = ['analysis', 'movie', 'planetarium', 'xdmf']
if 'halo_movie' not in w.get('unavailable', {}) and not w.get('visual_failures', {}).get('halo_movie'):
 req_keys.append('halo_movie')
if not allow_no_blender:
 req_keys.extend(['blender_scene', 'blender_movie'])
for key in req_keys:
 if key not in art:errors.append('Missing requested artifact: '+key)
analysis=Path(art.get('analysis',r/'missing'))
summary_path=analysis/'analysis_products_summary.json'
summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
late=min(summary.get('snapshots',{}).values(),key=lambda x:abs(x.get('redshift',1e9)),default={})
for product in ('hmf','pk','kinematics','subhalos'):
 if late.get('products',{}).get(product,{}).get('status')!='COMPLETE':errors.append('Latest epoch lacks '+product)
fit_reports=[json.loads(f.read_text()) for f in (analysis/'hmf_fitting_functions').glob('comparison_*.json')]
if not any(x.get('status')=='DIAGNOSTIC_ONLY' for x in fit_reports):errors.append('No fitting-function comparison was produced')
if w.get('navigation_error'):errors.append(w['navigation_error'])
check_files = ['movie']
if 'halo_movie' in art:
 check_files.append('halo_movie')
if not allow_no_blender:
 check_files.append('blender_scene')
for key in check_files:
 path=Path(art.get(key,r/'missing'))
 if not path.is_file() or path.stat().st_size==0:errors.append('Missing/empty artifact file: '+key)
if not allow_no_blender:
 blend=Path(art.get('blender_scene',r/'missing'))
 blender_movie=blend.with_name('blender_universe_movie.mp4')
 if not blender_movie.is_file() or blender_movie.stat().st_size==0:errors.append('Missing/empty Blender MP4')
report={'status':'FAIL' if errors else 'REQUIRED_EXPORTS_PRESENT','workflow_status':w.get('status'),'errors':errors,'sampling_or_input_omissions':w.get('unavailable',{}),'note':'Omissions remain visible; this gate does not establish convergence or certify every optional recipe.'}
path=r/'results/cluster_output_audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if errors:raise SystemExit(1)
