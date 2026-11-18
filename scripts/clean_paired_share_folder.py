#!/usr/bin/env python3
"""Remove explicit superseded generations and render caches from a finished atlas."""
import argparse,json,os,shutil
from pathlib import Path
from datetime import datetime,timezone

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);parser.add_argument('--apply',action='store_true');args=parser.parse_args();root=args.atlas.resolve()
 files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
 links=[p for p in root.rglob('*') if p.is_symlink()]
 protected=[p.resolve() for p in links if p.exists() and p.resolve().is_relative_to(root)]
 candidates={}
 def add(path,reason):
  if path.exists() and not path.is_symlink():candidates[path]=reason
 for folder in ('Previous_comparison_attempts','Previous_story_publications','Failed_attempts','Verification_attempts','Blender_film_frames'):
  add(root/'Source'/folder,'Superseded publication or regenerable rendering frames')
 for model in ('EDE','LCDM'):
  work=root/'Source/Native_workspaces'
  add(work/f'{model}_encounter/results/structure_maps','Intermediate density-map caches; rendered encounter and numeric publication retained')
  bundles=sorted((work/model/'results').glob('analysis_products_*'))
  for p in bundles[:-1]:add(p,'Superseded native analysis generation; current sealed bundle retained')
  for p in (root/'05_Blender_and_3D'/model).glob('*retained_interactive_universe'):add(p,'Superseded original Blender export; current rich scenes and framed entry retained')
  for p in (root/'05_Blender_and_3D'/model).glob('*full_86_epoch_interactive_universe'):add(p,'Incomplete earlier Blender staging directory')
  for p in (root/'05_Blender_and_3D'/model).glob('*all_saved_epochs_interactive/.scene_history'):add(p,'Previous Blender save generations')
  for p in (root/'05_Blender_and_3D'/model).glob('*all_saved_epochs_interactive/*_to_all_saved_epochs.blend*'):add(p,'Superseded Blender entry alias; current redshift-timeline entry retained')
 for p in root.rglob('*'):
  if p.is_dir() and (p.name=='__pycache__' or p.name.startswith(('.analysis_products_','.blender-timeline-'))):add(p,'Python cache or incomplete build stage')
 for p in files:
  if p.name in ('.DS_Store',) or p.suffix in ('.pyc','.blend1','.tmp','.part'):add(p,'Disposable system or temporary file')
  elif p.suffix=='.npy' and p.name.startswith(('camera_','frame_','particles_')) and '04_Movies' in p.parts:add(p,'Per-frame raster cache; final movie and scientific source volumes retained')
  elif p.suffix=='.jpg' and p.name.startswith(('film_','final_dome_')) and p.parent==root/'Source/Quality_review' and p.name!='final_dome_review.jpg':add(p,'Individual QA frame; reviewed contact sheets retained')
 for p in (root/'Source/Quality_review').glob('Guided_void_rasters'):add(p,'Per-frame guided-void projection cache; verified source volume retained')
 for p in (root/'Source/Quality_review').glob('Dome_numeric_display'):add(p,'Completed dome render staging area')
 # Keep one removal for each containing directory.
 ordered=[];selected=set();protected_set=set(protected)
 for p,reason in sorted(candidates.items(),key=lambda x:len(x[0].parts)):
  if any(parent in selected for parent in p.parents):continue
  if (p in protected_set if p.is_file() else any(target==p or target.is_relative_to(p) for target in protected)):raise ValueError(f'Published symlink still uses cleanup candidate {p}')
  ordered.append((p,reason));selected.add(p)
 removals=[]
 for p,reason in ordered:
  members=[p] if p.is_file() else [f for f in files if f.is_relative_to(p)]
  removals.append({'path':str(p.relative_to(root)),'reason':reason,'files':len(members),'bytes':sum(f.stat().st_size for f in members)})
 report={'schema':'finished-atlas-cleanup-v1','recorded_utc':datetime.now(timezone.utc).isoformat(),'state':'APPLIED' if args.apply else 'PLAN','package_before_bytes':sum(p.stat().st_size for p in files),'disk_free_before_bytes':shutil.disk_usage(root).free,'removals':removals,'removed_bytes':sum(r['bytes'] for r in removals),'external_rebuild_bindings':[]}
 # External source bindings are not needed to view the share copy. Keep a
 # compact provenance index instead of links that would break on another Mac.
 for p in links:
  target=p.resolve()
  if not target.is_relative_to(root):
   if not p.is_relative_to(root/'Source/Native_workspaces'):raise ValueError(f'External public dependency {p}')
   report['external_rebuild_bindings'].append({'path':str(p.relative_to(root)),'original_target':str(target),'kind':'directory' if target.is_dir() else 'file'})
 print(json.dumps({'candidates':len(ordered),'remove_gib':report['removed_bytes']/2**30,'external_bindings':len(report['external_rebuild_bindings'])},indent=2))
 status=root/'07_Status_and_Provenance';(status/'storage_cleanup_plan.json').write_text(json.dumps(report,indent=2)+'\n')
 if not args.apply:return
 for p,_ in ordered:
  if p.is_dir():shutil.rmtree(p)
  else:p.unlink()
 for row in report['external_rebuild_bindings']:
  p=root/row['path']
  if p.is_symlink():p.unlink()
 (status/'storage_cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
 (root/'Source/RAW_INPUTS.md').write_text('''# Raw inputs and removed caches

This is the rendered, portable comparison collection. The original simulation runs remain outside this folder in the research workspace. Raw-input symlinks used by build workspaces have been removed so they cannot become broken links when shared.

The complete original targets and removed-cache inventory are recorded in ../07_Status_and_Provenance/storage_cleanup.json. Reproduction of native analyses requires the original simulation runs and restoration of their workspace input bindings using prepare_paired_analysis.py. Viewing the published HTML, films, Blender scenes and ParaView timeline uses assets retained inside this folder.

Previous publication attempts, old scene saves, failed stages and per-frame raster caches were removed after final rendering and verification. Measured tables, published numeric source assets, current scene files and full-particle ParaView inputs remain.
''')
if __name__=='__main__':main()
