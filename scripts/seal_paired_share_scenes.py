#!/usr/bin/env python3
"""Seal compact final scene payloads without retaining duplicate rollback media."""
import argparse,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import file_sha256,atomic_json
from core.blender import verify_scene_publication

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);root=parser.parse_args().atlas.resolve();history=root/'Source/Storage_cleanup/scene_publication_metadata';history.mkdir(parents=True,exist_ok=True);rows=[];repo=Path(__file__).resolve().parents[1]
 for model in ('EDE','LCDM'):
  folder=root/'05_Blender_and_3D'/model/f'{model}_z_199_to_000_all_saved_epochs_interactive'
  # The only module change adds support for a final, directly hashed share
  # publication. Interactive scene behavior and packed geometry are unchanged.
  shutil.copy2(repo/'core/blender.py',folder/'blender.py')
  for path in (folder/'blender_universe.blend',folder/f'{model}_Explore_z_014p400_with_redshift_timeline.blend'):
   pointer=path.with_name(path.name+'.publication.json');old=json.loads(pointer.read_text());backup=history/(model+'_'+pointer.name)
   if not backup.exists():backup.write_bytes(pointer.read_bytes())
   aliases=old['aliases'];artifacts={name:file_sha256(folder/name) for name in aliases}
   atomic_json(pointer,{'schema':'blender-portable-scene-publication-v1','state':'COMPLETE','public_blend_name':path.name,'aliases':aliases,'artifacts':artifacts,'superseded_publication_record':str(backup.relative_to(root)),'storage_policy':'Final compressed scene and current assets retained once. Duplicate prior generations removed. Recovery requires rebuilding from retained manifests and original simulation inputs.','science_acceptance':old.get('science_acceptance','NOT_ACCEPTED')})
   verify_scene_publication(path);rows.append({'file':str(path.relative_to(root)),'sha256':file_sha256(path),'status':'PASS'})
 atomic_json(root/'07_Status_and_Provenance/scene_share_integrity.json',{'status':'PASS','scenes':rows});print('SEALED_PORTABLE_SCENES',len(rows))
if __name__=='__main__':main()
