#!/usr/bin/env python3
"""Check compact share payloads, relative 3D inputs and absence of stale links."""
import argparse,json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import file_sha256,atomic_json
from core.blender import verify_scene_publication

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);root=parser.parse_args().atlas.resolve();status=root/'07_Status_and_Provenance';failures=[];paths=list(root.rglob('*'));links=[str(p.relative_to(root)) for p in paths if p.is_symlink()]
 if links:failures.append({'unexpected_symlinks':links})
 unwanted=[str(p.relative_to(root)) for p in paths if p.name in ('__pycache__','.DS_Store') or p.name.endswith(('.staged.mp4','.packing.blend','.packing.h5','.blend1','.materializing'))]
 if unwanted:failures.append({'unfinished_or_temporary_files':unwanted})
 packed=json.loads((status/'lossless_xdmf_packing.json').read_text())
 if packed['status']!='PASS' or len(packed['files'])!=1376 or any(r['numeric_payload']!='BITWISE_IDENTICAL' for r in packed['files']):failures.append('Incomplete HDF5 payload verification')
 def check(row):
  path=root/row['path'];return None if path.is_file() and file_sha256(path)==row['sha256'] else row['path']
 with ThreadPoolExecutor(max_workers=4) as pool:
  bad=[p for p in pool.map(check,packed['files']) if p]
 if bad:failures.append({'packed_hdf5_hash_mismatch':bad})
 xml_items=0;scene_rows=[]
 for model in ('EDE','LCDM'):
  folder=root/'05_Blender_and_3D'/model/f'{model}_z_199_to_000_ParaView_XDMF'
  for xml in folder.glob('*.xdmf'):
   for node in ET.parse(xml).iter('DataItem'):
    if node.get('Format')!='HDF':continue
    filename=node.text.split(':',1)[0];path=(xml.parent/filename).resolve()
    if Path(filename).is_absolute() or not path.is_relative_to(root) or not path.is_file():failures.append({'xdmf_dependency':filename})
    xml_items+=1
  report=json.loads((status/f'{model}_paraview_load.json').read_text())
  if report['status']!='PASS' or len(report.get('checked_epochs',[]))!=86:failures.append(model+' ParaView fresh load incomplete')
  scene_dir=root/'05_Blender_and_3D'/model/f'{model}_z_199_to_000_all_saved_epochs_interactive'
  for scene in (scene_dir/'blender_universe.blend',scene_dir/f'{model}_Explore_z_014p400_with_redshift_timeline.blend'):
   verify_scene_publication(scene);scene_rows.append(str(scene.relative_to(root)))
  for kind in ('rich','entry'):
   report=json.loads((status/f'{model}_{kind}_compressed_scene_load.json').read_text())
   if report['status']!='PASS' or len(report['epochs'])!=86 or not report['halo_inspector_registered']:failures.append(model+' '+kind+' compressed scene load failed')
 regular=[p for p in paths if p.is_file() and not p.is_symlink()];unique={}
 for p in regular:
  s=p.stat();unique[(s.st_dev,s.st_ino)]=s.st_size
 cleanup=json.loads((status/'storage_cleanup.json').read_text());before=cleanup['package_before_bytes'];logical=sum(p.stat().st_size for p in regular)
 result={'status':'PASS' if not failures else 'FAIL','failures':failures,'before_bytes':before,'after_regular_file_bytes':logical,'after_unique_payload_bytes':sum(unique.values()),'reduction_bytes':before-logical,'reduction_percent':100*(before-logical)/before,'symbolic_links':len(links),'regular_files':len(regular),'relative_xdmf_data_references':xml_items,'lossless_hdf5_files_verified':len(packed['files']),'fresh_loaded_scene_files':scene_rows,'paraview_saved_epochs_checked_per_model':86,'share_guide':'../SHARING.md','storage_note':'Full-particle ParaView inputs, final scientific arrays, all 42 films and current interactive Blender scenes are retained. Prior generations and per-frame caches are removed. Ordinary files are used throughout; local hard links save disk blocks and are readable as regular files by ZIP tools.'}
 atomic_json(status/'share_ready_audit.json',result);print(json.dumps({k:result[k] for k in ('status','before_bytes','after_regular_file_bytes','reduction_percent','symbolic_links','lossless_hdf5_files_verified')},indent=2))
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
