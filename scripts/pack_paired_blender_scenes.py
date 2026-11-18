#!/usr/bin/env python3
"""Compress current scene files and fresh-load each before replacing it."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json,file_sha256

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);args=parser.parse_args();root=args.atlas.resolve();repo=Path(__file__).resolve().parents[1];status=root/'07_Status_and_Provenance';rows=[]
 for model in ('EDE','LCDM'):
  folder=root/'05_Blender_and_3D'/model/f'{model}_z_199_to_000_all_saved_epochs_interactive'
  for path in (folder/'blender_universe.blend',folder/f'{model}_Explore_z_014p400_with_redshift_timeline.blend'):
   stage=path.with_name('.'+path.stem+'.packing.blend');stem=model+('_entry' if 'Explore' in path.name else '_rich');saved=status/(stem+'_compressed_scene_save.json');loaded=status/(stem+'_compressed_scene_load.json');log=root/'Source/Storage_cleanup'/(stem+'_compression.log');log.parent.mkdir(exist_ok=True)
   before=file_sha256(path);size=path.stat().st_size
   with log.open('w') as stream:
    for script,arguments in (('compress_blender_share_scene.py',[str(stage),str(saved)]),('inspect_paired_blender_scene.py',[str(loaded)])):
     scene=path if script.startswith('compress') else stage
     subprocess.run(['/Applications/Blender.app/Contents/MacOS/Blender','--background',str(scene),'--enable-autoexec','--python',str(repo/'scripts'/script),'--',*arguments],stdout=stream,stderr=subprocess.STDOUT,check=True)
   s=json.loads(saved.read_text());j=json.loads(loaded.read_text())
   if j['status']!='PASS' or j['objects']!=s['before']['objects'] or j['range']!=s['before']['range'] or j['initial_saved_frame']!=s['before']['frame'] or len(j['epochs'])!=86:raise ValueError('Compressed scene state changed')
   if stage.stat().st_size>=size:stage.unlink();print('ALREADY_COMPACT',path,flush=True);continue
   stage.replace(path);j['scene']=str(path);atomic_json(loaded,j)
   row={'file':str(path.relative_to(root)),'before_sha256':before,'sha256':file_sha256(path),'before_bytes':size,'bytes':path.stat().st_size,'fresh_load_report':str(loaded.relative_to(root))};rows.append(row)
   print('BLENDER_PACKED',stem,'saved MiB',(size-row['bytes'])/2**20,flush=True)
   atomic_json(status/'lossless_blender_packing.json',{'status':'PASS','scenes':rows})
 audit=json.loads((status/'scene_publication_audit.json').read_text())
 for row in audit['scenes']:row['sha256']=file_sha256(root/row['file'])
 atomic_json(status/'scene_publication_audit.json',audit)
if __name__=='__main__':main()
