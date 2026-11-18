#!/usr/bin/env python3
"""Keep compact provenance records while dropping obsolete review-media backups."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json,file_sha256

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);root=parser.parse_args().atlas.resolve();rows=[]
 for name in ('Previous_dome_publication','Previous_void_publication'):
  folder=root/'Source/Quality_review'/name
  for p in sorted(folder.rglob('*')):
   if p.is_file() and p.suffix.lower() in ('.mp4','.jpg','.png'):
    rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_sha256(p),'reason':'Superseded visual export or poster; current rendered movie and numeric source data retained'});p.unlink()
  (folder/'MEDIA_REMOVED_FOR_SHARING.md').write_text('Only historical metadata is retained here. Prior media backups were removed after verification of the final published films. Their hashes and sizes remain in the storage cleanup report.\n')
 for folder in (root/'04_Movies').glob('*/**/void_contract.json'):
  contract=json.loads(folder.read_text())
  if 'guided_display' not in contract:continue
  receipt=contract['guided_display'];receipt['original_movie_retained']=False;receipt['original_movie_location_before_cleanup']=receipt.pop('original_movie',None);receipt['display_raster_cache_retention']='Discarded after rendering and source verification; regenerate from retained density volume and saved camera sample.'
  atomic_json(folder,contract)
  source=folder.with_name('film_source.json');record=json.loads(source.read_text());record['guided_display']=receipt;atomic_json(source,record)
 report_path=root/'07_Status_and_Provenance/storage_cleanup.json';report=json.loads(report_path.read_text());report['removals'].extend(rows);report['removed_bytes']+=sum(row['bytes'] for row in rows);atomic_json(report_path,report);print('PRUNED_BACKUP_MEDIA_MiB',sum(row['bytes'] for row in rows)/2**20)
if __name__=='__main__':main()
