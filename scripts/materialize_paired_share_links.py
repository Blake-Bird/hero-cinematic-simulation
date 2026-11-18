#!/usr/bin/env python3
"""Replace internal file symlinks by ordinary hard-linked files for portable ZIPs."""
import argparse,json,os
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('atlas',type=Path);root=parser.parse_args().atlas.resolve();rows=[]
 for path in root.rglob('*'):
  if not path.is_symlink():continue
  target=path.resolve()
  if not target.is_relative_to(root) or not target.is_file():raise ValueError(f'Nonportable dependency {path}')
  stage=path.with_name('.'+path.name+'.materializing');os.link(target,stage);os.replace(stage,path)
  rows.append({'path':str(path.relative_to(root)),'previous_target':str(target.relative_to(root)),'bytes':path.stat().st_size})
 report={'status':'PASS','definition':'Internal file symlinks replaced atomically with ordinary files using hard links on this volume. No payload change or duplicate disk allocation. ZIP tools read complete ordinary files; the shared folder no longer depends on symlink support.','files':rows}
 (root/'07_Status_and_Provenance/share_file_materialization.json').write_text(json.dumps(report,indent=2)+'\n');print('MATERIALIZED_SHARE_FILES',len(rows))
if __name__=='__main__':main()
