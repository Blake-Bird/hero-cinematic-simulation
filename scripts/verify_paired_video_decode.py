#!/usr/bin/env python3
"""Decode every public film through its final frame with FFmpeg."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    films=sorted(p for p in (root/'04_Movies').rglob('*.mp4')
                 if not any(part.startswith('.') for part in p.relative_to(root).parts))

    def decode(path):
        result=subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(path),
                               '-map','0:v:0','-an','-f','null','-'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        return {'film':str(path.relative_to(root)),'status':'PASS' if result.returncode==0 else 'FAIL',
                'diagnostic':result.stderr.strip()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(decode,films))
    report={'films':rows,'status':'PASS' if rows and all(r['status']=='PASS' for r in rows) else 'FAIL'}
    (root/'07_Status_and_Provenance/full_video_decode_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DECODED',len(rows),'films',report['status'])
    return report['status']!='PASS'


if __name__=='__main__':raise SystemExit(main())
