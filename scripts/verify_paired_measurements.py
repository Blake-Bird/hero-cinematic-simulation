#!/usr/bin/env python3
"""Verify every cached epoch against its source bytes and public assets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json,file_sha256
from core.paired_atlas import epoch_name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas',type=Path)
    root=parser.parse_args().atlas.resolve()
    measured_sources={}
    models={}
    for model,section in [('EDE','02_EDE'),('LCDM','03_LCDM')]:
        base=root/'06_Tables_and_Data/Measurements'/model
        manifest=json.loads((base/'measurement_manifest.json').read_text())
        epochs=manifest['epochs']
        if len(epochs)!=86 or [r['summary']['snapshot'] for r in epochs]!=list(range(86)):
            raise ValueError(f'{model} saved epoch inventory differs from 000–085')
        checked=[]
        for record in epochs:
            asset=base/record['numeric_asset']
            if file_sha256(asset)!=record['numeric_sha256']:
                raise ValueError(f'Changed numeric epoch {asset}')
            s=record['summary']
            directory=root/section/'Saved_epochs'/epoch_name(model,s['snapshot'],s['redshift'])
            label=directory.name
            required=[directory/'index.html',directory/(label+'_Interactive_3D.html')]
            required.extend(directory/(label+'_Dark_matter_'+axis+'.png') for axis in ('XY','XZ','YZ'))
            if any(not p.is_file() or not p.stat().st_size for p in required):
                raise ValueError(f'Incomplete public saved epoch {directory}')
            for source in record['source_files']:
                path=Path(source['path'])
                if path in measured_sources:
                    actual=measured_sources[path]
                else:
                    actual=file_sha256(path)
                    measured_sources[path]=actual
                if actual!=source['sha256']:
                    raise ValueError(f'Changed native source {path}')
            checked.append({'snapshot':s['snapshot'],'redshift':s['redshift'],'public_assets':len(required),
                            'numeric_sha256':record['numeric_sha256']})
        workspace=root/'Source/Native_workspaces'/model
        bindings=json.loads((workspace/'manifest.json').read_text())['analysis_relocation']
        original=Path(json.loads((workspace/'relocation.json').read_text())['original_run'])
        for filename,key in [('manifest.json','original_manifest_sha256'),('config.yaml','original_config_sha256')]:
            if file_sha256(original/filename)!=bindings[key]:
                raise ValueError(f'Original retained {model} {filename} changed')
        models[model]={'status':'PASS','epochs':checked,'original_manifest_and_config':'UNCHANGED'}
        print('VERIFIED_EPOCHS',model,len(checked),flush=True)
    atomic_json(root/'07_Status_and_Provenance/all_epoch_source_audit.json',
                {'models':models,'unique_native_sources_verified':len(measured_sources),'status':'PASS'})


if __name__=='__main__':main()
