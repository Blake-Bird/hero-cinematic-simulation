#!/usr/bin/env python3
"""Regenerate readable flights and linked halo stories; retain prior generations."""
from pathlib import Path
from datetime import datetime, timezone
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.cosmic_flythrough import make_cosmic_flythrough
from core.filament_flythrough import make_filament_flythrough
from core.analysis import halo_masses, group_field
from core.halo_story import export_linked_halo_story
from core.foundation import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--only',choices=['flights','tree','all'],default='all')
    args=parser.parse_args()
    root=args.run.resolve();analysis=root/'analysis';source=root/'analysis_source/visual_workspace'
    report={'schema':'readable-dmo-visual-refresh-v1','completed':{},'failures':{},'prior_generations':{}}
    status=analysis/'06_Data_and_provenance'/('Halo_story_refresh.json' if args.only=='tree' else 'Readable_visual_refresh.json')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=analysis/'06_Data_and_provenance/Previous_visual_attempts'

    def regenerate(name,out,operation):
        prior=None
        if out.exists():
            archive.mkdir(parents=True,exist_ok=True)
            prior=archive/(stamp+'_'+out.name)
            out.rename(prior);report['prior_generations'][name]=str(prior)
        try:
            result=operation()
            report['completed'][name]=str(result)
            print('COMPLETE',name,flush=True)
        except Exception as exc:
            # Keep the last working publication reachable if the fresh render fails.
            if prior is not None and prior.exists() and not out.exists():prior.rename(out)
            report['failures'][name]={'type':type(exc).__name__,'reason':str(exc)}
            print('FAIL',name,str(exc),flush=True)
        atomic_json(status,report)

    if args.only in ('flights','all'):
        for snap,z in ((21,14.4),(85,0.)):
            host=int(np.argmax(halo_masses(source,snap)))
            target=np.asarray(group_field(source,snap,['GroupPos']))[host]/1000
            for filament in (False,True):
                name=('Filament_tour' if filament else 'Halo_approach')+'_z'+str(z)
                out=analysis/'05_Movies_and_3D/Readable_flights'/name
                if filament:
                    def operation(out=out,snap=snap,host=host):
                        return make_filament_flythrough(source,out,snapshot=snap,host_row=host,
                            frames=240,fps=24,max_particles=250000,memory_gib=3.)
                else:
                    def operation(out=out,snap=snap,host=host,target=target):
                        return make_cosmic_flythrough(source,out,snapshot=snap,host_row=host,
                            start=(target+np.array([1.6,-1.4,.9])).tolist(),frames=240,width=1280,height=720,
                            fps=24,max_particles=250000,memory_gib=3.)
                regenerate(name,out,operation)
    if args.only in ('tree','all'):
        tree=sorted((root/'analysis_source/history_workspace/results').glob('analysis_products_*/merger_tree.json'))[-1]
        out=analysis/'04_Evolution/Halo_merger_story'
        regenerate('linked_tree_and_movie',out,lambda:export_linked_halo_story(tree,
            analysis/'05_Movies_and_3D/Cosmic_web_timeline_source.json',analysis,out,run_dir=root))
    return 1 if report['failures'] else 0


if __name__=='__main__':raise SystemExit(main())
