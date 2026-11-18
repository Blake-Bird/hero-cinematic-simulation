"""Relocate native-run analysis bindings without modifying transferred run inputs."""
from pathlib import Path
import json
import os
from core.foundation import file_sha256, atomic_json


def prepare(run, destination):
    run=Path(run).resolve();out=Path(destination).resolve()
    if out.exists():return out
    original=json.loads((run/'manifest.json').read_text())
    old=str(Path(original['gadget_handoff']['parameter_file']['path']).parent.parent)
    relocations=[]
    def path(value):
        if isinstance(value,str) and value.startswith(old+'/'):
            target=run/value[len(old)+1:]
            if target.exists():
                relocations.append({'recorded':value,'local':str(target)})
                return str(target)
        return value
    def walk(value):
        if isinstance(value,dict):return {path(k):walk(v) for k,v in value.items()}
        if isinstance(value,list):return [walk(v) for v in value]
        return path(value)
    out.mkdir(parents=True)
    (out/'data').symlink_to(os.path.relpath(run/'data',out),target_is_directory=True)
    (out/'config.yaml').write_bytes((run/'config.yaml').read_bytes())
    local=walk(original)
    local['analysis_relocation']={'original_manifest':str(run/'manifest.json'),'original_manifest_sha256':file_sha256(run/'manifest.json'),
        'original_config_sha256':file_sha256(run/'config.yaml'),'scope':'Existing path rebinding only; physics and native source bytes unchanged. Native IC prefix tokens remain their original recorded values.'}
    atomic_json(out/'manifest.json',local)
    input_dir=out/'input';input_dir.mkdir()
    for source in (run/'input').iterdir():
        if source.name.startswith('.'):continue
        target=input_dir/source.name
        if source.name=='halo_finding.json':atomic_json(target,walk(json.loads(source.read_text())))
        else:target.symlink_to(os.path.relpath(source,input_dir),target_is_directory=source.is_dir())
    atomic_json(out/'relocation.json',{'original_run':str(run),'original_manifest_sha256':file_sha256(run/'manifest.json'),'paths':relocations})
    return out


if __name__=='__main__':
    import sys
    print(prepare(sys.argv[1],sys.argv[2]))
