"""Export the measured ancestry of an explicitly selected final halo."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path


def verify_tree_bundle(directory):
    """Verify either a regular analysis tree or an explicitly selected tree."""
    from core.analysis import verify_analysis_bundle
    from core.foundation import file_sha256
    root=Path(directory).expanduser().resolve();path=root/'selected_tree_contract.json'
    if not path.exists():
        return verify_analysis_bundle(root,verify_sources=True)
    digest=file_sha256(path);record=json.loads(path.read_text())
    if record.get('schema')!='selected-merger-tree-v1' or not {'merger_tree.html','merger_tree.json'} <= set(record.get('artifacts',{})):
        raise ValueError('Malformed selected merger-tree contract')
    for name,expected in record['artifacts'].items():
        if not (root/name).is_file() or file_sha256(root/name)!=expected:raise ValueError('Selected tree artifact missing or changed: '+name)
    upstream=Path(record['analysis_bundle']).resolve()
    if file_sha256(upstream/'analysis_provenance.json')!=record['analysis_provenance_sha256']:
        raise ValueError('Selected tree source analysis changed')
    result=verify_analysis_bundle(upstream,verify_sources=True)
    if Path(result['source_run']).resolve()!=Path(record['run']).resolve():raise ValueError('Selected tree source run does not match')
    tree=json.loads((root/'merger_tree.json').read_text())
    if not tree.get('nodes') or tree['nodes'][0].get('id')!=record['halo_id']:raise ValueError('Selected tree root mismatch')
    if file_sha256(path)!=digest:raise ValueError('Selected tree contract changed')
    return {**result,'selected_tree_contract_sha256':digest,'source_analysis_bundle':str(upstream)}


def export_selected_tree(run_dir,analysis_bundle,destination,*,halo_id):
    from core.analysis import verify_analysis_bundle,load_cfg,main_progenitor_histories
    from core.merger_tree_visual import export_merger_tree
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,publish_directory
    root=Path(run_dir).expanduser().resolve();analysis=Path(analysis_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh selected-tree directory')
    if not isinstance(halo_id,str) or not halo_id.isascii() or not halo_id.isdecimal():raise ValueError('Halo ID must be an exact nonnegative decimal string')
    with native_run_lock(root,operation='selected merger tree'):
        provenance=verify_analysis_bundle(analysis,verify_sources=True)
        if Path(provenance['source_run']).resolve()!=root:raise ValueError('Analysis bundle belongs to another run')
        summary=json.loads((analysis/'analysis_products_summary.json').read_text());snapshots=summary.get('snapshots')
        if not isinstance(snapshots,dict) or not snapshots:raise ValueError('Analysis has no saved snapshot metadata')
        result=main_progenitor_histories(root,load_cfg(root),snapshots,final_halo_id=int(halo_id))
        if result.get('status')!='COMPLETE' or not result.get('_merger_tree'):
            raise ValueError('Selected tree unavailable: '+str(result.get('reason','no resolved progenitors')))
        tree=result['_merger_tree']
        if tree['nodes'][0]['id']!=str(int(halo_id)):raise ValueError('Selected tree root does not match requested halo')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.selected-tree-',dir=out.parent))
        try:
            export_merger_tree(tree,stage)
            verify_analysis_bundle(analysis,verify_sources=True)
            atomic_json(stage/'selected_tree_contract.json',{'schema':'selected-merger-tree-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),'analysis_bundle':str(analysis),
                'analysis_provenance_sha256':file_sha256(analysis/'analysis_provenance.json'),'halo_id':str(int(halo_id)),
                'final_snapshot':result['final_snapshot'],'particle_floor':result['particle_floor'],'source':result['source'],
                'algorithm_sha256':file_sha256(__file__),'artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out);return out/'merger_tree.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
