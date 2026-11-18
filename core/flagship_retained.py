"""Bind postprocessing and rendering to original completed workflow receipts."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def retained_operation_inputs(parameters):
    from .workflow import verify_completed_receipt,positive_number
    required={'operation','source_workflow','memory_gib','maximum_stages'}
    if set(parameters)!=required or parameters['operation'] not in {'postprocess','render'}:raise ValueError('Complete retained flagship operation required')
    memory=positive_number(parameters['memory_gib'],'retained operation memory');maximum=positive_number(parameters['maximum_stages'],'retained stage limit',True)
    reference=parameters['source_workflow']
    if set(reference)!={'path','sha256'}:raise ValueError('Actual retained workflow path/hash required')
    path=Path(reference['path']).resolve()
    if path.name!='workflow.json' or file_sha256(path)!=reference['sha256']:raise ValueError('Retained workflow changed')
    if path.stat().st_size*32>memory*1024**3/4:raise MemoryError('Retained workflow metadata exceeds allocation')
    state=json.loads(path.read_text())
    if state.get('schema')!='research-workflow-receipt-v1' or state.get('execution_status')!='COMPLETE':raise ValueError('Retained operation requires a completed original workflow')
    stages=state['stages']
    if not stages or len(stages)>maximum:raise MemoryError('Retained workflow stage census exceeds allocation')
    completed={};pins={str(path):reference['sha256']};preparations=[]
    for sid,record in stages.items():
        root=Path(record['generation']).resolve()
        if not root.is_relative_to(path.parent/'generations'):raise ValueError('Retained stage generation escapes its source workflow')
        receipt_path=root/'stage-receipt.json'
        if file_sha256(receipt_path)!=record['receipt_sha256']:raise ValueError('Retained stage receipt differs from workflow record')
        receipt=verify_completed_receipt(receipt_path)
        if receipt['stage_id']!=sid or receipt['fingerprint']!=record['fingerprint'] or Path(receipt['generation']).resolve()!=root:raise ValueError('Retained stage identity/fingerprint changed')
        completed[sid]=receipt
        _merge(pins,{str(receipt_path):record['receipt_sha256'],str(receipt_path.with_suffix('.sha256')):file_sha256(receipt_path.with_suffix('.sha256')),
            **receipt['source_hashes'],**receipt['input_hashes'],**{str(root/name):sha for name,sha in receipt['files_sha256'].items()}})
        if receipt['kind']=='flagship_runtime_measurements':
            source=root/'measurement-preparation.json'
            if receipt['files_sha256'].get('measurement-preparation.json')!=file_sha256(source):raise ValueError('Retained measurement preparation lacks original ownership')
            if source.stat().st_size*32>memory*1024**3/4:raise MemoryError('Retained measurement preparation exceeds allocation')
            preparations.append(json.loads(source.read_text()))
    if len(preparations)!=1:raise ValueError('One original connected measurement preparation required for retained operation')
    preparation=preparations[0]
    if preparation.get('schema')!='flagship-runtime-measurements-prepared-v1' or preparation.get('execution_status')!='RECIPES_PREPARED':raise ValueError('Original connected measurement preparation required')
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Retained original source or artifact changed')
    return preparation,completed,pins


def retained_operation_runner(parameters,context):
    from .workflow import resolve_values,safe_id
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Retained operation exceeds allocation')
    preparation,completed,pins=retained_operation_inputs(parameters)
    source_context=SimpleNamespace(completed=completed,resources=context.resources,base=Path(parameters['source_workflow']['path']).parent,generation=None)
    display=copy.deepcopy(preparation['display_continuation'])
    if display is None or display.get('kind')!='flagship_runtime_display':raise ValueError('Retained connected run needs its original native display continuation')
    display['stage_id']=safe_id(context.stage_id+'_display');display['parameters']['prefix']=safe_id(context.stage_id+'_display')
    display['outputs']={safe_id(context.stage_id+'_display_preparation'):'display-preparation.json'}
    display.pop('depends_on',None)
    if parameters['operation']=='render':
        display['parameters']['cases']=resolve_values(preparation['cases'],source_context)
        child=display
    else:
        request=copy.deepcopy(preparation['measurement_request']);request['prefix']=safe_id(context.stage_id+'_measure')
        request['continuation']=display
        child={'stage_id':safe_id(context.stage_id+'_measurements'),'kind':'flagship_runtime_measurements','parameters':request,
            'outputs':{safe_id(context.stage_id+'_measurement_preparation'):'measurement-preparation.json'}}
    child['input_files']=[{'path':name,'sha256':sha} for name,sha in pins.items()]
    atomic_json(context.generation/'retained-operation.json',{'schema':'flagship-retained-operation-v1','execution_status':'RECIPES_PREPARED',
        'operation':parameters['operation'],'source_workflow':parameters['source_workflow'],'source_hashes':pins,
        'completed_source_stage_count':len(completed),'followup_stage':child['stage_id'],'science_acceptance':'NOT_ACCEPTED'})
    return {'followups':[child],'measurements':{'completed_source_stages':len(completed),'science_acceptance':'NOT_ACCEPTED'}}
