"""Reconcile all102 registered graphs against actual callable producer routes."""
from pathlib import Path
import importlib
import inspect
import json
from .foundation import file_sha256


def audit_graph_producers(path=None):
    from core.research.display import scientific_figure_registry
    root=Path(__file__).resolve().parents[1];path=Path(path).resolve() if path else root/'configs/research/graph_producers.json'
    if path.stat().st_size>16*1024**2:raise MemoryError('Graph producer registry exceeds bounded metadata intake')
    record=json.loads(path.read_text())
    if record.get('schema')!='scientific-graph-producers-v1' or record.get('variant_count')!=102:raise ValueError('Exactly102 original graph producer entries required')
    expected={(family,name) for family,names in scientific_figure_registry()['families'].items() for name in names}
    keys=[(row['family'],row['variant']) for row in record['entries']]
    if len(keys)!=102 or len(set(keys))!=102 or set(keys)!=expected:raise ValueError('Graph producers omit, repeat or replace required registered variants')
    sources={str(path):file_sha256(path),str(Path(__file__).resolve()):file_sha256(__file__)};results=[]
    for row in record['entries']:
        for field in ['input_contract','required_measurement','source_contract','units','uncertainty_support','recipe_contract']:
            if not isinstance(row.get(field),str) or not row[field].strip():raise ValueError('Explicit graph '+field+' required')
        if set(row['artifacts'])!={'PNG','SVG','PDF','physical_arrays.npz','figure.json','gallery.json'}:raise ValueError('Complete three-format/numerical graph output contract required')
        if not row['producer_route']:raise ValueError('Graph cannot hide a missing producer behind an empty renderer')
        resolved=[]
        for node in [*row['producer_route'],row['renderer']]:
            if set(node)!={'module','function'}:raise ValueError('Exact callable graph producer module/function required')
            # This catalogue contains only original first-party import names.
            if not node['module'].startswith('core.'):raise ValueError('Graph producer must use a first-party source module')
            module=importlib.import_module(node['module']);function=getattr(module,node['function'],None)
            if not callable(function):raise ValueError('Graph producer callable is absent: '+node['module']+'.'+node['function'])
            source=Path(inspect.getsourcefile(function)).resolve()
            if not source.is_relative_to(root):raise ValueError('Graph producer resolves outside packaged source')
            sources[str(source)]=file_sha256(source)
            resolved.append({**node,'source':str(source),'line':inspect.getsourcelines(function)[1],'signature':str(inspect.signature(function))})
        results.append({'family':row['family'],'variant':row['variant'],'status':'CALLABLE_PRODUCER_CONNECTED','route':resolved})
    return {'schema':'scientific-graph-producer-audit-v1','status':'PASS','variant_count':102,'entries':results,'source_hashes':sources,
        'runtime_measurements':'NOT_IMPLIED','science_acceptance':'NOT_ACCEPTED',
        'scope':'Exactly102 callable measurement/export routes and declared source/units/support/artifact contracts. This check imports/inspects code; it does not execute producers or accept science.'}
