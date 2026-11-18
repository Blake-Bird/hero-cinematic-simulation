"""Optional history of reported failures and the measurements used to resolve them.

A fix can affect downstream products. This module follows those recorded
dependencies and checks supplied rerun results; a closed issue is not proof
that an entire simulation or physical model is accurate. Ordinary evolution
and analysis do not create this store automatically.
"""
from __future__ import annotations


from pathlib import Path
import json
import math
import time
import uuid
from core.foundation import atomic_json, canonical_digest, file_sha256
from core.native import native_run_lock

SEVERITIES = {'critical', 'major', 'minor', 'informational'}

def _dependency_graph(graph):
    from core.workflow import safe_id
    if not isinstance(graph,dict) or not graph:
        raise ValueError('Explicit complete product dependency graph required')
    normalized = {}
    for product,dependencies in graph.items():
        safe_id(product)
        if not isinstance(dependencies,list) or len(set(dependencies)) != len(dependencies):
            raise ValueError('Product dependencies must be distinct explicit IDs')
        normalized[product] = sorted(safe_id(q) for q in dependencies)
    if any(q not in normalized for values in normalized.values() for q in values):
        raise ValueError('Dependency graph contains an unknown upstream product')
    done,active = set(),set()
    def visit(product):
        if product in active:
            raise ValueError('Repair product dependency graph contains a cycle')
        if product in done:
            return
        active.add(product)
        for dependency in normalized[product]:
            visit(dependency)
        active.remove(product)
        done.add(product)
    for product in normalized:
        visit(product)
    return normalized


def _require_graph_extension(original,current):
    if original is not None and (not set(original) <= set(current) or
            any(not set(dependencies) <= set(current[product]) for product,dependencies in original.items())):
        raise ValueError('Repair dependency graph narrows recorded product/dependency scope')


def _require_revalidation_extension(original,current):
    """A later fix may add evidence obligations, never remove recorded ones."""
    if not original:
        return
    if not current or not set(original) <= set(current):
        raise ValueError('Repair fix drops previously frozen affected-product measurement obligations')
    for product,requirement in original.items():
        replacement = current[product]
        by_identity = {(row['stage_id'],row['product']):row for row in replacement['measurements']}
        for row in requirement['measurements']:
            identity = (row['stage_id'],row['product'])
            retained = by_identity.get(identity)
            if retained is None or any(retained.get(key) != row.get(key) for key in ('product_schema','measurement_origin')):
                raise ValueError('Repair fix narrows or substitutes a frozen scientific measurement producer')
            if not {canonical_digest(q) for q in row['conditions']} <= {canonical_digest(q) for q in retained['conditions']}:
                raise ValueError('Repair fix removes or weakens frozen measured revalidation predicates')


def _current_repair_receipt(product,reference,sources,fix_hash,graph_hash,requirement=None):
    path = Path(reference['path']).resolve()
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Changed repair receipt')
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'PASS' or receipt.get('execution_status') != 'COMPLETE':
        raise ValueError('Repair receipt is not a completed passing execution')
    bindings = receipt.get('source_hashes',{})
    if receipt.get('product_id') != product or any(bindings.get(p) != value for p,value in sources.items()):
        raise ValueError('Repair receipt belongs to another fix/product')
    if receipt.get('finding_event_hash') != fix_hash or receipt.get('dependency_graph_hash') != graph_hash:
        raise ValueError('Repair receipt does not bind the current fix and full dependency graph')
    if requirement is not None:
        if receipt.get('schema') != 'research-repair-receipt-v1' or receipt.get('revalidation_requirement_hash') != canonical_digest(requirement):
            raise ValueError('Repair receipt omits or changes the frozen measured revalidation requirement')
        decisions = receipt.get('decisions',[])
        count = sum(len(row['conditions']) for row in requirement['measurements'])
        if len(decisions) != count or any(row.get('passed') is not True for row in decisions):
            raise ValueError('Repair closure requires every frozen measured condition to pass')
        expected = [(index,condition) for index,row in enumerate(requirement['measurements']) for condition in row['conditions']]
        if [(row.get('measurement_index'),row.get('condition')) for row in decisions] != expected:
            raise ValueError('Repair receipt substitutes the frozen measurement predicates')
    for source,expected in bindings.items():
        if file_sha256(source) != expected:
            raise ValueError('Repair source version changed since revalidation')
    backing = receipt.get('files_sha256',{})
    if not backing:
        raise ValueError('Repair receipt has no retained measurements')
    for payload,expected in backing.items():
        if file_sha256(payload) != expected:
            raise ValueError('Repair measurement changed')
    return {'path':str(path),'sha256':reference['sha256']}


class FindingStore:
    def __init__(self, directory):
        self.root = Path(directory).resolve()

    def _events(self, finding_id):
        from core.workflow import safe_id
        safe_id(finding_id)
        folder = (self.root/finding_id).resolve()
        if folder.parent != self.root:
            raise ValueError('Finding history escaped its store')
        paths = sorted(folder.glob('*.json'))
        events = [json.loads(p.read_text()) for p in paths]
        prior = None
        previous_kind = None
        transitions = {None:{'OPEN'},'OPEN':{'FIXED_PENDING_REVALIDATION'},
            'FIXED_PENDING_REVALIDATION':{'FIXED_PENDING_REVALIDATION','CLOSED'},
            'CLOSED':{'REOPENED'},'REOPENED':{'FIXED_PENDING_REVALIDATION'}}
        for sequence, event in enumerate(events):
            if paths[sequence].name != f'{sequence:08d}.json' or event.get('schema') != 'research-finding-event-v1' or event.get('finding_id') != finding_id:
                raise ValueError('Finding history has an invalid identity or sequence path')
            if event.get('kind') not in transitions.get(previous_kind,set()):
                raise ValueError('Finding history has an invalid state transition')
            previous_kind = event['kind']
            if event['sequence'] != sequence or event['previous_event_hash'] != prior:
                raise ValueError('Finding history chain broken')
            content = {k: v for k, v in event.items() if k != 'event_hash'}
            if canonical_digest(content) != event['event_hash']:
                raise ValueError('Finding event changed')
            prior = event['event_hash']
        return events

    def _append(self, finding_id, kind, payload):
        with native_run_lock(self.root, operation='append finding history'):
            events = self._events(finding_id)
            record = {'schema': 'research-finding-event-v1', 'finding_id': finding_id,
                'sequence': len(events), 'previous_event_hash': events[-1]['event_hash'] if events else None,
                'kind': kind, 'payload': payload,'created_unix':time.time()}
            record['event_hash'] = canonical_digest(record)
            path = self.root/finding_id/(f'{len(events):08d}.json')
            if path.exists():
                raise FileExistsError('Finding event already exists')
            atomic_json(path, record)
            return record

    def record_failure(self, artifact_hash, trigger, severity, owning_requirement, reproduction,
                       *, affected_products, source_hashes, dependency_graph=None):
        from core.workflow import safe_id
        if (severity not in SEVERITIES or not isinstance(artifact_hash,str) or len(artifact_hash) != 64
                or any(q not in '0123456789abcdef' for q in artifact_hash)
                or not isinstance(trigger,str) or not trigger.strip()
                or not isinstance(owning_requirement,str) or not owning_requirement.strip() or not reproduction):
            raise ValueError('Complete artifact/trigger/severity/requirement/reproduction required')
        if not isinstance(affected_products,list) or not affected_products or not isinstance(source_hashes,dict) or not source_hashes:
            raise ValueError('Affected products and exact source hashes required')
        for product in affected_products:
            safe_id(product)
        if len(set(affected_products)) != len(affected_products):
            raise ValueError('Finding affected products must have distinct exact identities')
        for path, expected in source_hashes.items():
            if file_sha256(path) != expected:
                raise ValueError('Finding source changed before recording')
        graph = _dependency_graph(dependency_graph) if dependency_graph is not None else None
        if graph is not None and not set(affected_products) <= set(graph):
            raise ValueError('Finding products are outside the recorded dependency graph')
        identity = 'finding_'+uuid.uuid4().hex
        self._append(identity, 'OPEN', {'artifact_hash': artifact_hash, 'trigger': trigger, 'severity': severity,
            'owning_requirement': owning_requirement, 'reproduction': reproduction,
            'affected_products': sorted(set(affected_products)), 'source_hashes': source_hashes,
            'dependency_graph':graph})
        return identity

    def state(self, finding_id):
        events = self._events(finding_id)
        if not events or events[0]['kind'] != 'OPEN':
            raise ValueError('Finding has no origin')
        return {'finding_id': finding_id, **events[0]['payload'], 'status': events[-1]['kind'],
                'events': events, 'latest_event_hash': events[-1]['event_hash']}

    def list(self):
        if not self.root.exists():
            return []
        return [self.state(p.name) for p in sorted(self.root.iterdir()) if p.is_dir()]

    def affected_products(self, finding_id, dependency_graph):
        state = self.state(finding_id)
        dependency_graph = _dependency_graph(dependency_graph)
        _require_graph_extension(state.get('dependency_graph'),dependency_graph)
        fixed = next((q for q in reversed(state['events']) if q['kind'] == 'FIXED_PENDING_REVALIDATION'),None)
        if fixed:
            _require_graph_extension(fixed['payload'].get('dependency_graph'),dependency_graph)
        affected = set(state['affected_products'])
        if not affected <= set(dependency_graph):
            raise ValueError('Finding products absent from dependency graph')
        while True:
            additions = {p for p, dependencies in dependency_graph.items() if set(dependencies) & affected}
            expanded = affected | additions
            if expanded == affected:
                return sorted(expanded)
            affected = expanded

    def record_fix(self, finding_id, *, source_hashes, explanation, dependency_graph=None,
                   revalidation_requirements=None):
        with native_run_lock(self.root,operation='update finding state'):
            if not explanation or not source_hashes:
                raise ValueError('Fix explanation and current source hashes required')
            for path, expected in source_hashes.items():
                if file_sha256(path) != expected:
                    raise ValueError('Fix source changed')
            old = self.state(finding_id)
            if old['status'] == 'CLOSED':
                raise ValueError('Reopen finding before recording another fix')
            original = old.get('dependency_graph')
            prior = next((q for q in reversed(old['events']) if q['kind'] == 'FIXED_PENDING_REVALIDATION'),None)
            if prior and prior['payload'].get('dependency_graph') is not None:
                original = prior['payload']['dependency_graph']
            graph = _dependency_graph(dependency_graph) if dependency_graph is not None else original
            _require_graph_extension(original,graph or {})
            prior_requirements = prior['payload'].get('revalidation_requirements') if prior else None
            if revalidation_requirements is None and prior_requirements is not None:
                revalidation_requirements = prior_requirements
            if revalidation_requirements is not None:
                if graph is None or set(revalidation_requirements) != set(self.affected_products(finding_id,graph)):
                    raise ValueError('Freeze measured revalidation requirements for the complete affected product closure')
                for requirement in revalidation_requirements.values():
                    validate_revalidation_requirement(requirement)
                _require_revalidation_extension(prior_requirements,revalidation_requirements)
            return self._append(finding_id, 'FIXED_PENDING_REVALIDATION',
                {'source_hashes':source_hashes,'explanation':explanation,'dependency_graph':graph,
                 'dependency_graph_hash':canonical_digest(graph) if graph is not None else None,
                 'revalidation_requirements':revalidation_requirements})

    def close_only_with_current_receipts(self, finding_id, dependency_graph, receipts):
        with native_run_lock(self.root, operation='close finding with current evidence'):
            state = self.state(finding_id)
            if state['status'] != 'FIXED_PENDING_REVALIDATION':
                raise ValueError('Current fix must be recorded before closure')
            sources = state['events'][-1]['payload']['source_hashes']
            affected = self.affected_products(finding_id, dependency_graph)
            if set(receipts) != set(affected):
                raise ValueError('Repair revalidation must cover the complete affected dependency closure')
            fixed = state['events'][-1]
            graph = _dependency_graph(dependency_graph)
            bound = fixed['payload'].get('dependency_graph')
            if bound is None or graph != bound:
                raise ValueError('Record the complete dependency graph with the current fix before closure')
            graph_hash = canonical_digest(graph)
            retained = {product:_current_repair_receipt(product,receipts[product],sources,
                fixed['event_hash'],graph_hash,
                (fixed['payload'].get('revalidation_requirements') or {}).get(product)) for product in affected}
            return self._append(finding_id,'CLOSED',{'receipts':retained,
                'affected_products':affected,'fix_event_hash':fixed['event_hash'],
                'dependency_graph':graph,'dependency_graph_hash':graph_hash})

    def closure_current(self,finding_id):
        state = self.state(finding_id)
        if state['status'] != 'CLOSED':
            return {'current':False,'reason':'finding is open'}
        try:
            closure = state['events'][-1]['payload']
            fixed = next(q for q in state['events'] if q['event_hash'] == closure['fix_event_hash'])
            graph = _dependency_graph(closure['dependency_graph'])
            if graph != fixed['payload']['dependency_graph'] or canonical_digest(graph) != closure['dependency_graph_hash']:
                raise ValueError('Closure dependency scope differs from recorded fix')
            affected = self.affected_products(finding_id,graph)
            if set(closure['receipts']) != set(affected):
                raise ValueError('Closure omits affected downstream products')
            for product in affected:
                _current_repair_receipt(product,closure['receipts'][product],fixed['payload']['source_hashes'],
                    fixed['event_hash'],closure['dependency_graph_hash'],
                    (fixed['payload'].get('revalidation_requirements') or {}).get(product))
        except (OSError,ValueError,KeyError,StopIteration) as error:
            return {'current':False,'reason':str(error)}
        return {'current':True,'fix_event_hash':fixed['event_hash']}

    def reopen(self, finding_id, reason):
        with native_run_lock(self.root,operation='update finding state'):
            if not reason or self.state(finding_id)['status'] != 'CLOSED':
                raise ValueError('Only a closed finding can be reopened with a reason')
            return self._append(finding_id, 'REOPENED', {'reason': reason})

    def open_critical(self):
        blocked = []
        for state in self.list():
            if state['severity'] != 'critical':
                continue
            current = self.closure_current(state['finding_id'])
            if not current['current']:
                blocked.append({**state,'closure_freshness':current})
        return blocked


def validate_revalidation_requirement(requirement):
    """Freeze actual producer identities and measured predicates with the fix.

    This describes a future measurement; validation does not execute it or
    supply a passing result. A status-only predicate cannot close a finding.
    """
    from core.workflow import safe_id
    if not isinstance(requirement,dict) or requirement.get('schema') != 'finding-revalidation-requirement-v1':
        raise ValueError('Frozen finding-revalidation-requirement-v1 required')
    rows = requirement['measurements']
    maximum = requirement['maximum_measurements']
    if type(maximum) is not int or maximum < 1 or not isinstance(rows,list) or not 1 <= len(rows) <= maximum:
        raise ValueError('Bounded nonempty actual repair measurements required')
    identities = set()
    for row in rows:
        identity = (safe_id(row['stage_id']),safe_id(row['product']))
        if identity in identities:
            raise ValueError('Repeated repair measurement producer')
        identities.add(identity)
        if not isinstance(row.get('product_schema'),str) or not row['product_schema']:
            raise ValueError('Repair measurements require their exact scientific product schema')
        conditions = row['conditions']
        if not isinstance(conditions,list) or not conditions:
            raise ValueError('Repair measurement requires frozen physical/support predicates')
        measured = False
        for condition in conditions:
            path,operator = condition.get('path'),condition.get('operator')
            if not isinstance(path,list) or not path or any(not isinstance(key,str) and type(key) is not int for key in path):
                raise ValueError('Repair predicate must retain its exact measured field path')
            if operator not in {'minimum','maximum','exact','all_true','nonempty'}:
                raise ValueError('Unrecognized frozen repair measurement predicate')
            if operator in {'minimum','maximum'}:
                value = condition['value']
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('Finite frozen repair quantity threshold required')
            if operator == 'exact' and 'value' not in condition:
                raise ValueError('Exact repair predicate needs its frozen expected value')
            if operator != 'exact' and path[-1] not in {'status','execution_status','scientific_acceptance','science_acceptance'}:
                measured = True
        if not measured:
            raise ValueError('Scientific repair cannot be accepted from status strings alone')
    canonical_digest(requirement)
    return requirement


def validate_repair_parameters(parameters):
    from core.workflow import safe_id
    for key in ('finding_id','product_id'):
        safe_id(parameters[key])
    for key in ('finding_event_hash','dependency_graph_hash'):
        value = parameters[key]
        if not isinstance(value,str) or len(value) != 64 or any(q not in '0123456789abcdef' for q in value):
            raise ValueError('Repair producer must name the exact current fix and dependency scope')
    memory = parameters['memory_bytes']
    if type(memory) is not int or memory <= 0:
        raise ValueError('Positive exact repair metadata workspace required')
    requirement = validate_revalidation_requirement(parameters['requirement'])
    references = parameters['measurements']
    if not isinstance(references,list) or len(references) != len(requirement['measurements']):
        raise ValueError('Repair producer must retain every frozen measurement in declared order')
    return parameters


def repair_input_bindings(parameters):
    """Pin actual evidence and immutable fix events for public-stage resume."""
    from core.workflow import path_at
    from core.research.errors import _retained_json
    validate_repair_parameters(parameters)
    store = FindingStore(path_at(parameters['request_base'],parameters['finding_store']))
    state = store.state(parameters['finding_id'])
    if state['latest_event_hash'] != parameters['finding_event_hash'] or state['status'] != 'FIXED_PENDING_REVALIDATION':
        raise ValueError('Repair production requires the still-current recorded fix')
    fixed = state['events'][-1]
    product = parameters['product_id']
    graph = _dependency_graph(fixed['payload']['dependency_graph'])
    requirement = (fixed['payload'].get('revalidation_requirements') or {}).get(product)
    if (canonical_digest(graph) != parameters['dependency_graph_hash']
            or product not in store.affected_products(parameters['finding_id'],graph)
            or requirement != parameters['requirement']):
        raise ValueError('Repair producer changes the frozen product, graph or measured revalidation predicates')
    sources = dict(fixed['payload']['source_hashes'])
    for event in state['events']:
        path = store.root/state['finding_id']/f"{event['sequence']:08d}.json"
        sources[str(path)] = file_sha256(path)
    for reference in parameters['measurements']:
        _,bindings = _retained_json(reference,parameters['request_base'],parameters['memory_bytes'])
        for path,digest in bindings.items():
            if path in sources and sources[path] != digest:
                raise ValueError('Repair evidence refers to a different fix source version')
            sources[path] = digest
    return sources


def publish_repair_receipt(parameters,destination):
    """Evaluate retained measured predicates; never infer PASS from execution.

    Only fresh, sealed producer generations started after the recorded fix are
    admitted. Failed measured conditions are published as FAIL and therefore
    cannot close the finding. Every fix/source/event binding remains retained.
    """
    from core.workflow import path_at,verify_completed_receipt
    from core.research.errors import _retained_json
    from core.research.pilots import _condition
    validate_repair_parameters(parameters)
    base = parameters['request_base']
    store = FindingStore(path_at(base,parameters['finding_store']))
    state = store.state(parameters['finding_id'])
    sources = repair_input_bindings(parameters)
    fixed = state['events'][-1]
    since = fixed.get('created_unix')
    if not isinstance(since,(int,float)) or isinstance(since,bool) or not math.isfinite(since):
        raise ValueError('Record a new timed fix before producing scoped measured repair receipts')
    decisions,evidence,files = [],[],{}
    for index,(contract,reference) in enumerate(zip(parameters['requirement']['measurements'],parameters['measurements'])):
        receipt_path = path_at(base,reference['receipt'])
        producer = verify_completed_receipt(receipt_path)
        record,_ = _retained_json(reference,base,parameters['memory_bytes'])
        if (producer['stage_id'] != contract['stage_id'] or reference['product'] != contract['product']
                or record.get('schema') != contract['product_schema'] or record.get('synthetic_fixture') is True):
            raise ValueError('Repair evidence changes its frozen actual scientific producer contract')
        started = producer.get('started_unix')
        if isinstance(started,bool) or not isinstance(started,(int,float)) or not math.isfinite(started) or started < since:
            raise ValueError('Repair measurements must actually be rerun after the current recorded fix')
        if contract.get('measurement_origin') is not None and record.get('measurement_origin') != contract['measurement_origin']:
            raise ValueError('Repair measurement origin differs from the frozen physical control')
        root = Path(producer['generation']).resolve()
        payload = (root/producer['products'][reference['product']]['path']).resolve()
        files[str(payload)] = file_sha256(payload)
        files[str(receipt_path)] = file_sha256(receipt_path)
        evidence.append({'stage_id':producer['stage_id'],'product':reference['product'],
            'receipt':str(receipt_path),'receipt_sha256':file_sha256(receipt_path),
            'generation':str(root),'started_unix':started,'measurement':str(payload),'sha256':files[str(payload)]})
        for condition in contract['conditions']:
            decisions.append({'measurement_index':index,'condition':condition,'passed':bool(_condition(record,condition))})
    source = Path(__file__).resolve()
    sources[str(source)] = file_sha256(source)
    for path,digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Repair source or measured payload changed during scoped evaluation')
    if store.state(parameters['finding_id'])['latest_event_hash'] != fixed['event_hash']:
        raise ValueError('Finding fix changed during measured repair production')
    report = {'schema':'research-repair-receipt-v1','finding_id':state['finding_id'],
        'owning_requirement':state['owning_requirement'],'product_id':parameters['product_id'],
        'finding_event_hash':fixed['event_hash'],'dependency_graph_hash':parameters['dependency_graph_hash'],
        'revalidation_requirement_hash':canonical_digest(parameters['requirement']),
        'status':'PASS' if all(row['passed'] for row in decisions) else 'FAIL','execution_status':'COMPLETE',
        'decisions':decisions,'evidence':evidence,'source_hashes':sources,'files_sha256':files,
        'scope':'frozen measured predicates for this exact fix and affected product; closure remains a separate action'}
    destination = Path(destination).resolve()
    with native_run_lock(destination,operation='publish measured finding repair receipt'):
        if destination.exists():
            raise FileExistsError('Repair receipt generations are immutable')
        destination.mkdir(parents=True)
        output = destination/'repair-receipt.json'
        atomic_json(output,report)
    return report


def repair_receipt_runner(parameters,context):
    destination = Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Repair receipt escapes its owned scientific workflow generation')
    if parameters['memory_bytes'] > context.resources['memory_gib']*1024**3:
        raise MemoryError('Repair intake exceeds allocated workflow memory')
    report = publish_repair_receipt(parameters,destination)
    return {'measurements':{'repair_condition_count':len(report['decisions']),
        'passed_condition_count':sum(row['passed'] for row in report['decisions'])},
        'scientific_exclusions':[] if report['status'] == 'PASS' else
            [{'product_id':parameters['product_id'],'reason':'one or more frozen measured repair conditions failed'}]}


def repair_receipt_main():
    import argparse
    parser = argparse.ArgumentParser(description='Publish a current fix-bound receipt from actual retained rerun measurements')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 16*1024**2:
        raise MemoryError('Repair request exceeds bounded metadata intake')
    parameters = json.loads(args.request.read_text())
    parameters['request_base'] = str(args.request.resolve().parent)
    report = publish_repair_receipt(parameters,args.output)
    print(json.dumps(report,indent=2))
    return 0 if report['status'] == 'PASS' else 1


def execute_required_revalidation(store, finding_id, dependency_graph, workflows, destination, *,
                                  request_base, resume=False, receipt_products=None):
    """Preflight every affected rerun and collect actual scientific repair receipts.

    Workflow templates may reference {finding_id}, {finding_event_hash},
    {product_id}, and {dependency_graph_hash}. Receipt producers must issue
    measured PASS evidence themselves; dispatch never manufactures acceptance.
    """
    from core.workflow import execute_workflow,safe_id
    from copy import deepcopy
    state = store.state(finding_id)
    if state['status'] != 'FIXED_PENDING_REVALIDATION':
        raise ValueError('Revalidation requires a current recorded fix')
    fixed = state['events'][-1]
    graph = _dependency_graph(dependency_graph)
    if fixed['payload'].get('dependency_graph') != graph:
        raise ValueError('Record the complete dependency graph with the current fix before rerunning')
    graph_hash = canonical_digest(graph)
    affected = store.affected_products(finding_id,graph)
    if set(workflows) != set(affected):
        raise ValueError('Revalidation workflows must cover every affected product')
    names = receipt_products or {product:'repair_receipt' for product in affected}
    if set(names) != set(affected):
        raise ValueError('Each affected product requires its actual repair receipt product')
    def substitute(value,replacements):
        if isinstance(value,dict):
            return {key:substitute(item,replacements) for key,item in value.items()}
        if isinstance(value,list):
            return [substitute(item,replacements) for item in value]
        if isinstance(value,str):
            for key,replacement in replacements.items():
                value = value.replace('{'+key+'}',replacement)
        return value
    destination = Path(destination).resolve()/fixed['event_hash']
    with native_run_lock(destination,operation='complete affected-product revalidation'):
        prepared = {}
        for product in affected:
            safe_id(product)
            safe_id(names[product])
            workflow = substitute(deepcopy(workflows[product]),{'finding_id':finding_id,
                'finding_event_hash':fixed['event_hash'],'product_id':product,'dependency_graph_hash':graph_hash,
                'finding_store':str(store.root)})
            if names[product] not in workflow['requested_products']:
                raise ValueError('Revalidation workflow must request its scientific repair receipt')
            plan = execute_workflow(workflow,destination/product,request_base=request_base,dry_run=True)
            producers = [q for q in plan['stages'] if names[product] in q['outputs']]
            if len(producers) != 1:
                raise ValueError('Repair receipt needs one registered producer')
            prepared[product] = (workflow,producers[0])
        destination.mkdir(parents=True,exist_ok=True)
        results,receipts = {},{}
        dispatch = {'schema':'finding-revalidation-dispatch-v1','finding_id':finding_id,
            'finding_event_hash':fixed['event_hash'],'dependency_graph_hash':graph_hash,
            'affected_products':affected,'workflows':results,'receipts':receipts,
            'execution_status':'RUNNING','closure_status':'CURRENT_SCIENTIFIC_RECEIPTS_REQUIRED'}
        output = destination/'dispatch.json'
        if output.exists():
            old = json.loads(output.read_text())
            atomic_json(destination/'history'/(uuid.uuid4().hex+'.json'),old)
        atomic_json(output,dispatch)
        try:
            for product,(workflow,producer) in prepared.items():
                result = execute_workflow(workflow,destination/product,request_base=request_base,resume=resume)
                results[product] = result
                if result['execution_status'] != 'COMPLETE':
                    raise ValueError('Affected workflow has unfinished required execution')
                generation = Path(result['stages'][producer['stage_id']]['generation'])
                path = (generation/producer['outputs'][names[product]]).resolve()
                if not path.is_relative_to(generation):
                    raise ValueError('Repair receipt escaped its stage generation')
                reference = {'path':str(path),'sha256':file_sha256(path)}
                receipts[product] = _current_repair_receipt(product,reference,fixed['payload']['source_hashes'],
                    fixed['event_hash'],graph_hash,
                    (fixed['payload'].get('revalidation_requirements') or {}).get(product))
                atomic_json(output,dispatch)
            if store.state(finding_id)['latest_event_hash'] != fixed['event_hash']:
                raise ValueError('Finding fix changed during revalidation; historical reruns cannot close it')
            dispatch['execution_status'] = 'COMPLETE'
            dispatch['closure_status'] = 'CURRENT_RECEIPTS_COLLECTED_CLOSURE_REQUIRED'
            atomic_json(output,dispatch)
            return dispatch
        except BaseException as error:
            dispatch.update(execution_status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
                failure={'type':type(error).__name__,'message':str(error)})
            atomic_json(output,dispatch)
            raise


def findings_main():
    import argparse
    p = argparse.ArgumentParser(description='Append findings, record fixes, close with complete current revalidation')
    p.add_argument('action', choices=['record', 'fix', 'close', 'reopen', 'list', 'revalidate'])
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--request', type=Path)
    a = p.parse_args()
    if a.action != 'list' and a.request is None:
        p.error('--request is required for '+a.action)
    store = FindingStore(a.store)
    q = json.loads(a.request.read_text()) if a.request else {}
    if a.action == 'revalidate':
        q.setdefault('request_base',str(a.request.resolve().parent))
        print(json.dumps(execute_required_revalidation(store, **q), indent=2))
        return
    function = {'record': store.record_failure, 'fix': store.record_fix, 'close': store.close_only_with_current_receipts,
                'reopen': store.reopen, 'list': store.list}[a.action]
    print(json.dumps(function(**q), indent=2))
