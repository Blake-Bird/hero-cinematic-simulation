"""Resumable terminal questionnaires and prelaunch review without physics presets."""
from __future__ import annotations
import builtins
import json
import os
from pathlib import Path
import shutil

from core.foundation import atomic_json


class Revisit(Exception):
    """Restart only the pure questionnaire, replaying its validated input prefix."""


class Questionnaire:
    """Save every response; replay only while prompt labels still match.

    Branch changes discard the stale suffix. This wraps configuration collection
    only: solver execution must never be inside the replay callback.
    """
    def __init__(self, path, phase):
        self.path = Path(path).expanduser().resolve()
        self.phase = phase
        self.record = json.loads(self.path.read_text()) if self.path.exists() else {'schema': 'setup-draft-v1', 'phases': {}}
        if self.record.get('schema') != 'setup-draft-v1':
            raise ValueError('Unsupported setup draft')
        self.rows = self.record['phases'].setdefault(phase, [])
        self.cursor = 0

    def save(self):
        """Persist only the draft input history, independently of solver execution."""
        atomic_json(self.path, self.record)

    def run(self, callback):
        """Replay or edit a pure input callback, restoring input even on interruption."""
        original = builtins.input
        print('Setup draft: '+str(self.path))
        print('Commands at any prompt: :help, :back, :edit NUMBER, :list, :quit. Responses are saved automatically.')
        def read(prompt=''):
            if self.cursor < len(self.rows):
                row = self.rows[self.cursor]
                if row['prompt'] == prompt:
                    self.cursor += 1
                    print(prompt+row['answer']+' [saved]')
                    return row['answer']
                del self.rows[self.cursor:]
                self.save()
            while True:
                answer = original(prompt)
                if answer == ':help':
                    print('Enter the values and units printed above. :list shows answer numbers; :edit N revisits N and recollects later answers; :back revisits the last answer; :quit saves and exits.')
                elif answer == ':list':
                    for number, row in enumerate(self.rows, 1):
                        print(f"{number}. {row['prompt']}{row['answer']}")
                elif answer == ':quit':
                    self.save()
                    raise KeyboardInterrupt('Setup saved; rerun with the same --draft path')
                elif answer == ':back' or answer.startswith(':edit '):
                    try:
                        index = self.cursor-1 if answer == ':back' else int(answer.split()[1])-1
                        if not 0 <= index < len(self.rows):
                            raise ValueError()
                        del self.rows[index:]
                        self.save()
                        raise Revisit()
                    except (ValueError, IndexError):
                        print('Choose an existing answer number with :list.')
                else:
                    self.rows.append({'prompt': prompt, 'answer': answer})
                    self.cursor += 1
                    self.save()
                    return answer
        builtins.input = read
        try:
            while True:
                self.cursor = 0
                try:
                    result = callback()
                    print('\nReview '+self.phase+' settings')
                    print(json.dumps(result, indent=2, default=str))
                    answer = read('Accept these settings [yes, :back, :edit N, :quit]: ').strip().lower()
                    if answer == 'yes':
                        del self.rows[self.cursor-1:]
                        self.save()
                        return result
                    # Do not replay a rejected review response indefinitely.
                    del self.rows[self.cursor-1:]
                    self.save()
                    print('Use :list and :edit N to change settings, or enter yes.')
                except Revisit:
                    continue
        finally:
            builtins.input = original


def draft_path(value=None):
    from core.resources import runtime_paths
    return Path(value).expanduser().resolve() if value else runtime_paths().state_root/'drafts'/'interactive-setup.json'


def collect_simulation(source, path=None, *, outputs=True):
    """Walk through the existing physical inputs and selected output request files."""
    from commands import _gather_answers, choose_products, load_json, redshift_schedule
    from core.epochs import highlight_redshifts
    source = Path(source).expanduser().resolve()
    def collect():
        answers = _gather_answers(source)
        start = float(load_json(source)['parameters']['z_start'])
        redshift_schedule(answers['schedule'], start)
        while True:
            extra = input('Additional highlighted saved redshifts [comma list, or none]: ')
            try:
                answers['highlight_redshifts'] = highlight_redshifts(answers['schedule'], start, extra)
                break
            except ValueError as error:
                print(error)
        if outputs:
            print('\n[5/7] Figures and outputs')
            answers['graphs'] = choose_products()
            print('\n[6/7] Optional measurement tasks')
            collect_requests(answers)
        return answers
    # A different source invalidates the simulation phase, rather than rebinding old answers silently.
    session = Questionnaire(draft_path(path), 'simulation:'+str(source))
    return session.run(collect)


REQUESTS = {'stellar_components': ('stellar_search', '--stellar-search'),
            'stellar_histories': ('stellar_history', '--stellar-history'),
            'zoom_contamination': ('zoom_regions', '--zoom-regions'),
            'inference_diagnostics': ('inference_diagnostics', '--inference-diagnostics'),
            'radial_shapes': ('radial_shapes', '--radial-shapes')}


def collect_requests(answers):
    """Make additional requests explicit; saved-host selection can happen later."""
    from commands import load_json
    broad = answers['graphs'] in (['all'], ['available'])
    for product, (key, flag) in REQUESTS.items():
        if not broad and product not in answers['graphs']:
            continue
        if answers.get(key) is not None:
            continue
        while True:
            raw = input(f'{product}: request JSON path, or later (requires saved selections/data): ').strip()
            if raw == 'later':
                answers.setdefault('deferred_outputs', []).append(product)
                print(f'Pending {product}. After snapshots exist: analysis RUN --graphs {product} {flag} REQUEST.json')
                break
            try:
                answers[key] = load_json(Path(raw).expanduser())
                break
            except (OSError, ValueError) as error:
                print(error)
    print('Other experiments: compare-runs, trace-zoom, zoom-campaign, streams, and flagship have separate inputs. Use their --help commands; they do not run automatically.')


def launch_review(cfg, run_dir, products, answers):
    """Check selected tools and estimate snapshot storage before native preparation.

    Existing native resource/census checks remain authoritative. This estimate
    covers base particle payload only, not catalogues, renders or peak IC memory.
    """
    from commands import parse_products
    run_dir = Path(run_dir).resolve()
    selected = parse_products(products) if products else []
    explicit = set(selected) if selected not in (['all'], ['available']) else set()
    issues, pending = [], []
    if run_dir.exists():
        issues.append('Run directory already exists: '+str(run_dir))
    parent = run_dir.parent
    while not parent.exists():
        parent = parent.parent
    if not os.access(parent, os.W_OK):
        issues.append('Output ancestor is not writable: '+str(parent))
    sim = cfg['simulation']
    from core.resources import estimate_run_resources
    sizing = estimate_run_resources(cfg, len(sim['snapshot_redshift_list']))
    estimate = sizing['disk_bytes']
    free = shutil.disk_usage(parent).free
    if estimate > free:
        issues.append('Estimated base snapshot storage exceeds free disk space')
    tools = {}
    if explicit & {'blender', 'blender_movie'} or selected == ['all']:
        from core.visualization import _find_blender_executable
        try:
            tools['blender'] = _find_blender_executable()
            if not tools['blender']:
                issues.append('Blender executable missing')
        except Exception as error:
            issues.append('Blender unavailable: '+str(error))
    if cfg.get('hardware', {}).get('mpi_ranks', 1) > 1:
        tools['MPI'] = shutil.which('mpirun') or shutil.which('mpiexec')
        if not tools['MPI']:
            issues.append('MPI launcher missing')
    for product, (key, flag) in REQUESTS.items():
        if (product in explicit or selected == ['all']) and answers.get(key) is None:
            pending.append(f'{product}: supply {flag} REQUEST.json after selecting saved data')
    # A fresh interactive stellar host is deliberately collected after evolution.
    deferred = set(answers.get('deferred_outputs', []))
    blocking = [p for p in pending if p.split(':', 1)[0] not in deferred and not p.startswith('stellar_components:')]
    if selected == ['all']:
        blocking = [p for p in pending if p.split(':', 1)[0] not in deferred]
    issues.extend(blocking)
    review = {'schema': 'launch-review-v1', 'run': str(run_dir), 'requested_outputs': selected,
              'saved_redshifts': sim['snapshot_redshift_list'], 'highlight_redshifts': sim.get('highlight_redshifts', []),
              'estimated_storage_bytes': estimate, 'estimated_resources': sizing, 'free_disk_bytes': free, 'tools': tools,
              'pending': pending, 'explicitly_deferred_outputs': sorted(deferred), 'blocking': issues,
              'estimate_scope': 'Existing resource estimator: calibrated initial-species snapshot/IC allowance and reserve; excludes render storage and future particle growth. Native census/admission gates remain authoritative.'}
    print('\n[7/7] Launch review\n'+json.dumps(review, indent=2))
    from core.resources import runtime_paths
    import uuid
    path = runtime_paths().state_root/'drafts'/('launch-review-'+uuid.uuid4().hex+'.json')
    atomic_json(path, review)
    if issues:
        raise ValueError('Resolve launch review blockers before evolution; review: '+str(path))
    return review
