"""Observed first-party Python calls in the lab runner's current thread.

This is execution evidence, not branch coverage or scientific acceptance.
Imports are counted separately. Native code, other threads and isolated
library subprocesses retain their separate receipts and are not traced here.
"""
from pathlib import Path
import hashlib
import os
import sys


class PythonCalls:
    """Preserve any existing profiler and record calls without argument values."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.prefixes = tuple({str(Path(root).absolute()) + '/', str(self.root) + '/'})
        self.files = {}
        self.previous = None
        self.initial_sources = {}

    def __enter__(self):
        for directory, folders, names in os.walk(self.root):
            parent = Path(directory)
            folders[:] = [name for name in folders if name not in {'.git', '.venv', '__pycache__'}
                          and not (parent.name == 'core' and name == 'software')]
            for name in names:
                path = parent / name
                if path.suffix == '.py':
                    self.initial_sources[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.previous = sys.getprofile()
        sys.setprofile(self.observe)
        return self

    def observe(self, frame, event, argument):
        if self.previous is not None:
            self.previous(frame, event, argument)
        if event != 'call':
            return
        filename = frame.f_code.co_filename
        relative = next((filename[len(prefix):] for prefix in self.prefixes
                         if filename.startswith(prefix)), None)
        if relative is None:
            return
        if relative.startswith('core/software/') or not relative.endswith('.py'):
            return
        record = self.files.get(relative)
        if record is None:
            path = self.root / relative
            if not path.is_file():
                return
            record = {'calls': {}, 'module_events': 0, 'definition_events': 0,
                      'source_sha256': self.initial_sources.get(relative, hashlib.sha256(path.read_bytes()).hexdigest()),
                      'source_present_at_start': relative in self.initial_sources}
            self.files[relative] = record
        if frame.f_code.co_name == '<module>':
            record['module_events'] += 1
            return
        if not frame.f_code.co_flags & 2:  # CO_NEWLOCALS: class bodies are definitions.
            record['definition_events'] += 1
            return
        name = getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)
        key = name + ':' + str(frame.f_code.co_firstlineno)
        record['calls'][key] = record['calls'].get(key, 0) + 1

    def __exit__(self, kind, value, traceback):
        sys.setprofile(self.previous)

    def report(self):
        """Bind source revisions and keep import-only files distinguishable."""
        for relative, record in self.files.items():
            path = self.root / relative
            record['source_unchanged'] = (record['source_present_at_start'] and path.is_file() and
                hashlib.sha256(path.read_bytes()).hexdigest() == record['source_sha256'])
        return {
            'schema': 'research2-observed-python-calls-v1',
            'scope': 'Observed call events in the runner main Python thread during measurements. Module imports are separate; generator resumptions can count as calls. Other threads, native code and library subprocesses are excluded. Counts do not establish branch coverage or scientific acceptance.',
            'files_with_calls': sum(bool(row['calls']) for row in self.files.values()),
            'call_events': sum(sum(row['calls'].values()) for row in self.files.values()),
            'files': self.files,
        }
