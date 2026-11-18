"""DarkMatterSim Modular Suite Core Physics & Analysis Library.

Shared drivers, catalogue readers, numerical conventions and visualization
helpers used by the root entrypoints. Native engine source and required
physical tables are bundled separately under core/software.
"""

# Workflow and units
# Marks the shared scientific helper directory for package tools.
# Physical units are declared by the numerical modules rather than implicitly by package import.
# Root entrypoints import the shared modules; this initializer has no numerical side effects.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

# Consolidated historical drivers use direct sibling imports. Keep those
# available to package entry points as well; no solver or analysis is run here.
import sys as _core_sys
_core_sys.dont_write_bytecode = True
from pathlib import Path as _CorePath
_core_directory = str(_CorePath(__file__).resolve().parent)
if _core_directory not in _core_sys.path:
    _core_sys.path.insert(0, _core_directory)

# Consolidated extensions retain their original import namespaces. Scientific
# modules use separate globals even when their editable source shares a file.
# This prevents common helper names/constants from changing another algorithm.
import __future__ as _core_future
import importlib.abc as _core_import_abc
import importlib.util as _core_import_util
import json as _core_json
import types as _core_types

_layout_path = _CorePath(_core_directory).parent / 'configs' / 'core_source_layout.json'
_module_layout = {}
_module_future_flags = {}
if _layout_path.is_file():
    _layout_record = _core_json.loads(_layout_path.read_text(encoding='utf-8'))
    if _layout_record.get('schema') != 'research2-core-source-layout-v1':
        raise ImportError('Unknown consolidated core source layout')
    for _logical, _record in _layout_record['modules'].items():
        _physical = _CorePath(_record['source'])
        if (not _logical.isidentifier() or _physical.parent != _CorePath('core')
                or _physical.suffix != '.py' or _record.get('section') != _logical):
            raise ImportError('Unsafe consolidated core namespace descriptor')
        _module_layout[_logical] = _CorePath(_core_directory) / _physical.name
        _flags = 0
        for _feature in _record.get('future_features', []):
            if _feature not in _core_future.all_feature_names:
                raise ImportError('Unknown source future feature: ' + _feature)
            _flags |= getattr(_core_future, _feature).compiler_flag
        _module_future_flags[_logical] = _flags


def source_path(name):
    """Find an ordinary module's source or a remaining historical section owner."""
    logical = str(name).removeprefix('core.').removesuffix('.py')
    if logical in _module_layout:
        return _module_layout[logical]
    parts = logical.split('.')
    if not all(part.isidentifier() for part in parts):
        raise ValueError('A dotted Python module name is required')
    path = _CorePath(_core_directory).joinpath(*parts)
    source = path.with_suffix('.py')
    return source if source.is_file() else path / '__init__.py' if path.is_dir() else source


def source_section(name):
    """Return plain section code and its physical first line for tooling."""
    logical = str(name).removeprefix('core.').removesuffix('.py')
    path = source_path(logical)
    contents = path.read_text(encoding='utf-8')
    if logical not in _module_layout:
        return contents, 1
    start = '# @research2-module ' + logical + '\n'
    end = '# @research2-end-module ' + logical + '\n'
    if contents.count(start) != 1 or contents.count(end) != 1:
        raise ImportError('Consolidated module boundaries changed: ' + logical)
    offset = contents.index(start) + len(start)
    finish = contents.index(end, offset)
    return contents[offset:finish], contents.count('\n', 0, offset) + 1


class _ConsolidatedSourceLoader(_core_import_abc.InspectLoader):
    def __init__(self, logical):
        self.logical = logical

    def create_module(self, spec):
        return (_core_sys.modules.get('core.' + self.logical)
                or _core_sys.modules.get(self.logical)
                or _core_types.ModuleType('core.' + self.logical))

    def is_package(self, fullname):
        return False

    def get_filename(self, fullname):
        return str(source_path(self.logical))

    def get_source(self, fullname):
        source, line = source_section(self.logical)
        return '\n' * (line - 1) + source

    def get_code(self, fullname):
        return compile(self.get_source(fullname), self.get_filename(fullname), 'exec',
            flags=_module_future_flags[self.logical], dont_inherit=True)

    def exec_module(self, module):
        module.__name__ = 'core.' + self.logical
        module.__package__ = 'core'
        module.__file__ = self.get_filename(module.__name__)
        module.__loader__ = self
        module.__spec__ = _core_import_util.spec_from_loader(module.__name__, self, origin=module.__file__)
        if module.__dict__.get('_research2_section_loaded'):
            return
        module.__dict__['_research2_logical_source'] = self.logical
        module.__dict__['_research2_section_loaded'] = True
        _core_sys.modules[self.logical] = module
        _core_sys.modules['core.' + self.logical] = module
        try:
            exec(self.get_code(module.__name__), module.__dict__)
        except BaseException:
            module.__dict__.pop('_research2_section_loaded', None)
            for alias in (self.logical, 'core.' + self.logical):
                if _core_sys.modules.get(alias) is module:
                    del _core_sys.modules[alias]
            raise
        setattr(_core_sys.modules['core'], self.logical, module)


class _ConsolidatedSourceFinder(_core_import_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        logical = fullname.removeprefix('core.')
        if '.' in logical or logical not in _module_layout:
            return None
        loader = _ConsolidatedSourceLoader(logical)
        return _core_import_util.spec_from_loader(fullname, loader, origin=loader.get_filename(fullname))


if _module_layout and not any(getattr(finder, '_research2_section_finder', False) for finder in _core_sys.meta_path):
    _section_finder = _ConsolidatedSourceFinder()
    _section_finder._research2_section_finder = True
    _core_sys.meta_path.insert(0, _section_finder)
