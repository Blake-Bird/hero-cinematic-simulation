"""Rebuild the first-party API and CLI reading references without running workflows.

Run with the installed Python environment from any directory. These documents describe
source signatures and parser declarations; execution evidence is recorded separately.
"""
from pathlib import Path
import ast
import inspect
import importlib
import html
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cell(value):
    return ' '.join(str(value).split()).replace('|', '&#124;')



class ReturnExpressions(ast.NodeVisitor):
    """Collect explicit results without entering nested function scopes."""
    def __init__(self):
        self.results = []
        self.yields = False

    def visit_FunctionDef(self, node):
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Return(self, node):
        expression = ast.unparse(node.value) if node.value is not None else 'None'
        self.results.append((node.lineno, expression))

    def visit_Yield(self, node):
        self.yields = True

    visit_YieldFrom = visit_Yield


def result_description(node, relative):
    """Link each literal return site while preserving branch uncertainty."""
    if isinstance(node, ast.ClassDef):
        return 'Class. Inspect its methods and constructor.'
    visitor = ReturnExpressions()
    for statement in node.body:
        visitor.visit(statement)
    if visitor.yields:
        return 'Generator. Inspect yielded values and iteration conditions.'
    if not visitor.results:
        return 'No explicit return. Normal completion returns None.'
    descriptions = []
    for line, expression in visitor.results:
        label = expression if len(expression) <= 180 else expression[:177] + '...'
        safe = html.escape(cell(label)).replace('[', '&#91;').replace(']', '&#93;')
        descriptions.append(f'<code>{safe}</code> [line {line}](../{relative}#L{line})')
    return '<br>'.join(descriptions)


def parser_argument_rows(tree):
    """Keep argparse declarations attached to their literal subcommand path.

    Repeated flags on different parsers are separate declarations. This is a
    source reader; conditional parser behavior still requires live --help.
    """
    parsers = {}
    statements = sorted(ast.walk(tree), key=lambda node: getattr(node, 'lineno', 0))
    for node in statements:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        owner = ast.unparse(call.func.value)
        if call.func.attr == 'ArgumentParser':
            path = ''
        elif call.func.attr == 'add_subparsers':
            path = parsers.get(owner, '')
        elif call.func.attr == 'add_parser' and call.args:
            label = str(call.args[0].value) if isinstance(call.args[0], ast.Constant) else ast.unparse(call.args[0])
            path = ' '.join(filter(None, (parsers.get(owner, ''), label)))
        else:
            continue
        for target in node.targets:
            parsers[ast.unparse(target)] = path
    # Expand only literal string loops and string/type pairs. Evaluate no Python
    # code: a tiny expression reader accepts concatenation and str.replace only.
    def argument_text(node, binding):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in binding:
            return argument_text(binding[node.id], {})
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return argument_text(node.left, binding) + argument_text(node.right, binding)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'replace' and len(node.args) == 2 and not node.keywords):
            return argument_text(node.func.value, binding).replace(
                argument_text(node.args[0], binding), argument_text(node.args[1], binding))
        raise ValueError('Nonliteral parser argument')

    loop_arguments = {}
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or not isinstance(loop.iter, (ast.Tuple, ast.List)):
            continue
        targets = [loop.target] if isinstance(loop.target, ast.Name) else (
            loop.target.elts if isinstance(loop.target, (ast.Tuple, ast.List)) else [])
        if not targets or any(not isinstance(target, ast.Name) for target in targets):
            continue
        bindings = []
        for value in loop.iter.elts:
            values = [value] if isinstance(loop.target, ast.Name) else (
                value.elts if isinstance(value, (ast.Tuple, ast.List)) else [])
            if len(values) != len(targets):
                bindings = []; break
            bindings.append({target.id: value for target, value in zip(targets, values)})
        if not bindings:
            continue
        for statement in loop.body:
            for call in ast.walk(statement):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == 'add_argument' and call.args
                        and any(isinstance(node, ast.Name) and node.id in bindings[0]
                                for arg in call.args for node in ast.walk(arg))):
                    try:
                        expanded = [( [argument_text(arg, binding) for arg in call.args], binding)
                                    for binding in bindings]
                    except ValueError:
                        continue
                    loop_arguments[id(call)] = expanded
    rows = []
    for node in statements:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        args = [ast.literal_eval(a) if isinstance(a, ast.Constant) else ast.unparse(a) for a in node.args]
        kw = {a.arg: a.value for a in node.keywords}
        helptext = ast.literal_eval(kw['help']) if isinstance(kw.get('help'), ast.Constant) else ''
        for argument_names, binding in loop_arguments.get(id(node), [(args, {})]):
            constraints = '; '.join(k + '=' + ast.unparse(binding.get(v.id, v) if isinstance(v, ast.Name) else v)
                                    for k, v in kw.items()
                                    if k in {'required', 'default', 'choices', 'type', 'nargs', 'action'})
            rows.append((parsers.get(ast.unparse(node.func.value), ''), argument_names, helptext, constraints))
    return rows


def source_dependencies(paths, trees):
    """Resolve literal local imports and their reverse links for reading."""
    available = {path.relative_to(ROOT).as_posix(): path for path in paths}
    layout = json.loads((ROOT / 'configs/core_source_layout.json').read_text())['modules']

    def resolve(name):
        logical = name.removeprefix('core.')
        if logical in layout:
            return available.get(layout[logical]['source'])
        relative = name.replace('.', '/')
        return available.get(relative + '.py') or available.get(relative + '/__init__.py')

    dependencies = {}
    for path, tree in trees.items():
        package = path.relative_to(ROOT).parent.parts
        local = set()
        external = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = package[:len(package) - node.level + 1] if node.level else ()
                base = '.'.join((*parent, node.module)) if node.module else '.'.join(parent)
                names = [base, *(base + '.' + alias.name for alias in node.names)]
            else:
                continue
            for name in names:
                target = resolve(name)
                if target and target != path:
                    local.add(target)
                elif not target and (not isinstance(node, ast.ImportFrom) or node.level == 0):
                    top = name.split('.')[0]
                    if top and top not in sys.stdlib_module_names and top != 'core':
                        external.add(top)
        dependencies[path] = (sorted(local), sorted(external))
    callers = {path: [] for path in paths}
    for path, (imports, _) in dependencies.items():
        for imported in imports:
            callers[imported].append(path)
    return dependencies, callers


def parser_owner(handler):
    """Follow a simple imported-function forwarder without invoking its workflow.

    Stop at a handler with argparse declarations or a non-forwarding body.
    Only an unambiguous zero-argument return call to an imported first-party
    function is followed; more dynamic dispatch remains explicit in the guide.
    """
    chain = [handler]
    while len(chain) <= 8:
        tree = ast.parse(inspect.getsource(chain[-1]))
        if parser_argument_rows(tree):
            break
        function = tree.body[0]
        returns = [node for node in function.body if isinstance(node, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            break
        call = returns[0].value
        if not isinstance(call.func, ast.Name) or call.args or call.keywords:
            break
        imports = [(node.module, alias.name)
                   for node in function.body if isinstance(node, ast.ImportFrom)
                   and node.level == 0 and node.module
                   for alias in node.names if (alias.asname or alias.name) == call.func.id]
        if len(imports) != 1:
            break
        module_name, name = imports[0]
        if not (module_name.startswith('core.') or module_name == 'verification'):
            break
        target = getattr(importlib.import_module(module_name), name)
        source = inspect.getsourcefile(target)
        if (not inspect.isfunction(target) or target in chain or source is None
                or not Path(source).resolve().is_relative_to(ROOT)):
            break
        chain.append(target)
    return chain[-1], chain


def parser_group_choices(tree, namespace):
    """Describe loop-created subparsers from literal strings or string tuples.

    Read already loaded constants only; do not evaluate calls or dynamic iterables.
    """
    def strings(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.Name):
            value = namespace.get(node.id)
            if type(value) in (tuple, list) and all(type(item) is str for item in value):
                return list(value)
        if isinstance(node, ast.Starred):
            return strings(node.value)
        if isinstance(node, (ast.Tuple, ast.List)):
            return [value for child in node.elts for value in strings(child)]
        raise ValueError('Dynamic parser group')

    groups = {}
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
            continue
        if not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                   and call.func.attr == 'add_parser' and call.args
                   and isinstance(call.args[0], ast.Name) and call.args[0].id == loop.target.id
                   for statement in loop.body for call in ast.walk(statement)):
            continue
        try:
            groups[loop.target.id] = strings(loop.iter)
        except ValueError:
            continue
    return groups


def callable_notes(trees, notes_path=None):
    """Read reviewed API prose and reject stale or ambiguous implementation bindings.

    Notes bind a unique public top-level definition to its complete AST hash.
    Reading this documentation never imports or invokes the described callable.
    """
    path = Path(notes_path) if notes_path else ROOT / 'docs/python-api-notes.md'
    if not path.is_file():
        return {}
    notes = {}
    for heading, body in re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', path.read_text(), re.M | re.S):
        relative, separator, name = heading.partition('::')
        source = ROOT / relative
        if (not separator or Path(relative).is_absolute() or source not in trees
                or not name.isidentifier() or name.startswith('_')):
            raise ValueError('Unknown public API note target: ' + heading)
        definitions = [node for node in trees[source].body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and node.name == name]
        binding = re.search(r'<!-- implementation-sha256: ([0-9a-f]{64}) -->', body)
        if len(definitions) != 1 or binding is None:
            raise ValueError('API note needs a unique definition and AST binding: ' + heading)
        actual = hashlib.sha256(ast.dump(definitions[0], include_attributes=False).encode()).hexdigest()
        if actual != binding[1]:
            raise ValueError('Review API note after implementation change: ' + heading)
        description = (body[:binding.start()] + body[binding.end():]).strip()
        key = (source, name)
        if not description or key in notes:
            raise ValueError('Empty or duplicate public API note: ' + heading)
        notes[key] = description
    return notes


def build():
    import commands
    paths = sorted({*ROOT.glob('*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('optional_tests/*.py'),
                    *(p for p in (ROOT / 'core').rglob('*.py') if 'software' not in p.parts)})
    trees = {path: ast.parse(path.read_text()) for path in paths}
    notes = callable_notes(trees)
    definitions = [(path, node) for path, tree in trees.items() for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and not node.name.startswith('_')]
    described = sum(bool(ast.get_docstring(node) or notes.get((path, node.name)))
                    for path, node in definitions)
    dependencies, callers = source_dependencies(paths, trees)
    lines = ['# Python file and callable reference', '',
             'Generated by `python scripts/build_reference.py` from the first-party source. '
             'Use the [architecture guide](architecture.md#python-source-map) for callers and workflow ownership; '
             'use the [command reference](command-reference.md) to launch workflows. Do not execute library files individually.', '',
             'Each entry gives the module explanation, local import links, external packages and public top-level signatures. '
             'Import links are static source declarations, including conditional imports. Runtime dispatch and dynamic imports '
             'require the implementation and execution receipts. Parameters are inputs. '
             'The accompanying source documentation describes the implementation and return value where documented. '
             'An absent description is an explicit documentation gap, not a claim that the function has no output. '
             'Return sites reproduce source expressions and link to each branch. They describe source structure. '
             'Links open the actual implementation, including validation and file-writing branches. '
             'Dynamic logical namespaces in `configs/core_source_layout.json` can expose functions from consolidated files; '
             'use the namespace imported by the caller rather than guessing from a physical filename.', '',
             'Reviewed explanations can also come from the '
             '[maintained API notes](python-api-notes.md). Each note is checked against '
             'the definition’s full syntax tree before generation; a changed implementation '
             'requires its explanation to be reviewed.', '',
             f'Inventory: **{len(paths)} Python files**, excluding vendored software and standalone tests; optional lab helpers and checks are included. '
             '`verification.py` is included because it is also a public verification entry point.', '',
             f'Description coverage: **{described} of {len(definitions)} public top-level definitions** '
             f'have a source docstring or reviewed note; **{len(definitions)-described} remain undocumented**. '
             'This counts available explanations, not their scientific verification or runtime coverage. '
             'Methods, nested functions and private helpers are outside this count.', '', '## Files', '']
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f'- [{rel}](#{rel.replace("/", "-").replace(".", "-")})')
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        tree = trees[path]
        lines += ['', f'<a id="{rel.replace("/", "-").replace(".", "-")}"></a>', f'## {rel}', '',
                  f'[Read implementation](../{rel})', '', ast.get_docstring(tree) or 'No module docstring is present.', '']
        local, external = dependencies[path]
        def links(files):
            return ', '.join(f'[{file.relative_to(ROOT).as_posix()}](../{file.relative_to(ROOT).as_posix()})' for file in files)
        if local:
            lines += ['**Local imports** ' + links(local) + '.', '']
        if callers[path]:
            lines += ['**Imported by** ' + links(callers[path]) + '.', '']
        if external:
            lines += ['**External imports** ' + ', '.join(chr(96) + name + chr(96) for name in external) + '.', '']
        if not local and not callers[path]:
            lines += ['No literal local import links were resolved for this file.', '']
        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith('_')]
        if not defs:
            lines += ['This file supplies package initialization, registration, or forwarding; inspect its imports and callers.', '']
            continue
        lines += ['| Public definition and input signature | Implementation / result description | Explicit return sites |', '|---|---|---|']
        for node in defs:
            signature = node.name + ('(' + ast.unparse(node.args) + ')' if hasattr(node, 'args') else ' (class)')
            if getattr(node, 'returns', None) is not None:
                signature += ' → ' + ast.unparse(node.returns)
            doc = ast.get_docstring(node)
            description = cell(doc or notes.get((path, node.name), '')) or 'See the source for return values and side effects.'
            lines.append(f'| [`{cell(signature)}`](../{rel}#L{node.lineno}) | {description} | {result_description(node, rel)} |')
    (ROOT / 'docs/python-reference.md').write_text('\n'.join(lines) + '\n')
    cli = ['# Command reference', '',
           'Generated from `commands.COMMANDS` and the handlers’ argparse declarations by '
           '`python scripts/build_reference.py`. Run commands from the source folder with the installed environment active.', '',
           '```bash', 'python commands.py --help', 'python commands.py COMMAND --help', '```', '',
           'This is a lookup reference, not a sequence to run from top to bottom. '
           'The [operator guide](operator-guide.md) supplies workflow order and examples. '
           'Request JSON examples live in `configs/`; choose the file named by the relevant workflow guide. '
           'Paths, models, units, selected hosts, resource limits, and destinations must describe your experiment. '
           'Never reuse a published destination for a new experiment.', '',
           'Options below are source declarations, including conditional branches; the handler’s live help '
           'resolves its actual parser, required arguments, choices, and defaults. Simple imported-function '
           'forwarders are followed to their parser owner without invoking the workflow. Dynamic expressions '
           'remain visible as source expressions; use subcommand help for their concrete values.', '',
           f'**{len(commands.COMMANDS)} registered commands.**', '']
    for name, handler in sorted(commands.COMMANDS.items()):
        cli.append(f'- [{name}](#{name})')
    for name, handler in sorted(commands.COMMANDS.items()):
        src = Path(inspect.getsourcefile(handler)).resolve()
        relative = src.relative_to(ROOT).as_posix()
        owner, chain = parser_owner(handler)
        tree = ast.parse(inspect.getsource(owner))
        doc = inspect.getdoc(handler) or notes.get((src, handler.__name__))
        cli += ['', f'## {name}', '', f'`python commands.py {name} --help`', '',
                f'Implementation: [{relative}](../{relative}), `{handler.__name__}`.', '']
        if doc: cli += [doc, '']
        if owner is not handler:
            owner_source = Path(inspect.getsourcefile(owner)).resolve().relative_to(ROOT).as_posix()
            cli += [f'Parser owner: [{owner_source}](../{owner_source}#L{owner.__code__.co_firstlineno}), '
                    f'`{owner.__name__}`. The wrapper forwards to this function.', '']
        rows = parser_argument_rows(tree)
        for group, choices in parser_group_choices(tree, owner.__globals__).items():
            cli += [f'Parser group `{group}` accepts ' + ', '.join(f'`{value}`' for value in choices)
                    + f'. Inspect one with `python commands.py {name} SUBCOMMAND --help`.', '']
        if rows:
            cli += ['| Parser / subcommand | Argument | Meaning | Parser constraints |', '|---|---|---|---|']
            for subcommand, args, helptext, constraints in rows:
                cli.append(f'| {cell(subcommand) or "main"} | `{cell(", ".join(args))}` | {cell(helptext) or "See live help and implementation."} | {cell(constraints)} |')
        elif owner.__module__ == 'verification' and owner.__name__ == 'main':
            cli += ['Verification delegates test selection to `unittest.main`. Use '
                    '`python commands.py verify --help` for standard test-name, verbosity, '
                    'fail-fast and discovery options. The additional `--group NAME` switch '
                    'selects a retained domain group; repeat it to combine groups. With '
                    '`--group`, only `-v`/`--verbose` or `-q`/`--quiet` may accompany the groups. '
                    'See `TEST_GROUPS` in the parser-owner source for the available names.']
        else:
            cli += ['This handler delegates parsing or has an interactive interface. Follow the implementation link and its live help.']
    (ROOT / 'docs/command-reference.md').write_text('\n'.join(cli) + '\n')
    print(json.dumps({'python_files': len(paths), 'commands': len(commands.COMMANDS)}))


if __name__ == '__main__':
    build()
