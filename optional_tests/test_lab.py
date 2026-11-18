"""Focused checks for local UI authority, path confinement and truthful inventory.

Run: python -m unittest discover -s optional_tests -p 'test_*.py' -v
Numerical end-to-end runs are separately retained by lab.py; these checks target
serving boundaries and catalogue claims that the numerical run cannot cover.
"""
import http.client
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from optional_tests.catalogue import build_catalogue
from optional_tests.server import confined,JobStore,make_handler
from optional_tests.lab import LabRun,library_control


class LocalLabTests(unittest.TestCase):
    def test_source_map_explains_reviewed_functions_without_claiming_execution(self):
        from optional_tests.catalogue import source_inventory
        rows={row['path']:row for row in source_inventory()}
        source=rows['core/sidm.py']
        function=next(row for row in source['functions'] if row['name']=='measure_isolated_control_history')
        self.assertEqual(function['description_source'],'reviewed_note')
        self.assertIn('E=sum(m*v²)/2+sum(m*phi)/2',function['description'])
        self.assertIn('without h scaling',function['description'])
        self.assertEqual(source['status'],'NOT_EXECUTED_BY_THIS_LAB')
        documented=next(row for row in rows['optional_tests/lab.py']['functions'] if row['name']=='measure_density')
        self.assertEqual(documented['description_source'],'source_docstring')

    def test_source_map_rejects_an_explanation_bound_to_changed_implementation(self):
        import ast
        from optional_tests.catalogue import source_inventory
        parse=ast.parse
        def changed_tree(text):
            tree=parse(text)
            for node in tree.body:
                if isinstance(node,ast.ClassDef) and node.name=='SphericalFieldAdapter':
                    node.body.append(ast.Pass())
            return tree
        with patch('optional_tests.catalogue.ast.parse',side_effect=changed_tree):
            with self.assertRaisesRegex(ValueError,'Review API note after implementation change'):
                source_inventory()

    def test_runner_and_server_reject_outputs_resolving_inside_source(self):
        import uuid
        root=Path(__file__).resolve().parents[1]
        destination=root/('forbidden-lab-'+uuid.uuid4().hex)
        for factory in (LabRun,JobStore):
            with self.assertRaisesRegex(ValueError,'outside the source tree'):
                factory(destination)
            self.assertFalse(destination.exists())
        with tempfile.TemporaryDirectory() as name:
            alias=Path(name)/'source-alias';alias.symlink_to(root,target_is_directory=True)
            for factory in (LabRun,JobStore):
                with self.assertRaisesRegex(ValueError,'outside the source tree'):
                    factory(alias/destination.name)
                self.assertFalse(destination.exists())

    def test_advanced_worker_without_site_packages_retains_unavailable_report(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'advanced'
            script=Path(__file__).with_name('advanced_controls.py')
            process=subprocess.run([sys.executable,'-S',str(script),'shapes','--output',str(output)],
                cwd=name,capture_output=True,text=True,timeout=15)
            self.assertEqual(process.returncode,0,process.stderr)
            record=json.loads((output/'receipt.json').read_text())
            self.assertEqual(record['status'],'UNAVAILABLE')
            self.assertIn('optional_tests/advanced_controls.py',record['source_sha256'])
            self.assertTrue((output/'index.html').is_file())
            self.assertFalse((output/'data').exists())
            self.assertNotIn('href="figures/gallery/gallery.json"',(output/'index.html').read_text())

    def test_advanced_worker_rejects_source_tree_output_before_creation(self):
        import uuid
        script=Path(__file__).with_name('advanced_controls.py')
        output=script.parent/('forbidden-'+uuid.uuid4().hex)
        process=subprocess.run([sys.executable,'-S',str(script),'shapes','--output',str(output)],
            capture_output=True,text=True,timeout=15)
        self.assertEqual(process.returncode,2)
        self.assertIn('outside the source tree',process.stderr)
        self.assertFalse(output.exists())

    def test_physics_worker_without_site_packages_retains_unavailable_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'physics'
            script=Path(__file__).with_name('native_physics.py')
            process=subprocess.run([sys.executable,'-S',str(script),'gas',
                '--output',str(output),'--seed','42'],capture_output=True,text=True,timeout=15)
            self.assertEqual(process.returncode,0,process.stderr)
            record=json.loads((output/'receipt.json').read_text())
            self.assertEqual(record['status'],'UNAVAILABLE')
            self.assertIn('optional_tests/native_physics.py',record['source_sha256'])
            self.assertFalse((output/'arrays.npz').exists())

    def test_observed_calls_keep_imports_and_unused_definitions_separate(self):
        import importlib.util
        from optional_tests.execution import PythonCalls
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            path=root/'worker.py'
            path.write_text('def measured(x):\n    return x * x\n\ndef unused():\n    return 99\n\nclass Unused:\n    pass\n')
            before=sys.getprofile()
            with PythonCalls(root) as observed:
                spec=importlib.util.spec_from_file_location('proof_worker',path)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.assertEqual(module.measured(3),9)
                self.assertEqual(module.measured(4),16)
            self.assertIs(sys.getprofile(),before)
            report=observed.report();row=report['files']['worker.py']
            self.assertEqual(row['calls'],{'measured:1':2})
            self.assertEqual(row['module_events'],1)
            self.assertEqual(row['definition_events'],1)
            self.assertTrue(row['source_unchanged'])
            path.write_text('changed source\n')
            self.assertFalse(observed.report()['files']['worker.py']['source_unchanged'])

    def test_library_worker_without_site_packages_records_unavailable(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'worker'
            script=Path(__file__).with_name('software_checks.py')
            process=subprocess.run([sys.executable,'-S',str(script),'cloudpickle',
                '--output',str(output),'--seed','42'],capture_output=True,text=True,timeout=15)
            self.assertEqual(process.returncode,0,process.stderr)
            record=json.loads((output/'receipt.json').read_text())
            self.assertEqual(record['status'],'UNAVAILABLE')
            self.assertFalse((output/'arrays.npz').exists())

    def test_library_failure_retains_log_and_does_not_stop_later_steps(self):
        with tempfile.TemporaryDirectory() as name:
            run=LabRun(Path(name)/'results',profile='libraries')
            def failed_worker(*args,**kwargs):
                destination=run.output/'libraries'/'cloudpickle';destination.mkdir()
                (destination/'receipt.json').write_text(json.dumps({'status':'FAIL','reason':'test native exit','backend':'cloudpickle'}))
                kwargs['stdout'].write('retained backend failure\n')
                return subprocess.CompletedProcess(args[0],1)
            with patch('optional_tests.lab.subprocess.run',side_effect=failed_worker):
                run.step('library-cloudpickle','serialization',['cloudpickle'],[],[library_control],lambda:library_control(run,'cloudpickle'))
            run.step('later','later check',[],[],[library_control],lambda:{'executed':True})
            state=run.finish()
            self.assertEqual(state['status'],'FAILED')
            self.assertEqual([row['status'] for row in state['steps']],['FAIL','PASS'])
            from optional_tests.software_checks import CHECKS
            self.assertEqual(state['total_steps'],7+len(CHECKS))
            self.assertTrue(any(a['kind']=='log' for a in state['artifacts']))
            self.assertEqual(next(s for s in state['software'] if s['module']=='cloudpickle')['execution'],'CALCULATION_FAIL')

    def test_paths_reject_escape_and_absolute(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve()
            for bad in ['../private','/etc/passwd','']:
                with self.subTest(bad=bad),self.assertRaises(ValueError):confined(root,bad)
            external=root.parent/'outside-example'
            (root/'escape').symlink_to(root.parent,target_is_directory=True)
            with self.assertRaises(ValueError):confined(root,'escape/outside-example')
            self.assertEqual(confined(root,'safe/data.json'),root/'safe/data.json')

    def test_inventory_does_not_turn_inspection_into_execution(self):
        data=build_catalogue()
        self.assertEqual(len(data['graphs']),102)
        self.assertEqual(len({(g['family'],g['variant']) for g in data['graphs']}),102)
        for graph in data['graphs']:
            renderer, = [row for row in graph['implementation_route'] if row['role'] == 'renderer']
            self.assertGreater(renderer['line'], 1)
            self.assertTrue((Path(__file__).resolve().parents[1] / renderer['path']).is_file())
        self.assertTrue(all(s['status']=='NOT_EXECUTED_BY_THIS_LAB' for s in data['sources']))
        self.assertTrue(any(s['path']=='optional_tests/lab.py' for s in data['sources']))

    def test_archived_calls_do_not_change_live_execution_status(self):
        with tempfile.TemporaryDirectory() as name:
            here = Path(name)
            reference = here / 'reference_gallery'
            reference.mkdir()
            (reference / 'regression-calls.json').write_text(json.dumps({
                'scope': 'historical test trace',
                'original_sha256': 'a' * 64,
                'files': {'core/analysis.py': {'calls': {'ngp_density:1': 3}, 'tests': ['test_density']}},
            }))
            with patch('optional_tests.catalogue.HERE', here):
                sources = {row['path']: row for row in build_catalogue()['sources']}
        row = sources['core/analysis.py']
        self.assertEqual(row['status'], 'NOT_EXECUTED_BY_THIS_LAB')
        self.assertEqual(row['archived_regression']['calls'], {'ngp_density:1': 3})
        self.assertNotIn('archived_regression', sources['commands.py'])

    def test_campaign_import_uses_completed_merged_trace(self):
        from optional_tests.build_assets import import_campaign
        from optional_tests.catalogue import sha
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            evidence = base / 'evidence'; evidence.mkdir()
            output = base / 'optional'; output.mkdir()
            (evidence / 'graph-coverage.json').write_text(json.dumps({'entries': []}))
            def trace(label, count):
                path = evidence / ('python-execution-' + label + '.json')
                path.write_text(json.dumps({'files': {'core/analysis.py': {
                    'calls': {'measured:1': count}, 'tests': ['fixture_' + label]}}}))
                return path
            trace('refresh', 1)
            (evidence / 'regression-refresh-source-hashes.json').write_text('{}')
            selected = trace('complete', 7)
            summary = evidence / 'retained-tests-complete-summary.json'
            summary.write_text(json.dumps({'status': 'PASS', 'suites': ['application', 'verification'],
                                           'source_sha256': {'core/analysis.py': 'b' * 64}}))
            trace('pending', 999)
            (evidence / 'retained-tests-pending-summary.json').write_text(json.dumps({
                'status': 'RUNNING', 'suites': ['verification'], 'source_sha256': {}}))
            with patch('optional_tests.build_assets.HERE', output), patch(
                    'optional_tests.build_assets.source_inventory', return_value=[{'path': 'core/analysis.py'}]):
                self.assertEqual(import_campaign(evidence), 0)
            archived = json.loads((output / 'reference_gallery/regression-calls.json').read_text())
            self.assertEqual(archived['original_sha256'], sha(selected))
            self.assertEqual(archived['execution_summary_sha256'], sha(summary))
            row = archived['files']['core/analysis.py']
            self.assertEqual(row['calls'], {'measured:1': 7})
            self.assertEqual(row['execution_source_sha256'], 'b' * 64)

    def test_loopback_authority_and_mutation_token(self):
        with tempfile.TemporaryDirectory() as name:
            store=JobStore(Path(name)/'jobs')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(store,'secret','unused'))
            port=server.server_address[1];host=f'127.0.0.1:{port}'
            server.RequestHandlerClass=make_handler(store,'secret',host)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def request(method,path,body=None,headers=None):
                conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
                conn.request(method,path,body,headers or {});response=conn.getresponse()
                status=response.status;payload=json.loads(response.read());conn.close();return status,payload
            try:
                status,_=request('GET','/api/bootstrap',headers={'Host':'attacker.example'})
                self.assertEqual(status,403)
                status,data=request('GET','/api/bootstrap');self.assertEqual(status,200)
                self.assertEqual(data['token'],'secret')
                status,source=request('GET','/source?path=optional_tests/lab.py&line=1')
                self.assertEqual(status,200)
                from optional_tests.catalogue import sha
                self.assertEqual(source['sha256'],sha(Path(__file__).with_name('lab.py')))
                body=json.dumps({'profile':'quick','particles':32,'seed':42})
                status,_=request('POST','/api/run',body,{'Content-Type':'application/json'})
                self.assertEqual(status,403)
                status,_=request('POST','/api/run',body,{'X-Lab-Token':'secret','Origin':'https://attacker.example'})
                self.assertEqual(status,403)
                status,_=request('GET','/source?path=../private.py');self.assertEqual(status,403)
                status,_=request('POST','/api/run',json.dumps({'profile':'shell','particles':32,'seed':42}),{'X-Lab-Token':'secret'})
                self.assertEqual(status,400)
                self.assertFalse(store.jobs)
            finally:server.shutdown();server.server_close();thread.join(timeout=3)

if __name__=='__main__':unittest.main()
