"""Behavioral checks for epoch highlighting, replayable input and result navigation."""
import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.epochs import parse_schedule, highlight_redshifts, epoch_label
from core.setup_walkthrough import Questionnaire
from core.run_results import build_run_results
from core.foundation import atomic_json, file_sha256


class HighlightTests(unittest.TestCase):
    def test_stars_do_not_thin_movie_schedule(self):
        values, stars = parse_schedule('10, 5*, 2, 1*, 0*', 20)
        self.assertEqual(values, [10, 5, 2, 1, 0])
        self.assertEqual(stars, [5, 1, 0])
        self.assertEqual(highlight_redshifts('1:0:0.1', 1, '0.3,0'), [.3, 0])
        self.assertNotEqual(epoch_label(1, .00001), epoch_label(2, .00002))
        for text in ('nan', 'inf', '1:0:0.3', '1::1', '2*', '0**', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_schedule(text, 1)
        with self.assertRaises(ValueError):
            highlight_redshifts('1,0', 1, [.5])


class QuestionnaireTests(unittest.TestCase):
    def test_back_branch_changes_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'draft.json'
            def collect():
                mode = input('Mode: ')
                return {'mode': mode, 'value': input('A: ' if mode == 'a' else 'B: ')}
            original = builtins.input
            with patch('builtins.input', side_effect=['a', ':back', 'b', '17', 'yes']):
                self.assertEqual(Questionnaire(path, 'test').run(collect), {'mode': 'b', 'value': '17'})
            self.assertIs(builtins.input, original)
            with patch('builtins.input', side_effect=['yes']) as read:
                result = Questionnaire(path, 'test').run(collect)
            self.assertEqual(result['value'], '17')
            self.assertEqual(read.call_count, 1)

    def test_interruption_restores_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = builtins.input
            with patch('builtins.input', side_effect=[':quit']):
                with self.assertRaises(KeyboardInterrupt):
                    Questionnaire(Path(tmp)/'draft.json', 'x').run(lambda: input('Value: '))
            self.assertIs(builtins.input, original)


class RunNavigationTests(unittest.TestCase):
    def make_run(self, root):
        root.mkdir();(root/'config.yaml').write_text('simulation:\n  snapshot_redshift_list: [1.0, 0.0]\n  highlight_redshifts: [1.0]\n')
        folder = root/'results'/'analysis_a';folder.mkdir(parents=True)
        (folder/'figure.png').write_bytes(b'fixture')
        atomic_json(folder/'figure.figure.json', {'title': '<HMF>', 'scientific_metadata': {'snapshot': 0, 'redshift': 1.00001, 'product': 'hmf'},
             'artifacts': [{'path': 'figure.png', 'sha256': file_sha256(folder/'figure.png')}]})
        atomic_json(folder/'analysis_products_summary.json', {'snapshots': {'0': {'redshift': 1.00001}}, 'artifacts': ['figure.figure.json']})
        atomic_json(root/'results'/'workflow_20260101Z.json', {'run': str(root), 'status': 'COMPLETE', 'artifacts': {'analysis': str(folder)}})
        return folder

    def test_metadata_epochs_relocation_and_attempt_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();old=base/'old';self.make_run(old)
            atomic_json(old/'results'/'workflow_20260102Z.json', {'run': str(old), 'status': 'FAILED', 'artifacts': {}, 'failure': {'message': 'failed'}})
            root=base/'moved';old.rename(root)
            page=build_run_results(root, verify=True)
            report=json.loads((page.parent/'navigation.json').read_text())
            self.assertEqual(report['current_attempt'], 'workflow_20260102Z')
            self.assertEqual(report['attempts'][1]['items'], [])
            first=report['attempts'][0]
            self.assertEqual(first['items'][0]['snapshot'], 0)
            self.assertTrue(first['epochs'][0]['highlight'])
            self.assertFalse(report['issues'])
            epoch=page.parent/'navigation'/'workflow_20260101Z'/'snap_000_z_1.000'/'index.html'
            self.assertTrue(epoch.is_file())
            self.assertIn('&lt;HMF&gt;', epoch.read_text())
            self.assertNotIn(str(old), page.read_text())
            (root/'results'/'analysis_a'/'figure.png').write_bytes(b'changed')
            build_run_results(root, verify=True)
            report=json.loads((page.parent/'navigation.json').read_text())
            self.assertTrue(any('checksum' in row['reason'] for row in report['issues']))

    def test_path_escape_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'run';self.make_run(root)
            atomic_json(root/'results'/'workflow_20260102Z.json', {'run': str(root), 'status': 'COMPLETE', 'artifacts': {'movie': '../../outside.mp4'}})
            build_run_results(root)
            report=json.loads((root/'results'/'navigation.json').read_text())
            self.assertTrue(any('outside this run' in row['reason'] for row in report['issues']))

class ExecutionIsolationTests(unittest.TestCase):
    def test_optional_failure_does_not_prevent_visual_exports(self):
        import commands
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp).resolve();(run/'config.yaml').write_text('visualization: {}\n')
            status={'status':'RUNNING','product_status':'RUNNING','artifacts':{},'unavailable':{},'run':str(run),'products':['stellar_components','maps']}
            requests={key:None for key in ('stellar_search','stellar_history','zoom_regions','inference_diagnostics','radial_shapes')}
            requests['stellar_search']={'snapshot':0}
            def visuals(root, products, cfg, state, checkpoint, **kwargs):
                state['artifacts']['maps']='recorded-map.png'
            with patch('commands.validate_analysis_requests',return_value=requests), patch('commands.run_analysis_snapshot_ids',return_value=[0]), \
                 patch('halos.find_stellar_substructures',side_effect=ValueError('invalid host')), \
                 patch('core.visual_pipeline.execute_visual_products',side_effect=visuals) as render:
                result=commands._execute_run(run,['stellar_components','maps'],allow_incomplete=True,
                                            status=status,destination=run/'record.json')
            self.assertIn('invalid host',result['unavailable']['stellar_components'])
            self.assertEqual(result['artifacts']['maps'],'recorded-map.png')
            self.assertEqual(result['status'],'INCOMPLETE')
            render.assert_called_once()


class PreflightTests(unittest.TestCase):
    def test_insufficient_storage_blocks_before_execution(self):
        from core.setup_walkthrough import launch_review
        from collections import namedtuple
        usage=namedtuple('usage','total used free')(1,1,0)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve()
            cfg={'simulation':{'particles_per_dimension':32,'physics':'dm','snapshot_redshift_list':[1.,0.], 'pm_grid':64},'hardware':{'mpi_ranks':1}}
            with patch.dict('os.environ',{'RESEARCH2_STATE_ROOT':str(base/'state')}),patch('shutil.disk_usage',return_value=usage):
                with self.assertRaisesRegex(ValueError,'Resolve launch review blockers'):
                    launch_review(cfg,base/'run',['maps'],{})
            reviews=list((base/'state'/'drafts').glob('launch-review-*.json'))
            self.assertEqual(len(reviews),1)
            self.assertTrue(json.loads(reviews[0].read_text())['blocking'])

class DeferredRequestTests(unittest.TestCase):
    def test_explicit_deferral_runs_ready_products_and_stays_incomplete(self):
        import commands
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp).resolve();(run/'config.yaml').write_text('{}')
            plan={'decisions':{'maps':{'status':'APPLICABLE','reason':'ready'},'radial_shapes':{'status':'MISSING_INPUT','reason':'request required'}},
                  'selected':['maps'],'missing_inputs':{'radial_shapes':'request required'},'explicit_unavailable':{}}
            def execute(root, selected, **kwargs):
                self.assertEqual(selected,['maps'])
                state=kwargs['status'];self.assertIn('radial_shapes',state['unavailable'])
                state.update(status='INCOMPLETE',product_status='INCOMPLETE')
                return state
            with patch('commands.plan_saved_analysis',return_value=plan),patch('commands._execute_run',side_effect=execute):
                result=commands.process_run(run,['all'],deferred_products=['radial_shapes'],allow_incomplete=True)
            self.assertEqual(result['status'],'INCOMPLETE')
            self.assertEqual(result['requested_products'],['all'])
            self.assertEqual(result['product_states']['radial_shapes']['status'],'PENDING')
            self.assertTrue(result['retry_commands'])
            self.assertTrue((run/'results'/'index.html').exists())


class ConfigurationContractTests(unittest.TestCase):
    def test_frozen_output_settings_pass_native_validator(self):
        import yaml
        from core.evolution import validate_config_keys
        source=Path(__file__).resolve().parents[1]
        cfg=yaml.safe_load((source/'core/config.yaml').read_text())
        cfg['simulation'].update(snapshot_redshift_list=[5.,1.,0.], highlight_redshifts=[1.,0.])
        cfg['analysis'].update(requested_outputs=['all'], deferred_outputs=['radial_shapes'])
        self.assertTrue(validate_config_keys(cfg))
        cfg['simulation']['highlight_redshifts']=[2.]
        with self.assertRaisesRegex(ValueError,'highlight_redshifts'):
            validate_config_keys(cfg)


if __name__ == '__main__':
    unittest.main()
