import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.batch_analysis import run_batch_analysis


class BatchAnalysisTests(unittest.TestCase):
    def test_sequential_runs_and_reference_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve();runs=[p/'a',p/'b',p/'c']
            for r in runs:r.mkdir();(r/'config.yaml').write_text('{}')
            def run(root,products,**kwargs):
                if root==runs[1]:raise ValueError('missing data')
                return {'status':'COMPLETE','product_status':'COMPLETE','artifacts':{},'unavailable':{}}
            with patch('commands.process_run',side_effect=run) as process,patch('core.visualization.make_comparison_maps',return_value=p/'comparison') as compare:
                result=run_batch_analysis(runs,['hmf'],p/'out')
            self.assertEqual([x['status'] for x in result['runs']],['COMPLETE','FAILED','COMPLETE'])
            self.assertEqual([x['status'] for x in result['comparisons']],['SKIPPED','COMPLETE'])
            self.assertEqual(process.call_count,3);self.assertEqual(compare.call_count,1);self.assertEqual(result['status'],'INCOMPLETE')
            self.assertTrue((p/'out/batch_report.json').exists())

    def test_rejects_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'config.yaml').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unique'):run_batch_analysis([p,p],['hmf'],p/'out')
