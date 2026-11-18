import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.select_merger_tree import export_selected_tree,verify_tree_bundle


class SelectedTreeTests(unittest.TestCase):
    def test_selected_exact_id_and_source_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';root.mkdir();analysis=root/'analysis';analysis.mkdir()
            (analysis/'analysis_products_summary.json').write_text(json.dumps({'snapshots':{'0':{'redshift':1},'1':{'redshift':0}}}))
            (analysis/'analysis_provenance.json').write_text('{}')
            big='9007199254740993'
            tree={'schema':'adjacent-merger-tree-v1','nodes':[{'id':big,'snapshot':1,'redshift':0.,'scale_factor':1.,'mass_msun_h':10.,'epoch_index':1},{'id':'4','snapshot':0,'redshift':1.,'scale_factor':.5,'mass_msun_h':5.,'epoch_index':0}],'edges':[[1,0]],'box_mpc_h':1.,'source':'fixture','truncated':False,'max_nodes':10,'scope':'fixture'}
            history={'status':'COMPLETE','_merger_tree':tree,'final_snapshot':1,'particle_floor':1,'source':'fixture'}
            with patch('core.analysis.verify_analysis_bundle',return_value={'source_run':str(root)}) as verify,patch('core.analysis.load_cfg',return_value={}),patch('core.analysis.main_progenitor_histories',return_value=history) as measure:
                path=export_selected_tree(root,analysis,Path(temp)/'out',halo_id=big)
                self.assertEqual(measure.call_args.kwargs['final_halo_id'],int(big));self.assertEqual(verify.call_count,2)
            self.assertIn(big,path.read_text());self.assertEqual(json.loads((path.parent/'selected_tree_contract.json').read_text())['halo_id'],big)
            with patch('core.analysis.verify_analysis_bundle',return_value={'source_run':str(root),'data_contract':{}}):
                self.assertEqual(verify_tree_bundle(path.parent)['source_run'],str(root))
                (path.parent/'merger_tree.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'missing or changed'):verify_tree_bundle(path.parent)
            with self.assertRaises(ValueError):export_selected_tree(root,analysis,Path(temp)/'bad',halo_id='1.5')
