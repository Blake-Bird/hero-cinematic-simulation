import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.merger_tree_visual import build_merger_tree, export_merger_tree
from core.halo_formation_movie import tree_layout


def catalog(snapshot, scale, ids, descendants, masses, desc_scales):
    return {'snapshot':snapshot, 'scale':scale, 'redshift':1/scale-1,
            'ids':np.array(ids, dtype=np.int64), 'desc_ids':np.array(descendants, dtype=np.int64),
            'mass':np.array(masses, dtype=float), 'valid':np.ones(len(ids), dtype=bool),
            'desc_scale':np.array(desc_scales), 'source':'synthetic exact links'}


class MergerTreeVisualTests(unittest.TestCase):
    def setUp(self):
        self.catalogs = [
            catalog(0,.25,[101,102,103,104],[201,201,202,201],[5,3,2,1],[.5,.5,.5,1.]),
            catalog(1,.5,[201,202],[9007199254740993,9007199254740993],[9,4],[1.,1.]),
            catalog(2,1.,[9007199254740993],[-1],[15],[np.nan]),
        ]

    def test_branches_are_linked_and_skipped_epochs_are_not_invented(self):
        tree = build_merger_tree(self.catalogs, 0)
        self.assertEqual({n['id'] for n in tree['nodes']},
                         {'101','102','103','201','202','9007199254740993'})
        self.assertEqual(len(tree['edges']), 5)
        self.assertFalse(tree['truncated'])
        with tempfile.TemporaryDirectory() as temp:
            path = export_merger_tree(tree,temp)
            self.assertIn('9007199254740993',path.read_text())
            self.assertIn('Select a halo',path.read_text())
            self.assertTrue((Path(temp)/'merger_tree.json').exists())

    def test_node_limit_is_explicit(self):
        tree = build_merger_tree(self.catalogs, 0, max_nodes=2)
        self.assertTrue(tree['truncated'])
        self.assertEqual(len(tree['nodes']),2)
        with tempfile.TemporaryDirectory() as temp:
            self.assertIn('displayed tree is incomplete',export_merger_tree(tree,temp).read_text())

    def test_saved_position_and_tree_layout(self):
        self.catalogs[-1]['position_mpc_h']=np.array([[.1,.2,.3]])
        tree=build_merger_tree(self.catalogs,0,box_mpc_h=1.)
        self.assertEqual(tree['nodes'][0]['position_mpc_h'],[.1,.2,.3])
        self.assertEqual(len(tree_layout(tree)),len(tree['nodes']))
        with tempfile.TemporaryDirectory() as temp:
            page=export_merger_tree(tree,temp).read_text()
            self.assertIn('data-x="10"',page)
            self.assertIn('Saved XY box position',page)


if __name__ == '__main__':
    unittest.main()
