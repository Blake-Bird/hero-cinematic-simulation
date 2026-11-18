import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import imageio.v2 as imageio
from core.halo_collision_movie import select_merger_pair, make_halo_collision_movie


def node(identity,snapshot,scale,position,mass,velocity):
    return {'id':str(identity),'snapshot':snapshot,'scale_factor':scale,'redshift':1/scale-1,
            'position_mpc_h':position,'mass_msun_h':mass,'velocity_peculiar_km_s':velocity}


class HaloCollisionTests(unittest.TestCase):
    def test_linked_branch_selection_and_periodic_separation(self):
        tree={'box_mpc_h':10.,'nodes':[node(10,2,1.,[0,0,0],30,[0,0,0]),
             node(11,1,.5,[.1,1,1],10,[20,0,0]),node(12,1,.5,[9.9,1,1],5,[0,0,0]),
             node(13,0,.25,[.2,1,1],8,[10,0,0]),node(14,0,.25,[9.8,1,1],4,[0,0,0])],
             'edges':[[1,0],[2,0],[3,1],[4,2]]}
        selected=select_merger_pair(tree,'10')
        self.assertEqual([r['snapshot'] for r in selected['rows']],[0,1])
        self.assertAlmostEqual(selected['rows'][0]['separation_comoving_mpc_h'],.4)
        self.assertAlmostEqual(selected['rows'][1]['separation_comoving_mpc_h'],.2)
        self.assertEqual(selected['rows'][1]['relative_peculiar_speed_km_s'],20.)
        with self.assertRaisesRegex(ValueError,'No linked pair'):select_merger_pair(tree,'999')
        tree['nodes'][4].pop('position_mpc_h')
        with self.assertRaisesRegex(ValueError,'two common'):select_merger_pair(tree)

    def test_movie_from_synthetic_linked_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);analysis=base/'analysis';analysis.mkdir();maps=base/'maps';maps.mkdir()
            tree={'box_mpc_h':10.,'nodes':[node(10,2,1.,[0,0,0],30,[0,0,0]),
                node(11,1,.5,[.1,1,1],10,[20,0,0]),node(12,1,.5,[9.9,1,1],5,[0,0,0]),
                node(13,0,.25,[.2,1,1],8,[10,0,0]),node(14,0,.25,[9.8,1,1],4,[0,0,0])],
                'edges':[[1,0],[2,0],[3,1],[4,2]]}
            (analysis/'merger_tree.json').write_text(json.dumps(tree));files=[]
            scale=maps/'scale.json';scale.write_text(json.dumps({'maps':{'dm_xy':{'status':'COMPLETE','normalization':'global_fixed','vmin':0,'vmax':1}}}));files.append(scale)
            for snap,z in [(0,3.),(1,1.)]:
                data=maps/('dm_xy_%03d.hdf5'%snap)
                with h5py.File(data,'w') as f:f['log10_display']=np.ones((8,8))*.5
                figure=maps/('dm_xy_%03d.figure.json'%snap)
                figure.write_text(json.dumps({'scientific_metadata':{'snapshot':snap,'redshift':z,'box_size_mpc_h':10.,
                    'projection':'xy','slice':{'thickness_mpc_h':None},'density_colormap':'magma',
                    'overlay_layers':[],'numeric_asset':data.name}}));files.extend([data,figure])
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (maps/'map_contract.json').write_text(json.dumps({'schema':'map-publication-v1',
                'provenance':{'run_namespace':'control'},'artifacts':[{'path':p.name,'sha256':digest(p)} for p in files]}))
            report={'source_run':str(base),'data_contract':{'run_namespace':'control'}}
            with patch('core.analysis.verify_analysis_bundle',return_value=report),patch('core.foundation.verify_product_provenance'):
                movie=make_halo_collision_movie(analysis,maps,base/'movie',map_key='dm_xy',fps=1)
            receipt=json.loads((base/'movie/movie_contract.json').read_text())
            self.assertEqual([r['snapshot'] for r in receipt['frames']],[0,1])
            with imageio.get_reader(movie) as reader:self.assertEqual(reader.count_frames(),2)


if __name__=='__main__':unittest.main()
