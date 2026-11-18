import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
from core.dome_movie import make_dome_movie


class DomeMovieTests(unittest.TestCase):
    def test_chronological_saved_frames_and_fixed_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir()
            def load(run,snap,field,max_particles,observer_mpc_h=None,return_arrays=False):
                coords=np.array([[.1,0,1.],[0,.2,1.],[-.2,0,1.]])
                weights=np.array([1.,2.,3.])*(snap+1)
                source={'epoch':{'actual_scale_factor':.5 if snap==0 else 1.},'data_contract':{},'temperature_source':None}
                arrays={'particle_ids':np.array([1,2,3],dtype=np.uint64),'represented_mass_msun_h':weights}
                return coords,weights,1.,1. if snap==0 else 0.,source,arrays
            with patch('core.analysis.snapshot_ids',return_value=[0,1]),patch('core.visualization.load_particles_for_fisheye',side_effect=load),patch('core.foundation.verify_product_provenance'):
                movie=make_dome_movie(root,Path(tmp)/'out',snapshots=[1,0],resolution=32,max_particles=3,fps=1)
            receipt=json.loads((movie.parent/'planetarium_contract.json').read_text())
            self.assertEqual([x['snapshot'] for x in receipt['epochs']],[0,1])
            self.assertLess(receipt['display_log10_limits'][0],receipt['display_log10_limits'][1])
            with imageio.get_reader(movie) as reader:
                self.assertEqual(reader.count_frames(),2)
                self.assertLessEqual(int(reader.get_data(0)[0,0].max()),5)
            for row in receipt['epochs']:
                with np.load(movie.parent/row['numeric_asset'],allow_pickle=False) as data:
                    self.assertEqual(data['particle_ids'].tolist(),[1,2,3])


if __name__=='__main__':unittest.main()
