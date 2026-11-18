import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
from core.timeline_wall import make_timeline_wall


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class TimelineWallTests(unittest.TestCase):
    def test_saved_epoch_order_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'maps';root.mkdir();files=[]
            scale=root/'scale.json';scale.write_text(json.dumps({'maps':{'dm_xy':{'status':'COMPLETE','normalization':'global_fixed','vmin':0,'vmax':2}}}));files.append(scale)
            for i,z in enumerate([0.,2.,1.]):
                data=root/('dm_xy_%03d.hdf5'%i)
                with h5py.File(data,'w') as h:h['log10_display']=np.full((4,4),i,dtype=float)
                figure=root/('dm_xy_%03d.figure.json'%i)
                figure.write_text(json.dumps({'scientific_metadata':{'snapshot':i,'redshift':z,'field':'dm','projection':'xy',
                    'box_size_mpc_h':1.,'density_colormap':'magma','slice':{'thickness_mpc_h':None,'center_mpc_h':None,
                    'line_of_sight_axis':2,'assignment_kernel':'CIC','deposited_mass_msun_h':i},'overlay_layers':[],
                    'overlays':{},'numeric_asset':data.name}}))
                files.extend([figure,data])
            (root/'map_contract.json').write_text(json.dumps({'schema':'map-publication-v1','provenance':{'input_hashes':[]},'artifacts':[{'path':p.name,'sha256':digest(p)} for p in files]}))
            out=Path(temp)/'wall'
            with patch('core.foundation.verify_product_provenance') as verify:
                self.assertTrue(make_timeline_wall(root,out,map_key='dm_xy',run_dir=temp).exists())
                self.assertEqual(verify.call_count,2)
            receipt=json.loads((out/'timeline_contract.json').read_text())
            self.assertEqual([r['redshift'] for r in receipt['epochs']],[2.,1.,0.])
            self.assertTrue((out/'cosmic_timeline.pdf').is_file())
            (root/'dm_xy_000.hdf5').write_bytes(b'corrupt')
            with patch('core.foundation.verify_product_provenance'):
                with self.assertRaisesRegex(ValueError,'changed'):make_timeline_wall(root,Path(temp)/'bad',map_key='dm_xy',run_dir=temp)


if __name__ == '__main__':unittest.main()
