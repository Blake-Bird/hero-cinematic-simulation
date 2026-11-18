import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from core.resolution_reveal import make_resolution_reveal


def make_bundle(root,grid,z):
    root.mkdir();files=[]
    scale=root/'scale.json';scale.write_text(json.dumps({'maps':{'dm_xy':{'status':'COMPLETE'}}}));files.append(scale)
    data=root/'dm_xy_000.hdf5'
    with h5py.File(data,'w') as f:f['log10_display']=np.arange(grid*grid).reshape(grid,grid)/100
    meta={'snapshot':0,'redshift':z,'field':'dm','projection':'xy','box_size_mpc_h':10.,'overlay_layers':[],
          'measured_quantity_units':'h Msun / Mpc^2','slice':{'thickness_mpc_h':None,'center_mpc_h':None,'line_of_sight_axis':2,'assignment_kernel':'CIC'},
          'particles_per_dimension':grid,'particle_mass_msun_h':100.,'numeric_asset':data.name}
    figure=root/'dm_xy_000.figure.json';figure.write_text(json.dumps({'scientific_metadata':meta}));files.extend([data,figure])
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'map_contract.json').write_text(json.dumps({'schema':'map-publication-v1','provenance':{},'artifacts':[{'path':p.name,'sha256':digest(p)} for p in files]}))


class ResolutionRevealTests(unittest.TestCase):
    def test_different_grids_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'a';b=p/'b';a.mkdir();b.mkdir();ma=a/'maps';mb=b/'maps';make_bundle(ma,8,1.);make_bundle(mb,16,1.)
            with patch('core.foundation.verify_product_provenance'):
                page=make_resolution_reveal(a,ma,0,b,mb,0,p/'result',map_key='dm_xy')
                self.assertIn('type="range"',page.read_text())
                r=json.loads((page.parent/'resolution_contract.json').read_text())
                self.assertEqual([row['grid_shape'] for row in r['inputs']],[[8,8],[16,16]])
                (mb/'dm_xy_000.hdf5').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):
                    make_resolution_reveal(a,ma,0,b,mb,0,p/'bad',map_key='dm_xy')

    def test_mismatched_epoch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'a';b=p/'b';a.mkdir();b.mkdir();ma=a/'maps';mb=b/'maps';make_bundle(ma,8,1.);make_bundle(mb,8,2.)
            with patch('core.foundation.verify_product_provenance'):
                with self.assertRaisesRegex(ValueError,'redshift'):
                    make_resolution_reveal(a,ma,0,b,mb,0,p/'bad',map_key='dm_xy')


if __name__=='__main__':unittest.main()
