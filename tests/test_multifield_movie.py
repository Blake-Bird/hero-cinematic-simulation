import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np
from core.multifield_movie import make_multifield_movie


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class MultifieldMovieTests(unittest.TestCase):
    def test_synchronizes_epochs_and_records_missing_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'maps';root.mkdir();files=[]
            scale=root/'scale.json';scale.write_text(json.dumps({'maps':{k:{'status':'COMPLETE','normalization':'global_fixed','vmin':0,'vmax':2} for k in ('dm_xy','gas_xy')}}));files.append(scale)
            for field,snap,z in [('dm',0,2.),('dm',1,1.),('gas',1,1.)]:
                prefix='%s_xy_%03d'%(field,snap);data=root/(prefix+'.hdf5')
                with h5py.File(data,'w') as h:h['log10_display']=np.ones((4,4))
                figure=root/(prefix+'.figure.json');figure.write_text(json.dumps({'scientific_metadata':{
                    'snapshot':snap,'redshift':z,'field':field,'projection':'xy','box_size_mpc_h':1.,'density_colormap':'magma',
                    'slice':{'thickness_mpc_h':None,'center_mpc_h':None,'line_of_sight_axis':2,'assignment_kernel':'CIC'},
                    'overlay_layers':[],'overlays':{},'numeric_asset':data.name}}));files.extend([data,figure])
            (root/'map_contract.json').write_text(json.dumps({'schema':'map-publication-v1','provenance':{'input_hashes':[]},'artifacts':[{'path':p.name,'sha256':digest(p)} for p in files]}))
            out=Path(temp)/'out'
            with patch('core.foundation.verify_product_provenance') as verify:
                movie=make_multifield_movie(root,out,map_keys=['dm_xy','gas_xy'],fps=1,run_dir=temp)
                self.assertEqual(verify.call_count,2)
            report=json.loads((out/'movie_contract.json').read_text())
            self.assertEqual([e['snapshot'] for e in report['epochs']],[0,1])
            self.assertIsNone(report['epochs'][0]['fields']['gas_xy'])
            with imageio.get_reader(movie) as reader:self.assertEqual(reader.count_frames(),2)
            (root/'dm_xy_000.hdf5').write_bytes(b'changed')
            with patch('core.foundation.verify_product_provenance'):
                with self.assertRaisesRegex(ValueError,'changed'):make_multifield_movie(root,Path(temp)/'bad',map_keys=['dm_xy','gas_xy'],fps=1,run_dir=temp)

    def test_rejects_unbound_or_changed_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'maps';root.mkdir();contract=root/'map_contract.json'
            contract.write_text(json.dumps({'schema':'map-publication-v1','artifacts':[]}))
            with self.assertRaisesRegex(ValueError,'lacks simulation provenance'):
                make_multifield_movie(root,Path(temp)/'bad',map_keys=['dm','gas'],run_dir=temp)
            contract.write_text(json.dumps({'schema':'map-publication-v1','provenance':{'input_hashes':[]},'artifacts':[]}))
            with patch('core.foundation.verify_product_provenance',side_effect=ValueError('source changed')):
                with self.assertRaisesRegex(ValueError,'source changed'):
                    make_multifield_movie(root,Path(temp)/'bad',map_keys=['dm','gas'],run_dir=temp)


if __name__=='__main__':unittest.main()
