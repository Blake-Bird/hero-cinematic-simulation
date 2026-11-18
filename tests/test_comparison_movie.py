import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from core.comparison_movie import make_comparison_movie


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComparisonMovieTests(unittest.TestCase):
    def test_chronology_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';root.mkdir();files=[];figures=[]
            for i,z in enumerate([0.,1.]):
                numeric=root/('frame%d.hdf5'%i)
                with h5py.File(numeric,'w') as h:
                    h['measured_a']=np.ones((4,4))
                    h['measured_b']=np.ones((4,4))
                    h['log10_ratio_b_over_a']=np.zeros((4,4))
                figure=root/('frame%d.figure.json'%i)
                figure.write_text(json.dumps({'scientific_metadata':{'numeric_asset':numeric.name,
                    'redshift_a':z,'redshift_b':z,'snapshot_a':i,'snapshot_b':i,
                    'field':'dm','projection':'xy'}}))
                files += [numeric,figure];figures.append('frame%d.png'%i)
            summary=root/'comparison_maps_summary.json'
            summary.write_text(json.dumps({'box_mpc_h':1.,'maps':{'dm_xy':{'status':'COMPLETE',
                'fixed_color_scale_across_matched_redshifts':True,'figure_files':figures,
                'vmin':-1.,'vmax':1.,'log10_ratio_limit':1.}}}))
            files.append(summary)
            (root/'comparison_contract.json').write_text(json.dumps({'schema':'comparison-map-publication-v1',
                'provenance_by_run':[{'input_hashes':[]},{'input_hashes':[]}],
                'artifacts':[{'path':p.name,'sha256':digest(p)} for p in files]}))
            out=Path(temp)/'movie'
            with patch('core.foundation.verify_product_provenance') as verify:
                movie=make_comparison_movie(root,out,map_key='dm_xy',fps=1,runs=[temp,temp])
                self.assertEqual(verify.call_count,4)
            receipt=json.loads((out/'movie_contract.json').read_text())
            self.assertEqual([x['redshift_a'] for x in receipt['frames']],[1.,0.])
            with imageio.get_reader(movie) as reader:self.assertEqual(reader.count_frames(),2)
            (root/'frame0.hdf5').write_bytes(b'changed')
            with patch('core.foundation.verify_product_provenance'):
                with self.assertRaisesRegex(ValueError,'changed'):
                    make_comparison_movie(root,Path(temp)/'bad',map_key='dm_xy',fps=1,runs=[temp,temp])

    def test_rejects_missing_and_changed_source_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';root.mkdir();contract=root/'comparison_contract.json'
            contract.write_text(json.dumps({'schema':'comparison-map-publication-v1','artifacts':[]}))
            with self.assertRaisesRegex(ValueError,'Two source-run provenance'):
                make_comparison_movie(root,Path(temp)/'out',map_key='dm_xy',runs=[temp,temp])
            contract.write_text(json.dumps({'schema':'comparison-map-publication-v1','provenance_by_run':[{},{}],'artifacts':[]}))
            with patch('core.foundation.verify_product_provenance',side_effect=ValueError('source changed')):
                with self.assertRaisesRegex(ValueError,'source changed'):
                    make_comparison_movie(root,Path(temp)/'out',map_key='dm_xy',runs=[temp,temp])


if __name__ == '__main__':
    unittest.main()
