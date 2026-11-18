import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import yaml
from core.parameter_explorer import make_parameter_explorer,parameter_value


def fixture(root,value,has_hmf):
    root.mkdir();(root/'config.yaml').write_text(yaml.safe_dump({'cosmology':{'H0':value}}));maps=root/'maps';maps.mkdir();analysis=root/'analysis';analysis.mkdir()
    files=[];scale=maps/'scale.json';scale.write_text(json.dumps({'maps':{'dm_xy':{'status':'COMPLETE'}}}));files.append(scale)
    data=maps/'dm_xy_000.hdf5'
    with h5py.File(data,'w') as h:h['log10_display']=np.ones((8,8))
    meta={'field':'dm','projection':'xy','measured_quantity_units':'h Msun/Mpc²','snapshot':0,'redshift':1.,'box_size_mpc_h':10.,
          'slice':{'thickness_mpc_h':None,'center_mpc_h':None},'overlay_layers':[],'particles_per_dimension':8,
          'particle_mass_msun_h':100.,'numeric_asset':data.name}
    fig=maps/'dm_xy_000.figure.json';fig.write_text(json.dumps({'scientific_metadata':meta}));files.extend([data,fig])
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (maps/'map_contract.json').write_text(json.dumps({'schema':'map-publication-v1','provenance':{'run_namespace':root.name},
        'artifacts':[{'path':p.name,'sha256':digest(p)} for p in files]}))
    (analysis/'analysis_products_summary.json').write_text(json.dumps({'snapshots':{'0':{'redshift':1.,'epoch_contract':{'cosmic_time_gyr':5.}}}}))
    (analysis/'analysis_provenance.json').write_text('{}')
    with h5py.File(analysis/'analysis_products.h5','w') as h:
        if has_hmf:
            g=h.create_group('snap_000/hmf/data');g['mass_msun_h']=[1.,2.];g['dn_dlogM']=[.1,0.];g['counts']=np.array([3,1],dtype='u8');g['status']=np.bytes_('COMPLETE')
    return {'run':str(root),'map_bundle':str(maps),'map_key':'dm_xy','snapshot':0,'analysis_bundle':str(analysis)}


class ParameterExplorerTests(unittest.TestCase):
    def test_saved_runs_hmf_missingness_and_safe_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=fixture(p/'a',70.,True);b=fixture(p/'b','<script>',False)
            request={'schema':'parameter-explorer-request-v1','parameter':'cosmology.H0','runs':[a,b]}
            def verify(path,verify_sources):return {'source_run':str(Path(path).parent),'data_contract':{'run_namespace':Path(path).parent.name}}
            with patch('core.analysis.verify_analysis_bundle',side_effect=verify),patch('core.foundation.verify_product_provenance'):
                page=make_parameter_explorer(request,p/'explorer')
            text=page.read_text();self.assertIn('&lt;script&gt;',text);self.assertNotIn('<script> =',text)
            report=json.loads((page.parent/'explorer_contract.json').read_text())
            self.assertEqual(report['runs'][0]['binned_halo_count'],4)
            self.assertIsNone(report['runs'][1]['binned_halo_count'])
            self.assertTrue((page.parent/'hmf_000.png').is_file())
            self.assertIn('No supported HMF',text)

    def test_parameter_missing_and_non_scalar(self):
        self.assertIsNone(parameter_value({'a':{}},'a.b'))
        with self.assertRaisesRegex(ValueError,'scalar'):parameter_value({'a':[1]},'a')


if __name__=='__main__':unittest.main()
