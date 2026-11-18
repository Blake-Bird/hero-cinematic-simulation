import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from core.analysis import _initial_condition_paths,compare_initial_ic_phases


class InitialConditionHandoffTests(unittest.TestCase):
    def make_run(self,root):
        p=root/'native_000.hdf5'
        with h5py.File(p,'w') as f:
            h=f.create_group('Header');h.attrs['NumFilesPerSnapshot']=1;h.attrs['NumPart_Total']=np.array([0,8,0,0,0,0],dtype='u4');h.attrs['NumPart_ThisFile']=np.array([0,8,0,0,0,0],dtype='u4')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        (root/'manifest.json').write_text(json.dumps({'gadget_handoff':{'initial_conditions':[{'path':p.name,'sha256':digest}]}}))
        return p

    def test_reads_verified_handoff_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_run(root)
            self.assertEqual(_initial_condition_paths(root),[p.resolve()])
            with h5py.File(p,'a') as f:f.attrs['modified']=1
            with self.assertRaisesRegex(ValueError,'missing or changed'):_initial_condition_paths(root)
            cfg={'simulation':{'particles_per_dimension':2,'box_mpc_h':1}}
            result=compare_initial_ic_phases(root,root,cfg,cfg)
            self.assertEqual(result['status'],'FAIL');self.assertIn('provenance',result['reason'])

    def test_rejects_incomplete_handoff_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_run(root)
            with h5py.File(p,'a') as f:f['Header'].attrs['NumFilesPerSnapshot']=2
            manifest=json.loads((root/'manifest.json').read_text());manifest['gadget_handoff']['initial_conditions'][0]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'incomplete'):_initial_condition_paths(root)
