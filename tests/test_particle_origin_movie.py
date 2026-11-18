import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from core.particle_origin_movie import stable_cohort,matching_positions


class ParticleOriginTests(unittest.TestCase):
    def test_exact_id_selection_is_stable_and_matching_handles_missing(self):
        ids=np.array([2**63+1,9,2**53+1,5],dtype=np.uint64)
        chosen=stable_cohort(ids,3)
        np.testing.assert_array_equal(chosen,stable_cohort(ids[::-1],3))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'snapshot.hdf5'
            with h5py.File(path,'w') as f:
                g=f.create_group('PartType1');g['ParticleIDs']=ids[:3];g['Coordinates']=np.arange(9).reshape(3,3)
            found,pos=matching_positions([path],np.sort(ids),chunk_rows=2)
            self.assertEqual(found.tolist(),sorted(ids[:3].tolist()))
            self.assertEqual(pos.shape,(3,3))
            self.assertEqual(pos[np.flatnonzero(found==2**63+1)[0]].tolist(),[0,1,2])
            with h5py.File(path,'r+') as f:f['PartType1/ParticleIDs'][1]=ids[0]
            with self.assertRaisesRegex(ValueError,'Duplicate'):matching_positions([path],np.sort(ids),chunk_rows=2)


if __name__=='__main__':unittest.main()
