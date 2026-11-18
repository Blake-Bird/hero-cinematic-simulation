"""Transferred GADGET group-ordered snapshots retain their shard numbering."""
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from core.analysis import validate_gadget_snapshot_set
from core.foundation import KPC_CM, MSUN_G


class LegacySnapshotShardTests(unittest.TestCase):
    def files(self, root, prefix):
        paths = []
        for shard in range(2):
            path = root / f'{prefix}_021.{shard}.hdf5'
            counts = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint64)
            with h5py.File(path, 'w') as f:
                f.create_group('Header').attrs.update({
                    'Time': .5, 'Redshift': 1., 'BoxSize': 100.,
                    'MassTable': np.array([0, 1, 0, 0, 0, 0], dtype=float),
                    'NumFilesPerSnapshot': 2, 'NumPart_ThisFile': counts,
                    'NumPart_Total': counts*2})
                f.create_group('Parameters').attrs.update({
                    'UnitLength_in_cm': KPC_CM, 'UnitMass_in_g': 1e10*MSUN_G,
                    'UnitVelocity_in_cm_per_s': 1e5, 'HubbleParam': .7,
                    'Omega0': .3, 'OmegaLambda': .7, 'OmegaBaryon': .05,
                    'ComovingIntegrationOn': 1})
                p = f.create_group('PartType1')
                p['ParticleIDs'] = np.array([shard+1], dtype=np.uint64)
                for name, value, cgs, aa, hh in (
                    ('Coordinates', [[10.+shard, 20., 30.]], KPC_CM, 1., -1.),
                    ('Velocities', [[1., 2., 3.]], 1e5, .5, 0.)):
                    p[name] = np.array(value)
                    p[name].attrs.update(to_cgs=cgs, a_scaling=aa, h_scaling=hh)
            paths.append(path)
        return paths

    def test_group_ordered_and_ngenic_shards_are_accepted(self):
        for prefix in ('grouped_snapshot', 'ngenic_ics'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                validate_gadget_snapshot_set(self.files(Path(tmp), prefix), 21)

    def test_unknown_names_raise_value_error_instead_of_sort_type_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'numbered consecutively'):
                validate_gadget_snapshot_set(self.files(Path(tmp), 'unknown'), 21)


if __name__ == '__main__':
    unittest.main()
