"""Compare native isolated fixed/per-particle mass storage using real tiny HDF5.

These are authored reduction fixtures, not evolved or equilibrated halos.
"""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from core.foundation import KPC_CM, MSUN_G
from core.sidm_control_measurements import _native_epoch


class IsolatedHistoryUnitsTests(unittest.TestCase):
    def fixture(self, root, h, explicit):
        directory = root / 'data'
        directory.mkdir(parents=True)
        path = directory / 'snapshot_000.hdf5'
        positions = np.arange(96, dtype=float).reshape(32, 3) / 100
        velocities = np.tile([2., -3., 1.], (32, 1))
        masses = np.repeat([4., 6.], 16)
        potential = np.full(32, -11.)
        counts = np.array([0, 16, 16, 0, 0, 0], dtype=np.uint32)
        with h5py.File(path, 'w') as stream:
            stream.create_group('Header').attrs.update(
                Time=2., NumPart_Total=counts, NumPart_ThisFile=counts,
                NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32),
                MassTable=[0., 0. if explicit else 2., 0. if explicit else 3., 0., 0., 0.],
                Research2Generation='isolated-mass-unit-fixture', Research2ShardCount=1)
            stream.create_group('Parameters').attrs.update(
                ComovingIntegrationOn=0, PeriodicBoundariesOn=0, HubbleParam=h,
                UnitLength_in_cm=KPC_CM, UnitMass_in_g=2 * MSUN_G,
                UnitVelocity_in_cm_per_s=1e5, UnitTime_in_s=KPC_CM / 1e5)
            for kind, selection in ((1, slice(0, 16)), (2, slice(16, 32))):
                group = stream.create_group('PartType' + str(kind))
                group['ParticleIDs'] = np.arange(32, dtype=np.uint64)[selection] + np.uint64(2**60 + 1)
                for name, values, factor in (
                    ('Coordinates', positions[selection], KPC_CM),
                    ('Velocities', velocities[selection], 1e5),
                    ('Potential', potential[selection], 1e10),
                    *([('Masses', masses[selection] / 2, 2 * MSUN_G)] if explicit else []),
                ):
                    dataset = group.create_dataset(name, data=values, fletcher32=True)
                    dataset.attrs.update(to_cgs=factor, a_scaling=0., h_scaling=0.)
        path.with_suffix('.publication.json').write_text(json.dumps({
            'schema': 'native-hdf5-set-publication-v1', 'state': 'COMPLETE',
            'generation': 'isolated-mass-unit-fixture', 'shard_count': 1}))
        return positions, velocities, masses, potential

    def measure(self, root):
        return _native_epoch(root, 0, {
            'maximum_particles': 32, 'chunk_rows': 7, 'memory_gib': .01,
            'core_radius_proper_kpc': 10.,
        }, center=np.zeros(3), velocity_frame=np.zeros(3))

    def test_fixed_and_explicit_masses_give_identical_isolated_observables(self):
        for h in (.5, .7, 1.):
            with self.subTest(h=h), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                x, v, m, phi = self.fixture(root / 'fixed', h, False)
                self.fixture(root / 'explicit', h, True)
                fixed, explicit = self.measure(root / 'fixed'), self.measure(root / 'explicit')
                for key in ('mass_msun', 'kinetic_energy', 'potential_energy', 'total_energy',
                            'core_mass_msun', 'core_effective_particles'):
                    self.assertAlmostEqual(fixed[key], explicit[key], places=11, msg=key)
                self.assertAlmostEqual(fixed['mass_msun'], m.sum(), places=11)
                self.assertAlmostEqual(fixed['kinetic_energy'], .5 * np.sum(m * np.sum(v*v, axis=1)), places=11)
                self.assertAlmostEqual(fixed['potential_energy'], .5 * np.dot(m, phi), places=11)
                self.assertEqual(fixed['particle_count'], 32)
                self.assertEqual(fixed['identity_census_sha256'], explicit['identity_census_sha256'])
                np.testing.assert_allclose(fixed['linear_momentum'], np.sum(m[:, None] * v, axis=0), rtol=1e-14)
                np.testing.assert_allclose(fixed['orbital_angular_momentum'], np.sum(m[:, None] * np.cross(x, v), axis=0), rtol=1e-14)
                np.testing.assert_allclose(fixed['center_of_mass_proper_kpc'], np.sum(m[:, None] * x, axis=0) / m.sum(), rtol=1e-14)

    def test_isolated_fixed_mass_is_independent_of_hubble_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, h in enumerate((.25, .5, 1.)):
                path = root / str(index)
                self.fixture(path, h, False)
                rows.append(self.measure(path))
            for row in rows:
                self.assertAlmostEqual(row['mass_msun'], 160., places=11)
                self.assertAlmostEqual(row['total_energy'], 240., places=11)
                self.assertAlmostEqual(row['core_effective_particles'], 160.**2 / (16*4.**2 + 16*6.**2), places=11)


if __name__ == '__main__':
    unittest.main()
