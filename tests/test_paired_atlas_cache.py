"""Exercise cache reuse against real tiny HDF5 payloads and changed inputs.

Catalogue discovery/columns are fixtures; no native evolution is represented.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from core import paired_atlas
from core.foundation import MSUN_G, file_sha256


class PairedAtlasCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.run = self.root / 'run'
        self.run.mkdir()
        self.output = self.root / 'measurements'
        self.config = self.run / 'config.yaml'
        self.config.write_text(yaml.safe_dump({'cosmology': {'H0': 50.},
                                             'simulation': {'box_mpc_h': 10.}}))
        (self.run / 'manifest.json').write_text('{}')
        self.snapshot = self.run / 'snapshot.hdf5'
        self.catalogue = self.run / 'catalogue.fixture'
        self.catalogue.write_text('catalogue columns supplied by fixture')
        self.groups = [self.catalogue]
        with h5py.File(self.snapshot, 'w') as stream:
            header = stream.create_group('Header')
            header.attrs['MassTable'] = [0., 1., 0., 0., 0., 0.]
            parameters = stream.create_group('Parameters')
            parameters.attrs['HubbleParam'] = .5
            parameters.attrs['UnitMass_in_g'] = MSUN_G
            particles = stream.create_group('PartType1')
            particles['Coordinates'] = np.arange(96).reshape(32, 3) * 50.
            particles['Velocities'] = np.arange(96).reshape(32, 3) / 10.
            particles['ParticleIDs'] = np.arange(1, 33, dtype=np.uint64)
        for name, value in {
            'snapshot_ids': lambda run: [0],
            'read_snapshot_header': lambda run, snap: ({'Time': 1.}, [self.snapshot]),
            'group_files': lambda run, snap: list(self.groups),
            'halo_masses': lambda run, snap: np.array([1.]),
            'halo_particle_counts': lambda run, snap: np.array([32]),
            'group_field': lambda *args: None,
            'subhalo_field': lambda *args: None,
        }.items():
            self.stack.enter_context(patch('core.analysis.' + name, value))
        self.stack.enter_context(patch('core.visualization._velocity_conversion_km_s', return_value=1.))
        self.conversion = self.stack.enter_context(patch('core.halos._unit_conversion'))

    def measure(self, grid=16):
        return paired_atlas.measure_run(self.run, self.run, self.output, 'fixture', map_grid=grid)

    def cache(self):
        self.measure()
        return next(self.output.glob('fixture*.json'))

    def test_fresh_32_particle_measurement_and_unchanged_reuse(self):
        manifest = self.measure()
        before = {p.name: p.read_bytes() for p in self.output.glob('fixture*.npz')}
        record = json.loads(next(self.output.glob('fixture*.json')).read_text())
        with np.load(self.output / record['numeric_asset'], allow_pickle=False) as arrays:
            self.assertEqual(record['summary']['particle_count'], 32)
            self.assertAlmostEqual(record['summary']['total_dm_mass_msun'], 64.)
            self.assertAlmostEqual(arrays['dm_surface_density_msun_mpc2_xy'].sum() * (20 / 16)**2, 64.)
        self.assertEqual(self.measure(), manifest)
        self.assertEqual(self.conversion.call_count, 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.glob('fixture*.npz')})

    def test_changed_native_payload_is_rejected_and_cache_preserved(self):
        record = self.cache()
        before = record.read_bytes()
        with h5py.File(self.snapshot, 'r+') as stream:
            stream['PartType1/Velocities'][0, 0] = 999.
        with self.assertRaisesRegex(ValueError, 'cache inputs'):
            self.measure()
        self.assertEqual(record.read_bytes(), before)
        self.assertEqual(self.conversion.call_count, 1)

    def test_input_change_during_fresh_measurement_prevents_sealing(self):
        def change_catalogue(*args):
            self.catalogue.write_text('changed while measuring')
            return 1.
        with patch('core.visualization._velocity_conversion_km_s', side_effect=change_catalogue):
            with self.assertRaisesRegex(ValueError, 'source changed during generation'):
                self.measure()
        self.assertFalse(list(self.output.glob('fixture*.json')))
        self.assertFalse(list(self.output.glob('fixture*.sha256')))
        self.assertTrue(list(self.output.glob('fixture*.npz')))

    def test_changed_catalogue_inventory_and_grid_are_rejected(self):
        self.cache()
        with self.assertRaisesRegex(ValueError, 'cache inputs'):
            self.measure(grid=32)
        extra = self.run / 'extra-catalogue.fixture'
        extra.write_text('new shard')
        self.groups.append(extra)
        with self.assertRaisesRegex(ValueError, 'cache inputs'):
            self.measure()
        self.groups.pop()
        self.catalogue.write_text('changed catalogue')
        with self.assertRaisesRegex(ValueError, 'cache inputs'):
            self.measure()

    def test_changed_config_manifest_and_implementation_are_rejected(self):
        self.cache()
        for path in (self.config, self.run / 'manifest.json'):
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, 'cache inputs'):
                self.measure()
            path.write_bytes(original)
        implementation = Path(paired_atlas.__file__).with_name('analysis.py').resolve()
        def changed_digest(path):
            return '0' * 64 if Path(path).resolve() == implementation else file_sha256(path)
        with patch('core.paired_atlas.file_sha256', side_effect=changed_digest):
            with self.assertRaisesRegex(ValueError, 'cache inputs'):
                self.measure()

    def test_missing_seal_edited_metadata_and_legacy_binding_are_rejected(self):
        record = self.cache()
        seal = record.with_name(record.name + '.sha256')
        original, original_seal = record.read_bytes(), seal.read_bytes()
        seal.unlink()
        with self.assertRaisesRegex(ValueError, 'Unsealed'):
            self.measure()
        seal.write_bytes(original_seal)
        record.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed paired cache metadata'):
            self.measure()
        old = json.loads(original)
        old.pop('cache_binding')
        record.write_text(json.dumps(old))
        seal.write_text(file_sha256(record) + '\n')
        with self.assertRaisesRegex(ValueError, 'cache inputs'):
            self.measure()

    def test_corrupt_numeric_or_incomplete_pair_is_not_overwritten(self):
        record = self.cache()
        numeric = self.output / json.loads(record.read_text())['numeric_asset']
        numeric.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Existing paired cache changed'):
            self.measure()
        self.assertEqual(numeric.read_bytes(), b'broken')
        numeric.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete paired cache'):
            self.measure()
        self.assertTrue(record.is_file())
        self.assertFalse(numeric.exists())


if __name__ == '__main__':
    unittest.main()
