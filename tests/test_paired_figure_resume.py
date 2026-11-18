"""Actual tiny paired plots and bounded core-render resume corruption checks.

The numeric populations are deterministic authored fixtures, not native evolution.
"""
from contextlib import redirect_stdout
import io
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import paired_atlas_figures as plots
from core.foundation import file_sha256
from core.paired_atlas import epoch_name


class PairedFigureResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.original = Path(cls.workspace.name) / 'original'
        rng = np.random.default_rng(481)
        for name, scale in [('EDE', 1.), ('LCDM', 1.1)]:
            folder = cls.original / '06_Tables_and_Data/Measurements' / name
            folder.mkdir(parents=True)
            positions = rng.uniform(0, 2, (32, 3))
            speeds = np.linalg.norm(rng.normal(size=(32, 3)), axis=1)
            counts, edges = np.histogram(speeds, np.linspace(0, 5, 11))
            arrays = dict(fof_mass_msun=np.array([400., 800.])*scale,
                          fof_particle_count=np.array([400, 800]),
                          subhalo_mass_msun=np.array([350., 600.])*scale,
                          subhalo_vmax_km_s=np.array([1., 2., 3.])*scale,
                          velocity_histogram_edges_km_s=edges,
                          velocity_histogram_counts=counts)
            for projection, axes in [('xy', (0, 1)), ('xz', (0, 2)), ('yz', (1, 2))]:
                hist, _, _ = np.histogram2d(positions[:, axes[1]], positions[:, axes[0]],
                                            bins=8, range=((0, 2), (0, 2)))
                arrays['dm_surface_density_msun_mpc2_'+projection] = hist/(2/8)**2
            arrays.update(k_mpc_inverse=np.array([.1, 1., 10.]),
                          power_mpc3=np.array([10., 3., 1.])*scale,
                          power_sigma_gaussian_mpc3=np.array([1., .3, .1]),
                          power_mode_count=np.array([10, 20, 30]),
                          power_shot_mpc3=np.zeros(3))
            np.savez_compressed(folder / 'data.npz', **arrays)
            record = {'summary': {'snapshot': 0, 'redshift': 0.,
                                  'box_comoving_mpc': 2., 'particle_mass_mean_msun': 1.},
                      'numeric_asset': 'data.npz'}
            (folder / (name+'_z_000.json')).write_text(json.dumps(record))
        with redirect_stdout(io.StringIO()):
            cls.made = plots.render_available(cls.original)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Input paths are part of the binding, so retain the original location.
        # Work on the same atlas and restore exact bytes after every test.
        self.root = self.original
        self.backup = Path(self.temp.name) / 'backup'
        shutil.copytree(self.root, self.backup)
        self.addCleanup(self.restore)
        self.output = self.root/'01_Comparisons'/epoch_name('EDE_vs_LCDM', 0, 0.)
        self.manifest = self.output/'comparison_manifest.json'
        self.progress = self.root/'07_Status_and_Provenance/paired_figure_progress.json'

    def restore(self):
        shutil.rmtree(self.root)
        shutil.copytree(self.backup, self.root)

    def resume_rejected(self):
        before = self.progress.read_bytes()
        with self.assertRaises(ValueError):
            plots.render_available(self.root)
        self.assertEqual(self.progress.read_bytes(), before)

    def test_real_plots_and_unchanged_resume(self):
        self.assertEqual(len(self.made), 1)
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(len(manifest['figures']), 9)
        for name in manifest['figures']:
            for suffix in ('.png', '.pdf', '.svg'):
                self.assertGreater((self.output/Path(name).with_suffix(suffix)).stat().st_size, 100)
        # Independently check saved numeric exports, not just file existence.
        for path in self.output.glob('*_Density_*_and_log_ratio.npz'):
            with np.load(path, allow_pickle=False) as data:
                for model in ('EDE','LCDM'):
                    self.assertAlmostEqual(float(data[model+'_surface_density_msun_mpc2'].sum())*(2/8)**2,32.)
                np.testing.assert_allclose(data['log10_EDE_over_LCDM'],
                    np.log10(data['EDE_surface_density_msun_mpc2']/data['LCDM_surface_density_msun_mpc2']))
        with next(self.output.glob('*_FOF_halo_mass_function.csv')).open() as stream:
            rows=list(csv.DictReader(stream))
        for model in ('EDE','LCDM'):
            self.assertEqual(sum(int(row[model+'_count']) for row in rows),2)
            for row in rows:
                width=np.log10(float(row['mass_high_msun'])/float(row['mass_low_msun']))
                self.assertAlmostEqual(float(row[model+'_dn_dlog10M_mpc_inverse3']),int(row[model+'_count'])/8/width)
        with next(self.output.glob('*_Particle_velocity_distributions.csv')).open() as stream:
            rows=list(csv.DictReader(stream))
        for model in ('EDE','LCDM'):
            selected=[row for row in rows if row['run']==model]
            self.assertEqual(sum(int(row['particle_count']) for row in selected),32)
            self.assertAlmostEqual(sum(float(row['probability_density_per_km_s'])*
                (float(row['speed_high_km_s'])-float(row['speed_low_km_s'])) for row in selected),1.)
        before = {p.name: file_sha256(p) for p in self.output.iterdir() if p.is_file()}
        with patch.object(plots, 'render_pair', side_effect=AssertionError('unexpected rerender')):
            self.assertEqual(plots.render_available(self.root), self.made)
        self.assertEqual(before, {p.name: file_sha256(p) for p in self.output.iterdir() if p.is_file()})

    def test_changed_numeric_and_summary_rejected(self):
        folder = self.root/'06_Tables_and_Data/Measurements/EDE'
        for path, payload in [(folder/'data.npz', b'corrupt cache'),
                              (folder/'EDE_z_000.json', None)]:
            original = path.read_bytes()
            if payload is None:
                record = json.loads(original)
                record['summary']['particle_mass_mean_msun'] = 99.
                payload = json.dumps(record).encode()
            path.write_bytes(payload)
            with self.subTest(path=path.name):
                self.resume_rejected()
            self.assertEqual(path.read_bytes(), payload)
            path.write_bytes(original)

    def test_changed_or_missing_exports_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        for suffix in ('.png', '.pdf', '.svg', '.npz', '.csv'):
            path = next(self.output.glob('*'+suffix))
            original = path.read_bytes()
            with self.subTest(suffix=suffix, change='edit'):
                path.write_bytes(b'changed export')
                self.resume_rejected()
                self.assertEqual(path.read_bytes(), b'changed export')
            with self.subTest(suffix=suffix, change='missing'):
                path.unlink()
                self.resume_rejected()
            path.write_bytes(original)

    def test_legacy_or_changed_receipt_and_core_manifest_rejected(self):
        receipt = self.output/'core_render_receipt.json'
        seal = receipt.with_name(receipt.name+'.sha256')
        for path in (receipt, seal):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.unlink()
                self.resume_rejected()
                path.write_bytes(original+b'changed' if path==receipt else b'0'*64)
                self.resume_rejected()
                path.write_bytes(original)
        original = self.manifest.read_bytes()
        record = json.loads(original);record['physical_box_mpc'] = 999.
        self.manifest.write_text(json.dumps(record))
        self.resume_rejected()
        self.manifest.write_bytes(original)

    def test_changed_renderer_rejected_and_extension_preserved(self):
        source = Path(plots.__file__).resolve()
        def changed(path):
            return '0'*64 if Path(path).resolve()==source else file_sha256(path)
        with patch.object(plots, 'file_sha256', side_effect=changed):
            self.resume_rejected()
        record = json.loads(self.manifest.read_text())
        (self.output/'extension.png').write_bytes(b'separate producer fixture')
        record['figures'].append('extension.png')
        self.manifest.write_text(json.dumps(record))
        with patch.object(plots, 'render_pair', side_effect=AssertionError('unexpected rerender')):
            self.assertEqual(plots.render_available(self.root), [record])

    def test_incomplete_output_is_preserved(self):
        self.manifest.unlink()
        before = {p.name: file_sha256(p) for p in self.output.iterdir() if p.is_file()}
        self.resume_rejected()
        self.assertEqual(before, {p.name: file_sha256(p) for p in self.output.iterdir() if p.is_file()})

    def test_input_change_during_render_prevents_publication(self):
        records={};paths={}
        for name in ('EDE','LCDM'):
            folder=self.root/'06_Tables_and_Data/Measurements'/name
            records[name]=json.loads((folder/(name+'_z_000.json')).read_text())
            paths[name]=folder/'data.npz'
        out=Path(self.temp.name)/'interrupted-render'
        original_save=plots.save
        def change_after_first_export(*args,**kwargs):
            original_save(*args,**kwargs)
            with paths['EDE'].open('ab') as stream:stream.write(b'changed during rendering')
        with patch.object(plots,'save',side_effect=change_after_first_export):
            with self.assertRaisesRegex(ValueError,'source changed during rendering'):
                plots.render_pair(records,paths,out)
        self.assertFalse((out/'comparison_manifest.json').exists())
        self.assertFalse((out/'core_render_receipt.json').exists())
        self.assertTrue(list(out.glob('*.png')))


if __name__ == '__main__':
    unittest.main()
