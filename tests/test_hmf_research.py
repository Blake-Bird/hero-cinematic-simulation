import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import h5py
import yaml

from core.hmf_campaign import sampled_requests, score_rows, predict_sample, validate_request, saved_bundle_epoch
from core.hmf_learning import build_dataset, train_dataset, predict_density, load_source, cosmology_features
from core.hmf_campaign import mark_unusual_disagreement, export_candidates
from core.hmf_learning import select_resolution_rows
from core.gadget_hmf_data import export_gadget_measurements, export_gadget_tree

ROOT = Path(__file__).resolve().parents[1]


def source_fixture(root, h0, split=None, finder='gadget_fof_subfind', definition='fof'):
    root.mkdir()
    config = copy.deepcopy(json.loads((ROOT/'configs/hmf_scan.example.json').read_text())['config'])
    config['cosmology']['H0'] = h0
    config['halos'] = {'fof_linking_length': .2}
    meta = dict(target_kind='simulation_counts', config=config, finder=finder, physics='dmo')
    (root/'metadata.json').write_text(json.dumps(meta))
    fields = ['mass_lower_msun_h', 'mass_upper_msun_h', 'halo_count', 'volume_mpc_h3',
              'redshift', 'particle_mass_msun_h', 'minimum_halo_particles', 'mass_definition', 'feature_sigma']
    with (root/'hmf.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        rng = np.random.default_rng(int(h0*100))
        for z in (0, 1, 3):
            for lo in np.geomspace(1e10, 1e14, 12):
                hi = lo*1.5
                exposure = 1e5*np.log10(hi/lo)
                rate = np.exp(-2-.7*(np.log10(np.sqrt(lo*hi))-10)-.3*np.log1p(z)+.01*(h0-67))
                writer.writerow(dict(mass_lower_msun_h=lo, mass_upper_msun_h=hi,
                    halo_count=int(rng.poisson(exposure*rate)), volume_mpc_h3=1e5,
                    redshift=z, particle_mass_msun_h=1e7, minimum_halo_particles=100,
                    mass_definition=definition, feature_sigma=1/(1+z)))
    result = {'path': str(root)}
    if split:
        result['split'] = split
    return result


class HMFScreenTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT/'configs/hmf_scan.example.json').read_text())

    def test_sampling_reproducible_and_bounded(self):
        a, b = sampled_requests(self.spec), sampled_requests(self.spec)
        self.assertEqual(a, b)
        for request in a:
            for path, (lo, hi) in self.spec['sampling']['bounds'].items():
                section, key = path.split('.')
                self.assertTrue(lo <= request['config'][section][key] <= hi)
        self.spec['sampling']['bounds']['simulation.random_seed'] = [1, 2]
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            sampled_requests(self.spec)

    def test_validation(self):
        for z in ([-1], [np.nan], [0, 0]):
            spec = copy.deepcopy(self.spec)
            spec['redshifts'] = z
            with self.assertRaises(ValueError):
                validate_request(spec)
        self.spec['config']['cosmology']['N_ncdm'] = 1
        with self.assertRaisesRegex(ValueError, 'neutrino'):
            validate_request(self.spec)

    def test_grid_scans_every_requested_combination(self):
        self.spec['sampling'] = {'method': 'grid', 'grid': {
            'cosmology.H0': [64, 68, 72], 'cosmology.n_s': [.94, .98]}}
        requests = sampled_requests(self.spec)
        self.assertEqual(len(requests), 6)
        self.assertEqual({(r['config']['cosmology']['H0'], r['config']['cosmology']['n_s']) for r in requests},
                         {(h, n) for h in (64, 68, 72) for n in (.94, .98)})

    def test_unusual_formula_disagreement_and_gadget_handoff(self):
        rows = []
        for i in range(11):
            # A huge overall abundance alone must not select a cosmology.
            amplitude = 1e6 if i == 9 else 1.
            for fit, value in [('a', amplitude), ('b', amplitude*(20 if i == 10 else 2))]:
                rows.append(dict(sample_id=f'cosmo_{i:05d}', redshift=0., mass_lower_msun_h=1e10,
                    mass_upper_msun_h=1e11, fit=fit, k_covered=1, table_range=1, dn_dlog10M_h3_mpc3=value))
        scores = mark_unusual_disagreement(score_rows(rows))
        self.assertEqual([r['sample_id'] for r in scores if r['flag_unusual_disagreement']], ['cosmo_00010'])
        self.assertFalse(scores[9]['flag_unusual_disagreement'])
        with tempfile.TemporaryDirectory() as tmp:
            candidates = export_candidates(scores, sampled_requests(self.spec), tmp)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]['sample_id'], 'cosmo_00010')
            handoff = json.loads(Path(candidates[0]['linear_config']).read_text())
            expected = candidates[0]['config']['cosmology']
            self.assertAlmostEqual(handoff['Omega_b']*(handoff['H0']/100)**2, expected['omega_b'])
            self.assertEqual(handoff['z_start'], 49)
        # Uniformly large disagreement is usual in this ensemble.
        for row in rows:
            row['dn_dlog10M_h3_mpc3'] = 20 if row['fit'] == 'b' else 1
        self.assertFalse(any(r['flag_unusual_disagreement'] for r in mark_unusual_disagreement(score_rows(rows))))

    def test_spread_outliers_and_domain_mask(self):
        rows = []
        for sample, amplitude in enumerate((1., 1., 10.)):
            for fit, factor in (('a', 1.), ('b', 3.), ('excluded', 100000)):
                rows.append(dict(sample_id=str(sample), fit=fit, redshift=0.,
                    mass_lower_msun_h=1e10, mass_upper_msun_h=1e11, k_covered=1,
                    table_range=int(fit != 'excluded'), dn_dlog10M_h3_mpc3=amplitude*factor))
        scored = score_rows(rows)
        self.assertEqual(len(scored), 3)
        self.assertAlmostEqual(scored[-1]['cosmology_factor_from_median'], 10.)
        self.assertAlmostEqual(scored[-1]['fit_max_over_min'], 3.)
        self.assertTrue(scored[-1]['flag_cosmology_outlier'])
        self.assertIsNone(score_rows(rows[:3])[0]['cosmology_factor_from_median'])
        self.assertFalse(score_rows(rows[:1])[0]['flag_fit_disagreement'])
        extreme = copy.deepcopy(rows)
        for row in extreme:
            row['dn_dlog10M_h3_mpc3'] = 1e300 if row['fit'] == 'b' else 1e-300
        scored = score_rows(extreme)
        self.assertTrue(scored[0]['ratio_display_capped'])
        self.assertAlmostEqual(scored[0]['fit_log10_spread'], 600.)
        json.dumps(scored, allow_nan=False)

    def test_prediction_actual_z_and_masks(self):
        self.spec['redshifts'] = [0, 6]
        k = np.geomspace(1e-5, 1e3, 1000)
        p = 1e4*k**-.5/(1+k)**3
        with tempfile.TemporaryDirectory() as tmp:
            with patch('core.hmf_campaign.solve_epoch', return_value=(k, p)) as solve:
                rows = predict_sample(self.spec, 'one', Path(tmp)/'run')
            self.assertEqual([call.args[1] for call in solve.call_args_list], [0, 6])
            self.assertTrue(all(r['table_range'] == 0 for r in rows if r['redshift'] == 6 and r['fit'] in {'jenkins', 'warren'}))
            self.assertTrue((Path(tmp)/'run/predictions.csv').is_file())

    def test_saved_bundle_reuse_requires_epoch_and_cosmology(self):
        from core.cosmology import class_common
        from core.foundation import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            k = np.geomspace(1e-5, 100, 50)
            np.savetxt(root/'pk.dat', np.column_stack((k, k**-2)), header='redshift z=0')
            parameters = class_common(self.spec['config'], 0, Path('unused_'), 100)
            record = dict(model='lcdm', status='complete', checksums_sha256={'pk.dat': file_sha256(root/'pk.dat')},
                source_files={'z0': {'pk': 'pk.dat'}}, class_parameters={'z0': parameters})
            (root/'spectrum.json').write_text(json.dumps(record))
            self.spec['linear_bundle'] = str(root)
            actual, _ = saved_bundle_epoch(self.spec, 0, root/'reuse')
            np.testing.assert_array_equal(actual, k)
            with self.assertRaisesRegex(ValueError, 'no saved spectrum'):
                saved_bundle_epoch(self.spec, 1, root/'bad')
            self.spec['config']['cosmology']['H0'] = 70
            with self.assertRaisesRegex(ValueError, 'parameter differs'):
                saved_bundle_epoch(self.spec, 0, root/'bad')
            (root/'pk.dat').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                saved_bundle_epoch(self.spec, 0, root/'bad')


class HMFLearningTests(unittest.TestCase):
    def test_actual_gadget_catalogue_export_and_zero_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'run'
            (run/'data').mkdir(parents=True)
            config = copy.deepcopy(json.loads((ROOT/'configs/hmf_scan.example.json').read_text())['config'])
            config.update(simulation=dict(physics='dm', box_mpc_h=10., particles_per_dimension=8, random_seed=42),
                analysis=dict(hmf_mass_definition='fof', hmf_min_particles=100, hmf_mass_range_msun_h=[1e9, 1e12], mass_bins=3),
                halos=dict(fof_linking_length=.2))
            (run/'config.yaml').write_text(yaml.safe_dump(config))
            with h5py.File(run/'data/snapshot_000.hdf5', 'w') as f:
                h = f.create_group('Header').attrs
                h['Time'] = 1.
                h['BoxSize'] = 10000.
                h['NumPart_Total'] = [0, 8**3, 0, 0, 0, 0]
                h['MassTable'] = [0, .01, 0, 0, 0, 0]
            catalogue = run/'data/fof_subhalo_tab_000.hdf5'
            with h5py.File(catalogue, 'w') as f:
                h = f.create_group('Header').attrs
                h['Time'] = 1.
                h['Ngroups_Total'] = h['Ngroups_ThisFile'] = 4
                f['Group/GroupMass'] = [1., 2., 3., 5.]
                f['Group/GroupLen'] = [50, 200, 300, 500]
            meta = export_gadget_measurements(run, root/'measurement')
            self.assertEqual(meta['particle_count'], 512)
            with (root/'measurement/hmf.csv').open() as stream:
                bins = list(csv.DictReader(stream))
            self.assertEqual([int(r['halo_count']) for r in bins], [0, 3, 0])
            loaded, _ = load_source({'path': str(root/'measurement')})
            self.assertEqual([r['count'] for r in loaded], [3, 0])
            collection = export_gadget_tree(root, root/'collected')
            self.assertEqual(collection['exported_runs'], 1)
            self.assertEqual(collection['status'], 'COMPLETE')
            self.assertTrue((root/'collected/dataset-request.json').is_file())
            with h5py.File(catalogue, 'r+') as f:
                f['Header'].attrs['Ngroups_Total'] = 5
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                export_gadget_measurements(run, root/'bad')

    def test_resolution_comparison_uses_finest_without_double_counting(self):
        template = dict(cosmology_group='c', random_seed=42, box_size_mpc_h=100., redshift=0.,
                        mass_lower_msun_h=1e12, mass_upper_msun_h=1e13)
        low = dict(template, source='512', particles_per_dimension=512, count=90)
        high = dict(template, source='1024', particles_per_dimension=1024, count=100)
        retained, comparisons = select_resolution_rows([low, high])
        self.assertEqual(retained, [high])
        self.assertEqual(comparisons[0]['count_ratio_to_finest'], .9)
        other_seed = dict(low, random_seed=99)
        self.assertEqual(len(select_resolution_rows([low, high, other_seed])[0]), 2)

    def test_discovery_and_default_cosmology_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                source_fixture(root/f'c{i}', 60+i*2)
            report = build_dataset({'source_roots': [{'path': str(root)}]}, root/'dataset')
            self.assertEqual(report['cosmologies'], 5)
            config = json.loads((root/'c0/metadata.json').read_text())['config']
            first = cosmology_features(config)[1]
            config['cosmology']['engine'] = 'class'
            for key in ('N_ncdm', 'T_cmb', 'N_ur'):
                config['cosmology'].pop(key, None)
            self.assertEqual(first, cosmology_features(config)[1])

    def test_formula_predictions_are_excluded_from_training_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [source_fixture(root/f'c{i}', 60+i*2) for i in range(5)]
            from core.hmf_campaign import write_csv
            for source in sources:
                p = Path(source['path'])/'hmf.csv'
                with p.open() as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    row['feature_sheth_tormen_prediction'] = '123.0'
                write_csv(p, rows)
            report = build_dataset({'sources': sources}, root/'dataset')
            self.assertNotIn('feature_sheth_tormen_prediction', report['features'])
            self.assertIn('feature_sigma', report['features'])
            with self.assertRaisesRegex(ValueError, 'physical features'):
                build_dataset({'sources': sources, 'additional_physical_features': ['feature_sheth_tormen_prediction']}, root/'bad')

    def test_build_train_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [source_fixture(root/f'c{i}', 60+i*2) for i in range(10)]
            meta = build_dataset({'sources': sources}, root/'dataset')
            self.assertEqual(meta['cosmologies'], 10)
            model = train_dataset(root/'dataset', root/'model', degree=1)
            self.assertLess(model['metrics']['test']['mean_poisson_deviance'], 3.)
            with (root/'dataset/dataset.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            group_splits = {}
            for row in rows:
                group_splits.setdefault(row['cosmology_group'], set()).add(row['split'])
            self.assertTrue(all(len(s) == 1 for s in group_splits.values()))
            prediction, outside = predict_density(model, [r for r in rows if r['split'] == 'train'])
            self.assertTrue(np.all(prediction > 0))
            self.assertFalse(outside.any())
            bad = dict(rows[0], f_ede=.1)
            with self.assertRaisesRegex(ValueError, 'outside'):
                predict_density(model, [bad])
            _, outside = predict_density(model, [bad], allow_extrapolation=True)
            self.assertTrue(outside[0])

    def test_conventions_leakage_duplicates_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = source_fixture(root/'a', 67, 'train')
            b = source_fixture(root/'b', 67, 'test')
            c = source_fixture(root/'c', 70, 'validation')
            # Distinct realization, same cosmology: still cannot cross splits.
            table = root/'b/hmf.csv'
            with table.open() as stream:
                different = list(csv.DictReader(stream))
            different[0]['halo_count'] = str(int(different[0]['halo_count'])+1)
            from core.hmf_campaign import write_csv
            write_csv(table, different)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                build_dataset({'sources': [a, b, c]}, root/'bad')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                build_dataset({'sources': [a, a]}, root/'bad')
            p = root/'a/metadata.json'
            meta = json.loads(p.read_text())
            meta['target_kind'] = 'analytic_prediction'
            p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'simulation_counts'):
                load_source(a)

    def test_mixed_finder_and_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = source_fixture(root/'a', 67, 'train')
            b = source_fixture(root/'b', 69, 'test', finder='rockstar')
            with self.assertRaisesRegex(ValueError, 'separate datasets'):
                build_dataset({'sources': [a, b]}, root/'bad')
            meta = json.loads((root/'a/metadata.json').read_text())
            meta['config']['halos']['fof_linking_length'] = .3
            (root/'a/metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'linking_length'):
                load_source(a)

    def test_zero_counts_retained_and_corrupt_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [source_fixture(root/f'c{i}', 60+i*2) for i in range(5)]
            table = root/'c0/hmf.csv'
            with table.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['halo_count'] = '0'
            rows[1]['mass_lower_msun_h'] = '1e8'
            from core.hmf_campaign import write_csv
            write_csv(table, rows)
            loaded, _ = load_source(sources[0])
            self.assertEqual(len(loaded), len(rows)-1)
            self.assertEqual(loaded[0]['count'], 0)
            build_dataset({'sources': sources}, root/'dataset')
            with (root/'dataset/dataset.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'identity'):
                train_dataset(root/'dataset', root/'model')


if __name__ == '__main__':
    unittest.main()
