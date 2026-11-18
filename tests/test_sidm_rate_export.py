"""Regression for absent native rate metadata crossing the JSON boundary."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.sidm_products import fixed_cohort_interval, export_native_scattering_rates
from core.stream_common import read_product


class SIDMRateExportTests(unittest.TestCase):
    def test_unsupported_rates_publish_null_metadata_and_masked_nan_arrays(self):
        ids = np.array([2**60 + 1, 2**60 + 3], dtype=np.uint64)
        before = dict(particle_ids=ids, available=np.ones(2, bool),
                      mass_msun=np.ones(2), interaction_count=np.zeros(2, np.uint64),
                      exposure=np.zeros(2))
        for complete in (True, False):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as temp:
                after = dict(before, available=np.array([True, complete]),
                             interaction_count=np.array([2, 0], dtype=np.uint64),
                             exposure=np.array([.5, 0.]))
                row, raw = fixed_cohort_interval(
                    before, after, elapsed_interval_gyr=.25,
                    physics_interval_supported=False)
                arrays = dict(raw, interval_end_gyr=np.array([.25]))
                for name in ('complete_cohort', 'aggregate_supported',
                             'diagnostic_complete_cohort_rate_per_gyr',
                             'diagnostic_complete_cohort_expected_rate_per_gyr',
                             'supported_rate_per_gyr', 'supported_expected_rate_per_gyr'):
                    arrays[name] = np.asarray([row[name]])
                report = dict(schema='native-sidm-fixed-cohort-rates-v1',
                              source_origin='synthetic_control', epoch={'meaning': 'test interval'},
                              particle_namespace='test:exact-native-identities', intervals=[row],
                              definition=row['definition'], missing_policy='Unsupported rates remain masked.')
                target = Path(temp) / 'export'
                # Only native intake is substituted. The real interval estimator,
                # immutable numeric writer and figure exporter all execute.
                with patch('core.sidm_products.native_scattering_history',
                           return_value=(report, arrays, {})):
                    gallery = export_native_scattering_rates(
                        'unused-history.json', target, run_directory=temp)
                metadata = json.loads((target / 'scattering-rates.json').read_text())
                self.assertIsNone(metadata['intervals'][0]['supported_rate_per_gyr'])
                self.assertFalse(metadata['intervals'][0]['aggregate_supported'])
                _, saved, _ = read_product(target / 'scattering-rates.json', memory_gib=.1)
                self.assertTrue(np.isnan(saved['supported_rate_per_gyr']).all())
                np.testing.assert_array_equal(saved['particle_ids'], ids)
                np.testing.assert_array_equal(saved['interaction_count_increment'], [2, 0])
                self.assertEqual(len(gallery['figures']), 1)
                if complete:
                    self.assertEqual(metadata['intervals'][0]['diagnostic_complete_cohort_rate_per_gyr'], 4.)
                else:
                    self.assertIsNone(metadata['intervals'][0]['diagnostic_complete_cohort_rate_per_gyr'])


if __name__ == '__main__':
    unittest.main()
