import csv
import tempfile
import unittest
from pathlib import Path

from core.snapshot_tables import snapshot_rows, write_snapshot_tables


class SnapshotTableTests(unittest.TestCase):
    def test_actual_epochs_order_missing_and_csv_precision(self):
        snapshots = {'9': {'redshift':0., 'epoch_contract':{'actual_scale_factor':1.,
                         'cosmic_time_gyr':None, 'cosmic_time_source':{'status':'UNAVAILABLE'}}},
                     '2': {'redshift':1.05, 'requested_redshift':1.,
                         'epoch_contract':{'actual_scale_factor':1/2.05, 'cosmic_time_gyr':5.123456789,
                                           'cosmic_time_source':{'status':'MEASURED'}},
                         'products':{'<hmf>':{'status':'COMPLETE'}}}}
        with tempfile.TemporaryDirectory() as tmp:
            result = write_snapshot_tables(snapshots, tmp)
            self.assertEqual(result['rows'], 2)
            with (Path(tmp)/'snapshot_epochs.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['snapshot'] for row in rows], ['2','9'])
            self.assertEqual(rows[0]['cosmic_time_gyr'], '5.123456789')
            self.assertAlmostEqual(float(rows[0]['redshift_offset']), .05)
            self.assertEqual(rows[1]['cosmic_time_gyr'], '')
            page = (Path(tmp)/'snapshot_epochs.html').read_text()
            self.assertIn('Unavailable', page)
            self.assertIn('&lt;hmf&gt;', page)

    def test_requested_scale_factor_only_when_recorded(self):
        row = snapshot_rows({'1':{'redshift':1.,'epoch_contract':{'actual_scale_factor':.5,'requested_scale_factor':.4}}})[0]
        self.assertAlmostEqual(row['requested_redshift'],1.5)
        self.assertAlmostEqual(row['redshift_offset'],-.5)


if __name__ == '__main__':
    unittest.main()
