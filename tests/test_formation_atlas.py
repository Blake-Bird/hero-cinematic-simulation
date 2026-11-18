import json
import tempfile
import unittest
from pathlib import Path
from core.formation_atlas import select_comparable_histories, export_formation_atlas


def halo(identity,mass,redshift=None):
    return {'final_halo_id':identity,'final_mass_msun_h':mass,'formation_redshift':redshift,'merger_events':[],
            'track':[{'scale_factor':.5,'mass_msun_h':mass*.4,'halo_id':identity,'original_halo_id':identity,'snapshot':0,'redshift':1.},
                     {'scale_factor':1.,'mass_msun_h':mass,'halo_id':identity,'original_halo_id':identity,'snapshot':1,'redshift':0.}]}


class FormationAtlasTests(unittest.TestCase):
    def test_selects_largest_comparable_group_and_retains_unavailable_crossing(self):
        histories=[halo(1,1000),halo(2,100),halo(2**53+1,80, .7),halo(4,60)]
        selected=select_comparable_histories(histories)
        self.assertEqual([h['final_halo_id'] for h in selected],[2,2**53+1,4])
        with tempfile.TemporaryDirectory() as temp:
            result={'status':'COMPLETE','histories':histories,'source':'synthetic','final_snapshot':1,'particle_floor':0}
            image=export_formation_atlas(result,temp)
            self.assertTrue(image.exists())
            record=json.loads((Path(temp)/'halo_formation_atlas.json').read_text())
            self.assertIsNone(record['halos'][0]['formation_redshift'])
            self.assertEqual(record['halos'][1]['track'][0]['halo_id'],str(2**53+1))

    def test_single_history_returns_no_atlas(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(export_formation_atlas({'status':'COMPLETE','histories':[halo(1,5)]},temp))


if __name__=='__main__':unittest.main()
