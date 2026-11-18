import json
import tempfile
import unittest
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import verification as fixture
from core.stellar_stream_movie import make_stellar_stream_movie


class StellarStreamMovieTests(unittest.TestCase):
    def test_saved_identity_missingness_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);catalogue,ids=fixture.StellarHistoryTests().fixture(root)
            history=fixture.trace_stellar_candidates(root,catalogue=catalogue,snapshots=[0,1],candidate_ids=[1])
            movie=make_stellar_stream_movie(root,history['directory'],root/'movie',fps=1,max_particles=100)
            receipt=json.loads((movie.parent/'stellar_movie_contract.json').read_text())
            self.assertEqual([e['present_count'] for e in receipt['epochs']],[1,2])
            with np.load(movie.parent/receipt['epochs'][0]['numeric_asset']) as data:
                self.assertEqual(data['particle_ids'].tolist(),ids.tolist())
                self.assertEqual(data['present'].tolist(),[True,False])
            with imageio.get_reader(movie) as reader:self.assertEqual(reader.count_frames(),2)
            with open(Path(history['directory'])/'history.hdf5','ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                make_stellar_stream_movie(root,history['directory'],root/'bad',fps=1,max_particles=100)


if __name__=='__main__':unittest.main()
