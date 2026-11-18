import json
import tempfile
import unittest
from pathlib import Path
from core.results_gallery import create_results_gallery,local_href,verify_results_gallery


class ResultsGalleryTests(unittest.TestCase):
    def test_local_links_and_escaped_text(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);(base/'sample image.png').write_bytes(b'example');(base/'contract.json').write_text('{}')
            request={'schema':'visual-results-gallery-request-v1','title':'<title>','intro':'Safe & sound','items':[{'title':'<halo>','category':'Figures','description':'A & B','path':'sample image.png','contract':'contract.json'}]}
            result=create_results_gallery(request,base/'gallery',base=base)
            page=result.read_text();self.assertIn('&lt;halo&gt;',page);self.assertNotIn('<halo>',page)
            self.assertIn('../sample%20image.png',page);self.assertIn('A &amp; B',page)
            manifest=json.loads((base/'gallery/gallery_manifest.json').read_text());self.assertEqual(len(manifest['items']),1)
            self.assertTrue(manifest['items'][0]['source_sha256'])
            self.assertEqual(verify_results_gallery(base/'gallery')['status'],'INTEGRITY_VERIFIED')
            with self.assertRaises(FileExistsError):create_results_gallery(request,base/'gallery',base=base)
            (base/'sample image.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'missing or changed'):verify_results_gallery(base/'gallery')

    def test_rejects_remote_url(self):
        with tempfile.TemporaryDirectory() as temp:
            req={'schema':'visual-results-gallery-request-v1','title':'x','intro':'x','items':[{'title':'x','category':'x','description':'x','path':'https://example.com/a.png'}]}
            with self.assertRaises(FileNotFoundError):create_results_gallery(req,Path(temp)/'out',base=temp)
