"""Execute generated gallery JavaScript against a small DOM test double."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from core.research.display import write_combined_gallery, write_manifest_gallery


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed to execute generated gallery JavaScript')
class GallerySearchTests(unittest.TestCase):
    def check_script(self, page):
        scripts=re.findall(r'<script>(.*?)</script>',page,flags=re.S)
        self.assertEqual(len(scripts),1)
        harness=r'''
const vm=require('node:vm');
const assert=require('node:assert/strict');
const cards=[{dataset:{search:'density profile structure'}},
             {dataset:{search:'density power streams'}},
             {dataset:{search:'velocity profile structure'}}];
let inputHandler;
const document={getElementById(id){assert.equal(id,'filter');return {
  addEventListener(event,handler){assert.equal(event,'input');inputHandler=handler;}};},
  querySelectorAll(selector){assert.equal(selector,'article[data-search]');return cards;}};
vm.runInNewContext(SCRIPT,{document});
for(const [query,expected] of [
  ['density',[false,false,true]],
  ['density profile',[false,true,true]],
  ['  PROFILE\tDENSITY  ',[false,true,true]],
  ['structure\nprofile',[false,true,false]],
  ['absent density',[true,true,true]],
  [' \t\n ',[false,false,false]],
  ['', [false,false,false]]]){
  inputHandler.call({value:query});
  assert.deepEqual(cards.map(card=>card.hidden),expected,query);
}
'''
        result=subprocess.run([shutil.which('node'),'-e','const SCRIPT='+json.dumps(scripts[0])+';\n'+harness],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_scientific_gallery_search_splits_whitespace(self):
        with tempfile.TemporaryDirectory() as temp:
            write_manifest_gallery(temp,{'figures':[{
                'figure_id':'density-profile','family':'structure','variant':'density_profile',
                'science_status':'NOT_ACCEPTED','png':'density.png','pdf':'density.pdf',
                'svg':'density.svg','manifest':'density.figure.json','physical_arrays':'density.npz'}],
                'pending_measurements':[]})
            self.check_script((Path(temp)/'index.html').read_text())

    def test_combined_gallery_search_splits_whitespace(self):
        with tempfile.TemporaryDirectory() as temp:
            write_combined_gallery(temp,{'views':[],'entries':[{
                'family':'structure','variant':'density_profile','view_ids':[],
                'measurement_status':'PENDING_MEASURED_INPUT','missing_reason':'No supplied measurement'}]})
            self.check_script((Path(temp)/'index.html').read_text())


if __name__=='__main__':unittest.main()
