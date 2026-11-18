"""Final share receipts verify live assets without carrying rollback media."""
import json
from pathlib import Path
import pytest
from core.foundation import file_sha256
from core.blender import verify_scene_publication

def publication(tmp_path):
 scene=tmp_path/'universe.blend';scene.write_bytes(b'scene payload')
 asset=tmp_path/'preview.png';asset.write_bytes(b'preview payload')
 record={'schema':'blender-portable-scene-publication-v1','state':'COMPLETE','public_blend_name':scene.name,'aliases':[scene.name,asset.name],'artifacts':{p.name:file_sha256(p) for p in (scene,asset)}}
 pointer=scene.with_name(scene.name+'.publication.json');pointer.write_text(json.dumps(record));return scene,asset,pointer,record

def test_portable_scene_validates_without_history(tmp_path):
 scene,_,_,_=publication(tmp_path)
 assert verify_scene_publication(scene)['state']=='COMPLETE'
 assert not (tmp_path/'.scene_history').exists()

def test_portable_scene_rejects_changed_live_asset(tmp_path):
 scene,asset,_,_=publication(tmp_path);asset.write_bytes(b'changed')
 with pytest.raises(ValueError,match='artifact changed'):verify_scene_publication(scene)

def test_portable_scene_rejects_parent_path(tmp_path):
 scene,_,pointer,record=publication(tmp_path);record['aliases'].append('../outside');record['artifacts']['../outside']='unused';pointer.write_text(json.dumps(record))
 with pytest.raises(ValueError,match='escapes'):verify_scene_publication(scene)
