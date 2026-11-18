"""Review binding for a zoom target selection; no simulations are launched."""
import hashlib
import json

import pytest

from core.research.zooms import _verify_selection_preview


def _preview(tmp_path, value):
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(value))
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_reviewed_selection_requires_matching_live_catalogue_and_rules(tmp_path):
    selection = {'schema': 'zoom-target-selection-v1', 'snapshot': 3,
                 'rules': {'minimum_dm_particles': 1000},
                 'source_hashes': {'/run/catalogue': 'old'},
                 'targets': [{'host_row': 4}]}
    binding = _preview(tmp_path, selection)
    _verify_selection_preview(binding, tmp_path, selection)
    for changed in ({**selection, 'snapshot': 4},
                    {**selection, 'source_hashes': {'/run/catalogue': 'new'}},
                    {**selection, 'targets': [{'host_row': 5}]},
                    {**selection, 'rules': {'minimum_dm_particles': 500}}):
        with pytest.raises(ValueError, match='differs'):
            _verify_selection_preview(binding, tmp_path, changed)


def test_reviewed_selection_rejects_changed_file_bytes(tmp_path):
    selection = {'schema': 'zoom-target-selection-v1', 'targets': [{'host_row': 1}]}
    binding = _preview(tmp_path, selection)
    (tmp_path / binding['path']).write_text(json.dumps(selection, indent=2))
    with pytest.raises(ValueError, match='changed'):
        _verify_selection_preview(binding, tmp_path, selection)
