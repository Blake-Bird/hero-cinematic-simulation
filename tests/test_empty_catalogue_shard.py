"""Catalogue shards may contain particle IDs while declaring zero group rows."""
import h5py
import numpy as np
import pytest

from core.analysis import _read_match_catalogue_field


def test_declared_empty_group_shard_is_accepted(tmp_path):
    populated, empty = tmp_path/'groups.0.hdf5', tmp_path/'groups.1.hdf5'
    with h5py.File(populated,'w') as source:
        source.create_group('Header').attrs['Ngroups_ThisFile'] = 2
        source['Group/GroupMass'] = np.array([3.,5.])
    with h5py.File(empty,'w') as source:
        source.create_group('Header').attrs['Ngroups_ThisFile'] = 0
        source['IDs/ParticleIDs'] = np.arange(4,dtype=np.uint64)
    actual = _read_match_catalogue_field([populated,empty],'Group',['GroupMass'])
    np.testing.assert_array_equal(actual,[3.,5.])
    with h5py.File(empty,'r+') as source:
        source['Header'].attrs['Ngroups_ThisFile'] = 1
    with pytest.raises(ValueError,match='shard is missing'):
        _read_match_catalogue_field([populated,empty],'Group',['GroupMass'])
