"""Regression for native catalogue centres coinciding with massive particles."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.halos import (
    ReconstructedHostPotential,
    direct_host_field,
    fit_component_multipole_expansion,
)


def test_central_mass_field_and_persistence():
    agama = pytest.importorskip('agama')
    agama.setUnits(length=1, mass=1, velocity=1)
    masses = np.array([2.e8, 3.e8])
    samples = {'dm': {
        'offsets_proper_kpc': np.zeros((2, 3)),
        'velocities_peculiar_km_s': np.zeros((2, 3)),
        'masses_msun': masses,
        'particle_ids': np.array([10, 20], dtype=np.uint64),
    }}
    geometry = {'dm': {'kind': 'spheroidal', 'disk_rotation': np.eye(3)}}
    settings = dict(lmax=4, mmax=4, grid_size_r=20, grid_size_z=20,
                    rmin_kpc=.05, rmax_kpc=10., zmin_kpc=.05, zmax_kpc=10.)
    queries = np.array([[1., 0, 0], [0, 0, 3.], [4., -1., 2.]])
    binding = {'G_kpc_km2_s2_msun': float(agama.G)}
    with patch('core.halos.pinned_agama_backend', return_value=(agama, binding)):
        potential, _ = fit_component_multipole_expansion(samples, geometry, settings)
        expected = direct_host_field(np.zeros((2, 3)), masses, queries, gravity_constant=agama.G)
        for observed, reference in zip(potential.evaluate(queries), expected):
            np.testing.assert_allclose(observed, reference, rtol=1.e-12, atol=1.e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'coefficients'
            records = potential.export_coefficients(path)
            assert records[0]['central_point_mass_msun'] == masses.sum()
            reloaded = ReconstructedHostPotential.from_coefficients(path, records)
            for observed, reference in zip(reloaded.evaluate(queries), expected):
                np.testing.assert_allclose(observed, reference, rtol=1.e-12, atol=1.e-12)
            records[0]['central_point_mass_msun'] *= 2
            with pytest.raises(ValueError, match='parameters disagree'):
                ReconstructedHostPotential.from_coefficients(path, records)
