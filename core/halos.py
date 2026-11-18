"""Halos: consolidated scientific implementation."""
from __future__ import annotations

# ========================================================================
# halos: Run GADGET FOF/SUBFIND and Rockstar/Consistent Trees on existing snapshots.

# Workflow and units
# Runs Rockstar and Consistent Trees against existing snapshots; FoF/SUBFIND originate in the configured GADGET run.
# Native GADGET units convert to comoving Mpc/h and Msun/h using recorded parameters. Rockstar configuration explicitly matches the mass and velocity conventions.
# Phase-space detection follows Behroozi et al. arXiv:1110.4372. Catalogue IDs/parent links determine associations; proximity alone does not establish a merger history.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
import copy
import datetime
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from core.foundation import file_sha256
from core.native import native_engine_directory
from core.resources import runtime_paths

import h5py
import numpy as np
import yaml
from core.resources import cpu_thread_budget, mpi_ranks
from core.foundation import MPC_CM, MSUN_G
from core.native import native_run_owner, native_run_lock_fds, native_run_lock, native_owned_fds
from core.native import directory_owner
from core.foundation import atomic_json, atomic_text, atomic_copy
from core.data import staged_native_catalogues
from core.native import stop_owned_process_groups, invocation_record, verify_invocation_inputs, run_native_command

ROOT = Path(__file__).resolve().parent

# Block 11 uses the same proper-coordinate conventions as the native shape
# reader.  AGAMA is optional for earlier blocks, and mandatory for this stage.
AGAMA_REVISION = 'f302756b8af2b763db58e278e30478517dc8eea3'


def _halo_samples(sample):
    """Validate aligned complete samples; never turn sampling weights into mass."""
    from core.data import exact_uint64_ids
    x = np.asarray(sample['offsets_proper_kpc'], dtype=float)
    v = np.asarray(sample['velocities_peculiar_km_s'], dtype=float)
    m = np.asarray(sample['masses_msun'], dtype=float)
    ids = exact_uint64_ids(sample['particle_ids'])
    if (x.shape != (len(m), 3) or v.shape != x.shape or ids.shape != m.shape
            or not np.isfinite(x).all() or not np.isfinite(v).all()
            or not np.isfinite(m).all() or np.any(m <= 0)
            or not np.isfinite(m.sum()) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Complete component samples require finite aligned phase space, positive mass and unique exact IDs')
    return x, v, m, ids


def radial_shape_twists(shells, *, orientation_key):
    """Unsigned adjacent-axis angles only where both axes are resolved."""
    rows = []
    for inner,outer in zip(shells,shells[1:]):
        available = (inner.get('status')=='MEASURED' and outer.get('status')=='MEASURED'
                     and 'axes_columns' in inner and 'axes_columns' in outer)
        resolved = np.array(inner.get(orientation_key,[False]*3),dtype=bool)&np.array(outer.get(orientation_key,[False]*3),dtype=bool)
        angles = [None]*3
        if available:
            dot = np.sum(np.asarray(inner['axes_columns'])*np.asarray(outer['axes_columns']),axis=0)
            measured = np.degrees(np.arccos(np.clip(abs(dot),0,1)))
            angles = [float(value) if supported else None for value,supported in zip(measured,resolved)]
        rows.append({'unsigned_axis_twist_deg':angles,
                     'status':'MEASURED' if available and resolved.all() else 'PARTIALLY_RESOLVED' if available and resolved.any() else 'UNRESOLVED_ORIENTATION'})
    return rows


def spherical_velocity_variances(offsets, velocities, masses):
    """Return mass-weighted (var(v_r), var(v_theta), var(v_phi)).

    Offsets and velocities share the declared halo rest frame. The polar axis
    is simulation z; on its poles use e_phi=(0,1,0). Subtract each spherical
    component's measured mean, so ordered rotation is not random dispersion.
    Results have squared velocity units. A particle at the origin has no
    spherical basis; such a selection, or an empty one, returns three NaNs.
    Dispersion-based beta convention:
    https://docs.galpy.org/en/v1.12.0/tutorials/distribution_functions/spherical_dfs.html
    """
    x = np.asarray(offsets, float)
    v = np.asarray(velocities, float)
    m = np.asarray(masses, float)
    if x.ndim != 2 or x.shape[1] != 3 or v.shape != x.shape or m.shape != (len(x),):
        raise ValueError('Aligned N by 3 positions/velocities and N masses required')
    if not all(np.isfinite(a).all() for a in (x,v,m)) or np.any(m <= 0):
        raise ValueError('Finite phase space and positive spherical-moment masses required')
    radius = np.hypot.reduce(x, axis=1)
    if not np.isfinite(radius).all():
        raise FloatingPointError('Spherical moment radius exceeds numeric range')
    if not len(x) or np.any(radius == 0):
        return np.full(3, np.nan)
    radial = x / radius[:,None]
    cylindrical = np.hypot(radial[:,0], radial[:,1])
    azimuthal = np.zeros_like(radial)
    regular = cylindrical > 0
    azimuthal[regular,0] = -radial[regular,1] / cylindrical[regular]
    azimuthal[regular,1] = radial[regular,0] / cylindrical[regular]
    azimuthal[~regular,1] = 1.
    polar = np.cross(azimuthal, radial)
    components = np.column_stack([np.sum(v*basis,axis=1) for basis in (radial,polar,azimuthal)])
    weights = m / m.max()
    weights /= weights.sum()
    components -= np.sum(weights[:,None]*components,axis=0)
    variance = np.sum(weights[:,None]*components**2,axis=0)
    if not np.isfinite(variance).all():
        raise FloatingPointError('Spherical velocity variance exceeds numeric range')
    return variance



def measure_mass_density_kinematics_spin(components, radial_edges_kpc,
                                        bulk_velocity_km_s, *, gravity_constant=None):
    """Complete proper shell census, population covariance and enclosed spin.

    Velocities are peculiar, relative to the explicit host bulk velocity.
    Vc is the spherical-equivalent sqrt(G M(<r)/r), including for a disk.
    Poisson density errors are mass shot noise, not physical convergence.
    """
    from core.foundation import G_MPC_KMS2_MSUN
    g = G_MPC_KMS2_MSUN * 1000 if gravity_constant is None else float(gravity_constant)
    edges = np.asarray(radial_edges_kpc, dtype=float)
    bulk = np.asarray(bulk_velocity_km_s, dtype=float)
    if (edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)
            or bulk.shape != (3,) or not np.isfinite(bulk).all() or not np.isfinite(g) or g <= 0):
        raise ValueError('Declare increasing proper radial edges, finite bulk velocity and positive G')
    if not components or 'total' in components:
        raise ValueError('Named components must be nonempty; total is reserved')
    validated = {name: _halo_samples(s) for name, s in components.items()}
    identities = np.concatenate([s[3] for s in validated.values()])
    if len(np.unique(identities)) != len(identities):
        raise ValueError('Components overlap in the native particle ID namespace')
    all_samples = {**validated, 'total': tuple(np.concatenate([s[i] for s in validated.values()]) for i in range(4))}
    volume = 4 * np.pi / 3 * np.diff(edges**3)
    output = {}
    for name, (x, v, mass, ids) in all_samples.items():
        v = v - bulk
        radius = np.linalg.norm(x, axis=1)
        slots = np.searchsorted(edges, radius, side='right') - 1
        slots[radius == edges[-1]] = len(edges) - 2
        n = len(edges) - 1
        values = {
            'radius_proper_kpc': (edges[:-1] + edges[1:]) / 2,
            'radial_outer_edge_proper_kpc': edges[1:].copy(),
            'shell_count': np.zeros(n, dtype=np.uint64),
            'shell_mass_msun': np.zeros(n),
            'enclosed_mass_msun': np.zeros(n),
            'density_msun_kpc3': np.zeros(n),
            'density_shot_noise_msun_kpc3': np.zeros(n),
            'mean_velocity_km_s': np.full((n, 3), np.nan),
            'velocity_covariance_km2_s2': np.full((n, 3, 3), np.nan),
            'radial_velocity_variance_km2_s2': np.full(n, np.nan),
            'anisotropy_beta': np.full(n, np.nan),
            'angular_momentum_msun_kpc_km_s': np.zeros((n, 3)),
            'enclosed_angular_momentum_msun_kpc_km_s': np.zeros((n, 3)),
            'circular_velocity_spherical_km_s': np.zeros(n),
            'bullock_spin_enclosed': np.full(n, np.nan),
            'pseudo_phase_space_density': np.full(n, np.nan),
            'effective_count': np.zeros(n),
            'kinematics_supported': np.zeros(n, dtype=bool),
        }
        for j in range(n):
            shell = slots == j
            enclosed = radius <= edges[j+1]
            ms, me = mass[shell], mass[enclosed]
            total = float(ms.sum())
            values['shell_count'][j] = len(ms)
            values['shell_mass_msun'][j] = total
            values['enclosed_mass_msun'][j] = me.sum()
            values['density_msun_kpc3'][j] = total / volume[j]
            values['density_shot_noise_msun_kpc3'][j] = np.sqrt(np.sum(ms**2)) / volume[j]
            values['angular_momentum_msun_kpc_km_s'][j] = np.sum(ms[:,None]*np.cross(x[shell],v[shell]),axis=0)
            angular = np.sum(me[:,None]*np.cross(x[enclosed],v[enclosed]),axis=0)
            values['enclosed_angular_momentum_msun_kpc_km_s'][j] = angular
            vc = np.sqrt(g * me.sum() / edges[j+1])
            values['circular_velocity_spherical_km_s'][j] = vc
            if me.sum() > 0 and vc > 0:
                values['bullock_spin_enclosed'][j] = np.linalg.norm(angular)/(np.sqrt(2)*me.sum()*vc*edges[j+1])
            if not len(ms):
                continue
            neff = total**2 / np.sum(ms**2)
            values['effective_count'][j] = neff
            mean = np.average(v[shell], axis=0, weights=ms)
            dv = v[shell] - mean
            covariance = np.einsum('i,ij,ik->jk', ms/total, dv, dv)
            values['mean_velocity_km_s'][j] = mean
            values['velocity_covariance_km2_s2'][j] = covariance
            if neff < 2 or np.any(radius[shell] == 0):
                continue
            variance = spherical_velocity_variances(x[shell], v[shell], ms)
            values['radial_velocity_variance_km2_s2'][j] = variance[0]
            if variance[0] > 0:
                values['anisotropy_beta'][j] = 1 - (variance[1]+variance[2])/(2*variance[0])
            sigma = np.sqrt(np.trace(covariance)/3)
            if sigma > 0:
                values['pseudo_phase_space_density'][j] = total/volume[j]/sigma**3
            values['kinematics_supported'][j] = True
        output[name] = values
    return output


def direct_host_field(positions_proper_kpc, masses_msun, queries_proper_kpc,
                      *, gravity_constant, chunk_rows=4096):
    """Independent isolated Newtonian Phi, acceleration and signed Hessian.

    This is the unsoftened finite particle source used by the expansion.
    The sign is H_ij = d_i d_j Phi; acceleration = -grad(Phi).
    It is never a cosmological periodic or mean-density-subtracted solve.
    """
    x, m, q = map(lambda v: np.asarray(v,dtype=float),
                   (positions_proper_kpc,masses_msun,queries_proper_kpc))
    if (x.shape != (len(m),3) or q.ndim != 2 or q.shape[1] != 3
            or not len(m) or not len(q) or not all(np.isfinite(v).all() for v in (x,m,q))
            or np.any(m <= 0) or not np.isfinite(gravity_constant) or gravity_constant <= 0
            or type(chunk_rows) is not int or chunk_rows <= 0):
        raise ValueError('Finite proper source/query arrays, positive masses, G and workspace are required')
    phi, acc, hessian = np.zeros(len(q)), np.zeros_like(q), np.zeros((len(q),3,3))
    for k, point in enumerate(q):
        for start in range(0,len(m),chunk_rows):
            d = point-x[start:start+chunk_rows]
            r2 = np.sum(d*d,axis=1)
            if np.any(r2 == 0):
                raise ValueError('Direct field query coincides with an unsoftened source')
            weight = gravity_constant*m[start:start+chunk_rows]
            phi[k] -= np.sum(weight/np.sqrt(r2))
            acc[k] -= np.sum((weight/r2**1.5)[:,None]*d,axis=0)
            hessian[k] += np.eye(3)*np.sum(weight/r2**1.5) - 3*np.einsum('i,ij,ik->jk',weight/r2**2.5,d,d)
    if not all(np.isfinite(v).all() for v in (phi,acc,hessian)):
        raise ArithmeticError('Host field overflow')
    return phi, acc, hessian


def pinned_agama_backend():
    """Require the recorded upstream revision, not a moving PyPI shim."""
    import importlib.metadata
    import agama
    from core.foundation import file_sha256 as digest
    dist = importlib.metadata.distribution('agama')
    origin = json.loads(dist.read_text('direct_url.json') or '{}')
    from core.native import verified_python_backend
    pin_path = Path(__file__).resolve().parents[1]/'configs/streams.upstream.json'
    pin = next(item for item in json.loads(pin_path.read_text())['repositories'] if item['name']=='agama')
    if pin['revision'] != AGAMA_REVISION:
        raise ValueError('AGAMA physical backend and source pins disagree')
    installation = verified_python_backend('agama', pin, distribution=dist)
    if str(Path(agama.__file__).resolve()) not in installation['installed_assets']:
        raise ValueError('Imported AGAMA module is outside its verified distribution')
    agama.setUnits(length=1,mass=1,velocity=1)
    sources = {**installation['source_hashes'], str(pin_path):digest(pin_path),
               str(Path(agama.__file__).resolve()):digest(agama.__file__)}
    for entry in dist.files or []:
        if str(entry).endswith(('.so','.dylib','direct_url.json')):
            path = Path(dist.locate_file(entry)).resolve()
            if path.is_file():
                sources[str(path)] = digest(path)
    return agama, {'revision':AGAMA_REVISION,'source_url':'https://github.com/GalacticDynamics-Oxford/Agama',
                   'installed_asset_hashes':sources,'G_kpc_km2_s2_msun':float(agama.G),
                   'units':{'length':'proper kpc','mass':'Msun','velocity':'km/s'}}


class _CentralPointPotential:
    """An exact Newtonian point source represented by a zero-scale Plummer law."""

    def __init__(self, agama, mass_msun):
        self.mass_msun = float(mass_msun)
        if not np.isfinite(self.mass_msun) or self.mass_msun <= 0:
            raise ValueError('Central point mass must be finite and positive')
        self.potential = agama.Potential(type='Plummer', mass=self.mass_msun, scaleRadius=0)

    def eval(self, *args, **kwargs):
        return self.potential.eval(*args, **kwargs)

    def configuration(self):
        # AGAMA exports only the analytic type, so record its parameters explicitly.
        return f'[Potential]\ntype=Plummer\nmass={self.mass_msun:.17g}\nscaleRadius=0\n'

    def export(self, path):
        Path(path).write_text(self.configuration())


class ReconstructedHostPotential:
    """Independent spheroidal multipoles and disk cylindrical splines.

    disk_rotation rows are local disk axes in host proper Cartesian space.
    Stored native coefficients can be reloaded; particles are not needed for
    evaluation.  The representation covers only the declared finite source.
    """
    def __init__(self, components):
        self.components = tuple(components)

    def evaluate(self, positions_proper_kpc):
        q = np.asarray(positions_proper_kpc,dtype=float)
        if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
            raise ValueError('Host field queries require finite N by 3 proper positions')
        phi, acceleration, hessian = np.zeros(len(q)), np.zeros_like(q), np.zeros((len(q),3,3))
        for name, pot, rotation in self.components:
            local_q = q @ rotation.T
            p, a, derivative = pot.eval(local_q,pot=True,acc=True,der=True)
            local = np.zeros_like(hessian)
            for k,(i,j) in enumerate(((0,0),(1,1),(2,2),(0,1),(1,2),(0,2))):
                local[:,i,j] = local[:,j,i] = -derivative[:,k]
            # The pinned expansion's cylindrical-to-Cartesian derivative
            # conversion is singular on R=0. Scalar Phi remains regular.
            # Use fourth-order Cartesian differences of that SAME potential,
            # without perturbing the requested query coordinates.
            if str(pot).startswith(('Multipole','CylSpline')):
                polar = np.hypot(local_q[:,0],local_q[:,1]) <= 1e-5*np.maximum(1.,np.linalg.norm(local_q,axis=1))
                for index in np.flatnonzero(polar):
                    a[index],local[index] = _cartesian_potential_derivatives(pot,local_q[index])
            phi += p
            acceleration += a @ rotation
            hessian += np.einsum('ji,njk,kl->nil',rotation,local,rotation)
        if not all(np.isfinite(v).all() for v in (phi,acceleration,hessian)):
            raise ArithmeticError('Nonfinite reconstructed host field')
        return phi, acceleration, hessian

    def export_coefficients(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        records = []
        for index,(name,potential,rotation) in enumerate(self.components):
            path = directory/('component_%03d.ini'%index)
            potential.export(str(path))
            if '#other parameters are not stored' in path.read_text():
                raise ValueError('AGAMA analytic export is lossy; persist fitted expansion coefficients instead')
            records.append(_potential_coefficient_record(name, potential, path, rotation))
        return records

    @classmethod
    def from_coefficients(cls, directory, records):
        from core.foundation import file_sha256 as digest
        agama, binding = pinned_agama_backend()
        components = []
        for record in records:
            path = (Path(directory)/record['file']).resolve()
            if not path.is_relative_to(Path(directory).resolve()) or digest(path) != record['sha256']:
                raise ValueError('Potential coefficient binding mismatch')
            rotation = _host_rotation(record['disk_rotation'])
            if 'central_point_mass_msun' in record:
                potential = _CentralPointPotential(agama, record['central_point_mass_msun'])
                if path.read_text() != potential.configuration():
                    raise ValueError('Central point coefficient parameters disagree')
            else:
                potential = agama.Potential(str(path))
            components.append((record['name'],potential,rotation))
        if not components:
            raise ValueError('Empty reconstructed host potential')
        return cls(components)



def _potential_coefficient_record(name, potential, path, rotation):
    from core.foundation import file_sha256
    record = {'name': name, 'file': path.name, 'sha256': file_sha256(path), 'disk_rotation': rotation.tolist()}
    if isinstance(potential, _CentralPointPotential):
        record['central_point_mass_msun'] = potential.mass_msun
    return record


def _cartesian_potential_derivatives(potential, point):
    """Fourth-order differences, including Richardson mixed derivatives."""
    step = 1e-3*max(1.,float(np.linalg.norm(point)))
    origin = float(potential.potential(point))
    gradient = np.zeros(3)
    hessian = np.zeros((3,3))
    basis = np.eye(3)
    for i in range(3):
        values = potential.potential(point[None,:]+np.array([-2.,-1.,1.,2.])[:,None]*step*basis[i])
        gradient[i] = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        hessian[i,i] = (-values[3]+16*values[2]-30*origin+16*values[1]-values[0])/(12*step**2)
        for j in range(i):
            def mixed(h):
                corners = point+np.array([basis[i]+basis[j],basis[i]-basis[j],-basis[i]+basis[j],-basis[i]-basis[j]])*h
                values = potential.potential(corners)
                return (values[0]-values[1]-values[2]+values[3])/(4*h*h)
            hessian[i,j] = hessian[j,i] = (4*mixed(step/2)-mixed(step))/3
    return -gradient,hessian


def _host_rotation(value):
    rotation = np.asarray(value,dtype=float)
    if (rotation.shape != (3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T,np.eye(3),rtol=0,atol=1e-12)
            or not np.isclose(np.linalg.det(rotation),1,rtol=0,atol=1e-12)):
        raise ValueError('Disk axes must form a proper orthonormal rotation')
    return rotation


def fit_component_multipole_expansion(components, geometry, settings):
    """Fit the real native AGAMA expansions; no density-shape surrogate."""
    agama, binding = pinned_agama_backend()
    required = {'lmax','mmax','grid_size_r','grid_size_z','rmin_kpc','rmax_kpc','zmin_kpc','zmax_kpc'}
    if set(settings) != required or set(geometry) != set(components):
        raise ValueError('Declare expansion order, radial/vertical grid domain and every component geometry')
    for key,lower,upper in [('lmax',0,32),('mmax',0,32),('grid_size_r',5,1024),('grid_size_z',5,1024)]:
        if type(settings[key]) is not int or not lower <= settings[key] <= upper:
            raise ValueError('Unsupported expansion control '+key)
    for axis in ['r','z']:
        lo,hi = settings[axis+'min_kpc'],settings[axis+'max_kpc']
        if not np.isfinite([lo,hi]).all() or not 0 < lo < hi:
            raise ValueError('Positive increasing native grid domain required')
    models = []
    for name,sample in components.items():
        x,v,m,ids = _halo_samples(sample)
        definition = geometry[name]
        if set(definition) != {'kind','disk_rotation'} or definition['kind'] not in {'spheroidal','disk'}:
            raise ValueError('Select spheroidal multipole or disk cylindrical representation')
        rotation = _host_rotation(definition['disk_rotation'])
        if not len(m):
            continue
        at_origin = np.all(x == 0, axis=1)
        if np.any(at_origin):
            # The expansion excludes r=0; the exact point contribution is
            # added back with its original mass and position.
            models.append((f'{name}:central_point', _CentralPointPotential(agama, m[at_origin].sum()), rotation))
            x, m = x[~at_origin], m[~at_origin]
            if not len(m):
                continue
        common = dict(particles=(x @ rotation.T,m),gridSizeR=settings['grid_size_r'],
                      rmin=settings['rmin_kpc'],rmax=settings['rmax_kpc'],symmetry='None')
        if definition['kind'] == 'disk':
            potential = agama.Potential(type='CylSpline',mmax=settings['mmax'],
                gridSizeZ=settings['grid_size_z'],zmin=settings['zmin_kpc'],zmax=settings['zmax_kpc'],**common)
        else:
            potential = agama.Potential(type='Multipole',lmax=settings['lmax'],mmax=min(settings['mmax'],settings['lmax']),**common)
        models.append((name,potential,rotation))
    if not models:
        raise ValueError('Cannot reconstruct an empty physical population')
    return ReconstructedHostPotential(models), binding


def compare_with_independent_direct_force_samples(model, components, queries, budget, *, gravity_constant):
    """Frozen dimensional floors and worst-case force/tidal residuals."""
    if set(budget) != {'force_relative_tolerance','force_floor_km2_s2_kpc','tidal_relative_tolerance','tidal_floor_km2_s2_kpc2'}:
        raise ValueError('Declare separate force and tidal tolerances and dimensional floors')
    if not all(not isinstance(v,bool) and np.isfinite(v) and v > 0 for v in budget.values()):
        raise ValueError('Positive finite force/tidal error controls required')
    validated = [_halo_samples(s) for s in components.values()]
    truth = direct_host_field(np.concatenate([s[0] for s in validated]),
        np.concatenate([s[2] for s in validated]),queries,gravity_constant=gravity_constant)
    predicted = model.evaluate(queries)
    force_error = np.linalg.norm(predicted[1]-truth[1],axis=1)
    tidal_error = np.linalg.norm(predicted[2]-truth[2],axis=(1,2))
    force_limit = budget['force_relative_tolerance']*np.maximum(np.linalg.norm(truth[1],axis=1),budget['force_floor_km2_s2_kpc'])
    tidal_limit = budget['tidal_relative_tolerance']*np.maximum(np.linalg.norm(truth[2],axis=(1,2)),budget['tidal_floor_km2_s2_kpc2'])
    return {'status':'PASS' if np.all(force_error <= force_limit) and np.all(tidal_error <= tidal_limit) else 'FAIL',
            'worst_force_budget_fraction':float(np.max(force_error/force_limit)),
            'worst_tidal_budget_fraction':float(np.max(tidal_error/tidal_limit)),
            'reference_definition':'isolated unsoftened Newtonian finite source; not periodic simulation force'}, {
            'query_positions_proper_kpc':np.asarray(queries,dtype=float),
            'potential_reconstructed_km2_s2':predicted[0],'potential_direct_km2_s2':truth[0],
            'acceleration_reconstructed_km2_s2_kpc':predicted[1],'acceleration_direct_km2_s2_kpc':truth[1],
            'hessian_reconstructed_km2_s2_kpc2':predicted[2],'hessian_direct_km2_s2_kpc2':truth[2],
            'force_absolute_error':force_error,'tidal_absolute_error':tidal_error,
            'force_allowed_error':force_limit,'tidal_allowed_error':tidal_limit}


def measure_isopotential_geometry(model, reference_radii_kpc, *, radial_bracket_kpc,
                                  directions, maximum_shape_residual, minimum_eigenvalue_gap):
    """Ray-root isopotentials, independent of density moments and contours.

    Each level is the directional mean Phi at its reference radius. More than
    one supported root on any ray is ambiguous and never silently selected.
    """
    from scipy.optimize import brentq
    radii = np.asarray(reference_radii_kpc,dtype=float)
    bracket = np.asarray(radial_bracket_kpc,dtype=float)
    if (radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all()
            or bracket.shape != (2,) or not np.isfinite(bracket).all()
            or not 0 < bracket[0] < bracket[1] or np.any((radii <= bracket[0]) | (radii >= bracket[1]))
            or type(directions) is not int or not 24 <= directions <= 100000
            or not 0 < maximum_shape_residual < 1 or not 0 < minimum_eigenvalue_gap < 1):
        raise ValueError('Declare supported radii, root bracket, directions, residual and degeneracy limits')
    index = np.arange(directions)
    z = 1 - 2*(index+0.5)/directions
    angle = index*np.pi*(3-np.sqrt(5))
    rays = np.column_stack((np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z))
    design = np.column_stack((rays**2,2*rays[:,0]*rays[:,1],2*rays[:,1]*rays[:,2],2*rays[:,0]*rays[:,2]))
    scan = np.geomspace(*bracket,96)
    rows = []
    for radius in radii:
        level = float(np.mean(model.evaluate(rays*radius)[0]))
        roots = np.full(directions,np.nan)
        failures = []
        for j,ray in enumerate(rays):
            phi = model.evaluate(scan[:,None]*ray)[0]-level
            crossings = np.flatnonzero(phi[:-1]*phi[1:] < 0)
            exact = np.flatnonzero(phi == 0)
            if len(crossings)+len(exact) != 1:
                failures.append(j);continue
            if len(exact):
                roots[j] = scan[exact[0]]
            else:
                k = crossings[0]
                roots[j] = brentq(lambda r: float(model.evaluate((ray*r)[None,:])[0][0]-level),scan[k],scan[k+1])
        row = {'reference_radius_proper_kpc':float(radius),'potential_level_km2_s2':level,
               'ray_radii_proper_kpc':[float(v) if np.isfinite(v) else None for v in roots],'status':'UNSUPPORTED_RAY_BRACKET' if failures else 'MEASURED',
               'failed_directions':failures,'orientation_resolved':[False]*3}
        if not failures:
            coeff = np.linalg.lstsq(design,roots**-2,rcond=None)[0]
            tensor = np.array([[coeff[0],coeff[3],coeff[5]],[coeff[3],coeff[1],coeff[4]],[coeff[5],coeff[4],coeff[2]]])
            eig,axes = np.linalg.eigh(tensor)
            if eig[0] <= 0:
                row['status'] = 'NON_ELLIPSOIDAL'
            else:
                lengths = eig**-0.5
                residual = float(np.max(abs((design @ coeff)**-0.5/roots-1))) if np.all(design @ coeff > 0) else float('inf')
                gaps = np.diff(eig)/eig[-1]
                row.update(axis_lengths_proper_kpc=lengths.tolist(),axes_columns=axes.tolist(),
                           b_over_a=float(lengths[1]/lengths[0]),c_over_a=float(lengths[2]/lengths[0]),
                           triaxiality=float((lengths[0]**2-lengths[1]**2)/(lengths[0]**2-lengths[2]**2)) if gaps.sum()>minimum_eigenvalue_gap else None,
                           maximum_ray_shape_residual=residual,
                           orientation_resolved=[bool(gaps[0]>=minimum_eigenvalue_gap),bool(min(gaps)>=minimum_eigenvalue_gap),bool(gaps[1]>=minimum_eigenvalue_gap)],
                           status='MEASURED' if residual<=maximum_shape_residual else 'NON_ELLIPSOIDAL')
        rows.append(row)
    return {'schema':'host-isopotential-geometry-v1','levels':rows,
            'definition':'least-squares ellipsoid to independent ray roots of proper isolated Phi',
            'sampling_uncertainty':'not estimated; directional quadrature and field validation are separate',
            'science_acceptance':'NOT_ACCEPTED'}


def measure_declared_concentration(profiles, request):
    """NFW log-density fit with a measured total overdensity-radius bracket.

    Reference density and delta are explicitly supplied at the actual epoch;
    no LCDM virial formula or cosmology-independent M200 default is used.
    Failed/multiple crossings and inadequate radial support remain explicit.
    """
    from scipy.optimize import least_squares
    required = {'reference_density_msun_kpc3','overdensity_delta','definition_label',
                'fit_min_kpc','fit_max_kpc','rs_min_kpc','rs_max_kpc','minimum_effective_count'}
    if set(request) != required or not isinstance(request['definition_label'],str) or not request['definition_label']:
        raise ValueError('Declare the concentration model, actual reference density and full fit domain')
    if not all(not isinstance(v,bool) and np.isfinite(v) and v>0 for k,v in request.items() if k!='definition_label'):
        raise ValueError('Positive finite concentration controls required')
    if request['fit_min_kpc'] >= request['fit_max_kpc'] or request['rs_min_kpc'] >= request['rs_max_kpc']:
        raise ValueError('Increasing concentration fit and scale-radius bounds required')
    total = profiles['total']
    outer = np.asarray(total['radial_outer_edge_proper_kpc'])
    enclosed = total['enclosed_mass_msun']
    ratio = enclosed/(4*np.pi/3*outer**3)/(request['reference_density_msun_kpc3']*request['overdensity_delta'])
    # Exactly equal endpoints are included once, avoiding double bracket counts.
    exact = np.flatnonzero(ratio == 1)
    crossings = np.flatnonzero((ratio[:-1]>1)&(ratio[1:]<1))
    if len(exact)+len(crossings) != 1:
        return {'status':'AMBIGUOUS_OVERDENSITY' if len(exact)+len(crossings)>1 else 'UNBRACKETED_OVERDENSITY',
                'model':'NFW','definition':request,'components':{}}
    if len(exact):
        r_delta = float(outer[exact[0]])
    else:
        i = crossings[0]
        r_delta = float(np.exp(np.interp(0,[np.log(ratio[i+1]),np.log(ratio[i])],
                                             [np.log(outer[i+1]),np.log(outer[i])])))
    result = {'status':'MEASURED','model':'NFW log-density least squares with mass-shot-noise weights',
              'definition':request,'r_delta_proper_kpc':r_delta,'components':{},
              'scope':'fit conditional on NFW model; radius sampled at measured bin edges; not a convergence certificate'}
    for name,p in profiles.items():
        r,rho,error = p['radius_proper_kpc'],p['density_msun_kpc3'],p['density_shot_noise_msun_kpc3']
        mask = (r>=request['fit_min_kpc'])&(r<=request['fit_max_kpc'])&(rho>0)&(error>0)&(p['effective_count']>=request['minimum_effective_count'])
        row = {'status':'INSUFFICIENT_BINS','selected_bins':np.flatnonzero(mask).tolist()}
        if mask.sum() >= 5:
            rr,yy,ss = r[mask],np.log(rho[mask]),error[mask]/rho[mask]
            def residual(theta):
                u = rr/np.exp(theta[1])
                return (theta[0]-np.log(u)-2*np.log1p(u)-yy)/ss
            fit = least_squares(residual,[float(np.median(yy)),np.log(np.sqrt(request['rs_min_kpc']*request['rs_max_kpc']))],
                bounds=([-np.inf,np.log(request['rs_min_kpc'])],[np.inf,np.log(request['rs_max_kpc'])]))
            normal = fit.jac.T @ fit.jac
            rank = int(np.linalg.matrix_rank(normal))
            row.update(status='MEASURED' if fit.success and not np.any(fit.active_mask) and rank==2 else 'FIT_UNRESOLVED',
                       rs_proper_kpc=float(np.exp(fit.x[1])),concentration=float(r_delta/np.exp(fit.x[1])),
                       rho_s_msun_kpc3=float(np.exp(fit.x[0])),chi_squared=float(np.sum(fit.fun**2)),
                       degrees_of_freedom=int(mask.sum()-2),fit_rank=rank,
                       covariance_log_parameters=np.linalg.inv(normal).tolist() if rank==2 else None,
                       uncertainty_scope='local conditional shot-noise fit covariance, excludes model and resolution error')
        result['components'][name] = row
    return result


def load_native_host_component_samples(run_directory, snapshot, host_row, definitions,
                                       support_radius_kpc, *, memory_gib):
    """Read complete selected native populations without sampling or float IDs.

    Each component defaults to aperture_all. Exact subfind_bound_associated and
    subfind_central_bound selections preserve native object self-binding and
    association definitions, never a global host-binding claim.
    """
    from analysis import validate_gadget_snapshot_set
    from zoom import read_host_catalogue
    from core.data import (verify_native_hdf5_set,native_unit_factor,exact_uint64_ids,
                      run_epoch_contract,read_species_phase_space)
    from core.foundation import file_sha256 as digest, KPC_CM, MSUN_G
    if (type(snapshot) is not int or snapshot<0 or type(host_row) is not int or host_row<0
            or not np.isfinite([support_radius_kpc,memory_gib]).all()
            or min(support_radius_kpc,memory_gib)<=0 or not definitions):
        raise ValueError('Declare native host row, snapshot, complete support and memory')
    species_seen = set()
    for name,row in definitions.items():
        if (not isinstance(name,str) or not name.isidentifier() or name=='total'
                or {'species','kind','disk_rotation'}-set(row) or set(row)-{'species','kind','disk_rotation','membership'}
                or not isinstance(row['species'],list)
                or not row['species'] or any(type(s) is not int or not 0<=s<=5 for s in row['species'])
                or any(s in species_seen for s in row['species']) or len(set(row['species']))!=len(row['species'])):
            raise ValueError('Distinct component names and nonoverlapping explicit native species required')
        if row.get('membership','aperture_all') not in {'aperture_all','subfind_bound_associated','subfind_central_bound'}:
            raise ValueError('Unsupported component membership; global host binding is not inferred')
        species_seen.update(row['species'])
        _host_rotation(row['disk_rotation'])
        if row['kind'] not in {'spheroidal','disk'}:
            raise ValueError('Component geometry must be spheroidal or disk')
    run = Path(run_directory).resolve()
    files = _snapshot_files(run,snapshot)
    validate_gadget_snapshot_set(files,snapshot)
    native = verify_native_hdf5_set(files,require_publication=True)
    _unit_conversion(files)
    catalogue = read_host_catalogue(run,snapshot,memory_gib=memory_gib,host_row=host_row)
    center = np.asarray(catalogue['positions'][host_row])
    a,box,h = catalogue['frame']
    if support_radius_kpc >= box*a/h/2:
        raise ValueError('Host-local proper support must be strictly inside half the periodic box')
    sources = {str(Path(p).resolve()):digest(p) for p in files}
    for row in catalogue['sources']+catalogue['snapshot_sources']:
        sources[row['path']] = row['sha256']
    from core.data import native_publication_markers
    for marker in native_publication_markers(files):
        sources[str(marker.resolve())] = digest(marker)
    arrays = {name:[[],[],[],[]] for name in definitions}
    limit = int(memory_gib*1024**3)
    if limit < 16*1024**2:
        raise MemoryError('Native host reader requires at least 16 MiB')
    chunk = min(250000,max(1,limit//2048))
    selected = 0
    membership_ids = {};membership_records = {};membership_bytes = 0
    census = {name:{'aperture_count':0,'aperture_mass_msun':0.,'selected_count':0,'selected_mass_msun':0.}
              for name in definitions}
    for name,definition in definitions.items():
        mode = definition.get('membership','aperture_all')
        membership_records[name] = {'mode':mode,'species':{}}
        if mode!='aperture_all':
            for species in definition['species']:
                ids,record = native_subfind_bound_member_ids(run,snapshot,host_row,particle_type=species,
                    central_only=mode=='subfind_central_bound',memory_gib=(limit-membership_bytes)/1024**3)
                membership_ids[species] = ids;membership_records[name]['species'][str(species)] = record
                membership_bytes += ids.nbytes*4
                sources.update(record['source_hashes'])
    epoch = run_epoch_contract(run,snapshot,a)
    for path in files:
        with h5py.File(path,'r') as source:
            if source['Parameters'].attrs.get('ComovingIntegrationOn')!=1:
                raise ValueError('Native proper host conversion requires explicit cosmological comoving integration')
            for name,definition in definitions.items():
                for species in definition['species']:
                    group = source.get('PartType'+str(species))
                    if group is None:continue
                    for start in range(0,len(group['ParticleIDs']),chunk):
                        state = read_species_phase_space(group,epoch=epoch,start=start,
                            stop=min(start+chunk,len(group['ParticleIDs'])),
                            memory_bytes=limit-membership_bytes-selected*512-16*1024**2)
                        position = state['position_comoving_kpc_h']
                        d = (position-center+box/2)%box-box/2
                        d *= native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h)
                        raw_ids,mass = state['particle_ids'],state['mass_msun']
                        keep = (np.linalg.norm(d,axis=1)<=support_radius_kpc)&state['live_supported']
                        census[name]['aperture_count'] += int(keep.sum())
                        census[name]['aperture_mass_msun'] += float(mass[keep].sum())
                        if species in membership_ids:
                            members = membership_ids[species]
                            slot = np.searchsorted(members,raw_ids)
                            found = slot<len(members)
                            found[found] &= members[slot[found]]==raw_ids[found]
                            keep &= found
                        census[name]['selected_count'] += int(keep.sum())
                        census[name]['selected_mass_msun'] += float(mass[keep].sum())
                        if not np.any(keep):continue
                        selected += int(keep.sum())
                        if membership_bytes+selected*512+chunk*160+16*1024**2 > limit:
                            raise MemoryError('Complete component samples exceed declared workspace; reduce science aperture or raise memory')
                        ids = exact_uint64_ids(raw_ids[keep])
                        velocity = state['velocity_peculiar_kms']
                        for blocks,value in zip(arrays[name],[d[keep],velocity[keep],mass[keep],ids]):
                            blocks.append(value)
    components = {}
    for name,blocks in arrays.items():
        values = [np.concatenate(b) if b else np.empty((0,3) if i<2 else 0,dtype=np.uint64 if i==3 else float)
                  for i,b in enumerate(blocks)]
        components[name] = dict(zip(['offsets_proper_kpc','velocities_peculiar_km_s','masses_msun','particle_ids'],values))
        _halo_samples(components[name])
    for p,expected in sources.items():
        if digest(p)!=expected:raise RuntimeError('Native halo source changed during complete read')
    for name,row in census.items():
        row['excluded_count'] = row['aperture_count']-row['selected_count']
        row['excluded_mass_msun'] = row['aperture_mass_msun']-row['selected_mass_msun']
    return components, {'epoch':epoch,'component_membership':membership_records,'selection_census':census,
        'scale_factor':a,'h':h,'center_comoving_kpc_h':center.tolist(),
        'r200c_proper_kpc':float(catalogue['radii'][host_row]*a/h),
        'support_radius_proper_kpc':support_radius_kpc,
        'selection':'complete declared component membership within a minimum-image proper support sphere; default aperture retains substructure',
        'source_hashes':sources,'native_publication':native}


@directory_owner('destination')
def publish_halo_structure(request, destination, *, request_source_hashes=None):
    """Connected Block 11 native producer, with frozen calibration and holdout.

    Calibration locations choose refinement. Independent validation locations
    are used once after selection; failure never triggers tuning on the holdout.
    All attempted calibration results and failed geometric bins are retained.
    """
    from core.foundation import file_sha256 as digest, product_provenance
    from core.data import export_numeric_audit,coordinate_contract
    required = {'schema','run_directory','snapshot','host_row','components',
                'support_radius_proper_kpc','radial_edges_proper_kpc','bulk_velocity_peculiar_km_s',
                'shape_controls','expansion_sequence','calibration_positions_proper_kpc',
                'validation_positions_proper_kpc','field_error_budget','equipotential_controls',
                'concentration','memory_gib'}
    if set(request)!=required or request['schema']!='native-halo-structure-v1':
        raise ValueError('Declare the complete native-halo-structure-v1 request')
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.with_name(destination.name+'.data').exists():
        raise FileExistsError('Keep earlier host structure evidence')
    calibration = np.asarray(request['calibration_positions_proper_kpc'],dtype=float)
    validation = np.asarray(request['validation_positions_proper_kpc'],dtype=float)
    for values in [calibration,validation]:
        if values.ndim!=2 or values.shape[1]!=3 or len(values)<3 or not np.isfinite(values).all():
            raise ValueError('At least three finite calibration and withheld validation positions required')
    if set(map(tuple,calibration)) & set(map(tuple,validation)):
        raise ValueError('Calibration and withheld validation locations overlap')
    sequence = request['expansion_sequence']
    if not isinstance(sequence,list) or not 1<=len(sequence)<=12:
        raise ValueError('A finite, explicitly bounded expansion refinement sequence is required')
    for previous,current in zip(sequence,sequence[1:]):
        if (any(current[k]<previous[k] for k in ['lmax','mmax','grid_size_r','grid_size_z'])
                or all(current[k]==previous[k] for k in ['lmax','mmax','grid_size_r','grid_size_z'])):
            raise ValueError('Refinement must increase angular order or spatial grid resolution')
        if any(current[k]!=previous[k] for k in ['rmin_kpc','rmax_kpc','zmin_kpc','zmax_kpc']):
            raise ValueError('Expansion refinement cannot change the declared physical domain')
    components,context = load_native_host_component_samples(request['run_directory'],request['snapshot'],
        request['host_row'],request['components'],request['support_radius_proper_kpc'],memory_gib=request['memory_gib'])
    if request['radial_edges_proper_kpc'][-1]>request['support_radius_proper_kpc']:
        raise ValueError('Radial bins extend beyond the complete physical source support')
    geometry = {name:{k:v for k,v in row.items() if k not in {'species','membership'}} for name,row in request['components'].items()}
    history = []
    arrays = {}
    for iteration,settings in enumerate(sequence):
        particles = sum(len(s['masses_msun']) for s in components.values())
        grid_estimate = settings['grid_size_r']*((settings['lmax']+1)**2+settings['grid_size_z']*(2*settings['mmax']+1))*512
        if particles*512+grid_estimate+16*1024**2 > request['memory_gib']*1024**3:
            raise MemoryError('Expansion exceeds declared conservative workspace estimate')
        model,binding = fit_component_multipole_expansion(components,geometry,settings)
        check,data = compare_with_independent_direct_force_samples(model,components,calibration,
            request['field_error_budget'],gravity_constant=binding['G_kpc_km2_s2_msun'])
        history.append({'settings':settings,'calibration':check})
        for key,value in data.items():arrays['calibration_%02d_%s'%(iteration,key)] = value
        if check['status']=='PASS':break
    heldout,data = compare_with_independent_direct_force_samples(model,components,validation,
        request['field_error_budget'],gravity_constant=binding['G_kpc_km2_s2_msun'])
    for key,value in data.items():arrays['validation_'+key] = value
    profiles = measure_mass_density_kinematics_spin(components,request['radial_edges_proper_kpc'],
        request['bulk_velocity_peculiar_km_s'],gravity_constant=binding['G_kpc_km2_s2_msun'])
    for name,profile in profiles.items():
        profile['radial_outer_edge_proper_kpc'] = np.asarray(request['radial_edges_proper_kpc'][1:],dtype=float)
        for key,value in profile.items():arrays['profile_'+name+'_'+key] = value
    density_shapes = {}
    shape_components = {**components,'total':{
        key:np.concatenate([sample[key] for sample in components.values()])
        for key in ['offsets_proper_kpc','masses_msun']}}
    for name,sample in shape_components.items():
        density_shapes[name] = radial_density_shapes(sample['offsets_proper_kpc'],sample['masses_msun'],
            request['radial_edges_proper_kpc'],support_radius_kpc=request['support_radius_proper_kpc'],
            **request['shape_controls'])
        density_shapes[name]['radial_twists'] = radial_shape_twists(density_shapes[name]['shells'],
            orientation_key='orientation_sampling_resolved')
    shapes = measure_isopotential_geometry(model,**request['equipotential_controls'])
    shapes['radial_twists'] = radial_shape_twists(shapes['levels'],orientation_key='orientation_resolved')
    concentration = measure_declared_concentration(profiles,request['concentration']) if request['concentration'] is not None else {'status':'NOT_REQUESTED'}
    coefficients = []
    for index,(name,potential,rotation) in enumerate(model.components):
        path = destination.with_name(destination.name+'.component_%03d.ini'%index)
        if path.exists():raise FileExistsError('Keep earlier potential coefficients')
        potential.export(str(path))
        if '#other parameters are not stored' in path.read_text():
            raise ValueError('Only reloadable fitted potential coefficients may be published')
        coefficients.append(_potential_coefficient_record(name, potential, path, rotation))
    sources = {**context['source_hashes'],**(request_source_hashes or {}),
               str(Path(__file__).resolve()):digest(__file__)}
    for module in ['data.py','foundation.py','zoom.py','analysis.py']:
        path = ROOT/module;sources[str(path)] = digest(path)
    sources.update(binding['installed_asset_hashes'])
    # Every raw parent is retained with exact native IDs for remeasurement.
    for name,sample in components.items():
        for key,value in sample.items():arrays['parent_'+name+'_'+key] = value
    def units(key):
        if key.endswith('particle_ids'):return 'identity'
        if 'angular_momentum' in key:return 'Msun proper kpc km/s'
        if 'pseudo_phase_space' in key:return 'Msun/(proper kpc)^3/(km/s)^3'
        if 'density' in key:return 'Msun/(proper kpc)^3'
        if 'hessian' in key or 'tidal_' in key:return '(km/s)^2/(proper kpc)^2'
        if 'acceleration' in key or 'force_' in key:return '(km/s)^2/proper kpc'
        if 'potential_' in key:return '(km/s)^2'
        if 'variance' in key or 'covariance' in key:return '(km/s)^2'
        if 'mass' in key:return 'Msun'
        if 'velocity' in key or 'velocities' in key:return 'km/s'
        if 'kpc' in key:return 'proper kpc'
        return 'dimensionless'
    specs = {key:{'units':units(key),'quantity':key,
                  'missing_policy':'NaN for empty, degenerate or unsupported measurements; support flags retained'}
             for key in arrays}
    manifest_path = Path(request['run_directory'])/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest_path.is_file():sources[str(manifest_path.resolve())] = digest(manifest_path)
    coordinates = coordinate_contract(position_units='proper kpc',velocity_convention='peculiar km/s',
        origin='fixed native FOF centre recorded in global comoving kpc/h',
        periodic_wrapping='minimum-image source selection; subsequent field is isolated and proper')
    provenance = product_provenance(request['run_directory'],algorithm=__file__,
        selection=context['selection'],source_files=[{'path':p,'sha256':v} for p,v in context['source_hashes'].items()],
        coordinates=coordinates,snapshot=request['snapshot'])
    for entry in (provenance['initial_conditions']+list(provenance['boltzmann_inputs'].values())+
            [provenance['executable'],provenance['parameter_file'],provenance['compile_options']]):
        if entry.get('status')=='VERIFIED':sources[entry['path']]=entry['sha256']
    config_path = Path(request['run_directory'])/'config.yaml'
    if config_path.is_file():sources[str(config_path.resolve())]=digest(config_path)
    for key in arrays:
        if key.endswith('particle_ids'):specs[key]['identity_namespace'] = provenance['identity_namespaces']['native_particle']
    report = {'schema':'native-halo-structure-v1','status':'MEASURED',
              'field_validation_status':'PASS' if history[-1]['calibration']['status']=='PASS' and heldout['status']=='PASS' else 'UNRESOLVED',
              'request':request,'source_context':context,'source_hashes':sources,'backend':binding,'data_contract':provenance,
              'refinement_history':history,'heldout_field_validation':heldout,
              'potential_coefficients':coefficients,'density_shapes':density_shapes,
              'equipotential_shapes':shapes,'concentration':concentration,
              'field_scope':'finite-source isolated proper Newtonian reconstruction; cosmological periodic peculiar potential is a separate Block 12 product',
              'source_origin':'synthetic_control' if manifest.get('fixture') or manifest.get('synthetic_inputs') else 'native_snapshot',
              'simulation_resolution_convergence':'NOT_EVALUATED'}
    result = export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,
        context={'epoch':context['epoch'],'coordinates':coordinates,'selection':context['selection'],'provenance':provenance},source_hashes=sources)
    verify_halo_structure(destination)
    return result


def verify_halo_structure(path):
    """Verify sealed arrays, source/binary bindings and reloadable coefficients."""
    from core.data import verify_numeric_audit
    from core.foundation import file_sha256 as digest, verify_product_provenance
    path = Path(path).resolve()
    report = verify_numeric_audit(path)
    if report.get('schema')!='native-halo-structure-v1':
        raise ValueError('Not a native halo-structure product')
    for source,expected in report['source_hashes'].items():
        if digest(source)!=expected:raise ValueError('Halo structure source changed: '+source)
    verify_product_provenance(report['request']['run_directory'],report['data_contract'])
    model = ReconstructedHostPotential.from_coefficients(path.parent,report['potential_coefficients'])
    with np.load(path.with_name(path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
        values = model.evaluate(saved['validation_query_positions_proper_kpc'])
        for actual,key in zip(values,['potential_reconstructed_km2_s2','acceleration_reconstructed_km2_s2_kpc','hessian_reconstructed_km2_s2_kpc2']):
            expected = saved['validation_'+key]
            # Exported coefficients have finite decimal precision. Axis-safe
            # finite differences amplify that roundoff by 1/h^2; allow only
            # the calculated floating-point envelope, not a science budget.
            q = saved['validation_query_positions_proper_kpc']
            step = 1e-3*np.maximum(1.,np.linalg.norm(q,axis=1))
            order = 2 if 'hessian' in key else 1 if 'acceleration' in key else 0
            envelope = 256*np.finfo(float).eps*np.maximum(1.,abs(values[0]))/step**order
            envelope = envelope.reshape((-1,)+(1,)*(actual.ndim-1))
            if not np.all(abs(actual-expected) <= envelope + 5e-12*abs(expected)):
                raise ValueError('Reloaded host coefficients do not reproduce their measured field')
    return report


def plot_halo_structure(path, output_prefix):
    """Publish a registered, source-bound profile and field-validation figure."""
    from figures import export_figure,verify_figure_export
    from core.foundation import file_sha256 as digest
    import matplotlib.pyplot as plt
    path = Path(path).resolve()
    report = verify_halo_structure(path)
    before = digest(path)
    with np.load(path.with_name(path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
        fig,axes = plt.subplots(2,2,figsize=(10,8))
        names = [*report['request']['components'],'total']
        for name in names:
            prefix = 'profile_'+name+'_'
            r = saved[prefix+'radius_proper_kpc']
            density = saved[prefix+'density_msun_kpc3']
            good = density>0
            axes[0,0].loglog(r[good],density[good],label=name)
            axes[0,1].plot(r,saved[prefix+'circular_velocity_spherical_km_s'],label=name)
            axes[1,0].plot(r,saved[prefix+'anisotropy_beta'],label=name)
        force = saved['validation_force_absolute_error']/saved['validation_force_allowed_error']
        tides = saved['validation_tidal_absolute_error']/saved['validation_tidal_allowed_error']
        axes[1,1].plot(np.arange(len(force)),force,'o',label='force')
        axes[1,1].plot(np.arange(len(tides)),tides,'s',label='Hessian')
        axes[1,1].axhline(1,color='black',linestyle='--',label='frozen budget')
        for ax,title,label in zip(axes.ravel(),
                ['Complete component density','Spherical-equivalent circular speed','Local spherical anisotropy','Withheld field validation'],
                ['Density [Msun / proper kpc³]','Vc [km/s]','β = 1 − (σθ² + σφ²) / (2σr²)','Error / allowed error']):
            ax.set_title(title,fontsize=11);ax.set_ylabel(label,fontsize=10);ax.legend(fontsize=8)
            ax.tick_params(axis='both',which='both',labelsize=9)
        for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xlabel('Proper radius [kpc]',fontsize=10)
        axes[1,1].set_xlabel('Withheld query index',fontsize=10)
        from matplotlib.ticker import NullFormatter,MaxNLocator
        axes[0,0].xaxis.set_minor_formatter(NullFormatter())
        axes[1,1].xaxis.set_major_locator(MaxNLocator(integer=True,nbins=4))
        fig.subplots_adjust(hspace=.45,wspace=.32)
        metadata = {'source_product':str(path),'source_product_sha256':before,
                    'source_hashes':report['source_hashes'],'epoch':report['source_context']['epoch'],
                    'field_scope':report['field_scope'],'field_validation_status':report['field_validation_status'],
                    'science_acceptance':'NOT_ACCEPTED'}
        export_figure(fig,output_prefix,metadata=metadata,formats=['png','pdf'])
    if digest(path)!=before:raise ValueError('Host product changed during figure publication')
    verify_halo_structure(path)
    sidecar = Path(output_prefix).with_suffix('.figure.json')
    verify_figure_export(sidecar)
    return {'figure':str(sidecar),'sha256':digest(sidecar),'source_product_sha256':before}

class UnsupportedRockstarInput(ValueError):
    """Valid particle data outside the pinned finder's physical assumptions."""


def load_cfg(path):
    """Read the saved YAML configuration; physical parameters are never inferred from filenames."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def snapshot_numbers(run_dir):
    return list(snapshot_sets(run_dir))


def snapshot_sets(run_dir):
    from analysis import discover_snapshot_sets
    return discover_snapshot_sets(Path(run_dir) / "data")


def snapshot_file(run_dir, snap):
    found = snapshot_sets(run_dir).get(int(snap), [])
    return found[0] if found else None


def snapshot_blocks(run_dir, snap):
    return len(snapshot_sets(run_dir).get(int(snap), []))


def _snapshot_native_units(run_dir, snap):
    """Read positive native units from all shards; never assume absent units.

    These conversions apply to cosmological comoving GADGET data. Isolated
    non-comoving data uses the merger importer and physical kpc convention.
    Metadata inspection reads no particle coordinates or simulation payloads.
    """
    paths = snapshot_sets(run_dir).get(int(snap), [])
    if not paths:
        raise FileNotFoundError("No snapshot found for native unit inspection")
    reference = None
    names = ("UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s")
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "Parameters" not in handle:
                raise ValueError("Snapshot unit conversion requires explicit /Parameters: " + str(path))
            parameters = handle["Parameters"].attrs
            values = []
            for name in names:
                raw = np.asarray(parameters.get(name, []))
                if raw.size != 1:
                    raise ValueError("Snapshot needs a scalar unit declaration: " + name)
                value = float(raw.reshape(-1)[0])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Snapshot units must be finite and positive: " + name)
                values.append(value)
            comoving = np.asarray(parameters.get("ComovingIntegrationOn", []))
            if comoving.size != 1 or float(comoving.reshape(-1)[0]) != 1.:
                raise ValueError("Cosmological halo/dome unit conversion requires ComovingIntegrationOn=1")
            if reference is not None and not np.allclose(values, reference, rtol=1e-10, atol=0):
                raise ValueError("Snapshot shards disagree about native length/mass/velocity units")
            reference = tuple(values)
    return reference


def snapshot_units(run_dir, snap=0):
    """Convert recorded native length/mass to comoving Mpc/h and Msun/h.

    The h factors already belong to GADGET's comoving convention. All shards
    must declare the same units; no default kpc/h or 1e10 Msun/h is guessed.
    """
    length, mass, _ = _snapshot_native_units(run_dir, snap)
    return length/MPC_CM, mass/MSUN_G


def snapshot_dm_mass_msun_h(run_dir, snap=0):
    """Return the pinned Rockstar reader's constant DM mass, checking all shards."""
    paths = snapshot_sets(run_dir).get(int(snap), [])
    if not paths:
        return None
    _, mass_unit = snapshot_units(run_dir, snap)
    reference = None
    particle_count = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            header = handle["Header"].attrs
            table = np.asarray(header.get("MassTable", []), dtype=float)
            if table.shape != (6,) or np.any(~np.isfinite(table)) or np.any(table < 0):
                raise ValueError("Rockstar needs six finite non-negative Header/MassTable entries")
            dm_mass = float(table[1])
            if reference is not None and not math.isclose(dm_mass, reference, rel_tol=1e-10, abs_tol=0):
                raise ValueError("Snapshot shards disagree about constant DM mass")
            reference = dm_mass
            group = handle.get("PartType1")
            if group is not None and "Coordinates" in group:
                particle_count += len(group["Coordinates"])
    if not particle_count:
        return None
    if reference <= 0:
        raise UnsupportedRockstarInput("Rockstar requires positive constant Header/MassTable[1]; its pinned reader cannot process variable-mass DM. Use native GADGET FOF/SUBFIND")
    return reference*mass_unit


def snapshot_velocity_conversion(run_dir, snap):
    """Return native velocity units to km/s for the configured Rockstar reader.

    Rockstar's GADGET reader separately handles the snapshot's cosmological
    velocity convention; this factor only converts cm/s to km/s.
    """
    _, _, velocity = _snapshot_native_units(run_dir, snap)
    return velocity/1e5


@native_run_owner
def halo_background_table(run_dir, cfg, numbers, native_parameters=None, scale_factors=None):
    """Use the recorded H(a), or the native LCDM/radiation expansion law.

    Particle snapshots carry neither a scalar trajectory nor a general dark
    energy equation of state. An EDE/custom-table run needs its original table.
    For vanilla GADGET metadata, E^2=Omega_m/a^3+Omega_r/a^4+Omega_k/a^2+
    Omega_Lambda, with Omega_k=1-sum(Omega_i). These two-column tables describe
    only expansion; they deliberately contain no fabricated growth columns.
    """
    from evolution import file_sha256
    from core.foundation import G_CGS, MSUN_G, RHO_CRIT_MSUN_H_MPC_H3
    header = {}
    if native_parameters is None:
        path = snapshot_file(run_dir, numbers[0])
        with h5py.File(path, "r") as handle:
            parameters = dict(handle["Parameters"].attrs)
            header = dict(handle["Header"].attrs)
    else:
        parameters = native_parameters
    scales = list(scale_factors or [])
    for snap in numbers:
        with h5py.File(snapshot_file(run_dir, snap), "r") as handle:
            scales.append(float(handle["Header"].attrs["Time"]))
    if not scales or any(not math.isfinite(a) or not 0. < a <= 1. for a in scales):
        raise ValueError("Halo expansion requires finite snapshot scale factors in (0, 1]")
    table = cfg["cosmology"].get("gadget_background_table")
    recorded = parameters.get("CosmologyTableFile", "none")
    if isinstance(recorded, bytes):
        recorded = recorded.decode()
    if not table and str(recorded) not in ("none", "", "0"):
        table = str(recorded)
    if table:
        table = Path(table).expanduser().resolve()
        if not table.is_file():
            raise FileNotFoundError("Halo measurements need the original expansion table: " + str(table))
        from core.data import verify_numeric_table
        verify_numeric_table(table)
        values = np.loadtxt(table, ndmin=2)
        source = "recorded expansion table"
    else:
        if cfg.get("ede", {}).get("enabled") or cfg["cosmology"].get("model") == "ede":
            raise ValueError("EDE halo finding requires its original CLASS/AxiCLASS expansion table")
        om = float(parameters["Omega0"])
        ol = float(parameters["OmegaLambda"])
        radiation = float(parameters.get("OmegaRadiation", header.get("OmegaRadiation", 0.)))
        curvature = 1. - om - ol - radiation
        minimum = 1.e-4
        # The native spatial-distance initializer reaches z=300, and catalogue
        # histories also query slightly before the first saved epoch.
        minimum = min(minimum, 0.5*min(scales))
        a = np.geomspace(minimum, 1., 16385)
        e2 = om/a**3 + radiation/a**4 + curvature/a**2 + ol
        if not np.isfinite(e2).all() or np.any(e2 <= 0):
            raise ValueError("Snapshot metadata does not define a positive expansion history")
        values = np.column_stack((a, np.sqrt(e2)))
        table = Path(run_dir) / "input/halo_background.txt"
        from core.data import publish_numeric_table
        publish_numeric_table(table,values,{'a':{'units':'dimensionless','quantity':'Canonical scale factor'},
            'E':{'units':'dimensionless','quantity':'Native H(a)/H0 from saved matter/radiation/curvature/Lambda'}},
            algorithm=__file__,context={'canonical_time':'scale_factor','epoch_column':'a',
                'Omega_m':om,'Omega_Lambda':ol,'Omega_radiation':radiation,'Omega_curvature':curvature,
                'scope':'Native homogeneous expansion; no growth columns'},
            header='a E(a)=H(a)/H0; native snapshot matter/radiation/curvature/Lambda expansion; no growth columns')
        source = "native snapshot Omega_m/radiation/curvature/Lambda parameters"
    if (values.shape[1] < 2 or not np.isfinite(values[:, :2]).all()
            or np.any(values[:, :2] <= 0) or np.any(np.diff(values[:, 0]) <= 0)
            or not np.isclose(values[-1, 0], 1., rtol=0., atol=1.e-14)
            or not np.isclose(values[-1, 1], 1., rtol=1.e-6)):
        raise ValueError("Halo background table must provide increasing positive epochs through a=1, E=1")
    if values[0, 0] > min(scales):
        raise ValueError("Halo background table does not cover the earliest catalogue/snapshot epoch")
    record = {"path": str(table), "sha256": file_sha256(table), "source": source,
              "G_cgs": G_CGS, "adopted_solar_mass_g": MSUN_G,
              "rho_critical0_msun_h_per_mpc_h_cubed": RHO_CRIT_MSUN_H_MPC_H3,
              "interpolation": "log-linear H(a)/H0; exact power-law-segment time and distance integrals",
              "virial_overdensity": "Bryan-Norman LCDM fit evaluated using Omega_m(a); empirical convention outside LCDM, not an EDE spherical-collapse solution"}
    return table, record


def softening_kpc_h(cfg):
    value = cfg["simulation"].get("softening_kpc_h", "auto")
    if value != "auto":
        return float(value)
    n = cfg["simulation"].get("particles_per_dimension")
    if n is None:
        raise ValueError("Imported non-cubic particle sets need explicit simulation.softening_kpc_h for Rockstar")
    spacing = float(cfg["simulation"]["box_mpc_h"]) * 1000.0 / int(n)
    return spacing / 40.0


def rockstar_worker_count(cfg):
    """Use periodic Rockstar with a valid worker count inside the CPU budget."""
    return (cpu_thread_budget(cfg) // 8) * 8


@native_run_owner
def run_gadget_groups(run_dir, cfg, numbers=None, dry_run=False):
    """Build a finder binary and run restart flag 3 in an owned staging folder.

    GADGET rewrites the input snapshot in FOF order after finding groups. Keep
    that reordered copy separate for exact membership work; physical products
    continue reading the original snapshot with all its original fields.
    """
    from analysis import validate_gadget_snapshot_set, group_files
    from evolution import build_gadget, gadget_param_text, file_sha256

    run_dir = Path(run_dir).expanduser().resolve()
    if (run_dir / "data").is_symlink():
        raise ValueError("Halo finding needs an owned data directory; reimport the source with --folder")
    sets = snapshot_sets(run_dir)
    numbers = list(sets) if numbers is None else list(numbers)
    if not numbers:
        raise ValueError("No snapshots found for GADGET halo finding")
    records = {snap: validate_gadget_snapshot_set(sets[snap], snap) for snap in numbers}
    h = float(cfg["cosmology"]["H0"])/100.
    matter = (float(cfg["cosmology"]["omega_b"])+float(cfg["cosmology"]["omega_cdm"]))/h**2
    baryon = float(cfg["cosmology"]["omega_b"])/h**2
    for row in records.values():
        for name, requested in (("box_mpc_h", float(cfg["simulation"]["box_mpc_h"])),
                                ("hubble_param", h), ("omega_m", matter), ("omega_baryon", baryon)):
            if not math.isclose(row[name], requested, rel_tol=1.e-5, abs_tol=1.e-8):
                raise ValueError("Halo snapshot metadata disagrees with config.yaml on " + name)
    source_bytes = sum(path.stat().st_size for snap in numbers for path in sets[snap])
    # A native ordered copy, the retained membership copy, plus catalogue/I/O
    # margin. Original payloads are linked and need no initial full copy.
    required_bytes = 3 * source_bytes + 256 * 1024**2
    available_bytes = shutil.disk_usage(run_dir).free
    if not dry_run and available_bytes < required_bytes:
        raise OSError("Independent halo finding needs approximately %.3g GiB of free staging space; %.3g GiB available"
                      % (required_bytes / 1024**3, available_bytes / 1024**3))
    build_cfg = copy.deepcopy(cfg)
    if build_cfg["simulation"].get("particles_per_dimension") is None:
        if build_cfg["simulation"].get("softening_kpc_h", "auto") == "auto":
            raise ValueError("Non-cubic imported particle sets need explicit simulation.softening_kpc_h")
        # This is a compile-time PM geometry for flag 3, not an inferred IC load.
        count = max(row["particle_counts_by_type"][1] for row in records.values())
        build_cfg["simulation"]["particles_per_dimension"] = max(8, math.ceil(count ** (1/3)))
        if build_cfg["simulation"].get("pm_grid") == "auto":
            build_cfg["simulation"]["pm_grid"] = 32
    import tempfile
    (run_dir/'input').mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='halo_stage_',dir=run_dir/'input'))
    (stage / "data").mkdir(parents=True, exist_ok=True)
    (stage / "input").mkdir(exist_ok=True)
    # No Boltzmann output or primordial amplitude is needed to measure groups.
    # SO thresholds do need the actual expansion law. Retain a known EDE table,
    # and refuse to infer LCDM when snapshot metadata records a custom table.
    first = records[numbers[0]]
    with h5py.File(sets[numbers[0]][0], "r") as handle:
        native = dict(handle["Parameters"].attrs)
    recorded_table = native.get("CosmologyTableFile", "none")
    if isinstance(recorded_table, bytes):
        recorded_table = recorded_table.decode()
    table = build_cfg["cosmology"].get("gadget_background_table")
    if str(recorded_table) not in ("none", "", "0") and not table:
        table = str(recorded_table)
    if (table or build_cfg.get("ede", {}).get("enabled")) and (not table or not Path(table).expanduser().is_file()):
        raise ValueError("SO halo measurements need the original expansion table; set cosmology.gadget_background_table to its existing path")
    if table:
        build_cfg["cosmology"]["gadget_background_table"] = str(Path(table).expanduser().resolve())
    radiation = float(native.get("OmegaRadiation", max(0., 1. - first["omega_m"] - first["omega_lambda"])))
    build_cfg["cosmology"]["omega_radiation"] = radiation
    build_cfg["simulation"]["start_redshift"] = max(row["redshift"] for row in records.values())
    # Post-processing operates on synchronized saved epochs, which can differ
    # slightly from the requested redshifts. Do not carry an earlier requested
    # output into a parameter recipe whose start now comes from the snapshot.
    actual_schedule = sorted((records[snap]["redshift"] for snap in numbers), reverse=True)
    build_cfg["simulation"].update(snapshot_redshift_list=actual_schedule,
                                   snapshot_z_max=actual_schedule[0],
                                   snapshot_z_min=actual_schedule[-1],
                                   end_redshift=actual_schedule[-1])
    binary = build_gadget(build_cfg, stage, dry_run=dry_run)
    ranks = mpi_ranks(cfg)
    report = {"status": "DRY_RUN" if dry_run else "RUNNING", "finder": "GADGET-4 FOF/SUBFIND",
              "source_snapshots_are_preserved": True, "snapshots": [], "expansion_table": table,
              "estimated_staging_bytes": required_bytes, "mpi_ranks": ranks,
              "executable_sha256": None if dry_run else file_sha256(binary),
              "expansion_table_sha256": file_sha256(table) if table else None}
    from core.data import run_epoch_contract,coordinate_contract
    from core.foundation import product_provenance, verify_product_provenance
    report['science_acceptance']='NOT_ACCEPTED'
    report['staging_directory']=str(stage)
    report['data_contract']=product_provenance(run_dir,algorithm=__file__,
        selection='Native FOF/SUBFIND on exact explicitly selected saved particle snapshots',
        source_files=[path for snap in numbers for path in sets[snap]],
        coordinates=coordinate_contract(position_units='comoving kpc/h',
            origin='global simulation Cartesian box origin',periodic_wrapping='native periodic domain',
            velocity_convention='native particle velocities; catalogue GroupVel/SubhaloVel conventions differ'),
        helper_sources=[ROOT/'data.py'])
    report['data_contract']['coordinates']['native_length_unit_cm'] = float(native['UnitLength_in_cm'])
    report['data_contract']['coordinates']['native_to_comoving_kpc_h'] = float(native['UnitLength_in_cm']) / 3.085678e21
    from core.foundation import atomic_text
    atomic_json(stage/'input/halo_finding.json',report)
    for snap in numbers:
        shards = sets[snap]
        for previous_stage in group_files(stage, snap):
            previous_stage.unlink()
        # A new invocation may use a different shard layout. Remove only our
        # staged particle links/copies, including native bak- files, so an old
        # block cannot be mistaken for part of the new ordered snapshot.
        for staged_directory in (stage / "data", stage / ("data/snapdir_%03d" % snap)):
            if staged_directory.is_dir():
                for pattern in ("snapshot_%03d.hdf5" % snap, "snapshot_%03d.*.hdf5" % snap,
                                "bak-snapshot_%03d.hdf5" % snap, "bak-snapshot_%03d.*.hdf5" % snap):
                    for old in staged_directory.glob(pattern):
                        old.unlink()
        input_hashes = {str(path.resolve()): file_sha256(path) for path in shards} if not dry_run else {}
        directory = stage / "data" if len(shards) == 1 else stage / ("data/snapdir_%03d" % snap)
        directory.mkdir(exist_ok=True)
        for block, original in enumerate(shards):
            name = "snapshot_%03d.hdf5" % snap if len(shards) == 1 else "snapshot_%03d.%d.hdf5" % (snap, block)
            target = directory / name
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(original.resolve())
        atomic_text(stage / "input/output_times.txt", "%.17g\n" % records[snap]["scale_factor"])
        parameter_text = gadget_param_text(build_cfg, stage, shards[0], dry_run=dry_run,
                                           num_files_per_snapshot=len(shards))
        values = dict(line.split(None, 1) for line in parameter_text.splitlines())
        values["OmegaLambda"] = str(records[snap]["omega_lambda"])
        param = stage / ("input/groups_%03d.param" % snap)
        atomic_text(param, "".join(key + " " + str(value) + "\n" for key, value in values.items()))
        cmd = ["mpirun", "-np", str(ranks), str(binary), str(param), "3", str(snap)]
        if dry_run:
            print(" ".join(cmd))
            continue
        log = stage / "input" / ("groups_%03d.log" % snap)
        with open(log, "w", encoding="utf-8") as out:
            invocation = stage / 'input' / ('groups_%03d.invocation.json' % snap)
            run_native_command(cmd, cwd=stage / 'input', receipt_path=invocation,
                               inputs=[*shards, stage / 'input/output_times.txt', *([table] if table else [])],
                               stdout=out, stderr=subprocess.STDOUT, check=True)
        for path, before in input_hashes.items():
            if file_sha256(path) != before:
                raise RuntimeError("Halo finding modified an original snapshot: " + path)
        catalogues = group_files(stage, snap)
        if not catalogues:
            raise RuntimeError("GADGET wrote no group catalogue; see " + str(log))
        # The durable journal closes public discovery across all member renames.
        # Both native next bytes and prior catalogue/order bytes are immutable.
        previous=group_files(run_dir,snap)
        previous+=list((run_dir/'data').glob('grouped_snapshot_%03d.hdf5'%snap))
        previous+=list((run_dir/'data').glob('grouped_snapshot_%03d.*.hdf5'%snap))
        staged_outputs=[(path,path.relative_to(stage/'data')) for path in catalogues]
        for block in range(len(shards)):
            source=directory/('snapshot_%03d.hdf5'%snap if len(shards)==1 else 'snapshot_%03d.%d.hdf5'%(snap,block))
            if source.is_symlink() or not source.is_file():
                raise RuntimeError('GADGET did not write its ordered particle output')
            relative=Path('grouped_snapshot_%03d.hdf5'%snap if len(shards)==1 else 'grouped_snapshot_%03d.%d.hdf5'%(snap,block))
            staged_outputs.append((source,relative))
        ordered=[source for source,relative in staged_outputs if relative.name.startswith('grouped_snapshot_')]
        ordered_metadata=validate_gadget_snapshot_set(ordered,snap)
        if (ordered_metadata['particle_counts_by_type']!=records[snap]['particle_counts_by_type'] or
                not math.isclose(ordered_metadata['scale_factor'],records[snap]['scale_factor'],rel_tol=1e-12)):
            raise ValueError('Native group-ordered output changed the saved epoch or particle counts')
        from core.data import hdf5_scalar_attr
        totals={};summed={}
        for catalogue in catalogues:
            with h5py.File(catalogue,'r') as saved:
                header=saved['Header'].attrs
                if int(hdf5_scalar_attr(header['NumFiles'],'NumFiles'))!=len(catalogues):
                    raise ValueError('Native group catalogue shard set is incomplete')
                if not math.isclose(hdf5_scalar_attr(header['Time'],'Time'),records[snap]['scale_factor'],rel_tol=1e-12):
                    raise ValueError('Native catalogue epoch differs from its group-ordered particles')
                for species in ('Ngroups','Nsubhalos'):
                    total_key=species+'_Total';local_key=species+'_ThisFile'
                    if total_key not in header and local_key not in header:continue
                    total=int(hdf5_scalar_attr(header[total_key],total_key))
                    local=int(hdf5_scalar_attr(header[local_key],local_key))
                    if total<0 or local<0 or species in totals and totals[species]!=total:
                        raise ValueError('Native catalogue count metadata disagrees across shards')
                    totals[species]=total;summed[species]=summed.get(species,0)+local
        if any(summed[key]!=total for key,total in totals.items()):
            raise ValueError('Native catalogue shard counts do not sum to the declared generation')
        from core.data import publish_catalogue_set
        publication=publish_catalogue_set(run_dir,snap,staged_outputs,previous,
            epoch=run_epoch_contract(run_dir,snap,records[snap]['scale_factor']),provenance=report['data_contract'])
        outputs=[{'path':str(run_dir/'data'/relative),'sha256':file_sha256(run_dir/'data'/relative)}
                 for _,relative in staged_outputs]
        report["snapshots"].append({"snapshot": snap, "inputs_sha256": input_hashes, "outputs": outputs,
                                     'epoch_contract':run_epoch_contract(run_dir,snap,records[snap]['scale_factor']),
                                     "parameter_sha256": file_sha256(param), "log": str(log),
                                     "publication":publication,
                                     "native_invocation": {"path":str(invocation),"sha256":file_sha256(invocation)}})
        atomic_json(stage/'input/halo_finding.json',report)
    report["status"] = "DRY_RUN" if dry_run else "COMPLETE"
    verify_product_provenance(run_dir,report['data_contract'])
    atomic_json(stage/'input/halo_finding.json',report)
    atomic_json((run_dir / "input/halo_finding.json"),report)
    return report


@native_run_owner
def rockstar_config(run_dir, cfg):
    """Write finder units, cosmology, mass definition and file layout for the actual run."""
    run_dir = Path(run_dir).resolve()
    out = (run_dir / "data/rockstar").resolve()
    out.mkdir(parents=True, exist_ok=True)
    nums = snapshot_numbers(run_dir)
    if not nums:
        raise RuntimeError("No snapshots found for Rockstar")
    blocks = snapshot_blocks(run_dir, nums[0])
    if blocks < 1:
        raise RuntimeError("Could not determine GADGET snapshot block count")
    writers = rockstar_worker_count(cfg)
    if writers < 8:
        raise ValueError("Periodic Rockstar requires at least 8 CPU slots in this pinned version")
    # Validate real snapshot payloads before reporting a finder capability
    # limitation. Malformed particle data must still fail, not become a skip.
    sets = snapshot_sets(run_dir)
    from analysis import validate_gadget_snapshot_set
    records = {snapshot: validate_gadget_snapshot_set(sets[snapshot], snapshot) for snapshot in nums}
    from core.data import validate_signed_native_particle_ids
    try:
        validate_signed_native_particle_ids([path for paths in sets.values() for path in paths],species=1)
    except ValueError as exc:
        if 'signed-int64' not in str(exc):raise
        raise UnsupportedRockstarInput(str(exc)) from exc
    dm_mass = snapshot_dm_mass_msun_h(run_dir, nums[0])
    if dm_mass is None:
        raise RuntimeError("Could not read the DM particle mass from the GADGET HDF5 header")
    length_unit, mass_unit = snapshot_units(run_dir, nums[0])
    velocity_unit = snapshot_velocity_conversion(run_dir, nums[0])
    # The pinned parallel reader has one file pattern and conversion pair for
    # the whole timeline. Reject incompatible epochs rather than silently
    # mixing masses, units, or shard counts in temporal halo finding.
    for snapshot in nums:
        units = snapshot_units(run_dir, snapshot)
        if not all(math.isclose(left, right, rel_tol=1.e-10) for left, right in zip(units, (length_unit, mass_unit))):
            raise ValueError("Rockstar snapshots use inconsistent native units")
        if snapshot_blocks(run_dir, snapshot) != blocks:
            raise ValueError("Rockstar requires the same snapshot shard count at every epoch")
        if not math.isclose(snapshot_velocity_conversion(run_dir, snapshot), velocity_unit, rel_tol=1.e-10):
            raise ValueError("Rockstar snapshots use inconsistent native velocity units")
        mass = snapshot_dm_mass_msun_h(run_dir, snapshot)
        if mass is None or not math.isclose(mass, dm_mass, rel_tol=1.e-10):
            raise ValueError("Rockstar requires a constant DM particle mass across the timeline")
        with h5py.File(snapshot_file(run_dir, snapshot), "r") as handle:
            box = float(handle["Header"].attrs["BoxSize"])*length_unit
        if not math.isclose(box, float(cfg["simulation"]["box_mpc_h"]), rel_tol=1.e-6):
            raise ValueError("Rockstar snapshot box differs from config.yaml in Mpc/h")
    # Rockstar loops dense indices [0, NUM_SNAPS). A series starting at 7 or
    # containing gaps must not be assigned STARTING_SNAP=7 with NUM_SNAPS=2.
    # Stage zero-based file links and retain the original snapshot mapping for
    # measurements. Tree binaries and out_<index>.list keep native indexing.
    staging = run_dir / "input/rockstar_snapshots"
    staging.mkdir(parents=True, exist_ok=True)
    snapshot_map = []
    for index, snapshot in enumerate(nums):
        shards = sets[snapshot]
        record = records[snapshot]
        if snapshot_map and record["scale_factor"] <= snapshot_map[-1]["scale_factor"]:
            raise ValueError("Rockstar temporal snapshots must increase in scale factor with snapshot number")
        for block, original in enumerate(shards):
            with h5py.File(original, "r") as handle:
                count = int(handle["Header"].attrs["NumPart_ThisFile"][1])
                if count and "PartType1/Velocities" not in handle:
                    raise ValueError("Rockstar phase-space finding requires DM Velocities in " + str(original))
                if count:
                    ids = handle["PartType1/ParticleIDs"]
                    for offset in range(0, count, 1_000_000):
                        values = ids[offset:offset + 1_000_000]
                        if (values < 0).any() or (values > (1 << 63) - 1).any():
                            raise ValueError("Rockstar particle IDs must fit non-negative signed int64")
            name = "snapshot_%03d.hdf5" % index if blocks == 1 else "snapshot_%03d.%d.hdf5" % (index, block)
            target = staging / name
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                raise FileExistsError("Rockstar staging path contains an owned file: " + str(target))
            target.symlink_to(original.resolve())
        snapshot_map.append({"rockstar_index": index, "gadget_snapshot": snapshot,
                             "scale_factor": record["scale_factor"], "redshift": record["redshift"]})
    atomic_json(out/"snapshot_index.json",{"snapshots":snapshot_map})
    background, background_record = halo_background_table(run_dir, cfg, nums)
    atomic_json((out / "background_metadata.json"),background_record)
    strict = 1 if cfg["halos"].get("rockstar_strict_so_masses", True) else 0
    exact_membership = cfg['halos'].get('rockstar_exact_membership', False)
    if type(exact_membership) is not bool:
        raise ValueError('halos.rockstar_exact_membership must be a boolean')
    filename = "snapshot_<snap>.hdf5" if blocks == 1 else "snapshot_<snap>.<block>.hdf5"
    force_res = softening_kpc_h(cfg) / 1000.0
    box_size = float(cfg["simulation"]["box_mpc_h"])
    overlap = float(cfg["halos"].get("rockstar_overlap_length_mpc_h", 3.0))
    if not math.isfinite(overlap) or overlap <= 0:
        raise ValueError("Rockstar overlap length must be a positive finite Mpc/h value")
    # Rockstar requires L_box > 5*OVERLAP_LENGTH. Keep its periodic buffers
    # valid for small test boxes, while recording any reduction in the config.
    max_overlap = box_size / 6.0
    if overlap > max_overlap:
        print(
            "Rockstar overlap length %.6g Mpc/h exceeds the small-box limit; "
            "using %.6g Mpc/h. Check that this buffer still covers the largest "
            "halo you intend to analyze." % (overlap, max_overlap)
        )
        overlap = max_overlap
    text = '''PARALLEL_IO = 1
INBASE = "{inbase}"
OUTBASE = "{outbase}"
FILE_FORMAT = "AREPO"
BACKGROUND_TABLE = "{background}"
FILENAME = "{filename}"
NUM_BLOCKS = {blocks}
NUM_READERS = {readers}
NUM_WRITERS = {writers}
NUM_SNAPS = {num_snaps}
STARTING_SNAP = {starting_snap}
FORK_READERS_FROM_WRITERS = 1
FORK_PROCESSORS_PER_MACHINE = {writers}
PERIODIC = 1
OVERLAP_LENGTH = {overlap:.12g}
AREPO_LENGTH_CONVERSION = {length_unit:.12e}
AREPO_MASS_CONVERSION = {mass_unit:.12e}
AREPO_VELOCITY_CONVERSION = {velocity_unit:.12e}
PARTICLE_MASS = {particle_mass:.12e}
FORCE_RES = {force_res:.12e}
STRICT_SO_MASSES = {strict}
TEMPORAL_HALO_FINDING = 1
MIN_HALO_OUTPUT_SIZE = {min_halo}
MASS_DEFINITION = "vir"
MASS_DEFINITION2 = "200c"
MASS_DEFINITION3 = "200b"
MASS_DEFINITION4 = "500c"
GADGET_HALO_PARTICLE_TYPE = 1
RESCALE_PARTICLE_MASS = 0
FULL_PARTICLE_CHUNKS = {particle_chunks}
OUTPUT_EVERY_N_PARTICLES = 1
'''.format(
        inbase=staging,
        background=background,
        outbase=out,
        filename=filename,
        blocks=blocks,
        readers=min(blocks, writers),
        writers=writers,
        num_snaps=len(nums),
        starting_snap=0,
        particle_mass=dm_mass,
        length_unit=length_unit,
        mass_unit=mass_unit,
        velocity_unit=velocity_unit,
        force_res=force_res,
        strict=strict,
        overlap=overlap,
        min_halo=int(cfg["halos"]["min_particles"]),
        particle_chunks=writers if exact_membership else 0,
    )
    path = run_dir / "input/rockstar.cfg"
    atomic_text(path,text)
    return path


@staged_native_catalogues
@native_run_owner
def run_rockstar(run_dir, cfg, dry_run=False):
    """Build/run the bundled phase-space finder on complete supported particle snapshots."""
    run_dir = Path(run_dir).resolve()
    wall_seconds=cfg.get('halos',{}).get('rockstar_wall_seconds',3600.)
    if isinstance(wall_seconds,bool) or not isinstance(wall_seconds,(int,float)) or not math.isfinite(wall_seconds) or wall_seconds<=0:
        raise ValueError('Rockstar wall limit must be finite and positive')
    workers = rockstar_worker_count(cfg)
    if workers < 8:
        budget = cpu_thread_budget(cfg)
        reason = (
            "This pinned Rockstar version requires at least 8 worker CPU slots "
            "for periodic boundaries; the shared CPU budget is %d. FOF/SUBFIND can still run."
            % budget
        )
        status = {"status": "SKIPPED", "reason": reason, "cpu_thread_budget": budget}
        status_path = run_dir / "data/rockstar/rockstar_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(status_path,status)
        print("Rockstar skipped:", reason)
        return status
    candidates = [native_engine_directory("rockstar") / "rockstar", native_engine_directory("rockstar") / "rockstar-galaxies"]
    binary = next((p for p in candidates if p.exists()), None)
    try:
        conf = rockstar_config(run_dir, cfg)
    except UnsupportedRockstarInput as exc:
        status = {"status": "SKIPPED", "reason": str(exc), "capability": "constant-mass DM only"}
        status_path = run_dir / "data/rockstar/rockstar_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(status_path,status)
        print("Rockstar skipped:", str(exc))
        return status
    outbase = run_dir / "data/rockstar"
    auto = outbase / "auto-rockstar.cfg"
    if binary is None and not dry_run:
        raise RuntimeError("Rockstar executable not found")
    if dry_run:
        print(str(binary or candidates[0]), "-c", str(conf))
        print("then client:", str(binary or candidates[0]), "-c", str(auto))
        return {"status": "DRY_RUN", "writers": workers, "cpu_thread_budget": cpu_thread_budget(cfg)}
    if auto.exists():
        auto.unlink()
    server_log = run_dir / "input/rockstar_server.log"
    client_log = run_dir / "input/rockstar_client.log"
    receipt=run_dir/'input/rockstar_invocation.json'
    invocation=invocation_record([str(binary),'-c',str(conf)],run_dir/'input')
    atomic_json(receipt,invocation)
    try:
        with open(server_log, "w", encoding="utf-8") as so:
            server = subprocess.Popen([str(binary), "-c", str(conf)], cwd=run_dir / "input", stdout=so, stderr=subprocess.STDOUT,
                                      start_new_session=True,pass_fds=native_owned_fds())
            client=None;client_invocation=None
            try:
                invocation.update(status='SPAWNED',pid=server.pid)
                atomic_json(receipt,invocation)
                deadline = time.monotonic() + 180.0
                while not auto.exists() and time.monotonic() < deadline:
                    code = server.poll()
                    if code is not None:
                        raise RuntimeError("Rockstar server exited before creating auto-rockstar.cfg; see " + str(server_log))
                    time.sleep(0.25)
                if not auto.exists():
                    raise RuntimeError("Rockstar server did not create auto-rockstar.cfg within 180 s")
                client_receipt=run_dir/'input/rockstar_client_invocation.json'
                client_invocation=invocation_record([str(binary),'-c',str(auto)],run_dir/'input')
                atomic_json(client_receipt,client_invocation)
                with open(client_log, "w", encoding="utf-8") as co:
                    client = subprocess.Popen([str(binary), "-c", str(auto)], cwd=run_dir / "input", stdout=co,
                                              stderr=subprocess.STDOUT, start_new_session=True,pass_fds=native_owned_fds())
                    client_invocation.update(status='SPAWNED',pid=client.pid)
                    atomic_json(client_receipt,client_invocation)
                    client_deadline=time.monotonic()+wall_seconds
                    while client.poll() is None:
                        if time.monotonic()>=client_deadline:
                            raise TimeoutError('Rockstar exceeded declared client wall limit')
                        # A failed forked reader may leave both parents waiting
                        # forever for network data. Surface its native error.
                        for native_log in (client_log, server_log):
                            with native_log.open("rb") as stream:
                                stream.seek(max(0, native_log.stat().st_size - 65536))
                                if b"[Error]" in stream.read():
                                    raise RuntimeError("Rockstar reported a native error; see " + str(native_log))
                        if server.poll() is not None and server.returncode:
                            raise RuntimeError("Rockstar server failed; see " + str(server_log))
                        time.sleep(.5)
                    if client.returncode:
                        raise RuntimeError("Rockstar client failed; see " + str(client_log))
                    code = server.wait(timeout=30)
                if code != 0:
                    raise RuntimeError("Rockstar server failed; see " + str(server_log))
            finally:
                stop_owned_process_groups([client,server])
        verify_invocation_inputs(client_invocation)
        client_invocation.update(status='RETURNED',returncode=client.returncode)
        atomic_json(client_receipt,client_invocation)
        verify_invocation_inputs(invocation)
        invocation.update(status='RETURNED',returncode=code,client_returncode=client.returncode)
        atomic_json(receipt,invocation)
    except BaseException as exc:
        invocation.update(status='FAILED',failure={'type':type(exc).__name__,'message':str(exc)})
        atomic_json(receipt,invocation)
        raise
    status = {"status": "COMPLETE", "writers": workers, "cpu_thread_budget": cpu_thread_budget(cfg)}
    atomic_json((run_dir / "data/rockstar/rockstar_status.json"),status)
    return status


@staged_native_catalogues
@native_run_owner
def run_consistent_trees(run_dir, dry_run=False):
    """Construct temporal associations from Rockstar IDs/descendants using the run-matched expansion history."""
    run_dir = Path(run_dir).resolve()
    cfg = load_cfg(run_dir / "config.yaml")
    tree = native_engine_directory("consistent-trees")
    rockstar = native_engine_directory("rockstar")
    gen = rockstar / "scripts/gen_merger_cfg.pl"
    do_tree = tree / "do_merger_tree.pl"
    to_catalog = tree / "halo_trees_to_catalog.pl"
    rock_cfg = run_dir / "input/rockstar.cfg"
    for required in [gen, do_tree, to_catalog]:
        if not required.exists() and not dry_run:
            raise RuntimeError("Consistent Trees helper is missing: " + str(required))
    if dry_run:
        print("perl", gen, rock_cfg)
        print("perl", do_tree, "<generated merger_tree.cfg>")
        print("perl", to_catalog, "<generated merger_tree.cfg>")
        return {"status": "DRY_RUN"}
    if not (tree / "gravitational_consistency").is_file():
        from core.resources import build_jobs
        command = ["make", "-j", str(build_jobs(cfg))]
        if os.uname().sysname == "Darwin":
            compilers = [path for path in Path("/opt/homebrew/bin").glob("gcc-*")
                         if path.name.removeprefix("gcc-").isdigit()]
            if not compilers:
                raise RuntimeError("Consistent Trees needs GNU GCC with OpenMP on macOS")
            compiler = max(compilers, key=lambda path: int(path.name.split("-")[-1]))
            command.append("CC=" + str(compiler))
        with native_run_lock(tree,operation="consistent-trees shared build"):
            build_sources = sorted({path for pattern in ('*.c', '*.h', 'Makefile', '*.pl')
                                    for path in tree.glob(pattern) if path.is_file()})
            run_native_command(command, cwd=tree,
                receipt_path=run_dir / 'input/trees_build_invocation.json',
                inputs=build_sources, check=True)
    log = run_dir / "input/consistent_trees.log"
    with open(log, "w", encoding="utf-8") as out:
        run_native_command(['perl',str(gen),str(rock_cfg)],cwd=run_dir/'input',
            receipt_path=run_dir/'input/trees_generate_invocation.json',stdout=out,stderr=subprocess.STDOUT)
        cfg_candidates = [
            run_dir / "input/merger_tree.cfg",
            run_dir / "data/rockstar/merger_tree.cfg",
            run_dir / "data/rockstar/outputs/merger_tree.cfg",
        ]
        cfg_candidates += list((run_dir / "input").glob("*merger*.cfg"))
        cfg_candidates += list((run_dir / "data/rockstar").glob("*merger*.cfg"))
        merger_cfg = next((p for p in cfg_candidates if p.exists()), None)
        if merger_cfg is None:
            raise RuntimeError("Rockstar's gen_merger_cfg.pl produced no merger-tree configuration; see " + str(log))
        scale_file = merger_cfg.parent / "scales.txt"
        usable_scales = []
        if scale_file.is_file():
            usable_scales = [
                line for line in scale_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if len(usable_scales) < 2:
            status = {
                "status": "SKIPPED",
                "reason": "Rockstar produced fewer than two time outputs with enough halos for tree construction.",
                "usable_snapshots": len(usable_scales),
                "minimum_rockstar_halo_count": int(cfg["halos"]["min_particles"]),
            }
            status_path = run_dir / "data/rockstar/trees_status.json"
            atomic_json(status_path,status)
            print("Consistent Trees skipped: fewer than two Rockstar snapshots contain enough halos.")
            print("Tree status:", status_path)
            return status
        numbers = snapshot_numbers(run_dir)
        merger_text = merger_cfg.read_text(encoding="utf-8")
        native_parameters = None
        if not numbers:
            # Tree-only workflows may have catalogues and scales, but no
            # particle snapshots. The upstream generator recovered these
            # cosmological parameters directly from the first catalogue.
            def native_value(name):
                match = re.search(r'^' + name + r'\s*=\s*([\d.eE+-]+)', merger_text, re.MULTILINE)
                if not match:
                    raise ValueError("Tree-only catalogue metadata is missing " + name)
                return float(match.group(1))
            background_match = re.search(r'^BACKGROUND_TABLE\s*=\s*"([^"\n]+)"', rock_cfg.read_text(), re.MULTILINE)
            native_parameters = {"Omega0": native_value("Om"), "OmegaLambda": native_value("Ol"),
                                 "CosmologyTableFile": background_match.group(1) if background_match else "none"}
        scales = [float(line.split()[1]) for line in usable_scales]
        background, background_record = halo_background_table(run_dir, cfg, numbers, native_parameters, scales)
        # The upstream Perl generator discards unrecognised Rockstar options.
        # Carry the verified expansion table into native trees and Perl hlists.
        merger_text = re.sub(r'^BACKGROUND_TABLE\s*=.*\n?', '', merger_text, flags=re.MULTILINE)
        atomic_text(merger_cfg,merger_text + '\nBACKGROUND_TABLE = "' + str(background) + '"\n')
        atomic_json((merger_cfg.parent / "background_metadata.json"),background_record)
        run_native_command(['perl',str(do_tree),str(merger_cfg)],cwd=tree,
            receipt_path=run_dir/'input/trees_run_invocation.json',
            inputs=[p for p in tree.iterdir() if p.is_file() and os.access(p,os.X_OK)],stdout=out,stderr=subprocess.STDOUT)
        run_native_command(['perl',str(to_catalog),str(merger_cfg)],cwd=tree,
            receipt_path=run_dir/'input/trees_catalogue_invocation.json',
            inputs=[p for p in tree.iterdir() if p.is_file() and os.access(p,os.X_OK)],stdout=out,stderr=subprocess.STDOUT)
    return {"status": "COMPLETE"}


@staged_native_catalogues
@native_run_owner
def prepare_rockstar_host_catalogues(run_dir, cfg, parents_binary=None):
    """Create host/subhalo-labelled ASCII copies without changing tree inputs.

    Rockstar's normal catalogue does not necessarily include parent IDs. The
    upstream ``find_parents`` helper writes labelled catalogue rows to stdout;
    keeping those rows in a separate directory preserves the original files
    consumed by Consistent Trees.
    """
    run_dir = Path(run_dir).resolve()
    rockstar_dir = run_dir / "data/rockstar"
    catalogues = sorted(rockstar_dir.glob("out_*.list"), key=lambda p: _catalogue_snapshot_number(p))
    status_path = rockstar_dir / "host_catalogues_status.json"
    if not catalogues:
        status = {"status": "SKIPPED", "reason": "Rockstar produced no out_<snapshot>.list catalogues."}
        atomic_json(status_path,status)
        return status

    binary = Path(parents_binary) if parents_binary else native_engine_directory("rockstar") / "util/find_parents"
    if any(not _rockstar_catalogue_has_parent_id(path) for path in catalogues) and not binary.is_file():
        status = {
            "status": "SKIPPED",
            "reason": "Rockstar catalogues lack parent IDs and its find_parents utility is unavailable.",
            "catalogue_count": len(catalogues),
            "utility": str(binary),
        }
        atomic_json(status_path,status)
        return status

    host_dir = rockstar_dir / "host_catalogues"
    host_dir.mkdir(parents=True, exist_ok=True)
    box_size = float(cfg["simulation"]["box_mpc_h"])
    output_records = []
    for catalogue in catalogues:
        target = host_dir / catalogue.name
        if _rockstar_catalogue_has_parent_id(catalogue):
            atomic_copy(catalogue, target)
            output_records.append({"source": str(catalogue), "output": str(target), "parent_ids": "preserved"})
            continue

        header_line = _rockstar_catalogue_header(catalogue)
        if header_line is None:
            raise RuntimeError("Rockstar catalogue has no recognizable ASCII header: " + str(catalogue))
        header = header_line.lstrip("#").split()
        if not any(name.lower() in {"mvir", "m200c", "m200b", "mfof"} for name in header):
            raise RuntimeError("Rockstar catalogue header has no recognized mass field: " + str(catalogue))

        temp_rows = target.with_name(target.name + ".rows.tmp")
        log = target.with_name(target.name + ".parents.log")
        with open(temp_rows, "w", encoding="utf-8") as rows, open(log, "w", encoding="utf-8") as err:
            result = run_native_command(
                [str(binary), str(catalogue), "%.17g" % box_size],
                receipt_path=target.with_name(target.name+'.invocation.json'),
                cwd=catalogue.parent,
                stdout=rows,
                stderr=err,
                check=False,
                pass_fds=native_owned_fds(),
            )
        if result.returncode != 0:
            temp_rows.unlink(missing_ok=True)
            raise RuntimeError("Rockstar find_parents failed for %s; see %s" % (catalogue, log))

        source_row_count = 0
        with open(catalogue, "r", encoding="utf-8", errors="replace") as source_rows:
            for line in source_rows:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    source_row_count += 1
        row_count = 0
        output_header = None
        expected_columns = len(header) + 1
        with open(temp_rows, "r", encoding="utf-8", errors="replace") as rows:
            for line_number, line in enumerate(rows, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    candidate = stripped.lstrip("#").split()
                    lowered = [field.lower() for field in candidate]
                    if "id" in lowered and set(lowered).intersection({"mvir", "m200c", "m200b", "mfof"}):
                        if output_header is not None:
                            raise RuntimeError("find_parents emitted multiple ASCII headers in " + str(temp_rows))
                        expected_header = [field.lower() for field in header + ["PID"]]
                        if lowered != expected_header:
                            raise RuntimeError("find_parents changed the catalogue column order in " + str(temp_rows))
                        output_header = stripped
                    continue
                if len(stripped.split()) != expected_columns:
                    raise RuntimeError(
                        "find_parents emitted %d columns, expected %d at %s:%d"
                        % (len(stripped.split()), expected_columns, temp_rows, line_number)
                    )
                row_count += 1
        if row_count == 0 and source_row_count != 0:
            temp_rows.unlink(missing_ok=True)
            raise RuntimeError("find_parents produced an empty catalogue for " + str(catalogue))
        if row_count != source_row_count:
            temp_rows.unlink(missing_ok=True)
            raise RuntimeError(
                "find_parents preserved %d of %d halo rows for %s"
                % (row_count, source_row_count, catalogue)
            )

        staged = target.with_name(target.name + ".tmp")
        with open(staged, "w", encoding="utf-8") as output, open(temp_rows, "r", encoding="utf-8") as rows:
            if output_header is None:
                output.write("#" + " ".join(header + ["PID"]) + "\n")
            shutil.copyfileobj(rows, output)
        temp_rows.unlink()
        staged.replace(target)
        output_records.append({
            "source": str(catalogue), "output": str(target), "parent_ids": "computed",
            "row_count": row_count, "utility_log": str(log),
        })

    status = {
        "status": "COMPLETE",
        "method": "pinned Rockstar util/find_parents; host halos have PID=-1",
        "box_size_mpc_h": box_size,
        "catalogues": output_records,
    }
    atomic_json(status_path,status)
    return status


def _catalogue_snapshot_number(path):
    match = re.fullmatch(r"out_(\d+)\.list", Path(path).name)
    return int(match.group(1)) if match else -1


def _rockstar_catalogue_header(path):
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("#"):
                header = stripped.lstrip("#").split()
                lowered = {field.lower() for field in header}
                if "id" in lowered and lowered.intersection({"mvir", "m200c", "m200b", "mfof"}):
                    return stripped
    return None


def _rockstar_catalogue_has_parent_id(path):
    header = _rockstar_catalogue_header(path)
    return bool(header and "pid" in {field.lower() for field in header.lstrip("#").split()})


@native_run_owner
def run_all(run_dir, cfg, dry_run=False):
    """Run the enabled catalogue stages and record each stage outcome in the run manifest."""
    statuses = {}
    if cfg["halos"].get("fof") or cfg["halos"].get("subfind"):
        statuses["fof_subfind"] = run_gadget_groups(run_dir, cfg, dry_run=dry_run)
    rockstar_status = None
    if cfg["halos"].get("rockstar"):
        rockstar_status = run_rockstar(run_dir, cfg, dry_run)
        statuses["rockstar"] = rockstar_status
        if rockstar_status and rockstar_status.get("status") == "COMPLETE":
            host_catalogue_status = prepare_rockstar_host_catalogues(run_dir, cfg)
            statuses["rockstar_host_catalogues"] = host_catalogue_status
    if cfg["halos"].get("consistent_trees"):
        if rockstar_status and rockstar_status.get("status") == "SKIPPED":
            tree_status = {
                "status": "SKIPPED",
                "reason": "Rockstar was skipped, so this run has no new Rockstar catalogs for tree construction.",
            }
            status_path = Path(run_dir) / "data/rockstar/trees_status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(status_path,tree_status)
            statuses["consistent_trees"] = tree_status
            print("Consistent Trees skipped because Rockstar did not run.")
        else:
            statuses["consistent_trees"] = run_consistent_trees(run_dir, dry_run)
    return statuses


def halos_main():
    """Parse command arguments and dispatch this module's documented workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--folder", help="stage compatible external GADGET-4 snapshots into an independent run")
    p.add_argument("--config", help="optional YAML checked against imported snapshot metadata")
    p.add_argument("--output", help="new run folder for --folder")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", choices=["fof", "subfind", "rockstar", "consistent_trees"],
                   help="run one finder/tree stage using existing snapshots or catalogues")
    a = p.parse_args()
    if bool(a.run_dir) == bool(a.folder):
        p.error("pass either run_dir or --folder")
    if (a.config or a.output) and not a.folder:
        p.error("--config and --output apply to --folder")
    if a.folder:
        if a.only not in ("rockstar", "fof", "subfind"):
            p.error("external folder intake requires --only fof, subfind or rockstar")
        from analysis import prepare_hdf5_folder
        prepared = prepare_hdf5_folder(a.folder, a.config, a.output)
        a.run_dir = prepared["run_directory"]
    cfg = load_cfg(Path(a.run_dir) / "config.yaml")
    if a.only:
        for name in ("fof", "subfind", "rockstar", "consistent_trees"):
            cfg["halos"][name] = name == a.only
        # SUBFIND operates on FOF groups; compile/run that required parent stage.
        if a.only == "subfind":
            cfg["halos"]["fof"] = True
    run_all(a.run_dir, cfg, a.dry_run)


if __name__ == "__main__":
    halos_main()


def run_requested_native_finders(run_dir, cfg, *, requested, snapshots, velociraptor=None, dry_run=False):
    """Explicit public dispatch to existing native engines; no success-by-presence."""
    if not isinstance(requested,list) or not requested or len(set(requested))!=len(requested) or set(requested)-{'fof','subfind','rockstar','consistent_trees','velociraptor'}:
        raise ValueError('Choose supported native finder/tree stages explicitly')
    if 'consistent_trees' in requested and 'rockstar' not in requested:
        raise ValueError('Native Consistent Trees requires a requested Rockstar source campaign')
    statuses={}
    if 'velociraptor' in requested:
        if not isinstance(velociraptor,dict):raise ValueError('Declare native VELOCIraptor configuration and destination root')
        from halos import run_velociraptor
        options=dict(velociraptor);destination=Path(options.pop('destination'))
        statuses['velociraptor']={str(number):{'status':'DRY_RUN'} if dry_run else run_velociraptor(run_dir,destination/f'snapshot_{number:03d}',snapshot=number,**options) for number in snapshots}
    if set(requested)&{'fof','subfind'}:
        import copy
        local=copy.deepcopy(cfg);local['halos']['fof']=True;local['halos']['subfind']='subfind' in requested
        statuses['fof_subfind']=run_gadget_groups(run_dir,local,numbers=snapshots,dry_run=dry_run)
    if 'rockstar' in requested:
        if set(snapshots)!=set(snapshot_numbers(run_dir)):raise ValueError('Rockstar consumes the declared full saved epoch series; subset requires an owned imported run')
        statuses['rockstar']=run_rockstar(run_dir,cfg,dry_run=dry_run)
        if not dry_run and statuses['rockstar'].get('status')=='COMPLETE':statuses['rockstar_hosts']=prepare_rockstar_host_catalogues(run_dir,cfg)
    if 'consistent_trees' in requested:
        if not dry_run and statuses['rockstar'].get('status')!='COMPLETE':raise ValueError('No completed Rockstar campaign for native trees')
        statuses['consistent_trees']=run_consistent_trees(run_dir,dry_run=dry_run)
    return statuses

# ========================================================================
# galaxies: Resolved galaxy catalogues derived from GADGET-4 SUBFIND outputs.

# Workflow and units
# Reads SUBFIND bound-particle spans and writes resolved stellar system catalogues.
# Masses are Msun/h, formation times are scale factors, ages are Gyr, and stellar half-mass radii carry the documented comoving/physical convention.
# A stellar system requires actual star particles. Colours are not stellar population synthesis, and a nearby halo centre cannot substitute for a galaxy detection.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

from core.foundation import file_sha256
import uuid


SCHEMA = "darkmattersim-galaxy-catalogue-v1"
PARTICLE_TYPES = {"gas": 0, "dark_matter": 1, "stars": 4}
from core.foundation import KPC_CM
from core.data import convert_pinned_native,native_unit_factor,exact_uint64_ids


def _group_files(run_dir: Path, snapshot: int) -> list[Path]:
    from analysis import group_files

    return group_files(run_dir, int(snapshot))


def _snapshot_files(run_dir: Path, snapshot: int) -> list[Path]:
    from analysis import group_ordered_snapshot_files
    return group_ordered_snapshot_files(run_dir, snapshot)


def _check_file_series(paths: list[Path], header_name: str, total_name: str) -> None:
    """Compatibility adapter to the shared exact native shard count contract."""
    from core.data import validate_native_shard_counts
    if header_name!='Header':raise ValueError('Unsupported native series header')
    validate_native_shard_counts(paths,total_name)


def _read_catalogue_series(paths: list[Path], group: str, field: str, *, required=True) -> np.ndarray | None:
    chunks = []
    found_group = False
    found_field = False
    missing_populated = False
    trailing_shape = None
    for path in paths:
        with h5py.File(path, "r") as source:
            count_key = "Nsubhalos_ThisFile" if group == "Subhalo" else "Ngroups_ThisFile"
            expected_rows = source["Header"].attrs.get(count_key)
            if group not in source:
                missing_populated |= expected_rows is not None and int(expected_rows) > 0
                continue
            found_group = True
            if field not in source[group]:
                # Empty shards may omit datasets. A populated shard may not:
                # concatenating the others would misidentify catalogue rows.
                missing_populated |= expected_rows is None or int(expected_rows) > 0
                continue
            found_field = True
            array = np.asarray(source[group][field][:])
            if array.ndim == 0:
                raise ValueError("%s/%s must be an array" % (group, field))
            if expected_rows is not None and len(array) != int(expected_rows):
                raise ValueError("%s/%s rows disagree with the shard header" % (group, field))
            shape = array.shape[1:]
            if trailing_shape is None:
                trailing_shape = shape
            elif shape != trailing_shape:
                raise ValueError("%s/%s shard shapes disagree" % (group, field))
            chunks.append(array)
    if missing_populated:
        if required:
            raise ValueError("A populated catalogue shard is missing %s/%s" % (group, field))
        return None
    if not found_group or not found_field:
        if required and group == "Subhalo":
            # GADGET omits datasets for an explicitly empty subhalo catalogue.
            totals = []
            for path in paths:
                with h5py.File(path, "r") as source:
                    totals.append(source["Header"].attrs.get("Nsubhalos_Total"))
            if totals and all(value is not None and int(value) == 0 for value in totals):
                vector_fields = {"SubhaloPos", "SubhaloVel"}
                type_fields = {"SubhaloLenType", "SubhaloMassType", "SubhaloOffsetType", "SubhaloHalfmassRadType"}
                integer_fields = {"SubhaloGroupNr", "SubhaloRankInGr", "SubhaloLenType", "SubhaloOffsetType"}
                shape = (0, 3) if field in vector_fields else (0, 6) if field in type_fields else (0,)
                return np.empty(shape, dtype=np.uint64 if field in integer_fields else np.float64)
        if required:
            raise ValueError("SUBFIND catalogue is missing %s/%s" % (group, field))
        return None
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,) + (trailing_shape or ()))


def _snapshot_header_and_stars(paths: list[Path], snapshot: int):
    """Validate stellar metadata/masses in chunks without retaining all stars."""
    from core.data import snapshot_id_shards
    _, declared_star_total = snapshot_id_shards(paths, 4)
    headers = []
    star_count = 0
    for path in paths:
        with h5py.File(path, "r") as source:
            if "Header" not in source:
                raise ValueError("snapshot is missing /Header: %s" % path)
            header = source["Header"].attrs
            if "Time" not in header or "BoxSize" not in header:
                raise ValueError("snapshot header must contain Time and BoxSize")
            if not math.isfinite(float(header["Time"])) or float(header["Time"]) <= 0:
                raise ValueError("Every snapshot shard needs a finite positive scale factor")
            if not math.isfinite(float(header["BoxSize"])) or float(header["BoxSize"]) <= 0:
                raise ValueError("Every snapshot shard needs a finite positive box size")
            headers.append({
                "time": float(header["Time"]),
                "redshift": float(header.get("Redshift", 1.0 / float(header["Time"]) - 1.0)),
                "box_size": float(header["BoxSize"]),
                "hubble_param": float(header.get("HubbleParam", np.nan)),
                "num_files": int(header.get("NumFilesPerSnapshot", len(paths))),
            })
            if "PartType4" not in source:
                continue
            stars = source["PartType4"]
            if "ParticleIDs" not in stars:
                raise ValueError("PartType4/ParticleIDs is required to validate SUBFIND offsets")
            this_count = len(stars["ParticleIDs"])
            star_count += this_count
            if "Masses" in stars:
                masses = stars["Masses"]
                if masses.shape != (this_count,):
                    raise ValueError("PartType4/Masses must align with particle IDs")
                for start in range(0, this_count, 1_000_000):
                    mass = np.asarray(masses[start:start + 1_000_000], dtype=np.float64)
                    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
                        raise ValueError("PartType4/Masses must be finite and positive")
            else:
                mass_table = np.asarray(header.get("MassTable", np.zeros(6)), dtype=np.float64)
                if mass_table.shape != (6,) or not np.isfinite(mass_table[4]) or mass_table[4] <= 0:
                    raise ValueError("star particle masses are absent and MassTable[4] is not positive")
            if "StellarFormationTime" in stars:
                if stars["StellarFormationTime"].shape != (this_count,):
                    raise ValueError("PartType4/StellarFormationTime does not align with particle IDs")

    if not headers:
        raise ValueError("no numbered particle snapshot files found for snapshot %03d" % snapshot)
    if len({item["num_files"] for item in headers}) != 1 or headers[0]["num_files"] != len(paths):
        raise ValueError("snapshot shards are incomplete or disagree about NumFilesPerSnapshot")
    for key in ("time", "redshift", "box_size", "hubble_param"):
        vals = np.asarray([item[key] for item in headers], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) and not np.allclose(finite, finite[0], rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("snapshot shards disagree about %s" % key)
    header = headers[0]
    if declared_star_total != star_count:
        raise ValueError("Snapshot PartType4 rows disagree with the exact shard totals")
    header["star_count"] = star_count
    return header


def _bound_particle_slice(paths: list[Path], particle_type: int, offset: int,
                          count: int, fields: tuple[str, ...], *, records=None) -> dict[str, np.ndarray | None]:
    """Read one SUBFIND global-offset range across group-ordered shards.

    Offsets are defined over the concatenated particles of one type. Reading
    only the requested bound subhalo avoids a full gas-coordinate allocation
    for large hydro runs.
    """
    if int(offset) != offset or int(count) != count or offset < 0 or count < 0:
        raise ValueError("Bound particle spans need non-negative integer offsets and counts")
    remaining_start, remaining_end = int(offset), int(offset + count)
    from core.data import snapshot_id_shards
    if records is None:
        records, _ = snapshot_id_shards(paths, particle_type)
    total = records[-1]["stop"] if records else 0
    if remaining_end > total:
        raise ValueError("SUBFIND particle slice exceeds group-ordered snapshot")
    parts: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    absent = set()
    for record in records:
        if record["stop"] <= remaining_start or record["start"] >= remaining_end:
            continue
        with h5py.File(record["path"], "r") as source:
            group = source.get("PartType%d" % particle_type)
            size = record["count"]
            cursor = record["start"]
            lo = max(remaining_start, cursor)
            hi = min(remaining_end, cursor + size)
            if hi > lo:
                local = slice(lo - cursor, hi - cursor)
                for field in fields:
                    if field in group:
                        if group[field].ndim == 0 or len(group[field]) != size:
                            raise ValueError("Bound particle field does not align with Coordinates: " + field)
                        parts[field].append(np.asarray(group[field][local]))
                    else:
                        mass_table = source["Header"].attrs.get("MassTable", np.zeros(6))
                        if field == "Masses" and np.isfinite(mass_table[particle_type]) and mass_table[particle_type] > 0:
                            parts[field].append(np.full(hi - lo, mass_table[particle_type], dtype=np.float64))
                        elif field == "StellarFormationTime":
                            parts[field].append(np.full(hi - lo, np.nan, dtype=np.float64))
                        else:
                            absent.add(field)
    return {field: None if field in absent or not chunks else np.concatenate(chunks)
            for field, chunks in parts.items()}


def _weighted_median_scale_factor(scale_factor: np.ndarray, mass: np.ndarray, snapshot_a: float) -> float:
    """Find the mass-weighted median stellar formation scale factor, without treating it as an age."""
    valid = (
        np.isfinite(scale_factor) & (scale_factor > 0.0)
        & (scale_factor <= snapshot_a * (1.0 + 1.0e-8))
        & np.isfinite(mass) & (mass > 0.0)
    )
    if not np.any(valid):
        return math.nan
    a = scale_factor[valid]
    weights = mass[valid]
    order = np.argsort(a, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    return float(a[order[min(int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")), len(a) - 1)]])


def _unit_conversion(group_paths: list[Path]) -> tuple[float, float, dict]:
    """Check native cosmological catalogue units and return length/SFR factors.

    GADGET's sfr_eos.cc converts SphP.Sfr to Msun/year at calculation time;
    SUBFIND sums those values without returning them to code mass/time units.
    SubhaloVel is already peculiar velocity in native velocity units (no
    snapshot sqrt(a) factor). The pinned pipeline uses 1 km/s per unit.
    """
    reference = None
    for path in group_paths:
        with h5py.File(path, "r") as source:
            if "Parameters" not in source:
                raise ValueError("Every catalogue/particle shard needs GADGET /Parameters unit metadata")
            parameters = source["Parameters"].attrs
            values = tuple(float(parameters.get(name, math.nan)) for name in (
                "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"))
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("GADGET must declare positive length, mass and velocity units")
            if int(parameters.get("ComovingIntegrationOn", 0)) != 1:
                raise ValueError("This galaxy catalogue requires cosmological comoving snapshots")
            if reference is not None and not np.allclose(values, reference, rtol=1e-10, atol=0):
                raise ValueError("Catalogue/particle shards disagree about native units")
            reference = values
    if reference is None or not np.allclose(reference, (KPC_CM, 1e10*MSUN_G, 1e5), rtol=1e-8, atol=0):
        raise ValueError("Galaxy catalogues require the pinned kpc/h, 1e10 Msun/h and km/s units")
    return 1e-3, 1.0, {
        "length": "comoving Mpc/h (input: comoving kpc/h)",
        "mass": "Msun/h (input: 1e10 Msun/h)",
        "velocity": "peculiar km/s (SUBFIND already removes the cosmological velocity factor)",
        "sfr": "Msun/year (GADGET stores physical rates before SUBFIND sums them)",
        "native_sfr_to_msun_per_year": 1.0,
    }


from core.native import directory_owner


@native_run_owner
@directory_owner('output_path',default_relative='results')
def build_galaxy_catalogue(
    run_dir: str | Path,
    snapshot: int,
    *,
    minimum_star_particles: int = 100,
    minimum_dm_particles: int = 300,
    output_path: str | Path | None = None,
) -> dict:
    """Write a SUBFIND galaxy catalogue for one saved GADGET snapshot.

    Every luminous subhalo is retained.  ``Resolved`` records the explicit
    stellar and dark-matter particle-count selection; it never drops the raw
    subhalo row.  ``FormationRedshift`` is the redshift at which half of the
    present bound stellar mass had formed (mass-weighted median formation
    scale factor).
    """
    run_dir = Path(run_dir).expanduser().resolve()
    snapshot = int(snapshot)
    for name, value in (("minimum_star_particles", minimum_star_particles), ("minimum_dm_particles", minimum_dm_particles)):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
            raise ValueError("%s must be a non-negative integer" % name)
    group_paths = _group_files(run_dir, snapshot)
    snapshot_paths = _snapshot_files(run_dir, snapshot)
    if not group_paths or not snapshot_paths:
        raise FileNotFoundError("snapshot %03d needs both GADGET particle and FOF/SUBFIND files" % snapshot)
    source_records=[{'path':str(path.resolve()),'sha256':file_sha256(path)} for path in group_paths+snapshot_paths]
    _check_file_series(group_paths, "Header", "Nsubhalos_Total")
    _check_file_series(snapshot_paths, "Header", "NumPart_Total")
    snap = _snapshot_header_and_stars(snapshot_paths, snapshot)

    pos = _read_catalogue_series(group_paths, "Subhalo", "SubhaloPos")
    vel = _read_catalogue_series(group_paths, "Subhalo", "SubhaloVel")
    group_index = _read_catalogue_series(group_paths, "Subhalo", "SubhaloGroupNr")
    rank = _read_catalogue_series(group_paths, "Subhalo", "SubhaloRankInGr")
    lengths = _read_catalogue_series(group_paths, "Subhalo", "SubhaloLenType")
    masses = _read_catalogue_series(group_paths, "Subhalo", "SubhaloMassType")
    offsets = _read_catalogue_series(group_paths, "Subhalo", "SubhaloOffsetType")
    halfmass = _read_catalogue_series(group_paths, "Subhalo", "SubhaloHalfmassRadType")
    sfr = _read_catalogue_series(group_paths, "Subhalo", "SubhaloSFR", required=False)
    host_r200 = _read_catalogue_series(group_paths, "Group", "Group_R_Crit200", required=False)

    nsub = len(pos)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    group_index = np.asarray(group_index).reshape(-1)
    rank = np.asarray(rank).reshape(-1)
    lengths_raw = np.asarray(lengths)
    offsets_raw = np.asarray(offsets)
    masses = np.asarray(masses, dtype=np.float64)
    halfmass = np.asarray(halfmass, dtype=np.float64)
    if (pos.shape != (nsub, 3) or vel.shape != (nsub, 3)
            or any(len(array) != nsub for array in (group_index, rank, lengths_raw, offsets_raw, masses, halfmass))):
        raise ValueError("required SUBFIND galaxy fields are not aligned")
    if (lengths_raw.ndim != 2 or lengths_raw.shape[1] != 6
            or offsets_raw.ndim != 2 or offsets_raw.shape[1] != 6
            or masses.shape != (nsub, 6) or halfmass.shape != (nsub, 6)):
        raise ValueError("SUBFIND type fields must contain six GADGET particle types")
    if lengths_raw.dtype.kind not in "ui" or offsets_raw.dtype.kind not in "ui" or np.any(lengths_raw < 0):
        raise ValueError("SUBFIND particle lengths and offsets must be non-negative integers")
    lengths = lengths_raw.astype(np.int64, copy=False)
    if np.any(lengths_raw > np.iinfo(np.int64).max) or np.any(offsets_raw > np.iinfo(np.int64).max):
        raise ValueError("SUBFIND lengths and offsets exceed the supported index range")
    offsets = offsets_raw.astype(np.int64, copy=False)
    if group_index.dtype.kind not in "ui" or rank.dtype.kind not in "ui":
        raise ValueError("SUBFIND group IDs and ranks must use integer types")
    if np.any(group_index < 0) or np.any(rank < 0):
        raise ValueError("SUBFIND group IDs and ranks must be non-negative")

    star_total = int(snap["star_count"])
    star_count = lengths[:, 4]
    star_offset = offsets[:, 4]
    nonempty = star_count > 0
    if np.any(star_offset[nonempty] < 0) or np.any(star_count[nonempty] > star_total - star_offset[nonempty]):
        raise ValueError("SUBFIND star offsets extend beyond the group-ordered PartType4 snapshot")
    # Offsets must follow catalogue order and ranges must not overlap. Gaps are
    # valid: they represent FOF particles not bound to any SUBFIND subhalo.
    order = np.argsort(star_offset, kind="stable")
    occupied = order[star_count[order] > 0]
    if len(occupied) > 1 and np.any(star_offset[occupied[1:]] < star_offset[occupied[:-1]] + star_count[occupied[:-1]]):
        raise ValueError("SUBFIND stellar membership ranges overlap")

    if np.any(~np.isfinite(masses)) or np.any(masses < 0) or np.any(~np.isfinite(halfmass)) or np.any(halfmass < 0):
        raise ValueError("SUBFIND type masses and half-mass radii must be finite and non-negative")
    luminous = nonempty & (masses[:, 4] > 0)
    rows = np.flatnonzero(luminous)
    formation_redshift = np.full(len(rows), np.nan, dtype=np.float64)
    snapshot_a = float(snap["time"])
    if not math.isfinite(snapshot_a) or snapshot_a <= 0 or not math.isfinite(snap["box_size"]) or snap["box_size"] <= 0:
        raise ValueError("Galaxy snapshot needs a positive scale factor and comoving box size")
    if not math.isclose(float(snap["redshift"]), 1./snapshot_a-1., rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Galaxy snapshot scale factor disagrees with its redshift")

    for path in group_paths:
        with h5py.File(path, "r") as source:
            attrs = source["Header"].attrs
            for name, expected in (("Time", snapshot_a), ("Redshift", float(snap["redshift"]))):
                if name in attrs and not math.isclose(float(attrs[name]), expected, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("SUBFIND catalogue and particle snapshot epochs disagree")
    length_scale, sfr_scale, units = _unit_conversion([*group_paths, *snapshot_paths])
    if sfr is None:
        sfr_values = np.full(nsub, np.nan, dtype=np.float64)
        sfr_available = False
    else:
        sfr_values = np.asarray(sfr, dtype=np.float64).reshape(-1)
        if len(sfr_values) != nsub:
            raise ValueError("SubhaloSFR does not align with SUBFIND rows")
        if np.any(~np.isfinite(sfr_values)) or np.any(sfr_values < 0):
            raise ValueError("SubhaloSFR must be finite non-negative physical rates")
        sfr_available = True

    # Measure galaxy properties inside 0.1 of each host's catalogue R200c.
    # SUBFIND offsets restrict the membership to particles bound to that
    # subhalo, while the radius itself follows the measured host halo and
    # scales naturally with cosmology, mass resolution, and box size.
    aperture_radius = np.full(len(rows), np.nan, dtype=np.float64)
    aperture_stellar_mass = np.full(len(rows), np.nan, dtype=np.float64)
    aperture_gas_mass = np.full(len(rows), np.nan, dtype=np.float64)
    aperture_sfr = np.full(len(rows), np.nan, dtype=np.float64)
    aperture_formation_z = np.full(len(rows), np.nan, dtype=np.float64)
    aperture_star_count = np.zeros(len(rows), dtype=np.uint64)
    aperture_gas_count = np.zeros(len(rows), dtype=np.uint64)
    host_r200 = np.asarray(host_r200, dtype=float).reshape(-1) if host_r200 is not None else None
    if host_r200 is not None and len(host_r200):
        if np.any(group_index >= len(host_r200)):
            raise ValueError("SubhaloGroupNr references a missing FOF host for R200c aperture measurement")
    box_native = float(snap["box_size"])
    from core.data import snapshot_id_shards
    star_records, _ = snapshot_id_shards(snapshot_paths, 4)
    gas_records, _ = snapshot_id_shards(snapshot_paths, 0)
    for output_row, subhalo_row in enumerate(rows):
        star_slice = _bound_particle_slice(
            snapshot_paths, 4, int(star_offset[subhalo_row]), int(star_count[subhalo_row]),
            ("Coordinates", "Masses", "StellarFormationTime"), records=star_records,
        )
        star_mass_native = star_slice["Masses"]
        star_formation = star_slice["StellarFormationTime"]
        half_a = _weighted_median_scale_factor(star_formation, star_mass_native, snapshot_a)
        if np.isfinite(half_a):
            formation_redshift[output_row] = 1.0 / half_a - 1.0
        if host_r200 is None or not len(host_r200):
            continue
        host_radius = float(host_r200[group_index[subhalo_row]])
        if not math.isfinite(host_radius) or host_radius <= 0:
            continue
        radius = 0.1 * host_radius
        aperture_radius[output_row] = radius * length_scale
        center = pos[subhalo_row]

        star_xyz = star_slice["Coordinates"]
        if star_xyz is not None and len(star_xyz):
            if np.any(~np.isfinite(star_xyz)):
                raise ValueError("Bound stellar coordinates must be finite")
            delta = np.asarray(star_xyz, dtype=float) - center
            delta -= box_native * np.round(delta / box_native)
            in_aperture = np.einsum("ij,ij->i", delta, delta) <= radius * radius
            aperture_stellar_mass[output_row] = float(convert_pinned_native(np.sum(star_mass_native[in_aperture]),'mass_msun_h'))
            aperture_star_count[output_row] = int(np.count_nonzero(in_aperture))
            half_a_ap = _weighted_median_scale_factor(
                star_formation[in_aperture], star_mass_native[in_aperture], snapshot_a,
            )
            if math.isfinite(half_a_ap):
                aperture_formation_z[output_row] = 1.0 / half_a_ap - 1.0

        gas_count = int(lengths[subhalo_row, 0])
        gas_offset = int(offsets[subhalo_row, 0])
        if gas_count:
            gas_slice = _bound_particle_slice(
                snapshot_paths, 0, gas_offset, gas_count,
                ("Coordinates", "Masses", "StarFormationRate"), records=gas_records,
            )
            gas_xyz = gas_slice["Coordinates"]
            gas_native_mass = gas_slice["Masses"]
            if gas_xyz is not None and gas_native_mass is not None:
                if np.any(~np.isfinite(gas_xyz)) or np.any(~np.isfinite(gas_native_mass)) or np.any(gas_native_mass <= 0):
                    raise ValueError("Bound gas coordinates and masses must be finite with positive masses")
                delta = np.asarray(gas_xyz, dtype=float) - center
                delta -= box_native * np.round(delta / box_native)
                in_aperture = np.einsum("ij,ij->i", delta, delta) <= radius * radius
                aperture_gas_mass[output_row] = float(convert_pinned_native(np.sum(gas_native_mass[in_aperture]),'mass_msun_h'))
                aperture_gas_count[output_row] = int(np.count_nonzero(in_aperture))
                gas_sfr = gas_slice["StarFormationRate"]
                if gas_sfr is not None:
                    if np.any(~np.isfinite(gas_sfr)) or np.any(gas_sfr < 0):
                        raise ValueError("Bound gas SFR values must be finite and non-negative")
                    aperture_sfr[output_row] = float(np.sum(gas_sfr[in_aperture])) * sfr_scale

    dm_count = lengths[:, 1]
    resolved = (star_count >= int(minimum_star_particles)) & (dm_count >= int(minimum_dm_particles))
    galaxy_data = {
        "GalaxyID": rows.astype(np.uint64),
        "SubhaloID": rows.astype(np.uint64),
        "HostFOFID": group_index[rows].astype(np.uint64),
        "CentralFlag": (rank[rows] == 0),
        "Resolved": resolved[rows],
        "Position": np.mod(pos[rows] * length_scale, snap["box_size"] * length_scale),
        "Velocity": vel[rows],
        "StellarMass": convert_pinned_native(masses[rows, 4],'mass_msun_h'),
        "GasMass": convert_pinned_native(masses[rows, 0],'mass_msun_h'),
        "DMMass": convert_pinned_native(masses[rows, 1],'mass_msun_h'),
        "NStars": star_count[rows].astype(np.uint64),
        "NGas": lengths[rows, 0].astype(np.uint64),
        "NDM": dm_count[rows].astype(np.uint64),
        "StellarHalfMassRadius": halfmass[rows, 4] * length_scale,
        "SFR": sfr_values[rows] * sfr_scale,
        "FormationRedshift": formation_redshift,
        "ApertureRadius_Mpc_h": aperture_radius,
        "ApertureStellarMass": aperture_stellar_mass,
        "ApertureGasMass": aperture_gas_mass,
        "ApertureSFR": aperture_sfr,
        "ApertureFormationRedshift": aperture_formation_z,
        "ApertureNStars": aperture_star_count,
        "ApertureNGas": aperture_gas_count,
    }
    # A bound-subhalo gas fraction is reproducible from SUBFIND membership.
    # Unbound gas inside a visual aperture is not assigned to a galaxy.
    bound_baryons = galaxy_data["StellarMass"] + galaxy_data["GasMass"]
    galaxy_data["GasFraction"] = np.divide(
        galaxy_data["GasMass"], bound_baryons,
        out=np.full(len(rows), np.nan, dtype=np.float64), where=bound_baryons > 0,
    )
    aperture_baryons = galaxy_data["ApertureGasMass"] + galaxy_data["ApertureStellarMass"]
    galaxy_data["ApertureGasFraction"] = np.divide(
        galaxy_data["ApertureGasMass"], aperture_baryons,
        out=np.full(len(rows), np.nan, dtype=np.float64),
        where=np.isfinite(aperture_baryons) & (aperture_baryons > 0),
    )
    if any(len(values) != len(rows) or np.any(~np.isfinite(values)) for name, values in galaxy_data.items()
           if name in {"Position", "Velocity", "StellarMass", "GasMass", "DMMass", "StellarHalfMassRadius"}):
        raise ValueError("derived galaxy properties contain non-finite values or misaligned rows")

    destination = Path(output_path).expanduser().resolve() if output_path else run_dir / "results" / ("galaxies_%03d.hdf5" % snapshot)
    if destination.exists():raise FileExistsError('Galaxy catalogue destination already exists: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    from core.data import run_epoch_contract,coordinate_contract,write_dataset_contract
    from core.foundation import product_provenance, verify_product_provenance, publish_file, atomic_json
    epoch=run_epoch_contract(run_dir,snapshot,snapshot_a)
    coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
        periodic_wrapping='wrapped native fundamental box',velocity_convention='SUBFIND physical peculiar km/s')
    contract=product_provenance(run_dir,algorithm=__file__,snapshot=snapshot,
        source_files=source_records,coordinates=coordinates,
        selection='All luminous native SUBFIND rows; Resolved records declared particle count floors')
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["schema"] = SCHEMA
            output.attrs['data_contract_json']=json.dumps(contract,allow_nan=False)
            output.attrs['epoch_contract_json']=json.dumps(epoch,allow_nan=False)
            output.attrs['science_acceptance']='NOT_ACCEPTED'
            output.attrs["snapshot"] = snapshot
            output.attrs["redshift"] = float(snap["redshift"])
            output.attrs["scale_factor"] = snapshot_a
            output.attrs["box_size_mpc_h"] = float(snap["box_size"] * length_scale)
            output.attrs["selection"] = "all luminous SUBFIND subhalos; Resolved indicates the configured star and type-1 DM particle-count floors"
            output.attrs["minimum_star_particles"] = int(minimum_star_particles)
            output.attrs["minimum_dm_particles"] = int(minimum_dm_particles)
            output.attrs["central_definition"] = "SubhaloRankInGr == 0"
            output.attrs["formation_redshift_definition"] = "redshift at the mass-weighted median StellarFormationTime among currently bound star particles"
            output.attrs["sfr_source"] = "SubhaloSFR" if sfr_available else "unavailable in SUBFIND catalogue"
            output.attrs["unit_conversions_json"] = json.dumps(units, sort_keys=True)
            output.attrs["position_membership_source"] = "SUBFIND SubhaloOffsetType[4]/SubhaloLenType[4] into GADGET group-ordered PartType4 snapshot data"
            output.attrs["aperture_definition"] = (
                "bound PartType0/4 particles with periodic distance <= 0.1*FOF Group_R_Crit200; "
                "radius in Mpc/h; masses in Msun/h; SFR in Msun/year"
            )
            output.attrs["aperture_property_missing_policy"] = (
                "NaN when the host R200c, particle mass/SFR field, or valid stellar formation times are unavailable"
            )
            for name, values in galaxy_data.items():
                dataset = output.create_dataset(name,data=values,compression='gzip',shuffle=True,fletcher32=True)
                # Field-level units travel with a dataset even when a reader
                # extracts it without the catalogue's root metadata.
                if "Mass" in name and "Radius" not in name:
                    dataset.attrs["units"] = "Msun/h"
                elif name == "Position" or "Radius" in name:
                    dataset.attrs["units"] = "comoving Mpc/h"
                elif name == "Velocity":
                    dataset.attrs["units"] = "peculiar km/s"
                elif name in {"SFR", "ApertureSFR"}:
                    dataset.attrs["units"] = "Msun/year"
                else:
                    dataset.attrs["units"] = "dimensionless"
                if name in ('GalaxyID','SubhaloID','HostFOFID'):
                    unit='identity'
                elif name in ('NStars','NGas','NDM','ApertureNStars','ApertureNGas'):unit='particle count'
                elif name in ('Resolved','CentralFlag'):unit='boolean selection mask'
                else:unit=dataset.attrs['units']
                namespace=(contract['identity_namespaces']['halo'] if name=='HostFOFID'
                    else contract['identity_namespaces']['subhalo'] if name in ('GalaxyID','SubhaloID') else None)
                write_dataset_contract(dataset,units=unit,quantity='SUBFIND-derived '+name+'; definition retained in root metadata',
                    identity_namespace=namespace,epoch=epoch,provenance=contract,
                    coordinates=coordinates if name=='Position' or 'Radius' in name or name=='Velocity' else None,
                    ancestry={'semantics':'epoch-local native row; GalaxyID equals SubhaloID, no persistent merger identity'} if namespace else None,
                    missing_policy='NaN when the declared aperture/formation/SFR source is unavailable; zero counts are measured')
        verify_product_provenance(run_dir,contract)
        publish_file(temporary,destination)
    except BaseException as exc:
        if temporary.exists():atomic_json(temporary.with_name(temporary.name+'.failure.json'),
            {'status':'FAIL','science_acceptance':'NOT_ACCEPTED','failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise

    digest = file_sha256(destination)
    return {
        "status": "COMPLETE",
        "snapshot": snapshot,
        "redshift": float(snap["redshift"]),
        "galaxy_count": int(len(rows)),
        "resolved_galaxy_count": int(np.count_nonzero(resolved[rows])),
        "unresolved_luminous_subhalo_count": int(len(rows) - np.count_nonzero(resolved[rows])),
        "formation_redshift_measured_count": int(np.count_nonzero(np.isfinite(formation_redshift))),
        "sfr_status": "MEASURED" if sfr_available else "UNAVAILABLE",
        "path": str(destination),
        "sha256": digest,
    }


def verify_galaxy_catalogue(path, run_dir):
    """Verify a current sealed saved catalogue before reuse or visualization."""
    from core.data import verify_hdf5_dataset_payloads
    from core.foundation import verify_product_provenance
    verify_hdf5_dataset_payloads(path,require_contract=True)
    with h5py.File(path,'r') as source:
        if source.attrs.get('schema')!=SCHEMA:raise ValueError('Unsupported galaxy catalogue schema')
        contract=json.loads(source.attrs['data_contract_json'])
        verify_product_provenance(run_dir,contract)
        return {'status':'COMPLETE','snapshot':int(source.attrs['snapshot']),
            'redshift':float(source.attrs['redshift']),'galaxy_count':len(source['GalaxyID']),
            'resolved_galaxy_count':int(np.count_nonzero(source['Resolved'][:])),
            'minimum_star_particles':int(source.attrs['minimum_star_particles']),
            'minimum_dm_particles':int(source.attrs['minimum_dm_particles']),
            'path':str(Path(path).resolve()),'sha256':file_sha256(path),
            'science_acceptance':'NOT_ACCEPTED','reuse':'VERIFIED_IMMUTABLE_PRODUCT'}


@native_run_owner
def build_galaxy_catalogues(
    run_dir: str | Path,
    snapshots: list[int],
    *,
    minimum_star_particles: int = 100,
    minimum_dm_particles: int = 300,
) -> dict:
    """Build all requested per-snapshot catalogues and a checksummed index."""
    run_dir = Path(run_dir).expanduser().resolve()
    records = []
    for snapshot in sorted({int(value) for value in snapshots}):
        try:
            existing=run_dir/'results'/('galaxies_%03d.hdf5'%snapshot)
            if existing.exists():
                record=verify_galaxy_catalogue(existing,run_dir)
                if (record['minimum_star_particles']!=minimum_star_particles or
                        record['minimum_dm_particles']!=minimum_dm_particles):
                    raise ValueError('Completed galaxy catalogue uses different particle selection; choose a new output path')
                records.append(record)
                continue
            records.append(build_galaxy_catalogue(
                run_dir, snapshot,
                minimum_star_particles=minimum_star_particles,
                minimum_dm_particles=minimum_dm_particles,
            ))
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            records.append({"status": "UNAVAILABLE", "snapshot": int(snapshot), "reason": str(exc)})
    manifest = {
        "schema": SCHEMA,
        'science_acceptance':'NOT_ACCEPTED',
        "status": (
            "COMPLETE" if records and all(item.get("status") == "COMPLETE" for item in records)
            else "PARTIAL" if any(item.get("status") == "COMPLETE" for item in records)
            else "UNAVAILABLE"
        ),
        "run_directory": str(run_dir),
        "minimum_star_particles": int(minimum_star_particles),
        "minimum_dm_particles": int(minimum_dm_particles),
        "snapshots": records,
    }
    destination = run_dir / "results" / "galaxy_catalogue_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    from core.foundation import atomic_json
    atomic_json(destination,manifest)
    return manifest


def _stellar_search_controls(controls: dict) -> dict:
    """Validate explicit STF selections; distance is proper kpc, angle is pi radians.

    The published phase-space outlier method compares local velocity density
    with a smooth background, then links compatible outliers. Its thresholds
    select candidates; they do not turn a rendered filament into a discovery.
    Reference: Elahi et al. 2019, arXiv:1902.01010; vendored src/ui.cxx.
    """
    keys = {"minimum_particles", "velocity_neighbors", "physical_neighbors",
            "cell_fraction", "outlier_threshold", "significance_level",
            "velocity_ratio", "opening_angle_pi", "linking_length_kpc"}
    if set(controls) != keys:
        raise ValueError("STF controls require exactly: " + ", ".join(sorted(keys)))
    result = {}
    for key in keys:
        value = controls[key]
        if isinstance(value, (bool, str)) or not np.isscalar(value):
            raise ValueError("STF controls must be numeric scalars: " + key)
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("STF controls must be finite and positive: " + key)
        if key in {"minimum_particles", "velocity_neighbors", "physical_neighbors"}:
            if int(number) != number or number > np.iinfo(np.int32).max:
                raise ValueError("STF neighbor counts and minimum size must be exact positive int32")
            result[key] = int(number)
        else:
            result[key] = number
    if result["minimum_particles"] < 3 or result["velocity_neighbors"] < 4:
        raise ValueError("STF needs at least 3 members and 4 velocity neighbors")
    if result["physical_neighbors"] <= result["velocity_neighbors"]:
        raise ValueError("The physical neighbor pool must exceed the velocity neighbor count")
    if not 0 < result["cell_fraction"] <= 1 or not 0 < result["opening_angle_pi"] <= 1:
        raise ValueError("Cell fraction and opening angle/pi must be in (0,1]")
    if result["velocity_ratio"] <= 1:
        raise ValueError("The STF speed ratio must exceed one")
    return result


def _stellar_host_particles(run_dir: Path, snapshot: int, host: int, memory_gib: float):
    """Read exact FOF stellar membership and convert once to physical host coordinates.

    Includes stars outside SUBFIND-bound cores. The native adapter is a single
    nonperiodic host, not a whole cosmological volume disguised as one halo.
    Catalogue offsets address the concatenation of group-ordered PartType4.
    """
    from core.data import snapshot_id_shards
    groups, paths = _group_files(run_dir, snapshot), _snapshot_files(run_dir, snapshot)
    if not groups or not paths:
        raise FileNotFoundError("Stellar STF requires FOF catalogues and group-ordered snapshots")
    source_records=[{'path':str(path.resolve()),'sha256':file_sha256(path)} for path in paths+groups]
    _unit_conversion(groups + paths)
    _, star_total = snapshot_id_shards(paths, 4)
    declared = None
    rows = 0
    for path in groups:
        with h5py.File(path, "r") as source:
            header = source["Header"].attrs
            if any(key not in header for key in ("NumFiles", "Ngroups_Total", "Ngroups_ThisFile")):
                raise ValueError("FOF shards need complete file and group counts")
            values = tuple(header[key] for key in ("NumFiles", "Ngroups_Total", "Ngroups_ThisFile"))
            if any(np.asarray(value).ndim != 0 or np.asarray(value).dtype.kind not in "iu" or int(value) < 0
                   for value in values):
                raise ValueError("FOF header counts must be nonnegative integer scalars")
            files, total, this = map(int, values)
            if files != len(groups) or declared is not None and total != declared:
                raise ValueError("FOF catalogue shards are incomplete or disagree about totals")
            declared = total
            rows += this
    if rows != declared:
        raise ValueError("FOF shard rows do not sum to Ngroups_Total")
    lengths = _read_catalogue_series(groups, "Group", "GroupLenType")
    offsets = _read_catalogue_series(groups, "Group", "GroupOffsetType")
    positions = _read_catalogue_series(groups, "Group", "GroupPos")
    if (lengths.ndim != 2 or lengths.shape[1] != 6 or offsets.shape != lengths.shape
            or positions.shape != (len(lengths), 3)):
        raise ValueError("FOF host catalogue fields have incompatible shapes")
    if any(a.dtype.kind not in "iu" or np.any(a < 0) for a in (lengths, offsets)):
        raise ValueError("FOF counts/offsets must be nonnegative integers")
    if isinstance(host, bool) or int(host) != host or not 0 <= host < len(lengths):
        raise ValueError("Host must be a zero-based FOF catalogue row")
    start, count = int(offsets[host, 4]), int(lengths[host, 4])
    if count == 0 or start + count > star_total:
        raise ValueError("Selected FOF host has no stars or an invalid stellar span")
    # Covers Python arrays, native particle/tree/neighbor allocations and copies.
    # This is an allocation cap, not a claim of calibrated science resolution.
    if count > np.iinfo(np.uint32).max or count * 2048 > memory_gib * 1024**3:
        raise MemoryError("Selected host exceeds the STF allocation cap; increase --memory-gib")
    epoch = None
    for path in paths + groups:
        with h5py.File(path, "r") as source:
            header, parameters = source["Header"].attrs, source["Parameters"].attrs
            values = tuple(float(header.get(key, parameters.get(key, math.nan)))
                           for key in ("Time", "BoxSize", "HubbleParam"))
            a, box, h = values
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError("Every host/catalogue shard requires Time, BoxSize and HubbleParam")
            if "Redshift" in header and not math.isclose(float(header["Redshift"]), 1/a-1, abs_tol=1e-8):
                raise ValueError("Host epoch scale factor/redshift disagree")
            if epoch is not None and not np.allclose(values, epoch, rtol=1e-10, atol=0):
                raise ValueError("Host snapshot/catalogue epochs, box or h disagree")
            epoch = values
    fields = _bound_particle_slice(paths, 4, start, count,
                                   ("Coordinates", "Velocities", "Masses", "ParticleIDs"))
    if any(fields[key] is None for key in fields):
        raise ValueError("Host star Coordinates, Velocities, Masses and ParticleIDs are required")
    ids = fields["ParticleIDs"]
    if ids.shape != (count,) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("Star ParticleIDs must be exact nonnegative integers")
    ids = ids.astype(np.uint64, copy=False)
    if len(np.unique(ids)) != count:
        raise ValueError("Duplicated source star IDs cannot identify STF membership")
    xyz, velocity, masses = (np.asarray(fields[key], dtype=np.float64)
                             for key in ("Coordinates", "Velocities", "Masses"))
    if xyz.shape != (count, 3) or velocity.shape != xyz.shape or masses.shape != (count,):
        raise ValueError("Host particle fields have incompatible shapes")
    if any(np.any(~np.isfinite(array)) for array in (xyz, velocity, masses, positions)) or np.any(masses <= 0):
        raise ValueError("Host phase-space coordinates must be finite and stellar masses positive")
    a, box, h = epoch
    delta = (xyz - positions[host] + box/2) % box - box/2
    if np.any(np.ptp(delta, axis=0) >= box/2):
        raise ValueError("Host spans more than half the periodic box; its isolated geometry is ambiguous")
    # Pinned units: comoving kpc/h, 1e10 Msun/h, snapshot km/s/sqrt(a).
    xyz = delta * native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h)
    velocity = velocity * native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h)
    masses = masses * native_unit_factor('mass_msun',scale_factor=a,hubble_param=h)
    metadata = {"snapshot": int(snapshot), "host_fof_row": int(host), "scale_factor": a,
                "redshift": 1/a-1, "h": h, "box_comoving_kpc_h": box,
                "host_center_comoving_kpc_h": positions[host].tolist(),
                "source_files": [str(path) for path in paths + groups],
                'source_sha256':source_records,
                "selection": "all PartType4 members of the selected FOF host, including unbound stars",
                "source_particle_offset": start, "source_particle_count": count}
    if any(file_sha256(entry['path'])!=entry['sha256'] for entry in source_records):
        raise ValueError('Stellar host source changed during measurement')
    return xyz, velocity, masses, ids, metadata


def _stellar_native_membership(prefix: Path, count: int, empty: bool) -> np.ndarray:
    """Join native signed dense IDs to exact source rows, including unbound debris.

    Binding in a stars-only adapter omits the host DM potential; native binding
    labels therefore never become physical claims about stellar boundness.
    Both native membership lists are combined and checked for overlap/ranges.
    """
    membership = np.zeros(count, dtype=np.uint64)
    if empty:
        if prefix.with_suffix(".catalog_groups").exists():
            raise ValueError("Native empty selection unexpectedly has a group catalogue")
        return membership
    with h5py.File(str(prefix) + ".catalog_groups", "r") as f:
        sizes, bound, unbound = (np.asarray(f[key]) for key in ("Group_Size", "Offset", "Offset_unbound"))
        if any(a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < 0) for a in (sizes, bound, unbound)):
            raise ValueError("Invalid native STF membership offsets")
        n = len(sizes)
        if bound.shape != (n,) or unbound.shape != (n,) or int(f["Num_of_files"][0]) != 1:
            raise ValueError("Native STF membership requires complete serial catalogues")
    arrays = []
    for suffix in (".catalog_particles", ".catalog_particles.unbound"):
        with h5py.File(str(prefix) + suffix, "r") as f:
            ids = np.asarray(f["Particle_IDs"])
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 1) or np.any(ids > count):
                raise ValueError("Native STF IDs are outside the exact dense-ID map")
            arrays.append(ids.astype(np.int64) - 1)
    for offsets, array in zip((bound, unbound), arrays):
        ends = np.r_[offsets[1:], np.uint64(len(array))]
        if len(offsets) and (int(offsets[0]) != 0 or np.any(ends < offsets) or np.any(ends > len(array))):
            raise ValueError("Native STF membership lists have invalid spans")
    for i in range(n):
        indices = []
        for offsets, array in zip((bound, unbound), arrays):
            stop = int(offsets[i+1]) if i+1 < n else len(array)
            indices.append(array[int(offsets[i]):stop])
        index = np.concatenate(indices)
        if len(index) != int(sizes[i]) or len(np.unique(index)) != len(index) or np.any(membership[index]):
            raise ValueError("Native STF members are duplicated or disagree with Group_Size")
        membership[index] = i+1
    return membership


def _stellar_candidate_figure(destination: Path, xyz: np.ndarray, velocity: np.ndarray,
                              masses: np.ndarray, membership: np.ndarray, provenance: dict) -> list[str]:
    """Show measured host projections and velocity coherence without inventing streams.

    Both spatial panels integrate every selected host star's physical mass.
    Histogram pixels hold surface density in Msun/kpc²; there is no smoothing,
    population synthesis or inferred luminosity. Coloured dots carry the exact
    same candidate membership in all panels. The velocity view uses saved
    peculiar velocities, not trajectories integrated through a guessed field.
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from figures import export_figure

    bins = 128
    extent = float(np.max(np.abs(xyz))) * 1.04
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Stellar host has no finite spatial extent for a physical map")
    edges = np.linspace(-extent, extent, bins+1)
    area = float((edges[1]-edges[0])**2)
    maps = [np.histogram2d(xyz[:, 0], xyz[:, j], bins=(edges, edges), weights=masses)[0].T / area
            for j in (1, 2)]
    for image in maps:
        if not math.isclose(float(image.sum()*area), float(masses.sum()), rel_tol=1e-12):
            raise ArithmeticError("Stellar density projection lost host mass")
    positive = np.concatenate([image[image > 0] for image in maps])
    low, high = float(positive.min()), float(positive.max())
    if high <= low:
        # A constant field still needs a finite display interval, explicitly
        # recorded below; this padding changes no density measurement.
        low, high = low / math.sqrt(10), high * math.sqrt(10)
    candidate_ids = np.unique(membership[membership > 0])
    palette = plt.get_cmap("tab10")
    colors = {int(candidate): palette((int(candidate)-1) % 10) for candidate in candidate_ids}
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9}):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.7))
        fig.subplots_adjust(left=.055, right=.985, bottom=.265, top=.755, wspace=.30)
        fig.suptitle("Stellar phase-space substructure", fontsize=17, y=.98)
        fig.text(.5, .895,
                 f"Snapshot {provenance['snapshot']}  ·  FOF host {provenance['host_fof_row']}  ·  "
                 f"z = {provenance['redshift']:.3g}  ·  {len(xyz):,} stars  ·  {len(candidate_ids):,} candidates",
                 ha="center", fontsize=10)
        for ax, image, j in zip(axes[:2], maps, (1, 2)):
            ax.grid(False)
            ax.set_facecolor("#101827")
            artist = ax.imshow(np.ma.masked_less_equal(image, 0), origin="lower",
                               extent=(-extent, extent, -extent, extent), cmap="viridis",
                               norm=LogNorm(low, high), interpolation="nearest", rasterized=True)
            ax.set_xlabel("x [proper kpc]")
            ax.set_ylabel(("y" if j == 1 else "z") + " [proper kpc]")
            ax.set_title("Host-centred XY" if j == 1 else "Host-centred XZ")
            ax.set_aspect("equal", adjustable="box")
        axes[2].set_facecolor("#101827")
        axes[2].grid(False)
        axes[2].scatter(velocity[:, 0], velocity[:, 1], s=.7, c="#a1aab6", alpha=.23,
                        linewidths=0, rasterized=True)
        axes[2].set_xlabel(r"$v_x$ [km s$^{-1}$]")
        axes[2].set_ylabel(r"$v_y$ [km s$^{-1}$]")
        axes[2].set_title("Saved peculiar velocity")
        axes[2].set_aspect("equal", adjustable="box")
        for candidate in candidate_ids:
            selected = membership == candidate
            options = {"s": 5, "color": colors[int(candidate)], "edgecolors": "white",
                       "linewidths": .15, "rasterized": True}
            for ax, j in zip(axes[:2], (1, 2)):
                ax.scatter(xyz[selected, 0], xyz[selected, j], **options)
            axes[2].scatter(velocity[selected, 0], velocity[selected, 1], **options)
        cax = fig.add_axes((.125, .145, .445, .022))
        colorbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
        colorbar.set_label(r"Full-host stellar surface density $\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]", labelpad=3)
        shown = candidate_ids[:8]
        handles = [Line2D([], [], marker="o", linestyle="", markersize=4,
                          color=colors[int(candidate)], label=f"C{int(candidate)}") for candidate in shown]
        if handles:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .865),
                       ncol=len(handles), frameon=False, handletextpad=.4, columnspacing=1.4)
        else:
            fig.text(.5, .825, "No candidates meet the declared phase-space criteria", ha="center")
        notes = "Dots: measured membership; colour is a category, not stellar luminosity."
        if len(candidate_ids) > 8:
            notes += f" Legend shows 8 of {len(candidate_ids)} candidates; colours cycle after 10."
        fig.text(.5, .03, notes, ha="center", fontsize=8)
        metadata = {"provenance": provenance, "spatial_histogram_bins": bins,
                    "spatial_pixel_area_kpc2": area, "spatial_extent_kpc": [-extent, extent],
                    "surface_density_limits_msun_kpc2": [low, high],
                    "density_estimator": "full host mass-weighted histogram; no smoothing",
                    "host_mass_msun": float(masses.sum()), "scatter_particle_count": len(xyz),
                    "candidate_colors_rgba": {str(key): list(value) for key, value in colors.items()},
                    "category_identity": "local to this snapshot and host; no temporal matching"}
        return [str(path) for path in export_figure(fig, destination / "stellar_phase_space",
                                                    metadata=metadata, formats=["png", "pdf", "svg"],
                                                    dpi=300, layout="presentation")]


def _stellar_candidate_diagnostics(destination: Path, provenance: dict) -> list[str]:
    """Plot mass-weighted covariance diagnostics of the exact saved memberships.

    Principal RMS extents are sqrt(eigenvalues), not fitted stream lengths or
    half-mass radii. Velocity dispersion is sqrt(trace(covariance)/3), without
    subtracting a fitted velocity gradient or claiming equilibrium/binding.
    """
    from matplotlib import pyplot as plt
    from figures import export_figure

    from core.data import verify_hdf5_dataset_payloads
    verify_hdf5_dataset_payloads(destination/'candidates.hdf5')
    with h5py.File(destination / "candidates.hdf5", "r") as source:
        group = source["Candidates"]
        ids = np.asarray(group["CandidateID"])
        masses = np.asarray(group["Mass"])
        counts = np.asarray(group["ParticleCount"])
        eigenvalues = np.asarray(group["SpatialCovarianceEigenvalues"])
        velocity_covariance = np.asarray(group["VelocityCovariance"])
    if (np.any(~np.isfinite(masses)) or np.any(masses <= 0)
            or np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues < 0)
            or np.any(~np.isfinite(velocity_covariance))):
        raise ValueError("Invalid measured stellar candidate statistics")
    rms = np.sqrt(eigenvalues)
    dispersion = np.sqrt(np.maximum(np.trace(velocity_covariance, axis1=1, axis2=2), 0) / 3)
    major = rms[:, 2]
    resolved = major > 0
    ratios = np.divide(rms[:, :2], major[:, None], out=np.full((len(ids), 2), np.nan),
                       where=resolved[:, None])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    fig.suptitle(f"Measured stellar component diagnostics · snapshot {provenance['snapshot']}")
    for ax in axes:
        ax.grid(alpha=.2)
    if len(ids):
        axes[0].scatter(masses, major, label="Major principal RMS", marker="o")
        axes[0].scatter(masses, rms[:, 0], label="Minor principal RMS", marker="x")
        axes[0].legend(fontsize=8)
        axes[1].scatter(ratios[resolved, 1], ratios[resolved, 0])
        axes[2].scatter(masses, dispersion)
        for row, candidate in enumerate(ids):
            if len(ids) <= 20:
                axes[0].annotate(f"C{int(candidate)}", (masses[row], major[row]), fontsize=7)
                axes[2].annotate(f"C{int(candidate)}", (masses[row], dispersion[row]), fontsize=7)
    else:
        for ax in axes:
            ax.text(.5, .5, "No selected candidates", ha="center", va="center", transform=ax.transAxes)
    for ax in (axes[0], axes[2]):
        ax.set_xscale("log")
        ax.set_xlabel(r"Candidate stellar mass [M$_\odot$]")
    axes[0].set_ylabel("Principal RMS extent [proper kpc]")
    axes[1].set(xlim=(0, 1.05), ylim=(0, 1.05), xlabel="Intermediate / major RMS",
                ylabel="Minor / major RMS")
    axes[1].plot([0, 1], [0, 1], color="gray", linewidth=.6, linestyle=":")
    axes[2].set_ylabel(r"Mass-weighted 1-D dispersion [km s$^{-1}$]")
    fig.text(.5, .015, "Saved particle membership; morphology alone does not establish a tidal origin.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .93))
    metadata = {"provenance": provenance, "candidate_ids": ids.tolist(),
                "particle_counts": counts.tolist(), "mass_msun": masses.tolist(),
                "principal_rms_kpc_ascending": rms.tolist(),
                "velocity_dispersion_1d_kms": dispersion.tolist(),
                "shape_undefined_candidate_ids": ids[~resolved].tolist(),
                "measurement": "Mass-weighted population covariance; no Bessel correction, equilibrium assumption or fitted stream model"}
    return [str(path) for path in export_figure(fig, destination / "stellar_component_diagnostics",
                                               metadata=metadata, formats=["png", "pdf", "svg"],
                                               dpi=300, layout="presentation")]


def _stellar_candidate_blender_assets(destination: Path, xyz: np.ndarray,
                                      membership: np.ndarray, provenance: dict) -> Path:
    """Export saved physical positions and categorical colours for editable 3-D.

    PLY positions are float32 display coordinates in proper kpc. Original IDs,
    float64 coordinates and exact candidate membership remain in the HDF5 in
    the same row order. Only a 0..10 display category travels through float
    attributes; a large uint64 ParticleID is never converted to floating point.
    """
    from matplotlib import pyplot as plt
    from visualization import _write_blender_particle_ply
    from core.foundation import copy_blender_runtime

    categories = np.zeros(len(membership), dtype=np.uint8)
    selected = membership > 0
    categories[selected] = (membership[selected]-1) % 10 + 1
    ply = _write_blender_particle_ply(destination / "stellar_candidates.ply", xyz,
                                      stellar_candidate_color_bin=categories)
    from blender import verify_particle_ply_payload
    if verify_particle_ply_payload(ply)!=len(xyz):raise ValueError('Stellar candidate PLY payload count mismatch')
    copy_blender_runtime(destination)
    palette = [[.28, .30, .34, 1.]] + [list(plt.get_cmap("tab10")(i)) for i in range(10)]
    manifest = {"schema": "darkmattersim-stellar-candidate-blender-v1",
                "provenance": provenance, "ply": ply.name, "ply_sha256": file_sha256(ply),
                "catalogue": "candidates.hdf5",
                "catalogue_sha256": file_sha256(destination / "candidates.hdf5"),
                "particle_count": len(xyz), "candidate_count": int(membership.max(initial=0)),
                "scene_extent_kpc": 2 * float(np.max(np.abs(xyz))),
                "glyph_radius_extent_fraction": .0018,
                "length_unit": "proper kpc", "coordinate_precision": "float32 display only",
                "palette_srgb_rgba": palette,
                "membership": "vertex i corresponds to catalogue row i; exact uint64 IDs stay in HDF5",
                "color_categories": "0 unselected; 1+(CandidateID-1)%10 selected; colors repeat after 10",
                "time_display": "one exact saved snapshot; no interpolated motion"}
    manifest['assets']=[{'path':path.name,'sha256':file_sha256(path),'size_bytes':path.stat().st_size}
        for path in [ply,destination/'candidates.hdf5',*[destination/name for name in
            ('blender.py','blender.py','foundation.py')]]]
    path = destination / "stellar_candidates_blender.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    path.with_suffix('.json.sha256').write_text(file_sha256(path)+'\n')
    return path


@native_run_owner
def find_stellar_substructures(run_dir: Path, snapshot: int, host: int, *, controls: dict,
                              memory_gib: float = 4, threads: int = 1) -> dict:
    """Run the pinned STF on one measured host and save exact candidate membership.

    Stage dense positive IDs for the upstream signed-ID reader, preserving the
    original uint64 IDs separately. No particle is sampled, fabricated or moved
    to another species. Stellar ages/metallicity are not invented: the native
    build excludes optional stellar-population properties. Distances are proper
    host-centered kpc, masses Msun and velocities physical peculiar km/s.
    """
    import subprocess
    import tempfile
    from core.resources import cpu_thread_budget

    controls = _stellar_search_controls(controls)
    if isinstance(snapshot, bool) or int(snapshot) != snapshot or snapshot < 0:
        raise ValueError("Snapshot must be a nonnegative integer")
    if isinstance(threads, bool) or int(threads) != threads or not 1 <= threads <= cpu_thread_budget():
        raise ValueError("STF threads must fit the available CPU budget")
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError("STF memory_gib must be finite and positive")
    run_dir = Path(run_dir).expanduser().resolve()
    binary = native_engine_directory("velociraptor") / "build/stf"
    if not binary.is_file():
        raise FileNotFoundError("Build the bundled STF with python commands.py bootstrap --build-velociraptor")
    binary_sha=file_sha256(binary)
    xyz, velocities, masses, ids, metadata = _stellar_host_particles(run_dir, snapshot, host, memory_gib)
    n = len(ids)
    if n <= controls["physical_neighbors"] or controls["minimum_particles"] > n:
        raise ValueError("Selected host is smaller than the requested neighbor pool or minimum size")
    if n * controls["cell_fraction"] < 16 or n * controls["cell_fraction"] > n/8:
        raise ValueError("Each background cell needs >=16 stars and the host needs >=8 cells")
    native = {"HDF_name_convention": 1, "Input_includes_dm_particle": 0,
              "Input_includes_gas_particle": 0, "Input_includes_star_particle": 1,
              "Input_includes_bh_particle": 0, "Cosmological_input": 0, "h_val": 1,
              "Particle_search_type": 3, "Singlehalo_search": 1, "Search_for_substructure": 1,
              "FoF_search_type": 1, "Minimum_size": controls["minimum_particles"],
              "Length_input_unit_conversion_to_output_unit": 1,
              "Velocity_input_unit_conversion_to_output_unit": 1,
              "Mass_input_unit_conversion_to_output_unit": 1, "Length_unit_to_kpc": 1,
              "Velocity_to_kms": 1, "Mass_to_solarmass": 1, "Comoving_units": 0, "Period": 0,
              "Cell_fraction": controls["cell_fraction"], "Grid_type": 1,
              "Nsearch_velocity": controls["velocity_neighbors"],
              "Nsearch_physical": controls["physical_neighbors"],
              "Local_velocity_density_approximate_calculation": 0,
              "Outlier_threshold": controls["outlier_threshold"],
              "Significance_level": controls["significance_level"], "Velocity_ratio": controls["velocity_ratio"],
              "Velocity_opening_angle": controls["opening_angle_pi"],
              "Substructure_physical_linking_length": controls["linking_length_kpc"],
              "Iterative_searchflag": 0, "Unbind_flag": 0, "Binary_output": 2,
              "Separate_output_files": 0, "Calculate_radial_profiles": 0,
              "Calculate_aperture_quantities": 0, "Verbose": 0}
    destination = run_dir / "results/stellar_substructures" / f"snapshot_{snapshot:03d}_host_{host}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Stellar candidate output already exists: " + str(destination))
    from core.data import run_epoch_contract,coordinate_contract,write_dataset_contract
    from core.foundation import product_provenance, verify_product_provenance, publish_directory, atomic_json
    from core.native import native_run_lock_fds
    coordinates=coordinate_contract(position_units='proper kpc',
        origin='saved fixed native FOF center retained in global comoving kpc/h',
        periodic_wrapping='minimum image within unambiguous isolated host; native adapter is nonperiodic',
        velocity_convention='saved physical peculiar km/s; no host bulk subtraction')
    epoch=run_epoch_contract(run_dir,int(snapshot),metadata['scale_factor'])
    contract=product_provenance(run_dir,algorithm=__file__,selection=metadata['selection'],
        snapshot=int(snapshot),source_files=metadata['source_sha256'],coordinates=coordinates,
        helper_sources=[Path(__file__).with_name(name) for name in
            ('visualization.py','blender.py','foundation.py','blender.py','figures.py')])
    metadata.update(epoch_contract=epoch,data_contract=contract,science_acceptance='NOT_ACCEPTED')
    stage=Path(tempfile.mkdtemp(prefix='.stellar-stf-',dir=destination.parent))
    work=stage/'native_work';work.mkdir()
    try:
        with h5py.File(work / "stars.hdf5", "w") as f:
            header = f.create_group("Header")
            counts = np.zeros(6, dtype=np.uint32); counts[4] = n
            # Normalized noncosmological header is only a transport adapter.
            # Actual source epoch/cosmology stays in the output provenance.
            values = {"BoxSize": 0., "MassTable": np.zeros(6), "NumPart_ThisFile": counts,
                      "NumPart_Total": counts, "NumPart_Total_HighWord": np.zeros(6, dtype=np.uint32),
                      "Omega0": 0., "OmegaLambda": 0., "Redshift": 0., "Time": 1.,
                      "HubbleParam": 1., "NumFilesPerSnapshot": 1}
            for key, value in values.items(): header.attrs[key] = value
            p = f.create_group("PartType4")
            for key, value in {"Coordinates": xyz, "Velocities": velocities, "Masses": masses,
                               "ParticleIDs": np.arange(1, n+1, dtype=np.uint64)}.items():
                dataset=p.create_dataset(key,data=value,chunks=True,fletcher32=True)
                write_dataset_contract(dataset,units={'Coordinates':'proper kpc','Velocities':'km/s','Masses':'Msun','ParticleIDs':'identity'}[key],
                    quantity='STF noncosmological transport adapter: '+key,epoch=epoch,coordinates=coordinates,
                    identity_namespace=contract['identity_namespaces']['star_population']+':STF_dense_adapter' if key=='ParticleIDs' else None,
                    ancestry={'dense_id_mapping':'original uint64 saved IDs in DenseIDMap/OriginalParticleIDs; adapter ID = row+1'},provenance=contract)
            original=f.create_group('DenseIDMap').create_dataset('OriginalParticleIDs',data=ids,chunks=True,fletcher32=True)
            write_dataset_contract(original,units='identity',quantity='exact saved stellar identity at each dense adapter row',
                identity_namespace=contract['identity_namespaces']['star_population'],epoch=epoch,provenance=contract)
            f.attrs['scientific_epoch_contract_json']=json.dumps(epoch,allow_nan=False)
            f.attrs['adapter_header_semantics']='Time=1, HubbleParam=1, Cosmological_input=0 are transport only; actual saved epoch is in scientific_epoch_contract_json'
        configuration = "# Proper kpc, Msun, km/s; nonperiodic measured stellar host.\n"
        configuration += "\n".join(f"{key}={value}" for key, value in native.items()) + "\n"
        (work / "finder.cfg").write_text(configuration)
        prefix = work / "candidates"
        environment = dict(os.environ, OMP_NUM_THREADS=str(threads))
        invocation={'status':'STARTED','executable':str(binary),'executable_sha256':binary_sha,
            'configuration_sha256':file_sha256(work/'finder.cfg'),'input_sha256':file_sha256(work/'stars.hdf5'),
            'epoch_contract':epoch,'data_contract':contract,'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(work/'invocation.json',invocation)
        with (work / "finder.log").open("w") as log:
            subprocess.run([str(binary), "-I", "2", "-i", str(work / "stars"), "-C",
                            str(work / "finder.cfg"), "-o", str(prefix)], cwd=work, env=environment,
                           stdout=log,stderr=subprocess.STDOUT,check=True,pass_fds=native_run_lock_fds(run_dir))
        if file_sha256(binary)!=binary_sha:raise ValueError('STF executable changed during native invocation')
        if (file_sha256(work/'finder.cfg')!=invocation['configuration_sha256'] or
                file_sha256(work/'stars.hdf5')!=invocation['input_sha256']):
            raise ValueError('STF staged configuration or scientific input changed during invocation')
        atomic_json(work/'completion.json',{'status':'NATIVE_PROCESS_RETURNED_SUCCESS','science_acceptance':'NOT_ACCEPTED',
            'invocation_sha256':file_sha256(work/'invocation.json'),'native_log_sha256':file_sha256(work/'finder.log')})
        log_text = (work / "finder.log").read_text()
        membership = _stellar_native_membership(prefix, n, "STF_EMPTY_SELECTION" in log_text)
        candidate_count = int(membership.max(initial=0))
        metadata.update({"schema": "darkmattersim-stellar-stf-v1", "method": "VELOCIraptor/STF velocity-density outliers",
                         "method_reference": "https://arxiv.org/abs/1902.01010", "controls": controls,
                         "units": {"Coordinates": "proper kpc", "Velocities": "peculiar km/s", "Masses": "Msun"},
                         "candidate_count": candidate_count, "threads": int(threads),
                         "native_executable_sha256": binary_sha,
                         'native_work_sha256':{path.name:file_sha256(path) for path in work.iterdir() if path.is_file()},
                         "binding": "unbinding disabled; native stars-only binding labels are discarded",
                         "interpretation": "phase-space substructure candidates; candidate morphology is measured from member particles"})
        with h5py.File(stage / "candidates.hdf5", "w") as f:
            f.attrs["provenance_json"] = json.dumps(metadata, allow_nan=False)
            for key, value in {"ParticleIDs": ids, "CandidateID": membership, "Coordinates": xyz,
                               "Velocities": velocities, "Masses": masses}.items():
                dataset=f.create_dataset(key,data=value,compression='gzip',shuffle=True,fletcher32=True)
                namespace=(contract['identity_namespaces']['star_population'] if key=='ParticleIDs' else
                    contract['identity_namespaces']['progenitor']+f':STF:snapshot:{snapshot}:host:{host}' if key=='CandidateID' else None)
                write_dataset_contract(dataset,units=metadata['units'].get(key,'identity'),identity_namespace=namespace,
                    quantity='exact saved STF host '+key,epoch=epoch,coordinates=coordinates if key in ('Coordinates','Velocities') else None,
                    provenance=contract,missing_policy='CandidateID=0 denotes unselected; all saved parent stars retained',
                    ancestry={'semantics':'STF epoch-local candidate membership; no cross-epoch progenitor identity'} if key=='CandidateID' else None)
            group = f.create_group("Candidates")
            rows = []
            for candidate in range(1, candidate_count+1):
                selected = membership == candidate; mass = masses[selected]
                center = np.average(xyz[selected], weights=mass, axis=0)
                mean_velocity = np.average(velocities[selected], weights=mass, axis=0)
                delta = xyz[selected] - center; dv = velocities[selected] - mean_velocity
                covariance = (delta.T * mass) @ delta / mass.sum()
                velocity_covariance = (dv.T * mass) @ dv / mass.sum()
                eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
                rows.append((candidate, int(selected.sum()), float(mass.sum()), center, mean_velocity,
                             eigenvalues, velocity_covariance))
            for key, values, unit in (
                ("CandidateID", np.arange(1,candidate_count+1,dtype=np.uint64), "integer"),
                ("ParticleCount", np.asarray([r[1] for r in rows],dtype=np.uint64), "count"),
                ("Mass", np.asarray([r[2] for r in rows]), "Msun"),
                ("Center", np.asarray([r[3] for r in rows]).reshape(-1,3), "proper kpc"),
                ("MeanVelocity", np.asarray([r[4] for r in rows]).reshape(-1,3), "peculiar km/s"),
                ("SpatialCovarianceEigenvalues", np.asarray([r[5] for r in rows]).reshape(-1,3), "kpc^2; ascending"),
                ("VelocityCovariance", np.asarray([r[6] for r in rows]).reshape(-1,3,3), "(km/s)^2")):
                dataset=group.create_dataset(key,data=values,chunks=True,fletcher32=True)
                write_dataset_contract(dataset,units='identity' if key=='CandidateID' else unit,
                    identity_namespace=contract['identity_namespaces']['progenitor']+f':STF:snapshot:{snapshot}:host:{host}' if key=='CandidateID' else None,
                    quantity='mass-weighted saved STF membership summary: '+key,epoch=epoch,provenance=contract,
                    coordinates=coordinates if key in ('Center','MeanVelocity') else None)
            f.attrs['data_contract_json']=json.dumps(contract,allow_nan=False)
        _stellar_candidate_figure(stage, xyz, velocities, masses, membership, metadata)
        _stellar_candidate_diagnostics(stage, metadata)
        _stellar_candidate_blender_assets(stage, xyz, membership, metadata)
        (stage / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        (stage/'provenance.sha256').write_text(file_sha256(stage/'provenance.json')+'\n')
        (stage / "finder.cfg").write_text(configuration)
        (stage / "finder.log").write_text(log_text)
        verify_product_provenance(run_dir,contract)
        if file_sha256(binary)!=binary_sha:raise ValueError('STF executable changed before publication')
        publish_directory(stage,destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return {"directory": str(destination), "candidate_count": candidate_count, "particle_count": n,
            "catalogue": str(destination / "candidates.hdf5")}

def galaxies_main() -> None:
    """Parse command arguments and dispatch this module's documented workflow."""
    parser = argparse.ArgumentParser(description="Build resolved SUBFIND galaxy catalogues from GADGET snapshots.")
    parser.add_argument("run_dir", help="completed DarkMatterSim run directory")
    parser.add_argument("--snapshot", action="append", type=int, help="snapshot number to process; repeatable")
    parser.add_argument("--minimum-star-particles", type=int, default=100)
    parser.add_argument("--minimum-dm-particles", type=int, default=300)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.snapshot:
        snapshots = args.snapshot
    else:
        snapshots = sorted({
            int(match.group(1))
            for path in (run_dir / "data").glob("snapshot_*.hdf5")
            if (match := re.match(r"snapshot_(\d{3,})(?:\.|\.hdf5$)", path.name))
        })
    result = build_galaxy_catalogues(
        run_dir, snapshots,
        minimum_star_particles=args.minimum_star_particles,
        minimum_dm_particles=args.minimum_dm_particles,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    galaxies_main()


def galaxy_ownership_members(graph_directory, *, galaxy_namespace, galaxy_id, snapshot, particle_namespace):
    """Reusable exact graph intake; display IDs and native physical namespaces stay separate."""
    from halos import IdentityQueries
    return IdentityQueries(graph_directory).members_at_epoch((galaxy_namespace,galaxy_id,snapshot),particle_namespace=particle_namespace)

# ========================================================================
# subhalo_catalogue: Export every saved SUBFIND row, including unresolved and central objects.
import csv
import tempfile
from core.data import hdf5_scalar_attr, run_epoch_contract, coordinate_contract


def subhalo_table(run_dir, snapshot):
    paths = _group_files(Path(run_dir), snapshot)
    if not paths:
        return {"status": "SKIPPED", "reason": "SUBFIND catalogue unavailable"}
    snapshots = _snapshot_files(Path(run_dir), snapshot)
    if not snapshots:
        raise ValueError("Subhalo export requires its saved particle epoch")
    from core.foundation import file_sha256 as digest
    input_records = [{'path':str(p.resolve()),'sha256':digest(p)} for p in paths+snapshots]
    from analysis import validate_gadget_snapshot_set
    validate_gadget_snapshot_set(snapshots, snapshot)
    _check_file_series(paths, "Header", "Nsubhalos_Total")
    _unit_conversion(paths + snapshots)
    import yaml
    from core.resources import memory_limit_gb
    cfg = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    total = 0
    for path in paths:
        with h5py.File(path, "r") as source:
            total += int(source["Header"].attrs["Nsubhalos_ThisFile"])
    if total*1024 + 64*1024**2 > memory_limit_gb(cfg)*1024**3*.5:
        raise MemoryError("Complete subhalo catalogue exceeds half the configured host memory budget")
    with h5py.File(snapshots[0], "r") as source:
        hdr = source["Header"].attrs
        a = hdf5_scalar_attr(hdr["Time"], "Time")
        box = hdf5_scalar_attr(hdr["BoxSize"], "BoxSize")
        h = float(source["Parameters"].attrs["HubbleParam"])
    if not np.isfinite([a, box, h]).all() or min(a, box, h) <= 0:
        raise ValueError("Invalid snapshot epoch, box or h")
    for path in paths:
        with h5py.File(path, "r") as source:
            hdr = source["Header"].attrs
            values = (hdf5_scalar_attr(hdr["Time"], "Time"),
                      hdf5_scalar_attr(hdr["BoxSize"], "BoxSize"),
                      float(source["Parameters"].attrs["HubbleParam"]))
            if not np.allclose(values, (a, box, h), rtol=1e-10, atol=0):
                raise ValueError("SUBFIND and snapshot epoch/box/h disagree")
    fields = {}
    required = ("SubhaloPos", "SubhaloMass", "SubhaloHalfmassRad",
                "SubhaloGroupNr", "SubhaloLenType")
    optional = ("SubhaloVel", "SubhaloMassType", "SubhaloRankInGr",
                "SubhaloParentRank", "SubhaloVelDisp", "SubhaloVmax",
                "SubhaloVmaxRad", "SubhaloIDMostbound", "SubhaloSFR")
    for name in required + optional:
        # An optional field present on only some populated shards is corrupt,
        # rather than an unavailable property.
        present = False
        for path in paths:
            with h5py.File(path, "r") as source:
                present |= "Subhalo" in source and name in source["Subhalo"]
        fields[name] = _read_catalogue_series(paths, "Subhalo", name,
                                             required=name in required or present)
    n = len(fields["SubhaloMass"])
    for name, values in fields.items():
        if values is None:
            continue
        expected = (n, 3) if name in {"SubhaloPos", "SubhaloVel"} else (
            (n, 6) if name in {"SubhaloLenType", "SubhaloMassType"} else (n,))
        if values.shape != expected:
            raise ValueError(f"{name} has invalid row shape")
        if name in {"SubhaloGroupNr", "SubhaloLenType", "SubhaloIDMostbound",
                    "SubhaloRankInGr"}:
            if values.dtype.kind not in "iu" or np.any(values < 0):
                raise ValueError(f"{name} must contain nonnegative integers")
        elif name == "SubhaloParentRank":
            if values.dtype.kind not in "iu" or np.any(values < -1):
                raise ValueError("SubhaloParentRank must contain integer ranks or -1")
        elif not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")
        if name in {"SubhaloMass", "SubhaloMassType", "SubhaloHalfmassRad", "SubhaloVmaxRad"} and np.any(values < 0):
            raise ValueError(f"{name} contains negative mass/radius")
    if fields["SubhaloMassType"] is not None and not np.allclose(
            fields["SubhaloMassType"].sum(axis=1), fields["SubhaloMass"], rtol=2e-5, atol=0):
        raise ValueError("Subhalo species masses do not sum to bound mass")
    if fields['SubhaloIDMostbound'] is not None:
        fields['SubhaloIDMostbound'] = exact_uint64_ids(fields['SubhaloIDMostbound'])
    fields['SubhaloGroupNr']=fields['SubhaloGroupNr'].astype(np.uint64)
    from core.foundation import product_provenance
    coordinates = coordinate_contract(position_units='comoving kpc/h',origin='global simulation Cartesian box origin',
        periodic_wrapping='native coordinates wrapped to [0, box)',velocity_convention='SUBFIND peculiar km/s; no sqrt(a) correction')
    contract = product_provenance(run_dir,algorithm=__file__,selection='All native SUBFIND rows, including unresolved and central objects',
        source_files=input_records,coordinates=coordinates,snapshot=int(snapshot))
    return {"status": "COMPLETE", "schema": "darkmattersim-subhalos-v1", "snapshot": int(snapshot), "scale_factor": a,
            'epoch_contract':run_epoch_contract(run_dir,int(snapshot),a),'data_contract':contract,
            "redshift": 1/a-1, "h": h, "count": n,
            "fields": fields, "source_files": [str(p.resolve()) for p in paths],
            "units": {"native_mass": "1e10 Msun/h", "native_position_radius": "comoving kpc/h",
                      "native_velocity": "peculiar km/s; no sqrt(a) correction for SUBFIND"},
            "interpretation": "All rows retained, including centrals. Rows are epoch-local; use merger trees for identity. BH species mass is particle gravity mass."}


@native_run_owner
def export_subhalo_catalogue(run_dir, snapshot):
    """Publish an immutable human CSV and exact typed columns in one sealed bundle."""
    from core.data import numpy_archive_contract,verify_numpy_archive_contract
    from core.foundation import atomic_json, file_sha256 as digest, publish_directory, verify_product_provenance
    table=subhalo_table(run_dir,snapshot)
    if table['status']!='COMPLETE':return table
    fields,n=table['fields'],table['count']
    output=Path(run_dir)/'results';output.mkdir(parents=True,exist_ok=True)
    destination=output/f'subhalos_{snapshot:03d}'
    if destination.exists():raise FileExistsError('Subhalo bundle already exists: '+str(destination))
    columns={};specs={};csv_names=[];availability={}
    namespaces=table['data_contract']['identity_namespaces']
    def add(name,values,units,*,namespace=None,available=True,quantity=None):
        csv_names.append(name)
        columns[name]=np.asarray(values)
        specs[name]={'units':units,'quantity':quantity or 'native SUBFIND-derived '+name,
            'missing_policy':'none' if available else 'CSV empty; typed placeholders require '+name+'_available'}
        if namespace is not None:specs[name]['identity_namespace']=namespace
        if not available:
            availability[name]=np.zeros(n,dtype=bool)
            columns[name+'_available']=availability[name]
            specs[name+'_available']={'units':'boolean availability mask','quantity':'source availability for '+name}
    def optional(name,field,units,*,identity=False):
        data=fields[field]
        values=np.zeros(n,dtype=np.uint64) if identity and data is None else np.full(n,np.nan) if data is None else data
        add(name,values,units,namespace=namespaces['native_particle'] if identity else None,available=data is not None)
    add('snapshot',np.full(n,snapshot,dtype=np.uint64),'storage index',quantity='snapshot storage label; actual epoch is in the canonical contract')
    add('subhalo_row',np.arange(n,dtype=np.uint64),'identity',namespace=namespaces['subhalo'])
    add('host_fof_row',fields['SubhaloGroupNr'].astype(np.uint64),'identity',namespace=namespaces['halo'])
    ranks=fields['SubhaloRankInGr']
    add('central',np.zeros(n,dtype=bool) if ranks is None else ranks==0,'boolean central flag',available=ranks is not None,
        quantity='SubhaloRankInGr == 0; not a persistent identity')
    add('mass_msun_h',fields['SubhaloMass']*1e10,'Msun/h')
    add('half_mass_radius_comoving_kpc_h',fields['SubhaloHalfmassRad'],'comoving kpc/h')
    add('half_mass_radius_proper_kpc',fields['SubhaloHalfmassRad']*table['scale_factor']/table['h'],'proper kpc')
    for axis,index in zip('xyz',range(3)):
        name=axis+'_comoving_kpc_h';add(name,fields['SubhaloPos'][:,index],'comoving kpc/h')
        specs[name]['coordinates']=table['data_contract']['coordinates']
    for name,field,unit in [('vmax_km_s','SubhaloVmax','km/s'),('vmax_radius_comoving_kpc_h','SubhaloVmaxRad','comoving kpc/h'),
                          ('velocity_dispersion_km_s','SubhaloVelDisp','km/s')]:optional(name,field,unit)
    optional('most_bound_particle_id','SubhaloIDMostbound','identity',identity=True)
    optional('sfr_msun_year','SubhaloSFR','Msun/year')
    types=('gas','dm','coarse','type3','stars','bh_gravity')
    for index,kind in enumerate(types):add('n_'+kind,fields['SubhaloLenType'][:,index].astype(np.uint64),'particle count')
    mass_type=fields['SubhaloMassType']
    for index,kind in enumerate(types):
        add('mass_'+kind+'_msun_h',np.full(n,np.nan) if mass_type is None else mass_type[:,index]*1e10,
            'Msun/h',available=mass_type is not None)
    stage=Path(tempfile.mkdtemp(prefix='.subhalos-',dir=output))
    try:
        csv_path=stage/f'subhalos_{snapshot:03d}.csv'
        with csv_path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(csv_names)
            for row in range(n):
                writer.writerow(['' if name in availability and not availability[name][row] else columns[name][row].item()
                                 for name in csv_names])
        np.savez_compressed(stage/'columns.npz',**columns)
        archive=numpy_archive_contract(stage/'columns.npz',specs,epoch=table['epoch_contract'],
            provenance=table['data_contract'])
        table['artifact']=str(Path(destination.name)/csv_path.name)
        table['numeric_artifact']=str(Path(destination.name)/'columns.npz')
        metadata={key:value for key,value in table.items() if key!='fields'}
        metadata.update(csv_sha256=digest(csv_path),array_contract=archive,csv_columns=csv_names,
            science_acceptance='NOT_ACCEPTED',ancestry='Rows and FOF links are epoch-local; most-bound ID is in the native particle union')
        atomic_json(stage/'report.json',metadata)
        (stage/'report.sha256').write_text(digest(stage/'report.json')+'\n')
        verify_numpy_archive_contract(stage/'columns.npz',archive)
        verify_product_provenance(run_dir,table['data_contract'])
        publish_directory(stage,destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return table


def verify_subhalo_export(directory):
    """Verify both serialized representations before reading a published bundle."""
    from core.data import verify_numpy_archive_contract
    from core.foundation import file_sha256 as digest
    directory=Path(directory);path=directory/'report.json'
    if not (directory/'report.sha256').is_file() or (directory/'report.sha256').read_text().strip()!=digest(path):
        raise ValueError('Subhalo report checksum mismatch')
    report=json.loads(path.read_text())
    if report.get('schema')!='darkmattersim-subhalos-v1':raise ValueError('Unsupported subhalo schema')
    csv_path=directory/Path(report['artifact']).name
    if digest(csv_path)!=report['csv_sha256']:raise ValueError('Subhalo CSV checksum mismatch')
    verify_numpy_archive_contract(directory/'columns.npz',report['array_contract'])
    return report


def export_subhalo_ownership(run_dir, destination, *, selections, maximum_nodes, maximum_edges, histories=None, tree_intake=None, ssp_birth_audits=None, native_source_histories=None, request_source_hashes=None, reuse_completed=False):
    """Publish exact selected native memberships for all later ownership queries."""
    from halos import gadget_catalogue
    from halos import graph_from_catalogues
    from core.native import native_run_lock
    with native_run_lock(run_dir,operation='subhalo ownership graph'):
        catalogues=[]
        for selection in selections:
            if selection.get('finder')=='velociraptor':
                from halos import read_velociraptor_catalogue
                if set(selection)!={'finder','directory'}:raise ValueError('Select a sealed VELOCIraptor product')
                catalogues.append(read_velociraptor_catalogue(selection['directory']))
            elif selection.get('finder')=='rockstar':
                from halos import rockstar_catalogue
                catalogues.append(rockstar_catalogue(run_dir,**{k:v for k,v in selection.items() if k!='finder'}))
            else:catalogues.append(gadget_catalogue(run_dir,**selection))
        graph,sources=graph_from_catalogues(catalogues,maximum_nodes=maximum_nodes,maximum_edges=maximum_edges)
        if histories:
            from halos import add_stellar_history
            for history in histories:sources.update(add_stellar_history(graph,**history))
        if tree_intake:
            from halos import add_consistent_tree_history
            sources.update(add_consistent_tree_history(graph,run_dir,**tree_intake))
        for intake in ssp_birth_audits or []:
            sources.update(add_native_ssp_birth_audit(graph, **intake))
        for intake in native_source_histories or []:
            sources.update(add_native_hydro_source_history(graph,**intake))
        sources[str(Path(__file__).resolve())]=file_sha256(__file__)
        sources.update(request_source_hashes or {})
        return graph.publish(destination,source_hashes=sources,reuse_completed=reuse_completed)

# ========================================================================
# identity_graph: Exact identities, catalogue comparison and native tree diagnostics.
from contextlib import closing, contextmanager
from core.data import numpy_archive_contract, verify_numpy_archive_contract
from core.foundation import canonical_digest, publish_directory
from core.data import export_numeric_audit
from collections import defaultdict
import sqlite3


# Identity graph
#
# Exact, epoch-qualified ownership/ancestry graph with indexed reversible queries.
#
# Integer identities are encoded as eight-byte big-endian blobs in SQLite. SQLite's
# signed integer range therefore never truncates uint64 particle IDs. Native
# ancestry and derived confidence matches use separate edge kinds.

ANCESTRY={'descendant','birth_parent','mass_return','bh_merger','stripping'}
EDGE_KINDS=ANCESTRY|{'membership','bound_owner','parent_zoom','matching_confidence','host'}
KINDS={'particle','gas','parent_gas','ssp','bh','progenitor','halo','subhalo','galaxy','stream','tree','stream_realization'}


def identity(value):
    return int(exact_uint64_ids([value])[0])


def node_key(namespace, object_id, epoch):
    if not isinstance(namespace,str) or not namespace.strip():raise ValueError('Explicit nonempty namespace required')
    if type(epoch) is not int or epoch<0:raise ValueError('Epoch must be a nonnegative saved storage index')
    return (namespace,identity(object_id),epoch)


def _blob(value):return identity(value).to_bytes(8,'big')

def _decoded(row):return (row[0],int.from_bytes(row[1],'big'),int(row[2]))


class IdentityGraph:
    """Build exact observations; absent nodes never become invented detections."""
    def __init__(self, *, maximum_nodes, maximum_edges):
        if any(type(x) is not int or x<1 for x in (maximum_nodes,maximum_edges)):
            raise ValueError('Declare positive node/edge workspace limits')
        self.maximum_nodes=maximum_nodes;self.maximum_edges=maximum_edges
        self.nodes={};self.edges=[];self._edge_keys=set();self.epochs={}

    def add_epoch(self, storage_index, epoch_contract):
        if type(storage_index) is not int or storage_index<0:raise ValueError('Invalid storage index')
        if not isinstance(epoch_contract,dict) or not epoch_contract.get('canonical_time'):
            raise ValueError('Use the shared explicit epoch contract')
        if storage_index in self.epochs and self.epochs[storage_index]!=epoch_contract:
            raise ValueError('Epoch contract changed')
        self.epochs[storage_index]=dict(epoch_contract)

    def add_node(self, namespace, object_id, epoch, *, kind, status='DETECTED', attributes=None):
        key=node_key(namespace,object_id,epoch)
        if epoch not in self.epochs:raise ValueError('Declare shared epoch before observations')
        if kind not in KINDS or status not in {'DETECTED','UNBOUND','MISSING','AMBIGUOUS'}:
            raise ValueError('Unsupported observation kind/status')
        if key in self.nodes:raise ValueError('Duplicate exact observation identity')
        if len(self.nodes)>=self.maximum_nodes:raise MemoryError('Identity node budget exceeded')
        attributes={} if attributes is None else attributes
        if not isinstance(attributes,dict):raise ValueError('Observation attributes must be an object')
        json.dumps(attributes,allow_nan=False)
        self.nodes[key]={'kind':kind,'status':status,'attributes':dict(attributes)}
        return key

    def add_edge(self, source, target, kind, *, evidence):
        source=node_key(*source);target=node_key(*target)
        if source not in self.nodes or target not in self.nodes:raise ValueError('Edge endpoints require explicit observations')
        if kind not in EDGE_KINDS or not isinstance(evidence,dict) or not evidence:
            raise ValueError('Typed edge and explicit native/derived evidence required')
        if self.nodes[source]['status']=='MISSING' or self.nodes[target]['status']=='MISSING':
            raise ValueError('Missing observations cannot establish links')
        sk,tk=self.nodes[source]['kind'],self.nodes[target]['kind']
        if kind in {'membership','bound_owner'} and (sk not in {'particle','gas','parent_gas','ssp','bh'} or tk not in {'halo','subhalo','galaxy','stream','progenitor','stream_realization'}):raise ValueError('Membership requires particle and physical owner kinds')
        if kind=='birth_parent' and (sk not in {'gas','parent_gas'} or tk!='ssp'):raise ValueError('Birth ancestry requires parent gas and SSP')
        if kind=='bh_merger' and (sk!='bh' or tk!='bh'):raise ValueError('BH merger requires BH observations')
        if kind=='mass_return' and (sk!='ssp' or tk not in {'gas','parent_gas'}):raise ValueError('Mass-return ancestry requires SSP and recipient gas')
        if kind in {'membership','bound_owner','host'} and source[2]!=target[2]:
            raise ValueError('Ownership must refer to the same saved epoch')
        if kind in ANCESTRY:
            if evidence.get('origin')!='native_ancestry':raise ValueError('Physical ancestry requires native evidence; use matching_confidence for derived matches')
            a=self.epochs[source[2]];b=self.epochs[target[2]]
            if a['canonical_time']!=b['canonical_time']:raise ValueError('Ancestry cannot mix cosmological and isolated clocks')
            clock='actual_scale_factor' if a['canonical_time']=='scale_factor' else 'native_time_code'
            if a.get(clock) is None or b.get(clock) is None or b[clock]<a[clock]:
                raise ValueError('Ancestry must follow actual physical time')
        if kind=='parent_zoom' and evidence.get('origin')!='lagrangian_identity':
            raise ValueError('Parent/zoom links require recorded Lagrangian identity, not row matching')
        key=(source,target,kind)
        if key in self._edge_keys:raise ValueError('Duplicate typed graph edge')
        if len(self.edges)>=self.maximum_edges:raise MemoryError('Identity edge budget exceeded')
        json.dumps(evidence,allow_nan=False)
        self._edge_keys.add(key);self.edges.append({'source':source,'target':target,'kind':kind,'evidence':dict(evidence)})

    def add_membership(self, owner, particle_namespace, exact_particle_ids, *, bound, evidence, particle_kind='particle'):
        owner=node_key(*owner)
        if owner not in self.nodes:raise ValueError('Membership owner must exist')
        if type(bound) is not bool:raise ValueError('Bound membership must be explicit')
        ids=exact_uint64_ids(exact_particle_ids)
        with self._atomic_batch():
            for value in ids:
                key=node_key(particle_namespace,int(value),owner[2])
                if key not in self.nodes:self.add_node(*key,kind=particle_kind)
                elif self.nodes[key]['kind']!=particle_kind:raise ValueError('Particle namespace aliases incompatible native species')
                self.add_edge(key,owner,'bound_owner' if bound else 'membership',evidence=evidence)

    @contextmanager
    def _atomic_batch(self):
        """Rollback additions on a rejected batch without copying a large graph."""
        node_count = len(self.nodes); edge_count = len(self.edges)
        try:
            yield
        except BaseException:
            for edge in self.edges[edge_count:]:
                self._edge_keys.discard((edge['source'], edge['target'], edge['kind']))
            del self.edges[edge_count:]
            while len(self.nodes) > node_count: self.nodes.popitem()
            raise

    def add_membership_edges(self, records):
        """Attach exact native membership batches without inferring row identity."""
        required={'owner','particle_namespace','exact_particle_ids','bound','evidence'}
        with self._atomic_batch():
            for record in records:
                if not isinstance(record,dict) or not required <= set(record) or set(record)-required-{'particle_kind'}:
                    raise ValueError('Membership edge batch requires explicit owner, namespace, exact IDs, binding and evidence')
                self.add_membership(record['owner'],record['particle_namespace'],record['exact_particle_ids'],
                    bound=record['bound'],evidence=record['evidence'],particle_kind=record.get('particle_kind','particle'))

    def add_epoch_nodes(self, epoch, observations):
        """Import typed native observations; explicit missing rows remain nodes."""
        with self._atomic_batch():
            return [self.add_node(row['namespace'],row['id'],epoch,kind=row['kind'],
                status=row.get('status','DETECTED'),attributes=row.get('attributes',{})) for row in observations]

    def add_ancestry_edges(self, links):
        """Native lifecycle evidence is distinct from confidence/display links."""
        with self._atomic_batch():
            for link in links:
                if link['kind'] not in ANCESTRY:raise ValueError('Expected a physical ancestry edge')
                self.add_edge(link['source'],link['target'],link['kind'],evidence=link['evidence'])
            self.validate()

    def add_parent_zoom_edges(self, links):
        with self._atomic_batch():
            for link in links:
                if link.get('kind', 'parent_zoom') != 'parent_zoom': raise ValueError('Expected a parent/zoom identity edge')
                self.add_edge(link['source'],link['target'],'parent_zoom',evidence=link['evidence'])

    def validate_no_cross_namespace_row_joins(self):
        # Namespaces are part of every node and SQL key; row equality supplies
        # no edge. Every cross-namespace edge must carry explicit evidence.
        return self.validate()

    def validate(self):
        # Kahn's algorithm avoids recursion failure for long native histories.
        outgoing={key:[] for key in self.nodes};indegree={key:0 for key in self.nodes}
        for edge in self.edges:
            if edge['kind'] in ANCESTRY:
                outgoing[edge['source']].append(edge['target']);indegree[edge['target']]+=1
        queue=[key for key,n in indegree.items() if n==0];seen=0
        while queue:
            key=queue.pop();seen+=1
            for target in outgoing[key]:
                indegree[target]-=1
                if indegree[target]==0:queue.append(target)
        if seen!=len(self.nodes):raise ValueError('Native ancestry contains a cycle')
        return {'status':'PASS','nodes':len(self.nodes),'edges':len(self.edges)}

    @directory_owner('destination')
    def publish_indexed_queries(self, destination, *, source_hashes, reuse_completed=False):
        self.validate();destination=Path(destination).resolve()
        if not source_hashes:raise ValueError('Graph publication requires source-bound products')
        sources={str(Path(p).resolve()):h for p,h in source_hashes.items()}
        sources[str(Path(__file__).resolve())]=file_sha256(__file__)
        for name in ('data.py','foundation.py','foundation.py','foundation.py'):
            helper=Path(__file__).with_name(name).resolve();sources[str(helper)]=file_sha256(helper)
        for p,h in sources.items():
            if file_sha256(p)!=h:raise ValueError('Identity source changed')
        nodes=[{'key':list(k),**v} for k,v in sorted(self.nodes.items())]
        payload={'epochs':self.epochs,'nodes':nodes,'edges':self.edges,'sources':sources}
        fingerprint=canonical_digest(payload)
        if destination.exists():
            if not reuse_completed:raise FileExistsError(destination)
            result=verify_identity_graph(destination)
            if result['fingerprint']!=fingerprint:raise ValueError('Completed graph inputs differ')
            return result
        destination.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.identity-',dir=destination.parent))
        try:
            with closing(sqlite3.connect(stage/'index.sqlite')) as db, db:
                db.executescript('''CREATE TABLE nodes(namespace TEXT NOT NULL,id BLOB NOT NULL,epoch INTEGER NOT NULL,kind TEXT NOT NULL,status TEXT NOT NULL,attributes TEXT NOT NULL,PRIMARY KEY(namespace,id,epoch));
                CREATE TABLE edges(sns TEXT NOT NULL,sid BLOB NOT NULL,sepoch INTEGER NOT NULL,tns TEXT NOT NULL,tid BLOB NOT NULL,tepoch INTEGER NOT NULL,kind TEXT NOT NULL,evidence TEXT NOT NULL);
                CREATE INDEX outgoing ON edges(sns,sid,sepoch,kind);CREATE INDEX incoming ON edges(tns,tid,tepoch,kind);''')
                db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?,?)',[(k[0],_blob(k[1]),k[2],v['kind'],v['status'],json.dumps(v['attributes'],sort_keys=True)) for k,v in sorted(self.nodes.items())])
                db.executemany('INSERT INTO edges VALUES(?,?,?,?,?,?,?,?)',[(e['source'][0],_blob(e['source'][1]),e['source'][2],e['target'][0],_blob(e['target'][1]),e['target'][2],e['kind'],json.dumps(e['evidence'],sort_keys=True)) for e in self.edges])
            arrays={'object_ids':np.array([k[1] for k in sorted(self.nodes)],dtype=np.uint64),
                    'storage_epochs':np.array([k[2] for k in sorted(self.nodes)],dtype=np.uint64)}
            np.savez_compressed(stage/'nodes.npz',**arrays)
            contract=numpy_archive_contract(stage/'nodes.npz',{'object_ids':{'units':'identity','quantity':'exact IDs; each row namespace is in report nodes','identity_namespace':'graph:row-qualified-object'},
                'storage_epochs':{'units':'storage index','quantity':'epoch labels; physical clocks in report epochs'}},
                provenance={'sources':sources,'meaning':'Observation keys include namespace and actual epoch; no cross-namespace row joins'})
            report={'schema':'exact-identity-graph-v1','status':'COMPLETE','fingerprint':fingerprint,**payload,
                'array_contract':contract,'sqlite_sha256':file_sha256(stage/'index.sqlite'),'science_acceptance':'NOT_ACCEPTED'}
            atomic_json(stage/'report.json',report);atomic_text(stage/'report.sha256',file_sha256(stage/'report.json')+'\n')
            for p,h in sources.items():
                if file_sha256(p)!=h:raise ValueError('Identity source changed before publication')
            publish_directory(stage,destination);return report
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'failure.json',{'status':'FAIL','reason':str(exc)})
            raise


    publish = publish_indexed_queries

def verify_identity_graph(directory):
    root=Path(directory).resolve();path=root/'report.json'
    if file_sha256(path)!=(root/'report.sha256').read_text().strip():raise ValueError('Identity graph report changed')
    report=json.loads(path.read_text())
    if report.get('schema')!='exact-identity-graph-v1' or report.get('status')!='COMPLETE':raise ValueError('Incomplete identity graph')
    if file_sha256(root/'index.sqlite')!=report['sqlite_sha256']:raise ValueError('Identity graph index changed')
    verify_numpy_archive_contract(root/'nodes.npz',report['array_contract'])
    for p,h in report['sources'].items():
        if file_sha256(p)!=h:raise ValueError('Identity graph source changed')
    return report


class IdentityQueries:
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.report=verify_identity_graph(self.directory)

    def _rows(self,sql,parameters):
        # Verify before every query; callers cannot retain an unchecked changed index.
        verify_identity_graph(self.directory)
        with closing(sqlite3.connect((self.directory/'index.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
            return db.execute(sql,parameters).fetchall()

    def members_at_epoch(self,owner,*,particle_namespace,bound_only=False):
        key=node_key(*owner)
        kinds="('bound_owner')" if bound_only else "('bound_owner','membership')"
        rows=self._rows('SELECT DISTINCT sid FROM edges WHERE tns=? AND tid=? AND tepoch=? AND sns=? AND kind IN '+kinds+' ORDER BY sid',
            (key[0],_blob(key[1]),key[2],particle_namespace))
        return np.array([int.from_bytes(r[0],'big') for r in rows],dtype=np.uint64)

    def bound_owner(self,particle,*,owner_kind=None):
        key=node_key(*particle);rows=self._rows('SELECT e.tns,e.tid,e.tepoch,n.kind FROM edges e JOIN nodes n ON n.namespace=e.tns AND n.id=e.tid AND n.epoch=e.tepoch WHERE e.sns=? AND e.sid=? AND e.sepoch=? AND e.kind=?',(*_sql_key(key),'bound_owner'))
        owners=[_decoded(r[:3]) for r in rows if owner_kind is None or r[3]==owner_kind]
        observation=self._rows('SELECT status FROM nodes WHERE namespace=? AND id=? AND epoch=?',_sql_key(key))
        absent='UNDETECTED' if not observation or observation[0][0]=='MISSING' else 'UNBOUND' if observation[0][0]=='UNBOUND' else 'OWNERSHIP_UNAVAILABLE'
        return {'status':'BOUND' if len(owners)==1 else 'AMBIGUOUS' if owners else absent,'owners':owners}

    def descendants(self,node):
        key=node_key(*node);found=set();queue=[key]
        while queue:
            source=queue.pop()
            rows=self._rows("SELECT tns,tid,tepoch FROM edges WHERE sns=? AND sid=? AND sepoch=? AND kind IN ('descendant','birth_parent','bh_merger','mass_return','stripping')",_sql_key(source))
            for row in rows:
                target=_decoded(row)
                if target not in found:found.add(target);queue.append(target)
        return sorted(found)

    def birth_parent(self,node):
        key=node_key(*node)
        rows=self._rows('SELECT sns,sid,sepoch FROM edges WHERE tns=? AND tid=? AND tepoch=? AND kind=?',(*_sql_key(key),'birth_parent'))
        parents=[_decoded(r) for r in rows]
        return {'status':'KNOWN' if len(parents)==1 else 'AMBIGUOUS' if parents else 'UNAVAILABLE','parents':parents}

    def parent_zoom_match(self,node):
        key=node_key(*node)
        rows=self._rows('SELECT sns,sid,sepoch,tns,tid,tepoch,evidence FROM edges WHERE kind=? AND ((sns=? AND sid=? AND sepoch=?) OR (tns=? AND tid=? AND tepoch=?))',('parent_zoom',*_sql_key(key),*_sql_key(key)))
        matches=[{'source':_decoded(r[:3]),'target':_decoded(r[3:6]),'evidence':json.loads(r[6])} for r in rows]
        return {'status':'MATCHED' if len(matches)==1 else 'AMBIGUOUS' if matches else 'UNAVAILABLE','matches':matches}


def _sql_key(key):return key[0],_blob(key[1]),key[2]


def graph_from_catalogues(catalogues, *, maximum_nodes, maximum_edges):
    """Consume normalized measured catalogues; row equality never links epochs."""
    graph=IdentityGraph(maximum_nodes=maximum_nodes,maximum_edges=maximum_edges)
    sources={};particle_observations={}
    for catalogue in catalogues:
        validate_catalogue(catalogue)
        epoch=catalogue['epoch']['snapshot_storage_index']
        if epoch is None:raise ValueError('Ownership catalogue requires actual saved storage label')
        graph.add_epoch(epoch,catalogue['epoch'])
        for path,h in catalogue['source_hashes'].items():
            if path in sources and sources[path]!=h:raise ValueError('Catalogue graph source generation changed')
            sources[path]=h
        namespace=catalogue['namespace'];particle_namespace=catalogue['particle_namespace']
        for obj in catalogue['objects']:
            attributes={k:v for k,v in obj.items() if k!='members'}
            owner=node_key(namespace,obj['id'],epoch)
            if owner not in graph.nodes:
                graph.add_node(*owner,kind='halo' if catalogue['finder']=='fof' else 'subhalo',attributes=attributes)
            elif graph.nodes[owner]['attributes']!=attributes:
                raise ValueError('Species selections disagree about their native owner')
            if obj['members'] is None:continue
            graph.add_membership_edges([{'owner':owner,'particle_namespace':particle_namespace,
                'exact_particle_ids':obj['members'],'bound':catalogue['finder']!='fof' and 'NO_NATIVE_UNBINDING' not in obj['quality'],
                'particle_kind':catalogue.get('particle_kind','particle'),
                'evidence':{'origin':'native_membership','finder':catalogue['finder'],'source_hashes':catalogue['source_hashes']}}])
            for pid in obj['members']:particle_observations.setdefault((particle_namespace,pid),set()).add(epoch)
    for (namespace,pid),epochs in particle_observations.items():
        ordered=sorted(epochs,key=lambda e:graph.epochs[e]['actual_scale_factor'])
        for a,b in zip(ordered,ordered[1:]):
            graph.add_edge((namespace,pid,a),(namespace,pid,b),'descendant',
                evidence={'origin':'native_ancestry','basis':'exact saved particle identity continuity; no halo row ancestry inferred'})
    return graph,sources


# Finder comparison
#
# Compare independently defined catalogues without replacing science values.
#
# Exact overlap merit is N_shared^2/(N_left N_right). Phase-space cuts restrict
# candidates; they do not establish an identity. All ties and unmatched objects
# are retained. Centres are proper kpc and velocities peculiar km/s.

def validate_catalogue(catalogue):
    required={'finder','namespace','particle_namespace','epoch','box_proper_kpc','objects','source_hashes'}
    if not isinstance(catalogue,dict) or set(catalogue)-{'particle_kind'}!=required:raise ValueError('Incomplete normalized finder catalogue')
    if catalogue.get('particle_kind','particle') not in {'particle','gas','ssp','bh'}:raise ValueError('Unsupported native particle kind')
    if not all(isinstance(catalogue[k],str) and catalogue[k] for k in ('finder','namespace','particle_namespace')):
        raise ValueError('Finder and identity namespaces must be explicit')
    a=catalogue['epoch'].get('actual_scale_factor')
    if not isinstance(a,(float,int)) or isinstance(a,bool) or not math.isfinite(a) or a<=0:raise ValueError('Finder needs actual cosmological epoch')
    box=catalogue['box_proper_kpc']
    if not isinstance(box,(float,int)) or isinstance(box,bool) or not math.isfinite(box) or box<=0:raise ValueError('Positive proper periodic box required')
    ids=[]
    for obj in catalogue['objects']:
        if set(obj)-{'radius_definition','native_binding_status'}!={'id','position_proper_kpc','velocity_peculiar_kms','mass_msun','mass_definition','radius_proper_kpc','vmax_kms','central','host_id','members','quality'}:
            raise ValueError('Malformed normalized finder object')
        ids.append(obj['id'])
        for field in ('position_proper_kpc','velocity_peculiar_kms'):
            x=np.asarray(obj[field],dtype=float)
            if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('Finite three-dimensional phase coordinates required')
        for field in ('mass_msun','radius_proper_kpc','vmax_kms'):
            x=obj[field]
            if x is not None and (isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<0):raise ValueError('Invalid finder '+field)
        if not isinstance(obj['mass_definition'],str) or not obj['mass_definition']:raise ValueError('Explicit mass definition required')
        if obj['central'] is not None and type(obj['central']) is not bool:raise ValueError('Central status requires bool or unavailable')
        if obj['host_id'] is not None:exact_uint64_ids([obj['host_id']])
        if obj['members'] is not None:exact_uint64_ids(obj['members'])
        if not isinstance(obj['quality'],list):raise ValueError('Quality flags must be retained')
    exact_uint64_ids(ids)
    if not catalogue['source_hashes']:raise ValueError('Finder sources must be bound')
    for p,h in catalogue['source_hashes'].items():
        if file_sha256(p)!=h:raise ValueError('Finder input changed')
    return catalogue


def match_by_exact_membership_and_phase_space(left,right,request):
    validate_catalogue(left);validate_catalogue(right)
    required={'minimum_shared_particles','minimum_overlap_fraction','maximum_distance_proper_kpc','maximum_velocity_kms','tie_merit_tolerance','maximum_candidate_pairs','allow_phase_only'}
    if set(request)!=required:raise ValueError('Declare complete frozen matching thresholds')
    for k in ('minimum_shared_particles','maximum_candidate_pairs'):
        if type(request[k]) is not int or request[k]<1:raise ValueError('Positive integer matching limits required')
    for k in ('minimum_overlap_fraction','maximum_distance_proper_kpc','maximum_velocity_kms','tie_merit_tolerance'):
        x=request[k]
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<0:raise ValueError('Invalid matching threshold')
    if request['minimum_overlap_fraction']>1 or type(request['allow_phase_only']) is not bool:raise ValueError('Invalid overlap/phase policy')
    if left['epoch']['actual_scale_factor']!=right['epoch']['actual_scale_factor'] or left['box_proper_kpc']!=right['box_proper_kpc']:
        raise ValueError('Finder epoch or periodic box mismatch')
    if left['particle_namespace']!=right['particle_namespace']:raise ValueError('Cannot overlap identities across particle namespaces')
    if left['namespace']==right['namespace']:raise ValueError('Independent finder object namespaces must differ')
    if len(left['objects'])*len(right['objects'])>request['maximum_candidate_pairs']:raise MemoryError('Finder pair budget exceeded; partition the selected targets')
    box=left['box_proper_kpc'];candidates=[]
    for a in left['objects']:
        for b in right['objects']:
            delta=(np.asarray(a['position_proper_kpc'])-b['position_proper_kpc']+box/2)%box-box/2
            distance=float(np.linalg.norm(delta));speed=float(np.linalg.norm(np.asarray(a['velocity_peculiar_kms'])-b['velocity_peculiar_kms']))
            if distance>request['maximum_distance_proper_kpc'] or speed>request['maximum_velocity_kms']:continue
            measured=a['members'] is not None and b['members'] is not None
            shared=None;merit=None;fractions=None
            if measured:
                aa=exact_uint64_ids(a['members']);bb=exact_uint64_ids(b['members'])
                shared=int(len(np.intersect1d(aa,bb,assume_unique=True)))
                fractions=[shared/len(aa) if len(aa) else 0.,shared/len(bb) if len(bb) else 0.]
                if shared<request['minimum_shared_particles'] or min(fractions)<request['minimum_overlap_fraction']:continue
                merit=fractions[0]*fractions[1]
            elif not request['allow_phase_only']:continue
            same_mass=a['mass_definition']==b['mass_definition'] and a['mass_msun'] is not None and b['mass_msun'] is not None
            same_radius=a.get('radius_definition') is not None and a.get('radius_definition')==b.get('radius_definition') and a['radius_proper_kpc'] is not None and b['radius_proper_kpc'] is not None
            candidates.append({'left_id':a['id'],'right_id':b['id'],'shared_particles':shared,'overlap_fractions':fractions,'merit':merit,
                'distance_proper_kpc':distance,'velocity_difference_kms':speed,'status':'EXACT_OVERLAP_CANDIDATE' if measured else 'PHASE_ONLY_CANDIDATE',
                'left_mass_msun':a['mass_msun'],'right_mass_msun':b['mass_msun'],'mass_definitions':[a['mass_definition'],b['mass_definition']],
                'mass_difference_msun':b['mass_msun']-a['mass_msun'] if same_mass else None,
                'mass_comparison_status':'SAME_DEFINITION' if same_mass else 'INCOMPARABLE_OR_UNAVAILABLE',
                'radius_definitions':[a.get('radius_definition','UNDECLARED'),b.get('radius_definition','UNDECLARED')],
                'radii_proper_kpc':[a['radius_proper_kpc'],b['radius_proper_kpc']],
                'radius_difference_proper_kpc':b['radius_proper_kpc']-a['radius_proper_kpc'] if same_radius else None,
                'vmax_difference_kms':b['vmax_kms']-a['vmax_kms'] if a['vmax_kms'] is not None and b['vmax_kms'] is not None else None,
                'central_disagreement':a['central']!=b['central'] if a['central'] is not None and b['central'] is not None else None,
                'quality':sorted(set(a['quality']+b['quality']))})
    def best(rows):
        exact=[r for r in rows if r['merit'] is not None]
        if not exact:return []
        high=max(r['merit'] for r in exact)
        return [r for r in exact if high-r['merit']<=request['tie_merit_tolerance']]
    for row in candidates:
        aa=best([r for r in candidates if r['left_id']==row['left_id']]);bb=best([r for r in candidates if r['right_id']==row['right_id']])
        if row['merit'] is not None:
            row['status']='RECIPROCAL_UNIQUE_MATCH' if len(aa)==len(bb)==1 and aa[0] is row and bb[0] is row else 'AMBIGUOUS_OR_NONRECIPROCAL'
    matched_l={r['left_id'] for r in candidates if r['status']=='RECIPROCAL_UNIQUE_MATCH'};matched_r={r['right_id'] for r in candidates if r['status']=='RECIPROCAL_UNIQUE_MATCH'}
    return {'schema':'finder-comparison-v1','status':'COMPLETE','request':request,'candidates':candidates,
        'unmatched_or_ambiguous_left':[r['id'] for r in left['objects'] if r['id'] not in matched_l],
        'unmatched_or_ambiguous_right':[r['id'] for r in right['objects'] if r['id'] not in matched_r],
        'science_acceptance':'NOT_ACCEPTED','interpretation':'Recorded finder disagreements; no mass or native tree link is overwritten'}


def publish_finder_disagreements(destination,left,right,request):
    result=match_by_exact_membership_and_phase_space(left,right,request);rows=result['candidates']
    arrays={'left_id':np.array([r['left_id'] for r in rows],dtype=np.uint64),'right_id':np.array([r['right_id'] for r in rows],dtype=np.uint64),
        'distance_proper_kpc':np.array([r['distance_proper_kpc'] for r in rows],dtype=float),
        'velocity_difference_kms':np.array([r['velocity_difference_kms'] for r in rows],dtype=float)}
    specs={k:{'units':'identity' if k.endswith('_id') else 'proper kpc' if 'distance' in k else 'km/s','quantity':k} for k in arrays}
    specs['left_id']['identity_namespace']=left['namespace'];specs['right_id']['identity_namespace']=right['namespace']
    sources={**left['source_hashes'],**right['source_hashes'],str(Path(__file__).resolve()):file_sha256(__file__)}
    return export_numeric_audit(destination,{**result,'catalogues':[left,right]},arrays,specs,algorithm=__file__,
        context={'epoch':left['epoch'],'selection':'all declared independently measured finder objects'},source_hashes=sources)


# Tree integrity
#
# Preserve native tree links and publish diagnostic flags, never silent repairs.

def detect_disappearance_swap_jump_fragmentation(nodes,links,*,epochs,maximum_mass_ratio,maximum_position_jump_proper_kpc,box_proper_kpc,transient_maximum_detections,maximum_nodes=100000,maximum_links=200000,maximum_swap_pairs=1000000,link_origin='native_tree'):
    """Nodes are epoch-qualified; links are native ancestry, not catalogue rows.

    Gaps/large jumps are diagnostics, not proof of an unphysical evolution.
    Branch proposals must be published separately with the original links.
    """
    if link_origin not in {'native_tree','proposed_repair'}:raise ValueError('Declare native or proposed diagnostic link origin')
    if not isinstance(epochs,list) or not epochs or len(set(epochs))!=len(epochs) or any(type(e) is not int or e<0 for e in epochs):raise ValueError('Declare unique ordered physical-epoch storage labels')
    for x in (maximum_mass_ratio,maximum_position_jump_proper_kpc,box_proper_kpc):
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<=0:raise ValueError('Positive finite diagnostic thresholds required')
    if maximum_mass_ratio<=1 or type(transient_maximum_detections) is not int or transient_maximum_detections<1:raise ValueError('Invalid mass/transient limit')
    for limit in (maximum_nodes,maximum_links,maximum_swap_pairs):
        if type(limit) is not int or limit<1:raise ValueError('Positive integer diagnostic workspace limits required')
    if len(nodes)>maximum_nodes or len(links)>maximum_links:raise MemoryError('Tree diagnostic workspace exceeded')
    index={e:i for i,e in enumerate(epochs)};by_key={};flags=[];outgoing=defaultdict(list);incoming=defaultdict(list)
    def exact_key(value):
        if len(value)!=3 or not isinstance(value[0],str) or not value[0] or type(value[2]) is not int:raise ValueError('Exact namespace and integer storage epoch required')
        return value[0],int(exact_uint64_ids([value[1]])[0]),value[2]
    seen_links=set()
    for node in nodes:
        key=exact_key((node['namespace'],node['id'],node['epoch']))
        if key in by_key or key[2] not in index or not key[0]:raise ValueError('Duplicate/undeclared tree observation')
        if not math.isfinite(node['mass_msun']) or node['mass_msun']<0:raise ValueError('Invalid tree mass')
        if len(node['position_proper_kpc'])!=3 or not all(math.isfinite(x) for x in node['position_proper_kpc']):raise ValueError('Invalid tree position')
        by_key[key]=node
    def flag(kind,key,detail):flags.append({'kind':kind,'node':list(key),'detail':detail,'status':'DIAGNOSTIC_REQUIRES_REVIEW'})
    for link in links:
        source=exact_key(link['source']);target=exact_key(link['target'])
        if link.get('origin')!=link_origin:raise ValueError('Tree diagnostic link origin mismatch')
        if (source,target) in seen_links:raise ValueError('Duplicate native tree link')
        seen_links.add((source,target))
        if source not in by_key or target not in by_key:
            flag('BROKEN_LINK',source,{'original_link':link});continue
        if index[target[2]]<=index[source[2]]:flag('NONFORWARD_OR_CYCLIC_LINK',source,{'original_link':link});continue
        outgoing[source].append(target);incoming[target].append(source)
        a,b=by_key[source],by_key[target]
        if index[target[2]]-index[source[2]]>1:flag('MISSING_DETECTION_GAP',source,{'descendant':list(target)})
        ma,mb=a['mass_msun'],b['mass_msun']
        ratio=max(ma,mb)/min(ma,mb) if min(ma,mb)>0 else None
        if a.get('mass_definition')!=b.get('mass_definition'):flag('MASS_DEFINITION_CHANGE',source,{'original_mass_definitions':[a.get('mass_definition'),b.get('mass_definition')]})
        elif ratio is None or ratio>maximum_mass_ratio:flag('MASS_JUMP',source,{'descendant':list(target),'mass_ratio':ratio,'original_masses_msun':[ma,mb]})
        ba=a.get('box_proper_kpc',box_proper_kpc);bb=b.get('box_proper_kpc',box_proper_kpc)
        if not all(math.isfinite(x) and x>0 for x in (ba,bb)):raise ValueError('Invalid per-epoch proper box')
        displacement=[((x/ba-y/bb+.5)%1-.5)*bb for x,y in zip(a['position_proper_kpc'],b['position_proper_kpc'])]
        distance=math.sqrt(sum(x*x for x in displacement))
        if distance>maximum_position_jump_proper_kpc:flag('POSITION_JUMP',source,{'minimum_image_distance_proper_kpc':distance})
        if a.get('central') is not None and b.get('central') is not None and a['central']!=b['central']:flag('CENTRAL_SATELLITE_FLIP',source,{'descendant':list(target)})
        if a.get('host_id')!=b.get('host_id'):flag('HOST_CHANGE',source,{'original_host_ids':[a.get('host_id'),b.get('host_id')]})
        if a.get('core_particle_id') is not None and b.get('core_particle_id') is not None and a['core_particle_id']!=b['core_particle_id']:
            exact_uint64_ids([a['core_particle_id'],b['core_particle_id']]);flag('CORE_IDENTITY_CHANGE',source,{'descendant':list(target),'core_ids':[a['core_particle_id'],b['core_particle_id']]})
    for key,node in by_key.items():
        box=node.get('box_proper_kpc',box_proper_kpc)
        if not math.isfinite(box) or box<=0:raise ValueError('Invalid per-epoch proper box')
        if any(x<0 or x>=box for x in node['position_proper_kpc']):flag('PERIODIC_COORDINATE_OUTSIDE_BOX',key,{'original_position_proper_kpc':node['position_proper_kpc'],'box_proper_kpc':box})
        if node.get('phantom'):flag('PHANTOM_PREDICTION',key,{'detection':'predicted by native tree, not a measured halo'})
        if len(outgoing[key])>1:flag('FRAGMENTATION',key,{'native_descendants':[list(k) for k in outgoing[key]]})
        if not outgoing[key] and key[2]!=epochs[-1]:flag('DISAPPEARANCE',key,{'cause':'unresolved: merger, disruption, selection or detection failure'})
        if node.get('coarse_mass_fraction',0)>0:flag('CONTAMINATION',key,{'coarse_mass_fraction':node['coarse_mass_fraction']})
        if not incoming[key] and key[2] not in (epochs[0],epochs[-1]):
            trail=[key];current=key
            while len(outgoing[current])==1 and len(trail)<=transient_maximum_detections:
                current=outgoing[current][0];trail.append(current)
            if not outgoing[current] and current[2]!=epochs[-1] and len(trail)<=transient_maximum_detections:
                flag('TRANSIENT_DETECTION',key,{'detections':len(trail),'native_observations':[list(k) for k in trail]})
    # Reciprocal core exchanges provide an explicit swap diagnostic rather than
    # replacing either branch's mass or forcing a repaired link.
    valid_links=[(s,t) for s,targets in outgoing.items() for t in targets]
    if len(valid_links)*(len(valid_links)-1)//2>maximum_swap_pairs:raise MemoryError('Tree swap pair budget exceeded; partition diagnostic series')
    for i,(s,t) in enumerate(valid_links):
        for u,v in valid_links[i+1:]:
            a,b,c,d=[by_key[k] for k in (s,t,u,v)]
            if s[2]==u[2] and t[2]==v[2] and a.get('core_particle_id') is not None and c.get('core_particle_id') is not None and a['core_particle_id']!=c['core_particle_id'] and a['core_particle_id']==d.get('core_particle_id') and c['core_particle_id']==b.get('core_particle_id'):
                flag('RECIPROCAL_CORE_SWAP',s,{'original_links':[[list(s),list(t)],[list(u),list(v)]]})
    return {'schema':'tree-integrity-v1','status':'COMPLETE','original_nodes':nodes,'original_links':links,'flags':flags,
        'science_acceptance':'NOT_ACCEPTED','repair_policy':'Any proposed repaired links are a separate product; original links remain authoritative evidence'}


def publish_tree_repair_proposal(destination, original_audit, proposed_links, *, rationale, thresholds,
                                 request_source_hashes=None):
    """Publish reviewable alternative links alongside untouched native evidence."""
    from pathlib import Path
    import numpy as np
    from core.foundation import file_sha256
    from core.data import verify_numeric_audit,export_numeric_audit
    original_audit=Path(original_audit).resolve();original=verify_numeric_audit(original_audit)
    if original.get('schema')!='tree-integrity-v1' or not isinstance(rationale,str) or not rationale.strip():raise ValueError('Native original audit and explicit repair rationale required')
    # Diagnose the proposal independently; never mutate the measured product.
    diagnostic=detect_disappearance_swap_jump_fragmentation(original['original_nodes'],proposed_links,**{**thresholds,'link_origin':'proposed_repair'})
    payload={'schema':'tree-repair-proposal-v1','status':'PROPOSED_REQUIRES_REVIEW','original_nodes':original['original_nodes'],
        'original_links':original['original_links'],'proposed_links':proposed_links,'proposed_flags':diagnostic['flags'],
        'rationale':rationale,'science_acceptance':'NOT_ACCEPTED'}
    # Retain the complete verified original payload and its physical sources,
    # so changing an original array cannot leave the proposal apparently valid.
    original_data = original_audit.with_name(original_audit.name+'.data')
    node_contract = original['numeric_data']['array_contract']['datasets']['node_ids']
    sources = dict(__import__('json').loads(node_contract['provenance_json']).get('sources', {}))
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    for asset in [original_audit, original_audit.with_name(original_audit.name+'.sha256'),
                  original_data/'arrays.npz', original_data/'contract.json', original_data/'contract.sha256']:
        sources[str(asset)] = file_sha256(asset)
    for name, checksum in (request_source_hashes or {}).items():
        if file_sha256(Path(name)) != checksum:
            raise ValueError('Repair request source changed: '+name)
        if name in sources and sources[name] != checksum:
            raise ValueError('Conflicting repair source binding: '+name)
        sources[name] = checksum
    payload['source_hashes'] = sources
    payload['original_audit'] = str(original_audit)
    return export_numeric_audit(destination,payload,{'node_ids':np.array([n['id'] for n in original['original_nodes']],dtype=np.uint64)},
        {'node_ids':{'units':'identity','quantity':'original epoch-qualified node IDs','identity_namespace':'tree:epoch-qualified-native'}},
        algorithm=__file__,context={'epoch':__import__('json').loads(original['numeric_data']['array_contract']['datasets']['node_ids']['epoch_json'])},
        source_hashes=sources)


def verify_tree_repair_proposal(path):
    from core.data import verify_numeric_audit
    from core.foundation import file_sha256
    proposal = verify_numeric_audit(path)
    if proposal.get('schema') != 'tree-repair-proposal-v1':
        raise ValueError('Expected a separate tree repair proposal')
    for name, checksum in proposal['source_hashes'].items():
        if file_sha256(name) != checksum:
            raise ValueError('Tree repair proposal source changed: '+name)
    original = verify_numeric_audit(proposal['original_audit'])
    if original['original_nodes'] != proposal['original_nodes'] or original['original_links'] != proposal['original_links']:
        raise ValueError('Tree repair proposal changed original evidence')
    return proposal

# ========================================================================
# identity_history: Exact intake of existing sealed stellar histories, with missingness retained.
from core.data import verify_hdf5_dataset_payloads


def add_stellar_history(graph, path, *, particle_namespace, owner_namespace, maximum_bytes):
    path=Path(path).resolve()
    if type(maximum_bytes) is not int or maximum_bytes<1:raise ValueError('Declare a positive history workspace')
    checksum=file_sha256(path)
    sources={str(path):checksum,str(Path(__file__).resolve()):file_sha256(__file__)}
    verify_hdf5_dataset_payloads(path)
    with h5py.File(path) as f:
        if f['ParticleIDs'].size*128+64*1024**2>maximum_bytes:raise MemoryError('History graph workspace exceeded')
        ids=exact_uint64_ids(f['ParticleIDs'][:]);previous={}
        for name,group in sorted(f['Epochs'].items(),key=lambda item:json.loads(item[1].attrs['metadata_json'])['scale_factor']):
            epoch=int(name);meta=json.loads(group.attrs['metadata_json']);graph.add_epoch(epoch,{**meta['epoch_contract'],'snapshot_storage_index':epoch})
            for source,h in meta['source_sha256'].items():
                if file_sha256(source)!=h:raise ValueError('History native source changed')
                sources[source]=h
            present=group['Present'][:];rows=group['BoundSubhaloRow'][:];masses=group['Masses'][:]
            if any(len(x)!=len(ids) for x in (present,rows,masses)):raise ValueError('History identity vectors do not align')
            for i,pid in enumerate(ids):
                row=int(rows[i]);status='MISSING' if not bool(present[i]) else 'UNBOUND' if row==-1 else 'DETECTED'
                if (not present[i] and row!=-2) or (present[i] and row<-1 and row!=-3):raise ValueError('History missingness/membership sentinel mismatch')
                key=(particle_namespace,int(pid),epoch)
                if key in graph.nodes:raise ValueError('Duplicate history observation; choose explicit species intake once')
                graph.add_node(*key,kind='ssp',status=status,attributes={'mass_msun':float(masses[i]) if present[i] else None,
                    'birth_parent_status':'UNAVAILABLE_IN_THIS_NATIVE_HISTORY','bound_membership_status':'UNAVAILABLE' if row==-3 else 'MEASURED'})
                if row>=0:
                    owner=(owner_namespace,row,epoch)
                    if owner not in graph.nodes:graph.add_node(*owner,kind='subhalo')
                    graph.add_edge(key,owner,'bound_owner',evidence={'origin':'native_membership','history_sha256':checksum})
                if present[i]:
                    if int(pid) in previous:graph.add_edge(previous[int(pid)],key,'descendant',evidence={'origin':'native_ancestry','basis':'exact SSP identity continuity','history_sha256':checksum})
                    previous[int(pid)]=key
    if file_sha256(path)!=checksum:raise ValueError('History changed during graph intake')
    return sources


def add_native_ssp_birth_audit(graph, path, *, particle_namespace, maximum_bytes):
    """Attach frozen native parent IDs to selected, observed SSP graph nodes.

    A parent edge needs an existing gas observation at or before birth. If no
    such observation is selected, retain the measured parent ID as an attribute
    and leave the indexed parent query unavailable. No gas row is invented.
    """
    from stars import verify_native_ssp_birth_audit
    path=Path(path).resolve()
    if type(maximum_bytes) is not int or maximum_bytes<1:
        raise ValueError('Declare a positive SSP lineage workspace')
    raw=json.loads(path.read_text())
    count=raw.get('particle_count')
    if type(count) is not int or count<0 or count*512+len(graph.nodes)*192+16*1024**2>maximum_bytes:
        raise MemoryError('SSP lineage workspace exceeded')
    checksum=file_sha256(path)
    report=verify_native_ssp_birth_audit(path)
    if particle_namespace!=report['provenance']['identity_namespaces']['native_particle']:
        raise ValueError('SSP lineage must preserve the audited native particle namespace')
    epoch=report['request']['snapshot']
    if epoch not in graph.epochs or any(graph.epochs[epoch].get(k)!=v for k,v in report['epoch'].items()):
        raise ValueError('SSP lineage epoch differs from graph observation')
    payload=path.parent/report['numeric_data']['directory']/'arrays.npz'
    sources={**report['source_hashes'],str(path):checksum,str(payload):file_sha256(payload),
        str(Path(__file__).resolve()):file_sha256(__file__)}
    updates=[];links=[]
    parents={}
    for key,node in graph.nodes.items():
        if key[0]==particle_namespace and node['kind'] in {'gas','parent_gas'} and node['status'] in {'DETECTED','UNBOUND'}:
            clock=graph.epochs[key[2]]
            if clock['canonical_time']=='scale_factor':
                a=clock.get('actual_scale_factor')
                if isinstance(a,bool) or not isinstance(a,(int,float)) or not math.isfinite(a) or a<=0:
                    raise ValueError('SSP parent observation needs a finite positive actual epoch')
                parents.setdefault(key[1],[]).append((clock['actual_scale_factor'],key))
    with np.load(payload,allow_pickle=False) as arrays:
        if any(arrays[name].shape!=(count,) for name in ('ParticleIDs','birth_supported','SSPParentGasID','SSPBirthNativeTime')):
            raise ValueError('SSP lineage census count differs from its sealed arrays')
        for i,pid in enumerate(arrays['ParticleIDs']):
            key=(particle_namespace,int(pid),epoch)
            if key not in graph.nodes:continue  # Ownership graph is a selected population.
            node=graph.nodes[key]
            if node['kind']!='ssp' or node['status'] not in {'DETECTED','UNBOUND'}:
                raise ValueError('Native SSP birth requires an observed SSP endpoint')
            link=None
            if not arrays['birth_supported'][i]:
                attrs={'birth_parent_status':'LEGACY_BIRTH_UNAVAILABLE','ssp_birth_audit_sha256':checksum}
            else:
                parent=int(arrays['SSPParentGasID'][i]);birth=float(arrays['SSPBirthNativeTime'][i])
                candidates=[(a,k) for a,k in parents.get(parent,[]) if a<=birth]
                attrs={'native_birth_parent_id':parent,'native_birth_scale_factor':birth,
                    'native_birth_mode':int(arrays['SSPBirthMode'][i]),
                    'birth_parent_status':'MEASURED' if candidates else 'PARENT_ID_MEASURED_OBSERVATION_UNAVAILABLE',
                    'ssp_birth_audit_sha256':checksum}
                if candidates:
                    parent_key=max(candidates,key=lambda item:item[0])[1]
                    link={'source':parent_key,'target':key,'kind':'birth_parent',
                        'evidence':{'origin':'native_ancestry','basis':'frozen native SSPParentGasID and SSPBirthStateVersion=1',
                            'birth_scale_factor':birth,'audit_sha256':checksum,
                            'source_origin':report['source_origin'],
                            'parent_observation_scope':'latest selected gas observation at or before birth'}}
            if 'ssp_birth_audit_sha256' in node['attributes']:
                if any(node['attributes'].get(k)!=v for k,v in attrs.items()):
                    raise ValueError('Conflicting native SSP birth evidence')
                continue
            updates.append((key,attrs))
            if link is not None:links.append(link)
    for source,expected in sources.items():
        if file_sha256(source)!=expected:raise ValueError('SSP birth source changed during lineage intake')
    # Edge validation and cycle checks roll back as a batch; attributes are
    # applied only after every link has passed.
    graph.add_ancestry_edges(links)
    for key,attrs in updates:graph.nodes[key]['attributes'].update(attrs)
    return sources


def add_consistent_tree_history(graph, run_dir, **intake):
    """Use native tree IDs; predicted phantom rows never become detections."""
    from halos import consistent_tree_series
    from core.data import run_epoch_contract
    imported=consistent_tree_series(run_dir,**intake)
    for scale,epoch in intake['epochs_by_scale_factor'].items():
        graph.add_epoch(epoch,{**run_epoch_contract(run_dir,epoch,float(scale)),'snapshot_storage_index':epoch})
    for node in imported['nodes']:
        graph.add_node(node['namespace'],node['id'],node['epoch'],kind='tree',status='MISSING' if node['phantom'] else 'DETECTED',
            attributes={k:v for k,v in node.items() if k not in {'namespace','id','epoch'}})
    for link in imported['links']:
        source,target=tuple(link['source']),tuple(link['target'])
        unavailable=source not in graph.nodes or target not in graph.nodes or any(graph.nodes[k]['status']=='MISSING' for k in (source,target) if k in graph.nodes)
        if source in graph.nodes:
            graph.nodes[source]['attributes'].setdefault('native_descendant_links',[]).append({'target':list(target),'status':'MISSING_OR_PREDICTED_ENDPOINT' if unavailable else 'MEASURED_ENDPOINTS'})
        if unavailable:continue
        graph.add_edge(source,target,'descendant',evidence={'origin':'native_ancestry','basis':'original Consistent Trees link','source_hashes':imported['source_hashes']})
    return imported['source_hashes']

# ========================================================================
# finder_adapters: Normalize real saved native finder measurements through existing contracts.
from core.data import snapshot_id_shards, read_selected_group_ids


def gadget_catalogue(run_dir,snapshot,*,finder,particle_type,particle_namespace,memory_gib,selected_rows):
    """FOF inclusive or SUBFIND bound members from native ordered snapshots.

    Row IDs retain an explicit finder/run namespace and the saved epoch; they
    never become persistent tree IDs. Selecting rows is required to bound cost.
    """
    root=Path(run_dir).resolve()
    if finder not in {'fof','subfind'} or type(particle_type) is not int or particle_type not in range(6):raise ValueError('Choose FOF/SUBFIND and a native particle species')
    if isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive workspace required')
    rows=exact_uint64_ids(selected_rows,allow_empty=False)
    groups=_group_files(root,snapshot);snapshots=_snapshot_files(root,snapshot)
    if not groups or not snapshots:raise FileNotFoundError('Native catalogue and group-ordered snapshots required')
    _check_file_series(groups,'Header','Ngroups_Total' if finder=='fof' else 'Nsubhalos_Total')
    # Gate catalogue allocations using dataset metadata, before reading vectors.
    count=0
    for path in groups:
        with h5py.File(path) as f:
            name=('Group' if finder=='fof' else 'Subhalo')+'/'+('Group' if finder=='fof' else 'Subhalo')+'LenType'
            if name in f:count+=f[name].shape[0]
    if count*1024+64*1024**2>memory_gib*1024**3:raise MemoryError('Finder catalogue metadata exceeds declared workspace')
    sources={str(p.resolve()):file_sha256(p) for p in groups+snapshots}
    _unit_conversion(groups+snapshots)
    from analysis import validate_gadget_snapshot_set
    validate_gadget_snapshot_set(snapshots,snapshot)
    group='Group' if finder=='fof' else 'Subhalo';prefix=group
    lengths=_read_catalogue_series(groups,group,prefix+'LenType');offsets=_read_catalogue_series(groups,group,prefix+'OffsetType')
    n=len(lengths)
    if lengths.shape!=(n,6) or offsets.shape!=lengths.shape or any(x.dtype.kind not in 'iu' or np.any(x<0) for x in (lengths,offsets)) or np.any(rows>=n):raise ValueError('Invalid selected native membership spans')
    total=sum(int(lengths[int(r),particle_type]) for r in rows)
    if total*256+n*1024+64*1024**2>memory_gib*1024**3:raise MemoryError('Selected finder members exceed declared workspace')
    with h5py.File(snapshots[0]) as f:a=float(f['Header'].attrs['Time']);box=float(f['Header'].attrs['BoxSize']);h=float(f['Parameters'].attrs['HubbleParam'])
    for path in groups:
        with h5py.File(path) as f:
            if not np.allclose([float(f['Header'].attrs['Time']),float(f['Header'].attrs['BoxSize']),float(f['Parameters'].attrs['HubbleParam'])],[a,box,h],rtol=1e-12,atol=0):raise ValueError('Catalogue/snapshot frame mismatch')
    position=_read_catalogue_series(groups,group,prefix+'Pos');velocity=_read_catalogue_series(groups,group,prefix+'Vel')
    from core.data import convert_native_catalogue_kinematics
    with h5py.File(snapshots[0]) as source:
        parameters = source['Parameters'].attrs
        velocity = convert_native_catalogue_kinematics(velocity, prefix+'Vel',
            scale_factor=a, hubble_param=h, unit_length_cm=float(parameters['UnitLength_in_cm']),
            unit_velocity_cm_s=float(parameters['UnitVelocity_in_cm_per_s']))
    mass=_read_catalogue_series(groups,group,prefix+'Mass')
    radius=_read_catalogue_series(groups,group,'Group_R_Crit200' if finder=='fof' else 'SubhaloHalfmassRad',required=False)
    vmax=_read_catalogue_series(groups,group,'SubhaloVmax',required=False) if finder=='subfind' else None
    ranks=_read_catalogue_series(groups,group,'SubhaloRankInGr',required=False) if finder=='subfind' else None
    hosts=_read_catalogue_series(groups,group,'SubhaloGroupNr') if finder=='subfind' else None
    shards,_=snapshot_id_shards(snapshots,particle_type);objects=[]
    # Validate all nonempty spans, so subset intake cannot hide corrupt overlaps.
    spans=sorted((int(o[particle_type]),int(o[particle_type])+int(l[particle_type])) for o,l in zip(offsets,lengths) if l[particle_type])
    if any(b[0]<aa[1] for aa,b in zip(spans,spans[1:])):raise ValueError('Native finder membership spans overlap')
    for r in rows:
        r=int(r);ids,_=read_selected_group_ids(offsets,lengths,np.array([r],dtype=np.int64),shards,particle_type=particle_type,max_ids=int(lengths[r,particle_type]))
        objects.append({'id':r,'position_proper_kpc':(position[r]*a/h).tolist(),'velocity_peculiar_kms':velocity[r].tolist(),
            'mass_msun':float(mass[r]*1e10/h),'mass_definition':'FOF inclusive mass' if finder=='fof' else 'SUBFIND bound mass',
            'radius_proper_kpc':None if radius is None else float(radius[r]*a/h),
            'radius_definition':'R200c' if finder=='fof' else 'bound half-mass radius',
            'vmax_kms':None if vmax is None else float(vmax[r]),'central':True if finder=='fof' else None if ranks is None else bool(ranks[r]==0),
            'host_id':None if hosts is None else int(hosts[r]),'members':[int(i) for i in exact_uint64_ids(ids)],
            'quality':(['RADIUS_UNAVAILABLE'] if radius is None else [])+(['VMAX_UNAVAILABLE'] if vmax is None else [])})
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    result={'finder':finder,'namespace':str(root)+':'+finder+':catalogue-row','particle_namespace':particle_namespace,
        'particle_kind':{0:'gas',4:'ssp',5:'bh'}.get(particle_type,'particle'),
        'epoch':{**run_epoch_contract(root,snapshot,a),'snapshot_storage_index':snapshot},'box_proper_kpc':box*a/h,'objects':objects,'source_hashes':sources}
    return validate_catalogue(result)


def rockstar_catalogue(run_dir,snapshot,*,particle_namespace,memory_gib,selected_ids,mass_definition):
    """Pinned native ASCII object table and complete optional particle dumps.

    Missing/downsampled particle dumps remain unavailable; phase-only comparison
    must be explicitly enabled by its caller. Never confuse internal halo IDs
    in particle dumps with the external exact catalogue IDs.
    """
    from analysis import rockstar_host_table,rockstar_catalogue_index
    from core.data import rockstar_directory
    from halos import snapshot_sets
    root=Path(run_dir).resolve();base=rockstar_directory(root);index=rockstar_catalogue_index(root,snapshot)
    ids=exact_uint64_ids(selected_ids,allow_empty=False)
    if isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive workspace required')
    paths=[*base.glob('out_*.list'),*base.glob('host_catalogues/out_*.list'),*base.glob(f'halos_{index}.*.particles')]
    sets=snapshot_sets(root)
    if snapshot not in sets:raise FileNotFoundError('Saved Rockstar particle epoch required')
    paths.extend(sets[snapshot]);sources={str(p.resolve()):file_sha256(p) for p in paths}
    if sum(p.stat().st_size for p in paths)>memory_gib*1024**3/4:raise MemoryError('Rockstar ASCII selection exceeds workspace')
    table=rockstar_host_table(root,snapshot)
    if table is None:raise FileNotFoundError('Native parent-labelled Rockstar catalogue unavailable')
    if mass_definition not in table:raise ValueError('Requested native Rockstar mass definition unavailable')
    available=table['_canonical_identity_links']['id'];native_ids=available['values']
    if not available['available'].all():raise ValueError('Missing Rockstar object ID')
    exact_uint64_ids(native_ids)
    with h5py.File(sets[snapshot][0]) as f:a=float(f['Header'].attrs['Time']);h=float(f['Parameters'].attrs['HubbleParam']);box=float(f['Header'].attrs['BoxSize'])*a/h
    members={int(i):[] for i in ids};expected={};downsampled=set()
    for path in base.glob(f'halos_{index}.*.particles'):
        text=path.read_text();local_ids=set();halo_table=False
        for line in text.splitlines():
            if line.startswith('#Halo table begins here:'):halo_table=True;continue
            if line.startswith('#Particle table begins here:'):halo_table=False;continue
            if halo_table and line.startswith('#'):
                fields=line[1:].split()
                if len(fields)!=20:raise ValueError('Unknown pinned Rockstar halo membership header')
                external,internal,count=map(int,fields[:3])
                if count<0:raise ValueError('Negative native membership count')
                if external in members:
                    if external in expected:raise ValueError('Duplicate native membership halo across chunks')
                    expected[external]=count;local_ids.add(external)
                continue
            if not line.strip() or line.startswith('#'):continue
            fields=line.split()
            if len(fields)!=10:raise ValueError('Unknown pinned Rockstar particle dump format')
            particle,assigned,internal,external=map(int,fields[6:10])
            if particle<0:raise ValueError('Negative Rockstar particle identity')
            if external in members:members[external].append(particle)
        if '#Notes: Only particles' in text:downsampled.update(local_ids)
    objects=[]
    for identity_value in ids:
        found=np.flatnonzero(native_ids==identity_value)
        if len(found)!=1:raise ValueError('Requested Rockstar ID missing or ambiguous')
        r=int(found[0]);pid=int(table['pid'][r]) if 'pid' in table else None
        selected=int(identity_value);complete=selected in expected and selected not in downsampled
        if complete and len(members[selected])!=expected[selected]:raise ValueError('Truncated native Rockstar particle membership')
        objects.append({'id':int(identity_value),'position_proper_kpc':[float(table[k][r])*1000*a/h for k in ('x','y','z')],
            'velocity_peculiar_kms':[float(table[k][r]) for k in ('vx','vy','vz')],
            'mass_msun':float(table[mass_definition][r])/h,'mass_definition':'ROCKSTAR '+mass_definition,
            'radius_proper_kpc':float(table['rvir'][r])*a/h if 'rvir' in table else None,
            'radius_definition':'ROCKSTAR virial radius',
            'vmax_kms':float(table['vmax'][r]) if 'vmax' in table else None,'central':None if pid is None else pid<0,
            'host_id':None if pid is None or pid<0 else pid,'members':members[int(identity_value)] if complete else None,
            'quality':['NO_NATIVE_UNBINDING','NATIVE_ASSIGNED_MEMBERSHIP_BOUND_STATUS_UNAVAILABLE'] if complete else ['EXACT_MEMBERSHIP_UNAVAILABLE']})
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    return validate_catalogue({'finder':'rockstar','namespace':str(root)+':rockstar:native-id','particle_namespace':particle_namespace,
        'epoch':{**run_epoch_contract(root,snapshot,a),'snapshot_storage_index':snapshot},'box_proper_kpc':box,'objects':objects,'source_hashes':sources})

# ========================================================================
# tree_adapters: Import real Consistent Trees hlists without treating native rows as identity.
from analysis import rockstar_table_from_path


def consistent_tree_series(run_dir,*,catalogues,epochs_by_scale_factor,hubble_param,box_comoving_kpc_h,mass_column,memory_gib):
    if not catalogues or isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Declare nonempty native catalogue series and memory')
    paths=[Path(p).resolve() for p in catalogues]
    if sum(p.stat().st_size for p in paths)*16>memory_gib*1024**3:raise MemoryError('Native tree table workspace exceeded')
    if not math.isfinite(hubble_param) or hubble_param<=0 or not math.isfinite(box_comoving_kpc_h) or box_comoving_kpc_h<=0:raise ValueError('Positive declared native h and comoving box required')
    nodes=[];links=[];by_identity={};rows=[];sources={str(p):file_sha256(p) for p in paths}
    from halos import snapshot_sets
    from analysis import validate_gadget_snapshot_set
    saved=snapshot_sets(run_dir);clocks={}
    for scale,epoch in epochs_by_scale_factor.items():
        if type(epoch) is not int or epoch<0 or epoch not in saved:raise ValueError('Tree epoch mapping requires actual saved snapshots')
        meta=validate_gadget_snapshot_set(saved[epoch],epoch)
        if not math.isclose(float(scale),meta['scale_factor'],rel_tol=1e-12):raise ValueError('Tree saved clock mismatch')
        if not math.isclose(hubble_param,meta['hubble_param'],rel_tol=1e-12):raise ValueError('Tree saved Hubble parameter mismatch')
        with h5py.File(saved[epoch][0]) as f:
            if not math.isclose(box_comoving_kpc_h,float(f['Header'].attrs['BoxSize']),rel_tol=1e-12):raise ValueError('Tree saved comoving box mismatch')
        clocks[epoch]=meta['scale_factor']
        sources.update({str(p.resolve()):file_sha256(p) for p in saved[epoch]})
    namespace=str(Path(run_dir).resolve())+':consistent-trees:native-id'
    for path in paths:
        table=rockstar_table_from_path(path)
        if table is None:raise ValueError('Unrecognized or invalid native Consistent Trees header')
        required={'scale','id','desc_scale','desc_id','pid','x','y','z',mass_column}
        if not required<=set(table):raise ValueError('Native Consistent Trees fields missing')
        exact_uint64_ids(table['id'])
        for i,value in enumerate(table['id']):
            a=float(table['scale'][i]);epoch=epochs_by_scale_factor.get(format(a,'.12g'))
            if type(epoch) is not int:raise ValueError('Map every actual tree scale factor to its saved snapshot explicitly')
            key=(namespace,int(value),epoch)
            if key in by_identity:raise ValueError('Duplicate tree observation across catalogue files')
            if not math.isclose(clocks[epoch],a,rel_tol=1e-12):raise ValueError('Tree saved clock mismatch')
            node={'namespace':namespace,'id':int(value),'epoch':epoch,'mass_msun':float(table[mass_column][i])/hubble_param,
                'mass_definition':'Consistent Trees '+mass_column,'position_proper_kpc':[float(table[k][i])*1000*a/hubble_param for k in ('x','y','z')],
                'box_proper_kpc':box_comoving_kpc_h*a/hubble_param,'central':int(table['pid'][i])<0,
                'host_id':None if int(table['pid'][i])<0 else int(table['pid'][i]),'phantom':bool(table['phantom'][i]) if 'phantom' in table else False}
            # Pinned Consistent Trees assemble_halo_trees.c declares VX/Y/Z
            # physical peculiar km/s and Orig_halo_ID the original finder ID.
            # Missing columns remain unavailable; no finite differences or row
            # equality substitute for these native measurements/identities.
            if all(k in table for k in ('vx','vy','vz')):
                velocity=[float(table[k][i]) for k in ('vx','vy','vz')]
                if not np.isfinite(velocity).all():raise ValueError('Nonfinite native tree peculiar velocity')
                node['velocity_peculiar_kms']=velocity
            if 'orig_halo_id' in table:
                original=int(table['orig_halo_id'][i])
                if original>=0:
                    exact_uint64_ids([original]);node['original_finder_key']=[str(Path(run_dir).resolve())+':rockstar:native-id',original,epoch]
            by_identity[key]=node;nodes.append(node);rows.append((key,float(table['desc_scale'][i]),int(table['desc_id'][i])))
    for source,scale,desc in rows:
        if desc<0:continue
        epoch=epochs_by_scale_factor.get(format(scale,'.12g'))
        if epoch is None:
            links.append({'source':list(source),'target':[namespace,desc,-1],'origin':'native_tree','unmapped_descendant_scale_factor':scale})
        else:links.append({'source':list(source),'target':[namespace,desc,epoch],'origin':'native_tree'})
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native tree input changed')
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    units_source=Path(__file__).resolve().parent/'software/consistent-trees/src/assemble_halo_trees.c'
    sources[str(units_source)]=file_sha256(units_source)
    return {'nodes':nodes,'links':links,'source_hashes':sources,'namespace':namespace}
from core.data import write_dataset_contract
from core.cosmology import native_source_manifest,verify_native_source_manifest


def read_velociraptor_catalogue(directory):
    root=Path(directory).resolve();report=json.loads((root/'report.json').read_text())
    if file_sha256(root/'report.json')!=(root/'report.sha256').read_text().strip():raise ValueError('VELOCIraptor report seal changed')
    for name,h in report['artifacts'].items():
        if file_sha256(root/name)!=h:raise ValueError('VELOCIraptor native output changed')
    invocation=json.loads((root/'invocation.json').read_text())
    if invocation.get('status')!='RETURNED' or invocation.get('returncode')!=0:raise ValueError('VELOCIraptor native invocation did not succeed')
    for original,h in invocation['input_hashes'].items():
        relocated=report['published_input_paths'].get(original)
        path=root/relocated if relocated is not None else Path(original)
        if file_sha256(path)!=h:raise ValueError('VELOCIraptor executed input changed')
    verify_native_source_manifest(report['native_sources'])
    return validate_catalogue(report['catalogue'])


@native_run_owner
@directory_owner('destination')
def run_velociraptor(run_dir,destination,*,snapshot,particle_namespace,controls,memory_gib,threads,wall_hours):
    from halos import snapshot_sets
    from analysis import validate_gadget_snapshot_set
    from halos import _unit_conversion
    from core.foundation import G_MPC_KMS2_MSUN
    required={'Minimum_size','Nsearch_physical','Nsearch_velocity','Cell_fraction','Halo_3D_linking_length','Substructure_physical_linking_length','Search_for_substructure','Unbind_flag'}
    if set(controls)!=required:raise ValueError('Declare frozen native DM search controls')
    for key in ('Minimum_size','Nsearch_physical','Nsearch_velocity'):
        if type(controls[key]) is not int or controls[key]<2:raise ValueError('Positive native minimum/neighbor counts required')
    for key in ('Cell_fraction','Halo_3D_linking_length','Substructure_physical_linking_length'):
        if isinstance(controls[key],bool) or not math.isfinite(controls[key]) or controls[key]<=0:raise ValueError('Positive finite native linking/cell parameters required')
    if any(type(controls[k]) is not int or controls[k] not in (0,1) for k in ('Search_for_substructure','Unbind_flag')):raise ValueError('Declare native search/unbinding flags')
    if type(threads) is not int or not 1<=threads<=cpu_thread_budget():raise ValueError('Threads exceed available CPU budget')
    if any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in (memory_gib,wall_hours)):raise ValueError('Positive native workspace/time limits required')
    root=Path(run_dir).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    binary=native_engine_directory("velociraptor") / "build/stf"
    if not binary.is_file():raise FileNotFoundError('Build pinned VELOCIraptor via commands.py bootstrap --build-velociraptor')
    paths=snapshot_sets(root).get(snapshot)
    if not paths:raise FileNotFoundError('Complete saved native snapshot required')
    meta=validate_gadget_snapshot_set(paths,snapshot);_unit_conversion(paths)
    n=int(meta['particle_counts_by_type'][1])
    if n*1024+64*1024**2>memory_gib*1024**3:raise MemoryError('Native DM transport exceeds declared workspace')
    if n<=max(controls['Nsearch_physical'],controls['Nsearch_velocity']) or n<controls['Minimum_size']:raise ValueError('DM selection smaller than native requested neighbor/minimum pool')
    a=meta['scale_factor'];h=meta['hubble_param'];epoch={**run_epoch_contract(root,snapshot,a),'snapshot_storage_index':snapshot}
    sources={str(p.resolve()):file_sha256(p) for p in paths+[binary,Path(__file__).resolve()]}
    for name in ('data.py','foundation.py','foundation.py','foundation.py','cosmology.py','halos.py','halos.py','analysis.py','foundation.py','foundation.py'):
        helper=Path(__file__).with_name(name).resolve();sources[str(helper)]=file_sha256(helper)
    native_sources=native_source_manifest('velociraptor',binary.parent.parent)
    position=[];velocity=[];mass=[];ids=[];box=None;gravity=None
    for p in paths:
        with h5py.File(p) as f:
            box=float(f['Header'].attrs['BoxSize'])*a/h
            override=float(f['Parameters'].attrs.get('GravityConstantInternal',0.))
            current=G_MPC_KMS2_MSUN*1000 if override==0 else override/1e10
            if gravity is not None and current!=gravity:raise ValueError('Native snapshot shard gravity mismatch')
            gravity=current;group=f.get('PartType1')
            if group is None:continue
            position.append(group['Coordinates'][:]*native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h))
            velocity.append(group['Velocities'][:]*native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h))
            values=group['Masses'][:] if 'Masses' in group else np.full(len(group['ParticleIDs']),f['Header'].attrs['MassTable'][1])
            mass.append(values*native_unit_factor('mass_msun',scale_factor=a,hubble_param=h));ids.append(group['ParticleIDs'][:])
    position=np.concatenate(position);velocity=np.concatenate(velocity);mass=np.concatenate(mass);ids=exact_uint64_ids(np.concatenate(ids))
    destination.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.velociraptor-',dir=destination.parent))
    try:
        with h5py.File(stage/'particles.hdf5','w') as f:
            counts=np.array([0,n,0,0,0,0],dtype=np.uint32)
            f.create_group('Header').attrs.update(BoxSize=box,Time=1.,Redshift=0.,HubbleParam=1.,Omega0=0.,OmegaLambda=0.,NumFilesPerSnapshot=1,
                NumPart_Total=counts,NumPart_ThisFile=counts,NumPart_Total_HighWord=np.zeros(6,dtype=np.uint32),MassTable=np.zeros(6))
            group=f.create_group('PartType1')
            for name,values,units in [('Coordinates',position,'proper kpc'),('Velocities',velocity,'km/s'),('Masses',mass,'Msun'),('ParticleIDs',np.arange(1,n+1,dtype=np.uint64),'identity')]:
                dataset=group.create_dataset(name,data=values)
                write_dataset_contract(dataset,units=units,quantity='DM finder transport '+name,epoch=epoch,
                    identity_namespace=particle_namespace+':dense-map' if name=='ParticleIDs' else None,provenance={'source_hashes':sources,'header':'Time=1/HubbleParam=1 transport only'})
            dataset=f.create_dataset('DenseIDMap/OriginalParticleIDs',data=ids)
            write_dataset_contract(dataset,units='identity',quantity='reversible original DM IDs',epoch=epoch,identity_namespace=particle_namespace,provenance={'sources':sources})
        native={**controls,'HDF_name_convention':1,'Input_includes_dm_particle':1,'Input_includes_gas_particle':0,'Input_includes_star_particle':0,
            'Input_includes_bh_particle':0,'Cosmological_input':0,'h_val':1,'Particle_search_type':2,'Singlehalo_search':0,'FoF_search_type':1,
            'Length_input_unit_conversion_to_output_unit':1,'Velocity_input_unit_conversion_to_output_unit':1,'Mass_input_unit_conversion_to_output_unit':1,
            'Length_unit_to_kpc':1,'Velocity_to_kms':1,'Mass_to_solarmass':1,'Comoving_units':0,'Period':box,'Gravity':gravity,
            'Binary_output':2,'Separate_output_files':0,'Calculate_radial_profiles':0,'Calculate_aperture_quantities':0,'Verbose':0}
        atomic_text(stage/'finder.cfg','\n'.join(str(k)+'='+str(v) for k,v in native.items())+'\n')
        with (stage/'native.log').open('w') as log:
            run_native_command([str(binary),'-I','2','-i',str(stage/'particles'),'-C',str(stage/'finder.cfg'),'-o',str(stage/'catalogue')],
                cwd=stage,receipt_path=stage/'invocation.json',inputs=[*sources,stage/'particles.hdf5',stage/'finder.cfg'],
                env={**os.environ,'OMP_NUM_THREADS':str(threads)},stdout=log,stderr=__import__('subprocess').STDOUT,timeout=wall_hours*3600)
        objects=[]
        properties=stage/'catalogue.properties';groups=stage/'catalogue.catalog_groups'
        if not properties.exists():
            if 'STF_EMPTY_SELECTION' not in (stage/'native.log').read_text():raise ValueError('Native finder returned without a complete catalogue')
        else:
            with h5py.File(properties) as f,h5py.File(groups) as g,h5py.File(stage/'catalogue.catalog_particles') as particles:
                external=exact_uint64_ids(f['ID'][:]);offsets=g['Offset'][:];dense=exact_uint64_ids(particles['Particle_IDs'][:])
                if len(offsets)!=len(external) or (len(offsets) and (int(offsets[0])!=0 or np.any(np.diff(offsets.astype(np.int64))<0))):raise ValueError('Invalid native bound membership spans')
                if np.any(dense<1) or np.any(dense>n):raise ValueError('Native membership outside reversible dense map')
                ends=np.r_[offsets[1:],len(dense)];namespace=str(root)+':velociraptor:native-id'
                for i,value in enumerate(external):
                    selected=dense[int(offsets[i]):int(ends[i])];original=exact_uint64_ids(ids[selected.astype(np.int64)-1])
                    host=int(f['hostHaloID'][i])
                    objects.append({'id':int(value),'position_proper_kpc':[float(f[k][i]) for k in ('Xc','Yc','Zc')],
                        'velocity_peculiar_kms':[float(f[k][i]) for k in ('VXc','VYc','VZc')],'mass_msun':float(f['Mass_tot'][i]),
                        'mass_definition':'VELOCIraptor DM-only measured mass; bound members' if controls['Unbind_flag'] else 'VELOCIraptor DM-only selected mass; no unbinding',
                        'radius_proper_kpc':float(f['R_HalfMass'][i]),'radius_definition':'bound half-mass radius','vmax_kms':float(f['Vmax'][i]),'central':host<0,'host_id':None if host<0 else host,
                        'members':[int(x) for x in original],'quality':[] if controls['Unbind_flag'] else ['NO_NATIVE_UNBINDING']})
        sources[str(Path(__file__).with_name('halos.py'))]=file_sha256(Path(__file__).with_name('halos.py'))
        catalogue={'finder':'velociraptor','namespace':str(root)+':velociraptor:native-id','particle_namespace':particle_namespace,
            'epoch':epoch,'box_proper_kpc':box,'objects':objects,'source_hashes':sources}
        validate_catalogue(catalogue)
        artifacts={p.relative_to(stage).as_posix():file_sha256(p) for p in stage.rglob('*') if p.is_file()}
        verify_native_source_manifest(native_sources)
        invocation=json.loads((stage/'invocation.json').read_text())
        relocated={p:str(Path(p).relative_to(stage)) for p in invocation['input_hashes'] if Path(p).is_relative_to(stage)}
        report={'schema':'native-velociraptor-finder-v1','status':'COMPLETE','catalogue':catalogue,'controls':controls,'artifacts':artifacts,'native_sources':native_sources,
            'published_input_paths':relocated,'build_equivalence':'Executable and source bytes bound; compiler/source equivalence requires build record',
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(stage/'report.json',report);atomic_text(stage/'report.sha256',file_sha256(stage/'report.json')+'\n')
        publish_directory(stage,destination);return report
    except BaseException as exc:
        atomic_json(stage/'failure.json',{'status':'FAIL','reason':str(exc),'science_acceptance':'NOT_ACCEPTED'});raise

# ========================================================================
# halo_shape_profiles: Fixed-centre radial density shapes, with explicit window/estimator definitions.

NATIVE_SHAPE_FIELDS = {'snapshot','host_row','output_directory','radial_edges_kpc','support_radius_kpc',
    'window','estimator','minimum_particles','tolerance','maximum_iterations','minimum_axis_ratio',
    'bootstrap_realizations','seed','minimum_bootstrap_success_fraction','minimum_eigenvalue_gap',
    'maximum_orientation_error_deg'}


def validate_native_shape_request(request):
    """Validate the complete recipe before native reads or expensive evolution."""
    from pathlib import Path
    if (not isinstance(request,dict) or NATIVE_SHAPE_FIELDS-set(request)
            or set(request)-NATIVE_SHAPE_FIELDS-{'memory_gib','parent_selection','expected_scale_factor'}):
        raise ValueError('Radial shape request requires exactly '+', '.join(sorted(NATIVE_SHAPE_FIELDS))+'; optional memory_gib')
    result=dict(request)
    if result.get('parent_selection', 'complete_support_sphere') not in ('complete_support_sphere','fof_members','subfind_bound_associated','subfind_central_bound'):
        raise ValueError('parent_selection must be complete_support_sphere, fof_members, subfind_bound_associated or subfind_central_bound; none proves global host binding')
    if 'expected_scale_factor' in result:
        value=result['expected_scale_factor']
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0.:
            raise ValueError('expected_scale_factor must be finite and positive')
        result['expected_scale_factor']=float(value)
    for name,minimum in (('snapshot',0),('host_row',0),('minimum_particles',3),('maximum_iterations',1),
                         ('bootstrap_realizations',0),('seed',0)):
        result[name]=_integer(result[name],name,minimum)
    if 0<result['bootstrap_realizations']<32:
        raise ValueError('bootstrap requires at least 32 realizations or explicit zero')
    if result['window'] not in ('spherical_shell','iterative_ellipsoidal_shell') or result['estimator'] not in ('mass_quadrupole','reduced_mass_quadrupole'):
        raise ValueError('declare supported radial shape window and estimator')
    for name in ('support_radius_kpc','tolerance','memory_gib'):
        value=result.get(name,1.)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0.:
            raise ValueError(name+' must be finite and positive')
        result[name]=float(value)
    for name,upper in (('minimum_axis_ratio',1.),('minimum_bootstrap_success_fraction',1.),
                       ('minimum_eigenvalue_gap',1.),('maximum_orientation_error_deg',90.)):
        value=result[name]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or not 0.<value<=upper:
            raise ValueError(name+f' must lie in (0,{upper}]')
        result[name]=float(value)
    edges=np.asarray(result['radial_edges_kpc'],dtype=float)
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]<0. or np.any(np.diff(edges)<=0.):
        raise ValueError('radial_edges_kpc must be increasing finite nonnegative edges')
    if edges[-1]>result['support_radius_kpc']:
        raise ValueError('complete support must cover the outer radial edge')
    result['radial_edges_kpc']=edges.tolist()
    if not isinstance(result['output_directory'],(str,Path)) or not str(result['output_directory']).strip():
        raise ValueError('output_directory must be a nonempty path')
    result['output_directory']=str(Path(result['output_directory']).expanduser().resolve())
    history_workspace=(1+result['bootstrap_realizations'])*(len(edges)-1)*result['maximum_iterations']*512
    if history_workspace+64*1024**2>result['memory_gib']*1024**3:
        raise MemoryError('shape bootstrap/iteration history exceeds declared workspace')
    return result


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _axes(tensor):
    values, vectors = np.linalg.eigh(tensor)
    values, vectors = values[::-1], vectors[:, ::-1]
    if not np.isfinite(values).all() or values[-1] <= 32.*np.finfo(float).eps*values[0]:
        return None
    # Eigenvector sign is physically arbitrary; fix a right-handed convention.
    for j in (0, 1):
        if vectors[np.argmax(abs(vectors[:, j])), j] < 0.:
            vectors[:, j] *= -1.
    vectors[:, 2] = np.cross(vectors[:, 0], vectors[:, 1])
    return values, vectors, np.sqrt(values[1] / values[0]), np.sqrt(values[2] / values[0])


def _metric(q, s, axes):
    return (q*s)**(2./3.) * (axes * [1., 1./q**2, 1./s**2]) @ axes.T


def _shell(offsets, masses, lower, upper, *, window, estimator, support_radius,
           minimum_particles, tolerance, maximum_iterations, minimum_axis_ratio, initial_geometry=None):
    q, s, axes = 1., 1., np.eye(3)
    if initial_geometry is not None and window != 'spherical_shell':
        q, s, axes = initial_geometry
    previous_members = None
    history = []
    for iteration in range(1, maximum_iterations + 1):
        major = upper / (q*s)**(1./3.)
        if major > support_radius:
            return {'status': 'OUTSIDE_COMPLETE_SUPPORT', 'iterations': history,
                    'required_support_radius_kpc': float(major)}
        metric = np.eye(3) if window == 'spherical_shell' else _metric(q, s, axes)
        radius2 = np.einsum('ni,ij,nj->n', offsets, metric, offsets)
        members = (radius2 >= lower**2) & (radius2 < upper**2)
        if estimator == 'reduced_mass_quadrupole':
            members &= radius2 > 0.
        count = int(members.sum())
        census={'particle_count':count,'mass_msun':float(masses[members].sum())}
        if count < minimum_particles:
            return {'status': 'INSUFFICIENT_PARTICLES', **census, 'iterations': history}
        x, mass = offsets[members], masses[members]
        normalized_mass = mass / mass.max()
        effective_count = float(normalized_mass.sum()**2 / (normalized_mass @ normalized_mass))
        if effective_count < minimum_particles:
            return {'status': 'INSUFFICIENT_EFFECTIVE_MASS_COUNT', **census,
                    'effective_mass_count': effective_count, 'iterations': history}
        weight = mass / radius2[members] if estimator == 'reduced_mass_quadrupole' else mass
        if not np.isfinite(weight).all() or not np.isfinite(weight.sum()) or weight.sum() <= 0.:
            return {'status': 'INVALID_NUMERICAL_WEIGHTS', **census, 'iterations': history}
        weight = weight / weight.sum()
        tensor = (x.T * weight) @ x
        if not np.isfinite(tensor).all():
            return {'status': 'INVALID_NUMERICAL_TENSOR', **census, 'iterations': history}
        eigensystem = _axes(tensor)
        if eigensystem is None:
            return {'status': 'DEGENERATE_TENSOR', **census, 'iterations': history}
        values, new_axes, new_q, new_s = eigensystem
        if new_s < minimum_axis_ratio:
            return {'status': 'BELOW_DECLARED_AXIS_RATIO_DOMAIN', **census,
                    'b_over_a': float(new_q), 'c_over_a': float(new_s), 'iterations': history}
        new_metric = _metric(new_q, new_s, new_axes)
        change = float(np.linalg.norm(new_metric-metric) / np.linalg.norm(metric))
        membership_stable = previous_members is not None and np.array_equal(members, previous_members)
        history.append({'iteration': iteration, 'count': count, 'metric_fractional_change': change,
                        'membership_stable': membership_stable})
        if window == 'spherical_shell' or (change <= tolerance and membership_stable):
            denominator = values[0] - values[2]
            return {'status': 'MEASURED', **census,
                    'effective_mass_count': effective_count,
                    'mass_msun': float(mass.sum()), 'tensor': tensor.tolist(),
                    'eigenvalues': values.tolist(), 'axes_columns': new_axes.tolist(),
                    'b_over_a': float(new_q), 'c_over_a': float(new_s),
                    'triaxiality': float((values[0]-values[1])/denominator) if denominator > 32*np.finfo(float).eps*values[0] else None,
                    'selection_metric': metric.tolist(),
                    'selection_axes_columns': axes.tolist() if window != 'spherical_shell' else np.eye(3).tolist(),
                    'boundary_semiaxes_kpc': (upper*np.array([1.,q,s])/(q*s)**(1./3.)).tolist()
                    if window != 'spherical_shell' else [upper]*3,
                    'fitted_semiaxes_at_upper_radius_kpc': (upper*np.array([1.,new_q,new_s])/(new_q*new_s)**(1./3.)).tolist(),
                    'iterations': history}
        q, s, axes, previous_members = new_q, new_s, new_axes, members
    return {'status': 'NOT_CONVERGED', **census, 'iterations': history}


def _fit_profile(x, mass, edges, controls):
    """Follow the measured inner solution outward; every bootstrap repeats this."""
    results, geometry = [], None
    for lower, upper in zip(edges[:-1], edges[1:]):
        result = _shell(x, mass, lower, upper, initial_geometry=geometry, **controls)
        result.update(lower_radius_kpc=float(lower), upper_radius_kpc=float(upper),
                      initialization='sphere' if geometry is None else 'previous measured inner shell')
        geometry = ((result['b_over_a'], result['c_over_a'], np.asarray(result['axes_columns']))
                    if result['status'] == 'MEASURED' else None)
        results.append(result)
    return results


def radial_density_shapes(offsets_proper_kpc, masses_msun, radial_edges_kpc, *,
                          support_radius_kpc, window, estimator, minimum_particles,
                          tolerance, maximum_iterations, minimum_axis_ratio,
                          bootstrap_realizations, seed, minimum_bootstrap_success_fraction,
                          minimum_eigenvalue_gap, maximum_orientation_error_deg):
    """Measure all declared shells and preserve failed bins rather than omit them.

    Input must contain all members of the declared parent population inside
    support_radius_kpc, relative to the SAME fixed physical centre. Parent
    membership (all particles, FOF, bound host, substructure exclusions) must
    be supplied and recorded by the file adapter. A spherical window gives
    aperture moments, not intrinsic ellipsoidal axis ratios.
    """
    x, mass, edges = map(lambda v: np.asarray(v, dtype=float),
                         (offsets_proper_kpc, masses_msun, radial_edges_kpc))
    if (x.ndim != 2 or x.shape[1] != 3 or mass.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isfinite(mass).all() or np.any(mass <= 0.) or edges.ndim != 1 or len(edges) < 2
            or not np.isfinite(edges).all() or edges[0] < 0. or np.any(np.diff(edges) <= 0.)):
        raise ValueError('finite aligned positions/positive masses and increasing nonnegative radial edges required')
    if not np.isfinite(mass.sum()):
        raise ValueError('total mass overflows the declared numerical units')
    if window not in ('spherical_shell', 'iterative_ellipsoidal_shell') or estimator not in ('mass_quadrupole', 'reduced_mass_quadrupole'):
        raise ValueError('declare a supported shell window and density-moment estimator')
    for name, value in (('support radius', support_radius_kpc), ('tolerance', tolerance)):
        if isinstance(value, (bool,np.bool_)) or not np.isfinite(value) or value <= 0.:
            raise ValueError(f'{name} must be finite and positive')
    if support_radius_kpc < edges[-1]:
        raise ValueError('complete support must cover the outer radial edge')
    for name, value, lower, upper in (('minimum_axis_ratio', minimum_axis_ratio, 0., 1.),
            ('minimum_bootstrap_success_fraction', minimum_bootstrap_success_fraction, 0., 1.),
            ('minimum_eigenvalue_gap', minimum_eigenvalue_gap, 0., 1.),
            ('maximum_orientation_error_deg', maximum_orientation_error_deg, 0., 90.)):
        if isinstance(value, (bool,np.bool_)) or not np.isfinite(value) or not lower < value <= upper:
            raise ValueError(f'{name} must lie in ({lower}, {upper}]')
    minimum_particles = _integer(minimum_particles, 'minimum_particles', 3)
    maximum_iterations = _integer(maximum_iterations, 'maximum_iterations', 1)
    bootstrap_realizations = _integer(bootstrap_realizations, 'bootstrap_realizations', 0)
    seed = _integer(seed, 'seed', 0)
    if 0 < bootstrap_realizations < 32:
        raise ValueError('bootstrap requires at least 32 realizations or explicit zero')
    within = np.linalg.norm(x, axis=1) <= support_radius_kpc
    x, mass = x[within], mass[within]
    controls = dict(window=window, estimator=estimator, support_radius=support_radius_kpc,
                    minimum_particles=minimum_particles, tolerance=tolerance,
                    maximum_iterations=maximum_iterations, minimum_axis_ratio=minimum_axis_ratio)
    rng = np.random.default_rng(seed)
    shells = _fit_profile(x, mass, edges, controls)
    bootstraps = []
    for _ in range(bootstrap_realizations):
        indices = rng.integers(0, len(x), len(x))
        bootstraps.append(_fit_profile(x[indices], mass[indices], edges, controls))
    for shell_index, result in enumerate(shells):
        result.setdefault('particle_count',None)
        result.setdefault('mass_msun',None)
        result.setdefault('effective_mass_count',None)
        result['failed_bin']=result['status']!='MEASURED'
        result['estimator_convergence']={'status':'CONVERGED' if result['status']=='MEASURED' else result['status'],
                                         'kind':'shell estimator iteration; not simulation convergence'}
        result['simulation_convergence']={'status':'NOT_EVALUATED','scope':'requires independent resolution/force controls'}
        result['sampling_uncertainty']={'status':'UNAVAILABLE_FAILED_BIN','realizations':bootstrap_realizations,
                                       'failure_counts':{},'axis_ratio_percentile_16_50_84':None,
                                       'axis_angle_percentile_95_deg':None}
        result['orientation_sampling_resolved']=[False]*3
        if result['status'] == 'MEASURED':
            values = np.asarray(result['eigenvalues'])
            gaps = [(values[0]-values[1])/values[0], (values[1]-values[2])/values[1]]
            separated = np.array([gaps[0] >= minimum_eigenvalue_gap,
                                  min(gaps) >= minimum_eigenvalue_gap, gaps[1] >= minimum_eigenvalue_gap])
            result['orientation_eigenvalue_resolved']=separated.tolist()
            result['relative_eigenvalue_gaps'] = list(map(float, gaps))
            samples, angles, failures = [], [], {}
            for bootstrap in bootstraps:
                boot = bootstrap[shell_index]
                if boot['status'] != 'MEASURED':
                    failures[boot['status']] = failures.get(boot['status'], 0) + 1
                    continue
                samples.append([boot['b_over_a'], boot['c_over_a']])
                dot = np.sum(np.asarray(boot['axes_columns']) * np.asarray(result['axes_columns']), axis=0)
                angles.append(np.degrees(np.arccos(np.clip(abs(dot), 0., 1.))))
            success = len(samples) / bootstrap_realizations if bootstrap_realizations else None
            adequate = bootstrap_realizations > 0 and success >= minimum_bootstrap_success_fraction
            angle95 = np.percentile(angles, 95., axis=0) if adequate else None
            result['sampling_uncertainty'] = {'kind': 'particle bootstrap of full supported parent; entire inner-to-outer profile refitted',
                    'realizations': bootstrap_realizations, 'success_fraction': success, 'failure_counts': failures,
                    'status': 'ESTIMATED' if adequate else 'NOT_ESTIMATED' if not bootstrap_realizations else 'INSUFFICIENT_SUCCESS',
                    'axis_ratio_percentile_16_50_84': np.percentile(samples, [16.,50.,84.], axis=0).tolist() if adequate else None,
                    'axis_angle_percentile_95_deg': angle95.tolist() if adequate else None}
            result['orientation_sampling_resolved'] = (separated & (angle95 <= maximum_orientation_error_deg)).tolist() if adequate else [False]*3
    return {'schema': 'darkmattersim-radial-density-shape-v1', 'science_acceptance': 'NOT_ACCEPTED',
            'status': 'MEASURED' if all(s['status']=='MEASURED' for s in shells) else 'INCOMPLETE',
            'window': window, 'estimator': estimator, 'radius_convention': 'proper kpc; volume-equivalent radius for ellipsoidal shells',
            'fixed_centre': True, 'complete_support_radius_kpc': float(support_radius_kpc),
            'parent_count_in_support': len(x), 'seed': seed,
            'controls': {k:v for k,v in controls.items() if k not in ('window','estimator')},
            'orientation_controls': {'minimum_relative_eigenvalue_gap': minimum_eigenvalue_gap,
                                     'maximum_bootstrap_95_angle_deg': maximum_orientation_error_deg},
            'bootstrap_minimum_success_fraction': minimum_bootstrap_success_fraction,
            'initialization': 'sphere then previous measured inner shell; no proof of global estimator uniqueness',
            'bootstrap_profiles':bootstraps,
            'simulation_convergence':{'status':'NOT_EVALUATED'},
            'bootstrap_joint_axis_ratio_samples': [[[b['b_over_a'],b['c_over_a']] if b['status']=='MEASURED' else None
                                                    for b in profile] for profile in bootstraps],
            'algorithm_record':{'primary_sources':['https://arxiv.org/abs/1107.5582'],
                'owning_functions':['radial_density_shapes','_shell'],
                'equations':['S_ij=sum(m_k*x_ki*x_kj)/sum(m_k); explicit optional inverse-radius-square weighting',
                             'b/a=sqrt(lambda_2/lambda_1); c/a=sqrt(lambda_3/lambda_1)',
                             'volume-equivalent ellipsoid: a*b*c=r^3'],
                'assumptions':['fixed centre','complete declared parent support','substructure retained unless excluded in membership'],
                'numerical_domain':'positive finite masses; supported shells; eigenvector degeneracy explicitly flagged'},
            'scope': 'density moments; sampling errors do not include force/resolution/convergence error', 'shells': shells}


@native_run_owner
def native_subfind_bound_member_ids(run_dir,snapshot,host_row,*,particle_type=1,
                                    central_only=False,memory_gib=1.):
    """Exact native self-bound object memberships associated with one FOF host.

    Union membership does not prove that each satellite is globally bound to
    the host. Central-only selection removes all other native bound objects;
    neither selection replaces native unbinding with an aperture cut.
    """
    import h5py
    from core.data import snapshot_id_shards,read_selected_group_ids,exact_uint64_ids,hdf5_scalar_attr
    from core.foundation import file_sha256
    if (type(snapshot) is not int or snapshot<0 or type(host_row) is not int or host_row<0
            or type(particle_type) is not int or not 0<=particle_type<=5 or type(central_only) is not bool
            or isinstance(memory_gib,bool) or not np.isfinite(memory_gib) or memory_gib<=0):
        raise ValueError('Exact native host/species selection and positive workspace required')
    groups = _group_files(Path(run_dir),snapshot);snapshots = _snapshot_files(Path(run_dir),snapshot)
    if not groups or not snapshots:
        raise ValueError('SUBFIND selection requires complete catalogues and matching ordered snapshots')
    sources = {str(p.resolve()):file_sha256(p) for p in groups+snapshots}
    frame = None
    for path in groups+snapshots:
        with h5py.File(path) as source:
            values = (hdf5_scalar_attr(source['Header'].attrs['Time'],'Time'),
                hdf5_scalar_attr(source['Header'].attrs['BoxSize'],'BoxSize'),
                hdf5_scalar_attr(source['Parameters'].attrs['HubbleParam'],'HubbleParam'))
            if not np.isfinite(values).all() or min(values)<=0:
                raise ValueError('Finite positive native membership epoch/frame required')
            if frame is not None and not np.allclose(frame,values,rtol=1e-12,atol=0):
                raise ValueError('Native bound membership catalogue/snapshot epoch or frame mismatch')
            frame = values
    count,group_count = None,None
    for path in groups:
        with h5py.File(path) as source:
            value = source['Header'].attrs.get('Nsubhalos_Total')
            if np.asarray(value).ndim!=0 or np.asarray(value).dtype.kind not in 'ui' or int(value)<0:
                raise ValueError('Exact native subhalo total required')
            if count is not None and int(value)!=count:
                raise ValueError('Native subhalo totals disagree')
            count = int(value)
            value = source['Header'].attrs.get('Ngroups_Total')
            if np.asarray(value).ndim!=0 or np.asarray(value).dtype.kind not in 'ui' or int(value)<0:
                raise ValueError('Exact native FOF total required')
            if group_count is not None and int(value)!=group_count:
                raise ValueError('Native FOF totals disagree')
            group_count = int(value)
    catalogue_bytes = count*256+group_count*128
    if host_row>=group_count:raise ValueError('Selected native FOF host does not exist')
    if catalogue_bytes+64*1024**2>memory_gib*1024**3:
        raise MemoryError('Native bound membership catalogue exceeds workspace')
    _check_file_series(groups,'Header','Nsubhalos_Total')
    _check_file_series(groups,'Header','Ngroups_Total')
    _unit_conversion(groups+snapshots)
    association = _read_catalogue_series(groups,'Subhalo','SubhaloGroupNr')
    rank = _read_catalogue_series(groups,'Subhalo','SubhaloRankInGr')
    offsets = _read_catalogue_series(groups,'Subhalo','SubhaloOffsetType')
    lengths = _read_catalogue_series(groups,'Subhalo','SubhaloLenType')
    host_offsets = _read_catalogue_series(groups,'Group','GroupOffsetType')
    host_lengths = _read_catalogue_series(groups,'Group','GroupLenType')
    if (host_offsets.shape!=(group_count,6) or host_lengths.shape!=host_offsets.shape
            or host_offsets.dtype.kind not in 'ui' or host_lengths.dtype.kind not in 'ui'
            or np.any(host_offsets<0) or np.any(host_lengths<0)):
        raise ValueError('Exact native FOF species spans required')
    if (association.shape!=(count,) or rank.shape!=(count,) or association.dtype.kind not in 'ui'
            or rank.dtype.kind not in 'ui' or np.any(association<0) or np.any(association>=group_count) or np.any(rank<0)):
        raise ValueError('Exact nonnegative native host association and subhalo rank required')
    rows = np.flatnonzero(association==host_row)
    central = rows[rank[rows]==0]
    if len(central)!=1:
        raise ValueError('Native host requires one explicitly ranked central; no row-order inference')
    selected = central if central_only else rows
    if lengths.shape!=(count,6) or lengths.dtype.kind not in 'ui' or np.any(lengths<0):
        raise ValueError('Complete native species membership lengths required')
    total = sum(int(lengths[row,particle_type]) for row in selected)
    if catalogue_bytes+total*160+64*1024**2>memory_gib*1024**3:
        raise MemoryError('Exact bound membership ID payload exceeds workspace')
    records,population_count = snapshot_id_shards(snapshots,particle_type)
    if offsets.shape!=(count,6) or offsets.dtype.kind not in 'ui' or np.any(offsets<0):
        raise ValueError('Complete exact native species membership offsets required')
    fof_spans = sorted((int(host_offsets[row,particle_type]),
        int(host_offsets[row,particle_type])+int(host_lengths[row,particle_type]))
        for row in range(group_count) if host_lengths[row,particle_type])
    if any(stop>population_count for start,stop in fof_spans) or any(
            right[0]<left[1] for left,right in zip(fof_spans,fof_spans[1:])):
        raise ValueError('Native FOF memberships overlap or exceed the source population')
    host_start = int(host_offsets[host_row,particle_type])
    host_stop = host_start+int(host_lengths[host_row,particle_type])
    spans = sorted((int(offsets[row,particle_type]),
        int(offsets[row,particle_type])+int(lengths[row,particle_type]))
        for row in rows if lengths[row,particle_type])
    if any(stop>population_count for start,stop in spans) or any(
            right[0]<left[1] for left,right in zip(spans,spans[1:])):
        raise ValueError('Associated native bound memberships overlap or exceed the source population')
    if any(start<host_start or stop>host_stop for start,stop in spans):
        raise ValueError('Associated native bound membership lies outside its declared FOF host span')
    ids,_ = read_selected_group_ids(offsets,lengths,selected,records,particle_type=particle_type,max_ids=total)
    ids = np.sort(exact_uint64_ids(ids))
    if np.any(ids==0):raise ValueError('Native bound membership contains a zero identity')
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise RuntimeError('Native bound membership source changed during intake')
    return ids,{'definition':'native central self-bound SUBFIND membership' if central_only else
        'union of native self-bound SUBFIND objects associated with the selected FOF host',
        'global_host_binding':'NOT_ESTABLISHED; subgroup self-binding and FOF association are distinct',
        'host_row':host_row,'snapshot':snapshot,'particle_type':particle_type,
        'selected_subhalo_rows':selected.tolist(),'central_subhalo_row':int(central[0]),
        'selected_id_count':len(ids),'native_fof_span':[host_start,host_stop],
        'ownership_scope':'selected species native subhalo spans contained in disjoint FOF spans',
        'source_hashes':sources}


@directory_owner('output_directory')
def export_native_halo_shapes(run_dir, snapshot, host_row, *, output_directory,
                              memory_gib=1., **shape_request):
    """Complete type-1 density samples around a fixed native FOF centre.

    Default aperture selection retains substructure/non-bound matter. Optional
    exact FOF, associated self-bound SUBFIND union or central-only membership
    preserves each native definition; none proves global host binding. Coarse
    type-2 counts/masses are a separate aperture census. No particle sampling.
    """
    from pathlib import Path
    import json
    import tempfile
    import h5py
    from analysis import validate_gadget_snapshot_set
    from core.foundation import file_sha256
    from zoom import read_host_catalogue
    from halos import _snapshot_files, _group_files
    from core.data import native_unit_factor, exact_uint64_ids, run_epoch_contract, coordinate_contract
    from core.foundation import product_provenance, publish_directory, verify_product_provenance

    normalized=validate_native_shape_request(dict(snapshot=snapshot,host_row=host_row,output_directory=output_directory,
                                                  memory_gib=memory_gib,**shape_request))
    from core.foundation import canonical_digest
    request_sha256 = canonical_digest(normalized)
    snapshot,host_row,memory_gib=normalized.pop('snapshot'),normalized.pop('host_row'),normalized.pop('memory_gib')
    destination=Path(normalized.pop('output_directory'))
    parent_selection=normalized.pop('parent_selection','complete_support_sphere')
    expected_scale_factor=normalized.pop('expected_scale_factor',None)
    shape_request=normalized
    if destination.exists():
        raise FileExistsError(f'halo-shape destination already exists: {destination}')
    catalogue = read_host_catalogue(run_dir, snapshot, host_row=host_row, memory_gib=memory_gib)
    snapshots = catalogue['snapshots']
    validate_gadget_snapshot_set(snapshots, snapshot)
    a, box, h = catalogue['frame']
    if expected_scale_factor is not None and a!=expected_scale_factor:
        raise ValueError('Selected halo request does not match the exact saved epoch')
    member_ids=None
    membership_record=None
    if parent_selection in ('subfind_bound_associated','subfind_central_bound'):
        member_ids,membership_record = native_subfind_bound_member_ids(run_dir,snapshot,host_row,
            central_only=parent_selection=='subfind_central_bound',memory_gib=memory_gib)
    if parent_selection=='fof_members':
        from core.data import snapshot_id_shards,read_selected_group_ids
        count=int(catalogue['lengths'][host_row,1])
        if count*160+64*1024**2>memory_gib*1024**3:
            raise MemoryError('Exact FOF membership exceeds declared shape workspace')
        shards,_=snapshot_id_shards(snapshots,1)
        member_ids,_=read_selected_group_ids(catalogue['offsets'],catalogue['lengths'],
            np.array([host_row],dtype=np.int64),shards,particle_type=1,max_ids=count)
        member_ids=np.sort(member_ids)
    support = shape_request.get('support_radius_kpc')
    if isinstance(support, bool) or not isinstance(support, (int, float)) or not np.isfinite(support) or not 0. < support < box*a/h/2.:
        raise ValueError('proper support radius must fit strictly inside the periodic half-box')
    center = catalogue['positions'][host_row]
    history_workspace=(1+shape_request['bootstrap_realizations'])*(len(shape_request['radial_edges_kpc'])-1)*shape_request['maximum_iterations']*512
    membership_workspace=0 if member_ids is None else len(member_ids)*32
    workspace=history_workspace+membership_workspace+64*1024**2
    if workspace>memory_gib*1024**3:
        raise MemoryError('shape workspace exceeds declared memory')
    chunk_size=max(1,min(250_000,int((memory_gib*1024**3-workspace)//512)))
    selected_count=0
    aperture_count,aperture_mass = 0,0.
    offsets, masses, identifiers = [], [], []
    coarse_count, coarse_mass = 0, 0.
    for path in snapshots:
        with h5py.File(path) as source:
            for species in (1, 2):
                name = f'PartType{species}'
                if name not in source:
                    continue
                group = source[name]
                for start in range(0, len(group['Coordinates']), chunk_size):
                    stop = min(start+chunk_size, len(group['Coordinates']))
                    native = np.asarray(group['Coordinates'][start:stop], dtype=float)
                    relative = ((native-center+box/2.) % box-box/2.) * native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h)
                    if not np.isfinite(relative).all():
                        raise ValueError('nonfinite native DM coordinates')
                    take = np.linalg.norm(relative, axis=1) <= support
                    aperture = take.copy()
                    ids=None
                    if species==1:
                        ids=exact_uint64_ids(np.asarray(group['ParticleIDs'][start:stop]))
                        if member_ids is not None:
                            indices=np.searchsorted(member_ids,ids)
                            match=indices<len(member_ids)
                            match[match] &= member_ids[indices[match]]==ids[match]
                            take &= match
                        selected_count+=int(take.sum())
                        if selected_count*160+workspace+chunk_size*128>memory_gib*1024**3:
                            raise MemoryError('Selected DM parent exceeds declared shape workspace')
                    mass = (np.asarray(group['Masses'][start:stop], dtype=float) if 'Masses' in group
                            else np.full(stop-start, float(source['Header'].attrs['MassTable'][species]))) * native_unit_factor('mass_msun',scale_factor=a,hubble_param=h)
                    if not np.isfinite(mass[aperture]).all() or np.any(mass[aperture] <= 0.):
                        raise ValueError('native DM masses must be finite and positive')
                    if species==1:
                        aperture_count += int(aperture.sum())
                        aperture_mass += float(mass[aperture].sum())
                    if not np.any(take):continue
                    if species == 2:
                        coarse_count += int(take.sum())
                        coarse_mass += float(mass[take].sum())
                    else:
                        offsets.append(relative[take]); masses.append(mass[take]); identifiers.append(exact_uint64_ids(ids[take]))
    x = np.concatenate(offsets) if offsets else np.empty((0,3))
    mass = np.concatenate(masses) if masses else np.empty(0)
    ids = np.concatenate(identifiers) if identifiers else np.empty(0,dtype=np.uint64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate IDs in complete type-1 density-shape parent')
    result = radial_density_shapes(x, mass, **shape_request)
    source_records = catalogue['sources'] + catalogue['snapshot_sources']
    current_paths = sorted(str(p.resolve()) for p in _snapshot_files(Path(run_dir),snapshot)+_group_files(Path(run_dir),snapshot))
    if (current_paths != sorted(entry['path'] for entry in source_records)
            or any(file_sha256(entry['path']) != entry['sha256'] for entry in source_records)):
        raise RuntimeError('native halo sources changed during shape measurement')
    result.update(source_files=source_records, snapshot=int(snapshot), epoch_local_host_row=int(host_row),
                  native_request_sha256=request_sha256,
                  scale_factor=float(a), h=float(h), center_comoving_kpc_h=center.tolist(),
                  measured_r200c_proper_kpc=float(catalogue['radii'][host_row]*a/h),
                  parent_definition='all type-1 particles in complete proper support sphere; includes substructure and non-bound matter',
                  coarse_type2_count_in_support=coarse_count, coarse_type2_mass_msun_in_support=coarse_mass)
    result['parent_selection']=parent_selection
    result['selection_census'] = {'aperture_type1_count':aperture_count,
        'aperture_type1_mass_msun':aperture_mass,'selected_type1_count':len(ids),
        'selected_type1_mass_msun':float(mass.sum()),
        'excluded_type1_count':aperture_count-len(ids),
        'excluded_type1_mass_msun':aperture_mass-float(mass.sum()),
        'exclusion_definition':'membership rejection inside the declared aperture; not evidence of global unbinding'}
    if member_ids is not None:
        import hashlib
        result['parent_definition']='exact native FOF type-1 members in complete proper support sphere; includes member substructure'
        if membership_record is not None:
            result['parent_definition']=membership_record['definition']+' within complete proper support sphere'
            result['native_membership']=membership_record
        result['membership_count']=len(member_ids)
        result['membership_ids_sha256']=hashlib.sha256(member_ids.astype('<u8',copy=False).tobytes()).hexdigest()
    result['particle_read_chunk_size']=chunk_size
    result['epoch_contract'] = run_epoch_contract(run_dir,int(snapshot),a)
    coordinates = coordinate_contract(position_units='proper kpc',velocity_convention='no velocities in this density product',
        origin='fixed native FOF centre; centre retained separately in global comoving kpc/h',
        periodic_wrapping='minimum-image offsets in [-L/2,L/2); support strictly inside half box')
    result['data_contract'] = product_provenance(run_dir,algorithm=__file__,selection=result['parent_definition'],
        source_files=source_records,coordinates=coordinates,snapshot=int(snapshot))
    result['dataset_units'] = {'offsets_proper_kpc':'proper kpc','masses_msun':'Msun','particle_ids':'uint64 identity',
                              'radius':'proper kpc','axis_ratios':'dimensionless','orientation_sampling_error':'degrees'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.halo-shape-',dir=destination.parent))
    from core.foundation import atomic_json
    try:
        np.savez_compressed(stage/'parent.npz', particle_ids=ids, offsets_proper_kpc=x, masses_msun=mass)
        result['parent_sha256'] = file_sha256(stage/'parent.npz')
        from core.data import numpy_archive_contract
        result['array_contract']=numpy_archive_contract(stage/'parent.npz',{
            'particle_ids':{'units':'identity','identity_namespace':result['data_contract']['identity_namespaces']['dark_matter'],
                'quantity':'exact saved type-1 parent particle identity','missing_policy':'none'},
            'offsets_proper_kpc':{'units':'proper kpc','quantity':'minimum-image offsets from fixed native FOF center','missing_policy':'none'},
            'masses_msun':{'units':'Msun','quantity':'saved dynamical particle mass','missing_policy':'none'}},
            epoch=result['epoch_contract'],coordinates=coordinates,provenance={'report':'report.json'},memory_bytes=int(memory_gib*1024**3))
        result['implementation_sha256'] = file_sha256(__file__)
        (stage/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (stage/'report.sha256').write_text(file_sha256(stage/'report.json')+'\n')
        verify_product_provenance(run_dir,result['data_contract'])
        publish_directory(stage,destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return result


def verify_native_shape_export(run_dir, request):
    """Reuse only a complete, unchanged export of the exact native request.

    Failed scientific bins are valid retained measurements, never a PASS.
    Older exports without a request receipt cannot be resumed automatically.
    """
    import json
    from pathlib import Path
    from core.foundation import file_sha256
    from core.foundation import canonical_digest
    from core.foundation import verify_product_provenance
    from halos import _snapshot_files, _group_files
    normalized = validate_native_shape_request(request)
    directory = Path(normalized['output_directory'])
    report = json.loads((directory/'report.json').read_text())
    if ((directory/'report.sha256').read_text().strip() != file_sha256(directory/'report.json')
            or report.get('schema') != 'darkmattersim-radial-density-shape-v1'
            or report.get('native_request_sha256') != canonical_digest(normalized)
            or report.get('implementation_sha256') != file_sha256(__file__)
            or report.get('parent_sha256') != file_sha256(directory/'parent.npz')):
        raise ValueError('Shape export request, implementation or parent checksum changed')
    paths = _snapshot_files(Path(run_dir), normalized['snapshot']) + _group_files(Path(run_dir), normalized['snapshot'])
    records = report['source_files']
    if (sorted(str(p.resolve()) for p in paths) != sorted(r['path'] for r in records)
            or any(file_sha256(r['path']) != r['sha256'] for r in records)):
        raise ValueError('Shape export native source files changed')
    verify_product_provenance(run_dir,report['data_contract'])
    if report.get('array_contract') is not None:
        from core.data import verify_numpy_archive_contract
        verify_numpy_archive_contract(directory/'parent.npz',report['array_contract'],memory_bytes=int(normalized['memory_gib']*1024**3))
    return report


@directory_owner('shape_directory')
@directory_owner('output_directory')
def plot_radial_shapes(shape_directory, output_directory):
    """Plot measured density ratios, particle support and orientation uncertainty."""
    from pathlib import Path
    import json
    import tempfile
    from core.foundation import file_sha256
    from matplotlib import pyplot as plt
    from figures import apply_plot_style

    source, destination = Path(shape_directory).resolve(), Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f'shape figure destination already exists: {destination}')
    path = source/'report.json'
    source_hash = file_sha256(path)
    report = json.loads(path.read_text())
    if report.get('schema') != 'darkmattersim-radial-density-shape-v1':
        raise ValueError('radial shape figure requires a measured shape report')
    if file_sha256(source/'parent.npz') != report.get('parent_sha256'):
        raise ValueError('shape parent hash disagrees with report')
    if report.get('array_contract') is not None:
        from core.data import verify_numpy_archive_contract
        if not (source/'report.sha256').is_file() or (source/'report.sha256').read_text().strip()!=source_hash:
            raise ValueError('shape report checksum mismatch')
        verify_numpy_archive_contract(source/'parent.npz',report['array_contract'])
    shells = report['shells']
    radius = np.array([np.sqrt(s['lower_radius_kpc']*s['upper_radius_kpc']) if s['lower_radius_kpc']>0.
                       else s['upper_radius_kpc']/2. for s in shells])
    apply_plot_style({'figure_layout':'presentation','figure_formats':['png','pdf'],'figure_dpi':180})
    figure, axes = plt.subplots(1,3,figsize=(12.,3.8),constrained_layout=True)
    for j, (key,label,color) in enumerate((('b_over_a',r'$b/a$','#0072B2'),('c_over_a',r'$c/a$','#D55E00'))):
        values=np.array([s.get(key,np.nan) if s['status']=='MEASURED' else np.nan for s in shells])
        axes[0].plot(radius,values,'o-',color=color,label=label)
        intervals=np.full((len(shells),2),np.nan)
        for i,s in enumerate(shells):
            interval=s.get('sampling_uncertainty',{}).get('axis_ratio_percentile_16_50_84')
            if interval is not None: intervals[i]=[interval[0][j],interval[2][j]]
        axes[0].fill_between(radius,intervals[:,0],intervals[:,1],color=color,alpha=.18)
    labelled_failure = False
    for s in shells:
        if s['status']!='MEASURED':
            axes[0].axvspan(max(s['lower_radius_kpc'],radius.min()/2.),s['upper_radius_kpc'],color='gray',alpha=.12,
                           label='unmeasured shell' if not labelled_failure else None)
            labelled_failure = True
    axes[0].set(xlabel='proper radius [kpc]',ylabel='density-moment axis ratio',ylim=(0.,1.02),title='Radial density shape')
    axes[0].legend(fontsize=9.)
    count=np.array([s.get('particle_count') if s.get('particle_count') is not None else np.nan for s in shells],dtype=float)
    effective=np.array([s.get('effective_mass_count') if s.get('effective_mass_count') is not None else np.nan for s in shells],dtype=float)
    axes[1].plot(radius,np.where(count>0,count,np.nan),'o-',label='particle count')
    axes[1].plot(radius,np.where(effective>0,effective,np.nan),'s--',label='effective mass count')
    axes[1].axhline(report['controls']['minimum_particles'],color='black',ls=':',label='declared minimum')
    axes[1].set(xlabel='proper radius [kpc]',ylabel='particles in shell',yscale='log',title='Particle support')
    axes[1].legend(fontsize=8.)
    for j,label in enumerate(('major axis','intermediate axis','minor axis')):
        angles=np.array([s.get('sampling_uncertainty',{}).get('axis_angle_percentile_95_deg')
                         if s.get('sampling_uncertainty',{}).get('axis_angle_percentile_95_deg') is not None
                         else [np.nan]*3 for s in shells])
        axes[2].plot(radius,angles[:,j],'o-',label=label)
    axes[2].axhline(report['orientation_controls']['maximum_bootstrap_95_angle_deg'],color='black',ls=':',label='declared limit')
    axes[2].set(xlabel='proper radius [kpc]',ylabel='bootstrap 95% axis angle [degrees]',ylim=(0.,90.),title='Orientation sampling error')
    axes[2].legend(fontsize=8.)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(radius.min()*.75, max(s['upper_radius_kpc'] for s in shells)*1.05)
    description=(f"Native snapshot {report['snapshot']} · epoch-local host {report['epoch_local_host_row']}" if 'snapshot' in report
                 else 'Analytic density-moment control')
    selection = report.get('parent_selection')
    title = description+' · '+report['window'].replace('_',' ')
    if selection:title += '\nSelection: '+selection.replace('_',' ')
    figure.suptitle(title,fontsize=11.)
    metadata={'status':'DIAGNOSTIC_ONLY','science_acceptance':'NOT_ACCEPTED','source_report_sha256':source_hash,
              'implementation_sha256':file_sha256(__file__),
              'shading':'particle-bootstrap 16–84% intervals; gray spans mark unmeasured bins',
              'scope':'density moments and particle sampling; not potential shape or convergence uncertainty',
              'parent_selection':selection,'parent_definition':report.get('parent_definition'),
              'selection_census':report.get('selection_census')}
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        stage=Path(tempfile.mkdtemp(prefix='.shape-figure-',dir=destination.parent))
        from core.foundation import atomic_json, publish_directory
        try:
            from figures import export_figure
            export_figure(figure,stage/'radial_shapes.png',formats=['png','pdf'],dpi=180,close=False,
                metadata={'source_report_sha256':source_hash,'science_acceptance':'NOT_ACCEPTED',
                    'epoch_contract':report.get('epoch_contract'),'source_data_contract':report.get('data_contract'),
                    'quantity':'saved radial density-moment estimates and their sampling diagnostics'})
            if file_sha256(path)!=source_hash: raise RuntimeError('shape report changed during plotting')
            metadata['product_sha256']={p.name:file_sha256(p) for p in (stage/'radial_shapes.png',stage/'radial_shapes.pdf')}
            (stage/'report.json').write_text(json.dumps(metadata,indent=2)+'\n')
            publish_directory(stage,destination)
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'failure':{'type':type(exc).__name__,'message':str(exc)}})
            raise
    finally:
        plt.close(figure)
    return metadata


def join_radial_shapes_to_timeline(report, timeline):
    """Join a verified density measurement to one exact FOF detection/epoch.

    Input timeline positions and output semiaxes are comoving Mpc/h. No row
    continuity, potential shape or intermediate-time shapes are inferred.
    """
    import copy
    epochs = [e for e in timeline['snapshots'] if e['snapshot'] == report['snapshot']]
    if len(epochs) != 1 or epochs[0]['scale_factor'] != report['scale_factor']:
        raise ValueError('Radial shape join requires the exact saved epoch')
    epoch = epochs[0]
    matches = [i for i in epoch.get('halo_overlays', {}).get('halos', {}).get('items', [])
               if i.get('id') == report['epoch_local_host_row']]
    if (len(matches) != 1 or matches[0].get('id_source') != 'catalogue_row'
            or matches[0].get('finder') != 'GADGET-4 FOF'):
        raise ValueError('Radial shape join requires exactly one native FOF row')
    item = matches[0]
    centre = np.mod(np.asarray(report['center_comoving_kpc_h'])/1000., timeline['box_mpc_h'])
    if not np.allclose(centre, item['position_mpc_h'], rtol=0., atol=1.e-12):
        raise ValueError('Radial shape and Blender native centres/units disagree')
    conversion = report['h']/report['scale_factor']/1000.
    shape = {'kind':'radial_density_moments', 'science_acceptance':'NOT_ACCEPTED',
             'snapshot':report['snapshot'],'epoch_local_host_row':report['epoch_local_host_row'],
             'scale_factor':report['scale_factor'], 'units':'comoving Mpc/h',
             'native_request_sha256':report['native_request_sha256'],
             'scope':report['scope'], 'shells':copy.deepcopy(report['shells'])}
    for shell in shape['shells']:
        if shell['status'] == 'MEASURED':
            shell['fitted_semiaxes_comoving_mpc_h'] = (np.asarray(shell['fitted_semiaxes_at_upper_radius_kpc'])*conversion).tolist()
    item['radial_density_shapes'] = shape
    return shape


def validate_shape_request(request):
    """Accept legacy single-host requests or explicit selected halo/epoch batches."""
    if not isinstance(request,dict) or request.get('schema')!='radial-shape-batch-v1':
        return validate_native_shape_request(request)
    from pathlib import Path
    required={'schema','selections','shape_controls','output_directory','memory_gib','maximum_targets'}
    if set(request)!=required:
        raise ValueError('Batch shape request requires exactly '+', '.join(sorted(required)))
    maximum=_integer(request['maximum_targets'],'maximum_targets',1)
    selections=request['selections']
    if not isinstance(selections,list) or not selections or len(selections)>maximum:
        raise ValueError('Batch selections must be nonempty and within maximum_targets')
    controls=request['shape_controls']
    if not isinstance(controls,dict) or set(controls)!=(NATIVE_SHAPE_FIELDS-{'snapshot','host_row','output_directory'})|{'parent_selection'}:
        raise ValueError('Batch shape_controls require all estimator/sampling controls and explicit parent_selection')
    output=request['output_directory']
    if not isinstance(output,(str,Path)) or not str(output).strip():
        raise ValueError('Batch output_directory must be a nonempty path')
    destination=Path(output).expanduser().resolve()
    rows=[];identities=set()
    for target in selections:
        if not isinstance(target,dict) or set(target)!={'snapshot','host_row','scale_factor'}:
            raise ValueError('Each batch target requires snapshot, host_row and exact scale_factor')
        normalized=validate_native_shape_request(dict(controls,snapshot=target['snapshot'],host_row=target['host_row'],
            expected_scale_factor=target['scale_factor'],output_directory=destination/'profile',memory_gib=request['memory_gib']))
        key=(normalized['snapshot'],normalized['host_row'])
        if key in identities:
            raise ValueError('Duplicate selected halo/epoch')
        identities.add(key)
        rows.append({'snapshot':key[0],'host_row':key[1],'scale_factor':normalized['expected_scale_factor']})
    normalized_controls={key:normalized[key] for key in controls}
    return dict(schema=request['schema'],selections=rows,shape_controls=normalized_controls,
                output_directory=str(destination),memory_gib=normalized['memory_gib'],maximum_targets=maximum)


def shape_profile_requests(request):
    """Expand a batch without inferring cross-epoch identity from catalogue rows."""
    from pathlib import Path
    normalized=validate_shape_request(request)
    if normalized.get('schema')!='radial-shape-batch-v1':
        return [normalized]
    root=Path(normalized['output_directory'])
    return [validate_native_shape_request(dict(normalized['shape_controls'],snapshot=row['snapshot'],host_row=row['host_row'],
        expected_scale_factor=row['scale_factor'],memory_gib=normalized['memory_gib'],
        output_directory=root/'profiles'/f"snapshot_{row['snapshot']:03d}_host_{row['host_row']}"))
        for row in normalized['selections']]


def _verify_shape_figures(directory,report_sha256):
    import json
    from pathlib import Path
    from core.foundation import file_sha256
    from figures import verify_figure_export
    directory=Path(directory)
    record=json.loads((directory/'report.json').read_text())
    if (record.get('source_report_sha256')!=report_sha256 or record.get('implementation_sha256')!=file_sha256(__file__)
            or set(record.get('product_sha256',{}))!={'radial_shapes.png','radial_shapes.pdf'}):
        raise ValueError('Shape figure source or implementation changed')
    for name,sha in record['product_sha256'].items():
        if file_sha256(directory/name)!=sha:
            raise ValueError('Shape figure checksums changed')
    verify_figure_export(directory/'radial_shapes.figure.json')
    return record


@native_run_owner
@directory_owner('request_root')
def _export_shape_batch(run_dir,request,request_root):
    from pathlib import Path
    import json,tempfile
    from core.foundation import atomic_json, publish_directory, canonical_digest
    from core.foundation import file_sha256
    destination=Path(request_root)
    if destination.exists():
        raise FileExistsError('Shape batch destination already exists: '+str(destination))
    # Read and check every requested epoch before publishing any profile.
    from zoom import read_host_catalogue
    for row in request['selections']:
        catalogue=read_host_catalogue(run_dir,row['snapshot'],host_row=row['host_row'],memory_gib=request['memory_gib'])
        if catalogue['frame'][0]!=row['scale_factor']:
            raise ValueError('Selected halo request does not match the exact saved epoch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.radial-shape-batch-',dir=destination.parent))
    reports=[];figures=[];inspectors=[]
    try:
        for profile_request in shape_profile_requests(request):
            profile=Path(profile_request['output_directory']).relative_to(destination)
            staged=dict(profile_request,output_directory=str(stage/profile))
            report=export_native_halo_shapes(run_dir,**staged)
            # Bind the portable final location as the request identity before publication.
            report['native_request_sha256']=canonical_digest(profile_request)
            atomic_json(stage/profile/'report.json',report)
            (stage/profile/'report.sha256').write_text(file_sha256(stage/profile/'report.json')+'\n')
            figure=profile.with_name(profile.name+'_figures')
            plot_radial_shapes(stage/profile,stage/figure)
            sha=file_sha256(stage/profile/'report.json')
            row={'snapshot':report['snapshot'],'host_row':report['epoch_local_host_row'],
                 'scale_factor':report['scale_factor'],'report':(profile/'report.json').as_posix(),
                 'report_sha256':sha,'parent_sha256':report['parent_sha256'],
                 'status':report['status'],'failed_bins':[i for i,shell in enumerate(report['shells']) if shell['failed_bin']],
                 'sampling_status':[shell['sampling_uncertainty']['status'] for shell in report['shells']],
                 'simulation_convergence':'NOT_EVALUATED','figure_directory':figure.as_posix()}
            reports.append(row)
            for name in ('radial_shapes.png','radial_shapes.pdf'):
                figures.append({'path':(figure/name).as_posix(),'sha256':file_sha256(stage/figure/name),
                                'source_report':row['report'],'source_report_sha256':sha,
                                'snapshot':row['snapshot'],'host_row':row['host_row']})
            inspectors.append({'snapshot':row['snapshot'],'host_row':row['host_row'],'scale_factor':row['scale_factor'],
                               'report':row['report'],'sha256':sha,'finder':'GADGET-4 FOF',
                               'identity_scope':'epoch-local catalogue row; no cross-epoch row continuity assumed'})
        # Recheck earlier targets after the last calculation, before batch publication.
        from core.foundation import verify_product_provenance
        from halos import _snapshot_files,_group_files
        for row in reports:
            profile_report=json.loads((stage/row['report']).read_text())
            sources=profile_report['source_files']
            paths=_snapshot_files(Path(run_dir),row['snapshot'])+_group_files(Path(run_dir),row['snapshot'])
            if (sorted(str(p.resolve()) for p in paths)!=sorted(r['path'] for r in sources)
                    or any(file_sha256(r['path'])!=r['sha256'] for r in sources)):
                raise ValueError('Native batch sources changed before publication')
            verify_product_provenance(run_dir,profile_report['data_contract'])
        from figures import review_figure_folder
        review=review_figure_folder(stage)
        if review['status']!='COMPLETE':
            raise ValueError('Batch shape figure integrity review failed')
        result={'schema':'radial-shape-batch-product-v1','execution_status':'COMPLETE',
                'status':'MEASURED' if all(r['status']=='MEASURED' and all(s in ('ESTIMATED','NOT_ESTIMATED') for s in r['sampling_status']) for r in reports) else 'INCOMPLETE',
                'science_acceptance':'NOT_ACCEPTED','simulation_convergence':'NOT_EVALUATED',
                'request':request,'request_sha256':canonical_digest(request),'implementation_sha256':file_sha256(__file__),
                'profiles':reports,'figure_registry':figures,'inspector_references':inspectors,
                'figure_review_status':review['status']}
        atomic_json(stage/'batch.json',result)
        (stage/'batch.sha256').write_text(file_sha256(stage/'batch.json')+'\n')
        publish_directory(stage,destination)
        return result
    except BaseException as error:
        if stage.exists():
            atomic_json(stage/'export_failure.json',{'status':'FAILED','science_acceptance':'NOT_ACCEPTED',
                'failure':{'type':type(error).__name__,'message':str(error)}})
        raise


def export_shape_request(run_dir,request,*,resume=False):
    """Shared public analysis/CLI adapter for single and batch shape requests."""
    from pathlib import Path
    normalized=validate_shape_request(request)
    if normalized.get('schema')!='radial-shape-batch-v1':
        return verify_native_shape_export(run_dir,normalized) if resume and Path(normalized['output_directory']).exists() else export_native_halo_shapes(run_dir,**normalized)
    if resume and Path(normalized['output_directory']).exists():
        return verify_shape_request(run_dir,normalized)
    return _export_shape_batch(run_dir,normalized,normalized['output_directory'])


def verify_shape_request(run_dir,request):
    """Verify every profile, native source, figure and exact batch membership."""
    from pathlib import Path
    import json
    from core.foundation import canonical_digest
    from core.foundation import file_sha256
    normalized=validate_shape_request(request)
    if normalized.get('schema')!='radial-shape-batch-v1':
        return verify_native_shape_export(run_dir,normalized)
    root=Path(normalized['output_directory']);path=root/'batch.json'
    result=json.loads(path.read_text())
    expected=shape_profile_requests(normalized)
    if ((root/'batch.sha256').read_text().strip()!=file_sha256(path)
            or result.get('schema')!='radial-shape-batch-product-v1' or result.get('execution_status')!='COMPLETE'
            or result.get('request_sha256')!=canonical_digest(normalized)
            or result.get('implementation_sha256')!=file_sha256(__file__) or len(result['profiles'])!=len(expected)):
        raise ValueError('Shape batch request, implementation or completion seal changed')
    expected_figures=[]
    for req,row in zip(expected,result['profiles']):
        report=verify_native_shape_export(run_dir,req)
        if (row['snapshot'],row['host_row'],row['scale_factor'])!=(report['snapshot'],report['epoch_local_host_row'],report['scale_factor']):
            raise ValueError('Shape batch selected identity/epoch changed')
        profile=(Path(req['output_directory'])/'report.json').relative_to(root).as_posix()
        if row['report']!=profile or row['report_sha256']!=file_sha256(root/profile) or row['parent_sha256']!=report['parent_sha256']:
            raise ValueError('Shape batch profile binding changed')
        figure_directory=Path(req['output_directory']).with_name(Path(req['output_directory']).name+'_figures')
        if row['figure_directory']!=figure_directory.relative_to(root).as_posix():
            raise ValueError('Shape batch figure directory changed')
        _verify_shape_figures(figure_directory,row['report_sha256'])
        for name in ('radial_shapes.png','radial_shapes.pdf'):
            expected_figures.append({'path':(figure_directory/name).relative_to(root).as_posix(),
                'sha256':file_sha256(figure_directory/name),'source_report':profile,
                'source_report_sha256':row['report_sha256'],'snapshot':row['snapshot'],'host_row':row['host_row']})
    if result['figure_registry']!=expected_figures:
        raise ValueError('Shape batch figure registry is incomplete or changed')
    for figure in result['figure_registry']:
        path=(root/figure['path']).resolve()
        if not path.is_relative_to(root) or file_sha256(path)!=figure['sha256']:
            raise ValueError('Shape batch figure registry changed')
    expected_inspectors=[(r['snapshot'],r['host_row'],r['scale_factor'],r['report'],r['report_sha256']) for r in result['profiles']]
    if [(r['snapshot'],r['host_row'],r['scale_factor'],r['report'],r['sha256']) for r in result['inspector_references']]!=expected_inspectors:
        raise ValueError('Shape batch inspector references changed')
    return result


def shape_reports(run_dir,request):
    """Bounded iterator used by the Blender consumer; verify before joining."""
    verify_shape_request(run_dir,request)
    for profile in shape_profile_requests(request):
        yield verify_native_shape_export(run_dir,profile)


def add_native_hydro_source_history(graph,path,*,particle_namespace,maximum_bytes):
    """Insert measured source-operation states and ancestry into indexed graphs.

    Native event epochs are separate from saved snapshot indices. Event nodes
    use operation-specific namespaces so an ID-preserving conversion or merger
    cannot create a self-edge. Saved detections are linked only when observed.
    """
    from stars import verify_native_hydro_history
    from core.data import epoch_contract,isolated_epoch_contract
    from core.cosmology import run_proper_time_clock
    from core.foundation import SECONDS_PER_YEAR
    import zipfile
    path=Path(path).resolve();checksum=file_sha256(path);report=verify_native_hydro_history(path)
    if type(maximum_bytes) is not int or maximum_bytes<1:raise ValueError('Declare a positive source-lineage workspace')
    if particle_namespace!=report['provenance']['identity_namespaces']['native_particle']:raise ValueError('Source lineage must preserve the native ID namespace')
    payload=path.parent/report['numeric_data']['directory']/'arrays.npz'
    with zipfile.ZipFile(payload) as index:allocation=sum(entry.file_size for entry in index.infolist())
    rows=sum(i['event_count']+i['return_row_count']+i.get('bh_row_count',0) for i in report['intervals'])
    if allocation+rows*2048+len(graph.nodes)*256+16*1024**2>maximum_bytes:raise MemoryError('Native source lineage workspace exceeded')
    sources={**report['source_hashes'],str(path):checksum,str(payload):file_sha256(payload),str(Path(__file__).resolve()):file_sha256(__file__)}
    root=Path(report['request']['run_directory']).resolve();params=report['epochs'][0]['parameters'];comoving=bool(params['comoving'])
    clock=run_proper_time_clock(root) if comoving else None;operations=[]
    with np.load(payload,allow_pickle=False) as arrays:
        for ordinal,interval in enumerate(report['intervals']):
            prefix=f'interval_{ordinal:03d}_event_'
            for j in range(interval['event_count']):
                operations.append({'time':float(arrays[prefix+'birth_time'][j]),'kind':'birth','ordinal':ordinal,'row':j,
                    'parent':int(arrays[prefix+'parent_id'][j]),'child':int(arrays[prefix+'ssp_id'][j]),
                    'before_mass':float(arrays[prefix+'gas_mass_before'][j]),'after_mass':float(arrays[prefix+'locked_mass'][j])})
            groups={}
            for row in interval['stellar_return_rows']:groups.setdefault((row['native_time'],row['source_id'],row['source_counter']),[]).append(row)
            for key,group in groups.items():operations.append({'time':float(key[0]),'kind':'return','ordinal':ordinal,'counter':key[2],'source':key[1],'rows':group})
            for row in interval.get('black_hole_rows',[]):
                if row['kind']=='merger':operations.append({'time':float(row['native_time']),'kind':'merger','ordinal':ordinal,'row':row})
                elif row['kind']=='seed':operations.append({'time':float(row['native_time']),'kind':'seed','ordinal':ordinal,'row':row})
    operations.sort(key=lambda r:(r['time'],{'birth':0,'return':1,'seed':2,'merger':3}[r['kind']],r['ordinal']))
    previous={};new_epochs={};next_epoch=max(graph.epochs,default=-1)+1;unavailable=[]
    # _atomic_batch rolls back nodes/edges; epochs are staged and restored too.
    prior_epochs=dict(graph.epochs)
    try:
        with graph._atomic_batch():
            for op_index,op in enumerate(operations):
                event_epoch=next_epoch+op_index
                contract=epoch_contract(op['time'],clock=clock) if comoving else isolated_epoch_contract(op['time'],time_unit_gyr=params['unit_time_s']/SECONDS_PER_YEAR/1.e9,snapshot=event_epoch)
                contract={**contract,'native_source_operation_ordinal':op_index,'snapshot_storage_index':None}
                graph.add_epoch(event_epoch,contract)
                evidence={'origin':'native_ancestry','source_history_sha256':checksum,'native_time':op['time'],'operator':op['kind'],'source_origin':report['source_origin']}
                event_namespace=particle_namespace+':native-source:'+str(op_index)
                def node(pid,kind,side,mass=None):
                    key=(event_namespace+':'+side,pid,event_epoch)
                    graph.add_node(*key,kind=kind,attributes={'native_particle_namespace':particle_namespace,'native_particle_id':pid,'measured_source_state':side,'mass_msun':mass})
                    if side=='before' and pid in previous:
                        graph.add_edge(previous[pid],key,'descendant',evidence={**evidence,'basis':'exact native source identity continuity'})
                    if side=='after':previous[pid]=key
                    return key
                if op['kind']=='birth':
                    parent=node(op['parent'],'parent_gas','before',op['before_mass']);child=node(op['child'],'ssp','after',op['after_mass']);graph.add_edge(parent,child,'birth_parent',evidence=evidence)
                elif op['kind']=='return':
                    donors=[r for r in op['rows'] if r['species']==4]
                    if len(donors)!=1:raise ValueError('Source history lacks a unique stellar donor')
                    donor=donors[0];source=node(op['source'],'ssp','before',donor['mass_before_msun']);target=node(op['source'],'ssp','after',donor['mass_after_msun']);graph.add_edge(source,target,'descendant',evidence=evidence)
                    for row in op['rows']:
                        if row['species']!=0:continue
                        pre=node(row['particle_id'],'gas','before',row['mass_before_msun']);post=node(row['particle_id'],'gas','after',row['mass_after_msun']);graph.add_edge(pre,post,'descendant',evidence=evidence)
                        graph.add_edge(source,post,'mass_return',evidence={**evidence,'source_counter':op['counter'],'returned_mass_msun':row['mass_after_msun']-row['mass_before_msun'],'elements_after_msun':row['elements_after_msun']})
                elif op['kind']=='seed':
                    row=op['row'];parent=node(row['particle_id'],'parent_gas' if row['previous_species']==0 else 'bh','before',row['gravity_mass_msun']);child=node(row['particle_id'],'bh','after',row['gravity_mass_msun'])
                    graph.add_edge(parent,child,'descendant',evidence={**evidence,'basis':'native ID-preserving seed conversion','seed_physical_mass_msun':row['physical_mass_msun']})
                else:
                    row=op['row'];parents=[node(pid,'bh','before',mass) for pid,mass in zip(row['parent_ids'],row['gravity_mass_before_msun'])]
                    child=node(row['particle_id'],'bh','after',row['gravity_mass_after_msun'])
                    for parent in parents:graph.add_edge(parent,child,'bh_merger',evidence={**evidence,'counter':row['counter'],'radiated_mass_msun':row['radiated_mass_msun'],'radiated_momentum_msun_km_s':row['radiated_momentum_msun_km_s']})
                    for pid in row['parent_ids']:
                        if pid!=row['particle_id']:previous.pop(pid,None)
            # Source states link to existing saved observations only when their
            # actual clock follows the final native operation for that identity.
            for pid,source in previous.items():
                candidates=[]
                source_time=operations[int(source[0].split(':native-source:')[-1].split(':')[0])]['time']
                for key,record in graph.nodes.items():
                    if key[0]!=particle_namespace or key[1]!=pid or record['status']=='MISSING':continue
                    time=graph.epochs[key[2]].get('actual_scale_factor' if comoving else 'native_time_code')
                    if time is not None and time>=source_time:candidates.append((time,key))
                if candidates:
                    key=min(candidates,key=lambda r:(r[0],r[1]))[1]
                    graph.add_edge(source,key,'descendant',evidence={'origin':'native_ancestry','basis':'exact native source ID to observed saved state','source_history_sha256':checksum})
                else:unavailable.append(pid)
            graph.validate()
            for source,sha in sources.items():
                if file_sha256(source)!=sha:raise ValueError('Native source lineage changed before insertion')
    except BaseException:
        graph.epochs=prior_epochs;raise
    return sources


def read_native_subhalo_spans(run, snapshot, *, native_sources, maximum_subhalos, maximum_particles, memory_gib):
    """Read SUBFIND spans and validate fine-DM ordering before bound-ID reads.

    Offsets are exact particle indices and lengths are counts in each of the
    six Gadget species. Fine-DM spans must be disjoint and fit the complete
    ordered type-1 snapshot. Ancestry, stellar ownership and SIDM recovery
    share this catalogue intake; their later selections remain separate.
    """
    import h5py
    groups = _group_files(Path(run), snapshot); snapshots = _snapshot_files(Path(run), snapshot)
    if not groups or not snapshots:
        raise ValueError('Native satellite continuity needs the complete saved SUBFIND/ordered snapshot')
    for p in groups + snapshots:
        if native_sources.get(str(p.resolve())) != file_sha256(p):
            raise ValueError('Native satellite catalogue/snapshot is outside its owned evolution')
    _check_file_series(groups, 'Header', 'Nsubhalos_Total')
    _check_file_series(snapshots, 'Header', 'NumPart_Total')
    with h5py.File(groups[0]) as f:
        count = int(f['Header'].attrs['Nsubhalos_Total'])
    if count > maximum_subhalos or count * 512 > memory_gib * 1024**3 / 4:
        raise MemoryError('Native satellite catalogue exceeds explicit census/allocation')
    fields = {k: _read_catalogue_series(groups, 'Subhalo', k) for k in
              ('SubhaloGroupNr', 'SubhaloRankInGr', 'SubhaloOffsetType', 'SubhaloLenType')}
    for k, array in fields.items():
        shape = (count, 6) if k.endswith('Type') else (count,)
        if array.shape != shape or array.dtype.kind not in 'ui' or np.any(array < 0):
            raise ValueError('Native satellite spans/ranks/hosts require aligned exact integers')
    from .data import snapshot_id_shards
    records, total = snapshot_id_shards(snapshots, 1)
    if total > maximum_particles:
        raise MemoryError('Native satellite particle census exceeds original allocation')
    lengths = fields['SubhaloLenType'][:, 1]; starts = fields['SubhaloOffsetType'][:, 1]
    active = np.flatnonzero(lengths > 0); active = active[np.argsort(starts[active])]
    # Python integer addition rejects uint64 wrap before offset arithmetic.
    ends = np.asarray([int(starts[i]) + int(lengths[i]) for i in active], dtype=object)
    if any(end > total for end in ends) or (len(active) > 1 and any(ends[:-1] > starts[active[1:]])):
        raise ValueError('Native satellite member spans overlap or exceed the ordered species population')
    return fields, snapshots, records


