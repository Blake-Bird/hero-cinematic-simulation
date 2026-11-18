"""Saved-snapshot scientific measurements, catalogue statistics and publication.

Readers validate complete snapshot sets and their stored unit/epoch contracts.
Selected products publish immutable HDF5/figure bundles; the legacy full suite
retains its separate entry point. Numerical arrays, support masks and missing
measurements are authoritative; plots do not supply absent physics.
See docs/architecture.md for the pipeline and docs/calculations.md for equations.
"""
from __future__ import annotations

# ========================================================================
# analyze: Post-processing for DarkMatterSim.

# Workflow and units
# Reads complete, metadata-checked snapshot/catalogue shards and writes results/analysis.h5 plus figures.
# Positions are comoving Mpc/h after conversion from kpc/h; masses are Msun/h. Snapshot peculiar velocities require the GADGET sqrt(a) conversion.
# Mass definitions, resolution cuts, saved epochs and missing fields are recorded rather than silently substituted. HMF equations live in hmf.py; solver conventions follow the bundled GADGET manual.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
from contextlib import contextmanager
import copy
import csv
import datetime
import json
import math
import os
import re
import sys
import tempfile
import textwrap
from decimal import Decimal
from pathlib import Path
from core.native import native_run_owner, directory_owner
import shutil
import uuid

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import fft as scipy_fft
from scipy.spatial import cKDTree
from scipy.stats import beta, chi2, t as student_t
from figures import apply_plot_style, export_figure
from core.data import snapshot_total_counts as _snapshot_total_counts
from core.data import convert_pinned_native,dataset_to_cgs_factor
from core.foundation import RHO_CRIT_MSUN_H_MPC_H3, MSUN_G, SECONDS_PER_YEAR
from core.foundation import file_sha256 as _file_sha256


ANALYSIS_PROVENANCE_SCHEMA = "darkmattersim-analysis-provenance-v1"


def solve_periodic_peculiar_potential(density_contrast, *, scale_factor,
        mean_density_proper_msun_kpc3, box_comoving_kpc, smoothing_comoving_kpc,
        memory_gib):
    """Periodic peculiar Phi in (km/s)^2, without the homogeneous source.

    ∇_x² Phi = 4πG a² rho_bar(proper) delta, x in comoving kpc.
    Proper acceleration = -∇_x Phi/a; proper Hessian = ∂x_i∂x_j Phi/a².
    The DC mode is zero. Odd meshes avoid Nyquist derivative ambiguity.
    Gaussian smoothing acts on the source; there is no CIC deconvolution.
    """
    from core.foundation import G_MPC_KMS2_MSUN
    delta = np.asarray(density_contrast,dtype=float)
    if (delta.ndim!=3 or len(set(delta.shape))!=1 or delta.shape[0]<5
            or delta.shape[0]%2!=1 or not np.isfinite(delta).all()
            or np.min(delta)<-1-1e-12):
        raise ValueError('Peculiar potential requires a finite odd cubic contrast mesh with delta >= -1')
    parameters = [scale_factor,mean_density_proper_msun_kpc3,box_comoving_kpc,memory_gib]
    if (any(isinstance(v,(bool,np.bool_)) or not np.isfinite(v) or v<=0 for v in parameters)
            or isinstance(smoothing_comoving_kpc,(bool,np.bool_)) or not np.isfinite(smoothing_comoving_kpc)
            or smoothing_comoving_kpc<0 or smoothing_comoving_kpc>=box_comoving_kpc/2):
        raise ValueError('Explicit positive epoch, mean density, box, memory and supported smoothing required')
    if delta.size*512+16*1024**2 > memory_gib*1024**3:
        raise MemoryError('Peculiar field and eigenstructure exceed declared workspace')
    dc = float(delta.mean())
    if abs(dc)>1e-10*max(1.,float(np.max(abs(delta)))):
        raise ValueError('Density contrast must have zero mean; do not reinterpret a density grid as contrast')
    n = delta.shape[0]
    wave = 2*np.pi*scipy_fft.fftfreq(n,d=box_comoving_kpc/n)
    k = [wave[:,None,None],wave[None,:,None],wave[None,None,:]]
    k2 = k[0]**2+k[1]**2+k[2]**2
    source_k = scipy_fft.fftn(delta-dc,workers=1)
    source_k *= np.exp(-.5*k2*smoothing_comoving_kpc**2)
    source_k[0,0,0] = 0
    g = G_MPC_KMS2_MSUN*1000
    coefficient = 4*np.pi*g*mean_density_proper_msun_kpc3
    phi_k = np.zeros_like(source_k)
    np.divide(-coefficient*scale_factor**2*source_k,k2,out=phi_k,where=k2>0)
    phi = scipy_fft.ifftn(phi_k,workers=1).real
    smoothed = scipy_fft.ifftn(source_k,workers=1).real
    acceleration = np.empty(delta.shape+(3,))
    hessian = np.empty(delta.shape+(3,3))
    for i in range(3):
        acceleration[...,i] = scipy_fft.ifftn(-1j*k[i]*phi_k/scale_factor,workers=1).real
        for j in range(i+1):
            hessian[...,i,j] = hessian[...,j,i] = scipy_fft.ifftn(-k[i]*k[j]*phi_k/scale_factor**2,workers=1).real
    trace = np.trace(hessian,axis1=-2,axis2=-1)
    expected = coefficient*smoothed
    poisson_error = float(np.max(abs(trace-expected)))
    if poisson_error>1e-11*max(1.,float(np.max(abs(expected)))):
        raise ArithmeticError('Periodic Hessian failed its Poisson trace identity')
    trace_free = hessian-trace[...,None,None]*np.eye(3)/3
    return {'potential_km2_s2':phi,'acceleration_km2_s2_proper_kpc':acceleration,
            'hessian_km2_s2_proper_kpc2':hessian,'trace_free_hessian_km2_s2_proper_kpc2':trace_free,
            'density_contrast_smoothed':smoothed,'filtered_density_supported':smoothed>=-1-1e-12,
            'dimensionless_tidal_hessian':hessian/coefficient,
            'metadata':{'schema':'periodic-peculiar-field-v1','scale_factor':float(scale_factor),
                'mean_density_proper_msun_kpc3':float(mean_density_proper_msun_kpc3),
                'box_comoving_kpc':float(box_comoving_kpc),'smoothing_comoving_kpc':float(smoothing_comoving_kpc),
                'source_equation':'laplacian_comoving Phi = 4 pi G a^2 rho_bar_proper delta',
                'hessian_sign':'positive eigenvalues mean compression; acceleration Jacobian is minus the Hessian',
                'source_filter':'Gaussian exp(-k_comoving^2 R_comoving^2/2); no CIC deconvolution',
                'minimum_smoothed_contrast':float(np.min(smoothed)),
                'negative_smoothed_density_cells':int(np.sum(smoothed < -1-1e-12)),
                'density_filter_positivity':('UNRESOLVED_SPECTRAL_RINGING' if np.any(smoothed < -1-1e-12) else 'NONNEGATIVE'),
                'zero_mode':'homogeneous density removed; mean Phi=0','odd_mesh_required':True,
                'maximum_poisson_trace_error':poisson_error,'science_acceptance':'NOT_ACCEPTED'}}


def classify_eigenstructure(dimensionless_hessian, *, threshold, eigenvalue_gap_floor):
    """T-web classification from the FULL signed tensor, never trace-free."""
    tensor = np.asarray(dimensionless_hessian,dtype=float)
    if (tensor.shape[-2:]!=(3,3) or not np.isfinite(tensor).all()
            or not np.allclose(tensor,np.swapaxes(tensor,-1,-2),rtol=1e-12,atol=1e-14)
            or isinstance(threshold,bool) or not np.isfinite(threshold)
            or isinstance(eigenvalue_gap_floor,bool) or not np.isfinite(eigenvalue_gap_floor) or eigenvalue_gap_floor<=0):
        raise ValueError('Finite symmetric dimensionless full Hessian, threshold and degeneracy floor required')
    values,axes = np.linalg.eigh(tensor)
    classification = np.sum(values>threshold,axis=-1).astype(np.uint8)
    gaps = np.diff(values,axis=-1)
    axis_supported = np.stack([gaps[...,0]>=eigenvalue_gap_floor,
        np.min(gaps,axis=-1)>=eigenvalue_gap_floor,gaps[...,1]>=eigenvalue_gap_floor],axis=-1)
    return {'web_class':classification,'eigenvalues_ascending':values,'eigenvectors_columns':axes,
            'axis_supported':axis_supported,'filament_axis_supported':(classification==2)&axis_supported[...,0],
            'metadata':{'schema':'full-hessian-t-web-v1','threshold':float(threshold),
                'eigenvalue_gap_floor':float(eigenvalue_gap_floor),'class_names':['void','sheet','filament','knot'],
                'normalization':'H_proper/(4 pi G rho_bar_proper)',
                'filament_direction':'least-compressive eigenvector, only when exactly two eigenvalues exceed threshold'}}


def sample_periodic_tensor_grid(grid, positions_comoving_kpc, box_comoving_kpc):
    """Trilinear Cartesian tensor interpolation; eigensystems are fit AFTER it."""
    data = np.asarray(grid,dtype=float)
    p = np.asarray(positions_comoving_kpc,dtype=float)
    if (data.ndim<3 or len(set(data.shape[:3]))!=1 or not np.isfinite(data).all()
            or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
            or not np.isfinite(box_comoving_kpc) or box_comoving_kpc<=0):
        raise ValueError('Finite cubic periodic data and N by 3 comoving queries required')
    n = data.shape[0]
    coordinate = (p%box_comoving_kpc)*n/box_comoving_kpc
    base = np.floor(coordinate).astype(np.int64)
    fraction = coordinate-base
    output = np.zeros((len(p),)+data.shape[3:])
    for dx in [0,1]:
        for dy in [0,1]:
            for dz in [0,1]:
                step = np.array([dx,dy,dz])
                weight = np.prod(np.where(step,fraction,1-fraction),axis=1)
                values = data[(base[:,0]+dx)%n,(base[:,1]+dy)%n,(base[:,2]+dz)%n]
                output += values*weight.reshape((-1,)+(1,)*(data.ndim-3))
    return output


def measure_periodic_filament_regions(web_class, *, box_comoving_kpc, density_supported):
    """Six-connected class-2 voxel regions, including periodic face joins.

    These are threshold-defined regions, not a ridge skeleton, unique physical
    filament ancestry, or the 1DREAM extraction used in the alignment paper.
    """
    from scipy.ndimage import label,generate_binary_structure
    classification = np.asarray(web_class)
    supported = np.asarray(density_supported,dtype=bool)
    if (classification.ndim!=3 or len(set(classification.shape))!=1
            or supported.shape!=classification.shape or classification.dtype.kind not in 'iu'
            or np.any(classification>3) or not np.isfinite(box_comoving_kpc) or box_comoving_kpc<=0):
        raise ValueError('Explicit cubic web classes, density support and positive periodic comoving box required')
    regions,count = label((classification==2)&supported,generate_binary_structure(3,1))
    parent = np.arange(count+1)
    def find(value):
        while parent[value]!=value:
            parent[value] = parent[parent[value]];value = parent[value]
        return value
    for axis in range(3):
        left,right = np.take(regions,0,axis=axis),np.take(regions,-1,axis=axis)
        pairs = np.unique(np.column_stack([left.ravel(),right.ravel()]),axis=0)
        for one,two in pairs:
            if one and two:
                first,second = find(one),find(two)
                parent[max(first,second)] = min(first,second)
    roots = np.array([find(value) for value in range(count+1)])
    unique = np.unique(roots[1:]);mapping = np.zeros(count+1,dtype=np.uint64)
    if len(unique):mapping[1:] = np.searchsorted(unique,roots[1:]).astype(np.uint64)+1
    cells = mapping[regions]
    voxels = np.argwhere(cells>0)
    n = classification.shape[0];spacing = box_comoving_kpc/n
    population = np.bincount(cells.ravel().astype(np.int64),minlength=len(unique)+1)[1:]
    return {'component_ids':cells,'voxel_positions_comoving_kpc':voxels*spacing,
        'voxel_count':population,'component_volume_comoving_kpc3':population*spacing**3,
        'metadata':{'definition':'periodic face-connected full-Hessian class-2 cells with supported filtered density',
            'connectivity':6,'component_count':len(unique),'voxel_position':'spectral grid nodes, not shifted CIC centers',
            'grid_shape':list(classification.shape),'published_label_storage':'flat C-order uint64 vector; 0 means no classified region',
            'nearest_voxel_distance_error_bound_comoving_kpc':np.sqrt(3)*spacing/2,
            'identity_scope':'product/epoch/smoothing-local connected labels; no cross-epoch filament identity inferred',
            'geometry_scope':'voxel union around classified grid nodes; neither ridge skeleton nor 1DREAM'}}


def measure_filament_proximity(regions, queries_comoving_kpc, *, box_comoving_kpc):
    queries = np.asarray(queries_comoving_kpc,dtype=float)
    if queries.ndim!=2 or queries.shape[1]!=3 or not np.isfinite(queries).all():
        raise ValueError('Finite actual comoving filament queries required')
    positions = regions['voxel_positions_comoving_kpc']
    if not len(positions):
        return {'nearest_node_distance_comoving_kpc':np.full(len(queries),np.nan),
                'distance_lower_comoving_kpc':np.full(len(queries),np.nan),
                'distance_upper_comoving_kpc':np.full(len(queries),np.nan),'supported':np.zeros(len(queries),dtype=bool)}
    tree = cKDTree(positions,boxsize=box_comoving_kpc)
    distance,_ = tree.query(queries%box_comoving_kpc)
    bound = regions['metadata']['nearest_voxel_distance_error_bound_comoving_kpc']
    return {'nearest_node_distance_comoving_kpc':distance,
        'distance_lower_comoving_kpc':np.maximum(0,distance-bound),
        'distance_upper_comoving_kpc':distance,'supported':np.ones(len(queries),dtype=bool)}


def measure_filament_axis_alignment(vectors, axes, axis_supported, *, randomized_realizations, seed):
    """Alignment to unoriented filament axes; no preferred spine direction exists.

    Unsupported axes may be missing; a supported direction must be finite and
    nonzero. Zero-amplitude vectors remain unresolved, including degenerate
    shape orientations explicitly supplied as zero by the source consumer.
    """
    vectors = np.asarray(vectors,dtype=float)
    axes = np.asarray(axes,dtype=float)
    supported = np.asarray(axis_supported,dtype=bool)
    if (vectors.ndim!=2 or vectors.shape[1]!=3 or axes.shape!=vectors.shape
            or supported.shape!=(len(vectors),) or not np.isfinite(vectors).all()
            or type(randomized_realizations) is not int or not 32<=randomized_realizations<=10000
            or type(seed) is not int or not 0<=seed<2**64):
        raise ValueError('Aligned vector/axis population and explicit null realization count and seed required')
    norm = np.linalg.norm(axes,axis=1)
    if np.any(supported & (~np.isfinite(axes).all(axis=1) | (norm <= 0))):
        raise ValueError('Supported filament axes must be finite nonzero directions')
    unit = np.zeros_like(axes)
    unit[supported] = axes[supported]/norm[supported,None]
    amplitude = np.linalg.norm(vectors,axis=1)
    good = supported&(amplitude>0)
    cosine = np.full(len(vectors),np.nan)
    cosine[good] = abs(np.sum(vectors[good]/amplitude[good,None]*unit[good],axis=1))
    cosine = np.clip(cosine,0,1)
    rng = np.random.Generator(np.random.Philox(seed))
    null = np.full(randomized_realizations,np.nan)
    if good.any():
        for index in range(randomized_realizations):
            directions = rng.normal(size=(good.sum(),3))
            directions /= np.linalg.norm(directions,axis=1)[:,None]
            null[index] = np.mean(abs(np.sum(directions*unit[good],axis=1)))
    return {'cosine_unsigned':cosine,'angle_unsigned_deg':np.degrees(np.arccos(cosine)),
            'supported':good,'isotropic_null_mean_cosine':null,
            'metadata':{'seed':seed,'randomized_realizations':randomized_realizations,
                'supported_count':int(good.sum()),'isotropic_expected_mean_cosine':.5,
                'null_scope':'independent isotropic orientation draws at fixed local axes; excludes spatial/sample covariance',
                'status':'MEASURED' if good.any() else 'EMPTY_SUPPORTED_SELECTION'}}


def measure_spin_shape_filament_angles(vectors, eigensystem, *, randomized_realizations, seed):
    """T-web-specific adapter; topology-based axes retain separate definitions."""
    axes = np.asarray(eigensystem['eigenvectors_columns'],dtype=float)
    if axes.shape != (len(vectors),3,3):
        raise ValueError('Aligned T-web eigensystems required')
    return measure_filament_axis_alignment(vectors,axes[:,:,0],
        eigensystem['filament_axis_supported'],
        randomized_realizations=randomized_realizations,seed=seed)


def uncertainty_aware_log_density_derivative(radius_proper_kpc, density_msun_kpc3,
        log_density_covariance, *, window_bins, polynomial_degree, minimum_effective_count,
        effective_count):
    """Local GLS derivative and its exact linear propagated covariance.

    Unsupported/empty bins divide the radial support into separate segments;
    a smoothing window never crosses a missing bin. No log-zero imputation.
    """
    radius,density,count = map(lambda v:np.asarray(v,dtype=float),
        [radius_proper_kpc,density_msun_kpc3,effective_count])
    covariance = np.asarray(log_density_covariance,dtype=float)
    n = len(radius)
    if (radius.ndim!=1 or n<1 or density.shape!=radius.shape or count.shape!=radius.shape
            or not np.isfinite(radius).all() or np.any(radius<=0) or np.any(np.diff(radius)<=0)
            or not np.isfinite(count).all() or np.any(count<0)
            or covariance.shape!=(n,n) or not np.isfinite(covariance).all()
            or not np.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-14)
            or type(window_bins) is not int or window_bins<5 or window_bins%2!=1
            or type(polynomial_degree) is not int or not 2<=polynomial_degree<window_bins-1
            or not np.isfinite(minimum_effective_count) or minimum_effective_count<=0):
        raise ValueError('Declare increasing physical radii, aligned counts, covariance and supported local-polynomial controls')
    eig = np.linalg.eigvalsh(covariance)
    if eig[0]<-1e-12*max(1.,float(eig[-1])):
        raise ValueError('Log-density covariance is not positive semidefinite')
    valid = np.isfinite(density)&(density>0)&(count>=minimum_effective_count)&(np.diag(covariance)>0)
    operator = np.zeros((n,n))
    supported = np.zeros(n,dtype=bool)
    logarithm = np.log(radius)
    indices = np.flatnonzero(valid)
    groups = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    for group in groups:
        if len(group)<window_bins:continue
        for local,index in enumerate(group):
            start = min(max(0,local-window_bins//2),len(group)-window_bins)
            selected = group[start:start+window_bins]
            design = (logarithm[selected]-logarithm[index])[:,None]**np.arange(polynomial_degree+1)
            cov = covariance[np.ix_(selected,selected)]
            precision = np.linalg.pinv(cov,hermitian=True)
            normal = design.T @ precision @ design
            if np.linalg.matrix_rank(normal)<polynomial_degree+1:continue
            weights = np.linalg.solve(normal,design.T @ precision)[1]
            operator[index,selected] = weights
            supported[index] = True
    logs = np.zeros(n);logs[valid] = np.log(density[valid])
    slope = operator @ logs
    propagated = operator @ covariance @ operator.T
    slope[~supported] = np.nan
    error = np.sqrt(np.maximum(np.diag(propagated),0));error[~supported] = np.nan
    return {'logarithmic_slope':slope,'slope_standard_error':error,
            'slope_covariance':propagated,'derivative_operator':operator,'supported':supported,
            'metadata':{'window_bins':window_bins,'polynomial_degree':polynomial_degree,
                'definition':'GLS local polynomial in ln rho versus ln proper r; derivative at each saved bin center',
                'uncertainty_scope':'linear propagation of supplied log-density covariance; not simulation convergence'}}


def locate_supported_slope_feature(radius_proper_kpc, slope_product, *,
        search_interval_proper_kpc, minimum_depth, bootstrap_realizations, seed,
        maximum_fractional_radius_error, minimum_success_fraction):
    """Detect ambiguous valleys, quantify radius/depth/half-depth width.

    The operational baseline is the lower of the adjacent shoulders, and width
    is measured in ln r at half prominence. This is recorded separately from
    the paper's parametric turnover-based depth and width definition.
    """
    from scipy.signal import find_peaks,peak_widths
    radius = np.asarray(radius_proper_kpc,dtype=float)
    slope = np.asarray(slope_product['logarithmic_slope'],dtype=float)
    supported = np.asarray(slope_product['supported'],dtype=bool)
    covariance = np.asarray(slope_product['slope_covariance'],dtype=float)
    interval = np.asarray(search_interval_proper_kpc,dtype=float)
    if (radius.ndim!=1 or radius.shape!=slope.shape or supported.shape!=radius.shape
            or covariance.shape!=(len(radius),len(radius)) or interval.shape!=(2,)
            or not np.isfinite(radius).all() or np.any(radius<=0) or np.any(np.diff(radius)<=0)
            or not np.isfinite(interval).all() or not 0<interval[0]<interval[1]
            or not np.isfinite(minimum_depth) or minimum_depth<=0
            or type(bootstrap_realizations) is not int or not 32<=bootstrap_realizations<=10000
            or type(seed) is not int or not 0<=seed<2**64
            or not 0<maximum_fractional_radius_error<1 or not 0<minimum_success_fraction<=1):
        raise ValueError('Explicit supported feature search, depth, uncertainty and seed controls required')
    if not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-14):
        raise ValueError('Finite symmetric slope covariance required')
    eig,axes = np.linalg.eigh((covariance+covariance.T)/2)
    if eig[0]<-1e-12*max(1.,float(eig[-1])):
        raise ValueError('Slope covariance is not positive semidefinite')
    selected = supported&np.isfinite(slope)
    def candidates(values):
        output = []
        for group in np.split(np.flatnonzero(selected),np.flatnonzero(np.diff(np.flatnonzero(selected))>1)+1):
            if len(group)<5:continue
            peaks,properties = find_peaks(-values[group],prominence=minimum_depth)
            if not len(peaks):continue
            widths,height,left,right = peak_widths(-values[group],peaks,rel_height=.5,
                prominence_data=(properties['prominences'],properties['left_bases'],properties['right_bases']))
            for k,peak in enumerate(peaks):
                index = int(group[peak])
                if not interval[0]<=radius[index]<=interval[1]:continue
                xp = np.arange(len(group))
                lo = float(np.exp(np.interp(left[k],xp,np.log(radius[group]))))
                hi = float(np.exp(np.interp(right[k],xp,np.log(radius[group]))))
                output.append({'index':index,'radius_proper_kpc':float(radius[index]),
                    'minimum_logarithmic_slope':float(values[index]),'depth':float(properties['prominences'][k]),
                    'half_depth_inner_radius_proper_kpc':lo,'half_depth_outer_radius_proper_kpc':hi,
                    'half_depth_width_ln_radius':float(np.log(hi/lo)),
                    'radius_bin_bracket_proper_kpc':[float(radius[group[peak-1]]),float(radius[group[peak+1]])]})
        return output
    measured = candidates(slope)
    result = {'schema':'nonparametric-splashback-feature-v1','candidates':measured,
              'status':'AMBIGUOUS_MINIMA' if len(measured)>1 else 'NO_SUPPORTED_FEATURE' if not measured else 'MEASURED',
              'definition':'sampled radius of supported slope valley; prominence and half-prominence width in ln r',
              'paper_metric_equivalence':'not claimed; this uncertainty-aware nonparametric definition is distinct from fitted turnover-based depth',
              'science_acceptance':'NOT_ACCEPTED'}
    if len(measured)!=1:
        result['sampling_uncertainty']={'status':'UNAVAILABLE_UNRESOLVED_FEATURE'}
        return result
    rng = np.random.Generator(np.random.Philox(seed))
    transform = axes*np.sqrt(np.maximum(eig,0))
    samples = []
    failures = {'NO_FEATURE':0,'AMBIGUOUS':0}
    for _ in range(bootstrap_realizations):
        values = slope + transform @ rng.normal(size=len(radius))
        found = candidates(values)
        if len(found)!=1:
            failures['NO_FEATURE' if not found else 'AMBIGUOUS'] += 1;continue
        row = found[0];samples.append([row['radius_proper_kpc'],row['depth'],row['half_depth_width_ln_radius']])
    fraction = len(samples)/bootstrap_realizations
    uncertainty = {'status':'ESTIMATED' if fraction>=minimum_success_fraction else 'INSUFFICIENT_SUCCESS',
                   'definition':'correlated Gaussian log-density covariance propagated through the linear derivative operator',
                   'realizations':bootstrap_realizations,'seed':seed,'success_fraction':fraction,'failure_counts':failures}
    if samples:
        percentile = np.percentile(samples,[16,50,84],axis=0)
        fraction_error = float((percentile[2,0]-percentile[0,0])/(2*measured[0]['radius_proper_kpc']))
        uncertainty.update(radius_depth_width_percentile_16_50_84=percentile.tolist(),
                           fractional_radius_error=fraction_error)
        if fraction_error>maximum_fractional_radius_error or fraction<minimum_success_fraction:
            result['status']='UNCERTAINTY_UNRESOLVED'
    else:result['status']='UNCERTAINTY_UNRESOLVED'
    result['sampling_uncertainty']=uncertainty
    return result


def filament_transverse_profiles(density_grid, centers_mpc_h, tangents, radii_mpc_h, *,
        box_mpc_h, grid_origin_mpc_h, background_radius_range_mpc_h, angular_samples=32,
        excluded_centers=None, monotonic_tolerance=0., memory_bytes=64*1024**2):
    """Sample the unsmoothed scientific grid in tangent-normal planes.

    Grid origin is the position of element [0,0,0], not a guessed cell edge.
    Return the complete angle/radius samples and a declared half-excess diameter.
    Background is the mean over the supplied outer annulus. Off-axis peaks,
    rebounds, unresolved crossings and excluded junctions have no reported width.
    This geometric estimator does not certify field or topology convergence.
    """
    from scipy.ndimage import map_coordinates
    field = np.asarray(density_grid)
    centers = np.asarray(centers_mpc_h, dtype=float)
    direction = np.asarray(tangents, dtype=float)
    radii = np.asarray(radii_mpc_h, dtype=float)
    origin = np.asarray(grid_origin_mpc_h, dtype=float)
    background_range = np.asarray(background_radius_range_mpc_h, dtype=float)
    if (field.ndim != 3 or field.dtype.kind not in 'fiu' or min(field.shape) < 2
            or not np.isfinite(field).all() or np.any(field < 0)):
        raise ValueError('Finite nonnegative three-dimensional scientific density required')
    if (centers.ndim != 2 or centers.shape[1] != 3 or direction.shape != centers.shape
            or not np.isfinite(centers).all() or origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError('Aligned centers, tangents and declared grid origin required')
    if (not np.isfinite(box_mpc_h) or box_mpc_h <= 0 or radii.ndim != 1 or len(radii) < 4
            or not np.isfinite(radii).all() or radii[0] != 0 or np.any(np.diff(radii) <= 0)
            or radii[-1] >= box_mpc_h/2):
        raise ValueError('Ordered radii starting at zero and below half a box required')
    if (background_range.shape != (2,) or not np.isfinite(background_range).all()
            or not 0 < background_range[0] < background_range[1] <= radii[-1]):
        raise ValueError('Explicit resolved background annulus required')
    outer = (radii >= background_range[0]) & (radii <= background_range[1])
    if np.count_nonzero(outer) < 2:
        raise ValueError('Background annulus needs two or more sampled radii')
    if type(angular_samples) is not int or angular_samples < 8 or angular_samples > 4096:
        raise ValueError('Angular sample count must be between 8 and 4096')
    if not np.isfinite(monotonic_tolerance) or monotonic_tolerance < 0:
        raise ValueError('Nonnegative density-excess monotonicity tolerance required')
    excluded = np.zeros(len(centers), dtype=bool) if excluded_centers is None else np.asarray(excluded_centers)
    if excluded.shape != (len(centers),) or excluded.dtype.kind != 'b':
        raise ValueError('Explicit boolean junction/overlap exclusion mask required')
    # Account for persistent output and the one-center coordinate workspace.
    allocation = len(centers)*len(radii)*(angular_samples+3)*8 + len(radii)*angular_samples*128
    if type(memory_bytes) is not int or memory_bytes < allocation:
        raise MemoryError('Transverse profile workspace exceeds declared budget')
    angles = 2*np.pi*np.arange(angular_samples)/angular_samples
    samples = np.full((len(centers), len(radii), angular_samples), np.nan)
    status = np.full(len(centers), 'UNSUPPORTED_TANGENT', dtype='U32')
    for row, (center, tangent) in enumerate(zip(centers, direction)):
        norm = np.linalg.norm(tangent)
        if not np.isfinite(norm) or norm <= 64*np.finfo(float).eps:
            continue
        tangent = tangent/norm
        anchor = np.eye(3)[np.argmin(np.abs(tangent))]
        normal = np.cross(tangent, anchor); normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        ring = np.cos(angles)[:, None]*normal + np.sin(angles)[:, None]*binormal
        coordinates = center + radii[:, None, None]*ring[None, :, :]
        grid = np.mod((coordinates-origin)/box_mpc_h, 1.)*np.asarray(field.shape)
        samples[row] = map_coordinates(field, grid.reshape(-1, 3).T, output=np.float64, order=1,
            mode='grid-wrap', prefilter=False).reshape(len(radii), angular_samples)
        status[row] = 'EXCLUDED_JUNCTION' if excluded[row] else 'WIDTH_UNRESOLVED'
    profile = np.mean(samples, axis=2)
    background = np.mean(profile[:, outer], axis=1)
    excess = profile-background[:, None]
    diameter = np.full(len(centers), np.nan)
    for row in range(len(centers)):
        if status[row] != 'WIDTH_UNRESOLVED':
            continue
        e = excess[row]
        if e[0] <= 0:
            status[row] = 'NO_CENTRAL_EXCESS'; continue
        inner = radii < background_range[0]
        if np.any(np.diff(e[inner]) > monotonic_tolerance):
            status[row] = 'MULTIMODAL_PROFILE'; continue
        crossings = np.flatnonzero((e[:-1] > .5*e[0]) & (e[1:] <= .5*e[0]))
        if len(crossings) != 1:
            continue
        i = int(crossings[0])
        if radii[i+1] >= background_range[0]:
            continue
        width = radii[i]+(.5*e[0]-e[i])/(e[i+1]-e[i])*(radii[i+1]-radii[i])
        if width < box_mpc_h/max(field.shape):
            status[row] = 'SUBCELL_WIDTH'; continue
        diameter[row] = 2*width; status[row] = 'MEASURED'
    return {'radii_mpc_h': radii, 'angles_radians': angles, 'density_samples': samples,
        'mean_density_profile': profile, 'background_density': background, 'density_excess': excess,
        'half_excess_diameter_mpc_h': diameter, 'status': status,
        'width_supported': status == 'MEASURED',
        'definition': 'twice the first half-central-excess crossing; monotone pre-background profile',
        'background_definition': 'equal-radius and equal-angle mean over declared outer annulus',
        'science_acceptance': 'NOT_ACCEPTED'}


def measure_filament_graph(points_mpc_h, arcs, node_ids, endpoint_node_ids, *,
                           box_mpc_h, arc_image_offsets):
    """Geometry of declared native arcs; topology is never inferred from rounding.

    Arc images are explicit integer offsets. Returned distances are comoving
    Mpc/h. Curvature is inverse comoving Mpc/h, with endpoints unresolved.
    """
    from core.data import exact_uint64_ids
    nodes = exact_uint64_ids(node_ids)
    endpoints = np.asarray(endpoint_node_ids)
    points = np.asarray(points_mpc_h, dtype=float)
    if not np.isfinite(box_mpc_h) or box_mpc_h <= 0:
        raise ValueError('Positive periodic box required')
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite three-dimensional filament points required')
    if endpoints.shape != (len(arcs), 2) or endpoints.dtype.kind not in 'iu' or np.any(endpoints < 0):
        raise ValueError('Exact endpoint node IDs required for each native arc')
    endpoints = endpoints.astype(np.uint64, copy=False)
    if not np.isin(endpoints, nodes).all() or len(arc_image_offsets) != len(arcs):
        raise ValueError('Arc refers to absent nodes or image offsets')
    degree = np.zeros(len(nodes), dtype=np.uint64)
    lookup = {int(value): i for i, value in enumerate(nodes)}
    length, paths, tangents, curvature = [], [], [], []
    starts, stops, arc_indices, arclength_start = [], [], [], []
    for i, (indices, images) in enumerate(zip(arcs, arc_image_offsets)):
        indices = np.asarray(indices)
        images = np.asarray(images)
        if indices.ndim != 1 or len(indices) < 2 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= len(points)):
            raise ValueError('Arc needs valid ordered point indices')
        if images.shape != (len(indices), 3) or images.dtype.kind not in 'iu':
            raise ValueError('Recorded integer images must align with arc points')
        path = points[indices]+images*box_mpc_h
        steps = np.diff(path, axis=0)
        if np.any(np.abs(steps) >= box_mpc_h/2):
            raise ValueError('Arc segments must resolve their periodic image within half a box')
        distances = np.linalg.norm(steps, axis=1)
        if np.any(distances <= 0):
            raise ValueError('Repeated adjacent filament points do not define a tangent')
        direction = steps/distances[:, None]
        tangent = np.full(path.shape, np.nan)
        tangent[0], tangent[-1] = direction[0], direction[-1]
        curve = np.full(len(path), np.nan)
        if len(path) > 2:
            middle = direction[:-1]+direction[1:]
            norm = np.linalg.norm(middle, axis=1)
            good = norm > 64*np.finfo(float).eps
            tangent[1:-1][good] = middle[good]/norm[good, None]
            values = np.linalg.norm(np.diff(direction, axis=0), axis=1)/(.5*(distances[:-1]+distances[1:]))
            curve[1:-1][good] = values[good]
        for node in endpoints[i]:
            degree[lookup[int(node)]] += np.uint64(1)
        length.append(float(distances.sum()))
        paths.append(path)
        tangents.append(tangent)
        curvature.append(curve)
        starts.extend(path[:-1]); stops.extend(path[1:]); arc_indices.extend([i]*len(steps))
        arclength_start.extend(np.r_[0., np.cumsum(distances)[:-1]])
    return {'node_ids': nodes, 'node_degree': degree, 'arc_length_mpc_h': np.asarray(length),
        'arc_paths_mpc_h': paths, 'arc_tangents': tangents, 'arc_curvature_h_mpc': curvature,
        'segment_start_mpc_h': np.asarray(starts).reshape(-1, 3),
        'segment_stop_mpc_h': np.asarray(stops).reshape(-1, 3),
        'segment_arc_index': np.asarray(arc_indices, dtype=np.int64),
        'segment_arclength_start_mpc_h': np.asarray(arclength_start),
        'box_mpc_h': float(box_mpc_h), 'degree_definition': 'incident arcs; loops contribute twice',
        'science_acceptance': 'NOT_ACCEPTED'}


def nearest_filament_segments(queries_mpc_h, geometry, *, memory_bytes=64*1024**2,
                              maximum_candidates=2, ambiguity_distance_mpc_h=0.):
    """Periodic projections retaining nearest distinct-arc alternatives.

    Candidate rank is deterministic (native segment order breaks exact ties).
    Missing ranks use -1 indices, NaN measurements and a false support mask.
    Ambiguity is a geometric diagnostic, never a probabilistic membership.
    """
    import itertools
    query = np.asarray(queries_mpc_h, dtype=float)
    start, stop = geometry['segment_start_mpc_h'], geometry['segment_stop_mpc_h']
    box = geometry['box_mpc_h']
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all() or not len(start):
        raise ValueError('Finite query positions and nonempty resolved graph required')
    if (type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 64
            or isinstance(ambiguity_distance_mpc_h, bool)
            or not np.isfinite(ambiguity_distance_mpc_h) or ambiguity_distance_mpc_h < 0):
        raise ValueError('Bounded candidate count and nonnegative ambiguity distance required')
    output_bytes = len(query)*(maximum_candidates*128+128)
    if type(memory_bytes) is not int or memory_bytes < len(start)*256+output_bytes:
        raise MemoryError('Filament nearest-segment workspace too small')
    chunk = max(1, min(64, (memory_bytes-output_bytes)//(len(start)*256)))
    vector = stop-start
    norm2 = np.sum(vector*vector, axis=1)
    chosen, distances, fraction, ambiguous = [], [], [], []
    candidate_indices = np.full((len(query), maximum_candidates), -1, dtype=np.int64)
    candidate_fraction = np.full(candidate_indices.shape, np.nan)
    candidate_distance = np.full(candidate_indices.shape, np.nan)
    for first in range(0, len(query), chunk):
        q = query[first:first+chunk]
        best = np.full((len(q), len(start)), np.inf)
        along = np.zeros_like(best)
        offset = q[:, None, :]-start[None, :, :]
        offset -= box*np.rint(offset/box)
        for image in itertools.product((-1, 0, 1), repeat=3):
            relative = offset+np.asarray(image)*box
            u = np.clip(np.sum(relative*vector, axis=2)/norm2, 0., 1.)
            distance2 = np.sum((relative-u[:, :, None]*vector)**2, axis=2)
            better = distance2 < best
            best[better], along[better] = distance2[better], u[better]
        indices = np.argmin(best, axis=1)
        row = np.arange(len(q))
        chosen.extend(indices); distances.extend(np.sqrt(best[row, indices])); fraction.extend(along[row, indices])
        tolerance = 64*np.finfo(float).eps*np.maximum(1., best[row, indices])
        ties = best <= (np.sqrt(best[row, indices, None])+ambiguity_distance_mpc_h)**2+tolerance[:, None]
        arc = geometry['segment_arc_index']
        ambiguous.extend(np.any(ties & (arc[None, :] != arc[indices, None]), axis=1))
        # Retain the nearest segment per distinct arc, not adjacent segments of
        # the same arc. Work is bounded by the declared candidate count.
        remaining = best.copy()
        for rank in range(maximum_candidates):
            selected = np.argmin(remaining, axis=1)
            supported = np.isfinite(remaining[row, selected])
            output_rows = first+row[supported]
            candidate_indices[output_rows, rank] = selected[supported]
            candidate_fraction[output_rows, rank] = along[row[supported], selected[supported]]
            candidate_distance[output_rows, rank] = np.sqrt(best[row[supported], selected[supported]])
            remaining[(arc[None, :] == arc[selected, None]) & supported[:, None]] = np.inf
    chosen = np.asarray(chosen, dtype=np.int64)
    selected_vectors = vector[chosen]
    segment_length = np.linalg.norm(selected_vectors, axis=1)
    if len(chosen):
        tangent = selected_vectors/segment_length[:, None]
    else:
        tangent = np.empty((0, 3))
    valid = candidate_indices >= 0
    safe_indices = np.maximum(candidate_indices, 0)
    candidate_arcs = np.where(valid, geometry['segment_arc_index'][safe_indices], -1)
    candidate_lengths = np.sqrt(norm2[safe_indices])
    candidate_arclength = geometry['segment_arclength_start_mpc_h'][safe_indices]+candidate_fraction*candidate_lengths
    candidate_tangent = np.where(valid[..., None], vector[safe_indices]/candidate_lengths[..., None], np.nan)
    return {'segment_index': chosen, 'arc_index': geometry['segment_arc_index'][chosen],
        'distance_mpc_h': np.asarray(distances), 'segment_fraction': np.asarray(fraction),
        'arclength_mpc_h': geometry['segment_arclength_start_mpc_h'][chosen]+np.asarray(fraction)*segment_length,
        'tangent': tangent,
        'candidate_segment_index': candidate_indices, 'candidate_arc_index': candidate_arcs,
        'candidate_distance_mpc_h': candidate_distance, 'candidate_segment_fraction': candidate_fraction,
        'candidate_arclength_mpc_h': candidate_arclength, 'candidate_tangent': candidate_tangent,
        'candidate_supported': valid,
        'ambiguous_arc_assignment': np.asarray(ambiguous, dtype=bool)}


def read_bound_filament_graph(binding, run_directory, snapshot, snapshot_sources, *, memory_bytes):
    """Read sealed raw topology for the exact run/epoch used by a consumer.

    Smoothed VTK display paths are deliberately not a scientific input. The
    caller pins the publication manifest; every retained asset remains sealed.
    """
    from core.foundation import file_sha256 as digest, verify_product_provenance
    required = {'schema','path','sha256','maximum_candidates','ambiguity_distance_mpc_h'}
    if set(binding) != required or binding['schema'] != 'filament-skeleton-binding-v1':
        raise ValueError('Complete pinned filament skeleton binding required')
    path = Path(binding['path']).resolve()
    if path.stat().st_size > 4*1024**2 or digest(path) != binding['sha256']:
        raise ValueError('Filament publication manifest changed or exceeds bound')
    record = json.loads(path.read_text())
    if record.get('schema') != 'filament-publication-v1' or record['metadata']['snapshot'] != snapshot:
        raise ValueError('Filament product belongs to a different native epoch')
    verify_product_provenance(run_directory, record['provenance'])
    inputs = {entry['path']:entry['sha256'] for entry in record['provenance']['input_hashes']}
    if any(inputs.get(str(Path(p).resolve())) != checksum for p,checksum in snapshot_sources.items()):
        raise ValueError('Filament and consumer must bind the same native snapshot sources')
    sources = {str(path):binding['sha256']}
    for entry in record['assets']:
        asset = (path.parent/entry['path']).resolve()
        if not asset.is_relative_to(path.parent) or str(asset) in sources:
            raise ValueError('Filament asset escaped its publication or is duplicated')
        if asset.stat().st_size != entry['size_bytes'] or digest(asset) != entry['sha256']:
            raise ValueError('Filament publication asset changed')
        sources[str(asset)] = entry['sha256']
    numeric = path.parent/'skeleton.hdf5'
    if str(numeric) not in sources:
        raise ValueError('Filament publication does not seal its numeric graph')
    names = ['node_ids','endpoint_node_ids','arc_point_offsets',
             'arc_points_unwrapped_comoving_mpc_h','arc_image_offsets']
    with h5py.File(numeric,'r') as saved:
        group = saved['graph']
        estimate = (sum(group[name].size*group[name].dtype.itemsize for name in names)
            + len(group['arc_points_unwrapped_comoving_mpc_h'])*1024
            + len(group['arc_point_offsets'])*256+len(group['node_ids'])*64)
        if type(memory_bytes) is not int or memory_bytes < estimate:
            raise MemoryError('Raw filament topology exceeds consumer workspace')
        arrays = {name:group[name][:] for name in names}
    offsets = arrays['arc_point_offsets']
    points = arrays['arc_points_unwrapped_comoving_mpc_h']
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(points) or np.any(offsets > len(points))
            or np.any(offsets[1:] <= offsets[:-1]+1)):
        raise ValueError('Filament graph has invalid ordered arc offsets')
    box = record['metadata']['box_size_mpc_h']
    images = arrays['arc_image_offsets']
    wrapped = np.mod(points, box)
    if images.shape != points.shape or not np.allclose(wrapped+images*box,points,rtol=0,atol=1e-10*box):
        raise ValueError('Filament graph winding differs from recorded images')
    arcs = [np.arange(int(lo),int(hi)) for lo,hi in zip(offsets[:-1],offsets[1:])]
    geometry = measure_filament_graph(wrapped,arcs,arrays['node_ids'],arrays['endpoint_node_ids'],
        box_mpc_h=box,arc_image_offsets=[images[arc] for arc in arcs])
    return geometry, sources, record


def compare_first_apocenter_particles(particle_ids, proper_time_myr, radii_proper_kpc,
        reference_radius_proper_kpc, *, maximum_bracket_myr, profile_feature_radius_kpc,
        radial_derivatives_kpc_myr=None, valid_intervals=None):
    """First saved infall, first pericenter and subsequent first apocenter.

    Missing detections break orbital support. Close pairs of samples alone do
    not prove passage. Event timing is bracketed by actual saved proper times;
    no inferred simulation outputs or ID-row continuity are manufactured.
    """
    from core.data import exact_uint64_ids
    if radial_derivatives_kpc_myr is not None:
        return compare_continuous_first_apocenters(particle_ids, proper_time_myr,
            radii_proper_kpc, radial_derivatives_kpc_myr, reference_radius_proper_kpc,
            maximum_bracket_myr=maximum_bracket_myr,
            profile_feature_radius_kpc=profile_feature_radius_kpc, valid_intervals=valid_intervals)
    ids = exact_uint64_ids(particle_ids)
    time = np.asarray(proper_time_myr,dtype=float)
    radius = np.asarray(radii_proper_kpc,dtype=float)
    reference = np.asarray(reference_radius_proper_kpc,dtype=float)
    if (time.ndim!=1 or len(time)<5 or radius.shape!=(len(time),len(ids))
            or reference.shape!=time.shape or not np.isfinite(time).all()
            or np.any(np.diff(time)<=0) or not np.isfinite(reference).all() or np.any(reference<=0)
            or np.any(radius[np.isfinite(radius)]<0) or len(np.unique(ids))!=len(ids)
            or not np.isfinite(maximum_bracket_myr) or maximum_bracket_myr<=0
            or (profile_feature_radius_kpc is not None and (not np.isfinite(profile_feature_radius_kpc) or profile_feature_radius_kpc<=0))):
        raise ValueError('Exact identities, actual increasing proper times, aligned orbital radii and cadence budget required')
    rows = []
    accepted = []
    for column,identity in enumerate(ids):
        r = radius[:,column]
        row = {'particle_id':int(identity),'status':'NO_SAVED_FIRST_INFALL'}
        inward = np.flatnonzero(np.isfinite(r[:-1])&np.isfinite(r[1:])&(r[:-1]>reference[:-1])&(r[1:]<=reference[1:]))
        if len(inward):
            start = int(inward[0]+1)
            row['first_infall_bracket_myr'] = time[start-1:start+1].tolist()
            if not np.isfinite(r[start:]).all():
                row['status']='MISSING_ORBITAL_SUPPORT'
            else:
                peri = next((k for k in range(start,len(time)-1) if r[k]<r[k-1] and r[k]<=r[k+1]),None)
                apo = next((k for k in range(peri+1,len(time)-1) if r[k]>r[k-1] and r[k]>=r[k+1]),None) if peri is not None else None
                row['status']='NO_SAVED_FIRST_PERICENTER' if peri is None else 'NO_SAVED_FIRST_APOCENTER' if apo is None else 'MEASURED'
                if apo is not None:
                    bracket = time[apo-1:apo+2]
                    row.update(first_pericenter_bracket_myr=time[peri-1:peri+2].tolist(),
                               first_apocenter_bracket_myr=[float(bracket[0]),float(bracket[-1])],
                               first_apocenter_saved_radius_proper_kpc=float(r[apo]),
                               first_apocenter_radius_neighbor_range_kpc=[float(r[apo-1:apo+2].min()),float(r[apo-1:apo+2].max())])
                    if bracket[-1]-bracket[0]>maximum_bracket_myr:
                        row['status']='CADENCE_UNRESOLVED'
                    else:accepted.append(float(r[apo]))
        rows.append(row)
    report = {'schema':'first-apocenter-orbital-check-v1','particles':rows,
              'supported_count':len(accepted),'status':'MEASURED' if accepted else 'NO_SUPPORTED_APOCENTERS',
              'definition':'first maximum after the first minimum after the first inward crossing of the declared evolving reference radius',
              'cadence_budget_myr':float(maximum_bracket_myr),
              'uncertainty_scope':'saved-time brackets and neighboring sampled radii; no rigorous unobserved extremum-radius bound',
              'science_acceptance':'NOT_ACCEPTED'}
    if accepted:
        report['radius_percentile_16_50_84_kpc'] = np.percentile(accepted,[16,50,84]).tolist()
        report['median_radius_over_profile_feature'] = float(np.median(accepted)/profile_feature_radius_kpc) if profile_feature_radius_kpc is not None else None
    return report


def compare_continuous_first_apocenters(particle_ids, proper_time_myr, radii_proper_kpc,
        radial_derivatives_kpc_myr, reference_radius_proper_kpc, *, maximum_bracket_myr,
        profile_feature_radius_kpc=None, valid_intervals=None):
    """First passage events using proper-radius Hermite derivatives; no cadence certification."""
    from core.stream.trajectories import radial_history_events
    from core.data import exact_uint64_ids
    ids = exact_uint64_ids(particle_ids)
    times = np.asarray(proper_time_myr, dtype=float)
    radii = np.asarray(radii_proper_kpc, dtype=float)
    derivatives = np.asarray(radial_derivatives_kpc_myr, dtype=float)
    reference = np.asarray(reference_radius_proper_kpc, dtype=float)
    shape = (len(times), len(ids))
    if times.ndim != 1 or len(times) < 2 or radii.shape != shape or derivatives.shape != shape:
        raise ValueError('Aligned time by exact-particle radius and derivative arrays required')
    if len(np.unique(ids)) != len(ids) or reference.shape != times.shape:
        raise ValueError('Unique IDs and aligned reference boundary required')
    if profile_feature_radius_kpc is not None and (not np.isfinite(profile_feature_radius_kpc) or profile_feature_radius_kpc <= 0):
        raise ValueError('Positive profile feature radius required')
    masks = None if valid_intervals is None else np.asarray(valid_intervals)
    if masks is not None and (masks.dtype.kind != 'b' or
            masks.shape not in {(len(times)-1,), (len(times)-1, len(ids))}):
        raise ValueError('Shared or per-particle explicit boolean interval support required')
    rows, accepted = [], []
    for column, identity in enumerate(ids):
        mask = masks[:, column] if masks is not None and masks.ndim == 2 else masks
        events, supported, interval_support = radial_history_events(times, radii[:, column], derivatives[:, column],
            reference_kpc=reference, valid_intervals=mask, maximum_interval_myr=maximum_bracket_myr,return_support_report=True)
        row = {'particle_id': int(identity), 'events': events, 'supported_intervals': supported.tolist(),
               'interval_support':interval_support}
        if np.isfinite(radii[0, column]) and radii[0, column] <= reference[0]:
            row['status'] = 'LEFT_CENSORED'
        else:
            infall = next((event for event in events if event['kind'] == 'infall'), None)
            peri = next((event for event in events if infall and event['kind'] == 'pericenter'
                         and event['time_myr'] > infall['time_myr']), None)
            apo = next((event for event in events if peri and event['kind'] == 'apocenter'
                        and event['time_myr'] > peri['time_myr']), None)
            row['status'] = 'NO_SUPPORTED_FIRST_INFALL' if infall is None else (
                'NO_SUPPORTED_FIRST_PERICENTER' if peri is None else 'NO_SUPPORTED_FIRST_APOCENTER')
            if infall:
                row['first_infall_bracket_myr'] = infall['bracket_myr']
            if apo:
                last = int(np.searchsorted(times, apo['bracket_myr'][1]))
                continuous = supported[:last].all()
                bracket = apo['bracket_myr']
                row.update(first_pericenter_bracket_myr=peri['bracket_myr'],
                    first_apocenter_bracket_myr=bracket, first_apocenter_time_myr=apo['time_myr'],
                    first_apocenter_radius_proper_kpc=apo['radius_kpc'])
                row['status'] = 'MISSING_ORBITAL_SUPPORT' if not continuous else (
                    'CADENCE_UNRESOLVED' if bracket[1]-bracket[0] > maximum_bracket_myr else 'MEASURED')
                if row['status'] == 'MEASURED':
                    accepted.append(apo['radius_kpc'])
        rows.append(row)
    report = {'schema': 'first-apocenter-orbital-check-v2', 'particles': rows,
        'supported_count': len(accepted), 'status': 'MEASURED' if accepted else 'NO_SUPPORTED_APOCENTERS',
        'method': 'proper-radius cubic Hermite; radial velocity is its proper-time derivative',
        'cadence_budget_myr': float(maximum_bracket_myr),
        'uncertainty_scope': 'saved brackets; interpolation does not establish an error bound or resolve unseen passages',
        'science_acceptance': 'NOT_ACCEPTED'}
    from core.stream.trajectories import orbital_interpolation_contract
    report['algorithm_contract'] = orbital_interpolation_contract()
    report['interval_support_code_table'] = {name:index for index,name in enumerate([
        'SUPPORTED','EXPLICIT_UNSUPPORTED','NONFINITE_SAVED_STATE','CADENCE_UNRESOLVED',
        'NEGATIVE_RADIUS_INTERPOLANT','NONFINITE_RADIAL_POLYNOMIAL','RADIAL_ROOT_SOLVER_FAILED'])}
    if accepted:
        report['radius_percentile_16_50_84_kpc'] = np.percentile(accepted, [16, 50, 84]).tolist()
        report['median_radius_over_profile_feature'] = (float(np.median(accepted)/profile_feature_radius_kpc)
            if profile_feature_radius_kpc is not None else None)
    return report


def orbital_event_arrays(report, saved_times_myr, host_graph_keys):
    """Pack continuous events with exact identities and explicit graph-key brackets.

    Namespace indices address the report's frozen table; IDs alone never name
    hosts. Only supported first apocenters enter the aggregation mask.
    """
    from core.data import exact_uint64_ids
    times = np.asarray(saved_times_myr, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or len(host_graph_keys) != len(times)):
        raise ValueError('Event publication needs aligned actual epochs and host graph keys')
    from halos import node_key
    keys = [node_key(*key) for key in host_graph_keys]
    namespaces = sorted({key[0] for key in keys})
    kinds = {'stationary':0, 'infall':1, 'pericenter':2, 'apocenter':3}
    states = sorted({row['status'] for row in report['particles']})
    particle_ids, kind, time, radius, brackets, host_rows, selected, signs = [], [], [], [], [], [], [], []
    for row in report['particles']:
        for event in row.get('events', []):
            bracket = np.asarray(event['bracket_myr'], dtype=float)
            indices = np.searchsorted(times, bracket)
            if (bracket.shape != (2,) or np.any(indices >= len(times))
                    or not np.array_equal(times[indices], bracket)):
                raise ValueError('Event bracket must address actual saved epochs')
            name = event['kind']
            if name not in kinds or not bracket[0] <= event['time_myr'] <= bracket[1]:
                raise ValueError('Invalid event kind or bracketed time')
            particle_ids.append(row['particle_id']); kind.append(kinds[name])
            time.append(event['time_myr']); radius.append(event['radius_kpc'])
            brackets.append(bracket); host_rows.append(indices)
            selected.append(name == 'apocenter' and row['status'] == 'MEASURED'
                and event['time_myr'] == row.get('first_apocenter_time_myr'))
            signs.append({'infall':[0,0], 'pericenter':[-1,1],
                          'apocenter':[1,-1], 'stationary':[0,0]}[name])
    host_rows = np.asarray(host_rows, dtype=np.int64).reshape(-1, 2)
    report['event_array_contract'] = {'schema':'orbital-events-v2',
        'event_kind_codes':kinds, 'particle_status_codes':{name:i for i,name in enumerate(states)},
        'host_namespace_table':namespaces, 'host_key_definition':'namespace index + exact uint64 ID + saved epoch',
        'radial_sign_definition':'sign before/after turning point; zero for stationary or unknown (boundary crossing alone does not establish radial motion)',
        'boundary_direction_definition':'-1 inward relative to evolving boundary, 0 not a boundary event',
        'aggregation_mask':'supported_first_apocenter only; no unresolved particle enters quantiles'}
    interval_arrays = {}
    if 'interval_support_code_table' in report:
        codes = report['interval_support_code_table']
        interval_arrays['orbital_interval_support_code'] = np.asarray([
            [codes[interval['status']] for interval in row['interval_support']]
            for row in report['particles']],dtype=np.int16).reshape(len(report['particles']),len(times)-1).T
        interval_arrays['orbital_interval_supported'] = interval_arrays['orbital_interval_support_code']==codes['SUPPORTED']
        report['event_array_contract']['interval_support_code_table'] = codes
    return {**interval_arrays,'orbital_event_particle_ids':exact_uint64_ids(particle_ids, unique=False),
        'orbital_event_kind_code':np.asarray(kind, dtype=np.int8),
        'orbital_event_time_myr':np.asarray(time), 'orbital_event_radius_proper_kpc':np.asarray(radius),
        'orbital_event_bracket_myr':np.asarray(brackets).reshape(-1, 2),
        'orbital_event_host_epoch_rows':host_rows,
        'orbital_event_host_ids':np.asarray([[keys[i][1] for i in pair] for pair in host_rows], dtype=np.uint64).reshape(-1, 2),
        'orbital_event_host_epochs':np.asarray([[keys[i][2] for i in pair] for pair in host_rows], dtype=np.int64).reshape(-1, 2),
        'orbital_event_host_namespace_index':np.asarray([[namespaces.index(keys[i][0]) for i in pair]
            for pair in host_rows], dtype=np.int32).reshape(-1, 2),
        'orbital_event_radial_sign_before_after':np.asarray(signs, dtype=np.int8).reshape(-1, 2),
        'orbital_event_boundary_direction_code':np.asarray([-1 if value == kinds['infall'] else 0 for value in kind], dtype=np.int8),
        'orbital_event_supported_first_apocenter':np.asarray(selected, dtype=bool),
        'orbital_particle_failure_code':np.asarray([states.index(row['status']) for row in report['particles']], dtype=np.int16)}


def _bind_structure_graph_observation(product, key, graph, nodes, previous):
    """One authority for host identity, native sources, actual epochs and clocks."""
    from core.foundation import file_sha256 as digest
    from halos import node_key
    key = node_key(*key)
    if key not in nodes or nodes[key]['kind']!='halo' or nodes[key]['status']!='DETECTED':
        raise ValueError('History host is not a detected halo observation in the identity graph')
    if previous is not None and key not in graph.descendants(previous):
        raise ValueError('Host observations lack a graph ancestry path; catalogue rows cannot establish continuity')
    epoch = product['source_context']['epoch']
    graph_epoch = graph.report['epochs'].get(str(key[2]))
    if graph_epoch is None or graph_epoch['actual_scale_factor']!=epoch['actual_scale_factor']:
        raise ValueError('Graph and host source have different actual epochs')
    attributes = nodes[key]['attributes']
    center = np.asarray(product['source_context']['center_comoving_kpc_h'])*product['source_context']['scale_factor']/product['source_context']['h']
    if (attributes.get('id')!=product['request']['host_row']
            or not np.allclose(attributes.get('position_proper_kpc',[]),center,rtol=1e-12,atol=1e-12)):
        raise ValueError('Graph host identity/position does not bind the native FOF observation')
    for p,expected in product['source_context']['source_hashes'].items():
        if Path(p).suffix=='.hdf5' and graph.report['sources'].get(p)!=expected:
            raise ValueError('Graph does not bind the actual orbital snapshot/catalogue sources')
    time = epoch['cosmic_time_gyr']
    if time is None:
        raise ValueError('Host history requires a bound actual proper-time clock')
    # A clock path in an epoch contract must remain valid at the caller boundary.
    clock = epoch['cosmic_time_source']
    if clock.get('path') and digest(clock['path'])!=clock['sha256']:
        raise ValueError('Host history proper-time source changed')
    return epoch,time*1000


def _proper_clock_error_bounds_myr(epochs):
    bounds = np.full(len(epochs),np.nan)
    for index,epoch in enumerate(epochs):
        error = epoch.get('cosmic_time_numerical_error_gyr',{})
        if error.get('status')=='MODEL_CONDITIONAL_BOUND':
            value = error['absolute_bound_gyr']
            if isinstance(value,bool) or not np.isfinite(value) or value<0:
                raise ValueError('Invalid numerical proper-clock error bound')
            bounds[index] = value*1000
    return bounds


def compare_native_first_apocenters(history, *, profile_feature_radius_kpc, memory_gib):
    """Graph-bound host observations, exact particles, and CLASS proper clocks."""
    from halos import verify_halo_structure,IdentityQueries
    from core.data import exact_uint64_ids,exact_id_join
    from core.foundation import file_sha256 as digest
    required = {'structure_observations','identity_graph_directory','component','particle_ids','maximum_bracket_myr'}
    continuous = history.get('schema') == 'orbit-history-v2'
    if set(history)-required-{'schema'} or required-set(history) or history.get('schema', 'orbit-history-v1') not in {'orbit-history-v1', 'orbit-history-v2'} or not isinstance(history['structure_observations'],list) or len(history['structure_observations'])<5:
        raise ValueError('Orbital check requires at least five graph-bound native host observations')
    ids = exact_uint64_ids(history['particle_ids'])
    if not len(ids) or len(np.unique(ids))!=len(ids):
        raise ValueError('Select unique native orbital particle IDs')
    if len(ids)*len(history['structure_observations'])*256+16*1024**2 > memory_gib*1024**3:
        raise MemoryError('Native orbital history exceeds declared workspace')
    graph = IdentityQueries(history['identity_graph_directory'])
    nodes = {tuple(node['key']):node for node in graph.report['nodes']}
    graph_root = Path(history['identity_graph_directory']).resolve()
    sources = {str(graph_root/name):digest(graph_root/name) for name in ['report.json','report.sha256','index.sqlite','nodes.npz']}
    sources.update(graph.report['sources'])
    radii = np.full((len(history['structure_observations']),len(ids)),np.nan)
    derivatives = np.full_like(radii, np.nan) if continuous else None
    times,reference,epochs = [],[],[]
    namespace,run,previous = None,None,None
    for index,observation in enumerate(history['structure_observations']):
        if set(observation)!={'structure_product','graph_node'}:
            raise ValueError('Each orbital observation must declare a structure product and exact graph node')
        key = tuple(observation['graph_node'])
        path = Path(observation['structure_product']).resolve()
        product = verify_halo_structure(path)
        epoch,time = _bind_structure_graph_observation(product,key,graph,nodes,previous)
        previous = key
        current_run = Path(product['request']['run_directory']).resolve()
        current_namespace = product['data_contract']['identity_namespaces']['native_particle']
        if (run is not None and current_run!=run) or (namespace is not None and current_namespace!=namespace):
            raise ValueError('Orbital history changes native run or particle namespace')
        run,namespace = current_run,current_namespace
        times.append(time)
        reference.append(product['source_context']['r200c_proper_kpc'])
        epochs.append(epoch)
        component = history['component']
        if component not in product['request']['components']:
            raise ValueError('Orbital component is absent from the host structure product')
        if set(product['request']['components'][component]['species'])!={1}:
            raise ValueError('First-apocenter DM check requires the unchanged type-1 collisionless population')
        with np.load(path.with_name(path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
            available = saved['parent_'+component+'_particle_ids']
            joined = exact_id_join(ids,available,requested_namespace=namespace,available_namespace=current_namespace,require_all=False)
            good = joined>=0
            radii[index,good] = np.linalg.norm(saved['parent_'+component+'_offsets_proper_kpc'][joined[good]],axis=1)
            if continuous:
                from core.data import relative_proper_state
                from core.cosmology import run_proper_time_clock, class_background_table
                from core.foundation import C_SI
                _, _, clock = run_proper_time_clock(current_run)
                if clock['status'] != 'VERIFIED':
                    raise ValueError('Continuous cosmological orbit needs the actual source-bound H(a)')
                columns, table = class_background_table(clock['path'])
                if digest(clock['path']) != clock['sha256']:
                    raise RuntimeError('Continuous orbit background changed during intake')
                if 'H' not in columns:
                    raise ValueError('Continuous orbit background lacks H in inverse Mpc')
                scale = 1/(1+table[:, columns['z']])
                order = np.argsort(scale)
                actual = product['source_context']['scale_factor']
                if not scale.min() <= actual <= scale.max():
                    raise ValueError('Continuous orbit epoch outside H(a) source domain')
                expansion = float(np.interp(actual, scale[order], table[order, columns['H']]))*C_SI/1000.
                host_velocity = nodes[key]['attributes'].get('velocity_peculiar_kms')
                if host_velocity is None:
                    raise ValueError('Continuous host-relative orbit needs graph-bound center peculiar velocity')
                state = relative_proper_state(saved['parent_'+component+'_offsets_proper_kpc'][joined[good]],
                    saved['parent_'+component+'_velocities_peculiar_km_s'][joined[good]],
                    host_velocity, hubble_kms_mpc=expansion)
                derivatives[index,good] = state['radial_derivative_kpc_myr']
        sources.update(product['source_hashes'])
        for p in [path,path.with_name(path.name+'.sha256'),path.with_name(path.name+'.data')/'arrays.npz',
                  path.with_name(path.name+'.data')/'contract.json']:
            sources[str(p)] = digest(p)
    result = compare_first_apocenter_particles(ids,times,radii,reference,
        maximum_bracket_myr=history['maximum_bracket_myr'],profile_feature_radius_kpc=profile_feature_radius_kpc,
        radial_derivatives_kpc_myr=derivatives)
    clock_errors = _proper_clock_error_bounds_myr(epochs)
    bounded_count = 0
    for row in result['particles']:
        row['clock_bounded_cadence_supported'] = False
        if row['status']=='MEASURED':
            bracket = row['first_apocenter_bracket_myr']
            endpoints = np.searchsorted(times,bracket)
            if np.isfinite(clock_errors[endpoints]).all():
                expanded = [float(bracket[0]-clock_errors[endpoints[0]]),float(bracket[1]+clock_errors[endpoints[1]])]
                row['clock_expanded_apocenter_bracket_myr'] = expanded
                row['clock_bounded_cadence_supported'] = expanded[1]-expanded[0]<=history['maximum_bracket_myr']
                bounded_count += int(row['clock_bounded_cadence_supported'])
    result['clock_bounded_supported_count'] = bounded_count
    result['clock_scope'] = 'Supported count measures saved brackets; clock-bounded count additionally includes declared numerical clock errors. Unbounded clocks cannot establish cadence acceptance.'
    result.update(particle_namespace=namespace,epoch_contracts=epochs,
        graph_directory=str(graph_root),host_ancestry='explicit verified graph paths and native FOF position/source bindings')
    data = {'orbital_particle_ids':ids,'orbital_proper_time_myr':np.asarray(times),
        'orbital_proper_clock_error_myr':clock_errors,
        'orbital_particle_radius_proper_kpc':radii,'orbital_reference_radius_proper_kpc':np.asarray(reference)}
    if continuous:
        data['orbital_radial_derivative_kpc_myr'] = derivatives
        data.update(orbital_event_arrays(result, times,
            [entry['graph_node'] for entry in history['structure_observations']]))
        for path in [Path(__file__).with_name('streams.py'),
                     Path(__file__).resolve().parents[1]/'configs/runtime/orbit_interpolation.json']:
            sources[str(path)] = digest(path)
    return result, data, sources


def measure_radial_environment(grid, components, radial_edges_proper_kpc, *,
                               center_comoving_kpc, scale_factor, box_comoving_kpc,
                               threshold, eigenvalue_gap_floor, chunk_rows=8192,
                               filtered_density_contrast=None):
    """Mass-weight actual shell locations; diagonalize the mean full tensor.

    The retained tensor scatter is spatial variation, not uncertainty in a
    particle bootstrap or proof of environmental resolution convergence.
    """
    edges = np.asarray(radial_edges_proper_kpc, dtype=float)
    center = np.asarray(center_comoving_kpc, dtype=float)
    if (edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all()
            or edges[0]<0 or np.any(np.diff(edges)<=0) or center.shape!=(3,)
            or not np.isfinite(center).all() or not np.isfinite(scale_factor)
            or scale_factor<=0 or type(chunk_rows) is not int or chunk_rows<1):
        raise ValueError('Explicit proper shell edges, comoving center, scale factor and bounded chunks required')
    n = len(edges)-1
    count = np.zeros(n,dtype=np.int64)
    unsupported_density_count = np.zeros(n,dtype=np.int64)
    mass, mass2, moment = np.zeros(n),np.zeros(n),np.zeros(n)
    mean = np.zeros((n,3,3))
    for sample in components.values():
        offsets = np.asarray(sample['offsets_proper_kpc'],dtype=float)
        weights = np.asarray(sample['masses_msun'],dtype=float)
        if (offsets.shape!=(len(weights),3) or not np.isfinite(offsets).all()
                or not np.isfinite(weights).all() or np.any(weights<=0)):
            raise ValueError('Finite complete native shell positions and positive masses required')
        for start in range(0,len(weights),chunk_rows):
            x,w = offsets[start:start+chunk_rows],weights[start:start+chunk_rows]
            radius = np.linalg.norm(x,axis=1)
            bins = np.searchsorted(edges,radius,side='right')-1
            bins[radius==edges[-1]] = n-1
            values = sample_periodic_tensor_grid(grid,center+x/scale_factor,box_comoving_kpc)
            density_good = (sample_periodic_tensor_grid(filtered_density_contrast,center+x/scale_factor,box_comoving_kpc)>=-1-1e-12
                            if filtered_density_contrast is not None else np.ones(len(x),dtype=bool))
            for b in np.unique(bins[(bins>=0)&(bins<n)]):
                use = bins==b; v,weight = values[use],w[use]
                total = float(weight.sum()); local = np.einsum('n,nij->ij',weight,v)/total
                spread = float(np.sum(weight*np.sum((v-local)**2,axis=(1,2))))
                combined = mass[b]+total; difference = local-mean[b]
                moment[b] += spread+float(np.sum(difference**2))*mass[b]*total/combined
                mean[b] += difference*total/combined
                mass[b] = combined; mass2[b] += np.sum(weight**2); count[b] += use.sum()
                unsupported_density_count[b] += int((~density_good[use]).sum())
    supported = mass>0
    mean[~supported] = np.nan
    trace_free = mean-np.trace(mean,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)/3
    classes = np.full(n,255,dtype=np.uint8)
    values,axes = np.full((n,3),np.nan),np.full((n,3,3),np.nan)
    axis_support = np.zeros((n,3),dtype=bool); filament_support = np.zeros(n,dtype=bool)
    if supported.any():
        measured = classify_eigenstructure(mean[supported],threshold=threshold,eigenvalue_gap_floor=eigenvalue_gap_floor)
        for destination,key in [(classes,'web_class'),(values,'eigenvalues_ascending'),
                                (axes,'eigenvectors_columns'),(axis_support,'axis_supported'),
                                (filament_support,'filament_axis_supported')]:
            destination[supported] = measured[key]
    density_supported = supported&(unsupported_density_count==0)
    filament_support &= density_supported
    return {'mean_dimensionless_tidal_hessian':mean,'mean_dimensionless_trace_free_hessian':trace_free,
        'tensor_frobenius_scatter':np.sqrt(np.divide(moment,mass,out=np.full(n,np.nan),where=supported)),
        'mass_msun':mass,'particle_count':count,
        'effective_count':np.divide(mass**2,mass2,out=np.zeros(n),where=mass2>0),
        'radial_edges_proper_kpc':edges,'supported':supported,'web_class':classes,
        'filtered_density_supported':density_supported,'unsupported_density_count':unsupported_density_count,
        'eigenvalues_ascending':values,'eigenvectors_columns':axes,'axis_supported':axis_support,
        'filament_axis_supported':filament_support,
        'metadata':{'sampling':'complete native particles in proper spherical shells; mass-weight Cartesian tensor before diagonalization',
            'empty_web_class':255,'scatter_definition':'mass-weight population RMS Frobenius variation across shell locations',
            'filtered_density_checked':filtered_density_contrast is not None,
            'filtered_density_scope':'all actual native shell locations must have filtered delta >= -1 within 1e-12; raw tensors remain diagnostic where unsupported',
            'uncertainty_scope':'spatial variation only; neither sampling confidence interval nor simulation convergence'}}


@directory_owner('destination')
def publish_native_environment(request, destination, *, request_source_hashes=None):
    """Connected Block 12 native mesh, tidal/web, halo and splashback producer."""
    from core.data import (verify_native_hdf5_set,native_publication_markers,native_unit_factor,
                      run_epoch_contract,coordinate_contract,export_numeric_audit)
    from core.foundation import file_sha256 as digest, product_provenance
    from halos import _snapshot_files,_unit_conversion,verify_halo_structure
    required = {'schema','run_directory','snapshot','mesh_size','smoothing_comoving_kpc',
        'web_threshold','eigenvalue_gap_floor','halo_structure_products','splashback_component',
        'slope_controls','splashback_controls','alignment_controls','orbital_history','memory_gib'}
    if set(request)-{'slope_window_study','filament_skeleton'}!=required or request['schema']!='native-environment-v1':
        raise ValueError('Declare complete native-environment-v1 request')
    windows = request.get('slope_window_study',[request['slope_controls']['window_bins']])
    if (not isinstance(windows,list) or not windows or len(set(windows))!=len(windows)
            or any(type(window) is not int or window<5 or window%2!=1
                   or window<=request['slope_controls']['polynomial_degree'] for window in windows)
            or request['slope_controls']['window_bins'] not in windows):
        raise ValueError('Slope window study requires distinct supported odd windows including the primary window')
    if (type(request['snapshot']) is not int or request['snapshot']<0
            or type(request['mesh_size']) is not int or request['mesh_size']<5 or request['mesh_size']%2!=1
            or not isinstance(request['smoothing_comoving_kpc'],list) or not request['smoothing_comoving_kpc']
            or len(set(request['smoothing_comoving_kpc']))!=len(request['smoothing_comoving_kpc'])
            or not isinstance(request['halo_structure_products'],list)
            or isinstance(request['memory_gib'],bool) or not np.isfinite(request['memory_gib']) or request['memory_gib']<=0):
        raise ValueError('Explicit native epoch, odd grid, distinct smoothing list, product list and memory required')
    destination = Path(destination).resolve()
    if destination.exists() or destination.with_name(destination.name+'.data').exists():
        raise FileExistsError('Keep earlier environment evidence')
    n = request['mesh_size']
    if (n**3*(512+208*len(request['smoothing_comoving_kpc']))+32*1024**2
            > request['memory_gib']*1024**3):
        raise MemoryError('Requested retained smoothing fields exceed conservative memory budget')
    run = Path(request['run_directory']).resolve()
    files = _snapshot_files(run,request['snapshot'])
    validate_gadget_snapshot_set(files,request['snapshot'])
    native = verify_native_hdf5_set(files,require_publication=True)
    _unit_conversion(files)
    sources = {str(p.resolve()):digest(p) for p in files+native_publication_markers(files)}
    with h5py.File(files[0],'r') as source:
        if source['Parameters'].attrs.get('ComovingIntegrationOn')!=1:
            raise ValueError('Periodic peculiar potential requires explicit cosmological comoving integration')
        a = float(source['Header'].attrs['Time'])
        box_native = float(source['Header'].attrs['BoxSize'])
        h = float(source['Parameters'].attrs['HubbleParam'])
    box = box_native*native_unit_factor('position_comoving_kpc',scale_factor=a,hubble_param=h)
    chunk = min(250000,max(1,int(request['memory_gib']*1024**3)//1024))
    deposited,particle_count,shot = _snapshot_matter_cic_mesh(files,box_native/1000,n,chunk_size=chunk)
    total_mass = float(deposited.sum())*native_unit_factor('mass_msun',scale_factor=a,hubble_param=h)
    mean_density = total_mass/(box*a)**3
    contrast = deposited/float(np.mean(deposited))-1.
    arrays = {'density_contrast_unsmoothed':contrast}
    report = {'schema':'native-environment-v1','status':'MEASURED','request':request,
        'epoch':run_epoch_contract(run,request['snapshot'],a),'scale_factor':a,
        'box_comoving_kpc':box,'mean_density_proper_msun_kpc3':mean_density,
        'total_particle_mass_msun':total_mass,'massive_particle_rows':particle_count,
        'particle_mass_shot_noise_comoving_kpc3':float(shot*(1000/h)**3),
        'density_definition':'complete massive native particle CIC source; no constant background source in peculiar Poisson problem',
        'smoothing_fields':[],'halo_measurements':[],
        'first_apocenter_check':{'status':'NOT_REQUESTED','required_input':'complete native host-frame histories and source-bound proper times; separate orbital comparison API'},
        'simulation_resolution_convergence':'NOT_EVALUATED'}
    halo_products = []
    for source in request['halo_structure_products']:
        path = Path(source).resolve()
        structure = verify_halo_structure(path)
        if (Path(structure['request']['run_directory']).resolve()!=run
                or structure['request']['snapshot']!=request['snapshot']
                or structure['source_context']['scale_factor']!=a):
            raise ValueError('Environment and host structure must use the same actual native run and epoch')
        sources[str(path)] = digest(path)
        sources.update(structure['source_hashes'])
        for p in [path.with_name(path.name+'.sha256'),path.with_name(path.name+'.data')/'arrays.npz',
                  path.with_name(path.name+'.data')/'contract.json']:
            sources[str(p)] = digest(p)
        halo_products.append((path,structure))
    if request.get('filament_skeleton') is not None:
        binding = request['filament_skeleton']
        # Reserve the existing field workspace and divide the remainder between
        # retained graph geometry and projection workspace/output arrays.
        topology_budget = (int(request['memory_gib']*1024**3)
            - n**3*(512+208*len(request['smoothing_comoving_kpc']))-32*1024**2)//2
        geometry, graph_sources, skeleton = read_bound_filament_graph(binding,run,request['snapshot'],
            {str(p.resolve()):sources[str(p.resolve())] for p in files},
            memory_bytes=topology_budget)
        if not np.isclose(geometry['box_mpc_h'],box*h/1000,rtol=1e-12,atol=0):
            raise ValueError('Filament graph and environmental field use different physical boxes')
        centers = np.asarray([structure['source_context']['center_comoving_kpc_h']
            for _,structure in halo_products],dtype=float).reshape(-1,3)/1000
        assignments = nearest_filament_segments(centers,geometry,
            maximum_candidates=binding['maximum_candidates'],
            ambiguity_distance_mpc_h=binding['ambiguity_distance_mpc_h'],
            memory_bytes=topology_budget)
        arrays.update({'topology_halo_'+key:value for key,value in assignments.items()})
        arrays['topology_node_ids'] = geometry['node_ids']
        arrays['topology_node_degree'] = geometry['node_degree']
        arrays['topology_arc_length_mpc_h'] = geometry['arc_length_mpc_h']
        shells_requested = sum(len(structure['request']['radial_edges_proper_kpc'])-1
                               for _,structure in halo_products)
        alignment_bytes = binding['maximum_candidates']*(shells_requested*256+
            len(halo_products)*request['alignment_controls']['randomized_realizations']*16)
        if alignment_bytes+sum(value.nbytes for value in assignments.values()) > topology_budget:
            raise MemoryError('Filament candidate alignments exceed declared workspace')
        topology_alignments = []
        for halo_index,(path,structure) in enumerate(halo_products):
            prefix = 'topology_halo_%02d_'%halo_index
            with np.load(path.with_name(path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
                spin = saved['profile_total_enclosed_angular_momentum_msun_kpc_km_s']
                radius = saved['profile_total_radial_outer_edge_proper_kpc']
            shells = structure['density_shapes']['total']['shells']
            shape = np.asarray([np.asarray(shell['axes_columns'])[:,0]
                if shell.get('orientation_sampling_resolved',[False]*3)[0] else np.zeros(3)
                for shell in shells]).reshape(-1,3)
            # Candidate alignments retain geometric alternatives. The primary
            # scientific selection excludes ambiguous junction assignments.
            row = {'halo_structure':str(path),'alignments':{}}
            arrays[prefix+'spin_radius_proper_kpc'] = radius
            arrays[prefix+'shape_radius_proper_kpc'] = np.asarray(
                structure['request']['radial_edges_proper_kpc'])[1:]
            for candidate in range(binding['maximum_candidates']):
                flag = bool(assignments['candidate_supported'][halo_index,candidate])
                tangent = assignments['candidate_tangent'][halo_index,candidate]
                for name,vectors in [('spin',spin),('shape',shape)]:
                    aligned = measure_filament_axis_alignment(vectors,
                        np.repeat(tangent[None,:],len(vectors),axis=0),
                        np.full(len(vectors),flag),**request['alignment_controls'])
                    tag = prefix+'candidate_%02d_%s_'%(candidate,name)
                    for key,value in aligned.items():
                        if key!='metadata': arrays[tag+key] = value
                    row['alignments']['candidate_%02d_%s'%(candidate,name)] = aligned['metadata']
                    if candidate == 0:
                        unique = aligned['supported'] & ~assignments['ambiguous_arc_assignment'][halo_index]
                        arrays[prefix+name+'_unique_supported'] = unique
                        arrays[prefix+name+'_unique_cosine_unsigned'] = np.where(
                            unique,aligned['cosine_unsigned'],np.nan)
            topology_alignments.append(row)
        sources.update(graph_sources)
        report['filament_skeleton'] = {'manifest_sha256':binding['sha256'],
            'halo_structure_order':[str(path) for path,_ in halo_products],
            'halo_alignments':topology_alignments,
            'alignment_definition':'unsigned cosine to local unoriented arc tangent; candidate alternatives retained, unique selection masks junction ambiguity',
            'infall_alignment_status':'NOT_MEASURED; requires source-bound infall directions at matched continuous event epochs',
            'field_definition':skeleton['metadata']['input_field'],
            'estimator':'nearest periodic segment in unsmoothed native arcs; distinct-arc alternatives retained',
            'ambiguity_distance_mpc_h':binding['ambiguity_distance_mpc_h'],
            'relation_to_tweb':'separate estimators; source field, smoothing and persistence definitions retained',
            'science_acceptance':'NOT_ACCEPTED'}
    for index,smoothing in enumerate(request['smoothing_comoving_kpc']):
        field = solve_periodic_peculiar_potential(contrast,scale_factor=a,
            mean_density_proper_msun_kpc3=mean_density,box_comoving_kpc=box,
            smoothing_comoving_kpc=smoothing,memory_gib=request['memory_gib'])
        eigen = classify_eigenstructure(field['dimensionless_tidal_hessian'],
            threshold=request['web_threshold'],eigenvalue_gap_floor=request['eigenvalue_gap_floor'])
        filaments = measure_periodic_filament_regions(eigen['web_class'],box_comoving_kpc=box,
                                                     density_supported=field['filtered_density_supported'])
        for prefix,values in [('field',field),('web',eigen),('filament',filaments)]:
            for name,value in values.items():
                if name!='metadata':arrays['smoothing_%02d_%s_%s'%(index,prefix,name)] = value.reshape(-1) if name=='component_ids' else value
        report['smoothing_fields'].append({'index':index,'field':field['metadata'],'web':eigen['metadata'],
                                          'filaments':filaments['metadata']})
        for halo_index,(path,structure) in enumerate(halo_products):
            with np.load(path.with_name(path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
                center = np.asarray(structure['source_context']['center_comoving_kpc_h'])/h
                local = sample_periodic_tensor_grid(field['dimensionless_tidal_hessian'],center[None,:],box)
                local_eigen = classify_eigenstructure(local,threshold=request['web_threshold'],
                    eigenvalue_gap_floor=request['eigenvalue_gap_floor'])
                local_density = sample_periodic_tensor_grid(field['density_contrast_smoothed'],center[None,:],box)
                local_density_supported = local_density>=-1-1e-12
                local_eigen['filament_axis_supported'] &= local_density_supported
                radius = saved['profile_total_radial_outer_edge_proper_kpc']
                spins = saved['profile_total_enclosed_angular_momentum_msun_kpc_km_s']
                repeated = {k:np.repeat(v,len(radius),axis=0) for k,v in local_eigen.items() if k!='metadata'}
                aligned = measure_spin_shape_filament_angles(spins,repeated,**request['alignment_controls'])
                prefix = 'smoothing_%02d_halo_%02d_'%(index,halo_index)
                for key,value in local_eigen.items():
                    if key!='metadata':arrays[prefix+'center_'+key] = value
                for key,value in aligned.items():
                    if key!='metadata':arrays[prefix+'spin_'+key] = value
                row = {'halo_structure':str(path),'source_product_sha256':digest(path),'smoothing_index':index,
                    'web_class_at_center':int(local_eigen['web_class'][0]),
                    'center_filtered_density_contrast':float(local_density[0]),
                    'center_filtered_density_supported':bool(local_density_supported[0]),
                    'spin_alignment':aligned['metadata'],'spin_radius_proper_kpc':radius.tolist(),
                    'field_sampling':'trilinear interpolation of Cartesian tensor then diagonalization at actual host centre',
                    'radial_environment_evolution':'spin measured in each enclosed aperture about the local centre tensor'}
                proximity = measure_filament_proximity(filaments,center[None,:],box_comoving_kpc=box)
                for key,value in proximity.items():arrays[prefix+'filament_proximity_'+key] = value
                row['filament_proximity'] = {key:float(value[0]) if np.isfinite(value[0]) else None
                    for key,value in proximity.items() if key!='supported'}
                row['filament_proximity']['supported'] = bool(proximity['supported'][0])
                row['filament_proximity']['geometry_scope'] = filaments['metadata']['geometry_scope']
                components = {name:{key:saved['parent_'+name+'_'+key] for key in ('offsets_proper_kpc','masses_msun')}
                              for name in structure['request']['components']}
                radial = measure_radial_environment(field['dimensionless_tidal_hessian'],components,
                    structure['request']['radial_edges_proper_kpc'],center_comoving_kpc=center,
                    scale_factor=a,box_comoving_kpc=box,threshold=request['web_threshold'],
                    eigenvalue_gap_floor=request['eigenvalue_gap_floor'],
                    filtered_density_contrast=field['density_contrast_smoothed'])
                for key,value in radial.items():
                    if key!='metadata':arrays[prefix+'radial_'+key] = value
                radial_alignment = measure_spin_shape_filament_angles(spins,radial,**request['alignment_controls'])
                for key,value in radial_alignment.items():
                    if key!='metadata':arrays[prefix+'radial_spin_'+key] = value
                row['radial_environment'] = radial['metadata']
                row['radial_spin_alignment'] = radial_alignment['metadata']
                row['radial_environment_evolution'] = 'separate enclosed spin/center alignment and mass-weight shell environmental tensors'
                shape_vectors = []
                supported = []
                for shell in structure['density_shapes']['total']['shells']:
                    flag = shell.get('orientation_sampling_resolved',[False]*3)[0]
                    shape_vectors.append(np.asarray(shell['axes_columns'])[:,0] if flag else np.zeros(3))
                    supported.append(flag)
                shape_eigen = {k:np.repeat(v,len(shape_vectors),axis=0) for k,v in local_eigen.items() if k!='metadata'}
                shape = measure_spin_shape_filament_angles(np.asarray(shape_vectors),shape_eigen,**request['alignment_controls'])
                for key,value in shape.items():
                    if key!='metadata':arrays[prefix+'shape_'+key] = value
                row['shape_alignment'] = shape['metadata']
                component = request['splashback_component']
                if component not in structure['request']['components'] and component!='total':
                    raise ValueError('Splashback component absent from the native structure product')
                profile = 'profile_'+component+'_'
                rho = saved[profile+'density_msun_kpc3']
                shot_error = saved[profile+'density_shot_noise_msun_kpc3']
                covariance = np.zeros((len(rho),len(rho)))
                positive = rho>0
                covariance[np.diag_indices(len(rho))] = np.where(positive,np.divide(shot_error,rho,out=np.zeros_like(rho),where=positive)**2,0)
                slope = uncertainty_aware_log_density_derivative(saved[profile+'radius_proper_kpc'],rho,covariance,
                    effective_count=saved[profile+'effective_count'],**request['slope_controls'])
                feature = locate_supported_slope_feature(saved[profile+'radius_proper_kpc'],slope,**request['splashback_controls'])
                window_study=[]
                for window in windows:
                    alternative = slope if window==request['slope_controls']['window_bins'] else uncertainty_aware_log_density_derivative(
                        saved[profile+'radius_proper_kpc'],rho,covariance,effective_count=saved[profile+'effective_count'],
                        **(request['slope_controls']|{'window_bins':window}))
                    alternative_feature = feature if alternative is slope else locate_supported_slope_feature(
                        saved[profile+'radius_proper_kpc'],alternative,**request['splashback_controls'])
                    window_study.append({'window_bins':window,'feature':alternative_feature})
                    for key,value in alternative.items():
                        if key!='metadata':arrays[prefix+'window_%03d_'%window+key] = value
                measured_radii = [trial['feature']['candidates'][0]['radius_proper_kpc'] for trial in window_study
                                  if trial['feature']['status']=='MEASURED']
                for key,value in slope.items():
                    if key!='metadata':arrays[prefix+'splashback_'+key] = value
                row.update(splashback=feature,slope_definition=slope['metadata'],
                    slope_window_study=window_study,
                    slope_window_radius_range_proper_kpc=[min(measured_radii),max(measured_radii)] if measured_radii else None,
                    slope_window_sensitivity_status=('NOT_EVALUATED' if len(windows)<2 else
                        'MEASURED_ALL_WINDOWS' if len(measured_radii)==len(windows) else 'UNRESOLVED_WINDOWS_RETAINED'),
                    smoothing_scope='GLS density-derivative window study is separate from 3D Gaussian environmental smoothing; unresolved windows retained',
                    covariance_definition='independent shell mass shot noise transformed to log density; excludes correlated simulation and host-selection uncertainty')
                report['halo_measurements'].append(row)
    if request['orbital_history'] is not None:
        feature_radius = None
        final_structure = Path(request['orbital_history']['structure_observations'][-1]['structure_product']).resolve()
        matched_rows = [row for row in report['halo_measurements'] if Path(row['halo_structure']).resolve()==final_structure]
        if not matched_rows:
            raise ValueError('Orbital history must end at a host structure bound to this actual environmental epoch')
        if matched_rows:
            feature = matched_rows[0]['splashback']
            if feature['status']=='MEASURED':feature_radius = feature['candidates'][0]['radius_proper_kpc']
        orbital,data,bindings = compare_native_first_apocenters(request['orbital_history'],
            profile_feature_radius_kpc=feature_radius,memory_gib=request['memory_gib'])
        report['first_apocenter_check'] = orbital
        arrays.update(data);sources.update(bindings)
    coordinates = coordinate_contract(position_units='comoving kpc',velocity_convention='no velocity data in environmental tensor product',
        origin='periodic global Cartesian box origin',periodic_wrapping='uniform nodes in [0,L); trilinear periodic interpolation')
    provenance = product_provenance(run,algorithm=__file__,selection=report['density_definition'],
        source_files=[{'path':p,'sha256':v} for p,v in sources.items()],coordinates=coordinates,snapshot=request['snapshot'])
    for entry in (provenance['initial_conditions']+list(provenance['boltzmann_inputs'].values())+
            [provenance['executable'],provenance['parameter_file'],provenance['compile_options']]):
        if entry.get('status')=='VERIFIED':sources[entry['path']]=entry['sha256']
    if (run/'config.yaml').is_file():sources[str(run/'config.yaml')]=digest(run/'config.yaml')
    sources.update(request_source_hashes or {})
    for name in ['analysis.py','halos.py','foundation.py','data.py','cosmology.py']:
        path = Path(__file__).with_name(name).resolve();sources[str(path)] = digest(path)
    manifest_path = run/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest_path.is_file():sources[str(manifest_path)] = digest(manifest_path)
    report.update(data_contract=provenance,source_hashes=sources,native_publication=native,
        source_origin='synthetic_control' if manifest.get('fixture') or manifest.get('synthetic_inputs') else 'native_snapshot')
    def units(key):
        if key=='topology_node_ids':return 'identity'
        if key.startswith('topology_') and key.endswith('mpc_h'):return 'comoving Mpc/h'
        if key.startswith('topology_halo_') and key.endswith('radius_proper_kpc'):return 'proper kpc'
        if key.startswith('topology_halo_') and key.endswith('angle_unsigned_deg'):return 'degree'
        if key in {'orbital_particle_ids','orbital_event_particle_ids','orbital_event_host_ids'}:return 'identity'
        if key.startswith('orbital_event_') and key.endswith('myr'):return 'Myr'
        if key=='orbital_proper_time_myr':return 'Myr'
        if key=='orbital_proper_clock_error_myr':return 'Myr'
        if key=='orbital_radial_derivative_kpc_myr':return 'proper kpc/Myr'
        if key.startswith('orbital_') and key.endswith('kpc'):return 'proper kpc'
        if 'hessian_km2' in key:return '(km/s)^2/(proper kpc)^2'
        if 'acceleration_km2' in key:return '(km/s)^2/proper kpc'
        if 'potential_km2' in key:return '(km/s)^2'
        if 'angle_unsigned_deg' in key:return 'degrees'
        if key.endswith('mass_msun'):return 'Msun'
        if key.endswith('proper_kpc'):return 'proper kpc'
        if key.endswith('comoving_kpc'):return 'comoving kpc'
        if key.endswith('comoving_kpc3'):return '(comoving kpc)^3'
        return 'dimensionless'
    specifications = {key:{'units':units(key),'quantity':key,
        'missing_policy':'NaN for unsupported slopes or alignments; availability flags retained'} for key in arrays}
    for key in arrays:
        if key.endswith('filament_component_ids'):
            specifications[key]['units']='identity'
            specifications[key]['missing_policy']='0 means no classified filament region; repeated cell labels refer to product-local components'
            specifications[key]['identity_namespace']='environment-filament-regions:'+__import__('hashlib').sha256(
                json.dumps({'native_sources':native,'request':request,'smoothing':key.split('_')[1]},sort_keys=True).encode()).hexdigest()
    if 'orbital_particle_ids' in arrays:
        specifications['orbital_particle_ids']['identity_namespace']=report['first_apocenter_check']['particle_namespace']
    if 'orbital_event_particle_ids' in arrays:
        specifications['orbital_event_particle_ids']['identity_namespace']=report['first_apocenter_check']['particle_namespace']
        specifications['orbital_event_host_ids']['identity_namespace']='graph-host-keys:'+__import__('hashlib').sha256(
            json.dumps(report['first_apocenter_check']['event_array_contract']['host_namespace_table']).encode()).hexdigest()
    if 'orbital_proper_time_myr' in arrays:
        history_epoch={'schema':'orbit-epoch-series-v1','row_epoch_contracts':report['first_apocenter_check']['epoch_contracts'],
                       'canonical_time':'actual proper time; source-bound per row'}
        for key in arrays:
            if key.startswith('orbital_') and key!='orbital_particle_ids':specifications[key]['epoch']=history_epoch
            if key.startswith('orbital_event_'):
                specifications[key]['epoch']={'schema':'orbital-event-epochs-v2',
                    'saved_epoch_contracts':history_epoch['row_epoch_contracts'],
                    'canonical_time':'actual interpolated proper time from orbital_event_time_myr',
                    'bracket_array':'orbital_event_host_epoch_rows'}
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=__file__,
        context={'epoch':report['epoch'],'coordinates':coordinates,'provenance':provenance},source_hashes=sources,
        hdf5_arrays=[key for key in arrays if key.startswith('orbital_event_') or key=='orbital_particle_failure_code'])


def verify_native_environment(path):
    from core.data import verify_numeric_audit,verify_native_hdf5_set
    from core.foundation import file_sha256 as digest, verify_product_provenance
    from halos import _snapshot_files,verify_halo_structure
    report = verify_numeric_audit(path)
    if report.get('schema')!='native-environment-v1':
        raise ValueError('Not a native environment product')
    for source,expected in report['source_hashes'].items():
        if digest(source)!=expected:raise ValueError('Environment source changed: '+source)
    verify_product_provenance(report['request']['run_directory'],report['data_contract'])
    verify_native_hdf5_set(_snapshot_files(Path(report['request']['run_directory']),report['request']['snapshot']),require_publication=True)
    for path in report['request']['halo_structure_products']:verify_halo_structure(path)
    return report


def plot_native_environment(path, output_prefix, *, smoothing_index, halo_index=None):
    """Source-bound Cartesian field slices and explicitly selected halo data."""
    from figures import export_figure,verify_figure_export
    from core.foundation import file_sha256 as digest
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.ticker import FuncFormatter
    path = Path(path).resolve();report = verify_native_environment(path);before = digest(path)
    if type(smoothing_index) is not int or not 0<=smoothing_index<len(report['smoothing_fields']):
        raise ValueError('Choose an actual saved smoothing index')
    if halo_index is not None and (type(halo_index) is not int
            or not 0<=halo_index<len(report['request']['halo_structure_products'])):
        raise ValueError('Choose an actual saved halo index')
    with np.load(path.with_name(path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
        prefix = 'smoothing_%02d_'%smoothing_index
        full = saved[prefix+'field_hessian_km2_s2_proper_kpc2']
        web = saved[prefix+'web_web_class'];middle = len(web)//2
        spacing = report['box_comoving_kpc']/len(web)
        extent = [-spacing/2,report['box_comoving_kpc']-spacing/2]*2
        fig,axes = plt.subplots(2,2,figsize=(11,8.5))
        colors = ListedColormap(['#34495e','#85c1e9','#f4b942','#c0392b'])
        colors.set_bad('#eeeeee')
        masked_web = np.ma.masked_where(~saved[prefix+'field_filtered_density_supported'][:,:,middle].T,web[:,:,middle].T)
        image = axes[0,0].imshow(masked_web,origin='lower',extent=extent,
                                  cmap=colors,norm=BoundaryNorm(np.arange(-.5,4.5),4))
        bar = fig.colorbar(image,ax=axes[0,0],ticks=[0,1,2,3],fraction=.046,pad=.04)
        bar.ax.set_yticklabels(['void','sheet','filament','knot']);bar.ax.tick_params(labelsize=8)
        axes[0,0].set_title('Full Hessian web classification',fontsize=11)
        if np.ma.getmaskarray(masked_web).any():axes[0,0].text(.5,.97,'Gray: filtered density unsupported',ha='center',va='top',
                                                             transform=axes[0,0].transAxes,fontsize=8)
        trace = np.trace(full,axis1=-2,axis2=-1)[:,:,middle].T
        trace_span = float(np.ptp(trace));trace_scale = float(np.max(abs(trace)))
        roundoff = 128*np.finfo(float).eps*max(1.,float(np.max(abs(full))))
        uniform = trace_span<=roundoff
        limits = {'vmin':float(trace.mean())-roundoff,'vmax':float(trace.mean())+roundoff} if uniform else {}
        image = axes[0,1].imshow(trace,origin='lower',extent=extent,cmap='coolwarm',**limits)
        bar = fig.colorbar(image,ax=axes[0,1],fraction=.046,pad=.04)
        precision = 9 if uniform else min(12,max(2,int(np.ceil(-np.log10(trace_span/max(trace_scale,1e-300))))+2))
        if uniform:bar.set_ticks([float(trace.mean())])
        bar.formatter = FuncFormatter(lambda value,position,precision=precision:format(value,'.%dg'%precision));bar.update_ticks()
        bar.set_label('Trace H [(km/s)² / proper kpc²]',fontsize=9);bar.ax.tick_params(labelsize=8)
        axes[0,1].set_title('Signed full Hessian trace',fontsize=10)
        if uniform:axes[0,1].text(.5,.95,'Uniform within floating-point tolerance',ha='center',va='top',
                                 transform=axes[0,1].transAxes,fontsize=8)
        for ax in axes[0]:
            ax.set_xlabel('Comoving x [kpc]',fontsize=10);ax.set_ylabel('Comoving y [kpc]',fontsize=10)
        if halo_index is None:
            for ax,key,title in [(axes[1,0],'field_density_contrast_smoothed','Filtered density contrast'),
                    (axes[1,1],'field_trace_free_hessian_km2_s2_proper_kpc2','Trace-free tensor norm')]:
                value = saved[prefix+key]
                if value.ndim==5:value = np.sqrt(np.sum(value**2,axis=(-2,-1)))
                image = ax.imshow(value[:,:,middle].T,origin='lower',extent=extent,cmap='viridis')
                bar = fig.colorbar(image,ax=ax,fraction=.046,pad=.04)
                bar.formatter = FuncFormatter(lambda value,position:'%.2g'%value);bar.update_ticks();bar.ax.tick_params(labelsize=8)
                bar.set_label('Dimensionless δ' if value.ndim==3 and 'contrast' in key else '(km/s)² / proper kpc²',fontsize=9)
                ax.set_title(title,fontsize=11)
                ax.set_xlabel('Comoving x [kpc]',fontsize=10);ax.set_ylabel('Comoving y [kpc]',fontsize=10)
        else:
            row = next(row for row in report['halo_measurements'] if row['smoothing_index']==smoothing_index
                       and row['halo_structure']==str(Path(report['request']['halo_structure_products'][halo_index]).resolve()))
            hp = prefix+'halo_%02d_'%halo_index
            with np.load(Path(row['halo_structure']).with_name(Path(row['halo_structure']).name+'.data')/'arrays.npz',allow_pickle=False) as profile:
                radius = profile['profile_'+report['request']['splashback_component']+'_radius_proper_kpc']
                outer = profile['profile_total_radial_outer_edge_proper_kpc']
            slope = saved[hp+'splashback_logarithmic_slope'];covariance = saved[hp+'splashback_slope_covariance']
            error = np.sqrt(np.maximum(0,np.diag(covariance)))
            if np.isfinite(slope).any():
                axes[1,0].plot(radius,slope,color='#34495e');axes[1,0].fill_between(radius,slope-error,slope+error,alpha=.2,color='#34495e')
            else:axes[1,0].text(.5,.5,'No supported derivative bins',ha='center',va='center',transform=axes[1,0].transAxes)
            axes[1,0].set_title('Density slope: '+row['splashback']['status'].lower().replace('_',' '),fontsize=10)
            axes[1,0].set_xlim(float(radius.min()),float(radius.max()))
            if not np.isfinite(slope).any():axes[1,0].set_yticks([])
            axes[1,0].set_ylabel('d ln ρ / d ln r',fontsize=10)
            for name,label in [('spin','center filament'),('radial_spin','shell mean filament')]:
                angle = saved[hp+name+'_angle_unsigned_deg']
                axes[1,1].plot(outer,angle,'o-',label=label)
            axes[1,1].set_title('Enclosed spin alignment',fontsize=11)
            axes[1,1].set_ylabel('Unsigned angle [degrees]',fontsize=10);axes[1,1].set_ylim(0,90);axes[1,1].legend(fontsize=8)
            axes[1,1].set_xlim(float(outer.min()),float(outer.max()))
            if not np.isfinite(saved[hp+'spin_angle_unsigned_deg']).any() and not np.isfinite(saved[hp+'radial_spin_angle_unsigned_deg']).any():
                axes[1,1].text(.5,.45,'No supported filament axes',ha='center',va='center',transform=axes[1,1].transAxes)
            for ax in axes[1]:ax.set_xlabel('Proper radius [kpc]',fontsize=10)
        for ax in axes.ravel():ax.tick_params(labelsize=9)
        smoothing = report['request']['smoothing_comoving_kpc'][smoothing_index]
        origin = 'Synthetic control' if report['source_origin']=='synthetic_control' else 'Native environment'
        fig.suptitle('%s · a=%.5g · Gaussian R=%.5g comoving kpc · science NOT_ACCEPTED'%(origin,report['scale_factor'],smoothing),fontsize=10)
        fig.subplots_adjust(hspace=.4,wspace=.38,top=.9)
        export_figure(fig,output_prefix,formats=['png','pdf'],metadata={
            'source_product':str(path),'source_product_sha256':before,'source_hashes':report['source_hashes'],
            'epoch':report['epoch'],'smoothing_index':smoothing_index,'halo_index':halo_index,
            'slice_comoving_z_kpc':middle*report['box_comoving_kpc']/len(web),
            'trace_slice_numerically_uniform':uniform,'trace_display_roundoff_floor':roundoff,
            'uncertainty_scope':'shell mass shot noise only; no physical convergence claim','science_acceptance':'NOT_ACCEPTED'})
    if digest(path)!=before:raise ValueError('Environment changed during figure publication')
    verify_native_environment(path)
    sidecar = Path(output_prefix).with_suffix('.figure.json');verify_figure_export(sidecar)
    return {'figure':str(sidecar),'sha256':digest(sidecar)}


def measure_reorientation_series(vectors, supported, proper_time_myr, *, unsigned,
                                 maximum_interval_myr):
    """Adjacent saved orientations, with missingness and cadence kept explicit."""
    vectors = np.asarray(vectors,dtype=float)
    support = np.asarray(supported,dtype=bool)
    time = np.asarray(proper_time_myr,dtype=float)
    if (vectors.ndim!=3 or vectors.shape[-1]!=3 or support.shape!=vectors.shape[:2]
            or time.shape!=(len(vectors),) or len(time)<2 or not np.isfinite(time).all()
            or np.any(np.diff(time)<=0) or type(unsigned) is not bool
            or isinstance(maximum_interval_myr,bool) or not np.isfinite(maximum_interval_myr)
            or maximum_interval_myr<=0 or not np.isfinite(vectors[support]).all()):
        raise ValueError('Explicit supported orientation series, strictly increasing actual times and cadence limit required')
    norm = np.linalg.norm(vectors,axis=-1)
    available = support&np.isfinite(norm)&(norm>0)
    unit = np.divide(vectors,norm[...,None],out=np.zeros_like(vectors),where=available[...,None])
    paired = available[:-1]&available[1:]
    cosine = np.sum(unit[:-1]*unit[1:],axis=-1)
    if unsigned:cosine = abs(cosine)
    angle = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    angle[~paired] = np.nan
    cadence = np.diff(time)<=maximum_interval_myr
    return {'change_angle_deg':angle,'orientation_available':available,'adjacent_supported':paired,
        'cadence_supported':cadence,'accepted_interval':paired&cadence[:,None],
        'actual_time_brackets_myr':np.column_stack([time[:-1],time[1:]]),
        'metadata':{'unsigned_axes':unsigned,'maximum_interval_myr':maximum_interval_myr,
            'scope':'orientation differences at saved actual times; unresolved sub-cadence changes remain unknown',
            'missing_or_degenerate_intervals':int((~paired).sum()),'cadence_unresolved_intervals':int((~cadence).sum())}}


@directory_owner('destination')
def publish_environment_history(request, destination, *, request_source_hashes=None):
    """Join verified fields and structures through native graph ancestry."""
    from halos import IdentityQueries,verify_halo_structure
    from core.data import export_numeric_audit,coordinate_contract
    from core.foundation import file_sha256 as digest
    import zipfile
    if (set(request)!={'schema','identity_graph_directory','observations','maximum_interval_myr','memory_gib'}
            or request['schema']!='native-environment-history-v1'
            or not isinstance(request['observations'],list) or len(request['observations'])<2
            or isinstance(request['memory_gib'],bool) or not np.isfinite(request['memory_gib'])
            or request['memory_gib']<=0):
        raise ValueError('Declare complete native-environment-history-v1 with at least two actual observations')
    graph = IdentityQueries(request['identity_graph_directory'])
    nodes = {tuple(node['key']):node for node in graph.report['nodes']}
    root = Path(request['identity_graph_directory']).resolve()
    sources = {str(root/name):digest(root/name) for name in ['report.json','report.sha256','index.sqlite','nodes.npz']}
    sources.update(graph.report['sources'])
    budget = request['memory_gib']*1024**3
    times,epochs,spins,shapes,shape_support,masses = [],[],[],[],[],[]
    center_filaments,center_support,radial_filaments,radial_support = [],[],[],[]
    feature_radii,feature_support,progenitors,keys,origins = [],[],[],[],[]
    previous,run,namespace,settings = None,None,None,None
    edges = None
    for observation in request['observations']:
        if set(observation)!={'environment_product','halo_structure_product','graph_node'}:
            raise ValueError('Each history observation requires exact environment, structure and graph node')
        environment_path = Path(observation['environment_product']).resolve()
        structure_path = Path(observation['halo_structure_product']).resolve()
        for path in [environment_path,structure_path]:
            with zipfile.ZipFile(path.with_name(path.name+'.data')/'arrays.npz') as zipped:
                if max((info.file_size for info in zipped.infolist()),default=0)*4+16*1024**2>budget:
                    raise MemoryError('History source arrays exceed declared verification workspace')
        environment = verify_native_environment(environment_path)
        origins.append(environment['source_origin'])
        structure = verify_halo_structure(structure_path)
        key = tuple(observation['graph_node'])
        epoch,time = _bind_structure_graph_observation(structure,key,graph,nodes,previous)
        previous = key
        native_run = Path(structure['request']['run_directory']).resolve()
        native_namespace = structure['data_contract']['identity_namespaces']['native_particle']
        if (run is not None and run!=native_run) or (namespace is not None and namespace!=native_namespace):
            raise ValueError('Environmental history changes native run or particle namespace')
        run,namespace = native_run,native_namespace
        if (Path(environment['request']['run_directory']).resolve()!=run
                or environment['request']['snapshot']!=structure['request']['snapshot']
                or environment['epoch']['actual_scale_factor']!=epoch['actual_scale_factor']):
            raise ValueError('Environmental history joins different actual runs or epochs')
        current_settings = {name:environment['request'].get(name) for name in
                            ['mesh_size','smoothing_comoving_kpc','web_threshold','eigenvalue_gap_floor','splashback_component',
                             'slope_controls','splashback_controls','slope_window_study']}
        current_settings['structure_population_definition'] = {
            'components':{name:{'species':sorted(row['species']),
                'membership':row.get('membership','aperture_all')}
                for name,row in structure['request']['components'].items()},
            'support_radius_proper_kpc':structure['request']['support_radius_proper_kpc'],
            'density_shape_window':structure['request']['shape_controls']['window'],
            'density_shape_estimator':structure['request']['shape_controls']['estimator'],
            'centre_definition':'fixed native FOF centre at each graph-bound observation'}
        current_edges = structure['request']['radial_edges_proper_kpc']
        if (settings is not None and settings!=current_settings) or (edges is not None and edges!=current_edges):
            raise ValueError('History requires fixed proper shell edges and explicit matching field definitions; no silent interpolation')
        settings,edges = current_settings,current_edges
        halo_paths = [str(Path(p).resolve()) for p in environment['request']['halo_structure_products']]
        if halo_paths.count(str(structure_path))!=1:
            raise ValueError('History structure is not uniquely bound by its environmental product')
        halo_index = halo_paths.index(str(structure_path))
        n = len(edges)-1
        if len(request['observations'])*len(settings['smoothing_comoving_kpc'])*n*1024+16*1024**2>budget:
            raise MemoryError('Environmental history series exceeds declared workspace')
        with np.load(structure_path.with_name(structure_path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
            spins.append(saved['profile_total_enclosed_angular_momentum_msun_kpc_km_s'])
            masses.append(saved['profile_total_enclosed_mass_msun'])
        shell_shapes = structure['density_shapes']['total']['shells']
        shape_flags = np.array([shell.get('orientation_sampling_resolved',[False]*3)[0] for shell in shell_shapes])
        shapes.append(np.array([np.asarray(shell['axes_columns'])[:,0] if flag else np.zeros(3)
                                for shell,flag in zip(shell_shapes,shape_flags)]))
        shape_support.append(shape_flags)
        local_axes,local_flags,shell_axes,shell_flags,radii,radius_flags = [],[],[],[],[],[]
        with np.load(environment_path.with_name(environment_path.name+'.data')/'arrays.npz',allow_pickle=False) as saved:
            for index in range(len(settings['smoothing_comoving_kpc'])):
                prefix = 'smoothing_%02d_halo_%02d_'%(index,halo_index)
                # The center eigensystem is retained independently of spin/shape measurements.
                local_axes.append(saved[prefix+'center_eigenvectors_columns'][0,:,0])
                local_flags.append(bool(saved[prefix+'center_filament_axis_supported'][0]))
                shell_axes.append(saved[prefix+'radial_eigenvectors_columns'][:,:,0])
                shell_flags.append(saved[prefix+'radial_filament_axis_supported'])
                row = next(row for row in environment['halo_measurements']
                           if row['smoothing_index']==index and Path(row['halo_structure']).resolve()==structure_path)
                feature = row['splashback']; measured = feature['status']=='MEASURED'
                radii.append(feature['candidates'][0]['radius_proper_kpc'] if measured else np.nan)
                radius_flags.append(measured)
        center_filaments.append(local_axes);center_support.append(local_flags)
        radial_filaments.append(shell_axes);radial_support.append(shell_flags)
        feature_radii.append(radii);feature_support.append(radius_flags)
        parents = [edge for edge in graph.report['edges'] if edge['kind']=='descendant'
                   and tuple(edge['target'])==key and edge['evidence'].get('origin')=='native_ancestry']
        progenitors.append([edge['source'] for edge in parents]);keys.append(key)
        times.append(time);epochs.append(epoch)
        for path,product in [(environment_path,environment),(structure_path,structure)]:
            sources.update(product['source_hashes'])
            for asset in [path,path.with_name(path.name+'.sha256'),path.with_name(path.name+'.data')/'arrays.npz',
                          path.with_name(path.name+'.data')/'contract.json',path.with_name(path.name+'.data')/'contract.sha256']:
                sources[str(asset)] = digest(asset)
    time = np.asarray(times);spin = np.asarray(spins);shape = np.asarray(shapes)
    if not np.isfinite(time).all() or np.any(np.diff(time)<=0):raise ValueError('Actual history proper times must strictly increase')
    enclosed = np.asarray(masses)
    dt = np.diff(time)[:,None]
    scale = np.array([epoch['actual_scale_factor'] for epoch in epochs])
    if np.any(np.diff(scale)<=0):raise ValueError('Actual history scale factors must strictly increase')
    positive_mass = (enclosed[:-1]>0)&(enclosed[1:]>0)
    logarithmic_growth = np.full_like(enclosed[:-1],np.nan)
    logarithmic_growth[positive_mass] = np.log((enclosed[1:]/np.where(enclosed[:-1]>0,enclosed[:-1],1))[positive_mass])
    logarithmic_growth /= np.diff(np.log(scale))[:,None]
    center_axes,center_flags = np.asarray(center_filaments),np.asarray(center_support)
    radial_axes,radial_flags = np.asarray(radial_filaments),np.asarray(radial_support)
    arrays = {'proper_time_myr':time,'radial_edges_proper_kpc':np.asarray(edges),
        'spin_angular_momentum_msun_kpc_km_s':spin,'shape_major_axis':shape,
        'center_filament_axis':center_axes,'center_filament_supported':center_flags,
        'radial_filament_axis':radial_axes,'radial_filament_supported':radial_flags,
        'splashback_radius_proper_kpc':np.asarray(feature_radii),'splashback_supported':np.asarray(feature_support),
        'enclosed_aperture_mass_msun':enclosed,'aperture_mass_change_msun_per_myr':np.diff(enclosed,axis=0)/dt,
        'aperture_logarithmic_growth_per_ln_a':logarithmic_growth,'aperture_growth_supported':positive_mass,
        'native_progenitor_count':np.array([len(v) for v in progenitors]),
        'multiple_progenitor_observation':np.array([len(v)>1 for v in progenitors])}
    series = [('spin',spin,np.linalg.norm(spin,axis=-1)>0,False),
              ('shape',shape,np.asarray(shape_support),True)]
    for index in range(len(settings['smoothing_comoving_kpc'])):
        series += [('center_filament_%02d'%index,center_axes[:,index,None,:],center_flags[:,index,None],True),
                   ('radial_filament_%02d'%index,radial_axes[:,index],radial_flags[:,index],True)]
    measurements = {}
    clock_errors = _proper_clock_error_bounds_myr(epochs)
    arrays['proper_clock_error_myr'] = clock_errors
    bounded_clock = np.isfinite(clock_errors[:-1])&np.isfinite(clock_errors[1:])
    duration_bound = np.diff(time)+clock_errors[:-1]+clock_errors[1:]
    for name,vectors,flags,unsigned in series:
        measured = measure_reorientation_series(vectors,flags,time,unsigned=unsigned,
                                                maximum_interval_myr=request['maximum_interval_myr'])
        measurements[name] = measured['metadata']
        measured['clock_bounded_accepted_interval'] = measured['accepted_interval']&(
            bounded_clock&(duration_bound<=request['maximum_interval_myr']))[:,None]
        measurements[name]['clock_scope']='Saved cadence and clock-bounded acceptance are separate; unbounded source clocks remain unresolved'
        arrays.update({name+'_'+key:value for key,value in measured.items() if key!='metadata'})
    correlations=[]
    for index in range(len(settings['smoothing_comoving_kpc'])):
        feature = np.asarray(feature_radii)[:,index]
        available = np.asarray(feature_support)[:,index]&(enclosed[:,-1]>0)&np.isfinite(feature)
        x,y = enclosed[available,-1],feature[available]
        coefficient = None
        if len(x)>=3 and np.std(x)>0 and np.std(y)>0:coefficient = float(np.corrcoef(x,y)[0,1])
        correlations.append({'smoothing_index':index,'supported_observations':int(available.sum()),
            'pearson_coefficient':coefficient,'status':'MEASURED_DESCRIPTIVE' if coefficient is not None else 'UNRESOLVED',
            'scope':'saved splashback radius versus total mass in the largest fixed proper aperture; correlated temporal observations, no inference p-value'})
    sources.update(request_source_hashes or {})
    sources[str(Path(__file__).resolve())] = digest(__file__)
    report = {'schema':'native-environment-history-v1','status':'MEASURED','request':request,
        'field_definitions':settings,'epoch_contracts':epochs,'graph_nodes':keys,'native_progenitors':progenitors,
        'source_hashes':sources,'reorientations':measurements,
        'aperture_assembly_definition':'total enclosed mass in fixed source-declared proper apertures, dM/dt and d ln M/d ln a; signed loss retained, not catalogue M200 accretion',
        'splashback_assembly_correlations':correlations,
        'merger_scope':'multiple original native tree progenitors flag a merger observation; no unsourced mass ratio or major-merger classification',
        'source_origin':'synthetic_control' if 'synthetic_control' in origins else 'native_snapshot',
        'simulation_resolution_convergence':'NOT_EVALUATED'}
    epoch_series = {'schema':'environment-epoch-series-v1','canonical_time':'actual source-bound proper time',
                    'row_epoch_contracts':epochs}
    def units(key):
        if key.endswith('_myr'):return 'Myr'
        if key.endswith('_deg'):return 'degrees'
        if key.endswith('_proper_kpc'):return 'proper kpc'
        if key.endswith('_msun_kpc_km_s'):return 'Msun proper kpc km/s'
        if key.endswith('_msun'):return 'Msun'
        if key.endswith('_msun_per_myr'):return 'Msun/Myr'
        return 'dimensionless'
    specs = {key:{'units':units(key),'quantity':key,
                  'missing_policy':'NaN for unsupported or degenerate orientation/splashback; explicit support and cadence flags'} for key in arrays}
    for key in arrays:
        specs[key]['epoch'] = ({**epoch_series,'row_epoch_brackets':list(zip(epochs[:-1],epochs[1:]))}
                              if any(key.endswith(suffix) for suffix in ('change_angle_deg','adjacent_supported','accepted_interval',
                                                                        'cadence_supported','actual_time_brackets_myr')) else epoch_series)
        if key in {'aperture_mass_change_msun_per_myr','aperture_logarithmic_growth_per_ln_a','aperture_growth_supported'}:
            specs[key]['epoch'] = {**epoch_series,'row_epoch_brackets':list(zip(epochs[:-1],epochs[1:]))}
    coordinates = coordinate_contract(position_units='proper kpc',
        velocity_convention='no velocities published; spin uses peculiar km/s about the source-declared host bulk',
        origin='host-centered proper shells at each source epoch; orientation vectors use fixed global Cartesian axes',
        periodic_wrapping='proper host offsets from native minimum image; environmental fields remain periodic')
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,
                                context={'epoch':epoch_series,'coordinates':coordinates},source_hashes=sources)


def verify_environment_history(path):
    from core.data import verify_numeric_audit
    from core.foundation import file_sha256 as digest
    report = verify_numeric_audit(path)
    if report.get('schema')!='native-environment-history-v1':raise ValueError('Expected native environmental history')
    for source,expected in report['source_hashes'].items():
        if digest(source)!=expected:raise ValueError('Environment history source changed: '+source)
    for observation in report['request']['observations']:
        verify_native_environment(observation['environment_product'])
    return report

# ----------------------------- I/O helpers -----------------------------



def _file_record(path, base=None):
    """Record an actual input file path, size and digest for the analysis provenance."""
    path = Path(path).resolve()
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    if base is not None:
        try:
            record["relative_path"] = path.relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            pass
    return record


def _analysis_code_sha256():
    """Hash the numerical source that generated the analysis products."""
    return _file_sha256(Path(__file__).resolve())


def _analysis_input_records(run_dir):
    """Collect the actual saved input files used by analysis, preserving their provenance."""
    run_dir = Path(run_dir).resolve()
    records = {"config": None, "run_manifest": None, "hdf5_data": [], "other_native_data": []}
    config = run_dir / "config.yaml"
    manifest = run_dir / "manifest.json"
    if config.is_file():
        records["config"] = _file_record(config, run_dir)
    if manifest.is_file():
        records["run_manifest"] = _file_record(manifest, run_dir)
    data = run_dir / "data"
    if data.is_dir():
        data_root = data.resolve()
        records["hdf5_data"] = []
        for path in sorted(data_root.rglob("*.hdf5")):
            if not path.is_file():
                continue
            item = _file_record(path)
            item["relative_path"] = (Path("data") / path.relative_to(data_root)).as_posix()
            records["hdf5_data"].append(item)
    if data.is_dir():
        for path in sorted(data.resolve().rglob('*')):
            if path.is_file() and path.suffix != '.hdf5':
                item=_file_record(path)
                item['relative_path']=(Path('data')/path.relative_to(data.resolve())).as_posix()
                records['other_native_data'].append(item)
    return records


def _begin_analysis_contract(run_dir, products):
    from core.foundation import product_provenance
    from core.data import coordinate_contract
    verify_snapshot_intake(run_dir)
    inputs=_analysis_input_records(run_dir)
    contract=product_provenance(run_dir,algorithm=__file__,
        helper_sources=[Path(__file__).with_name(name) for name in (
            'figures.py','cosmology.py','halos.py','data.py','hmf_fitting_functions.py',
            'merger_tree_visual.py','snapshot_tables.py','halo_phase_portrait.py','formation_atlas.py')],
        selection={'products':list(products),'scope':'native saved data; no scientific acceptance implied'},
        source_files=inputs['hdf5_data']+inputs['other_native_data'],
        coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
            periodic_wrapping='native wrapped box; derived local offsets must declare their own convention',
            velocity_convention='snapshot v*sqrt(a) gives peculiar km/s; SUBFIND velocities already peculiar'))
    return inputs,contract


def _verify_analysis_inputs(run_dir, inputs, contract):
    from core.foundation import verify_product_provenance
    current=_analysis_input_records(run_dir)
    if current != inputs:
        raise ValueError('Analysis input inventory or content changed during measurement')
    verify_product_provenance(run_dir,contract)


@contextmanager
def _analysis_stage(run_dir):
    from core.foundation import atomic_json
    stage=Path(tempfile.mkdtemp(prefix='.analysis_products_',dir=run_dir))
    try:
        yield stage
    except BaseException as exc:
        atomic_json(stage/'analysis_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)},'source_run':str(run_dir)})
        raise
    else:
        shutil.rmtree(stage)


def _write_analysis_provenance(output_dir, run_dir, product_selection, created_utc, code_sha256=None, *, inputs=None, data_contract=None):
    """Hash analysis inputs and every output artifact into a sidecar record."""
    output_dir = Path(output_dir)
    provenance_path = output_dir / "analysis_provenance.json"
    code_sha256 = code_sha256 or _analysis_code_sha256()
    if inputs is None or data_contract is None:
        inputs,data_contract=_begin_analysis_contract(run_dir,product_selection or ["all"])
    _verify_analysis_inputs(run_dir,inputs,data_contract)
    artifacts = [
        _file_record(path, output_dir)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.resolve() != provenance_path.resolve() and path.name != "analysis_provenance.json.sha256"
    ]
    record = {
        "schema": ANALYSIS_PROVENANCE_SCHEMA,
        "created_utc": created_utc,
        "python_version": sys.version,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": code_sha256,
        "analysis_helper_sources": {name: _file_record(Path(__file__).with_name(name)) for name in (
            'data.py', 'halos.py', 'cosmology.py', 'foundation.py', 'merger_tree_visual.py',
            'snapshot_tables.py', 'halo_phase_portrait.py', 'formation_atlas.py')},
        "source_run": str(Path(run_dir).resolve()),
        "product_selection": product_selection,
        "inputs": inputs,
        "output_artifacts": artifacts,
    }
    from core.foundation import atomic_json, file_sha256 as digest
    record['data_contract']=data_contract
    record['science_acceptance']='NOT_ACCEPTED'
    atomic_json(provenance_path,record)
    receipt=provenance_path.with_name(provenance_path.name+'.sha256')
    with receipt.open('w') as stream:
        stream.write(digest(provenance_path)+'\n');stream.flush();os.fsync(stream.fileno())
    return record

def verify_analysis_bundle(output_dir, *, verify_sources=False):
    """Verify sealed artifact bytes and readable HDF5 chunks before reuse."""
    from core.foundation import file_sha256 as digest
    output_dir=Path(output_dir).resolve()
    path=output_dir/'analysis_provenance.json'
    receipt=path.with_name(path.name+'.sha256')
    if not receipt.is_file() or receipt.read_text().strip() != digest(path):
        raise ValueError('Analysis provenance checksum mismatch')
    record=json.loads(path.read_text())
    seen=set()
    for artifact in record['output_artifacts']:
        source=(output_dir/artifact['relative_path']).resolve()
        if not source.is_relative_to(output_dir) or source in seen:
            raise ValueError('Invalid or duplicate analysis artifact path')
        seen.add(source)
        if not source.is_file() or source.stat().st_size != artifact['size_bytes'] or digest(source) != artifact['sha256']:
            raise ValueError('Analysis artifact missing or changed: '+artifact['relative_path'])
        if source.suffix in {'.h5','.hdf5'}:
            with h5py.File(source) as handle:
                def read_payload(name,dataset):
                    if not isinstance(dataset,h5py.Dataset):
                        return
                    if dataset.ndim==0:
                        dataset[()]
                    elif dataset.chunks:
                        if dataset.size:
                            for chunk in dataset.iter_chunks():
                                dataset[chunk]
                    else:
                        tail=max(1,int(np.prod(dataset.shape[1:]))*max(1,dataset.dtype.itemsize))
                        step=max(1,8*1024**2//tail)
                        for start in range(0,len(dataset),step):
                            dataset[start:start+step]
                handle.visititems(read_payload)
    if verify_sources:
        _verify_analysis_inputs(record['source_run'],record['inputs'],record['data_contract'])
    return record

def load_cfg(run_dir):
    """Read the saved YAML configuration; physical parameters are never inferred from filenames."""
    with open(Path(run_dir) / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    apply_plot_style(cfg.get("visualization", {}))
    return cfg


def _public_native_path(path, root):
    # Retained recovery/history/staging bytes must never enter public discovery.
    return not any(part.startswith('.') for part in Path(path).relative_to(root).parts)


def find_hdf5_files(base, stem):
    """Find a numbered HDF5 series using numeric shard ordering rather than lexical ordering."""
    base = Path(base)
    candidates = []
    match = re.fullmatch(r"snapshot[_-]?(\d+)", str(stem))
    if match:
        target = int(match.group(1))
        for path in base.rglob("*.hdf5"):
            found = re.fullmatch(r"(?:snapshot|snap)[_-]?(\d+)(?:\.(\d+))?\.hdf5", path.name, re.IGNORECASE)
            if found and int(found.group(1)) == target:
                candidates.append(path)
    else:
        for pattern in [stem + ".hdf5", stem + ".*.hdf5", stem + "*.hdf5"]:
            candidates.extend(base.rglob(pattern))
    from core.data import verify_discovered_native_sets
    files = sorted({path for path in candidates if not path.name.startswith("bak-") and _public_native_path(path, base)}, key=_catalogue_shard_sort_key)
    return verify_discovered_native_sets(files)


def snapshot_ids(run_dir):
    """Return available saved snapshot numbers in chronological numeric order."""
    ids = set()
    root = Path(run_dir) / "data"
    for p in root.rglob("*.hdf5"):
        if not _public_native_path(p, root): continue
        m = re.fullmatch(r"(?:snapshot|snap)[_-]?(\d+)(?:\.\d+)?\.hdf5", p.name, re.IGNORECASE)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def group_ordered_snapshot_files(run_dir, snap):
    """Select the particle order matching Group/SubhaloOffsetType.

    Independent flag-3 finding keeps its rewritten particle copy separate
    from the scientific input. Ordinary GADGET output already has FOF order.
    Never use a separate copy for products that need original optional fields.
    """
    from core.data import assert_catalogue_publication_ready
    assert_catalogue_publication_ready(run_dir,int(snap))
    data = Path(run_dir) / "data"
    ordered = find_hdf5_files(data, "grouped_snapshot_%03d" % int(snap))
    files = ordered or find_hdf5_files(data, "snapshot_%03d" % int(snap))
    if files:
        from core.data import verify_discovered_native_sets
        verify_discovered_native_sets(files)
    return files


def match_requested_snapshot_redshifts(manifest, snapshots):
    """Pair manifest output targets to snapshot numbers only when the mapping is unambiguous."""
    manifest = manifest if isinstance(manifest, dict) else {}
    if "snapshot_redshifts_requested" not in manifest:
        return {
            "status": "NOT_RECORDED", "reason": "run manifest has no requested snapshot-redshift schedule",
            "requested_redshifts": None, "by_snapshot": {},
            "method": "unavailable; requested schedule was not recorded",
        }
    try:
        raw = manifest["snapshot_redshifts_requested"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("requested schedule must be a non-empty list")
        requested = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError):
        return {
            "status": "UNAVAILABLE", "reason": "run manifest contains an invalid requested snapshot-redshift schedule",
            "requested_redshifts": None, "by_snapshot": {},
            "method": "unavailable; invalid requested schedule",
        }
    if (not np.all(np.isfinite(requested)) or np.any(np.asarray(requested) < 0)
            or (len(requested) > 1 and np.any(np.diff(requested) >= 0))):
        return {
            "status": "UNAVAILABLE", "reason": "requested redshifts must be finite, non-negative, unique, and strictly descending",
            "requested_redshifts": requested, "by_snapshot": {},
            "method": "unavailable; requested schedule failed validation",
        }
    snapshot_numbers = [int(value) for value in snapshots]
    expected_numbers = list(range(len(requested)))
    if len(snapshot_numbers) != len(requested):
        return {
            "status": "UNAVAILABLE",
            "reason": "found %d snapshots but the manifest requests %d outputs" % (len(snapshot_numbers), len(requested)),
            "requested_redshifts": requested, "by_snapshot": {},
            "method": "unavailable; output count does not match the recorded schedule",
        }
    if snapshot_numbers != expected_numbers:
        return {
            "status": "UNAVAILABLE",
            "reason": "snapshot IDs %s are not the complete sequential output indices 0..%d" % (snapshot_numbers, len(requested) - 1),
            "requested_redshifts": requested, "by_snapshot": {},
            "method": "unavailable; snapshot IDs do not establish schedule order",
        }
    return {
        "status": "MATCHED", "reason": None,
        "requested_redshifts": requested,
        "by_snapshot": dict(zip(snapshot_numbers, requested)),
        "method": "snapshot ID equals the zero-based GADGET output-list index; applied only for a complete sequential output set",
    }


def _cosmology_time_samples(run_dir, model):
    """All analysis uses the same checksum-bound CLASS proper-time clock."""
    from core.cosmology import run_proper_time_clock
    scale, times, source = run_proper_time_clock(run_dir)
    return scale, times, source.get('path')


def read_snapshot_header(run_dir, snap):
    """Read the first numeric snapshot shard header and return the full discovered shard list."""
    files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%03d" % snap)
    if not files:
        files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%d" % snap)
    if not files:
        files = find_hdf5_files(Path(run_dir) / "data", "snap_%03d" % snap)
    if not files:
        files = find_hdf5_files(Path(run_dir) / "data", "snap_%d" % snap)
    if not files:
        raise FileNotFoundError("snapshot %d" % snap)
    with h5py.File(files[0], "r") as f:
        h = dict(f["Header"].attrs)
    return h, files


def discover_snapshot_sets(folder):
    """Find one numbered GADGET snapshot series in flat or snapdir-style folders.

    Do not merge files merely because their snapshot numbers match: a parent
    directory can contain several independent run trees.  A GADGET series is
    rooted at the common directory containing its flat snapshots or its
    ``snapdir_NNN`` snapshot directories.  Requiring one such root makes a
    broad parent-folder selection fail closed instead of fabricating a time
    series from separate simulations.
    """
    folder = Path(folder).expanduser().resolve()
    found = {}
    series_roots = set()
    pattern = re.compile(r"(?:snapshot|snap)[_-]?(\d+)(?:\.(\d+))?\.hdf5$", re.IGNORECASE)
    for path in folder.rglob("*.hdf5"):
        if not _public_native_path(path, folder): continue
        match = pattern.fullmatch(path.name)
        if match:
            snapshot_number = int(match.group(1))
            parent = path.parent
            snapdir_match = re.fullmatch(r"snapdir[_-]?(\d+)", parent.name, re.IGNORECASE)
            if snapdir_match:
                directory_snapshot = int(snapdir_match.group(1))
                if directory_snapshot != snapshot_number:
                    raise ValueError(
                        "Snapshot file %s is inside %s, but the directory and filename identify different snapshots"
                        % (path, parent.name)
                    )
                series_root = parent.parent
            else:
                series_root = parent
            series_roots.add(series_root.resolve())
            found.setdefault(snapshot_number, []).append(path)
    if len(series_roots) > 1:
        roots = ", ".join(str(path) for path in sorted(series_roots))
        raise ValueError(
            "The selected folder contains numbered GADGET snapshots from multiple directory trees (%s). "
            "Select one simulation's snapshot/data folder at a time." % roots
        )
    from core.data import verify_discovered_native_sets
    return {
        snap: verify_discovered_native_sets(sorted(paths, key=_catalogue_shard_sort_key))
        for snap, paths in sorted(found.items())
    }


def validate_gadget_snapshot_set(files, snapshot_id=None):
    """Validate cosmological GADGET-4 shards for this periodic-box analyzer.

    Isolated snapshots use the domain readers with their declared native time
    and proper coordinates; Header/Time must not become a scale factor here.
    """
    files = [Path(path).resolve() for path in files]
    if not files:
        raise ValueError("No HDF5 files were supplied for a snapshot")
    from core.data import verify_native_hdf5_set, retired_black_hole_carriers
    verify_native_hdf5_set(files)
    base_header = None
    # Python integers cannot wrap while summing shard counts near uint64 limits.
    counts_sum = [0] * 6
    shard_numbers = []
    parameter_baseline = None
    required_parameters = [
        "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s",
        "HubbleParam", "Omega0", "OmegaLambda", "OmegaBaryon",
    ]
    for path in files:
        match = re.fullmatch(r"(?:grouped_snapshot|ngenic_ics|snapshot|snap)[_-]?\d+\.(\d+)\.hdf5", path.name, re.IGNORECASE)
        shard_numbers.append(int(match.group(1)) if match else None)
        with h5py.File(path, "r") as h5:
            if "Header" not in h5 or "Parameters" not in h5:
                raise ValueError(
                    "%s must contain GADGET-4 Header and Parameters groups; supply complete GADGET-4 snapshots "
                    "with explicit units and cosmology" % path
                )
            header = h5["Header"].attrs
            required_header = [
                "BoxSize", "MassTable", "NumFilesPerSnapshot", "NumPart_ThisFile", "NumPart_Total",
            ]
            missing = [name for name in required_header if name not in header]
            if not any(name in header for name in ["Time", "Redshift"]):
                missing.append("Time or Redshift")
            if missing:
                raise ValueError("%s is missing required GADGET Header metadata: %s" % (path, ", ".join(missing)))
            box_values = np.asarray(header["BoxSize"], dtype=float)
            files_values = np.asarray(header["NumFilesPerSnapshot"])
            this_file_raw = np.asarray(header["NumPart_ThisFile"])
            if box_values.size != 1 or not np.isfinite(box_values).all() or float(box_values.reshape(-1)[0]) <= 0:
                raise ValueError("Header/BoxSize must be one finite positive value in %s" % path)
            if files_values.size != 1 or not np.issubdtype(files_values.dtype, np.integer) or int(files_values.reshape(-1)[0]) < 1:
                raise ValueError("Header/NumFilesPerSnapshot must be one positive integer in %s" % path)
            if this_file_raw.shape != (6,) or not np.issubdtype(this_file_raw.dtype, np.integer) or np.any(this_file_raw < 0):
                raise ValueError("Header/NumPart_ThisFile must contain six non-negative integer counts in %s" % path)
            box_internal = float(box_values.reshape(-1)[0])
            current = {name: np.asarray(header[name]) for name in required_header}
            current["NumPart_Total"] = _snapshot_total_counts(header)
            current["Time"] = np.asarray(header.get("Time", np.nan))
            current["Redshift"] = np.asarray(header.get("Redshift", np.nan))
            if base_header is None:
                base_header = current
            else:
                for name in ["BoxSize", "MassTable", "NumFilesPerSnapshot", "NumPart_Total"]:
                    if not np.array_equal(current[name], base_header[name]):
                        raise ValueError("Snapshot shards disagree on Header/%s" % name)
                for name in ["Time", "Redshift"]:
                    left, right = float(current[name].reshape(-1)[0]), float(base_header[name].reshape(-1)[0])
                    if np.isfinite(left) and np.isfinite(right) and not np.isclose(left, right, rtol=1e-10, atol=1e-12):
                        raise ValueError("Snapshot shards disagree on Header/%s" % name)

            this_file = this_file_raw.astype(np.uint64, copy=False)
            counts_sum = [total + int(count) for total, count in zip(counts_sum, this_file)]
            mass_table = np.asarray(header["MassTable"], dtype=float)
            if mass_table.shape != (6,) or np.any(~np.isfinite(mass_table)) or np.any(mass_table < 0):
                raise ValueError("Header/MassTable must be six finite non-negative masses in %s" % path)

            parameters = h5["Parameters"].attrs
            if 'ComovingIntegrationOn' in parameters:
                mode = np.asarray(parameters['ComovingIntegrationOn'])
                if mode.size != 1 or mode.dtype.kind not in 'iu' or int(mode.reshape(-1)[0]) != 1:
                    raise ValueError('Periodic-box analysis requires ComovingIntegrationOn=1; use the isolated domain reader for native-time snapshots')
            missing = [name for name in required_parameters if name not in parameters]
            if missing:
                raise ValueError("%s/Parameters is missing required units/cosmology: %s" % (path, ", ".join(missing)))
            values = {name: float(parameters[name]) for name in required_parameters}
            for header_name, parameter_name in [
                ("HubbleParam", "HubbleParam"), ("Omega0", "Omega0"),
                ("OmegaLambda", "OmegaLambda"), ("OmegaBaryon", "OmegaBaryon"),
            ]:
                if header_name in header:
                    header_value = float(header[header_name])
                    if not np.isclose(header_value, values[parameter_name], rtol=1e-8, atol=1e-12):
                        raise ValueError("%s Header/%s disagrees with Parameters/%s" % (path, header_name, parameter_name))
            if any(not np.isfinite(values[name]) or values[name] <= 0 for name in [
                "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s", "HubbleParam", "Omega0",
            ]) or any(not np.isfinite(values[name]) for name in ["OmegaLambda", "OmegaBaryon"]) or values["OmegaBaryon"] < 0:
                raise ValueError("Snapshot parameters contain non-positive or non-finite units/cosmology in %s" % path)
            if parameter_baseline is None:
                parameter_baseline = values
            elif any(not np.isclose(values[name], parameter_baseline[name], rtol=1e-10, atol=0.0) for name in required_parameters):
                raise ValueError("Snapshot shards disagree on Parameters units or cosmology")

            for ptype, count in enumerate(this_file):
                group_name = "PartType%d" % ptype
                if count == 0:
                    if group_name in h5 and len(h5[group_name].get("Coordinates", [])) != 0:
                        raise ValueError("%s has particles but Header/NumPart_ThisFile says zero" % group_name)
                    continue
                if group_name not in h5:
                    raise ValueError("%s is missing non-empty %s" % (path, group_name))
                group = h5[group_name]
                for required_field in ["Coordinates", "ParticleIDs"]:
                    if required_field not in group or group[required_field].shape[0] != count:
                        raise ValueError("%s/%s row count does not match Header/NumPart_ThisFile in %s" % (group_name, required_field, path))
                if group["Coordinates"].ndim != 2 or group["Coordinates"].shape[1] != 3:
                    raise ValueError("%s/Coordinates must have shape (N, 3) in %s" % (group_name, path))
                if not np.issubdtype(group["Coordinates"].dtype, np.number):
                    raise ValueError("%s/Coordinates must be numeric in %s" % (group_name, path))
                particle_ids = group["ParticleIDs"]
                if particle_ids.shape != (int(count),) or particle_ids.dtype.kind not in 'iu':
                    raise ValueError("%s/ParticleIDs must be an integer vector in %s" % (group_name, path))
                if particle_ids.dtype.kind == 'i':
                    for start in range(0, int(count), 1_000_000):
                        if np.any(particle_ids[start:start + 1_000_000] < 0):
                            raise ValueError("%s/ParticleIDs contains negative identities in %s" % (group_name, path))
                coordinates = group["Coordinates"]
                for start in range(0, count, 1_000_000):
                    block = np.asarray(coordinates[start:start + 1_000_000], dtype=float)
                    if np.any(~np.isfinite(block)):
                        raise ValueError("%s/Coordinates contains NaN or infinity in %s" % (group_name, path))
                    if np.any(block < -1.0e-6) or np.any(block > box_internal + 1.0e-6):
                        raise ValueError("%s/Coordinates contains points outside the periodic box in %s" % (group_name, path))
                if mass_table[ptype] == 0 and ("Masses" not in group or group["Masses"].shape[0] != count):
                    raise ValueError("%s requires one Masses value per particle because its MassTable entry is zero" % group_name)
                if "Masses" in group:
                    if group["Masses"].shape != (int(count),) or group["Masses"].dtype.kind not in 'fiu':
                        raise ValueError("%s/Masses must be a real numeric vector in %s" % (group_name, path))
                    masses_dataset = group["Masses"]
                    for start in range(0, count, 1_000_000):
                        block = np.asarray(masses_dataset[start:start + 1_000_000], dtype=float)
                        ids = group['ParticleIDs'][start:start + len(block)]
                        retired = retired_black_hole_carriers(ptype, ids, block)
                        if np.any(~np.isfinite(block)) or np.any((block <= 0) & ~retired):
                            raise ValueError("%s/Masses must contain finite positive values in %s" % (group_name, path))
                if "Velocities" in group:
                    if not np.issubdtype(group["Velocities"].dtype, np.number) or group["Velocities"].shape != (count, 3):
                        raise ValueError("%s/Velocities must be numeric with shape (N, 3) in %s" % (group_name, path))
                    velocities = group["Velocities"]
                    for start in range(0, count, 1_000_000):
                        block = np.asarray(velocities[start:start + 1_000_000], dtype=float)
                        if np.any(~np.isfinite(block)):
                            raise ValueError("%s/Velocities contains NaN or infinity in %s" % (group_name, path))
                for field, dataset in group.items():
                    if isinstance(dataset, h5py.Dataset) and dataset.ndim > 0 and dataset.shape[0] != count:
                        raise ValueError("%s/%s row count does not match its particle count in %s" % (group_name, field, path))

                # This analyzer's particle/catalog conversion routines support the
                # common GADGET units only; reject non-standard units instead of
                # silently applying kpc/h and 1e10 Msun/h factors to them.
                for field, expected_cgs, expected_a, expected_h in [
                    ("Coordinates", values["UnitLength_in_cm"], 1.0, -1.0),
                    ("Velocities", values["UnitVelocity_in_cm_per_s"], 0.5, 0.0),
                ]:
                    if field in group:
                        attrs = group[field].attrs
                        if not all(key in attrs for key in ["to_cgs", "a_scaling", "h_scaling"]):
                            raise ValueError("%s/%s is missing field-level unit metadata" % (group_name, field))
                        if not (
                            np.isclose(float(attrs["to_cgs"]), expected_cgs, rtol=1e-8)
                            and np.isclose(float(attrs["a_scaling"]), expected_a, atol=1e-12)
                            and np.isclose(float(attrs["h_scaling"]), expected_h, atol=1e-12)
                        ):
                            raise ValueError("%s/%s uses units this analysis pipeline does not support" % (group_name, field))
                if "Masses" in group:
                    attrs = group["Masses"].attrs
                    if not all(key in attrs for key in ["to_cgs", "a_scaling", "h_scaling"]):
                        raise ValueError("%s/Masses is missing field-level unit metadata" % group_name)
                    if not (
                        np.isclose(float(attrs["to_cgs"]), values["UnitMass_in_g"], rtol=1e-8)
                        and np.isclose(float(attrs["a_scaling"]), 0.0, atol=1e-12)
                        and np.isclose(float(attrs["h_scaling"]), -1.0, atol=1e-12)
                    ):
                        raise ValueError("%s/Masses uses units this analysis pipeline does not support" % group_name)
                for field, expected_cgs, expected_a, expected_h in [
                    ("Density", values["UnitMass_in_g"] / values["UnitLength_in_cm"]**3, -3.0, 2.0),
                    ("InternalEnergy", values["UnitVelocity_in_cm_per_s"]**2, 0.0, 0.0),
                    ("StarFormationRate", MSUN_G / SECONDS_PER_YEAR, 0.0, 0.0),
                ]:
                    if field in group:
                        attrs = group[field].attrs
                        if not all(key in attrs for key in ["to_cgs", "a_scaling", "h_scaling"]):
                            raise ValueError("%s/%s is missing field-level unit metadata" % (group_name, field))
                        if not (
                            # Accept upstream GADGET's older solar-mass unit
                            # as well as this workstation's adopted unit. The
                            # measurement converts the declared g/s factor.
                            (np.isclose(float(attrs["to_cgs"]), expected_cgs, rtol=1e-8)
                             or (field == "StarFormationRate" and np.isclose(float(attrs["to_cgs"]), 1.989e33 / SECONDS_PER_YEAR, rtol=1e-8)))
                            and np.isclose(float(attrs["a_scaling"]), expected_a, atol=1e-12)
                            and np.isclose(float(attrs["h_scaling"]), expected_h, atol=1e-12)
                        ):
                            raise ValueError("%s/%s uses units this analysis pipeline does not support" % (group_name, field))

    expected_files = int(np.asarray(base_header["NumFilesPerSnapshot"]).reshape(-1)[0])
    if expected_files != len(files):
        raise ValueError("Snapshot %s declares %d files but %d HDF5 shard(s) were found" % (snapshot_id, expected_files, len(files)))
    if expected_files == 1:
        if len(files) != 1 or shard_numbers[0] not in (None, 0):
            raise ValueError("Single-file snapshot %s must be named as one unsharded file or shard zero" % snapshot_id)
    elif None in shard_numbers or sorted(shard_numbers) != list(range(expected_files)):
        raise ValueError("Snapshot %s shards must be numbered consecutively from 0 to %d" % (snapshot_id, expected_files - 1))
    if counts_sum != [int(count) for count in base_header["NumPart_Total"]]:
        raise ValueError("Snapshot shard particle counts do not add up to Header/NumPart_Total")

    length_cm = parameter_baseline["UnitLength_in_cm"]
    mass_g = parameter_baseline["UnitMass_in_g"]
    velocity_cms = parameter_baseline["UnitVelocity_in_cm_per_s"]
    if not (
        np.isclose(length_cm, 3.0856775814913673e21, rtol=1e-8)
        and np.isclose(mass_g, 1.98847e43, rtol=1e-8)
        and np.isclose(velocity_cms, 1.0e5, rtol=1e-8)
    ):
        raise ValueError("Snapshot uses non-standard code units; this analysis path currently requires kpc/h, 1e10 Msun/h, and km/s")
    box_mpc_h = float(np.asarray(base_header["BoxSize"]).reshape(-1)[0]) * length_cm / 3.0856775814913673e24
    time = float(np.asarray(base_header["Time"]).reshape(-1)[0])
    redshift = float(np.asarray(base_header["Redshift"]).reshape(-1)[0])
    if not np.isfinite(time) or time <= 0:
        if not np.isfinite(redshift) or redshift <= -1:
            raise ValueError("Snapshot has neither a valid scale factor nor a valid redshift")
        time = 1.0 / (1.0 + redshift)
    if np.isfinite(redshift) and redshift > -1 and not np.isclose(
        redshift, 1.0 / time - 1.0, rtol=1e-7, atol=1e-9
    ):
        raise ValueError("Snapshot Header/Time and Header/Redshift are inconsistent")
    if not np.isfinite(box_mpc_h) or box_mpc_h <= 0:
        raise ValueError("Snapshot BoxSize and UnitLength do not define a positive Mpc/h box")
    totals = base_header["NumPart_Total"].astype(np.uint64)
    return {
        "snapshot": snapshot_id,
        "files": [str(path) for path in files],
        "file_count": expected_files,
        "particle_counts_by_type": [int(value) for value in totals],
        "scale_factor": time,
        "redshift": 1.0 / time - 1.0,
        "box_mpc_h": box_mpc_h,
        "hubble_param": parameter_baseline["HubbleParam"],
        "omega_m": parameter_baseline["Omega0"],
        "omega_lambda": parameter_baseline["OmegaLambda"],
        "omega_baryon": parameter_baseline["OmegaBaryon"],
        "units": {
            "length_in_cm_per_h": length_cm,
            "mass_in_g_per_h": mass_g,
            "velocity_in_cm_per_s": velocity_cms,
            "supported_convention": "kpc/h; 1e10 Msun/h; km/s",
        },
        "status": "PASS",
    }


def read_particle_fields(files, ptype, fields, max_points=None, return_sample_weights=False):
    """Read aligned particle fields using one reproducible row sample.

    Every requested field for a particle type is indexed by the same rows in
    each HDF5 shard. Fields absent from every non-empty shard return as None;
    inconsistent field availability or lengths across shards is an error rather
    than a silently truncated or misaligned result. When requested, the second
    return value gives each sampled row its inverse-inclusion weight, so sums
    and profiles remain unbiased when integer shard allocations differ.
    """
    fields = list(dict.fromkeys(fields))
    if not fields:
        raise ValueError("At least one particle field must be requested")
    if max_points is not None and int(max_points) <= 0:
        raise ValueError("max_points must be positive or None")

    group = "PartType%d" % int(ptype)
    records = []
    signatures = []
    for file in files:
        with h5py.File(file, "r") as f:
            if group not in f:
                continue
            g = f[group]
            if "Coordinates" in g:
                count = len(g["Coordinates"])
            elif any(name in g for name in fields):
                count = len(g[next(name for name in fields if name in g)])
            else:
                continue
            if count == 0:
                continue
            present = tuple(name for name in fields if name in g)
            signatures.append(present)
            for name in present:
                if len(g[name]) != count:
                    raise ValueError(
                        "%s/%s has %d rows but %s/Coordinates has %d in %s"
                        % (group, name, len(g[name]), group, count, file)
                    )
            records.append((file, count, present))

    if not records:
        result = {name: None for name in fields}
        return (result, np.empty(0, dtype=np.float64)) if return_sample_weights else result
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            "Requested fields are not consistently present across non-empty %s snapshot shards"
            % group
        )

    available = signatures[0]
    total_count = sum(count for _, count, _ in records)
    sample_count = total_count
    if max_points is not None:
        sample_count = min(total_count, int(max_points))

    take_counts = np.asarray([count for _, count, _ in records], dtype=int)
    if sample_count < total_count:
        if return_sample_weights and sample_count < len(records):
            raise ValueError(
                "max_points must include at least one sampled row from each non-empty shard "
                "when inverse-inclusion weights are requested"
            )
        if sample_count >= len(records):
            take_counts = np.ones(len(records), dtype=int)
            capacity = np.asarray([count - 1 for _, count, _ in records], dtype=int)
            remaining = sample_count - len(records)
            if remaining and np.sum(capacity) > 0:
                raw = capacity.astype(float) / np.sum(capacity) * remaining
                extra = np.minimum(np.floor(raw).astype(int), capacity)
                fractions = raw - np.floor(raw)
                take_counts += extra
                remaining -= int(np.sum(extra))
                while remaining:
                    eligible = take_counts < np.asarray([count for _, count, _ in records], dtype=int)
                    priority = np.where(eligible, fractions, -np.inf)
                    selected = int(np.argmax(priority))
                    take_counts[selected] += 1
                    fractions[selected] = -np.inf
                    remaining -= 1
        else:
            raw = take_counts.astype(float) / total_count * sample_count
            take_counts = np.floor(raw).astype(int)
            remaining = sample_count - int(take_counts.sum())
            if remaining:
                order = np.argsort(-(raw - take_counts))
                take_counts[order[:remaining]] += 1

    parts = {name: [] for name in available}
    sample_weights = []
    rng = np.random.default_rng(731271 + 101 * int(ptype))
    for (file, count, present), take in zip(records, take_counts):
        with h5py.File(file, "r") as f:
            if sample_count == total_count:
                indices = None
            elif take <= 0:
                continue
            else:
                indices = np.sort(rng.choice(count, size=int(take), replace=False))
            for name in present:
                dataset = f[group][name]
                parts[name].append(dataset[:] if indices is None else dataset[indices])
            if indices is None:
                sample_weights.append(np.ones(count, dtype=np.float64))
            else:
                sample_weights.append(np.full(int(take), count / float(take), dtype=np.float64))

    result = {name: None for name in fields}
    for name, values in parts.items():
        if values:
            result[name] = np.concatenate(values, axis=0)
    if return_sample_weights:
        weights = np.concatenate(sample_weights) if sample_weights else np.empty(0, dtype=np.float64)
        if len(weights) != sample_count or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise RuntimeError("Particle sample weights do not align with the selected rows")
        return result, weights
    return result


def read_particle_field(files, ptype, field, max_points=None):
    """Read one particle field through the aligned multi-field sampler."""
    result = read_particle_fields(files, ptype, [field], max_points=max_points)
    return None if result is None else result[field]


def summarize_particle_field(files, ptype, field, chunk_size=1_000_000, nonnegative=False):
    """Summarize a scalar field over every shard, optionally rejecting negatives."""
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    group = "PartType%d" % int(ptype)
    records = []
    present = []
    for file in files:
        with h5py.File(file, "r") as f:
            if group not in f:
                continue
            particles = f[group]
            count_dataset = particles.get("Coordinates")
            if count_dataset is None:
                count_dataset = particles.get(field)
            if count_dataset is None or len(count_dataset) == 0:
                continue
            count = len(count_dataset)
            has_field = field in particles
            present.append(has_field)
            if has_field:
                dataset = particles[field]
                if dataset.ndim != 1:
                    raise ValueError("%s/%s must contain one scalar per particle" % (group, field))
                if len(dataset) != count:
                    raise ValueError("%s/%s row count does not match its particle count in %s" % (group, field, file))
            records.append((file, count, has_field))

    if not records or not any(present):
        return None
    if not all(present):
        raise ValueError("%s is not consistently present across non-empty %s snapshot shards" % (field, group))

    partial_sums = []
    finite_count = 0
    invalid_count = 0
    below_zero_count = 0
    included_count = 0
    particle_count = 0
    for file, count, _ in records:
        particle_count += count
        with h5py.File(file, "r") as f:
            dataset = f[group][field]
            for start in range(0, count, chunk_size):
                values = np.asarray(dataset[start:start + chunk_size], dtype=float)
                finite = np.isfinite(values)
                finite_count += int(np.count_nonzero(finite))
                below_zero_count += int(np.count_nonzero(finite & (values < 0.0)))
                included = finite & ((values >= 0.0) if nonnegative else True)
                if np.any(included):
                    partial_sums.append(float(np.sum(values[included], dtype=np.float64)))
                    included_count += int(np.count_nonzero(included))
                invalid_count += int(values.size - np.count_nonzero(finite))

    return {
        "sum": math.fsum(partial_sums),
        "particle_count": particle_count,
        "finite_count": finite_count,
        "invalid_count": invalid_count,
        "below_zero_count": below_zero_count,
        "included_count": included_count,
    }


def _catalogue_shard_sort_key(path):
    path = Path(path)
    match = re.match(r"^(.*)\.(\d+)\.hdf5$", path.name)
    if match:
        return str(path.parent), match.group(1), int(match.group(2))
    return str(path.parent), path.name, -1


def group_files(run_dir, snap):
    """Discover numerically ordered FoF/SUBFIND catalogue shards for one saved epoch."""
    from core.data import assert_catalogue_publication_ready
    assert_catalogue_publication_ready(run_dir,int(snap))
    d = Path(run_dir) / "data"
    candidates = []
    patterns = [
        "fof_subhalo_tab_%03d*.hdf5" % snap,
        "group_tab_%03d*.hdf5" % snap,
        "groups_%03d/*.hdf5" % snap,
        "groups_%03d/*hdf5" % snap,
    ]
    for pattern in patterns:
        candidates.extend(d.rglob(pattern))
    # GADGET retains bak-<catalogue>.hdf5 on reruns. They contain old rows,
    # never additional shards; including them doubles halo counts and masses.
    from core.data import verify_discovered_native_sets
    files = sorted({path for path in candidates if not path.name.startswith(("bak-", "subhalo_desc_", "subhalo_prog_", "subhalo_treelink_")) and _public_native_path(path, d)}, key=_catalogue_shard_sort_key)
    return verify_discovered_native_sets(files)


def read_catalog_field(run_dir, snap, top, names):
    """Concatenate the selected HDF5 catalogue field across numeric shards without substituting another mass definition."""
    files = group_files(run_dir, snap)
    chosen = None
    for file in files:
        with h5py.File(file, "r") as f:
            if top in f:
                chosen = next((name for name in names if name in f[top]), None)
                if chosen:
                    break
    if chosen is None:
        return None
    arrays = []
    count_name = "Ngroups_ThisFile" if top == "Group" else "Nsubgroups_ThisFile"
    for file in files:
        with h5py.File(file, "r") as f:
            count = int(f["Header"].attrs.get(count_name, 0)) if "Header" in f else 0
            if top in f:
                count = max([count] + [len(d) for d in f[top].values() if isinstance(d, h5py.Dataset) and d.ndim])
            if top not in f or chosen not in f[top]:
                if count:
                    raise ValueError(f"Populated catalogue shard {file} lacks {top}/{chosen}; cannot align catalogue rows")
                continue
            data = f[top][chosen][:]
            if len(data) != count:
                raise ValueError(f"Catalogue field {top}/{chosen} has {len(data)} rows, expected {count} in {file}")
            arrays.append(data)
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def group_field(run_dir, snap, names):
    """Read a named FoF/SO group catalogue field, retaining its native units for explicit conversion."""
    return read_catalog_field(run_dir, snap, "Group", names)


def subhalo_field(run_dir, snap, names):
    """Read a named bound-subhalo catalogue field, retaining its native units for explicit conversion."""
    return read_catalog_field(run_dir, snap, "Subhalo", names)


def units_to_mpc_h(pos, box_mpc_h):
    """Convert pipeline-native comoving kpc/h positions to Mpc/h without applying h again."""
    if pos is None:
        return None
    # This pipeline fixes GADGET's UnitLength to 1 kpc/h and BoxSize to
    # 1000 * L. Convert from that declared unit; never infer units from values.
    return convert_pinned_native(pos,'position_comoving_mpc_h')


def mass_to_msun_h(m):
    """Convert pipeline-native 10^10 Msun/h masses to Msun/h without a second h factor."""
    if m is None:
        return None
    return convert_pinned_native(m,'mass_msun_h')


def simulation_resolution_metadata(cfg):
    """Record spacing, particle masses and numerical resolution quantities with explicit units."""
    simulation = cfg["simulation"]
    cosmology = cfg["cosmology"]
    box = float(simulation["box_mpc_h"])
    n_side = int(simulation["particles_per_dimension"])
    h = float(cosmology["H0"]) / 100.0
    if h <= 0 or box <= 0 or n_side <= 0:
        raise ValueError("Simulation box, particle count, and Hubble parameter must be positive")
    omega_m = (float(cosmology["omega_b"]) + float(cosmology["omega_cdm"])) / h**2
    if str(simulation["physics"]).lower() == "dm":
        omega_particle = omega_m
    else:
        omega_particle = float(cosmology["omega_cdm"]) / h**2
    particle_mass = omega_particle * RHO_CRIT_MSUN_H_MPC_H3 * box**3 / n_side**3
    spacing_mpc_h = box / n_side
    return {
        "box_mpc_h": box,
        "volume_mpc_h3": box**3,
        "particles_per_dimension": n_side,
        "particle_count": n_side**3,
        "particle_mass_msun_h": float(particle_mass),
        "mean_particle_spacing_mpc_h": float(spacing_mpc_h),
        "mean_particle_spacing_kpc_h": float(1000.0 * spacing_mpc_h),
        "omega_m": float(omega_m),
        "physics": str(simulation["physics"]).lower(),
    }


def redshift_from_header(header):
    """Convert the saved cosmological scale factor to dimensionless z=1/a-1."""
    a = float(header.get("Time", np.nan))
    if np.isfinite(a) and a > 0:
        return 1.0 / a - 1.0
    return float(header.get("Redshift", np.nan))


def savefig(path, metadata=None):
    """Save a publication-styled PNG, vector PDF/SVG, and figure metadata."""
    export_figure(plt.gcf(), path, metadata=metadata)


def load_wdm_reliability(run_dir):
    """Load the validated WDM mass limit, or explain why it is unavailable."""
    run_dir = Path(run_dir)
    report_path = run_dir / "input/wdm_resolution_validation.json"
    manifest_path = run_dir / "manifest.json"
    model = ""
    if manifest_path.is_file():
        try:
            model = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("model", "")).lower()
        except (OSError, ValueError):
            pass
    is_wdm = model == "wdm" or report_path.is_file()
    if not is_wdm:
        return None
    if not report_path.is_file():
        return {"status": "UNAVAILABLE", "reason": "WDM resolution validation report is missing"}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mass_limit = float(report["wang_white_spurious_mass_msun_h"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "UNAVAILABLE", "reason": "WDM resolution validation report is invalid"}
    if not np.isfinite(mass_limit) or mass_limit <= 0:
        return {"status": "UNAVAILABLE", "reason": "WDM resolution report has no positive finite mass limit"}
    return {
        "status": str(report.get("status", "UNKNOWN")),
        "mass_limit_msun_h": mass_limit,
        "half_mode_mass_msun_h": report.get("half_mode_mass_msun_h"),
        "reference": report.get("wang_white_reference", "Wang & White 2007 empirical estimate"),
        "resolution_warning": report.get("resolution_warning"),
    }


def shade_below_wdm_limit(mass_values, reliability):
    """Shade and label the empirical WDM mass range that is not trusted."""
    if not reliability:
        return
    axes = plt.gca()
    if "mass_limit_msun_h" not in reliability:
        axes.text(
            0.02, 0.97, "WDM reliability limit unavailable: " + reliability.get("reason", "unknown reason"),
            transform=axes.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        return

    values = np.asarray(mass_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return
    limit = float(reliability["mass_limit_msun_h"])
    xmin, xmax = float(np.min(values)), float(np.max(values))
    shade_end = min(limit, xmax)
    if shade_end > xmin:
        axes.axvspan(xmin, shade_end, color="crimson", alpha=0.16, zorder=0)
    if xmin <= limit <= xmax:
        axes.axvline(limit, color="crimson", linestyle="--", linewidth=1.4)
    half_mode = reliability.get("half_mode_mass_msun_h")
    if half_mode is not None:
        try:
            half_mode = float(half_mode)
        except (TypeError, ValueError):
            half_mode = None
        if half_mode is not None and np.isfinite(half_mode) and xmin <= half_mode <= xmax:
            axes.axvline(half_mode, color="tab:blue", linestyle=":", linewidth=1.2)
    annotation = "WDM numerical reliability limit: M < %.3g M_sun/h — do not interpret as physical" % limit
    if half_mode is not None and np.isfinite(half_mode):
        annotation += "\nHalf-mode mass: %.3g M_sun/h" % half_mode
    axes.text(
        0.02, 0.97, annotation,
        transform=axes.transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


# ----------------------------- halo statistics -----------------------------

def halo_masses(run_dir, snap):
    """Return the configured host mass without silently substituting definitions."""
    cfg = load_cfg(run_dir)
    definition = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    gadget_fields = {
        "fof": ["GroupMass"],
        "m200c": ["Group_M_Crit200"],
        "m200b": ["Group_M_Mean200"],
        # In the pinned GADGET-4 SUBFIND implementation, TopHat200 is the
        # Bryan-Norman redshift-dependent virial overdensity definition.
        "mvir": ["Group_M_TopHat200"],
    }
    if definition not in gadget_fields:
        raise ValueError("analysis.hmf_mass_definition must be one of: fof, m200c, m200b, mvir")
    return group_field(run_dir, snap, gadget_fields[definition])


def halo_particle_counts(run_dir, snap):
    counts = group_field(run_dir, snap, ["GroupLen"])
    if counts is not None:
        return np.asarray(counts, dtype=np.int64).reshape(-1)
    counts_by_type = group_field(run_dir, snap, ["GroupLenType"])
    if counts_by_type is None:
        return None
    counts_by_type = np.asarray(counts_by_type)
    if counts_by_type.ndim != 2:
        raise ValueError("GroupLenType must contain one row per halo and one column per particle type")
    return np.sum(counts_by_type, axis=1, dtype=np.int64)


def apply_hmf_particle_cut(masses, particle_counts, minimum):
    """Apply an inclusive halo-particle floor and report the retained sample."""
    minimum = int(minimum)
    if minimum < 0:
        raise ValueError("analysis.hmf_min_particles must be zero or greater")
    masses = np.asarray(masses)
    if masses.ndim != 1:
        raise ValueError("Host halo masses must be one-dimensional")
    if particle_counts is None:
        if minimum:
            return None, None, {
                "status": "UNAVAILABLE", "minimum_particles": minimum,
                "reason": "GADGET catalogue has neither GroupLen nor GroupLenType",
            }
        mask = np.ones(len(masses), dtype=bool)
        return masses, mask, {
            "status": "DISABLED", "minimum_particles": 0,
            "input_halo_count": int(len(masses)), "included_halo_count": int(len(masses)),
            "excluded_halo_count": 0,
        }
    counts = np.asarray(particle_counts)
    if counts.ndim != 1 or len(counts) != len(masses):
        raise ValueError("Host halo particle counts do not align with the mass catalogue")
    mask = counts >= minimum if minimum else np.ones(len(masses), dtype=bool)
    return masses[mask], mask, {
        "status": "APPLIED" if minimum else "DISABLED",
        "minimum_particles": minimum,
        "input_halo_count": int(len(masses)),
        "included_halo_count": int(np.count_nonzero(mask)),
        "excluded_halo_count": int(len(masses) - np.count_nonzero(mask)),
    }


def hmf_resolution_selection_report(masses, particle_counts, retained_mask, minimum_particles):
    """Describe the actual catalog cut without calling it a convergence limit."""
    mass = mass_to_msun_h(np.asarray(masses, dtype=float))
    if mass.ndim != 1:
        raise ValueError("HMF masses must be one-dimensional")
    mask = np.asarray(retained_mask, dtype=bool)
    if mask.shape != mass.shape:
        raise ValueError("HMF particle selection mask does not align with halo masses")
    good = mask & np.isfinite(mass) & (mass > 0)
    result = {
        "status": "CATALOG_SELECTION_ONLY",
        "minimum_particle_count_requested": int(minimum_particles),
        "minimum_particle_count_retained": None,
        "minimum_selected_mass_msun_h": float(np.min(mass[good])) if np.any(good) else None,
        "maximum_selected_mass_msun_h": float(np.max(mass[good])) if np.any(good) else None,
        "selected_halo_count": int(np.count_nonzero(good)),
        "convergence_completeness_mass_msun_h": None,
        "interpretation": "particle-count selection and observed catalog mass floor; convergence requires a matched-resolution study",
    }
    if particle_counts is not None:
        counts = np.asarray(particle_counts)
        if counts.ndim != 1 or counts.shape != mass.shape:
            raise ValueError("HMF particle counts do not align with halo masses")
        if np.any(good):
            result["minimum_particle_count_retained"] = int(np.min(counts[good]))
    return result


def rockstar_table_from_path(path):
    """Read one Rockstar or Consistent-Trees ASCII catalogue by its header."""
    path = Path(path)
    if not path.is_file():
        return None
    header = None
    data_rows = []
    unit_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t.startswith("#") and "unit" in t.lower() and ":" in t:
                unit_lines.append(t.lstrip("#").strip())
            if t.startswith("#") and "id" in t.lower() and any(
                mass_name in t.lower() for mass_name in ["mvir", "m200", "mfof"]
            ):
                header = [re.sub(r'\(\d+\)$','',name) for name in t.lstrip("#").split()]
                continue
            if not t or t.startswith("#") or header is None:
                continue
            data_rows.append(t.split())
    if not header:
        return None
    if any(len(row) != len(header) for row in data_rows):
        return None
    integer_fields = {
        "id", "desc_id", "descid", "pid", "upid", "desc_pid", "num_p", "numprog", "num_prog",
        "np", "orig_halo_id", "tree_root_id", "phantom", "num_child_particles", "flags",
    }
    table = {}
    for index, name in enumerate(header):
        key = name.lower()
        try:
            if key in integer_fields:
                parsed=[Decimal(row[index]) for row in data_rows]
                if any(not value.is_finite() or value!=value.to_integral_value() for value in parsed):
                    raise ValueError('Catalogue integer column has a fractional or nonfinite token')
                # This pinned native Rockstar format uses signed int64 links
                # and -1 sentinels. Reject overflow rather than wrapping IDs.
                table[key] = np.asarray([int(value) for value in parsed], dtype=np.int64)
            else:
                table[key] = np.asarray([float(row[index]) for row in data_rows], dtype=float)
        except (ValueError, ArithmeticError, OverflowError):
            return None
    if unit_lines:
        table["_unit_lines"] = tuple(unit_lines)
    from core.data import canonical_native_links
    table['_canonical_identity_links']={}
    for key in ('id','desc_id','descid','pid','upid','desc_pid','orig_halo_id','tree_root_id'):
        if key not in table:continue
        identities,available=canonical_native_links(table[key])
        table['_canonical_identity_links'][key]={'values':identities,'available':available,'native_missing_sentinel':-1}
    return table


def rockstar_table(run_dir, snap):
    """Read a Rockstar ASCII catalog and return {column_name: array}."""
    from core.data import rockstar_directory
    base = rockstar_directory(run_dir)
    snap = rockstar_catalogue_index(run_dir, snap)
    path = base / ("out_%d.list" % snap)
    if not path.exists():
        path = base / ("out_%03d.list" % snap)
    return rockstar_table_from_path(path)


def rockstar_catalogue_index(run_dir, snapshot):
    """Rockstar's dense time index need not equal the GADGET snapshot number."""
    from core.data import rockstar_directory
    mapping = rockstar_directory(run_dir)/"snapshot_index.json"
    if not mapping.is_file():
        return int(snapshot)
    recorded = json.loads(mapping.read_text(encoding="utf-8"))
    matches = [row["rockstar_index"] for row in recorded["snapshots"] if row["gadget_snapshot"] == int(snapshot)]
    if len(matches) != 1:
        raise ValueError("Rockstar snapshot mapping is missing or ambiguous for snapshot %s" % snapshot)
    return int(matches[0])


def rockstar_host_table(run_dir, snap):
    """Read the parent-labelled Rockstar copy when the pipeline has prepared one."""
    from core.data import rockstar_directory
    base = rockstar_directory(run_dir)/"host_catalogues"
    index = rockstar_catalogue_index(run_dir, snap)
    path = base / ("out_%d.list" % index)
    if not path.exists():
        path = base / ("out_%03d.list" % index)
    if path.exists():
        return rockstar_table_from_path(path)
    return rockstar_table(run_dir, snap)


def halo_mass_definition_comparison(run_dir, cfg, snap):
    """Compare Rockstar mass definitions for the same host-halo IDs."""
    table = rockstar_host_table(run_dir, snap)
    if table is None:
        return {"status": "SKIPPED", "reason": "no Rockstar catalogue is available"}
    definitions = {
        "mfof": "MFOF",
        "m200c": "M200c",
        "m200b": "M200b",
        "mvir": "Mvir",
    }
    missing = [field for field in ("id", "pid", *definitions) if field not in table]
    if missing:
        return {
            "status": "SKIPPED",
            "reason": "Rockstar host catalogues are missing fields required for a common-definition sample: "
            + ", ".join(missing),
        }
    halo_ids = np.asarray(table["id"])
    parent_ids = np.asarray(table["pid"])
    masses = {name: np.asarray(table[name], dtype=np.float64) for name in definitions}
    lengths = {len(halo_ids), len(parent_ids), *(len(values) for values in masses.values())}
    if halo_ids.ndim != 1 or parent_ids.ndim != 1 or len(lengths) != 1:
        return {"status": "SKIPPED", "reason": "Rockstar mass-definition rows and host IDs do not align"}
    if len(np.unique(halo_ids)) != len(halo_ids):
        return {"status": "SKIPPED", "reason": "Rockstar halo IDs are not unique within the snapshot"}
    minimum = int(cfg["analysis"]["hmf_min_particles"])
    if minimum < 0:
        raise ValueError("analysis.hmf_min_particles must be zero or greater")
    host_mask = parent_ids == -1
    counts = table.get("num_p", table.get("np"))
    if minimum:
        if counts is None:
            return {"status": "SKIPPED", "reason": "Rockstar catalogue has no particle count for the configured host cut"}
        counts = np.asarray(counts)
        if counts.ndim != 1 or len(counts) != len(halo_ids):
            return {"status": "SKIPPED", "reason": "Rockstar particle counts do not align with halo IDs"}
        host_mask &= counts >= minimum
    ids = halo_ids[host_mask].astype(np.int64, copy=False)
    selected_masses = {name: values[host_mask] for name, values in masses.items()}
    pairwise = {}
    field_names = list(definitions)
    for index, left in enumerate(field_names):
        for right in field_names[index + 1:]:
            a = selected_masses[left]
            b = selected_masses[right]
            valid = np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > 0)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio = a[valid] / b[valid]
            valid_ratio = np.isfinite(ratio) & (ratio > 0)
            ratio = ratio[valid_ratio]
            log_ratio = np.log10(ratio)
            key = left + "_over_" + right
            pairwise[key] = {
                "numerator": definitions[left],
                "denominator": definitions[right],
                "valid_halo_count": int(len(ratio)),
                "halo_id": ids[valid][valid_ratio],
                "mass_ratio": ratio,
                "log10_mass_ratio": log_ratio,
                "median_mass_ratio": float(np.median(ratio)) if len(ratio) else None,
                "mass_ratio_percentile_16_84": (
                    np.percentile(ratio, [16, 84]).astype(float).tolist() if len(ratio) else [None, None]
                ),
                "mean_log10_mass_ratio": float(np.mean(log_ratio)) if len(log_ratio) else None,
            }
    if not len(ids):
        return {"status": "SKIPPED", "reason": "no host halos remain after the configured particle cut"}
    return {
        "status": "COMPLETE",
        "halo_id": ids,
        "mass_msun_h": selected_masses,
        "particle_count_cut": minimum,
        "host_halo_count": int(len(ids)),
        "definitions": definitions,
        "pairwise": pairwise,
        "mass_component": "Rockstar host catalogue; all definitions refer to the same halo IDs",
        "interpretation": "Measured per-halo mass-definition ratios depend on redshift, cosmology, and the selected halo population; they are not universal conversion factors.",
    }


def mutual_nearest_periodic_matches(position_a, position_b, box_size, max_separation):
    """Return conservative one-to-one mutual-nearest matches in a periodic cube."""
    a = np.asarray(position_a, dtype=float)
    b = np.asarray(position_b, dtype=float)
    box_size = float(box_size)
    max_separation = float(max_separation)
    if a.ndim != 2 or a.shape[1] != 3 or b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("Halo-centre arrays must have shape (N, 3)")
    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError("Periodic matching box size must be finite and positive")
    if not np.isfinite(max_separation) or max_separation <= 0 or max_separation >= box_size / 2.0:
        raise ValueError("Halo-centre match distance must be positive and less than half the box size")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Halo-centre arrays must contain only finite positions")
    if not len(a) or not len(b):
        return {
            "index_a": np.empty(0, dtype=np.int64),
            "index_b": np.empty(0, dtype=np.int64),
            "distance": np.empty(0, dtype=float),
        }

    a = np.mod(a, box_size)
    b = np.mod(b, box_size)
    tree_a = cKDTree(a, boxsize=box_size)
    tree_b = cKDTree(b, boxsize=box_size)
    distance_a_to_b, nearest_b = tree_b.query(a, k=1)
    _, nearest_a = tree_a.query(b, k=1)
    a_indices = np.arange(len(a), dtype=np.int64)
    mutual = nearest_a[nearest_b] == a_indices
    keep = mutual & np.isfinite(distance_a_to_b) & (distance_a_to_b <= max_separation)
    return {
        "index_a": a_indices[keep],
        "index_b": np.asarray(nearest_b[keep], dtype=np.int64),
        "distance": np.asarray(distance_a_to_b[keep], dtype=float),
    }


def rockstar_masses(run_dir, snap, definition="fof"):
    table = rockstar_table(run_dir, snap)
    if not table:
        return None
    field = {"fof": "mfof", "m200c": "m200c", "m200b": "m200b", "mvir": "mvir"}.get(str(definition).lower())
    if field is None or field not in table:
        return None
    x = np.asarray(table[field], float)
    return x[np.isfinite(x) & (x > 0)]


def plot_concentration_mass(run_dir, cfg, snap):
    """Plot Rockstar's fitted virial concentration, c_vir = Rvir / Rs."""
    table = rockstar_table(run_dir, snap)
    if table is None:
        return {"status": "SKIPPED", "reason": "no Rockstar catalogue for this snapshot"}
    required = ["mvir", "rvir", "rs"]
    missing = [field for field in required if field not in table]
    if missing:
        return {"status": "SKIPPED", "reason": "Rockstar catalogue is missing " + ", ".join(missing)}

    mass = np.asarray(table["mvir"], dtype=float)
    rvir = np.asarray(table["rvir"], dtype=float)
    rs = np.asarray(table["rs"], dtype=float)
    if any(values.ndim != 1 or len(values) != len(mass) for values in [rvir, rs]):
        return {"status": "SKIPPED", "reason": "Rockstar Mvir, Rvir, and Rs rows do not align"}
    minimum = int(cfg["analysis"]["hmf_min_particles"])
    if minimum < 0:
        raise ValueError("analysis.hmf_min_particles must be zero or greater")
    if minimum:
        particle_count = table.get("num_p", table.get("np"))
        if particle_count is None:
            return {"status": "SKIPPED", "reason": "Rockstar catalogue has no Np/num_p field for the %d-particle cut" % minimum}
        counts = np.asarray(particle_count)
        if counts.ndim != 1 or len(counts) != len(mass):
            return {"status": "SKIPPED", "reason": "Rockstar mass and num_p rows do not align"}
        keep_particles = counts >= minimum
    else:
        keep_particles = np.ones(len(mass), dtype=bool)

    concentration = np.divide(rvir, rs, out=np.full_like(rvir, np.nan), where=rs > 0)
    good = (
        keep_particles & np.isfinite(mass) & (mass > 0)
        & np.isfinite(rvir) & (rvir > 0) & np.isfinite(rs) & (rs > 0)
        & np.isfinite(concentration) & (concentration > 0)
    )
    if np.count_nonzero(good) < 5:
        return {"status": "SKIPPED", "reason": "fewer than five valid concentration measurements after particle and radius checks"}

    masses = mass[good]
    values = concentration[good]
    plt.figure(figsize=(8, 6)); plt.loglog(masses, values, ".", alpha=0.55)
    plt.xlabel(r"$M_{vir}\ [M_\odot/h]$"); plt.ylabel(r"$c_{vir}=R_{vir}/r_s$")
    plt.title("Rockstar fitted virial concentration")
    savefig(Path(run_dir) / "results" / ("concentration_mass_%03d.png" % snap))
    result = {"status": "COMPLETE", "mass_msun_h": masses, "concentration_cvir": values}
    if "id" in table:
        ids = np.asarray(table["id"])
        if ids.ndim == 1 and len(ids) == len(mass):
            result["halo_id"] = ids[good]
    return result


def hmf_mass_edges(cfg):
    """Create logarithmic mass-bin edges in Msun/h using the declared analysis bounds."""
    analysis = cfg["analysis"]
    mass_range = analysis.get("hmf_mass_range_msun_h")
    bins = analysis.get("mass_bins")
    if not isinstance(mass_range, (list, tuple)) or len(mass_range) != 2:
        raise ValueError("analysis.hmf_mass_range_msun_h must explicitly contain [minimum, maximum]")
    lower, upper = (float(value) for value in mass_range)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("HMF mass range must be finite, positive, and increasing in Msun/h")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or int(bins) < 3:
        raise ValueError("analysis.mass_bins must explicitly be an integer of at least three")
    return np.logspace(np.log10(lower), np.log10(upper), int(bins) + 1)


def hmf_spatial_jackknife(masses, positions_mpc_h, box, mass_edges_msun_h):
    """Estimate HMF spatial sample variance by omitting declared box subvolumes."""
    box = float(box)
    mass = mass_to_msun_h(masses)
    positions = np.asarray(positions_mpc_h, dtype=float) if positions_mpc_h is not None else None
    edges = np.asarray(mass_edges_msun_h, dtype=float)
    result = {
        "status": "UNAVAILABLE",
        "method": "periodic spatial delete-one jackknife with 2x2x2 equal-volume regions",
        "regions": 8,
        "modes_larger_than_box": "not estimated",
    }
    if positions is None:
        result["reason"] = "halo positions are unavailable"
        return result
    if mass.ndim != 1 or positions.shape != (len(mass), 3):
        result["reason"] = "halo masses and three-dimensional positions do not align"
        return result
    if not np.isfinite(box) or box <= 0:
        result["reason"] = "box size must be finite and positive"
        return result
    good = np.isfinite(mass) & (mass > 0) & np.all(np.isfinite(positions), axis=1)
    mass = mass[good]
    positions = np.mod(positions[good], box)
    if len(mass) < 16:
        result["reason"] = "fewer than 16 valid halos are available for eight-region resampling"
        return result
    cells = np.floor(positions * (2.0 / box)).astype(np.int64)
    region_id = cells[:, 0] * 4 + cells[:, 1] * 2 + cells[:, 2]
    dlog = np.diff(np.log10(edges))
    dmass = np.diff(edges)
    volume = box**3
    estimates_log = []
    estimates_mass = []
    for region in range(8):
        keep = region_id != region
        remaining_volume = volume * (7.0 / 8.0)
        counts = np.histogram(mass[keep], bins=edges)[0]
        estimates_log.append(counts / remaining_volume / dlog)
        estimates_mass.append(counts / remaining_volume / dmass)
    estimates_log = np.asarray(estimates_log, dtype=float)
    estimates_mass = np.asarray(estimates_mass, dtype=float)
    mean_log = np.mean(estimates_log, axis=0)
    mean_mass = np.mean(estimates_mass, axis=0)
    factor = 7.0 / 8.0
    sigma_log = np.sqrt(factor * np.sum((estimates_log - mean_log) ** 2, axis=0))
    sigma_mass = np.sqrt(factor * np.sum((estimates_mass - mean_mass) ** 2, axis=0))
    result.update({
        "status": "COMPLETE",
        "valid_halo_count": int(len(mass)),
        "regions_with_halos": int(len(np.unique(region_id))),
        "sigma_dn_dlog10M_per_mpc_h3": sigma_log.tolist(),
        "sigma_dn_dM_h4_per_msun_mpc3": sigma_mass.tolist(),
        "interpretation": "within-box spatial covariance estimate; does not capture modes larger than the periodic box",
    })
    return result


def hmf_arrays(masses, box, mass_edges_msun_h):
    """Count measured halos per mass bin and normalize by periodic volume and bin width."""
    box = float(box)
    edges = np.asarray(mass_edges_msun_h, dtype=float)
    if not np.isfinite(box) or box <= 0:
        raise ValueError("HMF box size must be finite and positive")
    if (edges.ndim != 1 or len(edges) < 4 or np.any(~np.isfinite(edges))
            or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)):
        raise ValueError("HMF mass edges must be a strictly increasing positive vector in Msun/h")
    mass = mass_to_msun_h(masses)
    mass = mass[np.isfinite(mass) & (mass > 0)]
    if len(mass) < 5:
        return None
    counts, _ = np.histogram(mass, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dlog = np.diff(np.log10(edges))
    dn_dlog = counts / box**3 / dlog
    dn_dM = counts / box**3 / np.diff(edges)
    return centers, dn_dlog, dn_dM, counts, edges


def sheth_tormen_hmf_from_linear_power(k_h_mpc, p_mpc_h3, masses_msun_h, omega_m0,
                                       delta_c=1.686, A=0.3222, a=0.707, p=0.3):
    """Shared Sheth-Tormen reference; masses Msun/h, abundance (Mpc/h)^-3."""
    from core.cosmology import compute_sigma_and_dlnsigma, f_sheth_tormen, lagrangian_radius_from_mass
    if (not all(np.isfinite(v) and v > 0 for v in (delta_c, A, a))
            or not np.isfinite(p) or p < 0):
        raise ValueError("Sheth-Tormen parameters are invalid")
    masses = np.asarray(masses_msun_h, dtype=float)
    sigma, derivative = compute_sigma_and_dlnsigma(k_h_mpc, p_mpc_h3, masses, omega_m0)
    density = float(omega_m0)*RHO_CRIT_MSUN_H_MPC_H3
    abundance = math.log(10.)*density/masses*f_sheth_tormen(sigma, delta_c=delta_c, A=A, a=a, p=p)*derivative
    return {"sigma_linear_top_hat": sigma, "dln_sigma_inverse_dlnM": derivative,
            "dn_dlog10M_mpc_h3": abundance,
            "lagrangian_radius_mpc_h": lagrangian_radius_from_mass(masses, omega_m0),
            "mean_matter_density_msun_h_mpc_h3": density}


def _manifest_artifact_path(run_dir, record):
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        return None, "run manifest lacks a checksummed input artifact record"
    recorded = Path(str(record["path"])).expanduser()
    candidates = [recorded, Path(run_dir) / "input" / recorded.name]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None, "recorded input artifact is missing: " + str(recorded.name)
    if _file_sha256(path) != str(record["sha256"]):
        return None, "recorded input artifact checksum does not match: " + str(path.name)
    return path, None


def sheth_tormen_hmf_diagnostic(run_dir, cfg, model, target_redshift,
                                masses_msun_h, measured_dn_dlog10M=None):
    """Compare a measured HMF with an explicitly limited CLASS/ST diagnostic."""
    run_dir = Path(run_dir)
    masses_msun_h = np.asarray(masses_msun_h, dtype=float)
    if masses_msun_h.ndim != 1 or len(masses_msun_h) < 2:
        return {"status": "UNAVAILABLE", "reason": "at least two HMF mass-bin centers are required"}
    if np.any(~np.isfinite(masses_msun_h)) or np.any(masses_msun_h <= 0) or np.any(np.diff(masses_msun_h) <= 0):
        return {"status": "UNAVAILABLE", "reason": "HMF mass-bin centers must be finite, positive, and increasing"}
    if str(model).lower() != "lcdm":
        return {"status": "SKIPPED", "reason": "the Sheth-Tormen diagnostic is enabled only for the cold LCDM benchmark"}
    if str(cfg.get("simulation", {}).get("physics", "dm")).lower() != "dm":
        return {"status": "SKIPPED", "reason": "hydrodynamic halo abundances need a baryon-calibrated fit, not this DMO reference"}
    analysis = cfg.get("analysis", {})
    if str(analysis["hmf_mass_definition"]).lower() != "fof":
        return {"status": "SKIPPED", "reason": "Sheth-Tormen reference is restricted to the FOF mass definition"}
    linking_length = float(cfg.get("halos", {}).get("fof_linking_length", 0.2))
    if not math.isclose(linking_length, 0.2, rel_tol=0.0, abs_tol=1.0e-12):
        return {"status": "SKIPPED", "reason": "Sheth-Tormen reference is restricted to FOF linking length b=0.2"}
    cosmology = cfg.get("cosmology", {})
    if (str(cosmology.get("engine", "class")).lower() != "class"
            or str(cfg.get("spectrum", {}).get("source", "boltzmann")).lower() != "boltzmann"
            or int(cosmology.get("N_ncdm", 0) or 0) != 0):
        return {"status": "SKIPPED", "reason": "the reference requires a CLASS cold-matter LCDM linear spectrum without massive neutrinos or imported P(k)"}
    try:
        target_redshift = float(target_redshift)
    except (TypeError, ValueError, OverflowError):
        return {"status": "UNAVAILABLE", "reason": "snapshot redshift is invalid"}
    if not np.isfinite(target_redshift) or target_redshift < 0:
        return {"status": "UNAVAILABLE", "reason": "snapshot redshift must be finite and non-negative"}
    if target_redshift >= 10.0:
        return {"status": "SKIPPED", "reason": "Sheth-Tormen can strongly overpredict rare high-redshift halo abundances and is withheld at z >= 10"}

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "UNAVAILABLE", "reason": "a run manifest with checksummed CLASS P(k) and background artifacts is required"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("model", "")).lower() != "lcdm":
            return {"status": "SKIPPED", "reason": "run manifest does not identify a cold LCDM model"}
        artifacts = manifest.get("input_artifacts", {})
        pk_path, error = _manifest_artifact_path(run_dir, artifacts.get("linear_power_spectrum"))
        if error:
            return {"status": "UNAVAILABLE", "reason": error}
        background_path, error = _manifest_artifact_path(run_dir, artifacts.get("boltzmann_background"))
        if error:
            return {"status": "UNAVAILABLE", "reason": error}
        meta_path = run_dir / "input" / "lcdm_cosmology.json"
        if not meta_path.is_file():
            return {"status": "UNAVAILABLE", "reason": "the CLASS cosmology metadata sidecar is missing"}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (str(meta.get("model", "")).lower() != "lcdm"
                or str(meta.get("engine", "")).lower() != "class"
                or Path(str(meta.get("pk_file", ""))).name != pk_path.name
                or Path(str(meta.get("background_file", ""))).name != background_path.name):
            return {"status": "INCOMPATIBLE", "reason": "CLASS metadata sidecar disagrees with the checksummed run inputs"}
        if meta.get("pk_sha256") != artifacts["linear_power_spectrum"].get("sha256"):
            return {"status": "INCOMPATIBLE", "reason": "CLASS metadata and run manifest P(k) checksums differ"}
        pk_redshift_match = re.search(
            r"redshift\s+z\s*=\s*([-+0-9.eE]+)",
            pk_path.read_text(encoding="utf-8", errors="replace")[:4096], re.IGNORECASE,
        )
        if pk_redshift_match is None:
            return {"status": "UNAVAILABLE", "reason": "CLASS P(k) file does not record its reference redshift"}
        pk_redshift = float(pk_redshift_match.group(1))
        recorded_redshift = artifacts["linear_power_spectrum"].get("redshift")
        if (recorded_redshift is None or not np.isclose(pk_redshift, float(recorded_redshift), rtol=0.0, atol=5.1e-4)
                or not np.isclose(float(meta.get("output_redshift", np.nan)), pk_redshift, rtol=0.0, atol=5.1e-4)):
            return {"status": "INCOMPATIBLE", "reason": "P(k) reference redshift differs across the file, manifest, and cosmology metadata"}
        pk_table = np.loadtxt(pk_path, comments="#", ndmin=2)
        background = np.loadtxt(background_path, comments="#", ndmin=2)
        if pk_table.ndim != 2 or pk_table.shape[1] < 2 or background.ndim != 2 or background.shape[1] < 20:
            return {"status": "UNAVAILABLE", "reason": "CLASS P(k) or background table has an unsupported column layout"}
        k, power = pk_table[:, 0], pk_table[:, 1]
        z_table, growth = background[:, 0], background[:, 19]
        if (np.any(~np.isfinite(k)) or np.any(~np.isfinite(power)) or np.any(k <= 0) or np.any(power <= 0)
                or np.any(np.diff(k) <= 0) or np.any(~np.isfinite(z_table)) or np.any(z_table < 0)
                or np.any(~np.isfinite(growth)) or np.any(growth <= 0)):
            return {"status": "UNAVAILABLE", "reason": "CLASS spectrum/background contains invalid k, P(k), redshift, or growth values"}
        loga = np.log(1.0 / (1.0 + z_table))
        order = np.argsort(loga)
        loga, growth = loga[order], growth[order]
        if np.any(np.diff(loga) <= 0):
            return {"status": "UNAVAILABLE", "reason": "CLASS background redshifts are not unique"}
        target_loga = math.log(1.0 / (1.0 + target_redshift))
        input_loga = math.log(1.0 / (1.0 + pk_redshift))
        if target_loga < loga[0] or target_loga > loga[-1] or input_loga < loga[0] or input_loga > loga[-1]:
            return {"status": "UNAVAILABLE", "reason": "snapshot or input redshift lies outside the CLASS growth table"}
        D_target = float(np.interp(target_loga, loga, growth))
        D_input = float(np.interp(input_loga, loga, growth))
        if not np.isfinite(D_target / D_input) or D_input <= 0:
            return {"status": "UNAVAILABLE", "reason": "CLASS growth-factor interpolation failed"}
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        return {"status": "UNAVAILABLE", "reason": "could not validate CLASS analytic-HMF inputs: " + str(exc)}

    h = float(cosmology.get("H0", np.nan)) / 100.0
    omega_m0 = (
        float(cosmology.get("omega_b", 0.0)) + float(cosmology.get("omega_cdm", 0.0))
    ) / h**2 if np.isfinite(h) and h > 0 else float("nan")
    try:
        prediction = sheth_tormen_hmf_from_linear_power(
            k, power * (D_target / D_input) ** 2, masses_msun_h, omega_m0,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return {"status": "UNAVAILABLE", "reason": "could not evaluate the Sheth-Tormen diagnostic: " + str(exc)}
    radius = np.asarray(prediction["lagrangian_radius_mpc_h"], dtype=float)
    k_covered = (k[0] * radius <= 0.05) & (k[-1] * radius >= 20.0)
    if np.count_nonzero(k_covered) < 2:
        return {
            "status": "UNAVAILABLE",
            "reason": "fewer than two HMF mass bins have broad enough CLASS k coverage (k_min R <= 0.05 and k_max R >= 20)",
            "input_power_spectrum": str(pk_path),
            "input_power_spectrum_sha256": artifacts["linear_power_spectrum"]["sha256"],
            "input_power_redshift": pk_redshift, "output_redshift": target_redshift,
            "k_min_h_mpc": float(k[0]), "k_max_h_mpc": float(k[-1]),
            "supported_mass_bin_count": int(np.count_nonzero(k_covered)),
        }
    fit_rate = np.asarray(prediction["dn_dlog10M_mpc_h3"], dtype=float).copy()
    fit_rate[~k_covered] = np.nan
    report = {
        "status": "DIAGNOSTIC_ONLY", "reason": None,
        "input_power_spectrum": str(pk_path), "input_power_spectrum_sha256": artifacts["linear_power_spectrum"]["sha256"],
        "input_power_redshift": pk_redshift, "output_redshift": target_redshift,
        "growth_rescaling": (D_target / D_input) ** 2,
        "growth_factor_at_input_redshift": D_input, "growth_factor_at_output_redshift": D_target,
        "k_min_h_mpc": float(k[0]), "k_max_h_mpc": float(k[-1]),
        "omega_m0": omega_m0, "delta_c": 1.686,
        "fit": "Sheth-Tormen ellipsoidal-collapse multiplicity fit (A=0.3222, a=0.707, p=0.3; delta_c=1.686)",
        "fit_domain": "diagnostic only for collisionless LCDM, FOF b=0.2; not a precision prediction or validation criterion",
        "limitations": "only bins with k_min R <= 0.05 and k_max R >= 20 are reported; finite CLASS k-range is not extrapolated; fit is not calibrated for EDE, WDM, hydro, or z>=10; finite-volume and calibration errors are not included",
        "mass_msun_h": masses_msun_h,
        "input_k_coverage_mask": k_covered,
        "sigma_linear_top_hat": prediction["sigma_linear_top_hat"],
        "dln_sigma_inverse_dlnM": prediction["dln_sigma_inverse_dlnM"],
        "dn_dlog10M_mpc_h3": fit_rate,
    }
    if measured_dn_dlog10M is not None:
        measured = np.asarray(measured_dn_dlog10M, dtype=float)
        if measured.shape != masses_msun_h.shape:
            return {"status": "UNAVAILABLE", "reason": "measured HMF rates do not align with the mass-bin centers"}
        ratio = np.full(measured.shape, np.nan)
        valid = np.isfinite(measured) & (measured > 0)
        valid &= k_covered
        ratio[valid] = measured[valid] / report["dn_dlog10M_mpc_h3"][valid]
        report["measurement_over_fit_ratio"] = ratio
        report["compared_bin_count"] = int(np.count_nonzero(valid))
    return report


def poisson_rate_interval(counts, volume, widths, confidence=0.682689492137):
    """Exact Garwood central Poisson interval for number density per bin."""
    raw_counts = np.asarray(counts)
    if raw_counts.ndim != 1 or np.any(~np.isfinite(raw_counts)) or np.any(raw_counts != np.floor(raw_counts)):
        raise ValueError("Poisson interval counts must be finite integers")
    counts = raw_counts.astype(np.int64)
    widths = np.asarray(widths, dtype=float)
    if counts.ndim != 1 or widths.shape != counts.shape or np.any(counts < 0):
        raise ValueError("Poisson interval counts and bin widths must be aligned one-dimensional arrays")
    volume = float(volume)
    if not np.isfinite(volume) or volume <= 0 or np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Poisson interval volume and widths must be finite and positive")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Poisson interval confidence must be strictly between zero and one")
    alpha = 1.0 - confidence
    lower_count = np.zeros(counts.shape, dtype=float)
    positive = counts > 0
    lower_count[positive] = 0.5 * chi2.ppf(alpha / 2.0, 2.0 * counts[positive])
    upper_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (counts + 1))
    scale = volume * widths
    return lower_count / scale, upper_count / scale


def poisson_rate_ratio_interval(counts_a, counts_b, exposure_a, exposure_b, confidence=0.682689492137):
    """Exact conditional Clopper-Pearson interval for rate_b / rate_a.

    The counts are independent Poisson measurements with known exposures
    (volume times bin width). Conditioning on their sum gives a binomial
    likelihood and an exact interval for the rate ratio.
    """
    counts_a = np.asarray(counts_a)
    counts_b = np.asarray(counts_b)
    if counts_a.shape != counts_b.shape or counts_a.ndim != 1:
        raise ValueError("Rate-ratio counts must be aligned one-dimensional arrays")
    for counts in (counts_a, counts_b):
        if np.any(~np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("Rate-ratio counts must be finite non-negative integers")
    counts_a = counts_a.astype(np.int64)
    counts_b = counts_b.astype(np.int64)
    exposure_a = np.broadcast_to(np.asarray(exposure_a, dtype=float), counts_a.shape)
    exposure_b = np.broadcast_to(np.asarray(exposure_b, dtype=float), counts_b.shape)
    if (
        np.any(~np.isfinite(exposure_a)) or np.any(exposure_a <= 0)
        or np.any(~np.isfinite(exposure_b)) or np.any(exposure_b <= 0)
    ):
        raise ValueError("Rate-ratio exposures must be finite and positive")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Rate-ratio confidence must be strictly between zero and one")

    alpha = 1.0 - confidence
    total = counts_a + counts_b
    p_lower = np.zeros(total.shape, dtype=float)
    p_upper = np.ones(total.shape, dtype=float)
    positive_a = counts_a > 0
    positive_b = counts_b > 0
    p_lower[positive_a] = beta.ppf(
        alpha / 2.0, counts_a[positive_a], counts_b[positive_a] + 1
    )
    p_upper[positive_b] = beta.ppf(
        1.0 - alpha / 2.0, counts_a[positive_b] + 1, counts_b[positive_b]
    )
    lower = np.zeros(total.shape, dtype=float)
    upper = np.full(total.shape, np.inf, dtype=float)
    finite_lower = p_upper > 0
    lower[finite_lower] = (
        exposure_a[finite_lower] * (1.0 - p_upper[finite_lower])
        / (exposure_b[finite_lower] * p_upper[finite_lower])
    )
    finite_upper = p_lower > 0
    upper[finite_upper] = (
        exposure_a[finite_upper] * (1.0 - p_lower[finite_upper])
        / (exposure_b[finite_upper] * p_lower[finite_upper])
    )
    return lower, upper


def plot_hmfs(run_dir, cfg, snap, masses, reliability, positions_mpc_h, jackknife):
    """Plot catalogue abundances with the recorded finder, mass definition, completeness cuts and counting intervals."""
    resolution = simulation_resolution_metadata(cfg)
    edges = hmf_mass_edges(cfg)
    arr = hmf_arrays(masses, resolution["box_mpc_h"], edges)
    if arr is None:
        return None
    mass, dn_dlog, dn_dM, counts, edges = arr
    volume = resolution["volume_mpc_h3"]
    lower_log, upper_log = poisson_rate_interval(counts, volume, np.diff(np.log10(edges)))
    lower_mass, upper_mass = poisson_rate_interval(counts, volume, np.diff(edges))
    if not isinstance(jackknife, dict):
        raise ValueError("HMF spatial-jackknife metadata must be supplied explicitly")
    jack_sigma_log = np.asarray(jackknife.get("sigma_dn_dlog10M_per_mpc_h3", np.full_like(dn_dlog, np.nan)), dtype=float)
    jack_sigma_mass = np.asarray(jackknife.get("sigma_dn_dM_h4_per_msun_mpc3", np.full_like(dn_dM, np.nan)), dtype=float)
    analysis = cfg["analysis"]
    definition = str(analysis["hmf_mass_definition"]).lower()
    minimum_particles = int(analysis["hmf_min_particles"])
    particle_floor = minimum_particles * resolution["particle_mass_msun_h"]
    box = resolution["box_mpc_h"]
    z = redshift_from_header(read_snapshot_header(run_dir, snap)[0])
    common_metadata = {
        "product": "dark-matter-only halo mass function",
        "snapshot": int(snap),
        "redshift": float(z),
        "mass_definition": definition,
        "box_size_mpc_h": box,
        "volume_mpc_h3": volume,
        "particles_per_dimension": resolution["particles_per_dimension"],
        "particle_count": resolution["particle_count"],
        "particle_mass_msun_h": resolution["particle_mass_msun_h"],
        "mean_particle_spacing_mpc_h": resolution["mean_particle_spacing_mpc_h"],
        "minimum_halo_particles": minimum_particles,
        "particle_count_mass_floor_msun_h": float(particle_floor),
        "mass_edges_msun_h": edges.tolist(),
        "counts": counts.tolist(),
        "poisson_uncertainty": "68.27% central Garwood intervals",
        "spatial_jackknife": jackknife,
        "finite_volume_limitation": "within-box jackknife cannot estimate modes larger than this periodic box",
    }
    positive = counts > 0
    zero = ~positive
    def display_limits(lower, upper, value, sigma):
        measured_lower = lower[positive]
        candidates = np.r_[measured_lower[measured_lower > 0], upper[zero]]
        floor = float(np.min(candidates)) * 0.45 if len(candidates) else 1e-6
        top = np.r_[upper, value[positive] + np.nan_to_num(sigma[positive], nan=0.0)]
        ceiling = float(np.max(top)) * 1.8
        return floor, max(ceiling, floor * 10)
    log_limits = display_limits(lower_log, upper_log, dn_dlog, jack_sigma_log)
    mass_limits = display_limits(lower_mass, upper_mass, dn_dM, jack_sigma_mass)
    common_metadata["jackknife_display"] = "Band clipped at positive axis boundary when its lower limit reaches zero; exported numerical uncertainties are unchanged"
    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(positive):
        ax.errorbar(
            mass[positive], dn_dlog[positive],
            yerr=[dn_dlog[positive] - lower_log[positive], upper_log[positive] - dn_dlog[positive]],
            fmt="o", color="#244969", markeredgecolor="#ffffff", markeredgewidth=0.45,
            ecolor="#244969", elinewidth=1.1, capsize=2.4, label="Measured host abundance",
            zorder=4,
        )
    if jackknife["status"] == "COMPLETE":
        band = positive & np.isfinite(jack_sigma_log)
        if np.any(band):
            ax.fill_between(
                mass[band], np.maximum(dn_dlog[band] - jack_sigma_log[band], log_limits[0]),
                dn_dlog[band] + jack_sigma_log[band], color="#819794", alpha=0.15,
                linewidth=0, label=r"Within-box spatial jackknife $1\sigma$",
                zorder=1,
            )
    if np.any(zero):
        ax.scatter(mass[zero], upper_log[zero], marker="v", s=22, color="#b57547",
                   edgecolor="#10243a", linewidth=0.4, label="Zero-count 68% upper limit", zorder=5)
    if particle_floor > edges[0]:
        ax.axvspan(edges[0], min(particle_floor, edges[-1]), color="#718096", alpha=0.16,
                   label="Below configured particle-count floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*log_limits)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Halo mass $M$ [$M_\odot/h$]")
    ax.set_ylabel(r"$dn/d\log_{10}M$ [$(h/\mathrm{Mpc})^3$]")
    ax.set_title(
        f"{definition.upper()} halo mass function  |  z={z:.3f}  |  "
        f"L={box:g} Mpc/h, N={resolution['particles_per_dimension']}^3"
    )
    ax.legend(loc="best")
    ax.text(
        0.02, 0.02,
        rf"$m_p$={resolution['particle_mass_msun_h']:.3g} $M_\odot/h$   "
        rf"$V$={volume:.3g} $(\mathrm{{Mpc}}/h)^3$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#d7dee8", "alpha": 0.92, "pad": 3},
    )
    savefig(Path(run_dir) / "results" / ("HMF_%03d.png" % snap), metadata=common_metadata)

    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(positive):
        ax.errorbar(
            mass[positive], dn_dM[positive],
            yerr=[dn_dM[positive] - lower_mass[positive], upper_mass[positive] - dn_dM[positive]],
            fmt="o", color="#244969", markeredgecolor="#ffffff", markeredgewidth=0.45,
            ecolor="#244969", elinewidth=1.1, capsize=2.4, label="Measured host abundance",
            zorder=4,
        )
    if jackknife["status"] == "COMPLETE":
        band = positive & np.isfinite(jack_sigma_mass)
        if np.any(band):
            ax.fill_between(
                mass[band], np.maximum(dn_dM[band] - jack_sigma_mass[band], mass_limits[0]),
                dn_dM[band] + jack_sigma_mass[band], color="#819794", alpha=0.15,
                linewidth=0, label=r"Within-box spatial jackknife $1\sigma$",
                zorder=1,
            )
    if np.any(zero):
        ax.scatter(mass[zero], upper_mass[zero], marker="v", s=22, color="#b57547",
                   edgecolor="#10243a", linewidth=0.4, label="Zero-count 68% upper limit", zorder=5)
    if particle_floor > edges[0]:
        ax.axvspan(edges[0], min(particle_floor, edges[-1]), color="#718096", alpha=0.16,
                   label="Below configured particle-count floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*mass_limits)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Halo mass $M$ [$M_\odot/h$]")
    ax.set_ylabel(r"$dn/dM$ [$h^4\,M_\odot^{-1}\,\mathrm{Mpc}^{-3}$]")
    ax.set_title(
        f"Differential {definition.upper()} abundance  |  z={z:.3f}  |  "
        f"L={box:g} Mpc/h, N={resolution['particles_per_dimension']}^3"
    )
    ax.text(
        0.02, 0.02,
        rf"$m_p$={resolution['particle_mass_msun_h']:.3g} $M_\odot/h$   "
        rf"$V$={volume:.3g} $(\mathrm{{Mpc}}/h)^3$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#d7dee8", "alpha": 0.92, "pad": 3},
    )
    ax.legend(loc="best")
    savefig(Path(run_dir) / "results" / ("HMF_dn_dM_%03d.png" % snap), metadata=common_metadata)

    csv_path = Path(run_dir) / "results" / "hmf" / ("snapshot_%03d_hmf.csv" % snap)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "mass_lower_msun_h", "mass_upper_msun_h", "mass_center_msun_h", "halo_count",
            "dn_dlog10M_per_mpc_h3", "poisson68_lower_dn_dlog10M_per_mpc_h3",
            "poisson68_upper_dn_dlog10M_per_mpc_h3", "spatial_jackknife_sigma_dn_dlog10M_per_mpc_h3",
            "dn_dM_h4_per_msun_mpc3", "poisson68_lower_dn_dM_h4_per_msun_mpc3",
            "poisson68_upper_dn_dM_h4_per_msun_mpc3", "spatial_jackknife_sigma_dn_dM_h4_per_msun_mpc3",
            "redshift", "box_size_mpc_h", "volume_mpc_h3", "particle_mass_msun_h",
            "minimum_halo_particles", "mass_definition",
        ])
        for index in range(len(counts)):
            writer.writerow([
                edges[index], edges[index + 1], mass[index], int(counts[index]), dn_dlog[index],
                lower_log[index], upper_log[index], jack_sigma_log[index], dn_dM[index],
                lower_mass[index], upper_mass[index], jack_sigma_mass[index],
                z, box, volume, resolution["particle_mass_msun_h"], minimum_particles, definition,
            ])
    return arr


def _analytic_hmf_summary(report):
    return {
        key: value for key, value in report.items()
        if not isinstance(value, (np.ndarray, np.generic))
    }


def plot_analytic_hmf_check(run_dir, snap, measured_hmf, report, volume):
    """Compare measured abundances to the shared linear-spectrum reference in a compatible mass convention."""
    if not isinstance(report, dict) or report.get("status") != "DIAGNOSTIC_ONLY" or measured_hmf is None:
        return None
    cfg = load_cfg(run_dir)
    resolution = simulation_resolution_metadata(cfg)
    mass, measured, _, counts, edges = measured_hmf
    predicted = np.asarray(report["dn_dlog10M_mpc_h3"], dtype=float)
    if predicted.shape != np.asarray(measured).shape:
        raise ValueError("Sheth-Tormen prediction does not align with measured HMF bins")
    lower, upper = poisson_rate_interval(counts, volume, np.diff(np.log10(edges)))
    figure, (abundance_axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.04},
    )
    mass = np.asarray(mass)
    counts = np.asarray(counts)
    measured = np.asarray(measured)
    populated = (counts > 0) & np.isfinite(measured) & (measured > 0)
    empty = counts == 0
    fit_supported = np.isfinite(predicted) & (predicted > 0)
    abundance_axis.loglog(
        mass[fit_supported], predicted[fit_supported],
        color="#176b9b", linewidth=2.0, label="Sheth–Tormen FOF b=0.2 diagnostic",
        zorder=2,
    )
    abundance_axis.errorbar(
        mass[populated], measured[populated],
        yerr=[measured[populated] - lower[populated], upper[populated] - measured[populated]],
        fmt="o", color="#df6b2f", markeredgecolor="#ffffff", markeredgewidth=0.45,
        ecolor="#df9b6a", elinewidth=1.15, capsize=2.4, markersize=4.0,
        label="Measured HMF (68.27% Garwood interval)", zorder=4,
    )
    abundance_axis.scatter(
        mass[empty & fit_supported], upper[empty & fit_supported], marker="v", s=24,
        color="#29aaa4", edgecolor="#143748", linewidth=0.45,
        label="Zero-count 68% upper limit", zorder=5,
    )
    particle_floor = int(cfg["analysis"]["hmf_min_particles"]) * resolution["particle_mass_msun_h"]
    if particle_floor > edges[0]:
        for axis in (abundance_axis, ratio_axis):
            axis.axvspan(edges[0], min(particle_floor, edges[-1]), color="#718096", alpha=0.12,
                         label="Below configured particle-count floor")
    abundance_axis.set_ylabel(r"$dn/d\log_{10}M$ [$(h/\mathrm{Mpc})^3$]")
    abundance_axis.set_title(
        "HMF diagnostic at z=%.4g  |  L=%.5g Mpc/h  |  N=%d^3" % (
            float(report["output_redshift"]), resolution["box_mpc_h"],
            resolution["particles_per_dimension"],
        )
    )
    abundance_axis.text(
        0.02, 0.02,
        rf"$m_p$={resolution['particle_mass_msun_h']:.3g} $M_\odot/h$   "
        rf"$V$={volume:.3g} $(\mathrm{{Mpc}}/h)^3$",
        transform=abundance_axis.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#d7dee8", "alpha": 0.92, "pad": 3},
    )
    abundance_axis.legend(loc="best")

    ratio = np.asarray(report.get("measurement_over_fit_ratio", np.full_like(predicted, np.nan)), dtype=float)
    ratio_valid = populated & fit_supported & np.isfinite(ratio) & (ratio > 0)
    ratio_empty = empty & fit_supported & np.isfinite(upper) & (upper > 0)
    ratio_lower = np.full_like(ratio, np.nan)
    ratio_upper = np.full_like(ratio, np.nan)
    ratio_lower[ratio_valid] = lower[ratio_valid] / predicted[ratio_valid]
    ratio_upper[ratio_valid] = upper[ratio_valid] / predicted[ratio_valid]
    ratio_axis.fill_between(
        mass[ratio_valid], ratio_lower[ratio_valid], ratio_upper[ratio_valid],
        color="#df6b2f", alpha=0.16, linewidth=0,
        label="Garwood interval / fit",
    )
    ratio_axis.plot(
        mass[ratio_valid], ratio[ratio_valid], marker="o", linestyle="none",
        color="#df6b2f", markersize=4.0, label="Measured / fit",
    )
    ratio_axis.scatter(
        mass[ratio_empty], upper[ratio_empty] / predicted[ratio_empty], marker="v", s=22,
        color="#29aaa4", edgecolor="#143748", linewidth=0.4,
    )
    ratio_axis.axhline(1.0, color="#334155", linewidth=1.0, linestyle="--")
    ratio_axis.set_xscale("log")
    ratio_axis.set_yscale("log")
    ratio_axis.set_xlim(edges[0], edges[-1])
    ratio_axis.set_xlabel(r"Halo mass $M$ [$M_\odot/h$]")
    ratio_axis.set_ylabel("Measured / fit")
    ratio_axis.grid(True, which="both", axis="y", alpha=0.45)
    if np.any(ratio_valid):
        ratio_axis.legend(loc="best", fontsize=7)
    figure.align_ylabels((abundance_axis, ratio_axis))
    metadata = {
        "product": "analytic halo mass function diagnostic with residual ratio panel",
        "snapshot": int(snap),
        "redshift": float(report["output_redshift"]),
        "fit": report.get("fit"),
        "fit_domain": report.get("fit_domain"),
        "fit_limitations": report.get("limitations"),
        "residual_definition": "measured HMF divided by analytic diagnostic fit; shaded band is the Garwood Poisson interval divided by that fit",
        "compared_bin_count": int(np.count_nonzero(ratio_valid)),
        "zero_count_bin_count": int(np.count_nonzero(empty)),
        "box_size_mpc_h": resolution["box_mpc_h"],
        "volume_mpc_h3": float(volume),
        "particles_per_dimension": resolution["particles_per_dimension"],
        "particle_mass_msun_h": resolution["particle_mass_msun_h"],
        "mean_particle_spacing_mpc_h": resolution["mean_particle_spacing_mpc_h"],
        "minimum_halo_particles": int(cfg["analysis"]["hmf_min_particles"]),
        "particle_count_mass_floor_msun_h": float(particle_floor),
        "mass_definition": str(cfg["analysis"]["hmf_mass_definition"]),
        "mass_edges_msun_h": edges.tolist(),
        "abundance_unit": "(h/Mpc)^3",
        "mass_unit": "Msun/h",
        "finite_volume_variance": "not estimated from this single realization",
    }
    savefig(Path(run_dir) / "results" / ("HMF_analytic_check_%03d.png" % snap), metadata=metadata)
    return str(Path(run_dir) / "results" / ("HMF_analytic_check_%03d.png" % snap))


def plot_halo_finder_comparison(run_dir, cfg, snap):
    """Compare host-halo mass functions and periodically matched halo centres."""
    definition = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    if definition not in {"fof", "m200c", "m200b", "mvir"}:
        raise ValueError("analysis.hmf_mass_definition must be one of: fof, m200c, m200b, mvir")
    table = rockstar_host_table(run_dir, snap)
    if table is None:
        return {"status": "SKIPPED", "reason": "no Rockstar catalogue for this snapshot", "mass_definition": definition}
    parent_ids = table.get("pid", table.get("parent_id"))
    if parent_ids is None:
        return {
            "status": "SKIPPED",
            "reason": "Rockstar catalogue has no host/subhalo parent IDs; run the host-catalogue preparation step before comparing host mass functions",
            "mass_definition": definition,
        }
    parent_ids = np.asarray(parent_ids)
    if parent_ids.ndim != 1 or len(parent_ids) != len(table.get("id", [])):
        return {"status": "SKIPPED", "reason": "Rockstar parent IDs do not align with halo rows", "mass_definition": definition}

    rockstar_field = {"fof": "mfof", "m200c": "m200c", "m200b": "m200b", "mvir": "mvir"}[definition]
    if rockstar_field not in table:
        return {
            "status": "SKIPPED",
            "reason": "Rockstar catalogue has no %s field for the configured %s comparison" % (rockstar_field, definition),
            "mass_definition": definition,
        }
    hydro = str(cfg.get("simulation", {}).get("physics", "dm")).lower() == "hydro"
    if hydro and definition != "fof":
        return {
            "status": "SKIPPED",
            "reason": "Rockstar is configured to find type-1 dark-matter particles only; GADGET %s masses include all hydro species, so this is not a like-for-like mass comparison" % definition.upper(),
            "mass_definition": definition,
        }

    if hydro:
        gadget_mass_types = group_field(run_dir, snap, ["GroupMassType"])
        gadget_count_types = group_field(run_dir, snap, ["GroupLenType"])
        if gadget_mass_types is None or gadget_count_types is None:
            return {
                "status": "SKIPPED",
                "reason": "hydro FOF comparison needs GroupMassType and GroupLenType to select the same type-1 dark-matter particles as Rockstar",
                "mass_definition": definition,
            }
        gadget_mass_types = np.asarray(gadget_mass_types)
        gadget_count_types = np.asarray(gadget_count_types)
        if (
            gadget_mass_types.ndim != 2 or gadget_mass_types.shape[1] < 2
            or gadget_count_types.ndim != 2 or gadget_count_types.shape[1] < 2
            or len(gadget_mass_types) != len(gadget_count_types)
        ):
            return {"status": "SKIPPED", "reason": "GADGET per-type FOF mass/count fields are malformed", "mass_definition": definition}
        gadget_raw_mass = np.asarray(gadget_mass_types[:, 1], dtype=float)
        gadget_counts = np.asarray(gadget_count_types[:, 1], dtype=np.int64)
        mass_component = "type-1 dark matter only"
    else:
        gadget_raw_mass = halo_masses(run_dir, snap)
        gadget_counts = halo_particle_counts(run_dir, snap)
        if gadget_raw_mass is None:
            return {"status": "SKIPPED", "reason": "GADGET catalogue has no %s mass field" % definition, "mass_definition": definition}
        gadget_raw_mass = np.asarray(gadget_raw_mass, dtype=float)
        mass_component = "all gravitating particle types"

    gadget_positions = group_field(run_dir, snap, ["GroupPos"])
    if gadget_positions is None:
        return {"status": "SKIPPED", "reason": "GADGET catalogue has no FOF host positions", "mass_definition": definition}
    gadget_positions = np.asarray(gadget_positions, dtype=float)
    if gadget_positions.ndim != 2 or gadget_positions.shape[1] != 3 or len(gadget_positions) != len(gadget_raw_mass):
        return {"status": "SKIPPED", "reason": "GADGET host positions and masses do not align", "mass_definition": definition}

    rm_all = np.asarray(table[rockstar_field], float)
    if rm_all.ndim != 1 or len(rm_all) != len(parent_ids):
        return {"status": "SKIPPED", "reason": "Rockstar masses and parent IDs do not align", "mass_definition": definition}
    host_mask = parent_ids == -1
    rockstar_counts = table.get("num_p", table.get("np"))
    minimum = int(cfg["analysis"]["hmf_min_particles"])
    if minimum < 0:
        raise ValueError("analysis.hmf_min_particles must be zero or greater")
    if minimum:
        if rockstar_counts is None:
            return {"status": "SKIPPED", "reason": "Rockstar catalogue has no Np/num_p field for the %d-particle HMF cut" % minimum, "mass_definition": definition}
        rockstar_counts = np.asarray(rockstar_counts)
        if rockstar_counts.ndim != 1 or len(rockstar_counts) != len(rm_all):
            return {"status": "SKIPPED", "reason": "Rockstar mass and num_p rows do not align", "mass_definition": definition}
        host_mask &= rockstar_counts >= minimum
    if gadget_counts is None:
        if minimum:
            return {"status": "SKIPPED", "reason": "GADGET catalogue has no particle counts for the %d-particle finder-comparison cut" % minimum, "mass_definition": definition}
        gadget_mask = np.ones(len(gadget_raw_mass), dtype=bool)
    else:
        gadget_counts = np.asarray(gadget_counts)
        if gadget_counts.ndim != 1 or len(gadget_counts) != len(gadget_raw_mass):
            return {"status": "SKIPPED", "reason": "GADGET host masses and particle counts do not align", "mass_definition": definition}
        gadget_mask = gadget_counts >= minimum if minimum else np.ones(len(gadget_raw_mass), dtype=bool)
    gadget_mask &= np.isfinite(gadget_raw_mass) & (gadget_raw_mass > 0)
    rockstar_mask = host_mask & np.isfinite(rm_all) & (rm_all > 0)

    fm = mass_to_msun_h(gadget_raw_mass[gadget_mask])
    rm = rm_all[rockstar_mask]
    if len(fm) < 5 or len(rm) < 5:
        return {
            "status": "SKIPPED",
            "reason": "fewer than five valid host halos in one or both catalogues after matching mass-definition and particle-count selections",
            "mass_definition": definition,
        }
    lo, hi = max(fm.min(), rm.min()), min(fm.max(), rm.max())
    if hi <= lo:
        return {"status": "SKIPPED", "reason": "GADGET and Rockstar mass ranges do not overlap", "mass_definition": definition}
    edges = np.logspace(np.log10(lo), np.log10(hi), int(cfg["analysis"].get("mass_bins", 35)) + 1)
    edges[0] = np.nextafter(lo, 0.0)
    edges[-1] = np.nextafter(hi, np.inf)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dlog = np.diff(np.log10(edges))
    volume = float(cfg["simulation"]["box_mpc_h"]) ** 3
    nf = np.histogram(fm, bins=edges)[0] / volume / dlog
    nr = np.histogram(rm, bins=edges)[0] / volume / dlog
    goodf, goodr = nf > 0, nr > 0
    gadget_label = "GADGET %s" % definition.upper()
    rockstar_label = "Rockstar %s" % definition.upper()
    if hydro:
        gadget_label += " (type-1 dark matter)"
        rockstar_label += " (type-1 dark matter)"
    plt.figure(figsize=(8,6))
    if np.any(goodf): plt.loglog(centers[goodf], nf[goodf], marker="o", label=gadget_label)
    if np.any(goodr): plt.loglog(centers[goodr], nr[goodr], marker="s", label=rockstar_label)
    plt.xlabel(r"$M\ [M_\odot/h]$"); plt.ylabel(r"$dn/d\log_{10}M$"); plt.legend()
    plt.title("Host FOF mass functions; %s" % mass_component)
    savefig(Path(run_dir)/"results"/("HMF_%s_vs_Rockstar_%03d.png"%(definition.upper(),snap)))
    ratio = np.divide(nr, nf, out=np.full_like(nr, np.nan, dtype=float), where=(nf>0)&(nr>0))
    good=np.isfinite(ratio)
    if np.any(good):
        plt.figure(figsize=(8,5)); plt.semilogx(centers[good], ratio[good], marker="o"); plt.axhline(1.0, linewidth=1)
        plt.xlabel(r"$M\ [M_\odot/h]$"); plt.ylabel("Rockstar / GADGET %s HMF" % definition.upper())
        savefig(Path(run_dir)/"results"/("HMF_%s_vs_Rockstar_ratio_%03d.png"%(definition.upper(),snap)))

    position_names = ["x", "y", "z"]
    if any(name not in table for name in position_names):
        return {
            "status": "COMPLETE", "comparison_type": "host_hmf_only; per_halo_matches=false",
            "mass_definition": definition, "mass_component": mass_component,
            "mass": centers, "gadget_dn_dlogM": nf, "rockstar_dn_dlogM": nr, "ratio": ratio,
            "host_count_gadget": int(len(fm)), "host_count_rockstar": int(len(rm)),
            "per_halo_match_status": "SKIPPED", "per_halo_match_reason": "Rockstar catalogue lacks x/y/z centres",
        }
    rockstar_positions_all = np.column_stack([np.asarray(table[name], dtype=float) for name in position_names])
    if rockstar_positions_all.shape != (len(rm_all), 3):
        return {
            "status": "COMPLETE", "comparison_type": "host_hmf_only; per_halo_matches=false",
            "mass_definition": definition, "mass_component": mass_component,
            "mass": centers, "gadget_dn_dlogM": nf, "rockstar_dn_dlogM": nr, "ratio": ratio,
            "host_count_gadget": int(len(fm)), "host_count_rockstar": int(len(rm)),
            "per_halo_match_status": "SKIPPED", "per_halo_match_reason": "Rockstar centres do not align with catalogue rows",
        }

    box = float(cfg["simulation"]["box_mpc_h"])
    particles_per_dimension = cfg.get("simulation", {}).get("particles_per_dimension")
    spacing_fraction = float(cfg.get("analysis", {}).get("finder_match_radius_particle_spacings", 0.5))
    if (
        isinstance(particles_per_dimension, bool)
        or not isinstance(particles_per_dimension, int)
        or particles_per_dimension < 2
        or not math.isfinite(spacing_fraction)
        or spacing_fraction <= 0
    ):
        return {
            "status": "COMPLETE", "comparison_type": "host_hmf_only; per_halo_matches=false",
            "mass_definition": definition, "mass_component": mass_component,
            "mass": centers, "gadget_dn_dlogM": nf, "rockstar_dn_dlogM": nr, "ratio": ratio,
            "host_count_gadget": int(len(fm)), "host_count_rockstar": int(len(rm)),
            "per_halo_match_status": "SKIPPED",
            "per_halo_match_reason": "a valid particles_per_dimension and positive finder_match_radius_particle_spacings are required",
        }
    match_radius = spacing_fraction * box / particles_per_dimension
    if match_radius >= box / 2.0:
        return {
            "status": "COMPLETE", "comparison_type": "host_hmf_only; per_halo_matches=false",
            "mass_definition": definition, "mass_component": mass_component,
            "mass": centers, "gadget_dn_dlogM": nf, "rockstar_dn_dlogM": nr, "ratio": ratio,
            "host_count_gadget": int(len(fm)), "host_count_rockstar": int(len(rm)),
            "per_halo_match_status": "SKIPPED", "per_halo_match_reason": "match radius must be less than half the periodic box",
        }

    gadget_indices = np.flatnonzero(gadget_mask)
    rockstar_indices = np.flatnonzero(rockstar_mask)
    gadget_pos_mpc_h = units_to_mpc_h(gadget_positions[gadget_mask], box)
    rockstar_pos_mpc_h = rockstar_positions_all[rockstar_mask]
    finite_positions = np.all(np.isfinite(gadget_pos_mpc_h), axis=1)
    finite_rockstar_positions = np.all(np.isfinite(rockstar_pos_mpc_h), axis=1)
    gadget_indices = gadget_indices[finite_positions]
    rockstar_indices = rockstar_indices[finite_rockstar_positions]
    gadget_pos_mpc_h = gadget_pos_mpc_h[finite_positions]
    rockstar_pos_mpc_h = rockstar_pos_mpc_h[finite_rockstar_positions]
    match = mutual_nearest_periodic_matches(gadget_pos_mpc_h, rockstar_pos_mpc_h, box, match_radius)
    matched_gadget_indices = gadget_indices[match["index_a"]]
    matched_rockstar_indices = rockstar_indices[match["index_b"]]
    matched_gadget_mass = mass_to_msun_h(gadget_raw_mass[matched_gadget_indices])
    matched_rockstar_mass = rm_all[matched_rockstar_indices]
    mass_ratio = np.divide(
        matched_rockstar_mass, matched_gadget_mass,
        out=np.full(len(matched_rockstar_mass), np.nan), where=matched_gadget_mass > 0,
    )
    finite_ratio = np.isfinite(mass_ratio) & (mass_ratio > 0)
    mass_definition_pairs = {}
    candidate_mass_fields = {
        "fof": ("GroupMass", "mfof"),
        "m200c": ("Group_M_Crit200", "m200c"),
        "m200b": ("Group_M_Mean200", "m200b"),
        "mvir": ("Group_M_TopHat200", "mvir"),
    }
    for mass_name, (gadget_field_name, rockstar_field_name) in candidate_mass_fields.items():
        if hydro and mass_name != "fof":
            continue  # Rockstar is type-1-only in this pipeline; SO FOF masses include all species.
        if hydro and mass_name == "fof":
            gadget_all = mass_to_msun_h(gadget_raw_mass)
        else:
            gadget_all = group_field(run_dir, snap, [gadget_field_name])
            if gadget_all is None:
                continue
            gadget_all = np.asarray(gadget_all, dtype=float)
            if gadget_all.ndim != 1 or len(gadget_all) != len(gadget_raw_mass):
                continue
            gadget_all = mass_to_msun_h(gadget_all)
        rockstar_all = table.get(rockstar_field_name)
        if rockstar_all is None:
            continue
        rockstar_all = np.asarray(rockstar_all, dtype=float)
        if rockstar_all.ndim != 1 or len(rockstar_all) != len(rm_all):
            continue
        paired_gadget_mass = gadget_all[matched_gadget_indices]
        paired_rockstar_mass = rockstar_all[matched_rockstar_indices]
        valid_pair = (
            np.isfinite(paired_gadget_mass) & (paired_gadget_mass > 0)
            & np.isfinite(paired_rockstar_mass) & (paired_rockstar_mass > 0)
        )
        pair_ratio = paired_rockstar_mass[valid_pair] / paired_gadget_mass[valid_pair]
        if not len(pair_ratio):
            continue
        mass_definition_pairs[mass_name] = {
            "gadget_field": gadget_field_name,
            "rockstar_field": rockstar_field_name,
            "gadget_mass_msun_h": paired_gadget_mass[valid_pair],
            "rockstar_mass_msun_h": paired_rockstar_mass[valid_pair],
            "rockstar_to_gadget_ratio": pair_ratio,
            "gadget_catalogue_index": matched_gadget_indices[valid_pair],
            "rockstar_catalogue_index": matched_rockstar_indices[valid_pair],
            "rockstar_halo_id": (
                np.asarray(table["id"])[matched_rockstar_indices[valid_pair]]
                if "id" in table else np.full(np.count_nonzero(valid_pair), -1, dtype=np.int64)
            ),
            "valid_halo_count": int(len(pair_ratio)),
            "median_ratio": float(np.median(pair_ratio)),
            "ratio_percentile_16_84": np.percentile(pair_ratio, [16, 84]).astype(float).tolist(),
            "mean_log10_ratio": float(np.mean(np.log10(pair_ratio))),
        }
    subfind_central_pair = None
    if not hydro:
        # GroupFirstSub is GADGET's exact FOF-to-central-SUBFIND row link. Use
        # it instead of a positional guess for the two GADGET finder layers.
        first_sub = group_field(run_dir, snap, ["GroupFirstSub"])
        subhalo_mass = subhalo_field(run_dir, snap, ["SubhaloMass"])
        rockstar_fof_all = table.get("mfof")
        if first_sub is not None and subhalo_mass is not None and rockstar_fof_all is not None:
            first_sub = np.asarray(first_sub)
            subhalo_mass = np.asarray(subhalo_mass, dtype=float)
            rockstar_fof_all = np.asarray(rockstar_fof_all, dtype=float)
            if (first_sub.ndim == 1 and first_sub.dtype.kind in "iu"
                    and len(first_sub) == len(gadget_raw_mass)
                    and subhalo_mass.ndim == 1
                    and rockstar_fof_all.shape == rm_all.shape):
                matched_first_sub = first_sub[matched_gadget_indices].astype(np.int64, copy=False)
                valid_sub = (matched_first_sub >= 0) & (matched_first_sub < len(subhalo_mass))
                valid_sub &= np.isfinite(mass_to_msun_h(gadget_raw_mass[matched_gadget_indices]))
                valid_sub &= mass_to_msun_h(gadget_raw_mass[matched_gadget_indices]) > 0
                valid_sub &= np.isfinite(rockstar_fof_all[matched_rockstar_indices]) & (rockstar_fof_all[matched_rockstar_indices] > 0)
                if np.any(valid_sub):
                    group_mass = mass_to_msun_h(gadget_raw_mass[matched_gadget_indices[valid_sub]])
                    central_mass = mass_to_msun_h(subhalo_mass[matched_first_sub[valid_sub]])
                    rockstar_mass = rockstar_fof_all[matched_rockstar_indices[valid_sub]]
                    valid_mass = (
                        np.isfinite(group_mass) & (group_mass > 0)
                        & np.isfinite(central_mass) & (central_mass > 0)
                        & np.isfinite(rockstar_mass) & (rockstar_mass > 0)
                    )
                    if np.any(valid_mass):
                        group_mass = group_mass[valid_mass]
                        central_mass = central_mass[valid_mass]
                        rockstar_mass = rockstar_mass[valid_mass]
                        group_rows = matched_gadget_indices[valid_sub][valid_mass]
                        rockstar_rows = matched_rockstar_indices[valid_sub][valid_mass]
                        sub_rows = matched_first_sub[valid_sub][valid_mass]
                        subfind_central_pair = {
                            "gadget_group_index": group_rows,
                            "subfind_central_index": sub_rows,
                            "rockstar_catalogue_index": rockstar_rows,
                            "rockstar_halo_id": (
                                np.asarray(table["id"])[rockstar_rows]
                                if "id" in table else np.full(len(group_rows), -1, dtype=np.int64)
                            ),
                            "fof_mass_msun_h": group_mass,
                            "subfind_central_bound_mass_msun_h": central_mass,
                            "rockstar_mass_msun_h": rockstar_mass,
                            "subfind_to_fof_ratio": central_mass / group_mass,
                            "rockstar_to_fof_ratio": rockstar_mass / group_mass,
                            "subfind_to_rockstar_ratio": central_mass / rockstar_mass,
                        }
    match_result = {
        "per_halo_match_status": "COMPLETE",
        "per_halo_match_method": "mutual nearest periodic centre; unique one-to-one matches",
        "per_halo_match_max_separation_mpc_h": float(match_radius),
        "per_halo_match_radius_particle_spacings": spacing_fraction,
        "per_halo_match_gadget_count": int(len(gadget_indices)),
        "per_halo_match_rockstar_count": int(len(rockstar_indices)),
        "per_halo_match_count": int(len(match["index_a"])),
        "per_halo_match_gadget_fraction": float(len(match["index_a"]) / len(gadget_indices)) if len(gadget_indices) else 0.0,
        "per_halo_match_rockstar_fraction": float(len(match["index_b"]) / len(rockstar_indices)) if len(rockstar_indices) else 0.0,
        "per_halo_gadget_index": matched_gadget_indices,
        "per_halo_rockstar_index": matched_rockstar_indices,
        "per_halo_rockstar_id": np.asarray(table.get("id", np.full(len(rm_all), -1)))[matched_rockstar_indices],
        "per_halo_center_separation_mpc_h": match["distance"],
        "per_halo_gadget_mass_msun_h": matched_gadget_mass,
        "per_halo_rockstar_mass_msun_h": matched_rockstar_mass,
        "per_halo_rockstar_to_gadget_mass_ratio": mass_ratio,
        "per_halo_log10_rockstar_to_gadget_mass_ratio": np.log10(mass_ratio[finite_ratio]),
        "per_halo_mass_ratio_summary": {
            "median": float(np.median(mass_ratio[finite_ratio])) if np.any(finite_ratio) else None,
            "p16": float(np.percentile(mass_ratio[finite_ratio], 16)) if np.any(finite_ratio) else None,
            "p84": float(np.percentile(mass_ratio[finite_ratio], 84)) if np.any(finite_ratio) else None,
        },
        "per_halo_center_separation_summary_mpc_h": {
            "median": float(np.median(match["distance"])) if len(match["distance"]) else None,
            "p95": float(np.percentile(match["distance"], 95)) if len(match["distance"]) else None,
        },
        "per_halo_mass_definition_pairs": mass_definition_pairs,
        "per_halo_subfind_central_pair": subfind_central_pair,
    }
    return {
        "status": "COMPLETE", "comparison_type": "host_hmf_and_periodic_mutual_nearest_halo_matches",
        "mass_definition": definition,
        "mass_component": mass_component,
        "mass": centers, "gadget_dn_dlogM": nf, "rockstar_dn_dlogM": nr,
        "ratio": ratio,
        "host_count_gadget": int(len(fm)), "host_count_rockstar": int(len(rm)),
        **match_result,
    }

def plot_cumulative_hmf(run_dir, cfg, snap, masses, reliability):
    """Plot measured N(>=M)/V with the catalogue mass convention and explicit counting limits."""
    resolution = simulation_resolution_metadata(cfg)
    edges = hmf_mass_edges(cfg)
    all_masses = np.sort(mass_to_msun_h(masses))
    all_masses = all_masses[np.isfinite(all_masses) & (all_masses > 0)]
    thresholds_in_range = all_masses[(all_masses >= edges[0]) & (all_masses <= edges[-1])]
    if len(thresholds_in_range) < 5:
        return None
    thresholds = np.unique(thresholds_in_range)
    counts_gt = len(all_masses) - np.searchsorted(all_masses, thresholds, side="left")
    volume = resolution["volume_mpc_h3"]
    n_gt = counts_gt / volume
    lower, upper = poisson_rate_interval(counts_gt, volume, np.ones_like(n_gt))
    z = redshift_from_header(read_snapshot_header(run_dir, snap)[0])
    definition = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    minimum_particles = int(cfg["analysis"]["hmf_min_particles"])
    particle_floor = minimum_particles * resolution["particle_mass_msun_h"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(thresholds, lower, upper, color="#819794", alpha=0.18, linewidth=0,
                    label="68.27% Garwood Poisson interval")
    ax.plot(thresholds, n_gt, color="#244969", linewidth=2.1,
            label="Cumulative host abundance")
    if particle_floor > edges[0]:
        ax.axvspan(edges[0], min(particle_floor, edges[-1]), color="#718096",
                   alpha=0.16, label="Below configured particle-count floor")
    shade_below_wdm_limit(all_masses, reliability)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Threshold halo mass $M$ [$M_\odot/h$]")
    ax.set_ylabel(r"$n(\geq M)$ [$(h/\mathrm{Mpc})^3$]")
    ax.set_title(
        f"Cumulative {definition.upper()} abundance  |  z={z:.3f}  |  "
        f"L={resolution['box_mpc_h']:g} Mpc/h, N={resolution['particles_per_dimension']}^3"
    )
    ax.text(
        0.02, 0.02,
        rf"$m_p$={resolution['particle_mass_msun_h']:.3g} $M_\odot/h$   "
        rf"$V$={volume:.3g} $(\mathrm{{Mpc}}/h)^3$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#d7dee8", "alpha": 0.92, "pad": 3},
    )
    ax.legend(loc="best")
    savefig(Path(run_dir) / "results" / ("HMF_cumulative_%03d.png" % snap), metadata={
        "product": "cumulative host halo abundance", "snapshot": int(snap),
        "redshift": float(z), "mass_definition": definition,
        "mass_thresholds_msun_h": thresholds.tolist(), "host_counts_above_threshold": counts_gt.tolist(),
        "poisson_uncertainty": "68.27% central Garwood intervals",
        "box_size_mpc_h": resolution["box_mpc_h"], "volume_mpc_h3": volume,
        "particles_per_dimension": resolution["particles_per_dimension"],
        "particle_count": resolution["particle_count"],
        "particle_mass_msun_h": resolution["particle_mass_msun_h"],
        "mean_particle_spacing_mpc_h": resolution["mean_particle_spacing_mpc_h"],
        "minimum_halo_particles": minimum_particles,
        "particle_count_mass_floor_msun_h": float(particle_floor),
        "mass_range_msun_h": [float(edges[0]), float(edges[-1])],
        "abundance_unit": "(h/Mpc)^3",
    })
    return thresholds, n_gt


def rare_halo_table(cfg, masses):
    m = mass_to_msun_h(masses)
    m = m[np.isfinite(m) & (m > 0)]
    out = {str(t): int(np.sum(m >= float(t))) for t in cfg["analysis"]["rare_halo_thresholds_msun_h"]}
    out["maximum_mass_msun_h"] = float(np.max(m)) if len(m) else None
    out["halo_count"] = int(len(m))
    return out


def plot_spatial_distribution(run_dir, cfg, snap):
    """Plot measured halo positions within the comoving periodic box."""
    pos = group_field(run_dir, snap, ["GroupPos"])
    mass = halo_masses(run_dir, snap)
    if pos is None or mass is None:
        return None
    box = float(cfg["simulation"]["box_mpc_h"])
    pos = units_to_mpc_h(pos, box)
    mass = mass_to_msun_h(mass)
    n = min(len(pos), len(mass))
    good = np.isfinite(mass[:n]) & (mass[:n] > 0)
    if np.sum(good) < 3:
        return None
    size = 4 + 18 * (np.log10(mass[:n][good]) - np.log10(mass[:n][good]).min() + 0.1)
    plt.figure(figsize=(8, 8)); plt.scatter(pos[:n][good, 0], pos[:n][good, 1], s=size, alpha=0.45)
    plt.xlim(0, box); plt.ylim(0, box); plt.xlabel("x [Mpc/h]"); plt.ylabel("y [Mpc/h]")
    savefig(Path(run_dir) / "results" / ("halo_spatial_%03d.png" % snap))
    return pos[:n][good], mass[:n][good]


def catalogue_kinematic_metadata(run_dir, snap):
    """Check native units and catalogue/snapshot epochs before a field-specific conversion."""
    from halos import _snapshot_native_units
    header,files=read_snapshot_header(run_dir,snap)
    a=_snapshot_header_value(files,'Time')
    h=_snapshot_header_value(files,'HubbleParam')
    if h is None:h=_snapshot_parameter(files,'HubbleParam')
    length,_,velocity=_snapshot_native_units(run_dir,snap)
    if a is None or h is None or not np.isfinite([a,h]).all() or min(a,h)<=0:
        raise ValueError('Catalogue kinematics needs actual positive snapshot a and h')
    for path in group_files(run_dir,snap):
        with h5py.File(path,'r') as source:
            if 'Header' not in source or 'Time' not in source['Header'].attrs:
                raise ValueError('Catalogue kinematics needs an explicit catalogue epoch')
            if not np.isclose(float(source['Header'].attrs['Time']),a,rtol=1.e-10,atol=0.):
                raise ValueError('Catalogue and snapshot epochs disagree')
            parameters=source.get('Parameters')
            if parameters is None:raise ValueError('Catalogue kinematics needs explicit native unit parameters')
            for key,expected in [('UnitLength_in_cm',length),('UnitVelocity_in_cm_per_s',velocity),('HubbleParam',h)]:
                if key not in parameters.attrs or not np.isclose(float(parameters.attrs[key]),expected,rtol=1.e-10,atol=0.):
                    raise ValueError('Catalogue and snapshot kinematic units disagree: '+key)
            if int(parameters.attrs.get('ComovingIntegrationOn',-1))!=1:
                raise ValueError('Catalogue cosmological kinematics requires ComovingIntegrationOn=1')
    return {'scale_factor':a,'hubble_param':h,'unit_length_cm':length,'unit_velocity_cm_s':velocity}


def plot_halo_velocity_distribution(run_dir, snap):
    """Plot catalogue velocities in the declared physical peculiar-velocity convention."""
    vel = group_field(run_dir, snap, ["GroupVel"])
    field='GroupVel'
    if vel is None:
        vel = subhalo_field(run_dir, snap, ["SubhaloVel"])
        field='SubhaloVel'
    if vel is None:
        return None
    from core.data import convert_native_catalogue_kinematics
    metadata=catalogue_kinematic_metadata(run_dir,snap)
    velocity=convert_native_catalogue_kinematics(vel,field,**metadata)
    speed = np.linalg.norm(velocity, axis=1)
    speed = speed[np.isfinite(speed)]
    if not len(speed):
        return None
    density, edges = histogram_pdf(speed, 80)
    plt.figure(figsize=(8, 6)); plt.stairs(density, edges)
    plt.xlabel("halo peculiar speed [km/s]"); plt.ylabel("probability density [(km/s)^-1]")
    savefig(Path(run_dir) / "results" / ("halo_velocity_distribution_%03d.png" % snap))
    return {"bin_edges": edges, "probability_density": density, "sample_count": len(speed),
            'source_field':field,'native_unit_metadata':metadata,'velocity_convention':'physical peculiar km/s'}


def plot_subhalo_mass_function(run_dir, cfg, snap, reliability=None):
    """Count bound SUBFIND masses separately from the host halo mass function."""
    m = subhalo_field(run_dir, snap, ["SubhaloMass"])
    if m is None:
        return None
    m = mass_to_msun_h(m)
    m = m[np.isfinite(m) & (m > 0)]
    if len(m) < 5:
        return None
    edges = np.logspace(np.log10(m.min()), np.log10(m.max()), int(cfg["analysis"]["mass_bins"]) + 1)
    edges[0] = np.nextafter(m.min(), 0.0)
    edges[-1] = np.nextafter(m.max(), np.inf)
    count, _ = np.histogram(m, edges)
    cen = np.sqrt(edges[1:] * edges[:-1]); dlog = np.diff(np.log10(edges)); vol = float(cfg["simulation"]["box_mpc_h"])**3
    y = count / vol / dlog
    keep = count > 0
    plt.figure(figsize=(8, 6)); plt.loglog(cen[keep], y[keep], marker="o")
    shade_below_wdm_limit(cen[keep], reliability)
    plt.xlabel(r"$M_{sub}\ [M_\odot/h]$"); plt.ylabel(r"$dn_{sub}/d\log_{10}M$")
    savefig(Path(run_dir) / "results" / ("subhalo_HMF_%03d.png" % snap))
    return cen, y


def plot_spin_distribution(run_dir, snap):
    """Plot the catalogue specific angular-momentum norm; do not mislabel it as dimensionless spin."""
    spin = subhalo_field(run_dir, snap, ["SubhaloSpin"])
    mass = subhalo_field(run_dir, snap, ["SubhaloMass"])
    if spin is None or mass is None:
        return None
    # GroupSpin is not a registered field in the pinned FOF writer. Do not
    # assign SUBFIND's convention to an arbitrary imported field of that name.
    from core.data import convert_native_catalogue_kinematics
    physical=convert_native_catalogue_kinematics(spin,'SubhaloSpin',**catalogue_kinematic_metadata(run_dir,snap))
    j = np.linalg.norm(physical, axis=1)
    good = np.isfinite(j) & (j > 0)
    if not np.any(good):
        return None
    plt.figure(figsize=(8, 6)); plt.hist(np.log10(j[good]), bins=80, density=True)
    plt.xlabel(r"$\log_{10}(|j| / (\mathrm{proper\ kpc}\,\mathrm{km\,s}^{-1}))$"); plt.ylabel("probability per dex")
    savefig(Path(run_dir) / "results" / ("specific_angular_momentum_%03d.png" % snap))
    return j[good]


# ----------------------------- particle fields -----------------------------

def ngp_density(pos, box, grid):
    """Nearest-grid-point density mesh used only for transparent diagnostics."""
    hist, _ = np.histogramdd(pos, bins=int(grid), range=[[0, box], [0, box], [0, box]])
    return hist.astype(np.float32, copy=False)


def fft_power_spectrum(pos, box, grid=256):
    """Measure isotropically binned periodic density power, tracking FFT normalization and available modes."""
    grid = int(grid)
    hist = ngp_density(pos, box, grid)
    delta = (hist / hist.mean() - 1.0).astype(np.float32, copy=False)
    fk = scipy_fft.rfftn(delta, workers=1)
    kx = np.fft.fftfreq(grid, d=box / grid) * 2 * math.pi
    ky = kx
    kz = np.fft.rfftfreq(grid, d=box / grid) * 2 * math.pi
    k_squared_xy = kx[:, None]**2 + ky[None, :]**2
    bins = np.logspace(np.log10(2 * math.pi / box), np.log10(math.pi * grid / box), 45)
    # rfftn stores only k_z >= 0. Restore the missing -k_z modes when shell
    # averaging; k_z=0 and the even-grid Nyquist plane are self-conjugate.
    sumw = np.zeros(len(bins), dtype=np.float64)
    sumk = np.zeros(len(bins), dtype=np.float64)
    sump = np.zeros(len(bins), dtype=np.float64)
    normalization = box**3 / grid**6
    for iz, kz_value in enumerate(kz):
        kmag = np.sqrt(k_squared_xy + kz_value**2)
        idx = np.digitize(kmag.ravel(), bins)
        valid = (idx > 0) & (idx < len(bins))
        if not np.any(valid):
            continue
        weight = 1.0 if iz == 0 or (grid % 2 == 0 and iz == len(kz) - 1) else 2.0
        shell = idx[valid]
        multiplicity = np.bincount(shell, minlength=len(bins)) * weight
        sumw += multiplicity
        sumk += np.bincount(shell, weights=kmag.ravel()[valid] * weight, minlength=len(bins))
        mode_power = np.asarray(np.abs(fk[:, :, iz])**2, dtype=np.float64) * normalization
        sump += np.bincount(shell, weights=mode_power.ravel()[valid] * weight, minlength=len(bins))
    kval, pval = [], []
    for i in range(1, len(bins)):
        if sumw[i] > 0:
            kval.append(sumk[i] / sumw[i]); pval.append(sump[i] / sumw[i])
    return np.asarray(kval), np.asarray(pval)


def _deposit_cic_positions(density, positions, box, mesh_size, weights=None):
    """Deposit nonnegative particle weights conservatively onto a periodic 3-D cloud-in-cell mesh."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("CIC positions must be a finite (particles, 3) array")
    if weights is None:
        weights = np.ones(len(positions), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(positions):
            raise ValueError("CIC particle weights must align with positions")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("CIC particle weights must be finite and nonnegative")
    if not len(positions):
        return
    grid = np.mod(positions, box) * (float(mesh_size) / box)
    base = np.floor(grid).astype(np.int64)
    fraction = grid - base
    base %= mesh_size
    for dx in (0, 1):
        wx = fraction[:, 0] if dx else (1.0 - fraction[:, 0])
        ix = (base[:, 0] + dx) % mesh_size
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else (1.0 - fraction[:, 1])
            iy = (base[:, 1] + dy) % mesh_size
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else (1.0 - fraction[:, 2])
                iz = (base[:, 2] + dz) % mesh_size
                flat = (ix * mesh_size + iy) * mesh_size + iz
                deposited = np.bincount(
                    flat, weights=weights * wx * wy * wz, minlength=mesh_size**3,
                )
                density += deposited.reshape(density.shape)


def _snapshot_matter_cic_mesh(files, box_mpc_h, mesh_size, chunk_size=1_000_000):
    """Deposit every massive GADGET particle with its mass, streaming HDF5 chunks."""
    density = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float64)
    mass_table_reference = None
    mass_source_by_type = {}
    particle_count = 0
    total_mass = 0.0
    sum_squared_mass = 0.0
    for path in files:
        with h5py.File(path, "r") as h5:
            header = h5["Header"].attrs
            mass_table = np.asarray(header.get("MassTable", np.zeros(6)), dtype=float)
            if mass_table.shape != (6,) or not np.all(np.isfinite(mass_table)) or np.any(mass_table < 0):
                raise ValueError("Snapshot Header/MassTable must contain six finite nonnegative masses")
            if mass_table_reference is None:
                mass_table_reference = mass_table
            elif not np.allclose(mass_table, mass_table_reference, rtol=1.0e-10, atol=0.0):
                raise ValueError("Snapshot shards have inconsistent Header/MassTable values")
            for ptype in range(6):
                group_name = "PartType%d" % ptype
                if group_name not in h5:
                    continue
                particles = h5[group_name]
                if "Coordinates" not in particles:
                    raise ValueError("%s has no Coordinates dataset" % group_name)
                coordinates = particles["Coordinates"]
                if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                    raise ValueError("%s/Coordinates must have shape (particles, 3)" % group_name)
                count = len(coordinates)
                if count == 0:
                    continue
                variable_mass = "Masses" in particles
                source = "variable" if variable_mass else "constant"
                if ptype in mass_source_by_type and mass_source_by_type[ptype] != source:
                    raise ValueError("PartType%d mixes constant and variable masses across snapshot shards" % ptype)
                mass_source_by_type[ptype] = source
                if variable_mass:
                    masses = particles["Masses"]
                    if masses.ndim != 1 or len(masses) != count:
                        raise ValueError("%s/Masses must align with Coordinates" % group_name)
                    if mass_table[ptype] != 0:
                        raise ValueError("%s has both variable masses and a nonzero Header/MassTable entry" % group_name)
                else:
                    fixed_mass = float(mass_table[ptype])
                    if not np.isfinite(fixed_mass) or fixed_mass <= 0:
                        raise ValueError("%s has no valid particle-mass source" % group_name)
                    masses = None
                for start in range(0, count, int(chunk_size)):
                    stop = min(start + int(chunk_size), count)
                    positions = np.asarray(coordinates[start:stop], dtype=np.float64) / 1000.0
                    weights = (
                        np.asarray(masses[start:stop], dtype=np.float64)
                        if masses is not None else np.full(stop - start, fixed_mass, dtype=np.float64)
                    )
                    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                        raise ValueError("%s/Masses contains non-finite or negative values" % group_name)
                    _deposit_cic_positions(density, positions, box_mpc_h, mesh_size, weights)
                    particle_count += stop - start
                    total_mass += float(np.sum(weights, dtype=np.float64))
                    sum_squared_mass += float(np.dot(weights, weights))
    if particle_count == 0 or not np.isfinite(total_mass) or total_mass <= 0 or sum_squared_mass <= 0:
        raise ValueError("Snapshot contains no positive-mass particles for the matter field")
    if not np.isclose(float(np.sum(density)), total_mass, rtol=1.0e-10, atol=1.0e-10 * total_mass):
        raise RuntimeError("CIC matter deposition did not conserve the total particle mass")
    shot_noise = float(box_mpc_h**3 * sum_squared_mass / total_mass**2)
    return density, particle_count, shot_noise


def _exact_analysis_integer(value, label, minimum=1, maximum=None):
    """Reject fractional/bool analysis counts instead of silently truncating them."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be an integer" % label)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be an integer" % label) from exc
    if not np.isfinite(numeric) or numeric != math.floor(numeric) or numeric < minimum:
        raise ValueError("%s must be an integer >= %d" % (label, minimum))
    if maximum is not None and numeric > maximum:
        raise ValueError("%s must be an integer <= %d" % (label, maximum))
    return int(numeric)


def _cic_cross_power_bias(halo_density, matter_density, box_mpc_h, halo_count,
                          matter_shot_noise, min_modes=20, nyquist_fraction=0.25):
    """Estimate b_hm=P_hm/P_mm using periodic CIC meshes and a corrected matter spectrum."""
    halo_density = np.asarray(halo_density, dtype=np.float64)
    matter_density = np.asarray(matter_density, dtype=np.float64)
    if (halo_density.ndim != 3 or halo_density.shape != matter_density.shape
            or len(set(halo_density.shape)) != 1):
        raise ValueError("Halo and matter CIC meshes must be matching cubic arrays")
    mesh_size = int(halo_density.shape[0])
    if mesh_size < 8:
        raise ValueError("Halo-bias CIC mesh size must be at least 8")
    if (not np.isfinite(box_mpc_h) or box_mpc_h <= 0
            or not np.isfinite(matter_shot_noise) or matter_shot_noise < 0):
        raise ValueError("Halo-bias box and matter shot noise must be finite and valid")
    halo_count = _exact_analysis_integer(halo_count, "Halo-bias halo count")
    min_modes = _exact_analysis_integer(min_modes, "Halo-bias minimum mode count")
    if not np.isfinite(nyquist_fraction) or not 0 < nyquist_fraction <= 0.5:
        raise ValueError("Halo-bias Nyquist fraction must be in (0, 0.5]")
    if (not np.all(np.isfinite(halo_density)) or not np.all(np.isfinite(matter_density))
            or float(np.mean(halo_density)) <= 0 or float(np.mean(matter_density)) <= 0):
        raise ValueError("Halo and matter CIC meshes must have finite positive means")

    halo_contrast = halo_density / float(np.mean(halo_density)) - 1.0
    matter_contrast = matter_density / float(np.mean(matter_density)) - 1.0
    halo_contrast -= float(np.mean(halo_contrast))
    matter_contrast -= float(np.mean(matter_contrast))
    halo_fft = scipy_fft.rfftn(halo_contrast, workers=1)
    matter_fft = scipy_fft.rfftn(matter_contrast, workers=1)
    fundamental = 2.0 * math.pi / float(box_mpc_h)
    max_shell = int(math.floor(nyquist_fraction * mesh_size / 2.0))
    if max_shell < 2:
        raise ValueError("Halo-bias Nyquist cut leaves fewer than two Fourier shells")
    edges = fundamental * (np.arange(max_shell + 1, dtype=float) + 0.5)
    sum_modes = np.zeros(max_shell, dtype=np.int64)
    sum_k = np.zeros(max_shell, dtype=np.float64)
    sum_cross = np.zeros(max_shell, dtype=np.float64)
    sum_halo_auto = np.zeros(max_shell, dtype=np.float64)
    sum_matter_auto = np.zeros(max_shell, dtype=np.float64)
    frequency = np.fft.fftfreq(mesh_size) * mesh_size
    frequency_z = np.fft.rfftfreq(mesh_size) * mesh_size
    k_squared_xy = frequency[:, None]**2 + frequency[None, :]**2
    normalization = float(box_mpc_h)**3 / mesh_size**6
    halo_shot_noise = float(box_mpc_h)**3 / halo_count
    for iz, mode_z in enumerate(frequency_z):
        mode = np.sqrt(k_squared_xy + mode_z**2)
        shell = np.digitize(mode, np.arange(max_shell + 1, dtype=float) + 0.5) - 1
        valid = (shell >= 0) & (shell < max_shell) & (mode > 0)
        if not np.any(valid):
            continue
        window_x = np.sinc(frequency / mesh_size)**2
        window_y = np.sinc(frequency / mesh_size)**2
        window_z = np.sinc(mode_z / mesh_size)**2
        cic_window = window_x[:, None] * window_y[None, :] * window_z
        window_power = cic_window**2
        valid &= window_power > 1.0e-12
        if not np.any(valid):
            continue
        multiplicity = 1 if iz == 0 or (mesh_size % 2 == 0 and iz == len(frequency_z) - 1) else 2
        shell_ids = shell[valid]
        kval = mode[valid] * fundamental
        cross = (np.real(halo_fft[:, :, iz] * np.conj(matter_fft[:, :, iz]))
                 * normalization / window_power)[valid]
        halo_auto = (np.abs(halo_fft[:, :, iz])**2 * normalization / window_power)[valid]
        matter_auto = (np.abs(matter_fft[:, :, iz])**2 * normalization / window_power)[valid]
        sum_modes += np.bincount(shell_ids, minlength=max_shell).astype(np.int64) * multiplicity
        sum_k += np.bincount(shell_ids, weights=kval * multiplicity, minlength=max_shell)
        sum_cross += np.bincount(shell_ids, weights=cross * multiplicity, minlength=max_shell)
        sum_halo_auto += np.bincount(shell_ids, weights=halo_auto * multiplicity, minlength=max_shell)
        sum_matter_auto += np.bincount(shell_ids, weights=matter_auto * multiplicity, minlength=max_shell)

    centers = np.divide(sum_k, sum_modes, out=np.full(max_shell, np.nan), where=sum_modes > 0)
    cross_power = np.divide(sum_cross, sum_modes, out=np.full(max_shell, np.nan), where=sum_modes > 0)
    halo_auto_observed = np.divide(sum_halo_auto, sum_modes, out=np.full(max_shell, np.nan), where=sum_modes > 0)
    matter_auto_observed = np.divide(sum_matter_auto, sum_modes, out=np.full(max_shell, np.nan), where=sum_modes > 0)
    halo_auto = halo_auto_observed - halo_shot_noise
    matter_auto = matter_auto_observed - matter_shot_noise
    valid_modes = sum_modes >= min_modes
    bias = np.divide(cross_power, matter_auto, out=np.full(max_shell, np.nan),
                     where=valid_modes & np.isfinite(matter_auto) & (matter_auto > 0))
    bias_variance = np.full(max_shell, np.nan, dtype=np.float64)
    valid_bias = np.isfinite(bias) & (sum_modes > 0)
    pmm_signal = matter_auto[valid_bias]
    phm = cross_power[valid_bias]
    pmm_observed = matter_auto_observed[valid_bias]
    phh_observed = halo_auto_observed[valid_bias]
    modes = sum_modes[valid_bias].astype(np.float64)
    variance_cross = (phm**2 + phh_observed * pmm_observed) / modes
    variance_matter = 2.0 * pmm_observed**2 / modes
    covariance_cross_matter = 2.0 * phm * pmm_observed / modes
    variance = (
        variance_cross / pmm_signal**2
        + phm**2 * variance_matter / pmm_signal**4
        - 2.0 * phm * covariance_cross_matter / pmm_signal**3
    )
    bias_variance[valid_bias] = np.maximum(variance, 0.0)
    auto_bias = np.full(max_shell, np.nan, dtype=np.float64)
    auto_bias_variance = np.full(max_shell, np.nan, dtype=np.float64)
    cross_correlation = np.full(max_shell, np.nan, dtype=np.float64)
    valid_auto = (
        valid_modes & np.isfinite(halo_auto) & (halo_auto > 0)
        & np.isfinite(matter_auto) & (matter_auto > 0)
    )
    auto_bias[valid_auto] = np.sqrt(halo_auto[valid_auto] / matter_auto[valid_auto])
    cross_correlation[valid_auto] = (
        cross_power[valid_auto] / np.sqrt(halo_auto[valid_auto] * matter_auto[valid_auto])
    )
    auto_hh = halo_auto[valid_auto]
    auto_mm = matter_auto[valid_auto]
    auto_hh_observed = halo_auto_observed[valid_auto]
    auto_mm_observed = matter_auto_observed[valid_auto]
    auto_cross = cross_power[valid_auto]
    auto_modes = sum_modes[valid_auto].astype(np.float64)
    variance_hh = 2.0 * auto_hh_observed**2 / auto_modes
    variance_mm = 2.0 * auto_mm_observed**2 / auto_modes
    covariance_hh_mm = 2.0 * auto_cross**2 / auto_modes
    auto_variance = (auto_bias[valid_auto]**2 / 4.0) * (
        variance_hh / auto_hh**2 + variance_mm / auto_mm**2
        - 2.0 * covariance_hh_mm / (auto_hh * auto_mm)
    )
    auto_bias_variance[valid_auto] = np.maximum(auto_variance, 0.0)
    return {
        "k_h_Mpc": centers, "bias": bias, "bias_gaussian_68_sigma": np.sqrt(bias_variance),
        "bias_auto_hh_mm": auto_bias,
        "bias_auto_hh_mm_gaussian_68_sigma": np.sqrt(auto_bias_variance),
        "cross_correlation_coefficient_hm": cross_correlation,
        "cross_power_Mpc_h3": cross_power,
        "halo_power_Mpc_h3": halo_auto, "matter_power_Mpc_h3": matter_auto,
        "mode_count": sum_modes, "bin_edges_h_Mpc": edges,
        "grid_size": mesh_size,
        "matter_shot_noise_Mpc_h3": float(matter_shot_noise),
        "halo_shot_noise_Mpc_h3": halo_shot_noise,
        "nyquist_fraction": float(nyquist_fraction), "min_modes_per_bin": min_modes,
        "method": "periodic CIC halo-matter cross spectrum; b_hm=P_hm/P_mm; CIC-window deconvolved; all represented massive particle types mass-weighted in P_mm; variable-mass effective particle shot noise subtracted from P_mm and Poisson halo-center shot noise subtracted from P_hh; b_hh=sqrt(P_hh/P_mm) retained as an auto-spectrum cross-check",
    }


def read_gadget_power_spectrum_segments(path):
    """Read GADGET-4's concatenated folded power-spectrum sections."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    segments = []
    offset = 0
    while offset + 5 <= len(lines):
        try:
            time = float(lines[offset])
            count = int(lines[offset + 1])
            box_internal = float(lines[offset + 2])
            mesh = int(lines[offset + 3])
            growth = float(lines[offset + 4])
        except ValueError:
            break  # GADGET appends three summary values after the final section.
        if count < 1 or offset + 5 + count > len(lines):
            break
        row_lines = lines[offset + 5:offset + 5 + count]
        parsed_rows = []
        mode_counts = []
        mode_count_tokens_exact = []
        for line_number, line in enumerate(row_lines, offset + 6):
            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError("Malformed GADGET power-spectrum row at line %d in %s" % (line_number, path))
            try:
                row = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError("Non-numeric GADGET power-spectrum row at line %d in %s" % (line_number, path)) from exc
            mode_token = tokens[3]
            exact_token = bool(re.fullmatch(r"[+-]?\d+", mode_token))
            try:
                if exact_token:
                    mode_count = int(mode_token)
                else:
                    mode_value = float(mode_token)
                    if not math.isfinite(mode_value) or mode_value != math.floor(mode_value):
                        raise ValueError
                    mode_count = int(round(mode_value))
            except ValueError as exc:
                raise ValueError("GADGET Fourier-mode count must be a finite integer at line %d in %s" % (line_number, path)) from exc
            if mode_count < 0 or mode_count > np.iinfo(np.int64).max:
                raise ValueError("GADGET Fourier-mode count is outside the supported int64 range at line %d in %s" % (line_number, path))
            row[3] = float(mode_count)
            parsed_rows.append(row)
            mode_counts.append(mode_count)
            mode_count_tokens_exact.append(exact_token)
        rows = np.asarray(parsed_rows, dtype=float)
        if rows.shape != (count, 5) or not np.all(np.isfinite(rows)):
            raise ValueError("Malformed GADGET power-spectrum rows in " + str(path))
        segments.append({
            "time": time, "box_size_internal": box_internal, "mesh": mesh,
            "growth_factor": growth, "rows": rows,
            "mode_counts": np.asarray(mode_counts, dtype=np.int64),
            "mode_count_tokens_exact": np.asarray(mode_count_tokens_exact, dtype=bool),
        })
        offset += 5 + count
    if not segments:
        raise ValueError("No valid GADGET power-spectrum sections in " + str(path))
    return segments


def gadget_matter_power_spectrum(path, min_modes=8, fold_factor=16):
    """Return GADGET's CIC-deconvolved spectrum with raw, shot, signal, and Gaussian-error terms.

    The reported diagonal Gaussian mode-counting uncertainty is
    sqrt(2/N_modes) times the shot-included power.  Shot noise is subtracted
    from the signal estimate but retained in this variance.  This approximation
    excludes non-Gaussian covariance, fold-systematics, and resolution error.
    """
    segments = read_gadget_power_spectrum_segments(path)
    first = segments[0]
    box = first["box_size_internal"] / 1000.0  # Internal kpc/h to Mpc/h.
    mesh = first["mesh"]
    if isinstance(fold_factor, (bool, np.bool_)):
        raise ValueError("GADGET power-spectrum folding factor must be a positive integer")
    fold_factor_value = float(fold_factor)
    if (not math.isfinite(fold_factor_value) or fold_factor_value < 1
            or fold_factor_value != math.floor(fold_factor_value)):
        raise ValueError("GADGET power-spectrum folding factor must be a positive integer")
    fold_factor = int(fold_factor_value)
    if isinstance(min_modes, (bool, np.bool_)):
        raise ValueError("GADGET power-spectrum minimum mode count must be a positive integer")
    minimum_modes = float(min_modes)
    if (not math.isfinite(minimum_modes) or minimum_modes < 1
            or minimum_modes != math.floor(minimum_modes)):
        raise ValueError("GADGET power-spectrum minimum mode count must be a positive integer")
    min_modes = int(minimum_modes)
    if box <= 0 or mesh <= 0:
        raise ValueError("Invalid GADGET power-spectrum box or mesh metadata")
    kfund = 2.0 * math.pi / box
    upper_base = kfund * mesh / 8.0  # GADGET example's conservative quarter-Nyquist band.
    output = []
    exact_mode_tokens_used = []
    for index, segment in enumerate(segments[:3]):
        if segment["mesh"] != mesh or not math.isclose(segment["box_size_internal"], first["box_size_internal"], rel_tol=1e-7):
            raise ValueError("GADGET power-spectrum sections disagree on box or mesh")
        rows = segment["rows"]
        k = rows[:, 0] * 1000.0  # Internal h/kpc to h/Mpc.
        delta2_raw = rows[:, 1]
        delta2_shot = rows[:, 4]  # GADGET's variable-mass shot-noise estimate.
        delta2 = delta2_raw - delta2_shot
        power_native = rows[:, 2] * box**3
        modes = segment["mode_counts"].astype(np.float64)
        lower = kfund if index == 0 else upper_base * fold_factor ** (index - 1)
        upper = upper_base * fold_factor**index
        keep = (
            (k >= lower) & (k < upper) & (modes >= min_modes)
            & np.isfinite(k) & (k > 0)
            & np.isfinite(delta2_raw) & (delta2_raw >= 0)
            & np.isfinite(delta2_shot) & (delta2_shot >= 0)
            & np.isfinite(power_native) & (power_native >= 0)
            & np.isfinite(delta2) & np.isfinite(modes) & (modes > 0)
            & (modes == np.floor(modes))
        )
        if np.any(keep):
            selected_k = k[keep]
            selected_raw = delta2_raw[keep]
            selected_shot = delta2_shot[keep]
            selected_delta2 = delta2[keep]
            selected_modes = modes[keep]
            exact_mode_tokens_used.extend(segment["mode_count_tokens_exact"][keep].tolist())
            conversion = (2.0 * math.pi**2) / selected_k**3
            power_raw = selected_raw * conversion
            power_shot = selected_shot * conversion
            power_signal = selected_delta2 * conversion
            selected_native_power = power_native[keep]
            normalization_residual = (
                np.abs(power_raw - selected_native_power)
                / np.maximum(np.maximum(np.abs(power_raw), np.abs(selected_native_power)), np.finfo(float).tiny)
            )
            if np.any(normalization_residual > 1.0e-4):
                raise ValueError(
                    "GADGET power and dimensionless-power columns disagree by more than 1e-4; "
                    "check the native estimator version and units"
                )
            sigma_delta2_gaussian = np.sqrt(2.0 / selected_modes) * selected_raw
            sigma_power_gaussian = sigma_delta2_gaussian * conversion
            count = np.count_nonzero(keep)
            output.append(np.column_stack([
                selected_k, power_signal, selected_delta2, selected_modes,
                np.full(count, index), selected_raw, selected_shot,
                power_raw, power_shot, sigma_delta2_gaussian,
                sigma_power_gaussian, np.full(count, lower), np.full(count, upper),
                selected_native_power, normalization_residual,
            ]))
    if not output:
        return None
    result = np.concatenate(output, axis=0)
    result = result[np.argsort(result[:, 0])]
    type_match = re.fullmatch(r"powerspec_type(\d+)_\d+\.txt", Path(path).name, re.IGNORECASE)
    rounded_modes = np.rint(result[:, 3])
    if np.any(rounded_modes > np.iinfo(np.int64).max):
        raise ValueError("GADGET power-spectrum mode count exceeds the supported signed 64-bit range")
    return {
        "k_h_Mpc": result[:, 0], "P_Mpc_h3": result[:, 1], "delta2": result[:, 2],
        "mode_count": rounded_modes.astype(np.int64), "fold_segment": result[:, 4].astype(np.int8),
        "delta2_raw_shot_included": result[:, 5], "delta2_shot_noise": result[:, 6],
        "P_raw_shot_included_Mpc_h3": result[:, 7], "P_shot_noise_Mpc_h3": result[:, 8],
        "delta2_gaussian_1sigma": result[:, 9], "P_gaussian_1sigma_Mpc_h3": result[:, 10],
        "fold_band_lower_h_Mpc": result[:, 11], "fold_band_upper_h_Mpc": result[:, 12],
        "P_native_column_Mpc_h3": result[:, 13],
        "native_column_normalization_relative_residual": result[:, 14],
        "time": first["time"], "box_mpc_h": box, "pm_grid": mesh,
        "growth_factor": first["growth_factor"], "fold_factor": fold_factor,
        "k_fundamental_h_Mpc": kfund, "k_nyquist_h_Mpc": math.pi * mesh / box,
        "k_selection_upper_base_h_Mpc": upper_base,
        "selected_k_min_h_Mpc": float(result[0, 0]),
        "selected_k_max_h_Mpc": float(result[-1, 0]),
        "nonpositive_shot_subtracted_bin_count": int(np.count_nonzero(result[:, 1] <= 0)),
        "native_column_max_relative_normalization_residual": float(np.max(result[:, 14])),
        "native_column_normalization": "GADGET dimensionless and native P columns agree after P_native * box^3 in Mpc/h units; checked within 1e-4",
        "particle_selection": (
            "particle type %d only" % int(type_match.group(1)) if type_match
            else "all particle types; mass-weighted total"
        ),
        "uncertainty_method": (
            "diagonal Gaussian mode-counting approximation: sigma(P_signal)=sqrt(2/N_modes)*(P_signal+P_shot); "
            "shot noise is subtracted from the central value but retained in the variance; excludes non-Gaussian "
            "covariance, fold/mesh systematics, and resolution error"
        ),
        "mode_count_precision": (
            "mode counts were read from exact decimal integer tokens and retained as int64"
            if exact_mode_tokens_used and all(exact_mode_tokens_used) else
            "legacy non-integer-formatted mode-count tokens were rounded to int64; their source serialization precision is limited"
        ),
        "reliable_range_status": "not established by fold-band selection; requires observable-level convergence",
        "source": str(path),
    }


def _calibrate_ic_power_null(mode_counts, observed_median_deviation, observed_reduced_chi_squared, false_rejection_probability, trials):
    """Monte Carlo the finite-mode null using each bin's Gaussian-mode Gamma law."""
    modes = np.asarray(mode_counts, dtype=np.float64)
    if modes.ndim != 1 or len(modes) < 1 or np.any(~np.isfinite(modes)) or np.any(modes <= 0):
        raise ValueError("IC power null calibration needs positive finite mode counts")
    false_rejection_probability = float(false_rejection_probability)
    if not (0.0 < false_rejection_probability < 1.0):
        raise ValueError("IC power false_rejection_probability must be between zero and one")
    trials = int(trials)
    if trials < 10000:
        raise ValueError("IC power null calibration requires at least 10000 Monte Carlo trials")
    # Two statistics (median deviation and reduced chi-square) are Bonferroni-
    # combined below. Require enough Monte Carlo ranks to resolve that combined
    # p-value at the configured size; otherwise an arbitrarily bad spectrum
    # could be reported as statistically consistent just because the empirical
    # p-value has a coarse 1/(N+1) floor.
    minimum_resolvable_trials = int(math.ceil(2.0 / false_rejection_probability))
    if trials + 1 < minimum_resolvable_trials:
        raise ValueError(
            "IC power null calibration needs at least %d trials to resolve the configured false-rejection probability"
            % max(10000, minimum_resolvable_trials)
        )

    rng = np.random.default_rng(0x1C0FFEE)
    median_deviations = np.empty(trials, dtype=np.float64)
    reduced_chi_squared = np.empty(trials, dtype=np.float64)
    scale = 2.0 / modes
    shape = modes / 2.0
    batch_size = max(1, min(4096, 2_000_000 // len(modes)))
    for start in range(0, trials, batch_size):
        stop = min(start + batch_size, trials)
        ratios = rng.gamma(shape, scale, size=(stop - start, len(modes)))
        median_deviations[start:stop] = np.abs(np.median(ratios, axis=1) - 1.0)
        reduced_chi_squared[start:stop] = np.mean((ratios - 1.0) ** 2 / scale, axis=1)

    median_tail = (int(np.count_nonzero(median_deviations >= observed_median_deviation)) + 1) / (trials + 1)
    chi_squared_tail = (int(np.count_nonzero(reduced_chi_squared >= observed_reduced_chi_squared)) + 1) / (trials + 1)
    familywise_p = min(1.0, 2.0 * min(median_tail, chi_squared_tail))
    per_test_tail = false_rejection_probability / 2.0
    return {
        "method": "deterministic-seed Monte Carlo; independent Gaussian Fourier modes; bin power ratio ~ Gamma(Nmodes/2, 2/Nmodes)",
        "mode_counts_per_bin": modes.astype(int).tolist(),
        "trials": trials,
        "false_rejection_probability": false_rejection_probability,
        "minimum_empirical_single_statistic_p_value": float(1.0 / (trials + 1)),
        "minimum_empirical_two_statistic_familywise_p_value": float(min(1.0, 2.0 / (trials + 1))),
        "median_deviation_null_threshold": float(np.quantile(median_deviations, 1.0 - per_test_tail)),
        "reduced_chi_squared_null_threshold": float(np.quantile(reduced_chi_squared, 1.0 - per_test_tail)),
        "median_deviation_p_value": float(median_tail),
        "reduced_chi_squared_p_value": float(chi_squared_tail),
        "familywise_p_value_bonferroni": float(familywise_p),
        "status": "PASS" if familywise_p > float(false_rejection_probability) else "FAIL",
    }


def ic_species_density_reference(path, cfg, species):
    """Read explicitly named density columns; velocity is never delta_m.

    Generator v1 has five columns and no total density column. The separate
    validation reference has eight columns with an explicit delta_m. These
    schemas must not be inferred from an arbitrary numeric column count.
    """
    if species not in {'baryon','cdm','wdm'}:
        raise ValueError('Unknown IC target species')
    comments=[]
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith('#'):
                comments.append(line.strip())
    header='\n'.join(comments)
    table=np.loadtxt(path,comments='#',ndmin=2)
    if 'DarkMatterSim monofonIC transfer table v1' in header:
        model=cfg['cosmology']['model']
        if (table.shape[1]!=5 or 'gauge: N-body' not in header or f'Model: {model};' not in header
                or (species=='wdm' and model!='wdm') or (species=='cdm' and model=='wdm')):
            raise ValueError('Generator IC species table/model/schema mismatch')
        baryon=float(cfg['cosmology']['omega_b'])
        dark=float(cfg['cosmology']['omega_cdm'])
        if not all(math.isfinite(v) and v>0 for v in (baryon,dark)):
            raise ValueError('IC species reference requires positive baryon/dark physical densities')
        density=table[:,2 if species=='baryon' else 1]
        matter=(dark*table[:,1]+baryon*table[:,2])/(dark+baryon)
    elif 'k_h_per_Mpc delta_cdm delta_b delta_m delta_wdm' in header:
        if table.shape[1]!=8:
            raise ValueError('Species validation reference must have eight declared columns')
        density=table[:,{'cdm':1,'baryon':2,'wdm':4}[species]]
        matter=table[:,3]
    else:
        raise ValueError('Unrecognized IC species density-reference schema')
    k=table[:,0]
    if (len(k)<2 or np.any(~np.isfinite(k)) or np.any(k<=0) or np.any(np.diff(k)<=0)
            or np.any(~np.isfinite(density)) or np.any(~np.isfinite(matter))):
        raise ValueError('IC species density reference has invalid k/density values')
    return k,density,matter


def validate_ic_power_spectrum(gadget_spectrum_path, target_pk_path, cfg, target_transfer_file=None, target_species=None):
    """Compare a GADGET IC spectrum with its matching linear-theory target."""
    first = read_gadget_power_spectrum_segments(gadget_spectrum_path)[0]
    measured = first["rows"]
    box = first["box_size_internal"] / 1000.0
    mesh = first["mesh"]
    expected_box = float(cfg["simulation"]["box_mpc_h"])
    expected_mesh = cfg["simulation"]["pm_grid"]
    if expected_mesh == "auto":
        expected_mesh = cfg["simulation"]["particles_per_dimension"]
    if not math.isclose(box, expected_box, rel_tol=1e-6):
        raise ValueError("GADGET IC power spectrum box does not match the configured box")
    if mesh != int(expected_mesh):
        raise ValueError("GADGET IC power spectrum mesh does not match the configured PM grid")

    target = np.loadtxt(target_pk_path, comments="#", ndmin=2)
    if target.shape[1] < 2 or not np.all(np.isfinite(target[:, :2])) or np.any(target[:, :2] <= 0):
        raise ValueError("Target CLASS/AxiCLASS power spectrum is invalid")
    target_k, target_p = target[:, 0], target[:, 1]
    if np.any(np.diff(target_k) <= 0):
        raise ValueError("Target CLASS/AxiCLASS wavenumbers must be strictly increasing")
    measured_k = measured[:, 0] * 1000.0
    measured_delta2 = measured[:, 1]  # Do not Poisson-subtract a displaced lattice.
    modes = measured[:, 3]
    if (np.any(~np.isfinite(modes)) or np.any(modes <= 0)
            or np.any(modes != np.floor(modes))):
        raise ValueError("GADGET IC spectrum must report positive integer Fourier-mode counts")
    if np.any(~np.isfinite(measured_k)) or np.any(measured_k <= 0):
        raise ValueError("GADGET IC spectrum must report finite positive wavenumbers")
    kfund = 2.0 * math.pi / box
    kmax = kfund * mesh / 8.0
    target_delta2 = np.full_like(measured_k, np.nan, dtype=float)
    covered = (measured_k >= target_k[0]) & (measured_k <= target_k[-1])
    target_log_delta2 = np.log(target_k**3 * target_p / (2.0 * math.pi**2))
    target_delta2[covered] = np.exp(np.interp(np.log(measured_k[covered]), np.log(target_k), target_log_delta2))
    if target_transfer_file is not None:
        if target_species not in {"baryon", "cdm", "wdm"}:
            raise ValueError("target_species must be 'baryon', 'cdm', or 'wdm' when a transfer table is supplied")
        transfer_k,transfer_species,transfer_matter=ic_species_density_reference(target_transfer_file,cfg,target_species)
        transfer_covered = (measured_k >= transfer_k[0]) & (measured_k <= transfer_k[-1])
        covered &= transfer_covered
        log_transfer_k = np.log(transfer_k)
        t_species = np.interp(np.log(measured_k[transfer_covered]), log_transfer_k, transfer_species)
        t_matter = np.interp(np.log(measured_k[transfer_covered]), log_transfer_k, transfer_matter)
        nonzero_matter = np.abs(t_matter) > np.finfo(float).tiny
        transfer_indices = np.flatnonzero(transfer_covered)
        covered[transfer_indices[~nonzero_matter]] = False
        valid_indices = transfer_indices[nonzero_matter]
        target_delta2[valid_indices] *= (t_species[nonzero_matter] / t_matter[nonzero_matter])**2
    options = cfg.get("validation", {}).get("ic_power_spectrum", {})
    min_modes = int(options.get("min_modes_per_bin", 20))
    min_bins = int(options.get("min_bins", 5))
    use = (
        (measured_k >= kfund) & (measured_k <= kmax) & (modes >= min_modes)
        & np.isfinite(target_delta2) & (target_delta2 > 0) & (measured_delta2 > 0)
    )
    if int(np.count_nonzero(use)) < min_bins:
        raise RuntimeError("Too few well-populated Fourier bins are available for the IC power-spectrum check")
    ratio = measured_delta2[use] / target_delta2[use]
    median_ratio = float(np.median(ratio))
    median_deviation = abs(median_ratio - 1.0)
    reduced_chi_squared = float(np.mean((ratio - 1.0)**2 / (2.0 / modes[use])))
    null_calibration = _calibrate_ic_power_null(
        modes[use],
        median_deviation,
        reduced_chi_squared,
        float(options.get("false_rejection_probability", 0.001)),
        int(options.get("null_monte_carlo_trials", 100000)),
    )
    return {
        "status": null_calibration["status"],
        "estimator": "GADGET-4 restartflag 4; full mass-weighted particles; unfolded modes; no shot subtraction for lattice ICs",
        "target_spectrum": str(target_pk_path), "measured_spectrum": str(gadget_spectrum_path),
        "k_range_h_Mpc": [float(np.min(measured_k[use])), float(np.max(measured_k[use]))],
        "bins_compared": int(np.count_nonzero(use)), "modes_compared": int(np.sum(modes[use])),
        "median_measured_to_target_delta2": median_ratio, "median_fractional_deviation": median_deviation,
        "reduced_chi_squared": reduced_chi_squared,
        "finite_mode_null_calibration": null_calibration,
        "interpretation_limit": "Statistical consistency only; this null test does not set an observable-specific science tolerance or correct binning/estimator systematics.",
        **({"target_species": target_species, "target_transfer_file": str(target_transfer_file)} if target_transfer_file is not None else {}),
    }


def validate_ic_species_power_spectra(spectrum_dir, target_pk_path, target_transfer_file, cfg, dm_species="cdm"):
    """Check hydro species spectra with a familywise false-rejection bound.

    The configured IC-power false-rejection probability is the bound for the
    complete two-species gate. Bonferroni allocation keeps the bound valid
    whether or not the baryon and dark-matter estimates are correlated.
    """
    if dm_species not in {"cdm", "wdm"}:
        raise ValueError("dm_species must be 'cdm' or 'wdm'")
    species_names = ("baryon", dm_species)
    power_options = cfg.get("validation", {}).get("ic_power_spectrum", {})
    familywise_alpha = float(power_options.get("false_rejection_probability", 0.001))
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("IC-power familywise false-rejection probability must be between zero and one")
    species_alpha = familywise_alpha / len(species_names)
    species_cfg = dict(cfg)
    species_cfg["validation"] = dict(cfg.get("validation", {}))
    species_cfg["validation"]["ic_power_spectrum"] = dict(power_options)
    species_cfg["validation"]["ic_power_spectrum"]["false_rejection_probability"] = species_alpha
    root = Path(spectrum_dir)
    results = {}
    for ptype, species in ((0, "baryon"), (1, dm_species)):
        path = root / ("powerspec_type%d_000.txt" % ptype)
        if not path.is_file():
            raise FileNotFoundError("GADGET did not write the %s initial-condition power spectrum: %s" % (species, path))
        result = validate_ic_power_spectrum(
            path,
            target_pk_path,
            species_cfg,
            target_transfer_file=target_transfer_file,
            target_species=species,
        )
        result["particle_type"] = ptype
        result["allocated_familywise_false_rejection_probability"] = species_alpha
        results[species] = result
    species_p_values = [
        float(results[name]["finite_mode_null_calibration"]["familywise_p_value_bonferroni"])
        for name in species_names
    ]
    combined_p_value = min(1.0, len(species_names) * min(species_p_values))
    results["status"] = (
        "PASS" if combined_p_value > familywise_alpha
        and all(results[name]["status"] == "PASS" for name in species_names)
        else "FAIL"
    )
    results["interpretation"] = "GADGET PartType0 is baryons; PartType1 is %s in the supplied mixed-species hydro ICs" % dm_species.upper()
    results["familywise_false_rejection_probability"] = familywise_alpha
    results["species_tests"] = len(species_names)
    results["species_test_alpha"] = species_alpha
    results["familywise_p_value_bonferroni"] = combined_p_value
    results["multiple_test_correction"] = "Bonferroni across the two species; each species check Bonferroni-corrects median and reduced-chi-square diagnostics"
    return results


def plot_pk(run_dir, cfg, snap, files):
    """Plot measured periodic matter power and its mode/shot-noise metadata."""
    spectrum_dir = Path(run_dir) / "data/powerspecs"
    path = spectrum_dir / ("powerspec_%03d.txt" % snap)
    if not path.is_file():
        return None
    spectrum = gadget_matter_power_spectrum(path)
    if spectrum is None:
        return None
    k, p = spectrum["k_h_Mpc"], spectrum["P_Mpc_h3"]
    sigma = spectrum["P_gaussian_1sigma_Mpc_h3"]
    positive = np.isfinite(k) & np.isfinite(p) & (k > 0) & (p > 0)
    plt.figure(figsize=(8, 6))
    if np.any(positive):
        plt.loglog(k[positive], p[positive], label="shot-noise-subtracted signal")
        positive_lower = positive & (p - sigma > 0)
        if np.any(positive_lower):
            plt.fill_between(
                k[positive_lower], p[positive_lower] - sigma[positive_lower],
                p[positive_lower] + sigma[positive_lower], alpha=0.22,
                label="1σ Gaussian mode-counting band",
            )
    else:
        plt.text(0.5, 0.5, "No positive shot-noise-subtracted bins in the selected bands",
                 ha="center", va="center", transform=plt.gca().transAxes)
    crossing = int(np.count_nonzero(positive & (p - sigma <= 0)))
    nonpositive = spectrum["nonpositive_shot_subtracted_bin_count"]
    plt.title("GADGET-4 folded CIC-window-deconvolved matter spectrum")
    plt.xlabel(r"$k\ [h/\mathrm{Mpc}]$"); plt.ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
    plt.text(
        0.02, 0.02,
        "Measured variable-mass shot noise subtracted; %d nonpositive bins retained in HDF5; "
        "%d positive bins have a 1σ interval crossing zero. Gaussian diagonal errors omit nonlinear covariance."
        % (nonpositive, crossing), transform=plt.gca().transAxes, fontsize=8, va="bottom",
    )
    if np.any(positive):
        plt.legend()
    savefig(Path(run_dir) / "results" / ("Pk_%03d.png" % snap))
    particle_type_spectra = {}
    type_pattern = re.compile(r"powerspec_type(\d+)_%03d\.txt" % int(snap), re.IGNORECASE)
    for type_path in sorted(spectrum_dir.glob("powerspec_type*_%03d.txt" % int(snap))):
        match = type_pattern.fullmatch(type_path.name)
        if not match:
            continue
        particle_type = int(match.group(1))
        type_spectrum = gadget_matter_power_spectrum(type_path)
        if type_spectrum is not None:
            particle_type_spectra["type_%d" % particle_type] = type_spectrum
    spectrum["particle_type_spectra"] = particle_type_spectra
    if particle_type_spectra:
        plt.figure(figsize=(8, 6))
        for particle_type, type_spectrum in sorted(particle_type_spectra.items()):
            type_k = type_spectrum["k_h_Mpc"]
            type_power = type_spectrum["P_Mpc_h3"]
            type_sigma = type_spectrum["P_gaussian_1sigma_Mpc_h3"]
            type_positive = np.isfinite(type_k) & np.isfinite(type_power) & (type_k > 0) & (type_power > 0)
            if not np.any(type_positive):
                continue
            plt.loglog(type_k[type_positive], type_power[type_positive], label=particle_type)
            type_lower = type_positive & (type_power - type_sigma > 0)
            if np.any(type_lower):
                plt.fill_between(
                    type_k[type_lower], type_power[type_lower] - type_sigma[type_lower],
                    type_power[type_lower] + type_sigma[type_lower], alpha=0.16,
                )
        plt.title("GADGET-4 particle-type auto spectra (not additive to total)")
        plt.xlabel(r"$k\ [h/\mathrm{Mpc}]$")
        plt.ylabel(r"$P_{ii}(k)\ [(\mathrm{Mpc}/h)^3]$")
        plt.text(
            0.02, 0.02,
            "Each curve is a mass-weighted type auto-spectrum; total matter power also contains inter-type cross-power.",
            transform=plt.gca().transAxes, fontsize=8, va="bottom",
        )
        if plt.gca().lines:
            plt.legend()
            savefig(Path(run_dir) / "results" / ("Pk_particle_types_%03d.png" % snap))
        else:
            plt.close()
    return spectrum


def correlation_from_positions(pos, box, bins=35, minimum_separation=None):
    """Measure periodic-box xi(r), excluding pairs at or below minimum_separation."""
    box = float(box)
    bins = _exact_analysis_integer(bins, "Correlation bin count", minimum=3)
    points = np.asarray(pos, dtype=float)
    if not np.isfinite(box) or box <= 0:
        raise ValueError("Correlation box size must be finite and positive")
    if bins < 3:
        raise ValueError("Correlation needs at least three radial bins")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Correlation positions must have shape (particles, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return None
    points = np.mod(points, box)
    if minimum_separation is None:
        effective_grid = min(128, max(32, int(round(len(points) ** (1.0 / 3.0))) * 2))
        minimum_separation = box / effective_grid
        minimum_separation_source = (
            "catalog-size heuristic: box divided by effective grid %d (not a convergence-derived scale)"
            % effective_grid
        )
    else:
        minimum_separation_source = "explicit user-configured minimum separation in Mpc/h"
    if isinstance(minimum_separation, (bool, np.bool_)):
        raise ValueError("Correlation minimum separation must be a finite length in Mpc/h")
    minimum_separation = float(minimum_separation)
    maximum_separation = box / 2.0
    if not np.isfinite(minimum_separation) or not 0.0 < minimum_separation < maximum_separation:
        raise ValueError("Correlation minimum separation must be between zero and half the box size")

    bin_edges = np.geomspace(minimum_separation, maximum_separation, bins + 1)
    tree = cKDTree(points, boxsize=box)
    ordered_counts = np.asarray(
        tree.count_neighbors(tree, bin_edges, cumulative=False), dtype=np.int64,
    )
    if ordered_counts.shape != (bins + 1,):
        raise RuntimeError("Periodic pair counter returned an unexpected bin shape")
    close_ordered_count = int(ordered_counts[0]) - len(points)  # Remove self-pairs; remaining pairs are <= the cutoff.
    pair_ordered_counts = ordered_counts[1:]
    if (close_ordered_count < 0 or close_ordered_count % 2
            or np.any(pair_ordered_counts < 0) or np.any(pair_ordered_counts % 2)):
        raise RuntimeError("Periodic auto-pair counts did not reduce to nonnegative unordered pairs")
    excluded_close_pair_count = close_ordered_count // 2
    pair_counts = pair_ordered_counts // 2

    inner_edges = bin_edges[:-1]
    outer_edges = bin_edges[1:]
    shell_volumes = (4.0 * math.pi / 3.0) * (outer_edges**3 - inner_edges**3)
    random_pair_expectation = (len(points) * (len(points) - 1) / 2.0) * shell_volumes / box**3
    if np.any(~np.isfinite(random_pair_expectation)) or np.any(random_pair_expectation <= 0):
        raise RuntimeError("Analytic random-pair expectation is invalid")
    xi = pair_counts / random_pair_expectation - 1.0

    # Exact central Garwood intervals for pair counts, transformed to xi.
    alpha = 1.0 - 0.682689492137
    lower_count = np.zeros(pair_counts.shape, dtype=float)
    nonzero = pair_counts > 0
    lower_count[nonzero] = 0.5 * chi2.ppf(alpha / 2.0, 2.0 * pair_counts[nonzero])
    upper_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (pair_counts + 1))
    xi_lower = lower_count / random_pair_expectation - 1.0
    xi_upper = upper_count / random_pair_expectation - 1.0
    # Volume-weighted mean separation is a representative point for each bin.
    radius = 0.75 * (outer_edges**4 - inner_edges**4) / (outer_edges**3 - inner_edges**3)
    return {
        "r": radius,
        "xi": xi,
        "bin_edges": bin_edges,
        "pair_count": pair_counts,
        "excluded_close_pair_count": int(excluded_close_pair_count),
        "random_pair_expectation": random_pair_expectation,
        "xi_poisson_68_lower": xi_lower,
        "xi_poisson_68_upper": xi_upper,
        "input_point_count": int(np.asarray(pos).shape[0]),
        "used_point_count": int(len(points)),
        "excluded_nonfinite_point_count": int(np.asarray(pos).shape[0] - len(points)),
        "minimum_separation_mpc_h": float(minimum_separation),
        "minimum_separation_source": minimum_separation_source,
        "maximum_separation_mpc_h": float(maximum_separation),
        "method": (
            "direct periodic minimum-image unordered auto-pairs; self-pairs removed; distinct pairs below the "
            "configured minimum separation (inclusive) excluded; analytic uniform RR shell volumes; no mesh assignment or "
            "particle shot subtraction; r <= box/2"
        ),
    }


def plot_two_point(run_dir, cfg, snap):
    """Plot the measured spatial correlation function using the saved estimator and length units."""
    pos = group_field(run_dir, snap, ["GroupPos"])
    minimum_halos = _exact_analysis_integer(
        cfg.get("analysis", {}).get("correlation_min_halos", 20),
        "analysis.correlation_min_halos", minimum=2,
    )
    if pos is None:
        return {"status": "SKIPPED", "reason": "no FOF host positions are available"}
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return {"status": "SKIPPED", "reason": "FOF host positions are not an (N, 3) array"}
    finite_positions = np.all(np.isfinite(np.asarray(pos, dtype=float)), axis=1)
    finite_count = int(np.count_nonzero(finite_positions))
    if finite_count < minimum_halos:
        return {"status": "SKIPPED", "reason": "fewer than %d finite host halos" % minimum_halos}
    box = float(cfg["simulation"]["box_mpc_h"])
    pos = units_to_mpc_h(pos, box)
    result = correlation_from_positions(
        pos, box, bins=_exact_analysis_integer(
            cfg.get("analysis", {}).get("correlation_bins", 35),
            "analysis.correlation_bins", minimum=3,
        ),
        minimum_separation=cfg.get("analysis", {}).get("correlation_min_separation_mpc_h"),
    )
    if result is None:
        return {"status": "SKIPPED", "reason": "fewer than two finite host positions"}
    yerr = np.vstack([
        np.maximum(result["xi"] - result["xi_poisson_68_lower"], 0.0),
        np.maximum(result["xi_poisson_68_upper"] - result["xi"], 0.0),
    ])
    plt.figure(figsize=(8, 6)); plt.errorbar(result["r"], result["xi"], yerr=yerr, fmt="o-", markersize=3, capsize=2)
    plt.axhline(0, linewidth=1); plt.title("Periodic FOF host correlation; direct pairs and analytic RR")
    plt.xlabel(r"$r\ [\mathrm{Mpc}/h]$"); plt.ylabel(r"$\xi_{hh}(r)$")
    plt.text(
        0.02, 0.02,
        "Pairs at/below %.3g Mpc/h excluded; no mesh or shot-noise subtraction. Garwood intervals assume Poisson pairs; bin covariance is omitted."
        % result["minimum_separation_mpc_h"],
        transform=plt.gca().transAxes, fontsize=8, va="bottom",
    )
    savefig(Path(run_dir) / "results" / ("two_point_correlation_%03d.png" % snap))
    result["status"] = "COMPLETE"
    result["halo_count"] = int(result["used_point_count"])
    result["uncertainty_method"] = (
        "per-bin central Garwood interval assuming Poisson pair counts; shared-halo bin covariance and "
        "finite-volume uncertainty are not estimated"
    )
    return result


def plot_halo_bias(run_dir, cfg, snap, files):
    """Compare measured halo/matter clustering on the declared common scales."""
    hpos = group_field(run_dir, snap, ["GroupPos"])
    if hpos is None:
        return {"status": "SKIPPED", "reason": "no FOF host positions are available"}
    masses = halo_masses(run_dir, snap)
    if masses is None:
        return {"status": "SKIPPED", "reason": "no host-halo mass field is available for the configured mass definition"}
    counts = halo_particle_counts(run_dir, snap)
    minimum_particles = _exact_analysis_integer(
        cfg["analysis"]["hmf_min_particles"],
        "analysis.hmf_min_particles for halo bias",
    )
    _, particle_mask, filter_info = apply_hmf_particle_cut(masses, counts, minimum_particles)
    if particle_mask is None:
        return {"status": "SKIPPED", "reason": filter_info["reason"]}
    if np.asarray(hpos).ndim != 2 or np.asarray(hpos).shape[1] != 3 or len(hpos) != len(masses):
        return {"status": "SKIPPED", "reason": "FOF host positions and masses do not align"}
    selected_mass = mass_to_msun_h(np.asarray(masses, dtype=float))
    valid_halo = particle_mask & np.isfinite(selected_mass) & (selected_mass > 0)
    positions = np.asarray(hpos, dtype=float)[valid_halo]
    positions = positions[np.all(np.isfinite(positions), axis=1)]
    minimum_halos = _exact_analysis_integer(
        cfg.get("analysis", {}).get("halo_bias_min_halos", 20),
        "analysis.halo_bias_min_halos", minimum=2,
    )
    if len(positions) < minimum_halos:
        return {
            "status": "SKIPPED",
            "reason": "fewer than %d finite host halos remain after the %d-particle cut" % (minimum_halos, minimum_particles),
        }
    box = float(cfg["simulation"]["box_mpc_h"])
    grid = _exact_analysis_integer(
        cfg.get("analysis", {}).get("power_grid", 256),
        "analysis.power_grid for halo bias", minimum=16, maximum=256,
    )
    min_modes = _exact_analysis_integer(
        cfg.get("analysis", {}).get("halo_bias_min_modes_per_bin", 20),
        "analysis.halo_bias_min_modes_per_bin",
    )
    nyquist_fraction = float(cfg.get("analysis", {}).get("halo_bias_nyquist_fraction", 0.25))
    if not np.isfinite(nyquist_fraction) or not 0 < nyquist_fraction <= 0.5:
        raise ValueError("analysis.halo_bias_nyquist_fraction must be in (0, 0.5]")

    halo_density = np.zeros((grid, grid, grid), dtype=np.float64)
    halo_positions = units_to_mpc_h(positions, box)
    _deposit_cic_positions(halo_density, halo_positions, box, grid)
    try:
        matter_contrast, matter_particle_count, matter_shot_noise = _snapshot_matter_cic_mesh(files, box, grid)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "SKIPPED", "reason": "could not build the full mass-weighted matter mesh: " + str(exc)}
    result = _cic_cross_power_bias(
        halo_density, matter_contrast, box, len(positions), matter_shot_noise,
        min_modes=min_modes, nyquist_fraction=nyquist_fraction,
    )
    usable = np.isfinite(result["bias"])
    if np.count_nonzero(usable) < 2:
        return {
            "status": "SKIPPED",
            "reason": "fewer than two Fourier bins have enough modes and positive shot-noise-subtracted matter power",
            "halo_count": int(len(positions)), "matter_particle_count": int(matter_particle_count),
        }
    plt.figure(figsize=(8, 6)); plt.errorbar(
        result["k_h_Mpc"][usable], result["bias"][usable],
        yerr=result["bias_gaussian_68_sigma"][usable], fmt="o-", markersize=3, capsize=2,
        label=r"$b_{hm}=P_{hm}/P_{mm}$",
    )
    auto_usable = np.isfinite(result["bias_auto_hh_mm"])
    if np.any(auto_usable):
        plt.errorbar(
            result["k_h_Mpc"][auto_usable], result["bias_auto_hh_mm"][auto_usable],
            yerr=result["bias_auto_hh_mm_gaussian_68_sigma"][auto_usable],
            fmt="s--", markersize=3, capsize=2, label=r"$b_{hh}=\sqrt{P_{hh}/P_{mm}}$ (cross-check)",
        )
    plt.axhline(0.0, linewidth=1)
    mass_definition = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    plt.title("Host-halo bias from the CIC halo-matter cross spectrum")
    plt.xlabel(r"$k\ [h/\mathrm{Mpc}]$"); plt.ylabel(r"$b_{hm}(k)=P_{hm}(k)/P_{mm}(k)$")
    plt.text(
        0.02, 0.02,
        "CIC-window corrected; all massive particle types in $P_{mm}$; measured matter shot noise removed.\n"
        "%s-selected host mass; GroupLen >= %d (FOF membership); FOF GroupPos centers; k <= %.2f k_N.\n"
        "68%% diagonal Gaussian covariance only; mesh/aliasing, nonlinear, and volume errors are not included."
        % (mass_definition.upper(), minimum_particles, nyquist_fraction),
        transform=plt.gca().transAxes, fontsize=8, va="bottom",
    )
    plt.legend()
    savefig(Path(run_dir) / "results" / ("halo_bias_%03d.png" % snap))
    result.update({
        "status": "COMPLETE", "halo_count": int(len(positions)),
        "matter_particle_count": int(matter_particle_count),
        "minimum_halo_particles": minimum_particles,
        "mass_definition": mass_definition,
        "halo_particle_selection_definition": "GADGET GroupLen (total FOF membership count), even when selecting by an attached spherical-overdensity mass",
        "halo_center_definition": "GADGET GroupPos field: parent FOF group position",
        "selected_k_min_h_Mpc": float(np.min(result["k_h_Mpc"][usable])),
        "selected_k_max_h_Mpc": float(np.max(result["k_h_Mpc"][usable])),
        "uncertainty": (
            "68.27% diagonal Gaussian Fourier-mode covariance approximation using shot-included auto-power in the "
            "variance and shot-subtracted P_mm in the ratio; excludes residual assignment/aliasing, mesh-resolution, "
            "non-Gaussian, and finite-volume covariance"
        ),
    })
    return result


def plot_particle_velocity_distribution(run_dir, snap, files):
    """Plot saved particle peculiar speeds after the explicit cosmological velocity conversion."""
    vel = read_particle_field(files, 1, "Velocities", max_points=2_000_000)
    if vel is None:
        return None
    scale_factor = _snapshot_header_value(files, "Time")
    hubble_param = _snapshot_parameter(files, "HubbleParam")
    units=_snapshot_dataset_unit_attributes(files,1,'Velocities')
    if scale_factor is None or scale_factor <= 0 or hubble_param is None or units is None:
        return None
    speed = np.linalg.norm(np.asarray(vel, float), axis=1) * dataset_to_cgs_factor(units,scale_factor,hubble_param) / 1.0e5
    speed = speed[np.isfinite(speed)]
    if not len(speed):
        return None
    density, edges = histogram_pdf(speed, 100)
    plt.figure(figsize=(8, 6)); plt.stairs(density, edges)
    plt.xlabel("dark-matter peculiar speed [km/s]"); plt.ylabel("probability density per km/s")
    savefig(Path(run_dir) / "results" / ("particle_velocity_distribution_%03d.png" % snap))
    return {"bin_edges": edges, "probability_density": density, "sample_count": len(speed)}


def stable_histogram_edges(values, bins):
    """Finite, distinct float64 edges, including nearly constant measurements.

    Padding only resolves floating-point bin geometry; it supplies no samples.
    Empty/invalid measurements remain unavailable rather than becoming zero.
    """
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("Histogram bins must be a positive integer")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Histogram has no finite measurements")
    lo, hi = float(values.min()), float(values.max())
    scale = max(abs(lo), abs(hi), 1.)
    # Several ULPs per bin guarantee unique boundaries, even after log10(T)
    # maps temperatures that differ by only a few representable numbers.
    padding = scale * (8 * np.finfo(float).eps * int(bins))
    with np.errstate(over='ignore', invalid='ignore'):
        width = hi - lo
    if not np.isfinite(width):
        raise ValueError("Histogram range is too wide for finite float64 bin widths")
    if width < 2*padding:
        center = lo/2 + hi/2
        bound = np.finfo(float).max
        lo = max(-bound, center-padding)
        hi = min(bound, center+padding)
    with np.errstate(over='ignore', invalid='ignore'):
        edges = np.linspace(lo, hi, int(bins)+1)
        widths = np.diff(edges)
    if not np.isfinite(edges).all() or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("Histogram edges cannot be represented as distinct finite float64 values")
    return edges


def histogram_pdf(values, bins):
    """Normalized PDF; finite samples only, no synthetic replacement samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=stable_histogram_edges(values, bins))
    total = int(np.sum(counts))
    density = (counts / float(total)) / np.diff(edges)
    return density, edges


def periodic_delta(x, c, box):
    """Return a minimum-image separation within a comoving periodic box, preserving the input length unit."""
    d = x - c
    return d - box * np.round(d / box)


def profile_halo_scatter(profiles, value_key, radial_bins):
    """Summarize halo-to-halo profile scatter without calling it a CI."""
    radius_grid = np.logspace(-2.0, np.log10(2.0), int(radial_bins))
    curves = []
    for profile in profiles:
        radius = np.asarray(profile.get("radius_comoving_mpc_h", []), dtype=float)
        r200 = float(profile.get("r200c_comoving_mpc_h", np.nan))
        values = np.asarray(profile.get(value_key, []), dtype=float)
        if radius.ndim != 1 or values.shape != radius.shape or not np.isfinite(r200) or r200 <= 0:
            continue
        x = radius / r200
        good = np.isfinite(x) & (x > 0) & np.isfinite(values) & (values > 0)
        if np.count_nonzero(good) < 2:
            continue
        log_x = np.log10(x[good])
        log_y = np.log10(values[good])
        target = np.log10(radius_grid)
        inside = (target >= log_x[0]) & (target <= log_x[-1])
        curve = np.full(len(radius_grid), np.nan, dtype=float)
        curve[inside] = 10.0 ** np.interp(target[inside], log_x, log_y)
        curves.append(curve)
    if not curves:
        empty = np.full(len(radius_grid), np.nan, dtype=float)
        return {
            "status": "UNAVAILABLE", "reason": "fewer than two positive radial bins per halo",
            "radius_over_r200c": radius_grid, "valid_halo_count": np.zeros(len(radius_grid), dtype=np.int64),
            "lower_16": empty, "median": empty.copy(), "upper_84": empty.copy(),
        }
    values = np.asarray(curves, dtype=float)
    valid_count = np.sum(np.isfinite(values), axis=0)
    lower = np.full(len(radius_grid), np.nan, dtype=float)
    median = np.full(len(radius_grid), np.nan, dtype=float)
    upper = np.full(len(radius_grid), np.nan, dtype=float)
    for index in np.flatnonzero(valid_count >= 3):
        sample = values[np.isfinite(values[:, index]), index]
        lower[index], median[index], upper[index] = np.percentile(sample, [16.0, 50.0, 84.0])
    supported = valid_count >= 3
    status = "COMPLETE" if np.any(supported) else "UNAVAILABLE"
    return {
        "status": status,
        "reason": None if status == "COMPLETE" else "at least three selected halos must cover a radius bin",
        "radius_over_r200c": radius_grid,
        "valid_halo_count": valid_count,
        "lower_16": lower,
        "median": median,
        "upper_84": upper,
        "method": "16th, 50th, and 84th percentiles across the selected individual halos; descriptive halo-to-halo scatter, not a confidence interval",
        "selected_halo_count": int(len(curves)),
    }


def top_halo_profiles(run_dir, cfg, snap, files, max_halos=8):
    """Build explicitly unit-checked dark-matter profiles around valid SO halos.

    Group_R_Crit200 and snapshot coordinates are comoving in the project's
    GADGET convention. The 2 R200 aperture is only profiled when its outer
    radius fits inside the periodic box's inscribed sphere.
    """
    analysis_cfg = cfg.get("analysis", {})
    max_halos = _exact_analysis_integer(analysis_cfg.get("halo_profile_max_halos", max_halos), "halo_profile_max_halos")
    max_points = _exact_analysis_integer(analysis_cfg.get("halo_profile_max_particles", 4_000_000), "halo_profile_max_particles")
    radial_bins = _exact_analysis_integer(analysis_cfg.get("halo_profile_radial_bins", 24), "halo_profile_radial_bins", minimum=3)
    result = {
        "status": "SKIPPED", "reason": None, "velocity_status": "SKIPPED",
        "velocity_reason": None, "profiles": [], "velocity_dispersion_profiles": [],
    }
    try:
        box = float(cfg["simulation"]["box_mpc_h"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "simulation.box_mpc_h is unavailable: " + str(exc)
        return result
    if not np.isfinite(box) or box <= 0:
        result["reason"] = "simulation.box_mpc_h must be finite and positive"
        return result

    try:
        pos_h = group_field(run_dir, snap, ["GroupPos"])
        mass_h = group_field(run_dir, snap, ["Group_M_Crit200"])
        rad_h = group_field(run_dir, snap, ["Group_R_Crit200"])
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read aligned GADGET halo-catalogue fields: " + str(exc)
        return result
    if pos_h is None or mass_h is None or rad_h is None:
        result["reason"] = "GADGET GroupPos, Group_M_Crit200, and Group_R_Crit200 are required"
        return result
    pos_h = np.asarray(pos_h, dtype=float)
    mass_h = mass_to_msun_h(mass_h)
    rad_h = units_to_mpc_h(np.asarray(rad_h, dtype=float).reshape(-1, 1), box).ravel()
    mass_h = np.asarray(mass_h, dtype=float).reshape(-1)
    if pos_h.ndim != 2 or pos_h.shape[1:] != (3,) or len(pos_h) != len(mass_h) or len(rad_h) != len(mass_h):
        result["reason"] = "GADGET GroupPos, Group_M_Crit200, and Group_R_Crit200 are not row-aligned"
        return result
    pos_h = units_to_mpc_h(pos_h, box)
    valid_halos = (
        np.all(np.isfinite(pos_h), axis=1) & np.isfinite(mass_h) & (mass_h > 0)
        & np.isfinite(rad_h) & (rad_h > 0)
    )
    candidate_indices = np.flatnonzero(valid_halos)
    if not len(candidate_indices):
        result["reason"] = "no halos have finite centers and positive finite M200c and R200c"
        return result
    geometry_safe = 2.0 * rad_h[candidate_indices] <= box / 2.0
    safe_indices = candidate_indices[geometry_safe]
    result["skipped_halo_periodic_geometry_count"] = int(np.count_nonzero(~geometry_safe))
    if not len(safe_indices):
        result["reason"] = "all finite positive-mass halos have 2 R200c larger than the periodic box's inscribed sphere"
        return result
    # Sort only valid, periodic-safe rows so invalid or unmeasurable halos cannot
    # displace real profile candidates from the requested top-mass sample.
    order = safe_indices[np.argsort(mass_h[safe_indices])[::-1]][:max_halos]

    try:
        particle_fields, sample_weights = read_particle_fields(
            files, 1, ["Coordinates", "Velocities", "Masses"], max_points=max_points,
            return_sample_weights=True,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read aligned dark-matter particle fields: " + str(exc)
        return result
    pos = particle_fields["Coordinates"]
    vel = particle_fields["Velocities"]
    if pos is None or not len(pos):
        result["reason"] = "snapshot has no PartType1/Coordinates rows"
        return result
    scale_factor = _snapshot_header_value(files, "Time")
    hubble_param = _snapshot_header_value(files, "HubbleParam")
    if hubble_param is None:
        hubble_param = _snapshot_parameter(files, "HubbleParam")
    if scale_factor is None or not np.isfinite(scale_factor) or scale_factor <= 0:
        result["reason"] = "snapshot Header/Time is missing or invalid"
        return result
    if hubble_param is None or not np.isfinite(hubble_param) or hubble_param <= 0:
        result["reason"] = "snapshot HubbleParam is missing or invalid"
        return result
    try:
        particle_mass = _sampled_particle_masses_msun_h(
            files, 1, particle_fields["Masses"], len(pos), scale_factor, hubble_param,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not convert PartType1 masses to Msun/h: " + str(exc)
        return result
    sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.shape != (len(pos),) or not np.all(np.isfinite(sample_weights)) or np.any(sample_weights <= 0):
        result["reason"] = "dark-matter inverse-inclusion sample weights are invalid or misaligned"
        return result
    particle_mass *= sample_weights

    total_dm_count = 0
    try:
        for path in files:
            with h5py.File(path, "r") as h5:
                header = h5["Header"].attrs
                counts = np.asarray(header.get("NumPart_ThisFile", []))
                if counts.shape != (6,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
                    result["reason"] = "snapshot Header/NumPart_ThisFile is missing or invalid"
                    return result
                total_dm_count += int(counts[1])
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read snapshot particle counts: " + str(exc)
        return result
    if not np.isclose(np.sum(sample_weights), total_dm_count, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError("DM profile sample weights do not sum to the snapshot particle count")
    pos = units_to_mpc_h(pos, box)
    valid_positions = np.all(np.isfinite(pos), axis=1)
    result["invalid_particle_position_count"] = int(np.count_nonzero(~valid_positions))
    velocity_reason = None
    if vel is not None:
        try:
            velocity_units = _snapshot_dataset_unit_attributes(files, 1, "Velocities")
            velocity_factor = _dataset_to_cgs_factor(velocity_units, scale_factor, hubble_param)
            if velocity_factor is None or not np.isfinite(velocity_factor) or velocity_factor <= 0:
                raise ValueError("PartType1/Velocities lacks valid physical-unit metadata")
            vel = np.asarray(vel, dtype=float) * velocity_factor / 1.0e5
            if vel.shape != (len(pos), 3):
                raise ValueError("PartType1/Velocities must align as an N by 3 array")
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            velocity_reason = str(exc)
            vel = None
    else:
        velocity_reason = "PartType1/Velocities is unavailable"
    all_rho, all_sig = [], []
    phase_portrait_written = False
    fig1 = plt.figure(figsize=(8, 6)); ax1 = fig1.gca()
    fig2 = plt.figure(figsize=(8, 6)); ax2 = fig2.gca()
    skipped_geometry = result["skipped_halo_periodic_geometry_count"]
    skipped_low_count = 0
    for i in order:
        rmax = 2.0 * rad_h[i]
        d = periodic_delta(pos, pos_h[i], box)
        r = np.linalg.norm(d, axis=1)
        keep = valid_positions & (r < rmax)
        if np.sum(keep) < 100:
            skipped_low_count += 1
            continue
        minimum_radius = min(max(rmax / 100.0, 1.0e-4), rmax / 2.0)
        edges = np.logspace(np.log10(minimum_radius), np.log10(rmax), radial_bins + 1)
        mass_in_shell, _ = np.histogram(r[keep], edges, weights=particle_mass[keep])
        vol = 4 * math.pi / 3 * (edges[1:]**3 - edges[:-1]**3)
        # Uniform random sampling is corrected back to the full DM particle count.
        rho_comoving = mass_in_shell / vol
        cen = np.sqrt(edges[:-1] * edges[1:])
        count_in_shell = np.histogram(r[keep], edges)[0].astype(np.uint64)
        count_estimate = np.histogram(r[keep], edges, weights=sample_weights[keep])[0]
        inner = keep & (r < edges[0])
        halo_profile = {
            "halo_catalogue_row": int(i), "mass_msun_h": float(mass_h[i]),
            "center_group_pos_comoving_mpc_h": pos_h[i].copy(),
            "r200c_comoving_mpc_h": float(rad_h[i]),
            "radial_bin_edges_comoving_mpc_h": edges,
            "radius_comoving_mpc_h": cen,
            "sampled_particle_count_per_shell": count_in_shell,
            "particle_count_estimate_per_shell": count_estimate,
            "dm_mass_shell_msun_h": mass_in_shell,
            "dm_mass_density_comoving_msun_h_per_mpc_h3": rho_comoving,
            "dm_mass_density_proper_msun_h_per_mpc_h3": rho_comoving / scale_factor**3,
            "sampled_particles_in_aperture": int(np.count_nonzero(keep)),
            "estimated_particles_in_aperture": float(np.sum(sample_weights[keep])),
            "sampled_particles_inside_inner_cut": int(np.count_nonzero(inner)),
            "estimated_particles_inside_inner_cut": float(np.sum(sample_weights[inner])),
            "estimated_mass_inside_inner_cut_msun_h": float(np.sum(particle_mass[inner])),
            "profile_inner_radius_comoving_mpc_h": float(edges[0]),
            "profile_outer_radius_comoving_mpc_h": float(edges[-1]),
        }
        good = mass_in_shell > 0
        ax1.loglog(cen[good] / rad_h[i], halo_profile["dm_mass_density_proper_msun_h_per_mpc_h3"][good], alpha=0.65)
        all_rho.append(halo_profile)
        if vel is not None and len(vel) == len(pos):
            velocity_good = keep & np.all(np.isfinite(vel), axis=1)
            if not phase_portrait_written:
                from core.halo_phase_portrait import phase_space_histogram, export_phase_portrait
                portrait = phase_space_histogram(d, vel, sample_weights, float(rad_h[i]))
                if portrait is not None:
                    phase_path = Path(run_dir) / 'results' / ('halo_phase_space_%03d.png' % snap)
                    export_phase_portrait(portrait, phase_path, snapshot=snap, halo_row=i,
                                          redshift=1.0 / scale_factor - 1.0)
                    phase_portrait_written = True
            sig = np.full(radial_bins, np.nan, dtype=float)
            velocity_count = np.zeros(radial_bins, dtype=np.uint64)
            velocity_count_estimate = np.zeros(radial_bins, dtype=float)
            for j in range(len(edges) - 1):
                q = velocity_good & (r >= edges[j]) & (r < edges[j + 1])
                velocity_count[j] = np.count_nonzero(q)
                velocity_count_estimate[j] = float(np.sum(sample_weights[q]))
                if velocity_count[j] > 10:
                    weights = particle_mass[q]
                    mean_velocity = np.average(vel[q], axis=0, weights=weights)
                    sig[j] = np.sqrt(np.average(np.sum((vel[q] - mean_velocity)**2, axis=1), weights=weights))
            ax2.semilogx(cen / rad_h[i], sig, alpha=0.65)
            all_sig.append({
                "halo_catalogue_row": int(i), "mass_msun_h": float(mass_h[i]),
                "center_group_pos_comoving_mpc_h": pos_h[i].copy(),
                "r200c_comoving_mpc_h": float(rad_h[i]),
                "radial_bin_edges_comoving_mpc_h": edges,
                "radius_comoving_mpc_h": cen,
                "velocity_dispersion_3d_peculiar_km_s": sig,
                "sampled_velocity_particle_count_per_shell": velocity_count,
                "velocity_particle_count_estimate_per_shell": velocity_count_estimate,
                "definition": "mass-weighted 3-D peculiar-velocity RMS about the shell mean; sqrt(sum of three component variances), not the 1-D dispersion",
            })
    density_scatter = profile_halo_scatter(
        all_rho, "dm_mass_density_proper_msun_h_per_mpc_h3", radial_bins,
    )
    if density_scatter["status"] == "COMPLETE":
        supported = np.isfinite(density_scatter["median"])
        radius_grid = density_scatter["radius_over_r200c"]
        ax1.fill_between(
            radius_grid[supported], density_scatter["lower_16"][supported],
            density_scatter["upper_84"][supported], color="#64748b", alpha=0.18,
            linewidth=0, label="16–84% halo-to-halo scatter",
        )
        ax1.loglog(
            radius_grid[supported], density_scatter["median"][supported],
            color="#111827", linewidth=2.1,
            label="median of %d selected halos" % density_scatter["selected_halo_count"],
        )
        ax1.legend(loc="best", fontsize=7)
    velocity_scatter = profile_halo_scatter(
        all_sig, "velocity_dispersion_3d_peculiar_km_s", radial_bins,
    )
    if velocity_scatter["status"] == "COMPLETE":
        supported = np.isfinite(velocity_scatter["median"])
        radius_grid = velocity_scatter["radius_over_r200c"]
        ax2.fill_between(
            radius_grid[supported], velocity_scatter["lower_16"][supported],
            velocity_scatter["upper_84"][supported], color="#64748b", alpha=0.18,
            linewidth=0, label="16–84% halo-to-halo scatter",
        )
        ax2.loglog(
            radius_grid[supported], velocity_scatter["median"][supported],
            color="#111827", linewidth=2.1,
            label="median of %d selected halos" % velocity_scatter["selected_halo_count"],
        )
        ax2.legend(loc="best", fontsize=7)
    result.update({
        "selected_valid_halo_count": int(len(order)),
        "selected_halo_catalogue_rows": [int(value) for value in order],
        "skipped_halo_periodic_geometry_count": int(skipped_geometry),
        "skipped_halo_low_particle_count": int(skipped_low_count),
        "sampled_particle_count": int(len(pos)),
        "snapshot_particle_count": int(total_dm_count),
        "sampled_particle_weight_sum": float(np.sum(sample_weights)),
        "scale_factor": float(scale_factor),
        "particle_mass_conversion": "dataset to_cgs*a_scaling*h_scaling metadata (or consistent Header/MassTable and UnitMass), then grams to Msun/h; inverse-inclusion weights applied",
        "velocity_conversion": "PartType1/Velocities to_cgs*a_scaling*h_scaling metadata to km/s; GADGET-4 cosmological snapshots encode the sqrt(a) peculiar-velocity convention in a_scaling=1/2",
        "center_definition": "GADGET GroupPos; for this SUBFIND build the parent group position is reset to the primary subhalo potential minimum",
        "mass_definition": "GADGET Group_M_Crit200 (M200c); Group_R_Crit200 converted as comoving length",
        "density_definition": "inverse-inclusion-weighted dark-matter particle mass per spherical shell volume; proper density equals comoving density divided by a^3",
        "uncertainty_definition": "bands show descriptive 16th–84th percentile halo-to-halo scatter across the selected top-mass sample, not a confidence interval or sampling uncertainty",
        "density_profile_scatter": density_scatter,
        "velocity_profile_scatter": velocity_scatter,
        "velocity_reason": velocity_reason,
    })
    if all_rho:
        ax1.set_xlabel(r"$r/R_{200c}$"); ax1.set_ylabel(r"DM mass density [$M_\odot/h$ per $(\mathrm{Mpc}/h)^3$; proper]")
        ax1.set_title("Dark-matter radial mass-density profiles")
        fig1.tight_layout()
        export_figure(fig1, Path(run_dir) / "results" / ("halo_density_profiles_%03d.png" % snap), metadata={
            "snapshot": int(snap), "product": "dark-matter radial density profiles",
            "mass_definition": "M200c", "ensemble_band": density_scatter.get("method"),
        })
    plt.close(fig1)
    if all_sig:
        ax2.set_xlabel(r"$r/R_{200c}$"); ax2.set_ylabel(r"3-D peculiar velocity dispersion [km/s]")
        ax2.set_title("Dark-matter peculiar velocity dispersion profiles")
        fig2.tight_layout()
        export_figure(fig2, Path(run_dir) / "results" / ("velocity_dispersion_profiles_%03d.png" % snap), metadata={
            "snapshot": int(snap), "product": "dark-matter velocity-dispersion profiles",
            "definition": "mass-weighted 3-D peculiar-velocity RMS about shell mean",
            "ensemble_band": velocity_scatter.get("method"),
        })
    plt.close(fig2)
    if not all_rho:
        result["reason"] = "no selected halo has at least 100 finite-position sampled dark-matter particles within a periodic-safe 2 R200c aperture"
        return result
    result["status"] = "COMPLETE"
    result["profiles"] = all_rho
    result["velocity_dispersion_profiles"] = all_sig
    result["halo_count"] = len(all_rho)
    if any(np.any(np.isfinite(profile["velocity_dispersion_3d_peculiar_km_s"])) for profile in all_sig):
        result["velocity_status"] = "COMPLETE"
        result["velocity_reason"] = None
    else:
        result["velocity_status"] = "NO_VALID_VALUES"
        result["velocity_reason"] = velocity_reason or "fewer than eleven finite velocity samples fell in every radial shell"
    return result


def _sampled_particle_masses_msun_h(files, ptype, mass_values, sample_count,
                                    scale_factor, hubble_param):
    """Convert sampled particle masses to Msun/h without guessing their units."""
    if sample_count < 1:
        raise ValueError("Cannot convert an empty particle-mass sample")
    if mass_values is not None:
        units = _snapshot_dataset_unit_attributes(files, ptype, "Masses")
        factor = _dataset_to_cgs_factor(units, scale_factor, hubble_param)
        if factor is None or not np.isfinite(factor) or factor <= 0:
            raise ValueError("PartType%d/Masses lacks valid physical-unit metadata" % ptype)
        conversion = factor * hubble_param / 1.98847e33
        masses = np.asarray(mass_values, dtype=float) * conversion
    else:
        table_values = []
        for path in files:
            with h5py.File(path, "r") as h5:
                if "Header" not in h5:
                    raise ValueError("Snapshot Header is required for constant particle masses")
                table = np.asarray(h5["Header"].attrs.get("MassTable", []), dtype=float)
                if table.shape != (6,) or not np.all(np.isfinite(table)):
                    raise ValueError("Snapshot Header/MassTable must contain six finite values")
                table_values.append(float(table[int(ptype)]))
        if (not table_values or any(not np.isclose(value, table_values[0], rtol=1.0e-12, atol=0.0)
                                    for value in table_values[1:])
                or table_values[0] <= 0):
            raise ValueError("PartType%d has no consistent positive constant particle mass" % ptype)
        unit_mass = _snapshot_parameter(files, "UnitMass_in_g")
        if unit_mass is None or not np.isfinite(unit_mass) or unit_mass <= 0:
            raise ValueError("Parameters/UnitMass_in_g is required for constant particle masses")
        masses = np.full(sample_count, table_values[0] * unit_mass / 1.98847e33, dtype=float)
    if masses.shape != (sample_count,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("Sampled PartType%d masses must be finite and positive" % ptype)
    return masses


def top_gas_profiles(run_dir, cfg, snap, files, max_halos=8, max_points=4_000_000):
    """Build unit-checked gas density, temperature, and velocity profiles."""
    analysis_cfg = cfg.get("analysis", {})
    max_halos = _exact_analysis_integer(analysis_cfg.get("halo_profile_max_halos", max_halos), "halo_profile_max_halos")
    max_points = _exact_analysis_integer(analysis_cfg.get("halo_profile_max_particles", max_points), "halo_profile_max_particles")
    radial_bins = _exact_analysis_integer(analysis_cfg.get("halo_profile_radial_bins", 24), "halo_profile_radial_bins", minimum=3)
    result = {
        "status": "SKIPPED", "reason": None,
        "temperature_status": "SKIPPED", "temperature_reason": None,
        "velocity_status": "SKIPPED", "velocity_reason": None,
        "profiles": [],
    }
    try:
        box = float(cfg["simulation"]["box_mpc_h"])
        pos_h = group_field(run_dir, snap, ["GroupPos"])
        mass_h = group_field(run_dir, snap, ["Group_M_Crit200"])
        rad_h = group_field(run_dir, snap, ["Group_R_Crit200"])
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "simulation box or GADGET halo-catalogue input is invalid: " + str(exc)
        return result
    if not np.isfinite(box) or box <= 0:
        result["reason"] = "simulation.box_mpc_h must be finite and positive"
        return result
    if pos_h is None or mass_h is None or rad_h is None:
        result["reason"] = "GADGET GroupPos, Group_M_Crit200, and Group_R_Crit200 are required"
        return result
    try:
        pos_h = np.asarray(pos_h, dtype=float)
        rad_h = units_to_mpc_h(np.asarray(rad_h, dtype=float).reshape(-1, 1), box).ravel()
        mass_h = np.asarray(mass_to_msun_h(mass_h), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "GADGET halo-catalogue fields could not be converted: " + str(exc)
        return result
    if pos_h.ndim != 2 or pos_h.shape[1:] != (3,) or len(pos_h) != len(mass_h) or len(rad_h) != len(mass_h):
        result["reason"] = "GADGET GroupPos, Group_M_Crit200, and Group_R_Crit200 are not row-aligned"
        return result
    pos_h = units_to_mpc_h(pos_h, box)
    valid_halos = (
        np.all(np.isfinite(pos_h), axis=1) & np.isfinite(mass_h) & (mass_h > 0)
        & np.isfinite(rad_h) & (rad_h > 0)
    )
    candidate_rows = np.flatnonzero(valid_halos)
    if not len(candidate_rows):
        result["reason"] = "no finite-center halos with positive finite M200c and R200c"
        return result
    periodic_safe = 2.0 * rad_h[candidate_rows] <= box / 2.0
    result["skipped_halo_periodic_geometry_count"] = int(np.count_nonzero(~periodic_safe))
    candidate_rows = candidate_rows[periodic_safe]
    if not len(candidate_rows):
        result["reason"] = "all valid halos have 2 R200c larger than the periodic box's inscribed sphere"
        return result
    profile_order = candidate_rows[np.argsort(mass_h[candidate_rows])[::-1]][:max_halos]

    field_names = [
        "Coordinates", "Velocities", "Masses", "Temperature", "InternalEnergy",
        "ElectronAbundance", "StarFormationRate",
    ]
    try:
        gas_fields, sample_weights = read_particle_fields(
            files, 0, field_names, max_points=max_points, return_sample_weights=True,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read an aligned gas-particle sample: " + str(exc)
        return result
    gas_positions = gas_fields["Coordinates"]
    if gas_positions is None or not len(gas_positions):
        result["reason"] = "snapshot contains no gas coordinates"
        return result
    gas_positions = np.asarray(gas_positions, dtype=float)
    if gas_positions.ndim != 2 or gas_positions.shape[1:] != (3,):
        result["reason"] = "PartType0/Coordinates must be an N by 3 array"
        return result
    scale_factor = _snapshot_header_value(files, "Time")
    hubble_param = _snapshot_header_value(files, "HubbleParam")
    if hubble_param is None:
        hubble_param = _snapshot_parameter(files, "HubbleParam")
    if scale_factor is None or not np.isfinite(scale_factor) or scale_factor <= 0:
        result["reason"] = "Snapshot Header/Time is missing or invalid"
        return result
    if hubble_param is None or not np.isfinite(hubble_param) or hubble_param <= 0:
        result["reason"] = "Snapshot HubbleParam is missing or invalid"
        return result
    try:
        gas_mass = _sampled_particle_masses_msun_h(
            files, 0, gas_fields["Masses"], len(gas_positions), scale_factor, hubble_param,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = str(exc)
        return result
    sample_weights = np.asarray(sample_weights, dtype=float)
    if (sample_weights.shape != (len(gas_positions),) or not np.all(np.isfinite(sample_weights))
            or np.any(sample_weights <= 0)):
        result["reason"] = "gas inverse-inclusion sample weights are invalid or misaligned"
        return result
    total_gas_count = 0
    try:
        for path in files:
            with h5py.File(path, "r") as h5:
                counts = np.asarray(h5["Header"].attrs.get("NumPart_ThisFile", []))
                if counts.shape != (6,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
                    result["reason"] = "snapshot Header/NumPart_ThisFile is missing or invalid"
                    return result
                total_gas_count += int(counts[0])
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read gas particle counts: " + str(exc)
        return result
    if not np.isclose(np.sum(sample_weights), total_gas_count, rtol=1.0e-12, atol=1.0e-12):
        result["reason"] = "gas sample weights do not recover the snapshot particle count"
        return result
    gas_mass *= sample_weights
    gas_positions = units_to_mpc_h(gas_positions, box)
    position_good = np.all(np.isfinite(gas_positions), axis=1)
    result["invalid_gas_position_count"] = int(np.count_nonzero(~position_good))

    temperature = None
    temperature_source = None
    sfr = gas_fields["StarFormationRate"]
    star_formation_enabled = bool(cfg.get("hydro", {}).get("star_formation", False))
    if gas_fields["Temperature"] is not None:
        try:
            temperature_units = _snapshot_dataset_unit_attributes(files, 0, "Temperature")
            unit_names = set()
            for path in files:
                with h5py.File(path, "r") as h5:
                    if "PartType0/Temperature" in h5:
                        unit_name = h5["PartType0/Temperature"].attrs.get(
                            "unit", h5["PartType0/Temperature"].attrs.get("units"),
                        )
                        if isinstance(unit_name, bytes):
                            unit_name = unit_name.decode("utf-8", errors="replace")
                        unit_names.add(unit_name)
            temperature_factor = _dataset_to_cgs_factor(
                temperature_units, scale_factor, hubble_param,
            )
            if unit_names != {"K"} or temperature_factor is None or not np.isclose(
                temperature_factor, 1.0, rtol=1.0e-8, atol=1.0e-12,
            ):
                raise ValueError("PartType0/Temperature must carry explicit Kelvin units")
            temperature = np.asarray(gas_fields["Temperature"], dtype=float) * temperature_factor
            temperature_source = "stored Kelvin"
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            result["temperature_reason"] = str(exc)
    elif gas_fields["InternalEnergy"] is not None and gas_fields["ElectronAbundance"] is not None:
        if star_formation_enabled and sfr is None:
            result["temperature_reason"] = "cannot identify star-forming gas without StarFormationRate"
        else:
            try:
                energy_units = _snapshot_dataset_unit_attributes(files, 0, "InternalEnergy")
                if energy_units is None:
                    raise ValueError("PartType0/InternalEnergy lacks explicit unit metadata")
                energy_factor = _dataset_to_cgs_factor(
                    energy_units, scale_factor, hubble_param,
                )
                temperature = temperature_from_internal_energy(
                    gas_fields["InternalEnergy"], gas_fields["ElectronAbundance"],
                    internal_energy_to_cgs=energy_factor,
                )
                temperature_source = "derived from InternalEnergy and ElectronAbundance"
            except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
                result["temperature_reason"] = str(exc)
    else:
        result["temperature_reason"] = "neither unit-tagged Temperature nor complete InternalEnergy/ElectronAbundance is available"
    if temperature is not None:
        temp_good = np.isfinite(temperature) & (temperature > 0)
        if sfr is not None:
            temp_good &= np.isfinite(sfr) & (np.asarray(sfr) == 0.0)
        elif star_formation_enabled:
            temp_good[:] = False
            result["temperature_reason"] = "cannot identify star-forming gas without StarFormationRate"
        result["temperature_status"] = "COMPLETE" if np.any(temp_good) else "NO_VALID_VALUES"
        if not np.any(temp_good) and result["temperature_reason"] is None:
            result["temperature_reason"] = "no positive finite non-star-forming gas temperatures remain"
    velocity_km_s = None
    if gas_fields["Velocities"] is None:
        result["velocity_reason"] = "PartType0/Velocities is unavailable"
    else:
        try:
            velocity_units = _snapshot_dataset_unit_attributes(files, 0, "Velocities")
            velocity_factor = _dataset_to_cgs_factor(
                velocity_units, scale_factor, hubble_param,
            )
            if velocity_factor is None or not np.isfinite(velocity_factor) or velocity_factor <= 0:
                raise ValueError("PartType0/Velocities lacks valid physical-unit metadata")
            velocity_km_s = np.asarray(gas_fields["Velocities"], dtype=float) * velocity_factor / 1.0e5
            if velocity_km_s.shape != (len(gas_positions), 3):
                raise ValueError("PartType0/Velocities must align as an N by 3 array")
            result["velocity_status"] = "COMPLETE" if np.any(np.all(np.isfinite(velocity_km_s), axis=1)) else "NO_VALID_VALUES"
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            result["velocity_reason"] = str(exc)

    profiles = []
    skipped_low_particle_count = 0
    for halo_index in profile_order:
        maximum_radius = 2.0 * rad_h[halo_index]
        displacement = periodic_delta(gas_positions, pos_h[halo_index], box)
        radius = np.linalg.norm(displacement, axis=1)
        inside = position_good & (radius < maximum_radius)
        if np.count_nonzero(inside) < 100:
            skipped_low_particle_count += 1
            continue
        minimum_radius = min(max(maximum_radius / 100.0, 1.0e-4), maximum_radius / 2.0)
        edges = np.logspace(np.log10(minimum_radius),
                            np.log10(maximum_radius), radial_bins + 1)
        shell_volume = 4.0 * math.pi / 3.0 * (edges[1:]**3 - edges[:-1]**3)
        mass_shell, _ = np.histogram(radius[inside], edges, weights=gas_mass[inside])
        density_comoving = mass_shell / shell_volume
        density_proper = density_comoving / scale_factor**3
        particle_count = np.histogram(radius[inside], edges)[0].astype(np.uint64)
        particle_count_estimate = np.histogram(
            radius[inside], edges, weights=sample_weights[inside],
        )[0]
        inner = inside & (radius < edges[0])
        profile = {
            "halo_catalogue_row": int(halo_index), "mass_msun_h": float(mass_h[halo_index]),
            "center_group_pos_comoving_mpc_h": pos_h[halo_index].copy(),
            "r200c_comoving_mpc_h": float(rad_h[halo_index]),
            "radial_bin_edges_comoving_mpc_h": edges,
            "radius_comoving_mpc_h": np.sqrt(edges[:-1] * edges[1:]),
            "gas_particle_count": particle_count,
            "gas_particle_count_estimate": particle_count_estimate,
            "gas_mass_shell_msun_h": mass_shell,
            "gas_mass_density_comoving_msun_h_per_mpc_h3": density_comoving,
            "gas_mass_density_proper_msun_h_per_mpc_h3": density_proper,
            "sampled_particles_in_aperture": int(np.count_nonzero(inside)),
            "estimated_particles_in_aperture": float(np.sum(sample_weights[inside])),
            "estimated_gas_mass_in_aperture_msun_h": float(np.sum(gas_mass[inside])),
            "sampled_particles_inside_inner_cut": int(np.count_nonzero(inner)),
            "estimated_particles_inside_inner_cut": float(np.sum(sample_weights[inner])),
            "estimated_gas_mass_inside_inner_cut_msun_h": float(np.sum(gas_mass[inner])),
            "profile_inner_radius_comoving_mpc_h": float(edges[0]),
            "profile_outer_radius_comoving_mpc_h": float(edges[-1]),
        }
        if temperature is not None and result["temperature_status"] == "COMPLETE":
            valid_temperature = inside & temp_good
            thermal_mass, _ = np.histogram(radius[valid_temperature], edges, weights=gas_mass[valid_temperature])
            thermal_energy_proxy, _ = np.histogram(
                radius[valid_temperature], edges,
                weights=gas_mass[valid_temperature] * temperature[valid_temperature],
            )
            profile["gas_temperature_mass_weighted_K"] = np.divide(
                thermal_energy_proxy, thermal_mass,
                out=np.full(thermal_mass.shape, np.nan), where=thermal_mass > 0,
            )
            profile["gas_temperature_particle_count"] = np.histogram(
                radius[valid_temperature], edges,
            )[0].astype(np.uint64)
            profile["gas_temperature_particle_count_estimate"] = np.histogram(
                radius[valid_temperature], edges, weights=sample_weights[valid_temperature],
            )[0]
        if velocity_km_s is not None and result["velocity_status"] == "COMPLETE":
            valid_velocity = inside & np.all(np.isfinite(velocity_km_s), axis=1)
            sigma = np.full(len(edges) - 1, np.nan, dtype=float)
            velocity_count = np.zeros(len(edges) - 1, dtype=np.uint64)
            for shell_index in range(len(edges) - 1):
                in_shell = valid_velocity & (radius >= edges[shell_index]) & (radius < edges[shell_index + 1])
                velocity_count[shell_index] = np.count_nonzero(in_shell)
                if velocity_count[shell_index] < 10:
                    continue
                weights = gas_mass[in_shell]
                mean_velocity = np.average(velocity_km_s[in_shell], axis=0, weights=weights)
                sigma[shell_index] = np.sqrt(np.average(
                    np.sum((velocity_km_s[in_shell] - mean_velocity)**2, axis=1), weights=weights,
                ))
            profile["gas_velocity_dispersion_3d_peculiar_km_s"] = sigma
            profile["gas_velocity_particle_count"] = velocity_count
            profile["gas_velocity_particle_count_estimate"] = np.histogram(
                radius[valid_velocity], edges, weights=sample_weights[valid_velocity],
            )[0]
        profiles.append(profile)

    if not profiles:
        result["reason"] = "no selected halo has at least 100 finite-position sampled gas particles within a periodic-safe 2 R200c aperture"
        return result
    if not any("gas_temperature_mass_weighted_K" in profile and np.any(
        np.isfinite(profile["gas_temperature_mass_weighted_K"])
    ) for profile in profiles):
        result["temperature_status"] = "NO_VALID_VALUES"
        if result["temperature_reason"] is None:
            result["temperature_reason"] = "no valid temperature samples fell within the selected halo apertures"
    if not any("gas_velocity_dispersion_3d_peculiar_km_s" in profile and np.any(
        np.isfinite(profile["gas_velocity_dispersion_3d_peculiar_km_s"])
    ) for profile in profiles):
        result["velocity_status"] = "NO_VALID_VALUES"
        if result["velocity_reason"] is None:
            result["velocity_reason"] = "fewer than ten valid velocity samples fell in every radial shell"
    else:
        result["velocity_status"] = "COMPLETE"
        result["velocity_reason"] = None
    if result["temperature_status"] == "COMPLETE":
        result["temperature_reason"] = None
    profile_scatter = {
        "gas_mass_density_proper_msun_h_per_mpc_h3": profile_halo_scatter(
            profiles, "gas_mass_density_proper_msun_h_per_mpc_h3", radial_bins,
        ),
    }
    if result["temperature_status"] == "COMPLETE":
        profile_scatter["gas_temperature_mass_weighted_K"] = profile_halo_scatter(
            profiles, "gas_temperature_mass_weighted_K", radial_bins,
        )
    if result["velocity_status"] == "COMPLETE":
        profile_scatter["gas_velocity_dispersion_3d_peculiar_km_s"] = profile_halo_scatter(
            profiles, "gas_velocity_dispersion_3d_peculiar_km_s", radial_bins,
        )
    result.update({
        "status": "COMPLETE", "reason": None, "profiles": profiles,
        "profile_halo_to_halo_scatter": profile_scatter,
        "halo_count": len(profiles), "sampled_particle_count": int(len(gas_positions)),
        "snapshot_particle_count": int(total_gas_count),
        "sampled_particle_weight_sum": float(np.sum(sample_weights)),
        "selected_valid_halo_count": int(len(profile_order)),
        "selected_halo_catalogue_rows": [int(row) for row in profile_order],
        "skipped_halo_low_particle_count": int(skipped_low_particle_count),
        "scale_factor": float(scale_factor),
        "center_definition": "GADGET GroupPos; for this SUBFIND-enabled project configuration, the primary subhalo potential minimum",
        "mass_definition": "GADGET Group_M_Crit200 (M200c); R200c and particle positions are comoving",
        "particle_selection": "PartType0 gas particle masses inside a spherical 2 R200c aperture; all gas included for density and velocity, only positive finite non-star-forming temperatures included for temperature",
        "uncertainty_definition": "figures show descriptive 16th-84th percentile halo-to-halo scatter across the selected top-mass sample when at least three halos support a radial bin; this is not a confidence interval and does not estimate per-shell particle-sampling, Poisson, or spatial-jackknife uncertainty",
        "temperature_source": temperature_source,
        "temperature_reason": result["temperature_reason"],
        "velocity_reason": result["velocity_reason"],
        "density_definition": "inverse-inclusion-weighted gas particle mass per spherical shell volume (not an SPH-kernel density); proper density divides comoving density by a^3",
        "temperature_definition": "sample-weighted mass-weighted mean of positive finite non-star-forming gas temperature in each shell",
        "velocity_definition": "sample-weighted mass-weighted 3-D peculiar velocity RMS about the shell mean; no Hubble-flow term, and not the 1-D dispersion",
    })

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    for profile in profiles:
        density = profile["gas_mass_density_proper_msun_h_per_mpc_h3"]
        good = np.isfinite(density) & (density > 0)
        axes.loglog(
            profile["radius_comoving_mpc_h"][good] / profile["r200c_comoving_mpc_h"],
            density[good], color="#94a3b8", alpha=0.35, linewidth=0.8,
        )
    density_scatter = profile_scatter["gas_mass_density_proper_msun_h_per_mpc_h3"]
    if density_scatter["status"] == "COMPLETE":
        support = np.isfinite(density_scatter["median"])
        radius = density_scatter["radius_over_r200c"]
        axes.fill_between(
            radius[support], density_scatter["lower_16"][support],
            density_scatter["upper_84"][support], color="#56B4E9", alpha=0.20,
            linewidth=0, label="16–84% halo-to-halo scatter",
        )
        axes.loglog(
            radius[support], density_scatter["median"][support], color="#0072B2",
            linewidth=2.2, label="median of %d selected halos" % density_scatter["selected_halo_count"],
        )
        axes.legend(loc="best")
    axes.set_xlabel(r"$r/R_{200c}$"); axes.set_ylabel(r"Gas mass density [$M_\odot/h$ per $(\mathrm{Mpc}/h)^3$; proper]")
    axes.set_title("Gas radial mass-density profiles")
    savefig(Path(run_dir) / "results" / ("gas_density_profiles_%03d.png" % snap), metadata={
        "product": "gas radial mass-density profiles",
        "uncertainty": result["uncertainty_definition"],
        "profile_summary": density_scatter.get("method"),
    })
    if result["temperature_status"] == "COMPLETE":
        fig, axis = plt.subplots(figsize=(8, 6))
        for profile in profiles:
            if "gas_temperature_mass_weighted_K" not in profile:
                continue
            values = profile["gas_temperature_mass_weighted_K"]
            good = np.isfinite(values) & (values > 0)
            axis.loglog(
                profile["radius_comoving_mpc_h"][good] / profile["r200c_comoving_mpc_h"],
                values[good], color="#94a3b8", alpha=0.35, linewidth=0.8,
            )
        temperature_scatter = profile_scatter["gas_temperature_mass_weighted_K"]
        if temperature_scatter["status"] == "COMPLETE":
            support = np.isfinite(temperature_scatter["median"])
            radius = temperature_scatter["radius_over_r200c"]
            axis.fill_between(
                radius[support], temperature_scatter["lower_16"][support],
                temperature_scatter["upper_84"][support], color="#E69F00", alpha=0.20,
                linewidth=0, label="16–84% halo-to-halo scatter",
            )
            axis.loglog(
                radius[support], temperature_scatter["median"][support], color="#D55E00",
                linewidth=2.2, label="median of %d selected halos" % temperature_scatter["selected_halo_count"],
            )
            axis.legend(loc="best")
        axis.set_xlabel(r"$r/R_{200c}$"); axis.set_ylabel("mass-weighted gas temperature [K]")
        axis.set_title("Gas radial temperature profiles")
        savefig(Path(run_dir) / "results" / ("gas_temperature_profiles_%03d.png" % snap), metadata={
            "product": "gas radial temperature profiles",
            "uncertainty": result["uncertainty_definition"],
            "profile_summary": temperature_scatter.get("method"),
        })
    if result["velocity_status"] == "COMPLETE":
        fig, axis = plt.subplots(figsize=(8, 6))
        for profile in profiles:
            if "gas_velocity_dispersion_3d_peculiar_km_s" not in profile:
                continue
            values = profile["gas_velocity_dispersion_3d_peculiar_km_s"]
            good = np.isfinite(values) & (values > 0)
            axis.semilogx(
                profile["radius_comoving_mpc_h"][good] / profile["r200c_comoving_mpc_h"],
                values[good], color="#94a3b8", alpha=0.35, linewidth=0.8,
            )
        velocity_scatter = profile_scatter["gas_velocity_dispersion_3d_peculiar_km_s"]
        if velocity_scatter["status"] == "COMPLETE":
            support = np.isfinite(velocity_scatter["median"])
            radius = velocity_scatter["radius_over_r200c"]
            axis.fill_between(
                radius[support], velocity_scatter["lower_16"][support],
                velocity_scatter["upper_84"][support], color="#CC79A7", alpha=0.20,
                linewidth=0, label="16–84% halo-to-halo scatter",
            )
            axis.semilogx(
                radius[support], velocity_scatter["median"][support], color="#6A3D9A",
                linewidth=2.2, label="median of %d selected halos" % velocity_scatter["selected_halo_count"],
            )
            axis.legend(loc="best")
        axis.set_xlabel(r"$r/R_{200c}$"); axis.set_ylabel("3-D peculiar velocity dispersion [km/s]")
        axis.set_title("Gas radial velocity-dispersion profiles")
        savefig(Path(run_dir) / "results" / ("gas_velocity_dispersion_profiles_%03d.png" % snap), metadata={
            "product": "gas radial velocity-dispersion profiles",
            "uncertainty": result["uncertainty_definition"],
            "profile_summary": velocity_scatter.get("method"),
        })
    return result


def _shape_tensor_components(offsets, weights=None):
    """Return descending mass-quadrupole eigenpairs and axis ratios."""
    q = np.asarray(offsets, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) < 3:
        return None
    finite_positions = np.all(np.isfinite(q), axis=1)
    if weights is None:
        w = np.ones(len(q), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(q),) or np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("shape tensor weights must be finite, non-negative, and row-aligned")
    valid = finite_positions & (w > 0)
    q, w = q[valid], w[valid]
    if len(q) < 3 or not np.isfinite(np.sum(w)) or np.sum(w) <= 0:
        return None
    tensor = (q.T * w) @ q / float(np.sum(w))
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        return None
    eigenvalues = np.maximum(eigenvalues, 0.0)
    a2, b2, c2 = eigenvalues
    b_over_a = np.sqrt(b2 / a2)
    c_over_a = np.sqrt(c2 / a2)
    denominator = a2 - c2
    tolerance = np.finfo(float).eps * max(float(a2), float(c2), np.finfo(float).tiny) * 16.0
    triaxiality = np.nan if denominator <= tolerance else (a2 - b2) / denominator
    return eigenvalues, eigenvectors, (float(b_over_a), float(c_over_a), float(triaxiality))


def shape_tensor_metrics(offsets, weights=None):
    """Return b/a, c/a and T from a (optionally mass-weighted) shape tensor."""
    result = _shape_tensor_components(offsets, weights=weights)
    return None if result is None else result[2]


def trace_lagrangian_protohalo_shapes(run_dir, snap, min_particles=100, max_halos=40,
                                     max_particles=4_000_000, mass_definition="fof"):
    """Trace exact FoF type-1 memberships to IC positions and measure axis ratios.

    Coordinates are comoving Mpc/h. The mass-weighted covariance eigenvalues
    give b/a=sqrt(lambda_2/lambda_1), c/a=sqrt(lambda_3/lambda_1). This measures
    a declared halo sample; it does not classify every WDM halo as physical or
    numerical. Catalogue row numbers index the output arrays but never match
    particles. Only exact uint64 ParticleIDs establish Lagrangian membership.
    """
    from core.data import read_selected_group_ids, snapshot_id_shards
    from core.foundation import MPC_CM
    if min_particles < 3 or max_halos < 1 or max_particles < min_particles:
        raise ValueError("Protohalo tracing needs positive explicit sample limits and at least three particles")
    cfg = load_cfg(run_dir)
    if str(cfg["analysis"]["hmf_mass_definition"]) != str(mass_definition):
        raise ValueError("Protohalo mass ranking must use the saved HMF mass convention")
    raw_mass = halo_masses(run_dir, snap)
    offsets = group_field(run_dir, snap, ["GroupOffsetType"])
    lengths = group_field(run_dir, snap, ["GroupLenType"])
    if raw_mass is None or offsets is None or lengths is None:
        raise ValueError("Protohalo tracing requires exact GroupOffsetType and GroupLenType memberships")
    n_halos = len(raw_mass)
    if np.asarray(lengths).shape != (n_halos, 6) or np.asarray(offsets).shape != (n_halos, 6):
        raise ValueError("Protohalo catalogue masses and membership rows do not align")
    status = np.zeros(n_halos, dtype=np.uint8)
    status[np.asarray(lengths)[:, 1] < min_particles] = 2
    candidates = np.flatnonzero((status == 0) & np.isfinite(raw_mass) & (np.asarray(raw_mass) > 0))
    candidates = candidates[np.argsort(-np.asarray(raw_mass)[candidates], kind="stable")]
    rows, budget = [], 0
    for row in candidates:
        count = int(lengths[row, 1])
        if len(rows) >= max_halos:
            break
        if budget+count > max_particles:
            status[row] = 3
            continue
        rows.append(int(row))
        budget += count
    records, _ = snapshot_id_shards(group_ordered_snapshot_files(run_dir, snap))
    ids, labels = read_selected_group_ids(offsets, lengths, np.asarray(rows, dtype=np.int64), records,
                                          max_ids=int(max_particles))
    sort = np.argsort(ids)
    ids, labels = ids[sort], labels[sort]
    coordinates, masses = np.full((len(ids), 3), np.nan), np.full(len(ids), np.nan)
    found = np.zeros(len(ids), dtype=bool)
    ic_files = _initial_condition_paths(run_dir)
    if not ic_files:
        raise FileNotFoundError("Exact protohalo tracing needs the original IC particle files")
    box = float(cfg["simulation"]["box_mpc_h"])
    for path in ic_files:
        with h5py.File(path, "r") as source:
            group = source.get("PartType1")
            if group is None:
                continue
            if "ParticleIDs" not in group:
                raise ValueError("Initial particles have no exact ParticleIDs")
            params = source["Parameters"].attrs if "Parameters" in source else {}
            unit = float(params.get("UnitLength_in_cm", MPC_CM/1000.))/MPC_CM
            ic_box = float(source["Header"].attrs["BoxSize"])*unit
            if not np.isclose(ic_box, box, rtol=1.e-8):
                raise ValueError("IC and snapshot comoving boxes differ")
            count = len(group["ParticleIDs"])
            if group["ParticleIDs"].dtype.kind not in "ui" or group["Coordinates"].shape != (count, 3):
                raise ValueError("IC IDs/positions must be aligned integer IDs and (N,3) coordinates")
            for start in range(0, count, 1_000_000):
                stop = min(count, start+1_000_000)
                chunk_ids = np.asarray(group["ParticleIDs"][start:stop])
                if chunk_ids.dtype.kind == "i" and np.any(chunk_ids < 0):
                    raise ValueError("IC contains negative particle IDs")
                chunk_ids = chunk_ids.astype(np.uint64, copy=False)
                index = np.searchsorted(ids, chunk_ids)
                within = index < len(ids)
                selected = np.flatnonzero(within)
                selected = selected[ids[index[selected]] == chunk_ids[selected]]
                dest = index[selected]
                if np.any(found[dest]) or len(np.unique(dest)) != len(dest):
                    raise ValueError("IC ParticleIDs are duplicated in the traced membership")
                if not len(dest):
                    continue
                coordinates[dest] = np.asarray(group["Coordinates"][start:stop], dtype=float)[selected]*unit
                constant = float(np.asarray(source["Header"].attrs["MassTable"])[1])
                masses[dest] = (np.asarray(group["Masses"][start:stop], dtype=float)[selected]
                                if "Masses" in group else constant)
                found[dest] = True
    center = np.full((n_halos, 3), np.nan)
    counts = np.zeros(n_halos, dtype=np.int64)
    b_over_a, c_over_a = np.full(n_halos, np.nan), np.full(n_halos, np.nan)
    for row in rows:
        selected = labels == row
        counts[row] = int(np.count_nonzero(found[selected]))
        if not np.all(found[selected]):
            status[row] = 4
            continue
        pos, mass = coordinates[selected], masses[selected]
        if np.any(~np.isfinite(pos)) or np.any(~np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("Traced IC coordinates/masses must be finite with positive mass")
        # Unwrap around one actual member, then recenter by mass. Broad patches
        # spanning half the box are ambiguous and are rejected, never clipped.
        delta = periodic_delta(pos, pos[0], box)
        if np.any(np.ptp(delta, axis=0) >= box/2.):
            status[row] = 6
            continue
        mean = np.average(delta, axis=0, weights=mass)
        metrics = shape_tensor_metrics(delta-mean, weights=mass)
        if metrics is None:
            status[row] = 5
            continue
        center[row] = np.mod(pos[0]+mean, box)
        b_over_a[row], c_over_a[row] = metrics[:2]
        status[row] = 1
    return {"status": "COMPLETE", "snapshot": int(snap),
            "method": "exact FoF DM ParticleID tracing to ICs; periodic mass-weighted covariance",
            "status_codes": {"0": "not_selected", "1": "measured", "2": "below_particle_cut",
                             "3": "outside_memory_budget", "4": "missing_IC_ID", "5": "degenerate_shape",
                             "6": "ambiguous_periodic_extent"},
            "row_status": status, "particle_count": counts, "center_mpc_h": center,
            "b_over_a": b_over_a, "c_over_a": c_over_a, "sphericity_c_over_a": c_over_a.copy()}


def summarize_halo_shapes(profiles):
    """Summarize the measured selection, not the whole halo population.

    Scatter describes halo-to-halo variation; it is not a convergence error.
    Means of axis ratios do not specify an average spatial orientation.
    """
    summary = {"halo_count": len(profiles),
               "selection": "resolved halos from the configured highest-M200c selection",
               "uncertainty": "sample standard deviation across halos; no resolution/convergence error",
               "metrics": {}}
    masses = np.asarray([p["mass_msun_h"] for p in profiles], dtype=float)
    for name in ("b_over_a", "c_over_a", "triaxiality"):
        values = np.asarray([p[name] for p in profiles], dtype=float)
        valid = np.isfinite(values) & np.isfinite(masses) & (masses > 0)
        values, weights = values[valid], masses[valid]
        summary["metrics"][name] = {
            "count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
            "mass_weighted_mean": float(np.average(values, weights=weights)) if len(values) else None,
            "percentile_16": float(np.percentile(values, 16)) if len(values) else None,
            "percentile_84": float(np.percentile(values, 84)) if len(values) else None,
        }
    return summary


def halo_shapes(run_dir, cfg, snap, files, max_halos=40):
    """Measure mass-quadrupole shapes of type-1 matter in spherical R200c apertures."""
    analysis_cfg = cfg.get("analysis", {})
    max_halos = _exact_analysis_integer(analysis_cfg.get("halo_shape_max_halos", max_halos), "halo_shape_max_halos")
    max_points = _exact_analysis_integer(analysis_cfg.get("halo_shape_max_particles", 4_000_000), "halo_shape_max_particles")
    minimum_particles = _exact_analysis_integer(analysis_cfg.get("halo_shape_min_particles", 100), "halo_shape_min_particles", minimum=3)
    result = {
        "status": "SKIPPED", "reason": None, "profiles": [],
        "minimum_sampled_particles": minimum_particles,
    }
    try:
        box = float(cfg["simulation"]["box_mpc_h"])
        pos_h = group_field(run_dir, snap, ["GroupPos"])
        mass_h = group_field(run_dir, snap, ["Group_M_Crit200"])
        rad_h = group_field(run_dir, snap, ["Group_R_Crit200"])
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "simulation box or GADGET halo-catalogue input is invalid: " + str(exc)
        return result
    if not np.isfinite(box) or box <= 0:
        result["reason"] = "simulation.box_mpc_h must be finite and positive"
        return result
    if pos_h is None or mass_h is None or rad_h is None:
        result["reason"] = "GADGET GroupPos, Group_M_Crit200, and Group_R_Crit200 are required"
        return result
    try:
        pos_h = np.asarray(pos_h, dtype=float)
        mass_h = np.asarray(mass_to_msun_h(mass_h), dtype=float).reshape(-1)
        rad_h = units_to_mpc_h(np.asarray(rad_h, dtype=float).reshape(-1, 1), box).ravel()
    except (TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "GADGET halo-catalogue fields could not be converted: " + str(exc)
        return result
    if pos_h.ndim != 2 or pos_h.shape[1:] != (3,) or len(pos_h) != len(mass_h) or len(rad_h) != len(mass_h):
        result["reason"] = "GADGET GroupPos, Group_M_Crit200, and Group_R_Crit200 are not row-aligned"
        return result
    pos_h = units_to_mpc_h(pos_h, box)
    valid = (
        np.all(np.isfinite(pos_h), axis=1) & np.isfinite(mass_h) & (mass_h > 0)
        & np.isfinite(rad_h) & (rad_h > 0)
    )
    valid_rows = np.flatnonzero(valid)
    if not len(valid_rows):
        result["reason"] = "no finite-center halos with positive finite M200c and R200c"
        return result
    periodic_safe = rad_h[valid_rows] <= box / 2.0
    result["skipped_periodic_geometry_count"] = int(np.count_nonzero(~periodic_safe))
    valid_rows = valid_rows[periodic_safe]
    if not len(valid_rows):
        result["reason"] = "all valid halos have R200c larger than the periodic box's inscribed sphere"
        return result
    order = valid_rows[np.argsort(mass_h[valid_rows])[::-1]][:max_halos]
    try:
        fields, sample_weights = read_particle_fields(
            files, 1, ["Coordinates", "Masses"], max_points=max_points,
            return_sample_weights=True,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read aligned type-1 particle positions and masses: " + str(exc)
        return result
    pos = fields["Coordinates"]
    if pos is None or not len(pos):
        result["reason"] = "snapshot has no PartType1/Coordinates rows"
        return result
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1:] != (3,):
        result["reason"] = "PartType1/Coordinates must be an N by 3 array"
        return result
    scale_factor = _snapshot_header_value(files, "Time")
    hubble_param = _snapshot_header_value(files, "HubbleParam")
    if hubble_param is None:
        hubble_param = _snapshot_parameter(files, "HubbleParam")
    if scale_factor is None or not np.isfinite(scale_factor) or scale_factor <= 0:
        result["reason"] = "snapshot Header/Time is missing or invalid"
        return result
    if hubble_param is None or not np.isfinite(hubble_param) or hubble_param <= 0:
        result["reason"] = "snapshot HubbleParam is missing or invalid"
        return result
    try:
        particle_mass = _sampled_particle_masses_msun_h(
            files, 1, fields["Masses"], len(pos), scale_factor, hubble_param,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not convert PartType1 masses to Msun/h: " + str(exc)
        return result
    sample_weights = np.asarray(sample_weights, dtype=float)
    if (sample_weights.shape != (len(pos),) or not np.all(np.isfinite(sample_weights))
            or np.any(sample_weights <= 0)):
        result["reason"] = "type-1 inverse-inclusion sample weights are invalid or misaligned"
        return result
    total_particle_count = 0
    try:
        for path in files:
            with h5py.File(path, "r") as h5:
                counts = np.asarray(h5["Header"].attrs.get("NumPart_ThisFile", []))
                if counts.shape != (6,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
                    result["reason"] = "snapshot Header/NumPart_ThisFile is missing or invalid"
                    return result
                total_particle_count += int(counts[1])
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = "could not read type-1 particle counts: " + str(exc)
        return result
    if not np.isclose(np.sum(sample_weights), total_particle_count, rtol=1.0e-12, atol=1.0e-12):
        result["reason"] = "type-1 profile sample weights do not recover the snapshot particle count"
        return result
    pos = units_to_mpc_h(pos, box)
    position_valid = np.all(np.isfinite(pos), axis=1)
    profiles = []
    skipped_low_count = 0
    for row in order:
        displacement = periodic_delta(pos, pos_h[row], box)
        radius = np.linalg.norm(displacement, axis=1)
        selected = position_valid & (radius < rad_h[row])
        sampled_count = int(np.count_nonzero(selected))
        if sampled_count < minimum_particles:
            skipped_low_count += 1
            continue
        weights = particle_mass[selected] * sample_weights[selected]
        components = _shape_tensor_components(displacement[selected], weights=weights)
        if components is None:
            continue
        eigenvalues, eigenvectors, metrics = components
        profiles.append({
            "halo_catalogue_row": int(row), "mass_msun_h": float(mass_h[row]),
            "center_group_pos_comoving_mpc_h": pos_h[row].copy(),
            "r200c_comoving_mpc_h": float(rad_h[row]),
            "tensor_eigenvalues_mpc_h2": eigenvalues,
            "principal_axis_eigenvectors_columns": eigenvectors,
            "b_over_a": metrics[0], "c_over_a": metrics[1], "triaxiality": metrics[2],
            "sampled_particle_count": sampled_count,
            "estimated_particle_count": float(np.sum(sample_weights[selected])),
            "estimated_mass_msun_h": float(np.sum(weights)),
            "tensor_definition": "mass quadrupole sum(m_i x_i x_j)/sum(m_i), with particle masses multiplied by inverse-inclusion sampling weights; offsets are from GroupPos",
        })
    result.update({
        "sampled_particle_count": int(len(pos)), "snapshot_particle_count": int(total_particle_count),
        "sampled_particle_weight_sum": float(np.sum(sample_weights)),
        "invalid_particle_position_count": int(np.count_nonzero(~position_valid)),
        "selected_halo_catalogue_rows": [int(row) for row in order],
        "skipped_low_particle_count": int(skipped_low_count),
        "mass_definition": "GADGET Group_M_Crit200 (M200c)",
        "center_definition": "GADGET GroupPos; for this SUBFIND-enabled project configuration, the primary subhalo potential minimum",
        "particle_selection": "all PartType1 snapshot particles within a spherical R200c aperture around GroupPos, regardless of FOF/SUBFIND bound-membership flags",
        "axis_ratio_definition": "b/a=sqrt(lambda_2/lambda_1), c/a=sqrt(lambda_3/lambda_1), for descending mass-quadrupole tensor eigenvalues",
        "triaxiality_definition": "T=(lambda_1-lambda_2)/(lambda_1-lambda_3); NaN when the denominator is numerically degenerate",
        "uncertainty_definition": "no particle-sampling, finite-particle, or halo-to-halo uncertainty is estimated; mass-weighted tensor and sampled/estimated counts are retained",
    })
    if not profiles:
        result["reason"] = "no selected halo has the minimum number of finite-position PartType1 particles within periodic-safe R200c"
        return result
    result["status"] = "COMPLETE"
    result["profiles"] = profiles
    result["halo_count"] = len(profiles)
    result["population_summary"] = summarize_halo_shapes(profiles)
    masses = np.asarray([profile["mass_msun_h"] for profile in profiles])
    b_over_a = np.asarray([profile["b_over_a"] for profile in profiles])
    c_over_a = np.asarray([profile["c_over_a"] for profile in profiles])
    triaxiality = np.asarray([profile["triaxiality"] for profile in profiles])
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].semilogx(masses, b_over_a, "o", label="b/a")
    axes[0].semilogx(masses, c_over_a, "o", label="c/a")
    for name, label in (("b_over_a", "mean b/a"), ("c_over_a", "mean c/a")):
        axes[0].axhline(result["population_summary"]["metrics"][name]["mean"],
                        linestyle="--", alpha=.5, label=label)
    axes[0].set_ylabel("axis ratio"); axes[0].set_ylim(0.0, 1.05); axes[0].legend()
    valid_t = np.isfinite(triaxiality)
    axes[1].semilogx(masses[valid_t], triaxiality[valid_t], "o")
    axes[1].set_xlabel(r"$M_{200c}\ [M_\odot/h]$"); axes[1].set_ylabel("triaxiality T")
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle("Mass-weighted spherical-aperture shape tensor (top mass-limited sample)")
    savefig(Path(run_dir)/"results"/("halo_shape_triaxiality_%03d.png"%snap))
    return result


# ----------------------------- hydro diagnostics -----------------------------

def require_stored_temperature_kelvin(files, particle_type=0):
    """Require stored Temperature to explicitly declare physical Kelvin on every populated shard.

    A field named Temperature alone does not establish its units. This display
    path accepts unit/units='K' with identity CGS conversion and zero a/h
    exponents; differently scaled fields need an explicit preprocessing step.
    """
    found = False
    for path in files:
        with h5py.File(path, "r") as source:
            group = source.get("PartType%d" % particle_type)
            if group is None or not len(group.get("Coordinates", [])):
                continue
            if "Temperature" not in group:
                raise ValueError("Stored Temperature is missing on a populated shard")
            attrs = group["Temperature"].attrs
            unit = attrs.get("unit", attrs.get("units"))
            if isinstance(unit, bytes):
                unit = unit.decode("ascii", errors="strict")
            factor = float(attrs.get("to_cgs", 1.))
            a_scaling = float(attrs.get("a_scaling", 0.))
            h_scaling = float(attrs.get("h_scaling", 0.))
            if unit != "K" or not np.isclose(factor, 1., rtol=1.e-8, atol=0.) or a_scaling != 0. or h_scaling != 0.:
                raise ValueError("Stored Temperature needs explicit physical Kelvin units and identity conversion")
            found = True
    if not found:
        raise ValueError("No stored Kelvin temperature field was found")
    return 1.


def temperature_from_internal_energy(
    u, electron_abundance=None, gamma=5.0 / 3.0,
    unit_velocity_cm_per_s=None, hydrogen_mass_fraction=0.76,
    internal_energy_to_cgs=None,
):
    """Convert GADGET specific internal energy to K using its stored ionization.

    The GADGET build in this repository uses GAMMA=5/3 and
    HYDROGEN_MASSFRAC=0.76.  UnitVelocity_in_cm_per_s comes from the snapshot's
    Parameters group.  None is returned when the required particle or unit
    data is unavailable; invalid particle rows become NaN.
    """
    if electron_abundance is None or (unit_velocity_cm_per_s is None and internal_energy_to_cgs is None):
        return None
    gamma = float(gamma)
    energy_unit = (
        float(internal_energy_to_cgs) if internal_energy_to_cgs is not None
        else float(unit_velocity_cm_per_s) ** 2
    )
    hydrogen_fraction = float(hydrogen_mass_fraction)
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    if not np.isfinite(energy_unit) or energy_unit <= 0:
        raise ValueError("internal-energy conversion to erg/g must be finite and positive")
    if not np.isfinite(hydrogen_fraction) or not 0.0 < hydrogen_fraction < 1.0:
        raise ValueError("hydrogen_mass_fraction must be between zero and one")

    energy, electrons = np.broadcast_arrays(
        np.asarray(u, dtype=float), np.asarray(electron_abundance, dtype=float)
    )
    helium_to_hydrogen = (1.0 - hydrogen_fraction) / (4.0 * hydrogen_fraction)
    valid = np.isfinite(energy) & (energy >= 0.0) & np.isfinite(electrons) & (electrons >= 0.0)
    mean_molecular_weight = np.full(energy.shape, np.nan, dtype=float)
    np.divide(
        1.0 + 4.0 * helium_to_hydrogen,
        1.0 + helium_to_hydrogen + electrons,
        out=mean_molecular_weight,
        where=valid,
    )
    temperature = np.full(energy.shape, np.nan, dtype=float)
    temperature[valid] = (
        (gamma - 1.0) * energy[valid] * energy_unit
        * 1.67262192369e-24 / 1.380649e-16 * mean_molecular_weight[valid]
    )
    return temperature


def _snapshot_parameter(files, name):
    """Read one scalar Parameters attribute and reject inconsistent shards."""
    values = []
    missing = 0
    for path in files:
        with h5py.File(path, "r") as h5:
            parameters = h5.get("Parameters")
            if parameters is None or name not in parameters.attrs:
                missing += 1
                continue
            value = float(parameters.attrs[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Snapshot parameter %s must be finite and positive" % name)
            values.append(value)
    if missing and values:
        raise ValueError("Snapshot shards do not all contain Parameters/%s" % name)
    if not values:
        return None
    if any(not np.isclose(value, values[0], rtol=1.0e-10, atol=0.0) for value in values[1:]):
        raise ValueError("Snapshot shards have inconsistent Parameters/%s" % name)
    return values[0]


def _snapshot_header_value(files, name):
    """Read a consistent scalar Header attribute from every snapshot shard."""
    values = []
    for path in files:
        with h5py.File(path, "r") as h5:
            header = h5.get("Header")
            if header is None or name not in header.attrs:
                continue
            value = np.asarray(header.attrs[name])
            if value.size != 1:
                raise ValueError("Snapshot Header/%s must be scalar" % name)
            values.append(float(value.reshape(-1)[0]))
    if not values:
        return None
    if any(not np.isfinite(value) or not np.isclose(value, values[0], rtol=1.0e-10, atol=0.0) for value in values[1:]):
        raise ValueError("Snapshot shards have inconsistent or invalid Header/%s" % name)
    return values[0]


def _snapshot_dataset_unit_attributes(files, ptype, field):
    """Return identical GADGET-4 unit-conversion attributes for a field."""
    required = ["to_cgs", "a_scaling", "h_scaling"]
    records = []
    nonempty_shards = 0
    missing_field = 0
    incomplete_metadata = False
    for path in files:
        with h5py.File(path, "r") as h5:
            group = h5.get("PartType%d" % int(ptype))
            if group is None or "Coordinates" not in group or len(group["Coordinates"]) == 0:
                continue
            nonempty_shards += 1
            if field not in group:
                missing_field += 1
                continue
            dataset = group[field]
            attrs = dataset.attrs
            if any(key not in attrs for key in required):
                incomplete_metadata = True
                continue
            records.append({key: float(attrs[key]) for key in required})
    if not records:
        return None
    if missing_field and missing_field < nonempty_shards:
        raise ValueError("%s is not present in every non-empty snapshot shard" % field)
    if incomplete_metadata:
        return None
    baseline = records[0]
    if any(any(not np.isclose(row[key], baseline[key], rtol=1.0e-12, atol=0.0) for key in required) for row in records[1:]):
        raise ValueError("Snapshot shards have inconsistent unit conversions for PartType%d/%s" % (ptype, field))
    if any(not np.isfinite(value) for value in baseline.values()) or baseline["to_cgs"] <= 0:
        raise ValueError("PartType%d/%s has invalid unit conversion metadata" % (ptype, field))
    return baseline


def _snapshot_group_count(path, ptype):
    with h5py.File(path, "r") as h5:
        return len(h5["PartType%d/Coordinates" % int(ptype)]) if "PartType%d/Coordinates" % int(ptype) in h5 else 0


def _dataset_to_cgs_factor(unit_attributes, scale_factor, hubble_param):
    return dataset_to_cgs_factor(unit_attributes,scale_factor,hubble_param)


def _project_mass_weighted_gas_field(positions_kpc_h, values, masses, sample_weights, box_mpc_h, grid_size):
    """Project a sampled intensive gas field with inverse-inclusion and mass weights."""
    positions = convert_pinned_native(positions_kpc_h,'position_comoving_mpc_h')
    values = np.asarray(values, dtype=float)
    masses = np.asarray(masses, dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)
    if positions.shape != (len(values), 3) or masses.shape != values.shape or sample_weights.shape != values.shape:
        raise ValueError("gas map positions, values, masses, and sample weights do not align")
    good = (
        np.all(np.isfinite(positions), axis=1) & np.isfinite(values)
        & np.isfinite(masses) & (masses > 0)
        & np.isfinite(sample_weights) & (sample_weights > 0)
    )
    if not np.any(good):
        raise ValueError("no finite positive-mass gas samples are available")
    xy = np.mod(positions[good, :2], float(box_mpc_h))
    weights = masses[good] * sample_weights[good]
    numerator, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=int(grid_size),
        range=((0.0, float(box_mpc_h)), (0.0, float(box_mpc_h))),
        weights=weights * values[good],
    )
    denominator, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=int(grid_size),
        range=((0.0, float(box_mpc_h)), (0.0, float(box_mpc_h))),
        weights=weights,
    )
    image = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return image, int(np.count_nonzero(good))


def hydro_field_maps(run_dir, cfg, snap, files, particle_fields, sample_weights):
    """Render maps of GADGET's saved divergence, curl, and cooling/heating fields.

    Velocity gradients use saved a/h/CGS metadata when available. Older fields
    and the signed CoolingHeatingEnergy routine output explicitly retain their
    native code representation when physical normalization is unavailable.
    """
    result = {"status": "SKIPPED", "reason": None, "fields": {}}
    coordinates = particle_fields.get("Coordinates")
    if coordinates is None:
        result["reason"] = "PartType0/Coordinates is unavailable"
        return result
    simulation_cfg = cfg.get("simulation", {})
    try:
        box = float(simulation_cfg["box_mpc_h"])
    except (KeyError, TypeError, ValueError):
        result["reason"] = "simulation.box_mpc_h is unavailable"
        return result
    if not np.isfinite(box) or box <= 0:
        result["reason"] = "simulation.box_mpc_h must be finite and positive"
        return result
    grid_size = int(cfg.get("visualization", {}).get("hydro_map_grid", 256))
    masses = particle_fields.get("Masses")
    if masses is None:
        mass_table = None
        for path in files:
            with h5py.File(path, "r") as h5:
                if "Header" in h5 and "MassTable" in h5["Header"].attrs:
                    candidate = np.asarray(h5["Header"].attrs["MassTable"], dtype=float)
                    if candidate.shape == (6,) and candidate[0] > 0:
                        mass_table = float(candidate[0])
                        break
        if mass_table is None:
            result["reason"] = "PartType0/ Masses and a positive Header/MassTable[0] are unavailable"
            return result
        masses = np.full(len(coordinates), mass_table, dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.shape != (len(coordinates),):
        result["reason"] = "gas inverse-inclusion weights do not align with coordinates"
        return result

    scalar_fields = [
        ("VelocityDivergence", "compression", lambda values: -np.asarray(values, dtype=float),
         "-VelocityDivergence [snapshot code units]"),
        ("CoolingHeatingEnergy", "cooling_heating", lambda values: np.asarray(values, dtype=float),
         "CoolingHeatingEnergy [snapshot code units; native sign]"),
    ]
    plots = Path(run_dir) / "results"
    plots.mkdir(parents=True, exist_ok=True)
    a=_snapshot_header_value(files,'Time')
    h=_snapshot_header_value(files,'HubbleParam')
    if h is None:h=_snapshot_parameter(files,'HubbleParam')
    def field_conversion(source_name):
        attributes=_snapshot_dataset_unit_attributes(files,0,source_name)
        if source_name in ('VelocityDivergence','VelocityCurl') and attributes is not None:
            factor=_dataset_to_cgs_factor(attributes,a,h)
            return factor,'s^-1',attributes
        return 1.,'native '+source_name+' routine units (physical normalization unavailable)',attributes
    for source_name, output_name, transform, colorbar in scalar_fields:
        values = particle_fields.get(source_name)
        if values is None:
            result["fields"][output_name] = {"status": "SKIPPED", "reason": "PartType0/%s is unavailable" % source_name}
            continue
        factor,units,unit_attributes=field_conversion(source_name)
        values = transform(values)*factor
        colorbar=('-' if output_name=='compression' else '')+source_name+' ['+units+']'
        try:
            image, valid_count = _project_mass_weighted_gas_field(
                coordinates, values, masses, sample_weights, box, grid_size,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            result["fields"][output_name] = {"status": "SKIPPED", "reason": str(exc)}
            continue
        finite = image[np.isfinite(image)]
        if not finite.size:
            result["fields"][output_name] = {"status": "SKIPPED", "reason": "no finite projected pixels"}
            continue
        limit = float(np.percentile(np.abs(finite), 99.5))
        if not np.isfinite(limit) or limit <= 0:
            limit = max(float(np.max(np.abs(finite))), 1.0e-12)
        fig, ax = plt.subplots(figsize=(8, 7))
        mesh = ax.imshow(
            image.T, origin="lower", extent=(0.0, box, 0.0, box),
            cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest",
        )
        ax.grid(False)
        ax.set_xlabel("x [Mpc/h]"); ax.set_ylabel("y [Mpc/h]")
        if output_name == "compression":
            ax.set_title("Gas compression diagnostic — z=%.4g" % redshift_from_header(read_snapshot_header(run_dir, snap)[0]))
        else:
            ax.set_title("Signed gas cooling/heating routine field — snapshot %03d" % snap)
        fig.colorbar(mesh, ax=ax, label=colorbar)
        path = plots / ("hydro_%s_xy_%03d.png" % (output_name, snap))
        export_figure(fig, path, metadata={
            "snapshot": int(snap), "redshift": float(redshift_from_header(read_snapshot_header(run_dir, snap)[0])),
            "diagnostic": output_name, "units": units,
        })
        result["fields"][output_name] = {
            "status": "COMPLETE", "source_field": source_name,
            "valid_particle_count": valid_count, "grid_size": grid_size,
            "projection": "mass-weighted periodic xy pixel mean",
            "units": units,"source_unit_attributes":unit_attributes,"native_to_output_factor":factor,
            "figure": path.name, "pixel_values": image,
            "color_range_symmetric": [-limit, limit],
        }

    curl = particle_fields.get("VelocityCurl")
    if curl is None:
        result["fields"]["vorticity"] = {"status": "SKIPPED", "reason": "PartType0/VelocityCurl is unavailable"}
    else:
        factor,units,unit_attributes=field_conversion('VelocityCurl')
        curl = np.asarray(curl, dtype=float)*factor
        if curl.ndim == 2 and curl.shape[1] == 3:
            vorticity = np.linalg.norm(curl, axis=1)
        elif curl.ndim == 1:
            vorticity = np.abs(curl)
        else:
            result["fields"]["vorticity"] = {"status": "SKIPPED", "reason": "PartType0/VelocityCurl has an unsupported shape"}
            vorticity = None
        if vorticity is not None:
            try:
                image, valid_count = _project_mass_weighted_gas_field(
                    coordinates, vorticity, masses, sample_weights, box, grid_size,
                )
            except (ValueError, TypeError, OverflowError) as exc:
                result["fields"]["vorticity"] = {"status": "SKIPPED", "reason": str(exc)}
            else:
                image = np.where(image > 0, np.log10(image), np.nan)
                finite = image[np.isfinite(image)]
                if not finite.size:
                    result["fields"]["vorticity"] = {"status": "SKIPPED", "reason": "no positive finite projected curl values"}
                else:
                    low, high = np.percentile(finite, [0.5, 99.5])
                    fig, ax = plt.subplots(figsize=(8, 7))
                    mesh = ax.imshow(
                        image.T, origin="lower", extent=(0.0, box, 0.0, box),
                        cmap="magma", vmin=float(low), vmax=float(high), interpolation="nearest",
                    )
                    ax.grid(False)
                    ax.set_xlabel("x [Mpc/h]"); ax.set_ylabel("y [Mpc/h]")
                    ax.set_title("Gas vorticity magnitude — snapshot %03d" % snap)
                    fig.colorbar(mesh, ax=ax, label="log10(|VelocityCurl| / ("+units+"))")
                    path = plots / ("hydro_vorticity_xy_%03d.png" % snap)
                    export_figure(fig, path, metadata={
                        "snapshot": int(snap), "diagnostic": "vorticity magnitude",
                        "units": units,
                    })
                    result["fields"]["vorticity"] = {
                        "status": "COMPLETE", "source_field": "VelocityCurl",
                        "valid_particle_count": valid_count, "grid_size": grid_size,
                        "projection": "mass-weighted periodic xy pixel mean of curl magnitude",
                        "units": units,"source_unit_attributes":unit_attributes,"native_to_output_factor":factor,
                        "figure": path.name, "pixel_values_log10": image,
                        "color_range": [float(low), float(high)],
                    }
    if any(item.get("status") == "COMPLETE" for item in result["fields"].values()):
        result["status"] = "COMPLETE"
    else:
        result["reason"] = "none of the requested diagnostic fields could be projected"
    return result


def hydro_diagnostics(run_dir, cfg, snap, files):
    particle_fields, sample_weights = read_particle_fields(
        files, 0,
        ["Coordinates", "Masses", "Density", "Temperature", "InternalEnergy", "ElectronAbundance",
         "StarFormationRate", "VelocityDivergence", "VelocityCurl", "CoolingHeatingEnergy", "SmoothingLength"],
        max_points=2_000_000, return_sample_weights=True,
    )
    rho_code = particle_fields["Density"]
    temp = particle_fields["Temperature"]
    sfr_sample = particle_fields["StarFormationRate"]
    sfr = summarize_particle_field(files, 0, "StarFormationRate", nonnegative=True)
    result = {}
    scale_factor = _snapshot_header_value(files, "Time")
    hubble_param = _snapshot_header_value(files, "HubbleParam")
    if hubble_param is None:
        hubble_param = _snapshot_parameter(files, "HubbleParam")
    if scale_factor is None or scale_factor <= 0:
        result["gas_density_status"] = "SKIPPED"
        result["gas_density_reason"] = "Snapshot Header/Time is missing or invalid"
    if hubble_param is None or hubble_param <= 0:
        result["gas_density_status"] = "SKIPPED"
        result["gas_density_reason"] = "Snapshot HubbleParam is missing or invalid"

    if rho_code is None:
        result["gas_density_status"] = "SKIPPED"
        result["gas_density_reason"] = "PartType0/Density is unavailable"

    n_h_cm3 = None
    phase_density = None
    phase_density_units = None
    if rho_code is not None and scale_factor is not None and hubble_param is not None:
        try:
            density_units = _snapshot_dataset_unit_attributes(files, 0, "Density")
        except ValueError as exc:
            density_units = None
            result["gas_density_reason"] = str(exc)
        if density_units is None:
            result["gas_density_status"] = "SKIPPED"
            result.setdefault("gas_density_reason", "PartType0/Density unit metadata is unavailable")
        else:
            rho_to_cgs = _dataset_to_cgs_factor(density_units, scale_factor, hubble_param)
            if not np.isfinite(rho_to_cgs) or rho_to_cgs <= 0:
                result["gas_density_status"] = "SKIPPED"
                result["gas_density_reason"] = "PartType0/Density has an invalid physical-unit conversion"
            else:
                # GADGET's HydrogenMassFrac is fixed to 0.76 in this source build.
                n_h_cm3 = np.asarray(rho_code, dtype=float) * rho_to_cgs * 0.76 / 1.67262192369e-24
                phase_density = n_h_cm3
                phase_density_units = "proper n_H [cm^-3]"
                result["gas_density_status"] = "COMPLETE"
                result["gas_density_quantity"] = "proper hydrogen-nucleus number density n_H in cm^-3"

    temperature_source = "stored_temperature" if temp is not None else None
    phase_temp = None
    phase_temperature_units = None
    if temp is None:
        energy = particle_fields["InternalEnergy"]
        electrons = particle_fields["ElectronAbundance"]
        if energy is None:
            result["temperature_status"] = "SKIPPED"
            result["temperature_reason"] = "Temperature and InternalEnergy are unavailable"
        elif electrons is None:
            result["temperature_status"] = "SKIPPED"
            result["temperature_reason"] = "Temperature is unavailable and ElectronAbundance is required to derive it"
        else:
            try:
                energy_units = _snapshot_dataset_unit_attributes(files, 0, "InternalEnergy")
            except ValueError as exc:
                energy_units = None
                energy_unit_error = str(exc)
            else:
                energy_unit_error = None
            if bool(cfg.get("hydro", {}).get("star_formation", False)) and sfr_sample is None:
                result["temperature_status"] = "SKIPPED"
                result["temperature_reason"] = "Cannot identify star-forming gas without StarFormationRate"
            elif energy_units is not None and scale_factor is not None and hubble_param is not None:
                energy_to_cgs = _dataset_to_cgs_factor(energy_units, scale_factor, hubble_param)
                temp = temperature_from_internal_energy(
                    energy, electrons, internal_energy_to_cgs=energy_to_cgs,
                )
            elif energy_units is not None and energy_units["a_scaling"] == 0 and energy_units["h_scaling"] == 0:
                # A standalone GADGET-specific-energy unit has no a/h scaling;
                # these attributes are enough even when the snapshot header is
                # incomplete. Do not assume this for nonzero scale exponents.
                temp = temperature_from_internal_energy(
                    energy, electrons, internal_energy_to_cgs=energy_units["to_cgs"],
                )
            elif energy_units is None:
                try:
                    velocity_unit = _snapshot_parameter(files, "UnitVelocity_in_cm_per_s")
                except ValueError as exc:
                    velocity_unit = None
                    energy_unit_error = str(exc)
                if velocity_unit is not None:
                    temp = temperature_from_internal_energy(
                        energy, electrons, unit_velocity_cm_per_s=velocity_unit,
                    )
                    result["temperature_conversion_fallback"] = (
                        "Parameters/UnitVelocity_in_cm_per_s squared; assumes no additional scale-factor or h conversion"
                    )
                else:
                    temp = None
            else:
                temp = None
            if temp is None:
                if result.get("temperature_status") != "SKIPPED":
                    result["temperature_status"] = "SKIPPED"
                    result["temperature_reason"] = energy_unit_error or "Temperature is unavailable and InternalEnergy unit metadata is incomplete"
            else:
                phase_temp = np.asarray(temp, dtype=float)
                phase_temperature_units = "Kelvin"
                if sfr_sample is not None:
                    non_star_forming = np.isfinite(sfr_sample) & (sfr_sample == 0.0)
                    result["temperature_excluded_star_forming_count"] = int(
                        np.count_nonzero(np.isfinite(sfr_sample) & (sfr_sample > 0.0))
                    )
                    result["temperature_excluded_invalid_sfr_count"] = int(
                        np.count_nonzero(~np.isfinite(sfr_sample) | (sfr_sample < 0.0))
                    )
                    temp = np.where(non_star_forming, temp, np.nan)
                    phase_temp = temp
                    temperature_source = "internal_energy_derived_non_star_forming"
                else:
                    temperature_source = "internal_energy_derived"
                result["temperature_status"] = "COMPLETE"
    else:
        try:
            units = _snapshot_dataset_unit_attributes(files, 0, "Temperature")
            temperature_unit_error = None
        except ValueError as exc:
            units = None
            temperature_unit_error = str(exc)
        unit_name = None
        for path in files:
            with h5py.File(path, "r") as h5:
                if "PartType0/Temperature" in h5:
                    attrs = h5["PartType0/Temperature"].attrs
                    unit_name = attrs.get("unit", attrs.get("units", unit_name))
        if units is None or scale_factor is None or hubble_param is None or unit_name not in ("K", b"K"):
            temp = None
            temperature_source = None
            result["temperature_status"] = "SKIPPED"
            result["temperature_reason"] = temperature_unit_error or "Stored Temperature lacks explicit Kelvin unit metadata"
        else:
            temp = np.asarray(temp, dtype=float) * _dataset_to_cgs_factor(units, scale_factor, hubble_param)
            phase_temp = temp
            phase_temperature_units = "Kelvin"
            result["temperature_status"] = "COMPLETE"
            if sfr_sample is not None:
                non_star_forming = np.isfinite(sfr_sample) & (sfr_sample == 0.0)
                result["temperature_excluded_star_forming_count"] = int(
                    np.count_nonzero(np.isfinite(sfr_sample) & (sfr_sample > 0.0))
                )
                result["temperature_excluded_invalid_sfr_count"] = int(
                    np.count_nonzero(~np.isfinite(sfr_sample) | (sfr_sample < 0.0))
                )
                temp = np.where(non_star_forming, temp, np.nan)
                phase_temp = temp
    if temp is not None and sfr_sample is None and cfg.get("hydro", {}).get("star_formation", False):
        temp = phase_temp = None
        result["temperature_status"] = "SKIPPED"
        result["temperature_reason"] = "Physical gas temperature requires StarFormationRate to exclude effective-EOS gas"
    if temperature_source is not None:
        result["temperature_source"] = temperature_source
    if n_h_cm3 is not None:
        good_rho = np.isfinite(n_h_cm3) & (n_h_cm3 > 0)
        if np.any(good_rho):
            rho_log = np.log10(n_h_cm3[good_rho])
            rho_pdf, rho_edges = histogram_pdf(rho_log, 100)
            plt.figure(figsize=(8, 6)); plt.stairs(rho_pdf, rho_edges)
            plt.xlabel(r"$\log_{10} n_H$ [cm$^{-3}$; proper]"); plt.ylabel("probability density per dex")
            savefig(Path(run_dir) / "results" / ("gas_density_distribution_%03d.png" % snap))
            result["gas_density_log10_nH_cm3_bin_edges"] = rho_edges
            result["gas_density_probability_density"] = rho_pdf
    if temp is not None:
        good_temp = np.isfinite(temp) & (temp > 0)
        if np.any(good_temp):
            temp_log = np.log10(temp[good_temp])
            temp_pdf, temp_edges = histogram_pdf(temp_log, 100)
            plt.figure(figsize=(8, 6)); plt.stairs(temp_pdf, temp_edges)
            temp_label = (
                r"$\log_{10} T_{u}$ [K; derived, non-star-forming gas]"
                if temperature_source == "internal_energy_derived_non_star_forming"
                else r"$\log_{10} T_{u}$ [K; derived]"
                if temperature_source == "internal_energy_derived"
                else r"$\log_{10} T$ [K; stored]"
            )
            plt.xlabel(temp_label); plt.ylabel("probability density")
            savefig(Path(run_dir) / "results" / ("gas_temperature_distribution_%03d.png" % snap))
            result["temperature_log10_bin_edges"] = temp_edges
            result["temperature_probability_density"] = temp_pdf
        result["temperature_valid_sample_count"] = int(np.count_nonzero(good_temp))
        if not np.any(good_temp) and result.get("temperature_status") == "COMPLETE":
            result["temperature_status"] = "NO_VALID_VALUES"
            result["temperature_reason"] = "No positive finite gas temperatures remain after quality filters"
    if phase_density is not None and phase_temp is not None and len(phase_density) == len(phase_temp):
        good = (
            np.isfinite(phase_density) & (phase_density > 0)
            & np.isfinite(phase_temp) & (phase_temp > 0)
        )
        if np.any(good):
            rho_log = np.log10(phase_density[good]); temp_log = np.log10(phase_temp[good])
            phase, rho_edges, temp_edges = np.histogram2d(
                rho_log, temp_log, bins=(stable_histogram_edges(rho_log, 100),
                                       stable_histogram_edges(temp_log, 100)),
            )
            phase_estimate, _, _ = np.histogram2d(
                rho_log, temp_log, bins=(rho_edges, temp_edges),
                weights=np.asarray(sample_weights, dtype=float)[good],
            )
            displayed = np.ma.masked_where(phase_estimate.T <= 0, np.log10(np.maximum(phase_estimate.T, 1)))
            plt.figure(figsize=(8, 6)); mesh = plt.pcolormesh(rho_edges, temp_edges, displayed, shading="auto")
            plt.colorbar(mesh, label=r"$\log_{10}$ gas-particle count")
            plt.xlabel(r"$\log_{10} n_H$ [cm$^{-3}$; proper]" if phase_density_units == "proper n_H [cm^-3]" else r"$\log_{10}$ Density [snapshot units]")
            plt.ylabel(r"$\log_{10} T$ [K]" if phase_temperature_units == "Kelvin" else r"$\log_{10}$ Temperature [snapshot units]")
            savefig(Path(run_dir) / "results" / ("gas_phase_diagram_%03d.png" % snap))
            result["phase_log10_density_bin_edges"] = rho_edges
            result["phase_log10_temperature_bin_edges"] = temp_edges
            result["phase_density_units"] = phase_density_units
            result["phase_temperature_units"] = phase_temperature_units
            result["phase_particle_count_2d"] = phase.astype(np.uint64)
            result["phase_particle_count_estimate_2d"] = phase_estimate.astype(np.float64)
    if sfr is not None:
        result["sfr_total_snapshot_units"] = float(sfr["sum"])
        result["sfr_total_raw_code_value"] = float(sfr["sum"])
        result["sfr_particle_count"] = int(sfr["particle_count"])
        result["sfr_finite_count"] = int(sfr["finite_count"])
        result["sfr_invalid_value_count"] = int(sfr["invalid_count"])
        result["sfr_negative_value_count"] = int(sfr["below_zero_count"])
        result["sfr_summed_value_count"] = int(sfr["included_count"])
        result["sfr_status"] = (
            "PARTIAL_NONFINITE" if sfr["invalid_count"] else
            "PARTIAL_NEGATIVE" if sfr["below_zero_count"] else "COMPLETE"
        )
        result["sfr_quantity"] = "raw StarFormationRate dataset sum; see physical status and Msun/yr value"
        try:
            sfr_units = _snapshot_dataset_unit_attributes(files, 0, "StarFormationRate")
            sfr_unit_error = None
        except ValueError as exc:
            sfr_units = None
            sfr_unit_error = str(exc)
        sfr_factor = (
            _dataset_to_cgs_factor(sfr_units, scale_factor, hubble_param)
            if sfr_units is not None and scale_factor is not None and hubble_param is not None else None
        )
        msun_per_year_cgs = 1.98847e33 / 31557600.0
        if sfr_factor is None or not np.isfinite(sfr_factor) or sfr_factor <= 0:
            result["sfr_physical_status"] = "UNAVAILABLE"
            result["sfr_physical_reason"] = sfr_unit_error or "StarFormationRate lacks valid GADGET unit metadata"
        else:
            result["sfr_total_msun_per_year"] = sfr["sum"] * sfr_factor / msun_per_year_cgs
            result["sfr_physical_status"] = result["sfr_status"]
            result["sfr_physical_quantity"] = "sum of instantaneous gas-particle rates in Msun/yr"
            result["sfr_dataset_to_cgs_factor"] = sfr_factor
    result["hydro_field_maps"] = hydro_field_maps(
        run_dir, cfg, snap, files, particle_fields, sample_weights,
    )
    return result


def baryon_fractions(run_dir, snap):
    """Divide measured group gas/star mass by total group mass; the ratio is dimensionless."""
    gm = group_field(run_dir, snap, ["GroupMass"])
    gmt = group_field(run_dir, snap, ["GroupMassType"])
    if gm is None or gmt is None:
        return None
    total = mass_to_msun_h(gm)
    mt = np.asarray(gmt, float)
    # GroupMassType uses the same GADGET internal mass unit as GroupMass.
    mt = convert_pinned_native(mt,'mass_msun_h')
    gas = mt[:,0] if mt.shape[1] > 0 else np.zeros(len(total))
    star = mt[:,4] if mt.shape[1] > 4 else np.zeros(len(total))
    good = np.isfinite(total) & (total > 0)
    fg = np.divide(gas, total, out=np.full_like(total, np.nan), where=good)
    fs = np.divide(star, total, out=np.full_like(total, np.nan), where=good)
    plt.figure(figsize=(8,6)); plt.semilogx(total[good], fg[good], ".", alpha=.35, label="gas"); plt.semilogx(total[good], fs[good], ".", alpha=.35, label="stars")
    plt.xlabel(r"$M_{halo}\ [M_\odot/h]$"); plt.ylabel("mass fraction"); plt.legend()
    return total, fg, fs


def main_progenitor_histories(run_dir, cfg, snapshot_summary, *, final_halo_id=None):
    """Build main-progenitor tracks from Rockstar/Consistent-Trees descendant links."""
    minimum = int(cfg["analysis"]["hmf_min_particles"])
    if minimum < 0:
        raise ValueError("analysis.hmf_min_particles must be zero or greater")
    from core.data import rockstar_directory
    rockstar_dir = rockstar_directory(run_dir)
    consistent_catalogues = []
    for path in sorted(rockstar_dir.rglob("hlist_*.list")):
        table = rockstar_table_from_path(path)
        if table is None or "scale" not in table:
            continue
        scales = np.asarray(table["scale"], dtype=float)
        scales = scales[np.isfinite(scales) & (scales > 0)]
        if len(scales):
            consistent_catalogues.append((float(np.median(scales)), table))
    catalogs = []
    for snap_text, item in snapshot_summary.items():
        snap = int(snap_text)
        redshift = float(item["redshift"])
        if not np.isfinite(redshift) or redshift < 0:
            continue
        target_scale = 1.0 / (1.0 + redshift)
        tree_match = None
        if consistent_catalogues:
            nearest = min(consistent_catalogues, key=lambda entry: abs(entry[0] - target_scale))
            if abs(nearest[0] - target_scale) <= 1.0e-5 * max(1.0, target_scale):
                tree_match = nearest[1]
        table = tree_match if tree_match is not None else rockstar_table(run_dir, snap)
        if table is None or not all(name in table for name in ["id", "mvir"]):
            continue
        desc_name = "desc_id" if "desc_id" in table else "descid" if "descid" in table else None
        if desc_name is None:
            continue
        desc_scale_name = "desc_scale" if "desc_scale" in table else "descscale" if "descscale" in table else None
        if "num_p" not in table and "np" in table:
            table = dict(table, num_p=table["np"])
        if minimum and "num_p" not in table and "orig_halo_id" in table:
            original = rockstar_table(run_dir, snap)
            original_count = (original.get("num_p", original.get("np"))
                              if original is not None else None)
            if original_count is not None and "id" in original:
                counts_by_id = dict(zip(np.asarray(original["id"], dtype=np.int64),
                                        np.asarray(original_count, dtype=np.int64)))
                # Phantom or unmatched tree rows have no measured particle count.
                # Exclude them from the resolution cut rather than estimating Np.
                counts = np.array([counts_by_id.get(int(i), -1) for i in table["orig_halo_id"]], dtype=np.int64)
                table = dict(table, num_p=counts)
        if minimum and "num_p" not in table:
            return {"status": "SKIPPED", "reason": "No measured Rockstar particle counts can be matched to the tree catalogue"}
        ids = np.asarray(table["id"], dtype=np.int64)
        desc_ids = np.asarray(table[desc_name], dtype=np.int64)
        mass = np.asarray(table["mvir"], dtype=float)
        if len(ids) != len(desc_ids) or len(ids) != len(mass):
            return {"status": "SKIPPED", "reason": "Rockstar ID, descendant, and Mvir columns do not align"}
        if len(np.unique(ids)) != len(ids):
            return {"status": "SKIPPED", "reason": "Rockstar snapshot contains duplicate halo IDs"}
        if minimum:
            count = np.asarray(table["num_p"], dtype=np.int64)
            if count.ndim != 1 or len(count) != len(ids):
                return {"status": "SKIPPED", "reason": "Rockstar ID and num_p columns do not align"}
            particle_keep = count >= minimum
        else:
            particle_keep = np.ones(len(ids), dtype=bool)
        valid = particle_keep & (ids >= 0) & np.isfinite(mass) & (mass > 0)
        if not np.any(valid):
            continue
        desc_scale = np.asarray(table[desc_scale_name] if desc_scale_name else np.full(len(ids), np.nan), dtype=float)
        if desc_scale.ndim != 1 or len(desc_scale) != len(ids):
            return {"status": "SKIPPED", "reason": "Rockstar descendant scale and ID columns do not align"}
        position = None
        if all(name in table for name in ('x','y','z')):
            candidate = np.column_stack([np.asarray(table[name],dtype=float) for name in ('x','y','z')])
            if candidate.shape == (len(ids),3):
                position = candidate
        velocity = None
        if all(name in table for name in ('vx','vy','vz')):
            candidate = np.column_stack([np.asarray(table[name],dtype=float) for name in ('vx','vy','vz')])
            if candidate.shape == (len(ids),3):
                velocity = candidate
        catalogs.append({
            "snapshot": snap, "redshift": redshift, "scale": 1.0 / (1.0 + redshift),
            "ids": ids, "desc_ids": desc_ids, "desc_scale": desc_scale,
            "original_ids": np.asarray(table.get("orig_halo_id", ids), dtype=np.int64),
            "mass": mass, "valid": valid, "position_mpc_h": position, "velocity_peculiar_km_s": velocity,
            "source": "Consistent-Trees hlist" if tree_match is not None else "Rockstar descendant links",
        })
    catalogs.sort(key=lambda row: row["scale"])
    if len(catalogs) < 2:
        return {"status": "SKIPPED", "reason": "fewer than two Rockstar snapshots have descendant links and valid Mvir data"}

    final = catalogs[-1]
    final_indices = np.flatnonzero(final["valid"])
    if not len(final_indices):
        return {"status": "SKIPPED", "reason": "final Rockstar snapshot has no halos above the particle floor"}
    max_histories = int(cfg.get("analysis", {}).get("max_accretion_histories", 12))
    if max_histories <= 0:
        raise ValueError("analysis.max_accretion_histories must be positive")
    if final_halo_id is not None:
        if isinstance(final_halo_id,bool) or not isinstance(final_halo_id,(int,np.integer)) or final_halo_id<0:
            raise ValueError('Final halo ID must be an exact nonnegative integer')
        final_indices=final_indices[final['ids'][final_indices]==final_halo_id]
        if len(final_indices)!=1:
            return {"status":"SKIPPED","reason":"selected final halo ID is unavailable or below the particle floor"}
    else:
        final_indices = final_indices[np.argsort(final["mass"][final_indices])[::-1][:max_histories]]

    histories = []
    for final_index in final_indices:
        current_id = int(final["ids"][final_index])
        track = []
        merger_events = []
        for catalog_index in range(len(catalogs) - 1, -1, -1):
            catalog = catalogs[catalog_index]
            if catalog_index == len(catalogs) - 1:
                candidate = int(final_index)
            else:
                descendants = catalog["desc_ids"]
                matches = np.flatnonzero(catalog["valid"] & (descendants == current_id))
                if np.any(np.isfinite(catalog["desc_scale"][matches])):
                    scale_ok = ~np.isfinite(catalog["desc_scale"][matches]) | np.isclose(
                        catalog["desc_scale"][matches], catalogs[catalog_index + 1]["scale"], rtol=1.0e-5, atol=1.0e-7
                    )
                    matches = matches[scale_ok]
                if not len(matches):
                    break
                candidate = int(matches[np.argmax(catalog["mass"][matches])])
                main_mass = float(catalog["mass"][candidate])
                for secondary_index in matches:
                    if int(secondary_index) == candidate:
                        continue
                    secondary_mass = float(catalog["mass"][secondary_index])
                    if not np.isfinite(secondary_mass) or secondary_mass <= 0 or main_mass <= 0:
                        continue
                    merger_events.append({
                        "redshift": catalogs[catalog_index + 1]["redshift"],
                        "descendant_id": current_id,
                        "main_progenitor_id": int(catalog["ids"][candidate]),
                        "secondary_progenitor_id": int(catalog["ids"][secondary_index]),
                        "main_mass_msun_h": main_mass,
                        "secondary_mass_msun_h": secondary_mass,
                        "mass_ratio": secondary_mass / main_mass,
                    })
            track.append({
                "snapshot": catalog["snapshot"], "redshift": catalog["redshift"],
                "scale_factor": catalog["scale"], "halo_id": int(catalog["ids"][candidate]),
                "original_halo_id": int(catalog["original_ids"][candidate]),
                "mass_msun_h": float(catalog["mass"][candidate]),
            })
            current_id = int(catalog["ids"][candidate])
        track.reverse()
        if len(track) < 2:
            continue

        final_mass = float(track[-1]["mass_msun_h"])
        half_mass = 0.5 * final_mass
        formation_redshift = None
        for earlier, later in zip(track[:-1], track[1:]):
            m0, m1 = earlier["mass_msun_h"], later["mass_msun_h"]
            if m0 < half_mass <= m1:
                log0, log1 = math.log(m0), math.log(m1)
                fraction = (math.log(half_mass) - log0) / (log1 - log0) if log1 != log0 else 1.0
                a_half = earlier["scale_factor"] + fraction * (later["scale_factor"] - earlier["scale_factor"])
                formation_redshift = 1.0 / a_half - 1.0
                break
        histories.append({
            "final_halo_id": int(track[-1]["halo_id"]), "final_mass_msun_h": final_mass,
            "track": track, "formation_redshift": formation_redshift,
            "merger_events": merger_events,
        })
    if not histories:
        return {"status": "SKIPPED", "reason": "no final halos have a two-snapshot main-progenitor link"}
    from core.merger_tree_visual import build_merger_tree
    return {
        "status": "COMPLETE", "histories": histories, "final_snapshot": final["snapshot"],
        "particle_floor": minimum, "source": final["source"],
        "particle_count_source": "Rockstar Np/num_p; tree rows matched through Orig_halo_ID; unmatched rows excluded",
        "_merger_tree": build_merger_tree(
            catalogs, int(np.flatnonzero(final['ids'] == histories[0]['final_halo_id'])[0]),
            box_mpc_h=float(cfg['simulation']['box_mpc_h'])),
    }


def plot_mass_accretion_histories(run_dir, result):
    if not result or result.get("status") != "COMPLETE":
        return
    from core.formation_atlas import export_formation_atlas
    export_formation_atlas(result, Path(run_dir) / 'results')
    plt.figure(figsize=(9, 6))
    for history in result["histories"]:
        track = history["track"]
        scale = np.asarray([point["scale_factor"] for point in track])
        mass = np.asarray([point["mass_msun_h"] for point in track])
        plt.semilogy(scale, mass, marker="o", label="ID %d" % history["final_halo_id"])
    plt.xlabel("scale factor a"); plt.ylabel(r"main-progenitor $M_{vir}\ [M_\odot/h]$")
    plt.title("Main-progenitor mass accretion histories")
    plt.legend(fontsize=7, ncol=2)
    savefig(Path(run_dir) / "results" / "mass_accretion_histories.png")

    formed = [row for row in result["histories"] if row["formation_redshift"] is not None]
    if formed:
        mass = np.asarray([row["final_mass_msun_h"] for row in formed])
        zhalf = np.asarray([row["formation_redshift"] for row in formed])
        plt.figure(figsize=(8, 6)); plt.semilogx(mass, zhalf, "o")
        plt.xlabel(r"final $M_{vir}\ [M_\odot/h]$"); plt.ylabel("half-mass formation redshift")
        savefig(Path(run_dir) / "results" / "formation_redshift.png")


def plot_merger_histories(run_dir, result):
    if not result or result.get("status") != "COMPLETE":
        return
    from core.merger_tree_visual import export_merger_tree
    export_merger_tree(result.get('_merger_tree'), Path(run_dir) / 'results')
    plt.figure(figsize=(9, 6)); have_events = False
    for history in result["histories"]:
        events = history["merger_events"]
        if not events:
            continue
        have_events = True
        redshift = np.asarray([event["redshift"] for event in events])
        ratio = np.asarray([event["mass_ratio"] for event in events])
        plt.scatter(redshift, ratio, s=24, alpha=0.7, label="final halo %d" % history["final_halo_id"])
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1, label="major merger: 1:4")
    plt.yscale("log"); plt.xlabel("descendant redshift"); plt.ylabel("secondary / main progenitor mass")
    plt.title("Main-branch merger events")
    if have_events:
        plt.legend(fontsize=7, ncol=2)
    else:
        plt.gca().text(0.5, 0.5, "No linked secondary progenitors in the selected halos", ha="center", va="center", transform=plt.gca().transAxes)
    savefig(Path(run_dir) / "results" / "halo_merger_mass_ratios.png")


# ----------------------------- time evolution -----------------------------

def plot_time_evolution(run_dir, summary):
    rows = []
    for sid, item in summary["snapshots"].items():
        rare = item.get("rare_halos", {})
        rows.append((float(item["redshift"]), rare.get("maximum_mass_msun_h", np.nan), rare))
    if not rows:
        return
    rows.sort(reverse=True)
    z = np.asarray([x[0] for x in rows]); mmax = np.asarray([x[1] for x in rows], float)
    good = np.isfinite(mmax) & (mmax > 0)
    if np.any(good):
        plt.figure(figsize=(8,6)); plt.semilogy(z[good], mmax[good], marker="o"); plt.gca().invert_xaxis()
        plt.xlabel("redshift"); plt.ylabel(r"maximum halo mass [$M_\odot/h$]")
        savefig(Path(run_dir)/"results"/"maximum_halo_mass_vs_z.png")
    thresholds = load_cfg(run_dir).get("analysis", {}).get("rare_halo_thresholds_msun_h", [])
    plt.figure(figsize=(8,6)); any_line=False
    for t in thresholds:
        tf = float(t)
        y = np.asarray([x[2].get(str(t), np.nan) for x in rows], float)
        if np.any(np.isfinite(y)):
            plt.plot(z, y, marker="o", label=r"$M>%g$" % tf); any_line=True
    if any_line:
        plt.gca().invert_xaxis(); plt.xlabel("redshift"); plt.ylabel("halo count in box"); plt.legend(); savefig(Path(run_dir)/"results"/"rare_halo_abundance_vs_z.png")
    else:
        plt.close()


def create_analysis_report_pdf(results_dir, summary, cfg, source_run=None, model=None):
    """Create a concise run report followed by every top-level analysis figure."""
    from matplotlib.backends.backend_pdf import PdfPages

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "summary.pdf"
    temporary_path = results_dir / (".summary_" + uuid.uuid4().hex + ".tmp.pdf")
    figures = sorted(
        [*results_dir.glob("*.png"), *results_dir.glob("paper_figures/*.png")],
        key=lambda path: path.as_posix().lower(),
    )
    snapshots = summary.get("snapshots", {})
    skipped = summary.get("skipped", {})
    simulation = cfg.get("simulation", {})
    cosmology = cfg.get("cosmology", {})
    manifest = {}
    manifest_path = Path(source_run) / "manifest.json" if source_run else None
    if manifest_path is not None and manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        except (OSError, ValueError):
            manifest = {}
    selected = summary.get("products")
    analysis_scope = ", ".join(selected) if selected else "Full configured analysis"
    model_name = str(model or manifest.get("model") or cosmology.get("model", "unknown")).upper()
    run_name = (Path(source_run).name if source_run else results_dir.parent.name).replace("_", " ")
    box_size = simulation.get("box_mpc_h")
    particle_count = simulation.get("particles_per_dimension")
    resolution = "%s^3 particles per species" % particle_count if particle_count is not None else "Not recorded in config"
    if box_size is not None:
        box_text = "%s Mpc/h" % box_size
    else:
        box_text = "Not recorded in config"
    configured_particle_grid = simulation.get("particles_per_dimension")
    particle_grid_text = ("%s^3" % configured_particle_grid
                          if configured_particle_grid is not None else "Not recorded")
    validation = manifest.get("validation", {})
    validation_lines = []
    if isinstance(validation, dict):
        for name, result in sorted(validation.items()):
            if isinstance(result, dict):
                status = result.get("status", "details recorded")
                reason = result.get("reason")
                validation_lines.append("%s: %s%s" % (
                    name, status, " - " + str(reason) if reason else ""))
            else:
                validation_lines.append("%s: %s" % (name, result))
    if not validation_lines:
        validation_lines.append("No validation results were recorded in the run manifest.")
    validation_page_groups = []
    current_page = []
    used_height = 0.0
    for line in validation_lines:
        wrapped = textwrap.wrap(line, width=102) or [line]
        row_height = 0.035 * len(wrapped) + 0.014
        if current_page and used_height + row_height > 0.78:
            validation_page_groups.append(current_page)
            current_page = []
            used_height = 0.0
        current_page.append((wrapped, row_height))
        used_height += row_height
    if current_page:
        validation_page_groups.append(current_page)
    validation_pages = len(validation_page_groups)

    def draw_page_header(axis, title, page_number, total_pages):
        axis.text(0.07, 0.955, title, color="#183047", fontsize=19, weight="bold",
                  ha="left", va="top", transform=axis.transAxes)
        axis.plot([0.07, 0.93], [0.925, 0.925], color="#1b8a8f", linewidth=2,
                  transform=axis.transAxes, clip_on=False)
        axis.text(0.93, 0.025, "%d / %d" % (page_number, total_pages), color="#64748b",
                  fontsize=8, ha="right", va="bottom", transform=axis.transAxes)

    def save_report_page(pdf, title, page_number, total_pages, render):
        fig, axis = plt.subplots(figsize=(8.5, 11.0))
        axis.set_axis_off()
        draw_page_header(axis, title, page_number, total_pages)
        render(fig, axis)
        pdf.savefig(fig)
        plt.close(fig)

    skip_items = sorted((str(key), str(value)) for key, value in skipped.items())
    skip_page_groups = []
    if not skip_items:
        skip_page_groups.append([])
    else:
        current_page = []
        used_height = 0.0
        for key, reason in skip_items:
            key_lines = textwrap.wrap(key, width=28) or [key]
            reason_lines = textwrap.wrap(reason, width=78) or [reason]
            row_height = 0.035 * max(len(key_lines), len(reason_lines)) + 0.009
            if current_page and used_height + row_height > 0.78:
                skip_page_groups.append(current_page)
                current_page = []
                used_height = 0.0
            current_page.append((key, reason, key_lines, reason_lines, row_height))
            used_height += row_height
        if current_page:
            skip_page_groups.append(current_page)
    skip_pages = len(skip_page_groups)
    limitations = []
    for section in ("dark_matter_model", "dark_energy_approximation", "hydro_model"):
        details = manifest.get(section)
        if isinstance(details, dict):
            for name, value in details.items():
                if "limit" in name.lower() or "approx" in name.lower() or value is False:
                    limitations.append("%s: %s" % (name.replace("_", " "), value))
    warnings = manifest.get("warnings", [])
    if isinstance(warnings, list):
        limitations.extend("Run warning: " + str(item) for item in warnings)
    if not manifest:
        limitations.append("No run manifest was available, so run-specific assumptions and warnings cannot be reported.")
    if not limitations:
        limitations.append("No model limitations or warnings were recorded in the run manifest.")
    limitations = sorted(set(limitations))
    limitation_page_groups = []
    current_page = []
    used_height = 0.0
    for line in limitations:
        wrapped = textwrap.wrap(line, width=103) or [line]
        row_height = 0.035 * len(wrapped) + 0.014
        if current_page and used_height + row_height > 0.78:
            limitation_page_groups.append(current_page)
            current_page = []
            used_height = 0.0
        current_page.append((wrapped, row_height))
        used_height += row_height
    if current_page:
        limitation_page_groups.append(current_page)
    limitation_pages = len(limitation_page_groups)
    total_pages = 2 + validation_pages + limitation_pages + skip_pages + len(figures)
    try:
        with PdfPages(temporary_path) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = "Simulation analysis report - " + run_name
            metadata["Author"] = "DarkMatterSim"
            metadata["Subject"] = "Analysis status and exported figures"
            metadata["Creator"] = "DarkMatterSim analysis pipeline"

            def render_cover(fig, axis):
                axis.text(0.07, 0.86, "SIMULATION ANALYSIS", color="#1b8a8f", fontsize=10,
                          weight="bold", transform=axis.transAxes)
                axis.text(0.07, 0.80, run_name, color="#183047", fontsize=27, weight="bold",
                          transform=axis.transAxes)
                axis.text(0.07, 0.755, "%s model  |  %s snapshots analyzed" % (model_name, len(snapshots)),
                          color="#536579", fontsize=12, transform=axis.transAxes)
                axis.text(0.07, 0.69, "Run overview", color="#183047", fontsize=15, weight="bold",
                          transform=axis.transAxes)
                rows = [
                    ("Analysis scope", analysis_scope),
                    ("Box size", box_text),
                    ("Particle resolution", resolution),
                    ("Analysis figures", str(len(figures))),
                    ("Skipped products", str(len(skip_items))),
                    ("Report created (UTC)", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
                ]
                y = 0.645
                for label, value in rows:
                    axis.text(0.08, y, label, color="#536579", fontsize=10, weight="bold",
                              transform=axis.transAxes)
                    axis.text(0.39, y, value, color="#183047", fontsize=10,
                              transform=axis.transAxes)
                    y -= 0.045
                axis.text(0.07, 0.345, "Cosmology inputs", color="#183047", fontsize=15,
                          weight="bold", transform=axis.transAxes)
                cosmology_rows = [
                    ("H0", cosmology.get("H0")),
                    ("omega_b", cosmology.get("omega_b")),
                    ("omega_cdm", cosmology.get("omega_cdm")),
                    ("A_s", cosmology.get("A_s")),
                    ("n_s", cosmology.get("n_s")),
                ]
                y = 0.30
                for label, value in cosmology_rows:
                    axis.text(0.08, y, label, color="#536579", fontsize=10, weight="bold",
                              transform=axis.transAxes)
                    axis.text(0.39, y, "Not recorded" if value is None else str(value),
                              color="#183047", fontsize=10, transform=axis.transAxes)
                    y -= 0.04
                axis.text(0.07, 0.075,
                          "Numerical analysis products and detailed status are in the companion HDF5 and JSON files. "
                          "This report embeds the PNG figures found in the results folder.",
                          color="#536579", fontsize=9, wrap=True, transform=axis.transAxes)

            save_report_page(pdf, "Run summary", 1, total_pages, render_cover)

            def render_specification(fig, axis):
                axis.text(0.075, 0.88, "Simulation specification", color="#183047", fontsize=13,
                          weight="bold", transform=axis.transAxes)
                snapshot_schedule = manifest.get("snapshot_redshifts_requested")
                if isinstance(snapshot_schedule, list):
                    schedule_text = "%d requested redshifts; first=%s, last=%s" % (
                        len(snapshot_schedule), snapshot_schedule[0] if snapshot_schedule else "none",
                        snapshot_schedule[-1] if snapshot_schedule else "none",
                    )
                else:
                    schedule_text = "Requested schedule not recorded"
                sim_rows = [
                    ("Model / physics", "%s / %s" % (model_name, simulation.get("physics", "unknown"))),
                    ("Box / particle grid", "%s / %s" % (box_text, particle_grid_text)),
                    ("Start redshift / seed", "%s / %s" % (
                        manifest.get("start_redshift", simulation.get("start_redshift", "Not recorded")),
                        manifest.get("random_seed", simulation.get("random_seed", "Not recorded")))),
                    ("Snapshot outputs", schedule_text),
                    ("Halo finder settings", "FOF=%s; SUBFIND=%s; Rockstar=%s" % (
                        cfg.get("halos", {}).get("fof", "Not recorded"), cfg.get("halos", {}).get("subfind", "Not recorded"),
                        cfg.get("halos", {}).get("rockstar", "Not recorded"))),
                    ("HMF settings", "%s masses; minimum %s particles; %s bins" % (
                        cfg.get("analysis", {}).get("hmf_mass_definition", "FOF (default)"),
                        cfg["analysis"]["hmf_min_particles"],
                        cfg.get("analysis", {}).get("mass_bins", "not recorded"))),
                ]
                y = 0.835
                for label, value in sim_rows:
                    axis.text(0.08, y, label, color="#536579", fontsize=9, weight="bold",
                              transform=axis.transAxes)
                    axis.text(0.35, y, value, color="#183047", fontsize=9,
                              transform=axis.transAxes)
                    y -= 0.046

                axis.text(0.075, 0.055,
                          "Validation records and statuses are summarized on the following pages.",
                          color="#64748b", fontsize=8, wrap=True, transform=axis.transAxes)

            save_report_page(pdf, "Simulation setup and validation", 2, total_pages, render_specification)

            for page_index, page_items in enumerate(validation_page_groups):
                def render_validation(fig, axis, page_items=page_items):
                    y = 0.885
                    for wrapped, row_height in page_items:
                        axis.text(0.09, y, "- " + "\n  ".join(wrapped), color="#536579", fontsize=9,
                                  va="top", transform=axis.transAxes)
                        y -= row_height
                    axis.text(0.075, 0.065,
                              "Validation information is read from manifest.json; this analysis report does not rerun those checks.",
                              color="#64748b", fontsize=8, wrap=True, transform=axis.transAxes)

                title = "Recorded run validation"
                if validation_pages > 1:
                    title += " (page %d)" % (page_index + 1)
                save_report_page(pdf, title, 3 + page_index, total_pages, render_validation)

            for page_index, page_items in enumerate(limitation_page_groups):
                def render_limitations(fig, axis, page_items=page_items):
                    y = 0.885
                    for wrapped, row_height in page_items:
                        axis.text(0.09, y, "- " + "\n  ".join(wrapped), color="#536579", fontsize=9,
                                  va="top", transform=axis.transAxes)
                        y -= row_height
                    axis.text(0.075, 0.065,
                              "These notes are only as complete as the run manifest and configured model records.",
                              color="#64748b", fontsize=8, wrap=True, transform=axis.transAxes)

                title = "Known assumptions and limitations"
                if limitation_pages > 1:
                    title += " (page %d)" % (page_index + 1)
                save_report_page(pdf, title, 3 + validation_pages + page_index, total_pages, render_limitations)

            for page_index, page_items in enumerate(skip_page_groups):
                def render_status(fig, axis, page_items=page_items, page_index=page_index):
                    if not page_items:
                        axis.text(0.08, 0.84, "No products were marked skipped or unavailable.",
                                  color="#183047", fontsize=12, transform=axis.transAxes)
                        return
                    y = 0.885
                    for key, reason, key_lines, reason_lines, row_height in page_items:
                        axis.text(0.075, y, "\n".join(key_lines), color="#183047", fontsize=8.5,
                                  weight="bold", va="top", transform=axis.transAxes)
                        axis.text(0.31, y, "\n".join(reason_lines), color="#536579", fontsize=8.5,
                                  va="top", transform=axis.transAxes)
                        y -= row_height
                    if page_index == 0:
                        axis.text(0.075, 0.065,
                                  "A skipped result means its required catalog or validated input was unavailable; "
                                  "see analysis_summary.json for full context.",
                                  color="#64748b", fontsize=8, wrap=True, transform=axis.transAxes)

                title = "Analysis coverage" if skip_pages == 1 else "Analysis coverage (page %d)" % (page_index + 1)
                save_report_page(pdf, title, 3 + validation_pages + limitation_pages + page_index,
                                 total_pages, render_status)

            for index, image_path in enumerate(figures):
                page_number = 3 + validation_pages + limitation_pages + skip_pages + index
                figure_title = image_path.stem.replace("_", " ").title()

                def render_figure(fig, axis, image_path=image_path, figure_title=figure_title):
                    try:
                        image = plt.imread(image_path)
                        image_axis = fig.add_axes([0.08, 0.09, 0.84, 0.79])
                        image_axis.imshow(image)
                        image_axis.set_axis_off()
                    except (OSError, ValueError) as exc:
                        axis.text(0.08, 0.84, "Could not display %s: %s" % (image_path.name, exc),
                                  color="#9b2c2c", fontsize=10, wrap=True, transform=axis.transAxes)

                save_report_page(pdf, figure_title, page_number, total_pages, render_figure)
        os.replace(temporary_path, report_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return report_path


def build_star_formation_history(snapshot_summary):
    """Collect only complete, unit-converted whole-snapshot SFR measurements."""
    rows = []
    omitted = []
    for snapshot_text, item in snapshot_summary.items():
        try:
            snapshot = int(snapshot_text)
            redshift = float(item.get("actual_redshift", item.get("redshift", np.nan)))
        except (TypeError, ValueError, OverflowError):
            omitted.append({"snapshot": str(snapshot_text), "reason": "snapshot ID or actual redshift is invalid"})
            continue
        sfr = item.get("star_formation")
        if not isinstance(sfr, dict):
            omitted.append({"snapshot": snapshot, "reason": "no whole-snapshot star-formation measurement was recorded"})
            continue
        status = str(sfr.get("sfr_physical_status", "UNAVAILABLE"))
        try:
            rate = float(sfr["sfr_total_msun_per_year"])
        except (KeyError, TypeError, ValueError, OverflowError):
            rate = float("nan")
        if (status != "COMPLETE" or not np.isfinite(redshift) or redshift < 0
                or not np.isfinite(rate) or rate < 0):
            reason = sfr.get("sfr_physical_reason") or sfr.get("sfr_status") or status
            omitted.append({"snapshot": snapshot, "reason": str(reason)})
            continue
        rows.append((redshift, snapshot, rate))
    rows.sort(key=lambda row: row[0])
    return {
        "status": "COMPLETE" if rows else "SKIPPED",
        "reason": None if rows else "no snapshot has a complete nonnegative physical SFR measurement",
        "snapshot": [row[1] for row in rows],
        "redshift": [row[0] for row in rows],
        "total_sfr_msun_per_year": [row[2] for row in rows],
        "included_snapshot_count": len(rows),
        "omitted_snapshot_count": len(omitted),
        "omitted_snapshots": omitted,
        "quantity": "whole-box instantaneous sum of gas-particle StarFormationRate values",
        "unit": "Msun/year",
        "selection": "only snapshots with complete particle coverage and valid explicit SFR unit conversion; partial or unavailable snapshots are excluded and listed",
    }


def plot_star_formation_history(run_dir, history):
    """Plot saved star-formation rates in their declared units and at their saved epochs."""
    if not history or history.get("status") != "COMPLETE":
        return None
    redshift = np.asarray(history["redshift"], dtype=float)
    rate = np.asarray(history["total_sfr_msun_per_year"], dtype=float)
    if (redshift.ndim != 1 or rate.shape != redshift.shape or len(rate) == 0
            or np.any(~np.isfinite(redshift)) or np.any(~np.isfinite(rate)) or np.any(rate < 0)):
        raise ValueError("Star-formation history contains invalid redshifts or physical rates")
    plt.figure(figsize=(8, 6))
    if np.all(rate > 0):
        plt.semilogy(redshift, rate, marker="o")
    else:
        plt.plot(redshift, rate, marker="o")
        plt.gca().text(
            0.02, 0.97, "Linear y scale retains zero-rate snapshots",
            transform=plt.gca().transAxes, va="top", fontsize=8,
        )
    plt.gca().invert_xaxis()
    plt.xlabel("actual snapshot redshift")
    plt.ylabel(r"whole-box instantaneous SFR [$M_\odot$/yr]")
    plt.title("Star-formation history (%d complete snapshot%s)" % (
        len(rate), "" if len(rate) == 1 else "s",
    ))
    savefig(Path(run_dir) / "results" / "star_formation_rate_vs_redshift.png")
    return history


def write_star_formation_history_hdf5(parent, history, *, run_dir=None, snapshot_epochs=None, provenance=None):
    """Persist the saved star-formation history and its measurement/unit metadata."""
    group = parent.create_group("star_formation_history")
    group.attrs["status"] = history["status"]
    group.attrs["quantity"] = history["quantity"]
    group.attrs["unit"] = history["unit"]
    group.attrs["selection"] = history["selection"]
    group.attrs["included_snapshot_count"] = history["included_snapshot_count"]
    group.attrs["omitted_snapshot_count"] = history["omitted_snapshot_count"]
    if history["reason"]:
        group.attrs["reason"] = history["reason"]
    if history["snapshot"]:
        group.create_dataset("snapshot", data=history["snapshot"])
        group.create_dataset("redshift", data=history["redshift"])
        group.create_dataset("total_sfr_msun_per_year", data=history["total_sfr_msun_per_year"])
    if history["omitted_snapshots"]:
        omitted = group.create_group("omitted_snapshots")
        omitted.create_dataset(
            "snapshot", data=[str(row["snapshot"]) for row in history["omitted_snapshots"]],
            dtype=h5py.string_dtype("utf-8"),
        )
        omitted.create_dataset(
            "reason", data=[row["reason"] for row in history["omitted_snapshots"]],
            dtype=h5py.string_dtype("utf-8"),
        )
    from core.data import attach_time_series_contract
    from core.cosmology import run_proper_time_clock
    attach_time_series_contract(group,clock=run_proper_time_clock(run_dir) if run_dir is not None else None,
        snapshot_epochs=snapshot_epochs,provenance=provenance)


def _stellar_formation_clock(run_dir):
    """Use the canonical run clock; missing sources are explicit, corruption fails."""
    from core.cosmology import run_proper_time_clock
    scale, times, source = run_proper_time_clock(run_dir)
    return scale, times, source.get('reason')


def _field_present_in_every_star_shard(files, field):
    saw_stars = False
    for path in files:
        with h5py.File(path, "r") as h5:
            group = h5.get("PartType4")
            if group is None or "Coordinates" not in group or len(group["Coordinates"]) == 0:
                continue
            saw_stars = True
            if field not in group:
                return False
    return saw_stars


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    if not len(cumulative) or cumulative[-1] <= 0:
        return float("nan")
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")])


def build_stellar_particle_history(run_dir, snapshot_summary):
    """Count stars and derive age and formation cohorts from their birth scale factors."""
    discovered = discover_snapshot_sets(Path(run_dir) / "data")
    rows = []
    omitted = []
    for snapshot_text, item in snapshot_summary.items():
        try:
            snapshot = int(snapshot_text)
            redshift = float(item.get("actual_redshift", item.get("redshift", np.nan)))
        except (TypeError, ValueError, OverflowError):
            omitted.append({"snapshot": str(snapshot_text), "reason": "snapshot ID or actual redshift is invalid"})
            continue
        files = discovered.get(snapshot, [])
        if not files:
            omitted.append({"snapshot": snapshot, "reason": "no snapshot shards were found"})
            continue
        count = sum(_snapshot_group_count(path, 4) for path in files)
        if not np.isfinite(redshift) or redshift < 0:
            omitted.append({"snapshot": snapshot, "reason": "actual snapshot redshift is invalid"})
            continue
        rows.append((redshift, snapshot, int(count)))
    rows.sort(key=lambda row: row[0])
    any_stars = any(row[2] > 0 for row in rows)
    result = {
        "status": "COMPLETE" if any_stars else "SKIPPED",
        "reason": None if any_stars else "no PartType4 star particles were found in the analyzed snapshots",
        "snapshot": [row[1] for row in rows],
        "redshift": [row[0] for row in rows],
        "star_particle_count": [row[2] for row in rows],
        "included_snapshot_count": len(rows),
        "omitted_snapshot_count": len(omitted),
        "omitted_snapshots": omitted,
        "quantity": "number of PartType4 particle rows across the discovered snapshot shards",
        "unit": "particles",
    }
    if not any_stars:
        result["age_status"] = "SKIPPED"
        result["age_reason"] = "no PartType4 star particles were found in the analyzed snapshots"
        result["birth_cohort_status"] = "SKIPPED"
        return result

    scale_grid, time_grid, clock_reason = _stellar_formation_clock(run_dir)
    if scale_grid is None:
        result["age_status"] = "SKIPPED"
        result["age_reason"] = clock_reason
        result["birth_cohort_status"] = "SKIPPED"
        result["birth_cohort_reason"] = clock_reason
        return result
    formation_candidates = ("GFM_StellarFormationTime", "StellarFormationTime")
    initial_mass_candidates = ("GFM_InitialMass", "InitialMass")
    age_bins = int(load_cfg(run_dir).get("analysis", {}).get("stellar_age_bins", 32))
    age_snapshot_rows = []
    final_age_samples = None
    for redshift, snapshot, _count in rows:
        files = discovered.get(snapshot, [])
        if not files:
            continue
        formation_field = next(
            (name for name in formation_candidates if _field_present_in_every_star_shard(files, name)), None,
        )
        mass_field = next(
            (name for name in initial_mass_candidates if _field_present_in_every_star_shard(files, name)), None,
        )
        if mass_field is None and _field_present_in_every_star_shard(files, "Masses"):
            mass_field = "Masses"
        if formation_field is None or mass_field is None:
            age_snapshot_rows.append({
                "snapshot": snapshot, "redshift": redshift, "status": "SKIPPED",
                "reason": "a recognized stellar formation-scale-factor field or aligned stellar mass field is missing",
            })
            continue
        fields, inverse_weights = read_particle_fields(
            files, 4, [formation_field, mass_field], max_points=1_000_000,
            return_sample_weights=True,
        )
        birth_scale = np.asarray(fields[formation_field], dtype=float).reshape(-1)
        mass = np.asarray(fields[mass_field], dtype=float).reshape(-1) * 1.0e10
        inverse_weights = np.asarray(inverse_weights, dtype=float)
        current_scale = 1.0 / (1.0 + redshift)
        current_time = float(np.interp(current_scale, scale_grid, time_grid))
        valid = (
            np.isfinite(birth_scale) & (birth_scale > 0) & (birth_scale <= current_scale * (1.0 + 1.0e-6))
            & np.isfinite(mass) & (mass > 0) & np.isfinite(inverse_weights) & (inverse_weights > 0)
        )
        ages = current_time - np.interp(np.clip(birth_scale[valid], scale_grid[0], scale_grid[-1]), scale_grid, time_grid)
        weights = inverse_weights[valid]
        weighted_mass = mass[valid] * weights
        if len(ages):
            good_age = np.isfinite(ages) & (ages >= 0)
            ages = ages[good_age]
            weights = weights[good_age]
            weighted_mass = weighted_mass[good_age]
            age_snapshot_rows.append({
                "snapshot": snapshot, "redshift": redshift, "status": "COMPLETE",
                "mean_age_gyr_mass_weighted": float(np.average(ages, weights=weighted_mass)) if np.sum(weighted_mass) > 0 else float("nan"),
                "median_age_gyr_mass_weighted": _weighted_median(ages, weighted_mass),
                "young_mass_fraction_lt_100_myr": float(np.sum(weighted_mass[ages < 0.1]) / np.sum(weighted_mass)) if np.sum(weighted_mass) > 0 else float("nan"),
                "sampled_particle_count": int(len(ages)),
                "estimated_particle_count": float(np.sum(weights)),
                "estimated_stellar_mass_msun_h": float(np.sum(weighted_mass)),
                "formation_field": formation_field, "mass_field": mass_field,
            })
            if final_age_samples is None:
                final_age_samples = (snapshot, ages, weights, weighted_mass, current_time)
        else:
            age_snapshot_rows.append({"snapshot": snapshot, "redshift": redshift, "status": "SKIPPED", "reason": "no valid stellar formation times and masses"})

    complete_age_rows = [row for row in age_snapshot_rows if row.get("status") == "COMPLETE"]
    result["age_status"] = "COMPLETE" if complete_age_rows else "SKIPPED"
    result["age_reason"] = None if complete_age_rows else "no snapshot has valid stellar formation times and masses"
    result["age_history"] = age_snapshot_rows
    result["birth_cohort_status"] = "SKIPPED"
    result["birth_cohort_reason"] = result["age_reason"]
    if final_age_samples is not None:
        final_snapshot, ages, weights, weighted_mass, current_time = final_age_samples
        age_edges = np.linspace(0.0, max(current_time, float(np.max(ages)), 1.0e-6), age_bins + 1)
        age_counts, _ = np.histogram(ages, bins=age_edges, weights=weights)
        age_mass, _ = np.histogram(ages, bins=age_edges, weights=weighted_mass)
        formation_time = np.maximum(current_time - ages, 0.0)
        formation_edges = np.linspace(0.0, max(current_time, 1.0e-6), age_bins + 1)
        formation_counts, _ = np.histogram(formation_time, bins=formation_edges, weights=weights)
        formation_mass, _ = np.histogram(formation_time, bins=formation_edges, weights=weighted_mass)
        result.update({
            "age_distribution_snapshot": int(final_snapshot),
            "age_distribution_bin_edges_gyr": age_edges.tolist(),
            "age_distribution_estimated_particle_count": age_counts.tolist(),
            "age_distribution_surviving_mass_msun_h": age_mass.tolist(),
            "age_distribution_quantity": "surviving PartType4 particle ages from CLASS proper time; inverse-sampling corrected",
            "age_distribution_unit": "Gyr",
            "birth_cohort_status": "COMPLETE",
            "birth_cohort_method": "final available star snapshot binned by GFM formation scale factor; mass is initial mass when recorded, otherwise surviving mass",
            "birth_cohort_snapshot": int(final_snapshot),
            "birth_cohort_formation_time_bin_edges_gyr": formation_edges.tolist(),
            "birth_cohort_particle_count": formation_counts.tolist(),
            "birth_cohort_stellar_mass_msun_h": formation_mass.tolist(),
            "birth_cohort_mass_definition": "initial stellar particle mass when available, otherwise surviving PartType4 Masses",
        })
        result["birth_cohort_reason"] = None
    return result


def plot_stellar_particle_history(run_dir, history):
    """Plot measured stellar mass, count and formation-time statistics across saved outputs."""
    if not history or history.get("status") != "COMPLETE":
        return None
    redshift = np.asarray(history["redshift"], dtype=float)
    count = np.asarray(history["star_particle_count"], dtype=np.int64)
    if redshift.ndim != 1 or count.shape != redshift.shape or not len(count):
        raise ValueError("Stellar-particle history arrays do not align")
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0) or np.any(count < 0):
        raise ValueError("Stellar-particle history contains invalid redshifts or counts")
    plt.figure(figsize=(8, 6)); plt.plot(redshift, count, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("actual snapshot redshift")
    plt.ylabel("star-particle count in the simulation volume")
    plt.title("Stellar-particle count history")
    savefig(Path(run_dir) / "results" / "stellar_particle_count_vs_redshift.png")
    if history.get("age_status") == "COMPLETE":
        cohort_edges = np.asarray(history.get("birth_cohort_formation_time_bin_edges_gyr", []), dtype=float)
        cohort_mass = np.asarray(history.get("birth_cohort_stellar_mass_msun_h", []), dtype=float)
        if len(cohort_edges) == len(cohort_mass) + 1 and len(cohort_mass):
            centers = 0.5 * (cohort_edges[:-1] + cohort_edges[1:])
            plt.figure(figsize=(8, 6))
            plt.step(centers, cohort_mass, where="mid", color="#7c3aed", linewidth=2)
            plt.xlabel("stellar birth cosmic time [Gyr]")
            plt.ylabel(r"stellar cohort mass [$M_\odot/h$]")
            plt.title("Stellar birth-time cohorts at snapshot %03d" % history["birth_cohort_snapshot"])
            savefig(Path(run_dir) / "results" / "stellar_birth_cohorts.png")
        age_rows = [row for row in history.get("age_history", []) if row.get("status") == "COMPLETE"]
        if age_rows:
            redshift = np.asarray([row["redshift"] for row in age_rows], dtype=float)
            mean_age = np.asarray([row["mean_age_gyr_mass_weighted"] for row in age_rows], dtype=float)
            young_fraction = np.asarray([row["young_mass_fraction_lt_100_myr"] for row in age_rows], dtype=float)
            order = np.argsort(redshift)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(redshift[order], mean_age[order], marker="o", color="#2563eb", label="mass-weighted mean age")
            ax.set_xlabel("actual snapshot redshift")
            ax.set_ylabel("mean stellar age [Gyr]")
            ax.invert_xaxis()
            ax2 = ax.twinx()
            ax2.plot(redshift[order], young_fraction[order], marker="s", color="#f97316", label="mass < 100 Myr")
            ax2.set_ylabel("young stellar mass fraction")
            ax2.set_ylim(0.0, 1.0)
            ax.set_title("Stellar age evolution")
            savefig(Path(run_dir) / "results" / "stellar_age_vs_redshift.png")
    return history


def write_stellar_particle_history_hdf5(parent, history, *, run_dir=None, snapshot_epochs=None, provenance=None):
    """Write measured stellar history with physical units and field-availability metadata."""
    group = parent.create_group("stellar_particle_history")
    group.attrs["status"] = history["status"]
    group.attrs["quantity"] = history["quantity"]
    group.attrs["unit"] = history["unit"]
    group.attrs["included_snapshot_count"] = history["included_snapshot_count"]
    group.attrs["omitted_snapshot_count"] = history["omitted_snapshot_count"]
    if history["reason"]:
        group.attrs["reason"] = history["reason"]
    for key in ["age_status", "birth_cohort_status"]:
        if history.get(key) is not None:
            group.attrs[key] = history[key]
    for key in ["age_reason", "birth_cohort_reason", "birth_cohort_method", "birth_cohort_mass_definition",
                "age_distribution_quantity", "age_distribution_unit"]:
        if history.get(key):
            group.attrs[key] = history[key]
    if history["snapshot"]:
        group.create_dataset("snapshot", data=history["snapshot"])
        group.create_dataset("redshift", data=history["redshift"])
        group.create_dataset("star_particle_count", data=history["star_particle_count"])
    for source, name in [
        ("age_distribution_bin_edges_gyr", "age_distribution_bin_edges_gyr"),
        ("age_distribution_estimated_particle_count", "age_distribution_estimated_particle_count"),
        ("age_distribution_surviving_mass_msun_h", "age_distribution_surviving_mass_msun_h"),
        ("birth_cohort_formation_time_bin_edges_gyr", "birth_cohort_formation_time_bin_edges_gyr"),
        ("birth_cohort_particle_count", "birth_cohort_particle_count"),
        ("birth_cohort_stellar_mass_msun_h", "birth_cohort_stellar_mass_msun_h"),
    ]:
        if source in history:
            group.create_dataset(name, data=history[source])
    if history.get("age_distribution_snapshot") is not None:
        group.attrs["age_distribution_snapshot"] = history["age_distribution_snapshot"]
    if history.get("birth_cohort_snapshot") is not None:
        group.attrs["birth_cohort_snapshot"] = history["birth_cohort_snapshot"]
    if history.get("age_history"):
        age_group = group.create_group("age_history")
        age_rows = history["age_history"]
        age_group.create_dataset("snapshot", data=[row["snapshot"] for row in age_rows])
        age_group.create_dataset("redshift", data=[row["redshift"] for row in age_rows])
        age_group.create_dataset("status", data=[row["status"] for row in age_rows], dtype=h5py.string_dtype("utf-8"))
        for field in ["mean_age_gyr_mass_weighted", "median_age_gyr_mass_weighted", "young_mass_fraction_lt_100_myr", "estimated_particle_count", "estimated_stellar_mass_msun_h"]:
            age_group.create_dataset(field, data=[row.get(field, np.nan) for row in age_rows])
        age_group.attrs["young_age_threshold_gyr"] = 0.1
    if history["omitted_snapshots"]:
        omitted = group.create_group("omitted_snapshots")
        omitted.create_dataset(
            "snapshot", data=[str(row["snapshot"]) for row in history["omitted_snapshots"]],
            dtype=h5py.string_dtype("utf-8"),
        )
        omitted.create_dataset(
            "reason", data=[row["reason"] for row in history["omitted_snapshots"]],
            dtype=h5py.string_dtype("utf-8"),
        )
    from core.data import attach_time_series_contract
    from core.cosmology import run_proper_time_clock
    attach_time_series_contract(group,clock=run_proper_time_clock(run_dir) if run_dir is not None else None,
        snapshot_epochs=snapshot_epochs,provenance=provenance)


# ----------------------------- main per-run pass -----------------------------

def _merge_config(base, overlay):
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@directory_owner("folder")
def prepare_hdf5_folder(folder, config_path=None, output_dir=None, products=None):
    """Validate external snapshots and stage a writable, independent run.

    Snapshot payloads are linked individually and only opened for reading.
    Catalogue files are copied so rerunning a finder cannot overwrite the
    source catalogue. The data directory itself must never link to the source:
    subsequent stages write their catalogues, movies and maps beneath it.
    """
    source = Path(folder).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError("HDF5 snapshot folder does not exist: " + str(source))
    snapshots = discover_snapshot_sets(source)
    if not snapshots and (source / "data").is_dir():
        nested = discover_snapshot_sets(source / "data")
        if nested:
            source = (source / "data").resolve()
            snapshots = nested
    if not snapshots:
        raise ValueError("No numbered GADGET snapshot HDF5 files were found in " + str(source))

    from core.foundation import file_sha256 as digest, atomic_copy, atomic_json, atomic_text, publish_directory
    from core.native import native_run_lock
    from core.data import native_publication_copy_members
    intake_hashes = {str(path.resolve()): digest(path) for files in snapshots.values() for path in files}
    records = [validate_gadget_snapshot_set(files, snap) for snap, files in snapshots.items()]
    reference = records[0]
    for record in records[1:]:
        for key in ["box_mpc_h", "hubble_param", "omega_m", "omega_lambda", "omega_baryon"]:
            if not np.isclose(record[key], reference[key], rtol=1e-8, atol=1e-12):
                raise ValueError("Snapshot %d disagrees with snapshot %d on %s" % (record["snapshot"], reference["snapshot"], key))
        if record["units"] != reference["units"]:
            raise ValueError("Snapshot files do not use one consistent unit convention")
    scales = np.asarray([row["scale_factor"] for row in records], dtype=float)
    if len(np.unique(np.round(scales, 12))) != len(scales):
        raise ValueError("Two snapshot files have the same scale factor; check their snapshot numbering")
    if np.any(np.diff(scales) <= 0):
        raise ValueError(
            "Snapshot scale factors do not increase with snapshot number; the selected files may be mislabeled "
            "or may not form one chronological GADGET output series"
        )

    root = Path(__file__).resolve().parent
    intake_hashes[str(root / "config.yaml")] = digest(root / "config.yaml")
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    explicit_config = Path(config_path).expanduser().resolve() if config_path else None
    if explicit_config is None:
        for candidate in [source / "config.yaml", source.parent / "config.yaml"]:
            if candidate.is_file():
                explicit_config = candidate.resolve()
                break
    if explicit_config is not None:
        intake_hashes[str(explicit_config)] = digest(explicit_config)
        supplied = yaml.safe_load(explicit_config.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("External analysis config must contain a YAML mapping")
        _merge_config(cfg, supplied)

    cosmology = cfg.setdefault("cosmology", {})
    simulation = cfg.setdefault("simulation", {})
    h = float(reference["hubble_param"])
    omega_m = float(reference["omega_m"])
    omega_b = float(reference["omega_baryon"])
    box = float(reference["box_mpc_h"])
    counts_by_snapshot = [row["particle_counts_by_type"] for row in records]
    has_baryons = any(counts[0] > 0 or counts[4] > 0 for counts in counts_by_snapshot)
    if explicit_config is not None:
        from core.cosmology import omega_radiation_from_config
        config_h = float(cosmology.get("H0", np.nan)) / 100.0
        config_omega_m = (
            float(cosmology.get("omega_b", 0.0)) + float(cosmology.get("omega_cdm", 0.0))
        ) / config_h**2 if config_h > 0 else np.nan
        config_omega_b = float(cosmology.get("omega_b", 0.0)) / config_h**2 if config_h > 0 else np.nan
        config_omega_lambda = 1.0 - config_omega_m - omega_radiation_from_config(cfg) if np.isfinite(config_omega_m) else np.nan
        checks = [
            ("simulation.box_mpc_h", float(simulation.get("box_mpc_h", np.nan)), box),
            ("cosmology.H0/100", config_h, h),
            ("cosmology.Omega_m", config_omega_m, omega_m),
            ("cosmology.Omega_b", config_omega_b, omega_b),
            ("cosmology.Omega_Lambda", config_omega_lambda, float(reference["omega_lambda"])),
        ]
        for name, given, recorded in checks:
            if not np.isfinite(given) or not np.isclose(given, recorded, rtol=1e-5, atol=1e-8):
                raise ValueError("External config %s=%s disagrees with HDF5 metadata %s" % (name, given, recorded))
    else:
        cosmology.update({
            "model": "external", "engine": "GADGET-4 HDF5 metadata", "H0": 100.0 * h,
            "omega_b": omega_b * h**2, "omega_cdm": max(0.0, omega_m - omega_b) * h**2,
            "A_s": None, "n_s": None, "tau_reio": None, "N_ur": None,
            "N_ncdm": 0, "m_ncdm_eV": 0.0,
        })
    if not explicit_config:
        cosmology["model"] = "external"

    redshifts = [float(row["redshift"]) for row in records]
    simulation.update({
        "box_mpc_h": box,
        "physics": "hydro" if has_baryons else "dm",
        "start_redshift": max(redshifts),
        "snapshot_z_max": max(redshifts),
        "snapshot_z_min": min(redshifts),
    })
    dm_counts = [counts[1] for counts in counts_by_snapshot]
    dm_count = dm_counts[0] if dm_counts else 0
    dm_count_is_stable = bool(dm_counts) and all(count == dm_count for count in dm_counts)
    n_guess = int(round(dm_count ** (1.0 / 3.0))) if dm_count_is_stable and dm_count else 0
    cube_lattice = bool(dm_count_is_stable and dm_count and n_guess**3 == dm_count)
    simulation["particles_per_dimension"] = n_guess if cube_lattice else None
    has_sfr = False
    if has_baryons:
        for shard_set in snapshots.values():
            for path in shard_set:
                with h5py.File(path, "r") as h5:
                    if "PartType0/StarFormationRate" in h5:
                        has_sfr = True
                        break
            if has_sfr:
                break
    cfg.setdefault("hydro", {})["star_formation"] = has_sfr
    if cfg.get("analysis", {}).get("all", True) is not True and products is None:
        products = cfg.get("analysis", {}).get("products")
        if not products:
            raise ValueError("analysis.all=false needs analysis.products or explicit --product selections")

    if output_dir is None:
        runs_root = Path(str(cfg.get("project", {}).get("runs_dir", "runs"))).expanduser()
        if not runs_root.is_absolute():
            runs_root = root / runs_root
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = runs_root / ("imported_" + source.name.replace(" ", "_") + "_" + stamp)
    else:
        output = Path(output_dir).expanduser()
        if not output.is_absolute():
            output = root / output
    output = output.resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Analysis output must be outside the selected snapshot folder")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Import output folder already contains files: " + str(output))
    with native_run_lock(output, operation="external snapshot intake publication"):
        try:
            final_output = output
            final_output.parent.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=".snapshot-intake-", dir=final_output.parent))
            (output / "results").mkdir(exist_ok=True)
            (output / "input").mkdir(exist_ok=True)
            data = output / "data"
            data.mkdir()
            linked_snapshots = []
            for snap, shards in snapshots.items():
                for index, original in enumerate(shards):
                    name = "snapshot_%03d.hdf5" % snap if len(shards) == 1 else "snapshot_%03d.%d.hdf5" % (snap, index)
                    target = data / name
                    target.symlink_to(original.resolve())
                    linked_snapshots.append({"source": str(original.resolve()), "staged_file": name})
            for original, relative in native_publication_copy_members([(Path(row['source']), Path(row['staged_file'])) for row in linked_snapshots]):
                intake_hashes[str(original.resolve())] = digest(original)
                atomic_copy(original, data / relative)
            copied_catalogues = []
            for snap in snapshots:
                for pattern in ("fof_subhalo_tab_%03d*.hdf5", "group_tab_%03d*.hdf5", "groups_%03d/*.hdf5"):
                    copied_catalogues.extend(path for path in source.rglob(pattern % snap)
                                             if path.is_file() and not path.name.startswith("bak-") and _public_native_path(path, source))
            for original in sorted(set(copied_catalogues)):
                target = data / original.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                intake_hashes[str(original.resolve())] = digest(original)
                atomic_copy(original, target)
            for original, relative in native_publication_copy_members([(path, path.relative_to(source)) for path in sorted(set(copied_catalogues))]):
                intake_hashes[str(original.resolve())] = digest(original)
                atomic_copy(original, data / relative)
            atomic_text(output / 'config.yaml', yaml.safe_dump(cfg, sort_keys=False))
            manifest = {
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": "external", "external_snapshot_source": str(source),
                "external_config_source": str(explicit_config) if explicit_config else None,
                "random_seed": None, "particles_per_dimension": simulation["particles_per_dimension"],
                "dark_matter_particle_count_by_snapshot": dm_counts,
                "particle_lattice_inference": {
                    "status": "INFERRED_CUBE" if cube_lattice else "UNAVAILABLE",
                    "method": "cube root of the type-1 particle count in each snapshot, after confirming it is stable across snapshots",
                    "reason": None if cube_lattice else "type-1 counts are not stable or do not form an exact cubic lattice",
                },
                "box_mpc_h": box,
                "cosmology_from_hdf5": {
                    "H0_km_s_Mpc": 100.0 * h, "Omega_m": omega_m,
                    "Omega_b": omega_b, "Omega_Lambda": reference["omega_lambda"],
                },
                "snapshot_validation": records,
                "analysis_provenance": {
                    "analyzer": "DarkMatterSim analyze.py", "snapshot_units": reference["units"],
                    "snapshot_payloads_are_linked_for_reading": True,
                    "data_directory_is_owned_by_import": True,
                    "snapshot_links": linked_snapshots,
                    "copied_catalogues": [str(path.relative_to(source)) for path in sorted(set(copied_catalogues))],
                },
            }
            from core.data import epoch_contract
            from core.foundation import physical_constants_contract
            manifest['intake_contract'] = 'external-snapshot-intake-v1'
            manifest['epochs'] = {str(row['snapshot']): epoch_contract(row['scale_factor']) for row in records}
            manifest['adopted_constants'] = physical_constants_contract()
            manifest['algorithm_sha256'] = digest(__file__)
            manifest['config_sha256'] = digest(output / 'config.yaml')
            manifest['source_artifacts_sha256'] = intake_hashes
            manifest['science_acceptance'] = 'NOT_ACCEPTED'
            manifest['intake_integrity_scope'] = 'Header/epoch/unit/shard validation and source hashes; full payload scientific validation deferred'
            atomic_json(output / 'manifest.json', manifest)
            atomic_text(output / 'manifest.json.sha256', digest(output / 'manifest.json') + '\n')
            if any(digest(path) != expected for path, expected in intake_hashes.items()):
                raise ValueError('Snapshot intake source changed during staging')
            if final_output.exists(): final_output.rmdir()
            publish_directory(output, final_output)
            output = final_output
        except BaseException as exc:
            if output.exists() and output.name.startswith('.snapshot-intake-'):
                atomic_json(output / 'failure.json', {'status':'FAIL', 'science_acceptance':'NOT_ACCEPTED',
                    'failure':{'type':type(exc).__name__,'message':str(exc)}})
            raise
    preview = ", ".join("%03d (z=%.5g)" % (row["snapshot"], row["redshift"]) for row in records)
    print("Validated %d GADGET-4 snapshot(s): %s" % (len(records), preview))
    print("Analysis output: %s" % output)
    return {"run_directory": str(output), "validation": records, "products": products}


def verify_snapshot_intake(run_dir):
    """Check imported source bindings and completion before scientific reuse."""
    from core.foundation import file_sha256 as digest
    root = Path(run_dir).resolve()
    path = root / 'manifest.json'
    if not path.is_file(): return {'status': 'NOT_AN_IMPORTED_RUN'}
    before = digest(path)
    record = json.loads(path.read_text())
    if record.get('intake_contract') != 'external-snapshot-intake-v1':
        return {'status': 'LEGACY_OR_NATIVE_RUN'}
    seal = path.with_name(path.name + '.sha256')
    if not seal.is_file() or seal.read_text().strip() != before:
        raise ValueError('Snapshot intake completion seal missing or changed')
    if digest(root / 'config.yaml') != record['config_sha256']:
        raise ValueError('Imported snapshot configuration changed')
    for source, expected in record['source_artifacts_sha256'].items():
        if digest(source) != expected:
            raise ValueError('Imported snapshot source changed: ' + source)
    for link in record['analysis_provenance']['snapshot_links']:
        relative = Path(link['staged_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Snapshot intake link escapes its data directory')
        staged = root / 'data' / relative
        if not staged.is_symlink() or staged.resolve() != Path(link['source']).resolve():
            raise ValueError('Snapshot intake link changed')
    if digest(path) != before:
        raise ValueError('Snapshot intake manifest changed during verification')
    return {'status': 'INTEGRITY_VERIFIED', 'science_acceptance': 'NOT_ACCEPTED'}


def analyze_hdf5_folder(folder, config_path=None, output_dir=None, products=None,
                        include_pdf_report=True):
    """Stage compatible external snapshots and measure the selected products."""
    prepared = prepare_hdf5_folder(folder, config_path, output_dir, products)
    output = Path(prepared["run_directory"])
    selected = prepared.pop("products")
    prepared["analysis"] = (
        analyze_selected_products(output, selected, include_pdf_report=include_pdf_report)
        if selected else analyze_run(output, include_pdf_report=include_pdf_report)
    )
    return prepared


ANALYSIS_PRODUCTS = (
    "hmf", "finder", "concentration", "pk", "correlation", "bias",
    "kinematics", "spatial", "subhmf", "spin", "profiles", "shape",
    "hydro", "galaxies", "histories", "subhalos",
)


def normalize_analysis_products(products):
    """Validate product names supplied as a list or comma-separated string."""
    if isinstance(products, str):
        products = products.split(",")
    if not isinstance(products, (list, tuple, set)):
        raise ValueError("analysis products must be a list of names")
    selected = []
    expanded = []
    for product in products:
        expanded.extend(str(product).split(","))
    for product in expanded:
        name = str(product).strip().lower().replace("-", "_")
        if name and name not in selected:
            selected.append(name)
    invalid = sorted(set(selected) - set(ANALYSIS_PRODUCTS))
    if invalid:
        raise ValueError("unknown analysis product(s): %s. Choose from: %s" % (
            ", ".join(invalid), ", ".join(ANALYSIS_PRODUCTS),
        ))
    if not selected:
        raise ValueError("select at least one analysis product")
    return selected


def _write_product_hdf5(parent, name, value):
    """Write nested product arrays and metadata without lossy repr strings."""
    if value is None:
        parent.attrs[name + "_status"] = "UNAVAILABLE"
    elif isinstance(value, dict):
        group = parent.create_group(name)
        for key, item in value.items():
            _write_product_hdf5(group, str(key), item)
        if value.get('schema') == 'darkmattersim-subhalos-v1' and 'fields' in group:
            from core.data import write_dataset_contract
            conventions = value['data_contract']['coordinates']
            fields = group['fields']
            for key in fields:
                dataset=fields[key]
                if not isinstance(dataset,h5py.Dataset):
                    continue
                units = ('1e10 Msun/h' if 'Mass' in key and 'Halfmass' not in key else
                         'comoving kpc/h' if key in ('SubhaloPos','SubhaloHalfmassRad','SubhaloVmaxRad') else
                         'km/s' if key in ('SubhaloVel','SubhaloVelDisp','SubhaloVmax') else
                         'Msun/year' if key=='SubhaloSFR' else
                         'identity' if key in ('SubhaloIDMostbound','SubhaloGroupNr') else
                         'particle count' if key=='SubhaloLenType' else
                         'SUBFIND rank within FOF group')
                write_dataset_contract(dataset,units=units,
                    coordinates=conventions if key in ('SubhaloPos','SubhaloVel') else None,
                    identity_namespace=value['data_contract']['identity_namespaces']['native_particle']
                        if key=='SubhaloIDMostbound' else value['data_contract']['identity_namespaces']['halo']
                        if key=='SubhaloGroupNr' else None,
                    epoch=value['epoch_contract'],provenance=value['data_contract'],
                    quantity='saved native SUBFIND '+key+'; pinned conversion conventions in product units',
                    ancestry={'semantics':'epoch-local host FOF row; no persistent cross-epoch halo identity'} if key=='SubhaloGroupNr' else None,
                    missing_policy='optional fields absent are UNAVAILABLE; parent rank -1 denotes no parent')
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
        group = parent.create_group(name)
        for index, item in enumerate(value):
            _write_product_hdf5(group, "item_%04d" % index, item)
    else:
        try:
            array = np.asarray(value)
        except ValueError:
            group = parent.create_group(name)
            for index, item in enumerate(value):
                _write_product_hdf5(group, "item_%04d" % index, item)
            return
        if array.dtype.kind == "O":
            group = parent.create_group(name)
            for index, item in enumerate(array.tolist()):
                _write_product_hdf5(group, "item_%04d" % index, item)
            return
        if array.dtype.kind in {"U", "O"}:
            if array.ndim == 0:
                parent.create_dataset(name, data=str(array.item()), dtype=h5py.string_dtype("utf-8"))
            else:
                parent.create_dataset(name, data=array.astype(h5py.string_dtype("utf-8")))
        else:
            parent.create_dataset(name, data=array, chunks=True, fletcher32=True) if array.ndim and array.size else parent.create_dataset(name,data=array)


def _product_outcome(value, default_reason):
    if value is None:
        return "SKIPPED", default_reason
    if isinstance(value, dict) and value.get("status") in {"SKIPPED", "FAIL", "UNAVAILABLE"}:
        return value["status"], value.get("reason", default_reason)
    return "COMPLETE", None


def _analyze_one_product(stage, cfg, snap, files, product, reliability):
    """Run one named analysis product and return its scientific arrays."""
    if product == "subhalos":
        from halos import export_subhalo_catalogue
        return export_subhalo_catalogue(stage, snap)
    if product == "galaxies":
        if str(cfg.get("simulation", {}).get("physics", "dm")).lower() != "hydro":
            return {"status": "SKIPPED", "reason": "galaxy catalogues require a hydrodynamical run with star particles"}
        from halos import build_galaxy_catalogue

        destination = Path(stage) / "results" / ("galaxies_%03d.hdf5" % int(snap))
        result = build_galaxy_catalogue(
            stage, snap,
            minimum_star_particles=cfg.get("galaxies", {}).get("min_star_particles", 100),
            minimum_dm_particles=cfg.get("galaxies", {}).get("min_dm_particles", 300),
            output_path=destination,
        )
        result["artifact"] = destination.name
        result.pop("path", None)
        result.pop("sha256", None)
        return result
    if product == "hmf":
        raw = halo_masses(stage, snap)
        if raw is None:
            return None
        counts = halo_particle_counts(stage, snap)
        minimum = int(cfg["analysis"]["hmf_min_particles"])
        masses, retained, info = apply_hmf_particle_cut(raw, counts, minimum)
        resolution_selection = hmf_resolution_selection_report(raw, counts, retained, minimum) if retained is not None else {
            "status": "UNAVAILABLE", "reason": info["reason"],
        }
        if masses is None:
            return {"status": "SKIPPED", "reason": info["reason"], "particle_filter": info,
                    "resolution_selection": resolution_selection}
        raw_positions = group_field(stage, snap, ["GroupPos"])
        positions_mpc_h = None
        if (raw_positions is not None and retained is not None
                and np.asarray(raw_positions).shape == (len(raw), 3)
                and np.asarray(retained).shape == (len(raw),)):
            positions_mpc_h = units_to_mpc_h(raw_positions, float(cfg["simulation"]["box_mpc_h"]))[
                np.asarray(retained, dtype=bool)
            ]
        jackknife = hmf_spatial_jackknife(
            masses, positions_mpc_h, box=float(cfg["simulation"]["box_mpc_h"]),
            mass_edges_msun_h=hmf_mass_edges(cfg),
        )
        binned = plot_hmfs(stage, cfg, snap, masses, reliability, positions_mpc_h, jackknife)
        cumulative = plot_cumulative_hmf(stage, cfg, snap, masses, reliability)
        if binned is not None:
            header, _ = read_snapshot_header(stage, snap)
            snapshot_redshift = float(redshift_from_header(header))
            try:
                manifest = json.loads((Path(stage) / "manifest.json").read_text(encoding="utf-8"))
                run_model = str(manifest.get("model", "unknown")).lower()
            except (OSError, ValueError, TypeError):
                run_model = "unknown"
            analytic_hmf = sheth_tormen_hmf_diagnostic(
                stage, cfg, run_model, snapshot_redshift, binned[0], binned[1],
            )
            plot_analytic_hmf_check(
                stage, snap, binned, analytic_hmf, float(cfg["simulation"]["box_mpc_h"])**3,
            )
            from core.hmf_fitting_functions import write_hmf_fit_comparisons
            write_hmf_fit_comparisons(stage, cfg, snap, run_model, snapshot_redshift, binned)
        else:
            analytic_hmf = {"status": "SKIPPED", "reason": "measured HMF is unavailable"}
        return {
            "status": "COMPLETE" if binned is not None else "SKIPPED",
            "reason": None if binned is not None else "fewer than five valid halos or no mass range after the particle cut",
            "particle_filter": info,
            "resolution_selection": resolution_selection,
            "spatial_jackknife": jackknife,
            "analytic_sheth_tormen": analytic_hmf,
            "mass_msun_h": binned[0] if binned is not None else None,
            "dn_dlogM": binned[1] if binned is not None else None,
            "dn_dM": binned[2] if binned is not None else None,
            "counts": binned[3] if binned is not None else None,
            "mass_bin_edges_msun_h": binned[4] if binned is not None else None,
            "cumulative_mass_msun_h": cumulative[0] if cumulative is not None else None,
            "cumulative_n_gt": cumulative[1] if cumulative is not None else None,
            "rare_halos": rare_halo_table(cfg, masses),
        }
    if product == "finder":
        return plot_halo_finder_comparison(stage, cfg, snap)
    if product == "concentration":
        return plot_concentration_mass(stage, cfg, snap)
    if product == "pk":
        return plot_pk(stage, cfg, snap, files)
    if product == "correlation":
        return plot_two_point(stage, cfg, snap)
    if product == "bias":
        return plot_halo_bias(stage, cfg, snap, files)
    if product == "kinematics":
        particle = plot_particle_velocity_distribution(stage, snap, files)
        halo = plot_halo_velocity_distribution(stage, snap)
        return {"particle_velocity": particle, "halo_velocity": halo} if particle is not None or halo is not None else None
    if product == "spatial":
        result = plot_spatial_distribution(stage, cfg, snap)
        return {"position_mpc_h": result[0], "mass_msun_h": result[1]} if result is not None else None
    if product == "subhmf":
        result = plot_subhalo_mass_function(stage, cfg, snap, reliability)
        return {"mass_msun_h": result[0], "dn_dlogM": result[1]} if result is not None else None
    if product == "spin":
        result = plot_spin_distribution(stage, snap)
        return {"specific_angular_momentum": result} if result is not None else None
    if product == "profiles":
        result = top_halo_profiles(stage, cfg, snap, files)
        result["dark_matter_status"] = result["status"]
        result["dark_matter_reason"] = result.get("reason")
        gas_profiles = top_gas_profiles(stage, cfg, snap, files)
        result["gas_profiles"] = gas_profiles
        if result["status"] != "COMPLETE" and gas_profiles["status"] == "COMPLETE":
            result["status"] = "COMPLETE"
            result["reason"] = None
        elif result["status"] != "COMPLETE":
            result["reason"] = "dark-matter profiles: %s; gas profiles: %s" % (
                result.get("dark_matter_reason") or "unavailable",
                gas_profiles.get("reason") or "unavailable",
            )
        return result
    if product == "shape":
        return halo_shapes(stage, cfg, snap, files)
    if product == "hydro":
        diagnostic = hydro_diagnostics(stage, cfg, snap, files)
        fractions = baryon_fractions(stage, snap)
        if fractions is not None:
            plt.figure(figsize=(8, 6))
            total, gas, stars = fractions
            valid = np.isfinite(gas); plt.semilogx(total[valid], gas[valid], ".", alpha=.3, label="gas")
            valid = np.isfinite(stars); plt.semilogx(total[valid], stars[valid], ".", alpha=.3, label="stars")
            plt.legend(); plt.xlabel(r"$M_{halo}\ [M_\odot/h]$"); plt.ylabel("mass fraction")
            savefig(Path(stage) / "results" / ("baryon_fractions_%03d.png" % snap))
            diagnostic["baryon_fraction_halo_mass_msun_h"] = total
            diagnostic["gas_mass_fraction"] = gas
            diagnostic["stellar_mass_fraction"] = stars
        has_hydro_signal = fractions is not None or any(
            key.endswith("_status") and diagnostic.get(key) == "COMPLETE" for key in diagnostic
        )
        if not has_hydro_signal:
            return {"status": "SKIPPED", "reason": "no validated gas or baryon fields are available", "diagnostics": diagnostic}
        return {"status": "COMPLETE", "diagnostics": diagnostic}
    if product == "histories":
        # The history calculation is run once after the per-snapshot loop.
        return {"status": "DEFERRED"}
    raise AssertionError("unhandled analysis product: " + product)


@native_run_owner
def analyze_selected_products(run_dir, products, include_pdf_report=True, *, snapshots=None):
    """Analyze chosen products into a new, uniquely named bundle; preserve prior results."""
    run_dir = Path(run_dir).expanduser().resolve()
    selected = normalize_analysis_products(products)
    analysis_inputs,analysis_contract=_begin_analysis_contract(run_dir,selected)
    cfg = load_cfg(run_dir)
    source_data = run_dir / "data"
    if not source_data.is_dir():
        raise FileNotFoundError("run has no data directory: " + str(source_data))
    snaps = snapshot_ids(run_dir)
    if not snaps:
        raise ValueError("run contains no numbered HDF5 snapshots: " + str(run_dir))
    if snapshots is not None:
        requested = list(snapshots)
        if (not requested or any(type(s) is not int for s in requested)
                or len(set(requested)) != len(requested) or not set(requested) <= set(snaps)):
            raise ValueError('Select distinct existing integer snapshot IDs')
        snaps = sorted(requested)
        analysis_contract['selection']['snapshots'] = snaps
    reliability = load_wdm_reliability(run_dir)
    summary = {
        "source_run": str(run_dir), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "products": selected, "snapshots": {}, "skipped": {},
        "selected_snapshot_ids": snaps,
    }
    analysis_code_sha256 = _analysis_code_sha256()
    summary["analysis_provenance_schema"] = ANALYSIS_PROVENANCE_SCHEMA
    summary["analysis_script_sha256"] = analysis_code_sha256
    summary["analysis_provenance_file"] = "analysis_provenance.json"
    run_results = run_dir / "results"
    run_results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_dir = run_results / ("analysis_products_" + stamp + "_" + uuid.uuid4().hex[:8])
    with _analysis_stage(run_dir) as temporary:
        stage = Path(temporary)
        (stage / "data").symlink_to(source_data.resolve(), target_is_directory=True)
        for name in ("input",):
            path = run_dir / name
            if path.exists():
                (stage / name).symlink_to(path.resolve(), target_is_directory=True)
        shutil.copy2(run_dir / "config.yaml", stage / "config.yaml")
        if (run_dir / "manifest.json").is_file():
            shutil.copy2(run_dir / "manifest.json", stage / "manifest.json")
        output = stage / "results"
        output.mkdir()
        h5_path = output / "analysis_products.h5"
        with h5py.File(h5_path, "w") as h5out:
            h5out.attrs["data_contract_json"]=json.dumps(analysis_contract,allow_nan=False)
            h5out.attrs["science_acceptance"]="NOT_ACCEPTED"
            h5out.attrs["source_run"] = str(run_dir)
            h5out.attrs["products"] = json.dumps(selected)
            h5out.attrs["created_utc"] = summary["created_utc"]
            h5out.attrs["analysis_provenance_schema"] = ANALYSIS_PROVENANCE_SCHEMA
            h5out.attrs["analysis_script_sha256"] = analysis_code_sha256
            h5out.attrs["analysis_scope"] = "selected products; source snapshots are read-only"
            for snap in snaps:
                header, files = read_snapshot_header(stage, snap)
                redshift = float(redshift_from_header(header))
                snap_summary = {"redshift": redshift, "products": {}}
                snap_group = h5out.create_group("snap_%03d" % snap)
                snap_group.attrs["redshift"] = redshift
                from core.data import run_epoch_contract
                epoch = run_epoch_contract(run_dir,int(snap),float(np.asarray(header['Time']).reshape(-1)[0]))
                snap_group.attrs['epoch_contract_json'] = json.dumps(epoch,allow_nan=False)
                snap_summary['epoch_contract'] = epoch
                for product in selected:
                    if product == "histories":
                        continue
                    payload = _analyze_one_product(stage, cfg, snap, files, product, reliability)
                    status, reason = _product_outcome(payload, "required catalogue or particle fields are unavailable")
                    snap_summary["products"][product] = {"status": status, "reason": reason}
                    if product == "pk" and isinstance(payload, dict):
                        snap_summary["products"][product].update({
                            "estimator": "GADGET-4 native mass-weighted CIC-deconvolved folded spectrum",
                            "particle_selection": payload["particle_selection"],
                            "shot_noise": "GADGET-measured, variable-particle-mass estimate; subtracted from signal",
                            "uncertainty_method": payload["uncertainty_method"],
                            "mode_count_precision": payload["mode_count_precision"],
                            "particle_type_spectra": sorted(payload.get("particle_type_spectra", {})),
                            "total_spectrum_note": (
                                "the all-types total includes inter-type cross-power; type auto-spectra are not additive"
                            ),
                            "selected_k_min_h_Mpc": payload["selected_k_min_h_Mpc"],
                            "selected_k_max_h_Mpc": payload["selected_k_max_h_Mpc"],
                            "nonpositive_shot_subtracted_bin_count": payload["nonpositive_shot_subtracted_bin_count"],
                            "reliable_range_status": payload["reliable_range_status"],
                        })
                    if product == "hmf" and isinstance(payload, dict) and payload.get('rare_halos'):
                        snap_summary['rare_halos'] = payload['rare_halos']
                    product_group = snap_group.create_group(product)
                    product_group.attrs["status"] = status
                    if reason:
                        product_group.attrs["reason"] = str(reason)
                    if payload is not None:
                        _write_product_hdf5(product_group, "data", payload)
                        from core.data import attach_analysis_dataset_units,attach_analysis_identity_contracts
                        attach_analysis_identity_contracts(product_group['data'],product,snapshot=int(snap),
                            identity_namespaces=analysis_contract['identity_namespaces'],epoch=epoch,
                            provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json'})
                        attach_analysis_dataset_units(product_group['data'],product,epoch=epoch,
                            provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json',
                                'analysis_script_sha256':analysis_code_sha256,
                                'source_snapshot_storage_index':int(snap),
                                'source_inventory_sidecar':'analysis_provenance.json'})
                        if product == "hydro" and isinstance(payload, dict):
                            diagnostics = payload.get("diagnostics", {})
                            if isinstance(diagnostics, dict) and "sfr_physical_status" in diagnostics:
                                snap_summary["star_formation"] = {
                                    key: diagnostics[key] for key in [
                                        "sfr_physical_status", "sfr_total_msun_per_year",
                                        "sfr_physical_reason", "sfr_status",
                                    ] if key in diagnostics
                                }
                    if status != "COMPLETE":
                        summary["skipped"]["%s_%03d" % (product, snap)] = reason
                summary["snapshots"][str(snap)] = snap_summary
            if "hydro" in selected:
                sfr_history = build_star_formation_history(summary["snapshots"])
                plot_star_formation_history(stage, sfr_history)
                write_star_formation_history_hdf5(h5out, sfr_history,run_dir=run_dir,
                    snapshot_epochs={key:item['epoch_contract'] for key,item in summary['snapshots'].items()},
                    provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json','source_inventory_sidecar':'analysis_provenance.json'})
                summary["star_formation_history"] = sfr_history
                if sfr_history["status"] != "COMPLETE":
                    summary["skipped"]["star_formation_history"] = sfr_history["reason"]
                stellar_history = build_stellar_particle_history(stage, summary["snapshots"])
                plot_stellar_particle_history(stage, stellar_history)
                write_stellar_particle_history_hdf5(h5out, stellar_history,run_dir=run_dir,
                    snapshot_epochs={key:item['epoch_contract'] for key,item in summary['snapshots'].items()},
                    provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json','source_inventory_sidecar':'analysis_provenance.json'})
                summary["stellar_particle_history"] = stellar_history
                if stellar_history["status"] != "COMPLETE":
                    summary["skipped"]["stellar_particle_history"] = stellar_history["reason"]
            if "histories" in selected:
                history = main_progenitor_histories(stage, cfg, summary["snapshots"])
                status, reason = _product_outcome(history, "history catalogs are unavailable")
                hist_group = h5out.create_group("main_progenitor_histories")
                hist_group.attrs["status"] = status
                if reason:
                    hist_group.attrs["reason"] = str(reason)
                if history is not None:
                    from core.data import canonical_halo_history_payload,attach_time_series_contract
                    from core.cosmology import run_proper_time_clock
                    _write_product_hdf5(hist_group, "data", canonical_halo_history_payload(
                        {key:value for key,value in history.items() if key != '_merger_tree'}))
                    attach_time_series_contract(hist_group['data'],clock=run_proper_time_clock(run_dir),
                        snapshot_epochs={key:item['epoch_contract'] for key,item in summary['snapshots'].items()},
                        identity_namespaces=analysis_contract['identity_namespaces'],
                        provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json','source_inventory_sidecar':'analysis_provenance.json'})
                if status == "COMPLETE":
                    plot_mass_accretion_histories(stage, history)
                    plot_merger_histories(stage, history)
                else:
                    summary["skipped"]["histories"] = reason
        if 'hmf' in selected:
            plot_time_evolution(stage, summary)
        from figures import generate_single_run_story_figures, plot_saved_analysis_curves
        summary['saved_unit_figures']=plot_saved_analysis_curves(h5_path,output)
        summary["paper_figure_recipes"] = generate_single_run_story_figures(
            h5_path, output / "paper_figures", cfg.get("visualization", {}), source_run=stage,
        )
        if summary['paper_figure_recipes'].get('status') != 'COMPLETE':
            summary['skipped']['cosmic_web_story'] = (summary['paper_figure_recipes'].get('reason')
                or summary['paper_figure_recipes'].get('status') or 'Missing cosmic-web figure status')
        from core.snapshot_tables import write_snapshot_tables
        summary['snapshot_tables'] = write_snapshot_tables(summary['snapshots'], output)
        summary["output_directory"] = str(final_dir)
        if include_pdf_report:
            summary["report_pdf"] = "summary.pdf"
            create_analysis_report_pdf(output, summary, cfg, source_run=run_dir)
        from figures import review_figure_folder
        summary["figure_review"] = review_figure_folder(output, create_gallery=include_pdf_report)
        summary["artifacts"] = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
        (output / "analysis_products_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_analysis_provenance(
            output, run_dir, selected, summary["created_utc"], code_sha256=analysis_code_sha256, inputs=analysis_inputs, data_contract=analysis_contract,
        )
        from core.foundation import publish_directory
        verify_analysis_bundle(output,verify_sources=True)
        publish_directory(output,final_dir)
    print("Selected analysis bundle: " + str(final_dir))
    for snap, record in summary["snapshots"].items():
        statuses = ", ".join("%s=%s" % (name, detail["status"]) for name, detail in record["products"].items())
        if statuses:
            print("  snapshot %s (z=%.6g): %s" % (snap, record["redshift"], statuses))
    return summary


@native_run_owner
def analyze_run(run_dir, overwrite=False, products=None, include_pdf_report=True, *, full_analysis=False):
    """Measure explicitly selected products from saved particles/catalogues and write provenance with the outputs."""
    run_dir = Path(run_dir).expanduser().resolve()
    config_before_read=_file_sha256(run_dir/'config.yaml')
    cfg = load_cfg(run_dir)
    if full_analysis and products is not None:
        raise ValueError('Choose selected products or explicit full analysis')
    if products is not None:
        return analyze_selected_products(run_dir, products, include_pdf_report=include_pdf_report)
    if not full_analysis and cfg.get("analysis", {}).get("all", True) is not True:
        products = cfg.get("analysis", {}).get("products")
        if not products:
            raise ValueError("analysis.all=false needs analysis.products or explicit --product selections")
        return analyze_selected_products(run_dir, products, include_pdf_report=include_pdf_report)
    analysis_inputs,analysis_contract=_begin_analysis_contract(run_dir,["all"])
    if analysis_inputs['config']['sha256'] != config_before_read:
        raise ValueError('Analysis configuration changed while loading it')
    results = run_dir / "results"
    archived_results = None
    if results.exists() and any(results.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "Analysis results already exist; rerunning would replace them. Pass overwrite=True or use --overwrite "
                "to archive the existing results directory first."
            )
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived_results = run_dir / ("results_archive_" + stamp + "_" + uuid.uuid4().hex[:8])
        os.replace(results, archived_results)
    results.mkdir(exist_ok=True)
    summary = {"snapshots": {}, "skipped": {}}
    analysis_created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    analysis_code_sha256 = _analysis_code_sha256()
    summary["analysis_provenance_schema"] = ANALYSIS_PROVENANCE_SCHEMA
    summary["analysis_script_sha256"] = analysis_code_sha256
    summary["analysis_provenance_file"] = "analysis_provenance.json"
    summary["analysis_created_utc"] = analysis_created_utc
    if archived_results is not None:
        summary["previous_results_archived"] = str(archived_results)
    wdm_reliability = load_wdm_reliability(run_dir)
    if wdm_reliability:
        summary["wdm_reliability"] = wdm_reliability
    model = "unknown"
    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
                model = str(manifest.get("model", "unknown")).lower()
        except (OSError, ValueError):
            pass
    snapshots = snapshot_ids(run_dir)
    redshift_schedule = match_requested_snapshot_redshifts(manifest, snapshots)
    time_scale, time_values, time_source = _cosmology_time_samples(run_dir, model)
    def proper_time_at_scale_factor(scale_factor):
        if time_scale is None or time_values is None:
            return None
        from core.cosmology import interpolate_proper_time
        return interpolate_proper_time(time_scale,time_values,scale_factor)
    summary["snapshot_redshift_schedule"] = {
        "status": redshift_schedule["status"],
        "reason": redshift_schedule["reason"],
        "method": redshift_schedule["method"],
        "requested_snapshot_count": (
            len(redshift_schedule["requested_redshifts"])
            if redshift_schedule["requested_redshifts"] is not None else None
        ),
        "analyzed_snapshot_count": len(snapshots),
        "proper_time_source": time_source,
        "proper_time_status": "CLASS_BACKGROUND_INTERPOLATED" if time_source else "UNAVAILABLE",
    }
    if redshift_schedule["requested_redshifts"] is not None:
        summary["snapshot_redshift_schedule"]["requested_redshifts"] = redshift_schedule["requested_redshifts"]
    analysis_tmp = results / (".analysis_" + uuid.uuid4().hex + ".h5.tmp")
    h5out = h5py.File(analysis_tmp, "w")
    h5out.attrs['data_contract_json']=json.dumps(analysis_contract,allow_nan=False)
    h5out.attrs['science_acceptance']='NOT_ACCEPTED'
    h5out.attrs["model"] = model
    h5out.attrs["analysis_provenance_schema"] = ANALYSIS_PROVENANCE_SCHEMA
    h5out.attrs["analysis_script_sha256"] = analysis_code_sha256
    h5out.attrs["analysis_created_utc"] = analysis_created_utc
    h5out.attrs["hmf_mass_definition"] = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    h5out.attrs["hmf_min_particles"] = int(cfg["analysis"]["hmf_min_particles"])
    if wdm_reliability:
        h5out.attrs["wdm_reliability_status"] = wdm_reliability["status"]
        if "mass_limit_msun_h" in wdm_reliability:
            h5out.attrs["wdm_reliability_mass_limit_msun_h"] = wdm_reliability["mass_limit_msun_h"]
        else:
            h5out.attrs["wdm_reliability_reason"] = wdm_reliability["reason"]
    h5out.attrs["power_spectrum_estimator"] = "GADGET-4 full-particle folded CIC mesh; CIC-window deconvolution; measured shot-noise subtraction"
    h5out.attrs["power_spectrum_status"] = (
        "GADGET-4 native estimator; shot-included and shot-noise-subtracted P(k), measured shot noise, "
        "Gaussian mode-counting 1-sigma errors, mode counts, fold band, and particle selection are saved per snapshot"
    )
    h5out.attrs["snapshot_redshift_schedule_status"] = redshift_schedule["status"]
    h5out.attrs["snapshot_redshift_match_method"] = redshift_schedule["method"]
    if redshift_schedule["reason"] is not None:
        h5out.attrs["snapshot_redshift_schedule_reason"] = redshift_schedule["reason"]
    redshift_schedule_rows = []
    try:
        for snap in snapshots:
            header, files = read_snapshot_header(run_dir, snap)
            z = redshift_from_header(header)
            item = {"redshift": float(z)}
            g = h5out.require_group("snap_%03d" % snap); g.attrs["redshift"] = z
            from core.data import run_epoch_contract, hdf5_scalar_attr
            epoch=run_epoch_contract(run_dir,int(snap),hdf5_scalar_attr(header['Time'],'Time'))
            g.attrs['epoch_contract_json']=json.dumps(epoch,allow_nan=False)
            item['epoch_contract']=epoch
            from halos import export_subhalo_catalogue
            subhalos = export_subhalo_catalogue(run_dir, snap)
            _write_product_hdf5(g, "subhalos", subhalos)
            item["subhalos"] = {key: subhalos.get(key) for key in ("status", "reason", "count", "artifact")}
            if subhalos["status"] != "COMPLETE":
                summary["skipped"]["subhalos_%03d" % snap] = subhalos.get("reason")
            g.attrs["actual_redshift"] = z
            actual_a = epoch['actual_scale_factor']
            g.attrs["actual_scale_factor"] = actual_a
            g.attrs["scale_factor_units"] = "dimensionless"
            actual_time = proper_time_at_scale_factor(actual_a)
            if actual_time is not None:
                g.attrs["actual_cosmic_time_gyr"] = actual_time
                g.attrs["cosmic_time_units"] = "Gyr"
                item["actual_cosmic_time_gyr"] = actual_time
            g.attrs["requested_redshift_status"] = redshift_schedule["status"]
            if snap in redshift_schedule["by_snapshot"]:
                requested_z = float(redshift_schedule["by_snapshot"][snap])
                requested_a = 1.0 / (1.0 + requested_z)
                offset = float(z - requested_z)
                item["actual_redshift"] = float(z)
                item["requested_redshift"] = requested_z
                item["redshift_offset_actual_minus_requested"] = offset
                item["actual_scale_factor"] = actual_a
                item["requested_scale_factor"] = requested_a
                g.attrs["requested_redshift"] = requested_z
                g.attrs["redshift_offset_actual_minus_requested"] = offset
                g.attrs["requested_scale_factor"] = requested_a
                requested_time = proper_time_at_scale_factor(requested_a)
                if requested_time is not None:
                    g.attrs["requested_cosmic_time_gyr"] = requested_time
                    item["requested_cosmic_time_gyr"] = requested_time
                if requested_time is not None and actual_time is not None:
                    g.attrs["cosmic_time_offset_actual_minus_requested_gyr"] = actual_time - requested_time
                    item["cosmic_time_offset_actual_minus_requested_gyr"] = actual_time - requested_time
                redshift_schedule_rows.append((
                    snap, requested_z, float(z), offset, requested_a, actual_a,
                    requested_time if requested_time is not None else np.nan,
                    actual_time if actual_time is not None else np.nan,
                ))
            elif redshift_schedule["reason"] is not None:
                g.attrs["requested_redshift_reason"] = redshift_schedule["reason"]
            if wdm_reliability:
                g.attrs["wdm_reliability_status"] = wdm_reliability["status"]
                if "mass_limit_msun_h" in wdm_reliability:
                    g.attrs["wdm_reliability_mass_limit_msun_h"] = wdm_reliability["mass_limit_msun_h"]

            raw_masses = halo_masses(run_dir, snap)
            if raw_masses is None:
                summary["skipped"]["hmf_%03d" % snap] = "no host-halo mass field"
            else:
                particle_counts = halo_particle_counts(run_dir, snap)
                minimum = int(cfg["analysis"]["hmf_min_particles"])
                hmf_masses, retained, filter_info = apply_hmf_particle_cut(raw_masses, particle_counts, minimum)
                item["hmf_particle_filter"] = filter_info
                resolution_selection = hmf_resolution_selection_report(
                    raw_masses, particle_counts, retained, minimum,
                ) if retained is not None else {"status": "UNAVAILABLE", "reason": filter_info["reason"]}
                item["hmf_resolution_selection"] = resolution_selection
                g.attrs["hmf_min_particles"] = minimum
                g.attrs["hmf_resolution_selection_json"] = json.dumps(resolution_selection, sort_keys=True)
                g.create_dataset("host_halo_mass_unfiltered_msun_h", data=mass_to_msun_h(raw_masses))
                g.create_dataset("host_halo_in_hmf", data=retained if retained is not None else np.zeros(len(raw_masses), dtype=bool))
                if particle_counts is not None:
                    g.create_dataset("host_halo_particle_count", data=particle_counts)
                all_host_positions = group_field(run_dir, snap, ["GroupPos"])
                all_host_positions_mpc_h = None
                if all_host_positions is not None and np.asarray(all_host_positions).shape == (len(raw_masses), 3):
                    all_host_positions_mpc_h = units_to_mpc_h(
                        all_host_positions, float(cfg["simulation"]["box_mpc_h"]),
                    )
                if model == "wdm":
                    raw_mass_msun_h = mass_to_msun_h(raw_masses)
                    clean_mask = np.zeros(len(raw_masses), dtype=bool)
                    rejection_reason = np.full(len(raw_masses), 3, dtype=np.uint8)
                    if retained is not None:
                        rejection_reason[~retained] = 1  # below the configured minimum particle count
                    if wdm_reliability and "mass_limit_msun_h" in wdm_reliability and retained is not None:
                        wang_white_limit = float(wdm_reliability["mass_limit_msun_h"])
                        below_spurious_limit = np.isfinite(raw_mass_msun_h) & (raw_mass_msun_h < wang_white_limit)
                        clean_mask = np.asarray(retained, dtype=bool) & ~below_spurious_limit
                        rejection_reason[np.asarray(retained, dtype=bool) & below_spurious_limit] = 2
                        rejection_reason[clean_mask] = 0
                    g.create_dataset("wdm_clean_catalog_source_row_index", data=np.flatnonzero(clean_mask).astype(np.int64))
                    g.create_dataset("wdm_clean_catalog_host_mass_msun_h", data=raw_mass_msun_h[clean_mask])
                    if all_host_positions_mpc_h is not None:
                        g.create_dataset(
                            "wdm_clean_catalog_host_position_mpc_h",
                            data=all_host_positions_mpc_h[clean_mask],
                        )
                    if particle_counts is not None and len(particle_counts) == len(clean_mask):
                        g.create_dataset(
                            "wdm_clean_catalog_host_particle_count",
                            data=np.asarray(particle_counts)[clean_mask],
                        )
                    g.create_dataset("wdm_clean_catalog_in_raw_catalog", data=clean_mask)
                    g.create_dataset("wdm_raw_catalog_rejection_reason", data=rejection_reason)
                    g["wdm_clean_catalog_in_raw_catalog"].attrs["selection"] = (
                        "host passes the configured particle cut and the empirical Wang-White mass limit; "
                        "missing cut inputs produce an empty cleaned view"
                    )
                    g["wdm_raw_catalog_rejection_reason"].attrs["codes_json"] = json.dumps({
                        "0": "retained in Wang-White-cleaned sample",
                        "1": "rejected by minimum FOF particle count",
                        "2": "rejected below empirical Wang-White mass limit",
                        "3": "cleaning eligibility unavailable because a required cut or limit is missing",
                    }, sort_keys=True)
                    item["wdm_catalogue_quality"] = {
                        "status": "MEASURED" if wdm_reliability and "mass_limit_msun_h" in wdm_reliability and retained is not None else "UNAVAILABLE",
                        "raw_catalogue_host_count": int(len(raw_masses)),
                        "particle_cut_retained_count": int(np.count_nonzero(retained)) if retained is not None else None,
                        "wang_white_clean_count": int(np.count_nonzero(clean_mask)) if wdm_reliability and "mass_limit_msun_h" in wdm_reliability and retained is not None else None,
                        "wang_white_spurious_mass_msun_h": wdm_reliability.get("mass_limit_msun_h") if wdm_reliability else None,
                        "rejection_reason_codes": {"1": "minimum_particle_count", "2": "below_Wang_White_limit", "3": "cleaning_unavailable"},
                        "interpretation": "The cleaned view applies particle-count and Wang-White mass cuts only; it does not identify each numerical halo with particle tracing or a matched control run.",
                    }
                    protohalo_shape = {
                        "status": "NOT_RUN",
                        "reason": "Lagrangian tracing is limited to the final available snapshot to bound repeated IC reads",
                    }
                    if snapshots and snap == snapshots[-1]:
                        try:
                            protohalo_shape = trace_lagrangian_protohalo_shapes(
                                run_dir, snap,
                                min_particles=int(cfg.get("analysis", {}).get("halo_shape_min_particles", 100)),
                                max_halos=int(cfg.get("analysis", {}).get("halo_shape_max_halos", 40)),
                                max_particles=int(cfg.get("analysis", {}).get("halo_shape_max_particles", 4_000_000)),
                                mass_definition=str(cfg["analysis"]["hmf_mass_definition"]),
                            )
                        except (OSError, KeyError, TypeError, ValueError, OverflowError, ImportError) as exc:
                            protohalo_shape = {"status": "UNAVAILABLE", "reason": str(exc), "snapshot": int(snap)}
                    shape_status = str(protohalo_shape.get("status", "UNAVAILABLE"))
                    g.attrs["wdm_protohalo_shape_status"] = shape_status
                    if protohalo_shape.get("method"):
                        g.attrs["wdm_protohalo_shape_method"] = str(protohalo_shape["method"])
                    if protohalo_shape.get("reason"):
                        g.attrs["wdm_protohalo_shape_reason"] = str(protohalo_shape["reason"])
                    if protohalo_shape.get("row_status") is not None:
                        row_status = np.asarray(protohalo_shape["row_status"], dtype=np.uint8)
                        shape_count = np.asarray(protohalo_shape.get("particle_count", []), dtype=np.int64)
                        shape_center = np.asarray(protohalo_shape.get("center_mpc_h", []))
                        shape_arrays = [np.asarray(protohalo_shape.get(name, [])) for name in (
                            "b_over_a", "c_over_a", "sphericity_c_over_a",
                        )]
                        if (row_status.shape != (len(raw_masses),) or shape_count.shape != (len(raw_masses),)
                                or shape_center.shape != (len(raw_masses), 3)
                                or any(values.shape != (len(raw_masses),) for values in shape_arrays)):
                            g.attrs["wdm_protohalo_shape_status"] = "UNAVAILABLE"
                            g.attrs["wdm_protohalo_shape_reason"] = "protohalo arrays do not align with raw host catalogue rows"
                            shape_status = "UNAVAILABLE"
                            protohalo_shape["status"] = "UNAVAILABLE"
                            protohalo_shape["reason"] = "protohalo arrays do not align with raw host catalogue rows"
                        else:
                            g.create_dataset("wdm_protohalo_shape_status_code", data=row_status)
                            g.create_dataset("wdm_protohalo_traced_particle_count", data=shape_count)
                            g.create_dataset("wdm_protohalo_lagrangian_center_mpc_h", data=shape_center)
                            g.create_dataset("wdm_protohalo_b_over_a", data=shape_arrays[0])
                            g.create_dataset("wdm_protohalo_c_over_a", data=shape_arrays[1])
                            g.create_dataset("wdm_protohalo_sphericity_c_over_a", data=shape_arrays[2])
                            g["wdm_protohalo_shape_status_code"].attrs["codes_json"] = json.dumps(
                                protohalo_shape["status_codes"], sort_keys=True,
                            )
                            g["wdm_protohalo_sphericity_c_over_a"].attrs["definition"] = (
                                "minor-to-major semi-axis ratio derived from the periodic Lagrangian inertia-tensor eigenvalues assuming a uniform ellipsoid"
                            )
                            g["wdm_protohalo_sphericity_c_over_a"].attrs["selection"] = (
                                "traced host sample only; NaN means not measured"
                            )
                    item["wdm_catalogue_quality"]["protohalo_shape"] = {
                        key: value for key, value in protohalo_shape.items()
                        if key not in {"row_status", "particle_count", "center_mpc_h", "b_over_a", "c_over_a", "sphericity_c_over_a"}
                    }
                    item["wdm_catalogue_quality"]["interpretation"] += (
                        " Final-snapshot Lagrangian c/a diagnostics are saved for a bounded halo sample; "
                        "they do not yet drive cleaning because matched LCDM and higher-resolution evidence "
                        "is required to classify numerical halos."
                    )
                if all_host_positions_mpc_h is not None:
                    g.create_dataset(
                        "host_halo_positions_all_mpc_h",
                        data=all_host_positions_mpc_h,
                    )
                if hmf_masses is None:
                    summary["skipped"]["hmf_%03d" % snap] = filter_info["reason"]
                else:
                    hmf_positions_mpc_h = None
                    if (all_host_positions_mpc_h is not None and retained is not None
                            and np.asarray(retained).shape == (len(raw_masses),)
                            and len(all_host_positions_mpc_h) == len(raw_masses)):
                        hmf_positions_mpc_h = all_host_positions_mpc_h[
                            np.asarray(retained, dtype=bool)
                        ]
                    hmf_jackknife = hmf_spatial_jackknife(
                        hmf_masses, hmf_positions_mpc_h,
                        box=float(cfg["simulation"]["box_mpc_h"]),
                        mass_edges_msun_h=hmf_mass_edges(cfg),
                    )
                    hmf = plot_hmfs(
                        run_dir, cfg, snap, hmf_masses, wdm_reliability,
                        hmf_positions_mpc_h, hmf_jackknife,
                    )
                    cum = plot_cumulative_hmf(run_dir, cfg, snap, hmf_masses, wdm_reliability)
                    finder_cmp = plot_halo_finder_comparison(run_dir, cfg, snap)
                    item["rare_halos"] = rare_halo_table(cfg, hmf_masses)
                    if model == "wdm" and wdm_reliability and "mass_limit_msun_h" in wdm_reliability:
                        wang_white_limit = float(wdm_reliability["mass_limit_msun_h"])
                        cleaned_masses = np.asarray(hmf_masses, dtype=float)
                        cleaned_masses = cleaned_masses[
                            np.isfinite(mass_to_msun_h(cleaned_masses))
                            & (mass_to_msun_h(cleaned_masses) >= wang_white_limit)
                        ]
                        cleaned_hmf = hmf_arrays(
                            cleaned_masses, float(cfg["simulation"]["box_mpc_h"]), hmf_mass_edges(cfg),
                        )
                        if cleaned_hmf is not None:
                            for dataset_name, values in (
                                ("wdm_clean_hmf_mass_msun_h", cleaned_hmf[0]),
                                ("wdm_clean_hmf_dn_dlog10M_per_mpc_h3", cleaned_hmf[1]),
                                ("wdm_clean_hmf_counts", cleaned_hmf[3]),
                                ("wdm_clean_hmf_mass_bin_edges_msun_h", cleaned_hmf[4]),
                            ):
                                g.create_dataset(dataset_name, data=values)
                            item["wdm_catalogue_quality"]["cleaned_hmf_bin_count"] = int(len(cleaned_hmf[0]))
                    if hmf:
                        g.create_dataset("hmf_mass", data=hmf[0])
                        g.create_dataset("hmf_dn_dlogM", data=hmf[1])
                        g.create_dataset("hmf_dn_dM", data=hmf[2])
                        g.create_dataset("hmf_counts", data=hmf[3])
                        g.create_dataset("hmf_mass_bin_edges_msun_h", data=hmf[4])
                        g.attrs["hmf_spatial_jackknife_json"] = json.dumps(hmf_jackknife, sort_keys=True)
                        if hmf_jackknife.get("status") == "COMPLETE":
                            g.create_dataset(
                                "hmf_spatial_jackknife_sigma_dn_dlog10M",
                                data=np.asarray(hmf_jackknife["sigma_dn_dlog10M_per_mpc_h3"], dtype=float),
                            )
                            g.create_dataset(
                                "hmf_spatial_jackknife_sigma_dn_dM",
                                data=np.asarray(hmf_jackknife["sigma_dn_dM_h4_per_msun_mpc3"], dtype=float),
                            )
                        volume = float(cfg["simulation"]["box_mpc_h"])**3
                        lower_log, upper_log = poisson_rate_interval(hmf[3], volume, np.diff(np.log10(hmf[4])))
                        lower_mass, upper_mass = poisson_rate_interval(hmf[3], volume, np.diff(hmf[4]))
                        g.create_dataset("hmf_poisson_68_lower_dn_dlogM", data=lower_log)
                        g.create_dataset("hmf_poisson_68_upper_dn_dlogM", data=upper_log)
                        g.create_dataset("hmf_poisson_68_lower_dn_dM", data=lower_mass)
                        g.create_dataset("hmf_poisson_68_upper_dn_dM", data=upper_mass)
                        g.attrs["hmf_uncertainty_method"] = "68.27% exact central Garwood Poisson intervals per bin"
                        g.attrs["hmf_finite_volume_uncertainty"] = "not estimated from a single realization"
                        item["hmf_uncertainty"] = {
                            "poisson_interval": "68.27% central Garwood",
                            "finite_volume_sample_variance": "not estimated; requires multiple realizations or a validated covariance model",
                        }
                        analytic_hmf = sheth_tormen_hmf_diagnostic(
                            run_dir, cfg, model, z, hmf[0], hmf[1],
                        )
                        from core.hmf_fitting_functions import write_hmf_fit_comparisons
                        write_hmf_fit_comparisons(run_dir, cfg, snap, model, z, hmf)
                        analytic_summary = _analytic_hmf_summary(analytic_hmf)
                        item["analytic_hmf_sheth_tormen"] = analytic_summary
                        g.attrs["hmf_analytic_sheth_tormen_json"] = json.dumps(analytic_summary, sort_keys=True)
                        if analytic_hmf["status"] == "DIAGNOSTIC_ONLY":
                            for dataset, key in [
                                ("hmf_analytic_sheth_tormen_sigma_linear_top_hat", "sigma_linear_top_hat"),
                                ("hmf_analytic_sheth_tormen_dln_sigma_inverse_dlnM", "dln_sigma_inverse_dlnM"),
                                ("hmf_analytic_sheth_tormen_dn_dlog10M", "dn_dlog10M_mpc_h3"),
                                ("hmf_measurement_over_sheth_tormen_ratio", "measurement_over_fit_ratio"),
                            ]:
                                if key in analytic_hmf:
                                    g.create_dataset(dataset, data=analytic_hmf[key])
                            plot_analytic_hmf_check(
                                run_dir, snap, hmf, analytic_hmf, volume,
                            )
                        else:
                            summary["skipped"]["analytic_hmf_sheth_tormen_%03d" % snap] = analytic_summary["reason"]
                    else:
                        summary["skipped"]["hmf_%03d" % snap] = "fewer than five valid halos or no mass range after the particle-count cut"
                    if cum:
                        g.create_dataset("cumulative_mass", data=cum[0]); g.create_dataset("cumulative_n_gt", data=cum[1])
                    if finder_cmp["status"] == "COMPLETE":
                        for key, dataset in [
                            ("mass", "finder_comparison_mass"),
                            ("gadget_dn_dlogM", "finder_comparison_gadget_dn_dlogM"),
                            ("rockstar_dn_dlogM", "finder_comparison_rockstar_dn_dlogM"),
                            ("ratio", "finder_comparison_ratio"),
                        ]:
                            g.create_dataset(dataset, data=finder_cmp[key])
                        g.attrs["finder_comparison_mass_definition"] = finder_cmp["mass_definition"]
                        g.attrs["finder_comparison_mass_component"] = finder_cmp["mass_component"]
                        g.attrs["finder_comparison_type"] = finder_cmp["comparison_type"]
                        g.attrs["finder_comparison_host_count_gadget"] = finder_cmp["host_count_gadget"]
                        g.attrs["finder_comparison_host_count_rockstar"] = finder_cmp["host_count_rockstar"]
                        g.attrs["finder_comparison_per_halo_match_status"] = finder_cmp["per_halo_match_status"]
                        g.attrs["finder_comparison_per_halo_match_method"] = finder_cmp.get(
                            "per_halo_match_method", "unavailable",
                        )
                        if finder_cmp["per_halo_match_status"] == "COMPLETE":
                            for key, dataset in [
                                ("per_halo_gadget_index", "finder_match_gadget_index"),
                                ("per_halo_rockstar_index", "finder_match_rockstar_index"),
                                ("per_halo_rockstar_id", "finder_match_rockstar_id"),
                                ("per_halo_center_separation_mpc_h", "finder_match_center_separation_mpc_h"),
                                ("per_halo_gadget_mass_msun_h", "finder_match_gadget_mass_msun_h"),
                                ("per_halo_rockstar_mass_msun_h", "finder_match_rockstar_mass_msun_h"),
                                ("per_halo_rockstar_to_gadget_mass_ratio", "finder_match_rockstar_to_gadget_mass_ratio"),
                                ("per_halo_log10_rockstar_to_gadget_mass_ratio", "finder_match_log10_rockstar_to_gadget_mass_ratio"),
                            ]:
                                g.create_dataset(dataset, data=finder_cmp[key])
                            definition_group = g.create_group("finder_match_mass_definitions")
                            for definition_name, values in finder_cmp.get(
                                "per_halo_mass_definition_pairs", {},
                            ).items():
                                pair_group = definition_group.create_group(definition_name)
                                pair_group.attrs["gadget_field"] = values["gadget_field"]
                                pair_group.attrs["rockstar_field"] = values["rockstar_field"]
                                pair_group.attrs["valid_halo_count"] = values["valid_halo_count"]
                                pair_group.attrs["median_rockstar_to_gadget_ratio"] = values["median_ratio"]
                                pair_group.attrs["ratio_percentile_16_84"] = values["ratio_percentile_16_84"]
                                pair_group.attrs["mean_log10_rockstar_to_gadget_ratio"] = values["mean_log10_ratio"]
                                pair_group.attrs["mass_units"] = "Msun/h"
                                pair_group.attrs["halo_pairing_method"] = (
                                    "one-to-one reciprocal nearest periodic centres; Rockstar halo IDs are retained for audit"
                                )
                                pair_group.create_dataset("gadget_mass_msun_h", data=values["gadget_mass_msun_h"])
                                pair_group.create_dataset("rockstar_mass_msun_h", data=values["rockstar_mass_msun_h"])
                                pair_group.create_dataset("rockstar_to_gadget_mass_ratio", data=values["rockstar_to_gadget_ratio"])
                                pair_group.create_dataset("gadget_catalogue_index", data=values["gadget_catalogue_index"])
                                pair_group.create_dataset("rockstar_catalogue_index", data=values["rockstar_catalogue_index"])
                                pair_group.create_dataset("rockstar_halo_id", data=values["rockstar_halo_id"])
                            central_pair = finder_cmp.get("per_halo_subfind_central_pair")
                            if central_pair is not None:
                                central_group = g.create_group("finder_match_subfind_central")
                                central_group.attrs["membership_link"] = "Group/GroupFirstSub -> Subhalo row"
                                central_group.attrs["rockstar_pairing_method"] = (
                                    "one-to-one reciprocal nearest periodic centres"
                                )
                                central_group.attrs["mass_definitions"] = (
                                    "GADGET GroupMass (FOF), SUBFIND SubhaloMass (central bound mass), "
                                    "Rockstar MFOF; all masses in Msun/h"
                                )
                                for field_name, values in central_pair.items():
                                    central_group.create_dataset(field_name, data=values)
                            g.attrs["finder_match_max_separation_mpc_h"] = finder_cmp["per_halo_match_max_separation_mpc_h"]
                            g.attrs["finder_match_count"] = finder_cmp["per_halo_match_count"]
                            g.attrs["finder_match_gadget_fraction"] = finder_cmp["per_halo_match_gadget_fraction"]
                            g.attrs["finder_match_rockstar_fraction"] = finder_cmp["per_halo_match_rockstar_fraction"]
                        item["finder_comparison"] = {
                            "status": "COMPLETE",
                            "mass_definition": finder_cmp["mass_definition"],
                            "mass_component": finder_cmp["mass_component"],
                            "host_count_gadget": finder_cmp["host_count_gadget"],
                            "host_count_rockstar": finder_cmp["host_count_rockstar"],
                            "per_halo_match_status": finder_cmp["per_halo_match_status"],
                            "per_halo_match_method": finder_cmp.get("per_halo_match_method"),
                            "per_halo_match_count": finder_cmp.get("per_halo_match_count"),
                            "per_halo_match_gadget_fraction": finder_cmp.get("per_halo_match_gadget_fraction"),
                            "per_halo_match_rockstar_fraction": finder_cmp.get("per_halo_match_rockstar_fraction"),
                            "per_halo_match_max_separation_mpc_h": finder_cmp.get("per_halo_match_max_separation_mpc_h"),
                            "per_halo_mass_ratio_summary": finder_cmp.get("per_halo_mass_ratio_summary"),
                            "per_halo_center_separation_summary_mpc_h": finder_cmp.get("per_halo_center_separation_summary_mpc_h"),
                            "per_halo_match_reason": finder_cmp.get("per_halo_match_reason"),
                            "mass_definition_pairs": {
                                name: {
                                    key: value for key, value in record.items()
                                    if key not in {
                                        "gadget_mass_msun_h", "rockstar_mass_msun_h",
                                        "rockstar_to_gadget_ratio", "gadget_catalogue_index",
                                        "rockstar_catalogue_index", "rockstar_halo_id",
                                    }
                                }
                                for name, record in finder_cmp.get(
                                    "per_halo_mass_definition_pairs", {},
                                ).items()
                            },
                            "subfind_central_pair": (
                                {
                                    "status": "MEASURED",
                                    "matched_halo_count": int(len(finder_cmp["per_halo_subfind_central_pair"]["gadget_group_index"])),
                                    "membership_link": "Group/GroupFirstSub -> Subhalo row",
                                    "median_subfind_to_fof_ratio": float(np.median(
                                        finder_cmp["per_halo_subfind_central_pair"]["subfind_to_fof_ratio"]
                                    )),
                                    "median_rockstar_to_fof_ratio": float(np.median(
                                        finder_cmp["per_halo_subfind_central_pair"]["rockstar_to_fof_ratio"]
                                    )),
                                    "median_subfind_to_rockstar_ratio": float(np.median(
                                        finder_cmp["per_halo_subfind_central_pair"]["subfind_to_rockstar_ratio"]
                                    )),
                                }
                                if finder_cmp.get("per_halo_subfind_central_pair") is not None
                                else {"status": "UNAVAILABLE"}
                            ),
                        }
                    else:
                        summary["skipped"]["halo_finder_comparison_%03d" % snap] = finder_cmp["reason"]

            mass_definition_cmp = halo_mass_definition_comparison(run_dir, cfg, snap)
            if mass_definition_cmp["status"] == "COMPLETE":
                comparison_group = g.create_group("halo_mass_definition_comparison")
                comparison_group.attrs["status"] = "MEASURED"
                comparison_group.attrs["mass_component"] = mass_definition_cmp["mass_component"]
                comparison_group.attrs["particle_count_cut"] = mass_definition_cmp["particle_count_cut"]
                comparison_group.attrs["host_halo_count"] = mass_definition_cmp["host_halo_count"]
                comparison_group.attrs["interpretation"] = mass_definition_cmp["interpretation"]
                comparison_group.create_dataset("halo_id", data=mass_definition_cmp["halo_id"])
                mass_group = comparison_group.create_group("mass_msun_h")
                for field, values in mass_definition_cmp["mass_msun_h"].items():
                    dataset = mass_group.create_dataset(field, data=values)
                    dataset.attrs["definition"] = mass_definition_cmp["definitions"][field]
                    dataset.attrs["units"] = "Msun/h"
                pair_group = comparison_group.create_group("pairwise_ratios")
                summary_pairs = {}
                for pair_name, pair in mass_definition_cmp["pairwise"].items():
                    pair_output = pair_group.create_group(pair_name)
                    pair_output.attrs["numerator"] = pair["numerator"]
                    pair_output.attrs["denominator"] = pair["denominator"]
                    pair_output.attrs["valid_halo_count"] = pair["valid_halo_count"]
                    pair_output.attrs["median_mass_ratio"] = (
                        np.nan if pair["median_mass_ratio"] is None else pair["median_mass_ratio"]
                    )
                    pair_output.attrs["mass_ratio_percentile_16_84"] = np.asarray([
                        np.nan if value is None else value
                        for value in pair["mass_ratio_percentile_16_84"]
                    ], dtype=float)
                    pair_output.attrs["mean_log10_mass_ratio"] = (
                        np.nan if pair["mean_log10_mass_ratio"] is None else pair["mean_log10_mass_ratio"]
                    )
                    pair_output.create_dataset("halo_id", data=pair["halo_id"])
                    ratio_dataset = pair_output.create_dataset("mass_ratio", data=pair["mass_ratio"])
                    ratio_dataset.attrs["units"] = "dimensionless"
                    pair_output.create_dataset("log10_mass_ratio", data=pair["log10_mass_ratio"])
                    summary_pairs[pair_name] = {
                        key: value for key, value in pair.items()
                        if key not in {"halo_id", "mass_ratio", "log10_mass_ratio"}
                    }
                item["halo_mass_definition_comparison"] = {
                    "status": "MEASURED",
                    "host_halo_count": mass_definition_cmp["host_halo_count"],
                    "particle_count_cut": mass_definition_cmp["particle_count_cut"],
                    "definitions": mass_definition_cmp["definitions"],
                    "pairwise": summary_pairs,
                    "interpretation": mass_definition_cmp["interpretation"],
                }
            else:
                summary["skipped"]["halo_mass_definition_comparison_%03d" % snap] = mass_definition_cmp["reason"]

            concentration = plot_concentration_mass(run_dir, cfg, snap)
            if concentration["status"] == "COMPLETE":
                g.create_dataset("concentration_mass_msun_h", data=concentration["mass_msun_h"])
                g.create_dataset("concentration_cvir", data=concentration["concentration_cvir"])
                if "halo_id" in concentration:
                    g.create_dataset("concentration_halo_id", data=concentration["halo_id"])
                item["concentration"] = {"status": "COMPLETE", "halo_count": int(len(concentration["mass_msun_h"])), "definition": "c_vir = Rvir / Rs; mass = Mvir"}
            else:
                summary["skipped"]["concentration_mass_%03d" % snap] = concentration["reason"]

            pk = plot_pk(run_dir, cfg, snap, files)
            if pk:
                for key, dataset in [
                    ("k_h_Mpc", "pk_k"), ("P_Mpc_h3", "pk_P"), ("delta2", "pk_delta2"),
                    ("mode_count", "pk_mode_count"), ("fold_segment", "pk_fold_segment"),
                    ("delta2_raw_shot_included", "pk_delta2_raw_shot_included"),
                    ("delta2_shot_noise", "pk_delta2_shot_noise"),
                    ("P_raw_shot_included_Mpc_h3", "pk_P_raw_shot_included"),
                    ("P_shot_noise_Mpc_h3", "pk_P_shot_noise"),
                    ("delta2_gaussian_1sigma", "pk_delta2_gaussian_1sigma"),
                    ("P_gaussian_1sigma_Mpc_h3", "pk_P_gaussian_1sigma"),
                    ("fold_band_lower_h_Mpc", "pk_fold_band_lower_h_Mpc"),
                    ("fold_band_upper_h_Mpc", "pk_fold_band_upper_h_Mpc"),
                    ("P_native_column_Mpc_h3", "pk_P_native_column"),
                    ("native_column_normalization_relative_residual", "pk_native_normalization_relative_residual"),
                ]:
                    g.create_dataset(dataset, data=pk[key])
                particle_type_spectra = pk.get("particle_type_spectra", {})
                if particle_type_spectra:
                    type_group = g.create_group("pk_particle_type_spectra")
                    for particle_type, type_spectrum in sorted(particle_type_spectra.items()):
                        type_data = type_group.create_group(particle_type)
                        for key, value in type_spectrum.items():
                            if isinstance(value, np.ndarray):
                                type_data.create_dataset(key, data=value)
                            elif isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating)):
                                type_data.attrs[key] = value
                    type_group.attrs["interpretation"] = (
                        "mass-weighted particle-type auto-spectra; these do not sum to the all-types total "
                        "because the total also contains inter-type cross-power"
                    )
                g.attrs["pk_estimator"] = "GADGET-4 native full-particle folded estimator"
                g.attrs["pk_shot_noise_subtracted"] = True
                g.attrs["pk_shot_noise_source"] = "GADGET-measured variable-particle-mass shot-noise column"
                g.attrs["pk_uncertainty_method"] = pk["uncertainty_method"]
                g.attrs["pk_mode_count_precision"] = pk["mode_count_precision"]
                g.attrs["pk_native_column_normalization"] = pk["native_column_normalization"]
                g.attrs["pk_native_column_max_relative_normalization_residual"] = float(
                    pk["native_column_max_relative_normalization_residual"]
                )
                g.attrs["pk_particle_selection"] = pk["particle_selection"]
                g.attrs["pk_fold_factor"] = int(pk["fold_factor"])
                g.attrs["pk_box_size_mpc_h"] = float(pk["box_mpc_h"])
                g.attrs["pk_pm_grid"] = int(pk["pm_grid"])
                g.attrs["pk_fundamental_h_mpc"] = float(pk["k_fundamental_h_Mpc"])
                g.attrs["pk_nyquist_h_mpc"] = float(pk["k_nyquist_h_Mpc"])
                g.attrs["pk_selected_k_min_h_mpc"] = float(pk["selected_k_min_h_Mpc"])
                g.attrs["pk_selected_k_max_h_mpc"] = float(pk["selected_k_max_h_Mpc"])
                g.attrs["pk_nonpositive_signal_bin_count"] = int(pk["nonpositive_shot_subtracted_bin_count"])
                g.attrs["pk_reliable_range_status"] = "not established by the fold-band selection; requires observable-level convergence"
                g.attrs["pk_source"] = pk["source"]
                item["power_spectrum"] = {
                    "status": "COMPLETE",
                    "estimator": "GADGET-4 native mass-weighted CIC-deconvolved folded spectrum",
                    "particle_selection": pk["particle_selection"],
                    "shot_noise": "GADGET-measured, variable-particle-mass estimate; subtracted from signal",
                    "uncertainty_method": pk["uncertainty_method"],
                    "mode_count_precision": pk["mode_count_precision"],
                    "particle_type_spectra": sorted(particle_type_spectra),
                    "total_spectrum_note": (
                        "the all-types total includes inter-type cross-power; type auto-spectra are not additive"
                    ),
                    "native_column_normalization": pk["native_column_normalization"],
                    "native_column_max_relative_normalization_residual": pk[
                        "native_column_max_relative_normalization_residual"
                    ],
                    "selected_k_min_h_Mpc": pk["selected_k_min_h_Mpc"],
                    "selected_k_max_h_Mpc": pk["selected_k_max_h_Mpc"],
                    "nonpositive_shot_subtracted_bin_count": pk["nonpositive_shot_subtracted_bin_count"],
                    "reliable_range_status": "not established; requires observable-level convergence",
                }
            corr = plot_two_point(run_dir, cfg, snap)
            if corr["status"] == "COMPLETE":
                for source_key, dataset_name in [
                    ("r", "xi_r"), ("xi", "xi"), ("bin_edges", "xi_bin_edges"),
                    ("pair_count", "xi_pair_count"),
                    ("random_pair_expectation", "xi_random_pair_expectation"),
                    ("xi_poisson_68_lower", "xi_poisson_68_lower"),
                    ("xi_poisson_68_upper", "xi_poisson_68_upper"),
                ]:
                    g.create_dataset(dataset_name, data=corr[source_key])
                g.attrs["xi_estimator"] = corr["method"]
                g.attrs["xi_uncertainty_method"] = corr["uncertainty_method"]
                g.attrs["xi_catalog_halo_count"] = corr["used_point_count"]
                g.attrs["xi_input_point_count"] = corr["input_point_count"]
                g.attrs["xi_excluded_nonfinite_position_count"] = corr["excluded_nonfinite_point_count"]
                g.attrs["xi_excluded_close_pair_count"] = corr["excluded_close_pair_count"]
                g.attrs["xi_minimum_separation_mpc_h"] = corr["minimum_separation_mpc_h"]
                g.attrs["xi_minimum_separation_source"] = corr["minimum_separation_source"]
                g.attrs["xi_maximum_separation_mpc_h"] = corr["maximum_separation_mpc_h"]
                g.attrs["xi_covariance_status"] = "not estimated; pair counts share halos and are correlated between bins"
                item["two_point_correlation"] = {
                    "status": "COMPLETE", "halo_count": corr["halo_count"],
                    "excluded_close_pair_count": corr["excluded_close_pair_count"],
                    "minimum_separation_mpc_h": corr["minimum_separation_mpc_h"],
                    "minimum_separation_source": corr["minimum_separation_source"],
                    "maximum_separation_mpc_h": corr["maximum_separation_mpc_h"],
                    "estimator": corr["method"],
                    "uncertainty": corr["uncertainty_method"],
                }
            else:
                summary["skipped"]["two_point_correlation_%03d" % snap] = corr["reason"]
            bias = plot_halo_bias(run_dir, cfg, snap, files)
            if bias["status"] == "COMPLETE":
                for source_key, dataset_name in [
                    ("k_h_Mpc", "bias_k"), ("bias", "bias"),
                    ("bias_gaussian_68_sigma", "bias_gaussian_68_sigma"),
                    ("bias_auto_hh_mm", "bias_auto_hh_mm"),
                    ("bias_auto_hh_mm_gaussian_68_sigma", "bias_auto_hh_mm_gaussian_68_sigma"),
                    ("cross_correlation_coefficient_hm", "bias_cross_correlation_coefficient_hm"),
                    ("cross_power_Mpc_h3", "bias_cross_power_Mpc_h3"),
                    ("matter_power_Mpc_h3", "bias_matter_power_Mpc_h3"),
                    ("halo_power_Mpc_h3", "bias_halo_power_Mpc_h3"),
                    ("mode_count", "bias_mode_count"),
                    ("bin_edges_h_Mpc", "bias_bin_edges_h_Mpc"),
                ]:
                    g.create_dataset(dataset_name, data=bias[source_key])
                g.attrs["halo_bias_estimator"] = bias["method"]
                g.attrs["halo_bias_mass_definition"] = bias["mass_definition"]
                g.attrs["halo_bias_particle_selection_definition"] = bias["halo_particle_selection_definition"]
                g.attrs["halo_bias_center_definition"] = bias["halo_center_definition"]
                g.attrs["halo_bias_grid_size"] = bias["grid_size"]
                g.attrs["halo_bias_halo_count"] = bias["halo_count"]
                g.attrs["halo_bias_minimum_halo_particles"] = bias["minimum_halo_particles"]
                g.attrs["halo_bias_matter_particle_count"] = bias["matter_particle_count"]
                g.attrs["halo_bias_matter_shot_noise_Mpc_h3"] = bias["matter_shot_noise_Mpc_h3"]
                g.attrs["halo_bias_halo_shot_noise_Mpc_h3"] = bias["halo_shot_noise_Mpc_h3"]
                g.attrs["halo_bias_nyquist_fraction"] = bias["nyquist_fraction"]
                g.attrs["halo_bias_min_modes_per_bin"] = bias["min_modes_per_bin"]
                g.attrs["halo_bias_selected_k_min_h_mpc"] = bias["selected_k_min_h_Mpc"]
                g.attrs["halo_bias_selected_k_max_h_mpc"] = bias["selected_k_max_h_Mpc"]
                g.attrs["halo_bias_uncertainty"] = bias["uncertainty"]
                item["halo_bias"] = {
                    "status": "COMPLETE", "halo_count": bias["halo_count"],
                    "matter_particle_count": bias["matter_particle_count"],
                    "grid_size": bias["grid_size"], "estimator": bias["method"],
                    "mass_definition": bias["mass_definition"],
                    "halo_particle_selection_definition": bias["halo_particle_selection_definition"],
                    "halo_center_definition": bias["halo_center_definition"],
                    "selected_k_min_h_Mpc": bias["selected_k_min_h_Mpc"],
                    "selected_k_max_h_Mpc": bias["selected_k_max_h_Mpc"],
                    "uncertainty": bias["uncertainty"],
                }
            else:
                summary["skipped"]["halo_bias_%03d" % snap] = bias["reason"]
                item["halo_bias"] = {"status": "SKIPPED", "reason": bias["reason"]}
            particle_speed = plot_particle_velocity_distribution(run_dir, snap, files)
            if particle_speed is not None:
                g.create_dataset("dm_speed_bin_edges", data=particle_speed["bin_edges"])
                g.create_dataset("dm_speed_probability_density", data=particle_speed["probability_density"])
                g.attrs["dm_speed_sample_count"] = particle_speed["sample_count"]
            hv = plot_halo_velocity_distribution(run_dir, snap)
            if hv is not None:
                g.create_dataset("halo_speed_bin_edges", data=hv["bin_edges"])
                g.create_dataset("halo_speed_probability_density", data=hv["probability_density"])
                g.attrs["halo_speed_sample_count"] = hv["sample_count"]
            spatial = plot_spatial_distribution(run_dir, cfg, snap)
            if spatial is not None:
                g.create_dataset("host_halo_positions_mpc_h", data=spatial[0])
                g.create_dataset("host_halo_mass_spatial_msun_h", data=spatial[1])
            sub = plot_subhalo_mass_function(run_dir, cfg, snap, wdm_reliability)
            if sub:
                g.create_dataset("subhmf_mass", data=sub[0]); g.create_dataset("subhmf_dn", data=sub[1])
            j = plot_spin_distribution(run_dir, snap)
            if j is not None: g.create_dataset("specific_angular_momentum", data=j)
            profiles = top_halo_profiles(run_dir, cfg, snap, files)
            item["dark_matter_profiles"] = {
                "status": profiles["status"],
                "reason": profiles.get("reason"),
                "halo_count": profiles.get("halo_count", 0),
                "selected_valid_halo_count": profiles.get("selected_valid_halo_count", 0),
                "sampled_particle_count": profiles.get("sampled_particle_count", 0),
                "snapshot_particle_count": profiles.get("snapshot_particle_count", 0),
                "velocity_status": profiles.get("velocity_status", "SKIPPED"),
                "velocity_reason": profiles.get("velocity_reason"),
            }
            if profiles["status"] == "COMPLETE":
                profile_group = g.create_group("halo_profiles")
                profile_group.attrs["mass_definition"] = profiles["mass_definition"]
                profile_group.attrs["center_definition"] = profiles["center_definition"]
                profile_group.attrs["radius_coordinate"] = "comoving Mpc/h"
                profile_group.attrs["density_definition"] = profiles["density_definition"]
                profile_group.attrs["uncertainty_definition"] = profiles["uncertainty_definition"]
                profile_group.attrs["radial_profile_aperture"] = "2 R200c; profiles skipped if this exceeds L/2"
                profile_group.attrs["minimum_sampled_particles_in_aperture"] = 100
                profile_group.attrs["inner_radius_policy"] = "min(max(0.01 * outer radius, 1e-4 Mpc/h), outer radius/2); particles inside are excluded from shells and their sampled-weight estimates are recorded"
                profile_group.attrs["profile_status"] = profiles["status"]
                profile_group.attrs["selected_valid_halo_count"] = profiles["selected_valid_halo_count"]
                profile_group.attrs["sampled_particle_count"] = profiles["sampled_particle_count"]
                profile_group.attrs["snapshot_particle_count"] = profiles["snapshot_particle_count"]
                profile_group.attrs["sampled_particle_weight_sum"] = profiles["sampled_particle_weight_sum"]
                profile_group.attrs["invalid_particle_position_count"] = profiles.get("invalid_particle_position_count", 0)
                profile_group.attrs["skipped_halo_periodic_geometry_count"] = profiles.get("skipped_halo_periodic_geometry_count", 0)
                profile_group.attrs["skipped_halo_low_particle_count"] = profiles.get("skipped_halo_low_particle_count", 0)
                profile_group.attrs["particle_mass_conversion"] = profiles["particle_mass_conversion"]
                profile_group.attrs["scale_factor"] = profiles["scale_factor"]
                profile_group.attrs["proper_density_scale_factor_power"] = -3
                for index, profile_data in enumerate(profiles["profiles"]):
                    halo_profile = profile_group.create_group("halo_%03d" % index)
                    for name in [
                        "halo_catalogue_row", "mass_msun_h", "center_group_pos_comoving_mpc_h",
                        "r200c_comoving_mpc_h", "sampled_particles_in_aperture",
                        "estimated_particles_in_aperture", "sampled_particles_inside_inner_cut",
                        "estimated_particles_inside_inner_cut", "estimated_mass_inside_inner_cut_msun_h",
                        "profile_inner_radius_comoving_mpc_h", "profile_outer_radius_comoving_mpc_h",
                    ]:
                        halo_profile.attrs[name] = profile_data[name]
                    for name in [
                        "radial_bin_edges_comoving_mpc_h", "radius_comoving_mpc_h",
                        "sampled_particle_count_per_shell", "particle_count_estimate_per_shell",
                        "dm_mass_shell_msun_h", "dm_mass_density_comoving_msun_h_per_mpc_h3",
                        "dm_mass_density_proper_msun_h_per_mpc_h3",
                    ]:
                        halo_profile.create_dataset(name, data=profile_data[name])
                dispersion_group = g.create_group("velocity_dispersion_profiles")
                dispersion_group.attrs["status"] = profiles["velocity_status"]
                dispersion_group.attrs["definition"] = "mass-weighted 3-D peculiar-velocity RMS about the shell mean; sqrt(sum of three component variances), not the 1-D dispersion"
                dispersion_group.attrs["unit"] = "km/s"
                dispersion_group.attrs["velocity_conversion"] = profiles["velocity_conversion"]
                if profiles.get("velocity_reason"):
                    dispersion_group.attrs["reason"] = profiles["velocity_reason"]
                for index, profile in enumerate(profiles["velocity_dispersion_profiles"]):
                    halo_profile = dispersion_group.create_group("halo_%03d" % index)
                    halo_profile.attrs["halo_catalogue_row"] = profile["halo_catalogue_row"]
                    halo_profile.attrs["mass_msun_h"] = profile["mass_msun_h"]
                    halo_profile.attrs["r200c_comoving_mpc_h"] = profile["r200c_comoving_mpc_h"]
                    halo_profile.create_dataset("radial_bin_edges_comoving_mpc_h", data=profile["radial_bin_edges_comoving_mpc_h"])
                    halo_profile.create_dataset("radius_comoving_mpc_h", data=profile["radius_comoving_mpc_h"])
                    halo_profile.create_dataset("velocity_dispersion_3d_peculiar_km_s", data=profile["velocity_dispersion_3d_peculiar_km_s"])
                    halo_profile.create_dataset("sampled_velocity_particle_count_per_shell", data=profile["sampled_velocity_particle_count_per_shell"])
                    halo_profile.create_dataset("velocity_particle_count_estimate_per_shell", data=profile["velocity_particle_count_estimate_per_shell"])
            else:
                item["dark_matter_profiles"]["status"] = profiles["status"]
                summary["skipped"]["dark_matter_profiles_%03d" % snap] = profiles.get("reason")
            gas_profiles = top_gas_profiles(run_dir, cfg, snap, files)
            item["gas_profiles"] = {
                "status": gas_profiles["status"],
                "reason": gas_profiles.get("reason"),
                "temperature_status": gas_profiles["temperature_status"],
                "temperature_reason": gas_profiles["temperature_reason"],
                "velocity_status": gas_profiles["velocity_status"],
                "velocity_reason": gas_profiles["velocity_reason"],
                "halo_count": gas_profiles.get("halo_count", 0),
                "sampled_particle_count": gas_profiles.get("sampled_particle_count", 0),
                "snapshot_particle_count": gas_profiles.get("snapshot_particle_count", 0),
            }
            if gas_profiles["status"] == "COMPLETE":
                gas_group = g.create_group("gas_profiles")
                gas_group.attrs["status"] = gas_profiles["status"]
                gas_group.attrs["temperature_status"] = gas_profiles["temperature_status"]
                gas_group.attrs["velocity_status"] = gas_profiles["velocity_status"]
                gas_group.attrs["density_definition"] = gas_profiles["density_definition"]
                gas_group.attrs["temperature_definition"] = gas_profiles["temperature_definition"]
                gas_group.attrs["velocity_definition"] = gas_profiles["velocity_definition"]
                gas_group.attrs["temperature_source"] = gas_profiles.get("temperature_source") or "unavailable"
                gas_group.attrs["temperature_reason"] = gas_profiles.get("temperature_reason") or ""
                gas_group.attrs["velocity_reason"] = gas_profiles.get("velocity_reason") or ""
                gas_group.attrs["sampled_particle_count"] = gas_profiles["sampled_particle_count"]
                gas_group.attrs["snapshot_particle_count"] = gas_profiles["snapshot_particle_count"]
                gas_group.attrs["sampled_particle_weight_sum"] = gas_profiles["sampled_particle_weight_sum"]
                gas_group.attrs["profile_halo_count"] = gas_profiles["halo_count"]
                gas_group.attrs["selected_valid_halo_count"] = gas_profiles["selected_valid_halo_count"]
                gas_group.attrs["selected_halo_catalogue_rows"] = gas_profiles["selected_halo_catalogue_rows"]
                gas_group.attrs["center_definition"] = gas_profiles["center_definition"]
                gas_group.attrs["mass_definition"] = gas_profiles["mass_definition"]
                gas_group.attrs["particle_selection"] = gas_profiles["particle_selection"]
                gas_group.attrs["uncertainty_definition"] = gas_profiles["uncertainty_definition"]
                gas_group.attrs["scale_factor"] = gas_profiles["scale_factor"]
                gas_group.attrs["invalid_gas_position_count"] = gas_profiles.get("invalid_gas_position_count", 0)
                gas_group.attrs["skipped_halo_periodic_geometry_count"] = gas_profiles.get("skipped_halo_periodic_geometry_count", 0)
                gas_group.attrs["skipped_halo_low_particle_count"] = gas_profiles["skipped_halo_low_particle_count"]
                gas_group.attrs["minimum_sampled_gas_particles_in_aperture"] = 100
                gas_group.attrs["radial_profile_aperture"] = "2 R200c; profiles skipped if this exceeds L/2"
                gas_group.attrs["inner_radius_policy"] = "min(max(0.01 * outer radius, 1e-4 Mpc/h), outer radius/2); particles inside are excluded from shells and their sampled-weight estimates are recorded"
                gas_group.attrs["maximum_profile_halos"] = int(cfg.get("analysis", {}).get("halo_profile_max_halos", 8))
                gas_group.attrs["maximum_sampled_gas_particles"] = int(cfg.get("analysis", {}).get("halo_profile_max_particles", 4_000_000))
                gas_group.attrs["radial_bin_count"] = int(cfg.get("analysis", {}).get("halo_profile_radial_bins", 24))
                gas_group.attrs["density_unit"] = "Msun/h per (Mpc/h)^3"
                gas_group.attrs["temperature_unit"] = "K"
                gas_group.attrs["velocity_unit"] = "km/s"
                scatter_group = gas_group.create_group("halo_to_halo_scatter")
                scatter_group.attrs["interpretation"] = gas_profiles["uncertainty_definition"]
                for metric_name, scatter in gas_profiles.get("profile_halo_to_halo_scatter", {}).items():
                    metric_group = scatter_group.create_group(metric_name)
                    metric_group.attrs["status"] = scatter["status"]
                    if scatter.get("reason"):
                        metric_group.attrs["reason"] = scatter["reason"]
                    if scatter.get("method"):
                        metric_group.attrs["method"] = scatter["method"]
                    for dataset_name in (
                        "radius_over_r200c", "valid_halo_count", "lower_16", "median", "upper_84",
                    ):
                        if dataset_name in scatter:
                            metric_group.create_dataset(dataset_name, data=scatter[dataset_name])
                for index, halo_profile in enumerate(gas_profiles["profiles"]):
                    gas_halo = gas_group.create_group("halo_%03d" % index)
                    for attr in [
                        "halo_catalogue_row", "mass_msun_h", "center_group_pos_comoving_mpc_h",
                        "r200c_comoving_mpc_h", "sampled_particles_in_aperture",
                        "estimated_particles_in_aperture", "estimated_gas_mass_in_aperture_msun_h",
                        "sampled_particles_inside_inner_cut", "estimated_particles_inside_inner_cut",
                        "estimated_gas_mass_inside_inner_cut_msun_h", "profile_inner_radius_comoving_mpc_h",
                        "profile_outer_radius_comoving_mpc_h",
                    ]:
                        gas_halo.attrs[attr] = halo_profile[attr]
                    for name in [
                        "radial_bin_edges_comoving_mpc_h", "radius_comoving_mpc_h", "gas_particle_count",
                        "gas_particle_count_estimate", "gas_mass_shell_msun_h",
                        "gas_mass_density_comoving_msun_h_per_mpc_h3",
                        "gas_mass_density_proper_msun_h_per_mpc_h3",
                        "gas_temperature_mass_weighted_K", "gas_temperature_particle_count",
                        "gas_temperature_particle_count_estimate",
                        "gas_velocity_dispersion_3d_peculiar_km_s", "gas_velocity_particle_count",
                        "gas_velocity_particle_count_estimate",
                    ]:
                        if name in halo_profile:
                            gas_halo.create_dataset(name, data=halo_profile[name])
            else:
                summary["skipped"]["gas_profiles_%03d" % snap] = gas_profiles["reason"]
            shape = halo_shapes(run_dir, cfg, snap, files)
            item["halo_shape"] = {
                "status": shape["status"], "reason": shape.get("reason"),
                "halo_count": shape.get("halo_count", 0),
                "sampled_particle_count": shape.get("sampled_particle_count", 0),
                "snapshot_particle_count": shape.get("snapshot_particle_count", 0),
            }
            if shape["status"] == "COMPLETE":
                rows = shape["profiles"]
                g.create_dataset("shape_mass_m200c_msun_h", data=[row["mass_msun_h"] for row in rows])
                g.create_dataset("shape_b_over_a", data=[row["b_over_a"] for row in rows])
                g.create_dataset("shape_c_over_a", data=[row["c_over_a"] for row in rows])
                g.create_dataset("shape_triaxiality", data=[row["triaxiality"] for row in rows])
                g.create_dataset("shape_dm_particle_count", data=[row["sampled_particle_count"] for row in rows])
                g.attrs["shape_mass_definition"] = shape["mass_definition"]
                g.attrs["shape_method"] = shape["profiles"][0]["tensor_definition"]
                g.attrs["shape_triaxiality_definition"] = shape["triaxiality_definition"]
                shape_group = g.create_group("halo_shape_profiles")
                _write_product_hdf5(shape_group, "population_summary", shape["population_summary"])
                shape_group.attrs["status"] = shape["status"]
                shape_group.attrs["mass_definition"] = shape["mass_definition"]
                shape_group.attrs["center_definition"] = shape["center_definition"]
                shape_group.attrs["particle_selection"] = shape["particle_selection"]
                shape_group.attrs["axis_ratio_definition"] = shape["axis_ratio_definition"]
                shape_group.attrs["triaxiality_definition"] = shape["triaxiality_definition"]
                shape_group.attrs["tensor_definition"] = rows[0]["tensor_definition"]
                shape_group.attrs["uncertainty_definition"] = shape["uncertainty_definition"]
                shape_group.attrs["minimum_sampled_particles"] = shape["minimum_sampled_particles"]
                shape_group.attrs["sampled_particle_count"] = shape["sampled_particle_count"]
                shape_group.attrs["snapshot_particle_count"] = shape["snapshot_particle_count"]
                shape_group.attrs["sampled_particle_weight_sum"] = shape["sampled_particle_weight_sum"]
                shape_group.attrs["skipped_periodic_geometry_count"] = shape.get("skipped_periodic_geometry_count", 0)
                shape_group.attrs["skipped_low_particle_count"] = shape["skipped_low_particle_count"]
                for index, row in enumerate(rows):
                    halo_group = shape_group.create_group("halo_%03d" % index)
                    for name in [
                        "halo_catalogue_row", "mass_msun_h", "center_group_pos_comoving_mpc_h",
                        "r200c_comoving_mpc_h", "b_over_a", "c_over_a", "triaxiality",
                        "sampled_particle_count", "estimated_particle_count", "estimated_mass_msun_h",
                    ]:
                        halo_group.attrs[name] = row[name]
                    halo_group.create_dataset("tensor_eigenvalues_mpc_h2", data=row["tensor_eigenvalues_mpc_h2"])
                    halo_group.create_dataset("principal_axis_eigenvectors_columns", data=row["principal_axis_eigenvectors_columns"])
            else:
                summary["skipped"]["halo_shape_%03d" % snap] = shape.get("reason")
            hydro = hydro_diagnostics(run_dir, cfg, snap, files)
            for source_key, dataset_name in [
                ("gas_density_log10_nH_cm3_bin_edges", "gas_density_log10_nH_cm3_bin_edges"),
                ("gas_density_probability_density", "gas_density_probability_density"),
                ("temperature_log10_bin_edges", "temperature_log10_bin_edges"),
                ("temperature_probability_density", "temperature_probability_density"),
                ("phase_log10_density_bin_edges", "phase_log10_density_bin_edges"),
                ("phase_log10_temperature_bin_edges", "phase_log10_temperature_bin_edges"),
                ("phase_particle_count_2d", "phase_particle_count_2d"),
                ("phase_particle_count_estimate_2d", "phase_particle_count_estimate_2d"),
            ]:
                if source_key in hydro:
                    g.create_dataset(dataset_name, data=hydro[source_key], compression="gzip" if np.asarray(hydro[source_key]).ndim > 1 else None)
            if "phase_density_units" in hydro:
                g.attrs["phase_density_units"] = hydro["phase_density_units"]
            if "phase_temperature_units" in hydro:
                g.attrs["phase_temperature_units"] = hydro["phase_temperature_units"]
            field_map_result = hydro.get("hydro_field_maps", {})
            if isinstance(field_map_result, dict):
                field_map_group = g.create_group("hydro_field_maps")
                field_map_group.attrs["status"] = field_map_result.get("status", "SKIPPED")
                if field_map_result.get("reason"):
                    field_map_group.attrs["reason"] = str(field_map_result["reason"])
                field_summaries = {}
                for field_name, field_result in field_map_result.get("fields", {}).items():
                    target = field_map_group.create_group(str(field_name))
                    target.attrs["status"] = field_result.get("status", "SKIPPED")
                    target.attrs['source_unit_attributes_json']=json.dumps(field_result.get('source_unit_attributes'),allow_nan=False)
                    if field_result.get('native_to_output_factor') is not None:
                        target.attrs['native_to_output_factor']=field_result['native_to_output_factor']
                    for attr_name in [
                        "reason", "source_field", "valid_particle_count", "grid_size",
                        "projection", "units", "figure",
                    ]:
                        if field_result.get(attr_name) is not None:
                            target.attrs[attr_name] = field_result[attr_name]
                    for data_name in ["pixel_values", "pixel_values_log10"]:
                        if data_name in field_result:
                            target.create_dataset(data_name, data=field_result[data_name], compression="gzip")
                    for range_name in ["color_range", "color_range_symmetric"]:
                        if range_name in field_result:
                            target.create_dataset(range_name, data=field_result[range_name])
                    field_summaries[str(field_name)] = {
                        key: value for key, value in field_result.items()
                        if key not in {"pixel_values", "pixel_values_log10"}
                    }
                    if field_result.get("status") != "COMPLETE":
                        summary["skipped"]["hydro_%s_%03d" % (field_name, snap)] = field_result.get("reason", "unavailable")
                item["hydro_field_maps"] = {
                    "status": field_map_result.get("status", "SKIPPED"),
                    "reason": field_map_result.get("reason"),
                    "fields": field_summaries,
                }
            if "gas_density_status" in hydro:
                g.attrs["gas_density_status"] = hydro["gas_density_status"]
                g.attrs["gas_density_quantity"] = hydro.get("gas_density_quantity", "unavailable")
                if "gas_density_reason" in hydro:
                    g.attrs["gas_density_reason"] = hydro["gas_density_reason"]
                item["gas_density"] = {
                    key: hydro[key] for key in ["gas_density_status", "gas_density_quantity", "gas_density_reason"]
                    if key in hydro
                }
            if "temperature_status" in hydro:
                g.attrs["temperature_status"] = hydro["temperature_status"]
                if "temperature_source" in hydro:
                    g.attrs["temperature_source"] = hydro["temperature_source"]
                if "temperature_conversion_fallback" in hydro:
                    g.attrs["temperature_conversion_fallback"] = hydro["temperature_conversion_fallback"]
                if "temperature_reason" in hydro:
                    g.attrs["temperature_reason"] = hydro["temperature_reason"]
                if "temperature_valid_sample_count" in hydro:
                    g.attrs["temperature_valid_sample_count"] = hydro["temperature_valid_sample_count"]
                if "temperature_excluded_star_forming_count" in hydro:
                    g.attrs["temperature_excluded_star_forming_count"] = hydro["temperature_excluded_star_forming_count"]
                if "temperature_excluded_invalid_sfr_count" in hydro:
                    g.attrs["temperature_excluded_invalid_sfr_count"] = hydro["temperature_excluded_invalid_sfr_count"]
                if hydro.get("temperature_source", "").startswith("internal_energy_derived"):
                    g.attrs["temperature_gamma"] = 5.0 / 3.0
                    g.attrs["temperature_hydrogen_mass_fraction"] = 0.76
                item["gas_temperature"] = {
                    key: hydro[key] for key in [
                        "temperature_status", "temperature_source", "temperature_reason",
                        "temperature_conversion_fallback",
                        "temperature_valid_sample_count", "temperature_excluded_star_forming_count",
                        "temperature_excluded_invalid_sfr_count",
                    ] if key in hydro
                }
                if hydro.get("temperature_source", "").startswith("internal_energy_derived"):
                    item["gas_temperature"]["gamma"] = 5.0 / 3.0
                    item["gas_temperature"]["hydrogen_mass_fraction"] = 0.76
                if hydro["temperature_status"] != "COMPLETE":
                    summary["skipped"]["gas_temperature_%03d" % snap] = hydro.get(
                        "temperature_reason", hydro["temperature_status"]
                    )
            if "sfr_status" in hydro:
                # Preserve raw and physical results separately; never label a raw
                # dataset sum as Msun/yr unless its GADGET unit metadata passed.
                g.attrs["sfr_total_snapshot_units"] = hydro["sfr_total_snapshot_units"]
                g.attrs["sfr_total_raw_code_value"] = hydro["sfr_total_raw_code_value"]
                g.attrs["sfr_quantity"] = hydro["sfr_quantity"]
                g.attrs["sfr_particle_count"] = hydro["sfr_particle_count"]
                g.attrs["sfr_finite_count"] = hydro["sfr_finite_count"]
                g.attrs["sfr_invalid_value_count"] = hydro["sfr_invalid_value_count"]
                g.attrs["sfr_negative_value_count"] = hydro["sfr_negative_value_count"]
                g.attrs["sfr_summed_value_count"] = hydro["sfr_summed_value_count"]
                g.attrs["sfr_status"] = hydro["sfr_status"]
                g.attrs["sfr_physical_status"] = hydro.get("sfr_physical_status", "UNAVAILABLE")
                if "sfr_physical_reason" in hydro:
                    g.attrs["sfr_physical_reason"] = hydro["sfr_physical_reason"]
                if "sfr_dataset_to_cgs_factor" in hydro:
                    g.attrs["sfr_dataset_to_cgs_factor"] = hydro["sfr_dataset_to_cgs_factor"]
                sfr_summary = {
                    key: hydro[key] for key in [
                        "sfr_total_snapshot_units", "sfr_quantity", "sfr_particle_count", "sfr_finite_count",
                        "sfr_total_raw_code_value", "sfr_invalid_value_count", "sfr_negative_value_count",
                        "sfr_summed_value_count", "sfr_status", "sfr_physical_status",
                        "sfr_physical_reason", "sfr_physical_quantity", "sfr_dataset_to_cgs_factor", "sfr_reason",
                    ]
                    if key in hydro
                }
                if "sfr_total_msun_per_year" in hydro:
                    g.attrs["sfr_total_msun_per_year"] = hydro["sfr_total_msun_per_year"]
                    g.attrs["sfr_physical_quantity"] = hydro["sfr_physical_quantity"]
                    sfr_summary["sfr_total_msun_per_year"] = hydro["sfr_total_msun_per_year"]
                    sfr_summary["sfr_physical_quantity"] = hydro["sfr_physical_quantity"]
                item["star_formation"] = sfr_summary
            bf = baryon_fractions(run_dir, snap)
            if bf:
                plt.figure(figsize=(8,6)); good=np.isfinite(bf[1]); plt.semilogx(bf[0][good], bf[1][good], ".", alpha=.3, label="gas"); good=np.isfinite(bf[2]); plt.semilogx(bf[0][good], bf[2][good], ".", alpha=.3, label="stars"); plt.legend(); plt.xlabel(r"$M_{halo}\ [M_\odot/h]$"); plt.ylabel("mass fraction"); savefig(results/("baryon_fractions_%03d.png"%snap))
                g.create_dataset("baryon_fraction_halo_mass_msun_h", data=bf[0])
                g.create_dataset("gas_mass_fraction", data=bf[1])
                g.create_dataset("stellar_mass_fraction", data=bf[2])

            summary["snapshots"][str(snap)] = item
            from core.data import attach_legacy_analysis_units,attach_analysis_identity_contracts
            attach_analysis_identity_contracts(g,'legacy',snapshot=int(snap),
                identity_namespaces=analysis_contract['identity_namespaces'],epoch=epoch,
                provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json'})
            attach_legacy_analysis_units(g,epoch=epoch,
                provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json',
                    'source_snapshot_storage_index':int(snap),
                    'source_inventory_sidecar':'analysis_provenance.json'})
        schedule_group = h5out.create_group("snapshot_redshift_schedule")
        schedule_group.attrs["status"] = redshift_schedule["status"]
        schedule_group.attrs["match_method"] = redshift_schedule["method"]
        schedule_group.attrs["proper_time_status"] = "CLASS_BACKGROUND_INTERPOLATED" if time_source else "UNAVAILABLE"
        if time_source:
            schedule_group.attrs["proper_time_source"] = time_source
        if redshift_schedule["reason"] is not None:
            schedule_group.attrs["reason"] = redshift_schedule["reason"]
        if redshift_schedule["requested_redshifts"] is not None:
            schedule_group.create_dataset("requested_redshifts", data=redshift_schedule["requested_redshifts"])
        if redshift_schedule_rows:
            rows = np.asarray(redshift_schedule_rows, dtype=float)
            schedule_group.create_dataset("snapshot", data=rows[:, 0].astype(np.int64))
            schedule_group.create_dataset("requested_redshift", data=rows[:, 1]).attrs["units"] = "dimensionless"
            schedule_group.create_dataset("actual_redshift", data=rows[:, 2]).attrs["units"] = "dimensionless"
            schedule_group.create_dataset("offset_actual_minus_requested", data=rows[:, 3]).attrs["units"] = "dimensionless"
            schedule_group.create_dataset("requested_scale_factor", data=rows[:, 4]).attrs["units"] = "dimensionless"
            schedule_group.create_dataset("actual_scale_factor", data=rows[:, 5]).attrs["units"] = "dimensionless"
            if time_source:
                schedule_group.create_dataset("requested_cosmic_time_gyr", data=rows[:, 6]).attrs["units"] = "Gyr"
                schedule_group.create_dataset("actual_cosmic_time_gyr", data=rows[:, 7]).attrs["units"] = "Gyr"

        sfr_history = build_star_formation_history(summary["snapshots"])
        summary["star_formation_history"] = sfr_history
        if sfr_history["status"] == "COMPLETE":
            plot_star_formation_history(run_dir, sfr_history)
        else:
            summary["skipped"]["star_formation_history"] = sfr_history["reason"]
        write_star_formation_history_hdf5(h5out, sfr_history,run_dir=run_dir,
            snapshot_epochs={key:item['epoch_contract'] for key,item in summary['snapshots'].items()},
            provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json','source_inventory_sidecar':'analysis_provenance.json'})

        stellar_history = build_stellar_particle_history(run_dir, summary["snapshots"])
        summary["stellar_particle_history"] = stellar_history
        if stellar_history["status"] == "COMPLETE":
            plot_stellar_particle_history(run_dir, stellar_history)
        else:
            summary["skipped"]["stellar_particle_history"] = stellar_history["reason"]
        write_stellar_particle_history_hdf5(h5out, stellar_history,run_dir=run_dir,
            snapshot_epochs={key:item['epoch_contract'] for key,item in summary['snapshots'].items()},
            provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json','source_inventory_sidecar':'analysis_provenance.json'})

        history_result = main_progenitor_histories(run_dir, cfg, summary["snapshots"])
        if history_result["status"] == "COMPLETE":
            summary["mass_accretion_histories"] = {
                "status": "COMPLETE", "final_snapshot": history_result["final_snapshot"],
                "halo_count": len(history_result["histories"]),
                "formation_redshift_count": sum(row["formation_redshift"] is not None for row in history_result["histories"]),
                "source": history_result["source"],
            }
            all_mergers = [event for row in history_result["histories"] for event in row["merger_events"]]
            summary["merger_histories"] = {
                "status": "COMPLETE", "event_count": len(all_mergers),
                "major_event_count": sum(event["mass_ratio"] >= 0.25 for event in all_mergers),
                "source": history_result["source"],
            }
            histories_group = h5out.create_group("main_progenitor_histories")
            histories_group.attrs["final_snapshot"] = history_result["final_snapshot"]
            histories_group.attrs["particle_floor"] = history_result["particle_floor"]
            histories_group.attrs["mass_definition"] = "Mvir"
            histories_group.attrs["source"] = history_result["source"]
            for index, history in enumerate(history_result["histories"]):
                track_group = histories_group.create_group("halo_%04d" % index)
                from core.data import exact_uint64_ids
                final_id=exact_uint64_ids([history['final_halo_id']])[0]
                track_group.attrs["final_halo_id"] = final_id
                track_group.create_dataset('final_halo_id',data=final_id)
                track_group.attrs["final_mass_msun_h"] = history["final_mass_msun_h"]
                track_group.attrs["formation_redshift_status"] = "MEASURED" if history["formation_redshift"] is not None else "BEFORE_FIRST_SNAPSHOT_OR_UNMEASURED"
                if history["formation_redshift"] is not None:
                    track_group.attrs["formation_redshift_half_mass"] = history["formation_redshift"]
                track = history["track"]
                track_group.create_dataset("snapshot", data=[point["snapshot"] for point in track])
                track_group.create_dataset("redshift", data=[point["redshift"] for point in track])
                track_group.create_dataset("scale_factor", data=[point["scale_factor"] for point in track])
                track_group.create_dataset("halo_id", data=exact_uint64_ids([point["halo_id"] for point in track],unique=False))
                track_group.create_dataset("mvir_msun_h", data=[point["mass_msun_h"] for point in track])
                merger_group = track_group.create_group("merger_events")
                for event_index, event in enumerate(history["merger_events"]):
                    event_group = merger_group.create_group("event_%04d" % event_index)
                    for field in ["redshift", "main_mass_msun_h", "secondary_mass_msun_h", "mass_ratio"]:
                        event_group.attrs[field] = event[field]
                    for field in ['descendant_id','main_progenitor_id','secondary_progenitor_id']:
                        exact=exact_uint64_ids([event[field]])[0]
                        event_group.attrs[field]=exact
                        event_group.create_dataset(field,data=exact)
            from core.data import attach_time_series_contract
            from core.cosmology import run_proper_time_clock
            attach_time_series_contract(histories_group,clock=run_proper_time_clock(run_dir),
                snapshot_epochs={key:item['epoch_contract'] for key,item in summary['snapshots'].items()},
                identity_namespaces=analysis_contract['identity_namespaces'],
                provenance={'analysis_contract_hdf5_attribute':'/::data_contract_json','source_inventory_sidecar':'analysis_provenance.json'})
            plot_mass_accretion_histories(run_dir, history_result)
            plot_merger_histories(run_dir, history_result)
        else:
            summary["skipped"]["mass_accretion_histories"] = history_result["reason"]
            summary["skipped"]["merger_histories"] = history_result["reason"]
    except BaseException as exc:
        h5out.close()
        from core.foundation import atomic_json
        atomic_json(results/('analysis_failure_'+uuid.uuid4().hex+'.json'),
                    {'status':'FAIL','science_acceptance':'NOT_ACCEPTED','measurement':str(analysis_tmp),
                     'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    finally:
        if h5out.id.valid:
            h5out.close()

    galaxy_cfg = cfg.get("galaxies", {})
    if (str(cfg.get("simulation", {}).get("physics", "dm")).lower() == "hydro"
            and bool(galaxy_cfg.get("enabled", True))):
        from halos import build_galaxy_catalogues

        galaxy_manifest = build_galaxy_catalogues(
            run_dir,
            [int(value) for value in summary["snapshots"]],
            minimum_star_particles=galaxy_cfg.get("min_star_particles", 100),
            minimum_dm_particles=galaxy_cfg.get("min_dm_particles", 300),
        )
        statuses = [str(item.get("status", "UNAVAILABLE")) for item in galaxy_manifest["snapshots"]]
        galaxy_manifest["status"] = (
            "COMPLETE" if statuses and all(value == "COMPLETE" for value in statuses)
            else "PARTIAL" if any(value == "COMPLETE" for value in statuses)
            else "UNAVAILABLE"
        )
        summary["galaxy_catalogue"] = {
            "status": galaxy_manifest["status"],
            "path": "galaxy_catalogue_manifest.json",
            "snapshot_count": len(statuses),
            "galaxy_count": int(sum(item.get("galaxy_count", 0) for item in galaxy_manifest["snapshots"])),
            "resolved_galaxy_count": int(sum(item.get("resolved_galaxy_count", 0) for item in galaxy_manifest["snapshots"])),
            "unavailable_snapshots": [
                {"snapshot": item.get("snapshot"), "reason": item.get("reason")}
                for item in galaxy_manifest["snapshots"] if item.get("status") != "COMPLETE"
            ],
        }

    summary_tmp = results / (".analysis_summary_" + uuid.uuid4().hex + ".json.tmp")
    try:
        from core.snapshot_tables import write_snapshot_tables
        summary['snapshot_tables'] = write_snapshot_tables(summary['snapshots'], results)
        plot_time_evolution(run_dir, summary)
        from figures import generate_single_run_story_figures, plot_saved_analysis_curves
        summary['saved_unit_figures']=plot_saved_analysis_curves(analysis_tmp,results,final_source_name='analysis.h5')
        summary["paper_figure_recipes"] = generate_single_run_story_figures(
            analysis_tmp, results / "paper_figures", cfg.get("visualization", {}), source_run=run_dir,
        )
        if include_pdf_report:
            summary["report_pdf"] = "summary.pdf"
            create_analysis_report_pdf(results, summary, cfg, source_run=run_dir, model=model)
        from figures import review_figure_folder
        summary["figure_review"] = review_figure_folder(results, create_gallery=include_pdf_report)
        with open(summary_tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        _verify_analysis_inputs(run_dir,analysis_inputs,analysis_contract)
        with analysis_tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(analysis_tmp, results / "analysis.h5")
        os.replace(summary_tmp, results / "analysis_summary.json")
        _write_analysis_provenance(
            results, run_dir, ["all"], analysis_created_utc, code_sha256=analysis_code_sha256, inputs=analysis_inputs, data_contract=analysis_contract,
        )
        verify_analysis_bundle(results,verify_sources=True)
        from core.foundation import fsync_directory
        fsync_directory(results)
    except BaseException as exc:
        from core.foundation import atomic_json
        atomic_json(results/('analysis_failure_'+uuid.uuid4().hex+'.json'),
                    {'status':'FAIL','science_acceptance':'NOT_ACCEPTED','measurement':str(analysis_tmp),
                     'summary':str(summary_tmp),'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return summary


# ----------------------------- paired comparison -----------------------------

def _positive_series(group, x_name, y_name):
    if x_name not in group or y_name not in group:
        return None
    x = np.asarray(group[x_name][:], dtype=float)
    y = np.asarray(group[y_name][:], dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        return None
    return _clean_positive_series(x, y)


def _clean_positive_series(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        return None
    good = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    x, y = x[good], y[good]
    if len(x) < 2:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, first = np.unique(x, return_index=True)
    x, y = unique, y[first]
    return (x, y) if len(x) >= 2 else None


def _valid_hmf_mass_interval(mass_range):
    if not isinstance(mass_range, (list, tuple)) or len(mass_range) != 2:
        return False
    try:
        lower, upper = map(float, mass_range)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(lower) and np.isfinite(upper) and lower > 0 and upper > lower)


def _common_hmf_bin_comparison(group_a, group_b, volume_a, volume_b, bins, mass_range=None,
                               phase_agreement=None):
    """Re-bin two selected host catalogs on one shared mass interval."""
    if mass_range is None:
        return {"status": "SKIPPED", "reason": "a fixed, predeclared shared HMF mass interval is required"}
    required = ["host_halo_mass_unfiltered_msun_h", "host_halo_in_hmf"]
    if any(name not in group_a or name not in group_b for name in required):
        return None
    masses_a = np.asarray(group_a[required[0]][:], dtype=float)
    masses_b = np.asarray(group_b[required[0]][:], dtype=float)
    mask_a = np.asarray(group_a[required[1]][:], dtype=bool)
    mask_b = np.asarray(group_b[required[1]][:], dtype=bool)
    if masses_a.shape != mask_a.shape or masses_b.shape != mask_b.shape:
        return {"status": "SKIPPED", "reason": "saved HMF selection masks do not align with halo masses"}
    masses_a = masses_a[mask_a & np.isfinite(masses_a) & (masses_a > 0)]
    masses_b = masses_b[mask_b & np.isfinite(masses_b) & (masses_b > 0)]
    if len(masses_a) < 1 or len(masses_b) < 1:
        return {"status": "SKIPPED", "reason": "one or both selected catalogs contain no positive masses"}
    if not _valid_hmf_mass_interval(mass_range):
        raise ValueError("analysis.comparison_hmf_mass_range_msun_h must contain [minimum, maximum]")
    lower_mass, upper_mass = map(float, mass_range)
    if not np.isfinite(lower_mass) or not np.isfinite(upper_mass) or lower_mass <= 0 or upper_mass <= lower_mass:
        raise ValueError("analysis.comparison_hmf_mass_range_msun_h must be finite, positive, and increasing")
    volume_a, volume_b = float(volume_a), float(volume_b)
    if not np.isfinite(volume_a) or not np.isfinite(volume_b) or volume_a <= 0 or volume_b <= 0:
        raise ValueError("HMF comparison volumes must be finite and positive")
    binning = "shared logarithmic bins over the configured mass interval"
    bins = int(bins)
    if bins < 1:
        raise ValueError("common HMF comparison bin count must be positive")
    edges = np.geomspace(lower_mass, upper_mass, bins + 1)
    counts_a = np.histogram(masses_a, bins=edges)[0]
    counts_b = np.histogram(masses_b, bins=edges)[0]
    dlog = np.diff(np.log10(edges))
    rate_a = counts_a / (volume_a * dlog)
    rate_b = counts_b / (volume_b * dlog)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ratio = np.full(bins, np.nan)
    lower_ratio = np.full(bins, np.nan)
    upper_ratio = np.full(bins, np.nan)
    populated = (counts_a > 0) & (counts_b > 0)
    ratio[populated] = rate_b[populated] / rate_a[populated]
    lower_ratio[:], upper_ratio[:] = poisson_rate_ratio_interval(
        counts_a, counts_b, float(volume_a) * dlog, float(volume_b) * dlog
    )
    jackknife = {"status": "UNAVAILABLE", "reason": "measured matched initial-condition phases and equal periodic volumes are required"}
    if (isinstance(phase_agreement, dict) and phase_agreement.get("status") == "PASS"
            and np.isclose(float(volume_a), float(volume_b), rtol=1.0e-12, atol=0.0)):
        box_mpc_h = float(volume_a) ** (1.0 / 3.0)
        jackknife = _paired_hmf_jackknife_log_sigma(group_a, group_b, edges, box_mpc_h)
    jackknife_sigma = np.asarray(jackknife.get("log_ratio_sigma", []), dtype=float)
    if jackknife.get("status") == "COMPLETE" and jackknife_sigma.shape == ratio.shape:
        critical = float(student_t.ppf(0.5 + 0.682689492137 / 2.0, jackknife["jackknife_regions"] - 1))
        paired_lower = np.exp(np.log(ratio) - critical * jackknife_sigma)
        paired_upper = np.exp(np.log(ratio) + critical * jackknife_sigma)
        paired_lower[~populated] = np.nan
        paired_upper[~populated] = np.nan
        paired_covariance = jackknife["log_ratio_covariance"]
        covariance_a = jackknife["log_hmf_covariance_a"]
        covariance_b = jackknife["log_hmf_covariance_b"]
        cross_covariance = jackknife["log_hmf_cross_covariance_a_b"]
        jackknife_status = "COMPLETE"
        jackknife_method = jackknife["method"]
    else:
        paired_lower = np.full(bins, np.nan)
        paired_upper = np.full(bins, np.nan)
        paired_covariance = np.full((bins, bins), np.nan)
        covariance_a = np.full((bins, bins), np.nan)
        covariance_b = np.full((bins, bins), np.nan)
        cross_covariance = np.full((bins, bins), np.nan)
        jackknife_status = jackknife.get("status", "UNAVAILABLE")
        jackknife_method = jackknife.get("reason", "paired spatial jackknife did not produce aligned bins")

    return {
        "status": "COMPLETE",
        "mass_edges_msun_h": edges.tolist(),
        "mass_centers_msun_h": centers,
        "counts_a": counts_a,
        "counts_b": counts_b,
        "dn_dlogM_a": rate_a,
        "dn_dlogM_b": rate_b,
        "ratio_b_over_a": ratio,
        "ratio_poisson_68_lower": lower_ratio,
        "ratio_poisson_68_upper": upper_ratio,
        "poisson_interval": "68.27% exact conditional interval under the independent-Poisson count model; does not include paired sample covariance",
        "ratio_paired_jackknife_68_lower": paired_lower,
        "ratio_paired_jackknife_68_upper": paired_upper,
        "log_ratio_paired_jackknife_covariance": paired_covariance,
        "log_hmf_jackknife_covariance_a": covariance_a,
        "log_hmf_jackknife_covariance_b": covariance_b,
        "log_hmf_jackknife_cross_covariance_a_b": cross_covariance,
        "paired_jackknife_status": jackknife_status,
        "paired_jackknife_method_or_reason": jackknife_method,
        "finite_volume_sample_variance": "spatial delete-one jackknife covariance of log HMF rates; includes subvolume count variation and must not be added to Poisson intervals as an independent term",
        "binning": binning,
    }


def _resolution_source_compatibility(run_a, run_b):
    """Require matching solver, simulation-code, and analysis provenance."""
    def valid_digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    manifest_paths = [Path(run) / "manifest.json" for run in (run_a, run_b)]
    if not all(path.is_file() for path in manifest_paths):
        return {"status": "UNAVAILABLE", "reason": "both run manifests are required for a resolution claim"}
    try:
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
        if any(not isinstance(manifest, dict) for manifest in manifests):
            return {"status": "UNAVAILABLE", "reason": "run manifests must be JSON objects"}
        # Older inventories omitted nested scientific modules. Matching their
        # small subset cannot establish that the full implementations match.
        if any(manifest.get("project_source_inventory_version") != 2 for manifest in manifests):
            return {"status": "UNAVAILABLE", "reason": "complete nested-source provenance was not recorded by both runs"}
        required_files = {'evolution.py', "cosmology.py", "halos.py", "patches/gadget4-radiation.patch", "software-revisions"}
        hashes = [manifest.get("project_source_sha256") for manifest in manifests]
        if any(not isinstance(item, dict) for item in hashes):
            return {"status": "UNAVAILABLE", "reason": "run manifests lack project source checksums"}
        for item in hashes:
            if any(not isinstance(key, str) or not key or not valid_digest(value) for key, value in item.items()):
                return {"status": "UNAVAILABLE", "reason": "source inventory contains missing or malformed file identities"}
            missing = sorted(required_files - set(item))
            if missing:
                return {"status": "UNAVAILABLE", "reason": "run provenance lacks checksums for " + ", ".join(missing)}
        if set(hashes[0]) != set(hashes[1]):
            return {"status": "INCOMPATIBLE", "reason": "recorded project source file inventories differ"}
        changed = sorted(key for key in hashes[0] if hashes[0][key] != hashes[1][key])
        if changed:
            return {"status": "INCOMPATIBLE", "reason": "project source or setup inputs differ", "changed_files": changed}
        revisions_a = manifests[0].get("software_revisions")
        revisions_b = manifests[1].get("software_revisions")
        if not revisions_a or revisions_a != revisions_b:
            return {"status": "INCOMPATIBLE", "reason": "pinned upstream software revisions differ or are missing"}
        commits_a = manifests[0].get("software_commits")
        commits_b = manifests[1].get("software_commits")
        if not commits_a or commits_a != commits_b:
            return {"status": "INCOMPATIBLE", "reason": "built upstream commits differ or are missing"}
        inputs_a, inputs_b = manifests[0].get("input_artifacts"), manifests[1].get("input_artifacts")
        if not isinstance(inputs_a, dict) or not isinstance(inputs_b, dict):
            return {"status": "UNAVAILABLE", "reason": "run manifests lack input-artifact checksums"}
        spectrum_a, spectrum_b = inputs_a.get("linear_power_spectrum"), inputs_b.get("linear_power_spectrum")
        if not isinstance(spectrum_a, dict) or not isinstance(spectrum_b, dict):
            return {"status": "UNAVAILABLE", "reason": "both runs must record the exact linear input spectrum"}
        if any(not valid_digest(spectrum.get("sha256")) or isinstance(spectrum.get("redshift"), bool)
               or not isinstance(spectrum.get("redshift"), (int, float))
               or not math.isfinite(spectrum["redshift"]) for spectrum in (spectrum_a, spectrum_b)):
            return {"status": "UNAVAILABLE", "reason": "linear spectrum identity or reference redshift is missing or malformed"}
        if spectrum_a.get("sha256") != spectrum_b.get("sha256") or spectrum_a.get("redshift") != spectrum_b.get("redshift"):
            return {"status": "INCOMPATIBLE", "reason": "linear input-spectrum checksums or reference redshifts differ"}
        for artifact_name in ("transfer_functions", "boltzmann_background"):
            artifact_a, artifact_b = inputs_a.get(artifact_name), inputs_b.get(artifact_name)
            if (artifact_a is None) != (artifact_b is None):
                return {"status": "INCOMPATIBLE", "reason": "%s is present in only one run" % artifact_name}
            if artifact_a is not None and (
                not isinstance(artifact_a, dict) or not isinstance(artifact_b, dict)
                or not valid_digest(artifact_a.get("sha256")) or not valid_digest(artifact_b.get("sha256"))
                or artifact_a.get("sha256") != artifact_b.get("sha256")
            ):
                return {"status": "INCOMPATIBLE", "reason": "%s checksums differ" % artifact_name}
        gadget_build_hashes = []
        for manifest in manifests:
            handoff = manifest.get("gadget_handoff", {})
            if not isinstance(handoff, dict):
                return {"status": "UNAVAILABLE", "reason": "native executable handoff metadata is malformed"}
            compiled = handoff.get("compiled_options_file", {})
            executable = handoff.get("executable", {})
            compiled_hash = compiled.get("sha256") if isinstance(compiled, dict) else None
            executable_hash = executable.get("sha256") if isinstance(executable, dict) else None
            if not valid_digest(compiled_hash) or not valid_digest(executable_hash):
                return {"status": "UNAVAILABLE", "reason": "run manifests lack the exact GADGET executable and compile-option checksums"}
            gadget_build_hashes.append((compiled_hash, executable_hash))
        if gadget_build_hashes[0] != gadget_build_hashes[1]:
            return {"status": "INCOMPATIBLE", "reason": "GADGET executable or compile-option checksums differ"}
        hashes_analysis = []
        for run in (run_a, run_b):
            analysis_path = Path(run) / "results/analysis.h5"
            if not analysis_path.is_file():
                return {"status": "UNAVAILABLE", "reason": "both saved full analyses are required"}
            with h5py.File(analysis_path, "r") as handle:
                hashes_analysis.append(str(handle.attrs.get("analysis_script_sha256", "")))
        if not all(valid_digest(value) for value in hashes_analysis) or hashes_analysis[0] != hashes_analysis[1]:
            return {"status": "INCOMPATIBLE", "reason": "analysis-script checksums differ or are missing"}
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "UNAVAILABLE", "reason": "could not validate run provenance: " + str(exc)}
    return {
        "status": "MATCH",
        "project_source_checksums": "matching complete recorded first-party source and setup inventories",
        "project_source_files_compared": len(hashes[0]),
        "upstream_revisions": revisions_a,
        "linear_input_spectrum_sha256": spectrum_a["sha256"],
        "gadget_compiled_options_sha256": gadget_build_hashes[0][0],
        "gadget_executable_sha256": gadget_build_hashes[0][1],
        "analysis_script_sha256": hashes_analysis[0],
    }


def _hmf_resolution_pair_eligibility(cfg_a, cfg_b, phase, source_compatibility):
    """Accept only matched-phase runs that differ in numerical resolution controls."""
    if source_compatibility.get("status") != "MATCH":
        return {"status": "SKIPPED", "reason": source_compatibility.get("reason", "matching source provenance is required")}
    if phase.get("status") != "PASS":
        return {"status": "SKIPPED", "reason": "initial-condition phases did not pass the paired-run check"}
    setup = compare_requested_run_setups(cfg_a, cfg_b)
    categories = (
        "cosmology", "dark_energy", "warm_dark_matter", "linear_spectrum",
        "hydro", "halo_selection", "halo_finding",
    )
    def differs(row):
        return row.get("status") == "DIFFERENT" or row.get("run_a") != row.get("run_b")

    for category in categories:
        rows = setup.get(category, {})
        mismatches = [key for key, row in rows.items() if differs(row)]
        if mismatches:
            return {"status": "SKIPPED", "reason": "%s settings differ or are missing: %s" % (category, ", ".join(mismatches))}
    simulation = setup.get("simulation", {})
    allowed_resolution_changes = {
        "particles_per_dimension", "pm_grid", "softening_kpc_h", "softening_max_physical_kpc_h",
    }
    mismatches = [
        key for key, row in simulation.items()
        if key not in allowed_resolution_changes and differs(row)
    ]
    if mismatches:
        return {"status": "SKIPPED", "reason": "non-resolution simulation settings differ or are missing: " + ", ".join(mismatches)}
    n_a = cfg_a.get("simulation", {}).get("particles_per_dimension")
    n_b = cfg_b.get("simulation", {}).get("particles_per_dimension")
    if (isinstance(n_a, bool) or not isinstance(n_a, int)
            or isinstance(n_b, bool) or not isinstance(n_b, int) or n_a < 2 or n_b < 2 or n_a == n_b):
        return {"status": "SKIPPED", "reason": "particle grids must be recorded and differ between the two runs"}
    low_n, high_n = sorted((n_a, n_b))
    if high_n / low_n < 1.5:
        return {"status": "SKIPPED", "reason": "higher-resolution particle grid is less than 1.5 times finer per dimension"}
    analysis_a, analysis_b = cfg_a.get("analysis", {}), cfg_b.get("analysis", {})
    mass_range_a = analysis_a.get("comparison_hmf_mass_range_msun_h")
    mass_range_b = analysis_b.get("comparison_hmf_mass_range_msun_h")
    if (not isinstance(mass_range_a, (list, tuple)) or len(mass_range_a) != 2
            or mass_range_a != mass_range_b):
        return {"status": "SKIPPED", "reason": "both runs must declare the same fixed HMF convergence mass interval"}
    for key, default in [
        ("hmf_resolution_relative_tolerance", 0.05),
        ("hmf_resolution_confidence", 0.95),
        ("hmf_resolution_min_halos_per_bin", 20),
        ("hmf_resolution_required_consecutive_bins", 3),
    ]:
        if analysis_a.get(key, default) != analysis_b.get(key, default):
            return {"status": "SKIPPED", "reason": "HMF resolution-equivalence settings differ between runs"}
    return {
        "status": "READY", "particle_grid_a": n_a, "particle_grid_b": n_b,
        "low_resolution_run": "a" if n_a < n_b else "b",
        "high_resolution_run": "b" if n_a < n_b else "a",
        "mass_interval_msun_h": [float(mass_range_a[0]), float(mass_range_a[1])],
        "resolution_changes_allowed": sorted(allowed_resolution_changes),
    }


def _paired_hmf_jackknife_log_sigma(group_a, group_b, mass_edges, box_mpc_h, bins_per_axis=2):
    """Estimate paired log-HMF-ratio uncertainty by deleting periodic subvolumes."""
    fields = ("host_halo_mass_unfiltered_msun_h", "host_halo_in_hmf", "host_halo_positions_all_mpc_h")
    if any(name not in group_a or name not in group_b for name in fields):
        return {"status": "UNAVAILABLE", "reason": "full host masses, HMF selection masks, and positions are required"}
    edges = np.asarray(mass_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 3 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        return {"status": "UNAVAILABLE", "reason": "shared HMF mass edges are invalid"}
    box = float(box_mpc_h)
    if not np.isfinite(box) or box <= 0 or bins_per_axis < 2:
        return {"status": "UNAVAILABLE", "reason": "periodic box or jackknife geometry is invalid"}

    def cell_counts(group):
        mass = np.asarray(group[fields[0]][:], dtype=float)
        selected = np.asarray(group[fields[1]][:], dtype=bool)
        pos = np.asarray(group[fields[2]][:], dtype=float)
        if mass.ndim != 1 or selected.shape != mass.shape or pos.shape != (len(mass), 3):
            return None, "saved host masses, selections, and positions do not align"
        good = selected & np.isfinite(mass) & (mass > 0) & np.all(np.isfinite(pos), axis=1)
        mass, pos = mass[good], np.mod(pos[good], box)
        bin_index = np.searchsorted(edges, mass, side="right") - 1
        bin_index[mass == edges[-1]] = len(edges) - 2
        in_range = (bin_index >= 0) & (bin_index < len(edges) - 1)
        mass, pos, bin_index = mass[in_range], pos[in_range], bin_index[in_range]
        cell_xyz = np.floor(pos / box * bins_per_axis).astype(np.int64)
        cell_xyz = np.clip(cell_xyz, 0, bins_per_axis - 1)
        cell_index = (cell_xyz[:, 0] * bins_per_axis + cell_xyz[:, 1]) * bins_per_axis + cell_xyz[:, 2]
        counts = np.zeros((bins_per_axis**3, len(edges) - 1), dtype=np.int64)
        np.add.at(counts, (cell_index, bin_index), 1)
        return counts, None

    counts_a, error_a = cell_counts(group_a)
    counts_b, error_b = cell_counts(group_b)
    if error_a or error_b:
        return {"status": "UNAVAILABLE", "reason": error_a or error_b}
    total_a, total_b = np.sum(counts_a, axis=0), np.sum(counts_b, axis=0)
    if not np.array_equal(total_a, np.histogram(
            np.asarray(group_a[fields[0]][:])[np.asarray(group_a[fields[1]][:], dtype=bool)], bins=edges,
    )[0]) or not np.array_equal(total_b, np.histogram(
            np.asarray(group_b[fields[0]][:])[np.asarray(group_b[fields[1]][:], dtype=bool)], bins=edges,
    )[0]):
        return {"status": "UNAVAILABLE", "reason": "spatial jackknife counts do not reproduce the saved selected HMF counts"}
    if not np.all(total_a > 0) or not np.all(total_b > 0):
        return {"status": "UNAVAILABLE", "reason": "one or more shared HMF bins are empty in a full run"}
    loo_a, loo_b = total_a[None, :] - counts_a, total_b[None, :] - counts_b
    if np.any(loo_a <= 0) or np.any(loo_b <= 0):
        return {"status": "UNAVAILABLE", "reason": "a leave-one-subvolume-out HMF bin is empty"}
    log_hmf_a = np.log(loo_a)
    log_hmf_b = np.log(loo_b)
    log_ratio = log_hmf_b - log_hmf_a
    prefactor = (len(counts_a) - 1.0) / len(counts_a)

    def jackknife_covariance(samples):
        centered = samples - np.mean(samples, axis=0)
        return prefactor * centered.T @ centered

    covariance_a = jackknife_covariance(log_hmf_a)
    covariance_b = jackknife_covariance(log_hmf_b)
    mean_a = np.mean(log_hmf_a, axis=0)
    mean_b = np.mean(log_hmf_b, axis=0)
    cross_covariance = prefactor * (log_hmf_a - mean_a).T @ (log_hmf_b - mean_b)
    covariance = jackknife_covariance(log_ratio)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "status": "COMPLETE", "log_ratio_sigma": sigma,
        "log_ratio_covariance": covariance,
        "log_hmf_covariance_a": covariance_a,
        "log_hmf_covariance_b": covariance_b,
        "log_hmf_cross_covariance_a_b": cross_covariance,
        "log_hmf_sigma_a": np.sqrt(np.maximum(np.diag(covariance_a), 0.0)),
        "log_hmf_sigma_b": np.sqrt(np.maximum(np.diag(covariance_b), 0.0)),
        "jackknife_regions": int(len(counts_a)),
        "method": "%d^3 periodic spatial delete-one jackknife of matched-phase HMF ratios" % bins_per_axis,
    }


def hmf_resolution_completeness(group_a, group_b, common_hmf, cfg_a, cfg_b,
                                phase_agreement, source_compatibility, box_mpc_h):
    """Find a tested HMF mass interval only when a paired resolution test supports it."""
    eligibility = _hmf_resolution_pair_eligibility(
        cfg_a, cfg_b, phase_agreement, source_compatibility,
    )
    if eligibility["status"] != "READY":
        return {"status": "SKIPPED", "reason": eligibility["reason"]}
    if not isinstance(common_hmf, dict) or common_hmf.get("status") != "COMPLETE":
        return {"status": "SKIPPED", "reason": "shared-bin HMF comparison is unavailable"}
    analysis = cfg_a.get("analysis", {})
    tolerance = float(analysis.get("hmf_resolution_relative_tolerance", 0.05))
    confidence = float(analysis.get("hmf_resolution_confidence", 0.95))
    min_count = int(analysis.get("hmf_resolution_min_halos_per_bin", 20))
    required_bins = int(analysis.get("hmf_resolution_required_consecutive_bins", 3))
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("hmf_resolution_relative_tolerance must be finite and in (0, 1)")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("hmf_resolution_confidence must be finite and in (0, 1)")
    if min_count < 1 or required_bins < 2:
        raise ValueError("HMF resolution comparison requires positive per-bin counts and at least two consecutive bins")
    edges = np.asarray(common_hmf["mass_edges_msun_h"], dtype=float)
    low_key = "a" if eligibility["low_resolution_run"] == "a" else "b"
    high_key = "b" if low_key == "a" else "a"
    counts_low = np.asarray(common_hmf["counts_" + low_key], dtype=np.int64)
    counts_high = np.asarray(common_hmf["counts_" + high_key], dtype=np.int64)
    rate_low = np.asarray(common_hmf["dn_dlogM_" + low_key], dtype=float)
    rate_high = np.asarray(common_hmf["dn_dlogM_" + high_key], dtype=float)
    ratio = np.divide(rate_high, rate_low, out=np.full_like(rate_high, np.nan), where=rate_low > 0)
    jackknife = _paired_hmf_jackknife_log_sigma(group_a, group_b, edges, box_mpc_h)
    if jackknife["status"] != "COMPLETE":
        return {"status": "INCONCLUSIVE", "reason": jackknife["reason"], "eligibility": eligibility}
    sigma = np.asarray(jackknife["log_ratio_sigma"], dtype=float)
    critical = float(student_t.ppf(0.5 + confidence / 2.0, jackknife["jackknife_regions"] - 1))
    lower = np.exp(np.log(ratio) - critical * sigma)
    upper = np.exp(np.log(ratio) + critical * sigma)
    eligible_bins = (counts_low >= min_count) & (counts_high >= min_count)
    equivalent = (
        eligible_bins & np.isfinite(ratio) & np.isfinite(lower) & np.isfinite(upper)
        & (lower >= 1.0 - tolerance) & (upper <= 1.0 + tolerance)
    )
    adequate = np.flatnonzero(eligible_bins)
    start = None
    if len(adequate):
        last = int(adequate[-1])
        first = last
        while first > 0 and equivalent[first - 1]:
            first -= 1
        if equivalent[last] and last - first + 1 >= required_bins:
            start = first
    report = {
        "status": "INCONCLUSIVE",
        "reason": "the fixed interval does not contain the required consecutive bins whose paired jackknife confidence intervals lie within tolerance",
        "eligibility": eligibility,
        "relative_tolerance": tolerance,
        "confidence": confidence,
        "minimum_halos_per_resolution_per_bin": min_count,
        "required_consecutive_bins": required_bins,
        "jackknife": jackknife["method"],
        "ratio_interval_method": "two-sided Student-t interval on the log high-resolution/low-resolution ratio from periodic spatial delete-one jackknife",
        "mass_edges_msun_h": edges.tolist(),
        "counts_low_resolution": counts_low.tolist(),
        "counts_high_resolution": counts_high.tolist(),
        "ratio_high_over_low": ratio.tolist(),
        "ratio_lower": lower.tolist(),
        "ratio_upper": upper.tolist(),
        "equivalent_bin": equivalent.tolist(),
    }
    if start is not None:
        report.update({
            "status": "MEASURED_CONVERGED_RANGE",
            "reason": None,
            "completeness_mass_limit_msun_h": float(edges[start]),
            "validated_upper_mass_msun_h": float(edges[int(adequate[-1]) + 1]),
            "validated_bin_count": int(int(adequate[-1]) - start + 1),
        })
    return report


def _signed_series(group, x_name, y_name):
    if x_name not in group or y_name not in group:
        return None
    x = np.asarray(group[x_name][:], dtype=float)
    y = np.asarray(group[y_name][:], dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        return None
    good = np.isfinite(x) & (x > 0) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, first = np.unique(x, return_index=True)
    x, y = unique, y[first]
    return (x, y) if len(x) >= 2 else None


def _interp_signed_logx(xnew, x, y):
    xnew = np.asarray(xnew, dtype=float)
    out = np.full(xnew.shape, np.nan, dtype=float)
    good = np.isfinite(xnew) & (xnew > 0) & (xnew >= x[0]) & (xnew <= x[-1])
    if np.any(good):
        out[good] = np.interp(np.log(xnew[good]), np.log(x), y)
    return out


def _interp_loglog(xnew, x, y):
    """Interpolate a positive spectrum in log-x/log-y without extrapolating."""
    xnew = np.asarray(xnew, dtype=float)
    out = np.full(xnew.shape, np.nan, dtype=float)
    good = np.isfinite(xnew) & (xnew > 0) & (xnew >= x[0]) & (xnew <= x[-1])
    if np.any(good):
        out[good] = np.exp(np.interp(np.log(xnew[good]), np.log(x), np.log(y)))
    return out


def _read_match_catalogue_field(files, top, names, required=True):
    """Read one GADGET catalogue field across all numbered shards without gaps."""
    names = list(names)
    chosen = None
    any_top = False
    for path in files:
        with h5py.File(path, "r") as f:
            if top not in f:
                continue
            any_top = True
            chosen = next((name for name in names if name in f[top]), chosen)
            if chosen is not None:
                break
    if not any_top or chosen is None:
        if required:
            raise ValueError("catalogue is missing %s/%s" % (top, "/".join(names)))
        return None

    parts = []
    for path in files:
        with h5py.File(path, "r") as f:
            if top not in f or chosen not in f[top]:
                count_key = {"Group": "Ngroups_ThisFile", "Subhalo": "Nsubhalos_ThisFile"}.get(top)
                header = f.get("Header")
                if count_key and header is not None and header.attrs.get(count_key) == 0:
                    continue
                raise ValueError("catalogue shard is missing %s/%s: %s" % (top, chosen, path))
            parts.append(f[top][chosen][:])
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _read_cross_run_halo_catalogue(run_dir, snap, mass_definition, minimum_particles, max_catalogue_ids):
    """Read FOF host properties and their type-1 particle memberships.

    GADGET-4 writes /IDs/ParticleIDs ordered by FOF group, then SUBFIND subhalo
    rank, then particle type. The group and subhalo length vectors therefore
    let us recover each host's type-1 IDs exactly, including its unbound FOF
    particles. When that optional ID block is absent, the group-ordered
    snapshot plus GroupOffsetType/GroupLenType provides the same membership.
    """
    files = group_files(run_dir, snap)
    if not files:
        raise ValueError("no GADGET FOF/SUBFIND catalogue shards for snapshot %03d" % snap)

    group_mass_fields = {
        "fof": ["GroupMass"],
        "m200c": ["Group_M_Crit200"],
        "m200b": ["Group_M_Mean200"],
        "mvir": ["Group_M_TopHat200"],
    }
    mass_fields = group_mass_fields.get(str(mass_definition).lower())
    if mass_fields is None:
        raise ValueError("unsupported host mass definition for matching: %s" % mass_definition)

    # Validate split-catalogue bookkeeping and retain the physical box from the
    # catalogue header, which is independent evidence from the run manifest.
    expected_files = None
    total_ids = None
    total_groups = None
    box_sizes = []
    redshifts = []
    declared_id_counts = 0
    for path in files:
        with h5py.File(path, "r") as f:
            if "Header" not in f:
                raise ValueError("catalogue shard has no /Header group: %s" % path)
            header = f["Header"].attrs
            if "NumFiles" in header:
                count = int(header["NumFiles"])
                if expected_files is None:
                    expected_files = count
                elif expected_files != count:
                    raise ValueError("catalogue shards disagree about Header/NumFiles")
            if "Nids_Total" in header:
                count = int(header["Nids_Total"])
                if total_ids is None:
                    total_ids = count
                elif total_ids != count:
                    raise ValueError("catalogue shards disagree about Header/Nids_Total")
            if "Ngroups_Total" in header:
                count = int(header["Ngroups_Total"])
                if total_groups is None:
                    total_groups = count
                elif total_groups != count:
                    raise ValueError("catalogue shards disagree about Header/Ngroups_Total")
            if "Nids_ThisFile" in header:
                declared_id_counts += int(header["Nids_ThisFile"])
            if "BoxSize" in header:
                box_sizes.append(float(header["BoxSize"]))
            if "Redshift" in header:
                redshifts.append(float(header["Redshift"]))
    if expected_files is not None and expected_files != len(files):
        raise ValueError("catalogue is incomplete: Header/NumFiles=%d but found %d shards" % (expected_files, len(files)))
    if box_sizes and not np.allclose(box_sizes, box_sizes[0], rtol=1.0e-12, atol=0.0):
        raise ValueError("catalogue shards disagree about Header/BoxSize")
    if redshifts and not np.allclose(redshifts, redshifts[0], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("catalogue shards disagree about Header/Redshift")
    if total_groups == 0:
        return {
            "group_indices": np.empty(0, dtype=np.int64), "mass_msun_h": np.empty(0),
            "position_mpc_h": np.empty((0, 3)), "n_particles": np.empty(0, dtype=np.int64),
            "n_dm": np.empty(0, dtype=np.int64), "vmax": np.empty(0),
            "r200c_mpc_h": np.empty(0), "r200m_mpc_h": np.empty(0),
            "dm_ids": np.empty(0, dtype=np.uint64), "dm_group_indices": np.empty(0, dtype=np.int64),
            "membership_source": "no FOF host rows",
            "box_mpc_h": box_sizes[0] / 1000.0 if box_sizes else None,
            "redshift": redshifts[0] if redshifts else None,
        }

    mass = _read_match_catalogue_field(files, "Group", mass_fields)
    pos = _read_match_catalogue_field(files, "Group", ["GroupPos"])
    raw_len_type = np.asarray(_read_match_catalogue_field(files, "Group", ["GroupLenType"]))
    if raw_len_type.dtype.kind not in "ui":
        raise ValueError("Group/GroupLenType must use an integer type")
    len_type = raw_len_type.astype(np.int64, copy=False)
    if len_type.ndim != 2 or len_type.shape[1] != 6:
        raise ValueError("Group/GroupLenType must have the six GADGET particle-type columns")
    n_groups = len(len_type)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    if len(mass) != n_groups or pos.shape != (n_groups, 3):
        raise ValueError("host mass, position, and GroupLenType rows do not align")
    if np.any(len_type < 0):
        raise ValueError("Group/GroupLenType contains negative particle counts")
    group_len = np.sum(len_type, axis=1, dtype=np.int64)
    catalog_group_len = _read_match_catalogue_field(files, "Group", ["GroupLen"], required=False)
    if catalog_group_len is not None:
        raw_group_len = np.asarray(catalog_group_len)
        if raw_group_len.dtype.kind not in "ui" or not np.array_equal(raw_group_len.astype(np.int64, copy=False).reshape(-1), group_len):
            raise ValueError("Group/GroupLen disagrees with the sum of GroupLenType")
    raw_r200c = _read_match_catalogue_field(files, "Group", ["Group_R_Crit200"], required=False)
    raw_r200m = _read_match_catalogue_field(files, "Group", ["Group_R_Mean200"], required=False)
    r200c = np.full(n_groups, np.nan) if raw_r200c is None else np.asarray(raw_r200c, dtype=float).reshape(-1)
    r200m = np.full(n_groups, np.nan) if raw_r200m is None else np.asarray(raw_r200m, dtype=float).reshape(-1)
    if len(r200c) != n_groups or len(r200m) != n_groups:
        raise ValueError("spherical-overdensity radii do not align with host rows")

    # Central-subhalo Vmax is the available FOF-host Vmax proxy in this GADGET
    # output: GroupFirstSub points at rank-zero (central) SUBFIND subhalo.
    group_nsubs = _read_match_catalogue_field(files, "Group", ["GroupNsubs"], required=False)
    group_first_sub = _read_match_catalogue_field(files, "Group", ["GroupFirstSub"], required=False)
    subhalo_group = _read_match_catalogue_field(files, "Subhalo", ["SubhaloGroupNr"], required=False)
    subhalo_len_type = _read_match_catalogue_field(files, "Subhalo", ["SubhaloLenType"], required=False)
    subhalo_vmax = _read_match_catalogue_field(files, "Subhalo", ["SubhaloVmax"], required=False)
    have_subhalo_membership = all(value is not None for value in [group_nsubs, group_first_sub, subhalo_group, subhalo_len_type])
    if any(value is not None for value in [group_nsubs, group_first_sub, subhalo_group, subhalo_len_type]) and not have_subhalo_membership:
        raise ValueError("SUBFIND membership fields are incomplete; cannot decode the /IDs ordering safely")
    if have_subhalo_membership:
        raw_group_nsubs, raw_group_first_sub = np.asarray(group_nsubs), np.asarray(group_first_sub)
        raw_subhalo_group, raw_subhalo_len_type = np.asarray(subhalo_group), np.asarray(subhalo_len_type)
        if any(array.dtype.kind not in "ui" for array in [raw_group_nsubs, raw_group_first_sub, raw_subhalo_group, raw_subhalo_len_type]):
            raise ValueError("SUBFIND membership counts and indices must use integer types")
        group_nsubs = raw_group_nsubs.astype(np.int64, copy=False).reshape(-1)
        group_first_sub = raw_group_first_sub.astype(np.int64, copy=False).reshape(-1)
        subhalo_group = raw_subhalo_group.astype(np.int64, copy=False).reshape(-1)
        subhalo_len_type = raw_subhalo_len_type.astype(np.int64, copy=False)
        if len(group_nsubs) != n_groups or len(group_first_sub) != n_groups:
            raise ValueError("GroupNsubs and GroupFirstSub do not align with host rows")
        if subhalo_len_type.ndim != 2 or subhalo_len_type.shape[1] != len_type.shape[1] or len(subhalo_group) != len(subhalo_len_type):
            raise ValueError("SUBFIND membership arrays do not align by subhalo and particle type")
        if np.any(group_nsubs < 0) or np.any(subhalo_len_type < 0):
            raise ValueError("SUBFIND membership contains negative counts")
        if subhalo_vmax is not None:
            subhalo_vmax = np.asarray(subhalo_vmax, dtype=float).reshape(-1)
            if len(subhalo_vmax) != len(subhalo_group):
                raise ValueError("SubhaloVmax does not align with SubhaloGroupNr")
        else:
            subhalo_vmax = np.full(len(subhalo_group), np.nan)
    else:
        group_nsubs = np.zeros(n_groups, dtype=np.int64)
        group_first_sub = np.full(n_groups, -1, dtype=np.int64)
        subhalo_group = np.empty(0, dtype=np.int64)
        subhalo_len_type = np.empty((0, len_type.shape[1]), dtype=np.int64)
        subhalo_vmax = np.empty(0, dtype=float)

    minimum_particles = int(minimum_particles)
    eligible = (group_len >= minimum_particles) & np.isfinite(mass) & (mass > 0) & np.all(np.isfinite(pos), axis=1)
    eligible_indices = np.flatnonzero(eligible)
    if not len(eligible_indices):
        return {
            "group_indices": eligible_indices, "mass_msun_h": np.empty(0), "position_mpc_h": np.empty((0, 3)),
            "n_particles": np.empty(0, dtype=np.int64), "n_dm": np.empty(0, dtype=np.int64),
            "vmax": np.empty(0), "r200c_mpc_h": np.empty(0), "r200m_mpc_h": np.empty(0),
            "dm_ids": np.empty(0, dtype=np.uint64), "dm_group_indices": np.empty(0, dtype=np.int64),
            "membership_source": "no eligible FOF host rows",
            "box_mpc_h": box_sizes[0] / 1000.0 if box_sizes else None,
            "redshift": redshifts[0] if redshifts else None,
        }

    has_catalogue_ids = True
    catalogue_id_rows = 0
    for path in files:
        with h5py.File(path, "r") as f:
            if "IDs" not in f or "ParticleIDs" not in f["IDs"]:
                has_catalogue_ids = False
                break
            id_dataset = f["IDs/ParticleIDs"]
            if id_dataset.ndim != 1 or id_dataset.dtype.kind not in "ui":
                raise ValueError("/IDs/ParticleIDs must be a one-dimensional integer dataset")
            catalogue_id_rows += len(id_dataset)
    # Some GADGET builds create the optional IDs group but leave its dataset
    # empty. Treat that as absent only when the catalogue itself declares or
    # contains particles; partial ID output remains a hard integrity error.
    if has_catalogue_ids and catalogue_id_rows == 0 and (
        (total_ids is not None and total_ids > 0)
        or declared_id_counts > 0
        or int(np.sum(group_len, dtype=np.int64)) > 0
    ):
        has_catalogue_ids = False
    if (has_catalogue_ids and not have_subhalo_membership
            and np.any(np.sum(len_type[:, :1], axis=1) + np.sum(len_type[:, 2:], axis=1) > 0)):
        raise ValueError("FOF-only /IDs output does not preserve particle type; exact type-1 matching requires SUBFIND membership fields")
    id_upper_bound = (
        total_ids if total_ids is not None else int(np.sum(group_len, dtype=np.int64))
    ) if has_catalogue_ids else int(np.sum(len_type[eligible_indices, 1], dtype=np.int64))
    if id_upper_bound > int(max_catalogue_ids):
        raise OverflowError("catalogue membership needs %d type-1 IDs, above cross_run_match_max_catalogue_ids=%d" % (
            id_upper_bound, max_catalogue_ids,
        ))

    ids = None
    if has_catalogue_ids:
        id_parts = []
        for path in files:
            with h5py.File(path, "r") as f:
                values = f["IDs/ParticleIDs"][:]
                if values.dtype.kind not in "ui":
                    raise ValueError("/IDs/ParticleIDs must use an integer type")
                id_parts.append(values)
        ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.uint64)
        if declared_id_counts and declared_id_counts != len(ids):
            raise ValueError("sum of Header/Nids_ThisFile does not match /IDs/ParticleIDs length")
        if total_ids is not None and total_ids != len(ids):
            raise ValueError("Header/Nids_Total does not match concatenated /IDs/ParticleIDs length")
        if int(np.sum(group_len, dtype=np.int64)) != len(ids):
            raise ValueError("sum of GroupLenType does not match /IDs/ParticleIDs length")

    vmax_by_group = np.full(n_groups, np.nan)
    if have_subhalo_membership:
        for group_index, nsub in enumerate(group_nsubs):
            if nsub == 0:
                if group_first_sub[group_index] not in (-1, 0):
                    raise ValueError("a host with no subhalos has an invalid GroupFirstSub index")
                continue
            first = int(group_first_sub[group_index])
            if first < 0 or first + int(nsub) > len(subhalo_group):
                raise ValueError("GroupFirstSub and GroupNsubs point outside the SUBFIND arrays")
            if np.any(subhalo_group[first:first + int(nsub)] != group_index):
                raise ValueError("GroupFirstSub/Nsubs do not point to this host's contiguous subhalos")
            vmax_by_group[group_index] = subhalo_vmax[first]

    if not has_catalogue_ids:
        raw_offsets = _read_match_catalogue_field(files, "Group", ["GroupOffsetType"], required=False)
        if raw_offsets is None:
            raise ValueError("catalogue has no /IDs/ParticleIDs or Group/GroupOffsetType for exact type-1 membership")
        raw_offsets = np.asarray(raw_offsets)
        if raw_offsets.dtype.kind not in "ui" or raw_offsets.shape != len_type.shape:
            raise ValueError("Group/GroupOffsetType must be a six-column integer array aligned with GroupLenType")
        snapshot_files = group_ordered_snapshot_files(run_dir, snap)
        if not snapshot_files:
            raise FileNotFoundError("group-ordered snapshot is missing for catalogue particle membership")
        from core.data import read_selected_group_ids, snapshot_id_shards
        snapshot_records, _snapshot_count = snapshot_id_shards(snapshot_files)
        dm_host_rows = eligible_indices[len_type[eligible_indices, 1] > 0]
        dm_ids, dm_group_indices = read_selected_group_ids(
            raw_offsets, len_type, dm_host_rows, snapshot_records, max_ids=int(max_catalogue_ids),
        )
        if len(dm_ids) > int(max_catalogue_ids):
            raise OverflowError("selected host membership has %d type-1 IDs, above cross_run_match_max_catalogue_ids=%d" % (
                len(dm_ids), max_catalogue_ids,
            ))
        return {
            "group_indices": eligible_indices,
            "mass_msun_h": mass[eligible] * 1.0e10,
            "position_mpc_h": pos[eligible] / 1000.0,
            "n_particles": group_len[eligible],
            "n_dm": len_type[eligible, 1],
            "vmax": vmax_by_group[eligible],
            "r200c_mpc_h": r200c[eligible] / 1000.0,
            "r200m_mpc_h": r200m[eligible] / 1000.0,
            "dm_ids": dm_ids,
            "dm_group_indices": dm_group_indices,
            "membership_source": "GroupOffsetType/GroupLenType into group-ordered snapshot ParticleIDs",
            "box_mpc_h": box_sizes[0] / 1000.0 if box_sizes else None,
            "redshift": redshifts[0] if redshifts else None,
        }

    id_parts_for_hosts = []
    group_parts_for_hosts = []
    group_start = 0
    for group_index, total_count in enumerate(group_len):
        group_end = group_start + int(total_count)
        if eligible[group_index]:
            if have_subhalo_membership:
                nsub = int(group_nsubs[group_index])
                first = int(group_first_sub[group_index]) if nsub else -1
                cursor = group_start
                used_by_type = np.zeros(len_type.shape[1], dtype=np.int64)
                for sub_index in range(first, first + nsub):
                    counts = subhalo_len_type[sub_index]
                    sub_count = int(np.sum(counts, dtype=np.int64))
                    if cursor + sub_count > group_end:
                        raise ValueError("SUBFIND subhalo lengths extend beyond their parent FOF host")
                    dm_start = cursor + int(np.sum(counts[:1], dtype=np.int64))
                    dm_count = int(counts[1])
                    if dm_count:
                        id_parts_for_hosts.append(ids[dm_start:dm_start + dm_count])
                        group_parts_for_hosts.append(np.full(dm_count, group_index, dtype=np.int64))
                    used_by_type += counts
                    cursor += sub_count
                remainder = len_type[group_index] - used_by_type
                if np.any(remainder < 0) or cursor + int(np.sum(remainder, dtype=np.int64)) != group_end:
                    raise ValueError("SUBFIND lengths and FOF GroupLenType do not reconcile")
                dm_start = cursor + int(np.sum(remainder[:1], dtype=np.int64))
                dm_count = int(remainder[1])
                if dm_count:
                    id_parts_for_hosts.append(ids[dm_start:dm_start + dm_count])
                    group_parts_for_hosts.append(np.full(dm_count, group_index, dtype=np.int64))
            else:
                # Without SUBFIND, GADGET sorts FOF IDs by numeric particle ID,
                # not by particle type. This path is therefore safe only when
                # every particle in each host is type 1.
                dm_start = group_start
                dm_count = int(len_type[group_index, 1])
                if dm_count:
                    id_parts_for_hosts.append(ids[dm_start:dm_start + dm_count])
                    group_parts_for_hosts.append(np.full(dm_count, group_index, dtype=np.int64))
        group_start = group_end

    dm_ids = np.concatenate(id_parts_for_hosts) if id_parts_for_hosts else np.empty(0, dtype=ids.dtype)
    dm_group_indices = np.concatenate(group_parts_for_hosts) if group_parts_for_hosts else np.empty(0, dtype=np.int64)
    if len(dm_ids) != len(dm_group_indices):
        raise ValueError("decoded dark-matter IDs and host indices do not align")
    return {
        "group_indices": eligible_indices,
        "mass_msun_h": mass[eligible] * 1.0e10,
        "position_mpc_h": pos[eligible] / 1000.0,
        "n_particles": group_len[eligible],
        "n_dm": len_type[eligible, 1],
        "vmax": vmax_by_group[eligible],
        "r200c_mpc_h": r200c[eligible] / 1000.0,
        "r200m_mpc_h": r200m[eligible] / 1000.0,
        "dm_ids": dm_ids,
        "dm_group_indices": dm_group_indices,
        "membership_source": "GADGET /IDs/ParticleIDs catalogue block",
        "box_mpc_h": box_sizes[0] / 1000.0 if box_sizes else None,
        "redshift": redshifts[0] if redshifts else None,
    }


def _cross_run_identity(run_dir):
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return {
            "random_seed": int(manifest["random_seed"]),
            "particles_per_dimension": int(manifest["particles_per_dimension"]),
            "box_mpc_h": float(manifest["box_mpc_h"]),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def compare_requested_run_setups(cfg_a, cfg_b):
    """Report requested physics, resolution, and selection differences explicitly."""
    fields = {
        "cosmology": [
            "model", "H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio", "T_cmb",
            "N_ur", "N_ncdm", "m_ncdm_eV", "T_ncdm", "omega_radiation", "extra",
        ],
        "dark_energy": [
            "enabled", "parameterization", "f_ede", "z_c", "theta_i", "n_axion",
            "background_mode", "background_tolerance", "extra_axiclass",
        ],
        "warm_dark_matter": [
            "enabled", "mode", "model", "thermal_relic_mass_keV", "max_thermal_speed_fraction_c",
            "minimum_particles_per_half_mode",
        ],
        "linear_spectrum": ["source", "format", "k_unit", "p_unit", "redshift"],
        "simulation": [
            "physics", "box_mpc_h", "particles_per_dimension", "random_seed", "start_redshift",
            "start_redshift_guess", "end_redshift", "snapshot_z_max", "snapshot_z_min", "snapshot_dz",
            "lpt_order_dm", "lpt_order_hydro", "ic_engine_dm", "ic_engine_hydro", "pm_grid",
            "softening_kpc_h", "softening_max_physical_kpc_h", "accuracy",
            "max_size_timestep_ln_a", "err_tol_force_acc", "err_tol_int_accuracy", "err_tol_theta",
            "synchronized_outputs", "output_format",
        ],
        "hydro": [
            "cooling", "star_formation", "uv_background", "courant_factor", "require_species_transfer_functions",
            "allow_gadget_particle_split_approximation",
        ],
        "halo_selection": [
            "all", "hmf_mass_definition", "hmf_min_particles", "mass_bins", "max_accretion_histories",
            "hmf_resolution_relative_tolerance", "hmf_resolution_confidence",
            "hmf_resolution_min_halos_per_bin", "hmf_resolution_required_consecutive_bins",
            "power_grid", "halo_bias_min_halos", "halo_bias_min_modes_per_bin", "halo_bias_nyquist_fraction",
            "halo_profile_max_halos", "halo_profile_max_particles", "halo_profile_radial_bins",
            "correlation_bins", "correlation_min_halos", "correlation_min_separation_mpc_h",
            "density_grid", "rare_halo_thresholds_msun_h",
            "cross_run_match_min_shared_particles", "cross_run_match_min_overlap_fraction",
            "cross_run_match_max_catalogue_ids",
            "comparison_hmf_mass_range_msun_h",
        ],
        "halo_finding": [
            "fof", "subfind", "rockstar", "consistent_trees", "fof_linking_length", "min_particles",
            "rockstar_strict_so_masses", "rockstar_overlap_length_mpc_h",
        ],
    }
    sections = {
        "cosmology": "cosmology", "dark_energy": "ede", "warm_dark_matter": "wdm",
        "simulation": "simulation", "hydro": "hydro", "halo_selection": "analysis", "halo_finding": "halos",
        "linear_spectrum": "spectrum",
    }
    comparison = {}
    any_difference = False
    any_missing = False
    for category, keys in fields.items():
        section = sections[category]
        group_a = cfg_a.get(section, {}) if isinstance(cfg_a, dict) else {}
        group_b = cfg_b.get(section, {}) if isinstance(cfg_b, dict) else {}
        rows = {}
        for key in keys:
            value_a, value_b = group_a.get(key), group_b.get(key)
            if key not in group_a or key not in group_b:
                status = "UNAVAILABLE"
                any_missing = True
            elif value_a is None and value_b is None:
                status = "MATCH"
            elif value_a is None or value_b is None:
                status = "DIFFERENT"
                any_difference = True
            elif isinstance(value_a, (int, float)) and not isinstance(value_a, bool) and isinstance(value_b, (int, float)) and not isinstance(value_b, bool):
                same = np.isclose(float(value_a), float(value_b), rtol=1.0e-10, atol=1.0e-12)
                status = "MATCH" if same else "DIFFERENT"
            else:
                status = "MATCH" if value_a == value_b else "DIFFERENT"
            any_difference |= status == "DIFFERENT"
            rows[key] = {"run_a": value_a, "run_b": value_b, "status": status}
        comparison[category] = rows
    comparison["overall_status"] = "DIFFERENCES_REPORTED" if any_difference else "MATCH" if not any_missing else "PARTIAL_METADATA"
    comparison["scope"] = "requested YAML configuration only; resolved compiler/runtime details are recorded in each run manifest"
    return comparison


def _mutual_best_halo_matches(catalogue_a, catalogue_b, min_shared_particles, min_overlap_fraction):
    """Return deterministic reciprocal-best FOF matches from exact shared DM IDs."""
    ids_a = np.asarray(catalogue_a["dm_ids"])
    ids_b = np.asarray(catalogue_b["dm_ids"])
    host_a = np.asarray(catalogue_a["dm_group_indices"], dtype=np.int64)
    host_b = np.asarray(catalogue_b["dm_group_indices"], dtype=np.int64)
    if not len(ids_a) or not len(ids_b):
        return {"a": np.empty(0, dtype=np.int64), "b": np.empty(0, dtype=np.int64), "shared": np.empty(0, dtype=np.int64), "fraction_a": np.empty(0), "fraction_b": np.empty(0), "merit": np.empty(0), "candidate_edges": 0}
    order_a = np.argsort(ids_a, kind="stable")
    order_b = np.argsort(ids_b, kind="stable")
    sorted_a = ids_a[order_a]
    sorted_b = ids_b[order_b]
    if np.any(sorted_a[1:] == sorted_a[:-1]) or np.any(sorted_b[1:] == sorted_b[:-1]):
        raise ValueError("a type-1 particle ID appears more than once in the selected FOF hosts")
    lookup_b = np.searchsorted(sorted_b, sorted_a)
    in_range = lookup_b < len(sorted_b)
    common_a_positions = np.flatnonzero(in_range)
    common_a_positions = common_a_positions[sorted_b[lookup_b[in_range]] == sorted_a[in_range]]
    if not len(common_a_positions):
        return {"a": np.empty(0, dtype=np.int64), "b": np.empty(0, dtype=np.int64), "shared": np.empty(0, dtype=np.int64), "fraction_a": np.empty(0), "fraction_b": np.empty(0), "merit": np.empty(0), "candidate_edges": 0}
    host_a_for_common = host_a[order_a[common_a_positions]]
    host_b_for_common = host_b[order_b[lookup_b[common_a_positions]]]
    if max(int(np.max(host_a_for_common)), int(np.max(host_b_for_common))) >= 2**32:
        raise ValueError("FOF catalogue has too many groups for the compact overlap index")
    pair_key = (host_a_for_common.astype(np.uint64) << np.uint64(32)) | host_b_for_common.astype(np.uint64)
    unique_keys, shared = np.unique(pair_key, return_counts=True)
    candidate_a = (unique_keys >> np.uint64(32)).astype(np.int64)
    candidate_b = (unique_keys & np.uint64(0xffffffff)).astype(np.int64)
    eligible_rows = {int(index): row for row, index in enumerate(catalogue_a["group_indices"])}
    eligible_cols = {int(index): row for row, index in enumerate(catalogue_b["group_indices"])}
    count_a = np.asarray([catalogue_a["n_dm"][eligible_rows[int(index)]] for index in candidate_a], dtype=float)
    count_b = np.asarray([catalogue_b["n_dm"][eligible_cols[int(index)]] for index in candidate_b], dtype=float)
    fraction_a = shared / count_a
    fraction_b = shared / count_b
    merit = shared / np.sqrt(count_a * count_b)
    valid = (shared >= int(min_shared_particles)) & (fraction_a >= float(min_overlap_fraction)) & (fraction_b >= float(min_overlap_fraction))
    candidate_a, candidate_b = candidate_a[valid], candidate_b[valid]
    shared, fraction_a, fraction_b, merit = shared[valid], fraction_a[valid], fraction_b[valid], merit[valid]
    if not len(candidate_a):
        return {"a": np.empty(0, dtype=np.int64), "b": np.empty(0, dtype=np.int64), "shared": np.empty(0, dtype=np.int64), "fraction_a": np.empty(0), "fraction_b": np.empty(0), "merit": np.empty(0), "candidate_edges": 0}

    # Resolve ties by overlap merit, shared count, then stable catalogue index.
    order = np.lexsort((candidate_b, -shared, -merit, candidate_a))
    best_b_for_a = {}
    for row in order:
        best_b_for_a.setdefault(int(candidate_a[row]), int(candidate_b[row]))
    order = np.lexsort((candidate_a, -shared, -merit, candidate_b))
    best_a_for_b = {}
    for row in order:
        best_a_for_b.setdefault(int(candidate_b[row]), int(candidate_a[row]))
    mutual = np.asarray([
        best_b_for_a[int(a)] == int(b) and best_a_for_b[int(b)] == int(a)
        for a, b in zip(candidate_a, candidate_b)
    ], dtype=bool)
    return {
        "a": candidate_a[mutual], "b": candidate_b[mutual], "shared": shared[mutual],
        "fraction_a": fraction_a[mutual], "fraction_b": fraction_b[mutual], "merit": merit[mutual],
        "candidate_edges": int(len(candidate_a)),
    }


def _save_matched_halo_figures(outdir, pair_name, redshift, matches, catalogue_a, catalogue_b, box_mpc_h):
    group_rows_a = {int(index): row for row, index in enumerate(catalogue_a["group_indices"])}
    group_rows_b = {int(index): row for row, index in enumerate(catalogue_b["group_indices"])}
    rows_a = np.asarray([group_rows_a[int(index)] for index in matches["a"]], dtype=int)
    rows_b = np.asarray([group_rows_b[int(index)] for index in matches["b"]], dtype=int)
    mass_a = catalogue_a["mass_msun_h"][rows_a]
    mass_b = catalogue_b["mass_msun_h"][rows_b]
    mass_ratio = mass_b / mass_a
    pos_a = catalogue_a["position_mpc_h"][rows_a]
    pos_b = catalogue_b["position_mpc_h"][rows_b]
    delta = (pos_b - pos_a + 0.5 * box_mpc_h) % box_mpc_h - 0.5 * box_mpc_h
    displacement = np.linalg.norm(delta, axis=1)
    vmax_a, vmax_b = catalogue_a["vmax"][rows_a], catalogue_b["vmax"][rows_b]
    vmax_ratio = np.divide(vmax_b, vmax_a, out=np.full_like(vmax_b, np.nan), where=np.isfinite(vmax_a) & (vmax_a > 0) & np.isfinite(vmax_b) & (vmax_b > 0))
    x_mass = np.sqrt(mass_a * mass_b)

    plt.figure(figsize=(8, 5)); plt.scatter(x_mass, mass_ratio, s=12, alpha=0.55); plt.xscale("log"); plt.yscale("log"); plt.axhline(1.0, color="black", linewidth=1)
    plt.xlabel(r"geometric mean host mass [$M_\odot/h$]"); plt.ylabel("matched host mass ratio B / A")
    plt.title("Shared-particle FOF matches at z=%.6g" % redshift); savefig(outdir / ("matched_halo_mass_ratio_%s.png" % pair_name))

    plt.figure(figsize=(8, 5)); plt.scatter(x_mass, displacement, s=12, alpha=0.55); plt.xscale("log")
    plt.xlabel(r"geometric mean host mass [$M_\odot/h$]"); plt.ylabel("periodic position displacement [Mpc/h]")
    plt.title("Matched host position shifts at z=%.6g" % redshift); savefig(outdir / ("matched_halo_position_shift_%s.png" % pair_name))

    valid_vmax = np.isfinite(vmax_ratio) & (vmax_ratio > 0)
    if np.any(valid_vmax):
        plt.figure(figsize=(8, 5)); plt.scatter(x_mass[valid_vmax], vmax_ratio[valid_vmax], s=12, alpha=0.55); plt.xscale("log"); plt.yscale("log"); plt.axhline(1.0, color="black", linewidth=1)
        plt.xlabel(r"geometric mean host mass [$M_\odot/h$]"); plt.ylabel("central-subhalo Vmax ratio B / A")
        plt.title("Matched host Vmax shifts at z=%.6g" % redshift); savefig(outdir / ("matched_halo_vmax_ratio_%s.png" % pair_name))

    return {
        "mass_a": mass_a, "mass_b": mass_b, "mass_ratio": mass_ratio,
        "position_a": pos_a, "position_b": pos_b,
        "position_delta": delta, "position_displacement": displacement,
        "r200c_a": catalogue_a["r200c_mpc_h"][rows_a], "r200c_b": catalogue_b["r200c_mpc_h"][rows_b],
        "r200m_a": catalogue_a["r200m_mpc_h"][rows_a], "r200m_b": catalogue_b["r200m_mpc_h"][rows_b],
        "vmax_a": vmax_a, "vmax_b": vmax_b, "vmax_ratio": vmax_ratio,
    }


def _fof_main_progenitor_tracks(run_dir, snapshot_records, final_snapshot, final_group_indices,
                                mass_definition, minimum_particles, min_shared_particles,
                                min_overlap_fraction, max_catalogue_ids, expected_box_mpc_h):
    """Trace matched FOF hosts backward through exact shared type-1 particle IDs."""
    redshift_by_key = {key: float(redshift) for redshift, key in snapshot_records}
    if final_snapshot not in redshift_by_key:
        raise ValueError("final matched snapshot is missing from its analysis file")
    final_redshift = redshift_by_key[final_snapshot]
    ordered = sorted(
        [(redshift, key) for key, redshift in redshift_by_key.items()
         if redshift >= final_redshift - 1.0e-6 * max(1.0, abs(final_redshift))],
        key=lambda row: row[0], reverse=True,
    )
    final_position = next((index for index, (_, key) in enumerate(ordered) if key == final_snapshot), None)
    if final_position is None:
        raise ValueError("final matched snapshot was excluded from the FOF history time range")
    ordered = ordered[:final_position + 1]
    if len(ordered) < 2:
        raise ValueError("fewer than two analyzed snapshots are available for FOF formation histories")

    catalogues = []
    for expected_redshift, key in ordered:
        snap = int(key.split("_")[-1])
        catalogue = _read_cross_run_halo_catalogue(
            run_dir, snap, mass_definition, minimum_particles, max_catalogue_ids,
        )
        if catalogue["box_mpc_h"] is None or not np.isclose(
            catalogue["box_mpc_h"], expected_box_mpc_h, rtol=1.0e-6, atol=0.0,
        ):
            raise ValueError("FOF history catalogue BoxSize does not agree with its run manifest")
        if catalogue["redshift"] is None or abs(catalogue["redshift"] - expected_redshift) > 1.0e-6 * max(1.0, abs(expected_redshift)):
            raise ValueError("FOF history catalogue Redshift does not match its analysis snapshot")
        if np.any(np.diff(catalogue["dm_group_indices"]) < 0):
            raise ValueError("decoded FOF particle membership is not ordered by host index")
        catalogues.append(catalogue)

    final_catalogue = catalogues[-1]
    final_rows = {int(index): row for row, index in enumerate(final_catalogue["group_indices"])}
    final_group_indices = np.asarray(final_group_indices, dtype=np.int64)
    try:
        rows = np.asarray([final_rows[int(index)] for index in final_group_indices], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("a cross-run matched host is absent from its final FOF history catalogue") from exc

    track_count = len(final_group_indices)
    time_count = len(catalogues)
    mass_track = np.full((track_count, time_count), np.nan)
    group_track = np.full((track_count, time_count), -1, dtype=np.int64)
    shared_track = np.zeros((track_count, time_count), dtype=np.int64)
    for row, group_index in enumerate(final_group_indices):
        mass_track[row, -1] = final_catalogue["mass_msun_h"][rows[row]]
        group_track[row, -1] = group_index
    active_groups = final_group_indices.copy()

    for time_index in range(time_count - 2, -1, -1):
        current = catalogues[time_index + 1]
        past = catalogues[time_index]
        current_rows = {int(index): row for row, index in enumerate(current["group_indices"])}
        past_rows = {int(index): row for row, index in enumerate(past["group_indices"])}
        child_ids_parts, child_track_parts = [], []
        child_counts = np.zeros(track_count, dtype=np.int64)
        for track_index, group_index in enumerate(active_groups):
            if group_index < 0 or int(group_index) not in current_rows:
                continue
            lo = int(np.searchsorted(current["dm_group_indices"], group_index, side="left"))
            hi = int(np.searchsorted(current["dm_group_indices"], group_index, side="right"))
            if hi <= lo:
                continue
            particle_ids = current["dm_ids"][lo:hi]
            child_ids_parts.append(particle_ids)
            child_track_parts.append(np.full(len(particle_ids), track_index, dtype=np.int64))
            child_counts[track_index] = len(particle_ids)

        next_groups = np.full(track_count, -1, dtype=np.int64)
        if child_ids_parts and len(past["dm_ids"]):
            child_ids = np.concatenate(child_ids_parts)
            child_tracks = np.concatenate(child_track_parts)
            past_order = np.argsort(past["dm_ids"], kind="stable")
            past_sorted_ids = past["dm_ids"][past_order]
            if np.any(past_sorted_ids[1:] == past_sorted_ids[:-1]):
                raise ValueError("a type-1 particle ID appears more than once in a past FOF catalogue")
            past_sorted_groups = past["dm_group_indices"][past_order]
            past_locations = np.searchsorted(past_sorted_ids, child_ids)
            in_range = past_locations < len(past_sorted_ids)
            candidate_positions = np.flatnonzero(in_range)
            candidate_positions = candidate_positions[
                past_sorted_ids[past_locations[in_range]] == child_ids[in_range]
            ]
            if len(candidate_positions):
                track_for_particle = child_tracks[candidate_positions]
                parent_for_particle = past_sorted_groups[past_locations[candidate_positions]]
                if max(int(np.max(track_for_particle)), int(np.max(parent_for_particle))) >= 2**32:
                    raise ValueError("FOF history has too many tracks or groups for compact overlap indices")
                keys = (track_for_particle.astype(np.uint64) << np.uint64(32)) | parent_for_particle.astype(np.uint64)
                pair_keys, overlap_counts = np.unique(keys, return_counts=True)
                candidate_tracks = (pair_keys >> np.uint64(32)).astype(np.int64)
                candidate_groups = (pair_keys & np.uint64(0xffffffff)).astype(np.int64)
                candidate_parent_rows = np.asarray([past_rows[int(index)] for index in candidate_groups], dtype=np.int64)
                candidate_mass = past["mass_msun_h"][candidate_parent_rows]
                fractions = overlap_counts / child_counts[candidate_tracks]
                keep = (overlap_counts >= int(min_shared_particles)) & (fractions >= float(min_overlap_fraction))
                candidate_tracks = candidate_tracks[keep]
                candidate_groups = candidate_groups[keep]
                overlap_counts = overlap_counts[keep]
                candidate_mass = candidate_mass[keep]
                order = np.lexsort((candidate_groups, -candidate_mass, -overlap_counts, candidate_tracks))
                chosen = {}
                for candidate_row in order:
                    chosen.setdefault(int(candidate_tracks[candidate_row]), candidate_row)
                for track_index, candidate_row in chosen.items():
                    group_index = int(candidate_groups[candidate_row])
                    next_groups[track_index] = group_index
                    past_row = past_rows[group_index]
                    group_track[track_index, time_index] = group_index
                    mass_track[track_index, time_index] = past["mass_msun_h"][past_row]
                    shared_track[track_index, time_index] = int(overlap_counts[candidate_row])
        active_groups = next_groups

    redshifts = np.asarray([catalogue["redshift"] for catalogue in catalogues], dtype=float)
    scale_factors = 1.0 / (1.0 + redshifts)
    formation_redshifts = np.full(track_count, np.nan)
    formation_status = np.full(track_count, "NO_HALF_MASS_CROSSING_IN_TRACK", dtype=object)
    for row in range(track_count):
        valid_columns = np.flatnonzero(np.isfinite(mass_track[row]) & (mass_track[row] > 0))
        if len(valid_columns) < 2:
            formation_status[row] = "INSUFFICIENT_PROGENITOR_SNAPSHOTS"
            continue
        final_mass = float(mass_track[row, -1])
        if not np.isfinite(final_mass) or final_mass <= 0:
            formation_status[row] = "INVALID_FINAL_HOST_MASS"
            continue
        half_mass = 0.5 * final_mass
        if mass_track[row, valid_columns[0]] >= half_mass:
            formation_status[row] = "BEFORE_FIRST_DETECTED_PROGENITOR"
            continue
        for earlier, later in zip(valid_columns[:-1], valid_columns[1:]):
            if later != earlier + 1:
                continue
            m0, m1 = mass_track[row, earlier], mass_track[row, later]
            if m0 < half_mass <= m1 and m1 > m0:
                fraction = (math.log(half_mass) - math.log(m0)) / (math.log(m1) - math.log(m0))
                a_half = scale_factors[earlier] + fraction * (scale_factors[later] - scale_factors[earlier])
                formation_redshifts[row] = 1.0 / a_half - 1.0
                formation_status[row] = "MEASURED"
                break
    return {
        "redshift": redshifts,
        "group_index": group_track,
        "mass_msun_h": mass_track,
        "shared_particle_count": shared_track,
        "formation_redshift": formation_redshifts,
        "formation_status": formation_status,
    }


def _add_fof_formation_redshift_shift(run_a, run_b, snapshot_records_a, snapshot_records_b,
                                      final_snapshot_a, final_snapshot_b, pair_name, pair_out,
                                      pair_summary, summary, mass_definition, minimum_particles,
                                      min_shared_particles, min_overlap_fraction, max_catalogue_ids,
                                      identity_a, identity_b, outdir):
    """Add separately labeled FOF half-mass formation shifts for matched hosts."""
    if "matched_halos" not in pair_out or len(pair_out["matched_halos/group_index_a"]) == 0:
        result = {"status": "SKIPPED", "reason": "the final matched-redshift pair has no matched FOF hosts"}
        pair_summary.setdefault("secondary", {})["formation_redshift_shift"] = result
        summary["halo_matching"]["formation_redshift_shift"] = result
        return
    matched = pair_out["matched_halos"]
    try:
        tracks_a = _fof_main_progenitor_tracks(
            run_a, snapshot_records_a, final_snapshot_a, matched["group_index_a"][:],
            mass_definition, minimum_particles, min_shared_particles, min_overlap_fraction,
            max_catalogue_ids, identity_a["box_mpc_h"],
        )
        tracks_b = _fof_main_progenitor_tracks(
            run_b, snapshot_records_b, final_snapshot_b, matched["group_index_b"][:],
            mass_definition, minimum_particles, min_shared_particles, min_overlap_fraction,
            max_catalogue_ids, identity_b["box_mpc_h"],
        )
    except (OSError, KeyError, ValueError, OverflowError) as exc:
        result = {"status": "SKIPPED", "reason": str(exc)}
        pair_summary.setdefault("secondary", {})["formation_redshift_shift"] = result
        summary["halo_matching"]["formation_redshift_shift"] = result
        return

    formation_a = tracks_a["formation_redshift"]
    formation_b = tracks_b["formation_redshift"]
    shift = formation_b - formation_a
    valid_a = np.isfinite(formation_a)
    valid_b = np.isfinite(formation_b)
    valid_pair = valid_a & valid_b
    history = matched.create_group("fof_formation_history")
    history.attrs["method"] = "backward most-shared type-1-particle FOF progenitor; half-final-mass crossing interpolated linearly in log mass and scale factor"
    history.attrs["mass_definition"] = mass_definition
    history.attrs["minimum_particles_per_host"] = int(minimum_particles)
    history.attrs["minimum_shared_dark_matter_particles"] = int(min_shared_particles)
    history.attrs["minimum_overlap_fraction_of_descendant"] = float(min_overlap_fraction)
    history.attrs["formation_shift_sign"] = "B minus A; positive means B reaches half its final FOF mass at higher redshift"
    history.create_dataset("snapshot_redshift_a", data=tracks_a["redshift"])
    history.create_dataset("snapshot_redshift_b", data=tracks_b["redshift"])
    history.create_dataset("progenitor_group_index_a", data=tracks_a["group_index"])
    history.create_dataset("progenitor_group_index_b", data=tracks_b["group_index"])
    history.create_dataset("progenitor_mass_msun_h_a", data=tracks_a["mass_msun_h"])
    history.create_dataset("progenitor_mass_msun_h_b", data=tracks_b["mass_msun_h"])
    history.create_dataset("shared_particle_count_a", data=tracks_a["shared_particle_count"])
    history.create_dataset("shared_particle_count_b", data=tracks_b["shared_particle_count"])
    history.create_dataset("formation_redshift_half_mass_a", data=formation_a)
    history.create_dataset("formation_redshift_half_mass_b", data=formation_b)
    history.create_dataset("formation_redshift_shift_b_minus_a", data=shift)
    string_type = h5py.string_dtype(encoding="utf-8")
    history.create_dataset("formation_status_a", data=tracks_a["formation_status"], dtype=string_type)
    history.create_dataset("formation_status_b", data=tracks_b["formation_status"], dtype=string_type)

    figures = []
    if np.any(valid_pair):
        mass_a = np.asarray(matched["mass_a_msun_h"][:], dtype=float)[valid_pair]
        mass_b = np.asarray(matched["mass_b_msun_h"][:], dtype=float)[valid_pair]
        mass_axis = np.sqrt(mass_a * mass_b)
        plt.figure(figsize=(8, 5)); plt.scatter(mass_axis, shift[valid_pair], s=14, alpha=0.6); plt.xscale("log"); plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel(r"geometric mean matched host mass [$M_\odot/h$]"); plt.ylabel(r"$z_{1/2, B} - z_{1/2, A}$")
        plt.title("FOF half-mass formation-redshift shift at z=%.6g" % float(pair_summary["redshift_a"]))
        filename = "matched_halo_formation_redshift_shift_%s.png" % pair_name
        savefig(outdir / filename); figures.append(filename)

    if np.any(valid_a) or np.any(valid_b):
        plt.figure(figsize=(8, 5))
        if np.any(valid_a): plt.hist(formation_a[valid_a], bins=20, alpha=0.55, label="Run A")
        if np.any(valid_b): plt.hist(formation_b[valid_b], bins=20, alpha=0.55, label="Run B")
        plt.xlabel(r"FOF half-mass formation redshift $z_{1/2}$"); plt.ylabel("matched-host count"); plt.legend()
        plt.title("FOF formation-redshift distributions at z=%.6g" % float(pair_summary["redshift_a"]))
        filename = "matched_halo_formation_redshift_distribution_%s.png" % pair_name
        savefig(outdir / filename); figures.append(filename)

    if np.any(valid_pair):
        valid_shift = shift[valid_pair]
        result = {
            "status": "COMPLETE", "matched_host_count": int(len(formation_a)),
            "measured_count_a": int(np.count_nonzero(valid_a)),
            "measured_count_b": int(np.count_nonzero(valid_b)),
            "measured_shift_count": int(np.count_nonzero(valid_pair)),
            "median_shift_b_minus_a": float(np.median(valid_shift)),
            "percentile_16_84_shift_b_minus_a": [float(value) for value in np.percentile(valid_shift, [16, 84])],
            "status_counts_a": {str(value): int(np.count_nonzero(tracks_a["formation_status"] == value)) for value in np.unique(tracks_a["formation_status"])},
            "status_counts_b": {str(value): int(np.count_nonzero(tracks_b["formation_status"] == value)) for value in np.unique(tracks_b["formation_status"])},
            "figures": figures,
        }
    else:
        result = {
            "status": "SKIPPED", "reason": "fewer than one matched host has a measured half-mass crossing in both runs",
            "matched_host_count": int(len(formation_a)),
            "measured_count_a": int(np.count_nonzero(valid_a)),
            "measured_count_b": int(np.count_nonzero(valid_b)),
            "measured_shift_count": 0,
            "status_counts_a": {str(value): int(np.count_nonzero(tracks_a["formation_status"] == value)) for value in np.unique(tracks_a["formation_status"])},
            "status_counts_b": {str(value): int(np.count_nonzero(tracks_b["formation_status"] == value)) for value in np.unique(tracks_b["formation_status"])},
            "figures": figures,
        }
    pair_summary.setdefault("secondary", {})["formation_redshift_shift"] = result
    summary["halo_matching"]["formation_redshift_shift"] = {"pair": pair_name, **result}


def _initial_condition_paths(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Current native runs record the exact IC shards in the sealed GADGET
        # handoff. Do not substitute a similarly named file if those bytes are
        # missing or changed: that would compare a different realization.
        handoff = manifest.get("gadget_handoff", {})
        if isinstance(handoff, dict) and "initial_conditions" in handoff:
            entries = handoff["initial_conditions"]
            if not isinstance(entries, list) or not entries:
                raise ValueError("GADGET handoff has no initial-condition shards")
            paths = []
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
                    raise ValueError("GADGET handoff initial-condition record is malformed")
                path = Path(entry["path"]).expanduser()
                if not path.is_absolute():
                    path = run_dir / path
                path = path.resolve()
                if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                    raise ValueError("GADGET handoff initial-condition shard is missing or changed: " + str(path))
                paths.append(path)
            from core.data import validate_native_shard_counts
            validate_native_shard_counts(paths, "NumPart_Total")
            return paths
        try:
            source = manifest.get("initial_condition_source")
            if isinstance(source, dict):
                source = source.get("path")
            if source:
                from evolution import initial_condition_files
                paths = initial_condition_files(source)
                if paths:
                    return paths
        except (OSError, ValueError, KeyError, TypeError):
            pass
    paths = find_hdf5_files(run_dir / "data", "ics_gadget")
    if not paths:
        return []
    with h5py.File(paths[0], "r") as f:
        nfiles_value = np.asarray(f["Header"].attrs.get("NumFilesPerSnapshot", 1))
        if nfiles_value.size != 1:
            return []
        nfiles = int(nfiles_value.reshape(-1)[0])
    if nfiles > 1:
        first = paths[0]
        stem = first.name[:-5] if first.name.endswith(".hdf5") else first.name
        if stem.endswith(".0"):
            stem = stem[:-2]
        prefix = first.with_name(stem)
        paths = [Path(str(prefix) + ".%d.hdf5" % i) for i in range(nfiles)]
        if not all(path.is_file() for path in paths):
            return []
    return paths


def _deposit_cic_density_mesh(files, mesh_size):
    density = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float64)
    box = None
    particle_count = 0
    for path in files:
        with h5py.File(path, "r") as f:
            header = f["Header"].attrs
            box_value = np.asarray(header["BoxSize"])
            if box_value.size != 1:
                raise ValueError("paired IC Header/BoxSize must be scalar")
            this_box = float(box_value.reshape(-1)[0])
            if box is None:
                box = this_box
            elif not math.isclose(this_box, box, rel_tol=1.0e-10):
                raise ValueError("paired IC files disagree on BoxSize")
            mass_table = np.asarray(header.get("MassTable", np.zeros(6)), dtype=float)
            if mass_table.shape != (6,):
                raise ValueError("IC MassTable is not a six-element GADGET vector")
            for ptype in range(6):
                name = "PartType%d" % ptype
                if name not in f or "Coordinates" not in f[name]:
                    continue
                part = f[name]
                coords_ds = part["Coordinates"]
                if mass_table[ptype] > 0:
                    mass_ds = None
                    fixed_mass = float(mass_table[ptype])
                elif "Masses" in part:
                    mass_ds = part["Masses"]
                    fixed_mass = None
                    if len(mass_ds) != len(coords_ds):
                        raise ValueError("IC particle mass and coordinate counts disagree")
                else:
                    raise ValueError("IC particle species has neither MassTable mass nor Masses dataset")
                for start in range(0, len(coords_ds), 1_000_000):
                    stop = min(start + 1_000_000, len(coords_ds))
                    coords = np.mod(np.asarray(coords_ds[start:stop], dtype=np.float64), box)
                    masses = (np.asarray(mass_ds[start:stop], dtype=np.float64) if fixed_mass is None
                              else np.full(stop - start, fixed_mass, dtype=np.float64))
                    grid = coords * (mesh_size / box)
                    base = np.floor(grid).astype(np.int64)
                    frac = grid - base
                    base %= mesh_size
                    for dx in (0, 1):
                        wx = frac[:, 0] if dx else (1.0 - frac[:, 0])
                        ix = (base[:, 0] + dx) % mesh_size
                        for dy in (0, 1):
                            wy = frac[:, 1] if dy else (1.0 - frac[:, 1])
                            iy = (base[:, 1] + dy) % mesh_size
                            for dz in (0, 1):
                                wz = frac[:, 2] if dz else (1.0 - frac[:, 2])
                                iz = (base[:, 2] + dz) % mesh_size
                                flat = (ix * mesh_size + iy) * mesh_size + iz
                                density += np.bincount(flat, weights=masses * wx * wy * wz, minlength=mesh_size**3).reshape(density.shape)
                    particle_count += stop - start
    if box is None or particle_count == 0 or not np.isfinite(density).all() or density.sum() <= 0:
        raise ValueError("IC particle mass density could not be deposited on the comparison mesh")
    contrast = density / float(np.mean(density)) - 1.0
    contrast -= float(np.mean(contrast))
    return contrast, box, particle_count


def _measure_initial_ic_phase_agreement(files_a, files_b, cfg_a, cfg_b):
    """Measure low-k phases from two explicit Gadget IC file sets."""
    sim_a, sim_b = cfg_a.get("simulation", {}), cfg_b.get("simulation", {})
    n_a, n_b = int(sim_a.get("particles_per_dimension", 0)), int(sim_b.get("particles_per_dimension", 0))
    box_a, box_b = float(sim_a.get("box_mpc_h", 0)), float(sim_b.get("box_mpc_h", 0))
    if n_a < 2 or n_b < 2 or not math.isclose(box_a, box_b, rel_tol=1.0e-12):
        return {"status": "SKIPPED", "reason": "paired IC phase comparison requires valid particle grids and equal box sizes"}
    if not files_a or not files_b:
        return {"status": "UNAVAILABLE", "reason": "one or both IC sets do not expose readable Gadget HDF5 files"}
    options_a = cfg_a.get("validation", {}).get("ic_phase", {})
    options_b = cfg_b.get("validation", {}).get("ic_phase", {})
    mesh_max = min(int(options_a.get("max_mesh_size", 128)), int(options_b.get("max_mesh_size", 128)))
    if mesh_max < 8:
        return {"status": "UNAVAILABLE", "reason": "validation.ic_phase.max_mesh_size must be at least 8 in both runs"}
    mesh_size = min(n_a, n_b, mesh_max)
    try:
        contrast_a, ic_box_a, count_a = _deposit_cic_density_mesh(files_a, mesh_size)
        contrast_b, ic_box_b, count_b = _deposit_cic_density_mesh(files_b, mesh_size)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "FAIL", "reason": "initial-condition phase measurement failed: " + str(exc)}
    if not math.isclose(ic_box_a, ic_box_b, rel_tol=1.0e-10):
        return {"status": "FAIL", "reason": "initial-condition HDF5 BoxSize values differ"}
    expected_box_kpc_h = box_a * 1000.0
    if (not math.isclose(ic_box_a, expected_box_kpc_h, rel_tol=1.0e-6)
            or not math.isclose(ic_box_b, expected_box_kpc_h, rel_tol=1.0e-6)):
        return {"status": "FAIL", "reason": "initial-condition HDF5 BoxSize does not match the configured box"}
    fft_a, fft_b = scipy_fft.rfftn(contrast_a), scipy_fft.rfftn(contrast_b)
    integer_x = np.fft.fftfreq(mesh_size) * mesh_size
    integer_y = np.fft.fftfreq(mesh_size) * mesh_size
    integer_z = np.fft.rfftfreq(mesh_size) * mesh_size
    nx, ny, nz = np.meshgrid(integer_x, integer_y, integer_z, indexing="ij")
    mode_number = np.sqrt(nx**2 + ny**2 + nz**2)
    amp_a, amp_b = np.abs(fft_a), np.abs(fft_b)
    kmax_mode = mesh_size / 4.0
    keep = ((mode_number > 0) & (mode_number <= kmax_mode)
            & (amp_a > max(float(np.max(amp_a)) * 1.0e-10, np.finfo(float).tiny))
            & (amp_b > max(float(np.max(amp_b)) * 1.0e-10, np.finfo(float).tiny)))
    mode_count = int(np.count_nonzero(keep))
    min_modes = max(int(options_a.get("min_modes", 20)), int(options_b.get("min_modes", 20)))
    if mode_count < min_modes:
        return {"status": "UNAVAILABLE", "reason": "too few nonzero common low-k Fourier modes for a phase comparison", "common_mode_count": mode_count}
    cross = fft_a[keep] * np.conj(fft_b[keep])
    weights = np.abs(cross)
    nonzero = weights > 0
    cross, weights = cross[nonzero], weights[nonzero]
    phase = np.angle(cross)
    weight_sum = float(np.sum(weights))
    coherence = float(abs(np.sum(cross)) / weight_sum) if weight_sum > 0 else 0.0
    mean_phase = float(np.angle(np.sum(cross))) if weight_sum > 0 else float("nan")
    phase_rms = float(np.sqrt(np.sum(weights * phase**2) / weight_sum)) if weight_sum > 0 else float("inf")
    min_coherence = max(float(options_a.get("min_coherence", 0.99)), float(options_b.get("min_coherence", 0.99)))
    max_phase_rms = min(
        float(options_a.get("max_weighted_rms_phase_rad", 0.05)),
        float(options_b.get("max_weighted_rms_phase_rad", 0.05)),
    )
    passed = coherence >= min_coherence and phase_rms <= max_phase_rms
    return {
        "status": "PASS" if passed else "FAIL",
        "method": "CIC mass-weighted IC density; cross-phase over common modes below one-quarter mesh Nyquist",
        "particles_per_dimension_a": n_a, "particles_per_dimension_b": n_b,
        "particle_count_a": count_a, "particle_count_b": count_b,
        "mesh_size": mesh_size, "common_mode_count": mode_count,
        "k_max_h_Mpc": (2.0 * math.pi / box_a) * kmax_mode,
        "complex_phase_coherence": coherence,
        "weighted_mean_phase_rad": mean_phase,
        "weighted_rms_phase_rad": phase_rms,
        "min_coherence": min_coherence,
        "max_weighted_rms_phase_rad": max_phase_rms,
    }


def compare_initial_ic_file_phases(ic_a, ic_b, cfg_a, cfg_b):
    """Compare phases in explicit IC files, including ICs outside run folders.

    This compares the actual mass-weighted density fields and does not infer
    phase agreement from matching random-seed metadata.
    """
    try:
        from evolution import initial_condition_files
        files_a = initial_condition_files(ic_a)
        files_b = initial_condition_files(ic_b)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "UNAVAILABLE", "reason": "could not resolve IC files: " + str(exc)}
    return _measure_initial_ic_phase_agreement(files_a, files_b, cfg_a or {}, cfg_b or {})


def compare_initial_ic_phases(run_a, run_b, cfg_a=None, cfg_b=None):
    """Measure low-k initial phases; random-seed metadata is reported, never trusted as proof."""
    run_a, run_b = Path(run_a), Path(run_b)
    cfg_a = cfg_a or (load_cfg(run_a) if (run_a / "config.yaml").is_file() else {})
    cfg_b = cfg_b or (load_cfg(run_b) if (run_b / "config.yaml").is_file() else {})
    seed_a = cfg_a.get("simulation", {}).get("random_seed")
    seed_b = cfg_b.get("simulation", {}).get("random_seed")
    try:
        files_a, files_b = _initial_condition_paths(run_a), _initial_condition_paths(run_b)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "FAIL", "reason": "initial-condition provenance validation failed: " + str(exc),
                "random_seed_a": seed_a, "random_seed_b": seed_b,
                "random_seed_metadata_match": seed_a == seed_b if seed_a is not None and seed_b is not None else None}
    result = _measure_initial_ic_phase_agreement(files_a, files_b, cfg_a, cfg_b)
    result["random_seed_a"] = seed_a
    result["random_seed_b"] = seed_b
    result["random_seed_metadata_match"] = (
        seed_a == seed_b if seed_a is not None and seed_b is not None else None
    )
    return result


def _profile_on_normalized_radius(profile, dataset, radius_grid):
    if dataset not in profile or "radius_comoving_mpc_h" not in profile or "r200c_comoving_mpc_h" not in profile.attrs:
        return None
    radius = np.asarray(profile["radius_comoving_mpc_h"][:], dtype=float)
    values = np.asarray(profile[dataset][:], dtype=float)
    r200 = float(profile.attrs["r200c_comoving_mpc_h"])
    if radius.shape != values.shape or not np.isfinite(r200) or r200 <= 0:
        return None
    x = radius / r200
    good = np.isfinite(x) & (x > 0) & np.isfinite(values) & (values > 0)
    if np.count_nonzero(good) < 2:
        return None
    order = np.argsort(x[good], kind="stable")
    x, values = x[good][order], values[good][order]
    unique = np.r_[True, np.diff(x) > 0]
    x, values = x[unique], values[unique]
    output = np.full(len(radius_grid), np.nan, dtype=float)
    inside = (radius_grid >= x[0]) & (radius_grid <= x[-1])
    output[inside] = np.exp(np.interp(np.log(radius_grid[inside]), np.log(x), np.log(values)))
    return output


def _add_paired_hydro_snapshot_products(pair_out, group_a, group_b, pair_name, redshift, outdir):
    """Write phase-diagram, matched gas-profile and baryon-fraction comparisons."""
    secondary = {}
    phase_group = pair_out.create_group("gas_phase_diagram")
    phase_count_key = (
        "phase_particle_count_estimate_2d"
        if "phase_particle_count_estimate_2d" in group_a and "phase_particle_count_estimate_2d" in group_b
        else "phase_particle_count_2d"
    )
    phase_keys = [
        "phase_log10_density_bin_edges", "phase_log10_temperature_bin_edges",
        phase_count_key,
    ]
    if all(key in group_a and key in group_b for key in phase_keys):
        edges_r_a = np.asarray(group_a[phase_keys[0]][:], dtype=float)
        edges_r_b = np.asarray(group_b[phase_keys[0]][:], dtype=float)
        edges_t_a = np.asarray(group_a[phase_keys[1]][:], dtype=float)
        edges_t_b = np.asarray(group_b[phase_keys[1]][:], dtype=float)
        counts_a = np.asarray(group_a[phase_keys[2]][:], dtype=float)
        counts_b = np.asarray(group_b[phase_keys[2]][:], dtype=float)
        if (counts_a.shape == counts_b.shape and counts_a.shape == (len(edges_r_a) - 1, len(edges_t_a) - 1)
                and np.array_equal(edges_r_a, edges_r_b) and np.array_equal(edges_t_a, edges_t_b)
                and np.sum(counts_a) > 0 and np.sum(counts_b) > 0):
            fraction_a = counts_a / np.sum(counts_a)
            fraction_b = counts_b / np.sum(counts_b)
            ratio = np.full_like(fraction_a, np.nan)
            valid = (fraction_a > 0) & (fraction_b > 0)
            ratio[valid] = np.log10(fraction_b[valid] / fraction_a[valid])
            phase_group.attrs["status"] = "COMPLETE"
            phase_group.attrs["ratio_definition"] = "log10 normalized gas particle-count fraction B/A in common bins"
            phase_group.attrs["sample_weighting"] = (
                "inverse shard-inclusion weighted" if phase_count_key == "phase_particle_count_estimate_2d"
                else "unweighted sample counts; legacy analysis inputs"
            )
            phase_group.attrs["density_unit"] = str(group_a.attrs.get("phase_density_units", "snapshot analysis units"))
            phase_group.attrs["temperature_unit"] = str(group_a.attrs.get("phase_temperature_units", "K"))
            phase_group.create_dataset("log10_density_bin_edges", data=edges_r_a)
            phase_group.create_dataset("log10_temperature_bin_edges", data=edges_t_a)
            phase_group.create_dataset("fraction_a", data=fraction_a, compression="gzip")
            phase_group.create_dataset("fraction_b", data=fraction_b, compression="gzip")
            phase_group.create_dataset("log10_ratio_b_over_a", data=ratio, compression="gzip")
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
            extent = [edges_t_a[0], edges_t_a[-1], edges_r_a[0], edges_r_a[-1]]
            axes[0].imshow(np.log10(np.maximum(fraction_a.T, 1.0e-12)), origin="lower", extent=extent, aspect="auto", cmap="magma")
            axes[1].imshow(np.log10(np.maximum(fraction_b.T, 1.0e-12)), origin="lower", extent=extent, aspect="auto", cmap="magma")
            finite = np.abs(ratio[np.isfinite(ratio)])
            limit = max(float(np.percentile(finite, 98)) if finite.size else 0.05, 0.05)
            mesh = axes[2].imshow(np.ma.masked_invalid(ratio.T), origin="lower", extent=extent, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
            for axis in axes:
                axis.grid(False)
            axes[0].set_title("Run A: log10 gas fraction")
            axes[1].set_title("Run B: log10 gas fraction")
            axes[2].set_title("log10 normalized B/A")
            for axis in axes:
                axis.set_xlabel("log10 temperature [K]")
                axis.set_ylabel("log10 gas density [cm^-3]")
            fig.colorbar(mesh, ax=axes[2], label="log10 ratio")
            fig.suptitle("Gas phase diagram ratio at z=%.5g" % redshift)
            filename = "gas_phase_ratio_%s.png" % pair_name
            export_figure(fig, outdir / filename, metadata={
                "comparison_pair": pair_name, "redshift": float(redshift),
                "product": "paired gas phase diagram ratio",
            })
            secondary["gas_phase_diagram"] = {"status": "COMPLETE", "figure": filename, "valid_common_bins": int(np.count_nonzero(valid))}
        else:
            phase_group.attrs["status"] = "SKIPPED"
            phase_group.attrs["reason"] = "phase bins differ, counts are empty, or arrays are malformed"
            secondary["gas_phase_diagram"] = {"status": "SKIPPED", "reason": str(phase_group.attrs["reason"])}
    else:
        phase_group.attrs["status"] = "SKIPPED"
        phase_group.attrs["reason"] = "one or both snapshots lack gas phase-diagram counts"
        secondary["gas_phase_diagram"] = {"status": "SKIPPED", "reason": str(phase_group.attrs["reason"])}

    profile_group = pair_out.create_group("gas_profile_comparison")
    profile_group.attrs["status"] = "SKIPPED"
    profile_group.attrs["reason"] = "matched host gas profiles are unavailable"
    profile_name_a, profile_name_b = "gas_profiles", "gas_profiles"
    matched = pair_out.get("matched_halos")
    gas_a, gas_b = group_a.get(profile_name_a), group_b.get(profile_name_b)
    if matched is not None and gas_a is not None and gas_b is not None and "group_index_a" in matched and "group_index_b" in matched:
        rows_a = {int(profile.attrs.get("halo_catalogue_row", -1)): profile for profile in gas_a.values() if isinstance(profile, h5py.Group)}
        rows_b = {int(profile.attrs.get("halo_catalogue_row", -1)): profile for profile in gas_b.values() if isinstance(profile, h5py.Group)}
        radius_grid = np.geomspace(0.025, 2.0, 40)
        density_ratios, temperature_ratios, matched_rows, mass_pairs = [], [], [], []
        index_a = np.asarray(matched["group_index_a"][:], dtype=np.int64)
        index_b = np.asarray(matched["group_index_b"][:], dtype=np.int64)
        mass_a = np.asarray(matched["mass_a_msun_h"][:], dtype=float) if "mass_a_msun_h" in matched else np.full(len(index_a), np.nan)
        mass_b = np.asarray(matched["mass_b_msun_h"][:], dtype=float) if "mass_b_msun_h" in matched else np.full(len(index_b), np.nan)
        for i, (row_a, row_b) in enumerate(zip(index_a, index_b)):
            profile_a, profile_b = rows_a.get(int(row_a)), rows_b.get(int(row_b))
            if profile_a is None or profile_b is None:
                continue
            density_a = _profile_on_normalized_radius(profile_a, "gas_mass_density_comoving_msun_h_per_mpc_h3", radius_grid)
            density_b = _profile_on_normalized_radius(profile_b, "gas_mass_density_comoving_msun_h_per_mpc_h3", radius_grid)
            temp_a = _profile_on_normalized_radius(profile_a, "gas_temperature_mass_weighted_K", radius_grid)
            temp_b = _profile_on_normalized_radius(profile_b, "gas_temperature_mass_weighted_K", radius_grid)
            density_ratio = np.divide(density_b, density_a, out=np.full_like(radius_grid, np.nan), where=np.isfinite(density_a) & (density_a > 0) & np.isfinite(density_b) & (density_b > 0))
            temperature_ratio = np.divide(temp_b, temp_a, out=np.full_like(radius_grid, np.nan), where=np.isfinite(temp_a) & (temp_a > 0) & np.isfinite(temp_b) & (temp_b > 0))
            if np.any(np.isfinite(density_ratio)) or np.any(np.isfinite(temperature_ratio)):
                density_ratios.append(density_ratio)
                temperature_ratios.append(temperature_ratio)
                matched_rows.append((row_a, row_b))
                mass_pairs.append((mass_a[i], mass_b[i]))
        if density_ratios:
            density_array = np.asarray(density_ratios, dtype=float)
            temperature_array = np.asarray(temperature_ratios, dtype=float)
            profile_group.attrs["status"] = "COMPLETE"
            profile_group.attrs["ratio_definition"] = "matched-host B/A gas profile at common r/R200c; log interpolation within each measured profile range"
            profile_group.attrs["matched_profile_count"] = len(matched_rows)
            profile_group.create_dataset("radius_over_r200c", data=radius_grid)
            profile_group.create_dataset("matched_group_index_a_b", data=np.asarray(matched_rows, dtype=np.int64))
            profile_group.create_dataset("matched_mass_msun_h_a_b", data=np.asarray(mass_pairs, dtype=float))
            profile_group.create_dataset("gas_density_ratio_b_over_a", data=density_array, compression="gzip")
            profile_group.create_dataset("gas_temperature_ratio_b_over_a", data=temperature_array, compression="gzip")
            for label, array in [("gas_density", density_array), ("gas_temperature", temperature_array)]:
                valid_count = np.sum(np.isfinite(array), axis=0)
                median = np.full(array.shape[1], np.nan)
                for radial_index in np.flatnonzero(valid_count):
                    median[radial_index] = np.nanmedian(array[:, radial_index])
                profile_group.create_dataset(label + "_median_ratio", data=median)
                profile_group.create_dataset(label + "_valid_halo_count", data=valid_count)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
            for axis, array, title in [
                (axes[0], density_array, "gas density"),
                (axes[1], temperature_array, "gas temperature"),
            ]:
                valid_count = np.sum(np.isfinite(array), axis=0)
                median = np.full(array.shape[1], np.nan)
                low = np.full(array.shape[1], np.nan)
                high = np.full(array.shape[1], np.nan)
                for radial_index in np.flatnonzero(valid_count):
                    column = array[:, radial_index]
                    column = column[np.isfinite(column)]
                    median[radial_index] = np.median(column)
                    low[radial_index], high[radial_index] = np.percentile(column, [16, 84])
                good = np.isfinite(median) & (median > 0)
                axis.plot(radius_grid[good], median[good], color="#2563eb")
                band = good & np.isfinite(low) & np.isfinite(high) & (low > 0) & (high > 0)
                if np.any(band):
                    axis.fill_between(radius_grid[band], low[band], high[band], alpha=0.2)
                axis.axhline(1.0, color="#64748b", linewidth=0.8)
                axis.set_xscale("log"); axis.set_yscale("log")
                axis.set_xlabel(r"$r/R_{200c}$"); axis.set_ylabel("matched B / A")
                axis.set_title(title)
            fig.suptitle("Matched-host gas profile ratios at z=%.5g" % redshift)
            filename = "matched_gas_profile_ratio_%s.png" % pair_name
            export_figure(fig, outdir / filename, metadata={
                "comparison_pair": pair_name, "redshift": float(redshift),
                "product": "matched gas profile ratios",
            })
            secondary["gas_profiles"] = {"status": "COMPLETE", "figure": filename, "matched_profile_count": len(matched_rows)}
        else:
            secondary["gas_profiles"] = {"status": "SKIPPED", "reason": str(profile_group.attrs["reason"])}
    else:
        secondary["gas_profiles"] = {"status": "SKIPPED", "reason": str(profile_group.attrs["reason"])}

    baryon_group = pair_out.create_group("baryon_fraction_comparison")
    baryon_group.attrs["status"] = "SKIPPED"
    baryon_group.attrs["reason"] = "baryon fraction arrays are unavailable"
    required = ["baryon_fraction_halo_mass_msun_h", "gas_mass_fraction", "stellar_mass_fraction"]
    if all(key in group_a and key in group_b for key in required):
        mass_a, mass_b = np.asarray(group_a[required[0]][:], dtype=float), np.asarray(group_b[required[0]][:], dtype=float)
        gas_a, gas_b = np.asarray(group_a[required[1]][:], dtype=float), np.asarray(group_b[required[1]][:], dtype=float)
        star_a, star_b = np.asarray(group_a[required[2]][:], dtype=float), np.asarray(group_b[required[2]][:], dtype=float)
        good_a = np.isfinite(mass_a) & (mass_a > 0) & np.isfinite(gas_a) & np.isfinite(star_a)
        good_b = np.isfinite(mass_b) & (mass_b > 0) & np.isfinite(gas_b) & np.isfinite(star_b)
        if np.count_nonzero(good_a) >= 3 and np.count_nonzero(good_b) >= 3:
            lo, hi = max(np.min(mass_a[good_a]), np.min(mass_b[good_b])), min(np.max(mass_a[good_a]), np.max(mass_b[good_b]))
            if hi > lo:
                edges = np.geomspace(lo, hi, 9)
                centers = np.sqrt(edges[:-1] * edges[1:])
                medians = {}
                for name, masses, values, good in [
                    ("gas_a", mass_a, gas_a, good_a), ("gas_b", mass_b, gas_b, good_b),
                    ("stars_a", mass_a, star_a, good_a), ("stars_b", mass_b, star_b, good_b),
                ]:
                    med = np.full(len(centers), np.nan)
                    for index in range(len(centers)):
                        select = good & (masses >= edges[index]) & (masses < edges[index + 1])
                        if np.count_nonzero(select) >= 3:
                            med[index] = np.median(values[select])
                    medians[name] = med
                baryon_group.attrs["status"] = "COMPLETE"
                baryon_group.attrs["difference_definition"] = "median mass fraction in common halo-mass bins, run B minus run A"
                baryon_group.create_dataset("mass_bin_edges_msun_h", data=edges)
                baryon_group.create_dataset("mass_bin_centers_msun_h", data=centers)
                for name, values in medians.items():
                    baryon_group.create_dataset(name + "_median_fraction", data=values)
                baryon_group.create_dataset("gas_fraction_difference_b_minus_a", data=medians["gas_b"] - medians["gas_a"])
                baryon_group.create_dataset("stellar_fraction_difference_b_minus_a", data=medians["stars_b"] - medians["stars_a"])
                fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
                axis.semilogx(centers, medians["gas_b"] - medians["gas_a"], marker="o", label="gas")
                axis.semilogx(centers, medians["stars_b"] - medians["stars_a"], marker="s", label="stars")
                axis.axhline(0.0, color="#64748b", linewidth=0.8)
                axis.set_xlabel(r"FOF halo mass [$M_\odot/h$]"); axis.set_ylabel("median fraction B − A")
                axis.set_title("Baryon fraction differences at z=%.5g" % redshift); axis.legend()
                filename = "baryon_fraction_difference_%s.png" % pair_name
                export_figure(fig, outdir / filename, metadata={
                    "comparison_pair": pair_name, "redshift": float(redshift),
                    "product": "baryon fraction comparison",
                })
                secondary["baryon_fractions"] = {"status": "COMPLETE", "figure": filename}
            else:
                baryon_group.attrs["reason"] = "halo mass ranges do not overlap"
                secondary["baryon_fractions"] = {"status": "SKIPPED", "reason": str(baryon_group.attrs["reason"])}
        else:
            baryon_group.attrs["reason"] = "fewer than three valid halos in one or both runs"
            secondary["baryon_fractions"] = {"status": "SKIPPED", "reason": str(baryon_group.attrs["reason"])}
    else:
        secondary["baryon_fractions"] = {"status": "SKIPPED", "reason": str(baryon_group.attrs["reason"])}
    return secondary


def _add_paired_sfr_history(out, history_a, history_b, outdir):
    group = out.create_group("star_formation_history_comparison")
    group.attrs["status"] = "SKIPPED"
    group.attrs["reason"] = "physical SFR histories are unavailable in one or both runs"
    if history_a is None or history_b is None or not all(name in history_a and name in history_b for name in ("redshift", "total_sfr_msun_per_year")):
        return {"status": "SKIPPED", "reason": str(group.attrs["reason"])}
    z_a, z_b = np.asarray(history_a["redshift"][:], dtype=float), np.asarray(history_b["redshift"][:], dtype=float)
    sfr_a = np.asarray(history_a["total_sfr_msun_per_year"][:], dtype=float)
    sfr_b = np.asarray(history_b["total_sfr_msun_per_year"][:], dtype=float)
    if not len(z_a) or not len(z_b):
        return {"status": "SKIPPED", "reason": str(group.attrs["reason"])}
    pairs = []
    remaining = set(range(len(z_b)))
    for i, za in enumerate(z_a):
        choices = [(abs(za - z_b[j]), j) for j in remaining if np.isfinite(z_b[j])]
        if not choices:
            continue
        delta, j = min(choices)
        if delta <= 1.0e-6 * max(1.0, abs(za), abs(z_b[j])):
            pairs.append((float(za), float(sfr_a[i]), float(sfr_b[j])))
            remaining.remove(j)
    if not pairs:
        group.attrs["reason"] = "no physical SFR snapshots match in redshift"
        return {"status": "SKIPPED", "reason": str(group.attrs["reason"])}
    pairs.sort(key=lambda row: row[0])
    redshift, rate_a, rate_b = (np.asarray([row[index] for row in pairs], dtype=float) for index in range(3))
    ratio = np.divide(rate_b, rate_a, out=np.full_like(rate_a, np.nan), where=(rate_a > 0) & (rate_b > 0))
    group.attrs["status"] = "COMPLETE"
    group.attrs["ratio_definition"] = "physical whole-box instantaneous SFR B/A; zero-rate ratios are undefined"
    group.create_dataset("redshift", data=redshift)
    group.create_dataset("sfr_a_msun_per_year", data=rate_a)
    group.create_dataset("sfr_b_msun_per_year", data=rate_b)
    group.create_dataset("ratio_b_over_a", data=ratio)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
    axes[0].plot(redshift, rate_a, marker="o", label="Run A")
    axes[0].plot(redshift, rate_b, marker="s", label="Run B")
    if np.all(rate_a > 0) and np.all(rate_b > 0):
        axes[0].set_yscale("log")
    axes[0].set_ylabel(r"SFR [$M_\odot$/yr]"); axes[0].legend()
    axes[1].plot(redshift, ratio, marker="o")
    axes[1].axhline(1.0, color="#64748b", linewidth=0.8)
    axes[1].set_ylabel("B / A"); axes[1].set_xlabel("redshift")
    axes[1].invert_xaxis()
    fig.suptitle("Paired star-formation history")
    filename = "star_formation_history_comparison.png"
    export_figure(fig, outdir / filename, metadata={"product": "paired star formation history"})
    return {"status": "COMPLETE", "figure": filename, "matched_snapshot_count": len(pairs)}


def compare_runs(run_a, run_b, name="comparison"):
    """Own both inputs and publish one immutable comparison generation."""
    from core.native import native_run_lock
    from core.foundation import atomic_json, atomic_text, file_sha256 as digest, publish_directory
    run_a,run_b=Path(run_a).resolve(),Path(run_b).resolve()
    if not isinstance(name,str) or not name or Path(name).name!=name or name in ('.','..'):
        raise ValueError('Comparison name must be one directory component')
    destination=run_b.parent/name
    with native_run_lock(run_a,operation='comparison input A'),native_run_lock(run_b,operation='comparison input B'),native_run_lock(destination,operation='comparison publication'):
        if destination.exists():raise FileExistsError('Keep completed comparison; choose a fresh name')
        inputs_a,contract_a=_begin_analysis_contract(run_a,['comparison'])
        inputs_b,contract_b=_begin_analysis_contract(run_b,['comparison'])
        analysis_paths=[run/'results/analysis.h5' for run in (run_a,run_b)]
        analyses={str(path):digest(path) for path in analysis_paths}
        for run in (run_a,run_b):
            if (run/'results/analysis_provenance.json').is_file():verify_analysis_bundle(run/'results')
        stage=Path(tempfile.mkdtemp(prefix='.'+name+'-stage-',dir=destination.parent))
        try:
            atomic_json(stage/'comparison_inputs.json',{'run_a':contract_a,'run_b':contract_b,'analyses':analyses})
            _compare_runs_to_directory(run_a,run_b,stage)
            _verify_analysis_inputs(run_a,inputs_a,contract_a)
            _verify_analysis_inputs(run_b,inputs_b,contract_b)
            if any(digest(path)!=expected for path,expected in analyses.items()):
                raise ValueError('Saved analysis input changed during comparison')
            record={'schema':'paired-analysis-publication-v1','science_acceptance':'NOT_ACCEPTED',
                'inputs':analyses,'run_a':contract_a,'run_b':contract_b,
                'artifacts':{p.relative_to(stage).as_posix():digest(p) for p in stage.rglob('*') if p.is_file()
                    and '.native-run-locks' not in p.relative_to(stage).parts}}
            atomic_json(stage/'comparison_manifest.json',record)
            atomic_text(stage/'comparison_manifest.sha256',digest(stage/'comparison_manifest.json')+'\n')
            publish_directory(stage,destination)
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'type':type(exc).__name__,'message':str(exc),'source_analyses':analyses})
            raise
    return str(destination)


def _compare_runs_to_directory(run_a, run_b, outdir):
    """Compare only snapshots at the same redshift and export ratios with their inputs."""
    run_a, run_b = Path(run_a), Path(run_b)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tolerance = 1.0e-6
    identity_a, identity_b = _cross_run_identity(run_a), _cross_run_identity(run_b)
    cfg_a = load_cfg(run_a) if (run_a / "config.yaml").is_file() else {}
    cfg_b = load_cfg(run_b) if (run_b / "config.yaml").is_file() else {}
    setup_comparison = compare_requested_run_setups(cfg_a, cfg_b)
    initial_phase_agreement = compare_initial_ic_phases(run_a, run_b, cfg_a, cfg_b)
    resolution_source_status = _resolution_source_compatibility(run_a, run_b)
    match_settings_a = cfg_a.get("analysis", {})
    match_settings_b = cfg_b.get("analysis", {})
    min_shared_a = int(match_settings_a.get("cross_run_match_min_shared_particles", 10))
    min_shared_b = int(match_settings_b.get("cross_run_match_min_shared_particles", 10))
    min_fraction_a = float(match_settings_a.get("cross_run_match_min_overlap_fraction", 0.10))
    min_fraction_b = float(match_settings_b.get("cross_run_match_min_overlap_fraction", 0.10))
    max_ids_a = int(match_settings_a.get("cross_run_match_max_catalogue_ids", 30000000))
    max_ids_b = int(match_settings_b.get("cross_run_match_max_catalogue_ids", 30000000))
    if min_shared_a < 1 or min_shared_b < 1:
        raise ValueError("cross_run_match_min_shared_particles must be positive")
    if not (0.0 < min_fraction_a <= 1.0 and 0.0 < min_fraction_b <= 1.0):
        raise ValueError("cross_run_match_min_overlap_fraction must be in (0, 1]")
    if max_ids_a < 1 or max_ids_b < 1:
        raise ValueError("cross_run_match_max_catalogue_ids must be positive")
    matching_ready = True
    matching_reason = None
    if identity_a is None or identity_b is None:
        matching_ready, matching_reason = False, "both run manifests must record random_seed, particles_per_dimension, and box_mpc_h"
    elif identity_a["random_seed"] != identity_b["random_seed"]:
        matching_ready, matching_reason = False, "random seeds differ; particle IDs do not represent the same initial phases"
    elif identity_a["particles_per_dimension"] != identity_b["particles_per_dimension"]:
        matching_ready, matching_reason = False, "particle lattices differ; particle IDs cannot define direct halo counterparts"
    elif not np.isclose(identity_a["box_mpc_h"], identity_b["box_mpc_h"], rtol=1.0e-12, atol=0.0):
        matching_ready, matching_reason = False, "simulation box sizes differ; particle IDs cannot define direct halo counterparts"
    elif (min_shared_a, min_fraction_a, max_ids_a) != (min_shared_b, min_fraction_b, max_ids_b):
        matching_ready, matching_reason = False, "cross-run halo matching settings differ between the two run configurations"
    summary = {
        "run_a": str(run_a), "run_b": str(run_b),
        "redshift_tolerance": tolerance, "pairs": [],
        "unmatched_run_a": [], "unmatched_run_b": [],
        "heatmaps": {},
        "initial_ic_phase_agreement": initial_phase_agreement,
        "requested_setup_comparison": setup_comparison,
        "resolution_comparison_source_provenance": resolution_source_status,
        "halo_matching": {
            "status": "READY" if matching_ready else "SKIPPED",
            "reason": matching_reason,
            "minimum_shared_dark_matter_particles": min_shared_a,
            "minimum_overlap_fraction_per_host": min_fraction_a,
            "maximum_catalogue_ids_per_run": min(max_ids_a, max_ids_b),
            "identity_metadata_a": identity_a,
            "identity_metadata_b": identity_b,
            "match_policy": "one-to-one reciprocal best match by exact type-1 particle-ID overlap merit",
        },
    }

    with h5py.File(run_a / "results/analysis.h5", "r") as a, h5py.File(run_b / "results/analysis.h5", "r") as b, h5py.File(outdir / "analysis_comparison.h5", "w") as out:
        phase_group = out.create_group("initial_ic_phase_agreement")
        phase_group.attrs["report_json"] = json.dumps(initial_phase_agreement, sort_keys=True)
        model_a = str(a.attrs.get("model", "unknown"))
        model_b = str(b.attrs.get("model", "unknown"))
        definition_a = str(a.attrs.get("hmf_mass_definition", "unknown"))
        definition_b = str(b.attrs.get("hmf_mass_definition", "unknown"))
        min_particles_a = int(a.attrs.get("hmf_min_particles", -1))
        min_particles_b = int(b.attrs.get("hmf_min_particles", -1))
        range_a = cfg_a.get("analysis", {}).get("comparison_hmf_mass_range_msun_h")
        range_b = cfg_b.get("analysis", {}).get("comparison_hmf_mass_range_msun_h")
        range_a_valid = _valid_hmf_mass_interval(range_a)
        range_b_valid = _valid_hmf_mass_interval(range_b)
        range_compatible = range_a_valid and range_b_valid and list(range_a) == list(range_b)
        bins_a = cfg_a.get("analysis", {}).get("mass_bins")
        bins_b = cfg_b.get("analysis", {}).get("mass_bins")
        bins_compatible = (
            isinstance(bins_a, int) and not isinstance(bins_a, bool) and bins_a > 0
            and bins_a == bins_b
        )
        halo_selection_compatible = (
            definition_a == definition_b and min_particles_a >= 0
            and min_particles_a == min_particles_b
        )
        hmf_compatible = halo_selection_compatible and range_compatible and bins_compatible

        def snapshot_records(file):
            records = []
            for key in file.keys():
                if not key.startswith("snap_"):
                    continue
                try:
                    redshift = float(file[key].attrs["redshift"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(redshift):
                    records.append((redshift, key))
            return sorted(records)

        records_a, records_b = snapshot_records(a), snapshot_records(b)
        remaining_b = {key: z for z, key in records_b}
        pairs = []
        for za, key_a in records_a:
            choices = [(abs(za - zb), key_b, zb) for key_b, zb in remaining_b.items()]
            if not choices:
                summary["unmatched_run_a"].append({"snapshot": key_a, "redshift": za, "reason": "no unmatched snapshot remains in run B"})
                continue
            delta, key_b, zb = min(choices)
            if delta > tolerance * max(1.0, abs(za), abs(zb)):
                summary["unmatched_run_a"].append({"snapshot": key_a, "redshift": za, "reason": "no run B snapshot at matching redshift"})
                continue
            pairs.append((za, key_a, zb, key_b))
            del remaining_b[key_b]
        summary["unmatched_run_b"] = [
            {"snapshot": key, "redshift": z, "reason": "no run A snapshot at matching redshift"}
            for key, z in sorted(remaining_b.items())
        ]

        out.attrs["run_a"] = str(run_a)
        out.attrs["run_b"] = str(run_b)
        out.attrs["model_a"] = model_a
        out.attrs["model_b"] = model_b
        out.attrs["hmf_mass_definition_a"] = definition_a
        out.attrs["hmf_mass_definition_b"] = definition_b
        out.attrs["hmf_min_particles_a"] = min_particles_a
        out.attrs["hmf_min_particles_b"] = min_particles_b
        out.attrs["redshift_tolerance"] = tolerance

        hmf_entries, pk_entries = [], []
        pair_epochs={}
        pair_group = out.create_group("pairs")
        for index, (z, key_a, zb, key_b) in enumerate(pairs):
            pair_name = "pair_%04d" % index
            pair_out = pair_group.create_group(pair_name)
            pair_out.attrs["snapshot_a"] = key_a
            pair_out.attrs["snapshot_b"] = key_b
            pair_out.attrs["redshift_a"] = z
            pair_out.attrs["redshift_b"] = zb
            from core.data import run_epoch_contract
            pair_epochs[pair_name]={'run_a':run_epoch_contract(run_a,int(key_a.split('_')[-1]),1/(1+z)),
                'run_b':run_epoch_contract(run_b,int(key_b.split('_')[-1]),1/(1+zb))}
            pair_out.attrs['paired_epoch_contract_json']=json.dumps(pair_epochs[pair_name],allow_nan=False)
            pair_summary = {
                "id": pair_name, "snapshot_a": key_a, "snapshot_b": key_b,
                "redshift_a": z, "redshift_b": zb, "redshift_delta": abs(z - zb),
                "hmf_status": "SKIPPED", "pk_status": "SKIPPED",
            }
            group_a, group_b = a[key_a], b[key_b]
            secondary = {}

            box_a = cfg_a.get("simulation", {}).get("box_mpc_h")
            box_b = cfg_b.get("simulation", {}).get("box_mpc_h")
            common_hmf = None
            if hmf_compatible and box_a and box_b:
                common_hmf = _common_hmf_bin_comparison(
                    group_a, group_b, float(box_a)**3, float(box_b)**3,
                    bins_a, mass_range=range_a,
                    phase_agreement=initial_phase_agreement,
                )
                if common_hmf is None:
                    common_hmf = {
                        "status": "SKIPPED",
                        "reason": "saved host-halo masses and HMF selections are unavailable",
                    }
                pair_summary["hmf_common_bins_status"] = common_hmf["status"]
                if common_hmf["status"] == "COMPLETE":
                    common_group = pair_out.create_group("hmf_common_bins")
                    for source_key, dataset_name in [
                        ("mass_edges_msun_h", "mass_edges_msun_h"),
                        ("mass_centers_msun_h", "mass_centers_msun_h"),
                        ("counts_a", "counts_a"), ("counts_b", "counts_b"),
                        ("dn_dlogM_a", "dn_dlogM_a"), ("dn_dlogM_b", "dn_dlogM_b"),
                        ("ratio_b_over_a", "ratio_b_over_a"),
                        ("ratio_poisson_68_lower", "ratio_poisson_68_lower"),
                        ("ratio_poisson_68_upper", "ratio_poisson_68_upper"),
                        ("ratio_paired_jackknife_68_lower", "ratio_paired_jackknife_68_lower"),
                        ("ratio_paired_jackknife_68_upper", "ratio_paired_jackknife_68_upper"),
                        ("log_ratio_paired_jackknife_covariance", "log_ratio_paired_jackknife_covariance"),
                        ("log_hmf_jackknife_covariance_a", "log_hmf_jackknife_covariance_a"),
                        ("log_hmf_jackknife_covariance_b", "log_hmf_jackknife_covariance_b"),
                        ("log_hmf_jackknife_cross_covariance_a_b", "log_hmf_jackknife_cross_covariance_a_b"),
                    ]:
                        common_group.create_dataset(dataset_name, data=common_hmf[source_key])
                    common_group.attrs["poisson_interval"] = common_hmf["poisson_interval"]
                    common_group.attrs["finite_volume_sample_variance"] = common_hmf["finite_volume_sample_variance"]
                    common_group.attrs["paired_jackknife_status"] = common_hmf["paired_jackknife_status"]
                    common_group.attrs["paired_jackknife_method_or_reason"] = common_hmf["paired_jackknife_method_or_reason"]
                    common_group.attrs["binning"] = common_hmf["binning"]
                    pair_summary["hmf_common_bins"] = {
                        "bin_count": int(len(common_hmf["mass_centers_msun_h"])),
                        "empty_or_zero_count_bins_a": int(np.count_nonzero(common_hmf["counts_a"] == 0)),
                        "empty_or_zero_count_bins_b": int(np.count_nonzero(common_hmf["counts_b"] == 0)),
                        "poisson_interval": common_hmf["poisson_interval"],
                        "finite_volume_sample_variance": common_hmf["finite_volume_sample_variance"],
                        "paired_jackknife_status": common_hmf["paired_jackknife_status"],
                        "paired_jackknife_method_or_reason": common_hmf["paired_jackknife_method_or_reason"],
                        "binning": common_hmf["binning"],
                    }
                else:
                    pair_summary["hmf_common_bins_reason"] = common_hmf["reason"]
            else:
                pair_summary["hmf_common_bins_status"] = "SKIPPED"
                if not halo_selection_compatible:
                    pair_summary["hmf_common_bins_reason"] = "mass definition or minimum particle cut differs between runs"
                elif not range_compatible:
                    pair_summary["hmf_common_bins_reason"] = "both runs must predeclare the same finite, positive, increasing HMF comparison mass interval"
                elif not bins_compatible:
                    pair_summary["hmf_common_bins_reason"] = "both runs must use the same positive integer HMF bin count"
                else:
                    pair_summary["hmf_common_bins_reason"] = "both runs must have valid, positive simulation box sizes"
            resolution_report = hmf_resolution_completeness(
                group_a, group_b, common_hmf, cfg_a, cfg_b, initial_phase_agreement,
                resolution_source_status, float(box_a) if box_a else float("nan"),
            )
            pair_summary["hmf_resolution_completeness"] = resolution_report
            pair_out.attrs["hmf_resolution_completeness_json"] = json.dumps(
                resolution_report, sort_keys=True,
            )

            def compare_positive_curve(metric, x_a, y_a, x_b, y_b, xlabel):
                series_a = _clean_positive_series(x_a, y_a)
                series_b = _clean_positive_series(x_b, y_b)
                if series_a is None or series_b is None:
                    secondary[metric] = {"status": "SKIPPED", "reason": "fewer than two positive, finite data points in one or both runs"}
                    return
                xa, ya = series_a; xb, yb = series_b
                lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
                if hi <= lo:
                    secondary[metric] = {"status": "SKIPPED", "reason": "positive x ranges do not overlap"}
                    return
                xgrid = np.geomspace(lo, hi, 120)
                ratio = _interp_loglog(xgrid, xb, yb) / _interp_loglog(xgrid, xa, ya)
                pair_out.create_dataset(metric + "_x", data=xgrid)
                pair_out.create_dataset(metric + "_ratio_b_over_a", data=ratio)
                plt.figure(figsize=(8, 5)); plt.semilogx(xgrid, ratio); plt.axhline(1.0, linewidth=1)
                plt.xlabel(xlabel); plt.ylabel("B / A"); plt.title(metric.replace("_", " ") + " ratio at z=%.6g" % z)
                savefig(outdir / (metric + "_ratio_" + pair_name + ".png"))
                secondary[metric] = {"status": "COMPLETE"}

            if not hmf_compatible:
                pair_summary["hmf_reason"] = (
                    "mass definition or minimum particle cut differs between runs"
                    if not halo_selection_compatible
                    else "a shared, predeclared comparison_hmf_mass_range_msun_h and identical positive mass_bins are required"
                )
            elif common_hmf is None or common_hmf.get("status") != "COMPLETE":
                pair_summary["hmf_reason"] = "common-bin HMF counts are unavailable for the fixed comparison interval"
            else:
                centers = np.asarray(common_hmf["mass_centers_msun_h"], dtype=float)
                counts_a = np.asarray(common_hmf["counts_a"], dtype=np.int64)
                counts_b = np.asarray(common_hmf["counts_b"], dtype=np.int64)
                valid = (counts_a > 0) & (counts_b > 0) & np.isfinite(common_hmf["ratio_b_over_a"])
                if np.count_nonzero(valid) < 2:
                    pair_summary["hmf_reason"] = "fewer than two fixed common mass bins contain halos in both runs"
                else:
                    ratio = np.asarray(common_hmf["ratio_b_over_a"], dtype=float)
                    poisson_lower = np.asarray(common_hmf["ratio_poisson_68_lower"], dtype=float)
                    poisson_upper = np.asarray(common_hmf["ratio_poisson_68_upper"], dtype=float)
                    paired_lower = np.asarray(common_hmf["ratio_paired_jackknife_68_lower"], dtype=float)
                    paired_upper = np.asarray(common_hmf["ratio_paired_jackknife_68_upper"], dtype=float)
                    pair_out.create_dataset("hmf_mass_bin_centers_msun_h", data=centers)
                    pair_out.create_dataset("hmf_counts_a", data=counts_a)
                    pair_out.create_dataset("hmf_counts_b", data=counts_b)
                    pair_out.create_dataset("hmf_ratio_b_over_a", data=ratio)
                    pair_out.create_dataset("hmf_ratio_poisson_68_lower", data=poisson_lower)
                    pair_out.create_dataset("hmf_ratio_poisson_68_upper", data=poisson_upper)
                    pair_out.create_dataset("hmf_ratio_paired_jackknife_68_lower", data=paired_lower)
                    pair_out.create_dataset("hmf_ratio_paired_jackknife_68_upper", data=paired_upper)
                    axis = plt.figure(figsize=(8, 5)).gca()
                    if common_hmf["paired_jackknife_status"] == "COMPLETE":
                        axis.fill_between(
                            centers, paired_lower, paired_upper, step="mid", alpha=0.2,
                            label="paired spatial jackknife interval",
                        )
                    axis.errorbar(
                        centers[valid], ratio[valid],
                        yerr=[
                            ratio[valid] - poisson_lower[valid],
                            poisson_upper[valid] - ratio[valid],
                        ], fmt="o", capsize=2, label="independent-Poisson count interval",
                    )
                    axis.axhline(1.0, linewidth=1)
                    axis.set_xscale("log")
                    axis.set_xlabel(r"$M\ [M_\odot/h]$"); axis.set_ylabel("B / A")
                    axis.set_title("Fixed-bin host HMF ratio at z=%.6g" % z)
                    axis.legend()
                    savefig(outdir / ("HMF_ratio_%s.png" % pair_name))
                    hmf_entries.append((pair_name, z, np.asarray(common_hmf["mass_edges_msun_h"], dtype=float), ratio))
                    pair_summary["hmf_status"] = "COMPLETE"
                    pair_summary["hmf_mass_definition"] = definition_a
                    pair_summary["hmf_min_particles"] = min_particles_a
                    pair_summary["hmf_binning"] = common_hmf["binning"]

            pk_a = _positive_series(group_a, "pk_k", "pk_P")
            pk_b = _positive_series(group_b, "pk_k", "pk_P")
            if pk_a is None or pk_b is None:
                pair_summary["pk_reason"] = "one or both snapshots lack at least two positive P(k) bins"
            else:
                ka, pa = pk_a
                kb, pb = pk_b
                lo, hi = max(ka[0], kb[0]), min(ka[-1], kb[-1])
                if hi <= lo:
                    pair_summary["pk_reason"] = "positive P(k) ranges do not overlap"
                else:
                    k_grid = np.geomspace(lo, hi, 120)
                    ratio = _interp_loglog(k_grid, kb, pb) / _interp_loglog(k_grid, ka, pa)
                    pair_out.create_dataset("pk_k_h_mpc", data=k_grid)
                    pair_out.create_dataset("pk_ratio_b_over_a", data=ratio)
                    plt.figure(figsize=(8, 5)); plt.semilogx(k_grid, ratio); plt.axhline(1.0, linewidth=1)
                    plt.xlabel(r"$k\ [h/\mathrm{Mpc}]$"); plt.ylabel("B / A")
                    plt.title("Full-particle P(k) ratio at z=%.6g" % z)
                    savefig(outdir / ("Pk_ratio_%s.png" % pair_name))
                    pk_entries.append((pair_name, z, lo, hi, ka, pa, kb, pb))
                    pair_summary["pk_status"] = "COMPLETE"

            xi_a = _signed_series(group_a, "xi_r", "xi")
            xi_b = _signed_series(group_b, "xi_r", "xi")
            if xi_a is None or xi_b is None:
                secondary["xi"] = {"status": "SKIPPED", "reason": "one or both snapshots lack two finite correlation bins"}
            else:
                xa, ya = xi_a; xb, yb = xi_b
                lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
                if hi <= lo:
                    secondary["xi"] = {"status": "SKIPPED", "reason": "correlation-function ranges do not overlap"}
                else:
                    xgrid = np.geomspace(lo, hi, 120)
                    va = _interp_signed_logx(xgrid, xa, ya)
                    vb = _interp_signed_logx(xgrid, xb, yb)
                    floor = 1.0e-8 * max(float(np.max(np.abs(ya))), float(np.max(np.abs(yb))), np.finfo(float).tiny)
                    valid = np.isfinite(va) & np.isfinite(vb) & (va * vb > 0) & (np.abs(va) > floor) & (np.abs(vb) > floor)
                    ratio = np.full_like(xgrid, np.nan)
                    ratio[valid] = vb[valid] / va[valid]
                    if np.count_nonzero(valid) < 2:
                        secondary["xi"] = {"status": "SKIPPED", "reason": "fewer than two bins have same-sign correlation values away from zero"}
                    else:
                        pair_out.create_dataset("xi_r", data=xgrid)
                        pair_out.create_dataset("xi_ratio_b_over_a", data=ratio)
                        plt.figure(figsize=(8, 5)); plt.semilogx(xgrid, ratio); plt.axhline(1.0, linewidth=1)
                        plt.xlabel(r"$r\ [\mathrm{Mpc}/h]$"); plt.ylabel("B / A where both ξ values share a sign")
                        plt.title("Two-point correlation ratio at z=%.6g" % z)
                        savefig(outdir / ("xi_ratio_" + pair_name + ".png"))
                        secondary["xi"] = {"status": "COMPLETE", "valid_ratio_bins": int(np.count_nonzero(valid))}

            for metric, x_name, y_name, xlabel in [
                ("bias", "bias_k", "bias", r"$k\ [h/\mathrm{Mpc}]$"),
                ("subhalo_hmf", "subhmf_mass", "subhmf_dn", r"$M_{sub}\ [M_\odot/h]$"),
            ]:
                values_a = _positive_series(group_a, x_name, y_name)
                values_b = _positive_series(group_b, x_name, y_name)
                if values_a is None or values_b is None:
                    secondary[metric] = {"status": "SKIPPED", "reason": "one or both snapshots lack two positive data points"}
                else:
                    compare_positive_curve(metric, values_a[0], values_a[1], values_b[0], values_b[1], xlabel)

            concentration_a = (group_a["concentration_mass_msun_h"][:], group_a["concentration_cvir"][:]) if "concentration_mass_msun_h" in group_a and "concentration_cvir" in group_a else None
            concentration_b = (group_b["concentration_mass_msun_h"][:], group_b["concentration_cvir"][:]) if "concentration_mass_msun_h" in group_b and "concentration_cvir" in group_b else None
            if concentration_a is None or concentration_b is None:
                secondary["concentration"] = {"status": "SKIPPED", "reason": "one or both snapshots lack concentration data"}
            else:
                ma, ca = (np.asarray(v, dtype=float) for v in concentration_a)
                mb, cb = (np.asarray(v, dtype=float) for v in concentration_b)
                good_a = np.isfinite(ma) & (ma > 0) & np.isfinite(ca) & (ca > 0)
                good_b = np.isfinite(mb) & (mb > 0) & np.isfinite(cb) & (cb > 0)
                ma, ca, mb, cb = ma[good_a], ca[good_a], mb[good_b], cb[good_b]
                if len(ma) < 10 or len(mb) < 10:
                    secondary["concentration"] = {"status": "SKIPPED", "reason": "fewer than ten valid halos in one or both concentration samples"}
                else:
                    lo, hi = max(ma.min(), mb.min()), min(ma.max(), mb.max())
                    if hi <= lo:
                        secondary["concentration"] = {"status": "SKIPPED", "reason": "concentration mass ranges do not overlap"}
                    else:
                        edges = np.geomspace(lo, hi, 11)
                        centers = np.sqrt(edges[:-1] * edges[1:])
                        median_a = np.full(10, np.nan); median_b = np.full(10, np.nan)
                        count_a = np.zeros(10, dtype=int); count_b = np.zeros(10, dtype=int)
                        for bin_index in range(10):
                            right_a = (ma <= edges[bin_index + 1]) if bin_index == 9 else (ma < edges[bin_index + 1])
                            right_b = (mb <= edges[bin_index + 1]) if bin_index == 9 else (mb < edges[bin_index + 1])
                            in_a = (ma >= edges[bin_index]) & right_a
                            in_b = (mb >= edges[bin_index]) & right_b
                            count_a[bin_index], count_b[bin_index] = np.count_nonzero(in_a), np.count_nonzero(in_b)
                            if count_a[bin_index] >= 5: median_a[bin_index] = np.median(ca[in_a])
                            if count_b[bin_index] >= 5: median_b[bin_index] = np.median(cb[in_b])
                        valid = np.isfinite(median_a) & np.isfinite(median_b) & (median_a > 0) & (median_b > 0)
                        if np.count_nonzero(valid) < 2:
                            secondary["concentration"] = {"status": "SKIPPED", "reason": "fewer than two common mass bins contain five halos in each run"}
                        else:
                            ratio = median_b[valid] / median_a[valid]
                            pair_out.create_dataset("concentration_mass_bin_centers_msun_h", data=centers[valid])
                            pair_out.create_dataset("concentration_median_ratio_b_over_a", data=ratio)
                            pair_out.create_dataset("concentration_count_a", data=count_a)
                            pair_out.create_dataset("concentration_count_b", data=count_b)
                            plt.figure(figsize=(8, 5)); plt.semilogx(centers[valid], ratio, marker="o"); plt.axhline(1.0, linewidth=1)
                            plt.xlabel(r"$M_{vir}\ [M_\odot/h]$"); plt.ylabel("median concentration B / A")
                            plt.title("Concentration-mass ratio at z=%.6g" % z)
                            savefig(outdir / ("concentration_ratio_" + pair_name + ".png"))
                            secondary["concentration"] = {"status": "COMPLETE", "common_mass_bins": int(np.count_nonzero(valid))}

            for metric, edge_name, pdf_name, xlabel in [
                ("dm_velocity_pdf", "dm_speed_bin_edges", "dm_speed_probability_density", "dark-matter particle speed [snapshot units]"),
                ("halo_velocity_pdf", "halo_speed_bin_edges", "halo_speed_probability_density", "halo speed [catalogue units]"),
            ]:
                if edge_name not in group_a or pdf_name not in group_a or edge_name not in group_b or pdf_name not in group_b:
                    secondary[metric] = {"status": "SKIPPED", "reason": "one or both snapshots lack velocity distribution data"}
                    continue
                edges_a = np.asarray(group_a[edge_name][:], dtype=float); pdf_a = np.asarray(group_a[pdf_name][:], dtype=float)
                edges_b = np.asarray(group_b[edge_name][:], dtype=float); pdf_b = np.asarray(group_b[pdf_name][:], dtype=float)
                series_a = _clean_positive_series(0.5 * (edges_a[:-1] + edges_a[1:]), pdf_a)
                series_b = _clean_positive_series(0.5 * (edges_b[:-1] + edges_b[1:]), pdf_b)
                if series_a is None or series_b is None:
                    secondary[metric] = {"status": "SKIPPED", "reason": "velocity PDFs have fewer than two positive bins"}
                else:
                    compare_positive_curve(metric, series_a[0], series_a[1], series_b[0], series_b[1], xlabel)

            if not matching_ready:
                secondary["matched_halos"] = {"status": "SKIPPED", "reason": matching_reason}
            elif not halo_selection_compatible:
                secondary["matched_halos"] = {"status": "SKIPPED", "reason": "host mass definition or minimum particle cut differs between runs"}
            else:
                min_particles = min_particles_a
                max_catalogue_ids = min(max_ids_a, max_ids_b)
                try:
                    snap_a = int(key_a.split("_")[-1])
                    snap_b = int(key_b.split("_")[-1])
                    catalogue_a = _read_cross_run_halo_catalogue(run_a, snap_a, definition_a, min_particles, max_catalogue_ids)
                    catalogue_b = _read_cross_run_halo_catalogue(run_b, snap_b, definition_b, min_particles, max_catalogue_ids)
                    box_a, box_b = catalogue_a["box_mpc_h"], catalogue_b["box_mpc_h"]
                    if box_a is None or box_b is None:
                        raise ValueError("catalogue Header/BoxSize is required for periodic position shifts")
                    if catalogue_a["redshift"] is None or catalogue_b["redshift"] is None:
                        raise ValueError("catalogue Header/Redshift is required to validate the matched output time")
                    if not np.isclose(box_a, identity_a["box_mpc_h"], rtol=1.0e-6, atol=0.0) or not np.isclose(box_b, identity_b["box_mpc_h"], rtol=1.0e-6, atol=0.0):
                        raise ValueError("catalogue Header/BoxSize does not agree with its run manifest")
                    if abs(catalogue_a["redshift"] - z) > tolerance * max(1.0, abs(z)) or abs(catalogue_b["redshift"] - zb) > tolerance * max(1.0, abs(zb)):
                        raise ValueError("catalogue Header/Redshift does not match its analysis snapshot")
                    if not np.isclose(box_a, box_b, rtol=1.0e-12, atol=0.0):
                        raise ValueError("paired catalogue BoxSize values differ")
                    matches = _mutual_best_halo_matches(
                        catalogue_a, catalogue_b, min_shared_a, min_fraction_a,
                    )
                    unmatched_a = np.setdiff1d(catalogue_a["group_indices"], matches["a"], assume_unique=False)
                    unmatched_b = np.setdiff1d(catalogue_b["group_indices"], matches["b"], assume_unique=False)
                    matched_out = pair_out.create_group("matched_halos")
                    matched_out.attrs["mass_definition"] = definition_a
                    matched_out.attrs["minimum_particles_per_host"] = min_particles
                    matched_out.attrs["minimum_shared_dark_matter_particles"] = min_shared_a
                    matched_out.attrs["minimum_overlap_fraction_per_host"] = min_fraction_a
                    matched_out.attrs["box_mpc_h"] = box_a
                    matched_out.attrs["overlap_merit"] = "shared_count / sqrt(n_dm_a * n_dm_b)"
                    matched_out.attrs["membership_source_a"] = str(catalogue_a.get("membership_source", "unknown"))
                    matched_out.attrs["membership_source_b"] = str(catalogue_b.get("membership_source", "unknown"))
                    matched_out.create_dataset("group_index_a", data=matches["a"])
                    matched_out.create_dataset("group_index_b", data=matches["b"])
                    matched_out.create_dataset("shared_dark_matter_particle_count", data=matches["shared"])
                    rows_a = {int(index): row for row, index in enumerate(catalogue_a["group_indices"])}
                    rows_b = {int(index): row for row, index in enumerate(catalogue_b["group_indices"])}
                    matched_out.create_dataset("dark_matter_particle_count_a", data=[catalogue_a["n_dm"][rows_a[int(i)]] for i in matches["a"]])
                    matched_out.create_dataset("dark_matter_particle_count_b", data=[catalogue_b["n_dm"][rows_b[int(i)]] for i in matches["b"]])
                    matched_out.create_dataset("overlap_fraction_a", data=matches["fraction_a"])
                    matched_out.create_dataset("overlap_fraction_b", data=matches["fraction_b"])
                    matched_out.create_dataset("overlap_merit", data=matches["merit"])
                    matched_out.create_dataset("unmatched_group_indices_a", data=unmatched_a)
                    matched_out.create_dataset("unmatched_group_indices_b", data=unmatched_b)
                    if len(matches["a"]):
                        matched_values = _save_matched_halo_figures(outdir, pair_name, z, matches, catalogue_a, catalogue_b, box_a)
                        matched_out.create_dataset("mass_a_msun_h", data=matched_values["mass_a"])
                        matched_out.create_dataset("mass_b_msun_h", data=matched_values["mass_b"])
                        matched_out.create_dataset("mass_ratio_b_over_a", data=matched_values["mass_ratio"])
                        matched_out.create_dataset("position_a_mpc_h", data=matched_values["position_a"])
                        matched_out.create_dataset("position_b_mpc_h", data=matched_values["position_b"])
                        matched_out.create_dataset("position_delta_b_minus_a_mpc_h", data=matched_values["position_delta"])
                        matched_out.create_dataset("position_displacement_mpc_h", data=matched_values["position_displacement"])
                        matched_out.create_dataset("r200_critical_a_mpc_h", data=matched_values["r200c_a"])
                        matched_out.create_dataset("r200_critical_b_mpc_h", data=matched_values["r200c_b"])
                        matched_out.create_dataset("r200_mean_a_mpc_h", data=matched_values["r200m_a"])
                        matched_out.create_dataset("r200_mean_b_mpc_h", data=matched_values["r200m_b"])
                        matched_out.create_dataset("central_subhalo_vmax_a", data=matched_values["vmax_a"])
                        matched_out.create_dataset("central_subhalo_vmax_b", data=matched_values["vmax_b"])
                        matched_out.create_dataset("central_subhalo_vmax_ratio_b_over_a", data=matched_values["vmax_ratio"])
                        figures = ["matched_halo_mass_ratio_%s.png" % pair_name, "matched_halo_position_shift_%s.png" % pair_name]
                        if np.any(np.isfinite(matched_values["vmax_ratio"]) & (matched_values["vmax_ratio"] > 0)):
                            figures.append("matched_halo_vmax_ratio_%s.png" % pair_name)
                        matched_status = "COMPLETE"
                        matched_reason = None
                    else:
                        figures = []
                        matched_status = "SKIPPED"
                        matched_reason = "no reciprocal-best host matches passed the shared-particle and overlap-fraction thresholds"
                    secondary["matched_halos"] = {
                        "status": matched_status, "reason": matched_reason,
                        "eligible_host_count_a": int(len(catalogue_a["group_indices"])),
                        "eligible_host_count_b": int(len(catalogue_b["group_indices"])),
                        "candidate_edge_count": int(matches["candidate_edges"]),
                        "matched_host_count": int(len(matches["a"])),
                        "unmatched_host_count_a": int(len(unmatched_a)),
                        "unmatched_host_count_b": int(len(unmatched_b)),
                        "membership_source_a": catalogue_a.get("membership_source", "unknown"),
                        "membership_source_b": catalogue_b.get("membership_source", "unknown"),
                        "vmax_status": "COMPLETE" if any(name.startswith("matched_halo_vmax_ratio_") for name in figures) else "UNAVAILABLE_OR_NO_VALID_VALUES",
                        "figures": figures,
                    }
                except (OSError, KeyError, ValueError, OverflowError) as exc:
                    secondary["matched_halos"] = {"status": "SKIPPED", "reason": str(exc)}

            secondary.update(_add_paired_hydro_snapshot_products(
                pair_out, group_a, group_b, pair_name, z, outdir,
            ))
            pair_summary["secondary"] = secondary
            summary["pairs"].append(pair_summary)

        summary["star_formation_history_comparison"] = _add_paired_sfr_history(
            out, a.get("star_formation_history"), b.get("star_formation_history"), outdir,
        )

        if pairs:
            final_pair_index = min(range(len(pairs)), key=lambda index: (pairs[index][0] + pairs[index][2]) / 2.0)
            final_redshift, final_key_a, _, final_key_b = pairs[final_pair_index]
            final_pair_name = "pair_%04d" % final_pair_index
            final_pair_out = pair_group[final_pair_name]
            final_pair_summary = summary["pairs"][final_pair_index]
            if not matching_ready:
                formation_result = {"status": "SKIPPED", "reason": matching_reason}
                final_pair_summary["secondary"]["formation_redshift_shift"] = formation_result
                summary["halo_matching"]["formation_redshift_shift"] = {"pair": final_pair_name, **formation_result}
            elif not halo_selection_compatible:
                formation_result = {"status": "SKIPPED", "reason": "host mass definition or minimum particle cut differs between runs"}
                final_pair_summary["secondary"]["formation_redshift_shift"] = formation_result
                summary["halo_matching"]["formation_redshift_shift"] = {"pair": final_pair_name, **formation_result}
            else:
                _add_fof_formation_redshift_shift(
                    run_a, run_b, records_a, records_b, final_key_a, final_key_b,
                    final_pair_name, final_pair_out, final_pair_summary, summary,
                    definition_a, min_particles_a, min_shared_a, min_fraction_a,
                    min(max_ids_a, max_ids_b), identity_a, identity_b, outdir,
                )

        def build_heatmap(entries, x_name, y_name, filename, xlabel, h5_name):
            if len(entries) < 2:
                summary["heatmaps"][h5_name] = "SKIPPED: fewer than two valid matched redshifts"
                return
            lo = max(entry[2] for entry in entries)
            hi = min(entry[3] for entry in entries)
            if hi <= lo:
                summary["heatmaps"][h5_name] = "SKIPPED: snapshots have no common positive x range"
                return
            x = np.geomspace(lo, hi, 100)
            rows = []
            for _, _, _, _, xa, ya, xb, yb in entries:
                ratio = _interp_loglog(x, xb, yb) / _interp_loglog(x, xa, ya)
                rows.append(np.log10(ratio))
            redshifts = np.asarray([entry[1] for entry in entries], dtype=float)
            order = np.argsort(redshifts)
            matrix = np.vstack(rows)[order]
            redshifts = redshifts[order]
            out.create_dataset(h5_name + "_x", data=x)
            out.create_dataset(h5_name + "_redshift", data=redshifts)
            out.create_dataset(h5_name + "_log10_ratio", data=matrix)
            finite = matrix[np.isfinite(matrix)]
            limit = max(float(np.max(np.abs(finite))), 1.0e-6) if finite.size else 1.0
            plt.figure(figsize=(10, 7))
            mesh = plt.pcolormesh(np.log10(x), redshifts, matrix, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
            plt.colorbar(mesh, label=r"$\log_{10}(B/A)$")
            plt.xlabel(xlabel); plt.ylabel("redshift")
            savefig(outdir / filename)
            summary["heatmaps"][h5_name] = "COMPLETE"

        def build_hmf_heatmap(entries):
            name = "hmf_ratio"
            if len(entries) < 2:
                if not halo_selection_compatible:
                    reason = "host mass definition or minimum particle cut differs between runs"
                elif not range_compatible:
                    reason = "a shared, predeclared HMF comparison mass interval is required"
                elif not bins_compatible:
                    reason = "both runs must use the same positive integer HMF bin count"
                else:
                    reason = "fewer than two matched redshifts have at least two populated common HMF bins"
                summary["heatmaps"][name] = "SKIPPED: " + reason
                return
            mass_edges = entries[0][2]
            if any(not np.array_equal(entry[2], mass_edges) for entry in entries[1:]):
                summary["heatmaps"][name] = "SKIPPED: matched snapshots do not share identical fixed HMF bin edges"
                return
            redshifts = np.asarray([entry[1] for entry in entries], dtype=float)
            order = np.argsort(redshifts)
            redshifts = redshifts[order]
            if np.any(np.diff(redshifts) <= 0):
                summary["heatmaps"][name] = "SKIPPED: matched HMF snapshots have duplicate redshifts"
                return
            ratios = np.vstack([entry[3] for entry in entries])[order]
            with np.errstate(divide="ignore", invalid="ignore"):
                matrix = np.log10(ratios)
            finite = matrix[np.isfinite(matrix)]
            if finite.size == 0:
                summary["heatmaps"][name] = "SKIPPED: no shared mass bins are populated in both runs"
                return
            centers = np.sqrt(mass_edges[:-1] * mass_edges[1:])
            out.create_dataset(name + "_mass_edges_msun_h", data=mass_edges)
            out.create_dataset(name + "_mass_centers_msun_h", data=centers)
            out.create_dataset(name + "_redshift", data=redshifts)
            out.create_dataset(name + "_log10_ratio", data=matrix)
            limit = max(float(np.max(np.abs(finite))), 1.0e-6)
            plt.figure(figsize=(10, 7))
            mesh = plt.pcolormesh(
                np.log10(centers), redshifts, np.ma.masked_invalid(matrix),
                shading="nearest", cmap="coolwarm", vmin=-limit, vmax=limit,
            )
            plt.colorbar(mesh, label=r"$\log_{10}(B/A)$")
            plt.xlabel(r"$\log_{10}M\ [M_\odot/h]$"); plt.ylabel("redshift")
            savefig(outdir / "HMF_ratio_mass_redshift.png")
            summary["heatmaps"][name] = "COMPLETE"

        build_hmf_heatmap(hmf_entries)
        build_heatmap(pk_entries, "k", "redshift", "Pk_ratio_k_redshift.png", r"$\log_{10}k\ [h/\mathrm{Mpc}]$", "pk_ratio")

    from core.data import attach_comparison_contracts
    with h5py.File(outdir/'analysis_comparison.h5','r+') as comparison:
        context=json.loads((outdir/'comparison_inputs.json').read_text())
        attach_comparison_contracts(comparison,context,pair_epochs)
    from figures import generate_paired_story_figures
    summary["paper_figure_recipes"] = generate_paired_story_figures(
        outdir / "analysis_comparison.h5", outdir / "paper_figures",
        cfg_b.get("visualization", {}),
    )
    from core.foundation import atomic_json
    atomic_json(outdir / "comparison_summary.json",summary)
    return str(outdir)


def verify_comparison_bundle(directory):
    """Verify sealed paired artifacts and numeric payloads before consumption."""
    from core.foundation import file_sha256 as digest
    from core.data import verify_hdf5_dataset_payloads
    root=Path(directory).resolve();path=root/'comparison_manifest.json';before=digest(path)
    if (root/'comparison_manifest.sha256').read_text().strip()!=before:
        raise ValueError('Comparison manifest checksum mismatch')
    record=json.loads(path.read_text())
    if record.get('schema')!='paired-analysis-publication-v1':raise ValueError('Unknown comparison publication schema')
    for name,expected in record['artifacts'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or digest(target)!=expected:
            raise ValueError('Comparison artifact missing or changed: '+name)
        if target.suffix in ('.h5','.hdf5'):verify_hdf5_dataset_payloads(target,require_contract=True)
    if digest(path)!=before:raise RuntimeError('Comparison manifest changed during reading')
    return record


@directory_owner('comparison_dir')
def create_paired_analysis_report(run_a, run_b, comparison_dir, structure_map_dir=None):
    """Build one report with paired metrics, assessed convergence, and map overlays."""
    from matplotlib.backends.backend_pdf import PdfPages
    if (Path(comparison_dir)/'comparison_manifest.json').is_file():
        verify_comparison_bundle(comparison_dir)

    run_a, run_b = Path(run_a), Path(run_b)
    comparison_dir = Path(comparison_dir)
    summary_path = comparison_dir / "comparison_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("paired comparison summary is missing: " + str(summary_path))
    summary_before=_file_sha256(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if _file_sha256(summary_path)!=summary_before:raise ValueError("Paired summary changed while reading")
    from core.foundation import atomic_json, atomic_text, atomic_copy, file_sha256 as digest, publish_directory
    report = comparison_dir / "paired_analysis_report.pdf"
    if report.exists():raise FileExistsError('Keep completed paired report; use a fresh comparison generation')
    stage=Path(tempfile.mkdtemp(prefix='.paired-report-stage-',dir=comparison_dir))
    history=comparison_dir/'paired_reports';history.mkdir(exist_ok=True)
    final=history/uuid.uuid4().hex
    temporary=stage/'paired_analysis_report.pdf'
    source_hashes={str(summary_path.resolve()):summary_before}
    details = [
        "Scientific acceptance: NOT_ACCEPTED",
        "Run A: %s" % run_a,
        "Run B: %s" % run_b,
        "Matched redshift pairs: %d" % len(summary.get("pairs", [])),
        "Initial phase agreement: %s" % summary.get("initial_ic_phase_agreement", {}).get("status", "UNAVAILABLE"),
        "Halo matching: %s" % summary.get("halo_matching", {}).get("status", "UNAVAILABLE"),
        "Paired physical SFR: %s" % summary.get("star_formation_history_comparison", {}).get("status", "UNAVAILABLE"),
    ]
    for index, pair in enumerate(summary.get("pairs", [])):
        convergence = pair.get("hmf_resolution_completeness", {})
        mass_range = ""
        if convergence.get("status") == "MEASURED_CONVERGED_RANGE":
            mass_range = " | %.4g–%.4g M_sun/h" % (
                convergence.get("completeness_mass_limit_msun_h", float("nan")),
                convergence.get("validated_upper_mass_msun_h", float("nan")),
            )
        details.append(
            "%s (z=%.5g): HMF resolution %s%s%s" % (
                pair.get("id", "pair_%04d" % index), pair.get("redshift_a", float("nan")),
                convergence.get("status", "NOT_ASSESSED"), mass_range,
                " — " + str(convergence.get("reason")) if convergence.get("reason") else "",
            )
        )
    images = sorted(
        [*comparison_dir.glob("*.png"), *comparison_dir.glob("paper_figures/*.png")],
        key=lambda path: path.as_posix().lower(),
    )
    if structure_map_dir is not None:
        structure_map_dir = Path(structure_map_dir)
        map_summary_path = structure_map_dir / "comparison_maps_summary.json"
        map_files = set()
        if map_summary_path.is_file():
            source_hashes[str(map_summary_path.resolve())]=digest(map_summary_path)
            try:
                map_summary = json.loads(map_summary_path.read_text(encoding="utf-8"))
                def collect_map_files(value):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key == "figure_files" and isinstance(item, list):
                                map_files.update(str(name) for name in item)
                            else:
                                collect_map_files(item)
                    elif isinstance(value, list):
                        for item in value:
                            collect_map_files(item)
                collect_map_files(map_summary)
            except (OSError, ValueError):
                map_files = set()
        if map_files:
            for name in sorted(map_files):
                candidate=(structure_map_dir/name).resolve()
                if not candidate.is_relative_to(structure_map_dir.resolve()):
                    raise ValueError('Paired report map asset escapes its source directory')
                images.append(candidate)
        elif not map_summary_path.is_file():
            images.extend(sorted(structure_map_dir.glob("*.png"), key=lambda path: path.name.lower()))
    images = [path for path in images if path.is_file()]
    try:
        retained_images=[]
        for index,path in enumerate(images):
            expected=digest(path);source_hashes[str(path.resolve())]=expected
            sidecar=path.with_suffix('.figure.json')
            if sidecar.is_file():
                from figures import verify_figure_export
                verify_figure_export(sidecar)
                source_hashes[str(sidecar.resolve())]=digest(sidecar)
            copied=stage/'inputs'/('%04d_'%index+path.name)
            if atomic_copy(path,copied)!=expected:raise ValueError('Paired report image changed while copying')
            retained_images.append(copied)
        images=retained_images
        atomic_copy(summary_path,stage/'inputs/comparison_summary.json')
        values=comparison_dir/'analysis_comparison.h5'
        if values.is_file():
            source_hashes[str(values.resolve())]=digest(values)
            atomic_copy(values,stage/'inputs/analysis_comparison.h5')
        with PdfPages(temporary) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = "Paired simulation analysis report"
            metadata["Author"] = "DarkMatterSim"
            metadata["Subject"] = "Paired hydro, halo, convergence, and structure-map comparisons"
            fig, axis = plt.subplots(figsize=(8.5, 11.0))
            axis.set_axis_off()
            axis.text(0.07, 0.92, "PAIRED SIMULATION ANALYSIS", fontsize=10, color="#0f766e", weight="bold", transform=axis.transAxes)
            axis.text(0.07, 0.86, "%s vs %s" % (run_a.name, run_b.name), fontsize=22, color="#183047", weight="bold", transform=axis.transAxes)
            axis.plot([0.07, 0.93], [0.82, 0.82], color="#1b8a8f", linewidth=2, transform=axis.transAxes)
            y = 0.77
            for line in details:
                wrapped = textwrap.wrap(line, width=90) or [line]
                axis.text(0.08, y, "• " + "\n  ".join(wrapped), fontsize=9, color="#334155", va="top", transform=axis.transAxes)
                y -= 0.036 * len(wrapped) + 0.018
                if y < 0.10:
                    break
            axis.text(0.07, 0.06,
                      "Plots and machine-readable values are embedded in this report and saved in analysis_comparison.h5. "
                      "A convergence range appears only when the paired resolution test supports one.",
                      fontsize=8, color="#64748b", wrap=True, transform=axis.transAxes)
            pdf.savefig(fig); plt.close(fig)
            for image_path in images:
                try:
                    image = plt.imread(image_path)
                except (OSError, ValueError) as exc:
                    raise ValueError('Paired report image is unreadable: '+str(image_path)) from exc
                fig, axis = plt.subplots(figsize=(11.0, 8.5))
                axis.imshow(image)
                axis.set_title(image_path.name, fontsize=11, color="#183047")
                axis.axis("off")
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)
        if any(digest(path)!=expected for path,expected in source_hashes.items()):
            raise ValueError('Paired report input changed during rendering')
        if (comparison_dir/'comparison_manifest.json').is_file():verify_comparison_bundle(comparison_dir)
        record={'schema':'paired-report-publication-v1','science_acceptance':'NOT_ACCEPTED',
            'generation':final.relative_to(comparison_dir).as_posix(),'source_hashes':source_hashes,
            'numeric_source_status':'COPIED' if values.is_file() else 'UNAVAILABLE_LEGACY',
            'artifacts':{path.relative_to(stage).as_posix():digest(path) for path in stage.rglob('*') if path.is_file()}}
        atomic_json(stage/'report_manifest.json',record)
        atomic_text(stage/'report_manifest.sha256',digest(stage/'report_manifest.json')+'\n')
        publish_directory(stage,final)
        atomic_json(comparison_dir/'paired_report_publication.json',{'state':'PENDING','generation':record['generation']})
        atomic_copy(final/'paired_analysis_report.pdf',report)
        atomic_copy(final/'report_manifest.json',comparison_dir/'paired_report_manifest.json')
        atomic_copy(final/'report_manifest.sha256',comparison_dir/'paired_report_manifest.sha256')
        atomic_json(comparison_dir/'paired_report_publication.json',{'state':'COMPLETE','generation':record['generation']})
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'type':type(exc).__name__,'message':str(exc),'source_hashes':source_hashes})
        raise
    return report



def verify_paired_analysis_report(directory):
    from core.foundation import file_sha256 as digest
    root=Path(directory).resolve();pointer=json.loads((root/'paired_report_publication.json').read_text())
    if pointer.get('state')!='COMPLETE':raise ValueError('Paired report publication is pending')
    path=root/'paired_report_manifest.json';before=digest(path)
    if (root/'paired_report_manifest.sha256').read_text().strip()!=before:raise ValueError('Paired report manifest checksum mismatch')
    record=json.loads(path.read_text())
    if record.get('schema')!='paired-report-publication-v1':raise ValueError('Unsupported paired report publication schema')
    generation=(root/record['generation']).resolve()
    if pointer['generation']!=record['generation'] or not generation.is_relative_to(root/'paired_reports'):
        raise ValueError('Paired report generation mismatch')
    if digest(generation/'report_manifest.json')!=before:raise ValueError('Paired report archive manifest changed')
    for name,expected in record['artifacts'].items():
        target=(generation/name).resolve()
        if not target.is_relative_to(generation) or digest(target)!=expected:raise ValueError('Paired report asset changed: '+name)
    if digest(root/'paired_analysis_report.pdf')!=record['artifacts']['paired_analysis_report.pdf']:
        raise ValueError('Public paired report changed')
    if digest(path)!=before:raise RuntimeError('Paired report manifest changed while reading')
    return record

def main():
    """Parse command arguments and dispatch this module's documented workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--folder", help="import and analyze a folder of numbered GADGET-4 snapshot HDF5 files")
    p.add_argument("--config", help="optional project YAML to check against the snapshot metadata")
    p.add_argument("--output", help="analysis run directory for --folder; defaults under project.runs_dir")
    p.add_argument(
        "--product", action="append",
        help="run selected products (repeat or use comma-separated names; choices: %s)" % ", ".join(ANALYSIS_PRODUCTS),
    )
    p.add_argument("--compare-with")
    p.add_argument("--overwrite", action="store_true", help="archive and replace an existing analysis.h5")
    p.add_argument('--full-analysis',action='store_true',help='Explicit legacy full analysis; preserves the saved simulation configuration')
    a = p.parse_args()
    if a.compare_with and a.product:
        p.error("--compare-with requires the full saved analysis and cannot be combined with --product")
    if a.full_analysis and a.product:
        p.error('Choose --full-analysis or --product')
    if not a.folder and not a.compare_with and not a.product and not a.full_analysis:
        p.error("select the analysis products explicitly with --product")
    if a.folder:
        if a.run_dir:
            p.error("pass either run_dir or --folder, not both")
        analyze_hdf5_folder(a.folder, config_path=a.config, output_dir=a.output, products=a.product)
        return
    if not a.run_dir:
        p.error("run_dir is required unless --folder is used")
    if a.compare_with:
        for run in (a.run_dir, a.compare_with):
            if a.overwrite or not (Path(run) / "results/analysis.h5").is_file():
                analyze_run(run, overwrite=a.overwrite,full_analysis=True)
        comparison_dir = Path(compare_runs(a.compare_with, a.run_dir))
        from visualization import make_comparison_maps
        map_dir = comparison_dir / "structure_maps"
        make_comparison_maps(a.compare_with, a.run_dir, map_dir)
        create_paired_analysis_report(a.compare_with, a.run_dir, comparison_dir, map_dir)
    else:
        analyze_run(a.run_dir, overwrite=a.overwrite, products=a.product,full_analysis=a.full_analysis)


if __name__ == "__main__":
    main()
