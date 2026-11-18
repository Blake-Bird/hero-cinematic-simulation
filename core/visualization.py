"""Saved-particle maps, chronological movies and portable visual exports.

Field readers apply snapshot/catalogue units and selection contracts before
projection. Map/motion rendering and display normalization preserve source
measurements and explicit missingness. Saved-frame encoders and scene exporters
write their own provenance; none advances the native particle simulation.
"""
from __future__ import annotations

# ========================================================================
# visualization

# ========================================================================
# visualize: 2-D structure maps, movies, and ParaView/Blender handoff.

# Workflow and units
# Converts saved particle/catalogue fields into maps, movies, interactive views and portable Blender exports.
# Map distances are comoving Mpc/h, temperature is Kelvin and declared physical density is g/cm^3. Display normalization never changes stored measurements.
# ID matching preserves identity; periodic copies require a fundamental-box crop. Interpolated poses are display frames, not simulated states. Shared Blender nodes live in blender_scene.py.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from core.native import native_run_owner

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
from core.data import hdf5_scalar_attr as _hdf5_scalar_attr, particle_id_hash64 as _particle_id_hash64, convert_pinned_native
from blender import _available_memory_bytes as _available_ram_bytes
from core.foundation import copy_blender_runtime
from core.foundation import KPC_CM, MSUN_G
from figures import export_figure, semantic_cmap

from analysis import (
    load_cfg, require_stored_temperature_kelvin, _snapshot_parameter, find_hdf5_files, group_files, read_particle_fields,
    read_snapshot_header, simulation_resolution_metadata, snapshot_ids, temperature_from_internal_energy)

_RIDGE_PROJECTION_CACHE = {}
_DISPERSE_SKELETON_CACHE = {}


def _resolve_slice_settings(cfg, box, thickness_mpc_h=None, center_mpc_h=None):
    visualization = cfg["visualization"]
    supplied_by_caller = thickness_mpc_h is not None or center_mpc_h is not None
    if supplied_by_caller and (thickness_mpc_h is None or center_mpc_h is None):
        raise ValueError("provide both slice thickness and slice center, or neither")
    if not supplied_by_caller:
        mode = visualization["slice_mode"]
        if mode == "full_box":
            if visualization["slice_thickness_mpc_h"] is not None or visualization["slice_center_mpc_h"] is not None:
                raise ValueError("full_box slice mode cannot also declare slab dimensions")
            return None, None
        if mode != "slab":
            raise ValueError("visualization.slice_mode must explicitly be full_box or slab")
        thickness_mpc_h = visualization["slice_thickness_mpc_h"]
        center_mpc_h = visualization["slice_center_mpc_h"]
        if thickness_mpc_h is None or center_mpc_h is None:
            raise ValueError("slab mode requires explicit slice thickness and center in Mpc/h")
    thickness = float(thickness_mpc_h)
    if not np.isfinite(thickness) or thickness <= 0 or thickness > float(box):
        raise ValueError("slice thickness must be finite and in (0, box size]")
    if center_mpc_h is None:
        raise ValueError("an explicit slice center is required when a slab thickness is set")
    center = float(center_mpc_h)
    if not np.isfinite(center):
        raise ValueError("slice center must be finite")
    return thickness, float(np.mod(center, float(box)))


def _slice_tag(thickness_mpc_h, center_mpc_h):
    if thickness_mpc_h is None:
        return ""
    def encode(value):
        return ("%.6g" % float(value)).replace("-", "m").replace(".", "p")
    return "_slice_t%s_c%s" % (encode(thickness_mpc_h), encode(center_mpc_h))


def _scalar_float(value, name="value"):
    """Convert scalar or length-one HDF5 attributes consistently across NumPy versions."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("%s must be a scalar or length-one value" % name)
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _project_cic(xy, weights, grid, box):
    """Mass-conserving periodic cloud-in-cell deposition onto pixel centres."""
    xy = np.asarray(xy, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if xy.shape != (len(weights), 2) or grid < 1 or box <= 0:
        raise ValueError("CIC needs aligned positions/weights and positive grid/box")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(weights)):
        raise ValueError("CIC inputs must be finite")
    u = np.mod(xy, box) * grid / box - 0.5
    lower = np.floor(u).astype(np.int64)
    fraction = u - lower
    result = np.zeros(grid * grid, dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            wx = fraction[:, 0] if dx else 1 - fraction[:, 0]
            wy = fraction[:, 1] if dy else 1 - fraction[:, 1]
            ix = (lower[:, 0] + dx) % grid
            iy = (lower[:, 1] + dy) % grid
            result += np.bincount(ix * grid + iy, weights=weights * wx * wy,
                                  minlength=grid * grid)
    return result.reshape(grid, grid)


def _projected_density(run_dir, snap, field, projection,
                       slice_thickness_mpc_h=None, slice_center_mpc_h=None):
    """Project mass surface density, or mass-weighted gas temperature, by streaming shards."""
    field_types = {"dm": (1,), "gas": (0,), "stars": (4,), "total": (0, 1, 4), "gas_temperature": (0,)}
    if field not in field_types:
        raise ValueError("field must be one of: " + ", ".join(field_types))
    if projection not in {"xy", "xz", "yz"}:
        raise ValueError("projection must be one of: xy, xz, yz")

    cfg = load_cfg(run_dir)
    header, files = read_snapshot_header(run_dir, snap)
    box = float(cfg["simulation"]["box_mpc_h"])
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[projection]
    slice_thickness_mpc_h, slice_center_mpc_h = _resolve_slice_settings(
        cfg, box, slice_thickness_mpc_h, slice_center_mpc_h,
    )
    line_of_sight_axis = {"xy": 2, "xz": 1, "yz": 0}[projection]
    grid = int(cfg["analysis"]["density_grid"])
    if grid <= 0:
        raise ValueError("analysis.density_grid must be positive")

    weighted_mass = np.zeros((grid, grid), dtype=np.float64)
    weighted_temperature = np.zeros((grid, grid), dtype=np.float64) if field == "gas_temperature" else None
    temperature_mode = None
    temperature_velocity_unit = None
    star_forming_field_present = None
    scale_factor = None
    mass_unit_msun_h = None
    chunk_size = 1_000_000
    particle_types = field_types[field]
    for path in files:
        with h5py.File(path, "r") as h5:
            shard_header = h5["Header"].attrs
            if "Time" not in shard_header:
                raise ValueError("snapshot Header/Time is required to label map redshift")
            shard_scale = _scalar_float(shard_header["Time"], "snapshot Header/Time")
            if scale_factor is None:
                scale_factor = shard_scale
            elif not np.isclose(scale_factor, shard_scale, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("Snapshot shards have inconsistent scale factors for snapshot %03d" % snap)

            parameters = h5.get("Parameters")
            if parameters is None or "UnitMass_in_g" not in parameters.attrs or "UnitLength_in_cm" not in parameters.attrs:
                raise ValueError("Snapshot Parameters must explicitly declare UnitMass_in_g and UnitLength_in_cm")
            unit_length = float(parameters.attrs["UnitLength_in_cm"])
            if not np.isfinite(unit_length) or not np.isclose(unit_length, KPC_CM, rtol=1.0e-6, atol=0.0):
                raise ValueError("Snapshot UnitLength_in_cm must match the configured kpc/h coordinate unit")
            unit_mass = float(parameters.attrs["UnitMass_in_g"]) / MSUN_G
            if not np.isfinite(unit_mass) or unit_mass <= 0:
                raise ValueError("Snapshot has an invalid UnitMass_in_g conversion")
            if mass_unit_msun_h is None:
                mass_unit_msun_h = unit_mass
            elif not np.isclose(mass_unit_msun_h, unit_mass, rtol=1.0e-10, atol=0.0):
                raise ValueError("Snapshot shards use inconsistent mass units")

            if "MassTable" not in shard_header:
                raise ValueError("snapshot Header/MassTable is required to interpret particle masses")
            mass_table = np.asarray(shard_header["MassTable"], dtype=float).reshape(-1)
            for ptype in particle_types:
                group_name = "PartType%d" % ptype
                if group_name not in h5 or "Coordinates" not in h5[group_name]:
                    continue
                particles = h5[group_name]
                coordinates = particles["Coordinates"]
                count = len(coordinates)
                if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                    raise ValueError("%s/Coordinates must have shape (particles, 3)" % group_name)
                if count == 0:
                    continue
                if "Masses" in particles:
                    masses_dataset = particles["Masses"]
                    if masses_dataset.ndim != 1 or len(masses_dataset) != count:
                        raise ValueError("%s/Masses do not align with Coordinates" % group_name)
                else:
                    if ptype >= len(mass_table) or not np.isfinite(mass_table[ptype]) or mass_table[ptype] <= 0:
                        raise ValueError("%s has no Masses dataset or positive Header/MassTable entry" % group_name)
                    masses_dataset = None

                if field == "gas_temperature":
                    if "Temperature" in particles:
                        mode = "stored"
                        temp_dataset = particles["Temperature"]
                        electron_dataset = None
                        require_stored_temperature_kelvin(files, ptype)
                        sfr_dataset = particles.get("StarFormationRate")
                        if bool(cfg.get("hydro", {}).get("star_formation", False)) and sfr_dataset is None:
                            raise ValueError("Stored gas temperature needs StarFormationRate for effective-EOS exclusion")
                        if sfr_dataset is not None and (sfr_dataset.ndim != 1 or len(sfr_dataset) != count):
                            raise ValueError("StarFormationRate does not align with stored temperature")
                    elif "InternalEnergy" in particles and "ElectronAbundance" in particles:
                        mode = "derived"
                        temp_dataset = particles["InternalEnergy"]
                        if "UnitVelocity_in_cm_per_s" not in (parameters.attrs if parameters is not None else {}):
                            raise ValueError("Gas temperature conversion requires Parameters/UnitVelocity_in_cm_per_s")
                        velocity_unit = float(parameters.attrs["UnitVelocity_in_cm_per_s"])
                        if not np.isfinite(velocity_unit) or velocity_unit <= 0:
                            raise ValueError("Snapshot has an invalid UnitVelocity_in_cm_per_s conversion")
                        if temperature_velocity_unit is None:
                            temperature_velocity_unit = velocity_unit
                        elif not np.isclose(temperature_velocity_unit, velocity_unit, rtol=1.0e-10, atol=0.0):
                            raise ValueError("Snapshot shards use inconsistent UnitVelocity_in_cm_per_s conversions")
                        electron_dataset = particles["ElectronAbundance"]
                        if electron_dataset.ndim != 1 or len(electron_dataset) != count:
                            raise ValueError("PartType0/ElectronAbundance does not align with Coordinates")
                        sfr_dataset = particles.get("StarFormationRate")
                        has_sfr = sfr_dataset is not None
                        if star_forming_field_present is None:
                            star_forming_field_present = has_sfr
                        elif star_forming_field_present != has_sfr:
                            raise ValueError("Snapshot shards have inconsistent StarFormationRate availability")
                        if bool(cfg.get("hydro", {}).get("star_formation", False)) and sfr_dataset is None:
                            raise ValueError("Cannot derive a thermal gas-temperature map without StarFormationRate when star formation is enabled")
                        if sfr_dataset is not None and (sfr_dataset.ndim != 1 or len(sfr_dataset) != count):
                            raise ValueError("PartType0/StarFormationRate does not align with Coordinates")
                    else:
                        raise ValueError(
                            "Gas temperature needs Temperature, or both InternalEnergy and ElectronAbundance"
                        )
                    if temp_dataset.ndim != 1 or len(temp_dataset) != count:
                        raise ValueError("Gas temperature input does not align with Coordinates")
                    if temperature_mode is None:
                        temperature_mode = mode
                    elif temperature_mode != mode:
                        raise ValueError("Gas temperature representation differs between snapshot shards")

                for start in range(0, count, chunk_size):
                    stop = min(start + chunk_size, count)
                    xyz = np.asarray(coordinates[start:stop], dtype=float) / 1000.0
                    mass = (
                        np.asarray(masses_dataset[start:stop], dtype=float)
                        if masses_dataset is not None
                        else np.full(stop - start, mass_table[ptype], dtype=float)
                    ) * unit_mass
                    valid = np.all(np.isfinite(xyz), axis=1) & np.isfinite(mass) & (mass > 0)
                    if slice_thickness_mpc_h is not None:
                        line_of_sight = np.mod(xyz[:, line_of_sight_axis], box)
                        distance = line_of_sight - slice_center_mpc_h
                        distance -= box * np.round(distance / box)
                        valid &= np.abs(distance) <= slice_thickness_mpc_h / 2.0
                    if field == "gas_temperature":
                        input_temperature = np.asarray(temp_dataset[start:stop], dtype=float)
                        if mode == "derived":
                            electron = np.asarray(electron_dataset[start:stop], dtype=float)
                            input_temperature = temperature_from_internal_energy(
                                input_temperature, electron,
                                unit_velocity_cm_per_s=velocity_unit,
                            )
                            if input_temperature is None:
                                raise ValueError("Gas temperature conversion lacks the required internal-energy units or ionization data")
                        if sfr_dataset is not None:
                            sfr_values = np.asarray(sfr_dataset[start:stop], dtype=float)
                            valid &= np.isfinite(sfr_values) & (sfr_values == 0.0)
                        valid &= np.isfinite(input_temperature) & (input_temperature > 0)
                    if not np.any(valid):
                        continue
                    projected = xyz[valid][:, axes]
                    weighted_mass += _project_cic(projected, mass[valid], grid, box)
                    if field == "gas_temperature":
                        weighted_temperature += _project_cic(
                            projected, mass[valid] * input_temperature[valid], grid, box)

    if mass_unit_msun_h is None or not np.any(weighted_mass > 0):
        return None
    pixel_area = (box / grid) ** 2
    if field == "gas_temperature":
        values = np.divide(
            weighted_temperature, weighted_mass,
            out=np.full_like(weighted_temperature, np.nan), where=weighted_mass > 0,
        )
    else:
        values = weighted_mass / pixel_area
    img = np.full(values.T.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(values.T) & (values.T > 0)
    img[positive] = np.log10(values.T[positive])
    if "Time" not in header:
        raise ValueError("snapshot Header/Time is required to label map redshift")
    a = _scalar_float(header["Time"], "snapshot Header/Time")
    slice_info = {
        "thickness_mpc_h": slice_thickness_mpc_h,
        "center_mpc_h": slice_center_mpc_h,
        "line_of_sight_axis": line_of_sight_axis,
        "assignment_kernel": "periodic cloud-in-cell on pixel centres; no added smoothing",
        "deposited_mass_msun_h": float(weighted_mass.sum()),
        "empty_pixel_count": int(np.count_nonzero(weighted_mass == 0)),
    }
    z = 1.0 / a - 1.0
    if abs(z) < 1e-12:
        z = 0.0
    return img, box, z, axes, temperature_mode, slice_info, values.T


def _halo_overlay(run_dir, snap, axes, max_halos=250, line_of_sight_axis=None,
                  slice_center_mpc_h=None, slice_thickness_mpc_h=None):
    """Return the most massive GADGET FOF group centers for a clean overlay."""
    from analysis import group_field, units_to_mpc_h, mass_to_msun_h
    cfg = load_cfg(run_dir)
    pos = group_field(run_dir, snap, ["GroupPos"])
    mass = group_field(run_dir, snap, ["GroupMass", "Group_M_Crit200"])
    if pos is None or mass is None:
        return None, None
    pos = units_to_mpc_h(pos, float(cfg["simulation"]["box_mpc_h"]))
    mass = mass_to_msun_h(mass)
    good = np.isfinite(mass) & (mass > 0)
    if slice_thickness_mpc_h is not None and line_of_sight_axis is not None:
        los = np.mod(pos[:, line_of_sight_axis], float(cfg["simulation"]["box_mpc_h"]))
        distance = los - float(slice_center_mpc_h)
        distance -= float(cfg["simulation"]["box_mpc_h"]) * np.round(
            distance / float(cfg["simulation"]["box_mpc_h"]),
        )
        good &= np.abs(distance) <= float(slice_thickness_mpc_h) / 2.0
    pos, mass = pos[good], mass[good]
    if len(mass) > max_halos:
        idx = np.argpartition(mass, -max_halos)[-max_halos:]
        pos, mass = pos[idx], mass[idx]
    return pos[:, list(axes)], mass


def _periodic_density_ridge_map(run_dir, snap, projection_axes, grid_size=None,
                                slice_center_mpc_h=None, slice_thickness_mpc_h=None):
    """Return a 2-D projection of Hessian-classified periodic density ridges.

    Two negative Hessian eigenvalues identify filament-like ridges in the
    smoothed log-density field; this is a documented ridge proxy, not a
    persistence-filtered DisPerSE skeleton.
    """
    from scipy.ndimage import gaussian_filter
    cfg = load_cfg(run_dir)
    if grid_size is None:
        grid_size = int(cfg["visualization"]["filament_grid_size"])
    box = float(cfg["simulation"]["box_mpc_h"])
    cache_key = (
        str(Path(run_dir).resolve()), int(snap), tuple(projection_axes), int(grid_size),
        None if slice_center_mpc_h is None else float(slice_center_mpc_h),
        None if slice_thickness_mpc_h is None else float(slice_thickness_mpc_h),
    )
    if cache_key in _RIDGE_PROJECTION_CACHE:
        return _RIDGE_PROJECTION_CACHE[cache_key]
    files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%03d" % int(snap))
    density_ratio, density_metadata = _particle_mass_density_grid(
        files, 1, box, grid_size, return_density_ratio=True,
    )
    if int(density_metadata["particle_count"]) < 64:
        return None
    n = int(grid_size)
    if not np.all(np.isfinite(density_ratio)) or np.any(density_ratio < 0):
        return None
    log_density = np.log1p(density_ratio)
    smoothing_mpc_h = float(cfg["visualization"]["filament_smoothing_mpc_h"])
    sigma = smoothing_mpc_h / (box / n)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Hessian filament extraction requires positive physical smoothing in Mpc/h")
    hessian = np.empty((n, n, n, 3, 3), dtype=np.float32)
    for i in range(3):
        order = [0, 0, 0]
        order[i] = 2
        hessian[..., i, i] = gaussian_filter(log_density, sigma=sigma, order=order, mode="wrap")
        for j in range(i + 1, 3):
            order = [0, 0, 0]
            order[i] = order[j] = 1
            derivative = gaussian_filter(log_density, sigma=sigma, order=order, mode="wrap")
            hessian[..., i, j] = derivative
            hessian[..., j, i] = derivative
    eigenvalues = np.linalg.eigvalsh(hessian)
    finite = np.all(np.isfinite(eigenvalues), axis=-1)
    curvature = np.abs(eigenvalues[finite])
    if not curvature.size:
        return None
    threshold_factor = float(cfg["visualization"]["filament_curvature_threshold_factor"])
    threshold = max(float(np.percentile(curvature, 65.0)) * threshold_factor, np.finfo(float).eps)
    # eigvalsh is ascending: the middle eigenvalue marks collapse in two axes,
    # while the largest remains below the collapse threshold along the spine.
    ridge = finite & (eigenvalues[..., 1] < -threshold) & (eigenvalues[..., 2] >= -threshold)
    positive_density = density_ratio[density_ratio > 0]
    if not len(positive_density):
        return None
    density_percentile = float(cfg["visualization"]["filament_density_percentile"])
    ridge &= density_ratio >= np.percentile(positive_density, density_percentile)
    los_axis = ({0, 1, 2} - set(projection_axes)).pop()
    coordinates_1d = (np.arange(n, dtype=float) + 0.5) * box / n
    if slice_thickness_mpc_h is not None:
        distance = coordinates_1d - float(slice_center_mpc_h)
        distance -= box * np.round(distance / box)
        in_slab = np.abs(distance) <= float(slice_thickness_mpc_h) / 2.0
        reshape = [1, 1, 1]
        reshape[los_axis] = n
        ridge &= in_slab.reshape(reshape)
    projected = np.any(ridge, axis=los_axis)
    image = projected
    result = (coordinates_1d, image, {
        "method": "periodic Gaussian-smoothed log-density Hessian; two-axis collapse ridge proxy",
        "grid_size": n, "grid_cell_size_mpc_h": box / n,
        "smoothing_sigma_cells": sigma, "smoothing_sigma_mpc_h": smoothing_mpc_h,
        "curvature_threshold": threshold, "curvature_threshold_factor": threshold_factor,
        "density_percentile_of_occupied_cells": density_percentile,
        "ridge_cell_count_3d": int(np.count_nonzero(ridge)),
        "line_of_sight_axis": los_axis,
        "density_deposition": density_metadata["method"],
        "density_particle_count": density_metadata["particle_count"],
    })
    if len(_RIDGE_PROJECTION_CACHE) >= 600:
        _RIDGE_PROJECTION_CACHE.pop(next(iter(_RIDGE_PROJECTION_CACHE)))
    _RIDGE_PROJECTION_CACHE[cache_key] = result
    return result


def _write_disperse_fits(path, values, box_size_mpc_h):
    """Write a minimal standards-compliant FITS image without an Astropy dependency."""
    values = np.asarray(values, dtype=">f4")
    if values.ndim != 3 or len(set(values.shape)) != 1 or not np.all(np.isfinite(values)):
        raise ValueError("DisPerSE input must be a finite cubic 3-D scalar grid")
    grid = int(values.shape[0])
    cell = float(box_size_mpc_h) / grid
    cards = [
        "SIMPLE  =                    T / conforms to FITS standard",
        "BITPIX  =                  -32 / 32-bit floating point",
        "NAXIS   =                    3 / number of data axes",
        "NAXIS1  = %20d / x cells" % grid,
        "NAXIS2  = %20d / y cells" % grid,
        "NAXIS3  = %20d / z cells" % grid,
        "BSCALE  = %20d" % 1,
        "BZERO   = %20d" % 0,
        "CTYPE1  = 'LINEAR  '",
        "CTYPE2  = 'LINEAR  '",
        "CTYPE3  = 'LINEAR  '",
        "CUNIT1  = 'Mpc/h   '",
        "CUNIT2  = 'Mpc/h   '",
        "CUNIT3  = 'Mpc/h   '",
        "CRPIX1  = %20.12E" % 1.0,
        "CRPIX2  = %20.12E" % 1.0,
        "CRPIX3  = %20.12E" % 1.0,
        "CRVAL1  = %20.12E" % (cell / 2.0),
        "CRVAL2  = %20.12E" % (cell / 2.0),
        "CRVAL3  = %20.12E" % (cell / 2.0),
        "CDELT1  = %20.12E" % cell,
        "CDELT2  = %20.12E" % cell,
        "CDELT3  = %20.12E" % cell,
        "BUNIT   = 'log10(1+rho/rho_mean)'",
        "COMMENT   Periodic smoothed DM field; coordinates are comoving Mpc/h.",
        "END",
    ]
    header = "".join(card[:80].ljust(80) for card in cards).encode("ascii")
    header += b" " * ((-len(header)) % 2880)
    data = values.tobytes(order="C")
    data += b"\0" * ((-len(data)) % 2880)
    with open(path, "wb") as output:
        output.write(header)
        output.write(data)


def _disperse_executable(visualization, option, default_name):
    requested = visualization.get(option)
    executable = str(requested) if requested else shutil.which(default_name)
    if not executable:
        from core.native import native_engine_directory
        candidate = native_engine_directory('disperse')/'build/src/mse'/default_name
        if candidate.is_file():
            executable = str(candidate)
    if not executable or not (Path(executable).expanduser().is_file() or shutil.which(executable)):
        raise FileNotFoundError(
            "filament_method='disperse_grid' requires DisPerSE %s; install it or set visualization.%s"
            % (default_name, option)
        )
    return str(Path(executable).expanduser().resolve()) if Path(executable).expanduser().is_file() else str(Path(shutil.which(executable)).resolve())


@native_run_owner
def _load_disperse_skeleton(run_dir, snap, grid_size, persistence_cut):
    """Publish a source-bound skeleton; cache only verified completed bundles."""
    import uuid
    from core.data import coordinate_contract,run_epoch_contract,write_dataset_contract
    from core.foundation import product_provenance, verify_product_provenance, publish_directory, atomic_json, file_sha256 as digest, canonical_digest
    root=Path(run_dir).expanduser().resolve();cfg=load_cfg(root);visualization=cfg['visualization']
    representation=visualization.get('disperse_output_coordinate_representation')
    if representation not in ('comoving_mpc_h','grid_vertex_coordinates'):
        raise ValueError('DisPerSE output coordinates must explicitly declare comoving_mpc_h or grid_vertex_coordinates')
    files=find_hdf5_files(root/'data','snapshot_%03d'%int(snap))
    mse=_disperse_executable(visualization,'disperse_executable','mse')
    converter=_disperse_executable(visualization,'skelconv_executable','skelconv')
    contract=product_provenance(root,algorithm=__file__,source_files=files+[Path(mse),Path(converter)],snapshot=int(snap),
        helper_sources=[Path(__file__).with_name('analysis.py')],
        selection={'grid_size':int(grid_size),'persistence_cut':float(persistence_cut),
            'output_coordinate_representation':representation,'visualization':visualization},
        coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
            periodic_wrapping='native skeleton points wrapped to fundamental box; segments use minimum image',velocity_convention='not applicable to persistence skeleton'))
    key=canonical_digest(contract)
    if key in _DISPERSE_SKELETON_CACHE:
        cached=_DISPERSE_SKELETON_CACHE[key]
        manifest=Path(cached['metadata']['bundle_directory'])/'filament_contract.json'
        if digest(manifest)!=cached['metadata']['contract_sha256']:raise ValueError('Cached filament manifest changed')
        saved=json.loads(manifest.read_text())
        for item in saved['assets']:
            if digest(manifest.parent/item['path'])!=item['sha256']:raise ValueError('Cached filament asset changed')
        return cached
    parent=root/'results/filaments';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.filaments-',dir=parent));destination=parent/('snapshot_%03d_'%int(snap)+uuid.uuid4().hex)
    try:
        result=_compute_disperse_skeleton(root,snap,grid_size,persistence_cut,output_directory=stage)
        header,_=read_snapshot_header(root,snap)
        epoch=run_epoch_contract(root,int(snap),_hdf5_scalar_attr(header['Time'],'Time'))
        with h5py.File(stage/'skeleton.hdf5','w') as numeric:
            for name,values,units in [('points_comoving_mpc_h',result['points_mpc_h'],'comoving Mpc/h'),
                                      ('vtk_polyline_connectivity',result['lines'],'VTK connectivity: count followed by epoch-local point rows')]:
                dataset=numeric.create_dataset(name,data=values,chunks=True,fletcher32=True)
                write_dataset_contract(dataset,units=units,epoch=epoch,coordinates=contract['coordinates'],
                    quantity='DisPerSE '+name,provenance={'manifest':'filament_contract.json'},missing_policy='no missing points or indices')
            graph = numeric.create_group('graph')
            for name, values in result.get('graph_arrays', {}).items():
                units = ('comoving Mpc/h' if name.endswith('mpc_h') else 'h/Mpc' if name.endswith('h_mpc') else
                         'identity' if name in {'node_ids', 'endpoint_node_ids'} else 'native scalar fields' if name.startswith('native_') else 'dimensionless')
                dataset = graph.create_dataset(name, data=values)
                write_dataset_contract(dataset, units=units, epoch=epoch, coordinates=contract['coordinates'],
                    identity_namespace=(contract['identity_namespaces']['native_particle']+':filaments:'+str(snap)) if units == 'identity' else None,
                    quantity='Unsmoothed native filament '+name, provenance={'manifest': 'filament_contract.json'},
                    missing_policy='NaN for endpoint or degenerate curvature/tangents')
        from core.data import numpy_archive_contract
        npz_path=stage/result['metadata']['point_connectivity_file']
        npz_contract=numpy_archive_contract(npz_path,{
            'points_mpc_h':{'units':'comoving Mpc/h','quantity':'DisPerSE wrapped skeleton point positions'},
            'lines':{'units':'VTK connectivity','quantity':'Polyline count followed by epoch-local point row indices'},
            'metadata_json':{'units':'UTF-8 JSON','quantity':'Saved native/display skeleton metadata'}},
            epoch=epoch,coordinates=contract['coordinates'],provenance=contract)
        verify_product_provenance(root,contract)
        result['metadata'].update(epoch_contract=epoch,science_acceptance='NOT_ACCEPTED',
            bundle_directory=str(destination),contract_manifest='filament_contract.json')
        atomic_json(stage/'filament_contract.json',{'schema':'filament-publication-v1','science_acceptance':'NOT_ACCEPTED',
            'validation_state':'VALIDATION_PENDING','provenance':contract,'epoch':epoch,'metadata':result['metadata'],
            'numpy_contract':npz_contract,
            'assets':[{'path':path.relative_to(stage).as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size}
                for path in sorted(stage.rglob('*')) if path.is_file()]})
        publish_directory(stage,destination)
        result['metadata']['contract_sha256']=digest(destination/'filament_contract.json')
        if len(_DISPERSE_SKELETON_CACHE)>=100:_DISPERSE_SKELETON_CACHE.pop(next(iter(_DISPERSE_SKELETON_CACHE)))
        _DISPERSE_SKELETON_CACHE[key]=result
        return result
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


def read_disperse_ascii(path, *, memory_bytes=128*1024**2, maximum_nodes=1000000,
                        maximum_arcs=2000000, maximum_points=4000000):
    """Read native ANDSKEL topology, including critical data and persistence fields.

    Format follows pinned DisPerSE Save_ASCIIskel in src/C/NDskeleton.c.
    Node/arc IDs are native row IDs, never rounded-coordinate identities.
    """
    path = Path(path)
    if type(memory_bytes) is not int or memory_bytes <= 0 or path.stat().st_size*8 > memory_bytes:
        raise MemoryError('DisPerSE ASCII intake exceeds declared workspace')
    with path.open(encoding='ascii') as source:
        def line():
            while True:
                value = source.readline(65537)
                if not value or len(value) > 65536:
                    raise ValueError('Truncated or oversized DisPerSE record')
                value = value.strip()
                if value and not value.startswith('#'):
                    return value
        def count(limit):
            value = int(line())
            if not 0 <= value <= limit:
                raise ValueError('DisPerSE count exceeds its declared bound')
            return value
        if line() != 'ANDSKEL' or line() != '3':
            raise ValueError('Only native three-dimensional ANDSKEL is supported')
        box = line()
        match = re.fullmatch(r'BBOX \[([^]]+)\] \[([^]]+)\]', box)
        if match is None:
            raise ValueError('Native skeleton BBOX missing')
        origin = np.asarray([float(x) for x in match[1].split(',')])
        extent = np.asarray([float(x) for x in match[2].split(',')])
        if origin.shape != (3,) or extent.shape != (3,) or not np.isfinite([origin, extent]).all() or np.any(extent <= 0):
            raise ValueError('Invalid native skeleton bounding box')
        if line() != '[CRITICAL POINTS]':
            raise ValueError('Critical-point table missing')
        n = count(maximum_nodes)
        positions, types, values, pairs, boundaries, adjacency = [], [], [], [], [], []
        for _ in range(n):
            fields = line().split()
            if len(fields) != 7:
                raise ValueError('Invalid critical-point fields')
            types.append(int(fields[0])); positions.append([float(x) for x in fields[1:4]])
            values.append(float(fields[4])); pairs.append(int(fields[5])); boundaries.append(int(fields[6]))
            links = []
            for _ in range(count(maximum_arcs*2)):
                row = [int(x) for x in line().split()]
                if len(row) != 2:
                    raise ValueError('Invalid native adjacency row')
                links.append(row)
            adjacency.append(links)
        if line() != '[FILAMENTS]':
            raise ValueError('Filament table missing')
        m = count(maximum_arcs)
        endpoints, paths, points = [], [], []
        for _ in range(m):
            fields = [int(x) for x in line().split()]
            if len(fields) != 3 or not 0 <= fields[0] < n or not 0 <= fields[1] < n or fields[2] < 2:
                raise ValueError('Invalid native filament endpoints/count')
            if len(points)+fields[2] > maximum_points:
                raise MemoryError('Native filament points exceed allocation')
            start = len(points)
            for _ in range(fields[2]):
                row = [float(x) for x in line().split()]
                if len(row) != 3 or not np.isfinite(row).all():
                    raise ValueError('Invalid filament coordinates')
                points.append(row)
            endpoints.append(fields[:2]); paths.append(np.arange(start, len(points), dtype=np.int64))
        def data_table(tag, rows):
            if line() != tag:
                raise ValueError('Native skeleton data table missing: '+tag)
            columns = count(1024)
            names = [line() for _ in range(columns)]
            if len(set(names)) != columns:
                raise ValueError('Duplicate native skeleton data field')
            if not columns:
                return names, np.empty((rows, 0))
            rows_data = [[float(x) for x in line().split()] for _ in range(rows)]
            if any(len(row) != columns for row in rows_data):
                raise ValueError('Native skeleton data dimensions disagree')
            data = np.asarray(rows_data, dtype=float).reshape(rows, columns)
            if data.shape != (rows, columns):
                raise ValueError('Native skeleton data dimensions disagree')
            return names, data
        node_names, node_data = data_table('[CRITICAL POINTS DATA]', n)
        point_names, point_data = data_table('[FILAMENTS DATA]', len(points))
        if any(row.strip() and not row.lstrip().startswith('#') for row in source):
            raise ValueError('Unexpected trailing skeleton records')
    positions = np.asarray(positions).reshape(-1, 3)
    if not np.isfinite(positions).all() or not np.isfinite(values).all() or any(not 0 <= value <= 3 for value in types):
        raise ValueError('Invalid native critical geometry/type')
    if any(value >= n for value in pairs):
        raise ValueError('Persistence pair refers to an absent native node')
    for node, links in enumerate(adjacency):
        for other, arc in links:
            if not 0 <= other < n or not 0 <= arc < m or sorted(endpoints[arc]) != sorted([node, other]):
                raise ValueError('Native adjacency and arc endpoints disagree')
    expected = [[] for _ in range(n)]
    for arc, (left, right) in enumerate(endpoints):
        expected[left].append([right, arc]); expected[right].append([left, arc])
    if any(sorted(a) != sorted(b) for a, b in zip(adjacency, expected)):
        raise ValueError('Native arc is absent or duplicated in critical adjacency')
    return {'node_ids': np.arange(n, dtype=np.uint64), 'node_positions': positions,
        'node_types': np.asarray(types, dtype=np.int32), 'node_values': np.asarray(values),
        'persistence_pair_native_ids': np.asarray(pairs, dtype=np.int64),
        'node_boundary_flags': np.asarray(boundaries, dtype=np.int32),
        'endpoint_node_ids': np.asarray(endpoints, dtype=np.uint64).reshape(-1, 2),
        'points': np.asarray(points).reshape(-1, 3), 'arcs': paths,
        'node_data_names': node_names, 'node_data': node_data,
        'point_data_names': point_names, 'point_data': point_data,
        'bbox_origin': origin, 'bbox_extent': extent}


def _compute_disperse_skeleton(run_dir, snap, grid_size, persistence_cut, *, output_directory):
    """Compute or retrieve a periodic DisPerSE skeleton from a gridded DM field."""
    import pyvista as pv

    cfg = load_cfg(run_dir)
    visualization = cfg.get("visualization", {})
    box = float(cfg["simulation"]["box_mpc_h"])
    density_smoothing = float(visualization["disperse_density_smoothing_cells"])
    cache_key = (
        str(Path(run_dir).resolve()), int(snap), int(grid_size), float(persistence_cut),
        density_smoothing,
        str(visualization.get("disperse_executable") or "mse"),
        str(visualization.get("skelconv_executable") or "skelconv"),
        int(visualization["disperse_smoothing_cells"]),
    )
    files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%03d" % int(snap))
    density_ratio, density_metadata = _particle_mass_density_grid(
        files, 1, box, grid_size, return_density_ratio=True,
    )
    scientific_density_ratio = density_ratio
    if int(density_metadata["particle_count"]) < 64:
        raise ValueError("DisPerSE needs at least 64 dark-matter particles")
    n = int(grid_size)
    if not np.all(np.isfinite(density_ratio)) or np.any(density_ratio < 0):
        raise ValueError("DisPerSE density grid contains invalid mass density")
    if density_smoothing > 0:
        from scipy.ndimage import gaussian_filter
        density_ratio = gaussian_filter(density_ratio, sigma=density_smoothing, mode="wrap")
    mean_density = float(np.mean(density_ratio))
    if not np.isfinite(mean_density) or mean_density <= 0:
        raise ValueError("DisPerSE density grid has no positive mean density")
    # The mass grid is already normalized to its mean density. Normalize once
    # more after smoothing to preserve the intended log10(1 + rho/rho_mean)
    # scalar when the smoothing kernel is periodic.
    field = np.log10(1.0 + density_ratio / mean_density).astype(np.float32)
    mse = _disperse_executable(visualization, "disperse_executable", "mse")
    skelconv = _disperse_executable(visualization, "skelconv_executable", "skelconv")
    smoothing = int(visualization["disperse_smoothing_cells"])
    timeout = int(visualization["disperse_timeout_s"])
    if not np.isclose(float(persistence_cut),float(visualization['disperse_persistence_cut']),rtol=0.,atol=0.):
        raise ValueError('DisPerSE requested persistence cut disagrees with saved configuration')
    persistence_cut = float(persistence_cut)
    if not np.isfinite(density_smoothing) or density_smoothing < 0:
        raise ValueError("DisPerSE grid smoothing must be finite and non-negative")
    if not np.isfinite(persistence_cut) or persistence_cut < 0:
        raise ValueError("DisPerSE persistence cut must be finite and non-negative")
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_ndskl_path = output_dir / ("snapshot_%03d_disperse_raw.NDskl" % int(snap))
    density_field_path = output_dir / ("snapshot_%03d_disperse_log_density.fits" % int(snap))
    from contextlib import nullcontext
    from core.native import native_run_lock_fds
    from core.foundation import atomic_json
    work=output_dir/'native_work';work.mkdir()
    with nullcontext(work) as temporary_name:
        temporary = Path(temporary_name)
        field_path = temporary / "dm_log_overdensity.fits"
        _write_disperse_fits(field_path, field, box)
        shutil.copy2(field_path, density_field_path)
        compute = subprocess.run(
            [mse, str(field_path), "-outName", "dm_skeleton", "-outDir", str(temporary),
             "-periodicity", "111", "-cut", "%.8g" % float(persistence_cut),
             "-upSkl", "-forceLoops", "-no_saveMSC"],
            cwd=str(temporary), capture_output=True, text=True, timeout=timeout,pass_fds=native_run_lock_fds(run_dir),
        )
        atomic_json(output_dir/'mse_receipt.json',{'returncode':compute.returncode,'stdout':compute.stdout,'stderr':compute.stderr})
        skeletons = sorted(temporary.glob("*.NDskl"))
        if compute.returncode != 0 or not skeletons:
            raise RuntimeError(
                "DisPerSE mse failed to produce a skeleton (exit %d): %s"
                % (compute.returncode, (compute.stderr or compute.stdout)[-1800:])
            )
        shutil.copy2(skeletons[0], raw_ndskl_path)
        topology_conversion = subprocess.run(
            [skelconv, str(skeletons[0]), '-outName', 'dm_topology', '-outDir', str(temporary), '-to', 'NDskl_ascii'],
            cwd=str(temporary), capture_output=True, text=True, timeout=timeout,
            pass_fds=native_run_lock_fds(run_dir))
        atomic_json(output_dir/'topology_conversion_receipt.json', {'returncode': topology_conversion.returncode,
            'stdout': topology_conversion.stdout, 'stderr': topology_conversion.stderr})
        topology_files = sorted(temporary.glob('dm_topology*.a.NDskl'))
        if topology_conversion.returncode != 0 or len(topology_files) != 1:
            raise RuntimeError('Native unsmoothed skeleton topology conversion failed')
        topology = read_disperse_ascii(topology_files[0],
            memory_bytes=int(float(visualization.get('disperse_memory_gib', .125))*1024**3))
        topology_path = output_dir/('snapshot_%03d_disperse_topology.a.NDskl' % int(snap))
        shutil.copy2(topology_files[0], topology_path)
        converted = subprocess.run(
            [skelconv, str(skeletons[0]), "-outName", "dm_skeleton_lines", "-outDir", str(temporary),
             "-smooth", str(smoothing), "-breakdown", "-to", "vtp"],
            cwd=str(temporary), capture_output=True, text=True, timeout=timeout,pass_fds=native_run_lock_fds(run_dir),
        )
        atomic_json(output_dir/'skelconv_receipt.json',{'returncode':converted.returncode,'stdout':converted.stdout,'stderr':converted.stderr})
        vtp_files = sorted(temporary.glob("*.vtp"))
        if converted.returncode != 0 or not vtp_files:
            raise RuntimeError(
                "DisPerSE skelconv failed to produce VTP geometry (exit %d): %s"
                % (converted.returncode, (converted.stderr or converted.stdout)[-1800:])
            )
        mesh = pv.read(str(vtp_files[0]))
        points = np.asarray(mesh.points, dtype=float)
        lines = np.asarray(mesh.lines, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not len(lines):
        raise ValueError("DisPerSE VTP output contains no 3-D filament lines")
    finite_rows = np.all(np.isfinite(points), axis=1)
    if not np.all(finite_rows):
        raise ValueError("DisPerSE VTP output contains non-finite coordinates")
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    extent = upper - lower
    representation=visualization.get('disperse_output_coordinate_representation')
    if representation not in ('comoving_mpc_h','grid_vertex_coordinates'):
        raise ValueError('DisPerSE output coordinate representation is undeclared')
    coordinate_scale = box/float(n) if representation=='grid_vertex_coordinates' else 1.
    points = np.mod(points * coordinate_scale, box)
    from analysis import measure_filament_graph
    raw_points = np.mod(topology['points']*coordinate_scale, box)
    images = []
    for arc in topology['arcs']:
        steps = np.diff(raw_points[arc], axis=0)
        wraps = -np.rint(steps/box).astype(np.int64)
        images.append(np.vstack((np.zeros((1, 3), dtype=np.int64), np.cumsum(wraps, axis=0))))
    geometry = measure_filament_graph(raw_points, topology['arcs'], topology['node_ids'],
        topology['endpoint_node_ids'], box_mpc_h=box, arc_image_offsets=images)
    graph_arrays = {'node_ids': topology['node_ids'], 'node_types': topology['node_types'],
        'node_positions_comoving_mpc_h': np.mod(topology['node_positions']*coordinate_scale, box),
        'node_degree': geometry['node_degree'], 'endpoint_node_ids': topology['endpoint_node_ids'],
        'node_field_values': topology['node_values'],
        'persistence_pair_native_ids': topology['persistence_pair_native_ids'],
        'node_boundary_flags': topology['node_boundary_flags'],
        'arc_length_comoving_mpc_h': geometry['arc_length_mpc_h'],
        'arc_point_offsets': np.cumsum([0]+[len(arc) for arc in topology['arcs']], dtype=np.int64),
        'arc_points_unwrapped_comoving_mpc_h': np.concatenate(geometry['arc_paths_mpc_h']),
        'arc_image_offsets': np.concatenate(images), 'arc_tangents': np.concatenate(geometry['arc_tangents']),
        'arc_curvature_h_mpc': np.concatenate(geometry['arc_curvature_h_mpc']),
        'native_node_data': topology['node_data'], 'native_point_data': topology['point_data']}
    transverse = visualization.get('filament_transverse_profiles')
    transverse_metadata = None
    if transverse is not None:
        from analysis import filament_transverse_profiles
        if not isinstance(transverse, dict):
            raise ValueError('Filament transverse profiles require explicit algorithm settings')
        stride = transverse['sample_stride']
        if type(stride) is not int or stride < 1:
            raise ValueError('Positive integer topology-point sampling stride required')
        if not np.array_equal(np.asarray(transverse['grid_origin_mpc_h']), np.zeros(3)):
            raise ValueError('This CIC producer samples grid vertices at origin [0,0,0]')
        rows = np.arange(0, len(raw_points), stride, dtype=np.int64)
        centers = np.mod(graph_arrays['arc_points_unwrapped_comoving_mpc_h'][rows], box)
        node_positions = graph_arrays['node_positions_comoving_mpc_h']
        clearance = float(transverse['node_clearance_mpc_h'])
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError('Explicit finite nonnegative node exclusion radius required')
        # Evaluate one center at a time to avoid an unbounded centers x nodes allocation.
        excluded = np.zeros(len(rows), dtype=bool)
        for i, center in enumerate(centers):
            delta = node_positions-center
            delta -= box*np.rint(delta/box)
            excluded[i] = np.any(np.linalg.norm(delta, axis=1) <= clearance)
        profiles = filament_transverse_profiles(scientific_density_ratio, centers,
            graph_arrays['arc_tangents'][rows], transverse['radii_mpc_h'], box_mpc_h=box,
            grid_origin_mpc_h=transverse['grid_origin_mpc_h'],
            background_radius_range_mpc_h=transverse['background_radius_range_mpc_h'],
            angular_samples=transverse['angular_samples'], excluded_centers=excluded,
            monotonic_tolerance=transverse['monotonic_tolerance_density_ratio'],
            memory_bytes=int(float(transverse['memory_gib'])*1024**3))
        graph_arrays['transverse_topology_point_rows'] = rows
        for name, value in profiles.items():
            if isinstance(value, np.ndarray):
                graph_arrays['transverse_'+name] = value.astype('S32') if value.dtype.kind == 'U' else value
        transverse_metadata = {'settings': transverse, 'definition': profiles['definition'],
            'background_definition': profiles['background_definition'],
            'density_definition': 'unsmoothed full-particle CIC rho/rho_mean',
            'support_definition': 'native tangent, node clearance, profile monotonicity and resolved crossing',
            'science_acceptance': 'NOT_ACCEPTED'}
    polyline_count = 0
    line_cursor = 0
    while line_cursor < len(lines):
        point_count = int(lines[line_cursor])
        if point_count<2 or line_cursor+1+point_count>len(lines):
            raise ValueError('DisPerSE VTK connectivity has an incomplete or unsupported polyline')
        indices=lines[line_cursor+1:line_cursor+1+point_count]
        if np.any(indices<0) or np.any(indices>=len(points)):
            raise ValueError('DisPerSE VTK connectivity references an absent point row')
        line_cursor += point_count + 1
        polyline_count += int(point_count >= 2)
    polyline_path = output_dir / ("snapshot_%03d_disperse_skeleton.vtp" % int(snap))
    point_table_path = output_dir / ("snapshot_%03d_disperse_skeleton.npz" % int(snap))
    segment_table_path = output_dir / ("snapshot_%03d_disperse_segments.csv" % int(snap))
    metadata_path = output_dir / ("snapshot_%03d_disperse_metadata.json" % int(snap))
    pv.PolyData(points, lines=lines).save(str(polyline_path))
    metadata = {
        "method": "DisPerSE persistent topological skeleton on periodic log10(1+rho/rho_mean) grid",
        "persistence_cut_field_units": persistence_cut,
        "input_field_units": "dimensionless log10(1+rho/rho_mean)",
        "grid_size": n,
        "box_size_mpc_h": box,
        "grid_cell_size_mpc_h": box / n,
        "density_smoothing_sigma_cells": density_smoothing,
        "density_smoothing_sigma_mpc_h": density_smoothing * box / n,
        "smoothing_iterations_for_display_geometry": smoothing,
        "periodicity": "xyz",
        "coordinate_conversion": representation,"coordinate_scale_to_comoving_mpc_h":coordinate_scale,
        "input_field": "full-particle CIC mass-weighted dark-matter log10(1+rho/rho_mean)",
        "density_particle_count": density_metadata["particle_count"],
        "density_deposition": density_metadata["method"],
        "mse_executable": mse,
        "skelconv_executable": skelconv,
        "filament_polyline_count": polyline_count,
        "filament_definition": "ascending 1-dimensional persistence skeleton of the gridded DM field",
        "raw_ndskl_file": raw_ndskl_path.name,
        'raw_topology_file': topology_path.name,
        'graph_definition': 'native critical-point/arc IDs; unsmoothed centerlines; incident-arc degree',
        'periodic_graph_images': 'minimum image on resolved native segments, strict half-box support',
        'native_node_data_fields': topology['node_data_names'], 'native_point_data_fields': topology['point_data_names'],
        'transverse_profiles': transverse_metadata,
        "density_field_file": density_field_path.name,
        "polydata_file": polyline_path.name,
        "point_connectivity_file": point_table_path.name,
        "segment_table_file": segment_table_path.name,
        "snapshot": int(snap),
        "redshift": float(_snapshot_redshift(run_dir, snap)[0]),
    }
    np.savez_compressed(
        point_table_path,
        points_mpc_h=points,
        lines=lines,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with segment_table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "polyline_id", "segment_index", "x0_mpc_h", "y0_mpc_h", "z0_mpc_h",
            "x1_mpc_h", "y1_mpc_h", "z1_mpc_h", "length_mpc_h",
        ])
        cursor = 0
        polyline_id = 0
        while cursor < len(lines):
            count = int(lines[cursor])
            indices = lines[cursor + 1:cursor + 1 + count]
            cursor += count + 1
            for segment_index, (start_index, end_index) in enumerate(zip(indices[:-1], indices[1:])):
                start_point = points[int(start_index)]
                end_point = points[int(end_index)]
                delta = end_point - start_point
                delta -= box * np.round(delta / box)
                writer.writerow([
                    polyline_id, segment_index, *start_point.tolist(), *end_point.tolist(),
                    float(np.linalg.norm(delta)),
                ])
            polyline_id += 1
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"points_mpc_h": points, "lines": lines, "metadata": metadata, 'graph_arrays': graph_arrays}
    return result


def _disperse_filament_projection_map(run_dir, snap, projection_axes, grid_size=None,
                                      slice_center_mpc_h=None, slice_thickness_mpc_h=None):
    cfg = load_cfg(run_dir)
    visualization = cfg.get("visualization", {})
    n = int(grid_size if grid_size is not None else visualization["filament_grid_size"])
    persistence_cut = float(visualization["disperse_persistence_cut"])
    box = float(cfg["simulation"]["box_mpc_h"])
    skeleton = _load_disperse_skeleton(run_dir, snap, n, persistence_cut)
    occupancy = np.zeros((n, n), dtype=bool)
    points = skeleton["points_mpc_h"]
    lines = skeleton["lines"]
    los_axis = ({0, 1, 2} - set(projection_axes)).pop()
    cursor = 0
    centers = (np.arange(n, dtype=float) + 0.5) * box / n
    while cursor < len(lines):
        count = int(lines[cursor])
        indices = lines[cursor + 1:cursor + 1 + count]
        cursor += count + 1
        if count < 2 or np.any(indices < 0) or np.any(indices >= len(points)):
            continue
        vertices = points[indices]
        for start, finish in zip(vertices[:-1], vertices[1:]):
            delta = finish - start
            delta -= box * np.round(delta / box)
            samples = max(2, int(np.ceil(np.max(np.abs(delta)) * n / box)) + 1)
            fraction = np.linspace(0.0, 1.0, samples)
            xyz = np.mod(start[None, :] + fraction[:, None] * delta[None, :], box)
            visible = np.ones(samples, dtype=bool)
            if slice_thickness_mpc_h is not None:
                distance = xyz[:, los_axis] - float(slice_center_mpc_h)
                distance -= box * np.round(distance / box)
                visible &= np.abs(distance) <= float(slice_thickness_mpc_h) / 2.0
            projected = xyz[visible][:, list(projection_axes)]
            if len(projected):
                ix = np.floor(projected[:, 0] * n / box).astype(int) % n
                iy = np.floor(projected[:, 1] * n / box).astype(int) % n
                occupancy[ix, iy] = True
    metadata = dict(skeleton["metadata"])
    metadata["projection"] = "3-D skeleton projected along axis %d" % los_axis
    metadata["filament_cell_count_2d"] = int(np.count_nonzero(occupancy))
    if slice_thickness_mpc_h is not None:
        metadata["projection"] = "periodic skeleton segments sampled within the same line-of-sight slab"
        metadata["slice_thickness_mpc_h"] = float(slice_thickness_mpc_h)
        metadata["slice_center_mpc_h"] = float(slice_center_mpc_h)
    return centers, occupancy, metadata


def _periodic_skeleton_polydata(skeleton, box_size_mpc_h):
    """Build short, edge-wrapped VTK line segments from periodic skeleton paths."""
    import pyvista as pv

    box = float(box_size_mpc_h)
    source_points = np.asarray(skeleton["points_mpc_h"], dtype=float)
    source_lines = np.asarray(skeleton["lines"], dtype=np.int64)
    segments = []
    cursor = 0
    while cursor < len(source_lines):
        count = int(source_lines[cursor])
        indices = source_lines[cursor + 1:cursor + 1 + count]
        cursor += count + 1
        if count < 2 or np.any(indices < 0) or np.any(indices >= len(source_points)):
            continue
        for start, finish in zip(source_points[indices[:-1]], source_points[indices[1:]]):
            delta = finish - start
            delta -= box * np.round(delta / box)
            cuts = [0.0, 1.0]
            for axis in range(3):
                end = start[axis] + delta[axis]
                if delta[axis] > 0 and end >= box:
                    crossing = (box - start[axis]) / delta[axis]
                    if 0.0 < crossing < 1.0:
                        cuts.append(float(crossing))
                elif delta[axis] < 0 and end < 0.0:
                    crossing = -start[axis] / delta[axis]
                    if 0.0 < crossing < 1.0:
                        cuts.append(float(crossing))
            cuts = sorted(set(cuts))
            for low, high in zip(cuts[:-1], cuts[1:]):
                middle = start + ((low + high) / 2.0) * delta
                image_offset = np.floor(middle / box) * box
                a = np.clip(start + low * delta - image_offset, 0.0, box)
                b = np.clip(start + high * delta - image_offset, 0.0, box)
                segments.append((a, b))
    if not segments:
        return pv.PolyData()
    points = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
    lines = np.empty((len(segments), 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * len(segments), 2)
    lines[:, 2] = lines[:, 1] + 1
    return pv.PolyData(points, lines=lines.reshape(-1))


def _draw_single_run_structure_overlays(ax, run_dir, snap, axes, box,
                                        line_of_sight_axis=None,
                                        slice_center_mpc_h=None,
                                        slice_thickness_mpc_h=None,
                                        include_catalogues=True,
                                        overlay_layers=None):
    """Draw FOF, SUBFIND, Rockstar, rare-host and R200c layers on a map."""
    cfg = load_cfg(run_dir)
    visualization = cfg.get("visualization", {})
    if overlay_layers is None:
        overlay_layers = visualization.get(
            "structure_map_overlays",
        )
    overlay_layers = set(overlay_layers)
    statuses = {}
    if include_catalogues and visualization.get("structure_catalogue_overlays", True):
        try:
            mass_definition = str(cfg["analysis"]["hmf_mass_definition"])
            styles = {
                "subhalos": ("#ff7185", "x", 14, "SUBFIND subhalos"),
                "rockstar": ("#bd93f9", "^", 28, "Rockstar hosts"),
            }
            max_rows = int(cfg.get("visualization", {}).get("max_matched_halo_overlays", 1000))
            for kind, (color, marker, size, label) in styles.items():
                if kind not in overlay_layers:
                    continue
                layer = _read_catalogue_layer(run_dir, snap, kind, mass_definition)
                if layer is None:
                    statuses[kind] = "UNAVAILABLE"
                    continue
                positions = np.asarray(layer["position_mpc_h"], dtype=float)
                masses = np.asarray(layer["mass_msun_h"], dtype=float)
                keep = np.all(np.isfinite(positions), axis=1) & np.isfinite(masses) & (masses > 0)
                if slice_thickness_mpc_h is not None and line_of_sight_axis is not None:
                    distance = positions[:, line_of_sight_axis] - float(slice_center_mpc_h)
                    distance -= box * np.round(distance / box)
                    keep &= np.abs(distance) <= float(slice_thickness_mpc_h) / 2.0
                rows = np.flatnonzero(keep)
                if len(rows) > max_rows:
                    rows = rows[np.argpartition(masses[rows], -max_rows)[-max_rows:]]
                if len(rows):
                    ax.scatter(
                        positions[rows][:, list(axes)][:, 0], positions[rows][:, list(axes)][:, 1],
                        s=size, marker=marker, c=color, linewidths=0.6,
                        alpha=0.72, label=label, zorder=5,
                    )
                statuses[kind] = "COMPLETE" if len(rows) else "NO_OBJECTS_IN_VIEW"

            from analysis import group_field, units_to_mpc_h, mass_to_msun_h
            group_positions = group_field(run_dir, snap, ["GroupPos"]) if ({"fof", "rare_hosts", "r200c"} & overlay_layers) else None
            group_mass = group_field(run_dir, snap, ["GroupMass"]) if ({"fof", "rare_hosts", "r200c"} & overlay_layers) else None
            if group_positions is not None and group_mass is not None:
                host_pos = units_to_mpc_h(group_positions, box)
                host_mass = mass_to_msun_h(group_mass)
                good = np.all(np.isfinite(host_pos), axis=1) & np.isfinite(host_mass) & (host_mass > 0)
                if slice_thickness_mpc_h is not None and line_of_sight_axis is not None:
                    distance = host_pos[:, line_of_sight_axis] - float(slice_center_mpc_h)
                    distance -= box * np.round(distance / box)
                    good &= np.abs(distance) <= float(slice_thickness_mpc_h) / 2.0
                thresholds = sorted(set(
                    float(value) for value in cfg.get("analysis", {}).get("rare_halo_thresholds_msun_h", [])
                    if np.isfinite(float(value)) and float(value) > 0
                ))
                if thresholds and "rare_hosts" in overlay_layers:
                    bounds = thresholds + [np.inf]
                    markers = ["D", "P", "*", "X", "^", "s"]
                    colors = ["#fde68a", "#fb923c", "#f472b6", "#c4b5fd", "#86efac", "#fda4af"]
                    rare_shown = 0
                    for i in range(len(thresholds)):
                        category = good & (host_mass >= bounds[i]) & (host_mass < bounds[i + 1])
                        if not np.any(category):
                            continue
                        rows = np.flatnonzero(category)
                        if len(rows) > max_rows:
                            rows = rows[np.argpartition(host_mass[rows], -max_rows)[-max_rows:]]
                        rare_shown += int(len(rows))
                        ax.scatter(
                            host_pos[rows][:, list(axes)][:, 0], host_pos[rows][:, list(axes)][:, 1],
                            s=30 + 14 * i, marker=markers[i % len(markers)],
                            c=colors[i % len(colors)], edgecolors="#111827", linewidths=0.35,
                            alpha=0.95, label=("Rare FOF hosts: %.2g–%.2g M_sun/h" % (bounds[i], bounds[i + 1]))
                            if np.isfinite(bounds[i + 1]) else "Rare FOF hosts: ≥ %.2g M_sun/h" % bounds[i],
                            zorder=6,
                        )
                    statuses["rare_hosts"] = {
                        "status": "COMPLETE",
                        "thresholds_msun_h": thresholds,
                        "shown": rare_shown,
                    }
                elif "rare_hosts" in overlay_layers:
                    statuses["rare_hosts"] = "UNAVAILABLE: no positive rare-halo mass thresholds are configured"

                radii = group_field(run_dir, snap, ["Group_R_Crit200"]) if "r200c" in overlay_layers else None
                if radii is not None:
                    from matplotlib.patches import Circle
                    radii = np.asarray(radii, dtype=float).reshape(-1) / 1000.0
                    candidates = np.flatnonzero(good & np.isfinite(radii) & (radii > 0) & (radii < box / 2.0))
                    if len(candidates) > min(max_rows, 150):
                        candidates = candidates[np.argpartition(host_mass[candidates], -min(max_rows, 150))[-min(max_rows, 150):]]
                    for row in candidates:
                        projected_radius = float(radii[row])
                        if slice_thickness_mpc_h is not None and line_of_sight_axis is not None:
                            distance = host_pos[row, line_of_sight_axis] - float(slice_center_mpc_h)
                            distance -= box * np.round(distance / box)
                            nearest = max(abs(distance) - float(slice_thickness_mpc_h) / 2.0, 0.0)
                            if nearest >= projected_radius:
                                continue
                            projected_radius = math.sqrt(max(projected_radius**2 - nearest**2, 0.0))
                        center_2d = host_pos[row, list(axes)]
                        for shift_x in (-box, 0.0, box):
                            for shift_y in (-box, 0.0, box):
                                shifted = center_2d + np.asarray([shift_x, shift_y])
                                if (shifted[0] + projected_radius < 0 or shifted[0] - projected_radius > box
                                        or shifted[1] + projected_radius < 0 or shifted[1] - projected_radius > box):
                                    continue
                                ax.add_patch(Circle(
                                    shifted, projected_radius, fill=False,
                                    edgecolor="#e2e8f0", linewidth=0.45, alpha=0.55, zorder=4,
                                ))
                    statuses["r200c"] = "COMPLETE" if len(candidates) else "UNAVAILABLE"
        except (OSError, KeyError, ValueError, TypeError, OverflowError) as exc:
            statuses["catalogues"] = "UNAVAILABLE: " + str(exc)

    if visualization["filament_overlay"] and "filaments" in overlay_layers:
        try:
            filament_method = str(visualization["filament_method"])
            if filament_method == "disperse_grid":
                ridge_result = _disperse_filament_projection_map(
                    run_dir, snap, axes, slice_center_mpc_h=slice_center_mpc_h,
                    slice_thickness_mpc_h=slice_thickness_mpc_h,
                )
                filament_label = "DisPerSE persistent skeleton"
            else:
                ridge_result = _periodic_density_ridge_map(
                    run_dir, snap, axes, slice_center_mpc_h=slice_center_mpc_h,
                    slice_thickness_mpc_h=slice_thickness_mpc_h,
                )
                filament_label = "Hessian density-ridge proxy"
            if ridge_result is None:
                statuses["filaments"] = "UNAVAILABLE: insufficient dark-matter density samples"
            else:
                centers, ridge, method = ridge_result
                occupancy = ridge.astype(float)
                if slice_thickness_mpc_h is not None and filament_method != "disperse_grid":
                    method["projection"] = "periodic Hessian ridges restricted to the same line-of-sight slab"
                if np.any(occupancy):
                    if filament_method == "disperse_grid":
                        color = visualization["filament_color"]
                        ax.contour(centers, centers, occupancy.T, levels=[0.5],
                                   colors=[color], linewidths=4.2, alpha=0.10, zorder=2)
                        ax.contour(centers, centers, occupancy.T, levels=[0.5],
                                   colors=[color], linewidths=1.9, alpha=0.30, zorder=2.5)
                        ax.contour(centers, centers, occupancy.T, levels=[0.5],
                                   colors=[color], linewidths=0.80, alpha=0.98, zorder=3)
                    else:
                        ax.contour(
                            centers, centers, occupancy.T, levels=[0.5],
                            colors=[visualization["filament_color"]], linewidths=0.75,
                            alpha=0.92, zorder=3,
                        )
                    method["legend_label"] = filament_label
                    statuses["filaments"] = {"status": "COMPLETE", **method}
                else:
                    statuses["filaments"] = "NO_RIDGES"
        except (OSError, KeyError, ValueError, TypeError, OverflowError, ImportError,
                RuntimeError, subprocess.SubprocessError) as exc:
            statuses["filaments"] = "UNAVAILABLE: " + str(exc)
    return statuses


def _add_map_scale_bar(ax):
    """Add a compact, high-contrast physical scale bar inside a structure map."""
    from matplotlib import patheffects

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span = min(abs(float(xmax - xmin)), abs(float(ymax - ymin)))
    if not np.isfinite(span) or span <= 0:
        return None
    target = span / 5.0
    decade = 10.0 ** math.floor(math.log10(target))
    length = max((factor * decade for factor in (1.0, 2.0, 5.0, 10.0) if factor * decade <= target), default=decade)
    x0 = min(xmin, xmax) + 0.055 * abs(xmax - xmin)
    y0 = min(ymin, ymax) + 0.075 * abs(ymax - ymin)
    line, = ax.plot([x0, x0 + length], [y0, y0], color="white", linewidth=2.3, zorder=10)
    line.set_path_effects([patheffects.Stroke(linewidth=4.2, foreground="#10131a"), patheffects.Normal()])
    label = ax.text(
        x0 + length / 2.0, y0 + 0.022 * span, "%.3g Mpc/h" % length,
        color="white", fontsize=8, ha="center", va="bottom", zorder=10,
        bbox={"facecolor": "#10131a", "edgecolor": "none", "alpha": 0.65, "pad": 1.8},
    )
    label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="#10131a")])
    return float(length)


def density_map(run_dir, snap, field, projection, vmin, vmax, normalization_mode,
                overlay_halos=True, slice_thickness_mpc_h=None, slice_center_mpc_h=None,
                overlay_layers=None, map_window_mpc_h=None,
                output_suffix="", output_directory=None):
    projected = _projected_density(
        run_dir, snap, field, projection,
        slice_thickness_mpc_h, slice_center_mpc_h,
    )
    if projected is None:
        return None
    img, box, z, axes = projected[:4]
    slice_info = projected[5]
    slice_thickness_mpc_h = slice_info["thickness_mpc_h"]
    slice_center_mpc_h = slice_info["center_mpc_h"]
    cfg = load_cfg(run_dir)
    visualization = cfg["visualization"]
    resolution = simulation_resolution_metadata(cfg)
    density_grid = int(cfg["analysis"]["density_grid"])
    filament_grid = int(visualization["filament_grid_size"])
    outdir = Path(output_directory) if output_directory is not None else Path(run_dir) / "results" / "structure_maps"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ("%s_%s%s_%03d%s.png" % (
        field, projection, _slice_tag(slice_thickness_mpc_h, slice_center_mpc_h), snap, output_suffix,
    ))
    if overlay_layers is None:
        overlay_layers = visualization["structure_map_overlays"]
    overlay_layers = set(overlay_layers)
    if not overlay_halos:
        overlay_layers.clear()
    background = "#050713"
    foreground = "#f2f4ff"
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=background)
    ax.set_facecolor(background)
    cmap = visualization["temperature_cmap"] if field == "gas_temperature" else visualization["density_cmap"]
    import matplotlib
    display_cmap = matplotlib.colormaps[cmap].with_extremes(bad="#eeeae2" if field == "gas_temperature" else matplotlib.colormaps[cmap](0))
    image = ax.imshow(
        img, origin="lower", extent=[0, box, 0, box], interpolation="nearest",
        vmin=vmin, vmax=vmax, cmap=display_cmap,
    )
    ax.grid(False)
    if "fof" in overlay_layers:
        halo_positions, halo_masses = _halo_overlay(
            run_dir, snap, axes, line_of_sight_axis=slice_info["line_of_sight_axis"],
            slice_center_mpc_h=slice_center_mpc_h,
            slice_thickness_mpc_h=slice_thickness_mpc_h,
            max_halos=int(visualization["max_matched_halo_overlays"]),
        )
        if halo_positions is not None and len(halo_masses):
            log_mass = np.log10(halo_masses)
            sizes = np.clip(8.0 + 4.0 * (log_mass - 8.0), 9.0, 54.0)
            ax.scatter(
                halo_positions[:, 0], halo_positions[:, 1], s=sizes, marker="o",
                facecolors="none", edgecolors=foreground, linewidths=0.65, alpha=0.78,
                label=r"FOF centres (size $\propto \log_{10}M_{\rm FOF}$)", zorder=5,
            )
            fof_status = {"status": "COMPLETE", "shown": int(len(halo_masses))}
        else:
            fof_status = "UNAVAILABLE_OR_NO_HOSTS"
    else:
        fof_status = "DISABLED"
    overlay_status = _draw_single_run_structure_overlays(
        ax, run_dir, snap, axes, box,
        line_of_sight_axis=slice_info["line_of_sight_axis"],
        slice_center_mpc_h=slice_center_mpc_h,
        slice_thickness_mpc_h=slice_thickness_mpc_h,
        include_catalogues=overlay_halos,
        overlay_layers=overlay_layers,
    )
    overlay_status["fof"] = fof_status
    if overlay_status.get("filaments") and isinstance(overlay_status["filaments"], dict):
        ax.plot([], [], color=visualization["filament_color"], linewidth=1.2,
                label=overlay_status["filaments"].get("legend_label", "Filament skeleton"))
    if overlay_status.get("r200c") == "COMPLETE":
        ax.plot([], [], color="#e2e8f0", marker="o", markerfacecolor="none",
                linestyle="none", markersize=5, label="FOF $R_{200c}$ shells")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        legend = ax.legend(
            unique.values(), unique.keys(), loc="upper right", fontsize=7.5,
            frameon=True, fancybox=True, framealpha=0.82,
            facecolor="#090d1d", edgecolor="#66708a", borderpad=0.65,
        )
        legend.set_zorder(9)
        for text_item in legend.get_texts():
            text_item.set_color(foreground)
    ax.set_xlabel(projection[0] + " [Mpc/h]", color=foreground)
    ax.set_ylabel(projection[1] + " [Mpc/h]", color=foreground)
    ax.tick_params(axis="both", colors=foreground, direction="out", width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#8991a8")
        spine.set_alpha(0.55)
    ax.set_aspect("equal", adjustable="box")
    if field == "gas_temperature":
        colorbar_label = r"$\log_{10}(\langle T\rangle_M)$ [K]"
        title_field = "Gas temperature"
    else:
        names = {"dm": "Dark-matter", "gas": "Gas", "stars": "Stellar", "total": "Total-matter"}
        title_field = names[field] + " surface density"
        colorbar_label = r"$\log_{10}\Sigma$ [$hM_\odot\,\mathrm{Mpc}^{-2}$]"
    title = (
        f"{title_field}  |  z={z:.3f}  |  L={box:g} Mpc/h  |  "
        f"N={resolution['particles_per_dimension']}^3  |  "
        rf"$m_p$={resolution['particle_mass_msun_h']:.3g} $M_\odot/h$"
    )
    if slice_thickness_mpc_h is not None:
        title = "Slab-projected " + title
    ax.set_title(title, color=foreground, pad=10)
    if map_window_mpc_h is not None:
        if len(map_window_mpc_h) != 4:
            raise ValueError("map_window_mpc_h must be [xmin, xmax, ymin, ymax]")
        xmin, xmax, ymin, ymax = (float(value) for value in map_window_mpc_h)
        if not (0.0 <= xmin < xmax <= box and 0.0 <= ymin < ymax <= box):
            raise ValueError("map window must be ordered and contained within the periodic map box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    scale_bar_length = _add_map_scale_bar(ax)
    if overlay_status.get("filaments") and isinstance(overlay_status["filaments"], str):
        ax.text(0.01, 0.01, "Filament overlay: " + overlay_status["filaments"],
                color=foreground, fontsize=6.5, transform=ax.transAxes, va="bottom")
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.046)
    colorbar.set_label(colorbar_label, color=foreground)
    colorbar.ax.tick_params(colors=foreground)
    colorbar.outline.set_edgecolor("#8991a8")
    pixel_size = box / density_grid
    filament_cell_size = box / filament_grid
    map_metadata = {
        "snapshot": int(snap), "redshift": float(z), "field": field,
        "projection": projection, "slice": slice_info,
        "overlays": overlay_status,
        "density_colormap": cmap,
        "overlay_layers": sorted(overlay_layers),
        "map_window_mpc_h": map_window_mpc_h,
        "scale_bar_mpc_h": scale_bar_length,
        "normalization": normalization_mode,
        "color_limits_log10": [vmin, vmax],
        "box_size_mpc_h": box,
        "volume_mpc_h3": resolution["volume_mpc_h3"],
        "particles_per_dimension": resolution["particles_per_dimension"],
        "particle_count": resolution["particle_count"],
        "particle_mass_msun_h": resolution["particle_mass_msun_h"],
        "mean_particle_spacing_mpc_h": resolution["mean_particle_spacing_mpc_h"],
        "map_pixel_count_per_side": density_grid,
        "map_pixel_size_mpc_h": pixel_size,
        "filament_method": visualization["filament_method"],
        "filament_grid_cells_per_side": filament_grid,
        "filament_grid_cell_size_mpc_h": filament_cell_size,
        "map_array_shape_pixels": list(np.asarray(img).shape),
        "assignment_kernel": "periodic cloud-in-cell on pixel centres; no added smoothing",
        "empty_pixel_display": "lowest density colour; no artificial mass floor",
        "surface_density_unit": "h Msun / Mpc^2",
    }
    from core.data import run_epoch_contract, write_dataset_contract, coordinate_contract
    header,_=read_snapshot_header(run_dir,snap)
    epoch=run_epoch_contract(run_dir,int(snap),_hdf5_scalar_attr(header['Time'],'Time'))
    map_metadata['epoch_contract']=epoch
    map_metadata['numeric_asset']=out.with_suffix('.hdf5').name
    units='K' if field=='gas_temperature' else 'h Msun / Mpc^2'
    map_metadata['measured_quantity_units']=units
    with h5py.File(out.with_suffix('.hdf5'),'w') as numeric:
        numeric.attrs['science_acceptance']='NOT_ACCEPTED'
        numeric.attrs['pixel_axes_json']=json.dumps({'rows':int(axes[1]),'columns':int(axes[0]),
            'centres':'(index+0.5)*box/grid; periodic CIC pixels','pixel_size_mpc_h':box/density_grid})
        measured=numeric.create_dataset('measured',data=projected[6],chunks=True,fletcher32=True)
        log_image=numeric.create_dataset('log10_display',data=img,chunks=True,fletcher32=True)
        coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
            periodic_wrapping='periodic CIC pixels; optional centered line-of-sight slab',velocity_convention='not applicable to projected scalar field')
        context={'bundle_manifest':'map_contract.json','snapshot_storage_index':int(snap),
            'projection':projection,'slice':slice_info,'assignment':'periodic CIC; no smoothing'}
        write_dataset_contract(measured,units=units,epoch=epoch,coordinates=coordinates,provenance=context,
            quantity='CIC mass-weighted gas temperature' if field=='gas_temperature' else 'CIC projected mass surface density',
            missing_policy='temperature NaN when no supported mass; empty mass surface-density pixels are exact zero')
        write_dataset_contract(log_image,units='dimensionless log10(value / ('+units+'))',epoch=epoch,
            coordinates=coordinates,provenance=context,quantity='display logarithm of measured positive pixels',
            missing_policy='NaN when measured pixel is nonpositive or unavailable; no artificial floor')
    export_figure(fig, out, metadata=map_metadata, dpi=int(visualization["figure_dpi"]))
    out.with_suffix(".json").write_text(json.dumps(map_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return out


def make_structure_story_map(run_dir, snap, field, projections=None,
                             slice_thickness_mpc_h=None, slice_center_mpc_h=None,
                             map_window_mpc_h=None, output_directory=None):
    """Render a shared-scale, three-projection structure figure for one output."""
    cfg = load_cfg(run_dir)
    visualization = cfg["visualization"]
    projections = list(visualization["structure_story_projections"] if projections is None else projections)
    if not projections or any(projection not in {"xy", "xz", "yz"} for projection in projections):
        raise ValueError("structure-story projections must be selected from xy, xz, and yz")
    box = float(cfg["simulation"]["box_mpc_h"])
    resolution = simulation_resolution_metadata(cfg)
    density_grid = int(cfg["analysis"]["density_grid"])
    filament_grid = int(visualization["filament_grid_size"])
    slice_thickness_mpc_h, slice_center_mpc_h = _resolve_slice_settings(
        cfg, box, slice_thickness_mpc_h, slice_center_mpc_h,
    )
    projected = []
    for projection in projections:
        product = _projected_density(
            run_dir, snap, field, projection,
            slice_thickness_mpc_h, slice_center_mpc_h,
        )
        if product is not None:
            projected.append((projection, product))
    if not projected:
        return None
    percentiles = visualization["density_display_percentiles"]
    low_percentile, high_percentile = (float(percentiles[0]), float(percentiles[1]))
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("density_display_percentiles must satisfy 0 <= low < high <= 100")
    finite_chunks = [product[0][np.isfinite(product[0])] for _, product in projected]
    finite_chunks = [values for values in finite_chunks if len(values)]
    if not finite_chunks:
        return None
    pooled = np.concatenate(finite_chunks)
    vmin, vmax = float(np.percentile(pooled, low_percentile)), float(np.percentile(pooled, high_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        vmax = float(np.nextafter(vmin, np.inf))

    outdir = Path(output_directory) if output_directory is not None else Path(run_dir) / "results/structure_maps"
    outdir.mkdir(parents=True, exist_ok=True)
    cmap = visualization["temperature_cmap"] if field == "gas_temperature" else visualization["density_cmap"]
    overlay_layers = set(visualization["structure_map_overlays"])
    map_window = map_window_mpc_h if map_window_mpc_h is not None else visualization.get("map_window_mpc_h")
    if map_window is not None:
        if len(map_window) != 4:
            raise ValueError("map_window_mpc_h must be [xmin, xmax, ymin, ymax]")
        xmin, xmax, ymin, ymax = (float(value) for value in map_window)
        if not (0.0 <= xmin < xmax <= box and 0.0 <= ymin < ymax <= box):
            raise ValueError("map window must be ordered and contained within the periodic map box")
    fig, axes = plt.subplots(1, len(projected), figsize=(10.8, 4.2), squeeze=False, layout="constrained")
    axes = list(axes.flat)
    image = None
    projection_metadata = {}
    import matplotlib
    display_cmap = matplotlib.colormaps[cmap].with_extremes(bad="#eeeae2" if field == "gas_temperature" else matplotlib.colormaps[cmap](0))
    for axis, (projection, product) in zip(axes, projected):
        img, _box, redshift, plane_axes = product[:4]
        slice_info = product[5]
        image = axis.imshow(
            img, origin="lower", extent=[0, box, 0, box], interpolation="nearest",
            vmin=vmin, vmax=vmax, cmap=display_cmap,
        )
        axis.grid(False)
        fof_status = "DISABLED"
        if "fof" in overlay_layers:
            host_positions, host_masses = _halo_overlay(
                run_dir, snap, plane_axes,
                max_halos=max(1, int(visualization.get("max_matched_halo_overlays", 1000))),
                line_of_sight_axis=slice_info["line_of_sight_axis"],
                slice_center_mpc_h=slice_info["center_mpc_h"],
                slice_thickness_mpc_h=slice_info["thickness_mpc_h"],
            )
            if host_positions is not None and len(host_masses):
                log_mass = np.log10(host_masses)
                sizes = np.clip(8.0 + 4.0 * (log_mass - 8.0), 9.0, 54.0)
                axis.scatter(
                    host_positions[:, 0], host_positions[:, 1], s=sizes, marker="o",
                    facecolors="none", edgecolors="#f8fafc", linewidths=0.65,
                    alpha=0.78, label=r"FOF centres (size $\propto \log_{10}M$)", zorder=5,
                )
                fof_status = {"status": "COMPLETE", "shown": int(len(host_masses))}
            else:
                fof_status = "UNAVAILABLE_OR_NO_HOSTS"
        overlays = _draw_single_run_structure_overlays(
            axis, run_dir, snap, plane_axes, box,
            line_of_sight_axis=slice_info["line_of_sight_axis"],
            slice_center_mpc_h=slice_info["center_mpc_h"],
            slice_thickness_mpc_h=slice_info["thickness_mpc_h"],
            overlay_layers=overlay_layers,
        )
        overlays["fof"] = fof_status
        if isinstance(overlays.get("filaments"), dict):
            axis.plot([], [], color=visualization["filament_color"], linewidth=1.0,
                      label=overlays["filaments"].get("legend_label", "Filament skeleton"))
        if overlays.get("r200c") == "COMPLETE":
            axis.plot([], [], color="#e2e8f0", marker="o", markerfacecolor="none",
                      linestyle="none", markersize=5, label="FOF R200c shells")
        axis.set_xlabel(projection[0] + " [Mpc/h]")
        axis.set_ylabel(projection[1] + " [Mpc/h]")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(projection.upper())
        if slice_info["thickness_mpc_h"] is not None:
            axis.text(
                0.02, 0.98,
                "slab %.3g Mpc/h" % slice_info["thickness_mpc_h"],
                ha="left", va="top", fontsize=6, color="white", transform=axis.transAxes,
                bbox={"facecolor": "#10131a", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
            )
        if map_window is not None:
            axis.set_xlim(xmin, xmax); axis.set_ylim(ymin, ymax)
        scale_bar = _add_map_scale_bar(axis)
        projection_metadata[projection] = {
            "redshift": float(redshift), "slice": slice_info,
            "overlays": overlays, "scale_bar_mpc_h": scale_bar,
        }
    field_names = {"dm": "Dark-matter", "gas": "Gas", "stars": "Stellar", "total": "Total-matter"}
    if field == "gas_temperature":
        label = r"$\log_{10}(\langle T\rangle_M)$ [K]"
        field_title = "Mass-weighted gas temperature"
    else:
        label = r"$\log_{10}\Sigma$ [$hM_\odot/\mathrm{Mpc}^2$]"
        field_title = field_names.get(field, field)
    redshift = float(projected[0][1][2])
    fig.suptitle(
        "%s structure | z = %.4g\nL = %.5g Mpc/h  |  N = %d^3  |  "
        "$m_p$ = %.4g $M_\\odot/h$  |  %d^2 map pixels (%.5g Mpc/h/pixel)" % (
            field_title, redshift, box, resolution["particles_per_dimension"],
            resolution["particle_mass_msun_h"], density_grid, box / density_grid,
        ),
        weight="bold",
    )
    if image is not None:
        fig.colorbar(image, ax=axes, label=label, shrink=0.82, pad=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label_text in zip(handles, labels):
        unique.setdefault(label_text, handle)
    if unique:
        axes[0].legend(
            unique.values(), unique.keys(), loc="upper right", fontsize=5.5,
            frameon=True, fancybox=True, framealpha=0.88,
            facecolor="#111827", edgecolor="#cbd5e1", labelcolor="#f8fafc",
        )
    slice_tag = _slice_tag(slice_thickness_mpc_h, slice_center_mpc_h)
    out = outdir / ("structure_story_%s%s_%03d" % (field, slice_tag, int(snap)))
    export_figure(fig, out, metadata={
        "product": "multi-projection structure story map",
        "snapshot": int(snap), "redshift": redshift, "field": field,
        "box_size_mpc_h": box, "volume_mpc_h3": resolution["volume_mpc_h3"],
        "particles_per_dimension": resolution["particles_per_dimension"],
        "particle_count": resolution["particle_count"],
        "particle_mass_msun_h": resolution["particle_mass_msun_h"],
        "mean_particle_spacing_mpc_h": resolution["mean_particle_spacing_mpc_h"],
        "map_pixel_count_per_side": density_grid,
        "map_pixel_size_mpc_h": box / density_grid,
        "filament_grid_cells_per_side": filament_grid,
        "filament_grid_cell_size_mpc_h": box / filament_grid,
        "assignment_kernel": "periodic cloud-in-cell on pixel centres; no added smoothing",
        "empty_pixel_display": "lowest density colour; no artificial mass floor",
        "surface_density_unit": "h Msun / Mpc^2",
        "projections": [name for name, _ in projected],
        "normalization": "shared pooled percentile normalization across projections",
        "display_percentiles": [low_percentile, high_percentile],
        "color_limits": [vmin, vmax], "density_colormap": cmap,
        "overlay_layers": sorted(overlay_layers), "projection_metadata": projection_metadata,
        "map_window_mpc_h": map_window,
    }, dpi=int(visualization["figure_dpi"]))
    out.with_suffix(".json").write_text(json.dumps({
        "snapshot": int(snap), "redshift": redshift, "field": field,
        "box_size_mpc_h": box, "volume_mpc_h3": resolution["volume_mpc_h3"],
        "particles_per_dimension": resolution["particles_per_dimension"],
        "particle_count": resolution["particle_count"],
        "particle_mass_msun_h": resolution["particle_mass_msun_h"],
        "mean_particle_spacing_mpc_h": resolution["mean_particle_spacing_mpc_h"],
        "map_pixel_count_per_side": density_grid,
        "map_pixel_size_mpc_h": box / density_grid,
        "filament_grid_cells_per_side": filament_grid,
        "filament_grid_cell_size_mpc_h": box / filament_grid,
        "assignment_kernel": "periodic cloud-in-cell on pixel centres; no added smoothing",
        "empty_pixel_display": "lowest density colour; no artificial mass floor",
        "surface_density_unit": "h Msun / Mpc^2",
        "projections": [name for name, _ in projected],
        "normalization": "shared pooled percentile normalization across projections",
        "display_percentiles": [low_percentile, high_percentile],
        "color_limits": [vmin, vmax], "colormap": cmap,
        "overlays": projection_metadata, "overlay_layers": sorted(overlay_layers),
        "map_window_mpc_h": map_window,
    }, indent=2), encoding="utf-8")
    return out


@native_run_owner
def make_maps(run_dir, fields, projections,
              slice_thickness_mpc_h=None, slice_center_mpc_h=None,
              scale_mode=None, map_window_mpc_h=None, story_figure=None):
    """Publish an immutable map bundle with raw measurements and verified lineage."""
    import uuid
    from core.data import coordinate_contract
    from core.foundation import product_provenance, verify_product_provenance, publish_directory, atomic_json, file_sha256 as digest
    root=Path(run_dir).expanduser().resolve()
    if not snapshot_ids(root):return []
    if fields is None or projections is None:raise ValueError('Map fields and projections must be selected explicitly')
    fields=list(fields);projections=list(projections)
    sources=sorted((root/'data').rglob('*.hdf5'))
    sources+=sorted((root/'results').rglob('galaxies_*.hdf5')) if (root/'results').is_dir() else []
    contract=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name(name) for name in
        ('analysis.py','figures.py')],source_files=sources,
        selection={'fields':fields,'projections':projections,'scale_mode':scale_mode,'story_figure':story_figure,
            'slice_thickness_mpc_h':slice_thickness_mpc_h,'slice_center_mpc_h':slice_center_mpc_h,'map_window_mpc_h':map_window_mpc_h},
        coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
            periodic_wrapping='periodic projected box with declared optional slab',velocity_convention='not applicable to scalar maps'))
    parent=root/'results/structure_maps';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.maps-',dir=parent));destination=parent/('maps_'+uuid.uuid4().hex)
    try:
        made=_write_maps(root,fields,projections,slice_thickness_mpc_h,slice_center_mpc_h,
            scale_mode,map_window_mpc_h,story_figure,output_directory=stage)
        verify_product_provenance(root,contract)
        atomic_json(stage/'map_contract.json',{'schema':'map-publication-v1','science_acceptance':'NOT_ACCEPTED',
            'validation_state':'VALIDATION_PENDING','provenance':contract,
            'artifacts':[{'path':path.relative_to(stage).as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size}
                for path in sorted(stage.rglob('*')) if path.is_file()]})
        publish_directory(stage,destination)
        return [destination/Path(path).relative_to(stage) for path in made]
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


def _write_maps(run_dir, fields, projections,
              slice_thickness_mpc_h=None, slice_center_mpc_h=None,
              scale_mode=None, map_window_mpc_h=None, story_figure=None, *, output_directory):
    """Render maps with fixed global or per-snapshot percentile normalization."""
    snaps = snapshot_ids(run_dir)
    if not snaps:
        return []
    cfg = load_cfg(run_dir)
    visualization = cfg["visualization"]
    if fields is None or projections is None:
        raise ValueError("map fields and projections must be selected explicitly")
    fields = list(dict.fromkeys(fields))
    projections = list(dict.fromkeys(projections))
    scale_mode = str(visualization["density_scale_mode"] if scale_mode is None else scale_mode)
    if scale_mode not in {"global", "per_snapshot"}:
        raise ValueError("scale_mode must be global or per_snapshot")
    percentiles = visualization["density_display_percentiles"]
    low_percentile, high_percentile = float(percentiles[0]), float(percentiles[1])
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("density_display_percentiles must satisfy 0 <= low < high <= 100")
    if map_window_mpc_h is None:
        map_window_mpc_h = visualization.get("map_window_mpc_h")
    overlay_layers = visualization["structure_map_overlays"]
    box = float(cfg["simulation"]["box_mpc_h"])
    slice_thickness_mpc_h, slice_center_mpc_h = _resolve_slice_settings(
        cfg, box, slice_thickness_mpc_h, slice_center_mpc_h,
    )
    slice_suffix = _slice_tag(slice_thickness_mpc_h, slice_center_mpc_h)
    scales = {}
    outs = []
    for field in fields:
        for projection in projections:
            highs, lows = [], []
            snapshot_limits = {}
            temperature_sources = {}
            for snap in snaps:
                projected = _projected_density(
                    run_dir, snap, field, projection,
                    slice_thickness_mpc_h, slice_center_mpc_h,
                )
                if projected is not None:
                    if field == "gas_temperature":
                        temperature_sources[str(snap)] = projected[4]
                    finite = projected[0][np.isfinite(projected[0])]
                    if finite.size:
                        snapshot_limits[int(snap)] = [
                            float(np.percentile(finite, low_percentile)),
                            float(np.percentile(finite, high_percentile)),
                        ]
                        lows.append(snapshot_limits[int(snap)][0])
                        highs.append(snapshot_limits[int(snap)][1])
            if not highs:
                scales[field + "_" + projection + slice_suffix] = {"status": "UNAVAILABLE", "reason": "no finite map pixels"}
                continue
            if (scale_mode == "global" and field == "gas_temperature"
                    and len(set(temperature_sources.values())) > 1):
                scales[field + "_" + projection + slice_suffix] = {
                    "status": "UNAVAILABLE",
                    "reason": "temperature representation changes across snapshots; one fixed-scale series would mix unlike inputs",
                    "temperature_source_by_snapshot": temperature_sources,
                }
                continue
            key = field + "_" + projection + slice_suffix
            if scale_mode == "global":
                global_limits = [min(lows), max(highs)]
                if global_limits[1] <= global_limits[0]:
                    global_limits[1] = np.nextafter(global_limits[0], np.inf)
                scales[key] = {
                    "status": "COMPLETE", "normalization": "global_fixed",
                    "vmin": global_limits[0], "vmax": global_limits[1],
                }
            else:
                for snap, limits in snapshot_limits.items():
                    if limits[1] <= limits[0]:
                        limits[1] = np.nextafter(limits[0], np.inf)
                scales[key] = {
                    "status": "COMPLETE", "normalization": "per_snapshot_percentile",
                    "percentiles": [low_percentile, high_percentile],
                    "limits_by_snapshot": {str(snap): values for snap, values in snapshot_limits.items()},
                }
            if field == "gas_temperature":
                scales[key]["temperature_source_by_snapshot"] = temperature_sources
            for snap in snaps:
                limits = global_limits if scale_mode == "global" else snapshot_limits.get(int(snap))
                if limits is None:
                    continue
                map_kwargs = {
                    "vmin": limits[0], "vmax": limits[1],
                    "slice_thickness_mpc_h": slice_thickness_mpc_h,
                    "slice_center_mpc_h": slice_center_mpc_h,
                    "map_window_mpc_h": map_window_mpc_h,
                    "normalization_mode": "global_fixed" if scale_mode == "global" else "per_snapshot_percentile",
                    "output_directory":output_directory,
                }
                density_path = density_map(
                    run_dir, snap, field, projection, overlay_layers=[], **map_kwargs,
                )
                if density_path:
                    outs.append(density_path)
                if overlay_layers:
                    suffix = "_filaments" if set(overlay_layers) == {"filaments"} else "_structure"
                    overlay_path = density_map(
                        run_dir, snap, field, projection, overlay_layers=overlay_layers,
                        output_suffix=suffix, **map_kwargs,
                    )
                    if overlay_path:
                        outs.append(overlay_path)
    scale_data = {
        "simulation_resolution": simulation_resolution_metadata(cfg),
        "density_grid_pixels_per_side": int(cfg["analysis"]["density_grid"]),
        "filament_grid_cells_per_side": int(visualization["filament_grid_size"]),
        "filament_grid_cell_size_mpc_h": box / int(visualization["filament_grid_size"]),
        "normalization": scale_mode,
        "display_percentiles": [low_percentile, high_percentile],
        "map_window_mpc_h": map_window_mpc_h,
        "slice": {
            "status": "FULL_BOX_PROJECTION" if slice_thickness_mpc_h is None else "PERIODIC_CENTERED_SLAB",
            "thickness_mpc_h": slice_thickness_mpc_h,
            "center_mpc_h": slice_center_mpc_h,
            "center_rule": "explicit periodic coordinate" if slice_thickness_mpc_h is not None else None,
        },
        "maps": scales,
    }
    story_figures = []
    if (visualization["structure_story_figure"]
            if story_figure is None else bool(story_figure)):
        try:
            latest_snap = min(
                snaps,
                key=lambda snap: _snapshot_redshift(run_dir, snap)[0],
            )
            for field in fields:
                story_path = make_structure_story_map(
                    run_dir, latest_snap, field=field,
                    slice_thickness_mpc_h=slice_thickness_mpc_h,
                    slice_center_mpc_h=slice_center_mpc_h,
                    map_window_mpc_h=map_window_mpc_h,
                    output_directory=output_directory,
                )
                if story_path is not None:
                    story_figures.append(story_path.name + ".png")
            scale_data["story_figures"] = story_figures
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            scale_data["story_figures"] = {"status": "UNAVAILABLE", "reason": str(exc)}
    if len(scales) == 1 and "dm_xy" in scales and scales["dm_xy"].get("status") == "COMPLETE":
        scale_data.update({"vmin": scales["dm_xy"]["vmin"], "vmax": scales["dm_xy"]["vmax"]})
    scale_path = Path(output_directory) / "scale.json"
    scale_path.write_text(json.dumps(scale_data, indent=2), encoding="utf-8")
    from figures import review_figure_folder
    figure_review = review_figure_folder(scale_path.parent)
    scale_data["figure_review"] = {
        "status": figure_review["status"],
        "figure_count": figure_review["figure_count"],
        "gallery_pdf": figure_review.get("gallery_pdf"),
        "report_json": "figure_review.json",
    }
    scale_path.write_text(json.dumps(scale_data, indent=2), encoding="utf-8")
    return outs


@native_run_owner
def make_movie(run_dir, slice_thickness_mpc_h=None, slice_center_mpc_h=None, field="dm"):
    """Encode one consistent XY map sequence as a distinct scientific movie."""
    names = {
        "dm": "cosmic_web", "gas": "gas_evolution",
        "gas_temperature": "gas_temperature_evolution", "stars": "stellar_evolution",
    }
    if field not in names:
        raise ValueError("movie field must be dm, gas, gas_temperature, or stars")
    cfg = load_cfg(run_dir)
    box = float(cfg["simulation"]["box_mpc_h"])
    slice_thickness_mpc_h, slice_center_mpc_h = _resolve_slice_settings(
        cfg, box, slice_thickness_mpc_h, slice_center_mpc_h,
    )
    slice_suffix = _slice_tag(slice_thickness_mpc_h, slice_center_mpc_h)
    snapshots = snapshot_ids(run_dir)
    if not snapshots:
        return None
    generated=make_maps(
            run_dir, fields=[field], projections=["xy"],
            slice_thickness_mpc_h=slice_thickness_mpc_h,
            slice_center_mpc_h=slice_center_mpc_h,
            story_figure=False,
        )
    by_name={path.name:path for path in generated}
    # Star formation can begin after the first saved output. Omit only
    # epochs with genuinely absent species; a missing map for present particles
    # remains a failure rather than being silently dropped or synthesized.
    ptype={'dm':1,'gas':0,'gas_temperature':0,'stars':4}[field]
    included=[];omitted=[]
    for snap in snapshots:
        _, files=read_snapshot_header(run_dir,snap)
        present=False
        for path in files:
            with h5py.File(path) as source:
                key=f'PartType{ptype}/Coordinates'
                if key in source and len(source[key]):present=True
        if present:included.append(snap)
        else:omitted.append({'snapshot':snap,'reason':'No saved particles of requested species'})
    if not included:return None
    frames = [by_name.get("%s_xy%s_%03d.png" % (field, slice_suffix, snap)) for snap in included]
    available = [frame for frame in frames if frame is not None and frame.is_file()]
    if len(available) != len(included):
        raise RuntimeError("%s movie needs one XY map per species-present snapshot; found %d for %d outputs" % (
            field, len(available), len(included),
        ))
    outdir = Path(run_dir) / "results/movies"
    outdir.mkdir(parents=True, exist_ok=True)
    import uuid
    from core.foundation import atomic_json, publish_directory, file_sha256 as digest
    stage=Path(tempfile.mkdtemp(prefix='.movie-',dir=outdir))
    destination=outdir/(names[field]+'_'+uuid.uuid4().hex)
    out = stage / ("%s%s.mp4" % (names[field], slice_suffix))
    try:
        inputs=[{'path':str(path.resolve()),'sha256':digest(path)} for path in frames]
        writer = imageio.get_writer(out, fps=int(cfg["visualization"].get("movie_fps", 24)), codec="libx264", quality=8)
        try:
            for frame in frames:writer.append_data(imageio.imread(frame))
        finally:writer.close()
        if any(digest(Path(item['path']))!=item['sha256'] for item in inputs):
            raise ValueError('Movie source frame changed during encoding')
        if not out.is_file() or not out.stat().st_size:raise ValueError('Movie encoder produced no payload')
        source_contract=frames[0].parent/'map_contract.json'
        atomic_json(stage/'movie_contract.json',{'schema':'map-movie-publication-v1','science_acceptance':'NOT_ACCEPTED',
            'validation_state':'VALIDATION_PENDING','frames':inputs,'snapshot_storage_indices':included,'omitted_snapshots':omitted,
            'time_representation':'equal frame spacing for saved outputs; consult source epochs for physical time intervals',
            'source_map_contract':{'path':str(source_contract.resolve()),'sha256':digest(source_contract)},
            'movie':{'path':out.name,'sha256':digest(out),'size_bytes':out.stat().st_size}})
        publish_directory(stage,destination)
        return destination/out.name
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


@native_run_owner
def make_halo_growth_movie(run_dir, fps=None, maximum_halos=8, *, analysis_path=None):
    """Publish a sealed growth-history movie without replacing earlier exports."""
    import uuid
    from core.data import coordinate_contract,verify_hdf5_dataset_payloads
    from core.foundation import product_provenance, verify_product_provenance, publish_directory, atomic_json, file_sha256 as digest
    root=Path(run_dir).expanduser().resolve()
    source=Path(analysis_path).resolve() if analysis_path is not None else root/'results/analysis.h5'
    if not source.is_file():return None
    integrity=verify_hdf5_dataset_payloads(source)
    contract=product_provenance(root,algorithm=__file__,source_files=[source],
        selection={'product':'halo-growth movie','fps':fps,'maximum_halos':maximum_halos},
        coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='not applicable to mass-versus-scale-factor history plot',
            periodic_wrapping='not applicable to history plot',velocity_convention='not applicable to mass histories'))
    parent=root/'results/movies';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.halo-growth-',dir=parent));destination=parent/('halo_growth_'+uuid.uuid4().hex)
    try:
        movie=_write_halo_growth_movie(root,fps,maximum_halos,output_directory=stage,analysis_path=source)
        if movie is None:
            shutil.rmtree(stage);return None
        movie=Path(movie)
        if not movie.is_file() or not movie.stat().st_size:raise ValueError('Halo growth encoder produced no payload')
        verify_product_provenance(root,contract)
        atomic_json(stage/'halo_growth_contract.json',{'schema':'halo-growth-publication-v1',
            'science_acceptance':'NOT_ACCEPTED','validation_state':'VALIDATION_PENDING','provenance':contract,
            'source_integrity':integrity,'representation':'display of saved main-progenitor mass points and native linked merger events; no evolved interpolated halos',
            'assets':[{'path':path.relative_to(stage).as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size}
                for path in sorted(stage.rglob('*')) if path.is_file()]})
        publish_directory(stage,destination)
        return destination/movie.name
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


def _read_halo_movie_tracks(history):
    """Read both selected-analysis row records and legacy vector histories."""
    def scalar(group, key, default=np.nan):
        if key in group and isinstance(group[key],h5py.Dataset):
            return group[key][()]
        return group.attrs.get(key,default)
    selected = history.get('data/histories')
    container = selected if selected is not None else history
    tracks=[]
    for key in sorted(container):
        halo=container[key]
        if not isinstance(halo,h5py.Group):continue
        if selected is not None:
            rows=halo.get('track')
            if not isinstance(rows,h5py.Group):continue
            rows=[rows[name] for name in sorted(rows)]
            scale=np.asarray([scalar(row,'scale_factor') for row in rows],dtype=float)
            redshift=np.asarray([scalar(row,'redshift') for row in rows],dtype=float)
            mass=np.asarray([scalar(row,'mass_msun_h') for row in rows],dtype=float)
        else:
            if not all(name in halo for name in ('scale_factor','redshift','mvir_msun_h')):continue
            scale=np.asarray(halo['scale_factor'][:],dtype=float)
            redshift=np.asarray(halo['redshift'][:],dtype=float)
            mass=np.asarray(halo['mvir_msun_h'][:],dtype=float)
        if scale.ndim!=1 or scale.shape!=mass.shape or scale.shape!=redshift.shape:
            raise ValueError('Halo movie history vectors do not align')
        good=np.isfinite(scale)&(scale>0)&np.isfinite(redshift)&(redshift>=0)&np.isfinite(mass)&(mass>0)
        if np.count_nonzero(good)<2:continue
        if not np.allclose(scale[good],1/(1+redshift[good]),rtol=1e-8,atol=1e-12):
            raise ValueError('Halo movie scale factors disagree with saved redshifts')
        order=np.argsort(scale[good]);scale=scale[good][order];mass=mass[good][order];redshift=redshift[good][order]
        if np.any(np.diff(scale)<=0):raise ValueError('Halo movie history has repeated epochs')
        mergers=[]
        events=halo.get('merger_events')
        if isinstance(events,h5py.Group):
            for event in events.values():
                if not isinstance(event,h5py.Group):continue
                z=float(scalar(event,'redshift'));ratio=float(scalar(event,'mass_ratio'));main=float(scalar(event,'main_mass_msun_h'))
                if np.isfinite(z) and z>=0 and np.isfinite(ratio) and ratio>0 and np.isfinite(main) and main>0:
                    mergers.append((1/(1+z),main,ratio))
        tracks.append({'id':int(scalar(halo,'final_halo_id',len(tracks))),
                       'scale':scale,'redshift':redshift,'mass':mass,'mergers':mergers})
    return tracks


def _write_halo_growth_movie(run_dir, fps=None, maximum_halos=8, *, output_directory, analysis_path=None):
    """Animate Rockstar main-progenitor mass tracks and their linked mergers."""
    cfg = load_cfg(run_dir)
    fps = int(fps or cfg.get("visualization", {}).get("movie_fps", 24))
    analysis_path = Path(analysis_path) if analysis_path is not None else Path(run_dir) / "results/analysis.h5"
    if not analysis_path.is_file():
        return None
    tracks = []
    with h5py.File(analysis_path, "r") as h5:
        history = h5.get("main_progenitor_histories")
        if history is None:
            return None
        tracks = _read_halo_movie_tracks(history)
    if not tracks:
        return None
    tracks.sort(key=lambda row: np.max(row["mass"]), reverse=True)
    tracks = tracks[:max(1, int(maximum_halos))]
    scale_min = min(float(np.min(row["scale"])) for row in tracks)
    scale_max = max(float(np.max(row["scale"])) for row in tracks)
    if scale_max <= scale_min:
        return None
    frame_count = int(np.clip(max(48, 6 * max(len(row["scale"]) for row in tracks)), 48, 144))
    frame_scales = np.linspace(scale_min, scale_max, frame_count)
    all_masses = np.concatenate([row["mass"] for row in tracks])
    ymin = max(float(np.min(all_masses)) * 0.7, np.finfo(float).tiny)
    ymax = float(np.max(all_masses)) * 1.6
    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "halo_growth_mergers.mp4"
    from core.cosmology import run_proper_time_clock
    from core.data import epoch_contract
    from core.foundation import atomic_json
    clock=run_proper_time_clock(run_dir)
    atomic_json(outdir/'halo_growth_frames.json',{'schema':'history-display-frames-v1',
        'science_acceptance':'NOT_ACCEPTED','frame_spacing':'uniform scale-factor display control; not uniform physical time',
        'frames':[{'frame_index':index,'display_epoch':epoch_contract(float(a),clock=clock),
            'state_role':'display cutoff over saved history points; not a simulation output'} for index,a in enumerate(frame_scales)]})
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for frame_scale in frame_scales:
            fig, axis = plt.subplots(figsize=(10.24, 5.76), dpi=100, layout="constrained")
            for row in tracks:
                visible = row["scale"] <= frame_scale
                if np.count_nonzero(visible) >= 1:
                    axis.plot(row["scale"][visible], row["mass"][visible], marker="o", markersize=2.4,
                              linewidth=1.7, label="halo %d" % row["id"])
                for event_scale, main_mass, ratio in row["mergers"]:
                    if event_scale <= frame_scale:
                        axis.scatter(event_scale, main_mass, marker="*", s=35 + 140 * min(ratio, 1.0),
                                     color="#f97316" if ratio >= 0.25 else "#eab308",
                                     edgecolors="#7c2d12", linewidths=0.4, zorder=5)
            redshift = 1.0 / frame_scale - 1.0
            axis.set_yscale("log")
            axis.set_xlim(scale_min, scale_max)
            axis.set_ylim(ymin, ymax)
            axis.set_xlabel("scale factor a")
            axis.set_ylabel(r"main-progenitor $M_{vir}$ [$M_\odot/h$]")
            axis.set_title("Halo growth and merger history | z=%.4g" % redshift)
            axis.grid(alpha=0.2, which="both")
            if len(tracks) <= 8:
                axis.legend(fontsize=7, ncol=2)
            axis.text(0.99, 0.03, "★ major merger ≥ 1:4", ha="right", va="bottom",
                      transform=axis.transAxes, fontsize=8, color="#9a3412")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()
    return output


def _snapshot_redshift(run_dir, snap):
    header, _ = read_snapshot_header(run_dir, snap)
    scale = _scalar_float(header.get("Time", np.nan), "snapshot Header/Time")
    if np.isfinite(scale) and scale > 0:
        return 1.0 / scale - 1.0, header
    redshift = float(header.get("Redshift", np.nan))
    if np.isfinite(redshift):
        return redshift, header
    raise ValueError("Snapshot %03d has no valid scale factor or redshift" % snap)


def _match_snapshot_redshifts(run_a, run_b, tolerance):
    records_a = [(snap, *_snapshot_redshift(run_a, snap)) for snap in snapshot_ids(run_a)]
    records_b = [(snap, *_snapshot_redshift(run_b, snap)) for snap in snapshot_ids(run_b)]
    remaining_b = {snap: redshift for snap, redshift, _ in records_b}
    pairs, unmatched_a = [], []
    for snap_a, z_a, header_a in sorted(records_a, key=lambda row: row[1]):
        choices = [(abs(z_a - z_b), snap_b, z_b) for snap_b, z_b in remaining_b.items()]
        if not choices:
            unmatched_a.append({"snapshot": snap_a, "redshift": z_a, "reason": "no unmatched run-B snapshot remains"})
            continue
        delta, snap_b, z_b = min(choices)
        allowed = tolerance * max(1.0, abs(z_a), abs(z_b))
        if delta > allowed:
            unmatched_a.append({"snapshot": snap_a, "redshift": z_a, "reason": "no run-B snapshot at matching redshift"})
            continue
        header_b = next(header for candidate, redshift, header in records_b if candidate == snap_b)
        pairs.append({
            "snapshot_a": snap_a, "snapshot_b": snap_b,
            "redshift_a": z_a, "redshift_b": z_b,
            "redshift_delta": delta,
            "header_a": header_a, "header_b": header_b,
        })
        del remaining_b[snap_b]
    unmatched_b = [
        {"snapshot": snap, "redshift": redshift, "reason": "no run-A snapshot at matching redshift"}
        for snap, redshift in sorted(remaining_b.items())
    ]
    return pairs, unmatched_a, unmatched_b


def _log_ratio_map(image_a, image_b):
    """Return log10(B/A) only where both projected fields have support."""
    image_a = np.asarray(image_a, dtype=float)
    image_b = np.asarray(image_b, dtype=float)
    if image_a.shape != image_b.shape:
        raise ValueError("Comparison maps have different pixel dimensions")
    valid = np.isfinite(image_a) & np.isfinite(image_b)
    ratio = np.full(image_a.shape, np.nan, dtype=float)
    ratio[valid] = image_b[valid] - image_a[valid]
    return ratio, valid


def _snapshot_number(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return int(str(value).split("_")[-1])


def _load_matched_halo_overlay_data(run_b, outdir):
    candidates = [Path(outdir).parent / "analysis_comparison.h5", Path(run_b).parent / "comparison/analysis_comparison.h5"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {}, None, "analysis_comparison.h5 is unavailable; run the paired analysis before making halo overlays"
    pairs = {}
    try:
        with h5py.File(path, "r") as f:
            if "pairs" not in f:
                return {}, str(path), "analysis_comparison.h5 has no paired snapshot groups"
            for pair_name in sorted(f["pairs"]):
                group = f["pairs"][pair_name]
                try:
                    key = (_snapshot_number(group.attrs["snapshot_a"]), _snapshot_number(group.attrs["snapshot_b"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if "matched_halos" not in group:
                    pairs[key] = {"status": "SKIPPED", "reason": "no matched FOF hosts were exported for this snapshot pair"}
                    continue
                matched = group["matched_halos"]
                required = ["position_a_mpc_h", "position_b_mpc_h", "position_delta_b_minus_a_mpc_h", "mass_a_msun_h", "mass_b_msun_h"]
                missing = [name for name in required if name not in matched]
                if missing:
                    pairs[key] = {"status": "SKIPPED", "reason": "matched-host output lacks overlay fields: " + ", ".join(missing)}
                    continue
                arrays = {name: np.asarray(matched[name][:]) for name in required}
                count = len(arrays["mass_a_msun_h"])
                if any(len(value) != count for value in arrays.values()) or arrays["position_a_mpc_h"].shape != (count, 3) or arrays["position_b_mpc_h"].shape != (count, 3) or arrays["position_delta_b_minus_a_mpc_h"].shape != (count, 3):
                    pairs[key] = {"status": "SKIPPED", "reason": "matched-host overlay arrays have inconsistent shapes"}
                    continue
                for source, target in [
                    ("r200_critical_a_mpc_h", "r200_a"), ("r200_critical_b_mpc_h", "r200_b"),
                ]:
                    arrays[target] = np.asarray(matched[source][:], dtype=float) if source in matched else np.full(count, np.nan)
                    if len(arrays[target]) != count:
                        raise ValueError("matched-host R200c array does not align with positions")
                pairs[key] = {"status": "COMPLETE", "data": arrays, "matched_count": int(count)}
    except (OSError, KeyError, ValueError) as exc:
        return {}, str(path), "could not read matched-host overlay data: " + str(exc)
    return pairs, str(path), None


def _draw_periodic_arrow(ax, start, delta, box, color="#009E73", linewidth=0.8, alpha=0.8):
    """Draw the shortest periodic displacement as one or more boundary-safe segments."""
    position = np.mod(np.asarray(start, dtype=float), float(box))
    remaining = np.asarray(delta, dtype=float).copy()
    if position.shape != (2,) or remaining.shape != (2,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(remaining)):
        return
    for _ in range(4):
        crossing_time = 1.0
        crossing_axes = []
        for axis in range(2):
            if remaining[axis] > 0:
                candidate = (box - position[axis]) / remaining[axis]
            elif remaining[axis] < 0:
                candidate = (0.0 - position[axis]) / remaining[axis]
            else:
                continue
            if -1.0e-12 <= candidate < crossing_time - 1.0e-12:
                crossing_time = max(0.0, float(candidate))
                crossing_axes = [axis]
            elif abs(candidate - crossing_time) <= 1.0e-12 and candidate < 1.0:
                crossing_axes.append(axis)
        if crossing_time >= 1.0:
            endpoint = position + remaining
            ax.annotate("", xy=endpoint, xytext=position,
                        arrowprops={"arrowstyle": "->", "color": color, "linewidth": linewidth, "alpha": alpha},
                        annotation_clip=True)
            return
        endpoint = position + crossing_time * remaining
        if np.linalg.norm(endpoint - position) > 1.0e-12:
            ax.plot([position[0], endpoint[0]], [position[1], endpoint[1]], color=color, linewidth=linewidth, alpha=alpha)
        remaining *= 1.0 - crossing_time
        position = endpoint
        for axis in crossing_axes:
            position[axis] = 0.0 if delta[axis] > 0 else box


def _draw_matched_halo_overlay(axes, data, projection, box, maximum_halos,
                               slice_center_mpc_h=None, slice_thickness_mpc_h=None):
    projection_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[projection]
    mass_a = np.asarray(data["mass_a_msun_h"], dtype=float)
    mass_b = np.asarray(data["mass_b_msun_h"], dtype=float)
    positions_a = np.asarray(data["position_a_mpc_h"], dtype=float)
    positions_b = np.asarray(data["position_b_mpc_h"], dtype=float)
    delta = np.asarray(data["position_delta_b_minus_a_mpc_h"], dtype=float)
    good = np.isfinite(mass_a) & (mass_a > 0) & np.isfinite(mass_b) & (mass_b > 0)
    good &= np.all(np.isfinite(positions_a), axis=1) & np.all(np.isfinite(positions_b), axis=1) & np.all(np.isfinite(delta), axis=1)
    if slice_thickness_mpc_h is not None:
        los_axis = {"xy": 2, "xz": 1, "yz": 0}[projection]
        for positions in (positions_a, positions_b):
            distance = np.mod(positions[:, los_axis], box) - float(slice_center_mpc_h)
            distance -= box * np.round(distance / box)
            good &= np.abs(distance) <= float(slice_thickness_mpc_h) / 2.0
    candidates = np.flatnonzero(good)
    if len(candidates) > maximum_halos:
        order = np.argsort(np.sqrt(mass_a[candidates] * mass_b[candidates]), kind="stable")[::-1]
        candidates = candidates[order[:maximum_halos]]
    radius_a, radius_b = data["r200_a"], data["r200_b"]
    count_radii = 0
    for index in candidates:
        center_a = np.mod(positions_a[index, list(projection_axes)], box)
        center_b = np.mod(positions_b[index, list(projection_axes)], box)
        displacement = delta[index, list(projection_axes)]
        ax_a, ax_b, ax_diff = axes
        ax_a.plot(center_a[0], center_a[1], marker="o", markersize=3.5, markerfacecolor="none", markeredgecolor="#0072B2", linewidth=0)
        ax_b.plot(center_b[0], center_b[1], marker="x", markersize=3.5, color="#D55E00", linewidth=0)
        ax_diff.plot(center_a[0], center_a[1], marker="o", markersize=3.5, markerfacecolor="none", markeredgecolor="#0072B2", linewidth=0)
        ax_diff.plot(center_b[0], center_b[1], marker="x", markersize=3.5, color="#D55E00", linewidth=0)
        for ax in axes:
            _draw_periodic_arrow(ax, center_a, displacement, box)
        has_radius = False
        if np.isfinite(radius_a[index]) and radius_a[index] > 0:
            ax_a.add_patch(Circle(center_a, radius_a[index], fill=False, edgecolor="#0072B2", linewidth=0.85, alpha=0.85))
            ax_diff.add_patch(Circle(center_a, radius_a[index], fill=False, edgecolor="#0072B2", linewidth=0.85, alpha=0.85))
            has_radius = True
        if np.isfinite(radius_b[index]) and radius_b[index] > 0:
            ax_b.add_patch(Circle(center_b, radius_b[index], fill=False, edgecolor="#D55E00", linewidth=0.85, alpha=0.85))
            ax_diff.add_patch(Circle(center_b, radius_b[index], fill=False, edgecolor="#D55E00", linewidth=0.85, alpha=0.85))
            has_radius = True
        count_radii += int(has_radius)
    axes[2].legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#0072B2", label="Run A center / R200c"),
        Line2D([], [], marker="x", linestyle="none", color="#D55E00", label="Run B center / R200c"),
        Line2D([], [], color="#009E73", label="periodic center displacement"),
    ], loc="upper right", fontsize=7)
    return int(len(candidates)), int(count_radii)


def make_comparison_maps(run_a, run_b, outdir=None, fields=None, projections=None,
                         redshift_tolerance=1.0e-6, slice_thickness_mpc_h=None,
                         slice_center_mpc_h=None):
    """Own both native inputs and publish one immutable comparison-map bundle."""
    import uuid
    from contextlib import ExitStack
    from core.native import native_run_lock
    from core.data import coordinate_contract
    from core.foundation import product_provenance, verify_product_provenance, atomic_json, publish_directory, file_sha256 as digest
    roots=[Path(run_a).expanduser().resolve(),Path(run_b).expanduser().resolve()]
    parent=Path(outdir).expanduser().resolve() if outdir else roots[1].parent/'comparison_maps'
    with ExitStack() as owners:
        for root in sorted(set(roots)):owners.enter_context(native_run_lock(root,operation='comparison-map export'))
        _,overlay_source,_=_load_matched_halo_overlay_data(roots[1],parent)
        contracts=[]
        for root in roots:
            sources=sorted((root/'data').rglob('*.hdf5'))
            if overlay_source and Path(overlay_source).is_file():sources.append(Path(overlay_source))
            contracts.append(product_provenance(root,algorithm=__file__,source_files=sources,
                helper_sources=[Path(__file__).with_name('analysis.py')],
                selection={'fields':fields,'projections':projections,'redshift_tolerance':redshift_tolerance,
                    'slice_thickness_mpc_h':slice_thickness_mpc_h,'slice_center_mpc_h':slice_center_mpc_h},
                coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
                    periodic_wrapping='equal periodic boxes; declared equal line-of-sight slab',velocity_convention='not applicable to scalar comparison maps')))
        parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.comparison-maps-',dir=parent));destination=parent/('comparison_'+uuid.uuid4().hex)
        try:
            _write_comparison_maps(*roots,outdir=stage,fields=fields,projections=projections,
                redshift_tolerance=redshift_tolerance,slice_thickness_mpc_h=slice_thickness_mpc_h,
                slice_center_mpc_h=slice_center_mpc_h,_overlay_lookup_directory=parent)
            for root,contract in zip(roots,contracts):verify_product_provenance(root,contract)
            atomic_json(stage/'comparison_contract.json',{'schema':'comparison-map-publication-v1',
                'science_acceptance':'NOT_ACCEPTED','validation_state':'VALIDATION_PENDING','provenance_by_run':contracts,
                'artifacts':[{'path':path.relative_to(stage).as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size}
                    for path in sorted(stage.rglob('*')) if path.is_file()]})
            publish_directory(stage,destination)
            return destination
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'failure':{'type':type(exc).__name__,'message':str(exc)}})
            raise


def _write_comparison_maps(run_a, run_b, outdir=None, fields=None, projections=None,
                         redshift_tolerance=1.0e-6, slice_thickness_mpc_h=None,
                         slice_center_mpc_h=None, *, _overlay_lookup_directory=None):
    """Make matched-redshift, mass-weighted projected-field comparisons."""
    outdir = Path(outdir or (Path(run_b).parent / "comparison_maps"))
    outdir.mkdir(parents=True, exist_ok=True)
    run_a, run_b = Path(run_a), Path(run_b)
    cfg_a, cfg_b = load_cfg(run_a), load_cfg(run_b)
    box_a = float(cfg_a["simulation"]["box_mpc_h"])
    box_b = float(cfg_b["simulation"]["box_mpc_h"])
    grid_a = int(cfg_a["analysis"]["density_grid"])
    grid_b = int(cfg_b["analysis"]["density_grid"])
    visualization = cfg_a.get("visualization", {})
    density_cmap = visualization.get("density_cmap", semantic_cmap("density"))
    temperature_cmap = visualization.get("temperature_cmap", semantic_cmap("temperature"))
    difference_cmap = visualization.get("difference_cmap", semantic_cmap("difference"))
    if not np.isclose(box_a, box_b, rtol=1.0e-10, atol=0.0):
        raise ValueError("Comparison maps require equal configured box sizes")
    if not np.isclose(grid_a, grid_b, rtol=0.0, atol=0.0):
        raise ValueError("Comparison maps require the same density_grid in both run configs")
    slice_a = _resolve_slice_settings(cfg_a, box_a, slice_thickness_mpc_h, slice_center_mpc_h)
    slice_b = _resolve_slice_settings(cfg_b, box_b, slice_thickness_mpc_h, slice_center_mpc_h)
    if slice_a != slice_b:
        raise ValueError("Comparison maps require identical slice thickness and center in both runs")
    slice_thickness_mpc_h, slice_center_mpc_h = slice_a
    slice_suffix = _slice_tag(slice_thickness_mpc_h, slice_center_mpc_h)
    if not np.isfinite(redshift_tolerance) or redshift_tolerance < 0:
        raise ValueError("redshift_tolerance must be finite and nonnegative")
    overlay_limit_a = int(cfg_a.get("visualization", {}).get("max_matched_halo_overlays", 1000))
    overlay_limit_b = int(cfg_b.get("visualization", {}).get("max_matched_halo_overlays", 1000))
    if overlay_limit_a <= 0 or overlay_limit_b <= 0:
        raise ValueError("visualization.max_matched_halo_overlays must be positive")
    overlay_limit = min(overlay_limit_a, overlay_limit_b)
    fields = ["dm"] if fields is None else list(dict.fromkeys(fields))
    projections = ["xy"] if projections is None else list(dict.fromkeys(projections))
    pairs, unmatched_a, unmatched_b = _match_snapshot_redshifts(run_a, run_b, redshift_tolerance)
    matched_overlay_data, matched_overlay_source, matched_overlay_reason = _load_matched_halo_overlay_data(run_b, _overlay_lookup_directory or outdir)
    summary = {
        "run_a": str(run_a), "run_b": str(run_b), "box_mpc_h": box_a,
        "density_grid": grid_a, "redshift_tolerance": redshift_tolerance,
        "slice": {
            "status": "FULL_BOX_PROJECTION" if slice_thickness_mpc_h is None else "PERIODIC_CENTERED_SLAB",
            "thickness_mpc_h": slice_thickness_mpc_h,
            "center_mpc_h": slice_center_mpc_h,
        },
        "snapshot_pairing": "nearest unique redshift within relative tolerance",
        "unmatched_run_a": unmatched_a, "unmatched_run_b": unmatched_b,
        "pairs": [], "maps": {}, "matched_halo_overlays": {},
        "matched_halo_overlay_source": matched_overlay_source,
        "max_matched_halo_overlays_per_pair": overlay_limit,
        "random_seed_a": cfg_a.get("simulation", {}).get("random_seed"),
        "random_seed_b": cfg_b.get("simulation", {}).get("random_seed"),
    }
    summary["same_random_seed"] = summary["random_seed_a"] == summary["random_seed_b"]
    if not summary["same_random_seed"]:
        summary["phase_comparison_warning"] = "Runs use different random seeds; density differences include realization variance."
    for pair in pairs:
        for label, run_dir, snap in [
            ("a", run_a, pair["snapshot_a"]), ("b", run_b, pair["snapshot_b"]),
        ]:
            header_box = float(pair["header_" + label].get("BoxSize", box_a * 1000.0)) / 1000.0
            if not np.isclose(header_box, box_a, rtol=1.0e-8, atol=0.0):
                raise ValueError("Snapshot %03d in run %s disagrees with its configured box size" % (snap, run_dir))
    summary["pairs"] = [
        {key: pair[key] for key in ["snapshot_a", "snapshot_b", "redshift_a", "redshift_b", "redshift_delta"]}
        for pair in pairs
    ]

    for field in fields:
        for projection in projections:
            key = field + "_" + projection + slice_suffix
            lows, highs, difference_limits = [], [], []
            available_pairs = []
            skipped_pairs = []
            temperature_source_pairs = []
            for pair in pairs:
                image_a = _projected_density(
                    run_a, pair["snapshot_a"], field, projection,
                    slice_thickness_mpc_h, slice_center_mpc_h,
                )
                image_b = _projected_density(
                    run_b, pair["snapshot_b"], field, projection,
                    slice_thickness_mpc_h, slice_center_mpc_h,
                )
                if image_a is None or image_b is None:
                    missing = []
                    if image_a is None:
                        missing.append("run A has no particles for the requested field")
                    if image_b is None:
                        missing.append("run B has no particles for the requested field")
                    skipped_pairs.append({
                        "snapshot_a": pair["snapshot_a"], "snapshot_b": pair["snapshot_b"],
                        "redshift_a": pair["redshift_a"], "redshift_b": pair["redshift_b"],
                        "status": "SKIPPED", "reason": "; ".join(missing),
                    })
                    continue
                if field == "gas_temperature" and image_a[4] != image_b[4]:
                    skipped_pairs.append({
                        "snapshot_a": pair["snapshot_a"], "snapshot_b": pair["snapshot_b"],
                        "redshift_a": pair["redshift_a"], "redshift_b": pair["redshift_b"],
                        "status": "SKIPPED", "reason": "gas-temperature representations differ between runs",
                        "temperature_source_a": image_a[4], "temperature_source_b": image_b[4],
                    })
                    continue
                if field == "gas_temperature":
                    temperature_source_pairs.append({
                        "snapshot_a": pair["snapshot_a"], "snapshot_b": pair["snapshot_b"],
                        "source_a": image_a[4], "source_b": image_b[4],
                    })
                if image_a[0].shape != image_b[0].shape or not np.isclose(image_a[1], image_b[1], rtol=1.0e-10, atol=0.0):
                    raise ValueError("Matched comparison maps do not share a pixel grid and physical box")
                finite_a = image_a[0][np.isfinite(image_a[0])]
                finite_b = image_b[0][np.isfinite(image_b[0])]
                if not finite_a.size or not finite_b.size:
                    continue
                lows.append(min(float(np.percentile(finite_a, 0.5)), float(np.percentile(finite_b, 0.5))))
                highs.append(max(float(np.percentile(finite_a, 99.95)), float(np.percentile(finite_b, 99.95))))
                ratio, valid = _log_ratio_map(image_a[0], image_b[0])
                difference_limits.append(float(np.percentile(np.abs(ratio[valid]), 99.5)) if np.any(valid) else 0.0)
                available_pairs.append(pair)

            if field == "gas_temperature":
                source_pairs = {
                    (item["source_a"], item["source_b"])
                    for item in temperature_source_pairs
                }
                if len(source_pairs) > 1:
                    reason = "temperature representation changes across paired snapshots; one fixed-scale comparison would mix unlike inputs"
                    summary["maps"][key] = {
                        "status": "UNAVAILABLE", "reason": reason,
                        "temperature_source_by_pair": temperature_source_pairs,
                        "skipped_pairs": skipped_pairs,
                    }
                    summary["matched_halo_overlays"][key] = {
                        "status": "UNAVAILABLE", "reason": reason, "skipped_pairs": [],
                    }
                    continue

            if not available_pairs:
                summary["maps"][key] = {
                    "status": "UNAVAILABLE", "reason": "no matched redshifts contain the requested field in both runs",
                    "skipped_pairs": skipped_pairs,
                }
                summary["matched_halo_overlays"][key] = {
                    "status": "UNAVAILABLE", "reason": "no matched redshifts contain the requested field in both runs",
                    "skipped_pairs": skipped_pairs,
                }
                continue
            vmin, vmax = min(lows), max(highs)
            if vmax <= vmin:
                vmax = np.nextafter(vmin, np.inf)
            dlim = max(max(difference_limits), 1.0e-6)
            written, common_cells = [], []
            overlay_written, overlay_counts, radius_counts, overlay_skipped = [], [], [], []
            for index, pair in enumerate(available_pairs):
                image_a = _projected_density(
                    run_a, pair["snapshot_a"], field, projection,
                    slice_thickness_mpc_h, slice_center_mpc_h,
                )
                image_b = _projected_density(
                    run_b, pair["snapshot_b"], field, projection,
                    slice_thickness_mpc_h, slice_center_mpc_h,
                )
                ratio, valid = _log_ratio_map(image_a[0], image_b[0])
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
                extent = [0, box_a, 0, box_a]
                field_cmap = temperature_cmap if field == "gas_temperature" else density_cmap
                axes[0].imshow(image_a[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=field_cmap, interpolation="nearest")
                map_b = axes[1].imshow(image_b[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=field_cmap, interpolation="nearest")
                ratio_map = axes[2].imshow(ratio, origin="lower", extent=extent, vmin=-dlim, vmax=dlim, cmap=difference_cmap, interpolation="nearest")
                for axis in axes:
                    axis.grid(False)
                axes[0].set_title("%s | snapshot %03d | z=%.5g" % (run_a.name, pair["snapshot_a"], pair["redshift_a"]))
                axes[1].set_title("%s | snapshot %03d | z=%.5g" % (run_b.name, pair["snapshot_b"], pair["redshift_b"]))
                axes[2].set_title("log10(B/A); common nonzero support")
                for ax in axes:
                    ax.set_xlabel(projection[0] + " [Mpc/h]")
                    ax.set_ylabel(projection[1] + " [Mpc/h]")
                field_unit = (
                    r"$\log_{10}(\langle T\rangle_M)\ [\mathrm{K}]$"
                    if field == "gas_temperature"
                    else r"$\log_{10}\Sigma\ [hM_\odot/\mathrm{Mpc}^2]$"
                )
                fig.colorbar(map_b, ax=axes[:2], label=field_unit)
                fig.colorbar(ratio_map, ax=axes[2], label="log10(B/A)")
                slice_label = (
                    "" if slice_thickness_mpc_h is None else
                    " | LOS slab %.4g Mpc/h centered at %.4g Mpc/h"
                    % (slice_thickness_mpc_h, slice_center_mpc_h)
                )
                fig.suptitle("%s %s projection comparison%s" % (
                    field.replace("_", " ").title(), projection.upper(), slice_label,
                ))
                filename = "compare_%s_%s%s_pair_%04d.png" % (field, projection, slice_suffix, index)
                from core.data import write_dataset_contract,run_epoch_contract,coordinate_contract
                units='K' if field=='gas_temperature' else 'h Msun / Mpc^2'
                epochs=[]
                for root,snapshot in [(run_a,int(pair['snapshot_a'])),(run_b,int(pair['snapshot_b']))]:
                    header,_=read_snapshot_header(root,snapshot)
                    epochs.append(run_epoch_contract(root,snapshot,_hdf5_scalar_attr(header['Time'],'Time')))
                coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
                    periodic_wrapping='equal periodic projected boxes; identical declared slab',velocity_convention='not applicable to scalar comparison maps')
                with h5py.File((outdir/filename).with_suffix('.hdf5'),'w') as numeric:
                    numeric.attrs['science_acceptance']='NOT_ACCEPTED'
                    numeric.attrs['projection']=projection
                    numeric.attrs['pixel_axes_json']=json.dumps({'rows':image_a[3][1],'columns':image_a[3][0],
                        'pixel_size_mpc_h':box_a/grid_a,'centres':'(index+0.5)*box/grid'})
                    for name,values,unit,epoch in [('measured_a',image_a[6],units,epochs[0]),
                        ('measured_b',image_b[6],units,epochs[1]),('log10_ratio_b_over_a',ratio,'dimensionless',{'run_a':epochs[0],'run_b':epochs[1]}),
                        ('common_positive_support',valid,'boolean support mask',{'run_a':epochs[0],'run_b':epochs[1]})]:
                        dataset=numeric.create_dataset(name,data=values,chunks=True,fletcher32=True)
                        write_dataset_contract(dataset,units=unit,epoch=epoch,coordinates=coordinates,
                            provenance={'bundle_manifest':'comparison_contract.json','run_role':name},
                            quantity='saved comparison '+name,
                            missing_policy='ratio NaN outside common positive support; measured zero-density pixels retained')
                export_figure(fig, outdir / filename, metadata={
                    "snapshot_a": int(pair["snapshot_a"]), "snapshot_b": int(pair["snapshot_b"]),
                    "redshift_a": float(pair["redshift_a"]), "redshift_b": float(pair["redshift_b"]),
                    "field": field, "projection": projection, "normalization": "shared across both runs",
                    "ratio": "log10(B/A)",
                    "numeric_asset":Path(filename).with_suffix('.hdf5').name,'epoch_a':epochs[0],'epoch_b':epochs[1],
                }, dpi=int(visualization["figure_dpi"]))
                written.append(filename)
                common_cells.append(int(np.count_nonzero(valid)))

                pair_key = (int(pair["snapshot_a"]), int(pair["snapshot_b"]))
                overlay_record = matched_overlay_data.get(pair_key)
                if overlay_record is None:
                    overlay_skipped.append({
                        "snapshot_a": pair_key[0], "snapshot_b": pair_key[1], "status": "SKIPPED",
                        "reason": matched_overlay_reason or "analysis comparison has no matched-host record for this snapshot pair",
                    })
                    continue
                if overlay_record.get("status") != "COMPLETE" or not overlay_record.get("matched_count", 0):
                    overlay_skipped.append({
                        "snapshot_a": pair_key[0], "snapshot_b": pair_key[1], "status": "SKIPPED",
                        "reason": overlay_record.get("reason", "no matched hosts were exported for this snapshot pair"),
                    })
                    continue

                overlay_fig, overlay_axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
                overlay_axes[0].imshow(image_a[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=field_cmap, interpolation="nearest")
                overlay_map_b = overlay_axes[1].imshow(image_b[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=field_cmap, interpolation="nearest")
                overlay_ratio = overlay_axes[2].imshow(ratio, origin="lower", extent=extent, vmin=-dlim, vmax=dlim, cmap=difference_cmap, interpolation="nearest")
                for axis in overlay_axes:
                    axis.grid(False)
                shown, radius_count = _draw_matched_halo_overlay(
                    overlay_axes, overlay_record["data"], projection, box_a, overlay_limit,
                    slice_center_mpc_h, slice_thickness_mpc_h,
                )
                overlay_axes[0].set_title("Run A | matched FOF host centers")
                overlay_axes[1].set_title("Run B | matched FOF host centers")
                overlay_axes[2].set_title("log10(B/A) with paired centers and shifts")
                for ax in overlay_axes:
                    ax.set_xlabel(projection[0] + " [Mpc/h]")
                    ax.set_ylabel(projection[1] + " [Mpc/h]")
                overlay_fig.colorbar(overlay_map_b, ax=overlay_axes[:2], label=field_unit)
                overlay_fig.colorbar(overlay_ratio, ax=overlay_axes[2], label="log10(B/A)")
                overlay_fig.suptitle("%s %s map with matched halos%s | z=%.5g | %d shown of %d"
                                     % (field.replace("_", " ").title(), projection.upper(), slice_label,
                                        pair["redshift_a"], shown, overlay_record["matched_count"]))
                overlay_filename = "matched_halo_overlay_%s_%s%s_pair_%04d.png" % (field, projection, slice_suffix, index)
                export_figure(overlay_fig, outdir / overlay_filename, metadata={
                    "snapshot_a": int(pair["snapshot_a"]), "snapshot_b": int(pair["snapshot_b"]),
                    "redshift": float(pair["redshift_a"]), "field": field, "projection": projection,
                    "ratio": "log10(B/A)", "matched_halo_overlay_count": int(shown),
                }, dpi=int(visualization["figure_dpi"]))
                overlay_written.append(overlay_filename)
                overlay_counts.append(shown)
                radius_counts.append(radius_count)
            summary["maps"][key] = {
                "status": "COMPLETE", "figure_files": written,
                "vmin": vmin, "vmax": vmax, "log10_ratio_limit": dlim,
                "common_support_pixel_counts": common_cells,
                "skipped_pairs": skipped_pairs,
                "ratio_mask": "pixels without positive finite support in both runs are blank",
                "fixed_color_scale_across_matched_redshifts": True,
            }
            if field == "gas_temperature":
                summary["maps"][key]["temperature_source_by_pair"] = temperature_source_pairs
            summary["matched_halo_overlays"][key] = {
                "status": "COMPLETE" if overlay_written else "UNAVAILABLE",
                "reason": None if overlay_written else matched_overlay_reason or "no matched-host records contain usable positions",
                "figure_files": overlay_written,
                "matched_halo_counts": overlay_counts,
                "r200c_circle_counts": radius_counts,
                "maximum_halos_per_pair": overlay_limit,
                "r200c_status": "AVAILABLE" if any(radius_counts) else "UNAVAILABLE_OR_NO_VALID_RADII",
                "marker_policy": "all arrows use shortest periodic displacement; circles use Group_R_Crit200 where positive",
                "skipped_pairs": overlay_skipped,
            }

    summary_path = outdir / "comparison_maps_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from figures import review_figure_folder
    figure_review = review_figure_folder(outdir)
    summary["figure_review"] = {
        "status": figure_review["status"],
        "figure_count": figure_review["figure_count"],
        "gallery_pdf": figure_review.get("gallery_pdf"),
        "report_json": "figure_review.json",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outdir


@native_run_owner
def make_xdmf(run_dir):
    """Publish a portable, hash-bound XDMF bundle without replacing prior exports."""
    import uuid
    from analysis import validate_gadget_snapshot_set
    from core.data import coordinate_contract, run_epoch_contract
    from halos import _snapshot_native_units
    from core.foundation import atomic_json, file_sha256 as digest, product_provenance, publish_directory, verify_product_provenance
    root=Path(run_dir).expanduser().resolve()
    snapshots={snap:find_hdf5_files(root/'data','snapshot_%03d'%snap) for snap in snapshot_ids(root)}
    snapshots={snap:files for snap,files in snapshots.items() if files}
    if not snapshots:raise ValueError('XDMF export requires saved native snapshots')
    epochs={}
    for snap,files in snapshots.items():
        validate_gadget_snapshot_set(files,snap)
        # Geometry remains native. Reject absent/contradictory conversion
        # parameters rather than labelling arbitrary code lengths as kpc/h.
        length,_,_=_snapshot_native_units(root,snap)
        if not np.isclose(length,KPC_CM,rtol=1.e-10,atol=0.):
            raise ValueError('XDMF native geometry requires declared comoving kpc/h units')
        with h5py.File(files[0],'r') as source:
            epochs[str(snap)]=run_epoch_contract(root,snap,_hdf5_scalar_attr(source['Header'].attrs['Time'],'Time'))
    sources=[path for files in snapshots.values() for path in files]
    contract=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('data.py')],
        selection={'product':'portable native XDMF','particle_types':[0,1,4]},source_files=sources,
        coordinates=coordinate_contract(position_units='comoving kpc/h',origin='global simulation Cartesian box origin',
            periodic_wrapping='native wrapped periodic box',velocity_convention='native snapshot velocity; dataset metadata specifies conversion'))
    output=root/'results/3d'
    output.mkdir(parents=True,exist_ok=True)
    required=sum(path.stat().st_size for path in sources)+64*1024**2
    if shutil.disk_usage(output).free<required:
        raise OSError('Portable XDMF export requires input-copy bytes plus 64 MiB publication margin')
    stage=Path(tempfile.mkdtemp(prefix='.xdmf-',dir=output))
    destination=output/('xdmf_'+uuid.uuid4().hex)
    try:
        inputs=stage/'inputs';inputs.mkdir()
        copies={};records=[]
        for index,path in enumerate(sources):
            copied=inputs/('%06d_'%index+path.name)
            shutil.copy2(path,copied)
            checksum=digest(copied)
            source_checksum=next(item['sha256'] for item in contract['input_hashes'] if item['path']==str(path.resolve()))
            if checksum!=source_checksum:raise ValueError('XDMF input changed during copy: '+str(path))
            copies[str(path.resolve())]=copied
            records.append({'source':str(path.resolve()),'path':copied.relative_to(stage).as_posix(),'sha256':checksum})
        made=_write_xdmf_bundle(snapshots,stage,copies)
        verify_product_provenance(root,contract)
        atomic_json(stage/'xdmf_contract.json',{'schema':'portable-xdmf-v1','science_acceptance':'NOT_ACCEPTED',
            'validation_state':'VALIDATION_PENDING','provenance':contract,'epochs':epochs,'inputs':records,
            'time_coordinate':'scale factor (XDMF Time); canonical cosmic time in epochs',
            'geometry':'native comoving kpc/h; no coordinate rescaling',
            'particle_identity':'uint64 native IDs retained in HDF5; no floating-point ID conversion',
            'artifacts':[{'path':path.relative_to(stage).as_posix(),'sha256':digest(path)} for path in made]})
        publish_directory(stage,destination)
        return [destination/path.relative_to(stage) for path in made]
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


def _write_xdmf_bundle(snapshots, outdir, input_copies):
    """Write only into caller-owned staging; geometry and exact IDs stay native."""
    from xml.sax.saxutils import escape
    made = []
    temporal = {0: [], 1: [], 4: []}
    for snap, files in snapshots.items():
        if not files:
            continue
        by_type = {0: [], 1: [], 4: []}
        snapshot_time = None
        for file in files:
            rel = escape(input_copies[str(file.resolve())].relative_to(outdir).as_posix())
            with h5py.File(file, "r") as f:
                a = _hdf5_scalar_attr(f["Header"].attrs["Time"], "Time")
                if snapshot_time is None:
                    snapshot_time = a
                elif not np.isclose(snapshot_time, a, rtol=1.0e-10, atol=1.0e-12):
                    raise ValueError("Snapshot shards have inconsistent scale factors for snapshot %03d" % snap)
                for ptype in [0, 1, 4]:
                    g = "PartType%d" % ptype
                    if g not in f or "Coordinates" not in f[g]:
                        continue
                    n = len(f[g]["Coordinates"])
                    coordinates = f[g]["Coordinates"]
                    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                        raise ValueError("%s/Coordinates must have shape (particles, 3) in %s" % (g, file))
                    attrs = []
                    for name, dataset in f[g].items():
                        if name == "Coordinates" or dataset.ndim not in (1, 2) or len(dataset) != n:
                            continue
                        if dataset.dtype.kind not in "iuf":
                            continue
                        atype = "Vector" if dataset.ndim == 2 and dataset.shape[1] == 3 else "Scalar" if dataset.ndim == 1 else None
                        if atype is None:
                            continue
                        number_type = {"i": "Int", "u": "UInt", "f": "Float"}[dataset.dtype.kind]
                        dims = " ".join(str(x) for x in dataset.shape)
                        attrs.append(
                            '<Attribute Name="%s" AttributeType="%s" Center="Node">'
                            '<DataItem Dimensions="%s" NumberType="%s" Precision="%d" Format="HDF">%s:/%s/%s</DataItem></Attribute>'
                            % (escape(name,{'"':'&quot;'}), atype, dims, number_type, dataset.dtype.itemsize, rel, g, escape(name))
                        )
                    attr_text = '<Information Name="Coordinate units" Value="comoving kpc/h"/><Information Name="Time coordinate" Value="scale factor"/>' + "".join(attrs)
                    grid = (
                        '<Grid Name="snapshot_%03d_type%d_%s" GridType="Uniform">'
                        '<Topology TopologyType="Polyvertex" NumberOfElements="%d"/>'
                        '<Geometry GeometryType="XYZ"><DataItem Dimensions="%d 3" NumberType="Float" Precision="%d" Format="HDF">%s:/%s/Coordinates</DataItem></Geometry>%s</Grid>'
                        % (snap, ptype, escape(file.stem,{'"':'&quot;'}), n, n, coordinates.dtype.itemsize, rel, g, attr_text)
                    )
                    by_type[ptype].append(grid)
        if snapshot_time is None:
            continue
        for ptype, shard_grids in by_type.items():
            if not shard_grids:
                continue
            spatial = (
                '<Grid Name="snapshot_%03d_type%d" GridType="Collection" CollectionType="Spatial">'
                '<Time Value="%.12e"/>%s</Grid>'
                % (snap, ptype, snapshot_time, "".join(shard_grids))
            )
            x = outdir / ("snapshot_%03d_type%d.xdmf" % (snap, ptype))
            x.write_text('<?xml version="1.0" ?><Xdmf Version="3.0"><Domain>' + spatial + '</Domain></Xdmf>', encoding="utf-8")
            made.append(x)
            temporal[ptype].append(spatial)
    for ptype, grids in temporal.items():
        if not grids:
            continue
        x = outdir / ("timeline_type%d.xdmf" % ptype)
        x.write_text('<?xml version="1.0" ?><Xdmf Version="3.0"><Domain><Grid Name="timeline" GridType="Collection" CollectionType="Temporal">' + "".join(grids) + '</Grid></Domain></Xdmf>', encoding="utf-8")
        made.append(x)
    return made


def _rockstar_radius_scale_to_mpc_h(table):
    """Return the header-supported Rockstar radius conversion, or None if unknown."""
    lines = table.get("_unit_lines", ()) if isinstance(table, dict) else ()
    for line in lines:
        normalized = str(line).lower().replace(" ", "")
        if not any(token in normalized for token in ("radius", "radii", "lengths")):
            continue
        if "kpc/h" in normalized:
            return 1.0e-3
        if "mpc/h" in normalized:
            return 1.0
    return None


def _read_catalogue_layer(run_dir, snap, kind, mass_definition="fof"):
    """Read one aligned halo, subhalo, galaxy, or Rockstar layer for 3-D display."""
    if kind == "galaxies":
        path = Path(run_dir) / "results" / ("galaxies_%03d.hdf5" % int(snap))
        if not path.is_file():
            candidates = list((Path(run_dir) / "results").glob(
                "analysis_products_*/galaxies_%03d.hdf5" % int(snap)))
            if not candidates:
                return None
            path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        from core.data import verify_hdf5_dataset_payloads
        verify_hdf5_dataset_payloads(path)
        with h5py.File(path,'r') as contract_source:
            sealed='data_contract_json' in contract_source.attrs
        if sealed:
            from halos import verify_galaxy_catalogue
            verify_galaxy_catalogue(path,run_dir)
        with h5py.File(path, "r") as source:
            required = ("GalaxyID", "Resolved", "Position", "StellarMass", "StellarHalfMassRadius", "FormationRedshift")
            missing = [name for name in required if name not in source]
            if missing:
                raise ValueError("galaxy catalogue is missing " + ", ".join(missing))
            identifiers = np.asarray(source["GalaxyID"][:], dtype=np.uint64)
            resolved = np.asarray(source["Resolved"][:], dtype=bool)
            positions = np.asarray(source["Position"][:], dtype=float)
            masses = np.asarray(source["StellarMass"][:], dtype=float)
            halfmass = np.asarray(source["StellarHalfMassRadius"][:], dtype=float)
            formation_redshift = np.asarray(source["FormationRedshift"][:], dtype=float)
            host_fof_ids = (np.asarray(source["HostFOFID"][:], dtype=np.int64)
                            if "HostFOFID" in source else None)
            if any(len(array) != len(identifiers) for array in (resolved, masses, halfmass, formation_redshift)) or positions.shape != (len(identifiers), 3):
                raise ValueError("galaxy catalogue properties are not aligned")
            rows = np.flatnonzero(resolved)
            aperture_fields = {}
            for field in ("ApertureRadius_Mpc_h", "ApertureStellarMass", "ApertureGasMass",
                          "ApertureGasFraction", "ApertureSFR", "ApertureFormationRedshift"):
                aperture_fields[field] = np.asarray(source[field][:], dtype=float)[rows] if field in source else None
            return {
                "kind": "galaxies", "mass_definition": "stellar", "mass_field": "StellarMass",
                "position_mpc_h": positions[rows], "mass_msun_h": masses[rows],
                "row_index": rows.astype(np.int64), "particle_count": np.asarray(source["NStars"][:], dtype=np.int64)[rows],
                "GalaxyID": identifiers[rows].astype(np.int64),
                "HostFOFID": host_fof_ids[rows] if host_fof_ids is not None else None,
                "stellar_half_mass_radius_mpc_h": halfmass[rows],
                "formation_redshift": formation_redshift[rows],
                "aperture_properties": aperture_fields,
                "radius_scale_to_mpc_h": 1.0,
                "source_path": str(path.resolve()),
            }
    if kind == "rockstar":
        from analysis import rockstar_host_table
        table = rockstar_host_table(run_dir, int(snap))
        if not table:
            return None
        position_names = ("x", "y", "z")
        if any(name not in table for name in position_names):
            raise ValueError("Rockstar catalogue must contain x, y, and z coordinates")
        definition = str(mass_definition).lower()
        mass_name = {
            "fof": "mfof", "m200c": "m200c", "m200b": "m200b", "mvir": "mvir",
        }.get(definition)
        # Rockstar does not write an Mfof column in its standard catalogue.
        # Use its native spherical-overdensity Mvir, and report that exact
        # definition in the UI rather than dropping the independent detection.
        if definition == "fof" and mass_name not in table and "mvir" in table:
            mass_name = "mvir"
            definition = "mvir"
        if mass_name is None:
            raise ValueError("Unsupported Rockstar host mass definition: " + definition)
        if mass_name not in table:
            raise ValueError("Rockstar catalogue lacks the configured %s mass column" % mass_name)
        positions = np.column_stack([np.asarray(table[name], dtype=float) for name in position_names])
        masses = np.asarray(table[mass_name], dtype=float)
        if len(positions) != len(masses):
            raise ValueError("Rockstar positions and masses do not align")
        parent = np.asarray(table["pid"], dtype=np.int64) if "pid" in table else None
        host_mask = parent < 0 if parent is not None else np.ones(len(masses), dtype=bool)
        rows = np.flatnonzero(host_mask)
        result = {
            "kind": "rockstar", "mass_definition": definition, "mass_field": mass_name,
            "position_mpc_h": np.mod(positions[rows], float(load_cfg(run_dir)["simulation"]["box_mpc_h"])),
            "mass_msun_h": masses[rows], "row_index": rows.astype(np.int64),
            "particle_count": np.asarray(table.get("num_p", table.get("np", np.zeros(len(masses)))), dtype=np.int64)[rows],
            "RockstarID": np.asarray(table["id"], dtype=np.int64)[rows] if "id" in table else None,
            "pid": parent[rows] if parent is not None else None,
            "radius_scale_to_mpc_h": _rockstar_radius_scale_to_mpc_h(table),
            "source_table": table,
        }
        for field in ("m200c", "r200c", "m200b", "r200b", "mvir", "rvir", "vmax"):
            result[field] = np.asarray(table[field])[rows] if field in table else None
        return result
    if kind == "halos":
        top, position_name = "Group", "GroupPos"
        mass_choices = {
            "fof": ["GroupMass"],
            "m200c": ["Group_M_Crit200"],
            "m200b": ["Group_M_Mean200"],
            "mvir": ["Group_M_TopHat200"],
        }
        optional = [
            "Group_M_Crit200", "Group_M_Mean200", "Group_M_TopHat200",
            "Group_R_Crit200", "Group_R_Mean200", "GroupVel", "GroupLen",
            "GroupLenType", "GroupNsubs", "GroupFirstSub", "GroupVmax",
        ]
        vector_fields = {"GroupPos", "GroupVel"}
        six_type_fields = {"GroupLenType"}
    elif kind == "subhalos":
        top, position_name = "Subhalo", "SubhaloPos"
        # SUBFIND SubhaloMass is a bound-particle mass, not a spherical-
        # overdensity mass. Keep that definition explicit in exported layers.
        mass_choices = {"bound": ["SubhaloMass"]}
        optional = [
            "SubhaloGroupNr", "SubhaloRankInGr", "SubhaloParentRank",
            "SubhaloVmax", "SubhaloVel", "SubhaloSpin", "SubhaloHalfmassRad",
            "SubhaloLen", "SubhaloLenType", "SubhaloMassType", "SubhaloVelDisp", "SubhaloVmaxRad",
        ]
        vector_fields = {"SubhaloPos", "SubhaloVel", "SubhaloSpin"}
        six_type_fields = {"SubhaloLenType", "SubhaloMassType"}
    else:
        raise ValueError("kind must be halos, subhalos, or rockstar")

    definition = "bound" if kind == "subhalos" else str(mass_definition).lower()
    if definition not in mass_choices:
        raise ValueError(
            "SUBFIND subhalo layers use bound mass"
            if kind == "subhalos" else
            "analysis.hmf_mass_definition must be one of: fof, m200c, m200b, mvir"
        )
    files = group_files(run_dir, int(snap))
    if not files:
        return None
    from halos import _check_file_series, _unit_conversion
    _check_file_series(files, "Header", "Nsubhalos_Total" if kind=="subhalos" else "Ngroups_Total")
    _unit_conversion(files + read_snapshot_header(run_dir, int(snap))[1])

    shards = []
    for path in files:
        with h5py.File(path, "r") as h5:
            expected_rows = int(h5['Header'].attrs['Nsubhalos_ThisFile' if kind=='subhalos' else 'Ngroups_ThisFile'])
            if top not in h5:
                if expected_rows:raise ValueError('Populated native catalogue shard lacks '+top)
                continue
            table = h5[top]
            if position_name not in table:
                if expected_rows or any(dataset.size for dataset in table.values()):
                    raise ValueError("%s catalogue in %s has no %s field" % (top, path, position_name))
                continue
            position = table[position_name]
            if position.ndim != 2 or position.shape[1] != 3:
                raise ValueError("%s/%s must have shape (objects, 3) in %s" % (top, position_name, path))
            count = len(position)
            if count!=expected_rows:
                raise ValueError('Native catalogue position rows disagree with declared shard count')
            if count == 0:
                continue
            shards.append((Path(path), count, tuple(table.keys())))
    if not shards:
        return None

    mass_name = next(
        (name for name in mass_choices[definition] if all(name in names for _, _, names in shards)),
        None,
    )
    if mass_name is None:
        raise ValueError("%s catalogue lacks the configured %s mass field for snapshot %03d" % (top, definition, snap))

    fields = [position_name, mass_name] + [name for name in optional if name not in {position_name, mass_name}]
    result = {name: None for name in fields}
    for name in fields:
        availability = [name in names for _, _, names in shards]
        if name not in {position_name, mass_name} and not any(availability):
            continue
        if not all(availability):
            raise ValueError("%s/%s is missing from one or more non-empty catalogue shards" % (top, name))
        chunks = []
        for path, count, _ in shards:
            with h5py.File(path, "r") as h5:
                dataset = h5[top][name]
                if dataset.ndim == 0 or len(dataset) != count:
                    raise ValueError("%s/%s does not align with %s in %s" % (top, name, position_name, path))
                if name in vector_fields and (dataset.ndim != 2 or dataset.shape[1] != 3):
                    raise ValueError("%s/%s must have shape (objects, 3)" % (top, name))
                if name in six_type_fields and (dataset.ndim != 2 or dataset.shape[1] != 6):
                    raise ValueError("%s/%s must have the six GADGET particle-type columns" % (top, name))
                if name not in vector_fields | six_type_fields and dataset.ndim != 1:
                    raise ValueError("%s/%s must have one scalar per object" % (top, name))
                chunks.append(np.asarray(dataset[:]))
        result[name] = np.concatenate(chunks, axis=0)

    positions = np.asarray(result[position_name], dtype=float)
    masses = np.asarray(result[mass_name], dtype=float).reshape(-1)
    if len(positions) != len(masses):
        raise ValueError("%s positions and masses do not align" % top)
    result.update({
        "kind": kind,
        "mass_definition": definition,
        "mass_field": mass_name,
        "position_mpc_h": positions / 1000.0,
        "mass_msun_h": masses * 1.0e10,
        "row_index": np.arange(len(masses), dtype=np.int64),
    })
    velocity_field='GroupVel' if kind=='halos' else 'SubhaloVel'
    if result.get(velocity_field) is not None:
        from analysis import catalogue_kinematic_metadata
        from core.data import convert_native_catalogue_kinematics
        result['velocity_peculiar_km_s']=convert_native_catalogue_kinematics(result[velocity_field],velocity_field,
            **catalogue_kinematic_metadata(run_dir,snap))
    count_field = "GroupLen" if kind == "halos" else "SubhaloLen"
    type_count_field = "GroupLenType" if kind == "halos" else "SubhaloLenType"
    if result.get(count_field) is not None:
        result["particle_count"] = np.asarray(result[count_field], dtype=np.int64).reshape(-1)
    elif result.get(type_count_field) is not None:
        result["particle_count"] = np.sum(result[type_count_field], axis=1, dtype=np.int64)
    else:
        result["particle_count"] = None
    return result


def _catalogue_mass_limits(run_dir, snaps, kinds, mass_definition):
    """Find one stable logarithmic mass-slider range across the run's catalogs."""
    minimum, maximum = np.inf, -np.inf
    available_kinds = set()
    for snap in snaps:
        for kind in kinds:
            data = _read_catalogue_layer(run_dir, snap, kind, mass_definition)
            if data is None:
                continue
            mass = np.asarray(data["mass_msun_h"], dtype=float)
            good = np.isfinite(mass) & (mass > 0)
            if not np.any(good):
                continue
            minimum = min(minimum, float(np.min(mass[good])))
            maximum = max(maximum, float(np.max(mass[good])))
            available_kinds.add(kind)
    if not available_kinds:
        return None, []
    low, high = np.log10(minimum), np.log10(maximum)
    if high <= low:
        low, high = low - 0.5, high + 0.5
    return (float(low), float(high)), sorted(available_kinds)


def _read_snapshot_fields(files, ptype, fields, max_points, selected_ids=None):
    """Read aligned snapshot fields, sampling rows or matching stable IDs."""
    from core.data import exact_uint64_ids
    fields = list(dict.fromkeys(fields))
    if not fields:
        raise ValueError("At least one snapshot field must be requested")
    group = "PartType%d" % int(ptype)
    if selected_ids is None:
        requested = list(fields)
        if "ParticleIDs" not in requested:
            requested.append("ParticleIDs")
        data = read_particle_fields(files, ptype, requested, max_points=max_points)
        ids = data.get("ParticleIDs")
        if ids is None:
            if max_points is not None:
                raise ValueError("Timeline sampling requires ParticleIDs in the HDF5 snapshots")
            return data
        ids = exact_uint64_ids(ids)
        if ids.ndim != 1 or any(data[name] is not None and len(data[name]) != len(ids) for name in fields):
            raise ValueError("%s fields do not align with ParticleIDs" % group)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("%s contains duplicate ParticleIDs" % group)
        order = np.argsort(ids, kind="stable")
        for name in fields:
            if data[name] is not None:
                data[name] = np.asarray(data[name])[order]
        data["ParticleIDs"] = ids[order]
        return data

    wanted = np.sort(exact_uint64_ids(selected_ids))
    if not len(wanted):
        return {name: None for name in fields} | {"ParticleIDs": wanted}

    records = []
    signatures = []
    shapes = {}
    dtypes = {}
    for path in files:
        with h5py.File(path, "r") as h5:
            if group not in h5:
                continue
            particles = h5[group]
            if "Coordinates" not in particles:
                continue
            count = len(particles["Coordinates"])
            if count == 0:
                continue
            if "ParticleIDs" not in particles or particles["ParticleIDs"].ndim != 1 or len(particles["ParticleIDs"]) != count:
                raise ValueError("Timeline sampling requires aligned ParticleIDs in " + str(path))
            present = tuple(name for name in fields if name in particles)
            signatures.append(present)
            for name in present:
                dataset = particles[name]
                if dataset.ndim == 0 or len(dataset) != count:
                    raise ValueError("%s/%s does not align with Coordinates in %s" % (group, name, path))
                shapes.setdefault(name, dataset.shape[1:])
                dtypes.setdefault(name, dataset.dtype)
            records.append((path, count, present))
    if not records:
        return {name: None for name in fields} | {"ParticleIDs": np.empty(0, dtype=wanted.dtype)}
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Requested fields are not consistently present across non-empty %s snapshot shards" % group)

    available_fields = signatures[0]
    chunks = {name: [] for name in available_fields}
    matched_ids = []
    chunk_size = 1_000_000
    for path, count, present in records:
        with h5py.File(path, "r") as h5:
            particles = h5[group]
            id_dataset = particles["ParticleIDs"]
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                ids = exact_uint64_ids(id_dataset[start:stop])
                slot = np.searchsorted(wanted, ids)
                found = slot < len(wanted)
                found[found] &= wanted[slot[found]] == ids[found]
                rows = np.flatnonzero(found)
                if not len(rows):
                    continue
                matched_ids.append(ids[rows])
                for name in present:
                    chunks[name].append(np.asarray(particles[name][start + rows]))

    if not matched_ids:
        result = {name: None for name in fields}
        for name in available_fields:
            result[name] = np.empty((0,) + shapes[name], dtype=dtypes[name])
        result["ParticleIDs"] = np.empty(0, dtype=wanted.dtype)
        return result
    ids = np.concatenate(matched_ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("%s contains duplicate matched ParticleIDs" % group)
    order = np.argsort(ids, kind="stable")
    result = {name: None for name in fields}
    for name in available_fields:
        values = np.concatenate(chunks[name], axis=0)
        result[name] = values[order]
    result["ParticleIDs"] = ids[order]
    return result


def _read_snapshot_points(files, ptype, max_points, selected_ids=None):
    """Read one reproducible particle sample, optionally matching stable IDs."""
    data = _read_snapshot_fields(files, ptype, ["Coordinates"], max_points, selected_ids)
    xyz = data["Coordinates"]
    if xyz is None:
        return np.empty((0, 3), dtype=np.float32), data.get("ParticleIDs")
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("PartType%d/Coordinates must have shape (particles, 3)" % int(ptype))
    ids = data.get("ParticleIDs")
    if ids is not None and (ids.ndim != 1 or len(ids) != len(xyz)):
        raise ValueError("PartType%d/ParticleIDs do not align with Coordinates" % int(ptype))
    return xyz, ids


def _blender_halo_overlays(run_dir, snapshot, cfg, box_mpc_h, maximum_halos=250):
    """Return bounded catalogue rows for a synchronized Blender frame."""
    mass_definition = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    header, _ = read_snapshot_header(run_dir, int(snapshot))
    snapshot_scale_factor = float(header["Time"])
    snapshot_redshift = 1.0 / snapshot_scale_factor - 1.0
    thresholds = cfg.get("analysis", {}).get("rare_halo_thresholds_msun_h", [])
    try:
        rare_threshold = max(float(value) for value in thresholds if np.isfinite(float(value)) and float(value) > 0)
    except (TypeError, ValueError):
        rare_threshold = None
    output = {}
    for kind in ("halos", "subhalos", "rockstar", "galaxies"):
        if kind == "galaxies" and not bool(cfg.get("visualization", {}).get("blender_galaxy_overlays", True)):
            output[kind] = {"status": "DISABLED", "reason": "visualization.blender_galaxy_overlays is false", "items": []}
            continue
        item_limit = int(cfg.get("visualization", {}).get("blender_max_galaxies", maximum_halos)) if kind == "galaxies" else int(maximum_halos)
        if item_limit < 1:
            raise ValueError("Blender catalogue overlay limits must be positive")
        try:
            data = _read_catalogue_layer(run_dir, int(snapshot), kind, mass_definition)
        except Exception as exc:
            output[kind] = {"status": "UNAVAILABLE", "reason": str(exc), "items": []}
            continue
        if data is None:
            output[kind] = {"status": "UNAVAILABLE", "reason": "no catalogue is available", "items": []}
            continue
        positions = np.asarray(data["position_mpc_h"], dtype=float)
        masses = np.asarray(data["mass_msun_h"], dtype=float)
        if kind == "subhalos":
            # Inspect every subhalo by default. A deliberate display cap never
            # removes rows from the raw scientific CSV/HDF5 catalogue.
            limit = cfg.get("visualization", {}).get("blender_subhalo_limit")
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
                raise ValueError("blender_subhalo_limit must be null (all) or a positive integer")
            item_limit = len(masses) if limit is None else limit
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) != len(masses):
            output[kind] = {"status": "UNAVAILABLE", "reason": "catalogue positions and masses do not align", "items": []}
            continue
        rows = np.flatnonzero(np.isfinite(masses) & (masses > 0) & np.all(np.isfinite(positions), axis=1))
        rows = rows[np.argsort(-masses[rows], kind="stable")[:item_limit]]
        if kind == "rockstar" and data.get("RockstarID") is not None:
            identifiers = np.asarray(data["RockstarID"], dtype=np.int64)
            id_source = "RockstarID"
        elif kind == "galaxies" and data.get("GalaxyID") is not None:
            identifiers = np.asarray(data["GalaxyID"], dtype=np.int64)
            id_source = "GalaxyID (SUBFIND SubhaloID)"
        else:
            identifiers = np.asarray(data.get("row_index", np.arange(len(masses))), dtype=np.int64)
            id_source = "catalogue_row"
        radii = np.full(len(masses), np.nan, dtype=float)
        radius_source = None
        if kind == "halos" and data.get("Group_R_Crit200") is not None:
            radii = np.asarray(data["Group_R_Crit200"], dtype=float) / 1000.0
            radius_source = "GADGET Group_R_Crit200 [kpc/h]"
        elif kind == "rockstar" and data.get("r200c") is not None:
            radius_scale = data.get("radius_scale_to_mpc_h")
            if radius_scale is not None:
                radii = np.asarray(data["r200c"], dtype=float) * float(radius_scale)
                radius_source = "Rockstar R200c with header-declared units"
        items = []
        for row in rows:
            position = np.mod(positions[row], float(box_mpc_h))
            radius = float(radii[row]) if np.isfinite(radii[row]) and radii[row] > 0 else None
            mass = float(masses[row])
            items.append({
                "id": int(identifiers[row]),
                "id_source": id_source,
                "finder": {"halos": "GADGET-4 FOF", "subhalos": "GADGET-4 SUBFIND",
                           "rockstar": "Rockstar 6D"}.get(kind, kind),
                "subhalo": kind == "subhalos",
                "parent_id": (
                    int(data["pid"][row]) if kind == "rockstar" and data.get("pid") is not None
                    and int(data["pid"][row]) >= 0 else
                    int(data["SubhaloGroupNr"][row]) if kind == "subhalos" and data.get("SubhaloGroupNr") is not None else
                    int(data["HostFOFID"][row]) if kind == "galaxies" and data.get("HostFOFID") is not None else None
                ),
                "host_id": (
                    int(data["SubhaloGroupNr"][row]) if kind == "subhalos" and data.get("SubhaloGroupNr") is not None else
                    int(data["HostFOFID"][row]) if kind == "galaxies" and data.get("HostFOFID") is not None else
                    int(identifiers[row]) if kind == "rockstar" else
                    int(row) if kind == "halos" else None
                ),
                "subhalo_count": int(data["GroupNsubs"][row])
                    if kind == "halos" and data.get("GroupNsubs") is not None else None,
                "detection_redshift": float(snapshot_redshift),
                "snapshot": int(snapshot),
                "central": bool(data["SubhaloRankInGr"][row] == 0)
                    if kind == "subhalos" and data.get("SubhaloRankInGr") is not None else None,
                "velocity_km_s": [float(value) for value in data['velocity_peculiar_km_s'][row]]
                    if kind in ('halos','subhalos') and data.get('velocity_peculiar_km_s') is not None else
                    [float(data["source_table"][name][row]) for name in ("vx", "vy", "vz")]
                    if kind == "rockstar" and all(name in data.get("source_table", {}) for name in ("vx", "vy", "vz")) else None,
                "particle_count": int(data["particle_count"][row])
                    if data.get("particle_count") is not None else None,
                "position_mpc_h": [float(value) for value in position],
                "mass_msun_h": mass,
                "mass_definition": str(data.get("mass_definition", mass_definition)),
                "r200c_mpc_h": radius,
                "half_mass_radius_mpc_h": float(data["SubhaloHalfmassRad"][row]) / 1000.0
                    if kind == "subhalos" and data.get("SubhaloHalfmassRad") is not None else None,
                "vmax_km_s": float(data["SubhaloVmax"][row])
                    if kind == "subhalos" and data.get("SubhaloVmax") is not None else None,
                "vmax_radius_mpc_h": float(data["SubhaloVmaxRad"][row]) / 1000.0
                    if kind == "subhalos" and data.get("SubhaloVmaxRad") is not None else None,
                "velocity_dispersion_km_s": float(data["SubhaloVelDisp"][row])
                    if kind == "subhalos" and data.get("SubhaloVelDisp") is not None else None,
                "particle_count_by_type": [int(v) for v in data["SubhaloLenType"][row]]
                    if kind == "subhalos" and data.get("SubhaloLenType") is not None else None,
                "mass_by_type_msun_h": [float(v)*1e10 for v in data["SubhaloMassType"][row]]
                    if kind == "subhalos" and data.get("SubhaloMassType") is not None else None,
                "stellar_half_mass_radius_mpc_h": (
                    float(data["stellar_half_mass_radius_mpc_h"][row])
                    if kind == "galaxies" and np.isfinite(data["stellar_half_mass_radius_mpc_h"][row])
                    and data["stellar_half_mass_radius_mpc_h"][row] > 0 else None
                ),
                "n_stars": int(data["particle_count"][row]) if kind == "galaxies" else None,
                "formation_redshift": (
                    float(data["formation_redshift"][row])
                    if kind == "galaxies" and np.isfinite(data["formation_redshift"][row]) else None
                ),
                "aperture": {
                    name: float(values[row]) if values is not None and np.isfinite(values[row]) else None
                    for name, values in data.get("aperture_properties", {}).items()
                } if kind == "galaxies" else None,
                "rare": bool(kind != "galaxies" and rare_threshold is not None and mass >= rare_threshold),
            })
        output[kind] = {
            "status": "AVAILABLE" if items else "EMPTY",
            "mass_definition": str(data.get("mass_definition", mass_definition)),
            "mass_field": str(data.get("mass_field", "unknown")),
            "count_available": int(len(masses)),
            "count_exported": int(len(items)),
            "display_truncated": len(items) < int(np.sum(np.isfinite(masses) & (masses > 0) & np.all(np.isfinite(positions), axis=1))),
            "maximum_halos": item_limit,
            "identifier_source": id_source,
            "r200c_source": radius_source,
            "rare_mass_threshold_msun_h": rare_threshold,
            "items": items,
        }
    return output


def _snapshot_scale_factor(header):
    """Get a finite scale factor from a GADGET header without guessing units."""
    value = header.get("Time")
    try:
        scale_factor = float(np.asarray(value).reshape(-1)[0]) if value is not None else np.nan
    except (TypeError, ValueError, OverflowError, IndexError):
        scale_factor = np.nan
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        redshift = header.get("Redshift")
        try:
            redshift = float(np.asarray(redshift).reshape(-1)[0])
        except (TypeError, ValueError, OverflowError, IndexError):
            redshift = np.nan
        if not np.isfinite(redshift) or redshift <= -1.0:
            raise ValueError("Every timeline snapshot needs a positive Header/Time or finite Header/Redshift")
        scale_factor = 1.0 / (1.0 + redshift)
    return float(scale_factor)


def _timeline_bracket(scale_factors, value):
    """Return the output index and scale-factor interpolation fraction."""
    scale_factors = np.asarray(scale_factors, dtype=float)
    if scale_factors.ndim != 1 or not len(scale_factors) or np.any(~np.isfinite(scale_factors)):
        raise ValueError("timeline scale factors must be a non-empty finite vector")
    if len(scale_factors) == 1:
        return 0, 0.0
    value = float(np.clip(value, scale_factors[0], scale_factors[-1]))
    if value >= scale_factors[-1]:
        return len(scale_factors) - 1, 0.0
    left = max(int(np.searchsorted(scale_factors, value, side="right")) - 1, 0)
    width = scale_factors[left + 1] - scale_factors[left]
    if width <= 0:
        raise ValueError("timeline scale factors must be strictly increasing")
    return left, float((value - scale_factors[left]) / width)


def _interpolate_periodic_positions(start, end, fraction, box):
    """Interpolate wrapped positions along the shortest periodic displacement."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    box = float(box)
    fraction = float(fraction)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("periodic position arrays must both have shape (particles, 3)")
    if not np.isfinite(box) or box <= 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("periodic interpolation needs a positive box and fraction in [0, 1]")
    displacement = end - start
    displacement -= box * np.round(displacement / box)
    return np.mod(start + fraction * displacement, box)


def _velocity_grid_from_particles(files, ptype, selected_ids, cap, box_mpc_h, grid_size):
    """Deposit sampled particle-mass-weighted velocities onto a periodic vector grid."""
    from scipy.ndimage import gaussian_filter
    import pyvista as pv

    data = _read_snapshot_fields(
        files, ptype, ["Coordinates", "Velocities", "Masses"], cap, selected_ids,
    )
    coordinates, velocities, masses = (
        data.get("Coordinates"), data.get("Velocities"), data.get("Masses"),
    )
    if coordinates is None or velocities is None:
        return None, None, None, None, "coordinates or Velocities are unavailable"
    coordinates = np.mod(convert_pinned_native(coordinates,'position_comoving_mpc_h'), box_mpc_h)
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != coordinates.shape:
        return None, None, None, None, "Velocities must have three components aligned with Coordinates"
    mass_weighting = "particle-mass weighted"
    if masses is None:
        fixed_mass = _snapshot_constant_particle_mass(files, ptype)
        if fixed_mass is None:
            return None, None, None, None, "particle-mass-weighted velocity fields require known particle masses"
        else:
            weights = np.full(len(coordinates), fixed_mass, dtype=float)
    else:
        weights = np.asarray(masses, dtype=float)
        if weights.shape != (len(coordinates),):
            return None, None, None, None, "Masses must contain one value per sampled velocity"
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            return None, None, None, None, "Masses contain non-finite or negative values"
    good = np.all(np.isfinite(coordinates), axis=1) & np.all(np.isfinite(velocities), axis=1)
    # Stationary particles contribute mass to the mean velocity denominator.
    # Dropping them would systematically exaggerate streaming speeds.
    good &= np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(good) < 8:
        return None, None, None, None, "fewer than eight finite velocity samples are available"
    coordinates, velocities, weights = coordinates[good], velocities[good], weights[good]
    n = int(grid_size)
    if isinstance(grid_size, bool) or n != grid_size or not 8 <= n <= 192:
        raise ValueError("Velocity grid size must be an integer from 8 to 192")
    density, _ = np.histogramdd(
        coordinates, bins=n,
        range=((0, box_mpc_h), (0, box_mpc_h), (0, box_mpc_h)),
        weights=weights,
    )
    weighted_components = []
    for component in range(3):
        component_sum, _ = np.histogramdd(
            coordinates, bins=n,
            range=((0, box_mpc_h), (0, box_mpc_h), (0, box_mpc_h)),
            weights=weights * velocities[:, component],
        )
        weighted_components.append(component_sum)
    smooth_density = gaussian_filter(density, sigma=1.25, mode="wrap")
    vectors = np.zeros((n, n, n, 3), dtype=np.float32)
    for component, component_sum in enumerate(weighted_components):
        smooth_sum = gaussian_filter(component_sum, sigma=1.25, mode="wrap")
        vectors[..., component] = np.divide(
            smooth_sum, smooth_density, out=np.zeros_like(smooth_sum),
            where=smooth_density > float(np.max(smooth_density)) * 1.0e-5,
        )
    # One wrapped ghost cell on each face allows trilinear interpolation
    # across the physical box faces; it introduces no new velocity samples.
    vectors_with_faces = np.pad(vectors, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="wrap")
    grid = pv.ImageData(
        dimensions=(n+2, n+2, n+2), spacing=(box_mpc_h / n,) * 3,
        origin=(-box_mpc_h / (2.0 * n),) * 3,
    )
    grid.point_data["Velocity"] = vectors_with_faces.reshape((-1, 3), order="F")
    dense = np.argwhere(smooth_density > float(np.max(smooth_density)) * 0.08)
    if not len(dense):
        return None, None, None, None, "smoothed velocity field has no well-sampled seed cells"
    cell_centers = (dense.astype(float) + 0.5) * box_mpc_h / n
    return grid, cell_centers, coordinates, velocities, {"mass_weighting": mass_weighting}


def _particle_mass_density_grid(
    files, ptype, box_mpc_h, grid_size, chunk_size=250_000, return_density_ratio=False,
):
    """Build a periodic, mass-weighted CIC log-density contrast from one output.

    By default the stored scalar is log10(rho/<rho>) with a documented
    low-density floor. ``return_density_ratio`` returns the unfloored
    rho/<rho> grid for algorithms that need a physical density field.
    The normalization makes the visualization independent of the snapshot's
    mass-unit convention while retaining the spatial mass contrast.
    """
    from analysis import _deposit_cic_positions

    n = int(grid_size)
    box = float(box_mpc_h)
    if n < 8 or n > 192:
        raise ValueError("3-D density grid_size must be between 8 and 192")
    if not np.isfinite(box) or box <= 0:
        raise ValueError("3-D density requires a finite positive periodic box size")
    if int(ptype) < 0 or int(ptype) > 5:
        raise ValueError("GADGET particle type must be between 0 and 5")
    if int(chunk_size) <= 0:
        raise ValueError("3-D density chunk_size must be positive")
    density_mass = np.zeros((n, n, n), dtype=np.float64)
    total_mass = 0.0
    particle_count = 0
    mass_table_reference = None
    mass_source = None
    group_name = "PartType%d" % int(ptype)
    for path in files:
        with h5py.File(path, "r") as h5:
            if "Header" not in h5:
                raise ValueError("snapshot shard has no Header: " + str(path))
            parameters = h5.get("Parameters")
            if parameters is None or "UnitLength_in_cm" not in parameters.attrs:
                raise ValueError("Snapshot Parameters/UnitLength_in_cm is required for Mpc/h map coordinates")
            unit_length = float(parameters.attrs["UnitLength_in_cm"])
            if not np.isfinite(unit_length) or not np.isclose(unit_length, KPC_CM, rtol=1.0e-6, atol=0.0):
                raise ValueError("Snapshot UnitLength_in_cm must match the configured kpc/h coordinate unit")
            if "MassTable" not in h5["Header"].attrs:
                raise ValueError("snapshot Header/MassTable is required to interpret particle masses")
            raw_table = np.asarray(h5["Header"].attrs["MassTable"], dtype=float)
            if raw_table.shape != (6,) or not np.all(np.isfinite(raw_table)) or np.any(raw_table < 0):
                raise ValueError("snapshot Header/MassTable must contain six finite nonnegative masses")
            if mass_table_reference is None:
                mass_table_reference = raw_table
            elif not np.allclose(raw_table, mass_table_reference, rtol=1.0e-10, atol=0.0):
                raise ValueError("snapshot shards have inconsistent Header/MassTable values")
            if group_name not in h5:
                continue
            particles = h5[group_name]
            if "Coordinates" not in particles:
                raise ValueError(group_name + " has no Coordinates dataset")
            coordinates = particles["Coordinates"]
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(group_name + "/Coordinates must have shape (particles, 3)")
            count = len(coordinates)
            has_variable_mass = "Masses" in particles
            source = "variable" if has_variable_mass else "constant"
            if mass_source is not None and mass_source != source and count:
                raise ValueError(group_name + " mixes variable and constant masses across shards")
            if count == 0:
                continue
            mass_source = source
            if has_variable_mass:
                masses = particles["Masses"]
                if masses.ndim != 1 or len(masses) != count:
                    raise ValueError(group_name + "/Masses must align with Coordinates")
                if raw_table[int(ptype)] != 0:
                    raise ValueError(group_name + " has both variable Masses and nonzero Header/MassTable")
                fixed_mass = None
            else:
                fixed_mass = float(raw_table[int(ptype)])
                if not np.isfinite(fixed_mass) or fixed_mass <= 0:
                    raise ValueError(group_name + " has no valid particle-mass source")
                masses = None
            for start in range(0, count, int(chunk_size)):
                stop = min(start + int(chunk_size), count)
                positions = np.asarray(coordinates[start:stop], dtype=np.float64) / 1000.0
                weights = (
                    np.asarray(masses[start:stop], dtype=np.float64)
                    if masses is not None else np.full(stop - start, fixed_mass, dtype=np.float64)
                )
                if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                    raise ValueError(group_name + "/Masses contains non-finite or negative values")
                _deposit_cic_positions(density_mass, positions, box, n, weights=weights)
                total_mass += float(np.sum(weights, dtype=np.float64))
                particle_count += stop - start
    if particle_count == 0 or not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError(group_name + " contains no positive-mass particles for a density volume")
    deposited_mass = float(np.sum(density_mass, dtype=np.float64))
    if not np.isclose(deposited_mass, total_mass, rtol=1.0e-10, atol=1.0e-10 * total_mass):
        raise RuntimeError("periodic CIC density deposition did not conserve particle mass")
    mean_cell_mass = total_mass / float(n**3)
    density_ratio = density_mass / mean_cell_mass
    density_floor = 1.0e-4
    output_grid = (
        density_ratio.astype(np.float32)
        if return_density_ratio
        else np.log10(np.maximum(density_ratio, density_floor)).astype(np.float32)
    )
    metadata = {
        "method": "periodic cloud-in-cell mass assignment",
        "scalar": "rho/<rho>" if return_density_ratio else "log10(rho/<rho>)",
        "particle_type": int(ptype),
        "particle_count": int(particle_count),
        "grid_size": n,
        "box_mpc_h": box,
        "particle_mass_source": mass_source,
        "density_floor_rho_over_mean": density_floor,
        "total_particle_mass_snapshot_units": total_mass,
    }
    return output_grid, metadata


def _snapshot_particle_type_total(header, files, ptype):
    """Return a GADGET 64-bit particle count, or count rows from its shards."""
    from core.data import snapshot_total_counts
    if "NumPart_Total" in header:
        return int(snapshot_total_counts(header)[ptype])
    group = "PartType%d" % int(ptype)
    total = 0
    for path in files:
        with h5py.File(path, "r") as h5:
            if group in h5 and "Coordinates" in h5[group]:
                total += len(h5[group]["Coordinates"])
    return total


def _all_snapshot_shards_have_fields(snapshot_sources, ptype, required):
    group = "PartType%d" % int(ptype)
    found_particles = False
    for _, files in snapshot_sources.values():
        for path in files:
            with h5py.File(path, "r") as h5:
                if group not in h5 or "Coordinates" not in h5[group]:
                    continue
                particles = h5[group]
                count = len(particles["Coordinates"])
                if count == 0:
                    continue
                found_particles = True
                for name in required:
                    if name not in particles or particles[name].ndim == 0 or len(particles[name]) != count:
                        return False
                    if name == "Velocities" and (
                        particles[name].ndim != 2 or particles[name].shape[1] != 3
                    ):
                        return False
                    if name != "Velocities" and particles[name].ndim != 1:
                        return False
    return found_particles


def _temperature_field_supported(snapshot_sources, cfg):
    group = "PartType0"
    found_particles = False
    star_formation = bool(cfg.get("hydro", {}).get("star_formation", False))
    for _, files in snapshot_sources.values():
        modes = []
        for path in files:
            with h5py.File(path, "r") as h5:
                if group not in h5 or "Coordinates" not in h5[group]:
                    continue
                particles = h5[group]
                count = len(particles["Coordinates"])
                if count == 0:
                    continue
                found_particles = True
                if "Temperature" in particles:
                    name = "Temperature"
                    if particles[name].ndim != 1 or len(particles[name]) != count:
                        return False
                    modes.append("stored")
                elif "InternalEnergy" in particles and "ElectronAbundance" in particles:
                    names = ["InternalEnergy", "ElectronAbundance"]
                    if star_formation:
                        names.append("StarFormationRate")
                    if any(particles[name].ndim != 1 or len(particles[name]) != count for name in names if name in particles):
                        return False
                    if any(name not in particles for name in names):
                        return False
                    modes.append("derived")
                else:
                    return False
        if modes and len(set(modes)) > 1:
            return False
        if "derived" in modes and _snapshot_parameter(files, "UnitVelocity_in_cm_per_s") is None:
            return False
    return found_particles


def _available_3d_color_fields(snapshot_sources, ptype, cfg):
    fields = ["Uniform"]
    if _all_snapshot_shards_have_fields(snapshot_sources, ptype, ["Velocities"]):
        fields.append("Speed")
    if int(ptype) == 4 and _all_snapshot_shards_have_fields(
            snapshot_sources, ptype, ["StellarFormationTime"]):
        fields.append("Formation redshift")
    if int(ptype) == 0:
        if _all_snapshot_shards_have_fields(snapshot_sources, ptype, ["Density"]):
            fields.append("Density")
        if _temperature_field_supported(snapshot_sources, cfg):
            fields.append("Temperature")
    if _all_snapshot_shards_have_fields(snapshot_sources, ptype, ["Potential"]):
        fields.append("Potential")
    return fields


def _velocity_conversion_km_s(files, ptype, header):
    """Return the snapshot's declared particle-velocity scale in km/s, if complete."""
    try:
        from analysis import (
            _dataset_to_cgs_factor, _snapshot_dataset_unit_attributes,
            _snapshot_header_value,
        )
        units = _snapshot_dataset_unit_attributes(files, int(ptype), "Velocities")
        scale_factor = _snapshot_scale_factor(header)
        hubble_param = _snapshot_header_value(files, "HubbleParam")
        if hubble_param is None:
            hubble_param = _snapshot_parameter(files, "HubbleParam")
        if units is None or not np.isfinite(hubble_param) or hubble_param <= 0:
            return None
        factor = _dataset_to_cgs_factor(units, scale_factor, hubble_param) / 1.0e5
        return float(factor) if np.isfinite(factor) and factor > 0 else None
    except (OSError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def _read_3d_color_values(files, ptype, header, cfg, field, max_points, selected_ids):
    if field == "Uniform":
        return None, None, False
    if field == "Speed":
        data = _read_snapshot_fields(files, ptype, ["Velocities"], max_points, selected_ids)
        velocity = data["Velocities"]
        if velocity is None:
            return None, "Speed [snapshot units]", False
        velocity = np.asarray(velocity, dtype=float)
        if velocity.ndim != 2 or velocity.shape[1] != 3:
            raise ValueError("PartType%d/Velocities must have shape (particles, 3)" % int(ptype))
        conversion = _velocity_conversion_km_s(files, ptype, header)
        if conversion is not None:
            return np.linalg.norm(velocity, axis=1) * conversion, "Speed [km/s]", False
        return np.linalg.norm(velocity, axis=1), "Speed [snapshot units]", False
    if field == "Formation redshift":
        data = _read_snapshot_fields(files, ptype, ["StellarFormationTime"], max_points, selected_ids)
        formation_scale = data.get("StellarFormationTime")
        if formation_scale is None:
            return None, "Formation redshift [derived from scale factor]", False
        formation_scale = np.asarray(formation_scale, dtype=float)
        redshift = np.full(formation_scale.shape, np.nan, dtype=float)
        valid = np.isfinite(formation_scale) & (formation_scale > 0)
        redshift[valid] = 1.0 / formation_scale[valid] - 1.0
        return redshift, "Formation redshift", False
    if field == "Density":
        data = _read_snapshot_fields(files, ptype, ["Density"], max_points, selected_ids)
        values = data["Density"]
        return (None, "log10 Density [snapshot units]", True) if values is None else (np.asarray(values, dtype=float), "log10 Density [snapshot units]", True)
    if field == "Potential":
        data = _read_snapshot_fields(files, ptype, ["Potential"], max_points, selected_ids)
        values = data["Potential"]
        return (None, "Potential [snapshot units]", False) if values is None else (np.asarray(values, dtype=float), "Potential [snapshot units]", False)
    if field == "Temperature":
        stored_data = _read_snapshot_fields(files, ptype, ["Temperature", "StarFormationRate"], max_points, selected_ids)
        stored = stored_data["Temperature"]
        if stored is not None:
            require_stored_temperature_kelvin(files, ptype)
            values = np.asarray(stored, dtype=float)
            sfr = stored_data["StarFormationRate"]
            if sfr is not None:
                values = np.where(np.isfinite(sfr) & (np.asarray(sfr) == 0.), values, np.nan)
            elif bool(cfg.get("hydro", {}).get("star_formation", False)):
                return None, "Temperature [K]", True
            return values, "log10 Temperature [K; stored]", True
        data = _read_snapshot_fields(
            files, ptype, ["InternalEnergy", "ElectronAbundance", "StarFormationRate"],
            max_points, selected_ids,
        )
        energy = data["InternalEnergy"]
        electrons = data["ElectronAbundance"]
        if energy is None or electrons is None:
            return None, "Temperature [K]", True
        velocity_unit = _snapshot_parameter(files, "UnitVelocity_in_cm_per_s")
        if velocity_unit is None:
            return None, "Temperature [K]", True
        values = temperature_from_internal_energy(
            energy, electrons, unit_velocity_cm_per_s=velocity_unit,
        )
        sfr = data["StarFormationRate"]
        if sfr is not None:
            non_star_forming = np.isfinite(sfr) & (np.asarray(sfr, dtype=float) == 0.0)
            values = np.where(non_star_forming, values, np.nan)
            return values, "log10 Temperature [K; derived non-star-forming gas]", True
        if bool(cfg.get("hydro", {}).get("star_formation", False)):
            return None, "Temperature [K]", True
        return values, "log10 Temperature [K; derived]", True
    raise ValueError("Unsupported 3-D color field: " + str(field))


def open_interactive_3d(run_dir, snap=None, species="dm", max_points=None, full_resolution=False):
    """Open a timeline viewer with independently switchable gas, DM, and star layers.

    ``species`` is the initially visible layer. In sampled mode each layer uses
    up to ``max_points_3d`` particles selected by stable IDs across snapshots;
    full-resolution mode loads every particle of each layer the user enables.
    """
    import pyvista as pv

    run_dir = Path(run_dir)
    cfg = load_cfg(run_dir)
    box_mpc_h = float(cfg["simulation"]["box_mpc_h"])
    available = snapshot_ids(run_dir)
    if not available:
        raise FileNotFoundError("No HDF5 snapshots found under " + str(run_dir / "data"))
    snapshot_sources = {
        current: read_snapshot_header(run_dir, current)
        for current in available
    }
    timeline_rows = sorted(
        ((_snapshot_scale_factor(header), current) for current, (header, _) in snapshot_sources.items()),
    )
    scale_factors = np.asarray([row[0] for row in timeline_rows], dtype=float)
    available = [row[1] for row in timeline_rows]
    if len(scale_factors) > 1 and np.any(np.diff(scale_factors) <= 0.0):
        raise ValueError("Timeline snapshots must have distinct, increasing scale factors")
    cosmic_time_scale = cosmic_time_values = None
    cosmic_time_unavailable_reason = "validated CLASS proper-time table is unavailable"
    try:
        from analysis import _stellar_formation_clock
        cosmic_time_scale, cosmic_time_values, cosmic_time_unavailable_reason = _stellar_formation_clock(run_dir)
        if (cosmic_time_scale is not None and cosmic_time_values is not None
                and len(cosmic_time_scale) >= 2
                and len(cosmic_time_scale) == len(cosmic_time_values)
                and np.all(np.isfinite(cosmic_time_scale))
                and np.all(np.isfinite(cosmic_time_values))
                and np.all(np.diff(cosmic_time_scale) > 0)
                and np.all(np.diff(cosmic_time_values) >= 0)):
            cosmic_time_scale = np.asarray(cosmic_time_scale, dtype=float)
            cosmic_time_values = np.asarray(cosmic_time_values, dtype=float)
        else:
            cosmic_time_scale = cosmic_time_values = None
            if not cosmic_time_unavailable_reason:
                cosmic_time_unavailable_reason = "CLASS proper-time table failed viewer validation"
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        cosmic_time_scale = cosmic_time_values = None
        cosmic_time_unavailable_reason = str(exc)

    def cosmic_time_for_scale_factor(scale_factor):
        if (cosmic_time_scale is None or cosmic_time_values is None
                or not cosmic_time_scale[0] <= float(scale_factor) <= cosmic_time_scale[-1]):
            return None
        return float(np.interp(float(scale_factor), cosmic_time_scale, cosmic_time_values))

    timeline_cosmic_times_gyr = np.asarray([
        (value if value is not None else np.nan)
        for value in (cosmic_time_for_scale_factor(scale) for scale in scale_factors)
    ], dtype=float)
    has_cosmic_time_timeline = (
        len(timeline_cosmic_times_gyr) == len(scale_factors)
        and len(timeline_cosmic_times_gyr) > 1
        and np.all(np.isfinite(timeline_cosmic_times_gyr))
        and np.all(np.diff(timeline_cosmic_times_gyr) > 0)
    )

    snap = available[-1] if snap is None else int(snap)
    if snap not in available:
        raise ValueError("Snapshot %d is not present; available snapshots: %s" % (snap, available))
    species_types = {"gas": 0, "dm": 1, "stars": 4}
    if species not in species_types:
        raise ValueError("species must be one of: gas, dm, stars")
    selected_ptype = species_types[species]
    selected_species_name = species
    available_types = [
        (name, ptype) for name, ptype in species_types.items()
        if any(
            _snapshot_particle_type_total(header, files, ptype) > 0
            for header, files in snapshot_sources.values()
        )
    ]
    if not any(ptype == selected_ptype for _, ptype in available_types):
        raise ValueError("The run contains no particles of type " + species)
    if full_resolution:
        cap = None
    else:
        configured_cap = int(cfg.get("visualization", {}).get("max_points_3d", 3_000_000))
        cap = configured_cap if max_points is None else int(max_points)
        if cap <= 0:
            raise ValueError("max_points_3d must be positive; use full_resolution=True to render every particle")
    mass_definition = str(cfg["analysis"]["hmf_mass_definition"]).lower()
    mass_limits, available_catalogue_layers = _catalogue_mass_limits(
        run_dir, available, ["halos", "subhalos", "rockstar"], mass_definition,
    )
    max_catalogue_points = int(cfg.get("visualization", {}).get("max_catalogue_points_3d", 50000))
    if max_catalogue_points <= 0:
        raise ValueError("visualization.max_catalogue_points_3d must be positive")
    configured_mass_floor = cfg.get("visualization", {}).get("min_halo_mass_msun_h")
    if mass_limits is None:
        mass_floor_log10 = None
    elif configured_mass_floor is None:
        mass_floor_log10 = mass_limits[0]
    else:
        configured_mass_floor = float(configured_mass_floor)
        if not np.isfinite(configured_mass_floor) or configured_mass_floor <= 0:
            raise ValueError("visualization.min_halo_mass_msun_h must be finite and positive")
        mass_floor_log10 = float(np.clip(np.log10(configured_mass_floor), *mass_limits))
    plotter = pv.Plotter(window_size=(1500, 1000), theme=pv.themes.DarkTheme())
    plotter.set_background("#050914")
    plotter.add_axes(line_width=2)
    plotter.show_axes()
    plotter.add_mesh(
        pv.Cube(bounds=(0.0, box_mpc_h, 0.0, box_mpc_h, 0.0, box_mpc_h)),
        color="#64748b", style="wireframe", opacity=0.20,
        line_width=0.8, lighting=False, name="periodic_box_boundary", render=False,
    )
    current_index = available.index(snap)
    current_header, current_files = snapshot_sources[snap]

    def load_stable_sample(ptype, start_index):
        files_at_start = snapshot_sources[available[start_index]][1]
        if cap is None:
            xyz_at_start, ids = _read_snapshot_points(files_at_start, ptype, cap)
            return xyz_at_start, ids, ids
        sample_ids = _sample_timeline_id_union(run_dir, available, ptype, cap)
        xyz_at_start, display_ids = _read_snapshot_points(
            files_at_start, ptype, cap, selected_ids=sample_ids,
        )
        return xyz_at_start, sample_ids, display_ids

    def read_layer_sample(ptype, state, files):
        if cap is None or state.get("sample_ids") is None:
            return _read_snapshot_points(files, ptype, cap)
        return _read_snapshot_points(files, ptype, cap, selected_ids=state["sample_ids"])

    color_fields = _available_3d_color_fields(snapshot_sources, selected_ptype, cfg)
    appearance_state = {"point_size": 2.0, "opacity": 1.0}
    color_state = {
        "field": "Uniform", "scalar_bar_title": None, "locked_range": None,
        "lock_widget": None,
    }
    layer_colors = {"gas": "#52d6b5", "dm": "#a8d8ff", "stars": "#ffd166"}
    catalogue_colors = {"halos": "#ffe082", "subhalos": "#ff8a80", "rockstar": "#bb86fc"}

    def remove_color_bar():
        title = color_state["scalar_bar_title"]
        if title is not None:
            plotter.remove_scalar_bar(title, render=False)
            color_state["scalar_bar_title"] = None

    def add_layer_actor(ptype, state, xyz, files, header):
        if ptype == selected_ptype:
            remove_color_bar()
            state["color_range"] = None
        if not len(xyz):
            state["color_label"] = None
            state["color_valid_count"] = 0
            state["render_ids"] = np.empty(0, dtype=np.uint64)
            state["base_render_points"] = np.empty((0, 3), dtype=np.float32)
            state["shown"] = 0
            return None
        points = np.asarray(xyz, dtype=np.float32) / 1000.0
        if ptype == selected_ptype and color_state["field"] != "Uniform":
            values, title, logarithmic = _read_3d_color_values(
                files, ptype, header, cfg, color_state["field"], cap,
                state.get("display_ids"),
            )
            if values is not None:
                values = np.asarray(values, dtype=float)
                if values.ndim != 1 or len(values) != len(points):
                    raise ValueError("3-D color field %s does not align with the rendered particle sample" % color_state["field"])
                good = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
                if logarithmic:
                    good &= values > 0
                if np.any(good):
                    color_values = np.log10(values[good]) if logarithmic else values[good]
                    locked_range = color_state.get("locked_range")
                    if locked_range is None:
                        low, high = np.percentile(color_values, [1.0, 99.0])
                    else:
                        low, high = (float(locked_range[0]), float(locked_range[1]))
                    if not np.isfinite(low) or not np.isfinite(high):
                        good[:] = False
                    elif high <= low:
                        width = max(abs(float(low)) * 1.0e-6, 1.0e-6)
                        low, high = float(low) - width, float(high) + width
                if np.any(good):
                    color_values = np.log10(values[good]) if logarithmic else values[good]
                    cloud = pv.PolyData(points[good])
                    cloud.point_data["color_value"] = color_values
                    legend_title = {
                        "Speed [km/s]": "Speed [km/s]",
                        "Speed [snapshot units]": "Speed [code units]",
                        "log10 Density [snapshot units]": "log10 Density [code units]",
                        "Potential [snapshot units]": "Potential [code units]",
                        "log10 Temperature [K; stored]": "log10 T [K]",
                        "log10 Temperature [K; derived non-star-forming gas]": "log10 T [K]",
                        "log10 Temperature [K; derived]": "log10 T [K]",
                    }.get(title, title)
                    actor = plotter.add_points(
                        cloud, scalars="color_value", cmap="viridis", clim=(float(low), float(high)),
                        show_scalar_bar=False, point_size=appearance_state["point_size"],
                        opacity=appearance_state["opacity"], render_points_as_spheres=False,
                    )
                    plotter.add_scalar_bar(
                        title=legend_title, cmap="viridis", clim=(float(low), float(high)),
                        n_labels=5, vertical=True, position_x=0.88, position_y=0.16,
                        width=0.085, height=0.62, title_font_size=14, label_font_size=14,
                        render=False,
                    )
                    color_state["scalar_bar_title"] = legend_title
                    state["color_label"] = title
                    state["color_valid_count"] = int(np.count_nonzero(good))
                    state["color_range"] = [float(low), float(high)]
                    display_ids = state.get("display_ids")
                    state["render_ids"] = (
                        np.asarray(display_ids)[good].copy()
                        if display_ids is not None and len(display_ids) == len(points)
                        else None
                    )
                    state["base_render_points"] = points[good].copy()
                    state["shown"] = len(state["base_render_points"])
                    return actor
            state["color_label"] = title + " unavailable in this snapshot"
            state["color_valid_count"] = 0
        elif ptype == selected_ptype:
            state["color_label"] = "Uniform"
            state["color_valid_count"] = len(points)
        display_ids = state.get("display_ids")
        state["render_ids"] = (
            np.asarray(display_ids).copy()
            if display_ids is not None and len(display_ids) == len(points)
            else None
        )
        state["base_render_points"] = points.copy()
        state["shown"] = len(points)
        return plotter.add_points(
            pv.PolyData(points), color=layer_colors[state["name"]],
            point_size=appearance_state["point_size"], opacity=appearance_state["opacity"],
            render_points_as_spheres=False,
        )

    layer_state = {}
    for name, ptype in available_types:
        visible = ptype == selected_ptype
        state = {
            "name": name, "visible": visible, "actor": None,
            "sample_ids": None, "display_ids": None, "render_ids": None,
            "base_render_points": np.empty((0, 3), dtype=np.float32),
            "interpolation_ids": None, "timeline_sample_ids": None,
            "timeline_event_actors": [],
            "shown": 0,
        }
        if visible:
            xyz, sample_ids, display_ids = load_stable_sample(ptype, current_index)
            state["sample_ids"] = sample_ids
            state["display_ids"] = display_ids
            state["shown"] = len(xyz)
            state["actor"] = add_layer_actor(ptype, state, xyz, current_files, current_header)
        layer_state[ptype] = state

    mass_filter = {"log10_minimum": mass_floor_log10}
    catalogue_state = {
        kind: {"name": kind, "visible": False, "actor": None, "data": None,
               "catalog_rows": np.empty(0, dtype=np.int64), "shown": 0, "matching": 0}
        for kind in available_catalogue_layers
    }
    r200_state = {
        "visible": False, "actor": None, "shown": 0,
        "periodic_images": 0, "skipped_periodic": 0,
    }
    filament_state = {"visible": False, "actor": None, "shown": 0, "status": None, "metadata": None}
    density_state = {
        "visible": False, "actor": None, "grid": None, "cache": {}, "cache_order": [],
        "status": None, "scalar_bar_title": None,
    }
    density_grid_size = int(cfg.get("visualization", {}).get("density_volume_grid_size", 64))
    if density_grid_size < 16 or density_grid_size > 128:
        raise ValueError("visualization.density_volume_grid_size must be between 16 and 128")
    picked_selection = {"kind": None, "row": None, "halo_id": None}
    motion_state = {
        "playing": False, "timeline_value": float(scale_factors[current_index]),
        "fraction": 0.0, "playback_speed": 1.0, "timeline_slider": None,
        "loop": bool(cfg.get("visualization", {}).get("timeline_loop", True)),
        "play_widget": None, "loop_widget": None,
        "step_back_widget": None, "step_forward_widget": None,
        "follow_widget": None,
        "velocity_arrows": False, "streamlines": False, "velocity_status": None,
        "arrow_scale_box_fraction": 0.035,
        "arrow_actor": None, "streamline_actor": None,
        "follow": False, "follow_centers": None, "follow_center": None,
        "follow_anchor_halo_id": None, "follow_anchor_snapshot": None,
        "follow_track": None, "follow_status": None, "follow_marker_actor": None,
        "follow_shell_actor": None,
        "follow_trail_actor": None, "growth_chart": None, "growth_chart_snapshot": None,
        "growth_chart_track": None,
        "merger_marker_actor": None, "merger_status": None,
        "interpolation_status": None,
    }
    slice_state = {
        "visible": False, "center_mpc_h": box_mpc_h / 2.0,
        "thickness_mpc_h": box_mpc_h / 24.0,
        "actors": [], "plane_actors": [], "shown": 0,
    }

    def refresh_particle_slice():
        for actor in slice_state["actors"]:
            plotter.remove_actor(actor, render=False)
        slice_state["actors"] = []
        for actor in slice_state["plane_actors"]:
            plotter.remove_actor(actor, render=False)
        slice_state["plane_actors"] = []
        slice_state["shown"] = 0
        for state in layer_state.values():
            actor = state.get("actor")
            if actor is None:
                continue
            try:
                actor.GetProperty().SetOpacity(
                    appearance_state["opacity"] * (0.14 if slice_state["visible"] else 1.0),
                )
            except (AttributeError, RuntimeError, TypeError):
                pass
            if not slice_state["visible"]:
                continue
            try:
                points = np.asarray(actor.mapper.dataset.points, dtype=float)
            except (AttributeError, RuntimeError, TypeError):
                continue
            if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                continue
            distance = np.mod(
                points[:, 2] - slice_state["center_mpc_h"] + box_mpc_h / 2.0,
                box_mpc_h,
            ) - box_mpc_h / 2.0
            inside = np.isfinite(distance) & (np.abs(distance) <= slice_state["thickness_mpc_h"] / 2.0)
            if np.any(inside):
                highlight = pv.PolyData(points[inside].astype(np.float32, copy=False))
                slice_state["actors"].append(plotter.add_points(
                    highlight, color=layer_colors[state["name"]],
                    point_size=min(14.0, appearance_state["point_size"] + 2.0),
                    opacity=1.0, render_points_as_spheres=False, render=False,
                ))
                slice_state["shown"] += int(np.count_nonzero(inside))
        if slice_state["visible"]:
            half_width = slice_state["thickness_mpc_h"] / 2.0
            plane_centers = [slice_state["center_mpc_h"]]
            if slice_state["center_mpc_h"] - half_width < 0.0:
                plane_centers.append(box_mpc_h)
            if slice_state["center_mpc_h"] + half_width > box_mpc_h:
                plane_centers.append(0.0)
            for plane_center in dict.fromkeys(plane_centers):
                plane = pv.Plane(
                    center=(box_mpc_h / 2.0, box_mpc_h / 2.0, plane_center),
                    direction=(0.0, 0.0, 1.0), i_size=box_mpc_h, j_size=box_mpc_h,
                    i_resolution=1, j_resolution=1,
                )
                slice_state["plane_actors"].append(plotter.add_mesh(
                    plane, color="#67e8f9", opacity=0.12, show_edges=True,
                    edge_color="#67e8f9", line_width=1.1, lighting=False, render=False,
                ))

    def set_particle_point_size(value):
        appearance_state["point_size"] = float(np.clip(float(value), 1.0, 10.0))
        for state in layer_state.values():
            actor = state.get("actor")
            if actor is None:
                continue
            try:
                actor.GetProperty().SetPointSize(appearance_state["point_size"])
            except (AttributeError, RuntimeError, TypeError):
                pass
        if slice_state["visible"]:
            refresh_particle_slice()
        plotter.render()

    def set_particle_opacity(value):
        appearance_state["opacity"] = float(np.clip(float(value), 0.08, 1.0))
        refresh_particle_slice()
        plotter.render()

    def clear_picked_catalogue_info(render=False):
        picked_selection["kind"] = None
        picked_selection["row"] = None
        picked_selection["halo_id"] = None
        plotter.add_text(
            "Click a halo marker for its catalogue properties",
            position=(52, catalogue_y - 75), font_size=9, color="#f8fafc",
            name="picked_catalogue_info", render=render,
        )

    def add_catalogue_actor(kind, state, data):
        if data is None:
            state["data"] = None
            state["catalog_rows"] = np.empty(0, dtype=np.int64)
            state["shown"] = state["matching"] = 0
            return None
        positions = np.asarray(data["position_mpc_h"], dtype=float)
        masses = np.asarray(data["mass_msun_h"], dtype=float)
        good = np.all(np.isfinite(positions), axis=1) & np.isfinite(masses) & (masses > 0)
        if mass_filter["log10_minimum"] is not None:
            good &= np.log10(np.maximum(masses, np.finfo(float).tiny)) >= mass_filter["log10_minimum"]
        rows = np.flatnonzero(good)
        state["matching"] = len(rows)
        if len(rows) > max_catalogue_points:
            keep = np.argpartition(masses[rows], -max_catalogue_points)[-max_catalogue_points:]
            rows = rows[keep]
        if len(rows):
            rows = rows[np.argsort(masses[rows], kind="stable")]
            cloud = pv.PolyData(positions[rows].astype(np.float32, copy=False))
            cloud.point_data["dms_catalog_kind"] = np.full(
                len(rows), {"halos": 0, "subhalos": 1, "rockstar": 2}[kind], dtype=np.uint8,
            )
            cloud.point_data["dms_catalog_row"] = np.asarray(data["row_index"][rows], dtype=np.int64)
            state["actor"] = plotter.add_points(
                cloud, color=catalogue_colors[kind], point_size=10,
                render_points_as_spheres=True,
            )
        else:
            state["actor"] = None
        state["data"] = data
        state["catalog_rows"] = np.asarray(data["row_index"][rows], dtype=np.int64)
        state["shown"] = len(rows)
        return state["actor"]

    def add_r200_shells(current_snap):
        if r200_state["actor"] is not None:
            plotter.remove_actor(r200_state["actor"], render=False)
            r200_state["actor"] = None
        r200_state["shown"] = 0
        r200_state["periodic_images"] = 0
        r200_state["skipped_periodic"] = 0
        if not r200_state["visible"]:
            return
        data = _read_catalogue_layer(run_dir, current_snap, "halos", mass_definition)
        if data is None or data.get("Group_R_Crit200") is None:
            return
        positions = np.asarray(data["position_mpc_h"], dtype=float)
        radii = np.asarray(data["Group_R_Crit200"], dtype=float) / 1000.0
        masses = np.asarray(data["mass_msun_h"], dtype=float)
        good = (
            np.all(np.isfinite(positions), axis=1) & np.isfinite(radii) & (radii > 0)
            & np.isfinite(masses) & (masses > 0) & (radii < box_mpc_h / 2.0)
        )
        if mass_filter["log10_minimum"] is not None:
            good &= np.log10(np.maximum(masses, np.finfo(float).tiny)) >= mass_filter["log10_minimum"]
        candidate_rows = np.flatnonzero(good)
        maximum_shells = int(cfg.get("visualization", {}).get("r200_shell_max_count", 200))
        if maximum_shells <= 0:
            raise ValueError("visualization.r200_shell_max_count must be positive")
        if len(candidate_rows) > maximum_shells:
            keep = np.argpartition(masses[candidate_rows], -maximum_shells)[-maximum_shells:]
            candidate_rows = candidate_rows[keep]
        meshes = []
        for row in candidate_rows:
            center = np.mod(positions[row], box_mpc_h)
            radius = float(radii[row])
            per_axis_offsets = []
            for coordinate in center:
                offsets = [0.0]
                if coordinate < radius:
                    offsets.append(box_mpc_h)
                if box_mpc_h - coordinate < radius:
                    offsets.append(-box_mpc_h)
                per_axis_offsets.append(offsets)
            copies_for_halo = 0
            for offset_x in per_axis_offsets[0]:
                for offset_y in per_axis_offsets[1]:
                    for offset_z in per_axis_offsets[2]:
                        offset = np.asarray([offset_x, offset_y, offset_z], dtype=float)
                        image_center = center + offset
                        intersects_box = np.all(image_center + radius > 0.0) and np.all(image_center - radius < box_mpc_h)
                        if not intersects_box:
                            continue
                        meshes.append(pv.Sphere(
                            radius=radius, center=image_center,
                            theta_resolution=12, phi_resolution=8,
                        ))
                        copies_for_halo += 1
                        if np.any(offset != 0.0):
                            r200_state["periodic_images"] += 1
            if copies_for_halo:
                r200_state["shown"] += 1
            else:
                r200_state["skipped_periodic"] += 1
        if meshes:
            surface = pv.MultiBlock(meshes).combine(merge_points=False)
            r200_state["actor"] = plotter.add_mesh(
                surface, color="#f8fafc", style="wireframe", line_width=1.0,
                opacity=0.42, render=False,
            )

    def refresh_density_volume(current_snap):
        actor = density_state.get("actor")
        if actor is not None:
            plotter.remove_actor(actor, render=False)
            density_state["actor"] = None
        bar_title = density_state.get("scalar_bar_title")
        if bar_title:
            plotter.remove_scalar_bar(bar_title, render=False)
            density_state["scalar_bar_title"] = None
        density_state["grid"] = None
        density_state["status"] = None
        if not density_state["visible"]:
            return
        key = (int(current_snap), int(selected_ptype), int(density_grid_size))
        try:
            cached = density_state["cache"].get(key)
            if cached is None:
                log_density, metadata = _particle_mass_density_grid(
                    snapshot_sources[current_snap][1], selected_ptype,
                    box_mpc_h, density_grid_size,
                )
                spacing = box_mpc_h / density_grid_size
                volume = pv.ImageData(
                    dimensions=(density_grid_size,) * 3,
                    spacing=(spacing,) * 3,
                    origin=(0.5 * spacing,) * 3,
                )
                volume.point_data["log10_rho_over_mean"] = log_density.ravel(order="F")
                clim = np.percentile(log_density, [2.0, 99.7])
                if not np.all(np.isfinite(clim)) or clim[1] <= clim[0]:
                    clim = (float(np.min(log_density)), float(np.max(log_density)))
                if clim[1] <= clim[0]:
                    clim = (float(clim[0]) - 1.0e-6, float(clim[1]) + 1.0e-6)
                cached = (volume, (float(clim[0]), float(clim[1])), metadata)
                density_state["cache"][key] = cached
                density_state["cache_order"].append(key)
                while len(density_state["cache_order"]) > 2:
                    expired = density_state["cache_order"].pop(0)
                    density_state["cache"].pop(expired, None)
            volume, clim, metadata = cached
            density_state["grid"] = volume
            density_state["actor"] = plotter.add_volume(
                volume, scalars="log10_rho_over_mean", cmap="magma", clim=clim,
                opacity="sigmoid", opacity_unit_distance=box_mpc_h / density_grid_size,
                shade=True, show_scalar_bar=False, mapper="smart", render=False,
            )
            bar_title = r"$\log_{10}(\rho/\bar{\rho})$"
            plotter.add_scalar_bar(
                title=bar_title, cmap="magma", clim=clim, n_labels=5,
                vertical=True, position_x=0.88, position_y=0.16,
                width=0.085, height=0.62, title_font_size=13,
                label_font_size=12, color="#f8fafc", shadow=True,
                background_color="#111827", render=False,
            )
            density_state["scalar_bar_title"] = bar_title
            density_state["status"] = metadata
        except (OSError, KeyError, ValueError, TypeError, RuntimeError, ImportError) as exc:
            density_state["status"] = "unavailable: " + str(exc)

    def set_density_volume_visibility(visible):
        density_state["visible"] = bool(visible)
        refresh_density_volume(snap)
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_slice_visibility(visible):
        slice_state["visible"] = bool(visible)
        refresh_particle_slice()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_slice_center(value):
        slice_state["center_mpc_h"] = float(np.mod(float(value), box_mpc_h))
        refresh_particle_slice()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_slice_thickness(value):
        slice_state["thickness_mpc_h"] = float(np.clip(
            float(value), box_mpc_h / 500.0, box_mpc_h / 3.0,
        ))
        refresh_particle_slice()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_catalogue_visibility(kind, visible):
        state = catalogue_state[kind]
        state["visible"] = bool(visible)
        if state["actor"] is not None:
            plotter.remove_actor(state["actor"], render=False)
            state["actor"] = None
        if state["visible"]:
            data = _read_catalogue_layer(run_dir, snap, kind, mass_definition)
            add_catalogue_actor(kind, state, data)
        else:
            state["data"] = None
            state["catalog_rows"] = np.empty(0, dtype=np.int64)
            state["shown"] = state["matching"] = 0
            if picked_selection["kind"] == kind:
                clear_picked_catalogue_info(render=False)
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_r200_visibility(visible):
        r200_state["visible"] = bool(visible)
        add_r200_shells(snap)
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def refresh_filament_layer(snapshot):
        if filament_state["actor"] is not None:
            plotter.remove_actor(filament_state["actor"], render=False)
            filament_state["actor"] = None
        filament_state["shown"] = 0
        filament_state["metadata"] = None
        if not filament_state["visible"]:
            filament_state["status"] = None
            return
        try:
            visualization = cfg.get("visualization", {})
            skeleton = _load_disperse_skeleton(
                run_dir, snapshot, int(visualization["filament_grid_size"]),
                float(visualization["disperse_persistence_cut"]),
            )
            polydata = _periodic_skeleton_polydata(skeleton, box_mpc_h)
            if polydata.n_cells == 0:
                filament_state["status"] = "no filament segments in this saved output"
                return
            tube = polydata.tube(radius=max(box_mpc_h * 0.00065, 1.0e-4), n_sides=7)
            filament_state["actor"] = plotter.add_mesh(
                tube, color=visualization["filament_color"], opacity=0.93, smooth_shading=True,
                render=False,
            )
            filament_state["shown"] = int(skeleton["metadata"].get("filament_polyline_count", 0))
            filament_state["metadata"] = skeleton["metadata"]
            filament_state["status"] = "%s polylines; cut %.3g field units" % (
                format(filament_state["shown"], ","),
                float(skeleton["metadata"]["persistence_cut_field_units"]),
            )
        except (OSError, KeyError, ValueError, TypeError, OverflowError, ImportError,
                RuntimeError, subprocess.SubprocessError) as exc:
            filament_state["status"] = "unavailable: " + str(exc)

    def set_filament_visibility(visible):
        filament_state["visible"] = bool(visible)
        refresh_filament_layer(snap)
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_minimum_halo_mass(value):
        if mass_limits is None:
            return
        mass_filter["log10_minimum"] = float(np.clip(float(value), *mass_limits))
        for kind, state in catalogue_state.items():
            if not state["visible"]:
                continue
            if state["actor"] is not None:
                plotter.remove_actor(state["actor"], render=False)
                state["actor"] = None
            add_catalogue_actor(
                kind, state,
                _read_catalogue_layer(run_dir, snap, kind, mass_definition),
            )
        if r200_state["visible"]:
            add_r200_shells(snap)
        selected_kind = picked_selection["kind"]
        selected_row = picked_selection["row"]
        if selected_kind is not None:
            selected_state = catalogue_state.get(selected_kind, {})
            if (not selected_state.get("visible")
                    or selected_row not in set(selected_state.get("catalog_rows", []))):
                clear_picked_catalogue_info(render=False)
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def clear_velocity_actors():
        for key in ("arrow_actor", "streamline_actor"):
            actor = motion_state.get(key)
            if actor is not None:
                plotter.remove_actor(actor, render=False)
                motion_state[key] = None

    def refresh_velocity_layers():
        clear_velocity_actors()
        motion_state["velocity_status"] = None
        if not (motion_state["velocity_arrows"] or motion_state["streamlines"]):
            motion_state["velocity_status"] = None
            return
        state = layer_state[selected_ptype]
        if not state["visible"]:
            motion_state["velocity_status"] = "show the selected particle layer to display its velocity field"
            return
        grid, seed_cells, points, velocities, velocity_info = _velocity_grid_from_particles(
            current_files, selected_ptype, state.get("display_ids"), cap, box_mpc_h,
            int(cfg.get("visualization", {}).get("velocity_grid_size", 32)),
        )
        if isinstance(velocity_info, str):
            motion_state["velocity_status"] = velocity_info
            return
        mass_weighting = velocity_info.get("mass_weighting", "mass-weighted")
        maximum_seeds = int(cfg.get("visualization", {}).get("max_velocity_seeds_3d", 800))
        if motion_state["velocity_arrows"] and len(points):
            arrow_indices = np.arange(len(points))
            if len(arrow_indices) > maximum_seeds:
                arrow_indices = np.linspace(0, len(arrow_indices) - 1, maximum_seeds, dtype=int)
            vectors = velocities[arrow_indices]
            speed = np.linalg.norm(vectors, axis=1)
            nonzero = np.isfinite(speed) & (speed > 0)
            if np.any(nonzero):
                median_speed = max(float(np.median(speed[nonzero])), np.finfo(float).tiny)
                relative_length = np.clip(speed[nonzero] / median_speed, 0.3, 2.5)
                normalized = vectors[nonzero] / speed[nonzero, None] * relative_length[:, None]
                cloud = pv.PolyData(points[arrow_indices[nonzero]].astype(np.float32, copy=False))
                cloud.point_data["Velocity"] = normalized.astype(np.float32, copy=False)
                arrows = cloud.glyph(
                    orient="Velocity", scale="Velocity",
                    factor=box_mpc_h * motion_state["arrow_scale_box_fraction"],
                )
                motion_state["arrow_actor"] = plotter.add_mesh(
                    arrows, color="#ffcc66", opacity=0.85, lighting=False,
                    render=False,
                )
        if motion_state["streamlines"] and seed_cells is not None:
            if len(seed_cells) > maximum_seeds:
                seed_ids = np.linspace(0, len(seed_cells) - 1, maximum_seeds, dtype=int)
                seed_cells = seed_cells[seed_ids]
            try:
                streamlines = grid.streamlines_from_source(
                    pv.PolyData(np.asarray(seed_cells, dtype=np.float32)),
                    vectors="Velocity", integration_direction="both",
                    initial_step_length=0.35, max_step_length=0.8,
                    max_steps=700, max_length=box_mpc_h * 1.5,
                    terminal_speed=1.0e-8,
                    compute_vorticity=False,
                )
                if streamlines.n_points:
                    motion_state["streamline_actor"] = plotter.add_mesh(
                        streamlines, color="#39d9d2", line_width=1.35,
                        opacity=0.82, render=False,
                    )
            except (RuntimeError, ValueError, TypeError) as exc:
                motion_state["velocity_status"] = "streamline integration failed: " + str(exc)
        if motion_state["velocity_status"] is None:
            valid_speed = np.linalg.norm(velocities, axis=1)
            valid_speed = valid_speed[np.isfinite(valid_speed) & (valid_speed > 0)]
            median_code_speed = float(np.median(valid_speed)) if len(valid_speed) else float("nan")
            conversion = _velocity_conversion_km_s(current_files, selected_ptype, current_header)
            speed_label = (
                "median %.4g km/s" % (median_code_speed * conversion)
                if conversion is not None and np.isfinite(median_code_speed)
                else "median %.4g snapshot units" % median_code_speed
            )
            motion_state["velocity_status"] = (
                "%s velocity field; %s; arrow length at median speed is %.3g of box"
                % (mass_weighting, speed_label, motion_state["arrow_scale_box_fraction"])
            )

    def set_velocity_arrows(visible):
        motion_state["velocity_arrows"] = bool(visible)
        refresh_velocity_layers()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_streamlines(visible):
        motion_state["streamlines"] = bool(visible)
        refresh_velocity_layers()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_velocity_arrow_scale(value):
        motion_state["arrow_scale_box_fraction"] = float(np.clip(float(value), 0.005, 0.12))
        if motion_state["velocity_arrows"]:
            refresh_velocity_layers()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def build_rockstar_follow_centers(anchor_snapshot=None, anchor_halo_id=None):
        from analysis import rockstar_host_table

        configured_mass_definition = str(
            load_cfg(run_dir)["analysis"]["hmf_mass_definition"]
        ).lower()
        configured_mass_field = {
            "fof": "mfof", "m200c": "m200c", "m200b": "m200b", "mvir": "mvir",
        }.get(configured_mass_definition)
        catalogs = []
        for snapshot in available:
            table = rockstar_host_table(run_dir, snapshot)
            if not table or not all(name in table for name in ("id", "x", "y", "z")):
                catalogs.append(None)
                continue
            count = len(table["id"])
            parent = np.asarray(table.get("pid", np.full(count, -1)), dtype=np.int64)
            host = parent < 0
            ids = np.asarray(table["id"], dtype=np.int64)[host]
            positions = np.column_stack([np.asarray(table[key], dtype=float) for key in ("x", "y", "z")])[host]
            positions = np.mod(positions, box_mpc_h)
            if configured_mass_field is None or configured_mass_field not in table:
                catalogs.append(None)
                continue
            mass_name = configured_mass_field
            masses = np.asarray(table[mass_name], dtype=float)[host]
            desc_name = "desc_id" if "desc_id" in table else "descid" if "descid" in table else None
            desc_ids = np.asarray(table[desc_name], dtype=np.int64)[host] if desc_name else np.full(len(ids), -1, dtype=np.int64)
            desc_scale_name = "desc_scale" if "desc_scale" in table else "descscale" if "descscale" in table else None
            desc_scales = (
                np.asarray(table[desc_scale_name], dtype=float)[host]
                if desc_scale_name else np.full(len(ids), np.nan)
            )
            radius_scale_to_mpc_h = _rockstar_radius_scale_to_mpc_h(table)
            good = (ids >= 0) & np.all(np.isfinite(positions), axis=1) & np.isfinite(masses) & (masses > 0)
            catalogs.append({
                "ids": ids[good], "positions": positions[good], "masses": masses[good],
                "mass_field": mass_name,
                "desc_ids": desc_ids[good],
                "desc_scales": desc_scales[good],
                "has_desc_ids": desc_name is not None,
                "radius_scale_to_mpc_h": radius_scale_to_mpc_h,
                "r200c": np.asarray(table.get("r200c", np.full(count, np.nan)), dtype=float)[host][good]
                if "r200c" in table else np.full(np.count_nonzero(good), np.nan),
                "rvir": np.asarray(table.get("rvir", np.full(count, np.nan)), dtype=float)[host][good]
                if "rvir" in table else np.full(np.count_nonzero(good), np.nan),
            })
        valid_indices = [index for index, table in enumerate(catalogs) if table is not None and len(table["ids"])]
        if not valid_indices:
            return None, None
        descid_catalogue_count = sum(
            1 for table in catalogs if table is not None and table["has_desc_ids"]
        )
        if anchor_snapshot is not None and anchor_halo_id is not None:
            anchor_index = available.index(int(anchor_snapshot))
            anchor_catalogue = catalogs[anchor_index]
            matches = (np.flatnonzero(anchor_catalogue["ids"] == int(anchor_halo_id))
                       if anchor_catalogue is not None else np.empty(0, dtype=np.int64))
            if not len(matches):
                return None, None
            anchor_row = int(matches[np.argmax(anchor_catalogue["masses"][matches])])
            selection_mode = "selected Rockstar ID %d" % int(anchor_halo_id)
        else:
            anchor_index = valid_indices[-1]
            anchor_row = int(np.argmax(catalogs[anchor_index]["masses"]))
            selection_mode = "most massive Rockstar host at output %03d" % available[anchor_index]
        anchor = catalogs[anchor_index]
        track_rows = {anchor_index: anchor_row}

        def matching_descendant_rows(catalogue, descendant_id, destination_index):
            rows = np.flatnonzero(catalogue["desc_ids"] == int(descendant_id))
            if not len(rows):
                return rows
            scales = catalogue["desc_scales"][rows]
            finite = np.isfinite(scales)
            if np.any(finite):
                scale_ok = ~finite | np.isclose(
                    scales, scale_factors[destination_index], rtol=1.0e-5, atol=1.0e-7,
                )
                rows = rows[scale_ok]
            return rows

        current_id = int(anchor["ids"][anchor_row])
        for index in range(anchor_index - 1, -1, -1):
            previous = catalogs[index]
            if previous is None:
                break
            matches = matching_descendant_rows(previous, current_id, index + 1)
            if not len(matches):
                break
            row = int(matches[np.argmax(previous["masses"][matches])])
            track_rows[index] = row
            current_id = int(previous["ids"][row])

        current_index = anchor_index
        current_row = anchor_row
        while current_index + 1 < len(available):
            current_catalogue = catalogs[current_index]
            following = catalogs[current_index + 1]
            if current_catalogue is None or following is None:
                break
            descendant_id = int(current_catalogue["desc_ids"][current_row])
            if descendant_id < 0:
                break
            if np.isfinite(current_catalogue["desc_scales"][current_row]) and not np.isclose(
                current_catalogue["desc_scales"][current_row], scale_factors[current_index + 1],
                rtol=1.0e-5, atol=1.0e-7,
            ):
                break
            matches = np.flatnonzero(following["ids"] == descendant_id)
            if not len(matches):
                break
            current_index += 1
            current_row = int(matches[np.argmax(following["masses"][matches])])
            track_rows[current_index] = current_row

        centers = {}
        records = {}
        for index, row in track_rows.items():
            snapshot = available[index]
            catalogue = catalogs[index]
            centers[snapshot] = catalogue["positions"][row]
            records[snapshot] = {
                "rockstar_id": int(catalogue["ids"][row]),
                "mass_msun_h": float(catalogue["masses"][row]),
                "mass_field": catalogue["mass_field"],
                "r200c_mpc_h": (
                    float(catalogue["r200c"][row] * catalogue["radius_scale_to_mpc_h"])
                    if catalogue["radius_scale_to_mpc_h"] is not None
                    and np.isfinite(catalogue["r200c"][row]) else None
                ),
                "rvir_mpc_h": (
                    float(catalogue["rvir"][row] * catalogue["radius_scale_to_mpc_h"])
                    if catalogue["radius_scale_to_mpc_h"] is not None
                    and np.isfinite(catalogue["rvir"][row]) else None
                ),
            }
        merger_events = []
        for index, main_row in track_rows.items():
            following_index = index + 1
            if following_index not in track_rows or catalogs[index] is None or catalogs[following_index] is None:
                continue
            descendant_id = int(catalogs[following_index]["ids"][track_rows[following_index]])
            progenitor_catalogue = catalogs[index]
            if descendant_id < 0:
                continue
            progenitor_rows = matching_descendant_rows(progenitor_catalogue, descendant_id, following_index)
            if main_row not in progenitor_rows:
                continue
            secondary_rows = progenitor_rows[progenitor_rows != main_row]
            if len(secondary_rows) > 25:
                keep = np.argpartition(progenitor_catalogue["masses"][secondary_rows], -25)[-25:]
                secondary_rows = secondary_rows[keep]
            for secondary_row in secondary_rows:
                main_mass = float(progenitor_catalogue["masses"][main_row])
                secondary_mass = float(progenitor_catalogue["masses"][secondary_row])
                merger_events.append({
                    "progenitor_snapshot": int(available[index]),
                    "descendant_snapshot": int(available[following_index]),
                    "descendant_id": descendant_id,
                    "main_progenitor_id": int(progenitor_catalogue["ids"][main_row]),
                    "secondary_progenitor_id": int(progenitor_catalogue["ids"][secondary_row]),
                    "secondary_mass_msun_h": secondary_mass,
                    "main_mass_msun_h": main_mass,
                    "mass_ratio_to_main": secondary_mass / main_mass if main_mass > 0 else None,
                    "secondary_position_mpc_h": progenitor_catalogue["positions"][secondary_row].tolist(),
                })
        merger_events_by_snapshot = {}
        for event in merger_events:
            merger_events_by_snapshot.setdefault(event["progenitor_snapshot"], []).append(event)
        track = {
            "selection": selection_mode,
            "anchor_snapshot": int(available[anchor_index]),
            "anchor_rockstar_id": int(anchor["ids"][anchor_row]),
            "link_method": "exact Rockstar DescID descendant links",
            "matched_output_count": int(len(track_rows)),
            "total_output_count": int(len(available)),
            "exact_descendant_link_count": int(max(len(track_rows) - 1, 0)),
            "catalogues_with_desc_ids": int(descid_catalogue_count),
            "merger_event_count": int(len(merger_events)),
            "merger_events_by_progenitor_snapshot": merger_events_by_snapshot,
            "records_by_snapshot": records,
        }
        return centers, track

    def sync_follow_widget(visible):
        widget = motion_state.get("follow_widget")
        if widget is None:
            return
        try:
            widget.GetRepresentation().SetState(1 if visible else 0)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def update_merger_progenitors():
        actor = motion_state.get("merger_marker_actor")
        track = motion_state.get("follow_track")
        if not motion_state["follow"] or track is None:
            if actor is not None:
                plotter.remove_actor(actor, render=False)
                motion_state["merger_marker_actor"] = None
            motion_state["merger_status"] = None
            return
        events = track["merger_events_by_progenitor_snapshot"].get(snap, [])
        if not events:
            if actor is not None:
                plotter.remove_actor(actor, render=False)
                motion_state["merger_marker_actor"] = None
            if track["catalogues_with_desc_ids"]:
                motion_state["merger_status"] = "No other host DescID progenitors at this saved output"
            else:
                motion_state["merger_status"] = "DescID unavailable; merger candidates were not evaluated"
            return
        positions = np.mod(
            np.asarray([event["secondary_position_mpc_h"] for event in events], dtype=float),
            box_mpc_h,
        )
        actor = motion_state.get("merger_marker_actor")
        if actor is not None:
            plotter.remove_actor(actor, render=False)
        motion_state["merger_marker_actor"] = plotter.add_points(
            positions.astype(np.float32, copy=False), color="#ff8a65", point_size=12,
            render_points_as_spheres=True, render=False,
        )
        motion_state["merger_status"] = (
            "%d secondary host(s) point by DescID to the followed halo at output %03d; orange marks are progenitor positions"
            % (len(events), events[0]["descendant_snapshot"])
        )

    def move_camera_to_follow(center, reset=False):
        center = np.mod(np.asarray(center, dtype=float), box_mpc_h)
        if reset or motion_state["follow_center"] is None:
            plotter.camera.focal_point = center
            plotter.camera.position = center + np.asarray([1.55, -1.8, 1.35]) * box_mpc_h
            plotter.camera.view_up = (0.0, 0.0, 1.0)
        else:
            delta = center - motion_state["follow_center"]
            delta -= box_mpc_h * np.round(delta / box_mpc_h)
            plotter.camera.focal_point = np.asarray(plotter.camera.focal_point) + delta
            plotter.camera.position = np.asarray(plotter.camera.position) + delta
        motion_state["follow_center"] = center

    def update_follow_marker(center):
        actor = motion_state.get("follow_marker_actor")
        if center is None:
            if actor is not None:
                plotter.remove_actor(actor, render=False)
                motion_state["follow_marker_actor"] = None
            return
        point = np.asarray(center, dtype=np.float32).reshape(1, 3)
        if actor is None:
            cloud = pv.PolyData(point)
            motion_state["follow_marker_actor"] = plotter.add_points(
                cloud, color="#ffffff", point_size=18, render_points_as_spheres=True,
                render=False,
            )

            return
        try:
            dataset = actor.mapper.dataset
            dataset.points = point
            dataset.Modified()
        except (AttributeError, RuntimeError, TypeError):
            plotter.remove_actor(actor, render=False)
            cloud = pv.PolyData(point)
            motion_state["follow_marker_actor"] = plotter.add_points(
                cloud, color="#ffffff", point_size=18, render_points_as_spheres=True,
                render=False,
            )

    def update_follow_shell(center):
        actor = motion_state.get("follow_shell_actor")
        if actor is not None:
            plotter.remove_actor(actor, render=False)
            motion_state["follow_shell_actor"] = None
        track = motion_state.get("follow_track")
        if not motion_state["follow"] or track is None or center is None:
            return
        record = track["records_by_snapshot"].get(snap)
        radius = record.get("r200c_mpc_h") if record is not None else None
        if radius is None or not np.isfinite(radius) or radius <= 0 or radius >= box_mpc_h / 2.0:
            return
        shell = pv.Sphere(
            radius=float(radius), center=np.asarray(center, dtype=float),
            theta_resolution=28, phi_resolution=28,
        )
        motion_state["follow_shell_actor"] = plotter.add_mesh(
            shell, color="#f8fafc", style="wireframe", line_width=1.0,
            opacity=0.62, render=False,
        )

    def update_follow_trail(center):
        actor = motion_state.get("follow_trail_actor")
        if actor is not None:
            plotter.remove_actor(actor, render=False)
            motion_state["follow_trail_actor"] = None
        track = motion_state.get("follow_track")
        if not motion_state["follow"] or track is None or center is None:
            return
        records = track["records_by_snapshot"]
        centers = motion_state.get("follow_centers") or {}
        ordered = [snapshot for snapshot in available[:current_index + 1] if snapshot in records]
        if snap not in records or len(ordered) < 2:
            return
        wrapped = np.asarray([centers[snapshot] for snapshot in ordered], dtype=float)
        unwrapped = np.empty_like(wrapped)
        unwrapped[0] = wrapped[0]
        for index in range(1, len(wrapped)):
            delta = wrapped[index] - wrapped[index - 1]
            delta -= box_mpc_h * np.round(delta / box_mpc_h)
            unwrapped[index] = unwrapped[index - 1] + delta
        target = np.asarray(center, dtype=float)
        unwrapped += target - unwrapped[-1]
        if motion_state["fraction"] > 0 and current_index + 1 < len(available):
            unwrapped = np.vstack([unwrapped, target])
        if len(unwrapped) < 2:
            return
        path = pv.PolyData(unwrapped.astype(np.float32, copy=False))
        path.lines = np.concatenate(([len(unwrapped)], np.arange(len(unwrapped)))).astype(np.int64)
        motion_state["follow_trail_actor"] = plotter.add_mesh(
            path, color="#4dd0e1", line_width=3.0, render=False,
        )

    def remove_growth_chart():
        chart = motion_state.get("growth_chart")
        if chart is not None:
            try:
                plotter.remove_chart(chart)
            except (AttributeError, RuntimeError, TypeError):
                pass
        motion_state["growth_chart"] = None
        motion_state["growth_chart_snapshot"] = None
        motion_state["growth_chart_track"] = None

    def update_growth_chart():
        track = motion_state.get("follow_track")
        if not motion_state["follow"] or track is None:
            remove_growth_chart()
            return
        if (motion_state.get("growth_chart") is not None
                and motion_state.get("growth_chart_snapshot") == snap
                and motion_state.get("growth_chart_track") is track):
            return
        remove_growth_chart()
        samples = []
        for snapshot in available:
            record = track["records_by_snapshot"].get(snapshot)
            if record is None or not np.isfinite(record["mass_msun_h"]) or record["mass_msun_h"] <= 0:
                continue
            index = available.index(snapshot)
            samples.append((float(scale_factors[index]), float(record["mass_msun_h"]), record["mass_field"]))
        if len(samples) < 2 or len({row[2] for row in samples}) != 1:
            return
        x = np.asarray([row[0] for row in samples], dtype=float)
        y = np.log10(np.asarray([row[1] for row in samples], dtype=float))
        chart = pv.Chart2D(size=(0.21, 0.23), loc=(0.64, 0.035))
        chart.background_color = "#0b1220"
        chart.border_color = "#64748b"
        chart.title = "Halo mass growth"
        chart.x_axis.SetTitle("Scale factor a")
        chart.y_axis.SetTitle("log10 Mass [M_sun/h]")
        chart.x_axis.SetGridVisible(False)
        chart.y_axis.SetGridVisible(False)
        chart.GetTitleProperties().SetColor(0.9, 0.93, 0.97)
        chart.GetTitleProperties().SetFontSize(12)
        chart.GetTitleProperties().SetBold(True)
        chart.legend.visible = False
        chart.line(x, y, color="#4dd0e1", width=2.2)
        current = track["records_by_snapshot"].get(snap)
        if current is not None and np.isfinite(current["mass_msun_h"]) and current["mass_msun_h"] > 0:
            chart.scatter(
                [float(scale_factors[current_index])], [float(np.log10(current["mass_msun_h"]))],
                color="#ffffff", size=9,
            )
        for axis in (chart.x_axis, chart.y_axis):
            axis.GetLabelProperties().SetColor(0.9, 0.93, 0.97)
            axis.GetTitleProperties().SetColor(0.9, 0.93, 0.97)
        plotter.add_chart(chart)
        motion_state["growth_chart"] = chart
        motion_state["growth_chart_snapshot"] = snap
        motion_state["growth_chart_track"] = track

    def set_follow_camera(visible):
        motion_state["follow"] = bool(visible)
        if motion_state["follow"]:
            selected_id = (picked_selection.get("halo_id")
                           if picked_selection.get("kind") == "rockstar" else None)
            if selected_id is not None:
                motion_state["follow_anchor_halo_id"] = int(selected_id)
                motion_state["follow_anchor_snapshot"] = int(snap)
            else:
                selected_id = motion_state.get("follow_anchor_halo_id")
            anchor = (
                int(snap) if picked_selection.get("kind") == "rockstar"
                else motion_state.get("follow_anchor_snapshot")
            ) if selected_id is not None else None
            motion_state["follow_centers"], motion_state["follow_track"] = build_rockstar_follow_centers(
                anchor_snapshot=anchor, anchor_halo_id=selected_id,
            )
            if not motion_state["follow_centers"]:
                motion_state["follow"] = False
                motion_state["follow_status"] = "No exact Rockstar DescID track is available for this selection"
                motion_state["follow_centers"] = None
                motion_state["follow_track"] = None
                update_follow_marker(None)
                update_follow_shell(None)
                update_follow_trail(None)
                update_merger_progenitors()
                update_growth_chart()
                sync_follow_widget(False)
            else:
                center = motion_state["follow_centers"].get(snap)
                if center is None:
                    motion_state["follow"] = False
                    motion_state["follow_status"] = "The exact Rockstar track does not include this saved output"
                    motion_state["follow_centers"] = None
                    motion_state["follow_track"] = None
                    update_follow_marker(None)
                    update_follow_shell(None)
                    update_follow_trail(None)
                    update_merger_progenitors()
                    update_growth_chart()
                    sync_follow_widget(False)
                else:
                    move_camera_to_follow(center, reset=True)
                    update_follow_marker(center)
                    update_follow_shell(center)
                    update_follow_trail(center)
                    track = motion_state["follow_track"]
                    links = track["exact_descendant_link_count"]
                    if links:
                        motion_state["follow_status"] = (
                            "%s; %d exact DescID links across %d/%d outputs"
                            % (track["selection"], links, track["matched_output_count"],
                               track["total_output_count"])
                        )
                    elif track["catalogues_with_desc_ids"]:
                        motion_state["follow_status"] = (
                            "%s; DescID columns exist, but no adjacent exact link was found"
                            % track["selection"]
                        )
                    else:
                        motion_state["follow_status"] = (
                            "%s; DescID unavailable, showing this output only"
                            % track["selection"]
                        )
                    update_merger_progenitors()
                    update_growth_chart()
        else:
            motion_state["follow_center"] = None
            motion_state["follow_centers"] = None
            motion_state["follow_track"] = None
            motion_state["follow_status"] = None
            update_follow_marker(None)
            update_follow_shell(None)
            update_follow_trail(None)
            update_merger_progenitors()
            update_growth_chart()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def update_camera_follow(fraction=0.0):
        centers = motion_state.get("follow_centers")
        if not motion_state["follow"] or not centers:
            return
        if snap not in centers:
            motion_state["follow_status"] = "Exact DescID track has a gap at output %03d; camera follow is paused" % snap
            update_follow_marker(None)
            update_follow_shell(None)
            update_follow_trail(None)
            update_merger_progenitors()
            update_growth_chart()
            return
        center = np.asarray(centers[snap], dtype=float)
        next_index = min(current_index + 1, len(available) - 1)
        next_center = centers.get(available[next_index])
        if next_center is not None and fraction > 0:
            delta = np.asarray(next_center) - center
            delta -= box_mpc_h * np.round(delta / box_mpc_h)
            center = np.mod(center + fraction * delta, box_mpc_h)
        elif fraction > 0 and next_index != current_index:
            motion_state["follow_status"] = "Exact DescID link is missing at the next output; halo position is held"
        update_follow_marker(center)
        update_follow_shell(center)
        move_camera_to_follow(center)
        update_follow_trail(center)
        update_merger_progenitors()
        update_growth_chart()

    def set_playback(visible):
        motion_state["playing"] = bool(visible)
        if motion_state["playing"] and len(available) > 1:
            if motion_state["timeline_value"] >= scale_factors[-1]:
                seek_scale_factor(float(scale_factors[0]))
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_loop(visible):
        motion_state["loop"] = bool(visible)
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def clear_timeline_event_actors():
        for state in layer_state.values():
            for actor in state.get("timeline_event_actors", []):
                plotter.remove_actor(actor, render=False)
            state["timeline_event_actors"] = []

    def restore_particle_layers():
        clear_timeline_event_actors()
        for state in layer_state.values():
            actor = state.get("actor")
            base_points = state.get("base_render_points")
            if actor is None or base_points is None or not len(base_points):
                continue
            try:
                dataset = actor.mapper.dataset
                dataset.points = np.asarray(base_points, dtype=np.float32).copy()
                dataset.Modified()
            except (AttributeError, RuntimeError, TypeError):
                continue

    def interpolate_particle_layers(fraction):
        clear_timeline_event_actors()
        if fraction <= 0.0 or current_index + 1 >= len(available):
            motion_state["interpolation_status"] = None
            return
        next_files = snapshot_sources[available[current_index + 1]][1]
        box_kpc_h = box_mpc_h * 1000.0
        summaries = []
        for ptype, state in layer_state.items():
            if not state["visible"]:
                continue
            limit = int(cfg.get("visualization", {}).get("max_interpolation_points", 250_000))
            if limit <= 0:
                motion_state["interpolation_status"] = "Interpolation is disabled because max_interpolation_points is not positive"
                return
            current_ids = np.asarray(state.get("render_ids") if state.get("render_ids") is not None else [], dtype=np.uint64)
            total_displayed = int(len(current_ids))
            sample_ids = state.get("sample_ids")
            if cap is None or sample_ids is None:
                if state.get("timeline_sample_ids") is None:
                    state["timeline_sample_ids"] = _sample_timeline_id_union(
                        run_dir, available, ptype, limit,
                    )
                sample_ids = state["timeline_sample_ids"]
            if state.get("interpolation_ids") is None:
                state["interpolation_ids"] = _lowest_hash_ids(sample_ids, limit)
            interval_ids = np.asarray(state["interpolation_ids"], dtype=np.uint64)
            if not len(interval_ids):
                summaries.append((state["name"], 0, total_displayed, 0, 0, total_displayed))
                continue
            first = _read_snapshot_fields(
                current_files, ptype, ["Coordinates"], cap, selected_ids=interval_ids,
            )
            second = _read_snapshot_fields(
                next_files, ptype, ["Coordinates"], cap, selected_ids=interval_ids,
            )
            ids_a, ids_b = first.get("ParticleIDs"), second.get("ParticleIDs")
            if ids_a is None or ids_b is None:
                continue
            ids_a = np.asarray(ids_a, dtype=np.uint64)
            ids_b = np.asarray(ids_b, dtype=np.uint64)
            current_interval_ids = np.intersect1d(current_ids, interval_ids, assume_unique=True)
            common_ids, rows_current, rows_b = np.intersect1d(
                current_interval_ids, ids_b, assume_unique=True, return_indices=True,
            )
            matched_current = len(common_ids)
            left_only = max(len(current_interval_ids) - matched_current, 0)
            right_only_ids = np.setdiff1d(ids_b, ids_a, assume_unique=True)
            right_only = len(right_only_ids)
            held_by_cap = max(total_displayed - len(current_interval_ids), 0)
            total_interpolated = 0
            actor = state.get("actor")
            if actor is not None and matched_current:
                rows_a = np.searchsorted(ids_a, common_ids)
                if (np.any(rows_a >= len(ids_a))
                        or not np.array_equal(ids_a[rows_a], common_ids)):
                    summaries.append((state["name"], 0, total_displayed, left_only, right_only, held_by_cap))
                    continue
                xyz_a = np.asarray(first["Coordinates"], dtype=float)[rows_a]
                xyz_b = np.asarray(second["Coordinates"], dtype=float)[rows_b]
                if xyz_a.shape == xyz_b.shape == (len(common_ids), 3):
                    rendered_indices = np.searchsorted(current_ids, common_ids)
                    if (np.all(rendered_indices < len(current_ids))
                            and np.array_equal(current_ids[rendered_indices], common_ids)):
                        interpolated = _interpolate_periodic_positions(
                            xyz_a, xyz_b, fraction, box_kpc_h,
                        ) / 1000.0
                        try:
                            dataset = actor.mapper.dataset
                            updated_points = np.asarray(dataset.points, dtype=np.float32).copy()
                            if len(updated_points) == len(current_ids):
                                updated_points[rendered_indices] = interpolated.astype(np.float32, copy=False)
                                dataset.points = updated_points
                                dataset.Modified()
                                total_interpolated = len(common_ids)
                        except (AttributeError, RuntimeError, TypeError):
                            pass
            # An ID that first exists at the right output has no defensible
            # path to interpolate.  Show it at its exact right-output
            # position and fade it in across this interval.
            if len(right_only_ids):
                rows_right = np.searchsorted(ids_b, right_only_ids)
                if (np.all(rows_right < len(ids_b))
                        and np.array_equal(ids_b[rows_right], right_only_ids)):
                    event_points = (
                        np.asarray(second["Coordinates"], dtype=np.float32)[rows_right]
                        / np.float32(1000.0)
                    )
                    finite = np.all(np.isfinite(event_points), axis=1)
                    if np.any(finite):
                        event_cloud = pv.PolyData(event_points[finite])
                        event_actor = plotter.add_points(
                            event_cloud,
                            color=layer_colors[state["name"]],
                            point_size=appearance_state["point_size"] + 0.5,
                            opacity=float(np.clip(fraction, 0.0, 1.0))
                            * appearance_state["opacity"],
                            render_points_as_spheres=False,
                            render=False,
                        )
                        state["timeline_event_actors"].append(event_actor)
            state["interpolation_counts"] = {
                "matched": int(total_interpolated),
                "current_only_held": int(left_only),
                "next_only_at_saved_output": int(right_only),
                "outside_interpolation_cap": int(held_by_cap),
                "displayed": total_displayed,
            }
            summaries.append((
                state["name"], total_interpolated, total_displayed,
                left_only, right_only, held_by_cap,
            ))
        motion_state["interpolation_status"] = "Approximate periodic ID interpolation | " + "; ".join(
            "%s %s/%s matched; %s left-only held until the output boundary; %s next-only fading in at right-output positions; %s beyond interpolation cap"
            % (name.upper(), format(matched, ","), format(displayed, ","),
               format(left_only, ","), format(right_only, ","), format(held, ","))
            for name, matched, displayed, left_only, right_only, held in summaries
        ) if summaries else "No visible particle IDs were available for interpolation"

    def describe_picked_catalogue(point, picker):
        dataset = picker.GetDataSet()
        point_id = int(picker.GetPointId())
        if dataset is None or point_id < 0 or point_id >= dataset.n_points:
            return
        point_data = dataset.point_data
        if "dms_catalog_kind" not in point_data or "dms_catalog_row" not in point_data:
            return
        kind_code = int(point_data["dms_catalog_kind"][point_id])
        kind = {0: "halos", 1: "subhalos", 2: "rockstar"}.get(kind_code)
        if kind is None or kind not in catalogue_state:
            return
        state = catalogue_state[kind]
        data = state.get("data")
        row = int(point_data["dms_catalog_row"][point_id])
        if not state["visible"] or data is None or row < 0 or row >= len(data["mass_msun_h"]):
            return
        picked_selection["kind"] = kind
        picked_selection["row"] = row
        picked_selection["halo_id"] = (
            int(data["RockstarID"][row])
            if kind == "rockstar" and data.get("RockstarID") is not None else None
        )
        if picked_selection["halo_id"] is not None:
            motion_state["follow_anchor_halo_id"] = picked_selection["halo_id"]
            motion_state["follow_anchor_snapshot"] = int(snap)
        else:
            motion_state["follow_anchor_halo_id"] = None
            motion_state["follow_anchor_snapshot"] = None
        position = data["position_mpc_h"][row]
        lines = [
            {"halos": "FOF host", "subhalos": "SUBFIND subhalo", "rockstar": "Rockstar halo"}[kind]
            + " catalog row %d" % row,
            "Position: (%.5g, %.5g, %.5g) Mpc/h" % tuple(position),
            "Selected mass: %.6g M_sun/h (%s)" % (data["mass_msun_h"][row], data["mass_field"]),
        ]
        if kind == "halos":
            if data.get("Group_M_Crit200") is not None:
                lines.append("M200c: %.6g M_sun/h" % (data["Group_M_Crit200"][row] * 1.0e10))
            if data.get("Group_R_Crit200") is not None:
                lines.append("R200c: %.6g Mpc/h" % (data["Group_R_Crit200"][row] / 1000.0))
            if data.get("Group_M_Mean200") is not None:
                lines.append("M200m: %.6g M_sun/h" % (data["Group_M_Mean200"][row] * 1.0e10))
            if data.get("particle_count") is not None:
                lines.append("Particles: %s" % format(int(data["particle_count"][row]), ","))
            if data.get("GroupNsubs") is not None:
                lines.append("Subhalos: %d" % int(data["GroupNsubs"][row]))
            if data.get("GroupVmax") is not None:
                lines.append("Vmax: %.6g [catalog units]" % data["GroupVmax"][row])
            elif data.get("GroupFirstSub") is not None:
                sub_data = _read_catalogue_layer(run_dir, snap, "subhalos", mass_definition)
                first_sub = int(data["GroupFirstSub"][row])
                if (sub_data is not None and sub_data.get("SubhaloVmax") is not None
                        and 0 <= first_sub < len(sub_data["SubhaloVmax"])):
                    lines.append("Central Vmax: %.6g [catalog units]" % sub_data["SubhaloVmax"][first_sub])
        elif kind == "subhalos":
            if data.get("SubhaloGroupNr") is not None:
                lines.append("Parent FOF row: %d" % int(data["SubhaloGroupNr"][row]))
            if data.get("SubhaloRankInGr") is not None:
                rank = int(data["SubhaloRankInGr"][row])
                lines.append("Group rank: %d (%s)" % (rank, "central" if rank == 0 else "satellite"))
            if data.get("SubhaloVmax") is not None:
                lines.append("Vmax: %.6g [catalog units]" % data["SubhaloVmax"][row])
            if data.get("SubhaloHalfmassRad") is not None:
                lines.append("Half-mass radius: %.6g Mpc/h" % (data["SubhaloHalfmassRad"][row] / 1000.0))
            if data.get("particle_count") is not None:
                lines.append("Particles: %s" % format(int(data["particle_count"][row]), ","))
            if data.get("SubhaloSpin") is not None:
                lines.append("|Spin|: %.6g [catalog units]" % np.linalg.norm(data["SubhaloSpin"][row]))
        else:
            if data.get("RockstarID") is not None:
                lines.append("Rockstar ID: %d" % int(data["RockstarID"][row]))
                source = data.get("source_table")
                raw_rows = data.get("row_index")
                if source is not None and raw_rows is not None:
                    raw_row = int(raw_rows[row])
                    desc_name = next((name for name in ("desc_id", "descid") if name in source), None)
                    if desc_name is not None and 0 <= raw_row < len(source[desc_name]):
                        lines.append("Descendant ID: %d" % int(source[desc_name][raw_row]))
            if data.get("pid") is not None:
                lines.append("Parent Rockstar ID: %d" % int(data["pid"][row]))
            for field, label in (("m200c", "M200c"), ("m200b", "M200b"), ("mvir", "Mvir")):
                values = data.get(field)
                if values is not None and np.isfinite(values[row]) and values[row] > 0:
                    lines.append("%s: %.6g M_sun/h" % (label, values[row]))
                    break
            for field, label in (("r200c", "R200c"), ("rvir", "Rvir")):
                values = data.get(field)
                if values is not None and np.isfinite(values[row]) and values[row] > 0:
                    radius_scale = data.get("radius_scale_to_mpc_h")
                    if radius_scale is None:
                        lines.append("%s: %.6g [catalogue units; unit header unavailable]" % (label, values[row]))
                    else:
                        lines.append("%s: %.6g Mpc/h" % (label, values[row] * radius_scale))
                    break
            if data.get("particle_count") is not None:
                lines.append("Particles: %s" % format(int(data["particle_count"][row]), ","))
        if motion_state["follow"]:
            set_follow_camera(True)
        plotter.add_text(
            "\n".join(lines), position=(420, 730), font_size=8,
            color="#f8fafc", name="picked_catalogue_info", render=False,
        )
        plotter.render()

    def update_snapshot_label(header, current_snap, render=False):
        time_value = float(motion_state.get("timeline_value", header.get("Time", np.nan)))
        if not np.isfinite(time_value) or time_value <= 0:
            time_value = _snapshot_scale_factor(header)
        redshift = 1.0 / time_value - 1.0
        cosmic_time_gyr = cosmic_time_for_scale_factor(time_value)
        cosmic_time_label = ("t=%.4g Gyr" % cosmic_time_gyr
                             if cosmic_time_gyr is not None else "t unavailable")
        cosmic_time_reason = cosmic_time_unavailable_reason or (
            "scale factor lies outside the CLASS proper-time table coverage"
        )
        active = [state["name"].upper() for state in layer_state.values() if state["visible"]]
        active.extend(kind.upper() for kind, state in catalogue_state.items() if state["visible"])
        if motion_state["velocity_arrows"]:
            active.append("VELOCITY ARROWS")
        if motion_state["streamlines"]:
            active.append("STREAMLINES")
        if motion_state["follow"]:
            active.append("FOLLOW HALO")
        if density_state["visible"]:
            active.append(selected_species_name.upper() + " DENSITY VOLUME (saved output)")
        if r200_state["visible"]:
            active.append("R200C (%d shells, %d periodic copies)" % (
                r200_state["shown"], r200_state["periodic_images"],
            ))
        if filament_state["visible"]:
            active.append("3-D DISPERSE FILAMENTS")
        if slice_state["visible"]:
            active.append("Z-SLAB HIGHLIGHT")
        shown = sum(state["shown"] for state in layer_state.values() if state["visible"])
        shown += sum(state["shown"] for state in catalogue_state.values() if state["visible"])
        shown += r200_state["shown"]
        current_source_files = snapshot_sources[current_snap][1]
        total = 0
        for ptype, state in layer_state.items():
            if state["visible"]:
                total += _snapshot_particle_type_total(header, current_source_files, ptype)
        total += sum(state["matching"] for state in catalogue_state.values() if state["visible"])
        if total <= 0:
            total = shown
        sample_label = ("particle cap %s/layer" % format(cap, ",")) if cap is not None else "full particle layers"
        layer_label = ", ".join(active) if active else "no layers"
        color_label = selected_species_name.upper() + ": " + color_state["field"]
        if color_state["field"] != "Uniform":
            primary_state = layer_state[selected_ptype]
            source_label = primary_state.get("color_label", "")
            if "derived non-star-forming gas" in source_label:
                color_label = selected_species_name.upper() + ": T[K] derived non-SF"
            elif "derived" in source_label:
                color_label = selected_species_name.upper() + ": T[K] derived"
            elif "stored" in source_label and "Temperature" in source_label:
                color_label = selected_species_name.upper() + ": T[K] stored"
            color_label += " (locked across outputs)" if color_state.get("locked_range") is not None else " (auto/frame)"
            if primary_state.get("color_valid_count", 0) == 0:
                color_label += " | values unavailable"
        if motion_state["fraction"] > 0:
            snapshot_label = "output %03d→%03d α=%.2f" % (
                current_snap, available[min(current_index + 1, len(available) - 1)], motion_state["fraction"],
            )
        else:
            snapshot_label = "output %03d" % current_snap
        if motion_state["fraction"] > 0:
            snapshot_label += " | catalogue layers at left saved output"
        if motion_state["playing"]:
            playback_clock = "cosmic time" if has_cosmic_time_timeline else "scale factor"
            snapshot_label += " | PLAYING ×%.2g (%s)" % (
                motion_state["playback_speed"], playback_clock,
            )
        if motion_state.get("loop"):
            snapshot_label += " | LOOP"
        plotter.add_text(
            "%s | a=%.5f | z=%.3f | %s\n%s | %s/%s objects | %s | %s"
            % (snapshot_label, time_value, redshift, cosmic_time_label,
               layer_label.replace(", ", "+"), format(shown, ","), format(total, ","),
               color_label, sample_label),
            font_size=8, color="#f8fafc", name="snapshot_info", position=(420, 915), render=render,
        )
        if cosmic_time_gyr is None:
            plotter.add_text(
                "Cosmic time unavailable: " + str(cosmic_time_reason),
                position=(420, 880), font_size=6, color="#94a3b8",
                name="cosmic_time_status", render=render,
            )
        else:
            plotter.remove_actor("cosmic_time_status", render=False)
        if mass_limits is not None:
            plotter.add_text(
                "catalogue mass >= %.3g M_sun/h" % (10.0 ** mass_filter["log10_minimum"]),
                position=(1150, 572), font_size=7, color="#f8fafc",
                name="catalogue_mass_info", render=render,
            )
        if motion_state.get("velocity_status"):
            plotter.add_text(
                "Motion field: " + str(motion_state["velocity_status"]),
                position=(420, 620), font_size=7, color="#dbeafe",
                name="motion_status", render=render,
            )
        else:
            plotter.remove_actor("motion_status", render=False)
        if motion_state.get("interpolation_status"):
            plotter.add_text(
                str(motion_state["interpolation_status"]),
                position=(420, 600), font_size=7, color="#cbd5e1",
                name="interpolation_status", render=render,
            )
        else:
            plotter.remove_actor("interpolation_status", render=False)
        if motion_state.get("follow_status"):
            follow_status = str(motion_state["follow_status"])
            track = motion_state.get("follow_track")
            record = (track or {}).get("records_by_snapshot", {}).get(current_snap)
            if record is not None:
                mass_label = str(record["mass_field"]).replace("m", "M", 1)
                follow_status += "; ID %d; %s=%.4g M_sun/h" % (
                    record["rockstar_id"], mass_label, record["mass_msun_h"],
                )
                if record.get("r200c_mpc_h") is not None:
                    follow_status += "; R200c=%.4g Mpc/h" % record["r200c_mpc_h"]
            plotter.add_text(
                "Halo follow: " + follow_status,
                position=(420, 560), font_size=7, color="#fde68a",
                name="halo_follow_status", render=render,
            )
        else:
            plotter.remove_actor("halo_follow_status", render=False)
        if slice_state["visible"]:
            plotter.add_text(
                "Z slab: center=%.4g, thickness=%.4g Mpc/h; highlighted %s particles"
                % (slice_state["center_mpc_h"], slice_state["thickness_mpc_h"],
                   format(slice_state["shown"], ",")),
                position=(420, 490), font_size=7, color="#a5f3fc",
                name="particle_slice_status", render=render,
            )
        else:
            plotter.remove_actor("particle_slice_status", render=False)
        if motion_state.get("merger_status"):
            plotter.add_text(
                "Merger candidates: " + str(motion_state["merger_status"]),
                position=(420, 540), font_size=7, color="#fdba74",
                name="merger_candidate_status", render=render,
            )
        else:
            plotter.remove_actor("merger_candidate_status", render=False)
        if density_state.get("visible") and density_state.get("status"):
            volume_status = density_state["status"]
            if isinstance(volume_status, dict):
                volume_status = "%s CIC, %d^3 grid, %s particles; field at saved output" % (
                    selected_species_name.upper(), volume_status["grid_size"],
                    format(volume_status["particle_count"], ","),
                )
            plotter.add_text(
                "Density volume: " + str(volume_status),
                position=(420, 580), font_size=7, color="#cbd5e1",
                name="density_volume_status", render=render,
            )
        else:
            plotter.remove_actor("density_volume_status", render=False)
        if filament_state["visible"] and filament_state["status"]:
            plotter.add_text(
                "3-D filaments: " + str(filament_state["status"]),
                position=(420, 510), font_size=7,
                color="#67e8f9" if filament_state["actor"] is not None else "#fbbf24",
                name="filament_layer_status", render=render,
            )
        else:
            plotter.remove_actor("filament_layer_status", render=False)
    def set_layer_visibility(ptype, visible):
        state = layer_state[ptype]
        state["visible"] = bool(visible)
        if state["visible"] and state["actor"] is None:
            if state["sample_ids"] is None:
                xyz, sample_ids, display_ids = load_stable_sample(ptype, current_index)
                state["sample_ids"] = sample_ids
                state["display_ids"] = display_ids
            else:
                xyz, display_ids = read_layer_sample(ptype, state, current_files)
                state["display_ids"] = display_ids
            state["shown"] = len(xyz)
            state["actor"] = add_layer_actor(ptype, state, xyz, current_files, current_header)
        elif not state["visible"] and state["actor"] is not None:
            plotter.remove_actor(state["actor"], render=False)
            state["actor"] = None
            state["shown"] = 0
            if ptype == selected_ptype:
                remove_color_bar()
        if ptype == selected_ptype and (motion_state["velocity_arrows"] or motion_state["streamlines"]):
            refresh_velocity_layers()
        refresh_particle_slice()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def select_color_field(field):
        if field not in color_fields:
            raise ValueError("Unsupported 3-D color field: " + str(field))
        if field == color_state["field"]:
            return
        color_state["field"] = field
        color_state["locked_range"] = None
        if color_state.get("lock_widget") is not None:
            try:
                color_state["lock_widget"].GetRepresentation().SetState(0)
            except (AttributeError, RuntimeError, TypeError):
                pass
        state = layer_state[selected_ptype]
        if state["visible"]:
            if state["actor"] is not None:
                plotter.remove_actor(state["actor"], render=False)
                state["actor"] = None
            xyz, display_ids = read_layer_sample(selected_ptype, state, current_files)
            state["display_ids"] = display_ids
            state["shown"] = len(xyz)
            state["actor"] = add_layer_actor(selected_ptype, state, xyz, current_files, current_header)
        else:
            remove_color_bar()
        refresh_particle_slice()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    def set_locked_color_range(locked):
        state = layer_state[selected_ptype]
        if locked:
            current_range = state.get("color_range")
            if (color_state["field"] == "Uniform" or current_range is None
                    or len(current_range) != 2):
                color_state["locked_range"] = None
                if color_state.get("lock_widget") is not None:
                    try:
                        color_state["lock_widget"].GetRepresentation().SetState(0)
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                plotter.add_text(
                    "Choose an available scalar field before locking its range",
                    position=(420, 560), font_size=7, color="#fbbf24",
                    name="color_range_status", render=False,
                )
                plotter.render()
                return
            color_state["locked_range"] = (float(current_range[0]), float(current_range[1]))
        else:
            color_state["locked_range"] = None
            plotter.remove_actor("color_range_status", render=False)
        if state["visible"]:
            if state["actor"] is not None:
                plotter.remove_actor(state["actor"], render=False)
                state["actor"] = None
            xyz, display_ids = read_layer_sample(selected_ptype, state, current_files)
            state["display_ids"] = display_ids
            state["actor"] = add_layer_actor(selected_ptype, state, xyz, current_files, current_header)
        refresh_particle_slice()
        update_snapshot_label(current_header, snap, render=False)
        plotter.render()

    update_snapshot_label(current_header, snap)
    layer_y = 900
    plotter.add_text("LAYERS", position=(52, layer_y + 48), font_size=12, color="#f8fafc", name="layer_heading")
    for index, (name, ptype) in enumerate(available_types):
        y = layer_y - 48 * index
        plotter.add_checkbox_button_widget(
            lambda value, selected=ptype: set_layer_visibility(selected, value),
            value=(ptype == selected_ptype), position=(12, y), size=28,
            color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
        )
        plotter.add_text(name.upper(), position=(52, y + 5), font_size=11, color="#f8fafc", name="layer_label_" + name)
    catalogue_y = layer_y - 48 * len(available_types) - 32
    if catalogue_state:
        plotter.add_text("CATALOGS", position=(52, catalogue_y + 38), font_size=12, color="#f8fafc", name="catalogue_heading")
        for index, kind in enumerate(catalogue_state):
            y = catalogue_y - 42 * index
            plotter.add_checkbox_button_widget(
                lambda value, selected=kind: set_catalogue_visibility(selected, value),
                value=False, position=(12, y), size=28,
                color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
            )
            plotter.add_text(kind.upper(), position=(52, y + 5), font_size=11, color="#f8fafc", name="catalogue_label_" + kind)
        clear_picked_catalogue_info(render=False)
        plotter.enable_point_picking(
            callback=describe_picked_catalogue, picker="point", tolerance=0.025,
            left_clicking=True, show_message=False,
            use_picker=True, show_point=False,
        )
        r200_y = catalogue_y - 42 * len(catalogue_state) - 30
        plotter.add_checkbox_button_widget(
            set_r200_visibility, value=False, position=(12, r200_y), size=28,
            color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
        )
        plotter.add_text("R200C SHELLS", position=(52, r200_y + 5), font_size=10,
                         color="#f8fafc", name="r200_shell_label")
    else:
        r200_y = catalogue_y
    filament_y = r200_y - 42
    plotter.add_checkbox_button_widget(
        set_filament_visibility, value=False, position=(12, filament_y), size=25,
        color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
    )
    plotter.add_text("3-D FILAMENT SKELETON (DISPERSE)",
                     position=(52, filament_y + 4), font_size=8,
                     color="#f8fafc", name="filament_layer_label")
    density_y = filament_y - 42
    plotter.add_checkbox_button_widget(
        set_density_volume_visibility, value=False,
        position=(12, density_y), size=25,
        color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
    )
    plotter.add_text(selected_species_name.upper() + " DENSITY VOLUME",
                     position=(52, density_y + 4), font_size=9,
                     color="#f8fafc", name="density_volume_label")
    slice_y = density_y - 42
    plotter.add_checkbox_button_widget(
        set_slice_visibility, value=False, position=(12, slice_y), size=25,
        color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
    )
    plotter.add_text("HIGHLIGHT PERIODIC Z SLAB", position=(52, slice_y + 4), font_size=8,
                     color="#f8fafc", name="particle_slice_label")
    motion_y = slice_y - 42
    plotter.add_checkbox_button_widget(
        set_velocity_arrows, value=False, position=(12, motion_y), size=25,
        color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
    )
    plotter.add_text("VELOCITY ARROWS", position=(52, motion_y + 4), font_size=9,
                     color="#f8fafc", name="velocity_arrow_label")
    plotter.add_checkbox_button_widget(
        set_streamlines, value=False, position=(12, motion_y - 38), size=25,
        color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
    )
    plotter.add_text("VELOCITY STREAMLINES", position=(52, motion_y - 34), font_size=9,
                     color="#f8fafc", name="velocity_streamline_label")
    if len(available) > 1:
        motion_state["play_widget"] = plotter.add_checkbox_button_widget(
            set_playback, value=False, position=(12, motion_y - 76), size=25,
            color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
        )
        plotter.add_text("PLAY / INTERPOLATE", position=(52, motion_y - 72), font_size=9,
                         color="#f8fafc", name="playback_label")
        motion_state["loop_widget"] = plotter.add_checkbox_button_widget(
            set_loop, value=motion_state["loop"], position=(12, motion_y - 114), size=25,
            color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
        )
        plotter.add_text("LOOP TIMELINE", position=(52, motion_y - 110), font_size=9,
                         color="#f8fafc", name="timeline_loop_label")
    if "rockstar" in catalogue_state:
        motion_state["follow_widget"] = plotter.add_checkbox_button_widget(
            set_follow_camera, value=False, position=(12, motion_y - 152), size=25,
            color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
        )
        plotter.add_text("FOLLOW SELECTED ROCKSTAR (OR MAIN HOST)", position=(52, motion_y - 148), font_size=8,
                         color="#f8fafc", name="follow_camera_label")
    plotter.add_text("COLOR %s BY" % selected_species_name.upper(), position=(1150, 975), font_size=8, color="#f8fafc", name="color_field_heading")
    color_field_widget = plotter.add_text_slider_widget(
        select_color_field, color_fields, value=color_fields.index("Uniform"),
        pointa=(0.78, 0.91), pointb=(0.97, 0.91), color="#8ba4bd",
        style="modern", interaction_event="end",
    )
    color_representation = color_field_widget.GetRepresentation()
    color_representation.SetTitleHeight(0.014)
    color_representation.SetTubeWidth(0.006)
    color_representation.SetSliderWidth(0.012)
    plotter.add_slider_widget(
        set_particle_point_size, rng=(1.0, 10.0), value=appearance_state["point_size"],
        title="Particle point size [px]", pointa=(0.78, 0.81), pointb=(0.97, 0.81),
        color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
        slider_width=0.014, tube_width=0.006, style="modern", fmt="%.0f",
        interaction_event="end",
    )
    plotter.add_slider_widget(
        set_particle_opacity, rng=(0.08, 1.0), value=appearance_state["opacity"],
        title="Particle opacity", pointa=(0.78, 0.73), pointb=(0.97, 0.73),
        color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
        slider_width=0.014, tube_width=0.006, style="modern", fmt="%.2f",
        interaction_event="end",
    )
    color_state["lock_widget"] = plotter.add_checkbox_button_widget(
        set_locked_color_range, value=False, position=(1150, 855), size=24,
        color_on="#42d392", color_off="#9aa4b2", background_color="#172033",
    )
    plotter.add_text("LOCK COLOR RANGE", position=(1185, 860), font_size=8,
                     color="#f8fafc", name="lock_color_range_label")
    if mass_limits is not None:
        plotter.add_slider_widget(
            set_minimum_halo_mass,
            rng=mass_limits,
            value=mass_filter["log10_minimum"],
            title="Min log10 mass [Msun/h]",
            pointa=(0.78, 0.63), pointb=(0.97, 0.63),
            color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
            slider_width=0.018, tube_width=0.008, style="modern",
            fmt="%.2f", interaction_event="end",
        )

    plotter.add_slider_widget(
        set_velocity_arrow_scale,
        rng=(0.005, 0.12), value=motion_state["arrow_scale_box_fraction"],
        title="Arrow length [box fraction]",
        pointa=(0.78, 0.52), pointb=(0.97, 0.52),
        color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
        slider_width=0.014, tube_width=0.006, style="modern",
        fmt="%.3f", interaction_event="end",
    )
    plotter.add_slider_widget(
        set_slice_center,
        rng=(0.0, box_mpc_h), value=slice_state["center_mpc_h"],
        title="Z-slab center [Mpc/h]",
        pointa=(0.78, 0.42), pointb=(0.97, 0.42),
        color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
        slider_width=0.014, tube_width=0.006, style="modern",
        fmt="%.4g", interaction_event="end",
    )
    plotter.add_slider_widget(
        set_slice_thickness,
        rng=(box_mpc_h / 500.0, box_mpc_h / 3.0), value=slice_state["thickness_mpc_h"],
        title="Z-slab thickness [Mpc/h]",
        pointa=(0.78, 0.32), pointb=(0.97, 0.32),
        color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
        slider_width=0.014, tube_width=0.006, style="modern",
        fmt="%.4g", interaction_event="end",
    )

    if len(available) > 1:
        def activate_snapshot(index):
            nonlocal current_index, current_header, current_files, snap
            index = int(np.clip(index, 0, len(available) - 1))
            next_snap = available[index]
            next_header, next_files = snapshot_sources[next_snap]
            clear_timeline_event_actors()
            for ptype, state in layer_state.items():
                if not state["visible"]:
                    continue
                if state["actor"] is not None:
                    plotter.remove_actor(state["actor"], render=False)
                    state["actor"] = None
                next_xyz, next_ids = read_layer_sample(ptype, state, next_files)
                state["display_ids"] = next_ids
                state["interpolation_ids"] = None
                state["actor"] = add_layer_actor(ptype, state, next_xyz, next_files, next_header)
            current_index, current_header, current_files, snap = index, next_header, next_files, next_snap
            motion_state["timeline_value"] = float(scale_factors[index])
            motion_state["fraction"] = 0.0
            for kind, state in catalogue_state.items():
                if not state["visible"]:
                    continue
                if state["actor"] is not None:
                    plotter.remove_actor(state["actor"], render=False)
                    state["actor"] = None
                add_catalogue_actor(
                    kind, state,
                    _read_catalogue_layer(run_dir, next_snap, kind, mass_definition),
                )
            if r200_state["visible"]:
                add_r200_shells(next_snap)
            if filament_state["visible"]:
                refresh_filament_layer(next_snap)
            if density_state["visible"]:
                refresh_density_volume(next_snap)
            if motion_state["velocity_arrows"] or motion_state["streamlines"]:
                refresh_velocity_layers()
            refresh_particle_slice()
            update_camera_follow(0.0)
            clear_picked_catalogue_info(render=False)

        def sync_timeline_slider(value):
            widget = motion_state.get("timeline_slider")
            if widget is None:
                return
            try:
                motion_state["syncing_slider"] = True
                widget.GetRepresentation().SetValue(float(value))
            except (AttributeError, RuntimeError, TypeError):
                pass
            finally:
                motion_state["syncing_slider"] = False

        def seek_scale_factor(value, sync_slider=True):
            target = float(np.clip(float(value), scale_factors[0], scale_factors[-1]))
            index, fraction = _timeline_bracket(scale_factors, target)
            if index != current_index:
                activate_snapshot(index)
            else:
                restore_particle_layers()
            motion_state["timeline_value"] = target
            motion_state["fraction"] = fraction
            interpolate_particle_layers(fraction)
            refresh_particle_slice()
            update_camera_follow(fraction)
            if sync_slider:
                sync_timeline_slider(target)
            update_snapshot_label(current_header, snap, render=False)
            plotter.render()

        def select_timeline(value):
            if motion_state.get("syncing_slider"):
                return
            seek_scale_factor(float(value), sync_slider=False)

        def set_playback_speed(value):
            motion_state["playback_speed"] = float(np.clip(float(value), 0.25, 4.0))
            update_snapshot_label(current_header, snap, render=False)
            plotter.render()

        motion_state["timeline_slider"] = plotter.add_slider_widget(
            select_timeline,
            rng=(float(scale_factors[0]), float(scale_factors[-1])),
            value=float(scale_factors[current_index]),
            title="Scale factor a",
            pointa=(0.78, 0.10),
            pointb=(0.97, 0.10),
            color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
            slider_width=0.018, tube_width=0.008, style="modern",
            fmt="%.5f",
            interaction_event="end",
        )
        plotter.add_slider_widget(
            set_playback_speed,
            rng=(0.25, 4.0), value=motion_state["playback_speed"],
            title="Playback speed ×",
            pointa=(0.78, 0.21), pointb=(0.97, 0.21),
            color="#8ba4bd", title_color="#f8fafc", title_height=0.013,
            slider_width=0.014, tube_width=0.006, style="modern",
            fmt="%.2f", interaction_event="end",
        )

        def step_saved_output(direction):
            target_index = int(np.clip(current_index + int(direction), 0, len(available) - 1))
            seek_scale_factor(float(scale_factors[target_index]))

        def make_step_button(direction):
            holder = {"widget": None}

            def step_callback(pressed):
                if pressed:
                    step_saved_output(direction)
                widget = holder.get("widget")
                if widget is not None:
                    try:
                        widget.GetRepresentation().SetState(0)
                    except (AttributeError, RuntimeError, TypeError):
                        pass

            widget = plotter.add_checkbox_button_widget(
                step_callback, value=False,
                position=(1150, 25) if direction < 0 else (1350, 25),
                size=25, color_on="#42d392", color_off="#9aa4b2",
                background_color="#172033",
            )
            holder["widget"] = widget
            return widget

        motion_state["step_back_widget"] = make_step_button(-1)
        motion_state["step_forward_widget"] = make_step_button(1)
        plotter.add_text("PREV", position=(1185, 30), font_size=7, color="#f8fafc",
                         name="timeline_step_back_label")
        plotter.add_text("NEXT", position=(1385, 30), font_size=7, color="#f8fafc",
                         name="timeline_step_forward_label")
        plotter.add_text("← / → also steps between exact saved outputs",
                         position=(450, 40), font_size=7, color="#cbd5e1",
                         name="timeline_keyboard_hint")
        plotter.add_key_event("Left", lambda: step_saved_output(-1))
        plotter.add_key_event("Right", lambda: step_saved_output(1))

        def playback_timer(_step):
            if not motion_state["playing"]:
                return
            if has_cosmic_time_timeline:
                current_time = float(np.interp(
                    motion_state["timeline_value"], scale_factors, timeline_cosmic_times_gyr,
                ))
                time_step = float(np.median(np.diff(timeline_cosmic_times_gyr)))
                progress_time = current_time + time_step * 0.125 * motion_state["playback_speed"]
                if progress_time >= timeline_cosmic_times_gyr[-1]:
                    if motion_state["loop"]:
                        progress_time = float(timeline_cosmic_times_gyr[0])
                    else:
                        progress_time = float(timeline_cosmic_times_gyr[-1])
                        motion_state["playing"] = False
                progress = float(np.interp(
                    progress_time, timeline_cosmic_times_gyr, scale_factors,
                ))
            else:
                spacing = float(np.median(np.diff(scale_factors)))
                progress = motion_state["timeline_value"] + spacing * 0.125 * motion_state["playback_speed"]
                if progress >= scale_factors[-1]:
                    if motion_state["loop"]:
                        progress = float(scale_factors[0])
                    else:
                        progress = float(scale_factors[-1])
                        motion_state["playing"] = False
            seek_scale_factor(progress)
            if not motion_state["playing"] and motion_state.get("play_widget") is not None:
                try:
                    motion_state["play_widget"].GetRepresentation().SetState(0)
                except (AttributeError, RuntimeError, TypeError):
                    pass

        plotter.add_timer_event(max_steps=1_000_000, duration=140, callback=playback_timer)
    plotter.show()
    return layer_state[selected_ptype]["shown"]


def _snapshot_constant_particle_mass(files, ptype):
    """Read a positive constant particle mass from aligned GADGET headers."""
    for path in files:
        try:
            with h5py.File(path, "r") as h5:
                values = np.asarray(h5["Header"].attrs.get("MassTable", []), dtype=float)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if values.shape == (6,) and np.isfinite(values[int(ptype)]) and values[int(ptype)] > 0:
            return float(values[int(ptype)])
    return None


def _normalize_gas_timeline_samples(entries, raw_normalizers):
    """Use one density scale across exact-output frames when their field is consistent."""
    samples = []
    parsed = []
    sources = set()
    for entry in entries:
        if entry.get("empty_species") or not entry.get("ply"):
            continue
        normalizer_record = raw_normalizers.get(int(entry["snapshot"]))
        if isinstance(normalizer_record, dict):
            normalizer = normalizer_record.get("normalizer")
            source = normalizer_record.get("source")
        else:
            normalizer = normalizer_record
            source = "legacy Masses/SmoothingLength^3 proxy"
        if normalizer is None or not np.isfinite(normalizer) or normalizer <= 0:
            return {
                "status": "PER_SNAPSHOT_FALLBACK",
                "reason": "at least one saved output lacks a finite gas-density normalizer",
            }
        sources.add(str(source))
        path = Path(entry["ply"])
        raw = path.read_bytes()
        header, payload = raw.split(b"end_header\n", 1)
        rows = re.search(rb"element vertex (\d+)", header)
        if rows is None:
            raise ValueError("Gas timeline PLY has no vertex count: " + str(path))
        count = int(rows.group(1))
        properties = re.findall(rb"property\s+\w+\s+(\w+)", header)
        if b"density_weight" not in properties:
            raise ValueError("Gas timeline PLY has no density_weight property: " + str(path))
        property_count = len(properties)
        values = np.frombuffer(payload, dtype="<f4")
        if values.size != count * property_count:
            raise ValueError("Gas timeline PLY fields do not align with its vertex count: " + str(path))
        values = values.reshape(count, property_count).copy()
        weight_column = properties.index(b"density_weight")
        raw_density = values[:, weight_column].astype(np.float64) * float(normalizer) / 0.45
        valid = np.isfinite(raw_density) & (raw_density > 0)
        if np.any(valid):
            samples.append(raw_density[valid])
        parsed.append((path, header + b"end_header\n", values, weight_column, raw_density, valid))
    if len(sources) > 1:
        return {
            "status": "PER_SNAPSHOT_FALLBACK",
            "reason": "exact-output movie would mix unlike density fields: " + ", ".join(sorted(sources)),
        }
    if not samples:
        return {"status": "UNAVAILABLE", "reason": "no positive gas density weights were exported"}
    global_median = float(np.median(np.concatenate(samples)))
    if not np.isfinite(global_median) or global_median <= 0:
        return {"status": "UNAVAILABLE", "reason": "global gas density-weight median is invalid"}
    global_upper = float(np.percentile(np.concatenate(samples), 99.5))
    normalized_cap = max(global_upper / global_median, 1.0)
    for path, header, values, weight_column, raw_density, valid in parsed:
        weights = np.zeros(len(values), dtype=np.float32)
        weights[valid] = np.asarray(
            0.45 * np.minimum(raw_density[valid] / global_median, normalized_cap),
            dtype=np.float32,
        )
        values[:, weight_column] = weights
        with open(path, "wb") as output:
            output.write(header)
            output.write(values.astype("<f4", copy=False).tobytes(order="C"))
    return {
        "status": "COMPLETE",
        "source": next(iter(sources)) if sources else "unknown",
        "normalization": "one sampled-particle median across outputs maps to Blender density weight 0.45; upper values clipped at pooled 99.5th percentile",
        "raw_density_weight_median": global_median,
        "raw_density_weight_p99_5": global_upper,
        "density_weight_upper_clip": float(0.45 * normalized_cap),
        "valid_sample_count": int(sum(len(sample) for sample in samples)),
        "field_interpretation": "relative median-normalized density transfer; read the source and unit before physical interpretation",
    }


def _make_blender_snapshot_timeline_loader(
    run_dir, species="dm", max_points=300_000, *, full_resolution=False, memory_plan=None,
    requested_particle_mode=None, adaptive_fallback=None, output_directory=None, config=None,
):
    """Export the legacy exact-snapshot Blender timeline without interpolation."""
    outdir = Path(output_directory) if output_directory else Path(run_dir) / "results/3d"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else load_cfg(run_dir)
    visualization = cfg.get("visualization", {})
    available = snapshot_ids(run_dir)
    if not available:
        raise FileNotFoundError("No HDF5 snapshots found for Blender timeline export")
    if full_resolution:
        total_budget = None
        # The explicit full mode controls particle selection; retain the valid
        # public budget argument even though full mode does not subsample.
        per_snapshot = int(max_points)
        if memory_plan is None:
            memory_plan = _blender_timeline_memory_plan(
                run_dir, species, available,
                float(cfg.get("blender", {}).get("max_memory_fraction", 0.70)),
            )
        if not memory_plan.get("fits"):
            raise MemoryError(
                "Full-resolution Blender timeline is estimated to need %(estimated_peak_bytes)s bytes, "
                "above the configured RAM budget %(allowed_bytes)s" % memory_plan
            )
    else:
        total_budget = int(visualization.get("blender_sequence_max_points", 800_000))
        if total_budget < len(available):
            raise ValueError(
                "visualization.blender_sequence_max_points must allow at least one particle per saved output"
            )
        per_snapshot = min(int(max_points), max(1, total_budget // len(available)))
    manifest_entries = []
    gas_normalizers = {}
    bh_normalization = None
    if species == "black_holes" and full_resolution:
        bh_files = [path for snap in available for path in read_snapshot_header(run_dir, int(snap))[1]]
        bh_normalization = _black_hole_particle_mass_normalization(bh_files)
    for snap in available:
        empty = False
        try:
            make_blender_loader(
                run_dir, snap=snap, species=species, max_points=per_snapshot,
                sequence=False, _allow_render=False,
                _particle_mode="full" if full_resolution else "sampled",
                _black_hole_mass_normalization=bh_normalization, _output_directory=outdir,
            )
            item_status = json.loads(
                (outdir / ("blender_status_%s.json" % species)).read_text(encoding="utf-8")
            )
            if species == "gas":
                method = str(item_status.get("gas_density_method") or "")
                if "GADGET PartType0/Density" in method:
                    source = "GADGET PartType0/Density [proper g/cm^3]"
                elif "Masses/SmoothingLength^3 fallback" in method:
                    source = "Masses/SmoothingLength^3 relative proxy"
                else:
                    source = "equal-weight fallback"
                gas_normalizers[int(snap)] = {
                    "normalizer": item_status.get("gas_density_weight_normalizer_raw"),
                    "source": source,
                }
        except ValueError as exc:
            if "contains no particles" not in str(exc):
                raise
            empty = True
        header, _ = read_snapshot_header(run_dir, snap)
        scale_factor = _scalar_float(header.get("Time", np.nan), "snapshot Header/Time")
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("Snapshot %03d has no valid Header/Time scale factor" % snap)
        manifest_entries.append({
            "snapshot": int(snap),
            "scale_factor": scale_factor,
            "redshift": 1.0 / scale_factor - 1.0,
            "ply": None if empty else item_status["sample_ply"],
            "particle_count": 0 if empty else int(item_status.get(
                "particle_count", item_status.get("sampled_particle_count", 0),
            )),
            "empty_species": empty,
            "identity_asset": None if empty else item_status.get('identity_asset'),
            "halo_overlays": _blender_halo_overlays(
                run_dir, int(snap), cfg, float(cfg["simulation"]["box_mpc_h"]),
                maximum_halos=int(visualization.get("max_matched_halo_overlays", 250)),
            ),
        })

    gas_density_normalization = (
        _normalize_gas_timeline_samples(manifest_entries, gas_normalizers)
        if species == "gas" else None
    )

    box = float(cfg["simulation"]["box_mpc_h"])
    fps = int(visualization.get("movie_fps", 24))
    frames_per_snapshot = int(visualization.get("blender_frames_per_snapshot", 12))
    render_quality = str(visualization.get("blender_render_quality", "preview")).lower()
    manifest_path = outdir / ("blender_%s_timeline.json" % species)
    manifest = {
        "schema_version": 1,
        "species": species,
        "box_mpc_h": box,
        "fps": fps,
        "frames_per_snapshot": frames_per_snapshot,
        "render_quality": render_quality,
        "camera_orbit": bool(visualization.get("blender_timeline_camera_orbit", False)),
        "sampling": {
            "particle_mode": "full" if full_resolution else "sampled",
            "requested_particle_mode": requested_particle_mode or ("full" if full_resolution else "sampled"),
            "adaptive_fallback": adaptive_fallback,
            "per_snapshot_max_points": per_snapshot,
            "total_point_budget": total_budget,
            "memory_estimate": memory_plan,
            "interpretation": "each frame displays an exact saved snapshot; particle positions are not interpolated",
        },
        "gas_density_normalization": gas_density_normalization,
        "snapshots": manifest_entries,
        "black_hole_display": bh_normalization,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    scene_script = copy_blender_runtime(outdir)
    movie = outdir / ("%s_cinematic_timeline.mp4" % species)
    loader = outdir / ("blender_timeline_loader_%s.py" % species)
    loader.write_text(
        "# Run this in Blender's Scripting workspace to build and render the saved-output timeline.\n"
        "from pathlib import Path\n"
        "import sys\n"
        "base = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(base))\n"
        "from blender import build_timeline_scene\n"
        "build_timeline_scene(str(base / %r), str(base / %r))\n"
        % (manifest_path.name, movie.name),
        encoding="utf-8",
    )
    status = {
        "status": "EXPORTED",
        "timeline_manifest": manifest_path.name,
        "interactive_loader": loader.name,
        "scene_script": str(scene_script),
        "movie": movie.name,
        "snapshot_count": len(manifest_entries),
        "per_snapshot_max_points": per_snapshot,
        "total_point_budget": total_budget,
        "particle_mode": "full" if full_resolution else "sampled",
        "requested_particle_mode": requested_particle_mode or ("full" if full_resolution else "sampled"),
        "adaptive_fallback": adaptive_fallback,
        "memory_estimate": memory_plan,
        "render_quality": render_quality,
        "render_status": "NOT_REQUESTED",
        "motion_interpolation": "none; output frames show exact saved snapshots",
        "gas_density_normalization": gas_density_normalization,
    }
    if visualization.get("blender_render", False):
        executable_path = _find_blender_executable(visualization.get("blender_executable"))
        if executable_path is None:
            status["render_status"] = "SKIPPED"
            status["reason"] = "Blender executable was not found; run the generated timeline loader inside Blender"
        else:
            try:
                completed = subprocess.run(
                    [executable_path, "--background", "--python-exit-code", "1", "--python", str(scene_script), "--",
                     "--timeline", str(manifest_path.resolve()), str(movie.resolve())],
                    check=True, capture_output=True, text=True,
                )
                status["render_log_tail"] = completed.stdout[-2000:]
                status["movie_exists"] = movie.is_file()
                status["render_status"] = "COMPLETE" if movie.is_file() else "FAILED"
                if not movie.is_file():
                    status["reason"] = "Blender returned successfully but did not create the requested timeline movie"
            except (OSError, subprocess.CalledProcessError) as exc:
                status["render_status"] = "FAILED"
                status["reason"] = str(exc)
    (outdir / ("blender_status_%s.json" % species)).write_text(json.dumps(status, indent=2), encoding="utf-8")
    return loader




def _lowest_hash_ids(ids, limit):
    """Select the same deterministic bottom-hash sample regardless of row order."""
    values = np.unique(np.asarray(ids))
    if len(values) <= int(limit):
        return values
    hashes = _particle_id_hash64(values)
    chosen = np.argpartition(hashes, int(limit) - 1)[:int(limit)]
    order = np.lexsort((values[chosen], hashes[chosen]))
    return values[chosen[order]]


def _sample_timeline_id_union(run_dir, snapshots, ptype, limit):
    """Find a bounded, stable ID sample from the union of all saved outputs.

    Only the bottom ``limit`` hashes from each output can enter the bottom
    ``limit`` of the full union. This keeps memory bounded while retaining IDs
    that persist across snapshots and IDs that are born later in the run.
    """
    group = "PartType%d" % int(ptype)
    per_snapshot_candidates = []
    for snap in snapshots:
        files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%03d" % int(snap))
        best_ids = np.empty(0, dtype=np.uint64)
        best_hashes = np.empty(0, dtype=np.uint64)
        chunk_size = 1_000_000
        for path in files:
            with h5py.File(path, "r") as h5:
                if group not in h5 or "ParticleIDs" not in h5[group]:
                    continue
                dataset = h5[group]["ParticleIDs"]
                if dataset.ndim != 1:
                    raise ValueError("%s/ParticleIDs must be one-dimensional in %s" % (group, path))
                for start in range(0, len(dataset), chunk_size):
                    ids = np.asarray(dataset[start:min(start + chunk_size, len(dataset))])
                    if ids.dtype.kind not in "iu" or (ids.dtype.kind == "i" and np.any(ids < 0)):
                        raise ValueError("%s/ParticleIDs must contain non-negative integers" % group)
                    candidate_ids = np.concatenate((best_ids, ids.astype(np.uint64, copy=False)))
                    candidate_hashes = np.concatenate((best_hashes, _particle_id_hash64(ids)))
                    if len(candidate_ids) > int(limit):
                        take = np.argpartition(candidate_hashes, int(limit) - 1)[:int(limit)]
                        best_ids, best_hashes = candidate_ids[take], candidate_hashes[take]
                    else:
                        best_ids, best_hashes = candidate_ids, candidate_hashes
        if len(best_ids):
            per_snapshot_candidates.append(best_ids)
    if not per_snapshot_candidates:
        return np.empty(0, dtype=np.uint64)
    candidates = np.unique(np.concatenate(per_snapshot_candidates))
    return _lowest_hash_ids(candidates, limit)


def _write_blender_particle_ply(path, positions, radius=None, density_weight=None,
                                physical_density_g_cm3=None, temperature_log10=None,
                                stellar_age_gyr=None, black_hole_mass_weight=None,
                                stellar_candidate_color_bin=None):
    """Write a Blender-importable PLY with optional gas rendering attributes."""
    path = Path(path)
    points = np.asarray(positions, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
        raise ValueError("Blender timeline positions must be a finite (particles, 3) array")
    gas = radius is not None or density_weight is not None
    if gas and (radius is None or density_weight is None):
        raise ValueError("gas PLY exports require both smoothing radii and density weights")
    if gas:
        radius = np.asarray(radius, dtype="<f4").reshape(-1)
        density_weight = np.asarray(density_weight, dtype="<f4").reshape(-1)
        physical_density = (
            None if physical_density_g_cm3 is None else
            np.asarray(physical_density_g_cm3, dtype="<f4").reshape(-1)
        )
        if len(radius) != len(points) or len(density_weight) != len(points):
            raise ValueError("gas PLY rendering attributes must align with particle positions")
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0) or np.any(~np.isfinite(density_weight)):
            raise ValueError("gas PLY radii and density weights must be finite and radii positive")
        if physical_density is not None and (
                len(physical_density) != len(points)
                or np.any(~np.isfinite(physical_density))
                or np.any(physical_density < 0)):
            raise ValueError("physical gas densities must be finite, nonnegative, and aligned with positions")
    else:
        physical_density = None
    visual_fields = {}
    for name, values in (("temperature_log10", temperature_log10),
                         ("stellar_age_gyr", stellar_age_gyr),
                         ("black_hole_mass_weight", black_hole_mass_weight),
                         ("stellar_candidate_color_bin", stellar_candidate_color_bin)):
        if values is not None:
            array = np.asarray(values, dtype="<f4").reshape(-1)
            if len(array) != len(points) or np.any(~np.isfinite(array)):
                raise ValueError(name + " must be finite and align with PLY vertices")
            if name == "stellar_candidate_color_bin" and (
                    np.any(array < 0) or np.any(array > 10) or np.any(array != np.floor(array))):
                raise ValueError("Stellar candidate display bins must be exact integers from 0 through 10")
            visual_fields[name] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(points)
        header += "property float x\nproperty float y\nproperty float z\n"
        if gas:
            header += "property float radius\nproperty float density_weight\n"
            if physical_density is not None:
                header += "property float physical_density_g_cm3\n"
        for name in visual_fields:
            header += "property float " + name + "\n"
        header += "end_header\n"
        output.write(header.encode("ascii"))
        columns = [points]
        if gas:
            columns.extend((radius[:, None], density_weight[:, None]))
            if physical_density is not None:
                columns.append(physical_density[:, None])
        columns.extend(values[:, None] for values in visual_fields.values())
        np.column_stack(columns).astype("<f4", copy=False).tofile(output)
    return path


def _find_blender_executable(configured=None):
    """Find Blender on PATH or in common desktop installation locations."""
    candidates = [configured, shutil.which("blender")]
    if os.name == "nt":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if not root:
                continue
            base = Path(root)
            candidates.extend((base / "Blender Foundation").glob("Blender*/blender.exe"))
            candidates.extend((base / "Programs/Blender Foundation").glob("Blender*/blender.exe"))
    else:
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(str(candidate))
        if resolved:
            return str(Path(resolved).resolve())
    return None


def _blender_memory_estimate(particle_count, max_memory_fraction, available_bytes=None):
    """Conservatively estimate a Blender point scene before it is imported."""
    count = int(particle_count)
    fraction = float(max_memory_fraction)
    if count < 0 or not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("particle count and Blender memory fraction are invalid")
    available = _available_ram_bytes() if available_bytes is None else int(available_bytes)
    coordinate_bytes = count * 3 * np.dtype("<f4").itemsize
    # Mesh import and dependency-graph storage exceed the packed PLY payload.
    blender_overhead_bytes = coordinate_bytes * 16 + 1024 * 1024**2
    estimate = coordinate_bytes + blender_overhead_bytes
    budget = None if available is None else int(available * fraction)
    return {
        "particle_count": count,
        "coordinate_bytes": int(coordinate_bytes),
        "estimated_blender_overhead_bytes": int(blender_overhead_bytes),
        "estimated_peak_bytes": int(estimate),
        "available_ram_bytes": available,
        "max_memory_fraction": fraction,
        "allowed_bytes": budget,
        "fits": bool(budget is not None and estimate <= budget),
    }


def _blender_timeline_memory_plan(run_dir, species, snapshots, max_memory_fraction):
    ptype = {"gas": 0, "dm": 1, "stars": 4, "black_holes": 5}[species]
    total_particles = 0
    per_snapshot = {}
    for snap in snapshots:
        header, files = read_snapshot_header(run_dir, int(snap))
        count = _snapshot_particle_type_total(header, files, ptype)
        per_snapshot[int(snap)] = int(count)
        total_particles += int(count)
    report = _blender_memory_estimate(total_particles, max_memory_fraction)
    report["snapshot_particle_counts"] = per_snapshot
    return report


def _black_hole_particle_mass_normalization(files, chunk_size=1_000_000):
    """Measure a shared log-mass colour scale from saved PartType5 particles.

    These are dynamical particle masses, not inferred subgrid BH masses or
    accretion rates. Missing masses remain an unknown display category.
    """
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("Mass-normalization chunk size must be positive")
    low, high = math.inf, -math.inf
    missing = False
    count = 0
    for path in files:
        with h5py.File(path, "r") as source:
            particles = source.get("PartType5")
            if particles is None or not len(particles["Coordinates"]):
                continue
            size = len(particles["Coordinates"])
            if "Masses" in particles:
                masses = particles["Masses"]
                if masses.shape != (size,):
                    raise ValueError("PartType5/Masses must align with Coordinates")
                for start in range(0, size, chunk_size):
                    values = np.asarray(masses[start:start + chunk_size], dtype=float)
                    if np.any(~np.isfinite(values)) or np.any(values <= 0):
                        raise ValueError("PartType5 particle masses must be finite and positive")
                    low = min(low, float(np.log10(np.min(values))))
                    high = max(high, float(np.log10(np.max(values))))
            else:
                mass = float(source["Header"].attrs.get("MassTable", np.zeros(6))[5])
                if not np.isfinite(mass) or mass <= 0:
                    missing = True
                    continue
                low = min(low, math.log10(mass))
                high = max(high, math.log10(mass))
            count += size
    measured = count > 0
    return {
        "status": "MEASURED" if measured else "UNAVAILABLE",
        "source": "PartType5/Masses or Header/MassTable[5]",
        "source_unit": "snapshot dynamical particle mass units",
        "missing_mass_present": missing,
        "log10_mass_range_code_units": [low, high] if measured else None,
        "display_mapping": "shared log-mass colour scale; unknown masses are grey; glyph radius is not a horizon",
        "physics_scope": "No subgrid BH mass, accretion rate, radiative luminosity or feedback is inferred",
    }


def _write_blender_snapshot_ply_streaming(files, ptype, path, box_mpc_h, chunk_size=1_000_000,
                                           *, run_dir=None, header=None, black_hole_mass_normalization=None):
    """Stream every saved row, including measured thermal/age colours, with bounded RAM.

    Gas radii/positions are comoving Mpc/h. Uniform density_weight is a glyph
    display weight, not a physical density. Colours share the timeline's exact
    unit/EOS/formation-clock conversion; no chunk-dependent normalisation.
    """
    species = {0: "gas", 1: "dm", 4: "stars", 5: "black_holes"}[int(ptype)]
    bh_normalization = black_hole_mass_normalization
    if species == "black_holes" and bh_normalization is None:
        bh_normalization = _black_hole_particle_mass_normalization(files, chunk_size)
    fields = _blender_species_fields(species)
    group = "PartType%d" % int(ptype)
    records, signatures = [], []
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("Streaming chunk size must be positive")
    for raw_path in files:
        with h5py.File(raw_path, "r") as source:
            if group not in source:
                continue
            particles = source[group]
            if "Coordinates" not in particles:
                raise ValueError(group + "/Coordinates is missing")
            coordinates = particles["Coordinates"]
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(group + "/Coordinates must have shape (particles, 3)")
            if not len(coordinates):
                continue
            present = tuple(name for name in fields if name in particles)
            for name in present:
                dataset = particles[name]
                expected = (len(coordinates), 3) if name == "Coordinates" else (len(coordinates),)
                if dataset.shape != expected:
                    raise ValueError(group + "/" + name + " does not align with Coordinates")
            records.append((Path(raw_path), len(coordinates), present))
            signatures.append(present)
    if not records:
        raise ValueError("The selected snapshot contains no particles of type %d" % int(ptype))
    if any(present != signatures[0] for present in signatures):
        raise ValueError("Blender scalar fields must have consistent availability across non-empty shards")
    total = sum(count for _, count, _ in records)
    gas = species == "gas"
    properties = ["x", "y", "z"]
    if gas:
        properties.extend(["radius", "density_weight", "temperature_log10"])
    elif species == "stars":
        properties.append("stellar_age_gyr")
    elif species == "black_holes":
        properties.append("black_hole_mass_weight")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    radius_source = None
    identity_path=path.with_suffix('.identities.hdf5')
    from core.data import exact_uint64_ids, write_dataset_contract
    cursor=0
    with path.open("wb") as output, h5py.File(identity_path,'w') as identities:
        ids_dataset=identities.create_dataset('ParticleIDs',shape=(total,),dtype='uint64',chunks=True,fletcher32=True)
        write_dataset_contract(ids_dataset,units='identity',missing_policy='all PLY rows have an exact saved identity')
        identities.attrs['mapping']='ParticleIDs[row] is the native ID of PLY vertex row' 
        ply_header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % total
        ply_header += "".join("property float " + name + "\n" for name in properties)
        output.write((ply_header + "end_header\n").encode("ascii"))
        for path_item, count, present in records:
            with h5py.File(path_item, "r") as source:
                particles = source[group]
                for start in range(0, count, chunk_size):
                    stop = min(start + chunk_size, count)
                    data = {name: np.asarray(particles[name][start:stop]) if name in present else None
                            for name in fields}
                    exact=exact_uint64_ids(data['ParticleIDs'])
                    ids_dataset[cursor:cursor+len(exact)]=exact
                    cursor+=len(exact)
                    xyz = np.mod(convert_pinned_native(data['Coordinates'],'position_comoving_mpc_h'), box_mpc_h).astype("<f4")
                    if np.any(~np.isfinite(xyz)):
                        raise ValueError("Non-finite Blender coordinates in " + str(path_item))
                    attributes = _blender_species_attributes(
                        run_dir, files, ptype, species, box_mpc_h, header, data, xyz,
                    )
                    columns = [xyz]
                    if gas:
                        radius_source = attributes["radius_source"]
                        columns.extend([attributes["radius"][:, None],
                                        np.full((len(xyz), 1), .45, dtype="<f4"),
                                        attributes["temperature_log10"][:, None]])
                    elif species == "stars":
                        columns.append(attributes["stellar_age_gyr"][:, None])
                    elif species == "black_holes":
                        weight = np.full(len(xyz), -1., dtype="<f4")
                        raw = attributes.get("black_hole_log10_mass_code_units")
                        if raw is not None and bh_normalization["status"] == "MEASURED":
                            low, high = bh_normalization["log10_mass_range_code_units"]
                            weight[:] = .5 if high == low else np.clip((raw - low) / (high - low), 0., 1.)
                        columns.append(weight[:, None])
                    np.column_stack(columns).astype("<f4", copy=False).tofile(output)
    return {"particle_count": total, "identity_asset":identity_path.name, "gas_radius_source": radius_source,
            "gas_density_method": "uniform glyph display weight; not a physical density" if gas else None,
            "black_hole_mass_normalization": bh_normalization}


def _finish_blender_single_export(
    run_dir, outdir, species, snap, ply, box, fps, render_quality, status, *,
    allow_render=True, memory_plan=None,
):
    """Write the portable loader and optional Blender render for one snapshot."""
    visualization = load_cfg(run_dir).get("visualization", {})
    movie = Path(outdir) / ("snapshot_%03d_%s_cinematic.mp4" % (int(snap), species))
    scene_script = copy_blender_runtime(Path(outdir))
    loader = Path(outdir) / ("blender_loader.py" if species == "dm" else "blender_loader_%s.py" % species)
    memory_arguments = ""
    if memory_plan is not None:
        memory_arguments = ", particle_count=%d, max_memory_fraction=%r" % (
            int(memory_plan["particle_count"]), float(memory_plan["max_memory_fraction"]),
        )
    loader.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "base = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(base))\n"
        "from blender import build_scene\n"
        "build_scene(str(base / %r), str(base / %r), %r, %r, %r, %r, %r%s)\n"
        % (Path(ply).name, movie.name, box, species, fps, 240, render_quality, memory_arguments),
        encoding="utf-8",
    )
    status.update({
        "interactive_loader": loader.name,
        "scene_script": scene_script.name,
        "movie": movie.name,
        "render_quality": render_quality,
        "render_status": "NOT_REQUESTED",
    })
    if allow_render and visualization.get("blender_render", False):
        executable_path = _find_blender_executable(visualization.get("blender_executable"))
        if executable_path is None:
            status["render_status"] = "SKIPPED"
            status["reason"] = "Blender executable was not found; run the generated loader inside Blender"
        else:
            command = [
                executable_path, "--background", "--python-exit-code", "1", "--python", str(scene_script), "--",
                str(Path(ply).resolve()), str(movie.resolve()), str(box), species, str(fps), "240", render_quality,
            ]
            if memory_plan is not None:
                command.extend([
                    str(int(memory_plan["particle_count"])),
                    str(float(memory_plan["max_memory_fraction"])),
                ])
            try:
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                status["render_log_tail"] = completed.stdout[-2000:]
                status["movie_exists"] = movie.is_file()
                status["render_status"] = "COMPLETE" if movie.is_file() else "FAILED"
                if not movie.is_file():
                    status["reason"] = "Blender returned successfully but did not create the requested movie"
            except (OSError, subprocess.CalledProcessError) as exc:
                status["render_status"] = "FAILED"
                status["reason"] = str(exc)
    status_path = Path(outdir) / ("blender_status_%s.json" % species)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return loader


def _periodic_trajectory_copies(start, end, box, attributes=None):
    """Unwrap minimum-image tracks and add periodic images at crossed faces."""
    start = np.mod(np.asarray(start, dtype=float), float(box))
    end = np.mod(np.asarray(end, dtype=float), float(box))
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("periodic trajectories require aligned (particles, 3) endpoints")
    delta = end - start
    delta -= float(box) * np.round(delta / float(box))
    unwrapped_end = start + delta
    starts, ends = [], []
    copied_attributes = {key: [] for key in (attributes or {})}
    for row in range(len(start)):
        offsets = []
        for axis in range(3):
            if unwrapped_end[row, axis] < 0.0:
                offsets.append((0.0, float(box)))
            elif unwrapped_end[row, axis] >= float(box):
                offsets.append((0.0, -float(box)))
            else:
                offsets.append((0.0,))
        for ox in offsets[0]:
            for oy in offsets[1]:
                for oz in offsets[2]:
                    shift = np.asarray([ox, oy, oz], dtype=float)
                    starts.append(start[row] + shift)
                    ends.append(unwrapped_end[row] + shift)
                    for key, values in (attributes or {}).items():
                        copied_attributes[key].append(np.asarray(values)[row])
    start_result = np.asarray(starts, dtype="<f4").reshape(-1, 3)
    end_result = np.asarray(ends, dtype="<f4").reshape(-1, 3)
    attr_result = {
        key: np.asarray(values, dtype=np.asarray(attributes[key]).dtype) for key, values in copied_attributes.items()
    }
    return start_result, end_result, attr_result


def _estimate_blender_timeline_vertices(samples, species, box):
    """Count the actual interval mesh vertices, including periodic images."""
    total = 0
    for left, right in zip(samples[:-1], samples[1:]):
        matched, index_a, index_b = np.intersect1d(
            left["ids"], right["ids"], assume_unique=True, return_indices=True,
        )
        start = np.mod(left["positions"][index_a], float(box))
        end = np.mod(right["positions"][index_b], float(box))
        delta = end - start
        delta -= float(box) * np.round(delta / float(box))
        unwrapped = start + delta
        copies = np.ones(len(matched), dtype=np.int64)
        for axis in range(3):
            crosses = (unwrapped[:, axis] < 0.0) | (unwrapped[:, axis] >= float(box))
            copies[crosses] *= 2
        matched_vertices = int(np.sum(copies, dtype=np.int64))
        endpoint_events = int(len(left["ids"]) + len(right["ids"]) - 2 * len(matched))
        total += (2 if species == "gas" else 1) * matched_vertices + endpoint_events
    return int(total)


def _physical_gas_density_g_cm3(files, header, values):
    """Convert PartType0/Density with the snapshot's declared GADGET units."""
    if values is None:
        return None, "PartType0/Density is unavailable"
    try:
        from analysis import _dataset_to_cgs_factor, _snapshot_dataset_unit_attributes, _snapshot_header_value

        attributes = _snapshot_dataset_unit_attributes(files, 0, "Density")
        if attributes is None:
            return None, "PartType0/Density lacks complete declared unit-conversion attributes"
        scale_factor = _snapshot_scale_factor(header)
        hubble_param = _snapshot_header_value(files, "HubbleParam")
        if hubble_param is None:
            hubble_param = _snapshot_parameter(files, "HubbleParam")
        if hubble_param is None or not np.isfinite(hubble_param) or hubble_param <= 0:
            return None, "snapshot HubbleParam is unavailable for physical density conversion"
        factor = _dataset_to_cgs_factor(attributes, scale_factor, hubble_param)
        if not np.isfinite(factor) or factor <= 0:
            return None, "PartType0/Density has an invalid declared conversion to cgs"
        density = np.asarray(values, dtype=float).reshape(-1) * float(factor)
        if not len(density):
            return density.astype("<f4"), "GADGET PartType0/Density converted to proper g/cm^3"
        density[~np.isfinite(density) | (density < 0)] = np.nan
        if not np.any(np.isfinite(density) & (density > 0)):
            return None, "PartType0/Density has no positive finite physically converted values"
        density_float32 = density.astype("<f4")
        positive_density = np.isfinite(density) & (density > 0)
        if np.any(positive_density & (density_float32 == 0)):
            return None, "PartType0/Density underflows the Blender PLY single-precision field"
        if np.any(positive_density & ~np.isfinite(density_float32)):
            return None, "PartType0/Density overflows the Blender PLY single-precision field"
        return density_float32, "GADGET PartType0/Density converted to proper g/cm^3"
    except (OSError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        return None, "physical PartType0/Density conversion unavailable: " + str(exc)


def _blender_species_fields(species):
    """Keep identical saved scalar inputs for single-frame and timeline exports."""
    fields = ["Coordinates", "ParticleIDs"]
    if species == "gas":
        fields.extend(["SmoothingLength", "Masses", "Density", "Temperature",
                       "InternalEnergy", "ElectronAbundance", "StarFormationRate"])
    elif species == "stars":
        fields.append("StellarFormationTime")
    elif species == "black_holes":
        fields.append("Masses")
    return fields


def _read_timeline_species_sample(run_dir, snap, ptype, selected_ids, species, box, header=None):
    """Read one stable-ID sample with the fields needed by the Blender renderer."""
    files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%03d" % int(snap))
    fields = _blender_species_fields(species)
    data = _read_snapshot_fields(files, ptype, fields, None, selected_ids)
    coordinates = data.get("Coordinates")
    ids = np.asarray(data.get("ParticleIDs", np.empty(0, dtype=np.uint64)))
    if coordinates is None or not len(coordinates):
        return {
            "ids": ids, "positions": np.empty((0, 3), dtype="<f4"),
            "physical_density_g_cm3": np.empty(0, dtype="<f4") if species == "gas" else None,
            "density_proxy_mass_per_radius3": np.empty(0, dtype="<f4") if species == "gas" else None,
        }
    positions = np.mod(convert_pinned_native(coordinates,'position_comoving_mpc_h'), float(box)).astype("<f4")
    if len(ids) != len(positions):
        raise ValueError("Snapshot %03d particle IDs do not align with Coordinates" % int(snap))
    result = _blender_species_attributes(run_dir, files, ptype, species, box, header, data, positions)
    result["ids"] = ids
    return result


def _blender_species_attributes(run_dir, files, ptype, species, box, header, data, positions):
    """Convert aligned saved fields to display attributes with explicit physical units.

    Coordinates/radii use comoving Mpc/h; gas temperatures use K and stellar
    ages use the matched CLASS proper-time clock in Gyr. Unknown or effective
    EOS values retain grey sentinels rather than invented temperatures/ages.
    """
    for name in _blender_species_fields(species):
        values = data.get(name)
        if name != "Coordinates" and values is not None and np.asarray(values).shape != (len(positions),):
            raise ValueError(name + " must be a scalar per saved particle, aligned with Coordinates")
    result = {"positions": positions}
    if species == "black_holes":
        result["black_hole_mass_weight"] = np.full(len(positions), -1., dtype="<f4")
        mass = data.get("Masses")
        if mass is None:
            constant_mass = _snapshot_constant_particle_mass(files, ptype)
            if constant_mass is not None:
                mass = np.full(len(positions), constant_mass, dtype=float)
        if mass is not None:
            mass = np.asarray(mass, dtype=float).reshape(-1)
            if len(mass) != len(positions):
                raise ValueError("PartType5/Masses does not align with black-hole particle positions")
            valid = np.isfinite(mass) & (mass > 0.0)
            if not np.all(valid):
                raise ValueError("PartType5 particle masses must be finite and positive")
            log_mass = np.full(len(mass), np.nan, dtype="<f4")
            log_mass[valid] = np.log10(mass[valid]).astype("<f4")
            result["black_hole_log10_mass_code_units"] = log_mass
    if species == "stars":
        formation = data.get("StellarFormationTime")
        # Negative sentinel is explicitly an unknown display class, never an
        # age measurement. Use the run-matched CLASS clock, not a/a_snapshot.
        ages = np.full(len(positions), -1., dtype="<f4")
        if formation is not None and header is not None:
            from analysis import _stellar_formation_clock
            scale, proper_time, _reason = _stellar_formation_clock(run_dir)
            epoch = _snapshot_scale_factor(header)
            birth = np.asarray(formation, dtype=float)
            if scale is not None and proper_time is not None and scale[0] <= epoch <= scale[-1]:
                valid = np.isfinite(birth) & (birth >= scale[0]) & (birth <= epoch)
                now = np.interp(epoch, scale, proper_time)
                ages[valid] = np.maximum(0., now - np.interp(birth[valid], scale, proper_time)).astype("<f4")
        result["stellar_age_gyr"] = ages
    if species == "gas":
        temperature = data.get("Temperature")
        if temperature is not None:
            require_stored_temperature_kelvin(files, ptype)
        if temperature is None and data.get("InternalEnergy") is not None and data.get("ElectronAbundance") is not None:
            with h5py.File(files[0], "r") as h5:
                velocity_unit = float(h5["Parameters"].attrs["UnitVelocity_in_cm_per_s"])
            temperature = temperature_from_internal_energy(
                data["InternalEnergy"], data["ElectronAbundance"],
                unit_velocity_cm_per_s=velocity_unit,
            )
        if temperature is not None:
            temperature = np.asarray(temperature, dtype=float)
            known = np.isfinite(temperature) & (temperature > 0)
            if load_cfg(run_dir).get("hydro", {}).get("star_formation", False):
                sfr = data.get("StarFormationRate")
                if sfr is None:
                    raise ValueError("Gas thermal colours require StarFormationRate to identify effective-EOS gas")
                known &= np.isfinite(sfr) & (np.asarray(sfr) == 0.)
            log_temperature = np.full(len(temperature), -1.e30, dtype="<f4")
            log_temperature[known] = np.log10(temperature[known]).astype("<f4")
            result["temperature_log10"] = log_temperature
        result.setdefault("temperature_log10", np.full(len(positions), -1.e30, dtype="<f4"))
        smoothing = data.get("SmoothingLength")
        has_physical_smoothing = smoothing is not None
        if smoothing is None:
            radius = np.full(len(positions), max(float(box) * 0.003, 1.0e-4), dtype="<f4")
            radius_source = "fixed visualization radius; PartType0/SmoothingLength unavailable"
        else:
            radius = convert_pinned_native(smoothing,'position_comoving_mpc_h').reshape(-1)
            valid_radius = np.isfinite(radius) & (radius > 0)
            fallback_radius = max(float(box) * 0.003, 1.0e-4)
            radius[~valid_radius] = fallback_radius
            radius = np.maximum(radius, 1.0e-8).astype("<f4")
            radius_source = "PartType0/SmoothingLength [kpc/h converted to Mpc/h]"
        if len(radius) != len(positions):
            raise ValueError("PartType0/SmoothingLength does not align with Coordinates")
        mass = data.get("Masses")
        if mass is None:
            constant_mass = _snapshot_constant_particle_mass(files, ptype)
            if constant_mass is not None:
                mass = np.full(len(positions), constant_mass, dtype=float)
        density_physical, density_physical_source = _physical_gas_density_g_cm3(
            files, header, data.get("Density"),
        ) if header is not None else (None, "snapshot header is unavailable for physical density conversion")
        if density_physical is not None and len(density_physical) != len(positions):
            raise ValueError("PartType0/Density does not align with sampled Coordinates")
        if mass is None or not has_physical_smoothing:
            density_proxy = None
        else:
            mass = np.asarray(mass, dtype=float).reshape(-1)
            if len(mass) != len(positions):
                raise ValueError("PartType0/Masses does not align with Coordinates")
            density_proxy = mass / np.power(np.asarray(radius, dtype=float), 3)
            density_proxy[~np.isfinite(density_proxy) | (density_proxy <= 0)] = np.nan
        result.update({
            "radius": radius,
            "physical_density_g_cm3": density_physical,
            "physical_density_source": density_physical_source,
            "density_proxy_mass_per_radius3": density_proxy,
            "radius_source": radius_source,
        })
    return result


def _render_sealed_blender_bundle(stage, manifest_path, loader, status_path, config, *, timeline=False, enabled=True):
    """Render only after semantic export completion; retain tracked child evidence."""
    from core.native import run_native_command
    from core.foundation import atomic_copy, atomic_json, atomic_text, file_sha256 as digest
    from blender import verify_scene_publication
    stage, manifest_path, loader, status_path = map(Path, (stage, manifest_path, loader, status_path))
    if not enabled or not config.get('visualization', {}).get('blender_render', False):
        return
    source = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    executable = _find_blender_executable(config.get('visualization', {}).get('blender_executable'))
    if executable is None:
        status.update(render_status='SKIPPED', reason='Blender executable was not found; retained loader uses the sealed export')
    else:
        receipt = stage / 'blender_invocation.json'
        log = stage / 'blender_render.log'
        timeout = config.get('visualization', {}).get('blender_timeout_seconds')
        if timeout is not None and (isinstance(timeout, bool) or not np.isfinite(timeout) or timeout <= 0):
            raise ValueError('Blender timeout must be finite and positive')
        inputs = [manifest_path, manifest_path.with_name(manifest_path.name + '.sha256')]
        inputs.extend(stage / item['path'] for item in source['assets'])
        # These compatible public metadata aliases are resealed after rendering.
        # Keep their exact consumed bytes rather than invalidating the receipt.
        retained = stage / '.render_inputs'
        retained.mkdir()
        retained_aliases = {}
        for path in (manifest_path, manifest_path.with_name(manifest_path.name + '.sha256'), status_path):
            target = retained / path.name
            atomic_copy(path, target)
            retained_aliases[str(path.resolve())] = str(target.resolve())
        with log.open('wb') as stream:
            run_native_command([executable, '--background', '--python-exit-code', '1', '--python', str(loader.resolve())],
                cwd=stage, receipt_path=receipt, inputs=inputs, timeout=timeout, stdout=stream, stderr=subprocess.STDOUT)
        invocation = json.loads(receipt.read_text())
        invocation['consumed_input_hashes'] = dict(invocation['input_hashes'])
        invocation['retained_input_aliases'] = retained_aliases
        invocation['input_hashes'] = {retained_aliases.get(path, path): checksum
            for path, checksum in invocation['input_hashes'].items()}
        for path, checksum in invocation['input_hashes'].items():
            if digest(path) != checksum: raise ValueError('Retained Blender invocation input changed')
        invocation['input_retention'] = 'Exact pre-invocation bytes; public metadata aliases resealed after render'
        atomic_json(receipt, invocation)
        movie = stage / status['movie']
        blend = movie.with_name(manifest_path.stem + '.blend') if timeline else movie.with_suffix('.blend')
        verify_scene_publication(blend)
        if not movie.is_file(): raise FileNotFoundError('Tracked Blender invocation did not publish its requested movie')
        status.update(render_status='COMPLETE', movie_exists=True, scene=blend.name,
            native_invocation=receipt.name, native_invocation_sha256=digest(receipt), science_acceptance='NOT_ACCEPTED')
    atomic_json(status_path, status)
    # The input generation used by Blender is retained inside the scene archive;
    # the outer export now seals the complete rendered bundle, excluding self seals.
    seal = manifest_path.with_name(manifest_path.name + '.sha256')
    source['assets'] = [{'path': path.relative_to(stage).as_posix(), 'sha256': digest(path), 'size_bytes': path.stat().st_size}
        for path in sorted(stage.rglob('*')) if path.is_file() and path not in (manifest_path, seal)]
    atomic_json(manifest_path, source)
    atomic_text(seal, digest(manifest_path) + '\n')


@native_run_owner
def make_blender_timeline_loader(run_dir, species="dm", max_points=300_000, *, config=None, output_directory=None, manifest_transform=None):
    """Publish an immutable timeline bundle only after input and asset checks pass."""
    import datetime
    import uuid
    from core.foundation import product_provenance, verify_product_provenance, atomic_json, publish_directory, file_sha256 as digest
    from core.data import coordinate_contract, run_epoch_contract, exact_uint64_ids, numpy_archive_contract, write_dataset_contract
    from core.cosmology import run_proper_time_clock
    if species not in {'dm','gas','stars','black_holes'}:
        raise ValueError('Unsupported Blender species')
    if isinstance(max_points,bool) or not isinstance(max_points,(int,np.integer)) or max_points <= 0:
        raise ValueError('Blender sampled point budget must be a positive integer')
    run_dir = Path(run_dir).resolve()
    cfg = config if config is not None else load_cfg(run_dir)
    snapshots = snapshot_ids(run_dir)
    if not snapshots:
        raise FileNotFoundError('No saved snapshots for Blender timeline')
    clock = run_proper_time_clock(run_dir)
    contract = product_provenance(run_dir,algorithm=__file__,
        source_files=sorted((run_dir/'data').rglob('*.hdf5')),
        selection={'species':species,'max_points':max_points,'display_config':cfg.get('visualization',{}),
                   'blender_config':cfg.get('blender',{})},
        coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation box origin',
            periodic_wrapping='native wrapped snapshots; minimum-image display paths with cropped periodic images',
            velocity_convention='peculiar km/s; linear display interpolation in a is not dynamics'))
    ptype = {'dm':1,'gas':0,'stars':4,'black_holes':5}[species]
    epochs = {}
    for snapshot in snapshots:
        header, files = read_snapshot_header(run_dir,snapshot)
        from analysis import validate_gadget_snapshot_set
        from halos import _unit_conversion
        validate_gadget_snapshot_set(files,int(snapshot))
        _unit_conversion(files)
        actual_a=_hdf5_scalar_attr(header['Time'],'Time')
        epochs[int(snapshot)] = run_epoch_contract(run_dir,int(snapshot),actual_a,clock=clock)
        for path in files:
            with h5py.File(path) as source:
                a=_hdf5_scalar_attr(source['Header'].attrs['Time'],'Time')
                z=_hdf5_scalar_attr(source['Header'].attrs['Redshift'],'Redshift')
                if a != actual_a or not np.isclose(z,1./a-1.,rtol=1e-10,atol=1e-12):
                    raise ValueError('Blender native snapshot epoch metadata disagrees across shards or a/z')
        count = _snapshot_particle_type_total(header,files,ptype)
        required = count*64 + 64*1024**2
        available = _available_ram_bytes()
        fraction = float(cfg.get('blender',{}).get('max_memory_fraction',.70))
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Blender memory fraction must be in (0,1]')
        if available is None or required > available*fraction:
            raise MemoryError('Complete timeline ID validation exceeds the available memory budget')
        chunks=[]
        for path in files:
            with h5py.File(path) as source:
                group=source.get('PartType%d'%ptype)
                if group is None:
                    continue
                if 'ParticleIDs' not in group or group['ParticleIDs'].dtype != np.dtype('uint64'):
                    raise ValueError('Blender native ParticleIDs must be uint64')
                chunks.append(group['ParticleIDs'][:])
        ids=exact_uint64_ids(np.concatenate(chunks) if chunks else np.empty(0,dtype=np.uint64))
        if len(ids) != count:
            raise ValueError('Blender identity count disagrees with native Header')
    parent=run_dir/'results/3d'
    if output_directory is None:parent.mkdir(parents=True,exist_ok=True)
    destination=Path(output_directory).resolve() if output_directory else parent/(
        'timeline_'+species+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    if destination.exists():
        raise FileExistsError('Preserve the previous Blender bundle; choose a new destination')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.blender-timeline-',dir=destination.parent))
    try:
        export_cfg = dict(cfg)
        export_cfg['visualization'] = dict(cfg.get('visualization', {}), blender_render=False)
        loader=_export_blender_timeline(run_dir,species,max_points,config=export_cfg,output_directory=stage)
        manifest_path=stage/('blender_'+species+'_timeline.json')
        manifest=json.loads(manifest_path.read_text())
        manifest['skip_render'] = not bool(cfg.get('visualization', {}).get('blender_render', False))
        manifest['data_contract']=contract
        manifest['science_acceptance']='NOT_ACCEPTED'
        manifest['identity_namespace']=contract['identity_namespaces'][{'dm':'dark_matter','stars':'star_population','black_holes':'black_hole'}.get(species,species)]
        for entry in manifest['snapshots']:
            entry['epoch_contract']=epochs[entry['snapshot']]
            entry['cosmic_time_gyr']=entry['epoch_contract']['cosmic_time_gyr']
        if manifest_transform is not None:
            manifest_transform(manifest)
        for entry in manifest['snapshots']:
            if not entry.get('identity_asset'):continue
            identity_path=stage/entry['identity_asset']
            with h5py.File(identity_path,'r+') as identities:
                ids=identities['ParticleIDs']
                write_dataset_contract(ids,units='identity',identity_namespace=manifest['identity_namespace'],
                    epoch=entry['epoch_contract'],provenance=contract,quantity='Exact saved ParticleID following PLY rows',
                    ancestry={'relationship':'native snapshot rows; glyph copies do not create new particle identities'},
                    missing_policy='Every PLY vertex has an exact identity')
                entry['identity_contract']={'namespace':manifest['identity_namespace'],'epoch':entry['epoch_contract'],
                    'dtype':str(ids.dtype),'row_count':int(ids.size)}
        for interval in manifest.get('intervals',[]):
            if not interval.get('identity_asset'):continue
            identity_path=stage/interval['identity_asset']
            interval_epoch={'canonical_time':'scale_factor','start':epochs[interval['snapshot_start']],
                'end':epochs[interval['snapshot_end']],'interpolation':'minimum-image display interpolation in a; not dynamics'}
            interval['identity_contract']=numpy_archive_contract(identity_path,{name:{'units':'identity',
                'quantity':'Exact native particle identities: '+name,'identity_namespace':manifest['identity_namespace'],
                'ancestry':{'relationship':interval['identity_mapping'],'periodic_copies':'Repeated IDs are views of the same particle'},
                'missing_policy':'No sentinel; unmatched particles are explicit endpoint-only arrays'}
                for name in ('matched_vertex_ids','matched_ids','left_only_ids','right_only_ids')},
                epoch=interval_epoch,coordinates=contract['coordinates'],provenance=contract)
        manifest['assets']=[{'path':p.relative_to(stage).as_posix(),'sha256':digest(p),'size_bytes':p.stat().st_size}
                            for p in sorted(stage.rglob('*')) if p.is_file() and p != manifest_path]
        atomic_json(manifest_path,manifest)
        (stage/(manifest_path.name+'.sha256')).write_text(digest(manifest_path)+'\n')
        verify_product_provenance(run_dir,contract)
        from blender import verify_timeline_assets
        verify_timeline_assets(manifest_path)
        _render_sealed_blender_bundle(stage, manifest_path, loader, stage / ('blender_status_' + species + '.json'), cfg, timeline=True)
        verify_timeline_assets(manifest_path)
        verify_product_provenance(run_dir, contract)
        loader_name=Path(loader).name
        publish_directory(stage,destination)
        return destination/loader_name
    except BaseException as exc:
        if stage.exists():
            atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'failure':{'type':type(exc).__name__,'message':str(exc)},'destination':str(destination)})
        raise


def _export_blender_timeline(run_dir, species="dm", max_points=300_000, *, config=None, output_directory=None):
    """Export an ID-matched, interpolated Blender timeline with honest event handling."""
    if species not in {"dm", "gas", "stars", "black_holes"}:
        raise ValueError("species must be dm, gas, stars, or black_holes")
    cfg = config if config is not None else load_cfg(run_dir)
    visualization = cfg.get("visualization", {})
    blender = cfg.get("blender", {})
    requested_particle_mode = str(blender.get("particle_mode", "sampled")).strip().lower()
    if requested_particle_mode not in {"full", "sampled", "adaptive"}:
        raise ValueError("blender.particle_mode must be full, sampled, or adaptive")
    max_memory_fraction = float(blender.get("max_memory_fraction", 0.70))
    if not np.isfinite(max_memory_fraction) or not 0.0 < max_memory_fraction <= 1.0:
        raise ValueError("blender.max_memory_fraction must be finite and in (0, 1]")
    if not visualization.get("blender_interpolate_motion", True):
        exact_snapshots = snapshot_ids(run_dir)
        if not exact_snapshots:
            raise FileNotFoundError("No HDF5 snapshots found for Blender timeline export")
        exact_memory_plan = None
        exact_fallback = None
        exact_full = False
        if requested_particle_mode in {"full", "adaptive"}:
            exact_memory_plan = _blender_timeline_memory_plan(
                run_dir, species, exact_snapshots, max_memory_fraction,
            )
            if exact_memory_plan["fits"]:
                exact_full = True
            elif requested_particle_mode == "full":
                raise MemoryError(
                    "Full-resolution Blender timeline cannot fit the configured RAM budget: "
                    "estimated peak %(estimated_peak_bytes)s bytes, allowed %(allowed_bytes)s bytes "
                    "from %(available_ram_bytes)s available" % exact_memory_plan
                )
            else:
                exact_fallback = {
                    "reason": "estimated full scene exceeds the configured available-RAM fraction",
                    "memory_estimate": exact_memory_plan,
                }
        return _make_blender_snapshot_timeline_loader(
            run_dir, species, max_points=max_points, full_resolution=exact_full,
            memory_plan=exact_memory_plan, requested_particle_mode=requested_particle_mode,
            adaptive_fallback=exact_fallback, output_directory=output_directory, config=cfg,
        )
    outdir = Path(output_directory) if output_directory else Path(run_dir) / "results/3d"
    outdir.mkdir(parents=True, exist_ok=True)
    snapshots = snapshot_ids(run_dir)
    if not snapshots:
        raise FileNotFoundError("No HDF5 snapshots found for Blender timeline export")
    ptype = {"gas": 0, "dm": 1, "stars": 4, "black_holes": 5}[species]
    snapshot_rows = []
    for snap in snapshots:
        header, _files = read_snapshot_header(run_dir, snap)
        scale_factor = _snapshot_scale_factor(header)
        if "BoxSize" in header:
            snapshot_box = _scalar_float(header["BoxSize"], "snapshot Header/BoxSize") / 1000.0
            if not np.isclose(snapshot_box, float(cfg["simulation"]["box_mpc_h"]), rtol=1.0e-8, atol=0.0):
                raise ValueError(
                    "Snapshot %03d Header/BoxSize disagrees with the configured Blender timeline box" % int(snap)
                )
        snapshot_rows.append((scale_factor, int(snap), 1.0 / scale_factor - 1.0))
    snapshot_rows.sort()
    scale_factors = np.asarray([row[0] for row in snapshot_rows], dtype=float)
    if len(scale_factors) > 1 and np.any(np.diff(scale_factors) <= 0.0):
        raise ValueError("Blender timeline snapshots must have distinct, increasing scale factors")
    memory_plan = None
    adaptive_fallback = None
    resolved_particle_mode = "sampled"
    if requested_particle_mode in {"full", "adaptive"}:
        memory_plan = _blender_timeline_memory_plan(
            run_dir, species, [row[1] for row in snapshot_rows], max_memory_fraction,
        )
        if memory_plan["fits"]:
            resolved_particle_mode = "full"
        elif requested_particle_mode == "full":
            raise MemoryError(
                "Full-resolution Blender timeline cannot fit the configured RAM budget: "
                "estimated peak %(estimated_peak_bytes)s bytes, allowed %(allowed_bytes)s bytes "
                "from %(available_ram_bytes)s available" % memory_plan
            )
        else:
            adaptive_fallback = {
                "reason": "estimated full scene exceeds the configured available-RAM fraction",
                "memory_estimate": memory_plan,
            }
    if resolved_particle_mode == "full":
        return _make_blender_snapshot_timeline_loader(
            run_dir, species, max_points=None, full_resolution=True, memory_plan=memory_plan,
            requested_particle_mode=requested_particle_mode, output_directory=output_directory, config=cfg,
        )
    total_budget = int(visualization.get("blender_sequence_max_points", 800_000))
    intervals_count = max(len(snapshot_rows) - 1, 1)
    per_snapshot = min(int(max_points), max(1, total_budget // (2 * intervals_count)))
    if per_snapshot <= 0:
        raise ValueError("visualization.blender_sequence_max_points is too small for the requested timeline")
    if len(snapshot_rows) == 1:
        return _make_blender_snapshot_timeline_loader(
            run_dir, species, max_points=per_snapshot,
            requested_particle_mode=requested_particle_mode, adaptive_fallback=adaptive_fallback, output_directory=output_directory, config=cfg,
        )

    from core.cosmology import run_proper_time_clock, interpolate_proper_time
    cosmic_time_scale, cosmic_time_gyr, cosmic_time_status = run_proper_time_clock(run_dir)

    def proper_time_for_scale(scale_factor):
        if cosmic_time_scale is None:
            return None
        return interpolate_proper_time(cosmic_time_scale, cosmic_time_gyr, scale_factor)

    snap_ids = [row[1] for row in snapshot_rows]
    selected_ids = _sample_timeline_id_union(run_dir, snap_ids, ptype, per_snapshot)
    if not len(selected_ids):
        raise ValueError("No %s particle IDs were found in the saved outputs" % species)
    box = float(cfg["simulation"]["box_mpc_h"])
    samples = []
    for scale_factor, snap, redshift in snapshot_rows:
        header, _ = read_snapshot_header(run_dir, snap)
        sample = _read_timeline_species_sample(
            run_dir, snap, ptype, selected_ids, species, box, header=header,
        )
        sample.update({
            "scale_factor": float(scale_factor), "snapshot": int(snap), "redshift": float(redshift),
            "cosmic_time_gyr": proper_time_for_scale(float(scale_factor)),
        })
        samples.append(sample)

    scene_vertex_estimate = _estimate_blender_timeline_vertices(samples, species, box)
    while scene_vertex_estimate > total_budget and len(selected_ids) > 1:
        reduced_limit = max(
            1,
            min(len(selected_ids) - 1,
                int(len(selected_ids) * float(total_budget) / float(scene_vertex_estimate) * 0.95)),
        )
        if reduced_limit >= len(selected_ids):
            reduced_limit = len(selected_ids) - 1
        selected_ids = _lowest_hash_ids(selected_ids, reduced_limit)
        for sample in samples:
            mask = np.isin(sample["ids"], selected_ids, assume_unique=True)
            for field_name in (
                    "ids", "positions", "radius", "raw_density",
                    "physical_density_g_cm3", "density_proxy_mass_per_radius3",
                    "temperature_log10", "stellar_age_gyr",
                    "black_hole_log10_mass_code_units", "black_hole_mass_weight"):
                values = sample.get(field_name)
                if values is not None:
                    sample[field_name] = values[mask]
        scene_vertex_estimate = _estimate_blender_timeline_vertices(samples, species, box)
    if scene_vertex_estimate > total_budget:
        raise ValueError(
            "Blender timeline periodic-image geometry needs %d vertices, above the configured budget %d; increase visualization.blender_sequence_max_points"
            % (scene_vertex_estimate, total_budget)
        )

    gas_density_normalization = None
    if species == "gas":
        nonempty_samples = [sample for sample in samples if len(sample["ids"])]

        def complete_density_series(field_name):
            return bool(nonempty_samples) and all(
                sample.get(field_name) is not None
                and np.any(np.isfinite(sample[field_name]) & (sample[field_name] > 0))
                for sample in nonempty_samples
            )

        if complete_density_series("physical_density_g_cm3"):
            density_field = "physical_density_g_cm3"
            density_source = "GADGET PartType0/Density with declared proper g/cm^3 conversion"
            density_unit = "g/cm^3"
            density_is_physical = True
        elif complete_density_series("density_proxy_mass_per_radius3"):
            density_field = "density_proxy_mass_per_radius3"
            density_source = "Masses/SmoothingLength^3 fallback; relative proxy, not an absolute physical density"
            density_unit = "relative code-mass/(Mpc/h)^3 proxy"
            density_is_physical = False
        else:
            density_field = None
            density_source = "equal-weight fallback; no complete positive density field was available across outputs"
            density_unit = None
            density_is_physical = False

        for sample in samples:
            sample["raw_density"] = sample.get(density_field) if density_field else None
        raw_chunks = [sample["raw_density"][np.isfinite(sample["raw_density"])]
                      for sample in samples if sample.get("raw_density") is not None]
        raw_chunks = [chunk[chunk > 0] for chunk in raw_chunks if len(chunk)]
        if raw_chunks:
            pooled_density = np.concatenate(raw_chunks)
            global_median = float(np.median(pooled_density))
            upper_clip = float(np.percentile(pooled_density, 99.5))
            normalized_cap = max(upper_clip / global_median, 1.0)
            for sample in samples:
                raw = sample.get("raw_density")
                if raw is None:
                    sample["density_weight"] = np.full(len(sample["positions"]), 0.45, dtype="<f4")
                    sample["density_weight_source"] = "equal-weight fallback; density field unavailable"
                else:
                    valid = np.isfinite(raw) & (raw > 0)
                    weights = np.zeros(len(raw), dtype="<f4")
                    weights[valid] = np.asarray(
                        0.45 * np.minimum(raw[valid] / global_median, normalized_cap),
                        dtype="<f4",
                    )
                    sample["density_weight"] = weights
                    sample["density_weight_source"] = density_source
            gas_density_normalization = {
                "status": "COMPLETE",
                "source": density_source,
                "source_unit": density_unit,
                "normalization": "one sampled-particle median across all saved outputs maps to Blender density weight 0.45; values are capped at the pooled 99.5th percentile for stable volume rendering",
                "raw_density_weight_median": global_median,
                "raw_density_weight_p99_5": upper_clip,
                "density_weight_upper_clip": float(0.45 * normalized_cap),
                "valid_sample_count": int(sum(len(chunk) for chunk in raw_chunks)),
                "field_interpretation": (
                    "unit-converted proper gas mass density; relative, median-normalized Blender transfer scale"
                    if density_is_physical else
                    "relative mass-per-smoothed-volume proxy; not absolute physical density; Blender shading remains art-directed"
                ),
                "physical_density_available_for_every_nonempty_output": bool(density_is_physical),
            }
        else:
            for sample in samples:
                sample["density_weight"] = np.full(len(sample["positions"]), 0.45, dtype="<f4")
                sample["density_weight_source"] = density_source
            gas_density_normalization = {
                "status": "EQUAL_WEIGHT_FALLBACK",
                "reason": density_source,
            }

    black_hole_mass_normalization = None
    if species == "black_holes":
        mass_files = []
        for _scale, snapshot, _redshift in snapshot_rows:
            _header, files = read_snapshot_header(run_dir, snapshot)
            mass_files.extend(files)
        black_hole_mass_normalization = _black_hole_particle_mass_normalization(mass_files)
        mass_range = black_hole_mass_normalization["log10_mass_range_code_units"]
        for sample in samples:
            raw = sample.get("black_hole_log10_mass_code_units")
            if raw is None:
                continue
            weight = np.full(len(raw), -1., dtype="<f4")
            valid = np.isfinite(raw)
            if mass_range is not None:
                mass_low, mass_high = mass_range
                weight[valid] = (np.clip((raw[valid] - mass_low) / (mass_high - mass_low), 0., 1.)
                                 if mass_high > mass_low else .5)
            sample["black_hole_mass_weight"] = weight

    intervals = []
    frames_per_snapshot = int(visualization.get("blender_frames_per_snapshot", 12))
    for index, (left, right) in enumerate(zip(samples[:-1], samples[1:])):
        ids_a, ids_b = left["ids"], right["ids"]
        matched, index_a, index_b = np.intersect1d(
            ids_a, ids_b, assume_unique=True, return_indices=True,
        )
        left_only = np.setdiff1d(ids_a, matched, assume_unique=True)
        right_only = np.setdiff1d(ids_b, matched, assume_unique=True)
        left_lookup = np.searchsorted(ids_a, left_only)
        right_lookup = np.searchsorted(ids_b, right_only)
        match_start = left["positions"][index_a]
        match_end = right["positions"][index_b]
        start_match_attrs = {}
        end_match_attrs = {}
        if species == "gas":
            start_match_attrs = {
                "radius": left["radius"][index_a],
                "density_weight": left["density_weight"][index_a],
                "physical_density_g_cm3": (
                    np.nan_to_num(left["physical_density_g_cm3"][index_a], nan=0.0)
                    if left.get("physical_density_g_cm3") is not None else
                    np.zeros(len(index_a), dtype="<f4")
                ),
            }
            end_match_attrs = {
                "radius": right["radius"][index_b],
                "density_weight": right["density_weight"][index_b],
                "physical_density_g_cm3": (
                    np.nan_to_num(right["physical_density_g_cm3"][index_b], nan=0.0)
                    if right.get("physical_density_g_cm3") is not None else
                    np.zeros(len(index_b), dtype="<f4")
                ),
            }
        for field in ("temperature_log10", "stellar_age_gyr", "black_hole_mass_weight"):
            if left.get(field) is not None and right.get(field) is not None:
                start_match_attrs[field] = left[field][index_a]
                end_match_attrs[field] = right[field][index_b]
        start_match_attrs["particle_ids"] = matched
        trajectory_start, trajectory_end, trajectory_start_attrs = _periodic_trajectory_copies(
            match_start, match_end, box, start_match_attrs,
        )
        _unused_start, _unused_end, trajectory_end_attrs = _periodic_trajectory_copies(
            match_start, match_end, box, end_match_attrs,
        )
        prefix = "blender_%s_interval_%04d" % (species, index)
        interval = {
            "index": int(index),
            "snapshot_start": int(left["snapshot"]), "snapshot_end": int(right["snapshot"]),
            "scale_factor_start": float(left["scale_factor"]), "scale_factor_end": float(right["scale_factor"]),
            "redshift_start": float(left["redshift"]), "redshift_end": float(right["redshift"]),
            "cosmic_time_start_gyr": left.get("cosmic_time_gyr"),
            "cosmic_time_end_gyr": right.get("cosmic_time_gyr"),
            "frame_start": int(index * frames_per_snapshot + 1),
            "frame_end": int((index + 1) * frames_per_snapshot),
            "matched_particle_count": int(len(matched)),
            "left_only_event_count": int(len(left_only)), "right_only_event_count": int(len(right_only)),
            "periodic_image_count": int(len(trajectory_start) - len(matched)),
            "particle_event_policy": "unmatched particles remain at their exact endpoint positions and switch at the saved-output boundary; no unmatched trajectory is invented",
        }
        if len(matched):
            start_path = outdir / (prefix + "_matched_start.ply")
            end_path = outdir / (prefix + "_matched_end.ply")
            start_attrs = trajectory_start_attrs
            end_attrs = trajectory_end_attrs
            _write_blender_particle_ply(
                start_path, trajectory_start,
                start_attrs.get("radius"), start_attrs.get("density_weight"),
                start_attrs.get("physical_density_g_cm3"),
                start_attrs.get("temperature_log10"), start_attrs.get("stellar_age_gyr"),
                start_attrs.get("black_hole_mass_weight"),
            )
            _write_blender_particle_ply(
                end_path, trajectory_end,
                end_attrs.get("radius"), end_attrs.get("density_weight"),
                end_attrs.get("physical_density_g_cm3"),
                end_attrs.get("temperature_log10"), end_attrs.get("stellar_age_gyr"),
                end_attrs.get("black_hole_mass_weight"),
            )
            interval["matched_start_ply"] = start_path.name
            interval["matched_end_ply"] = end_path.name
        else:
            interval["matched_start_ply"] = None
            interval["matched_end_ply"] = None
        for side, ids, sample, rows in (
            ("left_only", left_only, left, left_lookup),
            ("right_only", right_only, right, right_lookup),
        ):
            if len(ids):
                event_path = outdir / (prefix + "_" + side + ".ply")
                _write_blender_particle_ply(
                    event_path, sample["positions"][rows],
                    sample.get("radius", None)[rows] if species == "gas" else None,
                    sample.get("density_weight", None)[rows] if species == "gas" else None,
                    (
                        np.nan_to_num(sample["physical_density_g_cm3"][rows], nan=0.0)
                        if species == "gas" and sample.get("physical_density_g_cm3") is not None else
                        np.zeros(len(rows), dtype="<f4") if species == "gas" else None
                    ),
                    sample.get("temperature_log10", None)[rows] if sample.get("temperature_log10") is not None else None,
                    sample.get("stellar_age_gyr", None)[rows] if sample.get("stellar_age_gyr") is not None else None,
                    sample.get("black_hole_mass_weight", None)[rows] if sample.get("black_hole_mass_weight") is not None else None,
                )
                interval[side + "_ply"] = event_path.name
            else:
                interval[side + "_ply"] = None
        interval["logical_particle_count"] = int(len(matched) + len(left_only) + len(right_only))
        interval["rendered_vertex_count"] = int(
            (2 if species == "gas" else 1) * len(trajectory_start) + len(left_only) + len(right_only)
        )
        identity_path = outdir / (prefix + "_identities.npz")
        np.savez(identity_path, matched_vertex_ids=trajectory_start_attrs["particle_ids"],
                 matched_ids=matched, left_only_ids=left_only, right_only_ids=right_only)
        interval["identity_asset"] = identity_path.name
        interval["identity_mapping"] = "matched_vertex_ids follows PLY vertex rows including periodic copies; endpoint-only IDs follow endpoint PLY rows"
        intervals.append(interval)

    manifest_path = outdir / ("blender_%s_timeline.json" % species)
    fps = int(visualization.get("movie_fps", 24))
    render_quality = str(visualization.get("blender_render_quality", "preview")).lower()
    manifest = {
        "schema_version": 3,
        "model": str(cfg["cosmology"]["model"]),
        "species": species,
        "box_mpc_h": box,
        "fps": fps,
        "frames_per_snapshot": frames_per_snapshot,
        "render_quality": render_quality,
        "camera_orbit": bool(visualization.get("blender_timeline_camera_orbit", False)),
        "sampling": {
            "particle_mode": "sampled",
            "requested_particle_mode": requested_particle_mode,
            "adaptive_fallback": adaptive_fallback,
            "per_snapshot_max_points": int(per_snapshot),
            "total_point_budget": int(total_budget),
            "total_rendered_vertex_estimate": int(scene_vertex_estimate),
            "selected_union_id_count": int(len(selected_ids)),
            "method": "deterministic SplitMix64 bottom-hash over the union of particle IDs across all outputs",
            "matched_motion": "exact ParticleID joins; shortest-image linear interpolation in scale factor",
            "birth_death_policy": "unmatched particles are displayed at endpoint positions and switch at saved-output boundaries",
            "periodic_boundary_policy": "minimum-image unwrapped tracks with temporary neighboring-box image copies at crossed faces",
        },
        "gas_density_normalization": gas_density_normalization,
        "black_hole_mass_normalization": black_hole_mass_normalization,
        "cosmic_time": cosmic_time_status,
        "scalar_display_conventions": {"stellar_age_gyr": "saved epoch proper time minus stellar formation proper time [Gyr]; negative sentinel is unknown/grey", "temperature_log10": "log10(T/K); -1e30 sentinel is unknown/effective-EOS grey; no temperature clipping"},
        "snapshots": [
            {"snapshot": int(sample["snapshot"]), "scale_factor": sample["scale_factor"],
             "redshift": sample["redshift"], "cosmic_time_gyr": sample.get("cosmic_time_gyr"),
             "sampled_particle_count": int(len(sample["ids"])),
             "halo_overlays": _blender_halo_overlays(
                 run_dir, int(sample["snapshot"]), cfg, box,
                 maximum_halos=int(visualization.get("max_matched_halo_overlays", 250)),
             )}
            for sample in samples
        ],
        "intervals": intervals,
        "frame_end": int(len(intervals) * frames_per_snapshot),
        "show_halo_ids_and_masses": bool(visualization.get("blender_galaxy_overlays", True)),
        "maximum_halo_labels_per_model_output": int(visualization.get("blender_max_galaxies", 12)),
        "skip_render": not bool(visualization.get("blender_render", False)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    scene_script = copy_blender_runtime(outdir)
    movie = outdir / ("%s_cinematic_timeline.mp4" % species)
    loader = outdir / ("blender_timeline_loader_%s.py" % species)
    loader.write_text(
        "# Run this in Blender's Scripting workspace to build and render the ID-matched timeline.\n"
        "from pathlib import Path\n"
        "import sys\n"
        "base = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(base))\n"
        "from blender import build_timeline_scene\n"
        "build_timeline_scene(str(base / %r), str(base / %r))\n"
        % (manifest_path.name, movie.name),
        encoding="utf-8",
    )
    status = {
        "status": "EXPORTED", "timeline_manifest": manifest_path.name,
        "interactive_loader": loader.name, "scene_script": str(scene_script),
        "movie": movie.name, "snapshot_count": len(samples),
        "interval_count": len(intervals), "selected_union_id_count": int(len(selected_ids)),
        "per_snapshot_max_points": int(per_snapshot), "total_point_budget": int(total_budget),
        "particle_mode": "sampled", "requested_particle_mode": requested_particle_mode,
        "adaptive_fallback": adaptive_fallback,
        "render_quality": render_quality, "render_status": "NOT_REQUESTED",
        "camera_orbit": bool(visualization.get("blender_timeline_camera_orbit", False)),
        "motion_interpolation": "exact ParticleID matches; periodic minimum-image linear interpolation in scale factor",
        "birth_death_policy": "endpoint-position events at saved-output boundaries; unmatched trajectories are not invented",
        "gas_density_normalization": gas_density_normalization,
    }
    if visualization.get("blender_render", False):
        executable_path = _find_blender_executable(visualization.get("blender_executable"))
        if executable_path is None:
            status["render_status"] = "SKIPPED"
            status["reason"] = "Blender executable was not found; run the generated timeline loader inside Blender"
        else:
            try:
                completed = subprocess.run(
                    [executable_path, "--background", "--python-exit-code", "1", "--python", str(scene_script), "--",
                     "--timeline", str(manifest_path.resolve()), str(movie.resolve())],
                    check=True, capture_output=True, text=True,
                )
                status["render_log_tail"] = completed.stdout[-2000:]
                status["movie_exists"] = movie.is_file()
                status["render_status"] = "COMPLETE" if movie.is_file() else "FAILED"
                if not movie.is_file():
                    status["reason"] = "Blender returned successfully but did not create the requested timeline movie"
            except (OSError, subprocess.CalledProcessError) as exc:
                status["render_status"] = "FAILED"
                status["reason"] = str(exc)
    (outdir / ("blender_status_%s.json" % species)).write_text(json.dumps(status, indent=2), encoding="utf-8")
    return loader


@native_run_owner
def make_blender_loader(run_dir, snap=None, species="dm", max_points=300_000,
                        sequence=None, _allow_render=True, _particle_mode=None,
                        _black_hole_mass_normalization=None, _output_directory=None):
    """Publish standalone scenes atomically; timeline callers own their outer stage."""
    import uuid
    from analysis import validate_gadget_snapshot_set
    from halos import _unit_conversion
    from core.data import coordinate_contract, run_epoch_contract, write_dataset_contract
    from core.foundation import product_provenance, verify_product_provenance, atomic_json, publish_directory, file_sha256 as digest
    root=Path(run_dir).expanduser().resolve()
    cfg=load_cfg(root)
    if species not in {'dm','gas','stars','black_holes'}:
        raise ValueError('Unsupported Blender species')
    if isinstance(max_points,bool) or not isinstance(max_points,(int,np.integer)) or max_points<=0:
        raise ValueError('Blender point budget must be a positive integer')
    if sequence is None:sequence=snap is None and bool(cfg.get('visualization',{}).get('blender_timeline',True))
    if sequence:
        return make_blender_timeline_loader(root,species,max_points=max_points)
    options={'sequence':False,'_allow_render':_allow_render,'_particle_mode':_particle_mode,
        '_black_hole_mass_normalization':_black_hole_mass_normalization}
    if _output_directory is not None:
        # The timeline's outer transaction verifies inputs and publishes all
        # scenes together. Do not nest a bundle inside its asset directory.
        return _export_blender_single(root,snap,species,max_points,_output_directory=_output_directory,**options)
    available=snapshot_ids(root)
    if not available:raise FileNotFoundError('No saved snapshots for Blender export')
    snap=available[-1] if snap is None else int(snap)
    if snap not in available:raise ValueError('Requested Blender snapshot is absent')
    header,files=read_snapshot_header(root,snap)
    validate_gadget_snapshot_set(files,snap);_unit_conversion(files)
    epoch=run_epoch_contract(root,snap,_hdf5_scalar_attr(header['Time'],'Time'))
    coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
        periodic_wrapping='PLY positions wrapped to native fundamental box',velocity_convention='not applicable to saved-position glyphs')
    contract=product_provenance(root,algorithm=__file__,snapshot=snap,
        helper_sources=[Path(__file__).with_name('blender.py'),Path(__file__).with_name('foundation.py')],
        source_files=files,coordinates=coordinates,
        selection={'species':species,'max_points':int(max_points),'particle_mode':_particle_mode or cfg.get('blender',{}).get('particle_mode','sampled'),
            'visualization':cfg.get('visualization',{}),'blender':cfg.get('blender',{})})
    parent=root/'results/3d';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.blender-single-',dir=parent))
    destination=parent/('snapshot_%03d_%s_'%(snap,species)+uuid.uuid4().hex)
    try:
        export_options = dict(options, _allow_render=False)
        loader=_export_blender_single(root,snap,species,max_points,_output_directory=stage,**export_options)
        status_path=stage/('blender_status_'+species+'.json')
        status=json.loads(status_path.read_text())
        namespace=contract['identity_namespaces'][{'dm':'dark_matter','gas':'gas','stars':'star_population','black_holes':'black_hole'}[species]]
        identity=stage/status['identity_asset']
        from core.data import exact_uint64_ids
        from blender import verify_particle_ply_payload, verify_single_scene_assets
        if verify_particle_ply_payload(stage/status['sample_ply'])!=status['particle_count']:
            raise ValueError('Blender PLY count disagrees with saved export status')
        with h5py.File(identity,'r+') as identities:
            ids=identities['ParticleIDs']
            # np.unique makes copies; account for identity verification as well
            # as the scene estimate before loading a full sidecar into RAM.
            available_ram=_available_ram_bytes()
            if available_ram is None:raise MemoryError('Available RAM is unknown for exact Blender identity verification')
            identity_budget=float(cfg.get('blender',{}).get('max_memory_fraction',.70))*available_ram
            if len(ids)*40+64*1024**2>identity_budget:
                raise MemoryError('Blender exact identity verification exceeds configured available-RAM fraction')
            exact_uint64_ids(ids[:])
            if len(ids)!=status['particle_count']:raise ValueError('Blender PLY identity count mismatch')
            write_dataset_contract(ids,units='identity',identity_namespace=namespace,epoch=epoch,
                provenance={'manifest':'blender_single_contract.json','native_inputs':contract['input_hashes']},
                quantity='exact native particle identity of corresponding PLY vertex row',
                missing_policy='one exact identity per saved vertex; no interpolated or synthetic vertices')
            identities.attrs['coordinate_conventions_json']=json.dumps(coordinates,sort_keys=True)
            identities.attrs['epoch_contract_json']=json.dumps(epoch,sort_keys=True,allow_nan=False)
        status.update(science_acceptance='NOT_ACCEPTED',validation_state='VALIDATION_PENDING',
            epoch_contract=epoch,identity_namespace=namespace,data_contract_manifest='blender_single_contract.json')
        atomic_json(status_path,status)
        loader=Path(loader)
        loader_text=loader.read_text()
        loader_text=loader_text.replace('from blender import build_scene\n',
            'from blender import build_scene, verify_single_scene_assets\n'
            "verify_single_scene_assets(base / 'blender_single_contract.json')\n",1)
        loader.write_text(loader_text,encoding='utf-8')
        verify_product_provenance(root,contract)
        assets=[{'path':path.relative_to(stage).as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size}
            for path in sorted(stage.rglob('*')) if path.is_file()]
        atomic_json(stage/'blender_single_contract.json',{'schema':'blender-single-publication-v1',
            'science_acceptance':'NOT_ACCEPTED','validation_state':'VALIDATION_PENDING','provenance':contract,
            'epoch':epoch,'coordinates':coordinates,'identity_namespace':namespace,'assets':assets,
            'representation':'saved snapshot positions represented as display glyphs; no inferred dynamics'})
        (stage/'blender_single_contract.json.sha256').write_text(digest(stage/'blender_single_contract.json')+'\n')
        verify_single_scene_assets(stage/'blender_single_contract.json')
        _render_sealed_blender_bundle(stage, stage/'blender_single_contract.json', loader, status_path, cfg, enabled=_allow_render)
        verify_single_scene_assets(stage/'blender_single_contract.json')
        verify_product_provenance(root, contract)
        publish_directory(stage,destination)
        return destination/Path(loader).name
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)},'destination':str(destination)})
        raise


def _export_blender_single(run_dir, snap=None, species="dm", max_points=300_000,
                        sequence=None, _allow_render=True, _particle_mode=None,
                        _black_hole_mass_normalization=None, _output_directory=None):
    """Export sampled or full-resolution particles and an optional Blender movie."""
    if species not in {"dm", "gas", "stars", "black_holes"}:
        raise ValueError("species must be dm, gas, stars, or black_holes")
    cfg = load_cfg(run_dir)
    visualization = cfg.get("visualization", {})
    blender = cfg.get("blender", {})
    requested_particle_mode = str(
        _particle_mode or blender.get("particle_mode", "sampled")
    ).strip().lower()
    if requested_particle_mode not in {"full", "sampled", "adaptive"}:
        raise ValueError("blender.particle_mode must be full, sampled, or adaptive")
    max_memory_fraction = float(blender.get("max_memory_fraction", 0.70))
    if not np.isfinite(max_memory_fraction) or not 0.0 < max_memory_fraction <= 1.0:
        raise ValueError("blender.max_memory_fraction must be finite and in (0, 1]")
    if sequence is None:
        sequence = snap is None and bool(visualization.get("blender_timeline", True))
    if sequence:
        return make_blender_timeline_loader(run_dir, species, max_points=max_points)
    outdir = Path(_output_directory) if _output_directory else Path(run_dir) / "results/3d"
    outdir.mkdir(parents=True, exist_ok=True)
    available = snapshot_ids(run_dir)
    if not available:
        raise FileNotFoundError("No HDF5 snapshots found for Blender export")
    snap = available[-1] if snap is None else int(snap)
    if snap not in available:
        raise ValueError("Snapshot %d is not present; available snapshots: %s" % (snap, available))
    ptype = {"gas": 0, "dm": 1, "stars": 4, "black_holes": 5}[species]
    files = find_hdf5_files(Path(run_dir) / "data", "snapshot_%03d" % snap)
    adaptive_fallback = None
    if requested_particle_mode in {"full", "adaptive"}:
        header, _ = read_snapshot_header(run_dir, snap)
        particle_count = _snapshot_particle_type_total(header, files, ptype)
        memory_plan = _blender_memory_estimate(particle_count, max_memory_fraction)
        if requested_particle_mode == "full" and not memory_plan["fits"]:
            raise MemoryError(
                "Full-resolution Blender export cannot fit the configured RAM budget: "
                "estimated peak %(estimated_peak_bytes)s bytes, allowed %(allowed_bytes)s bytes "
                "from %(available_ram_bytes)s available" % memory_plan
            )
        if memory_plan["fits"] and particle_count > 0:
            full_ply = outdir / ("snapshot_%03d_%s_full.ply" % (snap, species))
            export = _write_blender_snapshot_ply_streaming(
                files, ptype, full_ply, float(cfg["simulation"]["box_mpc_h"]),
                run_dir=run_dir, header=header,
                black_hole_mass_normalization=_black_hole_mass_normalization,
            )
            if int(export["particle_count"]) != int(particle_count):
                raise ValueError(
                    "Full-resolution Blender PLY count disagrees with the snapshot Header count"
                )
            box = float(cfg["simulation"]["box_mpc_h"])
            fps = int(visualization.get("movie_fps", 24))
            render_quality = str(visualization.get("blender_render_quality", "preview")).lower()
            status = {
                "status": "EXPORTED",
                "particle_mode": "full",
                "requested_particle_mode": requested_particle_mode,
                "adaptive_fallback": None,
                "sample_ply": full_ply.name,
                "particle_count": export["particle_count"], "identity_asset":export["identity_asset"],
                "coordinate_bytes": memory_plan["coordinate_bytes"],
                "memory_estimate": memory_plan,
                "scene_type": "one full-particle point mesh; geometry nodes render the vertices",
                "gas_radius_source": export["gas_radius_source"],
                "gas_density_method": export["gas_density_method"],
                "gas_physical_density_unit": None,
                "gas_physical_density_source": None,
                "black_hole_mass_normalization": export["black_hole_mass_normalization"],
                "scalar_display_conventions": {
                    "temperature_log10": "log10(T/K); unknown/effective-EOS gas is grey",
                    "stellar_age_gyr": "matched CLASS proper-time age in Gyr; unknown is grey",
                },
            }
            return _finish_blender_single_export(
                run_dir, outdir, species, snap, full_ply, box, fps, render_quality,
                status, allow_render=_allow_render, memory_plan=memory_plan,
            )
        if requested_particle_mode == "full":
            raise ValueError("The selected snapshot contains no particles of type " + species)
        adaptive_fallback = {
            "reason": "estimated full scene exceeds the configured available-RAM fraction",
            "memory_estimate": memory_plan,
        }
    header, _ = read_snapshot_header(run_dir, snap)
    fields = read_particle_fields(files, ptype, _blender_species_fields(species), max_points=max_points)
    coordinates = fields.get("Coordinates")
    if coordinates is None or not len(coordinates):
        raise ValueError("The selected snapshot contains no particles of type " + species)
    box = float(cfg["simulation"]["box_mpc_h"])
    xyz = np.mod(convert_pinned_native(coordinates,'position_comoving_mpc_h'), box).astype("<f4")
    attributes = _blender_species_attributes(run_dir, files, ptype, species, box, header, fields, xyz)
    bh_normalization = None
    if species == "black_holes":
        bh_normalization = _black_hole_mass_normalization or _black_hole_particle_mass_normalization(files)
        raw = attributes.get("black_hole_log10_mass_code_units")
        if raw is not None and bh_normalization["status"] == "MEASURED":
            low, high = bh_normalization["log10_mass_range_code_units"]
            attributes["black_hole_mass_weight"] = (
                np.full(len(xyz), .5, dtype="<f4") if high == low else
                np.clip((raw - low) / (high - low), 0., 1.).astype("<f4")
            )
    ply = outdir / ("snapshot_%03d_%s_sample.ply" % (snap, species))
    physical_density = attributes.get("physical_density_g_cm3")
    _write_blender_particle_ply(
        ply, xyz, attributes.get("radius"),
        np.full(len(xyz), .45, dtype="<f4") if species == "gas" else None,
        np.nan_to_num(physical_density, nan=0.) if physical_density is not None else None,
        attributes.get("temperature_log10"), attributes.get("stellar_age_gyr"),
        black_hole_mass_weight=attributes.get("black_hole_mass_weight"),
    )
    identity_path=ply.with_suffix('.identities.hdf5')
    from core.data import exact_uint64_ids,write_dataset_contract
    with h5py.File(identity_path,'w') as identities:
        ids=exact_uint64_ids(fields['ParticleIDs'])
        dataset=identities.create_dataset('ParticleIDs',data=ids,chunks=True,fletcher32=True)
        write_dataset_contract(dataset,units='identity',missing_policy='all PLY rows have an exact saved identity')
        identities.attrs['mapping']='ParticleIDs[row] is the native ID of PLY vertex row'
    status = {
        "status": "EXPORTED", "particle_mode": "sampled", "identity_asset":identity_path.name,
        "requested_particle_mode": requested_particle_mode, "adaptive_fallback": adaptive_fallback,
        "sample_ply": ply.name, "sampled_particle_count": len(xyz), "particle_count": len(xyz),
        "scene_type": "saved particle positions rendered as point glyphs",
        "gas_radius_source": attributes.get("radius_source"),
        "gas_density_method": "uniform glyph display weight; not a physical density" if species == "gas" else None,
        "gas_physical_density_unit": "g/cm^3" if physical_density is not None else None,
        "gas_physical_density_source": attributes.get("physical_density_source"),
        "black_hole_mass_normalization": bh_normalization,
        "scalar_display_conventions": {
            "temperature_log10": "log10(T/K); unknown/effective-EOS gas is grey",
            "stellar_age_gyr": "matched CLASS proper-time age in Gyr; unknown is grey",
        },
    }
    return _finish_blender_single_export(
        run_dir, outdir, species, snap, ply, box, int(visualization.get("movie_fps", 24)),
        str(visualization.get("blender_render_quality", "preview")).lower(), status,
        allow_render=_allow_render,
    )


def visualization_main():
    """Parse command arguments and dispatch this module's documented workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--compare-with")
    p.add_argument("--product", action="append", choices=[
        "maps", "movie", "gas_movie", "temperature_movie", "stars_movie",
        "halo_movie", "xdmf", "blender",
    ],
                   help="select one or more products explicitly")
    p.add_argument("--open-3d", action="store_true", help="Open the latest snapshot in an interactive 3-D window")
    p.add_argument("--open-3d-all", action="store_true", help="Render every particle in each visible species layer; can use substantial GPU memory")
    p.add_argument("--snapshot", type=int, help="Snapshot number for --open-3d")
    p.add_argument("--species", choices=["gas", "dm", "stars", "black_holes"],
                   help="required particle type for the interactive 3-D view or Blender export")
    p.add_argument(
        "--max-points", type=int,
        help="Maximum stable-ID sample per enabled particle layer; ignored by --open-3d-all",
    )
    p.add_argument("--field", choices=["dm", "gas", "gas_temperature", "stars", "total"], action="append", help="Map field; repeat to export several fields")
    p.add_argument("--projection", choices=["xy", "xz", "yz"], action="append", help="Map projection; repeat to export several orientations")
    p.add_argument("--slice-thickness-mpc-h", type=float,
                   help="Project only a periodic slab of this thickness along the map line of sight")
    p.add_argument("--slice-center-mpc-h", type=float,
                   help="Explicit slab center along the line of sight")
    args = p.parse_args()
    if args.open_3d or args.open_3d_all:
        if args.species == "black_holes":
            p.error("PartType5 display is supported by Blender exports; the interactive viewer supports gas, dm and stars")
        if args.species is None:
            p.error("the 3-D viewer requires an explicit --species")
        if args.snapshot is None:
            p.error("the 3-D viewer requires an explicit --snapshot")
        if args.slice_thickness_mpc_h is not None or args.slice_center_mpc_h is not None:
            p.error("slice options apply to 2-D map exports, not the interactive 3-D viewer")
        if not args.open_3d_all and args.max_points is None:
            p.error("the sampled 3-D viewer requires an explicit --max-points")
        open_interactive_3d(
            args.run_dir, args.snapshot, args.species, max_points=args.max_points,
            full_resolution=args.open_3d_all,
        )
        return
    if not args.product:
        p.error("select visualization products explicitly with --product")
    fields = args.field
    projections = args.projection
    if ("maps" in args.product or args.compare_with) and (fields is None or projections is None):
        p.error("map export and comparison require explicit --field and --projection selections")
    if "blender" in args.product and args.species is None:
        p.error("Blender export requires an explicit --species")
    if ((args.slice_thickness_mpc_h is None) != (args.slice_center_mpc_h is None)):
        p.error("slab exports require both --slice-thickness-mpc-h and --slice-center-mpc-h")
    products = set(args.product)
    if "maps" in products:
        make_maps(
            args.run_dir, fields, projections,
            args.slice_thickness_mpc_h, args.slice_center_mpc_h,
        )
    if "movie" in products:
        make_movie(args.run_dir, args.slice_thickness_mpc_h, args.slice_center_mpc_h)
    for product, field in (("gas_movie", "gas"), ("temperature_movie", "gas_temperature"),
                           ("stars_movie", "stars")):
        if product in products:
            make_movie(args.run_dir, args.slice_thickness_mpc_h, args.slice_center_mpc_h, field=field)
    if "halo_movie" in products:
        make_halo_growth_movie(args.run_dir)
    if "xdmf" in products:
        make_xdmf(args.run_dir)
    if "blender" in products:
        make_blender_loader(args.run_dir, species=args.species)
    if args.compare_with:
        make_comparison_maps(
            args.compare_with, args.run_dir, fields=fields, projections=projections,
            slice_thickness_mpc_h=args.slice_thickness_mpc_h,
            slice_center_mpc_h=args.slice_center_mpc_h,
        )
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from core.data import particle_id_hash64, snapshot_total_counts, isolated_epoch_contract, coordinate_contract
from core.native import directory_owner, native_run_lock
from core.foundation import atomic_json, atomic_text, file_sha256 as digest, canonical_digest, publish_directory
from core.data import export_model_arrays, verify_model_arrays
from core.foundation import KPC_CM as CM_PER_KPC, SECONDS_PER_YEAR, physical_constants_contract

SECONDS_PER_GYR = SECONDS_PER_YEAR * 1e9
SNAPSHOT_NAME = re.compile(r"^snapshot_(\d+)(?:\.(\d+))?\.(?:hdf5|h5)$", re.IGNORECASE)




def _retain_lowest_hash_ids(current: np.ndarray, additions: np.ndarray, limit: int) -> np.ndarray:
    """Retain the same bottom-hash ID sample regardless of snapshot order."""
    if not len(additions):
        return current
    ids = np.unique(np.concatenate((current, np.asarray(additions, dtype=np.uint64))))
    if len(ids) <= limit:
        return ids
    hashes = particle_id_hash64(ids)
    return ids[np.argpartition(hashes, limit - 1)[:limit]]


def _snapshot_groups(snapshot_dir: Path) -> list[dict]:
    groups: dict[int, list[tuple[int, Path]]] = {}
    for path in snapshot_dir.iterdir():
        match = SNAPSHOT_NAME.match(path.name)
        if match and path.is_file():
            snapshot = int(match.group(1))
            part = int(match.group(2) or 0)
            groups.setdefault(snapshot, []).append((part, path))
    result = []
    for snapshot, parts in sorted(groups.items()):
        parts.sort(key=lambda item: item[0])
        part_numbers = [part for part, _path in parts]
        if part_numbers != list(range(len(part_numbers))):
            raise ValueError("Snapshot %03d has missing or duplicate numbered HDF5 parts" % snapshot)
        result.append({"snapshot": snapshot, "files": [path for _part, path in parts]})
    if len(result) < 2:
        raise ValueError("Provide at least two HDF5 outputs named snapshot_NNN[.part].hdf5")
    return result


def _value(attrs, name):
    if name not in attrs:
        return None
    value = np.asarray(attrs[name]).reshape(-1)
    if len(value) != 1:
        raise ValueError("Snapshot metadata must be scalar: " + name)
    return value[0].item()


def _flag(value, name):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
        raise ValueError("%s must be encoded as a boolean flag" % name)
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(value)
    raise ValueError("%s must be encoded as a boolean flag" % name)


def _single_shard_metadata(path: Path) -> dict:
    """Read one shard's scalar timing/geometry/unit declarations."""
    with h5py.File(path, "r") as source:
        header = source["Header"].attrs
        parameters = source.get("Parameters")
        parameter_attrs = parameters.attrs if parameters is not None else {}
        raw_time = _value(header, "Time")
        if raw_time is None:
            raise ValueError("GADGET HDF5 Header/Time is required")
        time_code = float(raw_time)
        comoving_value = _value(parameter_attrs, "ComovingIntegrationOn")
        if comoving_value is None:
            comoving_value = _value(header, "ComovingIntegrationOn")
        periodic_value = _value(parameter_attrs, "PeriodicBoundariesOn")
        if periodic_value is None:
            periodic_value = _value(header, "PeriodicBoundariesOn")
        length_cm = _value(parameter_attrs, "UnitLength_in_cm")
        if length_cm is None:
            length_cm = _value(header, "UnitLength_in_cm")
        velocity_cms = _value(parameter_attrs, "UnitVelocity_in_cm_per_s")
        if velocity_cms is None:
            velocity_cms = _value(header, "UnitVelocity_in_cm_per_s")
        num_files = _value(header, "NumFilesPerSnapshot")
        redshift = _value(header, "Redshift")
        box_size = _value(header, "BoxSize")
    return {
        "time_code": time_code,
        "redshift": None if redshift is None or not math.isfinite(float(redshift)) else float(redshift),
        "comoving": _flag(comoving_value, "ComovingIntegrationOn"),
        "periodic": _flag(periodic_value, "PeriodicBoundariesOn"),
        "length_unit_cm": None if length_cm is None else float(length_cm),
        "velocity_unit_cm_s": None if velocity_cms is None else float(velocity_cms),
        "box_size_code_units": None if box_size is None else float(box_size),
        "file_count": num_files,
    }


def _snapshot_metadata(files: list[Path]) -> dict:
    """Require every shard to describe the same saved state and native units."""
    from core.data import verify_discovered_native_sets
    verify_discovered_native_sets(files)
    rows = [_single_shard_metadata(path) for path in files]
    if not rows:
        raise ValueError("A saved output needs at least one HDF5 shard")
    reference = rows[0]
    for row in rows:
        declared = row["file_count"]
        if declared is not None and (int(declared) != declared or int(declared) != len(files)):
            raise ValueError("Snapshot shard count disagrees with NumFilesPerSnapshot")
        for key in reference:
            if key == "file_count":
                continue
            left, right = reference[key], row[key]
            if (left is None) != (right is None):
                raise ValueError("Snapshot shards disagree about metadata availability: " + key)
            if left is None:
                continue
            if key in {"comoving", "periodic"}:
                consistent = left == right
            else:
                consistent = math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
            if not consistent:
                raise ValueError("Snapshot shards disagree about " + key)
    return {**reference, "file_count": len(files)}


def _read_type_fields(files: list[Path], particle_type: int, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    group_name = "PartType%d" % int(particle_type)
    declared_counts = []
    declared_totals = []
    chunks = {field: [] for field in fields}
    for path in files:
        with h5py.File(path, "r") as source:
            group = source.get(group_name)
            header = source["Header"].attrs
            if "NumPart_Total" in header:
                declared_totals.append(int(snapshot_total_counts(header)[particle_type]))
            counts = header.get("NumPart_ThisFile")
            expected = None
            if counts is not None:
                counts = np.asarray(counts)
                if counts.shape != (6,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
                    raise ValueError("NumPart_ThisFile must contain six non-negative integer counts")
                expected = int(counts[particle_type])
                declared_counts.append(expected)
            if group is None:
                if expected:
                    raise ValueError("A declared particle type is missing from a snapshot shard: " + group_name)
                continue
            if "Coordinates" not in group or "ParticleIDs" not in group:
                raise ValueError("%s needs both Coordinates and ParticleIDs in %s" % (group_name, path.name))
            size = len(group["Coordinates"])
            if expected is not None and size != expected:
                raise ValueError("Particle rows disagree with the shard header: " + group_name)
            for field in fields:
                if field == "ParticleIDs":
                    dataset_name = "ParticleIDs"
                else:
                    dataset_name = field
                if dataset_name not in group:
                    raise ValueError("%s/%s is absent in %s" % (group_name, dataset_name, path.name))
                if group[dataset_name].ndim == 0 or len(group[dataset_name]) != size:
                    raise ValueError("Particle fields are not aligned within a shard: " + group_name)
                chunks[field].append(np.asarray(group[dataset_name]))
    result = {}
    for field, values in chunks.items():
        if values:
            result[field] = np.concatenate(values, axis=0)
        elif field == "Coordinates":
            result[field] = np.empty((0, 3), dtype=np.float64)
        else:
            result[field] = np.empty(0, dtype=np.uint64)
    if declared_totals and (len(declared_totals) != len(files) or len(set(declared_totals)) != 1
                            or declared_totals[0] != len(result["ParticleIDs"])):
        raise ValueError("Particle rows do not cover consistent declared snapshot totals")
    if declared_counts and len(declared_counts) != len(files):
        raise ValueError("Particle-count headers are missing from some snapshot shards")
    if (result["Coordinates"].ndim != 2 or result["Coordinates"].shape[1] != 3
            or len(result["Coordinates"]) != len(result["ParticleIDs"])):
        raise ValueError("%s Coordinates and ParticleIDs must be aligned (particles, 3) data" % group_name)
    if np.any(~np.isfinite(result["Coordinates"])):
        raise ValueError("%s/Coordinates contains non-finite values" % group_name)
    ids = result["ParticleIDs"]
    if ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("%s/ParticleIDs must be nonnegative integer IDs" % group_name)
    result["ParticleIDs"] = ids.astype(np.uint64, copy=False)
    if len(ids) and len(np.unique(result["ParticleIDs"])) != len(ids):
        raise ValueError("%s contains duplicate IDs in one snapshot" % group_name)
    return result


def _write_position_ply(path: Path, positions_kpc: np.ndarray) -> None:
    points = np.asarray(positions_kpc, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
        raise ValueError("Exported GADGET coordinates must be finite (particles, 3) arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        ) % len(points)
        output.write(header.encode("ascii"))
        points.tofile(output)


def _export_to_directory(snapshots_dir: Path, output_dir: Path, *, max_particles: int,
           dark_type: int, stellar_type: int, gas_type: int,
           black_hole_type: int, length_unit_kpc: float | None,
           assume_noncomoving: bool, allow_periodic: bool) -> Path:
    """Export particle positions and an auditable saved-output manifest."""
    snapshots_dir = snapshots_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not snapshots_dir.is_dir():
        raise FileNotFoundError("GADGET snapshot directory does not exist: " + str(snapshots_dir))
    if isinstance(max_particles, bool) or int(max_particles) != max_particles or max_particles < 1:
        raise ValueError("max_particles must be a positive integer")
    supplied_types = {"dm": dark_type, "stars": stellar_type, "gas": gas_type, "black_holes": black_hole_type}
    if any(isinstance(value, (bool, np.bool_)) or int(value) != value for value in supplied_types.values()):
        raise ValueError("Particle type numbers must be integers, not rounded values or booleans")
    particle_types = {name: int(value) for name, value in supplied_types.items()}
    if any(value < 0 or value > 5 for value in particle_types.values()):
        raise ValueError("GADGET particle type numbers must be between 0 and 5")
    if len(set(particle_types.values())) != len(particle_types):
        raise ValueError("Dark matter, stars, gas, and black holes need distinct particle types")
    if length_unit_kpc is not None and (not math.isfinite(length_unit_kpc) or length_unit_kpc <= 0):
        raise ValueError("length_unit_kpc must be finite and positive")

    outputs = _snapshot_groups(snapshots_dir)
    source_hashes = {str(path): digest(path) for row in outputs for path in row['files']}
    identity_domain = 'isolated_saved_source:' + canonical_digest(source_hashes)
    metadata_rows = [_snapshot_metadata(row["files"]) for row in outputs]
    times = np.asarray([row["time_code"] for row in metadata_rows], dtype=float)
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("GADGET Header/Time values must be finite and strictly increasing")
    comoving_values = {row["comoving"] for row in metadata_rows if row["comoving"] is not None}
    if len(comoving_values) > 1:
        raise ValueError("ComovingIntegrationOn changes between snapshots")
    if comoving_values and True in comoving_values:
        raise ValueError("This importer is for isolated non-comoving galaxy runs; cosmological snapshots use the suite timeline exporter")
    comoving_missing = any(row["comoving"] is None for row in metadata_rows)
    if (not comoving_values or comoving_missing) and not assume_noncomoving:
        raise ValueError("Some snapshot files do not declare ComovingIntegrationOn=0; pass --assume-noncomoving only after checking the run")
    periodic_values = {row["periodic"] for row in metadata_rows if row["periodic"] is not None}
    if len(periodic_values) > 1:
        raise ValueError("PeriodicBoundariesOn changes between snapshots")
    periodic = next(iter(periodic_values)) if periodic_values else None
    periodic_missing = any(row["periodic"] is None for row in metadata_rows)
    if (periodic is True or periodic_missing) and not allow_periodic:
        raise ValueError("PeriodicBoundariesOn is enabled or unavailable; review the output geometry, then pass --allow-periodic to continue")

    declared_lengths = [row["length_unit_cm"] for row in metadata_rows if row["length_unit_cm"] is not None]
    if any(not math.isfinite(value) or value <= 0.0 for value in declared_lengths):
        raise ValueError("UnitLength_in_cm must be finite and positive")
    if declared_lengths and not np.allclose(declared_lengths, declared_lengths[0], rtol=1e-10, atol=0.0):
        raise ValueError("UnitLength_in_cm changes between snapshots")
    if 0 < len(declared_lengths) < len(metadata_rows) and length_unit_kpc is None:
        raise ValueError("UnitLength_in_cm is missing from some outputs; supply --length-unit-kpc after checking the parameter file")
    if declared_lengths:
        unit_kpc = float(declared_lengths[0]) / CM_PER_KPC
        length_source = "GADGET UnitLength_in_cm from HDF5 Parameters"
        if length_unit_kpc is not None and not math.isclose(unit_kpc, length_unit_kpc, rel_tol=1e-8):
            raise ValueError("--length-unit-kpc conflicts with the HDF5 UnitLength_in_cm declaration")
        if 0 < len(declared_lengths) < len(metadata_rows):
            length_source += "; checked against user-supplied --length-unit-kpc for missing outputs"
    elif length_unit_kpc is not None:
        unit_kpc = float(length_unit_kpc)
        length_source = "user-supplied --length-unit-kpc"
    else:
        raise ValueError("HDF5 has no UnitLength_in_cm; supply --length-unit-kpc from the GADGET parameter file")

    velocities = [row["velocity_unit_cm_s"] for row in metadata_rows if row["velocity_unit_cm_s"] is not None]
    if any(not math.isfinite(value) or value <= 0.0 for value in velocities):
        raise ValueError("UnitVelocity_in_cm_per_s must be finite and positive")
    if velocities and not np.allclose(velocities, velocities[0], rtol=1e-10, atol=0.0):
        raise ValueError("UnitVelocity_in_cm_per_s changes between snapshots")
    time_unit_gyr = None
    if len(velocities) == len(metadata_rows) and len(declared_lengths) == len(metadata_rows):
        time_unit_gyr = float(declared_lengths[0]) / float(velocities[0]) / SECONDS_PER_GYR
    elif len(velocities) == len(metadata_rows) and length_unit_kpc is not None:
        time_unit_gyr = float(length_unit_kpc) * CM_PER_KPC / float(velocities[0]) / SECONDS_PER_GYR

    # First pass: select one deterministic ID sample over the union of outputs.
    selected_ids = {name: np.empty(0, dtype=np.uint64) for name in particle_types}
    available_counts = {name: 0 for name in particle_types}
    origin_samples = []
    for output_index, output in enumerate(outputs):
        for name, particle_type in particle_types.items():
            data = _read_type_fields(output["files"], particle_type, ("Coordinates", "ParticleIDs"))
            ids, coords = data["ParticleIDs"], data["Coordinates"]
            available_counts[name] += len(ids)
            selected_ids[name] = _retain_lowest_hash_ids(selected_ids[name], ids, int(max_particles))
            if output_index == 0 and len(coords):
                origin_samples.append(np.asarray(coords, dtype=np.float64) * unit_kpc)
    if available_counts["dm"] == 0:
        raise ValueError("No dark-matter particles found in the configured GADGET particle type")
    if available_counts["stars"] == 0:
        raise ValueError("No stellar particles found; tidal debris needs actual stellar particles in the outputs")
    if not origin_samples:
        raise ValueError("The first output contains no particles in the selected species")
    initial_positions = np.concatenate(origin_samples, axis=0)
    origin_kpc = np.median(initial_positions, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    particle_dir = output_dir / "particles"
    output_records = []
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for output, meta in zip(outputs, metadata_rows):
        record = {
            "snapshot": int(output["snapshot"]),
            'epoch': isolated_epoch_contract(meta['time_code'], time_unit_gyr=time_unit_gyr, snapshot=int(output['snapshot'])),
            'numeric_products': {},
            "time_code": float(meta["time_code"]),
            "time_gyr": None if time_unit_gyr is None else float(meta["time_code"] * time_unit_gyr),
            "redshift_header": meta["redshift"],
            "source_files": [path.name for path in output["files"]],
            "particle_counts": {},
            "sampled_counts": {},
            "particle_files": {},
        }
        for name, particle_type in particle_types.items():
            data = _read_type_fields(output["files"], particle_type, ("Coordinates", "ParticleIDs"))
            ids = data["ParticleIDs"]
            coords = np.asarray(data["Coordinates"], dtype=np.float64) * unit_kpc
            if not len(coords):
                record["particle_counts"][name] = 0
                record["sampled_counts"][name] = 0
                record["particle_files"][name] = None
                continue
            coords -= origin_kpc
            lower = np.minimum(lower, np.min(coords, axis=0))
            upper = np.maximum(upper, np.max(coords, axis=0))
            keep = np.isin(ids, selected_ids[name], assume_unique=True)
            sampled = coords[keep]
            record["particle_counts"][name] = int(len(coords))
            record["sampled_counts"][name] = int(len(sampled))
            if len(sampled):
                path = particle_dir / ("%s_snapshot_%03d.ply" % (name, output["snapshot"]))
                _write_position_ply(path, sampled)
                coordinates = coordinate_contract(position_units='proper kpc',
                    origin='Source origin translated by first saved output component median: ' + json.dumps(origin_kpc.tolist()),
                    periodic_wrapping='native periodic positions retained without unwrapping' if periodic else 'isolated nonperiodic domain',
                    velocity_convention='No velocities exported; native isolated convention retained')
                context = {'epoch': record['epoch'], 'coordinates': coordinates,
                    'sources': source_hashes, 'science_acceptance': 'NOT_ACCEPTED',
                    'adopted_constants': physical_constants_contract(),
                    'helper_code_sha256': {'gadget_io': digest(Path(__file__).resolve().parents[1] / 'core/data.py')},
                    'identity_continuity': 'Source raw IDs retained; no IC ancestry evidence supplied'}
                bundle = export_model_arrays(path,
                    {'particle_ids': ids[keep], 'positions_kpc': sampled, 'ply_positions_kpc': sampled.astype('<f4')},
                    {'particle_ids': {'units': 'identity', 'quantity': 'Saved native particle identity',
                        'identity_namespace': identity_domain + ':PartType%d' % particle_type,
                        'ancestry': {'relation': 'unchanged raw saved source ID', 'species': particle_type,
                                     'initial_condition_ancestry': 'UNDECLARED'}},
                     'positions_kpc': {'units': 'proper kpc', 'quantity': 'Translated native saved particle position'},
                     'ply_positions_kpc': {'units': 'proper kpc', 'quantity': 'Float32 PLY display position',
                         'annotations': {'precision': 'float32 display cast; float64 scientific positions retained'}}},
                    algorithm=__file__, context=context)
                verify_model_arrays(bundle['directory'])
                record['numeric_products'][name] = {'directory': str(Path(bundle['directory']).relative_to(output_dir)),
                    'array_contract': bundle['array_contract']}

                record["particle_files"][name] = str(path.relative_to(output_dir))
            else:
                record["particle_files"][name] = None
        output_records.append(record)

    species_metadata = {}
    for name, particle_type in particle_types.items():
        if available_counts[name]:
            species_metadata[name] = {
                "particle_type": int(particle_type),
                "particle_records_across_outputs": int(available_counts[name]),
                "sampled_union_count": int(len(selected_ids[name])),
                "sampling": "lowest SplitMix64 hash of ParticleIDs over the union of saved outputs",
            }
    source_files = sorted({path for output in outputs for path in output["files"]})
    manifest = {
        "schema": "darkmattersim-isolated-gadget-merger-v1",
        "title": "Isolated GADGET merger saved-output scene",
        "source_directory": str(snapshots_dir),
        "source_file_records": [
            {"name": path.name, "path": str(path), "sha256": source_hashes[str(path)], "size_bytes": int(path.stat().st_size)} for path in source_files
        ],
        "source_snapshot_count": len(outputs),
        "particle_types": species_metadata,
        "outputs": output_records,
        "coordinate_units": "kpc",
        "length_unit_kpc_per_code_unit": float(unit_kpc),
        "length_unit_source": length_source,
        "time_unit_gyr_per_code_unit": time_unit_gyr,
        "time_unit_source": (
            "UnitLength_in_cm / UnitVelocity_in_cm_per_s" if time_unit_gyr is not None
            else "unknown; display Header/Time in GADGET code units"
        ),
        "comoving_integration": False,
        "periodic_boundaries": periodic,
        "periodic_boundary_review_override": bool(allow_periodic),
        "coordinate_transform": {
            "source": "GADGET PartTypeN/Coordinates multiplied by declared or user-supplied length unit",
            "translation_origin_kpc": [float(value) for value in origin_kpc],
            "origin_method": "component-wise median of the configured particle species in the first saved output",
            "rotation_or_rescaling": "none",
        },
        "scene_bounds_kpc": {
            "minimum": [float(value) for value in lower],
            "maximum": [float(value) for value in upper],
        },
        "particle_sampling": {
            "maximum_ids_per_species": int(max_particles),
            "selection_is_stable_between_outputs": True,
        },
        "animation_policy": "discrete saved outputs only; no interpolated trajectory or newly inferred stream membership",
        "scientific_limits": [
            "The particles and their dynamics come from the supplied GADGET HDF5 snapshots.",
            "This importer does not rerun gravity, classify bound versus stripped stars, or detect subhalos.",
            "Periodic coordinates are not unwrapped by this importer; review particle positions for box-edge splits.",
            "Point radii and species colors in Blender are display choices, not physical sizes or temperatures.",
            "Resolution, force softening, time spacing, initial conditions, and GADGET model choices limit what structure is resolved.",
            "PartType5 particles remain Newtonian GADGET markers and are not relativistic event horizons.",
        ],
    }
    manifest_path = output_dir / "merger_manifest.json"
    manifest['publication_contract'] = 'isolated-merger-publication-v1'
    manifest['science_acceptance'] = 'NOT_ACCEPTED'
    manifest['algorithm_sha256'] = digest(__file__)
    manifest['identity_domain'] = identity_domain
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('Isolated merger snapshot changed during export')
    manifest['artifacts'] = [{'path': path.relative_to(output_dir).as_posix(), 'sha256': digest(path)}
        for path in sorted(output_dir.rglob('*')) if path.is_file() and
        not any(part.startswith('.') for part in path.relative_to(output_dir).parts)]
    atomic_json(manifest_path, manifest)
    atomic_text(manifest_path.with_name(manifest_path.name + '.sha256'), digest(manifest_path) + '\n')
    return manifest_path


@directory_owner('output_dir')
def export(snapshots_dir: Path, output_dir: Path, **options) -> Path:
    """Publish one immutable actual-data merger export; retain failed stages."""
    snapshots_dir, output_dir = Path(snapshots_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists(): raise FileExistsError('Keep completed merger exports; choose a fresh destination')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.merger-export-', dir=output_dir.parent))
    try:
        with native_run_lock(snapshots_dir, operation='isolated snapshot export'):
            manifest = _export_to_directory(snapshots_dir, stage, **options)
            from blender import verify_export_manifest
            verify_export_manifest(manifest, schema='darkmattersim-isolated-gadget-merger-v1')
            publish_directory(stage, output_dir)
        return output_dir / manifest.name
    except BaseException as exc:
        if stage.exists(): atomic_json(stage / 'failure.json', {'status': 'FAIL',
            'science_acceptance': 'NOT_ACCEPTED', 'failure': {'type': type(exc).__name__, 'message': str(exc)}})
        raise


def export_gadget_merger_main() -> None:
    """Parse command arguments and dispatch this module's documented workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, type=Path, help="directory containing GADGET HDF5 snapshot_NNN files")
    parser.add_argument("--output", required=True, type=Path, help="directory for sampled PLY data and manifest")
    parser.add_argument("--max-particles", type=int, default=250_000, help="maximum stable ParticleID sample per species")
    parser.add_argument("--dark-type", type=int, default=1, help="GADGET particle type for collisionless halo particles")
    parser.add_argument("--stellar-type", type=int, default=2, help="GADGET particle type for stellar disk/bulge particles")
    parser.add_argument("--gas-type", type=int, default=0, help="GADGET particle type for gas (included only when present)")
    parser.add_argument("--black-hole-type", type=int, default=5, help="GADGET particle type for BH markers (included only when present)")
    parser.add_argument("--length-unit-kpc", type=float, help="physical kpc per coordinate code unit if HDF5 does not declare UnitLength_in_cm")
    parser.add_argument("--assume-noncomoving", action="store_true", help="allow snapshots without a ComovingIntegrationOn header after you verify the parameter file")
    parser.add_argument("--allow-periodic", action="store_true", help="allow periodic inputs after reviewing for box-edge splits")
    args = parser.parse_args()
    manifest = export(
        args.snapshots, args.output, max_particles=args.max_particles,
        dark_type=args.dark_type, stellar_type=args.stellar_type,
        gas_type=args.gas_type, black_hole_type=args.black_hole_type,
        length_unit_kpc=args.length_unit_kpc,
        assume_noncomoving=args.assume_noncomoving,
        allow_periodic=args.allow_periodic,
    )
    print("Merger particle manifest: " + str(manifest))

SUITE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SUITE), str(SUITE / "core")]
from core.data import exact_uint64_ids, run_epoch_contract
from core.foundation import product_provenance, verify_product_provenance

DEFAULT_RESOLUTION = 4096
DEFAULT_TILT_DEG = 23.


def rotate_pitch_tilt(coords, tilt_deg=DEFAULT_TILT_DEG):
    """Rotate comoving coordinates about X; +Z is zenith and +Y front."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or np.any(~np.isfinite(coords)):
        raise ValueError("Dome coordinates must be a finite (particles,3) array")
    if not np.isfinite(tilt_deg):
        raise ValueError("Dome tilt must be finite degrees")
    angle = math.radians(tilt_deg)
    result = coords.copy()
    result[:, 1] = coords[:, 1]*math.cos(angle)-coords[:, 2]*math.sin(angle)
    result[:, 2] = coords[:, 1]*math.sin(angle)+coords[:, 2]*math.cos(angle)
    return result


def project_equidistant_fisheye(coords_centered, canvas_size=DEFAULT_RESOLUTION, tilt_deg=DEFAULT_TILT_DEG):
    """Return pixel x,y and a hemisphere mask; positions at the observer are excluded.

    Image rows increase downward. +X maps right; +Y maps to the bottom rim.
    This orientation is shared by the Blender camera (local -Z optical axis).
    """
    if isinstance(canvas_size, bool) or int(canvas_size) != canvas_size or canvas_size < 16:
        raise ValueError("Canvas size must be an integer >=16")
    xyz = rotate_pitch_tilt(coords_centered, tilt_deg)
    distance = np.linalg.norm(xyz, axis=1)
    valid = distance > 0
    theta = np.zeros(len(xyz))
    theta[valid] = np.arccos(np.clip(xyz[valid, 2]/distance[valid], -1., 1.))
    valid &= theta <= math.pi/2.
    azimuth = np.arctan2(xyz[:, 0], xyz[:, 1])
    radius = canvas_size*theta/math.pi
    px = canvas_size/2.+radius*np.sin(azimuth)
    py = canvas_size/2.+radius*np.cos(azimuth)
    # A point exactly on the horizon belongs to the last pixel, not outside.
    maximum = np.nextafter(float(canvas_size), 0.)
    return np.clip(px, 0., maximum), np.clip(py, 0., maximum), valid


def load_particles_for_fisheye(run_dir, snap_idx, field="dm", max_particles=4_000_000, *, observer_mpc_h=None, return_metadata=False, return_arrays=False):
    """Read all snapshot shards with aligned mass/temperature sampling.

    No particle species is substituted for another. Inverse sample inclusion
    weights restore the represented mass when the point budget is limited.
    Temperature weights are two columns (represented mass, Kelvin), for the
    ratio sum(m*T)/sum(m) used by render_domemaster.
    """
    from analysis import (_snapshot_parameter, read_particle_fields,
                         read_snapshot_header, redshift_from_header, require_stored_temperature_kelvin, temperature_from_internal_energy)
    from halos import snapshot_units
    from visualization import _snapshot_constant_particle_mass
    import yaml

    if field not in {"dm", "gas", "gas_temperature"}:
        raise ValueError("Dome field must be dm, gas or gas_temperature")
    run_dir = Path(run_dir).resolve()
    header, files = read_snapshot_header(run_dir, int(snap_idx))
    source_hashes = {str(path.resolve()): digest(path) for path in files}
    from analysis import validate_gadget_snapshot_set
    validated = validate_gadget_snapshot_set(files, int(snap_idx))
    length_unit, mass_unit = snapshot_units(run_dir, snap_idx)
    box = float(np.asarray(header["BoxSize"]).reshape(-1)[0])*length_unit
    z = redshift_from_header(header)
    ptype = 1 if field == "dm" else 0
    names = ["Coordinates", "Masses", "ParticleIDs"]
    if field == "gas_temperature":
        names += ["Temperature", "InternalEnergy", "ElectronAbundance", "StarFormationRate"]
    values, inclusion = read_particle_fields(files, ptype, names, max_points=max_particles, return_sample_weights=True)
    if values["Coordinates"] is None or not len(values["Coordinates"]):
        raise ValueError("Requested %s particles are absent at snapshot %s" % (field, snap_idx))
    if values['ParticleIDs'] is None: raise ValueError('Dome export requires exact saved particle IDs')
    ids = exact_uint64_ids(values['ParticleIDs'])
    positions = np.asarray(values["Coordinates"], dtype=float)*length_unit
    if values["Masses"] is None:
        constant = _snapshot_constant_particle_mass(files, ptype)
        if constant is None:
            raise ValueError("Requested particle type lacks per-particle or positive constant masses")
        masses = np.full(len(positions), constant*mass_unit)
    else:
        masses = np.asarray(values["Masses"], dtype=float)*mass_unit
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("Dome particles need finite positive masses in Msun/h")
    weights = masses*inclusion
    temperature_source = None
    excluded_effective_eos = 0
    if field == "gas_temperature":
        if values["Temperature"] is not None:
            require_stored_temperature_kelvin(files, ptype)
            temperature = np.asarray(values["Temperature"], dtype=float)
            temperature_source = "PartType0/Temperature [K]"
        else:
            if values["InternalEnergy"] is None or values["ElectronAbundance"] is None:
                raise ValueError("Gas temperature requires Temperature or InternalEnergy+ElectronAbundance")
            temperature = temperature_from_internal_energy(
                values["InternalEnergy"], values["ElectronAbundance"],
                unit_velocity_cm_per_s=_snapshot_parameter(files, "UnitVelocity_in_cm_per_s"))
            if temperature is None:
                raise ValueError("Gas temperature conversion lacks snapshot velocity units")
            temperature_source = "ideal-gas InternalEnergy+ElectronAbundance; effective-EOS star-forming gas excluded"
        cfg = yaml.safe_load((run_dir/"config.yaml").read_text())
        if cfg.get("hydro", {}).get("star_formation", False):
            sfr = values["StarFormationRate"]
            if sfr is None:
                raise ValueError("Non-star-forming temperature selection needs StarFormationRate")
            reject = ~np.isfinite(sfr) | (np.asarray(sfr) != 0.)
            excluded_effective_eos = int(np.count_nonzero(reject))
            temperature = temperature.copy()
            temperature[reject] = np.nan
        valid = np.isfinite(temperature) & (temperature > 0)
        if not np.any(valid):
            raise ValueError("No gas has a usable physical temperature")
        positions, weights, temperature = positions[valid], weights[valid], temperature[valid]
        ids, masses, inclusion = ids[valid], masses[valid], inclusion[valid]
        weights = np.column_stack((weights, temperature))
    observer = np.full(3, box/2.) if observer_mpc_h is None else np.asarray(observer_mpc_h, dtype=float)
    if observer.shape != (3,) or np.any(~np.isfinite(observer)):
        raise ValueError("Observer must contain three finite comoving Mpc/h coordinates")
    observer = np.mod(observer, box)
    # Show one nearest periodic copy, without inventing repeated distant boxes.
    centered = positions-observer
    centered -= box*np.rint(centered/box)
    metadata = {"snapshot": int(snap_idx), "field": field, "redshift": float(z),
                "observer_mpc_h": observer.tolist(), "box_mpc_h": box,
                "sample_count": len(positions), "sample_budget": max_particles,
                "sampling": "shared stratified sample with inverse-inclusion mass weights",
                "temperature_source": temperature_source, "excluded_effective_eos_sample_count": excluded_effective_eos,
                "periodicity": "one minimum-image copy relative to observer"}
    coordinates = coordinate_contract(position_units='comoving Mpc/h',
        origin='Saved observer position: ' + json.dumps(observer.tolist()),
        periodic_wrapping='One minimum-image copy relative to observer',
        velocity_convention='No velocities exported')
    provenance = product_provenance(run_dir, algorithm=__file__, snapshot=int(snap_idx),
        source_files=[{'path': path, 'sha256': checksum} for path, checksum in source_hashes.items()],
        coordinates=coordinates, selection='Shared stratified sampled native particles; exact IDs and inverse inclusion weights retained')
    metadata['data_contract'] = provenance
    metadata['epoch'] = run_epoch_contract(run_dir, int(snap_idx), validated['scale_factor'])
    metadata['source_hashes'] = source_hashes
    metadata['identity_namespace'] = provenance['identity_namespaces']['dark_matter' if ptype == 1 else 'gas']
    arrays = {'particle_ids': ids, 'observer_relative_positions_mpc_h': centered,
              'mass_msun_h': masses, 'inverse_inclusion_weight': inclusion,
              'represented_mass_msun_h': weights[:, 0] if field == 'gas_temperature' else weights}
    if field == 'gas_temperature': arrays['temperature_kelvin'] = weights[:, 1]
    if any(digest(path) != checksum for path, checksum in source_hashes.items()):
        raise ValueError('Dome snapshot changed during sampling')
    result = (centered, weights, box, z)
    if return_arrays: return (*result, metadata, arrays)

    return (*result, metadata) if return_metadata else result


def _angular_raster(coords, weights, resolution, tilt, temperature):
    """Histogram mass or mass-weighted temperature with the correct image axes."""
    px, py, visible = project_equidistant_fisheye(coords, resolution, tilt)
    masses = weights[:, 0] if temperature else weights
    # Bin rows downward and columns rightward. At the circular horizon, the
    # containing square pixel can have its centre outside the black aperture.
    # Select the closest in-aperture neighbour in that case so a visible ray's
    # mass is not silently discarded by the display mask. This is pixel
    # quantization only, not a change to its continuous fisheye direction.
    ix = np.floor(px[visible]).astype(np.int64)
    iy = np.floor(py[visible]).astype(np.int64)
    radius2 = (resolution/2.)**2
    boundary = ((ix+.5-resolution/2.)**2 + (iy+.5-resolution/2.)**2) > radius2
    if np.any(boundary):
        offsets = np.asarray([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        cx = np.clip(ix[boundary, None]+offsets[:, 0], 0, resolution-1)
        cy = np.clip(iy[boundary, None]+offsets[:, 1], 0, resolution-1)
        inside = ((cx+.5-resolution/2.)**2 + (cy+.5-resolution/2.)**2) <= radius2
        distances = (cx+.5-px[visible][boundary, None])**2 + (cy+.5-py[visible][boundary, None])**2
        distances[~inside] = np.inf
        nearest = np.argmin(distances, axis=1)
        rows = np.arange(len(nearest))
        if np.any(~np.isfinite(distances[rows, nearest])):
            raise ValueError("A horizon ray has no in-aperture raster neighbour")
        ix[boundary], iy[boundary] = cx[rows, nearest], cy[rows, nearest]
    flat_pixels = iy*resolution + ix
    def deposit(w):
        return np.bincount(flat_pixels, weights=w[visible], minlength=resolution**2).reshape(resolution, resolution)
    mass_grid = deposit(masses)
    if temperature:
        numerator = deposit(masses*weights[:, 1])
        grid = np.divide(numerator, mass_grid, out=np.full_like(mass_grid, np.nan), where=mass_grid > 0)
    else:
        y, x = np.ogrid[:resolution, :resolution]
        theta = np.hypot(x+.5-resolution/2., y+.5-resolution/2.)*math.pi/resolution
        # dOmega/dA_pixel = sin(theta)/(theta*f^2), including its zenith limit.
        solid_angle = np.sinc(theta/math.pi)*(math.pi/resolution)**2
        grid = mass_grid/solid_angle
    return grid.astype(np.float32), visible


def render_domemaster(coords_centered, weights, canvas_size=DEFAULT_RESOLUTION, tilt_deg=DEFAULT_TILT_DEG,
                      field="dm", supersample=True, *, cmap=None, percentiles=(1., 99.9), return_metadata=False, return_arrays=False):
    """Colour an angular mass/temperature raster with black outside the dome.

    Supersampling doubles the requested resolution rather than assuming 4K.
    Palette limits and clipping are recorded; display colour is not physical
    emitted light. Temperature is sum(m*T)/sum(m) [K], never sum(T).
    """
    import matplotlib
    from PIL import Image
    if field not in {"dm", "gas", "gas_temperature"}:
        raise ValueError("Dome field must be dm, gas or gas_temperature")
    if isinstance(canvas_size, bool) or int(canvas_size) != canvas_size or not 16 <= canvas_size <= 8192:
        raise ValueError("Canvas size must be an integer between 16 and 8192")
    temperature = field == "gas_temperature"
    weights = np.asarray(weights, dtype=float)
    n = len(coords_centered)
    expected = (n, 2) if temperature else (n,)
    if weights.shape != expected or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Dome weights must be positive aligned masses, or (mass,temperature) rows")
    low_percentile, high_percentile = map(float, percentiles)
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Colour percentiles must satisfy 0 <= lower < upper <= 100")
    resolution = int(canvas_size)*(2 if supersample else 1)
    grid, visible = _angular_raster(coords_centered, weights, resolution, tilt_deg, temperature)
    y, x = np.ogrid[:resolution, :resolution]
    circle = (x+.5-resolution/2.)**2+(y+.5-resolution/2.)**2 <= (resolution/2.)**2
    occupied = circle & np.isfinite(grid) & (grid > 0)
    if not np.any(occupied):
        raise ValueError("No particles project onto the visible dome hemisphere")
    logged = np.log10(grid[occupied])
    low, high = np.percentile(logged, [low_percentile, high_percentile])
    high = float(high)
    collapsed_range = high == float(low)
    normal = np.zeros_like(grid)
    if collapsed_range:
        # Equal percentile limits can also occur with rare outliers. Preserve
        # their ordering while giving values at the limit a visible neutral tone.
        normal[occupied] = np.where(logged < low, 0., np.where(logged > high, 1., .5))
    else:
        normal[occupied] = np.clip((logged-low)/(high-low), 0., 1.)
    palette = cmap or ("inferno" if temperature else "magma")
    # A sequential palette preserves ordered scalar values; no categorical
    # node colours suggest unmeasured shocks, streams or stellar light.
    rgb = matplotlib.colormaps[palette](normal, bytes=True)[..., :3]
    rgb[~occupied] = 0
    if supersample:
        rgb = np.asarray(Image.fromarray(rgb).resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)).copy()
    # Reapply the final circular mask so Lanczos ringing never leaks off-dome.
    y, x = np.ogrid[:canvas_size, :canvas_size]
    outside = (x+.5-canvas_size/2.)**2+(y+.5-canvas_size/2.)**2 > (canvas_size/2.)**2
    rgb[outside] = 0
    metadata = {"projection": "180-degree equidistant fisheye; r=f*theta",
                "orientation": "+X right; +Y bottom; +Z centre before tilt",
                "resolution": [canvas_size, canvas_size], "tilt_degrees": float(tilt_deg),
                "supersampling_factor": 2 if supersample else 1, "palette": palette,
                "scalar": "mass-weighted non-star-forming gas temperature" if temperature else "angular mass density",
                "units": "K" if temperature else "(Msun/h)/sr", "display_transform": "log10",
                "display_percentiles": [low_percentile, high_percentile], "log10_colour_limits": [float(low), high],
                "collapsed_colour_range": collapsed_range,
                "collapsed_range_policy": "below/at/above equal limits map to 0/0.5/1; numeric raster unchanged",
                "visible_sample_count": int(np.count_nonzero(visible)),
                "horizon_pixel_policy": "nearest in-aperture pixel centre; preserves all visible ray mass",
                "interpretation": "simulation mass/temperature presentation; no luminosity, light cone, lensing or projector warp"}
    if return_arrays:
        return rgb, metadata, {"angular_scalar_raster": grid, "occupied_pixels": occupied,
            "visible_sample": visible, "output_rgb": rgb}
    return (rgb, metadata) if return_metadata else rgb


def configure_blender_fulldome_camera(bpy, scene, box_size_mpc_h, tilt_deg=DEFAULT_TILT_DEG, resolution=DEFAULT_RESOLUTION):
    """Orient Blender's local -Z camera toward the same tilted zenith as the raster."""
    if not np.isfinite(box_size_mpc_h) or box_size_mpc_h <= 0:
        raise ValueError("Box must be finite positive Mpc/h")
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    data = bpy.data.cameras.new("DomeMaster_Equidistant")
    data.type = "PANO"
    settings = data if hasattr(data, "panorama_type") else data.cycles
    settings.panorama_type = "FISHEYE_EQUIDISTANT"
    settings.fisheye_fov = math.pi
    camera = bpy.data.objects.new("DomeMaster_Camera", data)
    scene.collection.objects.link(camera)
    camera.location = (box_size_mpc_h/2.,)*3
    camera.rotation_euler = (math.pi+math.radians(tilt_deg), 0., 0.)
    scene.camera = camera


@native_run_owner
def export_domemaster(run_dir, snapshot, output, *, field='dm', max_particles=3_000_000,
                      observer_mpc_h=None, resolution=DEFAULT_RESOLUTION, tilt=DEFAULT_TILT_DEG,
                      supersample=False, cmap=None):
    """Export actual sampled inputs and raster values through the central contract."""
    from PIL import Image
    from core.data import export_image_product
    coords, weights, box, z, source, inputs = load_particles_for_fisheye(
        run_dir, snapshot, field, max_particles, observer_mpc_h=observer_mpc_h, return_arrays=True)
    rgb, display, raster = render_domemaster(coords, weights, resolution, tilt, field,
        supersample, cmap=cmap, return_arrays=True)
    specifications = {
        'particle_ids': {'units': 'identity', 'quantity': 'Exact saved sampled particle identity',
            'identity_namespace': source['identity_namespace'],
            'ancestry': {'relation': 'unchanged saved source ID', 'snapshot': snapshot}},
        'observer_relative_positions_mpc_h': {'units': 'comoving Mpc/h', 'quantity': 'Minimum-image position relative to observer'},
        'mass_msun_h': {'units': 'Msun/h', 'quantity': 'Native individual particle mass'},
        'represented_mass_msun_h': {'units': 'Msun/h', 'quantity': 'Mass times inverse sample inclusion probability'},
        'inverse_inclusion_weight': {'units': 'dimensionless', 'quantity': 'Inverse stratified sample inclusion probability'},
        'temperature_kelvin': {'units': 'K', 'quantity': 'Physical non-star-forming gas temperature'},
        'angular_scalar_raster': {'units': display['units'], 'quantity': display['scalar'],
            'missing_policy': 'Unoccupied temperature pixels are undefined and retained as NaN',
            'annotations': {'resolution': 'Raster before supersample reduction and display palette'}},
        'occupied_pixels': {'units': 'dimensionless', 'quantity': 'In-aperture finite positive scalar pixel indicator'},
        'visible_sample': {'units': 'dimensionless', 'quantity': 'Sample visible in selected dome hemisphere'},
        'output_rgb': {'units': 'uint8 display channel code', 'quantity': 'Final palette RGB display codes'} }
    arrays = {**inputs, **raster}
    specifications = {name: specifications[name] for name in arrays}
    report = {'schema_version': 3, 'source': source, 'display': display, 'code_sha256': digest(__file__)}
    verify_product_provenance(run_dir, source['data_contract'])
    published = export_image_product(output, Image.fromarray(rgb), report, arrays, specifications,
        algorithm=__file__, context={'epoch': source['epoch'], 'coordinates': source['data_contract']['coordinates'],
            'source_provenance': source['data_contract']}, source_hashes=source['source_hashes'])
    return {**published, 'artifact_registration':register_domemaster_artifact(run_dir, output)}


def register_domemaster_artifact(run_dir, output):
    """Verify and index a completed dome export; safe to retry after index failure."""
    from blender import verify_image_publication
    from core.foundation import register_artifacts, run_namespace_binding, canonical_digest
    run = Path(run_dir).resolve()
    results = run/'results'
    output = Path(output).resolve()
    report = verify_image_publication(output)
    if not output.is_relative_to(results):
        return {'status':'EXTERNAL_OUTPUT','reason':'Explicit output is outside the run results tree'}
    ownership = run_namespace_binding(run)
    if ownership['status'] != 'DECLARED':
        return {'status':'UNRESOLVED','reason':'Persistent run ownership is undeclared; image remains uncatalogued'}
    provenance = report['source']['data_contract']
    if provenance.get('run_namespace') != ownership['run_namespace']:
        raise ValueError('Dome product belongs to another run namespace')
    verify_product_provenance(run, provenance)
    archive = output.with_name(output.name+'.product')
    numeric = archive/report['numeric_data']['directory']
    namespace = ownership['run_namespace']
    recipe = canonical_digest({'algorithm_sha256':report['code_sha256'],
        'display':report['display'],'field':report['source']['field'],
        'observer_mpc_h':report['source']['observer_mpc_h'],
        'sampling':report['source']['sampling'],'sample_budget':report['source']['sample_budget']})
    relative = output.relative_to(results).as_posix()
    row = {'artifact_id':'domemaster-'+canonical_digest({'path':relative,'recipe':recipe,'sha256':report['output_sha256']}),
        'run_namespace':namespace,'catalogue_id':None,
        'product_generation':archive.relative_to(results).as_posix(),
        'object_selection':report['source']['field']+'; '+report['source']['sampling'],
        'epoch_selection':json.dumps(report['source']['epoch'],sort_keys=True,allow_nan=False),
        'numeric_sources':[{'relative_path':path.relative_to(results).as_posix(),
            'sha256':digest(path),'run_namespace':namespace}
            for path in (numeric/'arrays.npz',numeric/'contract.json')],
        'recipe_hash':recipe,'support_status':'UNRESOLVED',
        'display_transform':json.dumps(report['display'],sort_keys=True,allow_nan=False),
        'relative_path':relative,'sha256':report['output_sha256']}
    register_artifacts(results,[row])
    return {'status':'REGISTERED','artifact_id':row['artifact_id'],
        'index_relative_path':'artifact-index.json','science_acceptance':'NOT_ACCEPTED'}


def fulldome_planetarium_main():
    """Export a dome image and its complete scientific/display provenance."""
    from PIL import Image
    from analysis import snapshot_ids
    from commands import select_run
    from core.resources import runtime_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--state-root", type=Path, help="External runtime state containing runs")
    parser.add_argument("--snapshot", type=int)
    parser.add_argument("--tilt", type=float, default=DEFAULT_TILT_DEG)
    parser.add_argument("--res", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--supersample", action="store_true")
    parser.add_argument("--field", choices=["dm", "gas", "gas_temperature"], default="dm")
    parser.add_argument("--observer-mpc-h", nargs=3, type=float)
    parser.add_argument("--max-particles", type=int, default=3_000_000)
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 16 <= args.res <= 8192 or args.max_particles < 1:
        parser.error("Resolution must be 16..8192 and particle budget positive")
    paths = runtime_paths(SUITE, args.state_root)
    run_dir = (args.run_dir or select_run(paths.runs_root, "config.yaml")).resolve()
    snapshots = snapshot_ids(run_dir)
    if not snapshots:
        raise FileNotFoundError("No saved snapshots were found")
    snap = snapshots[-1] if args.snapshot is None else args.snapshot
    output = args.output or run_dir/'results/planetarium/stills'/f'domemaster_{args.res}_snap_{snap:03d}_{args.field}.png'
    report = export_domemaster(run_dir, snap, output, field=args.field, max_particles=args.max_particles,
        observer_mpc_h=args.observer_mpc_h, resolution=args.res, tilt=args.tilt,
        supersample=args.supersample, cmap=args.cmap)
    print(f"DomeMaster: {Path(output).resolve()}; science NOT_ACCEPTED; saved z={report['source']['redshift']:g}")

SCENE_COMMANDS = {
    'export-gadget-merger': export_gadget_merger_main,
    'fulldome-planetarium': fulldome_planetarium_main,
    'visualization': visualization_main,
}
